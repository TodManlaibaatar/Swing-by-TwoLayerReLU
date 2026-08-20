from dataclasses import replace
from pathlib import Path
import contextlib
import io

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from full_flow_diagnostic import (
    Config,
    run,
    analyze_capture,
    _interpolated_creation,
)


MU2_VALUES = (
    1.8,
    2.0,
    2.2,
)

SIGMA_VALUES = (
    0.12,
    0.15,
    0.18,
)


def interpolate_history_at(
    history,
    key,
    target_time,
):
    return float(
        np.interp(
            target_time,
            history["time"],
            history[key],
        )
    )


def run_cell(
    base_cfg,
    mu2,
    sigma,
):
    cfg = replace(
        base_cfg,
        mu2=mu2,
        sigma1=sigma,
        sigma2=sigma,

        # We only need the Phase-0 window.
        max_time=6.0,
        weak_field_max_time=6.0,

        # No per-cell figures.
        figure_dir=(
            f"phase0_sweep_tmp/"
            f"mu2_{mu2:.2f}_sigma_{sigma:.2f}"
        ),
    )

    # Suppress the long snapshot output for each sweep cell.
    with contextlib.redirect_stdout(
        io.StringIO()
    ):
        (
            history,
            weak_phi,
            final_diag,
        ) = run(cfg)

    creation = _interpolated_creation(
        history,
        "weak_H_true",
    )

    creation_without_W = (
        _interpolated_creation(
            history,
            "weak_H_without_W",
        )
    )

    capture = analyze_capture(
        history,
        cfg,
    )

    if creation is None:
        return {
            "mu2": mu2,
            "sigma": sigma,
            "created": False,
        }

    t_create, m_create = creation

    G1_create = interpolate_history_at(
        history,
        "G_A1",
        t_create,
    )

    GW_create = interpolate_history_at(
        history,
        "G_W",
        t_create,
    )

    LW_create = interpolate_history_at(
        history,
        "L_W",
        t_create,
    )

    radial_ratio = (
        GW_create / G1_create
        if abs(G1_create) > 1e-12
        else np.nan
    )

    if creation_without_W is None:
        delta_m_without_W = np.nan
    else:
        delta_m_without_W = (
            creation_without_W[1]
            - m_create
        )

    if capture is None:
        W_capture = 0.0
        W_persistent = 0.0
    else:
        W_capture = (
            capture["W"][
                "capture_fraction"
            ]
        )

        W_persistent = (
            capture["W"][
                "persistent_fraction"
            ]
        )

    return {
        "mu2": mu2,
        "sigma": sigma,
        "created": True,

        "t_create": t_create,
        "m_create": m_create,

        "delta_m_without_W": (
            delta_m_without_W
        ),

        "L_W_create": LW_create,

        "G_W_over_G_A1_create": (
            radial_ratio
        ),

        "W_capture_fraction": (
            W_capture
        ),

        "W_persistent_fraction": (
            W_persistent
        ),
    }


def metric_matrix(
    results,
    key,
):
    matrix = np.full(
        (
            len(SIGMA_VALUES),
            len(MU2_VALUES),
        ),
        np.nan,
    )

    for result in results:
        i = SIGMA_VALUES.index(
            result["sigma"]
        )

        j = MU2_VALUES.index(
            result["mu2"]
        )

        if key in result:
            matrix[i, j] = (
                result[key]
            )

    return matrix


def save_heatmap(
    matrix,
    title,
    filename,
    label,
):
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
    )

    ax.set_xticks(
        range(len(MU2_VALUES))
    )

    ax.set_xticklabels(
        [
            f"{x:.2f}"
            for x in MU2_VALUES
        ]
    )

    ax.set_yticks(
        range(len(SIGMA_VALUES))
    )

    ax.set_yticklabels(
        [
            f"{x:.2f}"
            for x in SIGMA_VALUES
        ]
    )

    ax.set_xlabel(
        r"$\mu_2$"
    )

    ax.set_ylabel(
        r"$\sigma_1=\sigma_2$"
    )

    ax.set_title(title)

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(label)

    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):
            value = matrix[i, j]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                )

    fig.tight_layout()

    fig.savefig(
        filename,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    output_dir = Path(
        "phase0_sweep_figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_cfg = Config()

    results = []

    for sigma in SIGMA_VALUES:
        for mu2 in MU2_VALUES:
            print(
                "running "
                f"mu2={mu2:.2f}, "
                f"sigma={sigma:.2f}"
            )

            result = run_cell(
                base_cfg,
                mu2,
                sigma,
            )

            results.append(result)

            print(result)

    # ---------------------------------------------------------
    # Save raw table.
    # ---------------------------------------------------------

    import csv

    keys = sorted(
        {
            key
            for result in results
            for key in result.keys()
        }
    )

    with open(
        output_dir
        / "phase0_sweep_results.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    # ---------------------------------------------------------
    # Heatmaps.
    # ---------------------------------------------------------

    save_heatmap(
        metric_matrix(
            results,
            "m_create",
        ),
        "P0-D: true weak-basin creation threshold",
        output_dir
        / "phase0D_m_create.pdf",
        r"$m_{\rm create}$",
    )

    save_heatmap(
        metric_matrix(
            results,
            "W_capture_fraction",
        ),
        "P0-D: W-seed capture fraction",
        output_dir
        / "phase0D_W_capture_fraction.pdf",
        "capture fraction",
    )

    save_heatmap(
        metric_matrix(
            results,
            "G_W_over_G_A1_create",
        ),
        "P0-D: radial-growth ratio at creation",
        output_dir
        / "phase0D_radial_ratio.pdf",
        r"$G_W/G_{A_1}$",
    )

    save_heatmap(
        metric_matrix(
            results,
            "delta_m_without_W",
        ),
        "P0-D: creation shift after removing W",
        output_dir
        / "phase0D_without_W_shift.pdf",
        r"$\Delta m_{\rm create}$",
    )

    print()
    print(
        "Saved P0-D outputs to "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()