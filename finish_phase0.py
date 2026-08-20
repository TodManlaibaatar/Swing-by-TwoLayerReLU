from dataclasses import replace
from pathlib import Path

import numpy as np

from full_flow_diagnostic import (
    Config,
    run,
    analyze_capture,
    print_capture_summary,
    plot_capture_diagnostics,
    _interpolated_creation,
)


# ============================================================
# Radial-envelope helper
# ============================================================

def radial_envelope(
    history,
    horizon,
    rhos=(0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90),
):
    """
    Compute

        C_rate(rho)
        =
        sup_{t <= horizon}
        [G_W(t) - rho G_A1(t)].

    Also reports the zero-intercept empirical slope threshold

        sup G_W / G_A1

    away from t=0.
    """

    times = np.asarray(
        history["time"]
    )

    G_A1 = np.asarray(
        history["G_A1"]
    )

    G_W = np.asarray(
        history["G_W"]
    )

    mask = (
        times <= horizon + 1e-12
    )

    t = times[mask]
    g1 = G_A1[mask]
    gw = G_W[mask]

    print()
    print("=" * 72)
    print(
        f"RADIAL ENVELOPE THROUGH t <= {horizon:.3f}"
    )
    print("=" * 72)

    # Ignore the tiny 0/0 region.
    ratio_mask = (
        g1 > 1.0
    )

    if np.any(ratio_mask):
        ratios = (
            gw[ratio_mask]
            / g1[ratio_mask]
        )

        print(
            "sup G_W/G_A1 for G_A1 > 1 "
            f"= {np.max(ratios):.6f}"
        )

    print()

    for rho in rhos:
        C_rate = float(
            np.max(
                gw
                - rho * g1
            )
        )

        # Since at t=0 both G's are zero,
        # numerical max can be tiny negative.
        C_rate = max(
            0.0,
            C_rate,
        )

        print(
            f"rho={rho:.2f}: "
            f"C_rate={C_rate:.6f}"
        )


# ============================================================
# Failed-corner timing diagnosis
# ============================================================

def separatrix_bound(
    history,
    cfg,
    j,
):
    """
    Return the best certified separatrix location at weak-field
    checkpoint j.

    If the repeller is directly visible, use it.

    If it has fallen below the grid floor while the stable weak
    branch still exists and V(left) > 0, then the grid floor is a
    conservative upper bound on phi_rep.
    """

    repeller = history[
        "weak_repeller_deg"
    ][j]

    stable = history[
        "weak_stable_deg"
    ][j]

    left_value = history[
        "weak_left_true"
    ][j]

    if np.isfinite(repeller):
        return (
            float(repeller),
            "observed",
        )

    if (
        np.isfinite(stable)
        and left_value > 0.0
    ):
        return (
            float(
                cfg.weak_branch_min_deg
            ),
            "grid-bound",
        )

    return (
        np.nan,
        "unavailable",
    )


def diagnose_failed_corner(
    history,
    cfg,
):
    """
    Determine why the fixed W seed fails or succeeds to enter the
    moving weak basin.
    """

    creation = _interpolated_creation(
        history,
        "weak_H_true",
    )

    print()
    print("=" * 72)
    print("FAILED-CORNER W-SEED TIMING DIAGNOSIS")
    print("=" * 72)

    if creation is None:
        print(
            "No weak basin was created."
        )
        return

    t_create, m_create = creation

    print(
        f"basin creation: "
        f"t={t_create:.6f}, "
        f"m_eff={m_create:.6f}"
    )

    times = np.asarray(
        history["weak_field_time"]
    )

    W_angles = np.asarray(
        history["W_angles_deg"]
    )[:len(times)]

    n_W = W_angles.shape[1]

    counts = {
        "never_entered_Q1": 0,
        "entered_before_creation": 0,
        "entered_after_creation_above_repeller": 0,
        "entered_after_creation_below_repeller": 0,
        "entered_after_creation_small_margin": 0,
        "entry_separatrix_unavailable": 0,
    }

    entries = []

    for i in range(n_W):
        entry_index = None

        for j in range(
            1,
            len(times),
        ):
            if (
                W_angles[j - 1, i] >= 90.0
                and W_angles[j, i] < 90.0
            ):
                entry_index = j
                break

        if entry_index is None:
            counts[
                "never_entered_Q1"
            ] += 1

            continue

        j = entry_index

        t_entry = float(
            times[j]
        )

        phi_entry = float(
            W_angles[j, i]
        )

        if t_entry < t_create:
            counts[
                "entered_before_creation"
            ] += 1

            entries.append(
                (
                    i,
                    t_entry,
                    phi_entry,
                    np.nan,
                    np.nan,
                    "before creation",
                )
            )

            continue

        phi_sep, sep_kind = (
            separatrix_bound(
                history,
                cfg,
                j,
            )
        )

        if not np.isfinite(phi_sep):
            counts[
                "entry_separatrix_unavailable"
            ] += 1

            entries.append(
                (
                    i,
                    t_entry,
                    phi_entry,
                    np.nan,
                    np.nan,
                    "separatrix unavailable",
                )
            )

            continue

        margin = (
            phi_entry
            - phi_sep
        )

        if margin >= (
            cfg.capture_margin_deg
        ):
            counts[
                "entered_after_creation_above_repeller"
            ] += 1

            status = (
                "safe side"
            )

        elif margin > 0.0:
            counts[
                "entered_after_creation_small_margin"
            ] += 1

            status = (
                "positive but uncertified margin"
            )

        else:
            counts[
                "entered_after_creation_below_repeller"
            ] += 1

            status = (
                "wrong side"
            )

        entries.append(
            (
                i,
                t_entry,
                phi_entry,
                phi_sep,
                margin,
                f"{status} ({sep_kind})",
            )
        )

    print()
    print("W-seed entry categories:")

    for key, value in counts.items():
        print(
            f"    {key}: "
            f"{value}/{n_W}"
        )

    print()
    print(
        "Final/minimum W angles by t="
        f"{times[-1]:.2f}:"
    )

    final_angles = W_angles[-1]

    min_angles = np.min(
        W_angles,
        axis=0,
    )

    print(
        "    final angle median = "
        f"{np.median(final_angles):.3f} deg"
    )

    print(
        "    final angle range = "
        f"{np.min(final_angles):.3f} to "
        f"{np.max(final_angles):.3f} deg"
    )

    print(
        "    median minimum angle = "
        f"{np.median(min_angles):.3f} deg"
    )

    print(
        "    global minimum angle = "
        f"{np.min(min_angles):.3f} deg"
    )

    if entries:
        print()
        print(
            "Q2 -> Q1 entry events:"
        )

        for (
            neuron,
            t_entry,
            phi_entry,
            phi_sep,
            margin,
            status,
        ) in entries:

            if np.isfinite(phi_sep):
                print(
                    f"    W[{neuron:02d}]: "
                    f"t={t_entry:.3f}, "
                    f"phi={phi_entry:.3f}, "
                    f"sep={phi_sep:.3f}, "
                    f"margin={margin:+.3f} deg, "
                    f"{status}"
                )
            else:
                print(
                    f"    W[{neuron:02d}]: "
                    f"t={t_entry:.3f}, "
                    f"phi={phi_entry:.3f}, "
                    f"{status}"
                )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 1. Canonical radial envelope through actual capture time
    # ========================================================

    print()
    print("#" * 72)
    print("FINAL P0 CHECK 1: CANONICAL RADIAL ENVELOPE")
    print("#" * 72)

    canonical_cfg = replace(
        Config(),
        mu1=3.0,
        mu2=2.0,
        sigma1=0.15,
        sigma2=0.15,

        max_time=3.0,
        weak_field_max_time=3.0,

        figure_dir=(
            "phase0_final_canonical"
        ),
    )

    (
        canonical_history,
        canonical_phi,
        canonical_final,
    ) = run(
        canonical_cfg
    )

    canonical_capture = (
        analyze_capture(
            canonical_history,
            canonical_cfg,
        )
    )

    print_capture_summary(
        canonical_capture
    )

    # First certified capture.
    radial_envelope(
        canonical_history,
        horizon=1.72,
    )

    # Median W-seed capture.
    radial_envelope(
        canonical_history,
        horizon=2.15,
    )

    # Slightly conservative extra horizon.
    radial_envelope(
        canonical_history,
        horizon=2.25,
    )

    # ========================================================
    # 2. Failed P0-D corner
    # ========================================================

    print()
    print("#" * 72)
    print(
        "FINAL P0 CHECK 2: FAILED CORNER "
        "(mu2=2.20, sigma=0.12)"
    )
    print("#" * 72)

    failed_cfg = replace(
        Config(),
        mu1=3.0,
        mu2=2.2,
        sigma1=0.12,
        sigma2=0.12,

        # Important: run all the way to 6.
        max_time=6.0,

        # Also keep evaluating the branch to 6 so zero capture at
        # t=3 cannot merely mean 'capture happens later.'
        weak_field_max_time=6.0,

        figure_dir=(
            "phase0_final_failed_corner"
        ),
    )

    (
        failed_history,
        failed_phi,
        failed_final,
    ) = run(
        failed_cfg
    )

    failed_capture = (
        analyze_capture(
            failed_history,
            failed_cfg,
        )
    )

    print_capture_summary(
        failed_capture
    )

    diagnose_failed_corner(
        failed_history,
        failed_cfg,
    )

    # Save the same trajectory figures for the negative-control cell.
    plot_capture_diagnostics(
        failed_history,
        failed_capture,
        failed_cfg,
    )

    print()
    print("=" * 72)
    print("FINAL P0 CHECKS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()