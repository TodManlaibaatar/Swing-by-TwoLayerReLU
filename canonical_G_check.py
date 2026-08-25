#!/usr/bin/env python3
"""
canonical_G_check.py

Canonical pre-grid gate for the near-boundary normal form.

Goal
----
Verify, on the canonical p=1 / psi=-85 degree Swing-by example, that

    dot E = eta_1^2 G_{1,-1}(z,t) + O(eta_1^3)

has the correct sign reversal and reversal time.

Run beside:
  - full_flow_diagnostic.py
  - theory_guided_swing_by_phase_diagram.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from full_flow_diagnostic import Config
import theory_guided_swing_by_phase_diagram as exp


def first_neg_to_pos_crossing(times, y, t_lo=None, t_hi=None):
    times = np.asarray(times, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    if t_lo is not None:
        valid &= times >= t_lo
    if t_hi is not None:
        valid &= times <= t_hi
    idx = np.where(valid)[0]
    for i, j in zip(idx[:-1], idx[1:]):
        if y[i] <= 0.0 and y[j] > 0.0:
            if y[j] == y[i]:
                return float(times[j])
            return float(times[i] - y[i] * (times[j] - times[i]) / (y[j] - y[i]))
    return np.nan


def main():
    cfg = replace(
        Config(),
        mu1=3.0,
        mu2=2.0,
        sigma1=0.15,
        sigma2=0.15,
        max_time=10.0,
    )

    t_create = 1.06411
    t_block = 4.0
    t_end = 10.0
    block_tol_deg = 15.0

    psi_deg = -85.0
    psi = math.radians(psi_deg)
    x = np.array([math.cos(psi), math.sin(psi)], dtype=float)

    outdir = Path("canonical_G_check")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CANONICAL NORMAL-FORM G CHECK")
    print("=" * 80)
    print(f"probe psi={psi_deg} deg, p=1, s=-1")
    print("No eta is tuned using OOD behavior.")

    run = exp.run_relu(cfg, log_every=50)

    branch = exp.track_stable_true_field_branch(
        run,
        t_start=t_create,
        t_end=t_end,
        weak_floor_deg=60.0,
        birth_repeller_floor_deg=20.0,
    )

    phi_block = exp.branch_arrays_for_times(branch, [t_block])[0]
    blocks = exp.define_final_blocks_from_branch(
        run,
        phi_block_deg=phi_block,
        t_block=t_block,
        tol_deg=block_tol_deg,
    )

    times = run["time"]
    j_block = exp.nearest_index(times, t_block)

    U_block = run["U"][j_block]
    W_block = run["W"][j_block]
    M_block = exp.block_matrix(U_block, W_block, blocks["strong"])

    m11 = float(M_block[0, 0])
    m12 = float(M_block[0, 1])
    m21 = float(M_block[1, 0])
    m22 = float(M_block[1, 1])

    # Two operational block-read definitions plus the parameter-formula value.
    eta_cross = abs(m12) / max(abs(m11), 1e-12)
    eta_sqrt = math.sqrt(abs(m22) / max(abs(m11), 1e-12))
    eta_formula = cfg.mu2 * cfg.sigma1 / (cfg.mu1 ** 2)

    eta_defs = {
        "eta_cross": eta_cross,
        "eta_sqrt": eta_sqrt,
        "eta_formula": eta_formula,
    }

    print()
    print(f"M^(1)(t_block={t_block:.2f}) =")
    print(M_block)
    print()
    for name, eta in eta_defs.items():
        print(f"{name:12s} = {eta:.8f}; z(-85 deg) = {x[0] / eta:.4f}")

    with (outdir / "canonical_eta_measurements.txt").open("w") as f:
        f.write(f"t_block={t_block}\n")
        f.write(f"phi_block_deg={phi_block}\n")
        f.write(f"M11={m11}\nM12={m12}\nM21={m21}\nM22={m22}\n")
        for name, eta in eta_defs.items():
            f.write(f"{name}={eta}\n")
            f.write(f"z_{name}={x[0] / eta}\n")

    t_start = 1.5
    j0 = exp.nearest_index(times, t_start)
    j1 = exp.nearest_index(times, t_end)
    idx = np.arange(j0, j1 + 1)
    t = times[idx]

    nT = len(idx)
    M1_hist = np.zeros((nT, 2, 2), dtype=float)
    dM1_hist = np.zeros((nT, 2, 2), dtype=float)
    loss_full = np.zeros(nT, dtype=float)
    loss_red = np.zeros(nT, dtype=float)
    edot_full = np.zeros(nT, dtype=float)
    edot_red = np.zeros(nT, dtype=float)

    for k, j in enumerate(idx):
        U = run["U"][j]
        W = run["W"][j]
        dU, dW, _ = exp.relu_rhs(U, W, run["X"])

        M1 = exp.block_matrix(U, W, blocks["strong"])
        dM1 = exp.block_matrix_dot(U, W, dU, dW, blocks["strong"])
        M1_hist[k] = M1
        dM1_hist[k] = dM1

        # For psi=-85 deg, strong selector is on and weak selector is off
        # throughout the relevant canonical reversal window.
        f_red = M1 @ x
        e_red = f_red - x
        df_red = dM1 @ x
        loss_red[k] = 0.5 * np.dot(e_red, e_red)
        edot_red[k] = np.dot(e_red, df_red)

        f_full, df_full = exp.full_probe_dynamics(U, W, dU, dW, x)
        e_full = f_full - x
        loss_full[k] = 0.5 * np.dot(e_full, e_full)
        edot_full[k] = np.dot(e_full, df_full)

    full_best, _ = exp.persistent_swing(t, loss_full)
    red_best, _ = exp.persistent_swing(t, loss_red)

    print()
    print("Measured canonical reversal:")
    print(
        f"    full:    swing={full_best['swing']:.8e}, "
        f"t_reversal={full_best['t_reversal']:.4f}"
    )
    print(
        f"    reduced: swing={red_best['swing']:.8e}, "
        f"t_reversal={red_best['t_reversal']:.4f}"
    )

    data = {
        "time": t,
        "loss_full": loss_full,
        "loss_reduced": loss_red,
        "edot_full": edot_full,
        "edot_reduced": edot_red,
    }
    summaries = []

    M11 = M1_hist[:, 0, 0]
    M12 = M1_hist[:, 0, 1]
    M21 = M1_hist[:, 1, 0]
    M22 = M1_hist[:, 1, 1]
    dM11 = dM1_hist[:, 0, 0]
    dM12 = dM1_hist[:, 0, 1]
    dM21 = dM1_hist[:, 1, 0]
    dM22 = dM1_hist[:, 1, 1]

    for name, eta in eta_defs.items():
        z = x[0] / eta
        s = -1.0

        a = M11
        b = M12 / eta
        c = M21 / eta
        d = M22 / (eta ** 2)
        da = dM11
        db = dM12 / eta
        dc = dM21 / eta
        dd = dM22 / (eta ** 2)

        A = (a - 1.0) * z + s * b
        B = da * z + s * db
        G = A * B - s * z * dc - dd
        eta2G = (eta ** 2) * G

        t_cross = first_neg_to_pos_crossing(t, eta2G, t_lo=2.5, t_hi=5.0)

        tr = full_best["t_reversal"]
        before = (t >= tr - 0.20) & (t < tr)
        after = (t > tr) & (t <= tr + 0.20)
        mean_before = float(np.mean(eta2G[before])) if np.any(before) else np.nan
        mean_after = float(np.mean(eta2G[after])) if np.any(after) else np.nan

        corr = (
            float(np.corrcoef(eta2G, edot_red)[0, 1])
            if np.std(eta2G) > 0 and np.std(edot_red) > 0
            else np.nan
        )

        local = (t >= 2.5) & (t <= 5.0)
        mae = float(np.mean(np.abs(eta2G[local] - edot_red[local])))
        scale = float(np.mean(np.abs(edot_red[local])))
        rel_mae = mae / max(scale, 1e-12)

        summaries.append({
            "eta_definition": name,
            "eta": eta,
            "z_at_m85": z,
            "normal_form_zero_crossing": t_cross,
            "full_t_reversal": full_best["t_reversal"],
            "reduced_t_reversal": red_best["t_reversal"],
            "zero_cross_minus_full_reversal": (
                t_cross - full_best["t_reversal"]
                if np.isfinite(t_cross) and np.isfinite(full_best["t_reversal"])
                else np.nan
            ),
            "mean_eta2G_before_full_reversal": mean_before,
            "mean_eta2G_after_full_reversal": mean_after,
            "corr_eta2G_vs_exact_reduced_edot": corr,
            "local_relative_MAE_2p5_to_5": rel_mae,
        })

        data[f"a_{name}"] = a
        data[f"b_{name}"] = b
        data[f"c_{name}"] = c
        data[f"d_{name}"] = d
        data[f"G_{name}"] = G
        data[f"eta2G_{name}"] = eta2G
        data[f"abs_error_{name}"] = np.abs(eta2G - edot_red)

        print()
        print(f"{name}:")
        print(f"    eta={eta:.8f}, z={z:.4f}")
        print(f"    eta^2 G zero crossing = {t_cross:.4f}")
        if np.isfinite(t_cross):
            print(
                "    crossing - full reversal = "
                f"{t_cross - full_best['t_reversal']:+.4f}"
            )
        print(
            f"    mean eta^2 G before/after reversal = "
            f"{mean_before:+.6e} / {mean_after:+.6e}"
        )
        print(f"    corr(eta^2 G, exact reduced dE/dt) = {corr:.6f}")
        print(f"    relative MAE on [2.5,5] = {rel_mae:.4f}")

    fieldnames = list(data.keys())
    with (outdir / "canonical_G_check.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for k in range(len(t)):
            writer.writerow([data[name][k] for name in fieldnames])

    with (outdir / "canonical_G_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    fig = plt.figure(figsize=(9, 5.2))
    plt.plot(t, edot_full, label="full ReLU $\\dot E$")
    plt.plot(t, edot_red, label="exact reduced $\\dot E$")
    for name in eta_defs:
        plt.plot(t, data[f"eta2G_{name}"], label=f"$\\eta^2 G$ ({name})")
    plt.axhline(0.0, linestyle=":")
    if np.isfinite(full_best["t_reversal"]):
        plt.axvline(full_best["t_reversal"], linestyle="--", label="full $t_{rev}$")
    plt.xlabel("time")
    plt.ylabel("$\\dot E$")
    plt.title("Canonical near-boundary normal-form check: $\\psi=-85^\\circ$")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "canonical_G_edot.pdf")
    plt.close(fig)

    primary = "eta_cross"
    fig = plt.figure(figsize=(9, 5.2))
    plt.plot(t, data[f"a_{primary}"], label="$a$")
    plt.plot(t, data[f"b_{primary}"], label="$b$")
    plt.plot(t, data[f"c_{primary}"], label="$c$")
    plt.plot(t, data[f"d_{primary}"], label="$d$")
    plt.axvline(t_block, linestyle="--", label="$t_{block}$")
    plt.xlabel("time")
    plt.ylabel("scaled coefficient")
    plt.title("Scaled strong-block coefficients using block-read $\\eta_1$")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "canonical_G_scaled_coefficients.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5.2))
    for name in eta_defs:
        plt.plot(t, data[f"abs_error_{name}"], label=f"{name}")
    plt.xlabel("time")
    plt.ylabel("$|\\eta^2G-\\dot E_{red}|$")
    plt.title("Normal-form derivative approximation error")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "canonical_G_error.pdf")
    plt.close(fig)

    print()
    print("=" * 80)
    print("GRID GATE")
    print("=" * 80)
    print("Proceed to the 3x3 test if a predeclared eta gives:")
    print("  (i) eta^2 G < 0 before reversal,")
    print(" (ii) eta^2 G > 0 after reversal,")
    print("(iii) zero crossing near t≈3.67 without tuning eta.")
    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
