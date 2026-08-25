#!/usr/bin/env python3
"""
theorem_interface_checks.py

Theorem-interface diagnostics for the new unconditional s=2 ReLU theory.

CHECK A:
    Post-capture neuronwise fixed-gate audit.
    - Freeze each neuron's *own* training-sample gate vector at t_W.
    - Measure later gate flips, mass-weighted gate flips, and exact
      frozen-gate reconstruction error.
    - Inspect clusterwise majority-bit consistency.
    - Summarize how many mass-dominant frozen gate patterns exist.

CHECK B:
    Direct ReLU OOD swing-by audit.
    - Evaluate the actual two-layer ReLU model on ++,+-,-+,-- probes.
    - Track full OOD loss trajectories, coordinate outputs, and learned
      weak representative gate state.
    - Detect local minima followed by later increases over the *entire*
      trajectory, not only after t_W.
    - No linear baseline is load-bearing here.

Assumes `full_flow_diagnostic.py` is in the same directory and exports
Config plus the low-level model helpers used below.
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

try:
    from full_flow_diagnostic import Config
except ImportError:
    from full_flow_diagnostic import Config


# ---------------------------------------------------------------------
# Core model helpers
# ---------------------------------------------------------------------

def make_data(cfg: Config):
    """
    Same s=2 SIM sample used throughout Phase 0.
    """
    rng = np.random.default_rng(cfg.seed)

    n = cfg.n_per_cluster

    x1 = np.column_stack([
        cfg.mu1 + cfg.sigma1 * rng.standard_normal(n),
        cfg.sigma2 * rng.standard_normal(n),
    ])
    x2 = np.column_stack([
        cfg.sigma1 * rng.standard_normal(n),
        cfg.mu2 + cfg.sigma2 * rng.standard_normal(n),
    ])

    X = np.concatenate([x1, x2], axis=0)
    labels = np.concatenate([
        np.zeros(n, dtype=int),
        np.ones(n, dtype=int),
    ])
    return X, labels


def init_model(cfg: Config):
    """
    Aligned-balanced isotropic initialization in S=R^2:
        U_i(0) = W_i(0) = eps * (cos phi_i, sin phi_i).
    """
    rng = np.random.default_rng(cfg.seed)
    phi = rng.uniform(-np.pi, np.pi, size=cfg.width)
    U = cfg.epsilon * np.column_stack([np.cos(phi), np.sin(phi)])
    W = U.copy()
    return U, W, phi


def relu_forward(U, W, X):
    """
    U, W : (h,2)
    X    : (N,2)
    returns F : (N,2)
    """
    Z = X @ U.T
    A = np.maximum(Z, 0.0)
    return A @ W


def relu_rhs(U, W, X):
    """
    Exact empirical gradient-flow vector field for
        L = (1/(2N)) sum_n ||f(x_n)-x_n||^2.
    """
    N = X.shape[0]
    Z = X @ U.T                     # (N,h)
    G = (Z > 0.0).astype(float)     # (N,h)
    A = np.maximum(Z, 0.0)          # (N,h)
    F = A @ W                       # (N,2)
    R = F - X                       # (N,2)

    dW = -(A.T @ R) / N             # (h,2)

    # For each neuron i:
    # dot u_i = - E[ <R_n, w_i> 1{z_ni>0} x_n ]
    RW = R @ W.T                    # (N,h), entry <R_n,w_i>
    dU = -((G * RW).T @ X) / N      # (h,2)
    return dU, dW, F


def projected_mass_per_neuron(U, W):
    return np.sum(U * U, axis=1) + np.sum(W * W, axis=1)


def gate_matrix(U, X):
    return (X @ U.T > 0.0).astype(np.int8)   # (N,h)


def weak_representative(U, W, weak_seed_mask=None):
    """
    Mass-weighted representative among high-angle Q1 neurons.
    If weak_seed_mask is provided, prefer that cohort; otherwise use all.
    """
    angles = np.degrees(np.arctan2(U[:, 1], U[:, 0]))
    mass = projected_mass_per_neuron(U, W)

    candidate = (angles > 70.0) & (angles < 100.0)
    if weak_seed_mask is not None:
        seeded = candidate & weak_seed_mask
        if np.any(seeded):
            candidate = seeded

    if not np.any(candidate):
        # fallback: use neuron closest to +e2
        idx = np.argmin(np.abs(angles - 90.0))
        vec = U[idx].copy()
    else:
        weights = mass[candidate]
        vec = np.sum(weights[:, None] * U[candidate], axis=0)

    norm = np.linalg.norm(vec)
    if norm <= 0:
        return np.array([0.0, 1.0])
    return vec / norm


# ---------------------------------------------------------------------
# Time stepping and checkpoint storage
# ---------------------------------------------------------------------

def run_relu(cfg: Config, log_every=50):
    X, labels = make_data(cfg)
    U, W, init_phi = init_model(cfg)

    dt = cfg.dt
    steps = int(round(cfg.max_time / dt))

    times = []
    U_hist = []
    W_hist = []
    F_hist = []

    for step in range(steps + 1):
        t = step * dt

        if step % log_every == 0:
            F = relu_forward(U, W, X)
            times.append(t)
            U_hist.append(U.copy())
            W_hist.append(W.copy())
            F_hist.append(F.copy())

        if step == steps:
            break

        dU, dW, _ = relu_rhs(U, W, X)
        U = U + dt * dU
        W = W + dt * dW

    return {
        "X": X,
        "labels": labels,
        "init_phi": init_phi,
        "time": np.asarray(times),
        "U": U_hist,
        "W": W_hist,
        "F": F_hist,
    }


def nearest_index(times, target):
    return int(np.argmin(np.abs(times - target)))


# ---------------------------------------------------------------------
# CHECK A
# ---------------------------------------------------------------------

def gate_flip_fraction(g_ref, g_now):
    return np.mean(g_ref != g_now, axis=0)


def frozen_gate_reconstruction(U, W, X, g_ref):
    Z = X @ U.T
    A_frozen = g_ref * Z
    return A_frozen @ W


def relative_rms(A, B):
    num = np.sqrt(np.mean((A - B) ** 2))
    den = np.sqrt(np.mean(A ** 2))
    return float(num / max(den, 1e-12))


def cluster_majority_errors(g_ref, labels):
    h = g_ref.shape[1]
    majority = np.zeros((2, h), dtype=np.int8)
    q = np.zeros((2, h), dtype=float)
    for p in (0, 1):
        gp = g_ref[labels == p]
        mean_on = gp.mean(axis=0)
        majority[p] = (mean_on >= 0.5).astype(np.int8)
        q[p] = np.mean(gp != majority[p][None, :], axis=0)
    return majority, q


def circular_angle_deg(U):
    return np.degrees(np.arctan2(U[:, 1], U[:, 0]))


def angle_family_summary(U, W, X, angle_tol_deg=5.0, boundary_quantile=0.05):
    """
    Group neurons by induced hyperplane angle rather than exact samplewise
    gate vector.  We bin angles at resolution `angle_tol_deg`, then for each
    mass-relevant group verify within-group gate agreement away from a
    boundary strip around the mass-weighted representative hyperplane.
    """
    angles = circular_angle_deg(U)
    mass = projected_mass_per_neuron(U, W)
    total_mass = float(np.sum(mass))
    gates = gate_matrix(U, X)

    # Hyperplanes are pi-periodic as gates up to complement, but orientation
    # matters because ReLU uses the positive half-space. Keep signed angle.
    keys = np.floor((angles + 180.0) / angle_tol_deg).astype(int)
    rows = []

    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        group_mass = float(np.sum(mass[idx]))
        if group_mass <= 0:
            continue

        # Mass-weighted representative direction.
        rep = np.sum(mass[idx, None] * U[idx], axis=0)
        nr = np.linalg.norm(rep)
        if nr <= 0:
            continue
        rep = rep / nr
        rep_angle = float(np.degrees(np.arctan2(rep[1], rep[0])))

        scores = np.abs(X @ rep)
        tau = float(np.quantile(scores, boundary_quantile))
        off = scores > tau

        max_pair_dis = 0.0
        mean_pair_dis = 0.0
        npairs = 0
        if len(idx) >= 2 and np.any(off):
            goff = gates[off][:, idx]
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    d = float(np.mean(goff[:, a] != goff[:, b]))
                    max_pair_dis = max(max_pair_dis, d)
                    mean_pair_dis += d
                    npairs += 1
        if npairs > 0:
            mean_pair_dis /= npairs

        rows.append({
            "angle_bin": int(key),
            "representative_angle_deg": rep_angle,
            "count": int(len(idx)),
            "mass_fraction": group_mass / max(total_mass, 1e-12),
            "boundary_tau": tau,
            "max_pairwise_gate_disagreement_off_boundary": max_pair_dis,
            "mean_pairwise_gate_disagreement_off_boundary": mean_pair_dis,
        })

    rows.sort(key=lambda r: r["mass_fraction"], reverse=True)
    cum = 0.0
    for rank, r in enumerate(rows, start=1):
        cum += r["mass_fraction"]
        r["rank"] = rank
        r["cumulative_mass_fraction"] = cum
    return rows


def weak_seed_mask_from_init(init_phi):
    deg = np.degrees(init_phi)
    return (deg >= 110.0) & (deg <= 170.0)


def weak_alignment_completion_time(run, tolerance_deg=10.0, required_mass_fraction=0.95):
    """
    Practical post-capture stabilization time:
    first logged time at which at least `required_mass_fraction` of W-seed
    projected mass lies within `tolerance_deg` of the mass-weighted W-seed
    representative, with that representative in the high-angle Q1 region.

    This avoids forcing never-captured / near-dead seed neurons to define t_W.
    """
    mask = weak_seed_mask_from_init(run["init_phi"])
    times = run["time"]

    for j, t in enumerate(times):
        U = run["U"][j]
        W = run["W"][j]
        angles = circular_angle_deg(U)
        mass = projected_mass_per_neuron(U, W)[mask]
        if np.sum(mass) <= 0:
            continue

        # Use W-seed neurons already in Q1/high-angle region.
        wm_angles = angles[mask]
        eligible = (wm_angles > 70.0) & (wm_angles < 90.5)
        if not np.any(eligible):
            continue

        Uwm = U[mask][eligible]
        mwm = mass[eligible]
        rep = np.sum(mwm[:, None] * Uwm, axis=0)
        nr = np.linalg.norm(rep)
        if nr <= 0:
            continue
        rep_angle = float(np.degrees(np.arctan2(rep[1], rep[0])))
        if not (75.0 <= rep_angle <= 95.0):
            continue

        # Circular difference in degrees.
        diff = np.abs(((wm_angles - rep_angle + 180.0) % 360.0) - 180.0)
        aligned_mass = float(np.sum(mass[diff <= tolerance_deg]))
        frac = aligned_mass / float(np.sum(mass))
        if frac >= required_mass_fraction:
            return float(t), rep_angle, frac

    return None, np.nan, np.nan


def audit_one_reference_time(run, t_ref, t_3, outdir, label):
    times = run["time"]
    X = run["X"]
    labels = run["labels"]

    jW = nearest_index(times, t_ref)
    j3 = nearest_index(times, t_3)

    U_W = run["U"][jW]
    W_W = run["W"][jW]
    g_ref = gate_matrix(U_W, X)
    mass_W = projected_mass_per_neuron(U_W, W_W)

    majority, q_cluster = cluster_majority_errors(g_ref, labels)
    angle_groups = angle_family_summary(
        U_W, W_W, X,
        angle_tol_deg=5.0,
        boundary_quantile=0.05,
    )

    rows = []
    for j in range(jW, j3 + 1):
        t = float(times[j])
        U = run["U"][j]
        W = run["W"][j]
        F = run["F"][j]

        g_now = gate_matrix(U, X)
        flip_i = gate_flip_fraction(g_ref, g_now)
        mass = projected_mass_per_neuron(U, W)

        F_frozen = frozen_gate_reconstruction(U, W, X, g_ref)
        err_all = relative_rms(F, F_frozen)
        err_c1 = relative_rms(F[labels == 0], F_frozen[labels == 0])
        err_c2 = relative_rms(F[labels == 1], F_frozen[labels == 1])

        rows.append({
            "time": t,
            "fraction_neurons_with_any_gate_flip": float(np.mean(flip_i > 0.0)),
            "mean_neuron_gate_flip_fraction": float(np.mean(flip_i)),
            "mass_weighted_gate_flip_fraction": float(
                np.sum(mass * flip_i) / max(np.sum(mass), 1e-12)
            ),
            "frozen_gate_relative_rms_error": err_all,
            "frozen_gate_relative_rms_error_cluster1": err_c1,
            "frozen_gate_relative_rms_error_cluster2": err_c2,
        })

    total_mass = float(np.sum(mass_W))
    q1_mass = float(np.sum(mass_W * q_cluster[0]) / max(total_mass, 1e-12))
    q2_mass = float(np.sum(mass_W * q_cluster[1]) / max(total_mass, 1e-12))

    print()
    print(f"Reference gate time {label}: t_ref={times[jW]:.6f}")
    print(
        "    max fraction neurons with any gate flip = "
        f"{max(r['fraction_neurons_with_any_gate_flip'] for r in rows):.6f}"
    )
    print(
        "    max mean neuron gate-flip fraction = "
        f"{max(r['mean_neuron_gate_flip_fraction'] for r in rows):.6e}"
    )
    print(
        "    max mass-weighted gate-flip fraction = "
        f"{max(r['mass_weighted_gate_flip_fraction'] for r in rows):.6e}"
    )
    print(
        "    max frozen-gate relative RMS error (all) = "
        f"{max(r['frozen_gate_relative_rms_error'] for r in rows):.6e}"
    )
    print(
        "    max frozen-gate relative RMS error (cluster 1) = "
        f"{max(r['frozen_gate_relative_rms_error_cluster1'] for r in rows):.6e}"
    )
    print(
        "    max frozen-gate relative RMS error (cluster 2) = "
        f"{max(r['frozen_gate_relative_rms_error_cluster2'] for r in rows):.6e}"
    )
    print(
        "    mass-weighted cluster-majority disagreement C1/C2 = "
        f"{q1_mass:.6e} / {q2_mass:.6e}"
    )

    print("    mass-dominant angle/gate families:")
    for r in angle_groups[:10]:
        print(
            f"      rank {r['rank']:2d}: angle={r['representative_angle_deg']:+7.2f} deg, "
            f"count={r['count']:3d}, mass={r['mass_fraction']:.4f}, "
            f"cum={r['cumulative_mass_fraction']:.4f}, "
            f"max gate-dis(off B)={r['max_pairwise_gate_disagreement_off_boundary']:.4f}"
        )

    csv_path = outdir / f"checkA_timeseries_{label}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fam_path = outdir / f"checkA_angle_families_{label}.csv"
    with fam_path.open("w", newline="") as f:
        fieldnames = [
            "rank", "angle_bin", "representative_angle_deg", "count",
            "mass_fraction", "cumulative_mass_fraction", "boundary_tau",
            "max_pairwise_gate_disagreement_off_boundary",
            "mean_pairwise_gate_disagreement_off_boundary",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in angle_groups:
            writer.writerow({k: r[k] for k in fieldnames})

    t = np.array([r["time"] for r in rows])

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(t, [r["mass_weighted_gate_flip_fraction"] for r in rows],
             label="mass-weighted gate flips")
    plt.plot(t, [r["mean_neuron_gate_flip_fraction"] for r in rows],
             label="mean neuron gate flips")
    plt.xlabel("time")
    plt.ylabel("fraction")
    plt.title(f"Check A: gate drift after {label}")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"checkA_gate_drift_{label}.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(t, [r["frozen_gate_relative_rms_error"] for r in rows], label="all")
    plt.plot(t, [r["frozen_gate_relative_rms_error_cluster1"] for r in rows],
             label="cluster 1")
    plt.plot(t, [r["frozen_gate_relative_rms_error_cluster2"] for r in rows],
             label="cluster 2")
    plt.xlabel("time")
    plt.ylabel("relative RMS error")
    plt.title(f"Check A: frozen-gate reconstruction after {label}")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"checkA_frozen_reconstruction_{label}.pdf")
    plt.close(fig)

    return rows, angle_groups


def check_A(run, candidate_times, t_3, outdir):
    print()
    print("=" * 76)
    print("CHECK A: NEW-THEORY POST-CAPTURE FIXED-GATE AUDIT")
    print("=" * 76)

    auto_t, auto_angle, auto_frac = weak_alignment_completion_time(
        run, tolerance_deg=10.0, required_mass_fraction=0.95
    )
    if auto_t is not None:
        print(
            f"automatic W-seed stabilization candidate: t={auto_t:.3f}, "
            f"rep angle={auto_angle:.3f} deg, aligned W-mass fraction={auto_frac:.4f}"
        )
    else:
        print("automatic W-seed stabilization candidate: not found")

    refs = list(candidate_times)
    if auto_t is not None and all(abs(auto_t - x) > 0.05 for x in refs):
        refs.append(auto_t)
    refs = sorted(set(round(float(x), 6) for x in refs if x < t_3))

    results = {}
    for k, tref in enumerate(refs):
        label = f"tref_{tref:.2f}".replace(".", "p")
        results[tref] = audit_one_reference_time(
            run, tref, t_3, outdir, label
        )
    return results

# ---------------------------------------------------------------------
# CHECK B
# ---------------------------------------------------------------------

def relu_probe(U, W, x):
    z = U @ x
    a = np.maximum(z, 0.0)
    return a @ W


def moving_average(y, window):
    window = max(1, int(window))
    if window == 1:
        return np.asarray(y, dtype=float)
    kernel = np.ones(window) / window
    # Edge-pad so output length matches input.
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    yp = np.pad(np.asarray(y, dtype=float), (pad_left, pad_right), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


def local_minima(y):
    out = []
    for i in range(1, len(y) - 1):
        if y[i] <= y[i - 1] and y[i] < y[i + 1]:
            out.append(i)
    return out


def persistent_swing(times, y, smooth_window=11, persistence_time=0.20, hold_fraction=0.80):
    """
    Search the smoothed trajectory for the largest local-minimum -> later
    elevation that persists for a full window.

    A candidate rebound at j with size Delta is certified if for every logged
    checkpoint in [t_j, t_j+persistence_time],
        y_smooth >= y_min + hold_fraction * Delta.
    """
    ys = moving_average(y, smooth_window)
    mins = local_minima(ys)
    dt_log = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    hold_steps = max(1, int(round(persistence_time / dt_log)))

    best = {
        "swing": 0.0,
        "t_min": np.nan,
        "loss_min": np.nan,
        "t_rebound": np.nan,
        "loss_rebound": np.nan,
        "normalized_swing": 0.0,
    }

    for i in mins:
        for j in range(i + 1, len(ys)):
            delta = float(ys[j] - ys[i])
            if delta <= best["swing"]:
                continue

            end = min(len(ys), j + hold_steps + 1)
            threshold = ys[i] + hold_fraction * delta
            if np.all(ys[j:end] >= threshold):
                best.update({
                    "swing": delta,
                    "t_min": float(times[i]),
                    "loss_min": float(ys[i]),
                    "t_rebound": float(times[j]),
                    "loss_rebound": float(ys[j]),
                })

    return best, ys


def weak_rep_after_creation(U, W, weak_seed_mask, min_candidate_mass_fraction=0.10):
    angles = circular_angle_deg(U)
    mass_all = projected_mass_per_neuron(U, W)
    mass = mass_all[weak_seed_mask]
    U_w = U[weak_seed_mask]
    ang_w = angles[weak_seed_mask]

    total = float(np.sum(mass))
    if total <= 0:
        return None

    candidate = (ang_w > 70.0) & (ang_w < 100.0)
    if not np.any(candidate):
        return None

    cand_mass = float(np.sum(mass[candidate]))
    if cand_mass / total < min_candidate_mass_fraction:
        return None

    rep = np.sum(mass[candidate, None] * U_w[candidate], axis=0)
    nr = np.linalg.norm(rep)
    if nr <= 0:
        return None
    return rep / nr


def check_B(run, t_create, t_ref_for_plot, t_3, outdir):
    times = run["time"]
    j3 = nearest_index(times, t_3)
    jref = nearest_index(times, t_ref_for_plot)

    X = run["X"]
    labels = run["labels"]
    mu1_hat = float(np.mean(X[labels == 0, 0]))
    mu2_hat = float(np.mean(X[labels == 1, 1]))

    probes = {
        "pp": np.array([mu1_hat, mu2_hat]),
        "pm": np.array([mu1_hat, -mu2_hat]),
        "mp": np.array([-mu1_hat, mu2_hat]),
        "mm": np.array([-mu1_hat, -mu2_hat]),
    }

    weak_mask = weak_seed_mask_from_init(run["init_phi"])
    rows = []
    summaries = []

    print()
    print("=" * 76)
    print("CHECK B: NEW-THEORY DIRECT ReLU OOD SWING-BY AUDIT")
    print("=" * 76)

    for name, xstar in probes.items():
        loss = []
        out1 = []
        out2 = []
        weak_gate = []
        weak_angle = []

        for j in range(0, j3 + 1):
            U = run["U"][j]
            W = run["W"][j]
            fstar = relu_probe(U, W, xstar)
            err = 0.5 * float(np.sum((fstar - xstar) ** 2))

            loss.append(err)
            out1.append(float(fstar[0]))
            out2.append(float(fstar[1]))

            if times[j] < t_create:
                weak_gate.append(np.nan)
                weak_angle.append(np.nan)
            else:
                rep = weak_rep_after_creation(U, W, weak_mask)
                if rep is None:
                    weak_gate.append(np.nan)
                    weak_angle.append(np.nan)
                else:
                    weak_gate.append(float(np.dot(rep, xstar) > 0.0))
                    weak_angle.append(
                        float(np.degrees(np.arctan2(rep[1], rep[0])))
                    )

            rows.append({
                "probe": name,
                "time": float(times[j]),
                "x1": float(xstar[0]),
                "x2": float(xstar[1]),
                "loss": err,
                "f1": float(fstar[0]),
                "f2": float(fstar[1]),
                "weak_gate_evolving_after_creation": weak_gate[-1],
                "weak_rep_angle_deg": weak_angle[-1],
            })

        loss = np.asarray(loss)
        t = times[:j3 + 1]
        best, loss_smooth = persistent_swing(
            t, loss,
            smooth_window=11,
            persistence_time=0.20,
            hold_fraction=0.80,
        )
        best["normalized_swing"] = (
            best["swing"] / max(float(np.dot(xstar, xstar)), 1e-12)
        )

        valid_gate = np.asarray(weak_gate, dtype=float)
        valid = np.isfinite(valid_gate)
        gate_switches = 0
        if np.sum(valid) >= 2:
            seq = valid_gate[valid].astype(int)
            gate_switches = int(np.sum(seq[1:] != seq[:-1]))

        summary = {
            "probe": name,
            "persistent_swing": best["swing"],
            "persistent_swing_over_xnorm2": best["normalized_swing"],
            "t_local_min": best["t_min"],
            "loss_local_min": best["loss_min"],
            "t_persistent_rebound": best["t_rebound"],
            "loss_persistent_rebound": best["loss_rebound"],
            "initial_loss": float(loss[0]),
            "loss_at_reference_t": float(loss[jref]),
            "loss_at_t3": float(loss[j3]),
            "post_creation_weak_gate_switches": gate_switches,
        }
        summaries.append(summary)

        print()
        print(f"Probe {name}: x*={xstar}")
        print(
            "    persistent local-min -> later rebound = "
            f"{summary['persistent_swing']:+.6e}"
        )
        print(
            "    normalized rebound Delta/||x*||^2 = "
            f"{summary['persistent_swing_over_xnorm2']:+.6e}"
        )
        if np.isfinite(summary["t_local_min"]):
            print(
                f"    min at t={summary['t_local_min']:.3f}, "
                f"L={summary['loss_local_min']:.6e}; "
                f"persistent rebound at t={summary['t_persistent_rebound']:.3f}, "
                f"L={summary['loss_persistent_rebound']:.6e}"
            )
        print(
            f"    L(0)={summary['initial_loss']:.6e}, "
            f"L(t_ref={times[jref]:.2f})={summary['loss_at_reference_t']:.6e}, "
            f"L(t_3)={summary['loss_at_t3']:.6e}"
        )
        print(
            f"    weak-representative gate switches after creation/availability = "
            f"{summary['post_creation_weak_gate_switches']}"
        )

        fig = plt.figure(figsize=(7, 4.5))
        plt.plot(t, loss, label="raw")
        plt.plot(t, loss_smooth, label="smoothed")
        plt.axvline(t_create, linestyle=":", label="$t_{create}$")
        plt.axvline(times[jref], linestyle="--", label="reference $t_W$")
        if np.isfinite(best["t_min"]):
            plt.scatter(
                [best["t_min"], best["t_rebound"]],
                [best["loss_min"], best["loss_rebound"]],
            )
        plt.xlabel("time")
        plt.ylabel(r"$\frac{1}{2}\|f_t(x^\star)-x^\star\|^2$")
        plt.title(f"Check B: ReLU OOD loss ({name})")
        plt.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"checkB_ood_loss_{name}.pdf")
        plt.close(fig)

        fig = plt.figure(figsize=(7, 4.5))
        plt.plot(t, out1, label="$f_1(x^\\star)$")
        plt.plot(t, out2, label="$f_2(x^\\star)$")
        plt.axhline(xstar[0], linestyle=":", label="$x_1^\\star$")
        plt.axhline(xstar[1], linestyle=":", label="$x_2^\\star$")
        plt.axvline(t_create, linestyle=":")
        plt.axvline(times[jref], linestyle="--")
        plt.xlabel("time")
        plt.ylabel("coordinate")
        plt.title(f"Check B: ReLU OOD outputs ({name})")
        plt.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"checkB_ood_outputs_{name}.pdf")
        plt.close(fig)

    csv_path = outdir / "checkB_ood_timeseries.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sum_path = outdir / "checkB_summary.csv"
    with sum_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    return summaries

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    cfg = replace(
        Config(),
        mu1=3.0,
        mu2=2.0,
        sigma1=0.15,
        sigma2=0.15,
        max_time=10.0,
    )

    # Phase-0 true-field basin creation for the canonical run.
    t_create = 1.06411

    # Sensitivity audit. 2.15 is retained only to show why median-capture
    # freezing is too early; later times test the stabilized regime.
    candidate_gate_times = [2.15, 3.0, 3.5, 4.0]

    # Use 3.5 as the reference marker in OOD plots; Check A also computes
    # an automatic W-seed stabilization candidate.
    t_ref_for_plot = 3.5
    t_3 = 10.0

    outdir = Path("theorem_interface_checks_v3")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("NEW-THEORY THEOREM INTERFACE CHECKS v3")
    print("=" * 76)
    print(
        f"canonical config: mu1={cfg.mu1}, mu2={cfg.mu2}, "
        f"sigma=({cfg.sigma1},{cfg.sigma2}), width={cfg.width}, "
        f"n/cluster={cfg.n_per_cluster}, epsilon={cfg.epsilon}, dt={cfg.dt}"
    )
    print(f"t_create={t_create}, candidate gate times={candidate_gate_times}, t_3={t_3}")
    print("model: two-layer ReLU only")

    run = run_relu(cfg, log_every=50)

    check_A(
        run,
        candidate_times=candidate_gate_times,
        t_3=t_3,
        outdir=outdir,
    )

    check_B(
        run,
        t_create=t_create,
        t_ref_for_plot=t_ref_for_plot,
        t_3=t_3,
        outdir=outdir,
    )

    print()
    print("=" * 76)
    print(f"Saved outputs to: {outdir.resolve()}")
    print("=" * 76)


if __name__ == "__main__":
    main()
