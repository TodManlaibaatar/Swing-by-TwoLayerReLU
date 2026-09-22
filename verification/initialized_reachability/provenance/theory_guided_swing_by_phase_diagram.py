#!/usr/bin/env python3
"""
theory_guided_swing_phase_diagram_v3.py

Clean paper-facing canonical experiment for the new unconditional s=2 ReLU theory.

This version fixes the branch-tracking error in v1.

KEY CORRECTION
--------------
The learned weak direction is NOT obtained by taking the eventual weak block at
t_block and averaging those neurons backward in time.  That mixes in neurons
that have not yet been recruited.

Instead, this script defines the theorem-facing weak branch phi_W(t) as the
stable Q1 root of the exact full-network angular test field

    V_true(phi;t)
      = - E_n[
            <f_t(x_n)-x_n, u(phi)>
            1{u(phi)^T x_n > 0}
            u_perp(phi)^T x_n
          ],

with
    u(phi)      = (cos phi, sin phi),
    u_perp(phi) = (-sin phi, cos phi).

The branch is born when the stable Q1 root first appears and is then tracked by
continuity.  Frozen final block membership is defined ONCE at t_block using the
tracked strong/weak directions and never changes afterward.

The script then does two things.

A. Corrected theory-guided OOD direction phase diagram
   ---------------------------------------------------
   For x*(psi)=R(cos psi,sin psi), positive homogeneity gives
       E_t(Rx)=R^2 E_t(x),
   so S(psi)=Delta_swing/R^2 is a complete directional OOD phase diagram.

   We compare:
       measured full-ReLU S_actual(psi)
   to
       reduced-model S_pred(psi)
   from
       f_red = d1 M^(1)(t)x + d2 M^(2)(t)x,

   where
       d1 = 1{x_1>0},
       d2 = 1{u(phi_W(t))^T x > 0}.

   The weak selector-switch sector and switch times therefore come from the
   tracked V_true branch, not from an eventual-block average.

B. Final canonical block-matrix mechanism decomposition
   ----------------------------------------------------
   For psi in {-85 deg, 176 deg, 33.7 deg}, log

       M^(1)(t), M^(2)(t),
       dot M^(1)(t), dot M^(2)(t),
       full and reduced OOD loss,
       full and reduced dE/dt,

   and the exact continuous-selector contribution

       C_{p,r,l}(t)
         = d_p e_r dot m_{r,l}^{(p)} x_l

   so that away from selector-switch times

       dot E_red
         = sum_{p,r,l} C_{p,r,l}.

   Since block membership is frozen at t_block,

       dot M = dot W_block^T U_block + W_block^T dot U_block

   has NO membership-derivative term.

Primary outputs
---------------
true_field_weak_branch.csv
true_field_weak_branch.pdf
postblock_weak_spread.csv
block_sensitivity.csv

phase_diagram.csv
phase_diagram_S_corrected.pdf
phase_diagram_positive_switch_wedge_corrected.pdf
switch_time_vs_direction_corrected.pdf
reversal_vs_switch_time_corrected.pdf

decomp_m85.csv / decomp_p176.csv / decomp_pp.csv
decomp_*_loss.pdf
decomp_*_edot.pdf
decomp_*_terms.pdf
decomposition_summary.csv

Run beside full_flow_diagnostic.py.
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


# =============================================================================
# Canonical s=2 ReLU simulation
# =============================================================================

def make_data(cfg: Config):
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
    rng = np.random.default_rng(cfg.seed)
    phi = rng.uniform(-np.pi, np.pi, size=cfg.width)
    U = cfg.epsilon * np.column_stack([np.cos(phi), np.sin(phi)])
    W = U.copy()
    return U, W, phi


def relu_forward(U, W, X):
    Z = X @ U.T
    A = np.maximum(Z, 0.0)
    return A @ W


def relu_rhs(U, W, X):
    """
    Exact empirical gradient-flow vector field for
        L = (1/(2N)) sum_n ||f(x_n)-x_n||^2.
    """
    N = X.shape[0]
    Z = X @ U.T
    G = (Z > 0.0).astype(float)
    A = np.maximum(Z, 0.0)
    F = A @ W
    R = F - X

    dW = -(A.T @ R) / N
    RW = R @ W.T
    dU = -((G * RW).T @ X) / N
    return dU, dW, F


def run_relu(cfg: Config, log_every=50):
    X, labels = make_data(cfg)
    U, W, init_phi = init_model(cfg)

    steps = int(round(cfg.max_time / cfg.dt))
    times, U_hist, W_hist = [], [], []

    for step in range(steps + 1):
        if step % log_every == 0:
            times.append(step * cfg.dt)
            U_hist.append(U.copy())
            W_hist.append(W.copy())

        if step == steps:
            break

        dU, dW, _ = relu_rhs(U, W, X)
        U = U + cfg.dt * dU
        W = W + cfg.dt * dW

    return {
        "X": X,
        "labels": labels,
        "init_phi": init_phi,
        "time": np.asarray(times, dtype=float),
        "U": U_hist,
        "W": W_hist,
    }


def nearest_index(times, target):
    return int(np.argmin(np.abs(np.asarray(times) - target)))


# =============================================================================
# Geometry helpers
# =============================================================================

def wrap_deg(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def angle_deg(U):
    return np.degrees(np.arctan2(U[:, 1], U[:, 0]))


def angular_distance_deg(a, b):
    return np.abs(wrap_deg(np.asarray(a) - b))


def projected_mass_per_neuron(U, W):
    return np.sum(U * U, axis=1) + np.sum(W * W, axis=1)


def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0 or np.sum(weights) <= 0:
        return np.nan

    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w)
    target = q * c[-1]
    return float(v[np.searchsorted(c, target, side="left")])


# =============================================================================
# Exact full-network V_true(phi;t) and stable-root branch tracking
# =============================================================================

def unit_direction_from_deg(phi_deg):
    phi = math.radians(float(phi_deg))
    return np.array([math.cos(phi), math.sin(phi)], dtype=float)


def v_true_scalar(phi_deg, X, residual):
    """
    V_true(phi;t)
      = - E[
          <residual,u(phi)>
          1{u(phi)^T x > 0}
          u_perp(phi)^T x
        ].
    """
    phi = math.radians(float(phi_deg))
    c, s = math.cos(phi), math.sin(phi)
    u = np.array([c, s], dtype=float)
    u_perp = np.array([-s, c], dtype=float)

    gate = (X @ u > 0.0)
    return -float(np.mean((residual @ u) * gate * (X @ u_perp)))


def v_true_grid(phi_deg_grid, X, residual):
    """
    Vectorized V_true on a grid of angles. Used only for root discovery /
    recovery, not every time step.
    """
    phi = np.radians(np.asarray(phi_deg_grid, dtype=float))
    Uphi = np.column_stack([np.cos(phi), np.sin(phi)])
    Uperp = np.column_stack([-np.sin(phi), np.cos(phi)])

    pre = X @ Uphi.T
    rdot = residual @ Uphi.T
    xperp = X @ Uperp.T
    return -np.mean(rdot * (pre > 0.0) * xperp, axis=0)


def bisect_v_true_root(a, b, X, residual, max_iter=40):
    fa = v_true_scalar(a, X, residual)
    fb = v_true_scalar(b, X, residual)

    if abs(fa) < 1e-12:
        return float(a)
    if abs(fb) < 1e-12:
        return float(b)
    if fa * fb > 0:
        return None

    lo, hi = float(a), float(b)
    flo, fhi = fa, fb

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = v_true_scalar(mid, X, residual)

        if abs(fm) < 1e-10 or abs(hi - lo) < 1e-6:
            return float(mid)

        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm

    return float(0.5 * (lo + hi))


def v_true_derivative(phi_deg, X, residual, h_deg=0.05):
    vp = v_true_scalar(phi_deg + h_deg, X, residual)
    vm = v_true_scalar(phi_deg - h_deg, X, residual)
    return float((vp - vm) / (2.0 * math.radians(h_deg)))


def stable_q1_roots(X, residual, phi_lo=0.25, phi_hi=89.95, grid_step=0.25):
    """
    Return all Q1 roots with dV_true/dphi < 0, i.e. locally stable roots for
    angular flow with sign convention dot phi proportional to V_true.
    """
    grid = np.arange(phi_lo, phi_hi + 0.5 * grid_step, grid_step)
    vals = v_true_grid(grid, X, residual)

    roots = []
    for k in range(len(grid) - 1):
        a, b = grid[k], grid[k + 1]
        fa, fb = vals[k], vals[k + 1]

        if fa == 0.0:
            root = float(a)
        elif fa * fb < 0.0:
            root = bisect_v_true_root(a, b, X, residual)
        else:
            continue

        if root is None:
            continue

        deriv = v_true_derivative(root, X, residual)
        if deriv < 0.0:
            if not roots or abs(root - roots[-1]["phi_deg"]) > 1e-3:
                roots.append({
                    "phi_deg": float(root),
                    "dV_dphi": float(deriv),
                })

    return roots



def all_q1_roots(X, residual, phi_lo=0.25, phi_hi=89.95, grid_step=0.25):
    """
    Return every Q1 root of V_true with its local stability classification.
    With dot(phi) proportional to V_true, dV/dphi < 0 is stable and
    dV/dphi > 0 is repelling.
    """
    grid = np.arange(phi_lo, phi_hi + 0.5 * grid_step, grid_step)
    vals = v_true_grid(grid, X, residual)

    roots = []
    for k in range(len(grid) - 1):
        a, b = grid[k], grid[k + 1]
        fa, fb = vals[k], vals[k + 1]

        root = None
        if abs(fa) < 1e-12:
            root = float(a)
        elif fa * fb < 0.0:
            root = bisect_v_true_root(a, b, X, residual)

        if root is None:
            continue

        deriv = v_true_derivative(root, X, residual)
        kind = "stable" if deriv < 0.0 else "repeller"

        if not roots or abs(root - roots[-1]["phi_deg"]) > 1e-3:
            roots.append({
                "phi_deg": float(root),
                "dV_dphi": float(deriv),
                "kind": kind,
            })

    return roots


def root_census(run, query_times, outdir):
    """
    Paper/debug census of all Q1 roots at selected canonical times.
    This makes branch identity explicit and prevents accidentally following
    the strong near-zero root.
    """
    rows = []
    X = run["X"]
    times = run["time"]

    print()
    print("Q1 V_true root census:")
    for tq in query_times:
        j = nearest_index(times, tq)
        U = run["U"][j]
        W = run["W"][j]
        residual = relu_forward(U, W, X) - X
        roots = all_q1_roots(X, residual, grid_step=0.10)

        desc = []
        for rank, r in enumerate(roots, start=1):
            rows.append({
                "requested_time": float(tq),
                "time": float(times[j]),
                "root_rank": rank,
                "phi_deg": r["phi_deg"],
                "dV_dphi": r["dV_dphi"],
                "kind": r["kind"],
            })
            desc.append(f"{r['phi_deg']:.3f}({r['kind']})")

        print(f"    t={times[j]:.3f}: " + (", ".join(desc) if desc else "no Q1 roots"))

    if rows:
        with (outdir / "true_field_root_census.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows


def track_stable_true_field_branch(
    run,
    t_start,
    t_end,
    weak_floor_deg=60.0,
    birth_repeller_floor_deg=20.0,
    discovery_step_deg=0.15,
    local_halfwidth_deg=5.0,
    local_step_deg=0.08,
):
    """
    Track the *weak* stable Q1 branch of V_true.

    Crucial branch-identity rule:
      - do NOT start from the strong stable root near 0 degrees;
      - declare weak-branch birth only when there is a high-angle stable root
        phi_s >= weak_floor_deg together with a lower repelling root, i.e. the
        saddle-node basin pair found in Phase 0;
      - after birth, track that high-angle stable root continuously.

    This fixes the v2 failure where the strong root near 0 degrees was
    mistakenly used as the first point of the weak branch.
    """
    times = run["time"]
    X = run["X"]
    j0 = nearest_index(times, t_start)
    j1 = nearest_index(times, t_end)

    rows = []
    prev_phi = None
    born = False

    for j in range(j0, j1 + 1):
        t = float(times[j])
        U = run["U"][j]
        W = run["W"][j]
        residual = relu_forward(U, W, X) - X

        chosen = None
        lower_repeller = np.nan
        root_count = 0

        if not born:
            roots = all_q1_roots(
                X, residual,
                grid_step=discovery_step_deg,
            )
            root_count = len(roots)

            stable_hi = [
                r for r in roots
                if r["kind"] == "stable" and r["phi_deg"] >= weak_floor_deg
            ]

            candidates = []
            for sroot in stable_hi:
                reps = [
                    r for r in roots
                    if r["kind"] == "repeller"
                    and birth_repeller_floor_deg <= r["phi_deg"] < sroot["phi_deg"]
                ]
                if reps:
                    nearest_rep = max(reps, key=lambda r: r["phi_deg"])
                    candidates.append((sroot, nearest_rep))

            if candidates:
                # The weak basin is the upper stable member of the pair.
                sroot, rep = max(candidates, key=lambda z: z[0]["phi_deg"])
                chosen = sroot
                lower_repeller = float(rep["phi_deg"])
                born = True

        else:
            # First try a local continuity search around the previous weak root.
            lo = max(weak_floor_deg, prev_phi - local_halfwidth_deg)
            hi = min(89.95, prev_phi + local_halfwidth_deg)

            grid = np.arange(lo, hi + 0.5 * local_step_deg, local_step_deg)
            vals = v_true_grid(grid, X, residual)

            local_roots = []
            for k in range(len(grid) - 1):
                if vals[k] * vals[k + 1] < 0.0:
                    root = bisect_v_true_root(grid[k], grid[k + 1], X, residual)
                    if root is None:
                        continue
                    deriv = v_true_derivative(root, X, residual)
                    if deriv < 0.0 and root >= weak_floor_deg:
                        local_roots.append({
                            "phi_deg": float(root),
                            "dV_dphi": float(deriv),
                            "kind": "stable",
                        })

            if local_roots:
                chosen = min(
                    local_roots,
                    key=lambda r: abs(r["phi_deg"] - prev_phi),
                )
                root_count = len(local_roots)
            else:
                # Robust fallback: full Q1 census, but still forbid the strong root.
                roots = all_q1_roots(
                    X, residual,
                    grid_step=discovery_step_deg,
                )
                root_count = len(roots)

                stable_hi = [
                    r for r in roots
                    if r["kind"] == "stable" and r["phi_deg"] >= weak_floor_deg
                ]
                if stable_hi:
                    chosen = min(
                        stable_hi,
                        key=lambda r: abs(r["phi_deg"] - prev_phi),
                    )

                if chosen is not None:
                    reps = [
                        r for r in roots
                        if r["kind"] == "repeller" and r["phi_deg"] < chosen["phi_deg"]
                    ]
                    if reps:
                        lower_repeller = float(max(reps, key=lambda r: r["phi_deg"])["phi_deg"])

        if chosen is None:
            rows.append({
                "index": j,
                "time": t,
                "phi_deg": np.nan,
                "dV_dphi": np.nan,
                "lower_repeller_deg": np.nan,
                "root_count": root_count,
                "branch_exists": 0,
            })
            continue

        prev_phi = float(chosen["phi_deg"])
        rows.append({
            "index": j,
            "time": t,
            "phi_deg": prev_phi,
            "dV_dphi": float(chosen["dV_dphi"]),
            "lower_repeller_deg": lower_repeller,
            "root_count": root_count,
            "branch_exists": 1,
        })

    valid = [r for r in rows if r["branch_exists"]]
    if not valid:
        raise RuntimeError(
            "No weak high-angle stable V_true branch was found. "
            "Inspect true_field_root_census.csv."
        )

    return rows

def branch_arrays_for_times(branch_rows, target_times):
    """
    Interpolate tracked phi_W(t) on requested target_times, using only valid
    branch points.
    """
    bt = np.asarray(
        [r["time"] for r in branch_rows if r["branch_exists"]],
        dtype=float,
    )
    bp = np.asarray(
        [r["phi_deg"] for r in branch_rows if r["branch_exists"]],
        dtype=float,
    )

    target_times = np.asarray(target_times, dtype=float)
    out = np.full_like(target_times, np.nan, dtype=float)

    valid = (target_times >= bt[0]) & (target_times <= bt[-1])
    out[valid] = np.interp(target_times[valid], bt, bp)
    return out


# =============================================================================
# Final blocks defined ONCE at t_block using tracked theorem directions
# =============================================================================

def define_final_blocks_from_branch(
    run,
    phi_block_deg,
    t_block=4.0,
    tol_deg=15.0,
):
    times = run["time"]
    j = nearest_index(times, t_block)
    U = run["U"][j]
    W = run["W"][j]

    ang = angle_deg(U)
    mass = projected_mass_per_neuron(U, W)

    d_strong = angular_distance_deg(ang, 0.0)
    d_weak = angular_distance_deg(ang, phi_block_deg)

    strong = (d_strong <= tol_deg) & (d_strong <= d_weak)
    weak = (d_weak <= tol_deg) & (d_weak < d_strong)
    remainder = ~(strong | weak)

    total_mass = float(np.sum(mass))

    return {
        "t_block": float(times[j]),
        "index": j,
        "strong": strong,
        "weak": weak,
        "remainder": remainder,
        "phi_block_deg": float(phi_block_deg),
        "strong_mass_fraction": float(
            np.sum(mass[strong]) / max(total_mass, 1e-12)
        ),
        "weak_mass_fraction": float(
            np.sum(mass[weak]) / max(total_mass, 1e-12)
        ),
        "remainder_mass_fraction": float(
            np.sum(mass[remainder]) / max(total_mass, 1e-12)
        ),
    }


def block_matrix(U, W, mask):
    if not np.any(mask):
        return np.zeros((2, 2), dtype=float)
    return W[mask].T @ U[mask]


def block_matrix_dot(U, W, dU, dW, mask):
    if not np.any(mask):
        return np.zeros((2, 2), dtype=float)
    return dW[mask].T @ U[mask] + W[mask].T @ dU[mask]


def postblock_weak_spread(run, blocks, branch_rows, t_start, t_end):
    times = run["time"]
    phi_interp = branch_arrays_for_times(branch_rows, times)
    j0 = nearest_index(times, t_start)
    j1 = nearest_index(times, t_end)

    rows = []
    weak = blocks["weak"]

    for j in range(j0, j1 + 1):
        phi = phi_interp[j]
        if not np.isfinite(phi):
            continue

        U = run["U"][j]
        W = run["W"][j]
        mass = projected_mass_per_neuron(U, W)[weak]
        a = angle_deg(U[weak])
        dev = wrap_deg(a - phi)

        q05 = weighted_quantile(dev, mass, 0.05)
        q50 = weighted_quantile(dev, mass, 0.50)
        q95 = weighted_quantile(dev, mass, 0.95)

        rows.append({
            "time": float(times[j]),
            "phi_true_root_deg": float(phi),
            "spread_q05_deg": q05,
            "spread_q50_deg": q50,
            "spread_q95_deg": q95,
            "spread_90_width_deg": q95 - q05,
        })

    return rows



def empirical_weak_family_vs_root(
    run,
    branch_rows,
    outdir,
    t_start,
    t_end,
    window_deg=10.0,
):
    """
    Diagnostic only (NOT used to define block membership):
    at each time, collect the current neurons lying within `window_deg` of the
    theorem stable root and report their mass-weighted representative angle.

    This tells us whether the actual mass-dominant weak family tracks the
    stable V_true root closely enough for the selector theorem.
    """
    times = run["time"]
    phi_interp = branch_arrays_for_times(branch_rows, times)
    j0 = nearest_index(times, t_start)
    j1 = nearest_index(times, t_end)

    rows = []
    for j in range(j0, j1 + 1):
        phi = phi_interp[j]
        if not np.isfinite(phi):
            continue

        U = run["U"][j]
        W = run["W"][j]
        ang = angle_deg(U)
        mass = projected_mass_per_neuron(U, W)
        total_mass = float(np.sum(mass))

        mask = angular_distance_deg(ang, phi) <= window_deg
        if not np.any(mask) or np.sum(mass[mask]) <= 0:
            continue

        vec = np.sum(mass[mask, None] * U[mask], axis=0)
        nr = np.linalg.norm(vec)
        if nr <= 0:
            continue

        emp_phi = float(np.degrees(np.arctan2(vec[1], vec[0])))
        dev = wrap_deg(ang[mask] - emp_phi)

        q05 = weighted_quantile(dev, mass[mask], 0.05)
        q95 = weighted_quantile(dev, mass[mask], 0.95)

        rows.append({
            "time": float(times[j]),
            "phi_root_deg": float(phi),
            "phi_empirical_deg": emp_phi,
            "empirical_minus_root_deg": float(wrap_deg(emp_phi - phi)),
            "selected_mass_fraction": float(
                np.sum(mass[mask]) / max(total_mass, 1e-12)
            ),
            "selected_count": int(np.sum(mask)),
            "empirical_90_width_deg": q95 - q05,
        })

    if rows:
        with (outdir / "weak_root_tracking_diagnostic.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        t = np.asarray([r["time"] for r in rows])
        pr = np.asarray([r["phi_root_deg"] for r in rows])
        pe = np.asarray([r["phi_empirical_deg"] for r in rows])

        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(t, pr, label="stable $V_{true}$ root")
        plt.plot(t, pe, label="empirical weak-family center")
        plt.xlabel("time")
        plt.ylabel("angle (degrees)")
        plt.title("Does the learned weak family track the theorem root?")
        plt.legend()
        fig.tight_layout()
        fig.savefig(outdir / "weak_root_tracking_diagnostic.pdf")
        plt.close(fig)

    return rows


# =============================================================================
# Persistent swing detector
# =============================================================================

def moving_average(y, window):
    y = np.asarray(y, dtype=float)
    window = max(1, int(window))
    if window == 1:
        return y.copy()

    kernel = np.ones(window, dtype=float) / window
    left = window // 2
    right = window - 1 - left
    yp = np.pad(y, (left, right), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


def local_minima(y):
    return [
        i for i in range(1, len(y) - 1)
        if y[i] <= y[i - 1] and y[i] < y[i + 1]
    ]


def persistent_swing(
    times,
    y,
    smooth_window=11,
    persistence_time=0.20,
    hold_fraction=0.80,
):
    times = np.asarray(times, dtype=float)
    ys = moving_average(y, smooth_window)
    mins = local_minima(ys)

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    hold_steps = max(1, int(round(persistence_time / dt)))

    best = {
        "swing": 0.0,
        "t_reversal": np.nan,
        "loss_min": np.nan,
        "t_rebound": np.nan,
        "loss_rebound": np.nan,
    }

    for i in mins:
        for j in range(i + 1, len(ys)):
            delta = float(ys[j] - ys[i])
            if delta <= best["swing"]:
                continue

            end = j + hold_steps + 1
            if end > len(ys):
                continue

            threshold = ys[i] + hold_fraction * delta
            if np.all(ys[j:end] >= threshold):
                best = {
                    "swing": delta,
                    "t_reversal": float(times[i]),
                    "loss_min": float(ys[i]),
                    "t_rebound": float(times[j]),
                    "loss_rebound": float(ys[j]),
                }

    return best, ys


# =============================================================================
# Selector geometry from the tracked stable V_true branch
# =============================================================================

def selector_scores(phi_deg, Xdir):
    """
    phi_deg: (T,), Xdir: (K,2)
    returns scores (T,K)
    """
    phi = np.radians(phi_deg)
    Uphi = np.column_stack([np.cos(phi), np.sin(phi)])
    return Uphi @ Xdir.T


def all_switch_times(times, score):
    times = np.asarray(times, dtype=float)
    score = np.asarray(score, dtype=float)
    out = []

    valid = np.isfinite(score)
    idx = np.where(valid)[0]

    for a, b in zip(idx[:-1], idx[1:]):
        s0, s1 = score[a], score[b]
        if s0 * s1 < 0.0:
            t0, t1 = times[a], times[b]
            tcross = t0 - s0 * (t1 - t0) / (s1 - s0)
            transition = (
                "on_to_off" if (s0 > 0 and s1 < 0) else "off_to_on"
            )
            out.append((float(tcross), transition))

    return out



def switch_nearest_time(times_and_transitions, target_time):
    if not times_and_transitions or not np.isfinite(target_time):
        return np.nan, "none", np.nan
    t, transition = min(
        times_and_transitions,
        key=lambda z: abs(z[0] - target_time),
    )
    return float(t), transition, float(target_time - t)


def switch_sign_prediction(x, M1, M2, transition):
    """
    Discrete-selector sign test at a switch.

    e0 = d1 M1 x - x, q = M2 x
    E_on  = 1/2 ||e0+q||^2
    E_off = 1/2 ||e0||^2

    Positive returned delta means the selector transition worsens OOD error.
    """
    d1 = float(x[0] > 0.0)
    e0 = d1 * (M1 @ x) - x
    q = M2 @ x

    E_off = 0.5 * float(np.dot(e0, e0))
    E_on = 0.5 * float(np.dot(e0 + q, e0 + q))

    if transition == "on_to_off":
        delta = E_off - E_on
    elif transition == "off_to_on":
        delta = E_on - E_off
    else:
        delta = np.nan

    return float(delta), E_on, E_off


def build_direction_grid(phi_min_deg, phi_max_deg):
    """
    720 global directions plus 0.05-degree refinement around the two swept
    stable-root selector sectors.
    """
    global_deg = np.linspace(-180.0, 180.0, 720, endpoint=False)

    lo = min(phi_min_deg, phi_max_deg)
    hi = max(phi_min_deg, phi_max_deg)
    intervals = [
        (lo - 90.0, hi - 90.0),
        (lo + 90.0, hi + 90.0),
    ]

    fine = []
    for a, b in intervals:
        fine.extend(np.arange(a - 1.0, b + 1.0001, 0.05))

    explicit_c = [-0.10, -0.05, -0.02, 0.0, 0.02]
    explicit = [math.degrees(math.atan2(c, 1.0)) for c in explicit_c]
    explicit += [wrap_deg(v + 180.0).item() for v in explicit]

    all_deg = np.concatenate([
        global_deg,
        np.asarray(fine, dtype=float),
        np.asarray(explicit, dtype=float),
    ])
    all_deg = np.unique(np.round(wrap_deg(all_deg), 6))
    all_deg.sort()
    return all_deg, intervals


# =============================================================================
# Full-model probe derivative
# =============================================================================

def full_probe_dynamics(U, W, dU, dW, x):
    """
    Return full ReLU output f(x) and its continuous-time derivative at x,
    away from exact gate-boundary instants.
    """
    z = U @ x
    gate = (z > 0.0).astype(float)
    a = np.maximum(z, 0.0)

    f = a @ W

    # d/dt [w_i sigma(u_i^T x)]
    # = dot w_i sigma(z_i) + w_i 1{z_i>0} dot u_i^T x.
    df = a @ dW + ((gate * (dU @ x))[:, None] * W).sum(axis=0)
    return f, df


# =============================================================================
# Corrected directional phase diagram
# =============================================================================

def corrected_phase_diagram(
    run,
    branch_rows,
    blocks,
    outdir,
    t_switch_end=6.0,
    t_eval_end=10.0,
    R=1.0,
):
    times = run["time"]

    valid_branch = [r for r in branch_rows if r["branch_exists"]]
    branch_birth = float(valid_branch[0]["time"])

    # Wedge prediction uses the theorem branch over the mechanism window.
    switch_rows = [
        r for r in valid_branch
        if r["time"] <= t_switch_end
    ]
    phi_switch = np.asarray([r["phi_deg"] for r in switch_rows])
    phi_min = float(np.min(phi_switch))
    phi_max = float(np.max(phi_switch))

    psi_deg, intervals = build_direction_grid(phi_min, phi_max)
    psi_rad = np.radians(psi_deg)
    Xdir = R * np.column_stack([np.cos(psi_rad), np.sin(psi_rad)])
    K = len(psi_deg)

    j0 = nearest_index(times, branch_birth)
    j1 = nearest_index(times, t_eval_end)
    eval_idx = np.arange(j0, j1 + 1)
    t_eval = times[eval_idx]
    phi_eval = branch_arrays_for_times(branch_rows, t_eval)

    if np.any(~np.isfinite(phi_eval)):
        raise RuntimeError(
            "Tracked stable branch does not cover the requested evaluation window."
        )

    T = len(eval_idx)
    actual_loss = np.empty((T, K), dtype=float)
    pred_loss = np.empty((T, K), dtype=float)
    actual_out = np.empty((T, K, 2), dtype=float)
    pred_out = np.empty((T, K, 2), dtype=float)

    weak_score = selector_scores(phi_eval, Xdir)

    for tt, j in enumerate(eval_idx):
        U = run["U"][j]
        W = run["W"][j]

        F = relu_forward(U, W, Xdir)
        actual_out[tt] = F
        actual_loss[tt] = 0.5 * np.sum((F - Xdir) ** 2, axis=1)

        M1 = block_matrix(U, W, blocks["strong"])
        M2 = block_matrix(U, W, blocks["weak"])

        d1 = (Xdir[:, 0] > 0.0).astype(float)
        d2 = (weak_score[tt] > 0.0).astype(float)

        Fp = (
            d1[:, None] * (Xdir @ M1.T)
            + d2[:, None] * (Xdir @ M2.T)
        )
        pred_out[tt] = Fp
        pred_loss[tt] = 0.5 * np.sum((Fp - Xdir) ** 2, axis=1)

    rows = []

    for k in range(K):
        x = Xdir[k]
        norm2 = float(np.dot(x, x))

        actual_best, _ = persistent_swing(t_eval, actual_loss[:, k])
        pred_best, _ = persistent_swing(t_eval, pred_loss[:, k])

        switches = all_switch_times(t_eval, weak_score[:, k])
        first_switch_t = switches[0][0] if switches else np.nan
        first_transition = switches[0][1] if switches else "none"
        last_switch_t = switches[-1][0] if switches else np.nan

        nearest_switch_t, nearest_transition, reversal_minus_switch = (
            switch_nearest_time(switches, actual_best["t_reversal"])
        )

        nearest_switch_delta = np.nan
        nearest_E_on = np.nan
        nearest_E_off = np.nan
        if np.isfinite(nearest_switch_t):
            js = nearest_index(t_eval, nearest_switch_t)
            Uj = run["U"][eval_idx[js]]
            Wj = run["W"][eval_idx[js]]
            M1j = block_matrix(Uj, Wj, blocks["strong"])
            M2j = block_matrix(Uj, Wj, blocks["weak"])
            nearest_switch_delta, nearest_E_on, nearest_E_off = (
                switch_sign_prediction(x, M1j, M2j, nearest_transition)
            )

        num = float(np.sum((actual_out[:, k] - pred_out[:, k]) ** 2))
        den = float(np.sum(actual_out[:, k] ** 2))
        rel_model_rms = math.sqrt(num / max(den, 1e-12))

        rows.append({
            "psi_deg": float(psi_deg[k]),
            "x1": float(x[0]),
            "x2": float(x[1]),
            "slope_x2_over_x1": (
                float(x[1] / x[0]) if abs(x[0]) > 1e-12 else np.nan
            ),
            "selector_switch_count": int(len(switches)),
            "first_switch_time": first_switch_t,
            "first_switch_transition": first_transition,
            "last_switch_time": last_switch_t,
            "nearest_switch_to_reversal": nearest_switch_t,
            "nearest_switch_transition": nearest_transition,
            "t_reversal_minus_nearest_switch": reversal_minus_switch,
            "nearest_switch_predicted_delta": nearest_switch_delta,
            "nearest_switch_E_on": nearest_E_on,
            "nearest_switch_E_off": nearest_E_off,
            "actual_persistent_swing": actual_best["swing"],
            "actual_S_normalized": actual_best["swing"] / max(norm2, 1e-12),
            "t_reversal_actual": actual_best["t_reversal"],
            "t_rebound_actual": actual_best["t_rebound"],
            "predicted_persistent_swing": pred_best["swing"],
            "predicted_S_normalized": pred_best["swing"] / max(norm2, 1e-12),
            "t_reversal_predicted": pred_best["t_reversal"],
            "t_rebound_predicted": pred_best["t_rebound"],
            "selector_model_relative_rms_over_time": rel_model_rms,
        })

    # Save CSV.
    with (outdir / "phase_diagram.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Console summary.
    print()
    print("=" * 80)
    print("CORRECTED DIRECTIONAL PHASE DIAGRAM")
    print("=" * 80)
    print(f"stable V_true branch birth: t={branch_birth:.4f}")
    print(
        f"stable branch through t={t_switch_end:.2f}: "
        f"phi_min={phi_min:.4f} deg, phi_max={phi_max:.4f} deg"
    )

    r1 = -1.0 / math.tan(math.radians(phi_min))
    r2 = -1.0 / math.tan(math.radians(phi_max))
    print(
        "positive-x selector-switch slope wedge: "
        f"x2/x1 in ({min(r1,r2):.5f}, {max(r1,r2):.5f})"
    )

    print("predicted stable-root selector sectors:")
    for a, b in intervals:
        print(f"    psi in [{a:.4f}, {b:.4f}] deg (mod 360)")

    actual_S = np.asarray([r["actual_S_normalized"] for r in rows])
    pred_S = np.asarray([r["predicted_S_normalized"] for r in rows])
    model_err = np.asarray(
        [r["selector_model_relative_rms_over_time"] for r in rows]
    )

    if np.std(actual_S) > 0 and np.std(pred_S) > 0:
        corr_all = float(np.corrcoef(actual_S, pred_S)[0, 1])
    else:
        corr_all = np.nan
    print(f"corr(S_actual,S_pred), all directions = {corr_all:.6f}")

    for thr in (0.02, 0.05, 0.10):
        m = model_err <= thr
        if np.sum(m) >= 3 and np.std(actual_S[m]) > 0 and np.std(pred_S[m]) > 0:
            corr = float(np.corrcoef(actual_S[m], pred_S[m])[0, 1])
            mae = float(np.mean(np.abs(actual_S[m] - pred_S[m])))
            print(
                f"    model RMS <= {thr:.2f}: n={np.sum(m)}, "
                f"corr={corr:.6f}, MAE(S)={mae:.6e}"
            )

    print()
    print("Explicit positive-x wedge probes x=(1,c):")
    for c in (-0.10, -0.05, -0.02, 0.0, 0.02):
        target = math.degrees(math.atan2(c, 1.0))
        rr = min(
            rows,
            key=lambda z: abs(wrap_deg(z["psi_deg"] - target).item()),
        )
        print(
            f"    c={c:+.3f}, psi={rr['psi_deg']:+.3f} deg: "
            f"switch_count={rr['selector_switch_count']}, "
            f"first_switch={rr['first_switch_time'] if np.isfinite(rr['first_switch_time']) else np.nan:.3f}, "
            f"S_actual={rr['actual_S_normalized']:.6e}, "
            f"S_pred={rr['predicted_S_normalized']:.6e}, "
            f"t_rev={rr['t_reversal_actual'] if np.isfinite(rr['t_reversal_actual']) else np.nan:.3f}"
        )

    switching_swinging = [
        r for r in rows
        if r["selector_switch_count"] > 0
        and r["actual_S_normalized"] > 1e-4
        and np.isfinite(r["t_reversal_actual"])
    ]
    print()
    print(
        "switching directions with actual persistent normalized swing > 1e-4: "
        f"{len(switching_swinging)}"
    )
    if switching_swinging:
        dt = np.asarray([
            r["t_reversal_minus_nearest_switch"]
            for r in switching_swinging
            if np.isfinite(r["t_reversal_minus_nearest_switch"])
        ])
        if len(dt):
            print(
                "t_reversal - nearest t_switch: "
                f"median={np.median(dt):+.4f}, mean_abs={np.mean(np.abs(dt)):.4f}"
            )

    # Main phase diagram.
    psi = np.asarray([r["psi_deg"] for r in rows])

    fig = plt.figure(figsize=(9, 4.8))
    plt.plot(psi, actual_S, label="full ReLU measured $S(\\psi)$")
    plt.plot(psi, pred_S, label="reduced-model predicted $S(\\psi)$")
    for a, b in intervals:
        if -180.0 <= a <= 180.0 and -180.0 <= b <= 180.0:
            plt.axvspan(min(a, b), max(a, b), alpha=0.12)
    plt.xlabel("probe direction $\\psi$ (degrees)")
    plt.ylabel("$S(\\psi)=\\Delta_{swing}/R^2$")
    plt.title("Corrected theory-guided OOD swing phase diagram")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "phase_diagram_S_corrected.pdf")
    plt.close(fig)

    # Fine positive-x switch wedge.
    pos_a = phi_min - 90.0
    pos_b = phi_max - 90.0
    lo, hi = min(pos_a, pos_b), max(pos_a, pos_b)
    zm = (psi >= lo - 2.0) & (psi <= hi + 2.0)

    fig = plt.figure(figsize=(8, 4.8))
    plt.plot(psi[zm], actual_S[zm], label="full ReLU")
    plt.plot(psi[zm], pred_S[zm], label="reduced model")
    plt.axvspan(lo, hi, alpha=0.12, label="stable-root switch sector")
    plt.xlabel("probe direction $\\psi$ (degrees)")
    plt.ylabel("$S(\\psi)$")
    plt.title("Corrected fine sweep around weak-selector switch wedge")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "phase_diagram_positive_switch_wedge_corrected.pdf")
    plt.close(fig)

    # Switch time by direction.
    switched = [r for r in rows if r["selector_switch_count"] > 0]
    fig = plt.figure(figsize=(8, 4.8))
    if switched:
        plt.scatter(
            [r["psi_deg"] for r in switched],
            [r["first_switch_time"] for r in switched],
            s=12,
        )
    plt.xlabel("probe direction $\\psi$ (degrees)")
    plt.ylabel("first $t_{switch}(\\psi)$")
    plt.title("Weak-selector switch time from stable $V_{true}$ branch")
    fig.tight_layout()
    fig.savefig(outdir / "switch_time_vs_direction_corrected.pdf")
    plt.close(fig)

    # Headline timing panel.
    fig = plt.figure(figsize=(5.5, 5.5))
    if switching_swinging:
        matched = [
            r for r in switching_swinging
            if np.isfinite(r["nearest_switch_to_reversal"])
        ]
        xs = np.asarray([r["nearest_switch_to_reversal"] for r in matched])
        ys = np.asarray([r["t_reversal_actual"] for r in matched])
        plt.scatter(xs, ys, s=16)
        diag_lo = min(np.min(xs), np.min(ys))
        diag_hi = max(np.max(xs), np.max(ys))
        plt.plot(
            [diag_lo, diag_hi],
            [diag_lo, diag_hi],
            linestyle="--",
            label="$t_{reversal}=t_{switch}$",
        )
        plt.legend()
    plt.xlabel("$t_{switch}(\\psi)$")
    plt.ylabel("$t_{reversal}(\\psi)$")
    plt.title("Does reversal track the stable-root selector switch?")
    fig.tight_layout()
    fig.savefig(outdir / "reversal_vs_switch_time_corrected.pdf")
    plt.close(fig)

    return rows, intervals


# =============================================================================
# Canonical block-matrix dE/dt decomposition
# =============================================================================

TERM_NAMES = [
    (1, 0, 0, "p1_m11"),
    (1, 0, 1, "p1_m12"),
    (1, 1, 0, "p1_m21"),
    (1, 1, 1, "p1_m22"),
    (2, 0, 0, "p2_m11"),
    (2, 0, 1, "p2_m12"),
    (2, 1, 0, "p2_m21"),
    (2, 1, 1, "p2_m22"),
]


def mechanism_decomposition(
    run,
    branch_rows,
    blocks,
    outdir,
    probes,
    t_end=10.0,
):
    times = run["time"]
    valid_branch = [r for r in branch_rows if r["branch_exists"]]
    t_start = float(valid_branch[0]["time"])
    j0 = nearest_index(times, t_start)
    j1 = nearest_index(times, t_end)
    idx = np.arange(j0, j1 + 1)
    t = times[idx]

    phi = branch_arrays_for_times(branch_rows, t)

    summaries = []

    for label, psi_deg in probes.items():
        psi = math.radians(psi_deg)
        x = np.array([math.cos(psi), math.sin(psi)], dtype=float)

        rows = []

        for tt, j in enumerate(idx):
            U = run["U"][j]
            W = run["W"][j]
            dU, dW, _ = relu_rhs(U, W, run["X"])

            M1 = block_matrix(U, W, blocks["strong"])
            M2 = block_matrix(U, W, blocks["weak"])
            dM1 = block_matrix_dot(U, W, dU, dW, blocks["strong"])
            dM2 = block_matrix_dot(U, W, dU, dW, blocks["weak"])

            uW = unit_direction_from_deg(phi[tt])
            d1 = float(x[0] > 0.0)
            d2 = float(np.dot(uW, x) > 0.0)

            f_red = d1 * (M1 @ x) + d2 * (M2 @ x)
            e_red = f_red - x
            loss_red = 0.5 * float(np.dot(e_red, e_red))

            df_red = d1 * (dM1 @ x) + d2 * (dM2 @ x)
            edot_red = float(np.dot(e_red, df_red))

            f_full, df_full = full_probe_dynamics(U, W, dU, dW, x)
            e_full = f_full - x
            loss_full = 0.5 * float(np.dot(e_full, e_full))
            edot_full = float(np.dot(e_full, df_full))

            rec = {
                "time": float(times[j]),
                "psi_deg": float(psi_deg),
                "phi_W_true_root_deg": float(phi[tt]),
                "d1": d1,
                "d2": d2,
                "loss_full": loss_full,
                "loss_reduced": loss_red,
                "edot_full": edot_full,
                "edot_reduced_continuous": edot_red,
            }

            for r in range(2):
                for l in range(2):
                    rec[f"M1_{r+1}{l+1}"] = float(M1[r, l])
                    rec[f"M2_{r+1}{l+1}"] = float(M2[r, l])
                    rec[f"dM1_{r+1}{l+1}"] = float(dM1[r, l])
                    rec[f"dM2_{r+1}{l+1}"] = float(dM2[r, l])

            contrib_sum = 0.0
            for p, r, l, name in TERM_NAMES:
                dp = d1 if p == 1 else d2
                dMp = dM1 if p == 1 else dM2
                c = float(dp * e_red[r] * dMp[r, l] * x[l])
                rec[name] = c
                contrib_sum += c

            rec["term_sum"] = contrib_sum
            rec["term_sum_minus_edot_reduced"] = contrib_sum - edot_red
            rows.append(rec)

        # Save per-probe CSV.
        with (outdir / f"decomp_{label}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        loss_full = np.asarray([r["loss_full"] for r in rows])
        loss_red = np.asarray([r["loss_reduced"] for r in rows])
        edot_full = np.asarray([r["edot_full"] for r in rows])
        edot_red = np.asarray([r["edot_reduced_continuous"] for r in rows])

        best_full, _ = persistent_swing(t, loss_full)
        best_red, _ = persistent_swing(t, loss_red)

        # Loss plot.
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(t, loss_full, label="full ReLU")
        plt.plot(t, loss_red, label="reduced block model")
        if np.isfinite(best_full["t_reversal"]):
            plt.axvline(
                best_full["t_reversal"],
                linestyle="--",
                label="$t_{reversal}$",
            )
        plt.xlabel("time")
        plt.ylabel("$E_\\psi(t)$")
        plt.title(f"OOD loss decomposition: $\\psi={psi_deg:.1f}^\\circ$")
        plt.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"decomp_{label}_loss.pdf")
        plt.close(fig)

        # dE/dt plot.
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(t, edot_full, label="full $\\dot E$")
        plt.plot(t, edot_red, label="reduced continuous-selector $\\dot E$")
        plt.axhline(0.0, linestyle=":")
        if np.isfinite(best_full["t_reversal"]):
            plt.axvline(best_full["t_reversal"], linestyle="--")
        plt.xlabel("time")
        plt.ylabel("$\\dot E_\\psi(t)$")
        plt.title(f"OOD error derivative: $\\psi={psi_deg:.1f}^\\circ$")
        plt.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"decomp_{label}_edot.pdf")
        plt.close(fig)

        # Entrywise contribution plot.
        fig = plt.figure(figsize=(9, 5.0))
        for _, _, _, name in TERM_NAMES:
            vals = np.asarray([r[name] for r in rows])
            if np.max(np.abs(vals)) > 1e-6:
                plt.plot(t, vals, label=name)
        plt.axhline(0.0, linestyle=":")
        if np.isfinite(best_full["t_reversal"]):
            plt.axvline(best_full["t_reversal"], linestyle="--")
        plt.xlabel("time")
        plt.ylabel("$e_r\\,\\dot m_{r\\ell}^{(p)}\\,x_\\ell$")
        plt.title(f"Entrywise $\\dot E$ contributions: $\\psi={psi_deg:.1f}^\\circ$")
        plt.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(outdir / f"decomp_{label}_terms.pdf")
        plt.close(fig)

        # Mechanism summary around actual reversal.
        summary = {
            "probe": label,
            "psi_deg": float(psi_deg),
            "full_swing": best_full["swing"],
            "full_t_reversal": best_full["t_reversal"],
            "reduced_swing": best_red["swing"],
            "reduced_t_reversal": best_red["t_reversal"],
        }

        if np.isfinite(best_full["t_reversal"]):
            tr = best_full["t_reversal"]
            before = (t >= tr - 0.20) & (t < tr)
            after = (t > tr) & (t <= tr + 0.20)

            for _, _, _, name in TERM_NAMES:
                vals = np.asarray([r[name] for r in rows])
                summary[f"{name}_mean_before"] = (
                    float(np.mean(vals[before])) if np.any(before) else np.nan
                )
                summary[f"{name}_mean_after"] = (
                    float(np.mean(vals[after])) if np.any(after) else np.nan
                )
        else:
            for _, _, _, name in TERM_NAMES:
                summary[f"{name}_mean_before"] = np.nan
                summary[f"{name}_mean_after"] = np.nan

        summaries.append(summary)

        print()
        print(f"Mechanism probe {label}: psi={psi_deg:.1f} deg")
        print(
            f"    full swing={best_full['swing']:.6e}, "
            f"t_reversal={best_full['t_reversal'] if np.isfinite(best_full['t_reversal']) else np.nan:.3f}"
        )
        print(
            f"    reduced swing={best_red['swing']:.6e}, "
            f"t_reversal={best_red['t_reversal'] if np.isfinite(best_red['t_reversal']) else np.nan:.3f}"
        )

        if np.isfinite(best_full["t_reversal"]):
            tr = best_full["t_reversal"]
            jrev = int(np.argmin(np.abs(t - tr)))
            vals = []
            for _, _, _, name in TERM_NAMES:
                vals.append((name, rows[jrev][name]))
            vals.sort(key=lambda z: abs(z[1]), reverse=True)
            print("    largest entrywise dE/dt terms at reversal:")
            for name, val in vals[:4]:
                print(f"        {name}: {val:+.6e}")

    # Save cross-probe summary.
    with (outdir / "decomposition_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    return summaries


# =============================================================================
# Main
# =============================================================================

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
    t_branch_end = 10.0
    t_switch_end = 6.0
    t_block = 4.0
    block_tol_deg = 15.0

    outdir = Path("theory_guided_swing_phase_diagram_v3")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PAPER-CLEAN THEORY-GUIDED SWING-BY EXPERIMENT v3")
    print("=" * 80)
    print(
        f"canonical: mu=({cfg.mu1},{cfg.mu2}), "
        f"sigma=({cfg.sigma1},{cfg.sigma2}), width={cfg.width}, "
        f"n/cluster={cfg.n_per_cluster}, eps={cfg.epsilon}, dt={cfg.dt}"
    )
    print(
        f"t_create={t_create}, t_block={t_block}, "
        f"t_switch_end={t_switch_end}, t_end={t_branch_end}"
    )
    print("model: two-layer ReLU only")
    print("weak theorem branch: stable Q1 root of V_true(phi;t)")

    run = run_relu(cfg, log_every=50)

    # -------------------------------------------------------------------------
    # 1) Correct theorem-facing branch tracker.
    # -------------------------------------------------------------------------
    root_census(
        run,
        query_times=[1.06, 1.07, 1.08, 1.12, 2.0, 3.0, 3.5, 4.0, 6.0, 10.0],
        outdir=outdir,
    )

    branch = track_stable_true_field_branch(
        run,
        t_start=t_create,
        t_end=t_branch_end,
        weak_floor_deg=60.0,
        birth_repeller_floor_deg=20.0,
    )

    valid = [r for r in branch if r["branch_exists"]]
    print()
    print("Stable V_true weak branch:")
    print(
        f"    first stable root: t={valid[0]['time']:.4f}, "
        f"phi={valid[0]['phi_deg']:.4f} deg"
    )
    print(
        f"    phi(t=6)={branch_arrays_for_times(branch, [6.0])[0]:.4f} deg"
    )
    print(
        f"    phi(t=10)={branch_arrays_for_times(branch, [10.0])[0]:.4f} deg"
    )

    with (outdir / "true_field_weak_branch.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(branch[0].keys()))
        writer.writeheader()
        writer.writerows(branch)

    # Branch plot.
    bt = np.asarray([r["time"] for r in valid])
    bp = np.asarray([r["phi_deg"] for r in valid])

    fig = plt.figure(figsize=(8, 4.8))
    plt.plot(bt, bp, label="stable Q1 root of $V_{true}$")
    plt.xlabel("time")
    plt.ylabel("$\\phi_W(t)$ (degrees)")
    plt.title("Corrected learned weak branch from the exact angular field")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "true_field_weak_branch.pdf")
    plt.close(fig)

    weak_track = empirical_weak_family_vs_root(
        run,
        branch,
        outdir,
        t_start=valid[0]["time"],
        t_end=t_branch_end,
        window_deg=10.0,
    )
    if weak_track:
        print()
        print("Stable-root vs empirical weak-family diagnostic:")
        for tq in (2.0, 3.0, 3.5, 4.0, 6.0, 10.0):
            rr = min(weak_track, key=lambda z: abs(z["time"] - tq))
            print(
                f"    t={rr['time']:.2f}: root={rr['phi_root_deg']:.3f} deg, "
                f"emp={rr['phi_empirical_deg']:.3f} deg, "
                f"gap={rr['empirical_minus_root_deg']:+.3f} deg, "
                f"selected mass={rr['selected_mass_fraction']:.3f}"
            )

    # -------------------------------------------------------------------------
    # 2) Define final blocks once at t_block from theorem directions.
    # -------------------------------------------------------------------------
    phi_block = branch_arrays_for_times(branch, [t_block])[0]

    print()
    print(f"Tracked weak direction at t_block={t_block:.2f}: {phi_block:.4f} deg")

    sensitivity = []
    for tol in (10.0, 15.0, 20.0):
        b = define_final_blocks_from_branch(
            run,
            phi_block_deg=phi_block,
            t_block=t_block,
            tol_deg=tol,
        )
        sensitivity.append({
            "t_block": b["t_block"],
            "tolerance_deg": tol,
            "phi_block_deg": b["phi_block_deg"],
            "strong_count": int(np.sum(b["strong"])),
            "weak_count": int(np.sum(b["weak"])),
            "remainder_count": int(np.sum(b["remainder"])),
            "strong_mass_fraction": b["strong_mass_fraction"],
            "weak_mass_fraction": b["weak_mass_fraction"],
            "remainder_mass_fraction": b["remainder_mass_fraction"],
        })
        print(
            f"    tol={tol:4.1f} deg: "
            f"strong mass={b['strong_mass_fraction']:.4f}, "
            f"weak mass={b['weak_mass_fraction']:.4f}, "
            f"remainder mass={b['remainder_mass_fraction']:.4f}"
        )

    with (outdir / "block_sensitivity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity[0].keys()))
        writer.writeheader()
        writer.writerows(sensitivity)

    blocks = define_final_blocks_from_branch(
        run,
        phi_block_deg=phi_block,
        t_block=t_block,
        tol_deg=block_tol_deg,
    )

    # Post-specialization angular spread = empirical boundary-smearing width.
    spread = postblock_weak_spread(
        run,
        blocks,
        branch,
        t_start=t_block,
        t_end=t_switch_end,
    )

    with (outdir / "postblock_weak_spread.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(spread[0].keys()))
        writer.writeheader()
        writer.writerows(spread)

    widths = np.asarray([r["spread_90_width_deg"] for r in spread])
    print(
        "post-block weak 90%-mass angular band width [t_block,6]: "
        f"median={np.median(widths):.4f} deg, max={np.max(widths):.4f} deg"
    )

    # -------------------------------------------------------------------------
    # 3) Corrected phase diagram and selector timing.
    # -------------------------------------------------------------------------
    corrected_phase_diagram(
        run,
        branch,
        blocks,
        outdir,
        t_switch_end=t_switch_end,
        t_eval_end=t_branch_end,
        R=1.0,
    )

    # -------------------------------------------------------------------------
    # 4) Final canonical matrix-entry mechanism decomposition.
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("CANONICAL BLOCK-MATRIX dE/dt DECOMPOSITION")
    print("=" * 80)

    probes = {
        "m85": -85.0,
        "p176": 176.0,
        "pp": 33.7,
    }

    mechanism_decomposition(
        run,
        branch,
        blocks,
        outdir,
        probes,
        t_end=t_branch_end,
    )

    print()
    print("=" * 80)
    print(f"Saved outputs to: {outdir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
