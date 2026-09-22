#!/usr/bin/env python3
"""
swing_grid_preregister.py

Two-stage preregistered 3x3 test for the near-boundary Swing-by normal form.

STAGE 1: preregister
--------------------
For each (mu2, sigma) cell, this stage uses ONLY the ID training trajectory to:
  1. track the stable high-angle weak branch phi_W(t),
  2. define frozen strong/weak blocks once at t_block,
  3. measure eta_p^blk and eta_p^gm in the ordered basis (e_p,e_q),
  4. record the parameter-level strong-alignment predictor mu2*sigma/mu1^2,
  5. record learned gate-boundary locations,
  6. compute the full reduced-model prospective phase curve S_red(psi),
  7. choose, from S_red only, the primary p=1 and p=2 near-boundary cores,
  8. record the finite-difference normal-form G zero crossing for the
     predicted p=1 peak,
  9. save the training trajectory required for later OOD evaluation,
 10. write a timestamped SHA-256 manifest and EXIT.

No full-ReLU OOD direction is evaluated in this stage.

STAGE 2: evaluate
-----------------
This stage first verifies every preregistered file hash. It then evaluates the
frozen full-ReLU trajectory on the OOD direction grid and reports:
  - S_max,p / (eta_p^blk)^2,
  - profile collapse S_p/(eta_p^blk)^2 vs theta_p/eta_p^blk,
  - eta_p^blk and eta_p^gm vs mu2*sigma/mu1^2,
  - predicted reduced-vs-actual phase agreement,
  - t_G=0, t_rev^red, t_rev^full for the predeclared p=1 core probe.

IMPORTANT BASIS CONVENTION
--------------------------
Every block is converted to the ordered basis (e_p,e_q) before extracting
normal-form entries:

  p=1:
      B^(1) = [[M11, M12],
               [M21, M22]]

  p=2:
      B^(2) = [[M22, M21],
               [M12, M11]]

Thus, in BOTH blocks,

      B^(p) = [[m_pp, m_pq],
               [m_qp, m_qq]]

and

  eta_p^blk :=
      max(
          |m_pq|/|m_pp|,
          |m_qp|/|m_pp|,
          sqrt(|m_qq|/|m_pp|)
      )

  eta_p^gm :=
      sqrt(|m_pq m_qp|) / |m_pp|.

The near-boundary normal-form coefficients keep the original unnormalized
convention:

      a_p = m_pp,
      b_p = m_pq / eta_p^blk,
      c_p = m_qp / eta_p^blk,
      d_p = m_qq / (eta_p^blk)^2.

The parameter-level formula

      eta_param = mu2 * sigma / mu1^2

is recorded as a separate strong-block prediction. It is NOT identified with
eta_p^blk by definition.

DEFAULT GRID
------------
    mu2  in {1.8, 2.0, 2.2}
    sigma in {0.12, 0.15, 0.18}
    mu1 = 3.0
    sigma1 = sigma2 = sigma
    width = 200
    n/cluster = 2000
    epsilon = 1e-3
    dt = 2e-4
    T = 10
    t_block = 4
    seed = 0

USAGE
-----
    python3 swing_grid_preregister.py self-test
    python3 swing_grid_preregister_patched.py preregister
    python3 swing_grid_preregister_patched.py refreeze
    python3 swing_grid_preregister_patched.py evaluate

If a pilot preregistration already exists, use `refreeze` to reuse its
saved trajectories and correct the post-specialization prediction window
WITHOUT retraining.  Evaluation uses only the corrected v1 preregistration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from full_flow_diagnostic import Config

try:
    import theory_guided_swing_by_phase_diagram as exp
except ImportError:
    import theory_guided_swing_phase_diagram_v3 as exp


# =============================================================================
# Frozen protocol
# =============================================================================

MU1 = 3.0
MU2_GRID = (1.8, 2.0, 2.2)
SIGMA_GRID = (0.12, 0.15, 0.18)

T_BLOCK = 4.0
T_END = 10.0
BRANCH_SEARCH_START = 0.90
BRANCH_STRIDE = 5              # run logging dt is 0.01 -> branch cadence 0.05
BLOCK_TOL_DEG = 15.0

# Fixed before OOD.
PSI_STEP_DEG = 0.5
PSI_GRID_DEG = np.arange(-180.0, 180.0, PSI_STEP_DEG)

# Normal-form core is predeclared as |theta| <= Z_CORE_MAX * eta.
Z_CORE_MAX = 4.0

# For the finite-difference G timing diagnostic.
G_SEARCH_HALF_WINDOW = 1.0

OUTPUT_ROOT = Path("swing_grid_preregistered")

# Post-specialization validity rule, frozen BEFORE any full-ReLU OOD evaluation.
#
# Let rho_p(t) be the projected-mass SHARE carried by the neurons that belong
# to the frozen t_block block.  Define t_W,p as the first time at which
#
#     rho_p(t) >= POSTSPEC_MASS_SHARE_FRACTION * rho_p(t_block)
#
# and the inequality remains true at every logged checkpoint through t_block.
# The common validity time is
#
#     t_W = max(branch_birth, t_W,1, t_W,2).
#
# This prevents a t_block-defined eventual block from being extrapolated
# backward into the pre-capture regime.
POSTSPEC_MASS_SHARE_FRACTION = 0.90

# The already-frozen pilot preregistration is preserved unchanged.
PILOT_PREREG_ROOT = OUTPUT_ROOT / "preregistration"

# Corrected preregistration/evaluation written by the `refreeze` stage.
PATCHED_PREREG_ROOT = OUTPUT_ROOT / "preregistration_v1_postspecialization"
PATCHED_EVAL_ROOT = OUTPUT_ROOT / "evaluation_v1_postspecialization"


# =============================================================================
# General helpers
# =============================================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def wrap_deg(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def slug_num(x):
    return f"{x:.3f}".replace("-", "m").replace(".", "p")


def cell_id(mu2, sigma):
    return f"mu2_{slug_num(mu2)}__sigma_{slug_num(sigma)}"


def sha256_file(path: Path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def scalar_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


def nearest_index(times, target):
    return int(np.argmin(np.abs(np.asarray(times, dtype=float) - float(target))))


def signed_angular_delta_deg(psi_deg, center_deg):
    """
    Signed shortest difference psi-center in [-180,180).
    """
    return wrap_deg(np.asarray(psi_deg, dtype=float) - float(center_deg))


def direction_vectors(psi_deg):
    r = np.radians(np.asarray(psi_deg, dtype=float))
    return np.column_stack([np.cos(r), np.sin(r)])


# =============================================================================
# Basis handling -- load-bearing p=2 protection
# =============================================================================

def ordered_block_matrix(M, p):
    """
    Re-express ambient (e1,e2) matrix M in the ordered basis (e_p,e_q).

    p=1 -> order [e1,e2]
    p=2 -> order [e2,e1]
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (2, 2):
        raise ValueError(f"Expected 2x2 block matrix, got {M.shape}")

    if p == 1:
        return M.copy()
    if p == 2:
        perm = [1, 0]
        return M[np.ix_(perm, perm)].copy()
    raise ValueError("p must be 1 or 2")


def block_alignment_stats(M, p):
    """
    Return normal-form block entries and operational alignment scales.
    """
    B = ordered_block_matrix(M, p)
    mpp = float(B[0, 0])
    mpq = float(B[0, 1])
    mqp = float(B[1, 0])
    mqq = float(B[1, 1])

    denom = max(abs(mpp), 1e-12)

    r_pq = abs(mpq) / denom
    r_qp = abs(mqp) / denom
    r_qq_sqrt = math.sqrt(abs(mqq) / denom)

    eta_blk = max(r_pq, r_qp, r_qq_sqrt)
    eta_gm = math.sqrt(abs(mpq * mqp)) / denom

    if eta_blk <= 0:
        a, b, c, d = mpp, np.nan, np.nan, np.nan
    else:
        a = mpp
        b = mpq / eta_blk
        c = mqp / eta_blk
        d = mqq / (eta_blk ** 2)

    return {
        "B_ordered": B.tolist(),
        "m_pp": mpp,
        "m_pq": mpq,
        "m_qp": mqp,
        "m_qq": mqq,
        "r_pq": r_pq,
        "r_qp": r_qp,
        "sqrt_r_qq": r_qq_sqrt,
        "eta_blk": eta_blk,
        "eta_gm": eta_gm,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
    }


def self_test_basis_and_eta():
    M = np.array([[1.0, 2.0], [3.0, 4.0]])

    B1 = ordered_block_matrix(M, 1)
    B2 = ordered_block_matrix(M, 2)

    assert np.array_equal(B1, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.array_equal(B2, np.array([[4.0, 3.0], [2.0, 1.0]]))

    s1 = block_alignment_stats(M, 1)
    s2 = block_alignment_stats(M, 2)

    assert s1["m_pp"] == 1.0
    assert s1["m_pq"] == 2.0
    assert s1["m_qp"] == 3.0
    assert s1["m_qq"] == 4.0

    assert s2["m_pp"] == 4.0
    assert s2["m_pq"] == 3.0
    assert s2["m_qp"] == 2.0
    assert s2["m_qq"] == 1.0

    # Canonical matrix arithmetic from the pre-grid G check.
    Mc = np.array([
        [0.9343428442973669, -0.010161874356094199],
        [-0.09422150743313977, 0.006105850333368189],
    ])
    sc = block_alignment_stats(Mc, 1)

    assert abs(sc["r_pq"] - 0.010875958881812819) < 1e-12
    assert abs(sc["r_qp"] - (0.09422150743313977 / 0.9343428442973669)) < 1e-12
    assert sc["eta_blk"] == sc["r_qp"]

    z = math.cos(math.radians(85.0)) / sc["eta_blk"]
    assert 0.8 < z < 0.9

    print("basis permutation test: PASS")
    print("canonical eta_blk arithmetic: PASS")
    print(f"canonical eta_blk={sc['eta_blk']:.8f}, z(-85)={z:.4f}")


# =============================================================================
# Branch tracking and frozen blocks
# =============================================================================

def downsample_run_for_branch(run, stride):
    idx = np.arange(0, len(run["time"]), stride, dtype=int)
    if idx[-1] != len(run["time"]) - 1:
        idx = np.concatenate([idx, [len(run["time"]) - 1]])

    return {
        "X": run["X"],
        "labels": run["labels"],
        "init_phi": run["init_phi"],
        "time": run["time"][idx],
        "U": [run["U"][i] for i in idx],
        "W": [run["W"][i] for i in idx],
    }


def track_weak_branch(run):
    branch_run = downsample_run_for_branch(run, BRANCH_STRIDE)

    branch = exp.track_stable_true_field_branch(
        branch_run,
        t_start=BRANCH_SEARCH_START,
        t_end=T_END,
        weak_floor_deg=60.0,
        birth_repeller_floor_deg=20.0,
    )
    return branch


def valid_branch_arrays(branch):
    bt = np.asarray(
        [r["time"] for r in branch if r["branch_exists"]],
        dtype=float,
    )
    bp = np.asarray(
        [r["phi_deg"] for r in branch if r["branch_exists"]],
        dtype=float,
    )
    if len(bt) == 0:
        raise RuntimeError("Weak branch has no valid points.")
    return bt, bp


def interp_branch(branch, target_times):
    bt, bp = valid_branch_arrays(branch)
    target_times = np.asarray(target_times, dtype=float)

    out = np.full_like(target_times, np.nan, dtype=float)
    mask = (target_times >= bt[0]) & (target_times <= bt[-1])
    out[mask] = np.interp(target_times[mask], bt, bp)
    return out


def save_branch_csv(path, branch):
    keys = list(branch[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(branch)



# =============================================================================
# ID-only post-specialization validity time
# =============================================================================

def projected_mass_share_history(Uhist, Whist, mask):
    """
    Projected-mass SHARE of a fixed neuron cohort:
        rho_A(t) = sum_{i in A}(||u_i||^2+||w_i||^2)
                   / sum_i(||u_i||^2+||w_i||^2).

    The cohort mask is frozen at t_block.  Using a share rather than absolute
    mass prevents ordinary radial growth of all neurons from being confused
    with cohort recruitment.
    """
    U = np.asarray(Uhist, dtype=float)
    W = np.asarray(Whist, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    mass = np.sum(U * U, axis=2) + np.sum(W * W, axis=2)
    total = np.sum(mass, axis=1)
    cohort = np.sum(mass[:, mask], axis=1)
    return cohort / np.maximum(total, 1e-12)


def first_persistent_fraction_time(
    times,
    share,
    target_share,
    threshold,
    t_lo,
    t_hi,
):
    """
    First time t in [t_lo,t_hi] such that share(s)/target_share >= threshold
    for EVERY logged s in [t,t_hi].
    """
    times = np.asarray(times, dtype=float)
    share = np.asarray(share, dtype=float)

    j0 = int(np.searchsorted(times, t_lo, side="left"))
    j1 = nearest_index(times, t_hi)

    if j0 > j1:
        raise RuntimeError("Invalid post-specialization search interval.")

    ratio = share / max(float(target_share), 1e-12)
    local = ratio[j0:j1 + 1]

    # suffix_min[k] = min(local[k:])
    suffix_min = np.minimum.accumulate(local[::-1])[::-1]
    ok = np.where(suffix_min >= threshold)[0]

    if len(ok) == 0:
        # t_block itself has ratio 1 by construction, so this should only occur
        # under numerical/pathological input.
        return float(times[j1]), float(ratio[j1]), float(suffix_min[-1])

    k = int(ok[0])
    j = j0 + k
    return float(times[j]), float(ratio[j]), float(suffix_min[k])


def compute_post_specialization_time(
    times,
    Uhist,
    Whist,
    strong_mask,
    weak_mask,
    branch_birth,
    t_block=T_BLOCK,
    threshold=POSTSPEC_MASS_SHARE_FRACTION,
):
    """
    ID-only common validity time for the frozen block reduction.

    t_W,strong / t_W,weak are based only on recruitment of the final frozen
    cohorts as measured by projected-mass share.  The common t_W is their max,
    also constrained to lie after weak-branch birth.
    """
    times = np.asarray(times, dtype=float)
    j_block = nearest_index(times, t_block)

    rho1 = projected_mass_share_history(Uhist, Whist, strong_mask)
    rho2 = projected_mass_share_history(Uhist, Whist, weak_mask)

    rho1_block = float(rho1[j_block])
    rho2_block = float(rho2[j_block])

    t1, ratio1, suffix1 = first_persistent_fraction_time(
        times, rho1, rho1_block, threshold, branch_birth, t_block
    )
    t2, ratio2, suffix2 = first_persistent_fraction_time(
        times, rho2, rho2_block, threshold, branch_birth, t_block
    )

    tW = max(float(branch_birth), t1, t2)

    jW = nearest_index(times, tW)

    return {
        "criterion": (
            "first persistent time where each frozen block's projected-mass "
            "share is at least threshold times its t_block share"
        ),
        "threshold_fraction_of_t_block_share": float(threshold),
        "branch_birth": float(branch_birth),
        "t_W_strong": float(t1),
        "t_W_weak": float(t2),
        "t_W": float(tW),
        "strong_share_t_block": rho1_block,
        "weak_share_t_block": rho2_block,
        "strong_share_t_W": float(rho1[jW]),
        "weak_share_t_W": float(rho2[jW]),
        "strong_ratio_t_W": float(rho1[jW] / max(rho1_block, 1e-12)),
        "weak_ratio_t_W": float(rho2[jW] / max(rho2_block, 1e-12)),
        "strong_ratio_at_own_onset": float(ratio1),
        "weak_ratio_at_own_onset": float(ratio2),
        "strong_persistent_suffix_min_at_own_onset": float(suffix1),
        "weak_persistent_suffix_min_at_own_onset": float(suffix2),
    }


# =============================================================================
# Reduced prediction from ID trajectory only
# =============================================================================

def block_histories(run, blocks):
    times = np.asarray(run["time"], dtype=float)
    M1 = np.empty((len(times), 2, 2), dtype=float)
    M2 = np.empty((len(times), 2, 2), dtype=float)

    for j in range(len(times)):
        U = run["U"][j]
        W = run["W"][j]
        M1[j] = exp.block_matrix(U, W, blocks["strong"])
        M2[j] = exp.block_matrix(U, W, blocks["weak"])

    return M1, M2


def phase_from_losses(times, losses):
    """
    losses shape = (T,K)
    """
    K = losses.shape[1]
    S = np.zeros(K, dtype=float)
    t_rev = np.full(K, np.nan, dtype=float)

    for k in range(K):
        best, _ = exp.persistent_swing(times, losses[:, k])
        S[k] = best["swing"]
        t_rev[k] = best["t_reversal"]

    return S, t_rev


def reduced_prediction(run, branch, blocks, M1_hist, M2_hist, t_start=None):
    times = np.asarray(run["time"], dtype=float)
    phi_all = interp_branch(branch, times)

    bt, _ = valid_branch_arrays(branch)
    if t_start is None:
        t_start = bt[0]
    t_start = max(float(t_start), float(bt[0]))
    j0 = nearest_index(times, t_start)
    j1 = nearest_index(times, T_END)

    idx = np.arange(j0, j1 + 1)
    t_eval = times[idx]
    phi_eval = phi_all[idx]

    if np.any(~np.isfinite(phi_eval)):
        raise RuntimeError("Weak branch does not cover reduced prediction window.")

    psi = PSI_GRID_DEG.copy()
    Xdir = direction_vectors(psi)
    K = len(psi)

    losses = np.empty((len(idx), K), dtype=float)

    d1 = (Xdir[:, 0] > 0.0).astype(float)

    for tt, j in enumerate(idx):
        uW = exp.unit_direction_from_deg(phi_eval[tt])
        d2 = (Xdir @ uW > 0.0).astype(float)

        M1 = M1_hist[j]
        M2 = M2_hist[j]

        Fp = (
            d1[:, None] * (Xdir @ M1.T)
            + d2[:, None] * (Xdir @ M2.T)
        )
        losses[tt] = 0.5 * np.sum((Fp - Xdir) ** 2, axis=1)

    Sred, t_rev_red = phase_from_losses(t_eval, losses)

    return {
        "times": t_eval,
        "indices": idx,
        "psi_deg": psi,
        "Sred": Sred,
        "t_rev_red": t_rev_red,
    }


def nearest_direction_index(psi_grid, target):
    delta = np.abs(wrap_deg(np.asarray(psi_grid) - float(target)))
    return int(np.argmin(delta))


def core_mask(psi_deg, center_deg, eta_blk):
    halfwidth_deg = math.degrees(Z_CORE_MAX * eta_blk)
    theta = signed_angular_delta_deg(psi_deg, center_deg)
    return np.abs(theta) <= halfwidth_deg


def normal_form_probe_psi_deg(p, eta_blk, z=1.0, s=-1.0):
    """
    Convert the predeclared normal-form core probe

        v = eta*z e_p + s sqrt(1-eta^2 z^2) e_q

    to an ambient angle.

    p=1 ordered basis (e1,e2):
        v = (eta*z, s*sqrt(...))

    p=2 ordered basis (e2,e1):
        ambient v = (s*sqrt(...), eta*z)

    The default z=1, s=-1 predicts the canonical sectors near -90 deg
    for p=1 and +180 deg for p=2 without maximizing over OOD behavior.
    """
    x = eta_blk * z
    if abs(x) >= 1.0:
        raise ValueError(
            f"normal-form probe invalid: |eta*z|={abs(x):.4f} >= 1"
        )
    q = s * math.sqrt(1.0 - x * x)

    if p == 1:
        v = np.array([x, q], dtype=float)
    elif p == 2:
        v = np.array([q, x], dtype=float)
    else:
        raise ValueError("p must be 1 or 2")

    return float(wrap_deg(math.degrees(math.atan2(v[1], v[0]))))


def predicted_primary_sector(p, prereg_center_deg, eta_blk, prediction, branch):
    """
    Freeze a theory-driven near-boundary sector BEFORE OOD.

    The sector center is the learned gate boundary. Its core is
        |theta| <= Z_CORE_MAX * eta_blk.

    Two distinct preregistered predictions are stored:
      1. the reduced-model maximum inside that frozen core (descriptive
         prediction from the ID-derived reduced system);
      2. the normal-form representative probe z=1, s=-1, which is selected
         from the expansion itself rather than by maximizing S_red.

    The latter is used for the G-timing diagnostic.
    """
    psi = prediction["psi_deg"]
    Sred = prediction["Sred"]
    t_red = prediction["t_rev_red"]

    mask = core_mask(psi, prereg_center_deg, eta_blk)
    if not np.any(mask):
        raise RuntimeError("Empty preregistered normal-form core.")

    cand = np.where(mask)[0]
    kmax = cand[int(np.argmax(Sred[cand]))]

    probe_psi = normal_form_probe_psi_deg(
        p=p,
        eta_blk=eta_blk,
        z=1.0,
        s=-1.0,
    )
    kprobe = nearest_direction_index(psi, probe_psi)

    return {
        "analysis_center_deg": float(prereg_center_deg),
        "core_halfwidth_deg": math.degrees(Z_CORE_MAX * eta_blk),
        "z_core_max": Z_CORE_MAX,

        "reduced_core_peak_psi_deg": float(psi[kmax]),
        "reduced_core_peak_Sred": float(Sred[kmax]),
        "reduced_core_peak_t_reversal": (
            scalar_or_none(t_red[kmax])
        ),

        "normal_form_probe_z": 1.0,
        "normal_form_probe_s": -1.0,
        "normal_form_probe_psi_deg": float(probe_psi),
        "normal_form_probe_grid_psi_deg": float(psi[kprobe]),
        "normal_form_probe_Sred": float(Sred[kprobe]),
        "normal_form_probe_t_reversal_red": (
            scalar_or_none(t_red[kprobe])
        ),
    }



def save_predicted_phase_csv(path, prediction):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["psi_deg", "Sred", "t_reversal_red"])
        for psi, s, tr in zip(
            prediction["psi_deg"],
            prediction["Sred"],
            prediction["t_rev_red"],
        ):
            w.writerow([
                float(psi),
                float(s),
                float(tr) if np.isfinite(tr) else "",
            ])


# =============================================================================
# Normal-form G timing from ID block history only
# =============================================================================

def ordered_hist(M_hist, p):
    return np.stack([ordered_block_matrix(M, p) for M in M_hist], axis=0)


def finite_difference_matrix_history(times, M_hist):
    times = np.asarray(times, dtype=float)
    return np.gradient(M_hist, times, axis=0, edge_order=2)


def all_neg_to_pos_crossings(times, y):
    times = np.asarray(times, dtype=float)
    y = np.asarray(y, dtype=float)

    crossings = []
    for i in range(len(times) - 1):
        if not np.isfinite(y[i]) or not np.isfinite(y[i + 1]):
            continue
        if y[i] <= 0.0 and y[i + 1] > 0.0:
            if y[i + 1] == y[i]:
                tc = times[i + 1]
            else:
                tc = times[i] - y[i] * (times[i + 1] - times[i]) / (y[i + 1] - y[i])
            crossings.append(float(tc))
    return crossings


def prereg_G_timing(times, M1_hist, eta_blk, psi_deg, target_t, t_valid_start=None):
    """
    Finite-difference implementation of eta^2 G for the p=1, s=-1 core.

    The choice of eta algebraically cancels from eta^2 G once the physical
    probe coordinate x_p is substituted, but eta_blk remains the preregistered
    normal-form scale and determines whether the probe lies at z=O(1).
    """
    B = ordered_hist(M1_hist, 1)
    dB = finite_difference_matrix_history(times, B)

    x = direction_vectors([psi_deg])[0]
    xp = float(x[0])

    # Determine s from sign of q coordinate in ordered (e1,e2) basis.
    s = 1.0 if x[1] >= 0.0 else -1.0

    a = B[:, 0, 0]
    mpq = B[:, 0, 1]
    mqp = B[:, 1, 0]
    mqq = B[:, 1, 1]

    da = dB[:, 0, 0]
    dmpq = dB[:, 0, 1]
    dmqp = dB[:, 1, 0]
    dmqq = dB[:, 1, 1]

    eta2G = (
        ((a - 1.0) * xp + s * mpq)
        * (da * xp + s * dmpq)
        - s * xp * dmqp
        - dmqq
    )

    crossings = all_neg_to_pos_crossings(times, eta2G)
    if t_valid_start is not None:
        crossings = [tc for tc in crossings if tc >= float(t_valid_start)]

    if crossings and target_t is not None and np.isfinite(target_t):
        near = [
            tc for tc in crossings
            if abs(tc - target_t) <= G_SEARCH_HALF_WINDOW
        ]
        if near:
            chosen = min(near, key=lambda tc: abs(tc - target_t))
        else:
            chosen = min(crossings, key=lambda tc: abs(tc - target_t))
    elif crossings:
        chosen = crossings[0]
    else:
        chosen = np.nan

    z = xp / max(eta_blk, 1e-12)

    return {
        "psi_deg": float(psi_deg),
        "eta_blk": float(eta_blk),
        "z": float(z),
        "s": float(s),
        "G_zero_crossing_fd": scalar_or_none(chosen),
        "all_G_zero_crossings_fd": crossings,
        "derivative_method": "np.gradient_on_logged_M_history",
        "validity_start_time": scalar_or_none(t_valid_start) if t_valid_start is not None else None,
    }


# =============================================================================
# State serialization
# =============================================================================

def save_training_state(path, run, branch, blocks, M1_hist, M2_hist):
    bt, bp = valid_branch_arrays(branch)

    np.savez_compressed(
        path,
        time=np.asarray(run["time"], dtype=float),
        U=np.stack(run["U"], axis=0),
        W=np.stack(run["W"], axis=0),
        branch_time=bt,
        branch_phi_deg=bp,
        strong_mask=np.asarray(blocks["strong"], dtype=np.uint8),
        weak_mask=np.asarray(blocks["weak"], dtype=np.uint8),
        remainder_mask=np.asarray(blocks["remainder"], dtype=np.uint8),
        M1=M1_hist,
        M2=M2_hist,
    )


def load_training_state(path):
    z = np.load(path)
    return {
        "time": z["time"],
        "U": z["U"],
        "W": z["W"],
        "branch_time": z["branch_time"],
        "branch_phi_deg": z["branch_phi_deg"],
        "strong_mask": z["strong_mask"].astype(bool),
        "weak_mask": z["weak_mask"].astype(bool),
        "remainder_mask": z["remainder_mask"].astype(bool),
        "M1": z["M1"],
        "M2": z["M2"],
    }



# =============================================================================
# Refreeze the pilot preregistration WITHOUT retraining
# =============================================================================

def branch_rows_from_state(state):
    return [
        {
            "time": float(t),
            "phi_deg": float(phi),
            "branch_exists": 1,
        }
        for t, phi in zip(state["branch_time"], state["branch_phi_deg"])
    ]


def run_view_from_state(state):
    """
    Minimal run-like object needed by reduced_prediction.
    """
    return {
        "time": np.asarray(state["time"], dtype=float),
        "U": state["U"],
        "W": state["W"],
    }


def refreeze_cell(source_cdir, target_root):
    old = read_json(source_cdir / "preregistration.json")
    cid = old["cell_id"]

    target_cdir = target_root / "cells" / cid
    target_cdir.mkdir(parents=True, exist_ok=True)

    state = load_training_state(
        source_cdir / old["files"]["training_state"]
    )
    run = run_view_from_state(state)
    branch = branch_rows_from_state(state)

    bt, bp = valid_branch_arrays(branch)
    phi_block = interp_branch(branch, [T_BLOCK])[0]

    M1_hist = np.asarray(state["M1"], dtype=float)
    M2_hist = np.asarray(state["M2"], dtype=float)

    j_block = nearest_index(state["time"], T_BLOCK)
    M1_block = M1_hist[j_block]
    M2_block = M2_hist[j_block]

    stat1 = block_alignment_stats(M1_block, p=1)
    stat2 = block_alignment_stats(M2_block, p=2)

    postspec = compute_post_specialization_time(
        state["time"],
        state["U"],
        state["W"],
        state["strong_mask"],
        state["weak_mask"],
        branch_birth=float(bt[0]),
        t_block=T_BLOCK,
        threshold=POSTSPEC_MASS_SHARE_FRACTION,
    )

    prediction = reduced_prediction(
        run,
        branch,
        blocks=None,
        M1_hist=M1_hist,
        M2_hist=M2_hist,
        t_start=postspec["t_W"],
    )

    p1_center = -90.0
    p2_center = float(wrap_deg(phi_block + 90.0))

    p1_sector = predicted_primary_sector(
        1, p1_center, stat1["eta_blk"], prediction, branch
    )
    p2_sector = predicted_primary_sector(
        2, p2_center, stat2["eta_blk"], prediction, branch
    )

    p1_G = prereg_G_timing(
        np.asarray(state["time"], dtype=float),
        M1_hist,
        stat1["eta_blk"],
        p1_sector["normal_form_probe_psi_deg"],
        p1_sector["normal_form_probe_t_reversal_red"],
        t_valid_start=postspec["t_W"],
    )

    # Copy immutable training/branch artifacts; recompute only theory-facing
    # predictions and metadata.
    source_state = source_cdir / old["files"]["training_state"]
    target_state = target_cdir / "training_state.npz"
    shutil.copy2(source_state, target_state)

    source_branch = source_cdir / old["files"]["weak_branch"]
    target_branch = target_cdir / "weak_branch.csv"
    shutil.copy2(source_branch, target_branch)

    pred_path = target_cdir / "predicted_phase_diagram.csv"
    save_predicted_phase_csv(pred_path, prediction)

    mu2 = float(old["config"]["mu2"])
    sigma = float(old["config"]["sigma1"])
    eta_param_strong = float(mu2 * sigma / (MU1 ** 2))

    new = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "cell_id": cid,
        "config": old["config"],
        "protocol": {
            "t_block": T_BLOCK,
            "t_end": T_END,
            "branch_search_start": old["protocol"]["branch_search_start"],
            "branch_stride": old["protocol"]["branch_stride"],
            "block_tol_deg": old["protocol"]["block_tol_deg"],
            "psi_step_deg": PSI_STEP_DEG,
            "z_core_max": Z_CORE_MAX,
            "post_specialization_mass_share_fraction": POSTSPEC_MASS_SHARE_FRACTION,
            "prediction_validity_start": postspec["t_W"],
            "full_ood_evaluated": False,
            "refrozen_from_pilot_without_retraining": True,
        },
        "correction": {
            "reason": (
                "Pilot reduced-model prediction began at weak-branch birth even "
                "though frozen block membership was defined at t_block, allowing "
                "eventual weak-block members to be extrapolated backward before "
                "recruitment. Corrected before any full-ReLU OOD evaluation."
            ),
            "source_preregistration_json_sha256": sha256_file(
                source_cdir / "preregistration.json"
            ),
        },
        "post_specialization": postspec,
        "weak_branch": {
            "birth_time": float(bt[0]),
            "phi_birth_deg": float(bp[0]),
            "phi_block_deg": float(phi_block),
            "phi_min_deg": float(np.min(bp)),
            "phi_max_deg": float(np.max(bp)),
            "boundary_minus_90_at_block_deg": float(wrap_deg(phi_block - 90.0)),
            "boundary_plus_90_at_block_deg": float(wrap_deg(phi_block + 90.0)),
        },
        "blocks": old["blocks"],
        "p1_strong_block": stat1,
        "p2_weak_block": stat2,
        "eta_param_strong": eta_param_strong,
        "predicted_gate_boundaries_deg": {
            "p1": [-90.0, 90.0],
            "p2_at_t_block": [
                float(wrap_deg(phi_block - 90.0)),
                float(wrap_deg(phi_block + 90.0)),
            ],
        },
        "predicted_primary_p1": p1_sector,
        "predicted_primary_p2": p2_sector,
        "predicted_p1_G_timing": p1_G,
        "files": {
            "training_state": target_state.name,
            "weak_branch": target_branch.name,
            "predicted_phase_diagram": pred_path.name,
        },
    }

    write_json(target_cdir / "preregistration.json", new)

    print()
    print("=" * 80)
    print(f"REFREEZE CELL {cid}")
    print("=" * 80)
    print(
        f"t_W={postspec['t_W']:.3f} "
        f"(strong={postspec['t_W_strong']:.3f}, weak={postspec['t_W_weak']:.3f})"
    )
    print(
        f"mass-share ratios at t_W: "
        f"strong={postspec['strong_ratio_t_W']:.3f}, "
        f"weak={postspec['weak_ratio_t_W']:.3f}"
    )
    print(
        f"p1 z=1: psi={p1_sector['normal_form_probe_psi_deg']:.2f}, "
        f"Sred={p1_sector['normal_form_probe_Sred']:.6e}, "
        f"tred={p1_sector['normal_form_probe_t_reversal_red']}"
    )
    print(
        f"p2 z=1: psi={p2_sector['normal_form_probe_psi_deg']:.2f}, "
        f"Sred={p2_sector['normal_form_probe_Sred']:.6e}, "
        f"tred={p2_sector['normal_form_probe_t_reversal_red']}"
    )

    return new


def refreeze_pilot():
    source_root = PILOT_PREREG_ROOT
    target_root = PATCHED_PREREG_ROOT

    if not source_root.exists():
        raise RuntimeError(
            f"Pilot preregistration not found at {source_root}. "
            "Run the original preregistration first or restore that directory."
        )

    if target_root.exists():
        raise RuntimeError(
            f"{target_root} already exists. Refusing to overwrite a corrected "
            "preregistration."
        )

    pilot_manifest = verify_manifest(source_root)

    target_root.mkdir(parents=True, exist_ok=False)

    correction = {
        "created_utc": utc_now(),
        "source_pilot_combined_sha256": pilot_manifest["combined_sha256"],
        "source_pilot_root": str(source_root),
        "full_ood_seen_before_correction": False,
        "correction": (
            "Introduce ID-only post-specialization validity time t_W based on "
            "persistent recruitment of frozen strong/weak cohorts. Recompute "
            "all reduced OOD predictions only on [t_W,T]. Reuse saved training "
            "trajectories; no retraining and no full-ReLU OOD evaluation."
        ),
        "mass_share_threshold": POSTSPEC_MASS_SHARE_FRACTION,
    }
    write_json(target_root / "correction_record.json", correction)

    # Copy and update the machine-readable protocol.
    old_protocol = read_json(source_root / "protocol.json")
    old_protocol["created_utc"] = utc_now()
    old_protocol["schema_version"] = 2
    old_protocol["source_pilot_combined_sha256"] = pilot_manifest["combined_sha256"]
    old_protocol["post_specialization_validity"] = {
        "mass_share_threshold": POSTSPEC_MASS_SHARE_FRACTION,
        "definition": (
            "t_W=max(branch_birth,t_W,strong,t_W,weak), where t_W,p is the "
            "first time the frozen block-p projected-mass share is at least "
            "threshold times its t_block share and remains so through t_block"
        ),
    }
    write_json(target_root / "protocol.json", old_protocol)

    for mu2 in MU2_GRID:
        for sigma in SIGMA_GRID:
            cid = cell_id(mu2, sigma)
            refreeze_cell(source_root / "cells" / cid, target_root)

    manifest = build_manifest(target_root)

    print()
    print("=" * 80)
    print("CORRECTED PREREGISTRATION FROZEN — NO RETRAINING")
    print("=" * 80)
    print(f"source pilot hash: {pilot_manifest['combined_sha256']}")
    print(f"new combined hash: {manifest['combined_sha256']}")
    print(f"timestamp: {manifest['created_utc']}")
    print(f"path: {target_root.resolve()}")
    print()
    print("The pilot preregistration remains unchanged.")
    print("No full-ReLU OOD phase diagram has been evaluated.")
    print("Now run:")
    print("    python3 swing_grid_preregister_patched.py evaluate")


# =============================================================================
# Stage 1: preregister
# =============================================================================

def make_cfg(mu2, sigma):
    return replace(
        Config(),
        mu1=MU1,
        mu2=float(mu2),
        sigma1=float(sigma),
        sigma2=float(sigma),
        width=200,
        n_per_cluster=2000,
        epsilon=1e-3,
        dt=2e-4,
        max_time=T_END,
        seed=0,
    )


def preregister_cell(mu2, sigma, root):
    cid = cell_id(mu2, sigma)
    cdir = root / "cells" / cid
    cdir.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg(mu2, sigma)

    print()
    print("=" * 80)
    print(f"PREREGISTER CELL {cid}")
    print("=" * 80)

    run = exp.run_relu(cfg, log_every=50)

    branch = track_weak_branch(run)
    bt, bp = valid_branch_arrays(branch)

    phi_block = interp_branch(branch, [T_BLOCK])[0]
    if not np.isfinite(phi_block):
        raise RuntimeError(
            f"{cid}: weak branch is not defined at t_block={T_BLOCK}."
        )

    blocks = exp.define_final_blocks_from_branch(
        run,
        phi_block_deg=phi_block,
        t_block=T_BLOCK,
        tol_deg=BLOCK_TOL_DEG,
    )

    M1_hist, M2_hist = block_histories(run, blocks)

    j_block = nearest_index(run["time"], T_BLOCK)
    M1_block = M1_hist[j_block]
    M2_block = M2_hist[j_block]

    stat1 = block_alignment_stats(M1_block, p=1)
    stat2 = block_alignment_stats(M2_block, p=2)

    eta_param_strong = float(mu2 * sigma / (MU1 ** 2))

    postspec = compute_post_specialization_time(
        run["time"],
        run["U"],
        run["W"],
        blocks["strong"],
        blocks["weak"],
        branch_birth=float(bt[0]),
        t_block=T_BLOCK,
        threshold=POSTSPEC_MASS_SHARE_FRACTION,
    )

    prediction = reduced_prediction(
        run, branch, blocks, M1_hist, M2_hist,
        t_start=postspec["t_W"],
    )

    # Preliminary centers frozen from ID gate geometry.
    p1_prelim_center = -90.0
    p2_prelim_center = float(wrap_deg(phi_block + 90.0))

    p1_sector = predicted_primary_sector(
        1, p1_prelim_center, stat1["eta_blk"], prediction, branch
    )
    p2_sector = predicted_primary_sector(
        2, p2_prelim_center, stat2["eta_blk"], prediction, branch
    )

    p1_G = prereg_G_timing(
        np.asarray(run["time"], dtype=float),
        M1_hist,
        stat1["eta_blk"],
        p1_sector["normal_form_probe_psi_deg"],
        p1_sector["normal_form_probe_t_reversal_red"],
        t_valid_start=postspec["t_W"],
    )

    # Save all preregistered objects BEFORE any full OOD evaluation.
    state_path = cdir / "training_state.npz"
    save_training_state(
        state_path, run, branch, blocks, M1_hist, M2_hist
    )

    branch_path = cdir / "weak_branch.csv"
    save_branch_csv(branch_path, branch)

    pred_path = cdir / "predicted_phase_diagram.csv"
    save_predicted_phase_csv(pred_path, prediction)

    prereg = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "cell_id": cid,
        "config": asdict(cfg),
        "protocol": {
            "t_block": T_BLOCK,
            "t_end": T_END,
            "branch_search_start": BRANCH_SEARCH_START,
            "branch_stride": BRANCH_STRIDE,
            "block_tol_deg": BLOCK_TOL_DEG,
            "psi_step_deg": PSI_STEP_DEG,
            "z_core_max": Z_CORE_MAX,
            "post_specialization_mass_share_fraction": POSTSPEC_MASS_SHARE_FRACTION,
            "prediction_validity_start": postspec["t_W"],
            "full_ood_evaluated": False,
        },
        "post_specialization": postspec,
        "weak_branch": {
            "birth_time": float(bt[0]),
            "phi_birth_deg": float(bp[0]),
            "phi_block_deg": float(phi_block),
            "phi_min_deg": float(np.min(bp)),
            "phi_max_deg": float(np.max(bp)),
            "boundary_minus_90_at_block_deg": float(wrap_deg(phi_block - 90.0)),
            "boundary_plus_90_at_block_deg": float(wrap_deg(phi_block + 90.0)),
        },
        "blocks": {
            "M1_ambient_t_block": M1_block.tolist(),
            "M2_ambient_t_block": M2_block.tolist(),
            "strong_mass_fraction": blocks["strong_mass_fraction"],
            "weak_mass_fraction": blocks["weak_mass_fraction"],
            "remainder_mass_fraction": blocks["remainder_mass_fraction"],
        },
        "p1_strong_block": stat1,
        "p2_weak_block": stat2,
        "eta_param_strong": eta_param_strong,
        "predicted_gate_boundaries_deg": {
            "p1": [-90.0, 90.0],
            "p2_at_t_block": [
                float(wrap_deg(phi_block - 90.0)),
                float(wrap_deg(phi_block + 90.0)),
            ],
        },
        "predicted_primary_p1": p1_sector,
        "predicted_primary_p2": p2_sector,
        "predicted_p1_G_timing": p1_G,
        "files": {
            "training_state": state_path.name,
            "weak_branch": branch_path.name,
            "predicted_phase_diagram": pred_path.name,
        },
    }

    prereg_path = cdir / "preregistration.json"
    write_json(prereg_path, prereg)

    print(
        f"t_W={postspec['t_W']:.3f} "
        f"(strong={postspec['t_W_strong']:.3f}, weak={postspec['t_W_weak']:.3f})"
    )
    print(
        f"phi_W(t_block)={phi_block:.3f} deg | "
        f"eta1_blk={stat1['eta_blk']:.5f} | "
        f"eta1_gm={stat1['eta_gm']:.5f} | "
        f"eta_param={eta_param_strong:.5f}"
    )
    print(
        f"eta2_blk={stat2['eta_blk']:.5f} | "
        f"eta2_gm={stat2['eta_gm']:.5f}"
    )
    print(
        f"pred p1 z=1 probe: psi={p1_sector['normal_form_probe_psi_deg']:.2f}, "
        f"Sred={p1_sector['normal_form_probe_Sred']:.6e}, "
        f"tred={p1_sector['normal_form_probe_t_reversal_red']}"
    )
    print(
        f"pred p2 z=1 probe: psi={p2_sector['normal_form_probe_psi_deg']:.2f}, "
        f"Sred={p2_sector['normal_form_probe_Sred']:.6e}, "
        f"tred={p2_sector['normal_form_probe_t_reversal_red']}"
    )

    return cid


def build_manifest(root):
    """
    Hash every preregistered artifact except the manifest itself.
    """
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "manifest.json":
            continue
        rel = str(path.relative_to(root))
        files.append({
            "path": rel,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })

    combined_payload = json.dumps(
        [{"path": x["path"], "sha256": x["sha256"]} for x in files],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    combined_hash = hashlib.sha256(combined_payload).hexdigest()

    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "purpose": "Frozen pre-OOD preregistration for 3x3 Swing-by scaling test",
        "full_ood_evaluated": False,
        "combined_sha256": combined_hash,
        "files": files,
    }

    write_json(root / "manifest.json", manifest)
    return manifest


def preregister_all():
    root = OUTPUT_ROOT / "preregistration"
    if root.exists():
        raise RuntimeError(
            f"{root} already exists. Refusing to overwrite a preregistration. "
            "Rename/archive it deliberately if you truly want a new run."
        )

    root.mkdir(parents=True, exist_ok=False)

    spec = {
        "created_utc": utc_now(),
        "grid": {
            "mu1": MU1,
            "mu2": list(MU2_GRID),
            "sigma": list(SIGMA_GRID),
        },
        "frozen_primary_tests": [
            "Smax_p / eta_blk_p^2 is O(1) and approximately stable across nearby cells",
            "S_p/eta_blk_p^2 approximately collapses vs theta_p/eta_blk_p in the preregistered normal-form core",
            "eta1_blk vs mu2*sigma/mu1^2",
            "eta1_gm vs mu2*sigma/mu1^2",
        ],
        "secondary_test": (
            "For the reduced-predicted p=1 core peak, compare finite-difference "
            "normal-form G zero crossing, reduced t_reversal, and later full t_reversal."
        ),
        "basis_convention": {
            "p1_order": ["e1", "e2"],
            "p2_order": ["e2", "e1"],
        },
        "eta_definition": (
            "max(|m_pq|/|m_pp|, |m_qp|/|m_pp|, sqrt(|m_qq|/|m_pp|))"
        ),
        "eta_gm_definition": "sqrt(|m_pq*m_qp|)/|m_pp|",
        "psi_step_deg": PSI_STEP_DEG,
        "z_core_max": Z_CORE_MAX,
        "t_block": T_BLOCK,
        "post_specialization_validity": {
            "mass_share_threshold": POSTSPEC_MASS_SHARE_FRACTION,
            "definition": (
                "t_W=max(branch_birth,t_W,strong,t_W,weak), with persistent "
                "frozen-cohort projected-mass share threshold through t_block"
            ),
        },
        "seed": 0,
    }
    write_json(root / "protocol.json", spec)

    for mu2 in MU2_GRID:
        for sigma in SIGMA_GRID:
            preregister_cell(mu2, sigma, root)

    manifest = build_manifest(root)

    print()
    print("=" * 80)
    print("PREREGISTRATION FROZEN")
    print("=" * 80)
    print(f"timestamp: {manifest['created_utc']}")
    print(f"combined SHA-256: {manifest['combined_sha256']}")
    print(f"path: {root.resolve()}")
    print()
    print("No full-ReLU OOD phase diagram has been evaluated.")
    print("Run the evaluation only with:")
    print("    python3 swing_grid_preregister.py evaluate")


# =============================================================================
# Manifest verification
# =============================================================================

def verify_manifest(root):
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Missing preregistration manifest.")

    manifest = read_json(manifest_path)

    failures = []
    actual_pairs = []

    for item in manifest["files"]:
        path = root / item["path"]
        if not path.exists():
            failures.append(f"missing: {item['path']}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != item["sha256"]:
            failures.append(
                f"hash mismatch: {item['path']} "
                f"expected={item['sha256']} actual={actual_hash}"
            )
        actual_pairs.append({
            "path": item["path"],
            "sha256": actual_hash,
        })

    combined_payload = json.dumps(
        sorted(actual_pairs, key=lambda x: x["path"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    combined = hashlib.sha256(combined_payload).hexdigest()

    if combined != manifest["combined_sha256"]:
        failures.append(
            "combined hash mismatch: "
            f"expected={manifest['combined_sha256']} actual={combined}"
        )

    if failures:
        raise RuntimeError(
            "Preregistration verification FAILED:\n  - "
            + "\n  - ".join(failures)
        )

    print("Preregistration manifest verification: PASS")
    print(f"Frozen combined SHA-256: {manifest['combined_sha256']}")
    return manifest


# =============================================================================
# Stage 2: full OOD evaluation
# =============================================================================

def evaluate_full_phase(state, t_start):
    times = state["time"]
    Uhist = state["U"]
    Whist = state["W"]

    bt = state["branch_time"]
    bp = state["branch_phi_deg"]

    t_start = max(float(t_start), float(bt[0]))
    j0 = nearest_index(times, t_start)
    j1 = nearest_index(times, T_END)
    idx = np.arange(j0, j1 + 1)
    t_eval = times[idx]

    psi = PSI_GRID_DEG.copy()
    Xdir = direction_vectors(psi)

    losses = np.empty((len(idx), len(psi)), dtype=float)

    for tt, j in enumerate(idx):
        U = Uhist[j]
        W = Whist[j]
        F = exp.relu_forward(U, W, Xdir)
        losses[tt] = 0.5 * np.sum((F - Xdir) ** 2, axis=1)

    Sactual, t_rev_actual = phase_from_losses(t_eval, losses)

    return {
        "times": t_eval,
        "psi_deg": psi,
        "Sactual": Sactual,
        "t_rev_actual": t_rev_actual,
    }


def load_predicted_phase(path):
    psi, S, tr = [], [], []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            psi.append(float(row["psi_deg"]))
            S.append(float(row["Sred"]))
            tr.append(float(row["t_reversal_red"]) if row["t_reversal_red"] else np.nan)

    return {
        "psi_deg": np.asarray(psi),
        "Sred": np.asarray(S),
        "t_rev_red": np.asarray(tr),
    }


def peak_in_frozen_core(phase, center_deg, eta_blk):
    psi = phase["psi_deg"]
    S = phase["Sactual"]
    tr = phase["t_rev_actual"]

    mask = core_mask(psi, center_deg, eta_blk)
    cand = np.where(mask)[0]
    if len(cand) == 0:
        return None

    k = cand[int(np.argmax(S[cand]))]
    theta_deg = float(signed_angular_delta_deg([psi[k]], center_deg)[0])

    return {
        "psi_deg": float(psi[k]),
        "Smax": float(S[k]),
        "t_reversal": scalar_or_none(tr[k]),
        "theta_deg": theta_deg,
        "theta_over_eta": math.radians(theta_deg) / max(eta_blk, 1e-12),
        "Smax_over_eta2": float(S[k] / max(eta_blk ** 2, 1e-12)),
    }


def save_actual_phase_csv(path, actual, predicted):
    if not np.allclose(actual["psi_deg"], predicted["psi_deg"]):
        raise RuntimeError("Actual and preregistered psi grids differ.")

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "psi_deg",
            "Sactual",
            "t_reversal_actual",
            "Sred_preregistered",
            "t_reversal_red_preregistered",
        ])
        for vals in zip(
            actual["psi_deg"],
            actual["Sactual"],
            actual["t_rev_actual"],
            predicted["Sred"],
            predicted["t_rev_red"],
        ):
            psi, sa, ta, sr, tr = vals
            w.writerow([
                float(psi),
                float(sa),
                float(ta) if np.isfinite(ta) else "",
                float(sr),
                float(tr) if np.isfinite(tr) else "",
            ])


def evaluate_cell(cdir, eval_root):
    prereg = read_json(cdir / "preregistration.json")
    cid = prereg["cell_id"]

    print()
    print("=" * 80)
    print(f"EVALUATE CELL {cid}")
    print("=" * 80)

    state = load_training_state(cdir / prereg["files"]["training_state"])
    predicted = load_predicted_phase(cdir / prereg["files"]["predicted_phase_diagram"])
    t_valid = float(prereg["post_specialization"]["t_W"])
    actual = evaluate_full_phase(state, t_start=t_valid)

    odir = eval_root / "cells" / cid
    odir.mkdir(parents=True, exist_ok=True)

    save_actual_phase_csv(odir / "actual_vs_predicted_phase.csv", actual, predicted)

    p1 = prereg["p1_strong_block"]
    p2 = prereg["p2_weak_block"]

    p1_center = prereg["predicted_primary_p1"]["analysis_center_deg"]
    p2_center = prereg["predicted_primary_p2"]["analysis_center_deg"]

    p1_peak = peak_in_frozen_core(actual, p1_center, p1["eta_blk"])
    p2_peak = peak_in_frozen_core(actual, p2_center, p2["eta_blk"])

    # Global phase agreement.
    corr = (
        float(np.corrcoef(actual["Sactual"], predicted["Sred"])[0, 1])
        if np.std(actual["Sactual"]) > 0 and np.std(predicted["Sred"]) > 0
        else np.nan
    )
    mae = float(np.mean(np.abs(actual["Sactual"] - predicted["Sred"])))

    # Full reversal at the preregistered p=1 predicted peak probe.
    p1_probe = prereg["predicted_primary_p1"]["normal_form_probe_psi_deg"]
    kprobe = nearest_direction_index(actual["psi_deg"], p1_probe)
    full_t_probe = (
        float(actual["t_rev_actual"][kprobe])
        if np.isfinite(actual["t_rev_actual"][kprobe])
        else None
    )

    result = {
        "schema_version": 1,
        "evaluated_utc": utc_now(),
        "cell_id": cid,
        "mu2": prereg["config"]["mu2"],
        "sigma": prereg["config"]["sigma1"],
        "eta_param_strong": prereg["eta_param_strong"],
        "prediction_validity_start_t_W": t_valid,
        "p1": {
            "eta_blk": p1["eta_blk"],
            "eta_gm": p1["eta_gm"],
            "center_deg_frozen": p1_center,
            "actual_core_peak": p1_peak,
            "normal_form_probe_psi_deg": p1_probe,
            "predicted_t_reversal_at_probe": prereg["predicted_primary_p1"]["normal_form_probe_t_reversal_red"],
            "G_zero_crossing_fd": prereg["predicted_p1_G_timing"]["G_zero_crossing_fd"],
            "full_t_reversal_at_predicted_probe": full_t_probe,
        },
        "p2": {
            "eta_blk": p2["eta_blk"],
            "eta_gm": p2["eta_gm"],
            "center_deg_frozen": p2_center,
            "actual_core_peak": p2_peak,
            "normal_form_probe_psi_deg": prereg["predicted_primary_p2"]["normal_form_probe_psi_deg"],
            "predicted_t_reversal_at_probe": prereg["predicted_primary_p2"]["normal_form_probe_t_reversal_red"],
        },
        "phase_agreement": {
            "global_corr_Sactual_Sred": scalar_or_none(corr),
            "global_MAE": mae,
        },
    }

    write_json(odir / "evaluation.json", result)

    print(
        f"p1: eta_blk={p1['eta_blk']:.5f}, "
        f"Smax/eta^2={p1_peak['Smax_over_eta2'] if p1_peak else np.nan:.4f}"
    )
    print(
        f"p2: eta_blk={p2['eta_blk']:.5f}, "
        f"Smax/eta^2={p2_peak['Smax_over_eta2'] if p2_peak else np.nan:.4f}"
    )
    print(f"global corr(Sactual,Sred)={corr:.4f}, MAE={mae:.3e}")

    return result, actual


# =============================================================================
# Aggregate evaluation plots
# =============================================================================

def aggregate_results(eval_root, results, actual_by_cell, prereg_root):
    # Summary CSV.
    rows = []
    collapse_p1 = []
    collapse_p2 = []

    for res in results:
        cid = res["cell_id"]
        prereg = read_json(prereg_root / "cells" / cid / "preregistration.json")
        actual = actual_by_cell[cid]

        for p in (1, 2):
            key = f"p{p}"
            block = prereg["p1_strong_block"] if p == 1 else prereg["p2_weak_block"]
            center = res[key]["center_deg_frozen"]
            eta = res[key]["eta_blk"]

            peak = res[key]["actual_core_peak"]
            rows.append({
                "cell_id": cid,
                "mu2": res["mu2"],
                "sigma": res["sigma"],
                "p": p,
                "eta_blk": eta,
                "eta_gm": res[key]["eta_gm"],
                "eta_param_strong": res["eta_param_strong"],
                "center_deg_frozen": center,
                "Smax_actual": peak["Smax"] if peak else np.nan,
                "Smax_over_eta2": peak["Smax_over_eta2"] if peak else np.nan,
                "psi_peak_actual": peak["psi_deg"] if peak else np.nan,
                "t_reversal_actual": peak["t_reversal"] if peak else np.nan,
                "global_corr_Sactual_Sred": res["phase_agreement"]["global_corr_Sactual_Sred"],
            })

            theta_deg = signed_angular_delta_deg(actual["psi_deg"], center)
            z = np.radians(theta_deg) / max(eta, 1e-12)
            scaled_S = actual["Sactual"] / max(eta ** 2, 1e-12)
            mask = np.abs(z) <= Z_CORE_MAX

            rec = {
                "cell_id": cid,
                "z": z[mask],
                "scaled_S": scaled_S[mask],
            }
            if p == 1:
                collapse_p1.append(rec)
            else:
                collapse_p2.append(rec)

    summary_path = eval_root / "scaling_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Amplitude ratio by cell, one chart per block.
    for p in (1, 2):
        rp = [r for r in rows if r["p"] == p]
        labels = [r["cell_id"] for r in rp]
        y = [r["Smax_over_eta2"] for r in rp]

        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(len(y)), y, marker="o")
        plt.xticks(range(len(y)), labels, rotation=45, ha="right")
        plt.ylabel(r"$S_{\max,p}/(\eta_p^{blk})^2$")
        plt.xlabel("grid cell")
        plt.title(f"Amplitude scaling diagnostic, block p={p}")
        fig.tight_layout()
        fig.savefig(eval_root / f"amplitude_scaling_p{p}.pdf")
        plt.close(fig)

    # Profile collapse, separately p=1 and p=2.
    for p, curves in ((1, collapse_p1), (2, collapse_p2)):
        fig = plt.figure(figsize=(8, 5))
        for rec in curves:
            order = np.argsort(rec["z"])
            plt.plot(
                rec["z"][order],
                rec["scaled_S"][order],
                label=rec["cell_id"],
            )
        plt.xlabel(r"$\theta_p/\eta_p^{blk}$")
        plt.ylabel(r"$S_p/(\eta_p^{blk})^2$")
        plt.title(f"Preregistered near-boundary profile collapse, p={p}")
        plt.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(eval_root / f"profile_collapse_p{p}.pdf")
        plt.close(fig)

    # Strong-block theory predictor: two SEPARATE scatter plots.
    strong = [r for r in rows if r["p"] == 1]
    x = np.asarray([r["eta_param_strong"] for r in strong])
    y_blk = np.asarray([r["eta_blk"] for r in strong])
    y_gm = np.asarray([r["eta_gm"] for r in strong])

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(x, y_blk)
    lo = min(np.min(x), np.min(y_blk))
    hi = max(np.max(x), np.max(y_blk))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(r"$\mu_2\sigma/\mu_1^2$")
    plt.ylabel(r"$\eta_1^{blk}$")
    plt.title("Strong block: max leakage scale vs parameter formula")
    fig.tight_layout()
    fig.savefig(eval_root / "strong_eta_blk_vs_param.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(x, y_gm)
    lo = min(np.min(x), np.min(y_gm))
    hi = max(np.max(x), np.max(y_gm))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(r"$\mu_2\sigma/\mu_1^2$")
    plt.ylabel(r"$\eta_1^{gm}$")
    plt.title("Strong block: geometric-mean leakage vs parameter formula")
    fig.tight_layout()
    fig.savefig(eval_root / "strong_eta_gm_vs_param.pdf")
    plt.close(fig)

    # Secondary G timing summary.
    g_rows = []
    for res in results:
        g_rows.append({
            "cell_id": res["cell_id"],
            "mu2": res["mu2"],
            "sigma": res["sigma"],
            "p1_predicted_probe_psi": res["p1"]["normal_form_probe_psi_deg"],
            "t_G_zero_fd": res["p1"]["G_zero_crossing_fd"],
            "t_rev_red_predicted": res["p1"]["predicted_t_reversal_at_probe"],
            "t_rev_full_at_predicted_probe": res["p1"]["full_t_reversal_at_predicted_probe"],
        })

    with (eval_root / "G_timing_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(g_rows[0].keys()))
        w.writeheader()
        w.writerows(g_rows)

    print()
    print("=" * 80)
    print("AGGREGATE OUTPUTS")
    print("=" * 80)
    print(f"summary: {summary_path.resolve()}")
    print(f"plots:   {eval_root.resolve()}")


def evaluate_all():
    prereg_root = PATCHED_PREREG_ROOT
    manifest = verify_manifest(prereg_root)

    eval_root = PATCHED_EVAL_ROOT
    if eval_root.exists():
        raise RuntimeError(
            f"{eval_root} already exists. Refusing to overwrite evaluation."
        )
    eval_root.mkdir(parents=True, exist_ok=False)

    eval_meta = {
        "started_utc": utc_now(),
        "verified_preregistration_combined_sha256": manifest["combined_sha256"],
    }
    write_json(eval_root / "evaluation_started.json", eval_meta)

    results = []
    actual_by_cell = {}

    for mu2 in MU2_GRID:
        for sigma in SIGMA_GRID:
            cid = cell_id(mu2, sigma)
            cdir = prereg_root / "cells" / cid
            result, actual = evaluate_cell(cdir, eval_root)
            results.append(result)
            actual_by_cell[cid] = actual

    aggregate_results(
        eval_root,
        results,
        actual_by_cell,
        prereg_root,
    )

    completed = {
        "completed_utc": utc_now(),
        "verified_preregistration_combined_sha256": manifest["combined_sha256"],
        "number_of_cells": len(results),
    }
    write_json(eval_root / "evaluation_completed.json", completed)

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"frozen prereg hash: {manifest['combined_sha256']}")
    print(f"evaluation path: {eval_root.resolve()}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["self-test", "preregister", "refreeze", "evaluate"],
    )
    args = parser.parse_args()

    if args.stage == "self-test":
        self_test_basis_and_eta()
        return

    if args.stage == "preregister":
        self_test_basis_and_eta()
        preregister_all()
        return

    if args.stage == "refreeze":
        self_test_basis_and_eta()
        refreeze_pilot()
        return

    if args.stage == "evaluate":
        self_test_basis_and_eta()
        evaluate_all()
        return


if __name__ == "__main__":
    main()
