#!/usr/bin/env python3
"""
fixed_gate_level1.py

Level-1 diagnostic: can fixed-gate least-squares geometry predict the learned
concept-block leakage from hyperparameters / learned gate geometry?

This is deliberately NOT a new training run.  It consumes the already-frozen
3x3 preregistration/evaluation artifacts:

  swing_grid_preregistered/
    preregistration_v1_postspecialization/
    evaluation_v1_postspecialization/

and solves the fixed-gate reduced least-squares problem

    min_{M1,M2} 1/2 E || g1 M1 x + g2 M2 x - x ||^2,

whose normal equations are

    M1 S1 + M2 H = S1,
    M1 H  + M2 S2 = S2,

with

    S1 = E[g1 x x^T],
    S2 = E[g2 x x^T],
    H  = E[g1 g2 x x^T].

Equivalently,

    [M1 M2] [[S1,H],[H,S2]] = [S1,S2].

The Gaussian moments are evaluated deterministically with tensor-product
Gauss-Hermite quadrature over the known SIM population distribution; they are
NOT estimated from the training samples.

VARIANTS
--------
A. ideal:
      g1 = 1{x1 > 0}
      g2 = 1{x2 > 0}  (phi2 = 90 deg)

   This is hyperparameter-only.

B1. measured_root:
      g1 = 1{x1 > 0}
      g2 = 1{u(phi_root(t))^T x > 0},

   where phi_root(t) is the already-measured stable V_true weak branch.

B2. measured_cohort:
      g1 = 1{x1 > 0}
      g2 = 1{u(phi_cohort(t))^T x > 0},

   where phi_cohort(t) is the projected-mass-weighted U-angle of the frozen
   weak block at that same time.

C. predicted_gate:
   intentionally NOT evaluated here because no hyperparameter-only prediction
   phi_hat_W(mu1,mu2,sigma,t) has yet been derived.  The output records this
   link as pending rather than inventing a predictor.

TIMES
-----
Compare the equilibrium block geometry against the actual network blocks at

    t_W,
    t_rev,p,
    t_block = 4,
    t = 10,

where t_rev,p is the actual full-ReLU reversal time of the preregistered core
peak for block p from the completed 3x3 evaluation.

FROZEN DECISION RULE
--------------------
Best case:
  ideal fixed point gets comparative statics and approximately matches
  eta_blk at t_rev.

Useful conditional case:
  measured-gate fixed point matches t_rev but ideal does not; gate-angle
  prediction is the principal missing link.

Partial success:
  equilibrium gets t=10 and/or parameter ordering but not t_rev; transient
  dynamics are required for quantitative Swing-by prediction.

Stop case:
  even measured-gate equilibrium gets the mu2/sigma comparative statics wrong;
  do not pursue equilibrium further in the current paper.

NEGATIVE PREDICTION FROZEN BEFORE THE SOLVE
-------------------------------------------
The isolated leading balance

    m21^(1) ~ -(mu1+mu2) sigma / (sqrt(2pi) mu1^2)

is predicted to FAIL quantitatively and in comparative statics.  The full
coupled normal equations are the object being tested.

AUDITABLE TWO-STAGE USAGE
-------------------------
    python3 fixed_gate_level1.py self-test
    python3 fixed_gate_level1.py freeze
    python3 fixed_gate_level1.py run

`freeze` writes the hypothesis/decision-rule record and hashes the source grid
artifacts before any fixed-point result is computed.

`run` verifies that frozen record, then performs the moment solves.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Paths / frozen protocol
# =============================================================================

SOURCE_PREREG = Path(
    "swing_grid_preregistered/preregistration_v1_postspecialization"
)
SOURCE_EVAL = Path(
    "swing_grid_preregistered/evaluation_v1_postspecialization"
)
OUT_ROOT = Path("swing_grid_level1_fixed_point")

MU1 = 3.0
T_BLOCK = 4.0
T_LATE = 10.0

# Deterministic population-moment quadrature.
GH_ORDER = 100

# Numerical condition threshold for the 4x4 gated Gram matrix.
COND_MAX = 1e12


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )


def read_json(path: Path):
    return json.loads(path.read_text())


def wrap_deg(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def nearest_index(times, target):
    return int(np.argmin(np.abs(np.asarray(times, dtype=float) - float(target))))


def scalar_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


def list_cells():
    cells_root = SOURCE_PREREG / "cells"
    if not cells_root.exists():
        raise RuntimeError(
            f"Missing corrected preregistration cells at {cells_root}"
        )
    return sorted([p for p in cells_root.iterdir() if p.is_dir()])


# =============================================================================
# Block basis / eta
# =============================================================================

def ordered_block_matrix(M, p):
    M = np.asarray(M, dtype=float)
    if M.shape != (2, 2):
        raise ValueError("Expected a 2x2 block matrix.")
    if p == 1:
        return M.copy()
    if p == 2:
        perm = [1, 0]
        return M[np.ix_(perm, perm)].copy()
    raise ValueError("p must be 1 or 2")


def block_stats(M, p):
    B = ordered_block_matrix(M, p)
    mpp = float(B[0, 0])
    mpq = float(B[0, 1])
    mqp = float(B[1, 0])
    mqq = float(B[1, 1])

    den = max(abs(mpp), 1e-12)
    r_pq = abs(mpq) / den
    r_qp = abs(mqp) / den
    sqrt_r_qq = math.sqrt(abs(mqq) / den)
    eta_blk = max(r_pq, r_qp, sqrt_r_qq)
    eta_gm = math.sqrt(abs(mpq * mqp)) / den

    return {
        "m_pp": mpp,
        "m_pq": mpq,
        "m_qp": mqp,
        "m_qq": mqq,
        "r_pq": r_pq,
        "r_qp": r_qp,
        "sqrt_r_qq": sqrt_r_qq,
        "eta_blk": eta_blk,
        "eta_gm": eta_gm,
    }


# =============================================================================
# Fixed-gate least-squares solver
# =============================================================================

def solve_fixed_gate_normal_equations(S1, S2, H, cond_max=COND_MAX):
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)
    H = np.asarray(H, dtype=float)

    for A, name in ((S1, "S1"), (S2, "S2"), (H, "H")):
        if A.shape != (2, 2):
            raise ValueError(f"{name} must be 2x2.")
        if not np.allclose(A, A.T, atol=1e-10, rtol=1e-10):
            raise ValueError(f"{name} is not symmetric.")

    G = np.block([[S1, H], [H, S2]])
    B = np.concatenate([S1, S2], axis=1)

    rank = int(np.linalg.matrix_rank(G))
    cond = float(np.linalg.cond(G))

    if rank < 4 or not np.isfinite(cond) or cond > cond_max:
        raise np.linalg.LinAlgError(
            f"Gated Gram matrix is singular/ill-conditioned: "
            f"rank={rank}, cond={cond:.3e}"
        )

    # X G = B  <=>  G^T X^T = B^T.
    X = np.linalg.solve(G.T, B.T).T
    M1 = X[:, :2]
    M2 = X[:, 2:]

    residual = X @ G - B
    residual_norm = float(np.linalg.norm(residual))
    relative_residual = residual_norm / max(float(np.linalg.norm(B)), 1e-12)

    return {
        "M1": M1,
        "M2": M2,
        "G": G,
        "rank": rank,
        "cond": cond,
        "residual_norm": residual_norm,
        "relative_residual": relative_residual,
    }


# =============================================================================
# Deterministic SIM population moments via Gauss-Hermite quadrature
# =============================================================================

class SIMMomentEngine:
    """
    Equal mixture:
        cluster 1: N((mu1,0), sigma^2 I)
        cluster 2: N((0,mu2), sigma^2 I)

    Tensor-product Gauss-Hermite quadrature is deterministic and independent
    of the finite training sample.
    """

    def __init__(self, mu1, mu2, sigma, order=GH_ORDER):
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.sigma = float(sigma)
        self.order = int(order)

        nodes, weights = np.polynomial.hermite.hermgauss(self.order)
        z = math.sqrt(2.0) * nodes
        w = weights / math.sqrt(math.pi)

        z1, z2 = np.meshgrid(z, z, indexing="ij")
        w1, w2 = np.meshgrid(w, w, indexing="ij")

        self.Z = np.column_stack([z1.ravel(), z2.ravel()])
        self.W = (w1 * w2).ravel()

        self._clusters = [
            np.array([self.mu1, 0.0]),
            np.array([0.0, self.mu2]),
        ]

        self._cache = {}

    @staticmethod
    def gate(X, phi_deg):
        phi = math.radians(float(phi_deg))
        u = np.array([math.cos(phi), math.sin(phi)])
        return (X @ u > 0.0)

    def moments(self, phi1_deg=0.0, phi2_deg=90.0):
        key = (round(float(phi1_deg), 9), round(float(phi2_deg), 9))
        if key in self._cache:
            return self._cache[key]

        S1 = np.zeros((2, 2), dtype=float)
        S2 = np.zeros((2, 2), dtype=float)
        H = np.zeros((2, 2), dtype=float)

        for mu in self._clusters:
            X = mu[None, :] + self.sigma * self.Z
            g1 = self.gate(X, phi1_deg)
            g2 = self.gate(X, phi2_deg)

            # weighted second moment of X over a mask
            def second(mask):
                wm = self.W * mask.astype(float)
                return np.einsum("n,ni,nj->ij", wm, X, X)

            S1 += 0.5 * second(g1)
            S2 += 0.5 * second(g2)
            H += 0.5 * second(g1 & g2)

        # Symmetrize against quadrature roundoff.
        S1 = 0.5 * (S1 + S1.T)
        S2 = 0.5 * (S2 + S2.T)
        H = 0.5 * (H + H.T)

        out = {"S1": S1, "S2": S2, "H": H}
        self._cache[key] = out
        return out


# =============================================================================
# Self-tests requested in the theory discussion
# =============================================================================

def self_test():
    # Basis permutation.
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(
        ordered_block_matrix(M, 2),
        np.array([[4.0, 3.0], [2.0, 1.0]])
    )

    # Disjoint gates: H=0 -> M1=M2=I.
    S1 = np.array([[2.0, 0.3], [0.3, 1.0]])
    S2 = np.array([[1.5, -0.2], [-0.2, 0.8]])
    H = np.zeros((2, 2))
    sol = solve_fixed_gate_normal_equations(S1, S2, H)
    assert np.allclose(sol["M1"], np.eye(2), atol=1e-12)
    assert np.allclose(sol["M2"], np.eye(2), atol=1e-12)
    assert sol["relative_residual"] < 1e-12

    # Identical gates: H=S1=S2 -> singular / non-identifiable.
    S = np.array([[1.2, 0.1], [0.1, 0.9]])
    G_identical = np.block([[S, S], [S, S]])
    assert np.linalg.matrix_rank(G_identical) < 4

    got_singular = False
    try:
        solve_fixed_gate_normal_equations(S, S, S)
    except np.linalg.LinAlgError:
        got_singular = True
    assert got_singular

    # Deterministic population moment sanity / symmetry.
    eng = SIMMomentEngine(3.0, 2.0, 0.15, order=50)
    mom = eng.moments(0.0, 90.0)
    for name in ("S1", "S2", "H"):
        assert np.allclose(mom[name], mom[name].T, atol=1e-12)

    # Quadrature convergence sanity at the canonical ideal gates.
    eng_hi = SIMMomentEngine(3.0, 2.0, 0.15, order=100)
    mom_hi = eng_hi.moments(0.0, 90.0)
    rels = {}
    for name in ("S1", "S2", "H"):
        rel = np.linalg.norm(mom[name] - mom_hi[name]) / max(
            np.linalg.norm(mom_hi[name]), 1e-12
        )
        rels[name] = rel

    print("basis permutation test: PASS")
    print("H=0 => M1=M2=I test: PASS")
    print("identical-gate singularity test: PASS")
    print("normal-equation residual test: PASS")
    print(
        "GH(50) vs GH(100) relative moment differences: "
        + ", ".join(f"{k}={v:.3e}" for k, v in rels.items())
    )


# =============================================================================
# Source artifact loading
# =============================================================================

def load_state(path):
    z = np.load(path)
    return {
        "time": z["time"],
        "U": z["U"],
        "W": z["W"],
        "branch_time": z["branch_time"],
        "branch_phi_deg": z["branch_phi_deg"],
        "strong_mask": z["strong_mask"].astype(bool),
        "weak_mask": z["weak_mask"].astype(bool),
        "M1": z["M1"],
        "M2": z["M2"],
    }


def interp_phi(state, t):
    bt = np.asarray(state["branch_time"], dtype=float)
    bp = np.asarray(state["branch_phi_deg"], dtype=float)
    if t < bt[0] or t > bt[-1]:
        return np.nan
    return float(np.interp(t, bt, bp))


def empirical_weak_cohort_phi(state, t):
    j = nearest_index(state["time"], t)
    U = np.asarray(state["U"][j], dtype=float)
    W = np.asarray(state["W"][j], dtype=float)
    mask = np.asarray(state["weak_mask"], dtype=bool)

    Um = U[mask]
    Wm = W[mask]

    un = np.linalg.norm(Um, axis=1)
    valid = un > 1e-12
    if not np.any(valid):
        return np.nan

    dirs = Um[valid] / un[valid, None]
    mass = (
        np.sum(Um[valid] * Um[valid], axis=1)
        + np.sum(Wm[valid] * Wm[valid], axis=1)
    )
    v = np.sum(mass[:, None] * dirs, axis=0)

    return float(math.degrees(math.atan2(v[1], v[0])))


def actual_blocks_at(state, t):
    j = nearest_index(state["time"], t)
    return (
        np.asarray(state["M1"][j], dtype=float),
        np.asarray(state["M2"][j], dtype=float),
        float(state["time"][j]),
    )


# =============================================================================
# Frozen hypothesis record
# =============================================================================

def source_hash_record():
    paths = [
        SOURCE_PREREG / "manifest.json",
        SOURCE_EVAL / "evaluation_completed.json",
        SOURCE_EVAL / "scaling_summary.csv",
        SOURCE_EVAL / "G_timing_summary.csv",
    ]
    out = {}
    for p in paths:
        if p.exists():
            out[str(p)] = sha256_file(p)
    return out


def freeze():
    if OUT_ROOT.exists():
        raise RuntimeError(
            f"{OUT_ROOT} already exists. Refusing to overwrite the Level-1 "
            "freeze/result directory."
        )
    OUT_ROOT.mkdir(parents=True, exist_ok=False)

    record = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "purpose": (
            "Predeclare the fixed-gate least-squares Level-1 diagnostic before "
            "computing any fixed-point result."
        ),
        "source_artifact_hashes": source_hash_record(),
        "moment_method": {
            "distribution": (
                "equal mixture N((mu1,0),sigma^2 I) and "
                "N((0,mu2),sigma^2 I)"
            ),
            "quadrature": "tensor-product Gauss-Hermite",
            "order": GH_ORDER,
        },
        "normal_equations": (
            "[M1 M2] [[S1,H],[H,S2]] = [S1,S2]"
        ),
        "variants": {
            "ideal": "phi1=0 deg, phi2=90 deg",
            "measured_root": (
                "phi1=0 deg, phi2=stable V_true root at comparison time"
            ),
            "measured_cohort": (
                "phi1=0 deg, phi2=mass-weighted frozen weak-cohort U angle"
            ),
            "predicted_gate": (
                "PENDING: not evaluated until an independent "
                "phi_hat_W(mu1,mu2,sigma,t) is derived"
            ),
        },
        "comparison_times": [
            "t_W",
            "t_rev_p1",
            "t_rev_p2",
            "t_block=4",
            "t=10",
        ],
        "negative_prediction": (
            "The isolated leading balance "
            "m21^(1)≈-(mu1+mu2)sigma/(sqrt(2pi)mu1^2) will fail "
            "quantitatively and in comparative statics; the full coupled "
            "normal equations are required."
        ),
        "decision_rule": {
            "best_case": (
                "ideal fixed point gets comparative statics and approximately "
                "matches eta_blk at t_rev"
            ),
            "useful_conditional_case": (
                "measured-gate fixed point matches t_rev while ideal does not; "
                "gate-angle prediction is the main missing link"
            ),
            "partial_success": (
                "fixed point predicts t=10 and/or parameter ordering but not "
                "t_rev; transient dynamics are required"
            ),
            "stop_case": (
                "even measured-gate equilibrium gets mu2/sigma comparative "
                "statics wrong; stop equilibrium work for the current paper"
            ),
        },
    }

    write_json(OUT_ROOT / "freeze.json", record)
    freeze_hash = sha256_file(OUT_ROOT / "freeze.json")
    (OUT_ROOT / "freeze.sha256").write_text(freeze_hash + "\n")

    print("LEVEL-1 FIXED-POINT TEST FROZEN")
    print(f"timestamp: {record['created_utc']}")
    print(f"freeze SHA-256: {freeze_hash}")
    print(f"path: {OUT_ROOT.resolve()}")
    print("No fixed-point solve has been performed.")
    print("Now run:")
    print("    python3 fixed_gate_level1.py run")


def verify_freeze():
    freeze_path = OUT_ROOT / "freeze.json"
    hash_path = OUT_ROOT / "freeze.sha256"
    if not freeze_path.exists() or not hash_path.exists():
        raise RuntimeError("Missing frozen Level-1 protocol. Run `freeze` first.")

    expected = hash_path.read_text().strip()
    actual = sha256_file(freeze_path)
    if actual != expected:
        raise RuntimeError(
            f"Freeze hash mismatch: expected {expected}, actual {actual}"
        )

    record = read_json(freeze_path)
    for pstr, expected_hash in record["source_artifact_hashes"].items():
        p = Path(pstr)
        if not p.exists():
            raise RuntimeError(f"Frozen source artifact is missing: {p}")
        actual_hash = sha256_file(p)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen source artifact changed: {p}\n"
                f"expected={expected_hash}\nactual={actual_hash}"
            )

    print(f"Level-1 freeze verification: PASS ({actual})")
    return record


# =============================================================================
# Per-cell fixed-point analysis
# =============================================================================

def solve_variant(engine, phi2_deg):
    mom = engine.moments(0.0, phi2_deg)
    sol = solve_fixed_gate_normal_equations(
        mom["S1"], mom["S2"], mom["H"]
    )

    return {
        "phi2_deg": float(phi2_deg),
        "S1": mom["S1"],
        "S2": mom["S2"],
        "H": mom["H"],
        **sol,
    }


def flatten_matrix(prefix, M):
    M = np.asarray(M)
    return {
        f"{prefix}_11": float(M[0, 0]),
        f"{prefix}_12": float(M[0, 1]),
        f"{prefix}_21": float(M[1, 0]),
        f"{prefix}_22": float(M[1, 1]),
    }


def append_comparison_rows(
    rows,
    cid,
    mu2,
    sigma,
    label,
    t_requested,
    t_actual,
    actual_M1,
    actual_M2,
    variant_name,
    fixed,
):
    a1 = block_stats(actual_M1, 1)
    a2 = block_stats(actual_M2, 2)
    f1 = block_stats(fixed["M1"], 1)
    f2 = block_stats(fixed["M2"], 2)

    for p, a, f, actual_M, pred_M in (
        (1, a1, f1, actual_M1, fixed["M1"]),
        (2, a2, f2, actual_M2, fixed["M2"]),
    ):
        row = {
            "cell_id": cid,
            "mu2": mu2,
            "sigma": sigma,
            "time_label": label,
            "time_requested": t_requested,
            "time_actual": t_actual,
            "variant": variant_name,
            "phi2_deg": fixed["phi2_deg"],
            "p": p,
            "eta_actual": a["eta_blk"],
            "eta_pred": f["eta_blk"],
            "eta_error": f["eta_blk"] - a["eta_blk"],
            "eta_abs_error": abs(f["eta_blk"] - a["eta_blk"]),
            "eta_ratio_pred_over_actual": (
                f["eta_blk"] / max(a["eta_blk"], 1e-12)
            ),
            "actual_eta_gm": a["eta_gm"],
            "pred_eta_gm": f["eta_gm"],
            "gram_cond": fixed["cond"],
            "normal_eq_relative_residual": fixed["relative_residual"],
            "matrix_relative_frob_error": float(
                np.linalg.norm(pred_M - actual_M)
                / max(np.linalg.norm(actual_M), 1e-12)
            ),
        }
        row.update(flatten_matrix("actual_M", actual_M))
        row.update(flatten_matrix("pred_M", pred_M))

        # ordered-basis entries
        for name in ("m_pp", "m_pq", "m_qp", "m_qq"):
            row[f"actual_{name}"] = a[name]
            row[f"pred_{name}"] = f[name]

        rows.append(row)


def evaluate_cell(cdir, rows, fixedpoint_rows, trajectory_rows):
    prereg = read_json(cdir / "preregistration.json")
    cid = prereg["cell_id"]

    eval_path = SOURCE_EVAL / "cells" / cid / "evaluation.json"
    if not eval_path.exists():
        raise RuntimeError(f"Missing completed evaluation for {cid}: {eval_path}")
    ev = read_json(eval_path)

    state = load_state(cdir / prereg["files"]["training_state"])

    mu1 = float(prereg["config"]["mu1"])
    mu2 = float(prereg["config"]["mu2"])
    sigma = float(prereg["config"]["sigma1"])

    engine = SIMMomentEngine(mu1, mu2, sigma, order=GH_ORDER)

    # Hyperparameter-only ideal fixed point: one solve per cell.
    ideal = solve_variant(engine, 90.0)

    # Leading-term negative prediction.
    leading_m21 = -(
        (mu1 + mu2) * sigma / (math.sqrt(2.0 * math.pi) * mu1 ** 2)
    )

    # Comparison times.  t_rev is p-specific; both are included.
    tW = float(prereg["post_specialization"]["t_W"])
    t_rev1 = ev["p1"]["actual_core_peak"]["t_reversal"]
    t_rev2 = ev["p2"]["actual_core_peak"]["t_reversal"]

    comparisons = [
        ("t_W", tW),
        ("t_rev_p1", float(t_rev1) if t_rev1 is not None else np.nan),
        ("t_rev_p2", float(t_rev2) if t_rev2 is not None else np.nan),
        ("t_block", T_BLOCK),
        ("t_10", T_LATE),
    ]

    print()
    print("=" * 88)
    print(f"LEVEL-1 CELL {cid}")
    print("=" * 88)
    print(
        f"mu2={mu2:.3f}, sigma={sigma:.3f}, "
        f"leading m21 prediction={leading_m21:+.5f}"
    )

    # One row summarizing the ideal fixed point itself.
    fixedpoint_rows.append({
        "cell_id": cid,
        "mu2": mu2,
        "sigma": sigma,
        "variant": "ideal",
        "phi2_deg": 90.0,
        "eta1_pred": block_stats(ideal["M1"], 1)["eta_blk"],
        "eta2_pred": block_stats(ideal["M2"], 2)["eta_blk"],
        "m21_p1_pred": float(ideal["M1"][1, 0]),
        "leading_m21_one_term": leading_m21,
        "gram_cond": ideal["cond"],
        "normal_eq_relative_residual": ideal["relative_residual"],
    })

    for label, t in comparisons:
        if not np.isfinite(t):
            continue

        actual_M1, actual_M2, t_actual = actual_blocks_at(state, t)
        phi_root = interp_phi(state, t_actual)
        phi_cohort = empirical_weak_cohort_phi(state, t_actual)

        # A. ideal
        append_comparison_rows(
            rows, cid, mu2, sigma, label, t, t_actual,
            actual_M1, actual_M2, "ideal", ideal
        )

        # B1. conditional on theorem-facing stable root
        root_fixed = solve_variant(engine, phi_root)
        append_comparison_rows(
            rows, cid, mu2, sigma, label, t, t_actual,
            actual_M1, actual_M2, "measured_root", root_fixed
        )

        # B2. conditional on actual frozen weak-cohort gate center
        cohort_fixed = solve_variant(engine, phi_cohort)
        append_comparison_rows(
            rows, cid, mu2, sigma, label, t, t_actual,
            actual_M1, actual_M2, "measured_cohort", cohort_fixed
        )

        a1 = block_stats(actual_M1, 1)
        a2 = block_stats(actual_M2, 2)

        print(
            f"{label:9s} t={t_actual:.3f} | "
            f"phi_root={phi_root:.2f}, phi_cohort={phi_cohort:.2f} | "
            f"eta1 actual={a1['eta_blk']:.4f}, "
            f"ideal={block_stats(ideal['M1'],1)['eta_blk']:.4f}, "
            f"root={block_stats(root_fixed['M1'],1)['eta_blk']:.4f}, "
            f"cohort={block_stats(cohort_fixed['M1'],1)['eta_blk']:.4f}"
        )

    # Trajectory comparison: actual eta(t) vs equilibrium conditional on
    # measured gate angles.  Sample every 0.25 time units after t_W.
    times = np.asarray(state["time"], dtype=float)
    sample_times = np.arange(tW, T_LATE + 1e-9, 0.25)

    for t in sample_times:
        actual_M1, actual_M2, ta = actual_blocks_at(state, t)
        phi_root = interp_phi(state, ta)
        phi_cohort = empirical_weak_cohort_phi(state, ta)

        root_fixed = solve_variant(engine, phi_root)
        cohort_fixed = solve_variant(engine, phi_cohort)

        trajectory_rows.append({
            "cell_id": cid,
            "mu2": mu2,
            "sigma": sigma,
            "time": ta,
            "phi_root_deg": phi_root,
            "phi_cohort_deg": phi_cohort,
            "eta1_actual": block_stats(actual_M1, 1)["eta_blk"],
            "eta2_actual": block_stats(actual_M2, 2)["eta_blk"],
            "eta1_ideal": block_stats(ideal["M1"], 1)["eta_blk"],
            "eta2_ideal": block_stats(ideal["M2"], 2)["eta_blk"],
            "eta1_root_fixed": block_stats(root_fixed["M1"], 1)["eta_blk"],
            "eta2_root_fixed": block_stats(root_fixed["M2"], 2)["eta_blk"],
            "eta1_cohort_fixed": block_stats(cohort_fixed["M1"], 1)["eta_blk"],
            "eta2_cohort_fixed": block_stats(cohort_fixed["M2"], 2)["eta_blk"],
        })


# =============================================================================
# Aggregate summaries / plots
# =============================================================================

def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize_rows(rows):
    summary = []

    for p in (1, 2):
        for label in ("t_W", "t_rev_p1", "t_rev_p2", "t_block", "t_10"):
            for variant in ("ideal", "measured_root", "measured_cohort"):
                rr = [
                    r for r in rows
                    if r["p"] == p
                    and r["time_label"] == label
                    and r["variant"] == variant
                ]
                if not rr:
                    continue

                actual = np.asarray([r["eta_actual"] for r in rr], dtype=float)
                pred = np.asarray([r["eta_pred"] for r in rr], dtype=float)
                mae = float(np.mean(np.abs(pred - actual)))
                rel = float(np.mean(np.abs(pred - actual) / np.maximum(actual, 1e-12)))
                corr = (
                    float(np.corrcoef(actual, pred)[0, 1])
                    if len(rr) >= 3
                    and np.std(actual) > 0
                    and np.std(pred) > 0
                    else np.nan
                )

                summary.append({
                    "p": p,
                    "time_label": label,
                    "variant": variant,
                    "n": len(rr),
                    "eta_MAE": mae,
                    "eta_mean_relative_abs_error": rel,
                    "eta_correlation": corr,
                    "mean_matrix_relative_frob_error": float(
                        np.mean([r["matrix_relative_frob_error"] for r in rr])
                    ),
                })

    return summary


def comparative_statics_table(rows):
    """
    Focus on p=1 at the actual p1 reversal and t=10.
    """
    out = []
    for label in ("t_rev_p1", "t_10"):
        for variant in ("ideal", "measured_root", "measured_cohort"):
            rr = [
                r for r in rows
                if r["p"] == 1
                and r["time_label"] == label
                and r["variant"] == variant
            ]
            if not rr:
                continue

            # Average slope across sigma within each mu2, and across mu2
            # within each sigma using simple least-squares lines.
            slopes_mu = []
            for sigma in sorted(set(r["sigma"] for r in rr)):
                sub = sorted(
                    [r for r in rr if r["sigma"] == sigma],
                    key=lambda z: z["mu2"],
                )
                if len(sub) >= 2:
                    x = np.asarray([r["mu2"] for r in sub])
                    y = np.asarray([r["eta_pred"] for r in sub])
                    slopes_mu.append(float(np.polyfit(x, y, 1)[0]))

            slopes_sig = []
            for mu2 in sorted(set(r["mu2"] for r in rr)):
                sub = sorted(
                    [r for r in rr if r["mu2"] == mu2],
                    key=lambda z: z["sigma"],
                )
                if len(sub) >= 2:
                    x = np.asarray([r["sigma"] for r in sub])
                    y = np.asarray([r["eta_pred"] for r in sub])
                    slopes_sig.append(float(np.polyfit(x, y, 1)[0]))

            out.append({
                "time_label": label,
                "variant": variant,
                "mean_deta_dmu2_pred": float(np.mean(slopes_mu)),
                "mean_abs_deta_dsigma_pred": float(np.mean(np.abs(slopes_sig))),
            })

    # Actual slopes separately.
    for label in ("t_rev_p1", "t_10"):
        rr = [
            r for r in rows
            if r["p"] == 1
            and r["time_label"] == label
            and r["variant"] == "ideal"
        ]
        if not rr:
            continue

        slopes_mu = []
        for sigma in sorted(set(r["sigma"] for r in rr)):
            sub = sorted(
                [r for r in rr if r["sigma"] == sigma],
                key=lambda z: z["mu2"],
            )
            x = np.asarray([r["mu2"] for r in sub])
            y = np.asarray([r["eta_actual"] for r in sub])
            slopes_mu.append(float(np.polyfit(x, y, 1)[0]))

        slopes_sig = []
        for mu2 in sorted(set(r["mu2"] for r in rr)):
            sub = sorted(
                [r for r in rr if r["mu2"] == mu2],
                key=lambda z: z["sigma"],
            )
            x = np.asarray([r["sigma"] for r in sub])
            y = np.asarray([r["eta_actual"] for r in sub])
            slopes_sig.append(float(np.polyfit(x, y, 1)[0]))

        out.append({
            "time_label": label,
            "variant": "ACTUAL",
            "mean_deta_dmu2_pred": float(np.mean(slopes_mu)),
            "mean_abs_deta_dsigma_pred": float(np.mean(np.abs(slopes_sig))),
        })

    return out


def make_scatter(rows, p, label, variant, filename):
    rr = [
        r for r in rows
        if r["p"] == p
        and r["time_label"] == label
        and r["variant"] == variant
    ]
    if not rr:
        return

    x = np.asarray([r["eta_actual"] for r in rr])
    y = np.asarray([r["eta_pred"] for r in rr])

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(x, y)
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(r"actual $\eta_p^{blk}$")
    plt.ylabel(r"fixed-point $\eta_{p,\infty}^{blk}$")
    plt.title(f"p={p}, {label}, {variant}")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / filename)
    plt.close(fig)


def make_comparative_mu_plot(rows, label, variant, filename):
    rr = [
        r for r in rows
        if r["p"] == 1
        and r["time_label"] == label
        and r["variant"] == variant
    ]
    if not rr:
        return

    fig = plt.figure(figsize=(7, 5))
    for sigma in sorted(set(r["sigma"] for r in rr)):
        sub = sorted(
            [r for r in rr if r["sigma"] == sigma],
            key=lambda z: z["mu2"],
        )
        x = [r["mu2"] for r in sub]
        ya = [r["eta_actual"] for r in sub]
        yp = [r["eta_pred"] for r in sub]
        plt.plot(x, ya, marker="o", label=f"actual sigma={sigma:.2f}")
        plt.plot(x, yp, marker="x", linestyle="--",
                 label=f"pred sigma={sigma:.2f}")
    plt.xlabel(r"$\mu_2$")
    plt.ylabel(r"$\eta_1^{blk}$")
    plt.title(f"Comparative statics at {label}: {variant}")
    plt.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / filename)
    plt.close(fig)


def make_trajectory_plot(traj_rows, cid, p, filename):
    rr = [r for r in traj_rows if r["cell_id"] == cid]
    if not rr:
        return

    t = np.asarray([r["time"] for r in rr])
    actual = np.asarray([r[f"eta{p}_actual"] for r in rr])
    ideal = np.asarray([r[f"eta{p}_ideal"] for r in rr])
    root = np.asarray([r[f"eta{p}_root_fixed"] for r in rr])
    cohort = np.asarray([r[f"eta{p}_cohort_fixed"] for r in rr])

    fig = plt.figure(figsize=(8, 5))
    plt.plot(t, actual, label="actual")
    plt.plot(t, ideal, label="ideal fixed point")
    plt.plot(t, root, label="root-conditioned fixed point")
    plt.plot(t, cohort, label="cohort-conditioned fixed point")
    plt.xlabel("time")
    plt.ylabel(rf"$\eta_{p}^{{blk}}$")
    plt.title(f"{cid}: actual vs conditional equilibrium, p={p}")
    plt.legend()
    fig.tight_layout()
    fig.savefig(OUT_ROOT / filename)
    plt.close(fig)


# =============================================================================
# Run
# =============================================================================

def run():
    verify_freeze()

    if (OUT_ROOT / "fixed_point_comparisons.csv").exists():
        raise RuntimeError(
            "Level-1 results already exist. Refusing to overwrite them."
        )

    rows = []
    fixedpoint_rows = []
    trajectory_rows = []

    for cdir in list_cells():
        evaluate_cell(cdir, rows, fixedpoint_rows, trajectory_rows)

    summary = summarize_rows(rows)
    comp = comparative_statics_table(rows)

    write_csv(OUT_ROOT / "fixed_point_comparisons.csv", rows)
    write_csv(OUT_ROOT / "fixed_point_summary.csv", summary)
    write_csv(OUT_ROOT / "fixed_point_cell_summary.csv", fixedpoint_rows)
    write_csv(OUT_ROOT / "comparative_statics.csv", comp)
    write_csv(OUT_ROOT / "eta_trajectory.csv", trajectory_rows)

    # Most decision-relevant figures.
    for p in (1, 2):
        for variant in ("ideal", "measured_root", "measured_cohort"):
            make_scatter(
                rows, p, f"t_rev_p{p}", variant,
                f"eta_scatter_p{p}_trev_{variant}.pdf"
            )
            make_scatter(
                rows, p, "t_10", variant,
                f"eta_scatter_p{p}_t10_{variant}.pdf"
            )

    for variant in ("ideal", "measured_root", "measured_cohort"):
        make_comparative_mu_plot(
            rows, "t_rev_p1", variant,
            f"comparative_mu2_trev_p1_{variant}.pdf"
        )
        make_comparative_mu_plot(
            rows, "t_10", variant,
            f"comparative_mu2_t10_p1_{variant}.pdf"
        )

    # Three representative cells: low, canonical, high.
    cells = [p.name for p in list_cells()]
    for cid in (
        "mu2_1p800__sigma_0p120",
        "mu2_2p000__sigma_0p150",
        "mu2_2p200__sigma_0p180",
    ):
        if cid in cells:
            for p in (1, 2):
                make_trajectory_plot(
                    trajectory_rows, cid, p,
                    f"trajectory_{cid}_p{p}.pdf"
                )

    completed = {
        "completed_utc": utc_now(),
        "freeze_sha256": sha256_file(OUT_ROOT / "freeze.json"),
        "n_comparison_rows": len(rows),
        "n_trajectory_rows": len(trajectory_rows),
        "predicted_gate_variant_status": (
            "not evaluated; no independent phi_hat_W has been derived"
        ),
    }
    write_json(OUT_ROOT / "completed.json", completed)

    print()
    print("=" * 88)
    print("LEVEL-1 FIXED-POINT DIAGNOSTIC COMPLETE")
    print("=" * 88)
    print(f"outputs: {OUT_ROOT.resolve()}")
    print("Decision files:")
    print("  fixed_point_summary.csv")
    print("  comparative_statics.csv")
    print("  fixed_point_cell_summary.csv")
    print("  representative trajectory PDFs")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["self-test", "freeze", "run"])
    args = parser.parse_args()

    if args.stage == "self-test":
        self_test()
        return
    if args.stage == "freeze":
        self_test()
        freeze()
        return
    if args.stage == "run":
        self_test()
        run()
        return


if __name__ == "__main__":
    main()
