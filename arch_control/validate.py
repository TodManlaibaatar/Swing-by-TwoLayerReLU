"""Validation against the archived canonical results (PROTOCOL.md Section 5).

V1  The E1 ReLU seed-0 run reproduces the archived canonical trajectory
    (swing_grid_preregistered/.../mu2_2p000__sigma_0p150/training_state.npz), and
    the persistent-increase statistic recomputed on the archived 1022 phase-diagram
    directions over the archived window reproduces phase_diagram.csv
    (198 in-cone / 824 off-cone / 230 above 1e-4).
V4  The vectorized detector equals the archived ``persistent_swing``.

    python -m arch_control.validate [--out arch_control/results]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import theory_guided_swing_by_phase_diagram as tg  # noqa: E402
from arch_control import metrics as M  # noqa: E402
from arch_control.experiments import experiment_specs  # noqa: E402

ARCH_STATE = REPO / "swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150/training_state.npz"
ARCH_PHASE = REPO / "theory_guided_swing_phase_diagram_v3/phase_diagram.csv"
ARCH_BRANCH = REPO / "theory_guided_swing_phase_diagram_v3/true_field_weak_branch.csv"


def detector_equivalence(n_series=300, seed=7):
    rng = np.random.default_rng(seed)
    times = np.arange(0, 10.0001, 0.01)
    Y = (np.cumsum(rng.standard_normal((len(times), n_series)) * 1e-3, axis=0)
         + np.sin(times)[:, None] * rng.uniform(0, 0.01, n_series))
    fast = M.persistent_increase(times, Y)
    ref = np.array([tg.persistent_swing(times, Y[:, k])[0]["swing"] for k in range(n_series)])
    return float(np.abs(ref - fast["swing"]).max())


def archive_reproduction(out: Path):
    spec = [s for s in experiment_specs("E1_canonical") if s.arch == "relu" and s.seed == 0][0]
    p = out / spec.exp / "runs" / f"{spec.run_id}.npz"
    if not p.exists():
        return {"status": "SKIPPED (E1 relu seed-0 run not found)"}
    z = np.load(p)
    U, W, times = z["U_hist"], z["W_hist"], z["times"]
    a = np.load(ARCH_STATE)
    L = len(a["time"])
    if not np.allclose(times[:L], a["time"]):
        return {"status": "FAIL (time grids differ)"}
    dU = float(np.abs(U[:L] - a["U"]).max()); dW = float(np.abs(W[:L] - a["W"]).max())
    rows = list(csv.DictReader(open(ARCH_PHASE)))
    psi = np.array([float(r["psi_deg"]) for r in rows])
    S_arch = np.array([float(r["actual_persistent_swing"]) for r in rows])
    br = [r for r in csv.DictReader(open(ARCH_BRANCH)) if r["branch_exists"] in ("True", "true", "1")]
    t_birth = float(br[0]["time"])
    j0 = int(np.argmin(np.abs(times - t_birth))); j1 = int(np.argmin(np.abs(times - 10.0)))
    P = np.column_stack([np.cos(np.radians(psi)), np.sin(np.radians(psi))])
    E = np.empty((j1 - j0 + 1, len(psi)))
    for k, j in enumerate(range(j0, j1 + 1)):
        F = np.maximum(P @ U[j].T, 0.0) @ W[j]
        E[k] = 0.5 * ((F - P) ** 2).sum(1)
    S = M.persistent_increase(times[j0:j1 + 1], E)["swing"]
    cone = M.in_closed_positive_cone(psi)
    res = dict(max_abs_dU=dU, max_abs_dW=dW, t_birth=t_birth,
               max_abs_dS=float(np.abs(S - S_arch).max()),
               n_directions=len(psi), n_in_cone=int(cone.sum()), n_off_cone=int((~cone).sum()),
               n_above_1em4=int((S > 1e-4).sum()), n_above_1em4_in_cone=int(((S > 1e-4) & cone).sum()),
               archived_above_1em4=int((S_arch > 1e-4).sum()))
    ok = (dU < 1e-9 and dW < 1e-9 and res["max_abs_dS"] < 1e-9
          and res["n_above_1em4"] == res["archived_above_1em4"] == 230
          and res["n_in_cone"] == 198 and res["n_above_1em4_in_cone"] == 0)
    res["status"] = "PASS" if ok else "FAIL"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    v4 = detector_equivalence()
    v1 = archive_reproduction(out)
    rep = {"V1_archive_reproduction": v1,
           "V4_detector_max_abs_diff": v4, "V4_status": "PASS" if v4 < 1e-12 else "FAIL"}
    (out / "VALIDATION.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0 if (v1.get("status", "").startswith(("PASS", "SKIPPED")) and v4 < 1e-12) else 1


if __name__ == "__main__":
    sys.exit(main())
