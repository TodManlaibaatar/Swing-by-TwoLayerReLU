#!/usr/bin/env python3
"""
swing_final_audits.py

Final publication-facing empirical audits for the two-concept Swing-by paper.

Components
----------
1) Multi-seed robustness on three representative cells:
       (mu2,sigma) = (1.8,.12), (2.0,.15), (2.2,.18)
   with seeds 0,1,2,3,4. Seed 0 reuses the completed frozen grid trajectory;
   seeds 1-4 are newly trained with exactly the same protocol.

2) Grid-wide reduction validity / boundary-layer audit on the completed 3x3
   seed-0 grid:
   - full network vs gates frozen at t_W on ID samples;
   - full network vs dynamic two-block selector model on ID samples;
   - OOD selector-model RMS error vs minimum angular distance to either learned
     gate boundary.

3) Aggregate sector-classification audit:
   full vs reduced labels on the 720-direction phase grid:
       swing,
       negligible,
       no_swing_improving,
       no_swing_worsening.
   Report all-direction and >=6 degree boundary-excluded agreement.

This script does NOT modify any prior preregistration or evaluation artifact.

Required repo files
-------------------
- swing_grid_preregister.py  (the corrected post-specialization version)
- full_flow_diagnostic.py
- theory_guided_swing_by_phase_diagram.py
  (or theory_guided_swing_phase_diagram_v3.py)

Required completed artifacts
----------------------------
swing_grid_preregistered/
  preregistration_v1_postspecialization/
  evaluation_v1_postspecialization/

Usage
-----
    python3 swing_final_audits.py self-test
    python3 swing_final_audits.py freeze
    python3 swing_final_audits.py multiseed
    python3 swing_final_audits.py audit
    python3 swing_final_audits.py summarize

`freeze` must be run before either empirical component.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import swing_grid_preregister as sg

if not hasattr(sg, "compute_post_specialization_time"):
    raise RuntimeError(
        "swing_grid_preregister.py is not the corrected post-specialization "
        "version. Replace it with the patched version before running this audit."
    )

exp = sg.exp

# ---------------------------------------------------------------------------
# Frozen audit protocol
# ---------------------------------------------------------------------------

REPRESENTATIVE_CELLS = (
    (1.8, 0.12),
    (2.0, 0.15),
    (2.2, 0.18),
)
SEEDS = (0, 1, 2, 3, 4)

SWING_THRESHOLD = 1e-4
NEGLIGIBLE_RANGE_THRESHOLD = 1e-4
BOUNDARY_EXCLUDE_DEG = 6.0

PREREG_ROOT = Path(
    "swing_grid_preregistered/preregistration_v1_postspecialization"
)
EVAL_ROOT = Path(
    "swing_grid_preregistered/evaluation_v1_postspecialization"
)
OUT_ROOT = Path("swing_final_audits")

MULTISEED_ROOT = OUT_ROOT / "multiseed"
AUDIT_ROOT = OUT_ROOT / "reduction_boundary_sector"

BOUNDARY_BINS = (0.0, 2.0, 4.0, 6.0, 10.0, 20.0, 180.0)


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

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


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True) + "\n")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cell_id(mu2, sigma):
    return sg.cell_id(mu2, sigma)


def angular_abs_deg(a, b):
    return np.abs(sg.wrap_deg(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def angular_error_deg(a, b):
    return float(angular_abs_deg([a], [b])[0])


def cfg_for(mu2, sigma, seed):
    return replace(
        sg.make_cfg(mu2, sigma),
        seed=int(seed),
    )


def source_hashes():
    out = {}
    for p in (
        PREREG_ROOT / "manifest.json",
        EVAL_ROOT / "evaluation_completed.json",
        EVAL_ROOT / "scaling_summary.csv",
        EVAL_ROOT / "G_timing_summary.csv",
    ):
        if p.exists():
            out[str(p)] = sha256_file(p)
    return out


# ---------------------------------------------------------------------------
# Frozen protocol / verification
# ---------------------------------------------------------------------------

def freeze():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    freeze_path = OUT_ROOT / "freeze.json"
    hash_path = OUT_ROOT / "freeze.sha256"

    if freeze_path.exists() or hash_path.exists():
        raise RuntimeError(
            f"{freeze_path} already exists. Refusing to overwrite the audit freeze."
        )

    record = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "source_hashes": source_hashes(),
        "components": {
            "multiseed": {
                "cells": [list(x) for x in REPRESENTATIVE_CELLS],
                "seeds": list(SEEDS),
                "seed0_reused": True,
                "new_training_seeds": [s for s in SEEDS if s != 0],
                "headline_metrics": [
                    "t_W",
                    "eta1_blk",
                    "eta2_blk",
                    "p1_peak_angle_error_deg",
                    "p2_peak_angle_error_deg",
                    "Smax1_over_eta1_sq",
                    "Smax2_over_eta2_sq",
                    "global_corr_Sactual_Sred",
                    "tG_minus_tfull_probe",
                    "tred_minus_tfull_probe",
                ],
            },
            "reduction_validity": {
                "reference_gate_time": "t_W",
                "id_frozen_gate_metric": (
                    "relative RMS of full output vs all-neuron gates frozen at t_W"
                ),
                "id_block_metric": (
                    "relative RMS of full output vs dynamic two-block selector model"
                ),
                "boundary_metric": (
                    "OOD selector-model output RMS vs minimum angular distance "
                    "to strong or dynamic weak gate boundary over [t_W,T]"
                ),
            },
            "sector_classification": {
                "swing_threshold": SWING_THRESHOLD,
                "negligible_range_threshold": NEGLIGIBLE_RANGE_THRESHOLD,
                "boundary_exclude_deg": BOUNDARY_EXCLUDE_DEG,
                "labels": [
                    "swing",
                    "negligible",
                    "no_swing_improving",
                    "no_swing_worsening",
                ],
            },
        },
    }
    write_json(freeze_path, record)
    digest = sha256_file(freeze_path)
    hash_path.write_text(digest + "\n")

    print("FINAL EMPIRICAL AUDIT PROTOCOL FROZEN")
    print(f"timestamp: {record['created_utc']}")
    print(f"SHA-256: {digest}")
    print(f"path: {OUT_ROOT.resolve()}")


def verify_freeze():
    freeze_path = OUT_ROOT / "freeze.json"
    hash_path = OUT_ROOT / "freeze.sha256"
    if not freeze_path.exists() or not hash_path.exists():
        raise RuntimeError("Run `python3 swing_final_audits.py freeze` first.")

    expected = hash_path.read_text().strip()
    actual = sha256_file(freeze_path)
    if actual != expected:
        raise RuntimeError(
            f"Freeze hash mismatch: expected {expected}, got {actual}"
        )

    record = read_json(freeze_path)
    for pstr, expected_hash in record["source_hashes"].items():
        p = Path(pstr)
        if not p.exists():
            raise RuntimeError(f"Frozen source artifact missing: {p}")
        got = sha256_file(p)
        if got != expected_hash:
            raise RuntimeError(
                f"Frozen source artifact changed:\n{p}\n"
                f"expected={expected_hash}\ngot={got}"
            )

    print(f"Final-audit freeze verification: PASS ({actual})")
    return record


# ---------------------------------------------------------------------------
# Shared trajectory helpers
# ---------------------------------------------------------------------------

def run_from_saved_state(state):
    return {
        "time": np.asarray(state["time"], dtype=float),
        "U": [u for u in np.asarray(state["U"])],
        "W": [w for w in np.asarray(state["W"])],
    }


def branch_from_saved_state(state):
    return sg.branch_rows_from_state(state)


def blocks_from_saved_state(state):
    return {
        "strong": np.asarray(state["strong_mask"], dtype=bool),
        "weak": np.asarray(state["weak_mask"], dtype=bool),
        "remainder": np.asarray(state["remainder_mask"], dtype=bool),
    }


def prepare_new_seed_run(mu2, sigma, seed):
    cfg = cfg_for(mu2, sigma, seed)
    run = exp.run_relu(cfg, log_every=50)

    branch = sg.track_weak_branch(run)
    bt, _ = sg.valid_branch_arrays(branch)

    phi_block = sg.interp_branch(branch, [sg.T_BLOCK])[0]
    if not np.isfinite(phi_block):
        raise RuntimeError(
            f"{cell_id(mu2,sigma)} seed={seed}: weak branch absent at t_block."
        )

    blocks = exp.define_final_blocks_from_branch(
        run,
        phi_block_deg=float(phi_block),
        t_block=sg.T_BLOCK,
        tol_deg=sg.BLOCK_TOL_DEG,
    )
    M1_hist, M2_hist = sg.block_histories(run, blocks)

    postspec = sg.compute_post_specialization_time(
        run["time"],
        run["U"],
        run["W"],
        blocks["strong"],
        blocks["weak"],
        branch_birth=float(bt[0]),
        t_block=sg.T_BLOCK,
        threshold=sg.POSTSPEC_MASS_SHARE_FRACTION,
    )

    return cfg, run, branch, blocks, M1_hist, M2_hist, postspec


def prepare_seed0_reuse(mu2, sigma):
    cid = cell_id(mu2, sigma)
    cdir = PREREG_ROOT / "cells" / cid
    prereg = read_json(cdir / "preregistration.json")
    state = sg.load_training_state(cdir / prereg["files"]["training_state"])
    run = run_from_saved_state(state)
    branch = branch_from_saved_state(state)
    blocks = blocks_from_saved_state(state)
    M1_hist = np.asarray(state["M1"], dtype=float)
    M2_hist = np.asarray(state["M2"], dtype=float)
    postspec = prereg["post_specialization"]
    cfg = cfg_for(mu2, sigma, 0)
    return cfg, run, branch, blocks, M1_hist, M2_hist, postspec, state


def analyze_multiseed_run(
    mu2, sigma, seed, cfg, run, branch, blocks, M1_hist, M2_hist, postspec
):
    times = np.asarray(run["time"], dtype=float)
    j_block = sg.nearest_index(times, sg.T_BLOCK)

    stat1 = sg.block_alignment_stats(M1_hist[j_block], p=1)
    stat2 = sg.block_alignment_stats(M2_hist[j_block], p=2)

    prediction = sg.reduced_prediction(
        run, branch, blocks, M1_hist, M2_hist,
        t_start=float(postspec["t_W"]),
    )

    phi_block = float(sg.interp_branch(branch, [sg.T_BLOCK])[0])
    p1_sector = sg.predicted_primary_sector(
        1, -90.0, stat1["eta_blk"], prediction, branch
    )
    p2_center = float(sg.wrap_deg(phi_block + 90.0))
    p2_sector = sg.predicted_primary_sector(
        2, p2_center, stat2["eta_blk"], prediction, branch
    )

    p1_G = sg.prereg_G_timing(
        times,
        M1_hist,
        stat1["eta_blk"],
        p1_sector["normal_form_probe_psi_deg"],
        p1_sector["normal_form_probe_t_reversal_red"],
        t_valid_start=float(postspec["t_W"]),
    )

    state_like = {
        "time": times,
        "U": np.stack(run["U"], axis=0),
        "W": np.stack(run["W"], axis=0),
        "branch_time": np.asarray([r["time"] for r in branch if r["branch_exists"]]),
        "branch_phi_deg": np.asarray([r["phi_deg"] for r in branch if r["branch_exists"]]),
    }
    actual = sg.evaluate_full_phase(state_like, t_start=float(postspec["t_W"]))

    p1_peak = sg.peak_in_frozen_core(
        actual, p1_sector["analysis_center_deg"], stat1["eta_blk"]
    )
    p2_peak = sg.peak_in_frozen_core(
        actual, p2_sector["analysis_center_deg"], stat2["eta_blk"]
    )

    corr = (
        float(np.corrcoef(actual["Sactual"], prediction["Sred"])[0, 1])
        if np.std(actual["Sactual"]) > 0 and np.std(prediction["Sred"]) > 0
        else np.nan
    )
    mae = float(np.mean(np.abs(actual["Sactual"] - prediction["Sred"])))

    p1_probe = float(p1_sector["normal_form_probe_psi_deg"])
    kp = sg.nearest_direction_index(actual["psi_deg"], p1_probe)
    t_full_probe = (
        float(actual["t_rev_actual"][kp])
        if np.isfinite(actual["t_rev_actual"][kp]) else np.nan
    )
    t_red_probe = p1_sector["normal_form_probe_t_reversal_red"]
    tG = p1_G["G_zero_crossing_fd"]

    row = {
        "cell_id": cell_id(mu2, sigma),
        "mu2": float(mu2),
        "sigma": float(sigma),
        "seed": int(seed),
        "t_W": float(postspec["t_W"]),
        "eta1_blk": float(stat1["eta_blk"]),
        "eta2_blk": float(stat2["eta_blk"]),
        "p1_pred_probe_psi_deg": p1_probe,
        "p1_actual_peak_psi_deg": (
            p1_peak["psi_deg"] if p1_peak is not None else np.nan
        ),
        "p1_peak_angle_error_deg": (
            angular_error_deg(p1_peak["psi_deg"], p1_probe)
            if p1_peak is not None else np.nan
        ),
        "p2_pred_probe_psi_deg": float(p2_sector["normal_form_probe_psi_deg"]),
        "p2_actual_peak_psi_deg": (
            p2_peak["psi_deg"] if p2_peak is not None else np.nan
        ),
        "p2_peak_angle_error_deg": (
            angular_error_deg(
                p2_peak["psi_deg"],
                p2_sector["normal_form_probe_psi_deg"],
            )
            if p2_peak is not None else np.nan
        ),
        "Smax1_over_eta1_sq": (
            p1_peak["Smax_over_eta2"] if p1_peak is not None else np.nan
        ),
        "Smax2_over_eta2_sq": (
            p2_peak["Smax_over_eta2"] if p2_peak is not None else np.nan
        ),
        "global_corr_Sactual_Sred": corr,
        "global_MAE": mae,
        "tG": float(tG) if tG is not None else np.nan,
        "tred_probe": (
            float(t_red_probe) if t_red_probe is not None else np.nan
        ),
        "tfull_probe": t_full_probe,
        "tG_minus_tfull_probe": (
            float(tG) - t_full_probe
            if tG is not None and np.isfinite(t_full_probe) else np.nan
        ),
        "tred_minus_tfull_probe": (
            float(t_red_probe) - t_full_probe
            if t_red_probe is not None and np.isfinite(t_full_probe) else np.nan
        ),
    }
    return row, actual, prediction


# ---------------------------------------------------------------------------
# Component 1: multi-seed robustness
# ---------------------------------------------------------------------------

def run_multiseed():
    verify_freeze()
    MULTISEED_ROOT.mkdir(parents=True, exist_ok=True)

    summary_path = MULTISEED_ROOT / "multiseed_summary.csv"
    if summary_path.exists():
        raise RuntimeError(
            f"{summary_path} already exists. Refusing to overwrite multi-seed results."
        )

    rows = []

    for mu2, sigma in REPRESENTATIVE_CELLS:
        cid = cell_id(mu2, sigma)
        for seed in SEEDS:
            print()
            print("=" * 88)
            print(f"MULTI-SEED {cid} seed={seed}")
            print("=" * 88)

            if seed == 0:
                cfg, run, branch, blocks, M1, M2, postspec, _ = (
                    prepare_seed0_reuse(mu2, sigma)
                )
            else:
                cfg, run, branch, blocks, M1, M2, postspec = (
                    prepare_new_seed_run(mu2, sigma, seed)
                )

                # Preserve newly trained trajectories for reproducibility.
                sdir = MULTISEED_ROOT / "states" / cid
                sdir.mkdir(parents=True, exist_ok=True)
                sg.save_training_state(
                    sdir / f"seed_{seed}.npz",
                    run, branch, blocks, M1, M2
                )

            row, _, _ = analyze_multiseed_run(
                mu2, sigma, seed, cfg, run, branch, blocks, M1, M2, postspec
            )
            rows.append(row)

            print(
                f"tW={row['t_W']:.3f}, "
                f"eta1={row['eta1_blk']:.4f}, eta2={row['eta2_blk']:.4f}, "
                f"corr={row['global_corr_Sactual_Sred']:.4f}"
            )
            print(
                f"p1 angle err={row['p1_peak_angle_error_deg']:.2f} deg, "
                f"p2 angle err={row['p2_peak_angle_error_deg']:.2f} deg, "
                f"|tG-tfull|={abs(row['tG_minus_tfull_probe']):.3f}"
            )

    write_csv(summary_path, rows)

    agg = []
    metric_names = [
        "t_W",
        "eta1_blk",
        "eta2_blk",
        "p1_peak_angle_error_deg",
        "p2_peak_angle_error_deg",
        "Smax1_over_eta1_sq",
        "Smax2_over_eta2_sq",
        "global_corr_Sactual_Sred",
        "global_MAE",
        "tG_minus_tfull_probe",
        "tred_minus_tfull_probe",
    ]
    for cid in sorted(set(r["cell_id"] for r in rows)):
        rr = [r for r in rows if r["cell_id"] == cid]
        for m in metric_names:
            a = np.asarray([r[m] for r in rr], dtype=float)
            a = a[np.isfinite(a)]
            agg.append({
                "cell_id": cid,
                "metric": m,
                "n": int(len(a)),
                "mean": float(np.mean(a)) if len(a) else np.nan,
                "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                "min": float(np.min(a)) if len(a) else np.nan,
                "max": float(np.max(a)) if len(a) else np.nan,
            })

    write_csv(MULTISEED_ROOT / "multiseed_aggregate.csv", agg)

    # Compact publication-facing figures.
    for metric, ylabel, fn in (
        ("global_corr_Sactual_Sred", "corr(Sactual, Sred)", "multiseed_phase_corr.pdf"),
        ("p1_peak_angle_error_deg", "p1 peak angle error (deg)", "multiseed_p1_angle_error.pdf"),
        ("tG_minus_tfull_probe", "tG - tfull", "multiseed_G_timing_error.pdf"),
    ):
        fig = plt.figure(figsize=(7, 4.5))
        labels = []
        data = []
        for mu2, sigma in REPRESENTATIVE_CELLS:
            cid = cell_id(mu2, sigma)
            labels.append(f"{mu2:.1f},{sigma:.2f}")
            data.append([r[metric] for r in rows if r["cell_id"] == cid])
        plt.boxplot(data, tick_labels=labels)
        plt.xlabel(r"$(\mu_2,\sigma)$")
        plt.ylabel(ylabel)
        plt.title("Multi-seed robustness")
        fig.tight_layout()
        fig.savefig(MULTISEED_ROOT / fn)
        plt.close(fig)

    print()
    print("MULTI-SEED ROBUSTNESS COMPLETE")
    print(f"summary: {summary_path.resolve()}")


# ---------------------------------------------------------------------------
# Component 2+3 shared full/reduced output audit
# ---------------------------------------------------------------------------

def reconstruct_seed0_data(prereg):
    cfg = cfg_for(
        float(prereg["config"]["mu2"]),
        float(prereg["config"]["sigma1"]),
        int(prereg["config"]["seed"]),
    )
    X, labels = exp.make_data(cfg)
    return X, labels


def relative_rms(F, G):
    num = float(np.sum((F - G) ** 2))
    den = float(np.sum(F ** 2))
    return math.sqrt(num / max(den, 1e-12))


def relative_rms_rows(F, G):
    num = np.sum((F - G) ** 2, axis=0)
    den = np.sum(F ** 2, axis=0)
    return np.sqrt(num / np.maximum(den, 1e-12))


def evaluate_id_reduction(prereg, state):
    X, labels = reconstruct_seed0_data(prereg)
    times = np.asarray(state["time"], dtype=float)
    Uhist = np.asarray(state["U"], dtype=float)
    Whist = np.asarray(state["W"], dtype=float)
    M1hist = np.asarray(state["M1"], dtype=float)
    M2hist = np.asarray(state["M2"], dtype=float)

    tW = float(prereg["post_specialization"]["t_W"])
    j0 = sg.nearest_index(times, tW)
    j1 = sg.nearest_index(times, sg.T_END)
    idx = np.arange(j0, j1 + 1)

    Uref = Uhist[j0]
    gate_ref = (X @ Uref.T > 0.0).astype(float)

    branch = branch_from_saved_state(state)
    phi_eval = sg.interp_branch(branch, times[idx])
    if np.any(~np.isfinite(phi_eval)):
        raise RuntimeError("Weak branch does not cover ID audit interval.")

    rows = []
    for tt, j in enumerate(idx):
        U = Uhist[j]
        W = Whist[j]

        F = exp.relu_forward(U, W, X)

        # All-neuron frozen-gate reconstruction.
        Z = X @ U.T
        F_frozen = (gate_ref * Z) @ W

        # Dynamic two-block selector reconstruction.
        d1 = (X[:, 0] > 0.0).astype(float)
        uW = exp.unit_direction_from_deg(phi_eval[tt])
        d2 = (X @ uW > 0.0).astype(float)
        F_block = (
            d1[:, None] * (X @ M1hist[j].T)
            + d2[:, None] * (X @ M2hist[j].T)
        )

        row = {
            "time": float(times[j]),
            "frozen_gate_rel_rms_all": relative_rms(F, F_frozen),
            "block_selector_rel_rms_all": relative_rms(F, F_block),
        }

        # make_data uses labels 0 and 1 for concept clusters 1 and 2.
        for cluster_name, label_value in ((1, 0), (2, 1)):
            mask = labels == label_value
            row[f"frozen_gate_rel_rms_cluster{cluster_name}"] = relative_rms(
                F[mask], F_frozen[mask]
            )
            row[f"block_selector_rel_rms_cluster{cluster_name}"] = relative_rms(
                F[mask], F_block[mask]
            )
        rows.append(row)

    return rows


def boundary_distance_over_window(psi_deg, phi_eval):
    psi = np.asarray(psi_deg, dtype=float)

    # Strong gate boundaries: +/-90 deg.
    dstrong = np.minimum(
        angular_abs_deg(psi, 90.0),
        angular_abs_deg(psi, -90.0),
    )

    # Weak gate boundaries move as phi(t) +/- 90 deg.
    dweak = np.full(len(psi), np.inf, dtype=float)
    for phi in np.asarray(phi_eval, dtype=float):
        d = np.minimum(
            angular_abs_deg(psi, float(sg.wrap_deg(phi + 90.0))),
            angular_abs_deg(psi, float(sg.wrap_deg(phi - 90.0))),
        )
        dweak = np.minimum(dweak, d)

    return np.minimum(dstrong, dweak)


def phase_losses_and_boundary(prereg, state):
    times = np.asarray(state["time"], dtype=float)
    Uhist = np.asarray(state["U"], dtype=float)
    Whist = np.asarray(state["W"], dtype=float)
    M1hist = np.asarray(state["M1"], dtype=float)
    M2hist = np.asarray(state["M2"], dtype=float)

    tW = float(prereg["post_specialization"]["t_W"])
    j0 = sg.nearest_index(times, tW)
    j1 = sg.nearest_index(times, sg.T_END)
    idx = np.arange(j0, j1 + 1)
    t_eval = times[idx]

    branch = branch_from_saved_state(state)
    phi_eval = sg.interp_branch(branch, t_eval)

    psi = sg.PSI_GRID_DEG.copy()
    Xdir = sg.direction_vectors(psi)

    T = len(idx)
    K = len(psi)
    actual_loss = np.empty((T, K), dtype=float)
    pred_loss = np.empty((T, K), dtype=float)
    actual_out = np.empty((T, K, 2), dtype=float)
    pred_out = np.empty((T, K, 2), dtype=float)

    d1 = (Xdir[:, 0] > 0.0).astype(float)

    for tt, j in enumerate(idx):
        U = Uhist[j]
        W = Whist[j]
        F = exp.relu_forward(U, W, Xdir)

        uW = exp.unit_direction_from_deg(phi_eval[tt])
        d2 = (Xdir @ uW > 0.0).astype(float)
        Fp = (
            d1[:, None] * (Xdir @ M1hist[j].T)
            + d2[:, None] * (Xdir @ M2hist[j].T)
        )

        actual_out[tt] = F
        pred_out[tt] = Fp
        actual_loss[tt] = 0.5 * np.sum((F - Xdir) ** 2, axis=1)
        pred_loss[tt] = 0.5 * np.sum((Fp - Xdir) ** 2, axis=1)

    num = np.sum((actual_out - pred_out) ** 2, axis=(0, 2))
    den = np.sum(actual_out ** 2, axis=(0, 2))
    model_rms = np.sqrt(num / np.maximum(den, 1e-12))

    dbound = boundary_distance_over_window(psi, phi_eval)

    return {
        "times": t_eval,
        "psi_deg": psi,
        "actual_loss": actual_loss,
        "pred_loss": pred_loss,
        "model_rms": model_rms,
        "boundary_distance_deg": dbound,
    }


def classify_direction(times, loss):
    loss = np.asarray(loss, dtype=float)
    S, _ = sg.phase_from_losses(
        np.asarray(times, dtype=float),
        loss[:, None],
    )
    swing = float(S[0])
    spread = float(np.max(loss) - np.min(loss))
    net_improvement = float(loss[0] - loss[-1])

    if swing >= SWING_THRESHOLD:
        label = "swing"
    elif spread < NEGLIGIBLE_RANGE_THRESHOLD:
        label = "negligible"
    elif net_improvement > 0:
        label = "no_swing_improving"
    else:
        label = "no_swing_worsening"

    return label, swing, spread, net_improvement


def run_grid_audit():
    verify_freeze()
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    master_id = []
    boundary_rows = []
    sector_rows = []

    for mu2 in sg.MU2_GRID:
        for sigma in sg.SIGMA_GRID:
            cid = cell_id(mu2, sigma)
            print()
            print("=" * 88)
            print(f"GRID AUDIT {cid}")
            print("=" * 88)

            cdir = PREREG_ROOT / "cells" / cid
            prereg = read_json(cdir / "preregistration.json")
            state = sg.load_training_state(
                cdir / prereg["files"]["training_state"]
            )

            # ID validity audit.
            id_rows = evaluate_id_reduction(prereg, state)
            c_out = AUDIT_ROOT / "cells" / cid
            write_csv(c_out / "id_reduction_over_time.csv", id_rows)

            for r in id_rows:
                master_id.append({
                    "cell_id": cid,
                    "mu2": mu2,
                    "sigma": sigma,
                    **r,
                })

            # OOD boundary + sector audit.
            ph = phase_losses_and_boundary(prereg, state)

            for k, psi in enumerate(ph["psi_deg"]):
                a_label, aS, aRange, aNet = classify_direction(
                    ph["times"], ph["actual_loss"][:, k]
                )
                p_label, pS, pRange, pNet = classify_direction(
                    ph["times"], ph["pred_loss"][:, k]
                )

                row = {
                    "cell_id": cid,
                    "mu2": mu2,
                    "sigma": sigma,
                    "psi_deg": float(psi),
                    "boundary_distance_deg": float(
                        ph["boundary_distance_deg"][k]
                    ),
                    "selector_model_relative_rms": float(ph["model_rms"][k]),
                    "actual_label": a_label,
                    "pred_label": p_label,
                    "match": int(a_label == p_label),
                    "actual_swing": aS,
                    "pred_swing": pS,
                    "actual_loss_range": aRange,
                    "pred_loss_range": pRange,
                    "actual_net_improvement": aNet,
                    "pred_net_improvement": pNet,
                }
                boundary_rows.append(row)
                sector_rows.append(row)

            cell_boundary = [
                r for r in boundary_rows if r["cell_id"] == cid
            ]
            all_acc = np.mean([r["match"] for r in cell_boundary])
            far = [
                r for r in cell_boundary
                if r["boundary_distance_deg"] >= BOUNDARY_EXCLUDE_DEG
            ]
            far_acc = np.mean([r["match"] for r in far]) if far else np.nan

            print(
                f"frozen-gate RMS max={max(r['frozen_gate_rel_rms_all'] for r in id_rows):.4f}, "
                f"block RMS max={max(r['block_selector_rel_rms_all'] for r in id_rows):.4f}"
            )
            print(
                f"sector agreement all={all_acc:.3f}, "
                f"boundary-excluded={far_acc:.3f}"
            )

    write_csv(AUDIT_ROOT / "id_reduction_all_cells.csv", master_id)
    write_csv(AUDIT_ROOT / "boundary_sector_all_directions.csv", sector_rows)

    # Per-cell ID summary.
    id_summary = []
    for cid in sorted(set(r["cell_id"] for r in master_id)):
        rr = [r for r in master_id if r["cell_id"] == cid]
        for metric in (
            "frozen_gate_rel_rms_all",
            "block_selector_rel_rms_all",
            "frozen_gate_rel_rms_cluster1",
            "frozen_gate_rel_rms_cluster2",
            "block_selector_rel_rms_cluster1",
            "block_selector_rel_rms_cluster2",
        ):
            a = np.asarray([r[metric] for r in rr], dtype=float)
            id_summary.append({
                "cell_id": cid,
                "metric": metric,
                "mean": float(np.mean(a)),
                "median": float(np.median(a)),
                "max": float(np.max(a)),
                "p95": float(np.quantile(a, 0.95)),
            })
    write_csv(AUDIT_ROOT / "id_reduction_summary.csv", id_summary)

    # Boundary bins aggregated over all cells.
    bin_rows = []
    for lo, hi in zip(BOUNDARY_BINS[:-1], BOUNDARY_BINS[1:]):
        rr = [
            r for r in boundary_rows
            if lo <= r["boundary_distance_deg"] < hi
        ]
        if not rr:
            continue
        a = np.asarray(
            [r["selector_model_relative_rms"] for r in rr],
            dtype=float,
        )
        bin_rows.append({
            "distance_lo_deg": lo,
            "distance_hi_deg": hi,
            "n": len(rr),
            "median_selector_model_rms": float(np.median(a)),
            "mean_selector_model_rms": float(np.mean(a)),
            "p90_selector_model_rms": float(np.quantile(a, 0.90)),
            "p95_selector_model_rms": float(np.quantile(a, 0.95)),
        })
    write_csv(AUDIT_ROOT / "boundary_layer_bins.csv", bin_rows)

    # Sector confusion and per-cell agreement.
    labels = [
        "swing",
        "negligible",
        "no_swing_improving",
        "no_swing_worsening",
    ]

    confusion_rows = []
    for scope_name, rr in (
        ("all", sector_rows),
        (
            f"boundary_ge_{BOUNDARY_EXCLUDE_DEG:g}deg",
            [
                r for r in sector_rows
                if r["boundary_distance_deg"] >= BOUNDARY_EXCLUDE_DEG
            ],
        ),
    ):
        for a in labels:
            for p in labels:
                n = sum(
                    1 for r in rr
                    if r["actual_label"] == a and r["pred_label"] == p
                )
                confusion_rows.append({
                    "scope": scope_name,
                    "actual_label": a,
                    "pred_label": p,
                    "count": n,
                })
    write_csv(AUDIT_ROOT / "sector_confusion.csv", confusion_rows)

    accuracy_rows = []
    for cid in sorted(set(r["cell_id"] for r in sector_rows)):
        rr = [r for r in sector_rows if r["cell_id"] == cid]
        for scope_name, sub in (
            ("all", rr),
            (
                f"boundary_ge_{BOUNDARY_EXCLUDE_DEG:g}deg",
                [
                    r for r in rr
                    if r["boundary_distance_deg"] >= BOUNDARY_EXCLUDE_DEG
                ],
            ),
        ):
            accuracy_rows.append({
                "cell_id": cid,
                "scope": scope_name,
                "n": len(sub),
                "accuracy": (
                    float(np.mean([r["match"] for r in sub]))
                    if sub else np.nan
                ),
            })
    write_csv(AUDIT_ROOT / "sector_accuracy_by_cell.csv", accuracy_rows)

    # Figures.
    fig = plt.figure(figsize=(7, 4.5))
    x = [
        0.5 * (r["distance_lo_deg"] + r["distance_hi_deg"])
        if r["distance_hi_deg"] < 100
        else 30.0
        for r in bin_rows
    ]
    med = [r["median_selector_model_rms"] for r in bin_rows]
    p90 = [r["p90_selector_model_rms"] for r in bin_rows]
    plt.plot(x, med, marker="o", label="median")
    plt.plot(x, p90, marker="x", label="90th percentile")
    plt.axvline(BOUNDARY_EXCLUDE_DEG, linestyle="--")
    plt.xlabel("minimum distance to learned gate boundary (deg)")
    plt.ylabel("selector-model relative RMS")
    plt.title("Boundary-layer localization of reduction error")
    plt.legend()
    fig.tight_layout()
    fig.savefig(AUDIT_ROOT / "boundary_layer_error.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4.5))
    cids = sorted(set(r["cell_id"] for r in accuracy_rows))
    xs = np.arange(len(cids))
    all_acc = [
        next(
            r["accuracy"] for r in accuracy_rows
            if r["cell_id"] == cid and r["scope"] == "all"
        )
        for cid in cids
    ]
    far_acc = [
        next(
            r["accuracy"] for r in accuracy_rows
            if r["cell_id"] == cid and r["scope"] != "all"
        )
        for cid in cids
    ]
    width = 0.38
    plt.bar(xs - width / 2, all_acc, width=width, label="all")
    plt.bar(xs + width / 2, far_acc, width=width, label=f">={BOUNDARY_EXCLUDE_DEG:g} deg")
    plt.xticks(xs, [c.replace("mu2_", "").replace("__sigma_", ",") for c in cids],
               rotation=35, ha="right")
    plt.ylabel("sector classification agreement")
    plt.ylim(0.0, 1.05)
    plt.title("Full vs reduced OOD sector classification")
    plt.legend()
    fig.tight_layout()
    fig.savefig(AUDIT_ROOT / "sector_accuracy_by_cell.pdf")
    plt.close(fig)

    print()
    print("GRID REDUCTION / BOUNDARY / SECTOR AUDIT COMPLETE")
    print(f"outputs: {AUDIT_ROOT.resolve()}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def run_summary():
    verify_freeze()

    out = {
        "created_utc": utc_now(),
        "freeze_sha256": sha256_file(OUT_ROOT / "freeze.json"),
    }

    mpath = MULTISEED_ROOT / "multiseed_summary.csv"
    if mpath.exists():
        with mpath.open() as f:
            mrows = list(csv.DictReader(f))

        def arr(name):
            a = np.asarray([float(r[name]) for r in mrows], dtype=float)
            return a[np.isfinite(a)]

        out["multiseed"] = {
            "n_runs": len(mrows),
            "mean_phase_corr": float(np.mean(arr("global_corr_Sactual_Sred"))),
            "min_phase_corr": float(np.min(arr("global_corr_Sactual_Sred"))),
            "mean_p1_angle_error_deg": float(np.mean(arr("p1_peak_angle_error_deg"))),
            "max_p1_angle_error_deg": float(np.max(arr("p1_peak_angle_error_deg"))),
            "mean_p2_angle_error_deg": float(np.mean(arr("p2_peak_angle_error_deg"))),
            "max_p2_angle_error_deg": float(np.max(arr("p2_peak_angle_error_deg"))),
            "mean_abs_G_timing_error": float(
                np.mean(np.abs(arr("tG_minus_tfull_probe")))
            ),
            "max_abs_G_timing_error": float(
                np.max(np.abs(arr("tG_minus_tfull_probe")))
            ),
        }

    ipath = AUDIT_ROOT / "id_reduction_all_cells.csv"
    spath = AUDIT_ROOT / "sector_accuracy_by_cell.csv"
    bpath = AUDIT_ROOT / "boundary_layer_bins.csv"

    if ipath.exists():
        with ipath.open() as f:
            rows = list(csv.DictReader(f))

        fg = np.asarray(
            [float(r["frozen_gate_rel_rms_all"]) for r in rows], dtype=float
        )
        br = np.asarray(
            [float(r["block_selector_rel_rms_all"]) for r in rows], dtype=float
        )
        out["id_reduction"] = {
            "frozen_gate_mean_rel_rms": float(np.mean(fg)),
            "frozen_gate_max_rel_rms": float(np.max(fg)),
            "block_selector_mean_rel_rms": float(np.mean(br)),
            "block_selector_max_rel_rms": float(np.max(br)),
        }

    if spath.exists():
        with spath.open() as f:
            rows = list(csv.DictReader(f))
        all_a = [
            float(r["accuracy"]) for r in rows if r["scope"] == "all"
        ]
        far_a = [
            float(r["accuracy"]) for r in rows if r["scope"] != "all"
        ]
        out["sector_classification"] = {
            "mean_accuracy_all": float(np.mean(all_a)),
            "min_accuracy_all": float(np.min(all_a)),
            "mean_accuracy_boundary_excluded": float(np.mean(far_a)),
            "min_accuracy_boundary_excluded": float(np.min(far_a)),
        }

    if bpath.exists():
        with bpath.open() as f:
            out["boundary_bins"] = list(csv.DictReader(f))

    write_json(OUT_ROOT / "final_audit_summary.json", out)

    print(json.dumps(out, indent=2))
    print()
    print(f"saved: {(OUT_ROOT / 'final_audit_summary.json').resolve()}")


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def self_test():
    sg.self_test_basis_and_eta()

    # Sector classifier.
    t = np.linspace(0.0, 2.0, 201)

    lab, _, _, _ = classify_direction(t, np.ones_like(t))
    assert lab == "negligible"

    lab, _, _, _ = classify_direction(t, np.linspace(1.0, 0.0, len(t)))
    assert lab == "no_swing_improving"

    lab, _, _, _ = classify_direction(t, np.linspace(0.0, 1.0, len(t)))
    assert lab == "no_swing_worsening"

    # Persistent decline -> local minimum -> rebound -> hold.
    loss = np.empty_like(t)
    loss[:81] = np.linspace(1.0, 0.0, 81)
    loss[81:121] = np.linspace(0.0, 0.5, 40)
    loss[121:] = 0.5
    lab, S, _, _ = classify_direction(t, loss)
    assert lab == "swing"
    assert S >= SWING_THRESHOLD

    # Boundary-distance sanity.
    psi = np.array([90.0, 0.0, -90.0])
    d = boundary_distance_over_window(psi, np.array([90.0]))
    assert abs(d[0]) < 1e-12
    assert abs(d[2]) < 1e-12

    # Frozen gate is exact at the reference instant.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 2))
    U = rng.normal(size=(7, 2))
    W = rng.normal(size=(7, 2))
    F = exp.relu_forward(U, W, X)
    Z = X @ U.T
    G = (Z > 0.0).astype(float)
    Ff = (G * Z) @ W
    assert np.allclose(F, Ff, atol=1e-12, rtol=1e-12)

    print("sector classifier tests: PASS")
    print("boundary-distance tests: PASS")
    print("frozen-gate exact-at-reference test: PASS")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["self-test", "freeze", "multiseed", "audit", "summarize"],
    )
    args = parser.parse_args()

    if args.stage == "self-test":
        self_test()
    elif args.stage == "freeze":
        self_test()
        freeze()
    elif args.stage == "multiseed":
        self_test()
        run_multiseed()
    elif args.stage == "audit":
        self_test()
        run_grid_audit()
    elif args.stage == "summarize":
        self_test()
        run_summary()


if __name__ == "__main__":
    main()
