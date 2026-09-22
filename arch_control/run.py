"""Run the pre-specified experiments.  Resumable; one .npz per run.

Examples
--------
    python -m arch_control.run --dry-run --all
    python -m arch_control.run --all --device cuda
    python -m arch_control.run --experiments E1_canonical E5_stepsize --device cpu
    python -m arch_control.run --smoke --out /tmp/ac_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

from arch_control.experiments import (ALL_EXPERIMENTS, NAMED_PROBES, RunSpec,  # noqa: E402
                                      experiment_specs, make_inputs, smoke_specs)
from arch_control.simulate import GRID_DEG, BatchInputs, simulate_batch  # noqa: E402
from arch_control import metrics as M  # noqa: E402

# Full 720-direction error curves are stored only where figures/re-analysis need them;
# per-direction statistics are always computed here and stored for every run.
STORE_GRID = ("E1_canonical", "E2_grid", "E5_stepsize", "SMOKE")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


FROZEN_FILES = ("PROTOCOL.md", "experiments.py", "simulate.py", "metrics.py", "run.py")


def git_dirty() -> bool:
    """True if any frozen protocol/code file is uncommitted or modified.

    Only the frozen files are checked, so results written under arch_control/results
    (or anywhere else) never block later invocations."""
    try:
        paths = [f"arch_control/{f}" for f in FROZEN_FILES]
        out = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain",
                                       "--untracked-files=all", "--", *paths],
                                      text=True, stderr=subprocess.DEVNULL)
        tracked = subprocess.check_output(["git", "-C", str(REPO), "ls-files", "--", *paths],
                                          text=True, stderr=subprocess.DEVNULL).split()
        return bool(out.strip()) or len(tracked) < len(paths)
    except Exception:
        return True


def protocol_hashes() -> dict:
    return {name: sha256_file(HERE / name) for name in
            ("PROTOCOL.md", "experiments.py", "simulate.py", "metrics.py")
            if (HERE / name).exists()}


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_path(out: Path, spec: RunSpec) -> Path:
    return out / spec.exp / "runs" / f"{spec.run_id}.npz"


def is_done(out: Path, spec: RunSpec) -> bool:
    p = run_path(out, spec)
    if not p.exists():
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            return json.loads(str(z["spec_json"])) == asdict(spec)
    except Exception:
        return False


def bytes_per_run(spec: RunSpec) -> int:
    N = 2 * spec.n_per_cluster
    return 8 * N * spec.width * 5 + 8 * 1000 * spec.width  # activations + slack


def chunk(specs, max_batch, mem_bytes):
    per = max(bytes_per_run(s) for s in specs)
    b = max(1, min(max_batch, mem_bytes // per))
    for i in range(0, len(specs), b):
        yield specs[i:i + b]


def cost_units(spec: RunSpec) -> float:
    return 2 * spec.n_per_cluster * spec.width * spec.t_end / spec.dt


def save_run(out: Path, spec: RunSpec, res, b: int, meta: dict) -> Path:
    p = run_path(out, spec)
    p.parent.mkdir(parents=True, exist_ok=True)
    Eg = res.E_grid[b].astype(np.float64)
    pg = M.persistent_increase(res.times, Eg)
    eg = M.endpoint_stats(res.times, Eg)
    arrays = dict(
        spec_json=np.array(json.dumps(asdict(spec), sort_keys=True)),
        times=res.times, grid_deg=GRID_DEG, named=np.array(NAMED_PROBES),
        E_named=res.E_named[b],
        loss_total=res.loss_total[b], loss_c1=res.loss_c1[b], loss_c2=res.loss_c2[b],
        grid_S=pg["swing"], grid_t_rev=pg["t_reversal"], grid_rise=M.raw_rise(Eg),
        grid_final_excess=eg["final_excess"], grid_t_min=eg["t_min"],
        grid_E0=eg["E0"], grid_Eend=eg["E_end"],
        antipodal_asym=np.array(M.antipodal_asymmetry(Eg)),
    )
    if spec.exp in STORE_GRID:
        arrays["E_grid"] = res.E_grid[b]
    if res.U_hist is not None:
        arrays["U_hist"] = res.U_hist[b]; arrays["W_hist"] = res.W_hist[b]
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, p)
    rec = dict(run_id=spec.run_id, spec=asdict(spec), sha256=sha256_file(p),
               batch_wall_seconds=res.wall_seconds, **meta)
    with open(out / spec.exp / "manifest.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="all protocol experiments, in priority order")
    g.add_argument("--experiments", nargs="+", choices=ALL_EXPERIMENTS)
    g.add_argument("--smoke", action="store_true", help="tiny end-to-end check (not a protocol run)")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | cpu")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                    help="float64 is the protocol setting (matches the archive)")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--mem-gb", type=float, default=6.0, help="activation memory budget per batch")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run even if arch_control/ has uncommitted changes")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        specs = smoke_specs()
    else:
        names = ALL_EXPERIMENTS if args.all else args.experiments
        specs = [s for n in names for s in experiment_specs(n)]

    pending = [s for s in specs if not is_done(out, s)]
    total_cost = sum(cost_units(s) for s in specs)
    todo_cost = sum(cost_units(s) for s in pending)
    by_exp = defaultdict(lambda: [0, 0, 0.0])
    for s in specs:
        by_exp[s.exp][0] += 1
    for s in pending:
        by_exp[s.exp][1] += 1; by_exp[s.exp][2] += cost_units(s)
    print("experiment              runs  pending  pending work (N*h*steps)")
    for k, (n, p, c) in by_exp.items():
        print(f"  {k:22s}{n:5d}  {p:7d}  {c:10.2e}")
    print(f"work units (N*h*steps): total {total_cost:.3g}, pending {todo_cost:.3g}")
    if args.dry_run:
        print("dry run: nothing executed")
        return 0
    if not args.smoke and git_dirty() and not args.allow_dirty:
        print("ERROR: the frozen files " + ", ".join(FROZEN_FILES) + " are not all committed "
              "and unmodified.  Commit them first (the manifest records the commit), or pass "
              "--allow-dirty.")
        return 2

    device = pick_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if device.startswith("mps") and dtype == torch.float64:
        print("ERROR: Apple MPS has no float64.  Use --device cpu (protocol) or "
              "--dtype float32 (not a protocol run; record it as a deviation).")
        return 2
    if args.dtype != "float64" and not args.smoke:
        print("WARNING: float32 is not the protocol setting; results are marked dtype=float32 "
              "in the manifest.")
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    meta = dict(device=device, dtype=args.dtype, torch=torch.__version__, numpy=np.__version__,
                python=platform.python_version(), host=platform.node(),
                gpu=(torch.cuda.get_device_name(torch.device(device))
                     if device.startswith("cuda") else None),
                git_commit=git_commit(), git_dirty=git_dirty(),
                protocol_sha256=protocol_hashes(), smoke=bool(args.smoke))
    print(f"device={device} dtype={args.dtype} torch={torch.__version__} commit={meta['git_commit'][:10]}")

    groups = defaultdict(list)
    for s in pending:
        groups[s.batch_key].append(s)
    # cheapest groups first so partial results appear early
    order = sorted(groups, key=lambda k: sum(cost_units(s) for s in groups[k]))
    mem = int(args.mem_gb * 2**30)
    done_cost = 0.0; t_start = time.time()
    for key in order:
        for batch in chunk(groups[key], args.max_batch, mem):
            arch, h, N, dt, t_end, log_dt, save = key
            print(f"[{time.strftime('%H:%M:%S')}] {batch[0].exp}: arch={arch} h={h} "
                  f"N={N} T={t_end} dt={dt} batch={len(batch)}", flush=True)
            ins = [make_inputs(s) for s in batch]
            inputs = BatchInputs(*(np.stack([x[i] for x in ins]) for i in range(5)))
            res = simulate_batch(arch, inputs, dt, t_end, log_dt, device=device, dtype=dtype,
                                 save_params=save,
                                 progress=lambda m: print(m, flush=True))
            for b, s in enumerate(batch):
                save_run(out, s, res, b, meta)
            done_cost += sum(cost_units(s) for s in batch)
            el = time.time() - t_start
            print(f"    saved {len(batch)} runs in {res.wall_seconds/60:.1f} min; "
                  f"overall {100*done_cost/max(todo_cost,1):.1f}% "
                  f"eta {el*(todo_cost-done_cost)/max(done_cost,1)/3600:.2f} h", flush=True)
    print("all requested runs complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
