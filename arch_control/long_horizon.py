"""E6_long_horizon: pre-registered amendment to PROTOCOL.md, added after E1-E5 were run.

Why
---
H3 showed that the ReLU -85 deg reversal has not recovered by t = 20.  But at t = 20 the
ReLU training loss is still about 1.7e-4 (linear: about 1e-26), so the ReLU network has not
converged.  E6 asks whether the reversal recovers later.

Design
------
E1's canonical matched pairs run to T = 100, with nothing else changed: same data and
initialization per seed, dt = 2e-4, log_dt = 0.01, float64, 10 seeds x {relu, linear}.
The frozen files (PROTOCOL.md, experiments.py, simulate.py, metrics.py, run.py) are imported,
not modified, so the protocol hashes recorded for E1-E5 still match.  Statistics use the
frozen detector and labels in metrics.py.

Disclosure
----------
Before this file was written, one exploratory ReLU run (seed 0, dt = 2e-3, T = 100, CPU)
lost 89.7% of the early -85 deg gain by t = 20 and 99.1% by t = 100, and E(-85 deg) never
decreased after t = 20.  The predictions below were set after seeing that single run.

Predictions (fixed by the commit that adds this file; evaluated on [0, 100])
-----------------------------------------------------------------------------
L1 (no recovery)  In >= 9/10 ReLU seeds the -85 deg probe is labelled `persistent` at 1e-4,
                  and E(-85 deg, t) never drops by more than 1e-6 for 20 <= s <= t <= 100.
L2 (gain lost)    In >= 9/10 ReLU seeds, (E(100) - E_min) / (E0 - E_min) >= 0.95 at -85 deg.
L3 (cone)         No ReLU seed has an in-cone direction labelled `persistent` at 1e-4.
L4 (linear)       No linear seed has any direction with S >= 1e-5 (H2a extended to T = 100).
C1 (consistency)  Not a prediction.  Truncated at t = 20, E6 should reproduce E1's -85 deg
                  statistics; checked when E1 results are present under --out.
Reported without prediction: fraction of the early gain lost at t = 20, 40, 60, 100;
off-cone persistent directions at T = 20 vs T = 100; training loss at t = 100.

Usage
-----
    python -m arch_control.long_horizon --dry-run
    python -m arch_control.long_horizon --device cuda --out $OUT
    python -m arch_control.long_horizon --analyze-only --out $OUT
    python -m arch_control.long_horizon --smoke --out /tmp/e6_smoke   # plumbing check only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

from arch_control import metrics as M  # noqa: E402
from arch_control import run as R  # noqa: E402  (frozen runner: provenance helpers only)
from arch_control.experiments import CANONICAL, NAMED_PROBES, RunSpec, make_inputs  # noqa: E402
from arch_control.simulate import GRID_DEG, BatchInputs, simulate_batch  # noqa: E402

EXP = "E6_long_horizon"
THIS = "long_horizon.py"
TAU = 1e-4
GRID_EVERY = 10          # store the 720-direction curves every 0.1 time units (figures only)


# ----------------------------------------------------------------------------- specs

def make_specs(smoke: bool = False) -> list[RunSpec]:
    if smoke:
        return [RunSpec("E6_SMOKE", arch, **CANONICAL, n_per_cluster=200, width=20,
                        epsilon=1e-3, init="iso", seed=s, dt=2e-4, t_end=2.0)
                for arch in ("relu", "linear") for s in (0, 1)]
    return [RunSpec(EXP, arch, **CANONICAL, n_per_cluster=2000, width=200, epsilon=1e-3,
                    init="iso", seed=s, dt=2e-4, t_end=100.0)
            for arch in ("relu", "linear") for s in range(10)]


def horizons(spec: RunSpec) -> tuple[float, float, tuple[float, ...]]:
    """(E1 horizon, full horizon, check times)."""
    if spec.t_end == 100.0:
        return 20.0, 100.0, (20.0, 40.0, 60.0, 100.0)
    h = spec.t_end / 2
    return h, spec.t_end, (h, spec.t_end)


# ----------------------------------------------------------------------------- provenance

def this_file_dirty() -> bool:
    p = f"arch_control/{THIS}"
    try:
        out = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain",
                                       "--untracked-files=all", "--", p],
                                      text=True, stderr=subprocess.DEVNULL)
        tracked = subprocess.check_output(["git", "-C", str(REPO), "ls-files", "--", p],
                                          text=True, stderr=subprocess.DEVNULL).split()
        return bool(out.strip()) or not tracked
    except Exception:
        return True


# ----------------------------------------------------------------------------- statistics

def _idx(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - t)))


def horizon_stats(times, E_grid, E_named, t_h) -> dict:
    """Frozen per-direction statistics on [0, t_h]."""
    k = _idx(times, t_h)
    t = times[:k + 1]
    out = {}
    for tag, Y in (("grid", np.asarray(E_grid[:k + 1], dtype=np.float64)),
                   ("named", np.asarray(E_named[:k + 1], dtype=np.float64))):
        p = M.persistent_increase(t, Y)
        e = M.endpoint_stats(t, Y)
        out[f"{tag}_S"] = p["swing"]
        out[f"{tag}_rise"] = M.raw_rise(Y)
        out[f"{tag}_final_excess"] = e["final_excess"]
        out[f"{tag}_t_min"] = e["t_min"]
        out[f"{tag}_Emin"] = e["E_min"]
        out[f"{tag}_Eend"] = e["E_end"]
    return out


def save_run(out: Path, spec: RunSpec, res, b: int, meta: dict) -> Path:
    p = R.run_path(out, spec)
    p.parent.mkdir(parents=True, exist_ok=True)
    t_split, t_full, _ = horizons(spec)
    Eg = res.E_grid[b]; En = res.E_named[b]
    arrays = dict(
        spec_json=np.array(json.dumps(asdict(spec), sort_keys=True)),
        times=res.times, grid_deg=GRID_DEG, named=np.array(NAMED_PROBES),
        E_named=En, loss_total=res.loss_total[b], loss_c1=res.loss_c1[b],
        loss_c2=res.loss_c2[b],
        times_sub=res.times[::GRID_EVERY], E_grid_sub=Eg[::GRID_EVERY],
        antipodal_asym=np.array(M.antipodal_asymmetry(Eg)),
        horizons=np.array([t_split, t_full]),
    )
    for tag, th in (("h_split", t_split), ("h_full", t_full)):
        for k, v in horizon_stats(res.times, Eg, En, th).items():
            arrays[f"{tag}__{k}"] = v
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, p)
    rec = dict(run_id=spec.run_id, spec=asdict(spec), sha256=R.sha256_file(p),
               batch_wall_seconds=res.wall_seconds, **meta)
    with open(out / spec.exp / "manifest.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return p


# ----------------------------------------------------------------------------- run

def run(args, specs) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pending = [s for s in specs if not R.is_done(out, s)]
    print(f"{specs[0].exp}: {len(specs)} runs, {len(pending)} pending "
          f"(T = {specs[0].t_end}, dt = {specs[0].dt})")
    if args.dry_run or not pending:
        return 0
    frozen_dirty = R.git_dirty(); mine_dirty = this_file_dirty()
    if not args.smoke and (frozen_dirty or mine_dirty) and not args.allow_dirty:
        print("ERROR: commit arch_control/long_horizon.py (and leave the frozen files "
              "unmodified) before running, so the manifest records the predictions' commit. "
              "Or pass --allow-dirty.")
        return 2
    device = R.pick_device(args.device)
    if device.startswith("mps"):
        print("ERROR: Apple MPS has no float64; use --device cpu or cuda.")
        return 2
    torch.use_deterministic_algorithms(True, warn_only=True)
    meta = dict(device=device, dtype="float64", torch=torch.__version__,
                numpy=np.__version__, python=platform.python_version(), host=platform.node(),
                gpu=(torch.cuda.get_device_name(torch.device(device))
                     if device.startswith("cuda") else None),
                git_commit=R.git_commit(), git_dirty=frozen_dirty,
                long_horizon_dirty=mine_dirty, protocol_sha256=R.protocol_hashes(),
                long_horizon_sha256=R.sha256_file(HERE / THIS), smoke=bool(args.smoke))
    print(f"device={device} commit={meta['git_commit'][:10]} "
          f"frozen_dirty={frozen_dirty} long_horizon_dirty={mine_dirty}")
    for arch in ("relu", "linear"):
        batch = [s for s in pending if s.arch == arch]
        if not batch:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {arch}: {len(batch)} seeds, T = {batch[0].t_end}",
              flush=True)
        ins = [make_inputs(s) for s in batch]
        inputs = BatchInputs(*(np.stack([x[i] for x in ins]) for i in range(5)))
        res = simulate_batch(arch, inputs, batch[0].dt, batch[0].t_end, batch[0].log_dt,
                             device=device, dtype=torch.float64, save_params=False,
                             progress=lambda m: print(m, flush=True))
        t0 = time.time()
        for b, s in enumerate(batch):
            save_run(out, s, res, b, meta)
        print(f"    simulated in {res.wall_seconds/60:.1f} min, statistics and save "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        del res
    return 0


# ----------------------------------------------------------------------------- analysis

def load(out: Path, exp: str) -> list[dict]:
    rows = []
    for p in sorted((out / exp / "runs").glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        with np.load(p, allow_pickle=False) as z:
            r = {k: z[k] for k in z.files}
        r["spec"] = json.loads(str(r["spec_json"]))
        rows.append(r)
    return rows


def run_row(r: dict) -> dict:
    spec = r["spec"]; t = r["times"]; named = list(r["named"])
    t_split, t_full = [float(x) for x in r["horizons"]]
    _, _, checks = horizons(RunSpec(**spec))
    im = named.index("m85")
    y = r["E_named"][:, im]
    E0 = float(y[0]); kmin = int(np.argmin(y)); Emin = float(y[kmin])
    ks = _idx(t, t_split)
    tail = y[ks:]
    max_drop = float((np.maximum.accumulate(tail) - tail).max())
    incone = M.in_closed_positive_cone(r["grid_deg"])
    row = dict(arch=spec["arch"], seed=spec["seed"], run_id=RunSpec(**spec).run_id,
               m85_E0=E0, m85_Emin=Emin, m85_t_min=float(t[kmin]),
               m85_max_drop_after_split=max_drop)
    for tc in checks:
        k = _idx(t, tc)
        row[f"m85_E@{tc:g}"] = float(y[k])
        row[f"gain_lost@{tc:g}"] = (float((y[k] - Emin) / (E0 - Emin))
                                    if (E0 > Emin and k >= kmin) else float("nan"))
        row[f"loss@{tc:g}"] = float(r["loss_total"][k])
    for tag, th in (("h_split", t_split), ("h_full", t_full)):
        g = lambda k: r[f"{tag}__{k}"]  # noqa: E731
        lab = M.classify(g("grid_rise"), g("grid_S"), g("grid_final_excess"), TAU)
        labn = M.classify(g("named_rise"), g("named_S"), g("named_final_excess"), TAU)
        row[f"m85_S@{th:g}"] = float(g("named_S")[im])
        row[f"m85_Eend@{th:g}"] = float(g("named_Eend")[im])
        row[f"m85_t_min@{th:g}"] = float(g("named_t_min")[im])
        row[f"m85_class@{th:g}"] = str(labn[im])
        row[f"n_off_persistent@{th:g}"] = int(((lab == "persistent") & ~incone).sum())
        row[f"n_in_persistent@{th:g}"] = int(((lab == "persistent") & incone).sum())
        row[f"n_off_S_ge_1e-04@{th:g}"] = int(((g("grid_S") >= 1e-4) & ~incone).sum())
        row[f"n_in_S_ge_1e-04@{th:g}"] = int(((g("grid_S") >= 1e-4) & incone).sum())
        row[f"n_any_S_ge_1e-05@{th:g}"] = int((g("grid_S") >= 1e-5).sum())
        row[f"max_in_final_excess@{th:g}"] = float(g("grid_final_excess")[incone].max())
    return row


def c1_check(out: Path, rows: list[dict], t_split: float) -> list[str]:
    p = out / "E1_canonical" / "run_summary.csv"
    if not p.exists():
        return [f"- **C1** skipped: `{p}` not found (E1 results not under --out)."]
    e1 = {(r["arch"], int(r["seed"])): r for r in csv.DictReader(open(p))}
    dS = dE = dT = 0.0; dcount = 0; n = 0
    for r in rows:
        k = (r["arch"], int(r["seed"]))
        if k not in e1:
            continue
        n += 1
        a = e1[k]
        dS = max(dS, abs(r[f"m85_S@{t_split:g}"] - float(a["m85_S"])))
        dE = max(dE, abs(r[f"m85_Eend@{t_split:g}"] - float(a["m85_Eend"])))
        dT = max(dT, abs(r[f"m85_t_min@{t_split:g}"] - float(a["m85_t_min"])))
        dcount = max(dcount, abs(r[f"n_off_S_ge_1e-04@{t_split:g}"] - int(a["n_off_S_ge_1e-04"])),
                     abs(r[f"n_in_S_ge_1e-04@{t_split:g}"] - int(a["n_in_S_ge_1e-04"])))
    ok = n > 0 and dS <= 1e-9 and dE <= 1e-9 and dT <= 1e-9
    return [f"- **C1** E6 truncated at t = {t_split:g} vs E1 ({n} matched runs): "
            f"max |dS(-85)| = {dS:.2e}, max |dE(-85, {t_split:g})| = {dE:.2e}, "
            f"max |dt_min| = {dT:.2e}, max count difference (S >= 1e-4, in/off cone) = "
            f"{dcount}: {'CONSISTENT' if ok else 'CHECK'}"]


def verdict(ok: bool, n: int = 1) -> str:
    return "n/a (no runs)" if n == 0 else ("PASS" if ok else "FAIL")


def analyze(out: Path, exp: str) -> int:
    runs = load(out, exp)
    if not runs:
        print(f"no runs under {out / exp}")
        return 1
    from arch_control.analyze import pm  # figure/table helpers (not frozen)
    rows = [run_row(r) for r in runs]
    t_split, t_full = [float(x) for x in runs[0]["horizons"]]
    _, _, checks = horizons(RunSpec(**runs[0]["spec"]))
    relu = [r for r in rows if r["arch"] == "relu"]
    lin = [r for r in rows if r["arch"] == "linear"]
    edir = out / exp; edir.mkdir(parents=True, exist_ok=True)

    with open(edir / "E6_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r["arch"], r["seed"])))

    need = 9 if len(relu) >= 10 else len(relu)
    l1 = sum(r[f"m85_class@{t_full:g}"] == "persistent"
             and r["m85_max_drop_after_split"] <= 1e-6 for r in relu)
    l2 = sum(bool(r[f"gain_lost@{t_full:g}"] >= 0.95) for r in relu)
    l3 = sum(r[f"n_in_persistent@{t_full:g}"] == 0 for r in relu)
    l4 = sum(r[f"n_any_S_ge_1e-05@{t_full:g}"] == 0 for r in lin)

    def rng(key, rs, pct=False, digits=3):
        v = np.array([r[key] for r in rs], dtype=float)
        if not np.isfinite(v).any():
            return "n/a"
        md, lo, hi = np.nanmedian(v), np.nanmin(v), np.nanmax(v)
        if pct:
            return f"{100*md:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
        return f"{md:.{digits}g} [{lo:.{digits}g}, {hi:.{digits}g}]"

    L = [f"# {exp}", "",
         f"runs analysed: {len(rows)} (ReLU {len(relu)}, linear {len(lin)}); "
         f"horizon split at t = {t_split:g}, full horizon T = {t_full:g}.", ""]
    man = edir / "manifest.jsonl"
    if man.exists():
        recs = [json.loads(x) for x in open(man)]
        commits = sorted({x["git_commit"][:10] for x in recs})
        L += [f"provenance: commits {commits}; frozen files dirty in any run: "
              f"{any(x['git_dirty'] for x in recs)}; long_horizon.py dirty in any run: "
              f"{any(x.get('long_horizon_dirty', True) for x in recs)}; "
              f"devices {sorted({str(x.get('gpu') or x['device']) for x in recs})}.", ""]
    L += ["## Pre-registered predictions", "",
          f"- **L1** -85 deg persistent on [0, {t_full:g}] and never drops by more than 1e-6 "
          f"after t = {t_split:g}: {verdict(l1 >= need, len(relu))} ({l1}/{len(relu)} ReLU seeds; "
          f"largest drop {max(r['m85_max_drop_after_split'] for r in relu):.2e})",
          f"- **L2** at least 95% of the early -85 deg gain lost by t = {t_full:g}: "
          f"{verdict(l2 >= need, len(relu))} ({l2}/{len(relu)}; median [min, max] "
          f"{rng(f'gain_lost@{t_full:g}', relu, pct=True)})",
          f"- **L3** no in-cone persistent direction: {verdict(l3 == len(relu), len(relu))} "
          f"({l3}/{len(relu)} ReLU seeds with none)",
          f"- **L4** no linear direction with S >= 1e-5: {verdict(l4 == len(lin), len(lin))} "
          f"({l4}/{len(lin)} linear seeds)"]
    L += c1_check(out, rows, t_split)
    L += ["", "## Reported without prediction", "",
          "| quantity (ReLU, -85 deg) | " + " | ".join(f"t = {tc:g}" for tc in checks) + " |",
          "|---|" + "---|" * len(checks),
          "| early gain lost, median [min, max] | " +
          " | ".join(rng(f"gain_lost@{tc:g}", relu, pct=True) for tc in checks) + " |",
          "| E(-85 deg), median [min, max] | " +
          " | ".join(rng(f"m85_E@{tc:g}", relu, digits=5) for tc in checks) + " |",
          "| training loss (ReLU), median [min, max] | " +
          " | ".join(rng(f"loss@{tc:g}", relu) for tc in checks) + " |",
          "| training loss (linear), median [min, max] | " +
          " | ".join(rng(f"loss@{tc:g}", lin) for tc in checks) + " |", "",
          f"- -85 deg minimum: E0 {rng('m85_E0', relu, digits=5)}, E_min {rng('m85_Emin', relu, digits=5)}, "
          f"t_min {rng('m85_t_min', relu)}",
          f"- ReLU off-cone directions labelled persistent at 1e-4 (mean ± 95% CI over seeds): "
          f"{pm([r[f'n_off_persistent@{t_split:g}'] for r in relu], tex=False)} at "
          f"T = {t_split:g}, {pm([r[f'n_off_persistent@{t_full:g}'] for r in relu], tex=False)} "
          f"at T = {t_full:g}",
          f"- ReLU in-cone directions labelled persistent: "
          f"{sum(r[f'n_in_persistent@{t_split:g}'] for r in relu)} in total at T = {t_split:g}, "
          f"{sum(r[f'n_in_persistent@{t_full:g}'] for r in relu)} at T = {t_full:g}; largest "
          f"in-cone final excess at T = {t_full:g}: "
          f"{max(r[f'max_in_final_excess@{t_full:g}'] for r in relu):.2e}",
          f"- ReLU -85 deg label at T = {t_split:g}: "
          f"{sorted({r[f'm85_class@{t_split:g}'] for r in relu})}; at T = {t_full:g}: "
          f"{sorted({r[f'm85_class@{t_full:g}'] for r in relu})}", ""]
    (edir / "E6_REPORT.md").write_text("\n".join(L) + "\n")
    figure(runs, edir, t_split)
    print("\n".join(L))
    print(f"wrote {edir / 'E6_REPORT.md'}, E6_summary.csv, fig_E6_long_horizon.pdf/png")
    return 0


def figure(runs: list[dict], edir: Path, t_split: float) -> None:
    from arch_control.analyze import COLOR, INK2, _band, _meta, _mpl
    plt = _mpl()
    relu = [r for r in runs if r["spec"]["arch"] == "relu"]
    lin = [r for r in runs if r["spec"]["arch"] == "linear"]
    if not relu:
        return
    im = list(relu[0]["named"]).index("m85")
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1))

    ax = axs[0]
    for r in relu:
        t = r["times"]; y = r["E_named"][:, im]; k = int(np.argmin(y))
        ax.plot(t, y, color=COLOR["relu"], lw=0.7, alpha=0.6)
        ax.plot(t[k], y[k], "o", ms=3, color=COLOR["relu"], markeredgecolor="white",
                markeredgewidth=0.6)
    ax.axvline(t_split, color=INK2, lw=0.6, ls=":")
    ax.text(t_split, 0.03, f" E1 horizon (t = {t_split:g})", transform=ax.get_xaxis_transform(),
            va="bottom", ha="left", color=INK2, fontsize=6.5)
    ax.set_xlabel("training time $t$"); ax.set_ylabel("$E(\\xi_{-85^\\circ}, t)$")
    ax.set_title(f"(a) ReLU at $-85^\\circ$, {len(relu)} seed{'s' if len(relu) != 1 else ''}", loc="left")
    ax.grid(True, axis="y")

    ax = axs[1]
    for r in relu:
        t = r["times"]; y = r["E_named"][:, im]; k = int(np.argmin(y))
        g = (y - y[k]) / (y[0] - y[k]); g[:k] = np.nan
        ax.plot(t, 100 * g, color=COLOR["relu"], lw=0.7, alpha=0.6)
    ax.axvline(t_split, color=INK2, lw=0.6, ls=":")
    ax.set_ylim(0, 102); ax.set_xlabel("training time $t$")
    ax.set_ylabel("early gain lost (%)")
    ax.set_title("(b) share of early gain lost", loc="left"); ax.grid(True, axis="y")

    ax = axs[2]
    for arch, rs in (("linear", lin), ("relu", relu)):
        if rs:
            _band(ax, rs[0]["times"], [r["loss_total"] for r in rs], arch, logy=True,
                  label="linear (clipped at $10^{-12}$)" if arch == "linear" else "ReLU")
    ax.axvline(t_split, color=INK2, lw=0.6, ls=":")
    ax.set_yscale("log"); ax.set_ylim(1e-12, 1)
    ax.set_xlabel("training time $t$"); ax.set_ylabel("training loss")
    ax.set_title("(c) training loss", loc="left"); ax.grid(True, axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(edir / f"fig_E6_long_horizon.{ext}", metadata=_meta(ext))
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check (not a protocol run)")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    specs = make_specs(args.smoke)
    if not args.analyze_only:
        rc = run(args, specs)
        if rc or args.dry_run:
            return rc
    return analyze(Path(args.out), specs[0].exp)


if __name__ == "__main__":
    sys.exit(main())
