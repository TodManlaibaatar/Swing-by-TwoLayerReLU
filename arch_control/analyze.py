"""Compute pre-specified statistics, tables, figures and REPORT.md.

    python -m arch_control.analyze                 # everything present under results/
    python -m arch_control.analyze --out /tmp/ac_smoke --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arch_control import metrics as M  # noqa: E402

# Categorical slots 1-3 of the validated reference palette (all-pairs safe);
# every series is also direct-labelled and/or line-styled (no colour-only identity).
COLOR = {"linear": "#2a78d6", "relu": "#eb6834", "tied": "#1baf7a"}
STYLE = {"linear": "-", "relu": "--", "tied": "-"}
LABEL = {"linear": "linear", "relu": "ReLU", "tied": "tied $UU^\\top$ (predecessor)"}
INK = "#0b0b0b"; INK2 = "#52514e"; GRIDC = "#d9d8d4"; CONE = "#efeee9"

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
       19: 2.093, 29: 2.045, 39: 2.023}


def t95(df):
    if df <= 0:
        return float("nan")
    keys = sorted(T95)
    for k in keys:
        if df <= k:
            return T95[k]
    return 1.96


def mean_ci(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(x.mean())
    hw = float(t95(n - 1) * x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return m, hw, n


# ----------------------------------------------------------------------------- loading

def load_run(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as z:
        r = {k: z[k] for k in z.files if k not in ("U_hist", "W_hist")}
    r["spec"] = json.loads(str(r.pop("spec_json")))
    r["named"] = [str(s) for s in r["named"]]
    return r


def run_metrics(r: dict, recompute: bool = False) -> tuple[dict, dict]:
    t = r["times"]; En = r["E_named"]
    deg = r["grid_deg"]; cone = M.in_closed_positive_cone(deg)
    if recompute and "E_grid" in r:
        Eg = r["E_grid"].astype(float)
        pg = M.persistent_increase(t, Eg); rg = M.raw_rise(Eg); eg = M.endpoint_stats(t, Eg)
        asym = M.antipodal_asymmetry(Eg)
    else:   # statistics stored by the runner (identical code path)
        pg = dict(swing=r["grid_S"], t_reversal=r["grid_t_rev"]); rg = r["grid_rise"]
        eg = dict(final_excess=r["grid_final_excess"], t_min=r["grid_t_min"])
        asym = float(r["antipodal_asym"])
    pn = M.persistent_increase(t, En); rn = M.raw_rise(En); en = M.endpoint_stats(t, En)
    row = dict(r["spec"])
    row["run_id"] = r.get("run_id", "")
    row["t_end_logged"] = float(t[-1])
    row["loss_final"] = float(r["loss_total"][-1])
    row["loss_max_increase"] = float(np.max(np.diff(r["loss_total"])))
    row["antipodal_asym"] = asym
    for tau in M.THRESHOLDS:
        s = f"{tau:.0e}"
        row[f"n_in_S_ge_{s}"] = int(((pg["swing"] >= tau) & cone).sum())
        row[f"n_off_S_ge_{s}"] = int(((pg["swing"] >= tau) & ~cone).sum())
        row[f"n_in_rise_ge_{s}"] = int(((rg >= tau) & cone).sum())
        row[f"n_off_rise_ge_{s}"] = int(((rg >= tau) & ~cone).sum())
    row["n_in"] = int(cone.sum()); row["n_off"] = int((~cone).sum())
    for j, name in enumerate(r["named"]):
        row[f"{name}_S"] = float(pn["swing"][j]); row[f"{name}_rise"] = float(rn[j])
        row[f"{name}_t_rev"] = float(pn["t_reversal"][j])
        row[f"{name}_t_min"] = float(en["t_min"][j]); row[f"{name}_E0"] = float(en["E0"][j])
        row[f"{name}_Emin"] = float(en["E_min"][j]); row[f"{name}_Eend"] = float(en["E_end"][j])
        row[f"{name}_final_excess"] = float(en["final_excess"][j])
        row[f"{name}_class_1e-04"] = str(M.classify(rn[j], pn["swing"][j],
                                                    en["final_excess"][j], 1e-4))
    per_dir = dict(S=pg["swing"], rise=rg, final_excess=eg["final_excess"], t_min=eg["t_min"])
    return row, per_dir


def analyze_runs(exp_dir: Path):
    rows, dirs, curves = [], {}, {}
    for p in sorted((exp_dir / "runs").glob("*.npz")):
        r = load_run(p)
        r["run_id"] = p.stem
        row, d = run_metrics(r)
        rows.append(row); dirs[p.stem] = d
        curves[p.stem] = dict(times=r["times"], E_named=r["E_named"], named=r["named"],
                              loss=r["loss_total"], grid_deg=r["grid_deg"])
    return rows, dirs, curves


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def fmt(x, digits=3, tex=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        m, e = f"{x:.{digits-1}e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}" if tex else f"{m}e{int(e)}"
    return f"{x:.{digits}g}"


def pm(vals, digits=3, tex=True):
    m, hw, n = mean_ci(vals)
    if n == 0:
        return "--"
    d = "$" if tex else ""
    if n == 1 or not math.isfinite(hw):
        return f"{d}{fmt(m, digits, tex)}{d}"
    return f"{d}{fmt(m, digits, tex)}\\pm{fmt(hw, 2, tex)}{d}" if tex else \
        f"{fmt(m, digits, tex)} ± {fmt(hw, 2, tex)}"


def write_tex(path: Path, header: list[str], body: list[list[str]], caption: str, label: str):
    cols = "l" * len(header)
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in body]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- plotting

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "grid.color": GRIDC, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })
    return plt


def _meta(ext):
    return {"CreationDate": None} if ext == "pdf" else {}


def _band(ax, t, Y, arch, label=None, logy=False, ls=None):
    Y = np.asarray(Y)
    med = np.median(Y, axis=0); lo = Y.min(axis=0); hi = Y.max(axis=0)
    if logy:
        floor = 1e-12; med = np.maximum(med, floor); lo = np.maximum(lo, floor); hi = np.maximum(hi, floor)
    ax.fill_between(t, lo, hi, color=COLOR[arch], alpha=0.18, linewidth=0)
    ax.plot(t, med, color=COLOR[arch], ls=ls or STYLE[arch], label=label or LABEL[arch])


def fig_E1(curves, dirs, rows, figdir: Path):
    plt = _mpl()
    by = defaultdict(list)
    for r in rows:
        by[r["arch"]].append(r["run_id"])
    if not by:
        return
    any_id = rows[0]["run_id"]; named = curves[any_id]["named"]
    ic = named.index("comp_emp"); im = named.index("m85")
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.1))
    ax = axs[0, 0]
    for arch in ("linear", "relu"):
        ids = by.get(arch, [])
        if ids:
            t = curves[ids[0]]["times"]
            _band(ax, t, [curves[i]["E_named"][:, ic] for i in ids], arch, logy=True)
    ax.set_yscale("log"); ax.set_xlabel("training time $t$"); ax.set_ylabel("$E(\\hat x/\\|\\hat x\\|, t)$")
    ax.set_title("(a) compositional probe", loc="left")
    ax.legend(loc="upper right"); ax.grid(True, axis="y")
    ax = axs[0, 1]
    for arch in ("linear", "relu"):
        ids = by.get(arch, [])
        if ids:
            t = curves[ids[0]]["times"]
            _band(ax, t, [curves[i]["E_named"][:, im] for i in ids], arch)
    ax.set_ylim(0, 0.52); ax.set_xlabel("training time $t$"); ax.set_ylabel("$E(\\xi_{-85^\\circ}, t)$")
    ax.set_title("(b) off-cone probe $-85^\\circ$", loc="left"); ax.grid(True, axis="y")
    for arch in ("linear", "relu"):
        ids = by.get(arch, [])
        if ids:
            c = curves[ids[0]]; k = int(0.62 * (len(c["times"]) - 1))
            yk = float(np.median([curves[i]["E_named"][k, im] for i in ids]))
            ax.annotate(LABEL[arch], (c["times"][k], yk), xytext=(0, -11 if arch == "relu" else 4),
                        textcoords="offset points", ha="center", color=INK2, fontsize=7)
    ax = axs[1, 0]
    ids = by.get("relu", [])
    for i in ids:
        c = curves[i]; y = c["E_named"][:, im]
        ax.plot(c["times"], y, color=COLOR["relu"], lw=0.7, alpha=0.6)
        k = int(np.argmin(y)); ax.plot(c["times"][k], y[k], "o", ms=3, color=COLOR["relu"],
                                        markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlabel("training time $t$"); ax.set_ylabel("$E(\\xi_{-85^\\circ}, t)$")
    ax.set_title(f"(c) ReLU at $-85^\\circ$, {len(ids)} seed(s)", loc="left")
    ax.grid(True, axis="y")
    ax = axs[1, 1]
    deg = curves[any_id]["grid_deg"]
    ax.axvspan(0, 90, color=CONE, zorder=0)
    ax.text(45, 0.95, "positive\ncone", transform=ax.get_xaxis_transform(), ha="center",
            va="top", color=INK2, fontsize=6.5)
    for arch in ("linear", "relu"):
        ids = by.get(arch, [])
        if ids:
            S = np.array([dirs[i]["S"] for i in ids])
            _band(ax, deg, np.maximum(S, 1e-8), arch, logy=True, ls="-")
    ax.axhline(1e-4, color=INK2, lw=0.6, ls=":")
    ax.text(-178, 1.25e-4, "$10^{-4}$ threshold", color=INK2, fontsize=6.5)
    ax.set_yscale("log"); ax.set_ylim(8e-9, 5e-2); ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlabel("probe angle $\\psi$ (deg)")
    ax.set_ylabel("persistent increase $S(\\psi)$")
    ax.set_title("(d) directional sweep ($10^{-8}$ = no increase)", loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"fig_E1_matched.{ext}", metadata=_meta(ext))
    plt.close(fig)


def fig_E3(rows, figdir: Path):
    plt = _mpl()
    datas = sorted({(r["mu1"], r["mu2"], r["sigma"]) for r in rows}, reverse=True)
    inits = ["iso", "pos"]
    widths = sorted({r["width"] for r in rows})
    s0s = sorted({round(r["width"] * r["epsilon"] ** 2, 10) for r in rows})
    fig, axs = plt.subplots(len(datas), 2, figsize=(5.5, 1.9 * len(datas) + 0.3), squeeze=False,
                            sharex=True, sharey=True)
    floor = 1e-10
    ls_s0 = {s0s[0]: ":", s0s[-1]: "-"} if s0s else {}
    mk_s0 = {s0s[0]: "o", s0s[-1]: "s"} if s0s else {}
    for a, data in enumerate(datas):
        for b, init in enumerate(inits):
            ax = axs[a, b]
            for arch in ("linear", "relu"):
                for s0 in s0s:
                    xs, ms, lo, hi = [], [], [], []
                    for h in widths:
                        v = [max(r["comp_pop_rise"], floor) for r in rows
                             if (r["mu1"], r["mu2"], r["sigma"]) == data and r["init"] == init
                             and r["arch"] == arch and r["width"] == h
                             and round(r["width"] * r["epsilon"] ** 2, 10) == s0]
                        if v:
                            xs.append(h); ms.append(np.median(v)); lo.append(min(v)); hi.append(max(v))
                    if xs:
                        off = 1.06 if arch == "relu" else 1 / 1.06
                        xx = np.array(xs) * off
                        ax.errorbar(xx, ms, yerr=[np.array(ms) - lo, np.array(hi) - ms],
                                    color=COLOR[arch], ls=ls_s0[s0], marker=mk_s0[s0], ms=3.5,
                                    lw=1.1, elinewidth=0.6, capsize=0,
                                    label=f"{LABEL[arch]}, $s_0={s0:g}$")
            ax.axhline(1e-4, color=INK2, lw=0.6, ls=":")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(floor / 2, 1)
            ax.set_xticks(widths); ax.set_xticklabels([str(w) for w in widths])
            mu1, mu2, sd = data
            ax.set_title(f"$\\mu=({mu1:g},{mu2:g})$, $\\sigma={sd:g}$, {init} init", loc="left")
            ax.grid(True, axis="y")
            if b == 0:
                ax.set_ylabel("compositional rise")
            if a == len(datas) - 1:
                ax.set_xlabel("width $h$")
    axs[0, 1].legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"fig_E3_regime.{ext}", metadata=_meta(ext))
    plt.close(fig)

    # paired comparison: same data, same initialization, activation changed
    key = lambda r: (r["mu1"], r["mu2"], r["sigma"], r["width"], r["epsilon"], r["init"], r["seed"])
    lin = {key(r): r for r in rows if r["arch"] == "linear"}
    pairs = [(lin[key(r)]["comp_pop_rise"], r["comp_pop_rise"], r) for r in rows
             if r["arch"] == "relu" and key(r) in lin]
    if pairs:
        fig, ax = plt.subplots(figsize=(2.9, 2.7))
        markers = {w: m for w, m in zip(widths, "o^sD")}
        for init, col in (("iso", COLOR["linear"]), ("pos", COLOR["relu"])):
            for w in widths:
                P = [(max(x, floor), max(y, floor)) for x, y, r in pairs
                     if r["init"] == init and r["width"] == w]
                if P:
                    P = np.array(P)
                    ax.plot(P[:, 0], P[:, 1], markers[w], ms=3.2, color=col, alpha=0.75,
                            markeredgecolor="white", markeredgewidth=0.4,
                            label=f"{init}, $h={w}$")
        ax.plot([floor, 1], [floor, 1], color=INK2, lw=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("linear compositional rise"); ax.set_ylabel("ReLU compositional rise")
        ax.set_title("same data & init, activation changed", loc="left")
        ax.legend(fontsize=5.5, ncol=2, loc="upper left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(figdir / f"fig_E3_paired.{ext}", metadata=_meta(ext))
        plt.close(fig)


def fig_E4(curves, rows, figdir: Path):
    plt = _mpl()
    conds = sorted({(r["width"], r["epsilon"]) for r in rows})
    if not conds:
        return
    fig, axs = plt.subplots(1, len(conds), figsize=(5.5, 1.7), sharey=True, squeeze=False)
    for ax, (w, e) in zip(axs[0], conds):
        tmax = 0.0
        for r in rows:
            if (r["width"], r["epsilon"]) != (w, e):
                continue
            c = curves[r["run_id"]]; j = c["named"].index("comp_pop"); y = c["E_named"][:, j]
            ax.plot(c["times"], y, color=COLOR["tied"], lw=0.7, alpha=0.7)
            k = int(np.argmax(y < 1e-3 * y[0])) if np.any(y < 1e-3 * y[0]) else len(y) - 1
            tmax = max(tmax, float(c["times"][k]))
        ax.set_xlim(0, 1.15 * tmax if tmax > 0 else None)
        ax.set_title(f"$d'={w}$, $\\omega={e:g}$", loc="left"); ax.set_xlabel("$t$")
        ax.grid(True, axis="y")
    axs[0, 0].set_ylabel("compositional $E$")
    fig.suptitle("positive control: predecessor's tied model $f=U^\\top Ux$, all-positive init "
                 "(each line a seed)", x=0.01, ha="left", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"fig_E4_positive_control.{ext}", metadata=_meta(ext))
    plt.close(fig)


def fig_E2(rows, figdir: Path):
    plt = _mpl()
    cells = sorted({(r["mu2"], r["sigma"]) for r in rows})
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(5.5, 1.9))
    for arch, dx in (("linear", -0.12), ("relu", 0.12)):
        for k, cell in enumerate(cells):
            v = [r["n_off_S_ge_1e-04"] for r in rows if r["arch"] == arch and (r["mu2"], r["sigma"]) == cell]
            ax.plot(np.full(len(v), k + dx), v, "o", ms=3.2, color=COLOR[arch], alpha=0.8,
                    markeredgecolor="white", markeredgewidth=0.4,
                    label=LABEL[arch] if k == 0 else None)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"$\\mu_2={a:g}$\n$\\sigma={b:g}$" for a, b in cells], fontsize=6)
    ax.set_ylabel("off-cone directions\nwith $S\\geq10^{-4}$"); ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"fig_E2_grid.{ext}", metadata=_meta(ext))
    plt.close(fig)


# ----------------------------------------------------------------------------- reports

def _md_table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return out + [""]


def _sci(x, tex=True):
    return f"${x:.1e}$" if tex else f"{x:.1e}"


def summarize(exp: str, rows: list[dict], tabdir: Path) -> list[str]:
    """Experiment-specific tables; returns REPORT.md lines."""
    lines = [f"## {exp}", "", f"runs analysed: {len(rows)}", ""]
    archs = [a for a in ("linear", "relu", "tied") if any(r["arch"] == a for r in rows)]
    plain = {"linear": "linear", "relu": "ReLU", "tied": "tied UU^T"}
    if exp in ("E1_canonical", "E2_grid", "SMOKE"):
        def body(tex):
            out = []
            for a in archs:
                R = [r for r in rows if r["arch"] == a]
                out.append([LABEL[a] if tex else plain[a], str(len(R)),
                            _sci(max(r["comp_pop_rise"] for r in R), tex),
                            _sci(max(r["comp_emp_rise"] for r in R), tex),
                            pm([r["n_in_S_ge_1e-04"] for r in R], tex=tex),
                            pm([r["n_off_S_ge_1e-04"] for r in R], tex=tex),
                            pm([r["n_in_S_ge_1e-05"] for r in R], tex=tex),
                            pm([r["n_off_S_ge_1e-05"] for r in R], tex=tex),
                            pm([r["m85_S"] for r in R], tex=tex),
                            pm([r["m85_t_min"] for r in R], tex=tex),
                            pm([r["m85_final_excess"] for r in R], tex=tex),
                            _sci(max(r["antipodal_asym"] for r in R), tex)])
            return out
        hdr_tex = ["arch", "runs", "comp rise (max)", "comp$_{\\rm emp}$ rise (max)",
                   "in $S{\\ge}10^{-4}$", "off $S{\\ge}10^{-4}$", "in $S{\\ge}10^{-5}$",
                   "off $S{\\ge}10^{-5}$", "$S(-85^\\circ)$", "$t_{\\min}(-85^\\circ)$",
                   "final excess $(-85^\\circ)$", "$\\max|E(\\psi)-E(\\psi{+}\\pi)|$"]
        hdr_md = ["arch", "runs", "comp rise (max)", "comp_emp rise (max)", "in S>=1e-4",
                  "off S>=1e-4", "in S>=1e-5", "off S>=1e-5", "S(-85)", "t_min(-85)",
                  "final excess(-85)", "max antipodal asym"]
        write_tex(tabdir / f"table_{exp}.tex", hdr_tex, body(True),
                  f"{exp.replace('_', ' ')}: matched architectures on identical data and "
                  "initialization (mean $\\pm$ 95\\% CI over seeds; counts are directions on "
                  "the 0.5$^\\circ$ grid; in/off = inside/outside the closed positive cone).",
                  f"tab:{exp}")
        lines += _md_table(hdr_md, body(False))
    if exp == "E2_grid":
        cells = sorted({(r["mu2"], r["sigma"]) for r in rows})
        def body2(tex):
            out = []
            for a in archs:
                for cell in cells:
                    R = [r for r in rows if r["arch"] == a and (r["mu2"], r["sigma"]) == cell]
                    out.append([LABEL[a] if tex else plain[a], f"{cell[0]:g}", f"{cell[1]:g}",
                                str(len(R)), _sci(max(r["comp_pop_rise"] for r in R), tex),
                                pm([r["n_in_S_ge_1e-04"] for r in R], tex=tex),
                                pm([r["n_off_S_ge_1e-04"] for r in R], tex=tex),
                                pm([r["m85_t_min"] for r in R], tex=tex)])
            return out
        write_tex(tabdir / "table_E2_cells.tex",
                  ["arch", "$\\mu_2$", "$\\sigma$", "seeds", "comp rise (max)",
                   "in $S{\\ge}10^{-4}$", "off $S{\\ge}10^{-4}$", "$t_{\\min}(-85^\\circ)$"],
                  body2(True), "Nine-cell grid, both architectures.", "tab:E2_cells")
        lines += _md_table(["arch", "mu2", "sigma", "seeds", "comp rise (max)", "in S>=1e-4",
                            "off S>=1e-4", "t_min(-85)"], body2(False))
    if exp == "E3_regime":
        groups = defaultdict(list)
        for r in rows:
            s0 = round(r["width"] * r["epsilon"] ** 2, 10)
            groups[(r["mu1"], r["mu2"], r["sigma"], r["init"], s0, r["width"], r["arch"])].append(r)
        def body3(tex):
            out = []
            for k in sorted(groups):
                R = groups[k]; mu1, mu2, sd, init, s0, h, a = k
                ntrans = sum(r["comp_pop_class_1e-04"] == "transient" for r in R)
                out.append([f"({mu1:g},{mu2:g}), {sd:g}", init, f"{s0:g}", str(h),
                            LABEL[a] if tex else plain[a],
                            pm([r["comp_pop_rise"] for r in R], tex=tex), f"{ntrans}/{len(R)}",
                            pm([r["n_off_S_ge_1e-04"] for r in R], tex=tex)])
            return out
        write_tex(tabdir / "table_E3_regime.tex",
                  ["data", "init", "$s_0$", "$h$", "arch", "comp rise", "transient",
                   "off $S{\\ge}10^{-4}$"], body3(True),
                  "Width / initialization sweep. ``transient'' = compositional rise "
                  "$\\ge10^{-4}$ that recovers (swing-by-like).", "tab:E3_regime")
        lines += _md_table(["data", "init", "s0", "h", "arch", "comp rise", "transient",
                            "off S>=1e-4"], body3(False))
        key = lambda r: (r["mu1"], r["mu2"], r["sigma"], r["width"], r["epsilon"], r["init"], r["seed"])
        lin = {key(r): r for r in rows if r["arch"] == "linear"}
        pairs = [(lin[key(r)]["comp_pop_rise"], r["comp_pop_rise"]) for r in rows
                 if r["arch"] == "relu" and key(r) in lin]
        big = [(x, y) for x, y in pairs if x >= 1e-4]
        if big:
            ratio = np.array([y / x for x, y in big])
            lines += [f"Paired compositional rise (ReLU / linear, same data and init), pairs with "
                      f"linear rise >= 1e-4: n={len(big)}, median ratio {np.median(ratio):.3f} "
                      f"(IQR {np.percentile(ratio, 25):.3f}-{np.percentile(ratio, 75):.3f}).", ""]
        if len(pairs) > 2:
            lx = np.log10(np.maximum([p[0] for p in pairs], 1e-10))
            ly = np.log10(np.maximum([p[1] for p in pairs], 1e-10))
            rk = lambda v: np.argsort(np.argsort(v))
            rho = np.corrcoef(rk(lx), rk(ly))[0, 1]
            lines += [f"Spearman rank correlation of paired compositional rises: {rho:.3f} "
                      f"(n={len(pairs)}).", ""]
    if exp == "E4_positive_control":
        conds = sorted({(r["width"], r["epsilon"]) for r in rows})
        def body4(tex):
            out = []
            for cond in conds:
                R = [r for r in rows if (r["width"], r["epsilon"]) == cond]
                ntr = sum(r["comp_pop_class_1e-04"] == "transient" for r in R)
                out.append([str(cond[0]), f"{cond[1]:g}", str(len(R)),
                            pm([r["comp_pop_rise"] for r in R], tex=tex),
                            pm([r["comp_pop_final_excess"] for r in R], tex=tex), f"{ntr}/{len(R)}"])
            return out
        write_tex(tabdir / "table_E4_positive_control.tex",
                  ["$d'$", "$\\omega$", "seeds", "comp rise", "final excess", "transient"], body4(True),
                  "Positive control: the predecessor's tied model $f=U^\\top Ux$ with "
                  "all-positive initialization.", "tab:E4")
        lines += _md_table(["d'", "omega", "seeds", "comp rise", "final excess", "transient"],
                           body4(False))
    return lines


def _expected(exp):
    from arch_control.experiments import experiment_specs
    try:
        return len(experiment_specs(exp))
    except KeyError:
        return 0


def _partial(rows_by_exp, exps):
    got = sum(len(rows_by_exp.get(e, [])) for e in exps)
    exp = sum(_expected(e) for e in exps)
    return "" if got >= exp else f" [PARTIAL: {got}/{exp} protocol runs analysed]"


def hypotheses(all_rows: dict, dirs_all: dict, curves_all: dict) -> list[str]:
    L = ["## Pre-specified hypotheses (PROTOCOL.md Section 4)", ""]
    wide = [r for e in ("E1_canonical", "E2_grid") for r in all_rows.get(e, [])]
    part = _partial(all_rows, ("E1_canonical", "E2_grid"))
    if wide:
        bad = [r for r in wide if r["comp_pop_rise"] > M.MONOTONE_TOL or r["comp_emp_rise"] > M.MONOTONE_TOL]
        L.append(f"- **H1** compositional probe monotone (raw rise <= {M.MONOTONE_TOL:g}) in every "
                 f"wide-regime run: {'PASS' if not bad else 'FAIL'} "
                 f"({len(wide)-len(bad)}/{len(wide)} runs; violations by arch: "
                 f"{dict((a, sum(r['arch']==a for r in bad)) for a in ('linear','relu'))})" + part)
        lin = [r for r in wide if r["arch"] == "linear"]; rel = [r for r in wide if r["arch"] == "relu"]
        a_bad = [r for r in lin if r["n_in_S_ge_1e-05"] + r["n_off_S_ge_1e-05"] > 0]
        b_bad = [r for r in rel if r["n_off_S_ge_1e-04"] == 0]
        L.append(f"- **H2a** linear runs have no direction with S >= 1e-5: "
                 f"{'PASS' if not a_bad else 'FAIL'} ({len(lin)-len(a_bad)}/{len(lin)})" + part)
        L.append(f"- **H2b** every ReLU run has an off-cone direction with S >= 1e-4: "
                 f"{'PASS' if not b_bad else 'FAIL'} ({len(rel)-len(b_bad)}/{len(rel)})" + part)
        L.append(f"- (reported, no prediction) ReLU in-cone directions with S >= 1e-4 per run: "
                 f"{[r['n_in_S_ge_1e-04'] for r in rel]}")
    e1 = [r for r in all_rows.get("E1_canonical", []) if r["arch"] == "relu"]
    if e1:
        ok = [r for r in e1 if r["m85_class_1e-04"] == "persistent"]
        verdict = ("INCOMPLETE" if len(e1) < 10 or max(r["t_end"] for r in e1) < 20
                   else ("PASS" if len(ok) >= 9 else "FAIL"))
        L.append(f"- **H3** ReLU -85 deg reversal persistent through t=20 in >= 9/10 seeds: "
                 f"{verdict} ({len(ok)}/{len(e1)} persistent)")
    e4 = all_rows.get("E4_positive_control", [])
    if e4:
        conds = sorted({(r["width"], r["epsilon"]) for r in e4})
        res = []
        for c in conds:
            R = [r for r in e4 if (r["width"], r["epsilon"]) == c]
            n = sum(r["comp_pop_class_1e-04"] == "transient" for r in R)
            res.append((c, n, len(R)))
        ok = all(n >= 0.9 * m for _, n, m in res)
        complete = len(res) == 4 and all(m >= 10 for _, _, m in res)
        L.append(f"- **P1** positive control detects swing-by (transient at 1e-4) in >= 90% of seeds "
                 f"per condition: {('PASS' if ok else 'FAIL') if complete else 'INCOMPLETE'} {res}")
    e5 = all_rows.get("E5_stepsize", [])
    if e5:
        for a in ("relu", "linear"):
            R = {r["dt"]: r for r in e5 if r["arch"] == a}
            if len(R) == 2:
                d1, d2 = sorted(R)
                i1 = R[d1]["run_id"]; i2 = R[d2]["run_id"]
                dS = float(np.abs(dirs_all["E5_stepsize"][i1]["S"] - dirs_all["E5_stepsize"][i2]["S"]).max())
                same = R[d1]["n_off_S_ge_1e-04"] == R[d2]["n_off_S_ge_1e-04"] and \
                    R[d1]["n_in_S_ge_1e-04"] == R[d2]["n_in_S_ge_1e-04"]
                L.append(f"- **V5** step size ({a}): max |dS| between dt={d1:g} and dt={d2:g} = {dS:.2e}; "
                         f"threshold counts identical: {same}")
    allr = [r for rs in all_rows.values() for r in rs]
    if allr:
        worst = max(r["loss_max_increase"] for r in allr)
        L.append(f"- **V3** training loss non-increasing at logged times in all runs: "
                 f"{'PASS' if worst <= 1e-12 else 'CHECK'} (max logged increase {worst:.2e})")
        lin = [r for r in allr if r["arch"] == "linear"]
        if lin:
            L.append(f"- **V2** linear antipodal symmetry max|E(psi)-E(psi+180)| = "
                     f"{max(r['antipodal_asym'] for r in lin):.2e}")
    L.append("")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    figdir = out / "figures"; tabdir = out / "tables"
    figdir.mkdir(parents=True, exist_ok=True); tabdir.mkdir(parents=True, exist_ok=True)
    exps = ["SMOKE"] if args.smoke else [d.name for d in sorted(out.iterdir())
                                         if (d / "runs").is_dir() and d.name != "SMOKE"]
    all_rows, all_dirs, all_curves = {}, {}, {}
    report = ["# arch_control report", ""]
    for exp in exps:
        rows, dirs, curves = analyze_runs(out / exp)
        if not rows:
            continue
        all_rows[exp], all_dirs[exp], all_curves[exp] = rows, dirs, curves
        write_csv(out / exp / "run_summary.csv", rows)
        np.savez_compressed(out / exp / "direction_stats.npz",
                            **{f"{k}__{m}": v[m] for k, v in dirs.items() for m in v})
        report += summarize(exp, rows, tabdir)
        if exp in ("E1_canonical", "SMOKE"):
            fig_E1(curves, dirs, [r for r in rows if r["arch"] in ("linear", "relu")], figdir)
        if exp == "E2_grid":
            fig_E2(rows, figdir)
        if exp == "E3_regime":
            fig_E3(rows, figdir)
        if exp == "E4_positive_control" or (exp == "SMOKE" and any(r["arch"] == "tied" for r in rows)):
            fig_E4(curves, [r for r in rows if r["arch"] == "tied"], figdir)
    report += hypotheses(all_rows, all_dirs, all_curves)
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"tables -> {tabdir}\nfigures -> {figdir}")


if __name__ == "__main__":
    main()
