#!/usr/bin/env python3
"""
pilot_mechanism_analysis.py

Post-hoc mechanism analysis of population_pilot.py runs. Reads a run directory
(labels.npz + summary.json) and re-evaluates the exact population field at the
saved label snapshots. Nothing here integrates new trajectories.

Subcommands (run from the repo root, next to population_pilot.py):

  ledger RUN          group decomposition of the strong family at the probe:
                      probe-active main block, probe-inactive "bridge" cohort,
                      weak family; their masses, mean angles, leakage onto the
                      cluster centres, output at the probe, and the
                      cluster-sourced input/output rotation of the main block.

  counterfactual RUN  static counterfactuals at each snapshot:
                        carrier-1pt : actual residual field, strong family collapsed
                                      to its mean angles (only the carrier changes)
                        carrier-2pt : same, probe-active and inactive parts collapsed
                                      separately
                        full-2block : strong and weak families each collapsed to a
                                      point, field recomputed
                        full-3block : main, bridge, weak each collapsed, field
                                      recomputed
                        no-leak     : full-3block with the weak block's input set to
                                      exactly 90 deg and the bridge moved onto the
                                      main block (removes the leakage onto cluster 1)

  p12 RUN             local one-sided rate of revision-2 P12 along the reference
                      path, ell(t) = sup_alpha lambda_max(A_Z(alpha)) (eq. 20.5, no
                      Gauss-Newton term), its time integral, the signed gate
                      coefficient h_gate (20.4), label-radius guard, projected
                      neuron density at both probe gate edges, and P8's L_M.

  sweep SWEEPDIR      ledger at the D1+D2 crossing for every S0_* run in a
                      population_pilot.py sweep directory.

  all RUN             ledger + counterfactual + p12.

Outputs: CSV files in RUN/analysis/ (or SWEEPDIR/analysis/), plus printed tables.
Units: physical time t; rates are physical (normalized rate x mu1^2), as in the pilot.
Angles in degrees. This is floating-point diagnosis, not a certificate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import population_pilot as pp  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz


# =============================================================================
# Loading and helpers
# =============================================================================

class Run:
    def __init__(self, run_dir, K=None, probe_deg=None):
        self.dir = Path(run_dir)
        with open(self.dir / "summary.json") as fh:
            self.summary = json.load(fh)
        p = self.summary["params"]
        self.rho, self.q, self.mu1 = p["rho"], p["q"], p["mu1"]
        self.rate = self.mu1 ** 2
        lab = np.load(self.dir / "labels.npz")
        self.t = lab["t"]
        self.states = lab["state"]
        self.L = int(lab["L"])
        self.K = int(K) if K else int(p.get("K", 4 * self.L))
        self.model = pp.Model(self.rho, self.q, self.L, self.K)
        self.probe_deg = float(probe_deg) if probe_deg is not None else float(p["focus_deg"][0])
        self.phi = math.radians(self.probe_deg)
        self.xi = np.array([math.cos(self.phi), math.sin(self.phi)])
        tag = f"{self.probe_deg:+.1f}"
        self.t_cross = self.summary.get("focus", {}).get(tag, {}).get("t_cross", float("nan"))

    def outdir(self):
        d = self.dir / "analysis"
        d.mkdir(exist_ok=True)
        return d


def circ_mean(ang, w):
    return math.atan2(np.sum(w * np.sin(ang)), np.sum(w * np.cos(ang)))


def deg(x):
    return math.degrees(x)


def groups(th, phi):
    """strong = |theta| < 45 deg; main = strong and active at the probe; bridge = strong, inactive;
    weak = |theta - 90 deg| < 45 deg."""
    tw = pp.wrap_pi(th)
    strong = np.abs(tw) < math.pi / 4
    act = np.cos(th - phi) > 0
    weak = np.abs(pp.wrap_pi(th - math.pi / 2)) < math.pi / 4
    return dict(strong=strong, main=strong & act, bridge=strong & ~act, weak=weak)


def output_of(mask, th, ps, w, x):
    z = np.maximum(np.cos(th[mask]) * x[0] + np.sin(th[mask]) * x[1], 0.0)
    return np.array([np.sum(w[mask] * z * np.cos(ps[mask])), np.sum(w[mask] * z * np.sin(ps[mask]))])


def R_at(model, ev, angles):
    """Residual matrices R_1, R_2 of the ACTUAL state's field, evaluated at arbitrary input angles."""
    g = model.grid
    h = ev["e_grid"][:, None, :, None] * g.wa.transpose(1, 0, 2)[:, :, None, :]
    R = pp.periodic_window_integral(g, h.reshape(g.K, 8), np.atleast_1d(np.asarray(angles, float)))
    R = R.reshape(-1, 2, 2, 2)
    return R[:, 0], R[:, 1]


def particle_D(xi, e, W, TH, PS, R1s, R2s):
    """(D_1, D_2) normalized, for explicit particles with given residual matrices and probe residual e."""
    a = np.stack([np.cos(TH), np.sin(TH)], 1)
    b = np.stack([np.cos(PS), np.sin(PS)], 1)
    z = a @ xi
    act = (z > 0).astype(float)
    zp = np.maximum(z, 0.0)
    out = []
    for R in (R1s, R2s):
        Ra = np.einsum("lij,lj->li", R, a)
        bRx = np.einsum("li,lij,j->l", b, R, xi)
        V = np.sum((W * zp)[:, None] * Ra + (W * act * bRx)[:, None] * b, 0)
        out.append(float(-0.5 * e @ V))
    return out


def probe_D(run, y):
    ev = run.model.evaluate(y)
    pr = run.model.probe_rates(ev, np.array([run.phi]))
    return ev, pr, run.rate * pr["D1"][0], run.rate * pr["D2"][0]


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"[wrote {path}]")


def select(run, tmin, tmax):
    return [(t, y) for t, y in zip(run.t, run.states) if tmin - 1e-9 <= t <= tmax + 1e-9]


# =============================================================================
# ledger
# =============================================================================

def ledger_row(run, t, y):
    m = run.model
    L = run.L
    th, ps, lm = y[:L], y[L:2 * L], y[2 * L:]
    w = np.exp(lm) / L
    G = groups(th, run.phi)
    ev = m.evaluate(y)
    row = dict(t=float(t))
    Mstrong = float(w[G["strong"]].sum())
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    for name in ("main", "bridge", "weak"):
        mk = G[name]
        M = float(w[mk].sum())
        row[f"{name}_mass"] = M
        if name != "weak":
            row[f"{name}_share_of_strong"] = M / Mstrong if Mstrong > 0 else float("nan")
        if M > 0:
            row[f"{name}_theta"] = deg(circ_mean(th[mk], w[mk]))
            row[f"{name}_psi"] = deg(circ_mean(ps[mk], w[mk]))
            row[f"{name}_f2_at_e1"] = output_of(mk, th, ps, w, e1)[1]     # leakage onto cluster-1 centre
            row[f"{name}_f1_at_e2"] = output_of(mk, th, ps, w, e2)[0]     # leakage onto cluster-2 centre
            fx = output_of(mk, th, ps, w, run.xi)
            row[f"{name}_f1_at_probe"], row[f"{name}_f2_at_probe"] = fx[0], fx[1]
        else:
            for k in ("theta", "psi", "f2_at_e1", "f1_at_e2", "f1_at_probe", "f2_at_probe"):
                row[f"{name}_{k}"] = float("nan")
    # probe residual and main-block channel velocities (cluster-sourced, mixture weight 1/2, physical units)
    f = np.sum((w * np.maximum(np.cos(th - run.phi), 0.0))[:, None] * ev["b"], 0)
    e = run.xi - f
    row["probe_e1"], row["probe_e2"] = e[0], e[1]
    mk = G["main"]
    if w[mk].sum() > 0:
        a, b = ev["a"][mk], ev["b"][mk]
        ap = np.stack([-a[:, 1], a[:, 0]], 1)
        bp = np.stack([-b[:, 1], b[:, 0]], 1)
        W = w[mk]
        for p, R in ((1, ev["R1"][mk]), (2, ev["R2"][mk])):
            vth = 0.5 * np.einsum("li,lij,lj->l", b, R, ap)
            vps = 0.5 * np.einsum("li,lij,lj->l", bp, R, a)
            row[f"main_thetadot_c{p}"] = run.rate * float(np.sum(W * vth) / W.sum())
            row[f"main_psidot_c{p}"] = run.rate * float(np.sum(W * vps) / W.sum())
        row["main_a_dot_probe"] = float(np.sum(W * (a @ run.xi)) / W.sum())
        row["main_b_dot_probe_residual"] = float(np.sum(W * (b @ e)) / W.sum())
    return row


def ledger(args):
    run = Run(args.run, args.K, args.probe)
    rows = [ledger_row(run, t, y) for t, y in select(run, args.tmin, args.tmax)]
    write_csv(run.outdir() / f"ledger_{run.probe_deg:+.0f}.csv", rows)
    print(f"\nLEDGER  probe {run.probe_deg:+.1f} deg   (t_cross = {run.t_cross:.3f})")
    print(f"{'t':>5} | {'main%':>5} {'th':>6} {'psi':>6} {'f2(e1)':>7} | {'brdg%':>5} {'th':>6} {'f2(e1)':>7} | "
          f"{'weak th':>7} {'f2(e1)':>7} | {'main f2(xi)':>11} {'a.xi':>6} | {'thdot c1':>9} {'thdot c2':>9} {'psidot c1':>9} {'psidot c2':>9}")
    for r in rows:
        print(f"{r['t']:5.2f} | {100 * r['main_share_of_strong']:5.1f} {r['main_theta']:+6.2f} {r['main_psi']:+6.2f} "
              f"{r['main_f2_at_e1']:+7.4f} | {100 * r['bridge_share_of_strong']:5.1f} {r['bridge_theta']:+6.2f} "
              f"{r['bridge_f2_at_e1']:+7.4f} | {r['weak_theta']:+7.2f} {r['weak_f2_at_e1']:+7.4f} | "
              f"{r['main_f2_at_probe']:+11.4f} {r.get('main_a_dot_probe', float('nan')):6.3f} | "
              f"{r.get('main_thetadot_c1', float('nan')):+9.2e} {r.get('main_thetadot_c2', float('nan')):+9.2e} "
              f"{r.get('main_psidot_c1', float('nan')):+9.2e} {r.get('main_psidot_c2', float('nan')):+9.2e}")
    return rows


# =============================================================================
# counterfactual
# =============================================================================

def counterfactual(args):
    run = Run(args.run, args.K, args.probe)
    m, L, phi, xi, rate = run.model, run.L, run.phi, run.xi, run.rate
    rows = []
    for t, y in select(run, args.tmin, args.tmax):
        th, ps, lm = y[:L], y[L:2 * L], y[2 * L:]
        w = np.exp(lm) / L
        G = groups(th, phi)
        st = G["strong"]
        if w[st].sum() < 1e-6:
            continue
        ev, pr, D1, D2 = probe_D(run, y)
        e = xi - pr["f"][0]
        rest = ~st
        c_rest = particle_D(xi, e, w[rest], th[rest], ps[rest], ev["R1"][rest], ev["R2"][rest])
        # carrier-1pt
        tb, pb = circ_mean(th[st], w[st]), circ_mean(ps[st], w[st])
        r1, r2 = R_at(m, ev, tb)
        c1 = particle_D(xi, e, np.array([w[st].sum()]), np.array([tb]), np.array([pb]), r1, r2)
        # carrier-2pt
        W, T, P, R1s, R2s = [], [], [], [], []
        for mk in (G["main"], G["bridge"]):
            if w[mk].sum() > 0:
                tt, pq = circ_mean(th[mk], w[mk]), circ_mean(ps[mk], w[mk])
                a1, a2 = R_at(m, ev, tt)
                W.append(w[mk].sum()); T.append(tt); P.append(pq); R1s.append(a1[0]); R2s.append(a2[0])
        c2 = particle_D(xi, e, np.array(W), np.array(T), np.array(P), np.array(R1s), np.array(R2s))

        def collapsed(masks):
            y2 = y.copy()
            for mk in masks:
                if mk.any():
                    y2[:L][mk] = circ_mean(th[mk], w[mk])
                    y2[L:2 * L][mk] = circ_mean(ps[mk], w[mk])
            return y2

        _, _, f2a, f2b = probe_D(run, collapsed([st, G["weak"]]))
        y3 = collapsed([G["main"], G["bridge"], G["weak"]])
        _, _, f3a, f3b = probe_D(run, y3)
        y4 = y3.copy()
        y4[:L][G["weak"]] = math.pi / 2
        if G["bridge"].any() and G["main"].any():
            y4[:L][G["bridge"]] = circ_mean(th[G["main"]], w[G["main"]])
            y4[L:2 * L][G["bridge"]] = circ_mean(ps[G["main"]], w[G["main"]])
        _, _, nla, nlb = probe_D(run, y4)
        row = dict(t=float(t), D1=D1, D2=D2,
                   carrier1_D1=rate * (c_rest[0] + c1[0]), carrier1_D2=rate * (c_rest[1] + c1[1]),
                   carrier2_D1=rate * (c_rest[0] + c2[0]), carrier2_D2=rate * (c_rest[1] + c2[1]),
                   full2_D1=f2a, full2_D2=f2b, full3_D1=f3a, full3_D2=f3b, noleak_D1=nla, noleak_D2=nlb)
        rows.append(row)
    write_csv(run.outdir() / f"counterfactual_{run.probe_deg:+.0f}.csv", rows)
    names = [("actual", "D1", "D2"), ("carrier-1pt", "carrier1_D1", "carrier1_D2"),
             ("carrier-2pt", "carrier2_D1", "carrier2_D2"), ("full-2block", "full2_D1", "full2_D2"),
             ("full-3block", "full3_D1", "full3_D2"), ("no-leak", "noleak_D1", "noleak_D2")]
    print(f"\nCOUNTERFACTUALS  probe {run.probe_deg:+.1f} deg  (each column: D1, D2, D1+D2; physical rates)")
    print(f"{'t':>5} " + " | ".join(f"{n:^29s}" for n, _, _ in names))
    for r in rows:
        print(f"{r['t']:5.2f} " + " | ".join(f"{r[a]:+9.2e} {r[b]:+9.2e} {r[a] + r[b]:+9.2e}" for _, a, b in names))
    print("\nfirst up-crossing of D1+D2 after t=1.5:")
    t = np.array([r["t"] for r in rows])
    for n, a, b in names:
        s = np.array([r[a] + r[b] for r in rows])
        print(f"   {n:12s} {pp.first_up_crossing(t, s, 1.5):.3f}")
    return rows


# =============================================================================
# p12
# =============================================================================

def p12(args):
    run = Run(args.run, args.K, args.probe)
    m, L = run.model, run.L
    g = m.grid

    def omega_mix(beta):
        return 0.5 * (pp.angular_weight(beta, g.c[0], run.q) + pp.angular_weight(beta, g.c[1], run.q))

    edges = [run.phi - math.pi / 2, run.phi + math.pi / 2]
    halfwin = math.radians(args.density_halfwidth)
    rows = []
    for t, y in select(run, 0.0, args.tmax):
        ev = m.evaluate(y)
        th, w, mm = ev["th"], ev["w"], ev["m"]
        a, b = ev["a"], ev["b"]
        n = np.stack([-a[:, 1], a[:, 0]], 1)
        Rm = 0.5 * (ev["R1"] + ev["R2"])
        hg = np.zeros(L)
        for eps in (+1, -1):
            beta = th + eps * math.pi / 2
            f = m.output_from_window(ev["srt"].window(ev["Pf"], beta), np.cos(beta), np.sin(beta))
            e = np.stack([np.cos(beta), np.sin(beta)], 1) - f
            hg += omega_mix(beta) * np.sum(b * e, 1)          # eq. (20.4), balanced states
        A = np.zeros((L, 4, 4))
        A[:, :2, :2] = hg[:, None, None] * n[:, :, None] * n[:, None, :]
        A[:, 2:, :2] = Rm
        A[:, :2, 2:] = np.transpose(Rm, (0, 2, 1))
        lam = np.linalg.eigvalsh(A)[:, -1]
        dens = []
        for ed in edges:
            dd = np.abs(pp.wrap_pi(th - ed))
            dens.append(float(np.sum(w[dd < halfwin]) / (2 * halfwin)))
        rows.append(dict(t=float(t), tau=run.rate * float(t), ell_loc=float(lam.max()),
                         hgate_max=float(hg.max()), hgate_min=float(hg.min()),
                         R_opnorm_max=float(np.linalg.norm(Rm, ord=2, axis=(1, 2)).max()),
                         min_input_radius=float(np.sqrt(mm.min())),
                         q01_input_radius=float(np.sqrt(np.quantile(mm, 0.01))),
                         density_edge_lo=dens[0], density_edge_hi=dens[1], S=float(np.sum(w))))
    write_csv(run.outdir() / "p12_rates.csv", rows)
    t = np.array([r["t"] for r in rows])
    tau = run.rate * t
    ell = np.array([r["ell_loc"] for r in rows])
    lo, hi = deg(edges[0]), deg(edges[1])
    print(f"\nP12 LOCAL ONE-SIDED RATE (normalized time units)   probe {run.probe_deg:+.1f}: gate edges {lo:+.1f}, {hi:+.1f} deg")
    print(f"{'t':>6} {'ell_loc':>8} {'hgate max':>9} {'hgate min':>9} {'max|R|':>8} {'min|U|':>8} "
          f"{'n(edge lo)':>10} {'n(edge hi)':>10} {'S':>6}")
    step = max(1, len(rows) // 16)
    for r in rows[::step]:
        print(f"{r['t']:6.2f} {r['ell_loc']:8.4f} {r['hgate_max']:9.4f} {r['hgate_min']:9.4f} {r['R_opnorm_max']:8.4f} "
              f"{r['min_input_radius']:8.4f} {r['density_edge_lo']:10.3g} {r['density_edge_hi']:10.3g} {r['S']:6.3f}")
    marks = sorted({1.0, 2.0, 3.0, 4.0, 6.0} | ({round(run.t_cross, 3)} if np.isfinite(run.t_cross) else set()))
    for T in marks:
        k = t <= T + 1e-9
        if k.sum() > 1:
            I = float(_trapz(ell[k], tau[k]))
            print(f"   int_0^{T:g} ell_loc dtau = {I:6.2f}   amplification bound e^I = {math.exp(I):.3g}")
    W = float(np.max(0.5 * (g.omega[0] + g.omega[1])))
    M2 = (1 + run.rho ** 2) / 2 + 2 * run.q ** 2
    Mmax = max(r["S"] for r in rows)
    H = M2 * (1 + Mmax)
    J = 2 * W * (1 + Mmax)
    F = H + M2 * Mmax + J
    LM = max(2 * H + 2 * (1 + Mmax) * M2, 2 * (1 + Mmax) * F)
    print(f"   P8 global rate L_M at M = max S = {Mmax:.3f}: {LM:.1f} per unit tau")
    return rows


# =============================================================================
# sweep
# =============================================================================

def sweep(args):
    base = Path(args.sweep_dir)
    rows = []
    for sub in sorted(base.glob("S0_*"), key=lambda p: float(p.name[3:])):
        run = Run(sub, args.K, args.probe)
        if not np.isfinite(run.t_cross):
            continue
        k = int(np.argmin(np.abs(run.t - run.t_cross)))
        r = ledger_row(run, run.t[k], run.states[k])
        r = dict(S0=run.summary["params"]["S0"], t_cross=run.t_cross, **r)
        rows.append(r)
    out = base / "analysis"
    out.mkdir(exist_ok=True)
    write_csv(out / "sweep_ledger.csv", rows)
    print("\nSWEEP LEDGER at the snapshot nearest the D1+D2 crossing")
    print(f"{'S0':>8} {'t_x':>6} {'t_snap':>6} | {'main%':>5} {'th':>6} {'psi':>6} | {'brdg%':>5} {'th':>6} | "
          f"{'weak th':>7} | {'leak f2(e1): bridge':>19} {'weak':>7} {'main':>7} | {'main f2(xi)':>11}")
    for r in rows:
        print(f"{r['S0']:8.0e} {r['t_cross']:6.2f} {r['t']:6.2f} | {100 * r['main_share_of_strong']:5.1f} "
              f"{r['main_theta']:+6.2f} {r['main_psi']:+6.2f} | {100 * r['bridge_share_of_strong']:5.1f} "
              f"{r['bridge_theta']:+6.2f} | {r['weak_theta']:+7.2f} | {r['bridge_f2_at_e1']:+19.4f} "
              f"{r['weak_f2_at_e1']:+7.4f} {r['main_f2_at_e1']:+7.4f} | {r['main_f2_at_probe']:+11.4f}")
    return rows


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run=True):
        if run:
            p.add_argument("run")
        p.add_argument("--probe", type=float, default=None, help="probe angle in deg (default: run's first focus probe)")
        p.add_argument("--K", type=int, default=None, help="data-direction grid (default: the run's K)")

    p = sp.add_parser("ledger"); common(p)
    p.add_argument("--tmin", type=float, default=1.0); p.add_argument("--tmax", type=float, default=6.0)
    p = sp.add_parser("counterfactual"); common(p)
    p.add_argument("--tmin", type=float, default=2.5); p.add_argument("--tmax", type=float, default=6.0)
    p = sp.add_parser("p12"); common(p)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--density-halfwidth", type=float, default=0.25, help="deg, window for the edge density")
    p = sp.add_parser("sweep"); common(p, run=False)
    p.add_argument("sweep_dir")
    p = sp.add_parser("all"); common(p)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--density-halfwidth", type=float, default=0.25)
    args = ap.parse_args(argv)

    if args.cmd == "ledger":
        ledger(args)
    elif args.cmd == "counterfactual":
        counterfactual(args)
    elif args.cmd == "p12":
        p12(args)
    elif args.cmd == "sweep":
        sweep(args)
    elif args.cmd == "all":
        tmax = args.tmax
        args.tmin, args.tmax = 1.0, 6.0
        ledger(args)
        args.tmin, args.tmax = 2.5, 6.0
        counterfactual(args)
        args.tmax = tmax
        p12(args)


if __name__ == "__main__":
    main()