#!/usr/bin/env python3
"""
population_pilot.py

Continuum (infinite-width, population-data) witness search for the planar
two-concept SIM with an untied, bias-free two-layer ReLU network trained by
gradient flow from aligned-balanced isotropic initialization.

It integrates the exact labeled-characteristic system of the foundations draft
and reports, along the trajectory,

    D_1(t, xi), D_2(t, xi)      exact cluster-sourced probe rates  (P6, 8.3-8.5)
    m_1, m_2, L_1, L_2          partition-free learning observables (P7, 9.1-9.2)
    strong/weak family angular statistics, probe-gate strip mass   (P9, 11.1)
    source x carrier split of D_p at focus probes

together with the archived per-direction statistics (S, rise, t_min,
final_excess) so the continuum can be compared with the h=200 seed ensemble.

THIS IS A FLOATING-POINT WITNESS SEARCH, NOT A CERTIFICATE.

-------------------------------------------------------------------------------
Model and units
-------------------------------------------------------------------------------
Physical data  P_1 = N(mu1 e1, sigma^2 I),  P_2 = N(mu2 e2, sigma^2 I), equal weights.
Network        f(x) = sum_i w_i (u_i^T x)_+ , loss (1/2) E||f(X) - X||^2.
Init           u_i = w_i = sqrt(S0/h) a_{alpha_i}, alpha_i ~ Unif; canonical
               h = 200, eps = 1e-3  =>  S0 = h eps^2 = 2e-4.

Normalized units: mu1 = 1, rho = mu2/mu1, q = sigma/mu1, tau = mu1^2 t_phys.
Parameters are identical in both unit systems, so the error at a UNIT probe is
identical, and physical rates are mu1^2 times normalized rates.
All times written to disk are PHYSICAL (column `t`), with `tau` alongside.
All D values written to disk are PHYSICAL rates (d/dt_phys), directly
comparable with the finite-network appendix numbers.

Continuum state: labels alpha_l = 2 pi (l + 1/2)/L carrying
(theta_l, psi_l, log m_l), quadrature weight 1/L:
    f(x) = (1/L) sum_l m_l b_{psi_l} (a_{theta_l}^T x)_+ .
Characteristics (draft 4.4):
    theta' = b^T R a_perp,   psi' = b_perp^T R a,   (log m)' = 2 b^T R a,
    R(theta) = (R_1 + R_2)/2,  R_p(theta) = E_p[e(X) X^T 1{a_theta^T X > 0}],
    e(x) = x - f(x).
R_p is evaluated through the exact angular reduction (draft 6.2, 6.11):
    R_p(theta) = int_{theta-pi/2}^{theta+pi/2} omega_p(beta) e(a_beta) a_beta^T dbeta
with omega_p in closed form (6.3-6.4), on a uniform beta grid of K points
(cumulative trapezoid, exact partial-cell interpolation). Window sums over
labels use sorted prefix sums, so the network output at any direction is exact
up to label quadrature. Cost per field evaluation is O((K + L) log L).

-------------------------------------------------------------------------------
Usage (run from the repo root so `archive` finds arch_control_summary/)
-------------------------------------------------------------------------------
  # 0. smoke test with all audits (~2 min)
  python population_pilot.py run --out pilot_smoke --L 4096 --K 16384 --tmax 6 --audit

  # 1. canonical run (rho=2/3, q=0.05, S0=2e-4), horizon 20 like E1_canonical
  python population_pilot.py run --out pilot_canonical --audit

  # 2. resolution check: double L and K, halve dt, then compare
  python population_pilot.py run --out pilot_canonical_2x --L 32768 --K 131072 --dt 0.0225
  python population_pilot.py compare pilot_canonical pilot_canonical_2x

  # 3. compare with the h=200 ReLU seed ensemble in the repo archive
  python population_pilot.py archive pilot_canonical

  # 4. initialization-scale sweep (eps = sqrt(S0/200) spans 3 decades);
  #    horizons are extended automatically for small S0
  python population_pilot.py sweep --out pilot_S0_sweep --L 8192 --K 32768 --tmax 10 \
      --S0-list 2e-8,2e-7,2e-6,2e-5,2e-4,2e-3

Outputs per run directory:
  summary.json        parameters, audits, clocks, focus-probe ledger, sector counts
  timeseries.csv      per save: m_p, L_p, mass identities, family statistics,
                      E/D1/D2 and the source x carrier x motion split at focus probes
  probes.npz          E, D1, D2 on the archived 0.5-deg grid + fine sector (T x P)
  direction_stats.npz S, t_rev, rise, t_min, final_excess per probe (archive definitions)
  sector.csv          per sector probe: S, t_min, D1+D2 up-crossing, D1/D2 at crossing,
                      share of the rate swing due to D1 decay
  labels.npz          label states (theta, psi, log m) every --snapshot-every saves
  *.png               focus-probe rates, strong-family angles, sector S
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    from scipy.special import ndtr as _ndtr

    def Phi(x):
        return _ndtr(x)
except Exception:  # pragma: no cover - scipy fallback
    _erfc = np.vectorize(math.erfc)

    def Phi(x):
        return 0.5 * _erfc(-np.asarray(x) / math.sqrt(2.0))


TWO_PI = 2.0 * math.pi
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x fallback
HALF_PI = 0.5 * math.pi

# Archived conventions (theory_guided_swing_by_phase_diagram.py, arch_control/metrics.py)
ARCHIVE_GRID_DEG = np.linspace(-180.0, 180.0, 720, endpoint=False)
SMOOTH_WINDOW = 11
PERSISTENCE_TIME = 0.20
HOLD_FRACTION = 0.80


# =============================================================================
# Gaussian angular calculus (draft P4)
# =============================================================================

def angular_weight(beta, c, q):
    """omega_p(beta) = int_0^inf r^3 p_p(r a_beta) dr for p_p = N(c, q^2 I_2).

    Closed form (draft 6.3-6.4). Not a probability density: integrates to
    |c|^2 + 2 q^2. The two terms are both O(exp(-|c|^2/2q^2)) when ell < 0,
    so the clipped value is exact to double precision.
    """
    ell = np.cos(beta) * c[0] + np.sin(beta) * c[1]
    kappa = ell / q
    c2 = float(c @ c)
    J0 = q * math.sqrt(TWO_PI) * Phi(kappa)
    pref = np.exp(-(c2 - ell * ell) / (2.0 * q * q))
    term1 = (ell ** 3 + 3.0 * ell * q * q) * J0 * pref
    term2 = q * q * (ell * ell + 2.0 * q * q) * math.exp(-c2 / (2.0 * q * q))
    return np.maximum((term1 + term2) / (TWO_PI * q * q), 0.0)


def halfspace_second_moment(theta, c, q):
    """Exact E[X X^T 1{a_theta^T X > 0}] for X ~ N(c, q^2 I_2). Used only in audits."""
    a = np.stack([np.cos(theta), np.sin(theta)], -1)
    ap = np.stack([-np.sin(theta), np.cos(theta)], -1)
    ell = a @ c
    cp = ap @ c
    k = ell / q
    Ph = Phi(k)
    ph = np.exp(-0.5 * k * k) / math.sqrt(TWO_PI)
    s_aa = (ell * ell + q * q) * Ph + ell * q * ph
    s_ap = cp * (ell * Ph + q * ph)
    s_pp = (cp * cp + q * q) * Ph
    M = (s_aa[:, None, None] * a[:, :, None] * a[:, None, :]
         + s_ap[:, None, None] * (a[:, :, None] * ap[:, None, :] + ap[:, :, None] * a[:, None, :])
         + s_pp[:, None, None] * ap[:, :, None] * ap[:, None, :])
    return M


class BetaGrid:
    """Uniform periodic grid on the data-direction circle."""

    def __init__(self, K, rho, q):
        self.K = int(K)
        self.dB = TWO_PI / self.K
        self.beta = self.dB * np.arange(self.K)
        self.cos = np.cos(self.beta)
        self.sin = np.sin(self.beta)
        self.A = np.stack([self.cos, self.sin], 1)                       # (K,2)
        self.c = [np.array([1.0, 0.0]), np.array([0.0, rho])]
        self.omega = np.stack([angular_weight(self.beta, cc, q) for cc in self.c], 0)  # (2,K)
        # integrand factor omega_p(beta) a_j(beta): (2,K,2)
        self.wa = self.omega[:, :, None] * self.A[None, :, :]


class SortedLabels:
    """Periodic half-circle window sums over labels sorted by input angle."""

    def __init__(self, theta):
        th = np.mod(theta, TWO_PI)
        self.order = np.argsort(th, kind="stable")
        self.th = th[self.order]
        self.L = th.size

    def prefix(self, vals):
        v = vals[self.order]
        P = np.empty((self.L + 1,) + v.shape[1:])
        P[0] = 0.0
        np.cumsum(v, axis=0, out=P[1:])
        return P

    def _G(self, P, x):
        n = np.floor(x / TWO_PI)
        r = x - TWO_PI * n
        idx = np.searchsorted(self.th, r, side="left")
        return n[:, None] * P[-1][None, :] + P[idx]

    def window(self, P, center):
        """Sum over labels with theta in [center - pi/2, center + pi/2) (mod 2 pi)."""
        return self._G(P, center + HALF_PI) - self._G(P, center - HALF_PI)


def periodic_window_integral(grid, h, theta):
    """int_{theta-pi/2}^{theta+pi/2} h(beta) dbeta for grid samples h (K,C),
    piecewise-linear interpolation of h (trapezoid with exact partial cells)."""
    K, dB = grid.K, grid.dB
    hn = np.roll(h, -1, axis=0)
    F = np.empty((K + 1, h.shape[1]))
    F[0] = 0.0
    np.cumsum(0.5 * dB * (h + hn), axis=0, out=F[1:])
    T = F[K]

    def Fint(x):
        n = np.floor(x / TWO_PI)
        r = x - TWO_PI * n
        k = np.minimum((r / dB).astype(np.int64), K - 1)
        k = np.maximum(k, 0)
        s = r - k * dB
        hk = h[k]
        return (n[:, None] * T[None, :] + F[k] + s[:, None] * hk
                + (0.5 * s * s / dB)[:, None] * (hn[k] - hk))

    return Fint(theta + HALF_PI) - Fint(theta - HALF_PI)


# =============================================================================
# Exact population field (draft P1, P2, P6, P7)
# =============================================================================

class Model:
    def __init__(self, rho, q, L, K):
        self.rho, self.q, self.L = float(rho), float(q), int(L)
        self.grid = BetaGrid(K, rho, q)

    def initial_state(self, S0):
        L = self.L
        alpha = TWO_PI * (np.arange(L) + 0.5) / L
        return np.concatenate([alpha, alpha.copy(), np.full(L, math.log(S0))])

    def unpack(self, y):
        L = self.L
        return y[:L], y[L:2 * L], y[2 * L:]

    def output_prefix(self, srt, w, ct, st, cp, sp):
        return srt.prefix(np.stack([w * cp * ct, w * sp * ct, w * cp * st, w * sp * st], 1))

    @staticmethod
    def output_from_window(Wf, cphi, sphi):
        return np.stack([cphi * Wf[:, 0] + sphi * Wf[:, 2],
                         cphi * Wf[:, 1] + sphi * Wf[:, 3]], 1)

    def evaluate(self, y):
        """Full field evaluation. Returns a dict (velocities, R_p, grid output)."""
        g_ = self.grid
        th, ps, lm = self.unpack(y)
        m = np.exp(lm)
        w = m / self.L
        ct, st, cp, sp = np.cos(th), np.sin(th), np.cos(ps), np.sin(ps)
        srt = SortedLabels(th)
        Pf = self.output_prefix(srt, w, ct, st, cp, sp)
        f_grid = self.output_from_window(srt.window(Pf, g_.beta), g_.cos, g_.sin)
        e_grid = g_.A - f_grid
        # h[k, p, i, j] = omega_p(beta_k) e_i(beta_k) a_j(beta_k)
        h = e_grid[:, None, :, None] * g_.wa.transpose(1, 0, 2)[:, :, None, :]
        R = periodic_window_integral(g_, h.reshape(g_.K, 8), th).reshape(self.L, 2, 2, 2)
        R1, R2 = R[:, 0], R[:, 1]
        Rm = 0.5 * (R1 + R2)
        a = np.stack([ct, st], 1)
        ap = np.stack([-st, ct], 1)
        b = np.stack([cp, sp], 1)
        bp = np.stack([-sp, cp], 1)
        Ra = np.einsum("lij,lj->li", Rm, a)
        Rap = np.einsum("lij,lj->li", Rm, ap)
        gg = np.sum(b * Ra, 1)
        vth = np.sum(b * Rap, 1)
        vps = np.sum(bp * Ra, 1)
        return dict(th=th, ps=ps, m=m, w=w, ct=ct, st=st, cp=cp, sp=sp, a=a, b=b,
                    srt=srt, Pf=Pf, f_grid=f_grid, e_grid=e_grid, R1=R1, R2=R2,
                    g=gg, vth=vth, vps=vps)

    def rhs(self, y):
        ev = self.evaluate(y)
        return np.concatenate([ev["vth"], ev["vps"], 2.0 * ev["g"]])

    # ------------------------------------------------------------------
    def observables(self, ev):
        """Learning observables and identity checks (normalized units)."""
        g_ = self.grid
        dB = g_.dB
        f, e = ev["f_grid"], ev["e_grid"]
        om1, om2 = g_.omega
        q2 = self.q ** 2
        m1 = dB * np.sum(om1 * g_.cos * f[:, 0]) / (1.0 + q2)
        m2 = dB * np.sum(om2 * g_.sin * f[:, 1]) / (self.rho ** 2 + q2)
        ee = np.sum(e * e, 1)
        L1 = 0.5 * dB * np.sum(om1 * ee)
        L2 = 0.5 * dB * np.sum(om2 * ee)
        ef = np.sum(e * f, 1)
        Sdot_data = dB * np.sum((om1 + om2) * ef)               # 2 E_P[e.f]
        w, gg, vth, vps = ev["w"], ev["g"], ev["vth"], ev["vps"]
        return dict(S=float(np.sum(w)), m1=float(m1), m2=float(m2),
                    L1=float(L1), L2=float(L2), L=float(0.5 * (L1 + L2)),
                    Sdot_labels=float(2.0 * np.sum(w * gg)), Sdot_data=float(Sdot_data),
                    Ldot_labels=float(-np.sum(w * (2 * gg * gg + vth * vth + vps * vps))))

    def probe_rates(self, ev, phis):
        """E, D_1, D_2 (normalized units) and f at unit probes xi = a_phi."""
        srt, w = ev["srt"], ev["w"]
        a, b, ct, st = ev["a"], ev["b"], ev["ct"], ev["st"]
        cols = []
        for R in (ev["R1"], ev["R2"]):
            Ra = np.einsum("lij,lj->li", R, a)
            RTb = np.einsum("lji,lj->li", R, b)
            cols += [w[:, None] * Ra * ct[:, None], w[:, None] * Ra * st[:, None],
                     (w[:, None, None] * b[:, :, None] * RTb[:, None, :]).reshape(-1, 4)]
        P = srt.prefix(np.concatenate(cols, 1))                  # (L+1, 16)
        Wv = srt.window(P, phis)
        Wf = srt.window(ev["Pf"], phis)
        cph, sph = np.cos(phis), np.sin(phis)
        xi = np.stack([cph, sph], 1)
        f = self.output_from_window(Wf, cph, sph)
        e = xi - f
        out = dict(f=f, E=0.5 * np.sum(e * e, 1))
        for p in (0, 1):
            o = 8 * p
            V = (cph[:, None] * Wv[:, o:o + 2] + sph[:, None] * Wv[:, o + 2:o + 4]
                 + np.einsum("qij,qj->qi", Wv[:, o + 4:o + 8].reshape(-1, 2, 2), xi))
            out[f"V{p + 1}"] = V
            out[f"D{p + 1}"] = -0.5 * np.sum(e * V, 1)
        return out

    def carrier_split(self, ev, phi, masks):
        """Source x carrier x motion decomposition of D_p at one probe (normalized units).

        With g_p = b^T R_p a, vth_p = b^T R_p a_perp, vps_p = b_perp^T R_p a,
        the per-label probe velocity from source p splits exactly into
            radial : 2 m (a.xi)_+ g_p b              (mass growth, both layers)
            inrot  : m 1{a.xi>0} (a_perp.xi) vth_p b (input-angle rotation)
            outrot : m (a.xi)_+ vps_p b_perp         (output-angle rotation)
        Near a probe gate edge (a.xi small, |a_perp.xi| ~ 1) the input-rotation
        term is amplified relative to the other two.
        """
        xi = np.array([math.cos(phi), math.sin(phi)])
        a, b, w = ev["a"], ev["b"], ev["w"]
        ap = np.stack([-a[:, 1], a[:, 0]], 1)
        bp = np.stack([-b[:, 1], b[:, 0]], 1)
        z = a @ xi
        zp = ap @ xi
        act = (z > 0).astype(float)
        zpos = np.maximum(z, 0.0)
        f = np.sum((w * zpos)[:, None] * b, 0)
        e = xi - f
        eb, ebp = b @ e, bp @ e
        res = {}
        for p, R in ((1, ev["R1"]), (2, ev["R2"])):
            Ra = np.einsum("lij,lj->li", R, a)
            gp = np.sum(b * Ra, 1)
            vps = np.sum(bp * Ra, 1)
            vth = np.einsum("li,lij,lj->l", b, R, ap)
            parts = dict(radial=2.0 * w * zpos * gp * eb,
                         inrot=w * act * zp * vth * eb,
                         outrot=w * zpos * vps * ebp)
            for name, mk in masks.items():
                tot = 0.0
                for term, c in parts.items():
                    val = float(-0.5 * np.sum(c[mk]))
                    res[f"D{p}_{name}_{term}"] = val
                    tot += val
                res[f"D{p}_{name}"] = tot
        return res


# =============================================================================
# Diagnostics helpers
# =============================================================================

def wrap_pi(x):
    return np.angle(np.exp(1j * x))


def weighted_quantile(v, wts, qs):
    if v.size == 0 or np.sum(wts) <= 0:
        return [float("nan")] * len(qs)
    o = np.argsort(v)
    cw = np.cumsum(wts[o])
    cw /= cw[-1]
    return [float(np.interp(qq, cw, v[o])) for qq in qs]


def family_masks(th):
    tw = wrap_pi(th)
    strong = np.abs(tw) < math.pi / 4
    weak = np.abs(wrap_pi(th - HALF_PI)) < math.pi / 4
    return dict(strong=strong, weak=weak, other=~(strong | weak))


def family_stats(ev, masks, focus_phis):
    th, ps, w = ev["th"], ev["ps"], ev["w"]
    out = {}
    for name, center in (("strong", 0.0), ("weak", HALF_PI)):
        mk = masks[name]
        wm = w[mk]
        M = float(np.sum(wm))
        out[f"{name}_mass"] = M
        if M <= 0:
            continue
        d_th = np.degrees(wrap_pi(th[mk] - center))
        d_ps = np.degrees(wrap_pi(ps[mk] - center))
        mean = float(np.sum(wm * d_th) / M)
        out[f"{name}_theta_mean_deg"] = mean
        out[f"{name}_theta_std_deg"] = float(math.sqrt(max(np.sum(wm * (d_th - mean) ** 2) / M, 0.0)))
        q05, q25, q50, q75, q95 = weighted_quantile(d_th, wm, [0.05, 0.25, 0.5, 0.75, 0.95])
        out.update({f"{name}_theta_q05_deg": q05, f"{name}_theta_q25_deg": q25,
                    f"{name}_theta_q50_deg": q50, f"{name}_theta_q75_deg": q75,
                    f"{name}_theta_q95_deg": q95})
        out[f"{name}_psi_mean_deg"] = float(np.sum(wm * d_ps) / M)
        for phi in focus_phis:
            tag = _tag(phi)
            act = np.cos(th[mk] - phi) > 0
            out[f"{name}_inactive_frac@{tag}"] = float(np.sum(wm[~act]) / M)
    S = float(np.sum(w))
    for phi in focus_phis:
        tag = _tag(phi)
        z = np.abs(np.cos(th - phi))
        for eta in (0.01, 0.03, 0.1):
            out[f"strip{eta:g}@{tag}"] = float(np.sum(w[z <= eta]) / S)
    return out


def _tag(phi_rad):
    return f"{math.degrees(phi_rad):+.1f}"


def moving_average(y, window=SMOOTH_WINDOW):
    left = window // 2
    right = window - 1 - left
    yp = np.pad(y, (left, right), mode="edge")
    c = np.concatenate([[0.0], np.cumsum(yp)])
    return (c[window:] - c[:-window]) / window


def persistent_swing(times, y):
    """Archived detector (theory_guided_swing_by_phase_diagram.persistent_swing)."""
    times = np.asarray(times, float)
    ys = moving_average(np.asarray(y, float))
    T = len(ys)
    dt = float(np.median(np.diff(times))) if T > 1 else 1.0
    hold = max(1, int(round(PERSISTENCE_TIME / dt)))
    best = (0.0, float("nan"), float("nan"))
    if T < hold + 3:
        return best
    from numpy.lib.stride_tricks import sliding_window_view
    wmin = sliding_window_view(ys, hold + 1).min(axis=1)
    jmax = T - hold - 1
    interior = np.arange(1, T - 1)
    mins = interior[(ys[interior] <= ys[interior - 1]) & (ys[interior] < ys[interior + 1])]
    for i in mins:
        if i + 1 > jmax:
            continue
        j = np.arange(i + 1, jmax + 1)
        delta = ys[j] - ys[i]
        ok = (delta > 0) & (wmin[j] >= ys[i] + HOLD_FRACTION * delta)
        if ok.any():
            dj = np.where(ok, delta, -np.inf)
            k = int(np.argmax(dj))
            if dj[k] > best[0]:
                best = (float(dj[k]), float(times[i]), float(times[j[k]]))
    return best


def direction_stats(times, E):
    """S, t_reversal, rise, t_min, final_excess per column of E (T x P)."""
    P = E.shape[1]
    S = np.zeros(P)
    t_rev = np.full(P, np.nan)
    for k in range(P):
        S[k], t_rev[k], _ = persistent_swing(times, E[:, k])
    rise = (E - np.minimum.accumulate(E, axis=0)).max(axis=0)
    imin = np.argmin(E, axis=0)
    return dict(S=S, t_rev=t_rev, rise=rise, t_min=np.asarray(times)[imin],
                final_excess=E[-1] - E[imin, np.arange(P)], E0=E[0], Emin=E[imin, np.arange(P)])


def first_up_crossing(t, y, t_lo):
    idx = np.where(t >= t_lo)[0]
    for i, j in zip(idx[:-1], idx[1:]):
        if y[i] <= 0.0 < y[j]:
            return float(t[i] - y[i] * (t[j] - t[i]) / (y[j] - y[i]))
    return float("nan")


def count_crossings(t, y, t_lo):
    s = np.sign(y[t >= t_lo])
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))


# =============================================================================
# Audits
# =============================================================================

def audit_initial(model, y0, S0, probe_phis):
    """Closed-form checks at t = 0 (draft P3 and the exact half-space moments)."""
    ev = model.evaluate(y0)
    obs = model.observables(ev)
    g_ = model.grid
    c0 = 1.0 - S0 / 4.0
    out = {}
    M2p = [1.0 + 2 * model.q ** 2, model.rho ** 2 + 2 * model.q ** 2]
    out["omega_mass_relerr"] = [float(abs(g_.dB * g_.omega[p].sum() - M2p[p]) / M2p[p]) for p in (0, 1)]
    out["f0_grid_relerr"] = float(np.max(np.abs(ev["f_grid"] - (S0 / 4) * g_.A)) / (S0 / 4))
    out["m_p0_relerr"] = [abs(obs["m1"] - S0 / 4) / (S0 / 4), abs(obs["m2"] - S0 / 4) / (S0 / 4)]
    out["L_p0_relerr"] = [abs(obs["L1"] - 0.5 * c0 ** 2 * M2p[0]) / (0.5 * c0 ** 2 * M2p[0]),
                          abs(obs["L2"] - 0.5 * c0 ** 2 * M2p[1]) / (0.5 * c0 ** 2 * M2p[1])]
    th = ev["th"]
    errs = []
    for p, R in ((0, ev["R1"]), (1, ev["R2"])):
        exact = c0 * halfspace_second_moment(th, g_.c[p], model.q)
        errs.append(float(np.max(np.abs(R - exact)) / np.max(np.abs(exact))))
    out["R_p0_vs_closed_form_relerr"] = errs
    pr = model.probe_rates(ev, probe_phis)
    xi = np.stack([np.cos(probe_phis), np.sin(probe_phis)], 1)
    out["f0_probe_relerr"] = float(np.max(np.abs(pr["f"] - (S0 / 4) * xi)) / (S0 / 4))
    out["Sdot_identity_relerr0"] = abs(obs["Sdot_labels"] - obs["Sdot_data"]) / abs(obs["Sdot_data"])
    return out


def audit_montecarlo(model, y, n_mc=2_000_000, n_labels=6, seed=0):
    """Monte Carlo transcription check of R_p(theta_l) at a few labels.

    f(X) is evaluated both by the sorted-window code path and, on a subsample,
    by a direct O(N L) sum. Reports |grid - MC| in units of the MC standard error.
    """
    rng = np.random.default_rng(seed)
    ev = model.evaluate(y)
    m = ev["m"]
    L = model.L
    pick = list(np.argsort(m)[-n_labels // 2:]) + list(rng.choice(L, n_labels - n_labels // 2, replace=False))
    srt, Pf, w = ev["srt"], ev["Pf"], ev["w"]
    out = {"labels": [int(i) for i in pick], "max_z": 0.0, "direct_f_maxerr": 0.0}
    for p in (0, 1):
        c = model.grid.c[p]
        X = c[None, :] + model.q * rng.standard_normal((n_mc, 2))
        r = np.hypot(X[:, 0], X[:, 1])
        ang = np.arctan2(X[:, 1], X[:, 0])
        Wf = srt.window(Pf, ang)
        fu = model.output_from_window(Wf, np.cos(ang), np.sin(ang))
        fX = r[:, None] * fu
        # direct check of the window code on a subsample
        sub = slice(0, 2000)
        z = X[sub] @ ev["a"].T
        f_dir = (np.maximum(z, 0.0) * w[None, :]) @ ev["b"]
        out["direct_f_maxerr"] = max(out["direct_f_maxerr"], float(np.max(np.abs(f_dir - fX[sub]))))
        e = X - fX
        R_grid = ev["R1"] if p == 0 else ev["R2"]
        for l in pick:
            gate = (X @ ev["a"][l]) > 0
            val = (e[:, :, None] * X[:, None, :]) * gate[:, None, None]
            mc = val.mean(0)
            se = val.std(0) / math.sqrt(n_mc) + 1e-300
            zsc = float(np.max(np.abs(R_grid[l] - mc) / se))
            out["max_z"] = max(out["max_z"], zsc)
    return out


# =============================================================================
# Run
# =============================================================================

def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mu1 = args.mu1
    rho = args.rho if args.rho is not None else args.mu2 / args.mu1
    q = args.q if args.q is not None else args.sigma / args.mu1
    S0 = args.S0
    tmax = args.tmax
    if args.auto_horizon and S0 < 2e-4:
        tmax = tmax + args.horizon_slope * math.log(2e-4 / S0) / mu1 ** 2
    save_dt = args.save_dt
    n_save = int(round(tmax / save_dt))
    dtau_save = mu1 ** 2 * save_dt
    n_sub = max(1, int(math.ceil(dtau_save / args.dt - 1e-9)))
    hstep = dtau_save / n_sub

    model = Model(rho, q, args.L, args.K)
    y = model.initial_state(S0)

    focus_deg = [float(v) for v in args.focus.split(",")]
    focus = np.radians(focus_deg)
    sec_lo, sec_hi, sec_step = [float(v) for v in args.sector.split(",")]
    sector_deg = np.round(np.arange(sec_lo, sec_hi + 1e-9, sec_step), 6)
    probe_deg = np.unique(np.round(np.concatenate([ARCHIVE_GRID_DEG, sector_deg, focus_deg]), 6))
    probe_phi = np.radians(probe_deg)
    i_grid = np.searchsorted(probe_deg, np.round(ARCHIVE_GRID_DEG, 6))
    i_sec = np.searchsorted(probe_deg, sector_deg)
    i_focus = np.searchsorted(probe_deg, np.round(focus_deg, 6))

    params = dict(rho=rho, q=q, S0=S0, eps_equiv_h200=math.sqrt(S0 / 200.0), mu1=mu1,
                  L=args.L, K=args.K, dt_tau=hstep, save_dt_phys=save_dt, tmax_phys=tmax,
                  focus_deg=focus_deg, sector=[sec_lo, sec_hi, sec_step])
    print(f"[pilot] rho={rho:.6g} q={q:.6g} S0={S0:.3g} L={args.L} K={args.K} "
          f"dtau={hstep:.4g} tmax_phys={tmax:.4g} saves={n_save + 1}", flush=True)

    audits = {}
    if args.audit:
        audits["initial"] = audit_initial(model, y, S0, probe_phi)
        print("[audit t=0]", json.dumps(audits["initial"]), flush=True)

    rate = mu1 ** 2
    T = n_save + 1
    P = probe_deg.size
    E_all = np.empty((T, P))
    D1_all = np.empty((T, P))
    D2_all = np.empty((T, P))
    rows = []
    snaps_t, snaps = [], []
    wall0 = time.time()

    def rk4(yv, hh):
        k1 = model.rhs(yv)
        k2 = model.rhs(yv + 0.5 * hh * k1)
        k3 = model.rhs(yv + 0.5 * hh * k2)
        k4 = model.rhs(yv + hh * k3)
        return yv + (hh / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    for s in range(T):
        t_phys = s * save_dt
        ev = model.evaluate(y)
        obs = model.observables(ev)
        pr = model.probe_rates(ev, probe_phi)
        E_all[s] = pr["E"]
        D1_all[s] = rate * pr["D1"]
        D2_all[s] = rate * pr["D2"]
        masks = family_masks(ev["th"])
        row = dict(t=t_phys, tau=rate * t_phys)
        row.update(obs)
        row["Sdot_labels"] *= rate
        row["Sdot_data"] *= rate
        row["Ldot_labels"] *= rate
        row.update(family_stats(ev, masks, focus))
        for k, phi in enumerate(focus):
            tag = _tag(phi)
            j = i_focus[k]
            row[f"E@{tag}"] = pr["E"][j]
            row[f"D1@{tag}"] = rate * pr["D1"][j]
            row[f"D2@{tag}"] = rate * pr["D2"][j]
            cs = model.carrier_split(ev, phi, masks)
            for key, val in cs.items():
                row[f"{key}@{tag}"] = rate * val
        rows.append(row)
        if args.snapshot_every > 0 and s % args.snapshot_every == 0:
            snaps_t.append(t_phys)
            snaps.append(y.copy())
        if s % max(1, int(round(1.0 / save_dt))) == 0 or s == T - 1:
            el = time.time() - wall0
            ftag = _tag(focus[0])
            print(f"  t={t_phys:6.2f} tau={rate * t_phys:7.2f}  S={obs['S']:.4f} m1={obs['m1']:.4f} "
                  f"m2={obs['m2']:.4f}  E{ftag}={row[f'E@{ftag}']:.6f} "
                  f"D1={row[f'D1@{ftag}']:+.3e} D2={row[f'D2@{ftag}']:+.3e}  [{el:6.0f}s]", flush=True)
        if s == T - 1:
            break
        for _ in range(n_sub):
            y = rk4(y, hstep)
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(f"non-finite state after t={t_phys + save_dt}")
    wall = time.time() - wall0

    times = np.array([r["t"] for r in rows])
    np.savez_compressed(out / "probes.npz", t=times, tau=rate * times, psi_deg=probe_deg,
                        E=E_all, D1=D1_all, D2=D2_all, i_grid=i_grid, i_sector=i_sec,
                        i_focus=i_focus)
    if snaps:
        np.savez_compressed(out / "labels.npz", t=np.array(snaps_t), state=np.array(snaps),
                            L=args.L, note="state = [theta(L), psi(L), log m(L)], weight 1/L")
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "timeseries.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    # ------------------------------------------------------------------ audits
    if args.audit:
        Ldot_fd = np.gradient(np.array([r["L"] for r in rows]), times)
        Ldot_lab = np.array([r["Ldot_labels"] for r in rows])
        Sd_lab = np.array([r["Sdot_labels"] for r in rows])
        Sd_dat = np.array([r["Sdot_data"] for r in rows])
        Ef = E_all[:, i_focus[0]]
        Df = D1_all[:, i_focus[0]] + D2_all[:, i_focus[0]]
        # trapezoid consistency: E_{k+1} - E_k vs dt (Edot_k + Edot_{k+1})/2
        dE = np.diff(Ef)
        dE_pred = save_dt * 0.5 * (Df[1:] + Df[:-1])
        audits["Edot_trapezoid_maxabs_err"] = float(np.max(np.abs(dE - dE_pred)))
        audits["Edot_trapezoid_rel_to_maxdE"] = float(np.max(np.abs(dE - dE_pred)) / (np.max(np.abs(dE)) + 1e-300))
        audits["Sdot_identity_max_relerr"] = float(np.max(np.abs(Sd_lab - Sd_dat) / (np.abs(Sd_dat) + 1e-12)))
        audits["Ldot_fd_vs_identity_max_relerr(t>0.1)"] = float(
            np.max(np.abs(Ldot_fd - Ldot_lab)[times > 0.1] / (np.abs(Ldot_lab[times > 0.1]) + 1e-12)))
        audits["loss_monotone_max_increase"] = float(np.max(np.diff([r["L"] for r in rows])))
        audits["montecarlo_final"] = audit_montecarlo(model, y, n_mc=args.mc_samples)
        print("[audit final]", json.dumps({k: v for k, v in audits.items() if k != "initial"}), flush=True)

    summary = summarize(out, params, rows, times, probe_deg, E_all, D1_all, D2_all,
                        i_grid, i_sec, i_focus, focus, args)
    summary["audits"] = audits
    summary["wall_seconds"] = wall
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    if not args.no_plots:
        try:
            make_plots(out, rows, times, probe_deg, E_all, D1_all, D2_all, i_sec, focus)
        except Exception as exc:  # plots are optional
            print(f"[plots skipped: {exc}]")
    print_summary(summary)
    return summary


def interp_at(times, y, t0):
    if not (times[0] <= t0 <= times[-1]):
        return float("nan")
    return float(np.interp(t0, times, y))


def summarize(out, params, rows, times, probe_deg, E, D1, D2, i_grid, i_sec, i_focus, focus, args):
    s = dict(params=params)
    m1 = np.array([r["m1"] for r in rows])
    m2 = np.array([r["m2"] for r in rows])
    clocks = {}
    for lev in (0.1, 0.5, 0.9):
        for name, arr in (("m1", m1), ("m2", m2)):
            k = np.where(arr >= lev)[0]
            clocks[f"t_{name}>={lev:g}"] = float(times[k[0]]) if k.size else float("nan")
    s["clocks_phys"] = clocks
    s["final"] = {k: rows[-1][k] for k in ("S", "m1", "m2", "L1", "L2", "L")}

    ds = direction_stats(times, E)
    np.savez_compressed(out / "direction_stats.npz", psi_deg=probe_deg, **ds)
    g = i_grid
    deg_g = probe_deg[g]
    rg = np.radians(deg_g)
    incone = (np.cos(rg) >= -1e-12) & (np.sin(rg) >= -1e-12)
    s["grid_counts"] = {f"n_{c}_S>={th:g}": int(np.sum((ds["S"][g] >= th) & (incone if c == "in" else ~incone)))
                        for th in (1e-4, 1e-5) for c in ("in", "off")}

    # per-probe sector table
    sec_rows = []
    delta = args.split_halfwidth
    for j in i_sec:
        Dsum = D1[:, j] + D2[:, j]
        tx = first_up_crossing(times, Dsum, args.cross_tmin)
        row = dict(psi_deg=float(probe_deg[j]), S=float(ds["S"][j]), t_rev=float(ds["t_rev"][j]),
                   rise=float(ds["rise"][j]), t_min=float(ds["t_min"][j]),
                   final_excess=float(ds["final_excess"][j]), t_cross=tx,
                   n_sign_changes=count_crossings(times, Dsum, args.cross_tmin))
        if np.isfinite(tx):
            d1x, d2x = interp_at(times, D1[:, j], tx), interp_at(times, D2[:, j], tx)
            row.update(D1_at_cross=d1x, D2_at_cross=d2x, opposite_signs_at_cross=bool(d1x < 0 < d2x))
            ta, tb = tx - delta, tx + delta
            dD1 = interp_at(times, D1[:, j], tb) - interp_at(times, D1[:, j], ta)
            dD2 = interp_at(times, D2[:, j], tb) - interp_at(times, D2[:, j], ta)
            row.update(dD1=dD1, dD2=dD2, strong_share_of_swing=dD1 / (dD1 + dD2) if dD1 + dD2 != 0 else float("nan"))
            # whole-window opposite signs between cross_tmin and t_min+delta
            win = (times >= tx - delta) & (times <= tx + delta)
            row["D1<0<D2_on_window"] = bool(np.all(D1[win, j] < 0) and np.all(D2[win, j] > 0))
        sec_rows.append(row)
    keys = sorted({k for r in sec_rows for k in r}, key=lambda k: (k != "psi_deg", k))
    with open(out / "sector.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in sec_rows:
            wr.writerow(r)
    Ssec = np.array([r["S"] for r in sec_rows])
    kbest = int(np.argmax(Ssec))
    s["sector"] = {
        "max_S": float(Ssec[kbest]), "psi_at_max_S": sec_rows[kbest]["psi_deg"],
        "n_probes": len(sec_rows),
        "n_S>=1e-4": int(np.sum(Ssec >= 1e-4)),
        "psi_range_S>=1e-4": ([float(min(r["psi_deg"] for r in sec_rows if r["S"] >= 1e-4)),
                               float(max(r["psi_deg"] for r in sec_rows if r["S"] >= 1e-4))]
                              if np.any(Ssec >= 1e-4) else None),
        "n_with_cross": int(sum(np.isfinite(r["t_cross"]) for r in sec_rows)),
        "n_D1<0<D2_at_cross": int(sum(bool(r.get("opposite_signs_at_cross")) for r in sec_rows)),
    }

    # focus probes
    s["focus"] = {}
    for k, phi in enumerate(focus):
        j = i_focus[k]
        tag = _tag(phi)
        Dsum = D1[:, j] + D2[:, j]
        tx = first_up_crossing(times, Dsum, args.cross_tmin)
        fr = dict(S=float(ds["S"][j]), t_rev=float(ds["t_rev"][j]), rise=float(ds["rise"][j]),
                  t_min=float(ds["t_min"][j]), final_excess=float(ds["final_excess"][j]),
                  E0=float(E[0, j]), Emin=float(ds["Emin"][j]), t_cross=tx,
                  n_sign_changes=count_crossings(times, Dsum, args.cross_tmin))
        for dtm in (0.5, 1.0, 3.0):
            fr[f"E(t_min+{dtm:g})-Emin"] = interp_at(times, E[:, j], fr["t_min"] + dtm) - fr["Emin"]
        for t0 in (3.4, 3.7, 3.9):
            fr[f"E({t0})"] = interp_at(times, E[:, j], t0)
            fr[f"D1({t0})"] = interp_at(times, D1[:, j], t0)
            fr[f"D2({t0})"] = interp_at(times, D2[:, j], t0)
        if np.isfinite(fr["D1(3.4)"]) and np.isfinite(fr["D1(3.9)"]):
            a1 = fr["D1(3.9)"] - fr["D1(3.4)"]
            a2 = fr["D2(3.9)"] - fr["D2(3.4)"]
            fr["strong_share_3.4_to_3.9"] = a1 / (a1 + a2) if a1 + a2 != 0 else float("nan")
        if np.isfinite(tx):
            ta, tb = tx - args.split_halfwidth, tx + args.split_halfwidth
            a1 = interp_at(times, D1[:, j], tb) - interp_at(times, D1[:, j], ta)
            a2 = interp_at(times, D2[:, j], tb) - interp_at(times, D2[:, j], ta)
            fr["strong_share_around_cross"] = a1 / (a1 + a2) if a1 + a2 != 0 else float("nan")
            fr["D1(t_cross)"] = interp_at(times, D1[:, j], tx)
            fr["D2(t_cross)"] = interp_at(times, D2[:, j], tx)
            # path-integral ledger from t_min to the end of the horizon
            tm = fr["t_min"]
            sel = times >= tm
            if sel.sum() > 1:
                fr["int_D1_tmin_to_end"] = float(_trapz(D1[sel, j], times[sel]))
                fr["int_D2_tmin_to_end"] = float(_trapz(D2[sel, j], times[sel]))
            # carriers at the crossing
            tt = np.array([r["t"] for r in rows])
            for key in rows[0]:
                if key.endswith(f"@{tag}") and key.startswith(("D1_", "D2_")):
                    fr[f"{key.split('@')[0]}(t_cross)"] = interp_at(tt, np.array([r[key] for r in rows]), tx)
            for key in ("strong_theta_mean_deg", "strong_theta_std_deg", "strong_theta_q05_deg",
                        "strong_theta_q95_deg", f"strong_inactive_frac@{tag}", f"strip0.03@{tag}", "m1", "m2"):
                if key in rows[0]:
                    fr[f"{key}(t_cross)"] = interp_at(tt, np.array([r.get(key, np.nan) for r in rows]), tx)
        s["focus"][tag] = fr
    return s


def print_summary(s):
    p = s["params"]
    print("\n" + "=" * 78)
    print(f"SUMMARY  rho={p['rho']:.4g} q={p['q']:.4g} S0={p['S0']:.3g} (eps@h200={p['eps_equiv_h200']:.3g}) "
          f"L={p['L']} K={p['K']}")
    print("clocks (physical t):", {k: round(v, 3) for k, v in s["clocks_phys"].items()})
    print("final:", {k: round(v, 5) for k, v in s["final"].items()})
    print("0.5-deg grid counts:", s["grid_counts"])
    print("sector:", s["sector"])
    for tag, fr in s["focus"].items():
        print(f"focus {tag} deg:")
        for k, v in fr.items():
            print(f"   {k:34s} {v}")
    if s.get("audits"):
        print("audits:", json.dumps(s["audits"], default=float))
    print("=" * 78)


# =============================================================================
# Plots
# =============================================================================

def make_plots(out, rows, times, probe_deg, E, D1, D2, i_sec, focus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = _tag(focus[0])
    j = int(np.searchsorted(probe_deg, round(math.degrees(focus[0]), 6)))
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)
    ax[0].plot(times, E[:, j], color="k")
    ax[0].set_ylabel(f"E at {tag} deg")
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].plot(times, D1[:, j], label="$D_1$ (strong-sourced)")
    ax[1].plot(times, D2[:, j], label="$D_2$ (weak-sourced)")
    ax[1].plot(times, D1[:, j] + D2[:, j], color="k", lw=1, label="$\\dot E$")
    ax[1].set_ylabel("physical rate")
    ax[1].legend(fontsize=8)
    ax[2].plot(times, [r["m1"] for r in rows], label="$m_1$")
    ax[2].plot(times, [r["m2"] for r in rows], label="$m_2$")
    ax[2].set_ylabel("learning")
    ax[2].set_xlabel("physical time t")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "focus_probe.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    ax[0].plot(times, [r.get("strong_theta_mean_deg", np.nan) for r in rows], label="mean")
    ax[0].fill_between(times, [r.get("strong_theta_q05_deg", np.nan) for r in rows],
                       [r.get("strong_theta_q95_deg", np.nan) for r in rows], alpha=0.25, label="5-95%")
    ax[0].axhline(math.degrees(focus[0]) + 90.0, color="r", lw=0.8, ls="--", label=f"gate edge of {tag}")
    ax[0].set_ylabel("strong input angle (deg)")
    ax[0].set_ylim(-10, 30)
    ax[0].legend(fontsize=8)
    ax[1].plot(times, [r.get(f"strong_inactive_frac@{tag}", np.nan) for r in rows])
    ax[1].set_ylabel(f"strong mass inactive at {tag}")
    ax[1].set_xlabel("physical time t")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "strong_family.png", dpi=140)
    plt.close(fig)

    ds = np.load(out / "direction_stats.npz")
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(probe_deg[i_sec], ds["S"][i_sec], ".-", ms=2)
    ax.set_xlabel("probe angle (deg)")
    ax.set_ylabel("S (persistent increase)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "sector_S.png", dpi=140)
    plt.close(fig)


# =============================================================================
# Compare / archive / sweep
# =============================================================================

def load_run(d):
    d = Path(d)
    pr = np.load(d / "probes.npz")
    with open(d / "summary.json") as fh:
        sm = json.load(fh)
    return pr, sm


def compare(args):
    (pa, sa), (pb, sb) = load_run(args.a), load_run(args.b)
    t = np.intersect1d(np.round(pa["t"], 8), np.round(pb["t"], 8))
    ia = np.searchsorted(np.round(pa["t"], 8), t)
    ib = np.searchsorted(np.round(pb["t"], 8), t)
    print(f"common saves: {t.size}, t in [{t[0]}, {t[-1]}]")
    for deg in [float(v) for v in args.probes.split(",")]:
        ja = int(np.argmin(np.abs(pa["psi_deg"] - deg)))
        jb = int(np.argmin(np.abs(pb["psi_deg"] - deg)))
        print(f"probe {deg:+.1f}:")
        for key in ("E", "D1", "D2"):
            A, B = pa[key][ia, ja], pb[key][ib, jb]
            scale = np.max(np.abs(B)) if key != "E" else np.max(np.abs(B - B[0])) + 1e-300
            print(f"   {key:3s} max|a-b| = {np.max(np.abs(A - B)):.3e}   relative to scale {scale:.3e}: "
                  f"{np.max(np.abs(A - B)) / scale:.3e}")
        Da = pa["D1"][ia, ja] + pa["D2"][ia, ja]
        Db = pb["D1"][ib, jb] + pb["D2"][ib, jb]
        print(f"   D1+D2 max|a-b| / max|D1| = {np.max(np.abs(Da - Db)) / np.max(np.abs(pb['D1'][ib, jb])):.3e}")
    for tag in sa["focus"]:
        if tag in sb["focus"]:
            fa, fb = sa["focus"][tag], sb["focus"][tag]
            print(f"focus {tag}: " + ", ".join(f"{k}: {fa.get(k)} vs {fb.get(k)}"
                                               for k in ("S", "t_min", "t_cross", "strong_share_around_cross")))
    print("clocks a:", sa["clocks_phys"])
    print("clocks b:", sb["clocks_phys"])


def archive(args):
    """Compare continuum per-direction statistics with the h=200 ReLU seed ensemble."""
    pr, sm = load_run(args.run)
    ds = np.load(Path(args.run) / "direction_stats.npz")
    arch = Path(args.archive)
    st = np.load(arch / "direction_stats.npz", allow_pickle=True)
    keys = [k for k in st.files if "__relu__" in k and k.endswith("__S")]
    if not keys:
        sys.exit("no ReLU runs found in archive direction_stats.npz")
    stems = [k[:-3] for k in keys]
    tmax = float(pr["t"][-1])
    print(f"archive ReLU runs: {len(stems)}; continuum horizon t={tmax:g} "
          f"(archive E1 horizon is 20: S/final_excess comparable only if equal)")
    g = pr["i_grid"]
    deg = pr["psi_deg"][g]
    for stat in ("S", "rise", "t_min", "final_excess"):
        A = np.stack([st[f"{s}__{stat}"] for s in stems], 0)       # (seeds, 720)
        med = np.median(A, 0)
        lo, hi = np.percentile(A, 10, 0), np.percentile(A, 90, 0)
        c = ds[stat][g]
        inside = np.mean((c >= lo) & (c <= hi))
        print(f"\n{stat}: fraction of 720 directions with continuum inside seed 10-90% band: {inside:.3f}")
        for d0 in [float(v) for v in args.probes.split(",")]:
            k = int(np.argmin(np.abs(deg - d0)))
            print(f"   {deg[k]:+7.1f} deg  continuum {c[k]: .4e}   seeds median {med[k]: .4e}  "
                  f"[10%,90%] = [{lo[k]: .4e}, {hi[k]: .4e}]")
    rsum = arch / "run_summary.csv"
    if rsum.exists():
        import csv as _csv
        with open(rsum) as fh:
            R = [r for r in _csv.DictReader(fh) if r["arch"] == "relu"]
        for col in ("m85_S", "m85_t_min", "m85_final_excess"):
            v = np.array([float(r[col]) for r in R])
            print(f"archive {col}: median {np.median(v):.4g}  min {v.min():.4g}  max {v.max():.4g}")


def sweep(args):
    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    for S0 in [float(v) for v in args.S0_list.split(",")]:
        sub = argparse.Namespace(**vars(args))
        sub.S0 = S0
        sub.out = str(base / f"S0_{S0:.0e}")
        sub.auto_horizon = True
        s = run(sub)
        tag = list(s["focus"].keys())[0]
        fr = s["focus"][tag]
        rows.append(dict(S0=S0, eps_equiv_h200=math.sqrt(S0 / 200.0),
                         t_m1_05=s["clocks_phys"]["t_m1>=0.5"], t_m2_05=s["clocks_phys"]["t_m2>=0.5"],
                         S_focus=fr["S"], rise_focus=fr["rise"], final_excess_focus=fr["final_excess"],
                         rebound_0p5=fr.get("E(t_min+0.5)-Emin"), rebound_1=fr.get("E(t_min+1)-Emin"),
                         rebound_3=fr.get("E(t_min+3)-Emin"), Emin_focus=fr["Emin"],
                         t_min_focus=fr["t_min"], t_cross_focus=fr["t_cross"],
                         D1_cross=fr.get("D1(t_cross)"), D2_cross=fr.get("D2(t_cross)"),
                         strong_share=fr.get("strong_share_around_cross"),
                         strong_std_deg_cross=fr.get("strong_theta_std_deg(t_cross)"),
                         strong_mean_deg_cross=fr.get("strong_theta_mean_deg(t_cross)"),
                         inactive_frac_cross=fr.get(f"strong_inactive_frac@{tag}(t_cross)"),
                         sector_max_S=s["sector"]["max_S"], sector_psi_max=s["sector"]["psi_at_max_S"]))
        with open(base / "sweep_summary.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
    print("\nSWEEP")
    for r in rows:
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


# =============================================================================
# CLI
# =============================================================================

def add_run_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--mu1", type=float, default=3.0, help="physical mu1 (sets time/rate conversion)")
    p.add_argument("--mu2", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--rho", type=float, default=None, help="override mu2/mu1")
    p.add_argument("--q", type=float, default=None, help="override sigma/mu1")
    p.add_argument("--S0", type=float, default=2e-4, help="initial total mass h*eps^2")
    p.add_argument("--L", type=int, default=16384, help="number of labels")
    p.add_argument("--K", type=int, default=65536, help="data-direction grid size")
    p.add_argument("--dt", type=float, default=0.045, help="max RK4 step in normalized time tau")
    p.add_argument("--tmax", type=float, default=20.0, help="physical horizon (E1_canonical uses 20)")
    p.add_argument("--save-dt", type=float, default=0.01, help="physical save spacing (archive: 0.01)")
    p.add_argument("--auto-horizon", action="store_true",
                   help="extend tmax by horizon_slope*log(2e-4/S0)/mu1^2 when S0 < 2e-4")
    p.add_argument("--horizon-slope", type=float, default=3.0)
    p.add_argument("--focus", default="-85", help="comma-separated focus probe angles (deg)")
    p.add_argument("--sector", default="-100,-60,0.1", help="lo,hi,step (deg) for the sector table")
    p.add_argument("--cross-tmin", type=float, default=1.5, help="ignore D1+D2 sign changes before this t")
    p.add_argument("--split-halfwidth", type=float, default=0.25, help="half-window for the D1/D2 swing split")
    p.add_argument("--snapshot-every", type=int, default=25, help="save label state every N saves (0=off)")
    p.add_argument("--audit", action="store_true", help="run closed-form, identity and Monte Carlo audits")
    p.add_argument("--mc-samples", type=int, default=2_000_000)
    p.add_argument("--no-plots", action="store_true")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    pr = sp.add_parser("run")
    add_run_args(pr)
    pc = sp.add_parser("compare")
    pc.add_argument("a")
    pc.add_argument("b")
    pc.add_argument("--probes", default="-85,-80,-88")
    pa = sp.add_parser("archive")
    pa.add_argument("run")
    pa.add_argument("--archive", default="arch_control_summary/E1_canonical")
    pa.add_argument("--probes", default="-85,-80,-88,-76,45,135")
    ps = sp.add_parser("sweep")
    add_run_args(ps)
    ps.add_argument("--S0-list", default="2e-8,2e-7,2e-6,2e-5,2e-4,2e-3")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args)
    elif args.cmd == "compare":
        compare(args)
    elif args.cmd == "archive":
        archive(args)
    elif args.cmd == "sweep":
        sweep(args)


if __name__ == "__main__":
    main()