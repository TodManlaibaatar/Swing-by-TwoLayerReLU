"""Pre-specified per-direction statistics (see PROTOCOL.md, Section 3).

``persistent_increase`` is a vectorized re-implementation of the archived
detector ``theory_guided_swing_by_phase_diagram.persistent_swing`` (11-sample
moving average, 0.20 time-unit persistence, 80% hold).  ``selftest.py`` checks
that it returns the same swing values as the archived function.
"""
from __future__ import annotations

import numpy as np

SMOOTH_WINDOW = 11
PERSISTENCE_TIME = 0.20
HOLD_FRACTION = 0.80
CONE_TOL = 1e-12
THRESHOLDS = (1e-4, 5e-5, 1e-5)   # archived reporting threshold and the two lower checks
MONOTONE_TOL = 1e-9                # raw sampled rise below this counts as monotone
RECOVERY_FRACTION = 0.10           # final excess <= 10% of the rise counts as "recovered"


def moving_average_cols(Y: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Edge-padded centred moving average along axis 0 (matches the archive)."""
    Y = np.asarray(Y, dtype=float)
    if window <= 1:
        return Y.copy()
    left = window // 2; right = window - 1 - left
    Yp = np.pad(Y, ((left, right),) + ((0, 0),) * (Y.ndim - 1), mode="edge")
    c = np.cumsum(np.concatenate([np.zeros((1,) + Yp.shape[1:]), Yp], axis=0), axis=0)
    return (c[window:] - c[:-window]) / window


def _sliding_min(y: np.ndarray, w: int) -> np.ndarray:
    """out[j] = min(y[j:j+w]) for j = 0..len(y)-w."""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(y, w).min(axis=1)


def persistent_increase(times: np.ndarray, Y: np.ndarray):
    """Largest persistent increase for each column of Y (shape T x K).

    Returns dict of arrays: swing, t_reversal, t_rebound (NaN when swing == 0).
    """
    times = np.asarray(times, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    T, K = Y.shape
    S = moving_average_cols(Y)
    dt = float(np.median(np.diff(times))) if T > 1 else 1.0
    hold = max(1, int(round(PERSISTENCE_TIME / dt)))
    swing = np.zeros(K); t_rev = np.full(K, np.nan); t_reb = np.full(K, np.nan)
    if T < hold + 3:
        return dict(swing=swing, t_reversal=t_rev, t_rebound=t_reb)
    jmax = T - hold - 1                       # last admissible rebound index
    for k in range(K):
        ys = S[:, k]
        interior = np.arange(1, T - 1)
        is_min = (ys[interior] <= ys[interior - 1]) & (ys[interior] < ys[interior + 1])
        mins = interior[is_min]
        if mins.size == 0:
            continue
        wmin = _sliding_min(ys, hold + 1)     # wmin[j] = min ys[j : j+hold+1]
        best = 0.0; bi = bj = -1
        for i in mins:
            if i + 1 > jmax:
                continue
            j = np.arange(i + 1, jmax + 1)
            delta = ys[j] - ys[i]
            ok = (delta > 0) & (wmin[j] >= ys[i] + HOLD_FRACTION * delta)
            if not ok.any():
                continue
            dj = np.where(ok, delta, -np.inf)
            m = int(np.argmax(dj))
            if dj[m] > best:
                best = float(dj[m]); bi = int(i); bj = int(j[m])
        if bi >= 0:
            swing[k] = best; t_rev[k] = times[bi]; t_reb[k] = times[bj]
    return dict(swing=swing, t_reversal=t_rev, t_rebound=t_reb)


def raw_rise(Y: np.ndarray) -> np.ndarray:
    """max_t [E(t) - min_{s<=t} E(s)] on the raw sampled curve (captures transient bumps)."""
    Y = np.asarray(Y, dtype=float)
    return (Y - np.minimum.accumulate(Y, axis=0)).max(axis=0)


def endpoint_stats(times: np.ndarray, Y: np.ndarray) -> dict:
    Y = np.asarray(Y, dtype=float)
    i = np.argmin(Y, axis=0)
    Emin = Y[i, np.arange(Y.shape[1])]
    return dict(t_min=np.asarray(times)[i], E_min=Emin, E_end=Y[-1],
                final_excess=Y[-1] - Emin, E0=Y[0])


def in_closed_positive_cone(deg: np.ndarray) -> np.ndarray:
    r = np.radians(np.asarray(deg, dtype=float))
    return (np.cos(r) >= -CONE_TOL) & (np.sin(r) >= -CONE_TOL)


def antipodal_asymmetry(E_grid: np.ndarray) -> float:
    """max_{t,psi} |E(psi) - E(psi + 180 deg)| on the 720-point grid (0 for linear nets)."""
    K = E_grid.shape[-1]
    return float(np.abs(E_grid[..., : K // 2] - E_grid[..., K // 2:]).max())


def classify(rise, swing, final_excess, tau):
    """Pre-specified labels: monotone / persistent / transient / other."""
    rise = np.asarray(rise); swing = np.asarray(swing); fe = np.asarray(final_excess)
    lab = np.full(rise.shape, "other", dtype=object)
    lab[rise <= MONOTONE_TOL] = "monotone"
    transient = (rise >= tau) & (fe <= RECOVERY_FRACTION * rise)
    persistent = (swing >= tau) & (fe > RECOVERY_FRACTION * rise)
    lab[transient] = "transient"
    lab[persistent] = "persistent"
    return lab
