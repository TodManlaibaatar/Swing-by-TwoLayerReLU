"""Batched full-batch gradient flow (forward Euler) for two-layer networks.

All architectures use the same parameterization

    f(x) = sum_i w_i * sigma(u_i^T x),      L = (1 / 2N) sum_n ||f(x_n) - x_n||^2,

with sigma = ReLU ("relu") or the identity ("linear").  The positive control
"tied" is the predecessor's symmetric model f(x) = sum_i u_i (u_i^T x) = U^T U x,
i.e. w_i is tied to u_i.

The ReLU and linear vector fields are exactly those of
``theory_guided_swing_by_phase_diagram.relu_rhs`` (masks from the current
pre-activations, strict inequality), so ReLU runs reproduce the archived
canonical trajectories up to floating-point summation order.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

GRID_DEG = np.linspace(-180.0, 180.0, 720, endpoint=False)  # archived 0.5-degree grid


def unit(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)])


@dataclass
class BatchInputs:
    X: np.ndarray            # (B, N, 2) training inputs (targets are the inputs)
    labels: np.ndarray       # (B, N) cluster labels 0/1
    U0: np.ndarray           # (B, h, 2)
    W0: np.ndarray           # (B, h, 2)
    named_probes: np.ndarray  # (B, K_named, 2) unit vectors


@dataclass
class BatchResult:
    times: np.ndarray        # (L,)
    E_grid: np.ndarray       # (B, L, 720) float32, directional OOD error on GRID_DEG
    E_named: np.ndarray      # (B, L, K_named) float64
    loss_total: np.ndarray   # (B, L) float64, training loss
    loss_c1: np.ndarray      # (B, L) float64, cluster-1 part (normalized by N)
    loss_c2: np.ndarray      # (B, L) float64, cluster-2 part (normalized by N)
    U_hist: Optional[np.ndarray] = None  # (B, L, h, 2) float64
    W_hist: Optional[np.ndarray] = None
    wall_seconds: float = 0.0


def simulate_batch(
    arch: str,
    inputs: BatchInputs,
    dt: float,
    t_end: float,
    log_dt: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    save_params: bool = False,
    progress: Optional[Callable[[str], None]] = print,
) -> BatchResult:
    if arch not in ("relu", "linear", "tied"):
        raise ValueError(f"unknown arch {arch!r}")
    n_steps = int(round(t_end / dt))
    log_every = int(round(log_dt / dt))
    if abs(n_steps * dt - t_end) > 1e-9 or abs(log_every * dt - log_dt) > 1e-12:
        raise ValueError("t_end and log_dt must be integer multiples of dt")
    if n_steps % log_every:
        raise ValueError("t_end must be an integer multiple of log_dt")

    dev = torch.device(device)
    X = torch.as_tensor(inputs.X, dtype=dtype, device=dev)
    U = torch.as_tensor(inputs.U0, dtype=dtype, device=dev).clone()
    W = U if arch == "tied" else torch.as_tensor(inputs.W0, dtype=dtype, device=dev).clone()
    B, N, _ = X.shape
    c1 = torch.as_tensor(inputs.labels == 0, dtype=dtype, device=dev)  # (B, N)
    c2 = 1.0 - c1

    grid = torch.as_tensor(np.stack([unit(d) for d in GRID_DEG]), dtype=dtype, device=dev)
    P = torch.cat([grid.unsqueeze(0).expand(B, -1, -1),
                   torch.as_tensor(inputs.named_probes, dtype=dtype, device=dev)], dim=1)
    K_grid = grid.shape[0]

    L = n_steps // log_every + 1
    times = np.arange(L) * log_dt
    E_grid = np.empty((B, L, K_grid), dtype=np.float32)
    E_named = np.empty((B, L, P.shape[1] - K_grid), dtype=np.float64)
    loss_total = np.empty((B, L)); loss_c1 = np.empty((B, L)); loss_c2 = np.empty((B, L))
    h = U.shape[1]
    U_hist = np.empty((B, L, h, 2)) if save_params else None
    W_hist = np.empty((B, L, h, 2)) if save_params else None

    def act(Z):
        return Z.clamp_min(0.0) if arch == "relu" else Z

    t0 = time.time(); li = 0
    report_every = max(1, n_steps // 20)
    for step in range(n_steps + 1):
        Z = torch.bmm(X, U.transpose(1, 2))            # (B, N, h)
        A = act(Z)
        F = torch.bmm(A, W)                            # (B, N, 2)
        R = F - X
        if step % log_every == 0:
            sq = 0.5 * (R * R).sum(-1)                 # (B, N)
            loss_total[:, li] = (sq.sum(1) / N).cpu().numpy()
            loss_c1[:, li] = ((sq * c1).sum(1) / N).cpu().numpy()
            loss_c2[:, li] = ((sq * c2).sum(1) / N).cpu().numpy()
            Fp = torch.bmm(act(torch.bmm(P, U.transpose(1, 2))), W)
            Ep = 0.5 * ((Fp - P) ** 2).sum(-1)         # (B, K)
            Ep = Ep.cpu().numpy()
            E_grid[:, li] = Ep[:, :K_grid]
            E_named[:, li] = Ep[:, K_grid:]
            if save_params:
                U_hist[:, li] = U.cpu().numpy(); W_hist[:, li] = W.cpu().numpy()
            li += 1
        if step == n_steps:
            break
        dW = -torch.bmm(A.transpose(1, 2), R) / N      # (B, h, 2)
        RW = torch.bmm(R, W.transpose(1, 2))           # (B, N, h)
        if arch == "relu":
            RW = RW * (Z > 0).to(dtype)
        dU = -torch.bmm(RW.transpose(1, 2), X) / N     # (B, h, 2)
        if arch == "tied":
            U = U + dt * (dU + dW)
            W = U
        else:
            U = U + dt * dU
            W = W + dt * dW
        if progress and step and step % report_every == 0:
            el = time.time() - t0
            progress(f"    step {step}/{n_steps}  ({100*step/n_steps:4.0f}%)  "
                     f"elapsed {el/60:6.1f} min  eta {el*(n_steps-step)/step/60:6.1f} min")
    return BatchResult(times, E_grid, E_named, loss_total, loss_c1, loss_c2,
                       U_hist, W_hist, wall_seconds=time.time() - t0)
