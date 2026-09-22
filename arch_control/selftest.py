"""Fast correctness checks (about 1 minute on CPU).  Run before the real experiments.

    python -m arch_control.selftest
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import theory_guided_swing_by_phase_diagram as tg  # noqa: E402
from arch_control.experiments import experiment_specs, make_inputs  # noqa: E402
from arch_control.simulate import BatchInputs, simulate_batch  # noqa: E402
from arch_control.validate import detector_equivalence  # noqa: E402


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def one_step(arch, spec):
    X, lab, U0, W0, named = make_inputs(spec)
    r = simulate_batch(arch, BatchInputs(X[None], lab[None], U0[None], W0[None], named[None]),
                       spec.dt, spec.dt * 1, spec.dt, save_params=True, progress=None)
    return X, U0, W0, r.U_hist[0, 1], r.W_hist[0, 1]


def main():
    ok = True
    ok &= check("detector equals archived persistent_swing", detector_equivalence(150) < 1e-12)

    spec = replace([s for s in experiment_specs("E1_canonical") if s.arch == "relu" and s.seed == 0][0],
                   t_end=2e-4)
    X, U0, W0, U1, W1 = one_step("relu", spec)
    dU, dW, _ = tg.relu_rhs(U0, W0, X)
    ok &= check("ReLU Euler step equals archived relu_rhs",
                np.abs(U1 - (U0 + spec.dt * dU)).max() < 1e-15 and np.abs(W1 - (W0 + spec.dt * dW)).max() < 1e-15)

    X, U0, W0, U1, W1 = one_step("linear", replace(spec, arch="linear"))
    N = len(X); A = X @ U0.T; R = A @ W0 - X
    dW = -(A.T @ R) / N; dU = -((R @ W0.T).T @ X) / N
    ok &= check("linear Euler step equals closed form",
                np.abs(U1 - (U0 + spec.dt * dU)).max() < 1e-15 and np.abs(W1 - (W0 + spec.dt * dW)).max() < 1e-15)

    tspec = [s for s in experiment_specs("E4_positive_control")][0]
    X, U0, _, U1, _ = one_step("tied", tspec)
    Xt = torch.tensor(X); Ut = torch.tensor(U0, requires_grad=True)
    loss = 0.5 * ((Xt @ Ut.T @ Ut - Xt) ** 2).sum(1).mean()
    loss.backward()
    g = Ut.grad.numpy()
    ok &= check("tied step equals autograd gradient descent",
                np.abs(U1 - (U0 - tspec.dt * g)).max() < 1e-14)

    # linear networks are odd: E(psi) = E(psi + 180 deg) exactly
    from arch_control.run import main as run_main
    from arch_control.analyze import main as analyze_main
    with tempfile.TemporaryDirectory() as d:
        rc = run_main(["--smoke", "--out", d, "--device", "cpu"])
        ok &= check("smoke run completes", rc == 0)
        analyze_main(["--smoke", "--out", d])
        rep = (Path(d) / "REPORT.md").read_text()
        ok &= check("smoke analysis writes report", "V2" in rep)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
