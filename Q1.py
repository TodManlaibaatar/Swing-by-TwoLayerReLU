"""
Q1.py

Phase-0 analysis on

    Q1 = (0, pi/2).

Goals:
    1. identify target-only critical points;
    2. map roots of V0(phi;m);
    3. detect residual-created saddle-node branches;
    4. classify root stability;
    5. inspect Gamma0 on the same (phi,m) grid.

Unlike Q2, K(phi)=cos(phi)J(phi) does not have a fixed sign in Q1.
Therefore branch creation is possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

from sim_fields import (
    SIMParams,
    QuadConfig,
    f_pop,
    fprime_pop,
    j_and_r_upper,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Q1Config:
    num_phi: int = 1200
    num_m: int = 500

    endpoint_eps: float = 1e-4

    # Replace with the theorem's actual 1 + delta_over.
    m_max: float = 1.10

    k_tol: float = 1e-10


@dataclass
class Q1Result:
    phi: np.ndarray
    m: np.ndarray

    F: np.ndarray
    Fprime: np.ndarray

    J: np.ndarray
    R: np.ndarray
    K: np.ndarray

    m_branch: np.ndarray

    V: np.ndarray
    Gamma: np.ndarray


# =============================================================================
# Core analysis
# =============================================================================

def analyze_q1(
    params: SIMParams,
    quad_config: QuadConfig,
    config: Q1Config,
) -> Q1Result:

    phi = np.linspace(
        config.endpoint_eps,
        np.pi / 2.0 - config.endpoint_eps,
        config.num_phi,
    )

    m_grid = np.linspace(
        0.0,
        config.m_max,
        config.num_m,
    )

    F = f_pop(phi, params)
    Fp = fprime_pop(phi, params)

    J = np.empty_like(phi)
    R = np.empty_like(phi)

    print("Computing deterministic upper-half-plane moments...")

    for i, angle in enumerate(phi):
        J[i], R[i] = j_and_r_upper(
            float(angle),
            params=params,
            quad_config=quad_config,
        )

    c = np.cos(phi)

    K = c * J

    # ---------------------------------------------------------
    # Scalar branch curve:
    #
    #     V0 = F' - m K = 0
    #     => m = F'/K
    #
    # only where K is numerically resolved.
    # ---------------------------------------------------------

    valid_branch = (
        np.isfinite(K)
        & (np.abs(K) > config.k_tol)
    )

    m_branch = np.full_like(
        phi,
        np.nan,
        dtype=float,
    )

    m_branch[valid_branch] = (
        Fp[valid_branch]
        / K[valid_branch]
    )

    # ---------------------------------------------------------
    # Full angular field grid.
    #
    # Shape:
    #     [num_m, num_phi].
    # ---------------------------------------------------------

    V = (
        Fp[None, :]
        - m_grid[:, None] * K[None, :]
    )

    # ---------------------------------------------------------
    # Full radial field grid.
    #
    # Gamma0 = 2F - m cos(phi) R(phi).
    # ---------------------------------------------------------

    Gamma = (
        2.0 * F[None, :]
        - m_grid[:, None]
        * c[None, :]
        * R[None, :]
    )

    result = Q1Result(
        phi=phi,
        m=m_grid,
        F=F,
        Fprime=Fp,
        J=J,
        R=R,
        K=K,
        m_branch=m_branch,
        V=V,
        Gamma=Gamma,
    )

    print_q1_report(
        result=result,
        params=params,
        config=config,
    )

    return result


# =============================================================================
# Root extraction
# =============================================================================

def roots_from_grid(
    phi: np.ndarray,
    values: np.ndarray,
):
    """
    Locate sign-changing roots using linear interpolation.

    This is for plotting / diagnostics.
    Fold locations are refined separately from m_branch.
    """
    roots = []

    for i in range(len(phi) - 1):
        y0 = values[i]
        y1 = values[i + 1]

        if not (
            np.isfinite(y0)
            and np.isfinite(y1)
        ):
            continue

        if y0 == 0.0:
            roots.append(phi[i])
            continue

        if y0 * y1 < 0.0:
            fraction = (
                -y0
                / (y1 - y0)
            )

            root = (
                phi[i]
                + fraction
                * (phi[i + 1] - phi[i])
            )

            roots.append(root)

    return roots


def root_stability_from_grid(
    phi: np.ndarray,
    values: np.ndarray,
    root: float,
):
    """
    Estimate sign of dV/dphi near a root.

    negative -> attracting
    positive -> repelling
    """
    derivative = np.gradient(
        values,
        phi,
    )

    index = np.argmin(
        np.abs(phi - root)
    )

    return derivative[index]


# =============================================================================
# Scalar branch extrema
# =============================================================================

def finite_branch_extrema(result: Q1Result, config: Q1Config):
    """
    Detect local extrema of m_branch on the grid.

    Only returns extrema in the physical m-window.
    """
    phi = result.phi
    branch = result.m_branch

    extrema = []

    for i in range(1, len(phi) - 1):
        if not np.all(
            np.isfinite(
                branch[i - 1:i + 2]
            )
        ):
            continue

        left = branch[i - 1]
        center = branch[i]
        right = branch[i + 1]

        if not (
            0.0
            <= center
            <= config.m_max
        ):
            continue

        if center < left and center < right:
            extrema.append(
                ("min", phi[i], center)
            )

        elif center > left and center > right:
            extrema.append(
                ("max", phi[i], center)
            )

    return extrema


# =============================================================================
# Reporting
# =============================================================================

def print_q1_report(
    result: Q1Result,
    params: SIMParams,
    config: Q1Config,
):
    print("=" * 72)
    print("Q1 ANALYSIS")
    print("=" * 72)

    print(
        f"mu1={params.mu1}, "
        f"mu2={params.mu2}, "
        f"sigma1={params.sigma1}, "
        f"sigma2={params.sigma2}"
    )
    print()

    # ---------------------------------------------------------
    # Target-only roots
    # ---------------------------------------------------------

    target_roots = roots_from_grid(
        result.phi,
        result.Fprime,
    )

    print("Target-only roots F'_pop(phi)=0:")

    for root in target_roots:
        stability = root_stability_from_grid(
            result.phi,
            result.Fprime,
            root,
        )

        kind = (
            "stable"
            if stability < 0
            else "repelling"
        )

        print(
            f"    phi = "
            f"{np.degrees(root):.6f} deg, "
            f"{kind}, "
            f"F'' ≈ {stability:.6f}"
        )

    print()

    # ---------------------------------------------------------
    # J sign
    # ---------------------------------------------------------

    j_roots = roots_from_grid(
        result.phi,
        result.J,
    )

    print("J(phi) sign-change locations:")

    if not j_roots:
        print("    none")
    else:
        for root in j_roots:
            print(
                f"    phi ≈ "
                f"{np.degrees(root):.6f} deg"
            )

    print()

    # ---------------------------------------------------------
    # Scalar branch extrema
    # ---------------------------------------------------------

    extrema = finite_branch_extrema(
        result,
        config,
    )

    print("Physical extrema of m_branch(phi):")

    if not extrema:
        print("    none detected")
    else:
        for kind, angle, level in extrema:
            print(
                f"    {kind}: "
                f"phi ≈ {np.degrees(angle):.6f} deg, "
                f"m ≈ {level:.9f}"
            )

    print()

    # ---------------------------------------------------------
    # Example root sets
    # ---------------------------------------------------------

    probe_ms = [
        0.0,
        0.50,
        0.55,
        0.56,
        0.58,
        0.60,
        0.62,
        0.65,
        0.80,
        1.00,
    ]

    print("Representative V0 roots:")

    for m_value in probe_ms:
        if m_value > config.m_max:
            continue

        index = np.argmin(
            np.abs(result.m - m_value)
        )

        values = result.V[index]

        roots = roots_from_grid(
            result.phi,
            values,
        )

        labels = []

        for root in roots:
            slope = root_stability_from_grid(
                result.phi,
                values,
                root,
            )

            kind = (
                "S"
                if slope < 0
                else "R"
            )

            labels.append(
                f"{np.degrees(root):.3f}° ({kind})"
            )

        print(
            f"    m={result.m[index]:.3f}: "
            + (
                ", ".join(labels)
                if labels
                else "no roots"
            )
        )

    print("=" * 72)


# =============================================================================
# Plotting
# =============================================================================

def plot_q1(result: Q1Result, config: Q1Config):
    phi_deg = np.degrees(result.phi)

    # ---------------------------------------------------------
    # 1. Target-only field
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        phi_deg,
        result.Fprime,
        linewidth=2,
    )

    plt.axhline(0.0, linewidth=1)

    plt.xlabel(r"$\phi$ (degrees)")
    plt.ylabel(r"$F_{\mathrm{pop}}'(\phi)$")
    plt.title(r"$Q_1$: target-only angular field")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 2. J and K
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        phi_deg,
        result.J,
        linewidth=2,
        label=r"$J(\phi)$",
    )

    plt.plot(
        phi_deg,
        result.K,
        linewidth=2,
        label=r"$K(\phi)=\cos(\phi)J(\phi)$",
    )

    plt.axhline(0.0, linewidth=1)

    plt.xlabel(r"$\phi$ (degrees)")
    plt.ylabel("value")
    plt.title(r"$Q_1$: residual coupling")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 3. Scalar branch curve
    # ---------------------------------------------------------

    mask = (
        np.isfinite(result.m_branch)
        & (result.m_branch >= 0.0)
        & (result.m_branch <= config.m_max)
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        phi_deg[mask],
        result.m_branch[mask],
        linewidth=2,
    )

    plt.xlabel(r"$\phi$ (degrees)")
    plt.ylabel(r"$m_{\mathrm{branch}}(\phi)$")
    plt.title(r"$Q_1$: physical scalar branch curve")

    plt.ylim(
        0.0,
        config.m_max,
    )

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 4. V0 heatmap
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.imshow(
        result.V,
        origin="lower",
        aspect="auto",
        extent=[
            phi_deg[0],
            phi_deg[-1],
            result.m[0],
            result.m[-1],
        ],
    )

    plt.colorbar(
        label=r"$V_0(\phi;m)$"
    )

    # zero contour
    plt.contour(
        phi_deg,
        result.m,
        result.V,
        levels=[0.0],
        linewidths=2,
    )

    plt.xlabel(r"$\phi$ (degrees)")
    plt.ylabel(r"$m$")
    plt.title(r"$Q_1$: angular branch diagram")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 5. Gamma0 heatmap
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.imshow(
        result.Gamma,
        origin="lower",
        aspect="auto",
        extent=[
            phi_deg[0],
            phi_deg[-1],
            result.m[0],
            result.m[-1],
        ],
    )

    plt.colorbar(
        label=r"$\Gamma_0(\phi;m)$"
    )

    plt.xlabel(r"$\phi$ (degrees)")
    plt.ylabel(r"$m$")
    plt.title(r"$Q_1$: radial field")

    plt.tight_layout()
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

def main():
    params = SIMParams(
        mu1=3.0,
        mu2=2.0,
        sigma1=0.15,
        sigma2=0.15,
    )

    quad_config = QuadConfig()

    config = Q1Config(
        num_phi=1200,
        num_m=500,
        m_max=1.10,
    )

    result = analyze_q1(
        params=params,
        quad_config=quad_config,
        config=config,
    )

    plot_q1(
        result=result,
        config=config,
    )


if __name__ == "__main__":
    main()