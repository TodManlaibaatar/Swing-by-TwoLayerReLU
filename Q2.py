"""
Q2.py

Phase-0 analysis of the coarse weak-neuron angular field in

    Q2 = (pi/2, pi).

We study

    V0(phi; m) = F'_pop(phi) - m K(phi),

where

    K(phi) = cos(phi) J(phi),

    J(phi)
      = E[
          1{x1 > 0} x1
          1{u(phi)^T x > 0}
          u_perp(phi)^T x
        ].

Analytic G0 result:
    For every strict phi in Q2,

        J(phi) < 0,
        K(phi) > 0,
        dV0/dm = -K(phi) < 0.

Hence Q2 equilibria can be represented by the scalar curve

    m_branch(phi) = F'_pop(phi) / K(phi).

The code below:
    1. computes F'_pop exactly;
    2. computes J by deterministic 1D Gaussian quadrature;
    3. computes K and the scalar branch curve;
    4. finds positive branch components and fold candidates;
    5. produces robust Q2 diagnostic plots.

No Monte Carlo is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import ndtr


# =============================================================================
# Configuration
# =============================================================================

SQRT_2PI = np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class SIMParams:
    """
    Two-concept Gaussian SIM parameters.

    Distribution:
        1/2 N((mu1, 0), diag(sigma1^2, sigma2^2))
      + 1/2 N((0, mu2), diag(sigma1^2, sigma2^2)).
    """

    mu1: float = 3.0
    mu2: float = 2.0
    sigma1: float = 0.15
    sigma2: float = 0.15


@dataclass(frozen=True)
class Q2Config:
    """
    Numerical configuration for Q2 analysis.
    """

    num_phi: int = 2000

    # Stay away from the literal endpoints.
    endpoint_eps: float = 1e-4

    # K tends to zero at both endpoints.
    # Do not divide by numerically unresolved K.
    k_division_tol: float = 1e-10

    # Numerical sign tolerance only.
    # Analytically, G0 already gives J < 0 and K > 0.
    sign_tol: float = 1e-12

    quad_epsabs: float = 1e-11
    quad_epsrel: float = 1e-10
    quad_limit: int = 250


@dataclass
class Q2Result:
    phi: np.ndarray
    fprime: np.ndarray
    J: np.ndarray
    K: np.ndarray
    m_branch: np.ndarray
    valid_branch_mask: np.ndarray

    phi_dagger: Optional[float]
    m_dagger: Optional[float]

    local_maxima: list[tuple[float, float]]
    positive_components: list[tuple[int, int]]


# =============================================================================
# Gaussian utilities
# =============================================================================

def std_normal_pdf(z):
    """
    Standard normal density phi(z).
    """
    z = np.asarray(z)
    return np.exp(-0.5 * z**2) / SQRT_2PI


def gaussian_pdf(x, mean, std):
    """
    N(mean, std^2) density evaluated at x.
    """
    z = (x - mean) / std
    return std_normal_pdf(z) / std


def upper_tail_prob(threshold, mean, std):
    """
    P[X > threshold] for X ~ N(mean, std^2).
    """
    z = (threshold - mean) / std
    return ndtr(-z)


def upper_truncated_first_moment(threshold, mean, std):
    """
    E[X 1{X > threshold}] for X ~ N(mean, std^2).

    Identity:

        E[X 1{X>a}]
          = mean * P(X>a)
            + std * phi((a-mean)/std).
    """
    z = (threshold - mean) / std
    p = ndtr(-z)

    return mean * p + std * std_normal_pdf(z)


# =============================================================================
# Exact population field F'_pop
# =============================================================================

def fprime_pop(phi, params: SIMParams):
    r"""
    Exact population angular derivative F'_pop(phi).

    Let

        u(phi) = (cos phi, sin phi),

        v(phi)
          = sigma1^2 cos^2(phi)
            + sigma2^2 sin^2(phi),

        m1 = mu1 cos(phi),
        m2 = mu2 sin(phi).

    Then

        4 F'_pop(phi)
        =
        -2 mu1 sin(phi)
          [m1 Phi(a1) + sqrt(v) phi(a1)]

        +2 mu2 cos(phi)
          [m2 Phi(a2) + sqrt(v) phi(a2)]

        +v'(phi)
          [Phi(a1) + Phi(a2)].
    """
    mu1 = params.mu1
    mu2 = params.mu2
    sigma1 = params.sigma1
    sigma2 = params.sigma2

    phi = np.asarray(phi)

    c = np.cos(phi)
    s = np.sin(phi)

    variance = sigma1**2 * c**2 + sigma2**2 * s**2
    std = np.sqrt(variance)

    m1 = mu1 * c
    m2 = mu2 * s

    a1 = m1 / std
    a2 = m2 / std

    variance_prime = (
        sigma2**2 - sigma1**2
    ) * np.sin(2.0 * phi)

    term1 = (
        -2.0
        * mu1
        * s
        * (
            m1 * ndtr(a1)
            + std * std_normal_pdf(a1)
        )
    )

    term2 = (
        2.0
        * mu2
        * c
        * (
            m2 * ndtr(a2)
            + std * std_normal_pdf(a2)
        )
    )

    term3 = (
        variance_prime
        * (
            ndtr(a1)
            + ndtr(a2)
        )
    )

    return 0.25 * (term1 + term2 + term3)


# =============================================================================
# J(phi)
# =============================================================================

def j_one_cluster(
    phi: float,
    mean_x1: float,
    mean_x2: float,
    params: SIMParams,
    config: Q2Config,
) -> float:
    r"""
    Compute the contribution to J(phi) from one Gaussian cluster.

    Recall

        J(phi)
        =
        E[
            1{x1>0} x1
            1{c x1 + s x2 > 0}
            (-s x1 + c x2)
        ].

    For fixed x1=t>0,

        c t + s x2 > 0

    is equivalent in Q2 to

        x2 > -(c/s)t.

    We integrate x2 analytically using truncated-Gaussian formulas
    and numerically integrate only over t=x1>0.
    """
    if not (np.pi / 2.0 < phi < np.pi):
        raise ValueError(
            "j_one_cluster is defined here only for phi in Q2=(pi/2, pi)."
        )

    c = np.cos(phi)
    s = np.sin(phi)

    sigma1 = params.sigma1
    sigma2 = params.sigma2

    def integrand(t: float) -> float:
        # x1 density
        density_x1 = gaussian_pdf(
            t,
            mean=mean_x1,
            std=sigma1,
        )

        # Gate:
        #
        #     c t + s x2 > 0
        #
        # =>  x2 > -(c/s)t.
        threshold_x2 = -(c / s) * t

        p_active = upper_tail_prob(
            threshold=threshold_x2,
            mean=mean_x2,
            std=sigma2,
        )

        ex2_active = upper_truncated_first_moment(
            threshold=threshold_x2,
            mean=mean_x2,
            std=sigma2,
        )

        # Conditional expectation:
        #
        # E[
        #   (-s t + c x2)
        #   1{x2 > threshold}
        # ].
        conditional_tangent = (
            -s * t * p_active
            + c * ex2_active
        )

        return (
            t
            * density_x1
            * conditional_tangent
        )

    value, _ = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=config.quad_epsabs,
        epsrel=config.quad_epsrel,
        limit=config.quad_limit,
    )

    return float(value)


def j_pop(
    phi: float,
    params: SIMParams,
    config: Q2Config,
) -> float:
    """
    Population J(phi) for the equally weighted two-cluster mixture.
    """

    # Concept-1 cluster:
    #
    #     mean = (mu1, 0).
    j_cluster1 = j_one_cluster(
        phi=phi,
        mean_x1=params.mu1,
        mean_x2=0.0,
        params=params,
        config=config,
    )

    # Concept-2 cluster:
    #
    #     mean = (0, mu2).
    j_cluster2 = j_one_cluster(
        phi=phi,
        mean_x1=0.0,
        mean_x2=params.mu2,
        params=params,
        config=config,
    )

    return 0.5 * (j_cluster1 + j_cluster2)


# =============================================================================
# K(phi) and scalar branch curve
# =============================================================================

def k_pop(
    phi: float,
    params: SIMParams,
    config: Q2Config,
) -> float:
    """
    K(phi) = cos(phi) J(phi).

    Analytically:
        K(phi) > 0 for every strict phi in Q2.
    """
    return (
        np.cos(phi)
        * j_pop(phi, params, config)
    )


def branch_level(
    phi: float,
    params: SIMParams,
    config: Q2Config,
) -> float:
    r"""
    Scalar Q2 branch curve

        m_branch(phi)
          = F'_pop(phi) / K(phi).

    Only use away from numerically unresolved endpoint regions.
    """
    K = k_pop(phi, params, config)

    if K <= config.k_division_tol:
        return np.nan

    return float(
        fprime_pop(phi, params) / K
    )


# =============================================================================
# Branch diagnostics
# =============================================================================

def _connected_true_components(mask: np.ndarray):
    """
    Return inclusive index intervals for connected True components.
    """
    components = []

    start = None

    for i, value in enumerate(mask):
        if value and start is None:
            start = i

        if start is not None:
            at_end = i == len(mask) - 1
            next_false = (
                not at_end
                and not mask[i + 1]
            )

            if at_end or next_false:
                components.append((start, i))
                start = None

    return components


def _grid_local_maxima(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
):
    """
    Return grid indices of local maxima using only valid neighbors.
    """
    indices = []

    for i in range(1, len(y) - 1):
        if not (
            valid[i - 1]
            and valid[i]
            and valid[i + 1]
        ):
            continue

        if (
            y[i] > y[i - 1]
            and y[i] > y[i + 1]
        ):
            indices.append(i)

    return indices


def _refine_local_maximum(
    left: float,
    right: float,
    params: SIMParams,
    config: Q2Config,
):
    """
    Refine a local maximum of m_branch(phi).
    """

    def objective(phi):
        value = branch_level(
            phi,
            params=params,
            config=config,
        )

        if not np.isfinite(value):
            return np.inf

        return -value

    result = minimize_scalar(
        objective,
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1e-11},
    )

    if not result.success:
        return None

    phi_star = float(result.x)
    m_star = -float(result.fun)

    return phi_star, m_star


# =============================================================================
# Main Q2 analysis
# =============================================================================

def analyze_q2(
    params: SIMParams,
    config: Q2Config,
) -> Q2Result:
    """
    Run the complete Q2 scalar-branch analysis.
    """

    phi = np.linspace(
        np.pi / 2.0 + config.endpoint_eps,
        np.pi - config.endpoint_eps,
        config.num_phi,
    )

    # F'_pop is exact and vectorized.
    Fp = fprime_pop(phi, params)

    # J requires 1D quadrature.
    J = np.array(
        [
            j_pop(
                float(angle),
                params=params,
                config=config,
            )
            for angle in phi
        ],
        dtype=float,
    )

    K = np.cos(phi) * J

    # ---------------------------------------------------------
    # Branch curve
    # ---------------------------------------------------------

    # Analytically K>0 on strict Q2.
    #
    # Numerically K may underflow near the endpoints,
    # so only divide where K is resolved.
    valid_branch = (
        np.isfinite(K)
        & (K > config.k_division_tol)
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
    # Local maxima
    # ---------------------------------------------------------

    local_max_indices = _grid_local_maxima(
        phi,
        m_branch,
        valid_branch,
    )

    local_maxima = []

    for i in local_max_indices:
        refined = _refine_local_maximum(
            left=float(phi[i - 1]),
            right=float(phi[i + 1]),
            params=params,
            config=config,
        )

        if refined is not None:
            local_maxima.append(refined)

    local_maxima.sort(
        key=lambda pair: pair[1],
        reverse=True,
    )

    # ---------------------------------------------------------
    # Global maximum m^dagger
    # ---------------------------------------------------------

    if np.any(valid_branch):
        valid_indices = np.where(valid_branch)[0]

        best_valid_position = np.nanargmax(
            m_branch[valid_branch]
        )

        best_grid_index = valid_indices[
            best_valid_position
        ]

        # Prefer refinement if interior to the valid region.
        if (
            best_grid_index > 0
            and best_grid_index < len(phi) - 1
            and valid_branch[best_grid_index - 1]
            and valid_branch[best_grid_index + 1]
        ):
            refined = _refine_local_maximum(
                left=float(phi[best_grid_index - 1]),
                right=float(phi[best_grid_index + 1]),
                params=params,
                config=config,
            )
        else:
            refined = None

        if refined is not None:
            phi_dagger, m_dagger = refined
        else:
            phi_dagger = float(
                phi[best_grid_index]
            )
            m_dagger = float(
                m_branch[best_grid_index]
            )

    else:
        phi_dagger = None
        m_dagger = None

    # ---------------------------------------------------------
    # Positive components
    # ---------------------------------------------------------

    positive_mask = (
        valid_branch
        & (m_branch > 0.0)
    )

    positive_components = (
        _connected_true_components(
            positive_mask
        )
    )

    result = Q2Result(
        phi=phi,
        fprime=Fp,
        J=J,
        K=K,
        m_branch=m_branch,
        valid_branch_mask=valid_branch,
        phi_dagger=phi_dagger,
        m_dagger=m_dagger,
        local_maxima=local_maxima,
        positive_components=positive_components,
    )

    print_q2_report(
        result=result,
        params=params,
        config=config,
    )

    return result


# =============================================================================
# Reporting
# =============================================================================

def print_q2_report(
    result: Q2Result,
    params: SIMParams,
    config: Q2Config,
):
    """
    Print a concise Phase-0 Q2 report.
    """

    J = result.J
    K = result.K
    Fp = result.fprime
    m_branch = result.m_branch

    print("=" * 72)
    print("Q2 ANALYSIS")
    print("=" * 72)

    print(
        f"mu1={params.mu1}, "
        f"mu2={params.mu2}, "
        f"sigma1={params.sigma1}, "
        f"sigma2={params.sigma2}"
    )
    print()

    # ---------------------------------------------------------
    # Analytic facts
    # ---------------------------------------------------------

    print("Analytic G0 facts:")
    print("    J(phi) < 0 for every strict phi in Q2.")
    print("    K(phi) > 0 for every strict phi in Q2.")
    print("    dV0/dm = -K(phi) < 0 in Q2.")
    print()

    # ---------------------------------------------------------
    # Numerical sanity checks
    # ---------------------------------------------------------

    print("Numerical sanity checks:")

    max_positive_J = np.max(
        np.maximum(J, 0.0)
    )

    max_negative_K = np.max(
        np.maximum(-K, 0.0)
    )

    print(
        "    largest numerical violation J > 0: "
        f"{max_positive_J:.3e}"
    )
    print(
        "    largest numerical violation K < 0: "
        f"{max_negative_K:.3e}"
    )
    print(
        "    max F'_pop(phi): "
        f"{np.max(Fp):.10f}"
    )
    print()

    if max_positive_J <= config.sign_tol:
        print(
            "PASS: numerical J is consistent "
            "with analytic J(phi) < 0."
        )
    else:
        print(
            "WARNING: positive numerical J detected."
        )

    if max_negative_K <= config.sign_tol:
        print(
            "PASS: numerical K is consistent "
            "with analytic K(phi) > 0."
        )
    else:
        print(
            "WARNING: negative numerical K detected."
        )

    print()

    # ---------------------------------------------------------
    # F' verdict
    # ---------------------------------------------------------

    if np.max(Fp) < 0.0:
        print("BASE-FIELD VERDICT:")
        print(
            "    F'_pop(phi) < 0 throughout sampled Q2."
        )
        print(
            "    Therefore V0(phi;m) < 0 "
            "for every m >= 0."
        )
        print(
            "    No physical Q2 equilibrium exists."
        )
    else:
        print("BASE-FIELD VERDICT:")
        print(
            "    F'_pop becomes positive somewhere in Q2."
        )
        print(
            "    A physical Q2 branch may exist."
        )

    print()

    # ---------------------------------------------------------
    # m^dagger
    # ---------------------------------------------------------

    if (
        result.phi_dagger is not None
        and result.m_dagger is not None
    ):
        print("Scalar branch maximum:")
        print(
            "    phi^dagger = "
            f"{result.phi_dagger:.10f} rad"
        )
        print(
            "                = "
            f"{np.degrees(result.phi_dagger):.6f} deg"
        )
        print(
            "    m^dagger   = "
            f"{result.m_dagger:.10f}"
        )
        print()

    # ---------------------------------------------------------
    # Local maxima
    # ---------------------------------------------------------

    print("Resolved local maxima of m_branch(phi):")

    if not result.local_maxima:
        print("    none detected")
    else:
        for index, (
            phi_star,
            m_star,
        ) in enumerate(
            result.local_maxima,
            start=1,
        ):
            print(
                f"    #{index}: "
                f"phi={phi_star:.8f} rad "
                f"({np.degrees(phi_star):.4f} deg), "
                f"m={m_star:.8f}"
            )

    print()

    # ---------------------------------------------------------
    # Physical positive components
    # ---------------------------------------------------------

    if not result.positive_components:
        print("PHYSICAL Q2 BRANCH VERDICT:")
        print(
            "    Positive component of "
            "m_branch(phi) is empty."
        )
        print(
            "    No Q2 equilibrium exists for m >= 0."
        )
        print(
            "    m_fold^W does not exist "
            "in the physical m >= 0 regime."
        )
    else:
        print("PHYSICAL Q2 BRANCH VERDICT:")
        print(
            f"    Found "
            f"{len(result.positive_components)} "
            f"positive component(s)."
        )

        for component_index, (
            start,
            end,
        ) in enumerate(
            result.positive_components,
            start=1,
        ):
            phi_start = result.phi[start]
            phi_end = result.phi[end]

            component_mask = np.zeros_like(
                result.valid_branch_mask,
                dtype=bool,
            )
            component_mask[start:end + 1] = True

            local_max_count = sum(
                (
                    phi_start
                    <= phi_star
                    <= phi_end
                )
                for phi_star, _ in result.local_maxima
            )

            print(
                f"    component #{component_index}: "
                f"{np.degrees(phi_start):.3f} deg "
                f"to {np.degrees(phi_end):.3f} deg"
            )

            print(
                f"        resolved local maxima: "
                f"{local_max_count}"
            )

            if local_max_count == 1:
                print(
                    "        numerically consistent "
                    "with unimodality."
                )
            else:
                print(
                    "        unimodality NOT certified."
                )

    print("=" * 72)


# =============================================================================
# Plotting
# =============================================================================

def plot_q2(result: Q2Result):
    """
    Produce the three core Q2 diagnostic plots.
    """

    phi_deg = np.degrees(result.phi)

    # ---------------------------------------------------------
    # F'_pop
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        phi_deg,
        result.fprime,
        linewidth=2,
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.xlabel(r"$\phi$ (degrees)")
    plt.ylabel(r"$F_{\mathrm{pop}}'(\phi)$")
    plt.title(r"$Q_2$: population angular field")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # K
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        phi_deg,
        result.K,
        linewidth=2,
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.xlabel(r"$\phi$ (degrees)")
    plt.ylabel(r"$K(\phi)=\cos(\phi)J(\phi)$")
    plt.title(
        r"$Q_2$: strong-fit suppression coefficient"
    )

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Scalar branch curve: full asymptotic view
    # ---------------------------------------------------------

    mask = (
        result.valid_branch_mask
        & np.isfinite(result.m_branch)
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        phi_deg[mask],
        result.m_branch[mask],
        linewidth=2,
    )

    plt.axhline(0.0, linewidth=1)

    if (
        result.phi_dagger is not None
        and result.m_dagger is not None
    ):
        plt.scatter(
            [np.degrees(result.phi_dagger)],
            [result.m_dagger],
            zorder=5,
        )

    plt.xlabel(r"$\phi$ (degrees)")
    plt.ylabel(r"$\mathfrak{m}(\phi)$")
    plt.title(r"$Q_2$: scalar branch curve — full view")

    plt.tight_layout()
    plt.show()


    # ---------------------------------------------------------
    # Scalar branch curve: physically relevant zoom
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        phi_deg[mask],
        result.m_branch[mask],
        linewidth=2,
    )

    plt.axhline(0.0, linewidth=1)

    if (
        result.phi_dagger is not None
        and result.m_dagger is not None
    ):
        plt.scatter(
            [np.degrees(result.phi_dagger)],
            [result.m_dagger],
            zorder=5,
            label=(
                rf"$m^\dagger={result.m_dagger:.3f}$"
            ),
        )

    # Show the part relevant to m >= 0 and nearby negative maxima.
    plt.ylim(-20.0, 2.0)

    plt.xlabel(r"$\phi$ (degrees)")
    plt.ylabel(r"$\mathfrak{m}(\phi)$")
    plt.title(r"$Q_2$: scalar branch curve — branch-scale view")

    if result.m_dagger is not None:
        plt.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# Optional equal-noise analytic check
# =============================================================================

def print_equal_noise_check(
    params: SIMParams,
    atol: float = 1e-14,
):
    """
    In the equal-noise case sigma1=sigma2,

        v'(phi)=0.

    For phi in Q2 both remaining terms in F'_pop are strictly
    negative, hence

        F'_pop(phi) < 0

    analytically throughout Q2.
    """

    if not np.isclose(
        params.sigma1,
        params.sigma2,
        atol=atol,
        rtol=0.0,
    ):
        return

    print()
    print("EQUAL-NOISE ANALYTIC CHECK:")
    print(
        "    sigma1 = sigma2, so v'(phi)=0."
    )
    print(
        "    Both mean-driven terms in F'_pop "
        "are strictly negative on Q2."
    )
    print(
        "    Therefore F'_pop(phi) < 0 "
        "analytically for all phi in Q2."
    )
    print(
        "    Combined with G0: "
        "V0(phi;m) < 0 for all m >= 0."
    )
    print()


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

    config = Q2Config(
        num_phi=2000,
        endpoint_eps=1e-4,
        k_division_tol=1e-10,
    )

    print_equal_noise_check(params)

    result = analyze_q2(
        params=params,
        config=config,
    )

    plot_q2(result)


if __name__ == "__main__":
    main()