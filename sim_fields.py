"""
sim_fields.py

Shared deterministic population-field utilities for the two-concept
Gaussian SIM analysis.

Distribution:
    1/2 N((mu1, 0), diag(sigma1^2, sigma2^2))
  + 1/2 N((0, mu2), diag(sigma1^2, sigma2^2)).

For upper-half-plane directions phi in (0, pi), define

    u(phi)      = (cos phi, sin phi),
    u_perp(phi) = (-sin phi, cos phi).

The coarse residual is

    rho_m(x) = -x + m 1{x1 > 0} x1 e1.

Angular field:
    V0(phi; m) = F'_pop(phi) - m cos(phi) J(phi).

Radial field:
    Gamma0(phi; m)
      = 2 F_pop(phi) - m cos(phi) R(phi),

where

    J(phi)
      = E[
          1{x1>0} x1
          1{u^T x>0}
          u_perp^T x
        ],

and

    R(phi)
      = E[
          1{x1>0} x1
          1{u^T x>0}
          u^T x
        ].

All x2 integrals are evaluated analytically. Only a one-dimensional
Gaussian integral over x1 remains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.integrate import quad
from scipy.special import ndtr


SQRT_2PI = np.sqrt(2.0 * np.pi)


# =============================================================================
# Parameters
# =============================================================================

@dataclass(frozen=True)
class SIMParams:
    mu1: float = 3.0
    mu2: float = 2.0
    sigma1: float = 0.15
    sigma2: float = 0.15


@dataclass(frozen=True)
class QuadConfig:
    epsabs: float = 1e-11
    epsrel: float = 1e-10
    limit: int = 250


# =============================================================================
# Gaussian utilities
# =============================================================================

def std_normal_pdf(z):
    z = np.asarray(z)
    return np.exp(-0.5 * z**2) / SQRT_2PI


def gaussian_pdf(x, mean, std):
    z = (x - mean) / std
    return std_normal_pdf(z) / std


def upper_tail_prob(threshold, mean, std):
    """
    P[X > threshold], X ~ N(mean, std^2).
    """
    z = (threshold - mean) / std
    return ndtr(-z)


def upper_truncated_first_moment(threshold, mean, std):
    """
    E[X 1{X > threshold}], X ~ N(mean, std^2).
    """
    z = (threshold - mean) / std
    p = ndtr(-z)

    return (
        mean * p
        + std * std_normal_pdf(z)
    )


# =============================================================================
# F_pop and F'_pop
# =============================================================================

def positive_gaussian_second_moment(mean, variance):
    """
    If Z ~ N(mean, variance), return E[(Z_+)^2].
    """
    std = np.sqrt(variance)
    a = mean / std

    return (
        (mean**2 + variance) * ndtr(a)
        + mean * std * std_normal_pdf(a)
    )


def f_pop(phi, params: SIMParams):
    """
    Population potential

        F_pop(phi) = 1/2 E[(u(phi)^T x)_+^2].

    For the equal two-cluster mixture this is 1/4 times the sum
    of the two cluster second moments.
    """
    phi = np.asarray(phi)

    c = np.cos(phi)
    s = np.sin(phi)

    variance = (
        params.sigma1**2 * c**2
        + params.sigma2**2 * s**2
    )

    m1 = params.mu1 * c
    m2 = params.mu2 * s

    h1 = positive_gaussian_second_moment(
        m1,
        variance,
    )

    h2 = positive_gaussian_second_moment(
        m2,
        variance,
    )

    return 0.25 * (h1 + h2)


def fprime_pop(phi, params: SIMParams):
    """
    Exact derivative F'_pop(phi).
    """
    phi = np.asarray(phi)

    c = np.cos(phi)
    s = np.sin(phi)

    variance = (
        params.sigma1**2 * c**2
        + params.sigma2**2 * s**2
    )

    std = np.sqrt(variance)

    m1 = params.mu1 * c
    m2 = params.mu2 * s

    a1 = m1 / std
    a2 = m2 / std

    variance_prime = (
        params.sigma2**2
        - params.sigma1**2
    ) * np.sin(2.0 * phi)

    term1 = (
        -2.0
        * params.mu1
        * s
        * (
            m1 * ndtr(a1)
            + std * std_normal_pdf(a1)
        )
    )

    term2 = (
        2.0
        * params.mu2
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

    return 0.25 * (
        term1
        + term2
        + term3
    )


# =============================================================================
# Upper-half-plane gated moments
# =============================================================================

def _check_upper_half(phi):
    if not (0.0 < phi < np.pi):
        raise ValueError(
            "This gated quadrature is defined for 0 < phi < pi."
        )


def _cluster_gated_moments(
    phi: float,
    mean_x1: float,
    mean_x2: float,
    params: SIMParams,
    quad_config: QuadConfig,
):
    """
    Compute for one Gaussian cluster:

        J_cluster
          = E[
              1{x1>0} x1
              1{u^T x>0}
              u_perp^T x
            ],

        R_cluster
          = E[
              1{x1>0} x1
              1{u^T x>0}
              u^T x
            ].

    For fixed x1=t>0 and sin(phi)>0,

        u^T x > 0
        iff
        x2 > -(cos(phi)/sin(phi)) t.

    We integrate x2 analytically.
    """
    _check_upper_half(phi)

    c = np.cos(phi)
    s = np.sin(phi)

    sigma1 = params.sigma1
    sigma2 = params.sigma2

    def integrands(t):
        density_x1 = gaussian_pdf(
            t,
            mean=mean_x1,
            std=sigma1,
        )

        threshold_x2 = -(c / s) * t

        p_active = upper_tail_prob(
            threshold_x2,
            mean=mean_x2,
            std=sigma2,
        )

        ex2_active = (
            upper_truncated_first_moment(
                threshold_x2,
                mean=mean_x2,
                std=sigma2,
            )
        )

        # Tangential component:
        #
        #     u_perp^T x = -s x1 + c x2.
        tangent = (
            -s * t * p_active
            + c * ex2_active
        )

        # Radial component:
        #
        #     u^T x = c x1 + s x2.
        radial = (
            c * t * p_active
            + s * ex2_active
        )

        common = t * density_x1

        return (
            common * tangent,
            common * radial,
        )

    def tangent_integrand(t):
        return integrands(t)[0]

    def radial_integrand(t):
        return integrands(t)[1]

    J_value, _ = quad(
        tangent_integrand,
        0.0,
        np.inf,
        epsabs=quad_config.epsabs,
        epsrel=quad_config.epsrel,
        limit=quad_config.limit,
    )

    R_value, _ = quad(
        radial_integrand,
        0.0,
        np.inf,
        epsabs=quad_config.epsabs,
        epsrel=quad_config.epsrel,
        limit=quad_config.limit,
    )

    return float(J_value), float(R_value)


def j_and_r_upper(
    phi: float,
    params: SIMParams,
    quad_config: QuadConfig,
):
    """
    Population J(phi) and R(phi), averaged over both clusters.
    """
    j1, r1 = _cluster_gated_moments(
        phi,
        mean_x1=params.mu1,
        mean_x2=0.0,
        params=params,
        quad_config=quad_config,
    )

    j2, r2 = _cluster_gated_moments(
        phi,
        mean_x1=0.0,
        mean_x2=params.mu2,
        params=params,
        quad_config=quad_config,
    )

    return (
        0.5 * (j1 + j2),
        0.5 * (r1 + r2),
    )


def j_upper(
    phi: float,
    params: SIMParams,
    quad_config: QuadConfig,
):
    J, _ = j_and_r_upper(
        phi,
        params,
        quad_config,
    )

    return J


def r_upper(
    phi: float,
    params: SIMParams,
    quad_config: QuadConfig,
):
    _, R = j_and_r_upper(
        phi,
        params,
        quad_config,
    )

    return R


# =============================================================================
# Coarse angular and radial fields
# =============================================================================

def k_upper(
    phi: float,
    params: SIMParams,
    quad_config: QuadConfig,
):
    """
    K(phi) = cos(phi) J(phi).
    """
    return (
        np.cos(phi)
        * j_upper(
            phi,
            params,
            quad_config,
        )
    )


def v0_upper(
    phi: float,
    m: float,
    params: SIMParams,
    quad_config: QuadConfig,
):
    """
    Coarse angular field

        V0(phi;m)
          = F'_pop(phi) - m cos(phi) J(phi).
    """
    return (
        fprime_pop(phi, params)
        - m
        * k_upper(
            phi,
            params,
            quad_config,
        )
    )


def gamma0_upper(
    phi: float,
    m: float,
    params: SIMParams,
    quad_config: QuadConfig,
):
    """
    Coarse radial field

        Gamma0(phi;m)
          = 2 F_pop(phi)
            - m cos(phi) R(phi).

    At m=0:

        Gamma0(phi;0) = 2 F_pop(phi).
    """
    _, R = j_and_r_upper(
        phi,
        params,
        quad_config,
    )

    return (
        2.0 * f_pop(phi, params)
        - m * np.cos(phi) * R
    )