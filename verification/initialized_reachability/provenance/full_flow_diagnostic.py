"""
full_flow_diagnostic.py

Full two-layer ReLU gradient-flow diagnostic for the s=2 SIM.

Purpose:
    Test whether the fixed strong cohort A1 remains angularly locked
    after m11^(1) passes the coarse weak-test-field fold near 0.627.

Tracks:
    - m11^(1)
    - strong cohort angle statistics
    - weak cohort angle statistics
    - strong gate agreement with 1{x1 > 0}
    - full training loss
    - P0-A strong-fit decomposition and without-W field ablation
    - P0-C projected-mass / cumulative radial-growth race
    - P0-B actual W/O trajectories against the moving true weak separatrix

This simulates the actual finite-sample two-layer network, not V0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

# Non-interactive backend: figures are saved to disk and never pop up.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    mu1: float = 3.0
    mu2: float = 2.0
    sigma1: float = 0.15
    sigma2: float = 0.15

    n_per_cluster: int = 2000
    width: int = 200
    epsilon: float = 1e-3

    dt: float = 2e-4
    max_time: float = 6.0
    seed: int = 0

    strong_half_width_deg: float = 20.0

    weak_min_deg: float = 110.0
    weak_max_deg: float = 170.0

    # Finer logging for capture timing.
    log_every: int = 50

    # Compute the true weak field long enough to observe capture.
    weak_field_max_time: float = 3.0

    # Wide grid for tracking the repeller.
    weak_branch_min_deg: float = 5.0

    # But H(t) is still the high-angle creation statistic.
    weak_H_min_deg: float = 60.0

    # Diagnostic capture margin.
    capture_margin_deg: float = 1.0

    figure_dir: str = "phase0_diagnostic_figures"
    save_legacy_figures: bool = False


# =============================================================================
# Data
# =============================================================================

def make_data(cfg: Config, rng: np.random.Generator):
    """
    Equal mixture:

        cluster 1: N((mu1, 0), diag(sigma1^2, sigma2^2))
        cluster 2: N((0, mu2), diag(sigma1^2, sigma2^2)).
    """

    n = cfg.n_per_cluster

    x1 = np.column_stack([
        rng.normal(cfg.mu1, cfg.sigma1, size=n),
        rng.normal(0.0, cfg.sigma2, size=n),
    ])

    x2 = np.column_stack([
        rng.normal(0.0, cfg.sigma1, size=n),
        rng.normal(cfg.mu2, cfg.sigma2, size=n),
    ])

    X = np.concatenate([x1, x2], axis=0)

    return X


# =============================================================================
# Initialization
# =============================================================================

def initialize(cfg: Config, rng: np.random.Generator):
    """
    Isotropic aligned-balanced initialization.

    Each row gets norm epsilon and random angle.
    """

    angles = rng.uniform(
        -np.pi,
        np.pi,
        size=cfg.width,
    )

    U = cfg.epsilon * np.column_stack([
        np.cos(angles),
        np.sin(angles),
    ])

    W = U.copy()

    return U, W, angles


def wrap_angle_deg(angle_rad):
    return np.degrees(
        np.mod(angle_rad, 2.0 * np.pi)
    )


def current_angles(U):
    return np.arctan2(
        U[:, 1],
        U[:, 0],
    )


def make_fixed_cohorts(
    initial_angles,
    cfg: Config,
):
    """
    Define cohorts ONCE from initialization.
    """

    deg = wrap_angle_deg(initial_angles)

    # Strong cohort near 0 degrees.
    strong = (
        (deg <= cfg.strong_half_width_deg)
        |
        (deg >= 360.0 - cfg.strong_half_width_deg)
    )

    weak = (
        (deg >= cfg.weak_min_deg)
        &
        (deg <= cfg.weak_max_deg)
    )

    return strong, weak


# =============================================================================
# Exact finite-sample gradient flow
# =============================================================================

def forward(X, U, W):
    """
    X: [N, 2]
    U: [H, 2]
    W: [H, 2]

    activations:
        Z[n,i] = ReLU(u_i^T x_n)

    output:
        F[n,:] = sum_i Z[n,i] w_i
    """

    pre = X @ U.T                # [N, H]
    Z = np.maximum(pre, 0.0)     # [N, H]

    F = Z @ W                    # [N, 2]

    return F, pre, Z


def gradients(X, U, W):
    """
    Exact empirical gradient-flow vector field for

        L = (1/(2N)) sum_n ||f(x_n)-x_n||^2.

    Returns dU/dt, dW/dt = negative gradients.
    """

    N = X.shape[0]

    F, pre, Z = forward(
        X,
        U,
        W,
    )

    residual = F - X            # [N, 2]

    # ---------------------------------------------------------
    # w_i dynamics:
    #
    #   dot w_i
    #     = -E[(f(x)-x) ReLU(u_i^T x)].
    # ---------------------------------------------------------

    dW = -(Z.T @ residual) / N

    # ---------------------------------------------------------
    # u_i dynamics:
    #
    #   dot u_i
    #     =
    #     -E[
    #       <f(x)-x, w_i>
    #       1{u_i^T x>0}
    #       x
    #     ].
    # ---------------------------------------------------------

    residual_dot_w = residual @ W.T    # [N, H]

    gates = (pre > 0.0).astype(float)

    coeff = residual_dot_w * gates     # [N, H]

    dU = -(coeff.T @ X) / N            # [H, 2]

    loss = 0.5 * np.mean(
        np.sum(residual**2, axis=1)
    )

    return dU, dW, loss


# =============================================================================
# Diagnostics
# =============================================================================
def cohort_output(X, U, W, mask):
    """
    Exact output contributed by a fixed cohort:

        f_C(x) = sum_{i in C} w_i ReLU(u_i^T x).

    Returns an array of shape [N, 2].
    """
    if not np.any(mask):
        return np.zeros(
            (X.shape[0], W.shape[1]),
            dtype=float,
        )

    pre = X @ U[mask].T
    Z = np.maximum(pre, 0.0)
    return Z @ W[mask]


def effective_fit_from_output(X, F):
    """
    Least-squares coefficient of an arbitrary output F along

        q(x) = 1{x1 > 0} x1 e1.

    The map F -> m_eff(F) is linear, so for a fixed partition
    A1 / W / O we should have, up to floating-point error,

        m_eff = m_eff^A1 + m_eff^W + m_eff^O.
    """
    gate = X[:, 0] > 0.0
    x1 = X[gate, 0]
    f1 = F[gate, 0]

    denominator = np.sum(x1**2)

    if denominator == 0.0:
        return np.nan

    return float(
        np.sum(x1 * f1) / denominator
    )


def effective_strong_fit(X, U, W):
    """
    Least-squares coefficient m_eff in

        f_1(x) ~ m_eff * 1{x1 > 0} * x1.

    This is the quantity most directly comparable to the m parameter
    used in the coarse weak-neuron field V0(phi; m).
    """
    F, _, _ = forward(X, U, W)
    return effective_fit_from_output(X, F)


def projected_cohort_mass(U, W, mask):
    """
    Projected cohort mass

        L_C = sum_{i in C} (||P_S u_i||^2 + ||P_S w_i||^2).

    This diagnostic is run directly in the 2D concept subspace, so
    P_S is the identity here.
    """
    if not np.any(mask):
        return 0.0

    return float(
        np.sum(U[mask] ** 2)
        + np.sum(W[mask] ** 2)
    )

def coarse_model_relative_error(X, U, W, m_eff):
    """
    Relative L2 error of the rank-one coarse model

        f_coarse(x)
          = m_eff * 1{x1 > 0} * x1 * e1.
    """
    F, _, _ = forward(X, U, W)

    coarse = np.zeros_like(F)

    gate = X[:, 0] > 0.0

    coarse[gate, 0] = (
        m_eff * X[gate, 0]
    )

    numerator = np.mean(
        np.sum(
            (F - coarse) ** 2,
            axis=1,
        )
    )

    denominator = np.mean(
        np.sum(
            F**2,
            axis=1,
        )
    )

    if denominator == 0.0:
        return np.nan

    return np.sqrt(
        numerator / denominator
    )


def strong_angular_decomposition(
    X,
    U,
    W,
    dU,
    m_eff,
    strong_mask,
):
    """
    Decompose the angular velocity of the fixed strong cohort A1.

    For each strong neuron, compare:

      actual
          exact angular velocity from the full gradient flow;

      true_actual_w
          exact residual + actual normalized w direction
          (should numerically equal actual);

      true_u_closure
          exact residual, but force w_hat = u_hat;

      coarse_actual_w
          coarse rank-one residual, but retain actual w_hat;

      coarse_u_closure
          coarse residual AND force w_hat = u_hat.
          This is the V0-style weak-neuron closure.

    This lets us distinguish:

      residual approximation error
          true_actual_w  vs coarse_actual_w

      w/u directional-closure error
          coarse_actual_w vs coarse_u_closure.
    """

    indices = np.where(strong_mask)[0]

    if len(indices) == 0:
        return None

    Us = U[indices]
    Ws = W[indices]
    dUs = dU[indices]

    # ---------------------------------------------------------
    # Norms and directions
    # ---------------------------------------------------------

    r = np.linalg.norm(
        Us,
        axis=1,
    )

    s = np.linalg.norm(
        Ws,
        axis=1,
    )

    u_hat = Us / r[:, None]
    w_hat = Ws / s[:, None]

    u_perp = np.column_stack([
        -u_hat[:, 1],
        u_hat[:, 0],
    ])

    # ---------------------------------------------------------
    # True network residual
    # ---------------------------------------------------------

    F, pre, _ = forward(
        X,
        U,
        W,
    )

    true_residual = F - X

    # ---------------------------------------------------------
    # Coarse residual:
    #
    # rho_m(x)
    #   = -x
    #     + m_eff 1{x1>0} x1 e1.
    # ---------------------------------------------------------

    coarse_residual = -X.copy()

    coordinate_gate = X[:, 0] > 0.0

    coarse_residual[
        coordinate_gate,
        0,
    ] += (
        m_eff
        * X[coordinate_gate, 0]
    )

    # ---------------------------------------------------------
    # Actual neuron gates
    # ---------------------------------------------------------

    strong_pre = pre[:, indices]

    gates = (
        strong_pre > 0.0
    ).astype(float)

    # u_perp^T x for every sample / strong neuron.
    tangent = X @ u_perp.T

    # ---------------------------------------------------------
    # Exact angular velocity directly from dU
    # ---------------------------------------------------------

    actual = (
        np.sum(
            u_perp * dUs,
            axis=1,
        )
        / r
    )

    # Common scale factor s_i / r_i.
    scale = s / r

    # ---------------------------------------------------------
    # Exact residual + actual w_hat
    #
    # This should reproduce `actual`.
    # ---------------------------------------------------------

    true_dot_w = (
        true_residual @ w_hat.T
    )

    field_true_actual_w = -np.mean(
        true_dot_w
        * gates
        * tangent,
        axis=0,
    )

    true_actual_w = (
        scale
        * field_true_actual_w
    )

    # ---------------------------------------------------------
    # Exact residual + forced w_hat = u_hat
    # ---------------------------------------------------------

    true_dot_u = (
        true_residual @ u_hat.T
    )

    field_true_u = -np.mean(
        true_dot_u
        * gates
        * tangent,
        axis=0,
    )

    true_u_closure = (
        scale
        * field_true_u
    )

    # ---------------------------------------------------------
    # Coarse residual + actual w_hat
    # ---------------------------------------------------------

    coarse_dot_w = (
        coarse_residual @ w_hat.T
    )

    field_coarse_actual_w = -np.mean(
        coarse_dot_w
        * gates
        * tangent,
        axis=0,
    )

    coarse_actual_w = (
        scale
        * field_coarse_actual_w
    )

    # ---------------------------------------------------------
    # Coarse residual + forced w_hat = u_hat
    #
    # This is the weak-field V0 closure.
    # ---------------------------------------------------------

    coarse_dot_u = (
        coarse_residual @ u_hat.T
    )

    field_coarse_u = -np.mean(
        coarse_dot_u
        * gates
        * tangent,
        axis=0,
    )

    coarse_u_closure = (
        scale
        * field_coarse_u
    )

    # ---------------------------------------------------------
    # w/u directional mismatch
    # ---------------------------------------------------------

    theta_u = np.arctan2(
        Us[:, 1],
        Us[:, 0],
    )

    theta_w = np.arctan2(
        Ws[:, 1],
        Ws[:, 0],
    )

    angle_difference = (
        theta_w
        - theta_u
        + np.pi
    ) % (2.0 * np.pi) - np.pi

    return {
        "actual": actual,
        "true_actual_w": true_actual_w,
        "true_u_closure": true_u_closure,
        "coarse_actual_w": coarse_actual_w,
        "coarse_u_closure": coarse_u_closure,
        "wu_angle_diff_rad": angle_difference,

        # Numerical identity check:
        "exact_reconstruction_error": (
            actual
            - true_actual_w
        ),
    }


def summarize_values(values):
    """
    Return median, 10th percentile, 90th percentile.
    """
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.10)),
        float(np.quantile(values, 0.90)),
    )

def weak_test_field(
    X,
    U,
    W,
    m_eff,
    phi_grid,
    weak_mask,
):
    """
    Angular field seen by a hypothetical infinitesimal test neuron

        w_hat = u_hat = (cos(phi), sin(phi)).

    Returns:

        V_true(phi)
            field generated by the actual network residual;

        V_coarse(phi)
            field generated by the rank-one coarse residual;

        V_without_W(phi)
            field generated by the SAME trained checkpoint after
            subtracting the fixed W cohort output from the residual.

    The last quantity is an instantaneous residual ablation, not a
    retraining intervention. It directly tests whether W is materially
    creating its own weak basin.
    """

    c = np.cos(phi_grid)
    s = np.sin(phi_grid)

    u_hat = np.column_stack([
        c,
        s,
    ])

    u_perp = np.column_stack([
        -s,
        c,
    ])

    # Full output and exact residual.
    F, _, _ = forward(
        X,
        U,
        W,
    )

    true_residual = F - X

    # Fixed-W output and residual with W removed at this checkpoint.
    F_W = cohort_output(
        X,
        U,
        W,
        weak_mask,
    )

    without_W_residual = (
        F
        - F_W
        - X
    )

    # Rank-one coarse residual.
    coarse_residual = -X.copy()

    coordinate_gate = (
        X[:, 0] > 0.0
    )

    coarse_residual[
        coordinate_gate,
        0,
    ] += (
        m_eff
        * X[coordinate_gate, 0]
    )

    # Test-neuron gate and tangent coordinate.
    pre = X @ u_hat.T

    gates = (
        pre > 0.0
    ).astype(float)

    tangent = X @ u_perp.T

    # Residual projections onto test output direction.
    true_radial = true_residual @ u_hat.T
    coarse_radial = coarse_residual @ u_hat.T
    without_W_radial = without_W_residual @ u_hat.T

    V_true = -np.mean(
        true_radial
        * gates
        * tangent,
        axis=0,
    )

    V_coarse = -np.mean(
        coarse_radial
        * gates
        * tangent,
        axis=0,
    )

    V_without_W = -np.mean(
        without_W_radial
        * gates
        * tangent,
        axis=0,
    )

    return (
        V_true,
        V_coarse,
        V_without_W,
    )

def roots_on_grid(phi, values):
    """
    Find sign-changing roots of a 1D field by linear interpolation.
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
            roots.append(
                float(phi[i])
            )

        elif y0 * y1 < 0.0:
            alpha = (
                -y0
                / (y1 - y0)
            )

            root = (
                phi[i]
                + alpha
                * (
                    phi[i + 1]
                    - phi[i]
                )
            )

            roots.append(
                float(root)
            )

    return roots

def classify_roots(
    phi,
    values,
    roots,
):
    """
    Classify roots of dot(phi)=V(phi).

    negative derivative -> stable
    positive derivative -> repelling
    """

    derivative = np.gradient(
        values,
        phi,
    )

    output = []

    for root in roots:
        index = np.argmin(
            np.abs(phi - root)
        )

        slope = derivative[index]

        kind = (
            "S"
            if slope < 0.0
            else "R"
        )

        output.append(
            (
                root,
                kind,
                float(slope),
            )
        )

    return output


def block_operator(U, W, mask):
    """
    Full 2x2 operator sum over a fixed cohort:

        M = sum_i w_i u_i^T.
    """

    Uc = U[mask]
    Wc = W[mask]

    return Wc.T @ Uc


def strong_gate_agreement(
    X,
    U,
    strong_mask,
):
    """
    Fraction of sample-neuron pairs for which the current strong gate
    agrees with the coordinate gate 1{x1 > 0}.
    """

    Us = U[strong_mask]

    if len(Us) == 0:
        return np.nan

    actual = (X @ Us.T) > 0.0

    target = (
        X[:, 0:1] > 0.0
    )

    return np.mean(
        actual == target
    )


def angular_summary(U, mask):
    if not np.any(mask):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    deg = wrap_angle_deg(
        current_angles(U[mask])
    )

    # For strong cohort, map angles close to 360 back to negative.
    deg_signed = (
        (deg + 180.0) % 360.0
    ) - 180.0

    return (
        np.median(deg_signed),
        np.quantile(deg_signed, 0.1),
        np.quantile(deg_signed, 0.9),
    )

def signed_angles_deg(U):
    """
    Projected neuron angles in (-180, 180].
    """
    return np.degrees(
        np.arctan2(
            U[:, 1],
            U[:, 0],
        )
    )


def individual_projected_mass(U, W, mask):
    """
    Per-neuron projected mass

        ell_i = ||u_i||^2 + ||w_i||^2

    in the 2D concept subspace.
    """
    return (
        np.sum(U[mask] ** 2, axis=1)
        + np.sum(W[mask] ** 2, axis=1)
    )


def select_weak_branch(
    phi_grid,
    values,
):
    """
    Identify the weak stable branch and the immediately preceding
    repelling branch.

    We select:
      - the highest-angle stable root;
      - the highest-angle repelling root below it.

    This avoids confusing the weak pair with any low-angle strong root.

    Returns degrees. NaN if a branch is absent.
    """

    roots = roots_on_grid(
        phi_grid,
        values,
    )

    info = classify_roots(
        phi_grid,
        values,
        roots,
    )

    stable = [
        (root, slope)
        for root, kind, slope in info
        if kind == "S"
    ]

    if not stable:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    stable_root, stable_slope = max(
        stable,
        key=lambda item: item[0],
    )

    repelling = [
        (root, slope)
        for root, kind, slope in info
        if (
            kind == "R"
            and root < stable_root
        )
    ]

    if repelling:
        rep_root, rep_slope = max(
            repelling,
            key=lambda item: item[0],
        )

        rep_deg = float(
            np.degrees(rep_root)
        )

        rep_slope = float(
            rep_slope
        )
    else:
        rep_deg = np.nan
        rep_slope = np.nan

    stable_deg = float(
        np.degrees(stable_root)
    )

    return (
        rep_deg,
        stable_deg,
        rep_slope,
        float(stable_slope),
    )


# =============================================================================
# Simulation
# =============================================================================

def run(cfg: Config):
    rng = np.random.default_rng(
        cfg.seed
    )

    X = make_data(
        cfg,
        rng,
    )

    U, W, initial_angles = initialize(
        cfg,
        rng,
    )

    strong_mask, weak_mask = make_fixed_cohorts(
        initial_angles,
        cfg,
    )

    other_mask = ~(
        strong_mask
        | weak_mask
    )

    # Initial masses used for cumulative radial-growth diagnostics.
    L_A1_0 = projected_cohort_mass(
        U,
        W,
        strong_mask,
    )

    L_W_0 = projected_cohort_mass(
        U,
        W,
        weak_mask,
    )

    L_O_0 = projected_cohort_mass(
        U,
        W,
        other_mask,
    )

    # Weak-field grid. 20 degrees is low enough to retain the moving
    # repeller around the creation window while avoiding the immediate
    # strong-axis neighborhood.
    weak_test_phi = np.linspace(
        np.radians(
            cfg.weak_branch_min_deg
        ),
        np.pi / 2.0 - 1e-4,
        1500,
    )

    print(
        f"fixed A1 size: {strong_mask.sum()} / {cfg.width}"
    )
    print(
        f"fixed W size:  {weak_mask.sum()} / {cfg.width}"
    )
    print(
        f"fixed O size:  {other_mask.sum()} / {cfg.width}"
    )

    num_steps = int(
        cfg.max_time / cfg.dt
    )

    history = {
        # Existing full-flow diagnostics.
        "time": [],
        "loss": [],
        "m_eff": [],
        "coarse_error": [],
        "m11": [],
        "m12": [],
        "m21": [],
        "m22": [],
        "strong_median_deg": [],
        "strong_q10_deg": [],
        "strong_q90_deg": [],
        "weak_median_deg": [],
        "gate_agreement": [],
        "phi_dot_actual": [],
        "phi_dot_true_u": [],
        "phi_dot_coarse_w": [],
        "phi_dot_coarse_u": [],
        "wu_angle_diff_deg": [],
        "angular_identity_error": [],

        # P0-A: exact m_eff decomposition under fixed cohorts.
        "m_eff_A1": [],
        "m_eff_W": [],
        "m_eff_O": [],
        "m_eff_additivity_error": [],

        # P0-C: projected masses and integrated growth.
        "L_A1": [],
        "L_W": [],
        "L_O": [],
        "G_A1": [],
        "G_W": [],
        "G_O": [],

        # Weak-field diagnostics through cfg.weak_field_max_time.
        "weak_field_time": [],
        "weak_field_m_eff": [],
        "weak_H_true": [],
        "weak_H_coarse": [],
        "weak_H_without_W": [],
        "weak_phi_max_true_deg": [],
        "weak_phi_max_coarse_deg": [],
        "weak_phi_max_without_W_deg": [],
        "weak_left_true": [],
        "weak_right_true": [],
        "weak_num_roots_true": [],
        "weak_num_roots_coarse": [],
        "weak_num_roots_without_W": [],
        "weak_remainder_sup": [],
        "weak_W_influence_sup": [],
        "weak_V_true": [],
        "weak_V_coarse": [],
        "weak_V_without_W": [],

        # P0-B: actual cohort trajectories and moving true weak branch.
        "W_angles_deg": [],
        "O_angles_deg": [],
        "W_individual_mass": [],
        "O_individual_mass": [],
        "weak_repeller_deg": [],
        "weak_stable_deg": [],
        "weak_repeller_slope": [],
        "weak_stable_slope": [],
    }

    crossed_0555 = False
    crossed_0627 = False

    phase0_snapshot_times = [
        0.90,
        1.00,
        1.02,
        1.04,
        1.06,
        1.08,
        1.10,
        1.12,
        1.14,
        1.16,
    ]

    for step in range(num_steps + 1):
        t = step * cfg.dt

        if step % cfg.log_every == 0:
            dU_log, _, loss = gradients(
                X,
                U,
                W,
            )

            M1 = block_operator(
                U,
                W,
                strong_mask,
            )

            # -------------------------------------------------
            # P0-A: exact strong-fit decomposition.
            # -------------------------------------------------

            F_full, _, _ = forward(
                X,
                U,
                W,
            )

            F_A1 = cohort_output(
                X,
                U,
                W,
                strong_mask,
            )

            F_W = cohort_output(
                X,
                U,
                W,
                weak_mask,
            )

            F_O = cohort_output(
                X,
                U,
                W,
                other_mask,
            )

            m_eff = effective_fit_from_output(
                X,
                F_full,
            )

            m_eff_A1 = effective_fit_from_output(
                X,
                F_A1,
            )

            m_eff_W = effective_fit_from_output(
                X,
                F_W,
            )

            m_eff_O = effective_fit_from_output(
                X,
                F_O,
            )

            additivity_error = abs(
                m_eff
                - m_eff_A1
                - m_eff_W
                - m_eff_O
            )

            # -------------------------------------------------
            # P0-C: projected cohort mass and cumulative growth.
            # -------------------------------------------------

            L_A1 = projected_cohort_mass(
                U,
                W,
                strong_mask,
            )

            L_W = projected_cohort_mass(
                U,
                W,
                weak_mask,
            )

            L_O = projected_cohort_mass(
                U,
                W,
                other_mask,
            )

            G_A1 = 0.5 * np.log(
                L_A1 / L_A1_0
            )

            G_W = 0.5 * np.log(
                L_W / L_W_0
            )

            G_O = 0.5 * np.log(
                L_O / L_O_0
            )

            coarse_error = coarse_model_relative_error(
                X,
                U,
                W,
                m_eff,
            )

            # -------------------------------------------------
            # Existing strong-neuron angular decomposition.
            # -------------------------------------------------

            angular_diag = strong_angular_decomposition(
                X=X,
                U=U,
                W=W,
                dU=dU_log,
                m_eff=m_eff,
                strong_mask=strong_mask,
            )

            strong_med, strong_q10, strong_q90 = (
                angular_summary(
                    U,
                    strong_mask,
                )
            )

            weak_med, _, _ = angular_summary(
                U,
                weak_mask,
            )

            agreement = strong_gate_agreement(
                X,
                U,
                strong_mask,
            )

            actual_med, _, _ = summarize_values(
                angular_diag["actual"]
            )

            true_u_med, _, _ = summarize_values(
                angular_diag["true_u_closure"]
            )

            coarse_w_med, _, _ = summarize_values(
                angular_diag["coarse_actual_w"]
            )

            coarse_u_med, _, _ = summarize_values(
                angular_diag["coarse_u_closure"]
            )

            wu_diff_deg = np.degrees(
                angular_diag["wu_angle_diff_rad"]
            )

            wu_diff_med, _, _ = summarize_values(
                wu_diff_deg
            )

            identity_error = np.max(
                np.abs(
                    angular_diag[
                        "exact_reconstruction_error"
                    ]
                )
            )

            # -------------------------------------------------
            # P0-B: save actual fixed-cohort trajectories.
            # These are logged at every ordinary checkpoint.
            # -------------------------------------------------

            history["W_angles_deg"].append(
                signed_angles_deg(
                    U[weak_mask]
                ).copy()
            )

            history["O_angles_deg"].append(
                signed_angles_deg(
                    U[other_mask]
                ).copy()
            )

            history["W_individual_mass"].append(
                individual_projected_mass(
                    U,
                    W,
                    weak_mask,
                ).copy()
            )

            history["O_individual_mass"].append(
                individual_projected_mass(
                    U,
                    W,
                    other_mask,
                ).copy()
            )

            # -------------------------------------------------
            # Direct weak-field self-consistency ablation
            # + P0-B moving separatrix.
            # -------------------------------------------------

            if t <= cfg.weak_field_max_time:
                (
                    V_true_weak,
                    V_coarse_weak,
                    V_without_W_weak,
                ) = weak_test_field(
                    X=X,
                    U=U,
                    W=W,
                    m_eff=m_eff,
                    phi_grid=weak_test_phi,
                    weak_mask=weak_mask,
                )

                # H(t) is the high-angle creation statistic:
                # max over [weak_H_min_deg, 90 deg].  The full wider
                # grid is still used below to locate the repeller.
                phi_deg_grid = np.degrees(
                    weak_test_phi
                )

                high_angle_mask = (
                    phi_deg_grid
                    >= cfg.weak_H_min_deg
                )

                high_indices = np.where(
                    high_angle_mask
                )[0]

                true_max_index = high_indices[
                    np.argmax(
                        V_true_weak[
                            high_angle_mask
                        ]
                    )
                ]

                coarse_max_index = high_indices[
                    np.argmax(
                        V_coarse_weak[
                            high_angle_mask
                        ]
                    )
                ]

                without_W_max_index = high_indices[
                    np.argmax(
                        V_without_W_weak[
                            high_angle_mask
                        ]
                    )
                ]

                H_true = float(
                    V_true_weak[true_max_index]
                )

                H_coarse = float(
                    V_coarse_weak[coarse_max_index]
                )

                H_without_W = float(
                    V_without_W_weak[
                        without_W_max_index
                    ]
                )

                phi_max_true_deg = float(
                    np.degrees(
                        weak_test_phi[
                            true_max_index
                        ]
                    )
                )

                phi_max_coarse_deg = float(
                    np.degrees(
                        weak_test_phi[
                            coarse_max_index
                        ]
                    )
                )

                phi_max_without_W_deg = float(
                    np.degrees(
                        weak_test_phi[
                            without_W_max_index
                        ]
                    )
                )

                left_true = float(
                    V_true_weak[0]
                )

                right_true = float(
                    V_true_weak[-1]
                )

                true_roots = roots_on_grid(
                    weak_test_phi,
                    V_true_weak,
                )

                coarse_roots = roots_on_grid(
                    weak_test_phi,
                    V_coarse_weak,
                )

                without_W_roots = roots_on_grid(
                    weak_test_phi,
                    V_without_W_weak,
                )

                true_root_info = classify_roots(
                    weak_test_phi,
                    V_true_weak,
                    true_roots,
                )

                coarse_root_info = classify_roots(
                    weak_test_phi,
                    V_coarse_weak,
                    coarse_roots,
                )

                without_W_root_info = classify_roots(
                    weak_test_phi,
                    V_without_W_weak,
                    without_W_roots,
                )

                (
                    weak_repeller_deg,
                    weak_stable_deg,
                    weak_repeller_slope,
                    weak_stable_slope,
                ) = select_weak_branch(
                    weak_test_phi,
                    V_true_weak,
                )

                remainder_sup = float(
                    np.max(
                        np.abs(
                            V_true_weak
                            - V_coarse_weak
                        )
                    )
                )

                W_influence_sup = float(
                    np.max(
                        np.abs(
                            V_true_weak
                            - V_without_W_weak
                        )
                    )
                )

                history["weak_field_time"].append(t)
                history["weak_field_m_eff"].append(m_eff)
                history["weak_H_true"].append(H_true)
                history["weak_H_coarse"].append(H_coarse)
                history["weak_H_without_W"].append(H_without_W)
                history["weak_phi_max_true_deg"].append(
                    phi_max_true_deg
                )
                history["weak_phi_max_coarse_deg"].append(
                    phi_max_coarse_deg
                )
                history["weak_phi_max_without_W_deg"].append(
                    phi_max_without_W_deg
                )
                history["weak_left_true"].append(left_true)
                history["weak_right_true"].append(right_true)
                history["weak_num_roots_true"].append(
                    len(true_roots)
                )
                history["weak_num_roots_coarse"].append(
                    len(coarse_roots)
                )
                history["weak_num_roots_without_W"].append(
                    len(without_W_roots)
                )
                history["weak_remainder_sup"].append(
                    remainder_sup
                )
                history["weak_W_influence_sup"].append(
                    W_influence_sup
                )
                history["weak_V_true"].append(
                    V_true_weak.copy()
                )
                history["weak_V_coarse"].append(
                    V_coarse_weak.copy()
                )
                history["weak_V_without_W"].append(
                    V_without_W_weak.copy()
                )
                history["weak_repeller_deg"].append(
                    weak_repeller_deg
                )
                history["weak_stable_deg"].append(
                    weak_stable_deg
                )
                history["weak_repeller_slope"].append(
                    weak_repeller_slope
                )
                history["weak_stable_slope"].append(
                    weak_stable_slope
                )

                # ---------------------------------------------
                # P0-A/C snapshot report around creation.
                # ---------------------------------------------

                snapshot_tolerance = (
                    0.5
                    * cfg.dt
                    * cfg.log_every
                    + 1e-12
                )

                if any(
                    abs(t - target)
                    < snapshot_tolerance
                    for target in phase0_snapshot_times
                ):
                    nonweak_fit = (
                        m_eff_A1
                        + m_eff_O
                    )

                    weak_to_nonweak = (
                        abs(m_eff_W)
                        / max(
                            abs(nonweak_fit),
                            1e-15,
                        )
                    )

                    print()
                    print("=" * 72)
                    print(
                        "PHASE-0 SNAPSHOT: "
                        f"t={t:.3f}, "
                        f"m_eff={m_eff:.6f}"
                    )
                    print("=" * 72)
                    print(
                        "m_eff decomposition: "
                        f"A1={m_eff_A1:+.6e}, "
                        f"W={m_eff_W:+.6e}, "
                        f"O={m_eff_O:+.6e}"
                    )
                    print(
                        "additivity error = "
                        f"{additivity_error:.3e}"
                    )
                    print(
                        "|m_eff^W| / |m_eff^A1 + m_eff^O| = "
                        f"{weak_to_nonweak:.6e}"
                    )
                    print(
                        "projected masses: "
                        f"L_A1={L_A1:.6e}, "
                        f"L_W={L_W:.6e}, "
                        f"L_O={L_O:.6e}"
                    )
                    print(
                        "cumulative growth: "
                        f"G_A1={G_A1:.6f}, "
                        f"G_W={G_W:.6f}, "
                        f"G_O={G_O:.6f}"
                    )
                    print(
                        f"H_true      = {H_true:+.6e} "
                        f"at {phi_max_true_deg:.3f} deg"
                    )
                    print(
                        f"H_without_W = {H_without_W:+.6e} "
                        f"at {phi_max_without_W_deg:.3f} deg"
                    )
                    print(
                        f"H_coarse    = {H_coarse:+.6e} "
                        f"at {phi_max_coarse_deg:.3f} deg"
                    )
                    print(
                        "sup |V_true - V_without_W| = "
                        f"{W_influence_sup:.6e}"
                    )
                    print(
                        "sup |V_true - V_coarse| = "
                        f"{remainder_sup:.6e}"
                    )

                    def _print_roots(label, root_info):
                        if not root_info:
                            print(f"{label} roots: none")
                            return

                        print(f"{label} roots:")
                        for root, kind, slope in root_info:
                            print(
                                "    "
                                f"{np.degrees(root):.3f} deg "
                                f"({kind}), "
                                f"slope={slope:+.6e}"
                            )

                    _print_roots(
                        "true",
                        true_root_info,
                    )
                    _print_roots(
                        "without-W",
                        without_W_root_info,
                    )
                    _print_roots(
                        "coarse",
                        coarse_root_info,
                    )

            # -------------------------------------------------
            # Save time-series history.
            # -------------------------------------------------

            m11 = M1[0, 0]

            history["time"].append(t)
            history["loss"].append(loss)
            history["m_eff"].append(m_eff)
            history["coarse_error"].append(coarse_error)
            history["m11"].append(m11)
            history["m12"].append(M1[0, 1])
            history["m21"].append(M1[1, 0])
            history["m22"].append(M1[1, 1])
            history["strong_median_deg"].append(strong_med)
            history["strong_q10_deg"].append(strong_q10)
            history["strong_q90_deg"].append(strong_q90)
            history["weak_median_deg"].append(weak_med)
            history["gate_agreement"].append(agreement)
            history["phi_dot_actual"].append(actual_med)
            history["phi_dot_true_u"].append(true_u_med)
            history["phi_dot_coarse_w"].append(coarse_w_med)
            history["phi_dot_coarse_u"].append(coarse_u_med)
            history["wu_angle_diff_deg"].append(wu_diff_med)
            history["angular_identity_error"].append(
                identity_error
            )

            history["m_eff_A1"].append(m_eff_A1)
            history["m_eff_W"].append(m_eff_W)
            history["m_eff_O"].append(m_eff_O)
            history["m_eff_additivity_error"].append(
                additivity_error
            )
            history["L_A1"].append(L_A1)
            history["L_W"].append(L_W)
            history["L_O"].append(L_O)
            history["G_A1"].append(G_A1)
            history["G_W"].append(G_W)
            history["G_O"].append(G_O)

            # Existing coarse thresholds retained as terminal checks.
            if (
                not crossed_0555
                and m_eff >= 0.554921
            ):
                crossed_0555 = True
                print()
                print(
                    f"m_eff crossed 0.554921 at t={t:.4f}"
                )
                print(
                    f"m_eff = {m_eff:.6f}, "
                    f"fixed A1 m11 = {m11:.6f}"
                )

            if (
                not crossed_0627
                and m_eff >= 0.627131
            ):
                crossed_0627 = True
                print()
                print(
                    f"m_eff crossed 0.627131 at t={t:.4f}"
                )
                print(
                    f"m_eff = {m_eff:.6f}, "
                    f"fixed A1 m11 = {m11:.6f}"
                )

        # Do not step past cfg.max_time.
        if step == num_steps:
            break

        dU, dW, _ = gradients(
            X,
            U,
            W,
        )

        U = U + cfg.dt * dU
        W = W + cfg.dt * dW

    for key in history:
        history[key] = np.asarray(
            history[key]
        )

    # ---------------------------------------------------------
    # Final O-cohort allocation diagnostics.
    # ---------------------------------------------------------

    other_angles_deg = (
        np.degrees(
            current_angles(U[other_mask])
        )
        + 180.0
    ) % 360.0 - 180.0

    other_mass_weights = (
        np.sum(U[other_mask] ** 2, axis=1)
        + np.sum(W[other_mask] ** 2, axis=1)
    )

    total_other_mass = np.sum(
        other_mass_weights
    )

    near_e1 = (
        np.abs(other_angles_deg) <= 10.0
    )

    near_e2 = (
        np.abs(other_angles_deg - 90.0) <= 10.0
    )

    other_mass_frac_e1 = (
        np.sum(other_mass_weights[near_e1])
        / total_other_mass
        if total_other_mass > 0.0
        else np.nan
    )

    other_mass_frac_e2 = (
        np.sum(other_mass_weights[near_e2])
        / total_other_mass
        if total_other_mass > 0.0
        else np.nan
    )

    final_diagnostics = {
        "other_angles_deg": other_angles_deg,
        "other_mass_weights": other_mass_weights,
        "other_mass_frac_e1": float(other_mass_frac_e1),
        "other_mass_frac_e2": float(other_mass_frac_e2),
        "other_count": int(other_mask.sum()),
    }

    return (
        history,
        weak_test_phi,
        final_diagnostics,
    )


# =============================================================================
# Plotting / Phase-0 summaries
# =============================================================================

def _prepare_figure_dir(cfg: Config):
    output_dir = Path(cfg.figure_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return output_dir


def _save_pdf(fig, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.tight_layout()
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _interpolated_creation(history, key):
    """
    First negative-to-positive crossing of a weak-field maximum.

    Linearly interpolates time and m_eff between logged checkpoints.
    Returns None if no crossing is observed.
    """
    values = history[key]
    times = history["weak_field_time"]
    mvals = history["weak_field_m_eff"]

    for i in range(len(values) - 1):
        y0 = values[i]
        y1 = values[i + 1]

        if y0 <= 0.0 and y1 > 0.0:
            alpha = (
                -y0 / (y1 - y0)
            )

            t_cross = (
                times[i]
                + alpha
                * (times[i + 1] - times[i])
            )

            m_cross = (
                mvals[i]
                + alpha
                * (mvals[i + 1] - mvals[i])
            )

            return (
                float(t_cross),
                float(m_cross),
            )

    return None


def analyze_capture(
    history,
    cfg: Config,
):
    """
    P0-B: analyze actual neuron capture by the moving true weak basin.

    The weak branch is read from the true-residual test field.  A neuron
    is certified captured at the first logged time after basin creation
    for which it lies in Q1 and at least capture_margin_deg above the
    moving repeller.

    When the repeller has moved below the left edge of the branch grid,
    a positive field value at that left edge certifies that the repeller
    lies below the grid.  In that case the grid edge is used as a
    conservative separatrix bound.

    Persistence is checked through cfg.weak_field_max_time, i.e. for as
    long as the moving field is explicitly tracked.
    """
    creation = _interpolated_creation(
        history,
        "weak_H_true",
    )

    if creation is None:
        return None

    t_create = float(
        creation[0]
    )

    times = np.asarray(
        history["weak_field_time"]
    )

    repeller = np.asarray(
        history["weak_repeller_deg"]
    )

    stable = np.asarray(
        history["weak_stable_deg"]
    )

    left_values = np.asarray(
        history["weak_left_true"]
    )

    # The weak-field checkpoints are the initial contiguous segment of
    # the ordinary logging checkpoints because both use the same cadence.
    n_field = len(times)

    W_angles = np.asarray(
        history["W_angles_deg"]
    )[:n_field]

    O_angles = np.asarray(
        history["O_angles_deg"]
    )[:n_field]

    W_mass_full = np.asarray(
        history["W_individual_mass"]
    )

    O_mass_full = np.asarray(
        history["O_individual_mass"]
    )

    left_edge = float(
        cfg.weak_branch_min_deg
    )

    margin_required = float(
        cfg.capture_margin_deg
    )

    def effective_separatrix(j):
        """
        Return a conservative separatrix angle at field checkpoint j.

        If the repeller is explicitly visible, use it.  If the stable
        weak root exists but the true field is already positive at the
        left grid edge, then the repeller has moved below that edge and
        the edge itself is a conservative upper bound.
        """
        if np.isfinite(
            repeller[j]
        ):
            return float(
                repeller[j]
            )

        if (
            np.isfinite(stable[j])
            and left_values[j] > 0.0
        ):
            return left_edge

        return np.nan

    def analyze_matrix(
        angles,
        mass_full,
    ):
        if angles.ndim != 2:
            raise ValueError(
                "capture angle history must be a 2D [time, neuron] array"
            )

        num_neurons = angles.shape[1]

        captured = np.zeros(
            num_neurons,
            dtype=bool,
        )

        persistent = np.zeros(
            num_neurons,
            dtype=bool,
        )

        capture_time = np.full(
            num_neurons,
            np.nan,
        )

        capture_angle = np.full(
            num_neurons,
            np.nan,
        )

        capture_repeller = np.full(
            num_neurons,
            np.nan,
        )

        capture_margin = np.full(
            num_neurons,
            np.nan,
        )

        entry_time = np.full(
            num_neurons,
            np.nan,
        )

        entry_angle = np.full(
            num_neurons,
            np.nan,
        )

        entry_repeller = np.full(
            num_neurons,
            np.nan,
        )

        entry_margin = np.full(
            num_neurons,
            np.nan,
        )

        # -----------------------------------------------------
        # Q2 -> Q1 entry times.
        # -----------------------------------------------------
        for i in range(num_neurons):
            for j in range(
                1,
                len(times),
            ):
                if (
                    angles[j - 1, i] >= 90.0
                    and angles[j, i] < 90.0
                ):
                    entry_time[i] = float(
                        times[j]
                    )
                    entry_angle[i] = float(
                        angles[j, i]
                    )

                    sep = effective_separatrix(
                        j
                    )

                    if np.isfinite(sep):
                        entry_repeller[i] = sep
                        entry_margin[i] = (
                            entry_angle[i]
                            - sep
                        )

                    break

        # -----------------------------------------------------
        # First certified capture.
        # -----------------------------------------------------
        for i in range(num_neurons):
            for j, t in enumerate(times):
                if t < t_create:
                    continue

                phi = float(
                    angles[j, i]
                )

                # Capture is certified in the first quadrant.
                if not (
                    0.0 < phi < 90.0
                ):
                    continue

                sep = effective_separatrix(
                    j
                )

                if not np.isfinite(sep):
                    continue

                gap = phi - sep

                if gap >= margin_required:
                    captured[i] = True
                    capture_time[i] = float(t)
                    capture_angle[i] = phi
                    capture_repeller[i] = sep
                    capture_margin[i] = gap
                    break

            if not captured[i]:
                continue

            # -------------------------------------------------
            # Persistence: after certification, the neuron must
            # never cross to the repeller's lower-angle side
            # during the tracked field window.
            #
            # A brief return to Q2 (phi >= 90) is not counted as
            # loss: the equal-noise Q2 field points downward and
            # the capture barrier concerns escape across the
            # lower-angle separatrix.
            # -------------------------------------------------
            start = int(
                np.searchsorted(
                    times,
                    capture_time[i],
                )
            )

            persistent_i = True

            for j in range(
                start,
                len(times),
            ):
                phi = float(
                    angles[j, i]
                )

                if phi >= 90.0:
                    continue

                if phi <= 0.0:
                    persistent_i = False
                    break

                sep = effective_separatrix(
                    j
                )

                if (
                    np.isfinite(sep)
                    and phi <= sep
                ):
                    persistent_i = False
                    break

            persistent[i] = (
                persistent_i
            )

        result = {
            "captured": captured,
            "persistent": persistent,
            "capture_time": capture_time,
            "capture_angle": capture_angle,
            "capture_repeller": capture_repeller,
            "capture_margin": capture_margin,
            "entry_time": entry_time,
            "entry_angle": entry_angle,
            "entry_repeller": entry_repeller,
            "entry_margin": entry_margin,
            "capture_fraction": float(
                np.mean(captured)
            ) if num_neurons > 0 else np.nan,
            "persistent_fraction": float(
                np.mean(
                    captured
                    & persistent
                )
            ) if num_neurons > 0 else np.nan,
        }

        if (
            mass_full.ndim == 2
            and mass_full.shape[1] == num_neurons
            and mass_full.shape[0] > 0
        ):
            final_mass = mass_full[-1]
            total_mass = float(
                np.sum(final_mass)
            )

            if total_mass > 0.0:
                result[
                    "captured_final_mass_fraction"
                ] = float(
                    np.sum(
                        final_mass[
                            captured
                        ]
                    )
                    / total_mass
                )

                result[
                    "persistent_final_mass_fraction"
                ] = float(
                    np.sum(
                        final_mass[
                            captured
                            & persistent
                        ]
                    )
                    / total_mass
                )

        return result

    W_result = analyze_matrix(
        W_angles,
        W_mass_full,
    )

    O_result = analyze_matrix(
        O_angles,
        O_mass_full,
    )

    return {
        "creation_time": t_create,
        "creation_m_eff": float(
            creation[1]
        ),
        "W": W_result,
        "O": O_result,
    }


def print_capture_summary(
    capture,
):
    print()
    print("=" * 72)
    print("P0-B CAPTURE SUMMARY")
    print("=" * 72)

    if capture is None:
        print(
            "No true weak-basin creation observed."
        )
        return

    print(
        "true basin creation: "
        f"t~{capture['creation_time']:.6f}, "
        f"m_eff~{capture['creation_m_eff']:.6f}"
    )

    def _print_cohort(
        label,
        result,
    ):
        print()
        print(label)
        print(
            "    capture fraction = "
            f"{result['capture_fraction']:.6f}"
        )
        print(
            "    persistent fraction = "
            f"{result['persistent_fraction']:.6f}"
        )

        if "captured_final_mass_fraction" in result:
            print(
                "    captured final-mass fraction = "
                f"{result['captured_final_mass_fraction']:.6f}"
            )
            print(
                "    persistent final-mass fraction = "
                f"{result['persistent_final_mass_fraction']:.6f}"
            )

        finite_capture = np.isfinite(
            result["capture_time"]
        )

        if np.any(finite_capture):
            margins = result[
                "capture_margin"
            ][finite_capture]

            print(
                "    median capture time = "
                f"{np.median(result['capture_time'][finite_capture]):.6f}"
            )
            print(
                "    earliest capture time = "
                f"{np.min(result['capture_time'][finite_capture]):.6f}"
            )
            print(
                "    median capture margin = "
                f"{np.median(margins):.3f} deg"
            )
            print(
                "    minimum capture margin = "
                f"{np.min(margins):.3f} deg"
            )

        finite_entry_margin = np.isfinite(
            result["entry_margin"]
        )

        if np.any(finite_entry_margin):
            print(
                "    median Q2->Q1 entry margin = "
                f"{np.median(result['entry_margin'][finite_entry_margin]):.3f} deg"
            )
            print(
                "    minimum Q2->Q1 entry margin = "
                f"{np.min(result['entry_margin'][finite_entry_margin]):.3f} deg"
            )

    _print_cohort(
        "Fixed W seed cohort:",
        capture["W"],
    )

    _print_cohort(
        "O transition cohort:",
        capture["O"],
    )


def print_phase0_summary(
    history,
    final_diagnostics,
):
    true_creation = _interpolated_creation(
        history,
        "weak_H_true",
    )

    without_W_creation = _interpolated_creation(
        history,
        "weak_H_without_W",
    )

    print()
    print("=" * 72)
    print("PHASE-0 SUMMARY")
    print("=" * 72)

    if true_creation is None:
        print(
            "true weak-field creation: not observed"
        )
    else:
        print(
            "true weak-field creation: "
            f"t~{true_creation[0]:.5f}, "
            f"m_eff~{true_creation[1]:.6f}"
        )

    if without_W_creation is None:
        print(
            "without-W creation: not observed"
        )
    else:
        print(
            "without-W creation: "
            f"t~{without_W_creation[0]:.5f}, "
            f"m_eff~{without_W_creation[1]:.6f}"
        )

    if (
        true_creation is not None
        and without_W_creation is not None
    ):
        print(
            "creation shift after removing W: "
            f"Delta t="
            f"{without_W_creation[0] - true_creation[0]:+.6e}, "
            f"Delta m="
            f"{without_W_creation[1] - true_creation[1]:+.6e}"
        )

    print(
        "max m_eff additivity error = "
        f"{np.max(history['m_eff_additivity_error']):.3e}"
    )

    print(
        "final O-mass within 10 deg of +e1: "
        f"{final_diagnostics['other_mass_frac_e1']:.6f}"
    )

    print(
        "final O-mass within 10 deg of +e2: "
        f"{final_diagnostics['other_mass_frac_e2']:.6f}"
    )

    # Empirical upper-envelope constants for a few candidate rho values.
    # Report both through the observed creation window and through t <= 2.
    for t_max in (1.12, 2.0):
        mask = history["time"] <= t_max + 1e-12

        if not np.any(mask):
            continue

        G1 = history["G_A1"][mask]
        GW = history["G_W"][mask]

        print()
        print(
            f"Radial envelope through t <= {t_max:.2f}:"
        )

        for rho in (0.25, 0.50, 0.75, 0.90):
            C_rate = np.max(
                GW - rho * G1
            )
            print(
                f"    rho={rho:.2f}: "
                f"C_rate={C_rate:.6f}"
            )


def plot_phase0_diagnostics(
    history,
    weak_test_phi,
    final_diagnostics,
    cfg: Config,
):
    """
    Save the NEW Phase-0 figures as PDF files.

    No matplotlib windows are opened. The older diagnostic figures are
    intentionally omitted by default because they have already served
    their purpose; this function focuses on P0-A/P0-C and the direct
    without-W ablation.
    """
    output_dir = _prepare_figure_dir(
        cfg
    )

    t = history["time"]
    tw = history["weak_field_time"]

    # ---------------------------------------------------------
    # 1. Exact m_eff decomposition.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        t,
        history["m_eff"],
        label=r"$m_{\mathrm{eff}}$",
    )
    ax.plot(
        t,
        history["m_eff_A1"],
        label=r"$m_{\mathrm{eff}}^{A_1}$",
    )
    ax.plot(
        t,
        history["m_eff_W"],
        label=r"$m_{\mathrm{eff}}^{W}$",
    )
    ax.plot(
        t,
        history["m_eff_O"],
        label=r"$m_{\mathrm{eff}}^{O}$",
    )

    ax.set_xlabel("time")
    ax.set_ylabel("effective strong-fit coefficient")
    ax.set_title(
        "P0-A: exact effective-fit decomposition"
    )
    ax.legend()

    _save_pdf(
        fig,
        output_dir / "phase0_m_eff_decomposition.pdf",
    )

    # ---------------------------------------------------------
    # 2. Direct self-consistency ablation.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        tw,
        history["weak_H_true"],
        label=r"$H_{\rm true}$",
    )
    ax.plot(
        tw,
        history["weak_H_without_W"],
        label=r"$H_{\setminus W}$",
    )
    ax.plot(
        tw,
        history["weak_H_coarse"],
        label=r"$H_{\rm coarse}$",
    )
    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel("time")
    ax.set_ylabel(
        r"$\max_\phi V^{\rm test}(\phi;t)$"
    )
    ax.set_title(
        "P0-A: weak-basin creation with and without W"
    )
    ax.legend()

    _save_pdf(
        fig,
        output_dir / "phase0_weak_field_ablation.pdf",
    )

    # ---------------------------------------------------------
    # 3. Direct angular influence of W.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        tw,
        history["weak_W_influence_sup"],
    )

    ax.set_xlabel("time")
    ax.set_ylabel(
        r"$\sup_\phi |V_{\rm true}-V_{\setminus W}|$"
    )
    ax.set_title(
        "P0-A: direct weak-cohort influence on test field"
    )

    _save_pdf(
        fig,
        output_dir / "phase0_W_field_influence.pdf",
    )

    # ---------------------------------------------------------
    # 4. Projected cohort masses.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        t,
        history["L_A1"],
        label=r"$L_{A_1}$",
    )
    ax.plot(
        t,
        history["L_W"],
        label=r"$L_W$",
    )
    ax.plot(
        t,
        history["L_O"],
        label=r"$L_O$",
    )

    ax.set_xlabel("time")
    ax.set_ylabel("projected cohort mass")
    ax.set_title(
        "P0-C: projected cohort mass"
    )
    ax.legend()

    _save_pdf(
        fig,
        output_dir / "phase0_projected_masses.pdf",
    )

    # ---------------------------------------------------------
    # 5. Integrated cohort growth versus time.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        t,
        history["G_A1"],
        label=r"$G_{A_1}$",
    )
    ax.plot(
        t,
        history["G_W"],
        label=r"$G_W$",
    )
    ax.plot(
        t,
        history["G_O"],
        label=r"$G_O$",
    )

    ax.set_xlabel("time")
    ax.set_ylabel(
        r"$\frac{1}{2}\log(L_C(t)/L_C(0))$"
    )
    ax.set_title(
        "P0-C: cumulative radial growth"
    )
    ax.legend()

    _save_pdf(
        fig,
        output_dir / "phase0_cumulative_growth.pdf",
    )

    # ---------------------------------------------------------
    # 6. Parametric radial-race plot G_W versus G_A1.
    # ---------------------------------------------------------

    mask = history["G_A1"] > 1.0

    fig, ax = plt.subplots(
        figsize=(6.5, 6.0)
    )

    if np.any(mask):
        ax.plot(
            history["G_A1"][mask],
            history["G_W"][mask],
            marker="o",
            markersize=2,
        )

        lo = min(
            float(np.min(history["G_A1"][mask])),
            float(np.min(history["G_W"][mask])),
        )
        hi = max(
            float(np.max(history["G_A1"][mask])),
            float(np.max(history["G_W"][mask])),
        )

        ax.plot(
            [lo, hi],
            [lo, hi],
            linestyle="--",
            label="slope 1 reference",
        )
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            r"No points with $G_{A_1}>1$",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel(r"$G_{A_1}$")
    ax.set_ylabel(r"$G_W$")
    ax.set_title(
        "P0-C: parametric radial race"
    )

    _save_pdf(
        fig,
        output_dir / "phase0_radial_race_parametric.pdf",
    )

    # ---------------------------------------------------------
    # 7-8. Final O angular allocation: count and mass weighted.
    # ---------------------------------------------------------

    other_angles = final_diagnostics[
        "other_angles_deg"
    ]
    other_weights = final_diagnostics[
        "other_mass_weights"
    ]

    bins = np.linspace(
        -180.0,
        180.0,
        73,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )
    ax.hist(
        other_angles,
        bins=bins,
    )
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(
        90.0,
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlabel("final O-cohort angle (degrees)")
    ax.set_ylabel("neuron count")
    ax.set_title(
        "P0-A: final O-cohort angular distribution"
    )

    _save_pdf(
        fig,
        output_dir / "phase0_other_angles_count.pdf",
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )
    ax.hist(
        other_angles,
        bins=bins,
        weights=other_weights,
    )
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(
        90.0,
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlabel("final O-cohort angle (degrees)")
    ax.set_ylabel("projected mass")
    ax.set_title(
        "P0-A: final O-cohort mass-weighted allocation"
    )

    _save_pdf(
        fig,
        output_dir / "phase0_other_angles_mass_weighted.pdf",
    )

    # ---------------------------------------------------------
    # 9. Weak-field snapshots: true / without-W / coarse.
    # ---------------------------------------------------------

    snapshot_dir = (
        output_dir
        / "weak_field_snapshots"
    )

    plot_weak_field_snapshots(
        history=history,
        weak_test_phi=weak_test_phi,
        output_dir=snapshot_dir,
        targets=(
            0.90,
            1.00,
            1.04,
            1.06,
            1.08,
            1.10,
            1.12,
            1.14,
            1.16,
        ),
    )

    print()
    print(
        "Saved Phase-0 PDF figures to: "
        f"{output_dir.resolve()}"
    )


def plot_capture_diagnostics(
    history,
    capture,
    cfg: Config,
):
    """
    Save P0-B capture figures as PDFs.  No interactive windows are opened.
    """
    if capture is None:
        return

    output_dir = _prepare_figure_dir(
        cfg
    )

    times = np.asarray(
        history["weak_field_time"]
    )

    n_field = len(times)

    W_angles = np.asarray(
        history["W_angles_deg"]
    )[:n_field]

    O_angles = np.asarray(
        history["O_angles_deg"]
    )[:n_field]

    repeller = np.asarray(
        history["weak_repeller_deg"]
    )

    stable = np.asarray(
        history["weak_stable_deg"]
    )

    # ---------------------------------------------------------
    # 1. W-seed trajectories against moving separatrix.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    captured_W = capture["W"][
        "captured"
    ]

    for i in range(
        W_angles.shape[1]
    ):
        if captured_W[i]:
            ax.plot(
                times,
                W_angles[:, i],
                linewidth=1.2,
                alpha=0.65,
            )
        else:
            ax.plot(
                times,
                W_angles[:, i],
                linewidth=0.7,
                alpha=0.20,
            )

    ax.plot(
        times,
        repeller,
        linewidth=2.5,
        label=r"$\phi_{\rm rep}(t)$",
    )

    ax.plot(
        times,
        stable,
        linewidth=2.5,
        label=r"$\phi_W(t)$",
    )

    ax.axhline(
        90.0,
        linestyle="--",
        linewidth=1,
        label=r"$90^\circ$",
    )

    ax.axvline(
        capture["creation_time"],
        linestyle="--",
        linewidth=1,
        label="basin creation",
    )

    ax.set_xlim(
        max(
            0.0,
            capture["creation_time"] - 0.35,
        ),
        cfg.weak_field_max_time,
    )

    ax.set_ylim(
        0.0,
        180.0,
    )

    ax.set_xlabel("time")
    ax.set_ylabel(
        "projected angle (degrees)"
    )

    ax.set_title(
        "P0-B: W-seed trajectories vs. moving weak separatrix"
    )

    ax.legend(
        loc="best"
    )

    _save_pdf(
        fig,
        output_dir
        / "phase0B_W_capture_trajectories.pdf",
    )

    # ---------------------------------------------------------
    # 2. O-transition trajectories against moving separatrix.
    #    To keep the figure legible, use light lines; this is a
    #    geometric diagnostic, while final O mass allocation is
    #    already shown by the P0-A mass-weighted histogram.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for i in range(
        O_angles.shape[1]
    ):
        ax.plot(
            times,
            O_angles[:, i],
            linewidth=0.55,
            alpha=0.12,
        )

    ax.plot(
        times,
        repeller,
        linewidth=2.5,
        label=r"$\phi_{\rm rep}(t)$",
    )

    ax.plot(
        times,
        stable,
        linewidth=2.5,
        label=r"$\phi_W(t)$",
    )

    ax.axhline(
        90.0,
        linestyle="--",
        linewidth=1,
        label=r"$90^\circ$",
    )

    ax.axvline(
        capture["creation_time"],
        linestyle="--",
        linewidth=1,
        label="basin creation",
    )

    ax.set_xlim(
        max(
            0.0,
            capture["creation_time"] - 0.35,
        ),
        cfg.weak_field_max_time,
    )

    ax.set_ylim(
        -10.0,
        180.0,
    )

    ax.set_xlabel("time")
    ax.set_ylabel(
        "projected angle (degrees)"
    )

    ax.set_title(
        "P0-B: O-transition trajectories vs. moving weak separatrix"
    )

    ax.legend(
        loc="best"
    )

    _save_pdf(
        fig,
        output_dir
        / "phase0B_O_capture_trajectories.pdf",
    )

    # ---------------------------------------------------------
    # 3. Capture-margin distributions.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    W_margins = capture["W"][
        "capture_margin"
    ]

    O_margins = capture["O"][
        "capture_margin"
    ]

    W_margins = W_margins[
        np.isfinite(W_margins)
    ]

    O_margins = O_margins[
        np.isfinite(O_margins)
    ]

    if len(W_margins) > 0:
        ax.hist(
            W_margins,
            bins=min(
                20,
                max(
                    5,
                    len(W_margins),
                ),
            ),
            alpha=0.55,
            label="W seed",
        )

    if len(O_margins) > 0:
        ax.hist(
            O_margins,
            bins=20,
            alpha=0.45,
            label="O transition",
        )

    ax.axvline(
        cfg.capture_margin_deg,
        linestyle="--",
        linewidth=1,
        label="certification margin",
    )

    ax.set_xlabel(
        r"$\phi_i-\phi_{\rm rep}$ at first certified capture (degrees)"
    )
    ax.set_ylabel("neuron count")
    ax.set_title(
        "P0-B: capture-margin distribution"
    )

    ax.legend(
        loc="best"
    )

    _save_pdf(
        fig,
        output_dir
        / "phase0B_capture_margins.pdf",
    )

    # ---------------------------------------------------------
    # 4. Moving branch alone, useful for theorem-margin reading.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        times,
        repeller,
        label=r"$\phi_{\rm rep}(t)$",
    )

    ax.plot(
        times,
        stable,
        label=r"$\phi_W(t)$",
    )

    ax.axhline(
        90.0,
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        capture["creation_time"],
        linestyle="--",
        linewidth=1,
        label="creation",
    )

    ax.set_xlabel("time")
    ax.set_ylabel("angle (degrees)")
    ax.set_title(
        "P0-B: moving true weak branch"
    )
    ax.legend()

    _save_pdf(
        fig,
        output_dir
        / "phase0B_moving_weak_branch.pdf",
    )

    print()
    print(
        "Saved P0-B PDF figures to: "
        f"{output_dir.resolve()}"
    )


def plot_weak_field_snapshots(
    history,
    weak_test_phi,
    output_dir,
    targets,
):
    """
    Save true / without-W / coarse field comparisons at selected times.
    """
    times = history["weak_field_time"]
    phi_deg = np.degrees(
        weak_test_phi
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for target in targets:
        index = np.argmin(
            np.abs(
                times - target
            )
        )

        t = times[index]
        m_eff = history[
            "weak_field_m_eff"
        ][index]

        V_true = history[
            "weak_V_true"
        ][index]
        V_without_W = history[
            "weak_V_without_W"
        ][index]
        V_coarse = history[
            "weak_V_coarse"
        ][index]

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            phi_deg,
            V_true,
            label="true residual",
        )
        ax.plot(
            phi_deg,
            V_without_W,
            label="true residual without W",
        )
        ax.plot(
            phi_deg,
            V_coarse,
            label="coarse residual",
        )
        ax.axhline(
            0.0,
            linewidth=1,
        )

        ax.set_xlabel(r"$\phi$ (degrees)")
        ax.set_ylabel(
            r"$V^{\rm test}(\phi;t)$"
        )
        ax.set_title(
            f"Weak test field: "
            f"t={t:.2f}, "
            f"m_eff={m_eff:.3f}"
        )
        ax.legend()

        filename = (
            f"weak_field_t_{t:.2f}.pdf"
        )

        _save_pdf(
            fig,
            output_dir / filename,
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    cfg = Config()

    (
        history,
        weak_test_phi,
        final_diagnostics,
    ) = run(cfg)

    print_phase0_summary(
        history,
        final_diagnostics,
    )

    plot_phase0_diagnostics(
        history,
        weak_test_phi,
        final_diagnostics,
        cfg,
    )

    capture = analyze_capture(
        history,
        cfg,
    )

    print_capture_summary(
        capture,
    )

    plot_capture_diagnostics(
        history,
        capture,
        cfg,
    )
