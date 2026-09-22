# Canonical crossing: local certificate and remaining entry gap

The initialized noisy theorem is **not closed**. The new result is a computer-assisted **local gradient-flow theorem**, conditional on entry into an explicit small neighborhood of a learned canonical state. It is stronger than checking derivative signs at numerical snapshots: the local continuous trajectory is enclosed through all training-gate changes.

## New result

For the supplied canonical binary dataset (200 neurons, 4000 samples, means 3 and 2, noise 0.15), start any compatible gradient-flow trajectory within Euclidean parameter distance **1e-6** of the supplied reference state at t=3.4. The local validator checks 2500 steps through t=3.9. Static neighborhood bounds then give, uniformly over probe angles within **1e-6 radians of −85 degrees**:

- E(3.4) − E(3.7) > 1.2e-4;
- E(3.9) − E(3.7) > 4e-5;
- a decreasing interval [3.4,3.40001] and an increasing interval [3.9,3.90001].

The fixed offline cohort contains 71 neurons. At entry its ungated m11 exceeds 0.9322, its input angles are within 15 degrees, and its output angles within 20 degrees. This is substantial learned mass and coarse strong-block geometry, not fine locking.

The exact certified error radii and execution status are in `LOCAL_FLOW_CERTIFICATE.json`. The theorem and proof are in `LOCAL_CROSSING_THEOREM.tex`, included in the standalone continuation document. No initialized reachability is inferred from restarting at the reference state.

## Mechanism resolved locally

The actual full OOD rate splits into contributions from the two training clusters, using the **same full-network residual and all neurons**. The strong-cluster contribution is negative at both endpoints. The weak-cluster contribution is positive at both endpoints and dominates at the later one. From the earlier to the later endpoint, the strong contribution becomes less negative and the weak contribution increases, each with strict certified margins. This is an exact signed field decomposition, not a retraining ablation.

`STATIC_WITNESS_CERTIFICATE.json` gives outward bounds throughout balls of radius 3.5e-5, including every possible training-gate change. Probe masks are verified to stay fixed only within those endpoint balls. `SNAPSHOT_CERTIFICATE.json` independently verifies full-network error gaps on the explicit sector.

## Exact remaining initialized inequality

For the flow from the original archived initialization, establish

    ||theta_GF(3.4) − theta_reference(3.4)||_2 <= 1e-6.

The local certificate then applies. A larger entry neighborhood might also work, but has not been certified here. No probability guarantee follows from this fixed dataset, fixed offline cohort, and reference neighborhood.

The attempted scalar enclosure from t=0 failed near t=1.68. This is a failure of a sufficient bound, not an obstruction to the initialized theorem. A retrospective event audit suggests that retaining which neuron each gate event affects would be substantially less costly. Its sampled sensitivity ratios are **not** validated trajectory bounds.

## Other proved content

- Exact weak-source square plus signed complement, target, mismatch and gate-weighted noise corrections.
- The repository statistic Q_G = eta² G is the derivative of a smooth block potential.
- Signed residual Duhamel comparison without exponential residual amplification; spectral and directional crossing criteria.
- Nonsmooth one-sided tube inequality, verified-step criterion, and transverse gate-event linearized sensitivity identity.
- Explicit smooth Hermite-defect and training-gate correction bounds used by the local validator.

The repository's Q_G is preserved. It is not replaced by the k-corrected proxy. The initial source audit shows that a positive-square argument alone does not settle the weak source: at t=3.6 the negative target term exceeds the square.

## Numerical provenance and limits

The early diagnostics reanalyze archived Euler snapshots only. After the author approved pursuing the canonical certificate, a new RK4 reference was computed with the **same dataset, same archived initial floating-point parameters, and step 0.0002**, with no resampling. RK4 accuracy by itself is not used as a proof premise. Its states and velocities define a piecewise cubic reference; verified defect and stability bounds establish the local continuous-flow enclosure.

The canonical dataset is treated as exact stored binary numbers. The theorem is not an iid Gaussian theorem. The cohort and witness times were selected retrospectively. The proof makes no preregistration, forecasting, unique-crossing, or specialization-causes-reversal claim.

The arithmetic contract is IEEE round-to-nearest basic operations and standard dot-product accumulation without unsafe fast-math. Interval trigonometry and exponentials use mpmath intervals. Exact-rational arithmetic tests and exact-symbolic derivative tests are included. The code is research proof code, not formally verified software.

The original user sources, existing paper, cleaned appendix, and 10-million-neuron simulation outputs are unchanged.
