# Canonical theory status

| Target | Status | Exact scope |
|---|---|---|
| Full-tube conditional timing theorem | **CLOSED** | All 2,500 steps; sector-wide sign separation and localization of every minimizer on [3.4,3.9], conditional on the existing 1.2e-5 entry bound. |
| Observable-entry theorem | **OPEN** | Exact fixed-pattern moment ODE and static observable formulas proved; aggregate-to-event continuation interface is not certified. |
| Initialized observable reachability | **OPEN** | No aggregate/event verifier from initialization is claimed. |
| Initialized canonical reversal | **OPEN** | Reaching the learned entry region remains necessary. |

## Closed timing conclusion

The repository statistic has a rigorous sign-transition bracket [3.5738, 3.6348]. Uniformly over the sector of half-width 1e-6 radians around -85 degrees, every full OOD-error minimizer on [3.4,3.9] is in [3.6384,3.7152]. The certified separation is **0.0036 = 18/5000**.

For the central probe the minimizer bracket is [3.6394,3.7136], giving separation **0.0046**. Neither uniqueness nor continuity of a G zero is required. These brackets are certificates, not fitted empirical timings.

The complete-tube margin separating the positive sector-rate suffix from zero is at least 1.9957e-8. The smallest certified sector probe-gate margin is greater than 1.40926e-5. Every sign-unresolved index and adjacent strict bound is in `TIMING_FULL_TUBE_CERTIFICATE.json`. Training gates that can switch are explicitly over-enclosed with all compatible selectors; none is frozen by assumption.

The independent 3.64 witness is also reproduced. Its Q_G-positive / full-rate-negative interval is [3.64,3.640006]. Some rounded numbers in the submitted message were inward at their last displayed digit; the theorem uses outward rounding instead, without changing any conclusion.

## Exact quotient result

The three moment equations hold with minus signs for residual f-x and plus signs for residual x-f. An exact symbolic check verifies transposes and normalization. The census is diagnostic only: 53 training patterns initially and 134 at entry, with 185211 adjacent-reference sign differences. Those differences are not claimed to be validated event counts.

## First unclosed obligation

For an event neuron i, aggregate continuation needs a validated rank-one transfer bound `||K_i-Khat_i|| <= b_i`, together with event-time/orientation control. It cannot be inferred from the group-moment error alone. The exact counterexample and static-to-dynamic dependency audit are in `TRAINING_PATTERN_QUOTIENT.tex` and `OBSERVABLE_ENTRY_AUDIT.md`.

The best next refinement is to retain boundary-neuron rank-one states while propagating probe- and cohort-refined group moments for the interior. This is not a theorem-scope pivot and does not revive full-parameter coercivity.

## Provenance

The preceding 74-file archive passed its SHA-256 manifest. The existing flow certificate and original appendix were not changed or rerun. The theorem refers to the supplied exact binary dataset; the original historical archive has no independent raw-data hash. Canonical parameters, cohort, initialization, probe, and training dynamics were unchanged. No statistical experiments were run.
