# Research audit — canonical initialized reachability

## Outcome

The initialization-to-reversal theorem is **not closed**. The accepted new result enlarges the sufficient entry radius at time 3.4 from 1e-6 to 1.2e-5. It preserves the local full-network reversal, explicit sector, and signed strong/weak-cluster interpretation. This remains a conditional theorem.

The authoritative independently replayed initialized comparison and its precise failure are in `INITIAL_FLOW_CERTIFICATE.json` and the status section of the LaTeX supplement. A failure of this upper comparison does not imply that the actual error is large, that initialized reversal fails, or that a pivot is required.

## Exact failed sufficient inequality

The full replay from radius zero validates 4,974 intervals, through t=0.9948. All strict tube inequalities pass. At the last step, the outward comparison moves from 1.1982855683396031e-5 to 1.2003084494454761e-5, exceeding the selected 1.2e-5 entry budget for the first time. The enclosing tube still has more than 3.4221508e-6 slack.

The failed inequality is **comparison_radius(4974) <= 0.000012**, not the tube-validity inequality, and not a lower bound on true trajectory error. This nondecreasing comparison cannot certify later entry without a different estimate. It does not rule out a successful initialized certificate by another method.

## Verified mathematics

1. The sharper gate estimate keeps the reference positive conormal weight and treats input/output parameter perturbations quadratically. Its correction is `Gamma = max_i(C_i + D R K_i)/2 + D sqrt(sum_i ||w_i||² K_i²)`. The proof retains the favorable sign of `-J*J`, though it does not exploit its magnitude.
2. Signed coordinate intervals for uncertain-gate residual matrices avoid adding a second full absolute contribution for already-counted gates.
3. The static sign, cubic-reference defect, and one-sided step arguments from the supplied package were audited and retained. Exact symbolic tests independently verify the smooth derivatives and all Hermite midpoint identities.
4. The first training event is validated nonlinearly from exact initial weights. It is sample 3971, neuron 36 (zero based), inactive to active, strictly between 35888/10^10 and 35891/10^10. Every competing training gate is separated in the frozen comparison interval, and both normal velocities are positive. This establishes isolation and transversality of that event, not continuation through every later event.

## Computer-assisted results

- All 2,500 local intervals were independently revalidated with wider prescribed tubes.
- Scalar replay starting at the upward enclosure of exact decimal 0.000012 passes every strict tube inequality. Error radii at 3.7 and 3.9 are below 2.828e-5 and 4.251e-5.
- The sector width remains 2e-6 radians about −85 degrees. The uniformly certified improvement and rebound exceed 1.2e-4 and 4e-5.
- Strong learned mass at entry exceeds 0.9323. Input and output cones have half-angles 15 and 20 degrees respectively.
- The strong-cluster full OOD rate is negative at both endpoints. The weak-cluster contribution is positive and dominates at the later endpoint. Both use all neurons, the same full-network residual, and normalization 1/N.
- The exact repository Q_G has opposite endpoint signs. No k-corrected replacement is used.
- Portable input checks, 553 exact-rational arithmetic checks, and symbolic derivative checks pass.

These results are conditional on the stated floating-point arithmetic contract and apply to every compatible Clarke flow satisfying entry. They do not assume a global unique smooth trajectory or persistent training gates.

## Input contract and scientific scope

Commit `ae16f9c0e4e51269c624ffb196184568f2cfe6d9` is pinned. Four source files exactly match their committed bytes. The exact archived initial rows, cohort rule, normalization, parameter ordering, and probe definition were verified.

The historical archive has no raw training matrix and no independent training-matrix hash. The supplied binary X matches the committed generator under the recorded runtime; historical bitwise identity is not independently proved. No result should be advertised as a certificate of unarchived bits. The theorem concerns the exact supplied X and archived initial weights.

Re-evaluating the initial trigonometric routine changes some bits (maximum observed difference 4.87890977618477e-19). The verifier uses archived weights instead. The data and initialization routines each restart seed 0; no implementation-level independence claim is made.

## Numerical observations, not proofs

A floating-point feasibility run of the sharpened global bound lasted to time 2.2652 before its proposed tube exceeded 0.03, compared with the older method's failure near 1.6804. This is a method diagnostic, not an enclosure of gradient flow. Its free comparison slopes and proposed radii may be used as inputs only after the interval verifier checks them.

The empirical interpolated G/full-reversal times are not promoted to theorem constants. No new statistical experiments were run, and no canonical training parameter was changed.

## Independent normalization audit

A suspected double-normalization error proved to be a parenthesis-reading error during review. The original certainly-active and uncertain contributions both had exactly one factor 1/N. An independent computation of the smooth rates agreed within 1.8e-15. The delivered expression makes the single normalization explicit. Accumulation was replayed from radius zero using independently recomputed rates; no intermediate error estimate is assumed as an input.

## What remains open

1. Initialized entry into the independently specified learned-state neighborhood at time 3.4. Neither the interval prefix nor the first-event certificate reaches it.
2. A replacement enclosure preserving enough directional contraction, resolving enough nonlinear gate events, or bounding only sufficient observables. The present failed sufficient comparison does not rule out any of these.
3. The interval-wide signed discrepancy inequality in `TIMING_STATUS.tex`. Endpoint signs do not imply timing order, a unique crossing, or minimizer localization.
4. A high-probability theorem at canonical parameters. The existing extremely asymmetric finite-sample regime does not automatically apply to this instance.
5. Parameter-dependent timing and positive-cone directional claims. These were not pursued ahead of initialized entry.

No initialized theorem, timing lag, high-probability guarantee, or theorem-scope pivot is claimed. The original appendix and abstract remain unchanged.
