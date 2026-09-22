# Observable entry: exact reductions and remaining continuation inequality

## Status: OPEN

The full-tube timing theorem is closed under the existing Euclidean entry assumption. An independently certified aggregate-only entry set implying that theorem has **not** been constructed. A condition asserting the desired later signs would not fix this.

The training-pattern moment equations are proved in `TRAINING_PATTERN_QUOTIENT.tex`. They give exact dynamics on an interval with fixed training patterns. They do not, alone, define a complete hybrid OOD state across pattern changes.

## What can be replaced exactly

Refine the partition by training pattern g, fixed strong-cohort flag c, and fixed probe-active flag s. For each subgroup a, store K_a = [[U_a,M_a^T],[M_a,W_a]]. On a window where these labels are fixed:

- Every training output is sum_a g_a(n) M_a x_n.
- The ungated strong matrix is sum_{a:c=1} M_a.
- The actual probe matrix is sum_{a:s=1} M_a.
- A_g = mean_n g(n) (f(x_n)-x_n) x_n^T.
- dM_a/dt = -A_g U_a - W_a A_g.
- Q_G and the full error/rate are then exact algebraic functions of the aggregate state.

Thus the **static** forward outputs, strong mass, signed cluster contributions, and full-rate discrepancy can be bounded from moments, without recovering every neuron's parameters. The same construction works separately for the two training clusters, always retaining the full residual and normalization 1/N. It applies across a sector when its probe labels are certified uniformly.

The canonical reference has no training-pattern group with mixed central-probe signs at the inspected 3.4, 3.64, and 3.9 knots. That diagnostic does not replace a tube-wide certificate. It does have a mixed-cohort training-pattern group of size 16 at entry, and size 15 at the other inspected knots, so cohort refinement is necessary for the current Q_G interface.

## What the existing proof still uses individual geometry for

1. Training/probe hyperplane margins: a group covariance does not specify each summand's margin.
2. The time and orientation of a neuron's transfer between pattern groups.
3. The rank-one moment moved at a transfer.
4. The original per-neuron cone assertion. Group small second moments can imply mass-weighted concentration, but not that every member is in the cone.

The Euclidean local proof handles these through its actual tube and compatible-selector bounds. Those estimates cannot be reused unchanged for arbitrary states merely sharing group moments.

## The first unclosed continuation inequality

At a transfer of neuron i from g to g', let K_i=y_i y_i^T and let hats denote the reference state. If D_g bounds the group-moment error immediately before transfer, a synchronized transfer gives

    ||Delta K_g^+|| <= D_g + ||K_i - Khat_i||.

If actual and reference transfers are not synchronized, their membership mismatch adds the corresponding rank-one moment until both transfers have occurred. Therefore a new verifier needs a separately validated bound

    ||K_i - Khat_i|| <= b_i

and a validated interval for event timing and orientation. The tempting replacement

    ||K_i - Khat_i|| <= C ||K_g - Khat_g||

is structurally false for every universal finite C: different rank-one decompositions can have identical K_g and different summands. The exact balanced counterexample in the quotient lemma also shows that aggregate training moments plus individual probe signs do not identify the probe-active aggregate.

This is an obstruction only to an **unaugmented** aggregate state. It is not an obstruction to the user's proposed mixed aggregate / boundary-neuron method, nor to initialized reversal. No numerical slack for the missing b_i guard is reported because no such guard has been constructed; presenting one would be invented evidence. The already-completed timing guards and all their neighboring margins are reported separately in the timing JSON.

## Single next refinement

Use probe- and cohort-refined pattern moments for interior neurons, while retaining rank-one states and oriented margins for neurons entering a certified boundary buffer. Validate the transfer guard and event-time enclosure **before** allocating an initialized quotient replay. The concrete target is b_i plus the asynchronous transfer interval above. Small aggregate errors alone cannot supply it.

An alternative exact representation between events is a shared fundamental matrix: y_i(t)=T_g(t)y_i(t_entry), with dT_g/dt=-B_g T_g. This retains the within-group geometry that K_g discards. It is an algebraic representation, not a validated initialized bound; its conditioning and cost must be checked before claiming an advantage.

No massive Phase D verifier was started. No global J^T J coercivity argument was used. The supplied Bernstein prefix improvement was not downgraded or overwritten; the reported frontier near 1.0158 is not independently recertified in this package.
