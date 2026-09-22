# Full-tube canonical timing certificate

**Phase A is closed. Initialized reachability remains open.** See `THEORY_STATUS.md` for the four requested status distinctions.

## Result

Conditional on the existing radius-0.000012 entry at t=3.4, the repository Q_G sign-transition bracket is [3.5738,3.6348]. Every full-error minimizer on [3.4,3.9], uniformly over the sector of half-width 0.000001 radians about −85 degrees, lies in [3.6384,3.7152]. The separation is 18/5000 = 0.0036. For the central probe the minimizer bracket is [3.6394,3.7136], giving 0.0046.

These are rigorous sign-separated brackets, not unique-zero claims or empirical interpolation estimates. The reported t=3.64 signed witness is independently reproduced too.

## Mathematical files

- `TIMING_FULL_TUBE_THEOREM.tex`: insertion-ready conditional theorem and proof.
- `CANONICAL_TIMING_AND_QUOTIENT.tex`: compiled supplement source.
- `CANONICAL_TIMING_AND_QUOTIENT_STANDALONE.tex`: single-file source.
- `TRAINING_PATTERN_QUOTIENT.tex`: exact fixed-pattern ODE, residual sign convention, and counterexample to an aggregate-only transfer bound.
- `APPENDIX_TIMING_INSERTION.tex`: insertion instructions. No original appendix or abstract was modified.
- `OBSERVABLE_ENTRY_AUDIT.md`: exact static replacements and missing event-transfer guard; it does not claim Phase C closed.

## Numerical files and ambiguity semantics

- `TIMING_FULL_TUBE_CERTIFICATE.csv`: 2,500 full-step enclosures for Q_G, q_E, and the directly bounded discrepancy Xi, with actual per-step radii and probe-margin certificates.
- `TIMING_FULL_TUBE_CERTIFICATE.json`: all sign-unresolved indices, exact rational bracket endpoints, neighboring strict margins, input hashes and scope.
- `POINT_TIMING_WITNESS.json`: the reproduced interval [3.64,3.640006].
- `TIMING_GEOMETRY_CHECKS.json`: independent all-step probe checks and representative full-step recomputations.
- `TRAINING_PATTERN_CENSUS.csv/.json`: diagnostic reference counts only.

`ENCLOSED` means the actual rate has a rigorous interval over the entire reference tube. It does **not** mean every training gate is constant. `training_ambiguous_pairs` records potentially varying training selectors; the static correction encloses all of them. There are no unresolved center masks or probe gates. A rate interval containing zero is sign-unresolved and is explicitly listed in the JSON. Treating a variable training selector as fixed would be invalid; that is not done here.

The source ledger's nonnegative scalar comparison is nondecreasing within each step. Its terminal `radius_upper` therefore bounds the error throughout that step. We used this certified radius, not one globally enlarged radius or merely a sampled error. We then added cubic-reference motion and midpoint-rounding bounds to obtain a midpoint ball. Probe signs were checked on the tighter actual reference tube using Bernstein coefficients.

## Reproduction

Packages: Python, NumPy, mpmath, SymPy. Use standard IEEE round-to-nearest arithmetic, square roots and BLAS without unsafe fast-math. Interval arithmetic, gamma bounds, and directed outward enlargements are inherited unchanged from the audited certificate. The tests supplement, rather than replace, the real-arithmetic proof.

Run from this directory. Use a copy to preserve the original reports, since verifiers overwrite their output files.

```sh
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python3 code/check_manifest.py
python3 code/verify_interval_arithmetic.py
python3 code/verify_local_formulas.py
python3 code/check_quotient_algebra.py
python3 code/check_timing_geometry.py
python3 code/check_point_witness.py
python3 code/summarize_timing.py
```

To recompute all 2,500 static timing enclosures (about three minutes on the recorded host):

```sh
python3 code/certify_full_tube.py --workers 3
python3 code/summarize_timing.py
python3 code/check_timing_geometry.py
```

This does not integrate training or rerun/alter the certified flow. All reference coefficients, exact X, fixed cohort, and existing flow enclosures required by the timing proof are included under `prior/`. Prior proof text and optional flow-verification source are included for audit; no flow rerun was performed in this work.

The diagnostic census uses the larger original 0-to-4 reference arrays at `work/crossing-closure/canonical-reference`. Those large arrays are not duplicated here; their hashes and the complete diagnostic CSV are supplied. `code/pattern_census.py` can reproduce that diagnostic from the original workspace. It is not a dependency of the timing theorem.

```sh
pdflatex -interaction=nonstopmode -halt-on-error CANONICAL_TIMING_AND_QUOTIENT.tex
pdflatex -interaction=nonstopmode -halt-on-error CANONICAL_TIMING_AND_QUOTIENT.tex
```

## Provenance and unchanged claims

Repository commit: `ae16f9c0e4e51269c624ffb196184568f2cfe6d9`.
Canonical h=200, N=4000, means (3,0)/(0,2), sigma=0.15, seed=0, epsilon=0.001, 71-neuron ID cohort, central probe −85 degrees, loss normalization 1/(2N), and parameter order U then W are unchanged. Q_G has exactly the repository definition, without k correction.

The prior 74-file archive passed its manifest before use. The exact binary dataset matches the pinned generator under the recorded runtime, but the historical training archive contains no independent raw-X hash. This theorem is for the supplied binary instance, not an iid high-probability event. Exact initialized entry has not been certified. The user's improved Bernstein initialization prefix is not overwritten; it is not independently recertified here.

No statistical experiments, parameter changes, global J^T J coercivity argument, or scope pivot were used.
