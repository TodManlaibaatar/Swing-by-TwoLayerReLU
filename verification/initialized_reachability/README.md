# Canonical continuation and initialized reachability audit

**The initialized canonical reversal theorem remains open.** The positive result is a conditional local theorem with entry radius `0.000012` at time 3.4, twelve times the former sufficient radius. The paper's original appendix was not modified.

## Contents

- `INITIALIZED_CANONICAL_AUDIT_STANDALONE.tex` (single-file source), modular `INITIALIZED_CANONICAL_AUDIT.tex`, and PDF: complete mathematical supplement.
- `CANONICAL_CONTINUATION_BODY.tex`, `appendix-integration-patch.tex`: conditional insertion only.
- `RESEARCH_AUDIT.md`: proved / conditional / computational / unproved distinctions.
- `WIDE_*CERTIFICATE.json`, `WIDE_LOCAL_FLOW_ENCLOSURE.csv`: enlarged local certificate.
- `INITIAL_FLOW_CERTIFICATE.json`, `INITIAL_FLOW_ENCLOSURE.csv`: authoritative initialized prefix and failed sufficient comparison.
- `FIRST_EVENT_CERTIFICATE.json`: independently validated first training-gate event.
- `INPUT_AUDIT.json`, `PORTABLE_INPUT_CHECKS.json`: provenance and exact-array checks.
- `inputs/`: exact binary instances and cubic-reference coefficients. These coefficients are not assumed accurate; the verifier bounds their defect.
- `provenance/`: four unmodified repository scripts from the pinned commit. Their comments/numerical interpretations are not proof premises.
- `code/`: executable verifiers.
- `SHA256_MANIFEST.json`: integrity hashes; `code/check_manifest.py` checks them.

## Exact instance and provenance

Commit: `ae16f9c0e4e51269c624ffb196184568f2cfe6d9`.
Parameters: h=200, N=4000, means (3,0)/(0,2), sigma=0.15, seed=0, epsilon=0.001. Loss is mean squared reconstruction error divided by two. Parameters are all U rows followed by all W rows. The strong cohort has 71 neurons and uses the repository's ID-only t=4 rule. The central unit probe is −85 degrees. Q_G is the repository's uncorrected statistic, not the k-corrected variant.

The initial weights equal the archived rows exactly. Re-evaluating their trigonometric construction can change bits, so regeneration is not used for initialization. The old archive contains no raw X or independent X hash. Supplied X matches the committed generator under this runtime; historical bitwise dataset identity is not independently established. Every theorem here concerns the supplied exact binary X. Data and initialization routines both restart seed 0; no independence or high-probability claim follows.

## Reproduction

Run from this directory in a **copy** if you want to preserve the delivered outputs. Python packages: NumPy, mpmath, SymPy, SciPy and Matplotlib (the latter two for importing the original provenance script). TeX requires a conventional pdfLaTeX installation.

```sh
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/tmp/swingby-certificate-mpl
python3 code/check_manifest.py
python3 code/audit_portable_inputs.py
python3 code/verify_interval_arithmetic.py
python3 code/verify_local_formulas.py
python3 code/check_recomputed_steps.py
python3 code/certify_first_event.py
python3 code/certify_wide_witnesses.py --data inputs/canonical_witness_states.npz --output WIDE_STATIC_WITNESS_CERTIFICATE.json
python3 code/certify_wide_snapshots.py --data inputs/canonical_snapshot_states.npz --output WIDE_SNAPSHOT_CERTIFICATE.json
```

The full 2,500-step local validation takes several minutes:

```sh
python3 code/validate_wide_local.py --reference inputs/canonical_late_reference.npz --radii WIDE_LOCAL_TUBE_RADII.csv --output . --workers 3
python3 code/replay_wide_enclosure.py --directory .
```

The first command constructs interval coefficients and checks the smaller `1e-5` entry. The second independently checks the same prescribed tubes starting at the selected `1.2e-5` entry, using 60-digit interval exponentials and exact rational steps. No hypothesis on integrator accuracy is used.

The initialized verifier starts at exact radius zero. Its purpose is to expose the actual limit of this sufficient enclosure, not to generate a success claim:

```sh
python3 code/validate_initial_segment.py --reference inputs/canonical_initial_reference.npz --radii INITIAL_SEGMENT_TUBES.csv --output . --workers 3
```

The direct run writes `INITIAL_SEGMENT_CERTIFICATE.json` and `INITIAL_SEGMENT_ENCLOSURE.csv`. The delivered authoritative joined-and-replayed result is `INITIAL_FLOW_CERTIFICATE.json`. To independently recompute all smooth rates:

```sh
python3 code/recompute_smooth_rates.py --reference inputs/canonical_initial_reference.npz --radii INITIAL_SEGMENT_TUBES.csv --output . --workers 3
```

For a fast scalar replay of the delivered interval coefficient audit:

```sh
python3 code/replay_initial_enclosure.py --directory .
```

The input prefix contains all knots through time 1.1. The failed comparison occurs earlier. Its proof is not extrapolated to time 3.4. The nondecreasing upper comparison exceeding the entry budget is **not** a lower bound on actual trajectory error.

```sh
pdflatex -interaction=nonstopmode -halt-on-error INITIALIZED_CANONICAL_AUDIT.tex
pdflatex -interaction=nonstopmode -halt-on-error INITIALIZED_CANONICAL_AUDIT.tex
```

## Arithmetic contract

Inputs are exact stored Float64 numbers. Basic operations and square roots must use standard IEEE round-to-nearest behavior without unsafe fast-math. Dot/sum errors use explicit gamma bounds with outward rounding; interval transcendental operations use mpmath. Exact-rational arithmetic tests and symbolic derivative tests are regression checks, not substitutes for the inequalities proved in the supplement. All actual training selectors, including compatible endpoint selectors, are over-enclosed. No global uniqueness claim is required by the local theorem.

A normalization concern raised during audit was a parenthesis-reading error, not an error in the original computation. The certainly-active and uncertain contributions each received exactly one factor 1/N. An independent recomputation of all smooth rates agreed within 1.8e-15; the expression was rewritten more explicitly, and the scalar comparison was replayed from radius zero.

The floating-point prototype profiles are diagnostics only. They are not invoked by any proof. Their role is choosing candidate radii and free comparison slopes, which are subsequently checked by interval inequalities.
