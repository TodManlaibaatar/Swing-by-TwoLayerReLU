# Reproduce the local certificate

This bundle contains exact binary proof inputs. No repository checkout or training rerun is required for the local certificate. The canonical initialized reachability statement is **not proved**.

Dependencies: Python 3.12, NumPy, mpmath; SymPy for the separate identity checks. NumPy 2.5.3 and mpmath 1.3.0 were used for the static report; see the JSON report for the recorded versions. Use IEEE floating-point operations without unsafe fast-math. CPU and BLAS implementation can change the precise interval widths; each run must actually pass its inequalities.

From this directory:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 code/certify_witness_balls.py --data canonical_witness_states.npz
python3 code/certify_snapshots.py --data canonical_snapshot_states.npz
python3 code/verify_interval_arithmetic.py
python3 code/verify_local_formulas.py
```

These checks take seconds. For the continuous local trajectory enclosure:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 code/validate_late_tube.py --reference canonical_late_reference.npz --radii LOCAL_TUBE_RADII.csv --output . --workers 4
python3 code/replay_local_enclosure.py --directory .
```

The full local validation took about nine minutes on the working machine. It computes independent uniform bounds in parallel, then propagates scalar error bounds in time order. It is **not** a fresh simulation. Each of the 2500 tube-invariance checks must pass. A validation failure must not be treated as a certificate.

`canonical_late_reference.npz` supplies the 2501 states and velocities defining the cubic reference, plus the full dataset. These arrays were checked byte-for-value against the inputs used in the executed validation. `LOCAL_TUBE_RADII.csv` supplies the proposed tubes. Their origin in a preliminary calculation is irrelevant to validity: the final validator verifies every proposed tube.

The scalar replay independently reevaluates the recurrence at 60 decimal digits with interval exponentials, exact rational timestep, and an upward enclosure of the decimal initial radius. It also verifies the required middle and terminal snapshot bounds. `validate_late_tube_executed.py` preserves the original workspace-input version used for the run; the portable version changes input loading, the radius-table location, and the initial scalar rounding only.

Compile the proof from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error SWINGBY_CROSSING_CONTINUATION.tex
pdflatex -interaction=nonstopmode -halt-on-error SWINGBY_CROSSING_CONTINUATION.tex
```

Keep the three included TeX modules beside the main source. The final PDF has 11 pages, with no unresolved references or LaTeX warnings.

## Provenance and research diagnostics

- `STATIC_WITNESS_CERTIFICATE.json`: static signed cluster contributions, full rate, and learned-block geometry.
- `SNAPSHOT_CERTIFICATE.json`: full-network error gaps and sector transport.
- `LOCAL_FLOW_CERTIFICATE.json` / `LOCAL_FLOW_ENCLOSURE.csv`: local continuous-flow validation and every step bound.
- `CANONICAL_WITNESS_PROVENANCE.json` and `SOURCE_HASHES.json`: source/reference provenance.
- `SHA256_MANIFEST.json`: hashes of bundle payload files, excluding the manifest itself and build intermediates.
- Files with `DIAGNOSTIC` or `PROTOTYPE` in their names are explicitly **not certificates**. In particular the initialization-to-3.4 tube failed; the local interval validation does not repair that gap.

Historical analysis scripts in `code/` expect the original workspace layout and archived repository files. They are included for provenance; the commands above are the portable certificate entry points. The full RK4-reference generator is also included, but its numerical accuracy is not assumed by the proof.
