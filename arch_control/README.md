# arch_control: matched linear-vs-ReLU control experiments

This code tests whether the monotone compositional probe and the off-cone
reversals come from the ReLU activation or from the width and initialization
regime. The hypotheses, metrics and decision rules are in
[`PROTOCOL.md`](PROTOCOL.md). Freeze them with a commit before running.

## Setup

```bash
cd Swing-by-TwoLayerReLU            # repository root; the code imports the archived modules
pip install torch numpy matplotlib   # any recent torch; CUDA strongly recommended
python -m arch_control.selftest      # about 1 min; all lines must say PASS
```

## Workflow at a glance

1. Unzip into the repository root. Run `python -m arch_control.selftest` locally;
   CPU is fine.
2. **Freeze commit, pushed before any run.** It contains `arch_control/`, with
   `PROTOCOL.md` and the code. The runner refuses to start unless the frozen
   files are committed and unmodified.
3. Run on a GPU, for example Colab: open `colab_run.ipynb`, which clones the
   pushed repo. Write results to Google Drive; the runner is resumable.
4. **Results commit.** It contains the summaries: tables, figures, `REPORT.md`,
   `VALIDATION.json`, manifests and `run_summary.csv`. The raw `runs/` arrays
   are gitignored; archive them separately for the anonymous supplement.

## Run

```bash
git add arch_control && git commit -m "arch_control: freeze protocol"   # BEFORE any protocol run

python -m arch_control.run --all --dry-run          # run counts and work per experiment

# priority order given the deadline: headline, validation, step size, positive control
python -m arch_control.run --experiments E1_canonical E5_stepsize E4_positive_control --device cuda
python -m arch_control.validate                     # V1 + V4; writes results/VALIDATION.json
python -m arch_control.run --experiments E2_grid E3_regime --device cuda

python -m arch_control.analyze                      # tables, figures, REPORT.md
```

The runner is resumable: finished runs are skipped. It prints an ETA after the
first batch. Useful options:

* `--device cuda:1` selects a GPU.
* `--max-batch` / `--mem-gb` limit the batch size.
* `--threads N` sets the CPU thread count.
* `--out DIR` writes results elsewhere, for example Google Drive on Colab. Pass
  the same `--out` to `validate` and `analyze`.
* `--allow-dirty` bypasses the frozen-files check. Any run made this way must be
  recorded as a deviation.

## Compute

| experiment | runs | work (N x h x steps) |
|---|---|---|
| E1_canonical | 20 | 1.6e12 |
| E5_stepsize | 4 | 2.4e11 |
| E4_positive_control | 40 | 1.4e11 |
| E2_grid | 90 | 3.6e12 |
| E3_regime | 640 | 2.3e13 |

The protocol dtype is float64, matching the archive.

* **CPU.** A 2-thread cloud CPU processed about 7e7 units/s, so the full suite
  would take about 115 h there. E1+E5 would take about 7 h. A many-core
  workstation is several times faster, because runs are batched.
* **E4.** E4 is tiny (width 2 or 4) but has 300k steps per run, so it is
  dominated by Python per-step overhead. On the same 2-thread CPU it took about
  10 min per batch, 2 batches in total. A GPU does not speed it up much.
* **GPU.** The dynamics are memory-bound batched matmuls, so a GPU is the
  intended device. Consumer cards have slow float64 but still run far faster
  than CPU.
* **Apple MPS.** There is no float64 on MPS. Use `--device cpu`, or
  `--dtype float32`, which is not a protocol run.

## Outputs (`arch_control/results/`)

```
<exp>/runs/<run_id>.npz    E_grid (T x 720), named probes, losses, spec; params for E1 seed 0
<exp>/manifest.jsonl       run id, spec, sha256, device, dtype, versions, git commit, protocol hashes
<exp>/run_summary.csv      one row per run: counts per threshold, named-probe S / rise / t_min / excess
<exp>/direction_stats.npz  per-direction S, rise, final excess, t_min
tables/*.tex               booktabs tables (E1, E2, E3, E4)
figures/*.pdf|png          fig_E1_matched (headline), fig_E2_grid, fig_E3_regime,
                           fig_E3_paired, fig_E4_positive_control
REPORT.md                  all tables plus H1-H3 / P1 / V2-V5 outcomes
VALIDATION.json            V1 archive reproduction, V4 detector equivalence
```

## Notes

* The 720-direction grid is not the archived 1022-direction phase-diagram grid,
  which adds 0.05 degree refinement near the weak-selector sectors. Off-cone
  counts are therefore not comparable to "230". V1 reproduces the 230
  separately.
* Nothing in `experiments.py`, `metrics.py` or `simulate.py` should change after
  the freezing commit. Doing so changes run ids and protocol hashes. Log it in
  PROTOCOL.md Section 6.
