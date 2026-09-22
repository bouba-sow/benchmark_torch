# CLAUDE.md

Guidance for Claude Code in this repository. Details live in `README.md` and `docs/`
(`architecture.md`, `running.md`, `data.md`, `dependencies.md`).

## What this is

A NeuroAI benchmark: how well LLM hidden states (embedding track, ridge) and scalar
surprisal (surprisal track, pooled LME) predict human neural and behavioral responses, across
model families, sizes and checkpoints. 2x2 design: {neural, behavioral} x {embedding,
surprisal}. Everything is identical across models; only per-dataset configs differ.

## Environment

Use the project interpreter: `.venv/bin/python` (Python 3.12, `uv sync`). The system Python
is older and rejects some syntax. LME fits need R via `pymer4`. Do not run `uv sync` as a
side effect of unrelated work.

## Sources of truth

* Which datasets are in the benchmark: `neuroai/analysis/benchmark_common.py::DATASETS`. A dataset
  having an adapter or config does not make it active. Check that file first.
* Which models are swept: `neuroai/models/registry.py::LPP_FAMILIES`. SLURM array sizes are
  derived from it; never hard-code model counts.
* How to submit jobs: `scripts/submit.py` and `scripts/slurm/stages.yaml`
  (defaults: 2 h, no QoS; dry run unless `--submit`).

## Working rules

* Do not touch `outputs/`, `outputs_jeanzay/` or `data/` unless explicitly asked; they hold
  expensive results and raw data. Use synthetic fixtures or a temporary output root to test.
* Restructuring is behavior-preserving: do not change formulas, filters, splits, layer
  conventions, ceilings or stimulus alignment while reorganizing.
* Surprisal headline = held-out `delta_loglik_test / n_obs_test` at the best layer,
  normalized by a delta-log-likelihood ceiling (older prose that says delta-Pearson-r is
  stale). Across-dataset composites use the median.
* Sweep CSVs (`*_lme_by_layer.csv`) can mix 26- and 29-column rows; read them with
  `neuroai/analysis/benchmark_common.py::read_lme_by_layer_csv`, never plain `pd.read_csv`.
* Noise-ceiling tables: reports read `outputs/noise_ceilings/noise_ceilings_summary.csv`
  first; `ceilings/correlation.py` writes to `outputs_jeanzay/noise_ceilings` by default, so
  pass `--out-dir` explicitly.
* Vendored trees (`forgetting-transformer/`, `moduleformer/`, `stickbreaking-attention/`)
  are patched and imported by the model loaders; do not replace them with upstream copies.
* Set the `*_CACHE_DIR` variables (see `scripts/slurm/jz/jz_cache_env.sh`) before running a
  script by hand; unset `${VAR}` in a config path creates a literal `${VAR}` folder.
* Large per-word or per-electrode arrays go to `$SCRATCH` when it is set
  (`neuroai/benchmarks/base.py::word_predictions_root`); do not let them land in `$WORK`.

## Paper

`paperwriting/` is the LaTeX source. When a table, figure or number changes, regenerate it
from the benchmark outputs (`scripts/figures/`, `outputs/benchmark_summary/`,
`outputs/noise_ceilings/`) rather than editing the LaTeX by hand.
