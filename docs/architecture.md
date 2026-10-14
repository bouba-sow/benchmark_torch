# Architecture

```
datasets/  ->  benchmarks/  ->  regressors/  ->  scripts/summarize/  ->  report + figures
adapter        one module per    ridge.py (embedding track)   per-dataset scores
per corpus     dataset+track     lme.py / *_pooled_lme.py     + noise-ceiling normalization
                                 (surprisal track)
models/registry.py: every model, checkpoint and family (array index = model x checkpoint)
features/: hidden-state and surprisal extraction, disk caches
```

* **`neuroai/datasets/`**: one adapter per corpus. It maps raw stimuli and responses to
  word-level tables (and, for neural data, to events/epochs).
* **`neuroai/benchmarks/`**: wires an adapter and a regressor into a runnable benchmark for
  one dataset and track (`<dataset>_embed.py`, `<dataset>_lme.py`, ...).
* **`neuroai/regressors/`**: the scoring methods. `ridge.py` (nested-CV ridge on hidden
  states), `lme.py` and the per-dataset pooled-LME modules (held-out Δ log-likelihood of
  reading time or neural amplitude on scalar surprisal), and the `*_ceiling.py` noise-ceiling
  estimators.
* **`neuroai/models/registry.py`**: `LPP_FAMILIES` is the single source of truth for what is
  swept. `flatten_family_tasks(family)` gives the (model, checkpoint) list whose index is the
  SLURM array index, which is why `scripts/submit.py` never hard-codes model counts.
* **`neuroai/analysis/benchmark_common.py`**: the dataset registry (`DATASETS`), model groups and
  colours, ceiling routing and the ragged-CSV reader used by all summarizers.

## Metrics

* Embedding track: cross-validated Pearson/Spearman r between predicted and observed
  responses at the best layer, divided by the dataset's split-half correlation ceiling.
* Surprisal track: held-out Δ log-likelihood per observation
  (`delta_loglik_test / n_obs_test`) of the model with surprisal against the baseline
  without it, divided by a Δ log-likelihood noise ceiling. The reported layer (or lag / fROI-layer)
  is the one with the best TRAINING Δ log-likelihood (`delta_loglik_train`), then scored on the
  held-out set, so the layer is never selected on the data it is scored on
  (`neuroai.analysis.benchmark_common.select_best_row`; set `LAYER_SELECTION = "test"` to
  reproduce the earlier, optimistic max-held-out rule).
* Across datasets: median of the normalized scores; model groups are LLM,
  Memory-Constrained, Constrained-BabyLM and BabyLM.

## Adding a dataset

1. Adapter in `neuroai/datasets/`, benchmark module(s) in `neuroai/benchmarks/`, a config
   in `configs/`, and `run_<dataset>_*_sweep.py` scripts following an existing dataset.
2. A shared runner under `scripts/slurm/jz/` and one entry per stage in
   `scripts/slurm/stages.yaml`.
3. Register it in `neuroai/analysis/benchmark_common.py::DATASETS` and add a loader in
   `scripts/summarize/benchmark.py`; add a noise-ceiling handler in
   `scripts/ceilings/correlation.py`.
