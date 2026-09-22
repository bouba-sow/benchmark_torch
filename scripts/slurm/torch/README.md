# NYU Torch job scripts

Converted from `../jz/` (Jean Zay) — see `docs/torch_cluster.md` for the cluster reference this
conversion is based on. Jobs are submitted with `python scripts/submit.py` (stage table:
`../stages.yaml`, which points at the runners here); the scripts here are what those jobs run.

```
embedding/{neural,behavioral}/<dataset>.sh      runner for the embedding track (extract + ridge fit)
surprisal/{neural,behavioral}/                  runners for the surprisal track
    <dataset>_extract.sh / <dataset>_fit.sh       neural: GPU surprisal extraction / CPU LME fit
    <dataset>.sh                                  behavioral: both stages in one runner
    lpp_refit.sh                                  LPP refit with saved word predictions (best/last layer)
surprisal/babylm_checkpoints_prefetch.sh        BabyLM checkpoint surprisal prefetch (array over dataset groups)
perplexity/<dataset>.sh                         perplexity for every registry model (see below)
perplexity/pythia_checkpoints.sh                Pythia per-training-checkpoint perplexity (NS SPR, UCL, ZuCo Reading, Nieuwland; array over sizes)
utils/                                          setup, ceilings, DLT, data prep
torch_cache_env.sh                              cache layout + R module, sourced by the jobs
```

The runners are not run directly: they define the functions that `submit.py` calls once per
array task (`<dataset>_<...>_run_array_task`, plus a `*_run_summary` for summaries).

## Differences from Jean Zay (`../jz/`)

* Single `$SCRATCH` filesystem, no `$WORK`/`$SCRATCH` split — `BENCHMARK_ROOT` defaults to
  `$SCRATCH/Documents/benchmark`.
* Accounts: `torch_pr_287_cds` / `torch_pr_287_general` (`--account=`, required). `stages.yaml`'s
  `cpu`/`gpu` profiles use `torch_pr_287_general` with `--partition=cs` (CPU) and
  `--constraint='h100|h200|a100'` (GPU, **L40S excluded** — no `sm_89` kernels in this repo's
  CUDA 12.4 wheel).
* **No SLURM-managed MPS on Torch** (`sinfo` lists no `mps` feature on any partition). The
  Broderick embedding-fit job (`embedding/neural/broderick.sh`) forces `N_SUBJECTS_BATCH=1`
  (sequential per-subject) instead of Jean Zay's concurrent multi-subject batching — see the
  script's "CUDA MPS note" header for why.
* R via `module load r/4.6.1` (Torch), not `r/4.5.2 + gcc/11.5.0` (Jean Zay) — no separate gcc
  module needed here.
* Env scrub (`unset LD_LIBRARY_PATH PYTHONPATH`) added to `torch_cache_env.sh` — Torch's login
  Anaconda base pollutes `LD_LIBRARY_PATH` with an old `libcrypto` that breaks `import hashlib`
  under `.venv` otherwise.
* Output roots renamed `outputs_jeanzay/` -> `outputs_torch/` throughout, so the two clusters'
  results never collide on disk.
* `forgetting-transformer`/`alibi`/`stickbreaking`/`hard_{2,3,5}gram` families are currently
  **broken on Torch regardless of CPU/GPU** — Triton JIT-compiles a CUDA shim at import time and
  Torch compute nodes have no Python dev headers (`Python.h: No such file`, unresolved; see
  `docs/torch_cluster.md` "Known gotchas" #4).

## Perplexity

Datasets whose surprisal extraction already stores perplexity (GECO, MECO, Provo, OneStop,
Natural Stories fMRI, Tuckute, Fedorenko 2016, Pereira) need nothing here. For the others
(Broderick, LPP, MEG-MASC, Natural Stories SPR, Nieuwland, Podcast, UCL Reading, ZuCo Reading)
there is one script per dataset that computes it for every registry model, then the two
CBR-RNN checkpoints, then rebuilds the summary. Finished models are skipped, so a re-run after a
time-out simply continues.

```bash
sbatch scripts/slurm/torch/perplexity/nieuwland.sh
PPL_EXTRA_ARGS="--family opt" sbatch --gres=gpu:2 scripts/slurm/torch/perplexity/nieuwland.sh  # 20B+ models
```

## utils/

```
ceilings/correlation.sh [dataset ...]         noise ceilings, correlation scale (embedding track)
ceilings/dll.sh <dataset|all> [--measure M]   noise ceilings, delta-log-likelihood scale (surprisal track)
dlt/costs.sh [dataset ...]                    dependency-locality costs (export words, spaCy parse)
dlt/analysis.sh corr|loglik|difficulty ...    analyses combining DLT costs with the sweep results
setup/prefetch_models_hf.sh                   Hugging Face weights + Harvard-Oxford atlas (needs internet)
setup/dlt_env.sh                              one-time .venv-dlt (spaCy) setup
data_prep/megmasc_prime_cache.sh              MEG-MASC epoch cache
data_prep/podcast_electrode_selection.sh      Podcast electrode selection
```

The header of each script lists its arguments and defaults; the dataset is always an argument, so
there is one script per job type rather than one per dataset.
