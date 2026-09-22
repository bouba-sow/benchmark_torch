# Jean Zay job scripts

Jobs are submitted with `python scripts/submit.py` (stage table: `../stages.yaml`); the
scripts here are what those jobs run.

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
jz_cache_env.sh                                 cache layout + R/GCC modules, sourced by the jobs
```

The runners are not run directly: they define the functions that `submit.py` calls once per
array task (`<dataset>_<...>_run_array_task`, plus a `*_run_summary` for summaries).

## Perplexity

Datasets whose surprisal extraction already stores perplexity (GECO, MECO, Provo, OneStop,
Natural Stories fMRI, Tuckute, Fedorenko 2016, Pereira) need nothing here. For the others
(Broderick, LPP, MEG-MASC, Natural Stories SPR, Nieuwland, Podcast, UCL Reading, ZuCo Reading)
there is one script per dataset that computes it for every registry model, then the two
CBR-RNN checkpoints, then rebuilds the summary. Finished models are skipped, so a re-run after a
time-out simply continues.

```bash
sbatch scripts/slurm/jz/perplexity/nieuwland.sh
PPL_EXTRA_ARGS="--family opt" sbatch --gres=gpu:2 scripts/slurm/jz/perplexity/nieuwland.sh  # 20B+ models
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
data_prep/podcast/electrode_selection.sh      Podcast electrode selection
```

The header of each script lists its arguments and defaults; the dataset is always an argument, so
there is one script per job type rather than one per dataset.
