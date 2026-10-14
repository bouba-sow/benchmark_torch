# Running the benchmark

## One entry point per stage

`scripts/submit.py` submits any (dataset, track, stage) for any registry family. Stages are
listed in `scripts/slurm/stages.yaml` (`python scripts/submit.py list`):

| Track | Stage | What it does |
|---|---|---|
| embedding | `fit` | extract hidden states and fit the ridge model, one array task per model/checkpoint |
| surprisal | `extract` | GPU: per-layer logit-lens surprisal and perplexity, cached |
| surprisal | `fit` | CPU: pooled mixed-effects fit per layer (R/pymer4), needs the `extract` cache |
| surprisal | `refit` | LPP only: refit the last (and best) layer with saved word predictions, for the ROI figure (`--env REFIT_LAYERS=last` for last layer only; see the runner header for the prep scripts) |

```bash
python scripts/submit.py geco surprisal extract --family gpt2 pythia          # dry run (default)
python scripts/submit.py geco surprisal fit --family gpt2 --measure GP --submit
python scripts/submit.py provo embedding --all-families --submit
python scripts/submit.py lpp embedding --family opt --indices 0-3,7 --submit  # a subset of models
python scripts/submit.py geco surprisal fit --family gpt2 --summary --submit   # summary step only
python scripts/submit.py tuckute embedding --family pythia_checkpoints --submit
```

* The array is built from the model registry (`neuroai/models/registry.py`), so every model
  and checkpoint of a family is covered. Use `--indices` to run a subset, e.g. after some
  models already finished.
* `--measure FP|GP|SP` (eye-tracking datasets) and `--task NR|SR|TSR` (ZuCo Reading) pick
  the variant.
* Models with at least 20 B parameters (`large_model` in `stages.yaml`) automatically get
  2 GPUs and are submitted as their own array.
* `--all-families` skips `defaults.excluded_families` (checkpoint sweeps and MoLM); name
  such families explicitly with `--family`.
* Chain stages with `--dependency afterok:<jobid>`: `extract` must finish before `fit`.

## Job scripts

`scripts/slurm/jz/` holds what the jobs run, organized as `embedding/` and `surprisal/` (each
with `neural/` and `behavioral/` runners), `perplexity/` (one script per dataset, plus per-checkpoint
Pythia), and `utils/` (setup, ceilings, DLT, data prep). See `scripts/slurm/jz/README.md`.
Perplexity is only computed there for datasets whose surprisal extraction does not already
store it:

```bash
sbatch scripts/slurm/jz/perplexity/naturalstories.sh
```

## Resources

Default: 2 h, no QoS. Profiles (`cpu`, `gpu`) and per-stage values live in
`scripts/slurm/stages.yaml`; override any of them on the command line: `--time`, `--account`,
`--partition`, `--qos`, `--gres`, `--cpus`, `--constraint`, `--mem`, `--max-concurrent`,
`--env KEY=VAL` (for example `--env N_WORKERS=8` for the LME worker count).

The account names in `stages.yaml` are those of the original cluster; change them once for
your site.

## Without SLURM

Every stage is a plain Python script driven by a config, for example
`scripts/run/surprisal/behavioral/provo_fit.py --config configs/surprisal/behavioral/provo_fp.yaml`, with the same
`--family`/`--array-index` arguments the runners pass. `python scripts/run/<track>/<neural|behavioral>/<dataset>….py --help`.

## After the sweeps

```bash
python scripts/maintenance/audit_sweep_gaps.py          # which registry models still lack results, with submit.py commands
python scripts/summarize/benchmark.py       # outputs/benchmark_summary/<dataset>/
python scripts/summarize/report.py # reports/benchmark_report.md
```

Noise ceilings are precomputed by `scripts/ceilings/correlation.py` (correlation scale) and
`scripts/ceilings/dll_*.py` (Δ log-likelihood scale); the report reads
`outputs/noise_ceilings/`.

## DLT costs (optional)

Dependency-locality costs use spaCy in a separate environment (`.venv-dlt`) because its
dependencies conflict with the main one: export the words with the main environment, parse
with `.venv-dlt`, then join (`scripts/dlt/costs.py`, `scripts/dlt/join_costs.py`).
