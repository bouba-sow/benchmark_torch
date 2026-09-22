# NeuroAI benchmark: do LLM representations predict human brain and behavior?

Code for a benchmark that measures how well the internal representations of language
models predict human **neural** and **behavioral** responses to language, across many
model families, sizes and training checkpoints, and compares human-constrained models
(BabyLM-scale, memory-/architecturally-constrained) with standard LLMs.

## The 2×2 design

Every result comes from one cell of a 2×2 grid, and every model is treated identically;
only the per-dataset configuration changes.

|                  | **Embedding track**                                   | **Surprisal track**                                                     |
|------------------|-------------------------------------------------------|-------------------------------------------------------------------------|
| **Neural**       | ridge regression of the neural signal on hidden states | pooled linear mixed-effects model on per-layer scalar surprisal         |
| **Behavioral**   | ridge regression of reading times on hidden states     | pooled linear mixed-effects model on per-layer scalar surprisal         |

Scores are normalized by each dataset's noise ceiling. Embedding track: Pearson/Spearman r
against a split-half correlation ceiling. Surprisal track: held-out Δ log-likelihood per
observation against a Δ log-likelihood ceiling. Across-dataset composites use the median.

## Datasets

| Modality | Datasets |
|---|---|
| Behavioral (reading times) | Natural Stories SPR, UCL Reading, GECO, MECO, Provo, ZuCo Reading (NR/SR/TSR), OneStop (first-pass; go-past and scan-path variants for GECO/MECO/Provo/OneStop) |
| fMRI | Le Petit Prince, Natural Stories fMRI, Pereira, Tuckute |
| iEEG / ECoG | Podcast iEEG, Fedorenko 2016 |
| EEG / MEG | Nieuwland N400, Szewczyk & Federmeier N400, Broderick EEG, MEG-MASC (Gwilliams) |

The authoritative list (slugs, metrics, which tracks are reported) is
`neuroai/analysis/benchmark_common.py::DATASETS`.

The raw datasets are too large for this repository. `data/` is expected to hold them
(see `docs/data.md` for the expected layout and where to obtain each dataset).

## Models

`neuroai/models/registry.py` lists every model and checkpoint. Groups:
standard LLMs (GPT-2, GPT-Neo/NeoX, OPT, Pythia, OLMo-2, Qwen3, Llama-3.2, Mistral),
memory-/architecturally-constrained models (Mamba, Mamba2, RWKV-4, BLOOM, FoX, CBR-RNN,
ALiBi, forgetting-transformer, stick-breaking, n-gram baselines) and BabyLM-scale GPT-2
checkpoints.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/); PyTorch uses CUDA 12.4 wheels. The
surprisal track fits mixed models with R through `pymer4`, so R must be available.

```bash
uv sync
source .venv/bin/activate
```

Dependency-locality (DLT) costs use spaCy in a separate environment (`.venv-dlt`), because
of a dependency conflict with the main environment; see `docs/running.md`.

## Running

Each dataset/track has a sweep script driven by a YAML config in `configs/`:

```bash
# embedding track (ridge on hidden states)
python scripts/run/embedding/neural/lpp.py --config configs/embedding/neural/lpp.yaml

# surprisal track (pooled LME)
python scripts/run/surprisal/behavioral/provo_fit.py --config configs/surprisal/behavioral/provo_fp.yaml
```

On a SLURM cluster every dataset, track and stage goes through one command. Array sizes
come from the model registry, so a family always covers all of its models:

```bash
python scripts/submit.py list                                        # every stage
python scripts/submit.py geco surprisal extract --family gpt2 pythia # dry run
python scripts/submit.py geco surprisal fit --family gpt2 --measure GP --submit
python scripts/submit.py provo embedding --all-families --submit
```

Defaults are 2 h and no QoS; everything is overridable (`--time`, `--gres`, `--cpus`,
`--account`, `--qos`, `--env KEY=VAL`, ...) or in `scripts/slurm/stages.yaml`.
See `docs/running.md`.

Then aggregate and report:

```bash
python scripts/summarize/benchmark.py       # per-dataset scores -> outputs/benchmark_summary/
python scripts/summarize/report.py # reports/benchmark_report.md
```

`scripts/figures/` builds the paper figures from `outputs/benchmark_summary/`.

## Repository layout

```
neuroai/            library: datasets/ (adapters), benchmarks/ (per dataset+track), regressors/
                    (ridge, LME, ceilings), models/ (registry), features/, roi/, plotting/
configs/            embedding/ and surprisal/ x neural/ and behavioral/: <dataset>[_<measure>][_insample].yaml;
                    subjects/ (frozen subject lists)
scripts/submit.py   SLURM submission for every stage        scripts/slurm/stages.yaml  stage table
scripts/run/        sweeps: embedding/ and surprisal/ x neural/ and behavioral/ (<dataset>.py, _extract.py, _fit.py)
scripts/perplexity/ ceilings/ dlt/      perplexity per dataset, noise ceilings, dependency-locality costs
scripts/summarize/  figures/ analysis/  per-dataset summaries + report, paper figures, matched-scale analyses
scripts/data/ maintenance/ dev/         data preparation, audits and diagnostics
scripts/slurm/jz/   job runners: embedding/ surprisal/ (neural, behavioral), perplexity/, utils/
neuroai/analysis/   shared helpers: dataset registry (benchmark_common.py), plot style, DLT, matched-scale stats
docs/               architecture, running, data, methodology notes; docs/history/ = old notes
reports/            generated benchmark report and figures
paperwriting/       LaTeX source of the paper
```

Vendored, patched third-party code lives in `forgetting-transformer/`, `moduleformer/`
and `stickbreaking-attention/`; see `docs/dependencies.md`.

## License and citation

MIT licensed (see `LICENSE`). The vendored trees keep their own licences. Add a citation entry before release.
