# Benchmark restructuring plan for Claude

Prepared 2026-09-18. This is a plan, not an implementation. The only project file created by this audit is this document.

## 1. Instructions and scope

The user asked Codex to understand and simplify the file organization, identify unused files, and give Claude a clear implementation handoff. **Do not apply this plan until the user tells Claude to do so.**

**`outputs/` and `outputs_jeanzay/` are excluded.** During this audit they were not traversed or read. References to their paths in source code were examined to understand dependencies. During implementation, do not move, rename, delete, rewrite, regenerate, or migrate anything in those directories. Preserve the existing output filenames, schemas, model directory names, and default destinations in the code. Do not run the summarizers, report generators, repair tools, or jobs against production outputs as a validation step. Tests should use synthetic fixtures and an isolated temporary output root. If a change needs access to existing results for verification, defer that change until the user expands the scope.

This is a behavior-preserving reorganization. Do not change formulas, data filters, train/test splits, layer conventions, layer-selection rules, noise ceilings, stimulus alignment, model registry membership, or model loading patches as part of it. Scientific corrections deserve a separate task.

Recommendations are deliberately staged. Fewer copies of logic and fewer commands to remember matter more than minimizing the literal number of Python files. Do not replace 100 readable modules with one giant switch statement or add an elaborate plugin framework.

## 2. Audit coverage and limits

The audit enumerated the top-level workspace and recursively inventoried first-party `scripts/`, `neuroai/`, `configs/`, `docs/`, `notebooks/`, `reports/`, and `paperwriting/`. It scanned first-party text sources for references, parsed Python source, compared top-level function ASTs, checked selected runtime dependencies and editable-install metadata, and read representative launchers, scientific implementations, configuration files, and project guidance. The appendix inventories individual files and groups the full shell-script list by directory.

Large raw datasets, trained weights, virtual environments, caches, and third-party trees were assessed at the directory/dependency level, not exhaustively read or hashed. PDFs and notebooks were not scientifically reviewed. Notebook names and text references were considered, but cells were not executed. No benchmark, GPU inference, R fit, job submission, cache cleanup, or remote sync was run. This is not a numerical equivalence audit or a proof that every apparently unreferenced file is unused.

Observed first-party footprint, excluding bytecode:

| Area | Observed inventory | Main issue |
|---|---:|---|
| `scripts/` | 2,001 files: 220 Python, 1,780 shell, 1 Markdown | Large repeated launcher matrix; analysis and maintenance mixed with production commands |
| `scripts/slurm/` | 1,758 shell scripts | Per-family, per-dataset, per-stage copies |
| `neuroai/` | 125 Python files | Parallel implementations and shared logic hidden in dataset-specific modules |
| `configs/` | 84 files | Dataset/method/measure variants plus experimental configurations |
| `docs/` | 2 files before this plan | Project knowledge concentrated in a large `CLAUDE.md` and historical reports |
| `notebooks/` | 7 notebooks excluding checkpoints | Exploration and historical analyses mixed together |
| `paperwriting/` | 38 inventoried files | Source, curated figures, and TeX build products mixed |
| `reports/` | 235 inventoried files | Generated reports, figures, and historical material |

All 345 first-party Python files in `neuroai/` and the top level of `scripts/` parsed successfully under the project's Python 3.12 interpreter. The system Python is 3.9 and initially rejected syntax in `neuroai/evaluation/metrics.py`; **that is not a source bug**. Use `.venv/bin/python` for project checks.

The visible `.git/` is empty and the previous `git status` failed; the handoff also says this workspace is not under Git. Do not assume a checkout can recover deleted files. Make a code-only recoverable snapshot or establish version control before implementing moves. Do not include the excluded output trees, raw datasets, environments, or caches in that snapshot.

## 3. Highest-value changes, in order

1. Write a real README and a short architecture/runbook reference; explicitly identify current commands and current scoring conventions.
2. Consolidate the SLURM launcher matrix into a small submission interface, resource profiles, and shared runtime scripts. Keep old launch paths temporarily compatible.
3. Consolidate duplicate summarization and sweep orchestration code; retain the distinct scientific adapters.
4. Resolve competing LPP implementations and the feature/ROI copies only after equivalence checks.
5. Separate development, repair, data preparation, and retired workflows from routine benchmark use.
6. Organize loose documents and scratch files; leave raw datasets, costly caches, installed environments, and patched dependencies in place initially.

The biggest file-count reduction will come from item 2, not deleting dataset adapters.

## 4. Proposed layout

Keep the existing import root `neuroai/`; do not introduce a `src/` migration at the same time. Keep data and result paths stable. This is the intended eventual layout, not a requirement to create empty directories:

```text
benchmark/
  README.md
  CLAUDE.md                    short working rules + links
  pyproject.toml
  uv.lock
  .python-version
  .gitignore
  neuroai/
    datasets/                  scientific dataset loading and alignment
    benchmarks/                scientific orchestration, retain dataset modules
    features/                  feature extraction and feature cache contracts
    models/                    registry and model adapters
    regressors/                ridge, LME, ceilings
    evaluation/                existing metrics
    roi/                       one implementation per ROI responsibility
    plotting/                  reusable visual components
    analysis/                  shared readers, score policy, aggregation, DLT
    cli/                       small argument parsing/dispatch layer
    paths.py                   centralized path resolution, no import-time mkdir
    sweep.py                   extend existing shared sweep utilities
  configs/
    datasets/                  scientific configurations, grouped by dataset
    experiments/               named ablations and historical variants
    clusters/                  resource/runtime profiles, separate from science
  scripts/
    submit.py                  proposed common scheduler submission command
    slurm/                     minimal generic stage runners + runtime setup
    maintenance/               explicit repair/cache tools, never auto-run
    data/                      download, conversion, alignment utilities
    figures/                   paper-specific figure entry points
    analysis/                  standalone exploratory analyses
    dev/                       diagnostics and hardware checks
    [temporary old wrappers]   remove only after caller migration
  tests/                       small synthetic regression/compatibility tests
  docs/                        architecture, methodology, running, dependencies
    history/                   dated handoffs and old methodological notes
  papers/                      reference papers, including BabyLM subdirectory
  notebooks/                   working exploration; historical subdirectory if useful
  archive/                     documented superseded source, not a deletion dump
  data/                        unchanged initially
  reports/                     preserved initially
  paperwriting/                preserved initially
  outputs/                     EXCLUDED, unchanged
  outputs_jeanzay/             EXCLUDED, unchanged
  [existing dependency roots] unchanged until separately validated
```

New CLI names are proposals. A minimal eventual interface could be `python -m neuroai run`, `extract`, `summarize`, `perplexity`, and `ceiling`, with dataset, track, measure/task, family, and checkpoint options. Do not implement every subcommand in one pass. Start with the repeated behavioral LME workflows. Keep heavy imports inside the selected command so help/list/dry-run do not load models, R, or data.

## 5. Root directory disposition

| Existing item(s) | Recommendation | Conditions / evidence |
|---|---|---|
| `neuroai/`, `scripts/`, `configs/` | Keep and consolidate in phases below | These contain the working benchmark |
| `README.md` | Fill in place | Currently empty; should be the human entry point |
| `CLAUDE.md` | Keep in place; shorten after extracting durable docs | Preserve patch rationale and scientific cautions; fix outdated host-specific directions |
| `HANDOFF_20260914.md` | Move to `docs/history/HANDOFF_20260914.md` | Preserve the recent fixes and unresolved work; update links |
| `naturalstories_lme_pymer4_summary.md` | Move to `docs/history/` | Treat as historical until checked against current formulas |
| `pyproject.toml`, `uv.lock`, `.python-version` | Keep together | Do not casually upgrade dependencies during refactor |
| `.gitignore` | Extend narrowly | Add project bytecode, notebook checkpoints, crash dumps, `.venv-dlt/`, cache variants, and TeX build intermediates; do not ignore useful source or all data by a blind rule |
| `.claude/`, `.agents/`, `.codex/`, `.git/` | Leave alone | Tool/session state is not benchmark source; empty visible directories do not imply deletable state |
| `rsync_to_nyu.sh`, `rsync_to_nyu.filter` | Later move together to `scripts/maintenance/` | Update command docs and relative filter lookup; never run rsync during this task; it includes `--delete` |
| `fedorenko.py` | Rename/move to `scripts/data/extract_fedorenko2016.py` | Actual extraction utility, not junk; its docstring already uses this name |
| `tmp.py` | Preserve as `scripts/data/download_nieuwland.py` after adding a safe main entry point | It downloads Nieuwland data and executes work at top level; do not import it as a test |
| `lag36.ipynb` | Move to `notebooks/lag36.ipynb` after path checks | Name alone does not establish obsolescence |
| Root PDFs | Move to `papers/` preserving names initially | Update references; preserve unpublished drafts |
| `babylm/` | Move its five reference PDFs to `papers/babylm/` | This is reference literature, not the model registry |
| `papers/` | Keep as reference library | Existing four PDFs |
| `NYU_meeting_4_6_2026_LME_Natural_Stories.pptx` | Move to `docs/presentations/` | Human-authored/research artifact; retain |
| `references.bib` | Keep until bibliographic use reconciled | The manuscript uses its own `paperwriting/iclr2027_conference.bib`; distinct bibliographies are not automatically duplicates |
| `reports/`, `paperwriting/` | Preserve current paths in initial phases | Source contains hardcoded figure destinations and LaTeX includes; no regeneration in this scope |
| `logs/`, root `submit_geco_lme_dev.log`, `submit_provo_lme_dev.log` | Retain or archive dated logs under `logs/history/` | They document previous runs; establish retention policy before removal |
| `.debug_scratch/` | Move reviewed diagnostics to `scripts/dev/` or historical archive | LPP checks import `neuroai.lpp_llm_features`; update with any consolidation |
| `_scattermoe_test/` | KEEP in place for now | Modified on audit date; MoLM performance/correctness experiments, local ScatterMoE dependency, launch scripts and profiling logs |
| `core-claude-*`, `core-hf-*` listed below | Explicit cleanup candidates | Only after retaining any wanted crash-debugging evidence; not scientific input |
| `data/`, `q3zws/`, `urvax/`, `Szewczyk2022/`, `ds002345-fmriprep/` | Keep in place pending separate dataset provenance audit | Similar names and root/data copies do not prove identity |
| `.venv/`, `.venv-dlt/`, `.uv-cache/` | Keep in place | DLT has intentional isolated environment; do not move virtualenvs |
| `.cache/`, `.brainio_cache/`, `.cache_brainscore_language/` | Keep in place | Expensive or externally sourced artifacts; filenames alone do not establish reproducibility |
| Literal `${...}` cache directories | Quarantine candidates only after cache-key/content verification | Evidence of unexpanded environment variables; see below |
| `NeMo/`, `language/`, `forgetting-transformer/`, `moduleformer/`, `stickbreaking-attention/`, `kenlm/`, `wheels/` | Dependency-level decisions below | Do not bulk-delete vendored repositories |

The two root files `_Paper__Human_LLM_Convergence_BenchMark.pdf` and `_Paper__Human_LLM_Convergence_BenchMark-1.pdf` have identical sizes but **different SHA-256 hashes**. Keep both; they are not byte-identical duplicates.

Explicit crash-dump candidates: `core-claude-109746-6`, `core-claude-2204950-6`, `core-claude-333240-6`, `core-claude-3924980-6`, `core-claude-409680-6`, `core-claude-504476-6`, `core-hf-1555120-6`. These are sparse files: apparent sizes exaggerate recoverable disk space. Report allocated blocks if estimating savings. Prefer a recoverable, individually enumerated cleanup; never a broad `core*` deletion rule applied to source trees.

Literal cache directories at the root are `${BRODERICK_FEATURE_CACHE_DIR}`, `${FEDORENKO2016_LME_FEATURE_CACHE_DIR}`, `${PODCAST_EPOCH_CACHE_DIR}`, `${PODCAST_FEATURE_CACHE_DIR}`. Do not merge by basename: compare model revision, alignment, extraction settings and array schema. Add unresolved-variable detection before new writes so these directories cannot recur. Preserve valid old caches and their names unless a separate migration is approved.

## 6. Consolidate SLURM without changing jobs

### Evidence

There are 480 shell scripts directly under `scripts/slurm/jz/`, 1,055 under its `surprisal/` subtree, 170 under `embedding/`, 25 under the separate `embed/pereira/` subtree, 22 under `perplexity/`, and 6 under `oberon/`. An additional 22 shell scripts live directly in `scripts/`.

Many family wrappers already source a dataset-specific common runner, e.g. `scripts/slurm/jz/surprisal/lpp/surprisal_mistral.sh`. Others repeat extensive runtime setup, e.g. `tuckute_embed_fox.sh` even retains a Pythia comment. Consolidate the existing mechanism instead of inventing another family of generated files.

### Required implementation

Build a small submission resolver with explicit inputs: cluster, dataset, track, stage, family/model selection, checkpoint selection, measure/task, experiment profile, and resource override. Use the existing model registry task resolvers; do not hardcode family sizes again. **Some current arrays index models, others model × checkpoint tasks. Preserve this mapping per workflow.**

Separate three concerns:

| Concern | Home | Examples |
|---|---|---|
| Scientific configuration | Dataset/experiment config | Formula, RT measure, split, context, regression grid |
| Cluster resources | Cluster profile + sparse workload exceptions | Account, partition, QoS, GPU type/count, CPUs, memory, time, concurrency |
| Runtime | Shared stage runner | Environment activation, cache variables, offline mode, threading, monitoring, command execution |

Do not expect shell variables inside `#SBATCH` directives to expand. The Python submitter should pass resolved options as an argument list to `sbatch`, or render a job script before submission. Default dry-run must only print the exact command, environment, task map, config, and resources; it must not create caches/log directories, query model hubs, or submit work.

Preserve CPU/GPU extraction-versus-fitting splits, cache-only requirements, `SUMMARY_ONLY`, `EXTRACT_ONLY`, last/best-layer refits, checkpoint selectors, task NR/SR/TSR, FP/GP/SP, multi-GPU OPT jobs, process/thread limits, `--dependency` behavior, and per-task completion markers. Keep exceptions as explicit records rather than flattening them into a single resource default.

Inventory sourced common scripts and environment exports before replacing wrappers. `jz_cache_env.sh` both loads R/GCC modules and creates directories, including an output path; never source it during a supposedly read-only dry-run. Refactor path computation separately from environment activation and directory creation.

Keep `zuco_dev_babysitter.sh` and its state contract intact until its referenced scripts have equivalents. It constructs script/config paths dynamically, submits successor jobs, and manages state. Static Python imports cannot establish whether its dependencies are unused. Check active jobs before replacing their runtime sources; do not cancel or resubmit them as part of cleanup.

Initial compatibility wrappers should preserve existing path, arguments, environment interpretation and `#SBATCH` behavior. Keep a migration table from each old shell path to its new resolved job specification. Remove wrappers only after all local references are migrated and queued/external usage is accounted for. Do not call an old script unused because it is only invoked by `sbatch` manually.

Acceptance: dry-run equivalence for at least one behavioral extraction/fit pair, LPP extraction/fit, a neural embedding job, checkpoint array, in-sample Provo ablation, ZuCo task variant, large multi-GPU model, and an excluded-dataset explicit run. Compare the resolved task tuples and resources, not just printed command shape.

## 7. Source consolidation: exact decisions

### 7.1 LPP has two real implementations

| File(s) | Evidence | Decision |
|---|---|---|
| `neuroai/lpp_encoding.py` | 2,036 lines; used by `run_lpp_checkpoint_sweep.py`, `run_lpp_encoding.py`, `join_lpp_dlt.py`, `benchmarks/lpp_lme.py`, `regressors/lpp_persubject.py` | Treat current call path as the baseline |
| `neuroai/benchmarks/fmri.py` | 1,868 lines; no static Python importer found in audited package/scripts; many identical functions to the above | Candidate superseded copy, NOT a drop-in replacement just because its location looks cleaner |
| `neuroai/lpp_llm_features.py` | 1,322 lines; used by active LPP encoding and `compute_lpp_perplexity.py`, plus debug checks | Active; do not replace with an unconditional re-export |
| `neuroai/features/llm.py` | 3,425 lines; newer general extraction/model-loading implementation | Keep; compare loader, hidden-layer, token-pooling, cache and alignment contracts against LPP copy |
| `neuroai/lpp_roi.py`, `neuroai/roi/lpp.py` | Both imported; four substantial AST-identical functions | Consolidate shared ROI functions first, keeping import aliases and file-relative path behavior |
| `neuroai/lpp_model_registry.py` | Already a 32-line compatibility re-export | Leave during migration; update local callers to `models.registry` before eventual removal |

The two encoders differ in feature-loader imports: top-level LPP calls `neuroai.lpp_llm_features`, whereas `benchmarks/fmri.py` calls `features.llm`. Moving the newer-looking file over the active one could change results. First compare complete function/class definitions and their callers; preserve active behavior. Possible eventual destination: `neuroai/benchmarks/lpp.py`, with old import paths forwarding until compatibility is demonstrated.

Identical LPP functions found include `word_hrf_basis_matrix`, `picture_block_nuisance_design`, `_extract_timeseries_masked`, `_build_layer_designs`, `_select_voxel_indices`, and others. Identical feature helpers include TextGrid parsing, word matching, Chinese pooling and some tokenizer/cache helpers. Identical helpers are evidence for extraction, not evidence that whole modules are equivalent. Relative paths and module-global state still matter.

### 7.2 Houston and excluded datasets

Houston is retired from the canonical reporting registry, but `neuroai/datasets/__init__.py` eagerly exports Houston symbols. Its implementation exists in both `pipeline.py`/`neural_pipeline.py` and `benchmarks/behavioral.py`/`benchmarks/ieeg.py`. `run_behavioral_encoding.py` imports the former pair despite its broad name. Keep shared code until its consumers are separated; then archive the Houston-specific runners/configs as a complete historical workflow. Remove eager legacy imports from package initialization only with compatibility checks.

Do not remove Ryskin, Michaelov, Broderick, SPaRRKULEE, MEG-MASC, or ZuCo EEG adapters solely because the report registry excludes them. Record a status and exclusion reason in a lightweight dataset registry. Keep explicit experimental runs possible. Szewczyk has scripts and recent handoff work but no current canonical report entry; classify it as experimental/unregistered, not abandoned.

`neuroai/benchmarks/eeg_erp.py` serves Nieuwland and exports `extract_sentence_features` consumed by other N400 encoders. A dataset-status filter must not archive shared dependencies with an excluded experiment.

### 7.3 Behavioral LME and embedding orchestration

GECO, MECO, Provo, OneStop, and ZuCo Reading LME adapters already import `UCLReadingLmeConfig` and `run_layer_sweep` from `benchmarks/ucl_reading_lme.py`. Extract this actual shared machinery into a clearly named shared behavioral module, with a compatibility import from the old location. Keep each corpus's data preparation, edge filtering, measure selection and grouping rules in its own adapter.

Consolidate repeated argument parsing, configuration resolution, model/checkpoint loops, logging, resume bookkeeping and summary dispatch in `run_*_lme_sweep.py` and `run_*_embed_sweep.py`. Extend `neuroai/sweep.py` rather than creating several competing sweep frameworks. Start with GECO/MECO/Provo/OneStop, which already share infrastructure. Natural Stories and neural datasets can retain distinct adapters.

The identical `_himalaya_gpu_warmup` in `meco_embed.py`, `ucl_embed.py`, and `zuco_reading_embed.py` is a small extraction opportunity. Keep backend choice, target averaging and ridge grids unchanged. Dataset-specific methods are not duplication merely because they share a name.

### 7.4 Other possible unused modules

No static first-party Python importer was found for `neuroai/datasets/base.py`, `features/base.py`, `models/dlt.py`, or `regressors/michaelov_lme.py`. These are review candidates, not verified unused files. Search notebook cells, dynamic imports, docs, scripts in data/vendor trees, and external workflows before archiving.

`models/dlt.py` deserves particular care: DLT scripts explicitly document duplicated algorithm logic to avoid environment/import conflicts. Preserve `.venv-dlt` separation. A future shared pure DLT helper must be importable without Torch, NeuralSet, Transformers, or R. Prove token-level cost equivalence before deduplicating it.

`models/surprisal.py` and `features/surprisal.py` share code, but the former remains used by Houston/behavioral compatibility paths and the latter has additional logic. Consolidate verified common primitives; do not assume signature and output equivalence for the complete module.

## 8. Analysis, aggregation and paper scripts

Move reusable code out of executable scripts so paper scripts do not import another paper script to obtain dataset paths or loaders.

Suggested focused modules under `neuroai/analysis/`: `registry.py`, `readers.py`, `scores.py`, `aggregate.py`, `dlt.py`, and `matched_scale.py`. Reuse `neuroai/plotting/` for shared styles and panels. Do not create one new file per tiny function.

Split responsibilities currently combined in `scripts/_benchmark_common.py`: dataset metadata/model grouping; CSV compatibility parsing; score/ceiling routing; plot styles/statistics. Keep the registry metadata import lightweight and independent of matplotlib and output reads. Retain a temporary re-export at the original script path for consumers.

Strong deduplication evidence from AST comparison:

- Ten `compute_*_perplexity.py` scripts have an identical `collect_summary` function. Centralize bookkeeping while preserving each corpus's text construction and perplexity convention.
- GECO/MECO/Provo/UCL/ZuCo embedding summarizers share `_collect_family`, `_plot_scaling`, `_draw_layer_profiles_ax`, and `_plot_layer_profiles_by_group`.
- Behavioral LME summarizers repeat registry metadata, metadata attachment, family collection and scaling helpers.
- Fedorenko/Michaelov/Nieuwland/Ryskin/Tuckute LME summaries share substantial family-summary and plotting logic.
- `summarize_lpp.py` is large but includes ROI/parcels/trajectory-specific behavior. Do not discard it just because `summarize_benchmark.py` also exists.

**Concrete misleading “obsolete” file:** `build_paper_figure1_surprisal_last_pareto.py` declares its CLI superseded and warns it can overwrite a correct figure with stale scoring. But `build_behavioral_surprisal_layerwise_by_model.py` and `build_paper_figure4b_layerwise_loglik.py` still import `BEHAVIORAL_DELTA_SPEC` from it. First move live metadata/loaders to the shared reader module, update imports, then archive the superseded figure entry point. Keep `build_paper_pareto_figures.py` as the current Pareto implementation. Do not run either as part of restructuring.

Keep `build_paper_figure10_provo_ohschuler.py` and `build_paper_figure11_provo_formula_ablation.py` as named ablation workflows. A version number or an old-formula filename is not proof of redundancy. Likewise compare `plot_dlt_hard_bars_v2.py` and `plot_dlt_hard_word_bars.py` before retiring either.

`test_mc_depth_effect.py` and `test_mc_perplexity_mixed_model.py` are scientific analysis scripts, not ordinary unit tests. Place them under `scripts/analysis/` with descriptive `analyze_...` names; avoid automatic test collection that reads real results.

### Preserve the actual metric, not stale prose

The earlier assistant explanation relied on historical documentation and incorrectly described the current surprisal headline as ΔPearson r. Current `summarize_benchmark.py::_best_last_dll_per_obs` selects by `delta_loglik_test` and scores `delta_loglik_test / n_obs_test`. `scripts/_benchmark_common.py` routes surprisal tracks to DLL ceilings. Δcorrelation survives in legacy fields in these paths. Existing methodology/report prose contains conflicting descriptions.

Preserve the implementation during this task. Explicitly document metric family, denominator, layer-selection scope (global or per-unit), aggregation order, target averaging, fit/test split, and ceiling type for each track. Keep external column names such as `ceiling_r` stable even when misleading; add internal metadata rather than renaming existing output columns. Do not assume one universal best-layer rule. Current NS-fMRI surprisal aggregation, for example, selects per fROI before summing.

Keep the ragged historical CSV reader and degenerate-DLL filtering in `_benchmark_common.py`. Those are deliberate compatibility/scoring behavior, not clutter. Verify with synthetic mixed-width CSV rows and edge cases. Do not replace the reader with `on_bad_lines='skip'`.

## 9. Configuration and path policy

Group scientific YAML files by dataset only after every caller has a migration mapping. Preserve fully resolved values. A modest shared-defaults mechanism with explicit named overrides is sufficient; avoid introducing an additional configuration framework solely for cosmetic cleanup.

Keep FP/GP/SP, NR/SR/TSR, `nofreq`, in-sample/out-of-sample, Oh–Schuler/original-formula, embedding/LME and CBR-RNN distinctions. Store experimental profiles explicitly. Subject manifests (`megmasc_subjects.json`, `sparrkulee_subjects.json`, `onestop_subjects_120.txt`) are scientific inputs and must remain available.

`configs/provo_lme_sweep_fp_oldformula_insample.yaml` and `configs/provo_lme_sweep_fp_insample.yaml` are byte-identical text in this snapshot, but both have shell consumers. Consolidate their underlying values only while preserving the two experiment aliases and their resolved job identity. Do not delete one filename before its callers are migrated.

Create a single tested path resolver for `BENCHMARK_ROOT`, cache roots and dataset defaults. Its import must not create directories. Reject unresolved `${VAR}` before a write, and distinguish required versus optional variables. Preserve environment override precedence. Separate this work from changing actual cache locations.

Moving scripts deeper breaks frequent `Path(__file__).resolve().parent.parent` and `parents[1]` root calculations. Some tools load registry source dynamically by filename or import sibling scripts after changing `sys.path`. Inventory all such references, including shell `source`, generated path strings, configs, notebooks, docs, sync filters and paper scripts. Root discovery must be independent of nesting depth and must work without a valid `.git/`.

## 10. Dependencies, environments and data

| Item | Observed dependency | Recommendation |
|---|---|---|
| `forgetting-transformer/` | Editable uv workspace member; installed `.pth` points to this exact root; `features/llm.py` inserts its `src` path | Keep in place. Record upstream revision and local cache/rotary/model-loader patches. A later move requires updating sources, workspace, lockfile, loaders and editable install together |
| `moduleformer/` | `features/llm.py::_load_moduleformer_direct` inserts the local directory and imports vendored classes | Keep with patches; document provenance and license. Not an unused repo |
| `stickbreaking-attention/` | Current main environment metadata points to a Git-installed package, not this checkout | Local checkout is only a review candidate; check other environments, patches and manual workflows before archiving |
| `kenlm/` | `features/kenlm_surprisal.py` dynamically loads local `model.py`; trained language-model data also live here | Keep wrapper and artifacts; later separate code from weights only with explicit path compatibility |
| `NeMo/` | Related to ASR/alignment preparation; SPaRRKULEE scripts activate a separate `acsr` environment | Optional preparation dependency; determine actual editable imports/local patches before archiving. Do not infer unused from `pyproject.toml` alone |
| `language/` | Brain-Score checkout, package metadata and historical scoring artifact | Preserve as external/reference workflow pending environment/notebook/provenance check |
| `wheels/` | Contains two compiled Python 3.12 wheels; pyproject points to similarly named wheels in the parent project's absolute path | Verify hashes and build provenance before choosing canonical location; neither copy is presumed redundant |
| `.venv-dlt/` | Explicit two-stage DLT workflow isolates spaCy/Transformers dependency conflict | Keep; document separate installation and execution commands |

Do not relocate all third-party trees to `vendor/` in phase one. That makes the root look cleaner but breaks working imports and editable environments. A documented dependency inventory is initially safer and simpler. If moved later, retain licenses, revisions, patch records and install verification; do not replace patched copies with fresh upstream checkouts.

`pyproject.toml` still has a placeholder description. Update metadata, but separate dependency cleanup from restructuring. Several directly imported scientific packages are not explicitly listed there; audit direct runtime imports before adjusting dependency groups. Preserve resolved versions and compiled-wheel compatibility. Do not use dependency installation or `uv sync` as an incidental validation step on this working environment.

Raw data duplicates need a separate manifest comparing relative paths, sizes, checksums where feasible, provenance, annotations and preprocessing versions. Examples requiring review: root `q3zws/` versus `data/q3zws/`; root `Szewczyk2022/` and three similarly named data directories; `data/tuckute/` versus `data/tuckute_fmri/`; `data/UCL/` versus `data/UCL_reading/`; Pereira raw/merged/experiment files. Do not flatten BIDS/OSF layouts, correct `perreira_fmri` spelling in place, or assume transformed files duplicate their sources.

Top-level `data/inspect_pereira.py`, `data/inspect_stimuli.py`, and `data/merge_pereira.py` can eventually move to `scripts/data/` after reading their path assumptions. Dataset download `.sh` manifests in `data/` should remain with provenance until reviewed. No raw-data content equivalence was established by this audit.

## 11. Development, maintenance, reports and documents

Move repair commands out of routine script discovery, with explicit manual-use documentation. This includes `clean_podcast_lme_corrupted_rows.py`, `recover_misdirected_scores.py`, `rename_gpt2_lme_csv_columns.py`, `prune_per_word_intermediate_layers.py`, `prep_lpp_lme_bestlayer_refit.py`, `prep_lpp_lme_lastlayer_refit.py`, and `clear_hard_ngram_cache.sh`. **Moving their source is permitted later; executing them against outputs/caches is not part of this plan.** Their names are not evidence that the repair is still needed.

Retain diagnostic coverage: `verify_fox_transformer_layers.py`, `verify_molm_layers.py`, `debug_hard_ngram_identical.py`, `check_torch_gpu_arch.py`, `.debug_scratch/*`, and the current MoLM experiment. Make GPU/network requirements explicit; keep them out of default unit tests.

Reference material: preserve both root benchmark PDFs, distinct notebook copies and bibliographies until compared. `notebooks/benchmark_overview copy.ipynb` is a comparison candidate, not a confirmed duplicate. Notebook checkpoints may contain unsaved work and need review before cleanup.

Keep `paperwriting/` paths stable. `main_iclr2027.tex` includes `sections/*` and `figures/*`; style files are manuscript dependencies. `.aux`, `.blg`, `.log`, and `.out` are eventual generated-file cleanup candidates after a reproducible build is established. Retain `.bbl` until submission/build requirements are known. Preserve curated PDFs and manuscript figures. Reports and paper figures have overlapping copies, but no generated-figure equivalence was established.

Documentation to write:

- `README.md`: purpose, dataset/track overview, supported interpreter, setup prerequisites, one small example, extraction versus fitting, submitting jobs, where results live, and links to deeper docs.
- `docs/architecture.md`: responsibilities and source of truth for datasets, models, scoring and paths.
- `docs/methodology.md`: current implementation-derived conventions, including dataset-specific differences. Label old methodology report as historical rather than silently treating it as current.
- `docs/running.md`: cluster environments, explicit offline/cache stages, resume behavior, resource overrides and dry-run examples.
- `docs/dependencies.md`: vendor patches, compiled wheels, R and DLT environment setup, provenance.
- `docs/history/`: old handoffs and methodological snapshots, retained for interpretation.

Reduce `CLAUDE.md` to working rules and these links after documents exist. Its local/NFS sync directions refer to a different host setup; on this Jean Zay workspace a local documentation edit does not call for syncing back to the same cluster. Do not add an automatic remote sync to this refactor.

## 12. Implementation phases and completion gates

### Phase A — Baseline and documentation

Record a code-only file/hash inventory, current CLI options/defaults and config values, static callers and scientific invariants. Make a recoverable source snapshot. Fill README and architecture/runbook. Record which commands have import-time side effects so even `--help` is not blindly executed. No scientific changes or file deletions.

Gate: a newcomer can identify the current entry points, and every planned move has a source, destination, caller-update list and rollback record.

### Phase B — Low-risk organization

Move loose literature, historical notes and standalone preparation/dev scripts according to the table. Update path references. Keep raw data, vendor trees, environments, caches, reports, paper and outputs in place. Crash cleanup is a distinct, explicitly enumerated decision; it need not block source reorganization.

Gate: original source is recoverable, links resolve, preparation/dev scripts use the intended project root, and no production computation was triggered.

### Phase C — Shared analysis and behavioral orchestration

Extract verified identical helpers and shared readers first. Migrate imports away from scripts-as-libraries. Separate live constants from the superseded Pareto CLI. Consolidate a small behavioral sweep slice while retaining old entry-point wrappers.

Gate: synthetic input gives equal score fields, row order where significant, layer IDs, trial counts, CSV schema and resolved settings. Preserve float values within justified tolerance; no unexplained metric changes.

### Phase D — Cluster launcher consolidation

Implement dry-run resolver, profiles and common stage runners. Migrate one representative workflow at a time. Record each old script's replacement specification. Keep resource exceptions and dynamic babysitter consumers. Only remove obsolete wrappers after the dependency and scheduler checks pass.

Gate: matched old/new resolved commands, environment and task mapping across the representative matrix in section 6. No actual submissions required for this scope.

### Phase E — Riskier implementation consolidation

Resolve LPP feature/encoder copies, shared ROI code and Houston legacy exports. Keep compatibility modules where needed. Do not replace active LPP algorithms with an alternate implementation without numerical comparison. If synthetic tests cannot establish the relevant equivalence, keep both and document the unresolved issue; production-output inspection requires a later scope expansion.

Gate: matching alignments, layer indexing, shapes, context boundaries, cache keys and toy regression outputs; affected imports still work; class module paths used by caching/serialization remain compatible.

### Phase F — Optional later cleanup

Data deduplication, vendor relocation, dependency regrouping, cache migration, historical result cleanup and broad manuscript/report regeneration are later tasks. Output directories remain excluded until the user explicitly changes that instruction. Do not make full completion of the safe source cleanup depend on these optional tasks.

## 13. Required validation and rollback

Use Python 3.12. Prefer AST parsing for broad syntax checks because it does not execute modules or write bytecode. Use `bash -n` for changed shell sources; never source them as a syntax test. Add focused synthetic tests for:

1. Old/new fully resolved configs and environment precedence, including unresolved variables.
2. Model/family/checkpoint-to-array mapping and model revision identity.
3. Historical mixed-width CSV normalization and DLL score selection/filtering.
4. Deterministic toy ridge/LME-frame preparation and baseline/full formulas; R numerical checks only in the appropriate environment.
5. Token-to-word alignment, word/section reset boundaries, layer counts, final normalization and subword aggregation for consolidated feature paths.
6. Cache-key compatibility and unchanged output schemas/resume identifiers.
7. Dry-run purity: no file writes, output traversal, network access, R/model initialization or scheduler invocation.

Archive the original file before retiring a divergent implementation. Each phase should have a reversible patch or commit, with its caller migration list. Avoid keeping both full implementations active indefinitely under “new”/“old” names; use a compatibility wrapper once equivalence is established.

Final handoff from Claude should enumerate moved/merged/archived/deleted paths, explicitly deferred files and reasons, compatibility wrappers still present, validation results and the rollback location. Confirm that `outputs/` and `outputs_jeanzay/` remained untouched. The goal is a small, documented operational interface and one implementation of each shared behavior—not a cosmetic reshuffle alone.

## 14. Inventory appendix

The following inventory is a routing aid, not an instruction to bulk-move files. Specific cautions and phase gates above override category recommendations. Individual shell filenames are listed under their parent directories because all require command/resource equivalence before retirement. “Review” never means “safe to delete.”


### A. Every top-level Python script (220)

Proposed routing is intentionally conditional. Excluded or experimental dataset status is orthogonal to this table; preserve such runners for explicit use until their full workflow is archived.

| Current file under `scripts/` | Proposed treatment |
|---|---|
| `_analyze_neural_trends_figs.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `_benchmark_common.py` | Shared library extraction → neuroai/analysis/ or neuroai/plotting/; retain compatibility imports |
| `_dlt_difficulty.py` | Shared library extraction → neuroai/analysis/ or neuroai/plotting/; retain compatibility imports |
| `_matched_scale_stats.py` | Shared library extraction → neuroai/analysis/ or neuroai/plotting/; retain compatibility imports |
| `_paper_style.py` | Shared library extraction → neuroai/analysis/ or neuroai/plotting/; retain compatibility imports |
| `analyze_last_layer_behavioral_dissociation.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `analyze_neural_trends.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `analyze_perplexity_matched_scaling.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `audit_missing_models.py` | Operational audit → scripts/maintenance/; not run against excluded outputs |
| `audit_sweep_gaps.py` | Operational audit → scripts/maintenance/; not run against excluded outputs |
| `build_behavioral_surprisal_layerwise_by_model.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `build_dlt_new_schema_scores.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `build_lpp_persubject_bold_table.py` | Preparation/conversion → scripts/data/; preserve data paths |
| `build_matched_scale_master_table.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `build_matched_scale_master_table_embed.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `build_matched_scale_regression.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `build_matched_scale_regression_by_method.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `build_matched_scale_report.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `build_matched_scale_simple_regression.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `build_paper_corr_ceiling_table.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure10_provo_ohschuler.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure11_provo_formula_ablation.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure1_surprisal_last_pareto.py` | Extract live imports first; archive superseded CLI (section 8) |
| `build_paper_figure2_hard_dlt_pareto.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure3_family_scaling.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure4_layerwise.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure4b_layerwise_loglik.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure5_roi_comparison.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure6_dlt_threshold_grid.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure8_scale_matched_comparison.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure9_pythia_checkpoint_scaling.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure_babylm_checkpoint_scaling.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure_pereira_scaling.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_figure_roi_scaling.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_noise_ceiling_summary.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_paper_pareto_figures.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `build_podcast_electrode_roi_map.py` | Preparation/conversion → scripts/data/; preserve data paths |
| `check_torch_gpu_arch.py` | Diagnostics → scripts/dev/; retain, explicit runtime requirements |
| `clean_podcast_lme_corrupted_rows.py` | Maintenance → scripts/maintenance/; DO NOT EXECUTE |
| `compare_michaelov_real_vs_ceiling.py` | Diagnostics → scripts/dev/; retain, explicit runtime requirements |
| `compute_broderick_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_critical_word_dlt_costs.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `compute_delta_loglik_noise_ceiling.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_dlt_costs.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `compute_dlt_delta_loglik.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `compute_dlt_neural_surprisal_corr.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `compute_dlt_podcast_delta_loglik.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `compute_dlt_podcast_surprisal_corr.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `compute_dlt_surprisal_corr.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `compute_ecog_delta_loglik_noise_ceiling.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_fedorenko_parcels_from_voxels.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `compute_fmri_delta_loglik_noise_ceiling.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_houston_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_lpp_delta_loglik_noise_ceiling.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_lpp_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_megmasc_delta_loglik_noise_ceiling.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_megmasc_noise_ceiling.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_megmasc_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_michaelov_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_n400_delta_loglik_noise_ceiling.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_naturalstories_kenlm_surprisal.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `compute_naturalstories_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_nieuwland_dlt_costs.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `compute_nieuwland_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_noise_ceilings.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_podcast_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_pythia_checkpoint_perplexity_3datasets.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_pythia_checkpoint_perplexity_n400.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_ryskin_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_sparrkulee_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_sparrkulee_whisper.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `compute_ucl_reading_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `compute_zuco_perplexity.py` | Perplexity/ceiling command adapter; keep dataset-specific computation |
| `count_hard_ngram_params.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `debug_hard_ngram_identical.py` | Diagnostics → scripts/dev/; retain, explicit runtime requirements |
| `dlt_critical_word_core.py` | Shared library extraction → neuroai/analysis/ or neuroai/plotting/; retain compatibility imports |
| `export_nieuwland_items_from_predictions.py` | Preparation/conversion → scripts/data/; preserve data paths |
| `generate_benchmark_report.py` | Report command → shared analysis/reporting; preserve current destinations |
| `generate_model_report.py` | Report command → shared analysis/reporting; preserve current destinations |
| `join_dlt_costs.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `join_lpp_dlt.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `join_ns_fmri_dlt.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `join_zuco_eeg_dlt.py` | DLT shared analysis + command; preserve separate DLT environment and write behavior |
| `lpp_family_slurm.py` | Reuse registry task resolver in submission interface; temporary compatibility command |
| `make_benchmark_figure.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `make_neuralset_slide.py` | Paper/presentation command → scripts/figures/; preserve destinations |
| `parse_mfa_textgrids.py` | Preparation/conversion → scripts/data/; preserve data paths |
| `plot_average_embed_scaling.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `plot_dlt_difficulty_scaling.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `plot_dlt_hard_bars_v2.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `plot_dlt_hard_word_bars.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `plot_feature_mode_comparison.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `plot_houston_neural_brain_windows.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `plot_inverted_u.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `plot_megmasc_timecourse.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `prefetch_babylm_gpt2_group.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prefetch_geco_surprisal.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prefetch_lpp_surprisal.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prefetch_meco_surprisal.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prefetch_naturalstories_surprisal.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prefetch_ns_fmri_surprisal.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prefetch_onestop_surprisal.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prefetch_podcast_lme_surprisal.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prefetch_provo_surprisal.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prefetch_ucl_reading_surprisal.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prefetch_zuco_reading_surprisal.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `prep_lpp_lme_bestlayer_refit.py` | Maintenance → scripts/maintenance/; DO NOT EXECUTE |
| `prep_lpp_lme_lastlayer_refit.py` | Maintenance → scripts/maintenance/; DO NOT EXECUTE |
| `prepare_mfa_corpus.py` | Preparation/conversion → scripts/data/; preserve data paths |
| `prime_megmasc_epoch_cache.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `prune_per_word_intermediate_layers.py` | Maintenance → scripts/maintenance/; DO NOT EXECUTE |
| `recover_misdirected_scores.py` | Maintenance → scripts/maintenance/; DO NOT EXECUTE |
| `rename_gpt2_lme_csv_columns.py` | Maintenance → scripts/maintenance/; DO NOT EXECUTE |
| `run_behavioral_encoding.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_broderick_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_broderick_lme_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_broderick_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_fedorenko2016_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_fedorenko2016_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_fedorenko2016_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_geco_embed_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_geco_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_houston_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_houston_electrode_selection.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_lpp_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_lpp_encoding.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_lpp_lme_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_lpp_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_meco_embed_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_meco_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_megmasc_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_megmasc_lme_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_megmasc_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_michaelov_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_michaelov_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_michaelov_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_naturalstories_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_naturalstories_embed_gpu_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_naturalstories_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_nieuwland_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_nieuwland_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_nieuwland_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_ns_fmri_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_ns_fmri_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_onestop_embed_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_onestop_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_pereira_embed_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_pereira_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_pereira_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_podcast_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_podcast_electrode_selection.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_podcast_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_provo_embed_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_provo_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_ryskin_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_ryskin_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_ryskin_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_sparrkulee_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_sparrkulee_lme_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_sparrkulee_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_szewczyk_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_szewczyk_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_szewczyk_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_tuckute_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_tuckute_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_tuckute_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_ucl_reading_embed_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_ucl_reading_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_zuco_eeg_checkpoint_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_zuco_eeg_lme_surprisal_extract.py` | Extraction command adapter → shared CLI; preserve cache/stage distinctions |
| `run_zuco_eeg_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_zuco_reading_embed_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `run_zuco_reading_lme_sweep.py` | Run command adapter → shared CLI gradually; preserve scientific adapter |
| `summarize_benchmark.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_broderick.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_broderick_lme.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_fedorenko2016.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_fedorenko2016_lme.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_geco_embed_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_geco_lme_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_lpp.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_lpp_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_lpp_lme.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_meco_embed_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_meco_lme_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_meco_pythia_checkpoints.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_michaelov.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_michaelov_lme.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_naturalstories_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_naturalstories_lme_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_nieuwland.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_nieuwland_lme.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_ns_fmri.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_ns_fmri_lme.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_onestop_lme_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_podcast.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_podcast_lme.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_podcast_surprisal.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_provo_embed_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_provo_lme_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_provo_pythia_checkpoints.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_ryskin.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_ryskin_lme.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_sparrkulee.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_szewczyk.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_tuckute.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_tuckute_lme.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_ucl_reading_embed_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_ucl_reading_lme_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_zuco_eeg.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_zuco_eeg_lme.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_zuco_reading_embed_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_zuco_reading_lme_families.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `summarize_zuco_reading_pythia_checkpoints.py` | Aggregation → shared readers/aggregation + thin command; never run on production outputs here |
| `test_mc_depth_effect.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `test_mc_perplexity_mixed_model.py` | Retain analysis utility → scripts/analysis/ after caller/path checks |
| `transcribe_narratives_whisper.py` | Preparation/conversion → scripts/data/; preserve data paths |
| `verify_fox_transformer_layers.py` | Diagnostics → scripts/dev/; retain, explicit runtime requirements |
| `verify_molm_layers.py` | Diagnostics → scripts/dev/; retain, explicit runtime requirements |
| `whisper_json_to_textgrid.py` | Preparation/conversion → scripts/data/; preserve data paths |

### B. Package inventory

Keep scientific modules in place initially. Apply only the specific consolidation decisions in section 7; no blanket module deletion. Each list is relative to its heading.

#### `neuroai/` (9)

- `__init__.py`
- `lpp_encoding.py`
- `lpp_llm_features.py`
- `lpp_model_registry.py`
- `lpp_roi.py`
- `neural_pipeline.py`
- `perplexity.py`
- `pipeline.py`
- `sweep.py`

#### `neuroai/benchmarks/` (47)

- `__init__.py`
- `base.py`
- `behavioral.py`
- `broderick.py`
- `broderick_lme.py`
- `eeg_erp.py`
- `fedorenko2016.py`
- `fedorenko2016_lme.py`
- `fmri.py`
- `geco_embed.py`
- `geco_lme.py`
- `ieeg.py`
- `lpp_lme.py`
- `meco_embed.py`
- `meco_lme.py`
- `megmasc.py`
- `megmasc_lme.py`
- `n400_michaelov.py`
- `n400_michaelov_lme.py`
- `n400_nieuwland_lme.py`
- `n400_ryskin.py`
- `n400_ryskin_lme.py`
- `n400_szewczyk.py`
- `n400_szewczyk_lme.py`
- `natural_stories.py`
- `natural_stories_embed_gpu.py`
- `natural_stories_fmri.py`
- `natural_stories_fmri_lme.py`
- `natural_stories_lme.py`
- `onestop_embed.py`
- `onestop_lme.py`
- `pereira_embed.py`
- `pereira_lme.py`
- `podcast.py`
- `podcast_lme.py`
- `provo_embed.py`
- `provo_lme.py`
- `sparrkulee.py`
- `sparrkulee_lme.py`
- `subject_average.py`
- `tuckute_lme.py`
- `ucl_embed.py`
- `ucl_reading_lme.py`
- `zuco_eeg.py`
- `zuco_eeg_lme.py`
- `zuco_reading_embed.py`
- `zuco_reading_lme.py`

#### `neuroai/datasets/` (24)

- `__init__.py`
- `base.py`
- `broderick.py`
- `fedorenko2016.py`
- `geco.py`
- `houston.py`
- `le_petit_prince.py`
- `meco.py`
- `megmasc.py`
- `michaelov.py`
- `natural_stories.py`
- `natural_stories_fmri.py`
- `nieuwland.py`
- `onestop.py`
- `pereira.py`
- `podcast.py`
- `provo.py`
- `ryskin.py`
- `sparrkulee.py`
- `szewczyk.py`
- `tuckute.py`
- `ucl_reading.py`
- `zuco.py`
- `zuco_reading.py`

#### `neuroai/evaluation/` (2)

- `__init__.py`
- `metrics.py`

#### `neuroai/features/` (9)

- `__init__.py`
- `base.py`
- `cbr_rnn.py`
- `hidden_states.py`
- `kenlm_surprisal.py`
- `llm.py`
- `logit_lens.py`
- `surprisal.py`
- `word_frequency.py`

#### `neuroai/models/` (8)

- `__init__.py`
- `_checkpoints.py`
- `behavioral.py`
- `cbr_rnn.py`
- `dlt.py`
- `embedding.py`
- `registry.py`
- `surprisal.py`

#### `neuroai/plotting/` (3)

- `__init__.py`
- `lpp.py`
- `scaling.py`

#### `neuroai/regressors/` (20)

- `__init__.py`
- `broderick_lme.py`
- `ecog_ceiling.py`
- `fmri_ceiling.py`
- `fmri_lme.py`
- `ieeg_pooled_lme.py`
- `lme.py`
- `lpp_ceiling.py`
- `lpp_persubject.py`
- `lpp_pooled_lme.py`
- `megmasc_lme.py`
- `michaelov_lme.py`
- `n400_pooled_lme.py`
- `pereira_pooled_lme.py`
- `podcast_lme.py`
- `ridge.py`
- `sparrkulee_lme.py`
- `tuckute_pooled_lme.py`
- `ucl_reading_lme.py`
- `zuco_eeg_lme.py`

#### `neuroai/roi/` (3)

- `__init__.py`
- `lpp.py`
- `podcast.py`

Additional package-local artifacts: `neuroai/.cache/harvard_oxford_parcels/fedorenko_36c0a998c55b2858.json` and its `.npz` pair. Preserve initially; they reinforce the need to audit file-relative cache roots before consolidating ROI modules. These were inventoried by path, not interpreted.

### C. All configuration files (84)

Initial action: retain. Later route each scientific config into `configs/datasets/<dataset>/`, named ablations into `configs/experiments/<dataset>/`, and retain subject manifests alongside their dataset configurations. Keep old config aliases until shell and Python callers are migrated. These are distinct parameterizations unless equivalence is explicitly demonstrated.

- `configs/broderick_checkpoint_sweep.yaml`
- `configs/broderick_lme_sweep.yaml`
- `configs/fedorenko2016_checkpoint_sweep.yaml`
- `configs/fedorenko2016_lme_sweep.yaml`
- `configs/geco_embed_sweep_fp.yaml`
- `configs/geco_embed_sweep_gp.yaml`
- `configs/geco_lme_sweep_fp.yaml`
- `configs/geco_lme_sweep_fp_insample.yaml`
- `configs/geco_lme_sweep_gp.yaml`
- `configs/houston_rt.yaml`
- `configs/lpp_cbr_rnn.yaml`
- `configs/lpp_cbr_rnn_sweep.yaml`
- `configs/lpp_cbr_rnn_sweep_surprisal.yaml`
- `configs/lpp_checkpoint_sweep.yaml`
- `configs/lpp_checkpoint_sweep_surprisal.yaml`
- `configs/lpp_encoding.yaml`
- `configs/lpp_lme_sweep.yaml`
- `configs/meco_embed_sweep_fp.yaml`
- `configs/meco_embed_sweep_gp.yaml`
- `configs/meco_embed_sweep_sp.yaml`
- `configs/meco_lme_sweep_fp.yaml`
- `configs/meco_lme_sweep_fp_insample.yaml`
- `configs/meco_lme_sweep_gp.yaml`
- `configs/meco_lme_sweep_sp.yaml`
- `configs/megmasc_checkpoint_sweep.yaml`
- `configs/megmasc_lme_sweep.yaml`
- `configs/megmasc_subjects.json`
- `configs/michaelov_checkpoint_sweep.yaml`
- `configs/michaelov_lme_sweep.yaml`
- `configs/naturalstories_checkpoint_sweep.yaml`
- `configs/naturalstories_embed_gpu_sweep.yaml`
- `configs/naturalstories_lme_sweep.yaml`
- `configs/naturalstories_lme_sweep_insample.yaml`
- `configs/nieuwland_checkpoint_sweep.yaml`
- `configs/nieuwland_lme_sweep.yaml`
- `configs/ns_fmri_checkpoint_sweep.yaml`
- `configs/ns_fmri_lme_sweep.yaml`
- `configs/ns_fmri_lme_sweep_nofreq.yaml`
- `configs/onestop_embed_sweep_fp.yaml`
- `configs/onestop_embed_sweep_gp.yaml`
- `configs/onestop_embed_sweep_sp.yaml`
- `configs/onestop_lme_sweep_fp.yaml`
- `configs/onestop_lme_sweep_gp.yaml`
- `configs/onestop_lme_sweep_sp.yaml`
- `configs/onestop_subjects_120.txt`
- `configs/pereira_embed_sweep.yaml`
- `configs/pereira_lme_sweep.yaml`
- `configs/podcast_checkpoint_sweep.yaml`
- `configs/podcast_checkpoint_sweep_surprisal.yaml`
- `configs/podcast_lme_sweep.yaml`
- `configs/provo_embed_sweep_fp.yaml`
- `configs/provo_embed_sweep_gp.yaml`
- `configs/provo_embed_sweep_sp.yaml`
- `configs/provo_lme_sweep_fp.yaml`
- `configs/provo_lme_sweep_fp_insample.yaml`
- `configs/provo_lme_sweep_fp_ohschuler_insample.yaml`
- `configs/provo_lme_sweep_fp_ohschuler_oos.yaml`
- `configs/provo_lme_sweep_fp_oldformula_insample.yaml`
- `configs/provo_lme_sweep_gp.yaml`
- `configs/provo_lme_sweep_sp.yaml`
- `configs/ryskin_checkpoint_sweep.yaml`
- `configs/ryskin_lme_sweep.yaml`
- `configs/scaling_pythia.yaml`
- `configs/sparrkulee_checkpoint_sweep.yaml`
- `configs/sparrkulee_lme_sweep.yaml`
- `configs/sparrkulee_subjects.json`
- `configs/szewczyk_checkpoint_sweep.yaml`
- `configs/szewczyk_lme_sweep.yaml`
- `configs/tuckute_checkpoint_sweep.yaml`
- `configs/tuckute_lme_sweep.yaml`
- `configs/ucl_reading_embed_sweep.yaml`
- `configs/ucl_reading_lme_sweep.yaml`
- `configs/ucl_reading_lme_sweep_insample.yaml`
- `configs/zuco_eeg_checkpoint_sweep.yaml`
- `configs/zuco_eeg_lme_sweep.yaml`
- `configs/zuco_reading_embed_sweep_nr.yaml`
- `configs/zuco_reading_embed_sweep_sr.yaml`
- `configs/zuco_reading_embed_sweep_tsr.yaml`
- `configs/zuco_reading_lme_sweep_nr.yaml`
- `configs/zuco_reading_lme_sweep_nr_insample.yaml`
- `configs/zuco_reading_lme_sweep_sr.yaml`
- `configs/zuco_reading_lme_sweep_sr_insample.yaml`
- `configs/zuco_reading_lme_sweep_tsr.yaml`
- `configs/zuco_reading_lme_sweep_tsr_insample.yaml`

### D. Full shell inventory, grouped by directory (1,780)

For every entry: record its command, sourced helpers, environment, array mapping, resource exceptions, state files and consumers before replacing it. Do not bulk-delete any listed group. Main SLURM wrappers migrate to the submission resolver in section 6; common scripts migrate to shared runtimes; diagnostics/preparation/repair scripts follow their functional category; Oberon remains a historical cluster profile until usage is resolved. The one-off `_gen_szewczyk_embed.sh` should be retired only after the launcher generator/resolver covers its settings. Root-level sync scripts are covered in section 5.

<details>
<summary>scripts/ — 22 shell files</summary>

- `_gen_szewczyk_embed.sh`
- `clear_hard_ngram_cache.sh`
- `job_dlt_corr.sh`
- `job_dlt_corr_podcast.sh`
- `job_dlt_loglik.sh`
- `job_dlt_podcast_loglik.sh`
- `job_houston_checkpoint_sweep_jz.sh`
- `job_houston_checkpoint_sweep_oberon_array.sh`
- `job_lpp_checkpoint_sweep_jz.sh`
- `job_lpp_checkpoint_sweep_oberon_array.sh`
- `job_verify_lme_fix.sh`
- `job_verify_opt66b_provo.sh`
- `job_verify_opt66b_provo_2gpu.sh`
- `job_verify_opt66b_provo_2gpu_lme.sh`
- `job_verify_ucl_reading_paper_repl.sh`
- `prefetch_lpp_jz.sh`
- `prefetch_pythia_deduped_hf.sh`
- `recompute_all_dll_ceilings_insample.sh`
- `run_behavioral_dlt_costs_cpu.sh`
- `run_narratives_mfa_align.sh`
- `run_nieuwland_dlt_costs_cpu.sh`
- `run_podcast_dlt.sh`

</details>

<details>
<summary>scripts/slurm/jz/ — 480 shell files</summary>

- `behavioral_delta_loglik_ceiling_extra_measures.sh`
- `broderick_alibi.sh`
- `broderick_babylm.sh`
- `broderick_cbr_rnn.sh`
- `broderick_embed_bloom.sh`
- `broderick_embed_fox.sh`
- `broderick_embed_rwkv4.sh`
- `broderick_forgetting.sh`
- `broderick_gpt2.sh`
- `broderick_gptneo.sh`
- `broderick_hard_2gram.sh`
- `broderick_hard_3gram.sh`
- `broderick_hard_5gram.sh`
- `broderick_llama32.sh`
- `broderick_mamba.sh`
- `broderick_mamba2.sh`
- `broderick_mistral.sh`
- `broderick_olmo2.sh`
- `broderick_opt.sh`
- `broderick_perplexity.sh`
- `broderick_pythia.sh`
- `broderick_qwen3.sh`
- `broderick_stickbreaking.sh`
- `compute_noise_ceilings_correlation_path.sh`
- `compute_noise_ceilings_remaining.sh`
- `compute_noise_ceilings_sparrkulee.sh`
- `compute_podcast_ecog_ceiling.sh`
- `count_hard_ngram_params.sh`
- `debug_hard_ngram_identical.sh`
- `dll_ceiling_common_env.sh`
- `dlt_costs_cpu.sh`
- `dlt_costs_stage2_meco_zuco_lpp.sh`
- `dlt_costs_stage2_ryskin_michaelov.sh`
- `dlt_costs_zuco_eeg_cpu.sh`
- `dlt_difficulty_csv_export_cpu.sh`
- `dlt_difficulty_zuco_eeg_csv_cpu.sh`
- `download_pereira_dropbox.sh`
- `fedorenko2016_alibi.sh`
- `fedorenko2016_babylm.sh`
- `fedorenko2016_cbr_rnn.sh`
- `fedorenko2016_embed_bloom.sh`
- `fedorenko2016_embed_fox.sh`
- `fedorenko2016_embed_rwkv4.sh`
- `fedorenko2016_forgetting.sh`
- `fedorenko2016_gpt2.sh`
- `fedorenko2016_gptneo.sh`
- `fedorenko2016_hard_2gram.sh`
- `fedorenko2016_hard_3gram.sh`
- `fedorenko2016_hard_5gram.sh`
- `fedorenko2016_llama32.sh`
- `fedorenko2016_mamba.sh`
- `fedorenko2016_mamba2.sh`
- `fedorenko2016_mistral.sh`
- `fedorenko2016_olmo2.sh`
- `fedorenko2016_opt.sh`
- `fedorenko2016_pythia.sh`
- `fedorenko2016_qwen3.sh`
- `fedorenko2016_stickbreaking.sh`
- `houston_electrode_selection.sh`
- `houston_electrode_selection_bipolar.sh`
- `houston_neural_sweep.sh`
- `houston_perplexity.sh`
- `job_broderick_jz_common.sh`
- `job_fedorenko2016_jz_common.sh`
- `job_houston_checkpoint_sweep_jz.sh`
- `job_lpp_jz_common.sh`
- `job_megmasc_jz_common.sh`
- `job_michaelov_jz_common.sh`
- `job_naturalstories_delta_loglik_ceiling.sh`
- `job_naturalstories_jz_common.sh`
- `job_nieuwland_jz_common.sh`
- `job_ns_fmri_jz_common.sh`
- `job_podcast_jz_common.sh`
- `job_ryskin_jz_common.sh`
- `job_sparrkulee_jz_common.sh`
- `job_szewczyk_jz_common.sh`
- `job_zuco_eeg_jz_common.sh`
- `jz_cache_env.sh`
- `lpp_alibi.sh`
- `lpp_babylm.sh`
- `lpp_cbr_rnn.sh`
- `lpp_cbr_rnn_perplexity.sh`
- `lpp_embed_bloom.sh`
- `lpp_embed_fox.sh`
- `lpp_embed_rwkv4.sh`
- `lpp_forgetting.sh`
- `lpp_gpt2.sh`
- `lpp_gptneo.sh`
- `lpp_hard_2gram.sh`
- `lpp_hard_3gram.sh`
- `lpp_hard_5gram.sh`
- `lpp_llama32.sh`
- `lpp_mamba.sh`
- `lpp_mamba2.sh`
- `lpp_mistral.sh`
- `lpp_olmo2.sh`
- `lpp_opt.sh`
- `lpp_perplexity.sh`
- `lpp_perplexity_alibi.sh`
- `lpp_perplexity_babylm.sh`
- `lpp_perplexity_forgetting.sh`
- `lpp_perplexity_mamba2_force.sh`
- `lpp_perplexity_stickbreaking.sh`
- `lpp_pythia.sh`
- `lpp_qwen3.sh`
- `lpp_stickbreaking.sh`
- `megmasc_alibi.sh`
- `megmasc_babylm.sh`
- `megmasc_cbr_rnn.sh`
- `megmasc_dll_ceiling.sh`
- `megmasc_embed_bloom.sh`
- `megmasc_embed_fox.sh`
- `megmasc_embed_rwkv4.sh`
- `megmasc_forgetting.sh`
- `megmasc_gpt2.sh`
- `megmasc_gptneo.sh`
- `megmasc_hard_2gram.sh`
- `megmasc_hard_3gram.sh`
- `megmasc_hard_5gram.sh`
- `megmasc_llama32.sh`
- `megmasc_mamba.sh`
- `megmasc_mamba2.sh`
- `megmasc_mistral.sh`
- `megmasc_noise_ceiling.sh`
- `megmasc_olmo2.sh`
- `megmasc_opt.sh`
- `megmasc_prime_cache.sh`
- `megmasc_pythia.sh`
- `megmasc_qwen3.sh`
- `megmasc_stickbreaking.sh`
- `michaelov_alibi.sh`
- `michaelov_babylm.sh`
- `michaelov_cbr_rnn.sh`
- `michaelov_forgetting.sh`
- `michaelov_gpt2.sh`
- `michaelov_gptneo.sh`
- `michaelov_hard_2gram.sh`
- `michaelov_hard_3gram.sh`
- `michaelov_hard_5gram.sh`
- `michaelov_llama32.sh`
- `michaelov_mamba.sh`
- `michaelov_mamba2.sh`
- `michaelov_mistral.sh`
- `michaelov_olmo2.sh`
- `michaelov_opt.sh`
- `michaelov_perplexity.sh`
- `michaelov_pythia.sh`
- `michaelov_qwen3.sh`
- `michaelov_stickbreaking.sh`
- `naturalstories_alibi.sh`
- `naturalstories_alibi_lme.sh`
- `naturalstories_babylm.sh`
- `naturalstories_babylm_lme.sh`
- `naturalstories_cbr_rnn_lme.sh`
- `naturalstories_forgetting.sh`
- `naturalstories_forgetting_lme.sh`
- `naturalstories_gpt2.sh`
- `naturalstories_gpt2_lme.sh`
- `naturalstories_gptneo.sh`
- `naturalstories_gptneo_lme.sh`
- `naturalstories_hard_2gram.sh`
- `naturalstories_hard_2gram_lme.sh`
- `naturalstories_hard_3gram.sh`
- `naturalstories_hard_3gram_lme.sh`
- `naturalstories_hard_5gram.sh`
- `naturalstories_hard_5gram_lme.sh`
- `naturalstories_llama32.sh`
- `naturalstories_llama32_lme.sh`
- `naturalstories_mamba.sh`
- `naturalstories_mamba2.sh`
- `naturalstories_mamba2_lme.sh`
- `naturalstories_mamba_lme.sh`
- `naturalstories_mistral.sh`
- `naturalstories_mistral_lme.sh`
- `naturalstories_olmo2.sh`
- `naturalstories_olmo2_lme.sh`
- `naturalstories_opt.sh`
- `naturalstories_opt_lme.sh`
- `naturalstories_perplexity.sh`
- `naturalstories_pythia.sh`
- `naturalstories_pythia_lme.sh`
- `naturalstories_qwen3.sh`
- `naturalstories_qwen3_lme.sh`
- `naturalstories_stickbreaking.sh`
- `naturalstories_stickbreaking_lme.sh`
- `nieuwland_alibi.sh`
- `nieuwland_babylm.sh`
- `nieuwland_cbr_rnn.sh`
- `nieuwland_dlt_costs_cpu.sh`
- `nieuwland_embed_bloom.sh`
- `nieuwland_embed_fox.sh`
- `nieuwland_embed_rwkv4.sh`
- `nieuwland_forgetting.sh`
- `nieuwland_gpt2.sh`
- `nieuwland_gptneo.sh`
- `nieuwland_hard_2gram.sh`
- `nieuwland_hard_3gram.sh`
- `nieuwland_hard_5gram.sh`
- `nieuwland_llama32.sh`
- `nieuwland_mamba.sh`
- `nieuwland_mamba2.sh`
- `nieuwland_mistral.sh`
- `nieuwland_olmo2.sh`
- `nieuwland_opt.sh`
- `nieuwland_perplexity.sh`
- `nieuwland_pythia.sh`
- `nieuwland_qwen3.sh`
- `nieuwland_stickbreaking.sh`
- `ns_fmri_alibi_gpu.sh`
- `ns_fmri_babylm_gpu.sh`
- `ns_fmri_cbr_rnn_gpu.sh`
- `ns_fmri_embed_bloom.sh`
- `ns_fmri_embed_fox.sh`
- `ns_fmri_embed_rwkv4.sh`
- `ns_fmri_forgetting_gpu.sh`
- `ns_fmri_gpt2_cpu.sh`
- `ns_fmri_gpt2_gpu.sh`
- `ns_fmri_gptneo_gpu.sh`
- `ns_fmri_hard_2gram_gpu.sh`
- `ns_fmri_hard_3gram_gpu.sh`
- `ns_fmri_hard_5gram_gpu.sh`
- `ns_fmri_llama32_gpu.sh`
- `ns_fmri_mamba2_gpu.sh`
- `ns_fmri_mamba_gpu.sh`
- `ns_fmri_mistral_gpu.sh`
- `ns_fmri_olmo2_gpu.sh`
- `ns_fmri_opt_gpu.sh`
- `ns_fmri_pythia_gpu.sh`
- `ns_fmri_qwen3_gpu.sh`
- `ns_fmri_stickbreaking_gpu.sh`
- `ns_perplexity_alibi.sh`
- `ns_perplexity_babylm.sh`
- `ns_perplexity_cbr_rnn.sh`
- `ns_perplexity_forgetting.sh`
- `ns_perplexity_gpt2.sh`
- `ns_perplexity_gptneo.sh`
- `ns_perplexity_hard_2gram.sh`
- `ns_perplexity_hard_3gram.sh`
- `ns_perplexity_hard_5gram.sh`
- `ns_perplexity_llama32.sh`
- `ns_perplexity_mamba.sh`
- `ns_perplexity_mamba2.sh`
- `ns_perplexity_mistral.sh`
- `ns_perplexity_olmo2.sh`
- `ns_perplexity_opt.sh`
- `ns_perplexity_opt6630_force_2gpu.sh`
- `ns_perplexity_opt_2gpu.sh`
- `ns_perplexity_pythia.sh`
- `ns_perplexity_qwen3.sh`
- `ns_perplexity_stickbreaking.sh`
- `pereira_delta_loglik_ceiling.sh`
- `podcast_alibi.sh`
- `podcast_babylm.sh`
- `podcast_cbr_rnn.sh`
- `podcast_electrode_selection_cpu.sh`
- `podcast_embed_bloom.sh`
- `podcast_embed_fox.sh`
- `podcast_embed_rwkv4.sh`
- `podcast_forgetting.sh`
- `podcast_gpt2.sh`
- `podcast_gptneo.sh`
- `podcast_hard_2gram.sh`
- `podcast_hard_3gram.sh`
- `podcast_hard_5gram.sh`
- `podcast_llama32.sh`
- `podcast_mamba.sh`
- `podcast_mamba2.sh`
- `podcast_mistral.sh`
- `podcast_olmo2.sh`
- `podcast_olmo2_ckpt.sh`
- `podcast_opt.sh`
- `podcast_perplexity.sh`
- `podcast_pythia.sh`
- `podcast_pythia_ckpt.sh`
- `podcast_qwen3.sh`
- `podcast_stickbreaking.sh`
- `prefetch_dlt_spacy_trf.sh`
- `prefetch_lpp_jz.sh`
- `prefetch_models_hf.sh`
- `prefetch_models_hf_prepost.sh`
- `prefetch_sparrkulee_nfa.sh`
- `prefetch_sparrkulee_whisper.sh`
- `ryskin_alibi.sh`
- `ryskin_babylm.sh`
- `ryskin_cbr_rnn.sh`
- `ryskin_forgetting.sh`
- `ryskin_gpt2.sh`
- `ryskin_gptneo.sh`
- `ryskin_hard_2gram.sh`
- `ryskin_hard_3gram.sh`
- `ryskin_hard_5gram.sh`
- `ryskin_llama32.sh`
- `ryskin_mamba.sh`
- `ryskin_mamba2.sh`
- `ryskin_mistral.sh`
- `ryskin_olmo2.sh`
- `ryskin_opt.sh`
- `ryskin_perplexity.sh`
- `ryskin_pythia.sh`
- `ryskin_qwen3.sh`
- `ryskin_stickbreaking.sh`
- `setup_rpy2_jz.sh`
- `sparrkulee_alibi.sh`
- `sparrkulee_babylm.sh`
- `sparrkulee_cbr_rnn.sh`
- `sparrkulee_forgetting.sh`
- `sparrkulee_gpt2.sh`
- `sparrkulee_gptneo.sh`
- `sparrkulee_hard_2gram.sh`
- `sparrkulee_hard_3gram.sh`
- `sparrkulee_hard_5gram.sh`
- `sparrkulee_llama32.sh`
- `sparrkulee_mamba.sh`
- `sparrkulee_mamba2.sh`
- `sparrkulee_mistral.sh`
- `sparrkulee_nfa.sh`
- `sparrkulee_olmo2.sh`
- `sparrkulee_opt.sh`
- `sparrkulee_perplexity.sh`
- `sparrkulee_pythia.sh`
- `sparrkulee_qwen3.sh`
- `sparrkulee_stickbreaking.sh`
- `sparrkulee_whisper.sh`
- `submit_behavioral_measure_lme.sh`
- `submit_behavioral_measure_lme_dev.sh`
- `submit_geco_lme_dev.sh`
- `submit_provo_lme_dev.sh`
- `submit_provo_ohschuler_dev.sh`
- `szewczyk_alibi.sh`
- `szewczyk_babylm.sh`
- `szewczyk_cbr_rnn.sh`
- `szewczyk_embed_alibi.sh`
- `szewczyk_embed_babylm.sh`
- `szewczyk_embed_bloom.sh`
- `szewczyk_embed_cbr_rnn.sh`
- `szewczyk_embed_forgetting.sh`
- `szewczyk_embed_fox.sh`
- `szewczyk_embed_gpt2.sh`
- `szewczyk_embed_gptneo.sh`
- `szewczyk_embed_gptneo_20b.sh`
- `szewczyk_embed_hard_2gram.sh`
- `szewczyk_embed_hard_3gram.sh`
- `szewczyk_embed_hard_5gram.sh`
- `szewczyk_embed_llama32.sh`
- `szewczyk_embed_mamba.sh`
- `szewczyk_embed_mamba2.sh`
- `szewczyk_embed_mistral.sh`
- `szewczyk_embed_olmo2.sh`
- `szewczyk_embed_opt.sh`
- `szewczyk_embed_opt_66b.sh`
- `szewczyk_embed_pythia.sh`
- `szewczyk_embed_qwen3.sh`
- `szewczyk_embed_rwkv4.sh`
- `szewczyk_embed_stickbreaking.sh`
- `szewczyk_forgetting.sh`
- `szewczyk_gpt2.sh`
- `szewczyk_gptneo.sh`
- `szewczyk_hard_2gram.sh`
- `szewczyk_hard_3gram.sh`
- `szewczyk_hard_5gram.sh`
- `szewczyk_llama32.sh`
- `szewczyk_mamba.sh`
- `szewczyk_mamba2.sh`
- `szewczyk_mistral.sh`
- `szewczyk_olmo2.sh`
- `szewczyk_opt.sh`
- `szewczyk_pythia.sh`
- `szewczyk_qwen3.sh`
- `szewczyk_stickbreaking.sh`
- `tuckute_alibi.sh`
- `tuckute_babylm.sh`
- `tuckute_cbr_rnn.sh`
- `tuckute_embed_bloom.sh`
- `tuckute_embed_fox.sh`
- `tuckute_embed_rwkv4.sh`
- `tuckute_forgetting.sh`
- `tuckute_gpt2.sh`
- `tuckute_gptneo.sh`
- `tuckute_hard_2gram.sh`
- `tuckute_hard_3gram.sh`
- `tuckute_hard_5gram.sh`
- `tuckute_llama32.sh`
- `tuckute_mamba.sh`
- `tuckute_mamba2.sh`
- `tuckute_mistral.sh`
- `tuckute_olmo2.sh`
- `tuckute_opt.sh`
- `tuckute_pythia.sh`
- `tuckute_qwen3.sh`
- `tuckute_stickbreaking.sh`
- `ucl_reading_perplexity.sh`
- `verify_fox_transformer_layers_gpu.sh`
- `verify_layer_batch_molm_gpu.sh`
- `verify_molm_layers_gpu.sh`
- `zuco_dev_babysitter.sh`
- `zuco_eeg_alibi.sh`
- `zuco_eeg_alibi_cpu.sh`
- `zuco_eeg_alibi_gpu.sh`
- `zuco_eeg_babylm_gpu.sh`
- `zuco_eeg_cbr_rnn.sh`
- `zuco_eeg_cbr_rnn_cpu.sh`
- `zuco_eeg_cbr_rnn_gpu.sh`
- `zuco_eeg_forgetting.sh`
- `zuco_eeg_forgetting_cpu.sh`
- `zuco_eeg_forgetting_gpu.sh`
- `zuco_eeg_gpt2.sh`
- `zuco_eeg_gpt2_cpu.sh`
- `zuco_eeg_gpt2_gpu.sh`
- `zuco_eeg_gptneo.sh`
- `zuco_eeg_gptneo_cpu.sh`
- `zuco_eeg_gptneo_gpu.sh`
- `zuco_eeg_hard_2gram.sh`
- `zuco_eeg_hard_2gram_cpu.sh`
- `zuco_eeg_hard_2gram_gpu.sh`
- `zuco_eeg_hard_3gram.sh`
- `zuco_eeg_hard_3gram_cpu.sh`
- `zuco_eeg_hard_3gram_gpu.sh`
- `zuco_eeg_hard_5gram.sh`
- `zuco_eeg_hard_5gram_cpu.sh`
- `zuco_eeg_hard_5gram_gpu.sh`
- `zuco_eeg_llama32.sh`
- `zuco_eeg_llama32_cpu.sh`
- `zuco_eeg_llama32_gpu.sh`
- `zuco_eeg_mamba.sh`
- `zuco_eeg_mamba2.sh`
- `zuco_eeg_mamba2_cpu.sh`
- `zuco_eeg_mamba2_gpu.sh`
- `zuco_eeg_mamba_cpu.sh`
- `zuco_eeg_mamba_gpu.sh`
- `zuco_eeg_mistral.sh`
- `zuco_eeg_mistral_cpu.sh`
- `zuco_eeg_mistral_gpu.sh`
- `zuco_eeg_olmo2.sh`
- `zuco_eeg_olmo2_cpu.sh`
- `zuco_eeg_olmo2_gpu.sh`
- `zuco_eeg_opt.sh`
- `zuco_eeg_opt_cpu.sh`
- `zuco_eeg_opt_gpu.sh`
- `zuco_eeg_prefetch_alibi.sh`
- `zuco_eeg_prefetch_cbr_rnn.sh`
- `zuco_eeg_prefetch_forgetting.sh`
- `zuco_eeg_prefetch_gpt2.sh`
- `zuco_eeg_prefetch_gptneo.sh`
- `zuco_eeg_prefetch_hard_2gram.sh`
- `zuco_eeg_prefetch_hard_3gram.sh`
- `zuco_eeg_prefetch_hard_5gram.sh`
- `zuco_eeg_prefetch_llama32.sh`
- `zuco_eeg_prefetch_mamba.sh`
- `zuco_eeg_prefetch_mamba2.sh`
- `zuco_eeg_prefetch_mistral.sh`
- `zuco_eeg_prefetch_olmo2.sh`
- `zuco_eeg_prefetch_opt.sh`
- `zuco_eeg_prefetch_pythia.sh`
- `zuco_eeg_prefetch_qwen3.sh`
- `zuco_eeg_prefetch_stickbreaking.sh`
- `zuco_eeg_pythia.sh`
- `zuco_eeg_pythia_cpu.sh`
- `zuco_eeg_pythia_gpu.sh`
- `zuco_eeg_qwen3.sh`
- `zuco_eeg_qwen3_cpu.sh`
- `zuco_eeg_qwen3_gpu.sh`
- `zuco_eeg_stickbreaking.sh`
- `zuco_eeg_stickbreaking_cpu.sh`
- `zuco_eeg_stickbreaking_gpu.sh`
- `zuco_perplexity.sh`
- `zuco_reading_alibi_prefetch.sh`
- `zuco_reading_babylm_prefetch.sh`
- `zuco_reading_cbr_rnn_prefetch.sh`
- `zuco_reading_forgetting_prefetch.sh`
- `zuco_reading_gpt2_prefetch.sh`
- `zuco_reading_gptneo_prefetch.sh`
- `zuco_reading_llama32_prefetch.sh`
- `zuco_reading_mamba2_prefetch.sh`
- `zuco_reading_mamba_prefetch.sh`
- `zuco_reading_mistral_prefetch.sh`
- `zuco_reading_olmo2_prefetch.sh`
- `zuco_reading_opt_prefetch.sh`
- `zuco_reading_pythia_prefetch.sh`
- `zuco_reading_qwen3_prefetch.sh`
- `zuco_reading_stickbreaking_prefetch.sh`

</details>

<details>
<summary>scripts/slurm/jz/embed/pereira/ — 3 shell files</summary>

- `alpha_calibration_gpt2.sh`
- `alpha_calibration_gpt2_wide.sh`
- `embed_opt66b.sh`

</details>

<details>
<summary>scripts/slurm/jz/embed/pereira/families/ — 22 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `babylm_gpt2_checkpoints.sh`
- `bloom.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `fox.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `qwen3.sh`
- `rwkv4.sh`
- `stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/embedding/geco/ — 24 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `bloom.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `fox.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_geco_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_ckpt.sh`
- `qwen3.sh`
- `rwkv4.sh`
- `stickbreaking.sh`
- `submit_all.sh`

</details>

<details>
<summary>scripts/slurm/jz/embedding/meco/ — 24 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `bloom.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `fox.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_meco_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_ckpt.sh`
- `qwen3.sh`
- `rwkv4.sh`
- `stickbreaking.sh`
- `submit_all.sh`

</details>

<details>
<summary>scripts/slurm/jz/embedding/naturalstories/ — 22 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `bloom.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `fox.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_naturalstories_embed_gpu_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `qwen3.sh`
- `rwkv4.sh`
- `stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/embedding/onestop/ — 27 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `bloom.sh`
- `bloom_large.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `fox.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_onestop_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `opt_66b.sh`
- `pythia.sh`
- `pythia_ckpt.sh`
- `qwen3.sh`
- `rwkv4.sh`
- `rwkv4_large.sh`
- `stickbreaking.sh`
- `submit_all.sh`

</details>

<details>
<summary>scripts/slurm/jz/embedding/provo/ — 24 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `bloom.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `fox.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_provo_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_ckpt.sh`
- `qwen3.sh`
- `rwkv4.sh`
- `stickbreaking.sh`
- `submit_all.sh`

</details>

<details>
<summary>scripts/slurm/jz/embedding/ucl_reading/ — 24 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `bloom.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `fox.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_ucl_reading_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_ckpt.sh`
- `qwen3.sh`
- `rwkv4.sh`
- `stickbreaking.sh`
- `submit_all.sh`

</details>

<details>
<summary>scripts/slurm/jz/embedding/zuco_reading/ — 25 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `bloom.sh`
- `cbr_rnn.sh`
- `cpu_fit.sh`
- `forgetting.sh`
- `fox.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_zuco_reading_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_ckpt.sh`
- `qwen3.sh`
- `rwkv4.sh`
- `stickbreaking.sh`
- `submit_all.sh`

</details>

<details>
<summary>scripts/slurm/jz/perplexity/ — 22 shell files</summary>

- `collect_pythia_checkpoint_perplexity_3ds_summary.sh`
- `collect_pythia_checkpoint_perplexity_n400_summary.sh`
- `constrained_models_fedorenko2016.sh`
- `constrained_models_lpp.sh`
- `constrained_models_michaelov.sh`
- `constrained_models_nieuwland.sh`
- `constrained_models_ns_fmri.sh`
- `constrained_models_podcast.sh`
- `constrained_models_ryskin.sh`
- `constrained_models_tuckute.sh`
- `naturalstories_new_constrained_families.sh`
- `pythia_checkpoint_perplexity_3ds_1.4b.sh`
- `pythia_checkpoint_perplexity_3ds_12b.sh`
- `pythia_checkpoint_perplexity_3ds_14m.sh`
- `pythia_checkpoint_perplexity_3ds_160m.sh`
- `pythia_checkpoint_perplexity_3ds_1b.sh`
- `pythia_checkpoint_perplexity_3ds_2.8b.sh`
- `pythia_checkpoint_perplexity_3ds_31m.sh`
- `pythia_checkpoint_perplexity_3ds_410m.sh`
- `pythia_checkpoint_perplexity_3ds_6.9b.sh`
- `pythia_checkpoint_perplexity_3ds_70m.sh`
- `pythia_checkpoint_perplexity_n400.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/ — 22 shell files</summary>

- `babylm_gpt2_checkpoints_prefetch_g1.sh`
- `babylm_gpt2_checkpoints_prefetch_g2.sh`
- `babylm_gpt2_checkpoints_prefetch_g3.sh`
- `babylm_gpt2_group_g1.sh`
- `babylm_gpt2_group_g2.sh`
- `babylm_gpt2_group_g3.sh`
- `babylm_gpt2_group_g4.sh`
- `babylm_gpt2_group_g5.sh`
- `babylm_gpt2_group_g6.sh`
- `babylm_gpt2_group_g7.sh`
- `lpp_cbr_rnn.sh`
- `lpp_gpt2.sh`
- `lpp_gptneo.sh`
- `lpp_llama32.sh`
- `lpp_mamba.sh`
- `lpp_mamba2.sh`
- `lpp_mistral.sh`
- `lpp_olmo2.sh`
- `lpp_opt.sh`
- `lpp_pythia.sh`
- `lpp_qwen3.sh`
- `prefetch_lpp_surprisal_jz.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/broderick/ — 44 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_broderick_lme_jz_common.sh`
- `job_broderick_surprisal_jz_common.sh`
- `llama32.sh`
- `lme_bloom.sh`
- `lme_fox.sh`
- `lme_rwkv4.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_bloom.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_fox.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_pythia.sh`
- `surprisal_qwen3.sh`
- `surprisal_rwkv4.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/fedorenko2016/ — 47 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `babylm_gpt2_checkpoints.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_fedorenko2016_lme_jz_common.sh`
- `job_fedorenko2016_surprisal_jz_common.sh`
- `llama32.sh`
- `lme_bloom.sh`
- `lme_fox.sh`
- `lme_rwkv4.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_checkpoints.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_bloom.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_fox.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_pythia.sh`
- `surprisal_pythia_checkpoints.sh`
- `surprisal_qwen3.sh`
- `surprisal_rwkv4.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/geco/ — 56 shell files</summary>

- `geco_alibi_gpu.sh`
- `geco_alibi_lme_cpu.sh`
- `geco_babylm_gpt2_checkpoints_gpu.sh`
- `geco_babylm_gpt2_checkpoints_lme_cpu_oos_parallel.sh`
- `geco_babylm_gpu.sh`
- `geco_babylm_lme_cpu.sh`
- `geco_bloom_gp_lme_cpu.sh`
- `geco_bloom_lme_cpu.sh`
- `geco_bloom_prefetch.sh`
- `geco_cbr_rnn_gpu.sh`
- `geco_cbr_rnn_lme_cpu.sh`
- `geco_forgetting_gpu.sh`
- `geco_forgetting_lme_cpu.sh`
- `geco_fox_gp_lme_cpu.sh`
- `geco_fox_lme_cpu.sh`
- `geco_fox_prefetch.sh`
- `geco_gpt2_gpu.sh`
- `geco_gpt2_lme_cpu.sh`
- `geco_gptneo_gpu.sh`
- `geco_gptneo_lme_cpu.sh`
- `geco_hard_2gram_gpu.sh`
- `geco_hard_2gram_lme_cpu.sh`
- `geco_hard_3gram_gpu.sh`
- `geco_hard_3gram_lme_cpu.sh`
- `geco_hard_5gram_gpu.sh`
- `geco_hard_5gram_lme_cpu.sh`
- `geco_llama32_gpu.sh`
- `geco_llama32_lme_cpu.sh`
- `geco_mamba2_gpu.sh`
- `geco_mamba2_lme_cpu.sh`
- `geco_mamba_gpu.sh`
- `geco_mamba_lme_cpu.sh`
- `geco_mistral_gpu.sh`
- `geco_mistral_lme_cpu.sh`
- `geco_molm_prefetch.sh`
- `geco_olmo2_gpu.sh`
- `geco_olmo2_lme_cpu.sh`
- `geco_opt_gpu.sh`
- `geco_opt_lme_cpu.sh`
- `geco_pythia_checkpoints_gpu.sh`
- `geco_pythia_checkpoints_lme_cpu.sh`
- `geco_pythia_checkpoints_lme_cpu_insample.sh`
- `geco_pythia_checkpoints_lme_cpu_insample_parallel.sh`
- `geco_pythia_checkpoints_lme_cpu_insample_parallel_tail.sh`
- `geco_pythia_checkpoints_lme_cpu_oos_parallel.sh`
- `geco_pythia_checkpoints_lme_cpu_oos_parallel_tail.sh`
- `geco_pythia_gpu.sh`
- `geco_pythia_lme_cpu.sh`
- `geco_qwen3_gpu.sh`
- `geco_qwen3_lme_cpu.sh`
- `geco_rwkv4_gp_lme_cpu.sh`
- `geco_rwkv4_lme_cpu.sh`
- `geco_rwkv4_prefetch.sh`
- `geco_stickbreaking_gpu.sh`
- `geco_stickbreaking_lme_cpu.sh`
- `job_geco_lme_jz_common.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/lpp/ — 54 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `babylm_gpt2_checkpoints.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_lpp_lme_jz_common.sh`
- `job_lpp_lme_lastbest_jz_common.sh`
- `job_lpp_lme_lastlayer_jz_common.sh`
- `job_lpp_surprisal_jz_common.sh`
- `lastbest_refit.sh`
- `lastlayer_refit.sh`
- `llama32.sh`
- `lme_bloom.sh`
- `lme_fox.sh`
- `lme_molm.sh`
- `lme_rwkv4.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_checkpoints.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_babylm_gpt2_checkpoints.sh`
- `surprisal_bloom.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_fox.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_molm.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_pythia.sh`
- `surprisal_pythia_checkpoints.sh`
- `surprisal_qwen3.sh`
- `surprisal_rwkv4.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/meco/ — 57 shell files</summary>

- `job_meco_lme_jz_common.sh`
- `meco_alibi_gpu.sh`
- `meco_alibi_lme_cpu.sh`
- `meco_babylm_gpt2_checkpoints_gpu.sh`
- `meco_babylm_gpt2_checkpoints_lme_cpu_oos_parallel.sh`
- `meco_babylm_gpu.sh`
- `meco_babylm_lme_cpu.sh`
- `meco_bloom_gp_lme_cpu.sh`
- `meco_bloom_lme_cpu.sh`
- `meco_bloom_prefetch.sh`
- `meco_bloom_sp_lme_cpu.sh`
- `meco_cbr_rnn_gpu.sh`
- `meco_cbr_rnn_lme_cpu.sh`
- `meco_forgetting_gpu.sh`
- `meco_forgetting_lme_cpu.sh`
- `meco_fox_gp_lme_cpu.sh`
- `meco_fox_lme_cpu.sh`
- `meco_fox_prefetch.sh`
- `meco_fox_sp_lme_cpu.sh`
- `meco_gpt2_gpu.sh`
- `meco_gpt2_lme_cpu.sh`
- `meco_gptneo_gpu.sh`
- `meco_gptneo_lme_cpu.sh`
- `meco_hard_2gram_gpu.sh`
- `meco_hard_2gram_lme_cpu.sh`
- `meco_hard_3gram_gpu.sh`
- `meco_hard_3gram_lme_cpu.sh`
- `meco_hard_5gram_gpu.sh`
- `meco_hard_5gram_lme_cpu.sh`
- `meco_llama32_gpu.sh`
- `meco_llama32_lme_cpu.sh`
- `meco_mamba2_gpu.sh`
- `meco_mamba2_lme_cpu.sh`
- `meco_mamba_gpu.sh`
- `meco_mamba_lme_cpu.sh`
- `meco_mistral_gpu.sh`
- `meco_mistral_lme_cpu.sh`
- `meco_molm_prefetch.sh`
- `meco_olmo2_gpu.sh`
- `meco_olmo2_lme_cpu.sh`
- `meco_opt_gpu.sh`
- `meco_opt_lme_cpu.sh`
- `meco_pythia_checkpoints_gpu.sh`
- `meco_pythia_checkpoints_lme_cpu.sh`
- `meco_pythia_checkpoints_lme_cpu_insample.sh`
- `meco_pythia_checkpoints_lme_cpu_insample_parallel.sh`
- `meco_pythia_checkpoints_lme_cpu_oos_parallel.sh`
- `meco_pythia_gpu.sh`
- `meco_pythia_lme_cpu.sh`
- `meco_qwen3_gpu.sh`
- `meco_qwen3_lme_cpu.sh`
- `meco_rwkv4_gp_lme_cpu.sh`
- `meco_rwkv4_lme_cpu.sh`
- `meco_rwkv4_prefetch.sh`
- `meco_rwkv4_sp_lme_cpu.sh`
- `meco_stickbreaking_gpu.sh`
- `meco_stickbreaking_lme_cpu.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/megmasc/ — 44 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_megmasc_lme_jz_common.sh`
- `job_megmasc_surprisal_jz_common.sh`
- `llama32.sh`
- `lme_bloom.sh`
- `lme_fox.sh`
- `lme_rwkv4.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_bloom.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_fox.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_pythia.sh`
- `surprisal_qwen3.sh`
- `surprisal_rwkv4.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/michaelov/ — 42 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `babylm_gpt2_checkpoints.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_michaelov_lme_jz_common.sh`
- `job_michaelov_surprisal_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_checkpoints.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_babylm_gpt2_checkpoints.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_pythia.sh`
- `surprisal_pythia_checkpoints.sh`
- `surprisal_qwen3.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/naturalstories/ — 56 shell files</summary>

- `job_naturalstories_lme_jz_common.sh`
- `naturalstories_alibi_lme_cpu.sh`
- `naturalstories_alibi_lme_prefetch.sh`
- `naturalstories_babylm_gpt2_checkpoints_lme_cpu_oos_parallel.sh`
- `naturalstories_babylm_gpt2_checkpoints_prefetch.sh`
- `naturalstories_babylm_lme_cpu.sh`
- `naturalstories_babylm_lme_prefetch.sh`
- `naturalstories_bloom_lme_cpu.sh`
- `naturalstories_bloom_lme_prefetch.sh`
- `naturalstories_cbr_rnn_lme_cpu.sh`
- `naturalstories_cbr_rnn_lme_prefetch.sh`
- `naturalstories_forgetting_lme_cpu.sh`
- `naturalstories_forgetting_lme_prefetch.sh`
- `naturalstories_fox_lme_cpu.sh`
- `naturalstories_fox_lme_prefetch.sh`
- `naturalstories_gpt2_lme_cpu.sh`
- `naturalstories_gpt2_lme_prefetch.sh`
- `naturalstories_gptneo_lme_cpu.sh`
- `naturalstories_gptneo_lme_prefetch.sh`
- `naturalstories_hard_2gram_lme_cpu.sh`
- `naturalstories_hard_2gram_lme_prefetch.sh`
- `naturalstories_hard_3gram_lme_cpu.sh`
- `naturalstories_hard_3gram_lme_prefetch.sh`
- `naturalstories_hard_5gram_lme_cpu.sh`
- `naturalstories_hard_5gram_lme_prefetch.sh`
- `naturalstories_llama32_lme_cpu.sh`
- `naturalstories_llama32_lme_prefetch.sh`
- `naturalstories_lme_prefetch_all.sh`
- `naturalstories_mamba2_lme_cpu.sh`
- `naturalstories_mamba2_lme_prefetch.sh`
- `naturalstories_mamba_lme_cpu.sh`
- `naturalstories_mamba_lme_prefetch.sh`
- `naturalstories_mistral_lme_cpu.sh`
- `naturalstories_mistral_lme_prefetch.sh`
- `naturalstories_molm_lme_prefetch.sh`
- `naturalstories_olmo2_lme_cpu.sh`
- `naturalstories_olmo2_lme_prefetch.sh`
- `naturalstories_opt_lme_cpu.sh`
- `naturalstories_opt_lme_prefetch.sh`
- `naturalstories_pythia_checkpoints_lme_cpu.sh`
- `naturalstories_pythia_checkpoints_lme_cpu_insample.sh`
- `naturalstories_pythia_checkpoints_lme_cpu_insample_parallel.sh`
- `naturalstories_pythia_checkpoints_lme_cpu_insample_parallel_tail.sh`
- `naturalstories_pythia_checkpoints_lme_cpu_insample_parallel_tail2.sh`
- `naturalstories_pythia_checkpoints_lme_cpu_oos_parallel.sh`
- `naturalstories_pythia_checkpoints_lme_cpu_oos_parallel_tail.sh`
- `naturalstories_pythia_checkpoints_lme_cpu_oos_parallel_tail2.sh`
- `naturalstories_pythia_checkpoints_prefetch.sh`
- `naturalstories_pythia_lme_cpu.sh`
- `naturalstories_pythia_lme_prefetch.sh`
- `naturalstories_qwen3_lme_cpu.sh`
- `naturalstories_qwen3_lme_prefetch.sh`
- `naturalstories_rwkv4_lme_cpu.sh`
- `naturalstories_rwkv4_lme_prefetch.sh`
- `naturalstories_stickbreaking_lme_cpu.sh`
- `naturalstories_stickbreaking_lme_prefetch.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/nieuwland/ — 48 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `babylm_gpt2_checkpoints.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_nieuwland_lme_jz_common.sh`
- `job_nieuwland_surprisal_jz_common.sh`
- `llama32.sh`
- `lme_bloom.sh`
- `lme_fox.sh`
- `lme_rwkv4.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_checkpoints.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_babylm_gpt2_checkpoints.sh`
- `surprisal_bloom.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_fox.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_pythia.sh`
- `surprisal_pythia_checkpoints.sh`
- `surprisal_qwen3.sh`
- `surprisal_rwkv4.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/ns_fmri/ — 66 shell files</summary>

- `alibi_lme.sh`
- `alibi_lme_cpu.sh`
- `alibi_prefetch.sh`
- `babylm_gpt2_checkpoints_lme_cpu.sh`
- `babylm_gpt2_checkpoints_prefetch.sh`
- `babylm_lme.sh`
- `babylm_lme_cpu.sh`
- `babylm_prefetch.sh`
- `cbr_rnn_lme.sh`
- `cbr_rnn_lme_cpu.sh`
- `cbr_rnn_prefetch.sh`
- `forgetting_lme.sh`
- `forgetting_lme_cpu.sh`
- `forgetting_prefetch.sh`
- `gpt2_lme.sh`
- `gpt2_lme_cpu.sh`
- `gpt2_prefetch.sh`
- `gptneo_lme.sh`
- `gptneo_lme_cpu.sh`
- `gptneo_prefetch.sh`
- `hard_2gram_lme.sh`
- `hard_2gram_lme_cpu.sh`
- `hard_2gram_prefetch.sh`
- `hard_3gram_lme.sh`
- `hard_3gram_lme_cpu.sh`
- `hard_3gram_prefetch.sh`
- `hard_5gram_lme.sh`
- `hard_5gram_lme_cpu.sh`
- `hard_5gram_prefetch.sh`
- `job_ns_fmri_lme_jz_common.sh`
- `llama32_lme.sh`
- `llama32_lme_cpu.sh`
- `llama32_prefetch.sh`
- `lme_bloom_cpu.sh`
- `lme_fox_cpu.sh`
- `lme_rwkv4_cpu.sh`
- `mamba2_lme.sh`
- `mamba2_lme_cpu.sh`
- `mamba2_prefetch.sh`
- `mamba_lme.sh`
- `mamba_lme_cpu.sh`
- `mamba_prefetch.sh`
- `mistral_lme.sh`
- `mistral_lme_cpu.sh`
- `mistral_prefetch.sh`
- `ns_fmri_lme_prefetch_all.sh`
- `olmo2_lme.sh`
- `olmo2_lme_cpu.sh`
- `olmo2_prefetch.sh`
- `opt_lme.sh`
- `opt_lme_cpu.sh`
- `opt_prefetch.sh`
- `pythia_checkpoints_lme_cpu.sh`
- `pythia_checkpoints_prefetch.sh`
- `pythia_lme.sh`
- `pythia_lme_cpu.sh`
- `pythia_prefetch.sh`
- `qwen3_lme.sh`
- `qwen3_lme_cpu.sh`
- `qwen3_prefetch.sh`
- `stickbreaking_lme.sh`
- `stickbreaking_lme_cpu.sh`
- `stickbreaking_prefetch.sh`
- `surprisal_bloom.sh`
- `surprisal_fox.sh`
- `surprisal_rwkv4.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/onestop/ — 51 shell files</summary>

- `job_onestop_lme_jz_common.sh`
- `onestop_alibi_gpu.sh`
- `onestop_alibi_lme_cpu.sh`
- `onestop_babylm_gpu.sh`
- `onestop_babylm_lme_cpu.sh`
- `onestop_bloom_gp_lme_cpu.sh`
- `onestop_bloom_lme_cpu.sh`
- `onestop_bloom_prefetch.sh`
- `onestop_bloom_sp_lme_cpu.sh`
- `onestop_cbr_rnn_gpu.sh`
- `onestop_cbr_rnn_lme_cpu.sh`
- `onestop_forgetting_gpu.sh`
- `onestop_forgetting_lme_cpu.sh`
- `onestop_fox_gp_lme_cpu.sh`
- `onestop_fox_lme_cpu.sh`
- `onestop_fox_prefetch.sh`
- `onestop_fox_sp_lme_cpu.sh`
- `onestop_gpt2_gpu.sh`
- `onestop_gpt2_lme_cpu.sh`
- `onestop_gptneo_gpu.sh`
- `onestop_gptneo_lme_cpu.sh`
- `onestop_hard_2gram_gpu.sh`
- `onestop_hard_2gram_lme_cpu.sh`
- `onestop_hard_3gram_gpu.sh`
- `onestop_hard_3gram_lme_cpu.sh`
- `onestop_hard_5gram_gpu.sh`
- `onestop_hard_5gram_lme_cpu.sh`
- `onestop_llama32_gpu.sh`
- `onestop_llama32_lme_cpu.sh`
- `onestop_mamba2_gpu.sh`
- `onestop_mamba2_lme_cpu.sh`
- `onestop_mamba_gpu.sh`
- `onestop_mamba_lme_cpu.sh`
- `onestop_mistral_gpu.sh`
- `onestop_mistral_lme_cpu.sh`
- `onestop_molm_prefetch.sh`
- `onestop_olmo2_gpu.sh`
- `onestop_olmo2_lme_cpu.sh`
- `onestop_opt_gpu.sh`
- `onestop_opt_lme_cpu.sh`
- `onestop_pythia_gpu.sh`
- `onestop_pythia_lme_cpu.sh`
- `onestop_qwen3_gpu.sh`
- `onestop_qwen3_lme_cpu.sh`
- `onestop_rwkv4_gp_lme_cpu.sh`
- `onestop_rwkv4_lme_cpu.sh`
- `onestop_rwkv4_prefetch.sh`
- `onestop_rwkv4_sp_lme_cpu.sh`
- `onestop_stickbreaking_gpu.sh`
- `onestop_stickbreaking_lme_cpu.sh`
- `submit_onestop_lme.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/pereira/ — 31 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `babylm_gpt2_checkpoints.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_pereira_lme_jz_common.sh`
- `llama32.sh`
- `lme_all_families.sh`
- `lme_bloom.sh`
- `lme_fox.sh`
- `lme_rwkv4.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `qwen3.sh`
- `smoketest_lme_gpt2.sh`
- `smoketest_surprisal_gpt2.sh`
- `stickbreaking.sh`
- `surprisal_all_families.sh`
- `surprisal_bloom.sh`
- `surprisal_fox.sh`
- `surprisal_opt66b.sh`
- `surprisal_rwkv4.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/podcast/ — 47 shell files</summary>

- `alibi.sh`
- `babylm_gpt2_checkpoints.sh`
- `babylm_prefetch.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_podcast_lme_jz_common.sh`
- `job_podcast_surprisal_jz_common.sh`
- `llama32.sh`
- `lme_alibi.sh`
- `lme_babylm.sh`
- `lme_bloom.sh`
- `lme_cbr_rnn.sh`
- `lme_forgetting.sh`
- `lme_fox.sh`
- `lme_gpt2.sh`
- `lme_gptneo.sh`
- `lme_hard_2gram.sh`
- `lme_hard_3gram.sh`
- `lme_hard_5gram.sh`
- `lme_llama32.sh`
- `lme_mamba.sh`
- `lme_mamba2.sh`
- `lme_mistral.sh`
- `lme_olmo2.sh`
- `lme_opt.sh`
- `lme_pythia.sh`
- `lme_qwen3.sh`
- `lme_rwkv4.sh`
- `lme_stickbreaking.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_checkpoints.sh`
- `pythia_checkpoints_prefetch.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_bloom.sh`
- `surprisal_fox.sh`
- `surprisal_rwkv4.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/provo/ — 61 shell files</summary>

- `job_provo_lme_jz_common.sh`
- `provo_alibi_gpu.sh`
- `provo_alibi_lme_cpu.sh`
- `provo_babylm_gpt2_checkpoints_gpu.sh`
- `provo_babylm_gpt2_checkpoints_lme_cpu_oos_parallel.sh`
- `provo_babylm_gpu.sh`
- `provo_babylm_lme_cpu.sh`
- `provo_bloom_gp_lme_cpu.sh`
- `provo_bloom_lme_cpu.sh`
- `provo_bloom_prefetch.sh`
- `provo_bloom_sp_lme_cpu.sh`
- `provo_cbr_rnn_gpu.sh`
- `provo_cbr_rnn_lme_cpu.sh`
- `provo_forgetting_gpu.sh`
- `provo_forgetting_lme_cpu.sh`
- `provo_fox_gp_lme_cpu.sh`
- `provo_fox_lme_cpu.sh`
- `provo_fox_prefetch.sh`
- `provo_fox_sp_lme_cpu.sh`
- `provo_gpt2_gpu.sh`
- `provo_gpt2_lme_cpu.sh`
- `provo_gptneo_gpu.sh`
- `provo_gptneo_lme_cpu.sh`
- `provo_hard_2gram_gpu.sh`
- `provo_hard_2gram_lme_cpu.sh`
- `provo_hard_3gram_gpu.sh`
- `provo_hard_3gram_lme_cpu.sh`
- `provo_hard_5gram_gpu.sh`
- `provo_hard_5gram_lme_cpu.sh`
- `provo_llama32_gpu.sh`
- `provo_llama32_lme_cpu.sh`
- `provo_mamba2_gpu.sh`
- `provo_mamba2_lme_cpu.sh`
- `provo_mamba_gpu.sh`
- `provo_mamba_lme_cpu.sh`
- `provo_mistral_gpu.sh`
- `provo_mistral_lme_cpu.sh`
- `provo_molm_prefetch.sh`
- `provo_olmo2_gpu.sh`
- `provo_olmo2_lme_cpu.sh`
- `provo_opt_gpu.sh`
- `provo_opt_lme_cpu.sh`
- `provo_pythia_checkpoints_gpu.sh`
- `provo_pythia_checkpoints_lme_cpu.sh`
- `provo_pythia_checkpoints_lme_cpu_insample.sh`
- `provo_pythia_checkpoints_lme_cpu_ohschuler_insample.sh`
- `provo_pythia_checkpoints_lme_cpu_ohschuler_insample_parallel.sh`
- `provo_pythia_checkpoints_lme_cpu_ohschuler_oos.sh`
- `provo_pythia_checkpoints_lme_cpu_ohschuler_oos_parallel.sh`
- `provo_pythia_checkpoints_lme_cpu_oldformula_insample_parallel.sh`
- `provo_pythia_checkpoints_lme_cpu_oldformula_oos_parallel.sh`
- `provo_pythia_gpu.sh`
- `provo_pythia_lme_cpu.sh`
- `provo_qwen3_gpu.sh`
- `provo_qwen3_lme_cpu.sh`
- `provo_rwkv4_gp_lme_cpu.sh`
- `provo_rwkv4_lme_cpu.sh`
- `provo_rwkv4_prefetch.sh`
- `provo_rwkv4_sp_lme_cpu.sh`
- `provo_stickbreaking_gpu.sh`
- `provo_stickbreaking_lme_cpu.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/ryskin/ — 42 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `babylm_gpt2_checkpoints.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_ryskin_lme_jz_common.sh`
- `job_ryskin_surprisal_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_checkpoints.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_babylm_gpt2_checkpoints.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_pythia.sh`
- `surprisal_pythia_checkpoints.sh`
- `surprisal_qwen3.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/sparrkulee/ — 38 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_sparrkulee_lme_jz_common.sh`
- `job_sparrkulee_surprisal_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_pythia.sh`
- `surprisal_qwen3.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/szewczyk/ — 46 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_szewczyk_lme_jz_common.sh`
- `job_szewczyk_surprisal_jz_common.sh`
- `llama32.sh`
- `lme_bloom.sh`
- `lme_fox.sh`
- `lme_opt.sh`
- `lme_rwkv4.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_bloom.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_fox.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_opt_66b.sh`
- `surprisal_pythia.sh`
- `surprisal_qwen3.sh`
- `surprisal_rwkv4.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/tuckute/ — 48 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `babylm_gpt2_checkpoints.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_tuckute_lme_jz_common.sh`
- `job_tuckute_surprisal_jz_common.sh`
- `llama32.sh`
- `lme_bloom.sh`
- `lme_fox.sh`
- `lme_rwkv4.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `pythia_checkpoints.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_babylm_gpt2_checkpoints.sh`
- `surprisal_bloom.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_fox.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_pythia.sh`
- `surprisal_pythia_checkpoints.sh`
- `surprisal_qwen3.sh`
- `surprisal_rwkv4.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/ucl_reading/ — 52 shell files</summary>

- `job_ucl_reading_lme_jz_common.sh`
- `ucl_reading_alibi_lme_cpu.sh`
- `ucl_reading_alibi_prefetch.sh`
- `ucl_reading_babylm_gpt2_checkpoints_lme_cpu_oos_parallel.sh`
- `ucl_reading_babylm_gpt2_checkpoints_prefetch.sh`
- `ucl_reading_babylm_lme_cpu.sh`
- `ucl_reading_babylm_prefetch.sh`
- `ucl_reading_bloom_lme_cpu.sh`
- `ucl_reading_bloom_prefetch.sh`
- `ucl_reading_cbr_rnn_lme_cpu.sh`
- `ucl_reading_cbr_rnn_prefetch.sh`
- `ucl_reading_forgetting_lme_cpu.sh`
- `ucl_reading_forgetting_prefetch.sh`
- `ucl_reading_fox_lme_cpu.sh`
- `ucl_reading_fox_prefetch.sh`
- `ucl_reading_gpt2_lme_cpu.sh`
- `ucl_reading_gpt2_prefetch.sh`
- `ucl_reading_gptneo_lme_cpu.sh`
- `ucl_reading_gptneo_prefetch.sh`
- `ucl_reading_hard_2gram_lme_cpu.sh`
- `ucl_reading_hard_2gram_prefetch.sh`
- `ucl_reading_hard_3gram_lme_cpu.sh`
- `ucl_reading_hard_3gram_prefetch.sh`
- `ucl_reading_hard_5gram_lme_cpu.sh`
- `ucl_reading_hard_5gram_prefetch.sh`
- `ucl_reading_llama32_lme_cpu.sh`
- `ucl_reading_llama32_prefetch.sh`
- `ucl_reading_mamba2_lme_cpu.sh`
- `ucl_reading_mamba2_prefetch.sh`
- `ucl_reading_mamba_lme_cpu.sh`
- `ucl_reading_mamba_prefetch.sh`
- `ucl_reading_mistral_lme_cpu.sh`
- `ucl_reading_mistral_prefetch.sh`
- `ucl_reading_molm_prefetch.sh`
- `ucl_reading_olmo2_lme_cpu.sh`
- `ucl_reading_olmo2_prefetch.sh`
- `ucl_reading_opt_lme_cpu.sh`
- `ucl_reading_opt_prefetch.sh`
- `ucl_reading_pythia_checkpoints_lme_cpu.sh`
- `ucl_reading_pythia_checkpoints_lme_cpu_insample.sh`
- `ucl_reading_pythia_checkpoints_lme_cpu_insample_parallel.sh`
- `ucl_reading_pythia_checkpoints_lme_cpu_insample_parallel_tail.sh`
- `ucl_reading_pythia_checkpoints_lme_cpu_oos_parallel.sh`
- `ucl_reading_pythia_checkpoints_prefetch.sh`
- `ucl_reading_pythia_lme_cpu.sh`
- `ucl_reading_pythia_prefetch.sh`
- `ucl_reading_qwen3_lme_cpu.sh`
- `ucl_reading_qwen3_prefetch.sh`
- `ucl_reading_rwkv4_lme_cpu.sh`
- `ucl_reading_rwkv4_prefetch.sh`
- `ucl_reading_stickbreaking_lme_cpu.sh`
- `ucl_reading_stickbreaking_prefetch.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/zuco_eeg/ — 38 shell files</summary>

- `alibi.sh`
- `babylm.sh`
- `cbr_rnn.sh`
- `forgetting.sh`
- `gpt2.sh`
- `gptneo.sh`
- `hard_2gram.sh`
- `hard_3gram.sh`
- `hard_5gram.sh`
- `job_zuco_eeg_lme_jz_common.sh`
- `job_zuco_eeg_surprisal_jz_common.sh`
- `llama32.sh`
- `mamba.sh`
- `mamba2.sh`
- `mistral.sh`
- `olmo2.sh`
- `opt.sh`
- `pythia.sh`
- `qwen3.sh`
- `stickbreaking.sh`
- `surprisal_alibi.sh`
- `surprisal_babylm.sh`
- `surprisal_cbr_rnn.sh`
- `surprisal_forgetting.sh`
- `surprisal_gpt2.sh`
- `surprisal_gptneo.sh`
- `surprisal_hard_2gram.sh`
- `surprisal_hard_3gram.sh`
- `surprisal_hard_5gram.sh`
- `surprisal_llama32.sh`
- `surprisal_mamba.sh`
- `surprisal_mamba2.sh`
- `surprisal_mistral.sh`
- `surprisal_olmo2.sh`
- `surprisal_opt.sh`
- `surprisal_pythia.sh`
- `surprisal_qwen3.sh`
- `surprisal_stickbreaking.sh`

</details>

<details>
<summary>scripts/slurm/jz/surprisal/zuco_reading/ — 65 shell files</summary>

- `job_zuco_reading_lme_jz_common.sh`
- `zuco_reading_alibi_gpu.sh`
- `zuco_reading_alibi_lme_cpu.sh`
- `zuco_reading_babylm_gpt2_checkpoints_gpu.sh`
- `zuco_reading_babylm_gpt2_checkpoints_lme_cpu_oos_parallel.sh`
- `zuco_reading_babylm_gpu.sh`
- `zuco_reading_babylm_lme_cpu.sh`
- `zuco_reading_bloom_nr_lme_cpu.sh`
- `zuco_reading_bloom_nr_prefetch.sh`
- `zuco_reading_bloom_sr_lme_cpu.sh`
- `zuco_reading_bloom_sr_prefetch.sh`
- `zuco_reading_bloom_tsr_lme_cpu.sh`
- `zuco_reading_bloom_tsr_prefetch.sh`
- `zuco_reading_cbr_rnn_gpu.sh`
- `zuco_reading_cbr_rnn_lme_cpu.sh`
- `zuco_reading_forgetting_gpu.sh`
- `zuco_reading_forgetting_lme_cpu.sh`
- `zuco_reading_fox_nr_lme_cpu.sh`
- `zuco_reading_fox_nr_prefetch.sh`
- `zuco_reading_fox_sr_lme_cpu.sh`
- `zuco_reading_fox_sr_prefetch.sh`
- `zuco_reading_fox_tsr_lme_cpu.sh`
- `zuco_reading_fox_tsr_prefetch.sh`
- `zuco_reading_gpt2_gpu.sh`
- `zuco_reading_gpt2_lme_cpu.sh`
- `zuco_reading_gptneo_gpu.sh`
- `zuco_reading_gptneo_lme_cpu.sh`
- `zuco_reading_hard_2gram_gpu.sh`
- `zuco_reading_hard_2gram_lme_cpu.sh`
- `zuco_reading_hard_3gram_gpu.sh`
- `zuco_reading_hard_3gram_lme_cpu.sh`
- `zuco_reading_hard_5gram_gpu.sh`
- `zuco_reading_hard_5gram_lme_cpu.sh`
- `zuco_reading_llama32_gpu.sh`
- `zuco_reading_llama32_lme_cpu.sh`
- `zuco_reading_mamba2_gpu.sh`
- `zuco_reading_mamba2_lme_cpu.sh`
- `zuco_reading_mamba_gpu.sh`
- `zuco_reading_mamba_lme_cpu.sh`
- `zuco_reading_mistral_gpu.sh`
- `zuco_reading_mistral_lme_cpu.sh`
- `zuco_reading_molm_nr_prefetch.sh`
- `zuco_reading_molm_sr_prefetch.sh`
- `zuco_reading_molm_tsr_prefetch.sh`
- `zuco_reading_olmo2_gpu.sh`
- `zuco_reading_olmo2_lme_cpu.sh`
- `zuco_reading_opt_gpu.sh`
- `zuco_reading_opt_lme_cpu.sh`
- `zuco_reading_pythia_checkpoints_gpu.sh`
- `zuco_reading_pythia_checkpoints_lme_cpu.sh`
- `zuco_reading_pythia_checkpoints_lme_cpu_insample.sh`
- `zuco_reading_pythia_checkpoints_lme_cpu_insample_parallel.sh`
- `zuco_reading_pythia_checkpoints_lme_cpu_oos_parallel.sh`
- `zuco_reading_pythia_gpu.sh`
- `zuco_reading_pythia_lme_cpu.sh`
- `zuco_reading_qwen3_gpu.sh`
- `zuco_reading_qwen3_lme_cpu.sh`
- `zuco_reading_rwkv4_nr_lme_cpu.sh`
- `zuco_reading_rwkv4_nr_prefetch.sh`
- `zuco_reading_rwkv4_sr_lme_cpu.sh`
- `zuco_reading_rwkv4_sr_prefetch.sh`
- `zuco_reading_rwkv4_tsr_lme_cpu.sh`
- `zuco_reading_rwkv4_tsr_prefetch.sh`
- `zuco_reading_stickbreaking_gpu.sh`
- `zuco_reading_stickbreaking_lme_cpu.sh`

</details>

<details>
<summary>scripts/slurm/oberon/ — 6 shell files</summary>

- `job_houston_checkpoint_sweep_oberon.sh`
- `job_lpp_checkpoint_sweep_oberon.sh`
- `narratives_mfa_align.sh`
- `narratives_whisper_transcribe.sh`
- `podcast_perplexity_oberon.sh`
- `sparrkulee_nfa_align.sh`

</details>

### E. Documents, notebooks, manuscript, reports and diagnostics

The following exact inventory is retained or organized under the rules above. Report/manuscript binaries are inventory entries, not certified regenerable artifacts. Keeping the list does not imply that every report is current. Notebook/checkpoint duplicates require content comparison.

<details>
<summary>docs/ — 2 inventoried files</summary>

- `docs/megmasc_design.md`
- `docs/pereira_embedding_design.md`

</details>

<details>
<summary>notebooks/ — 7 inventoried files</summary>

- `notebooks/LLM alignment.ipynb`
- `notebooks/benchmark_overview copy.ipynb`
- `notebooks/benchmark_overview.ipynb`
- `notebooks/brainmaps.ipynb`
- `notebooks/houston_pythia_checkpoint_analysis.ipynb`
- `notebooks/neuralset_demo.ipynb`
- `notebooks/scaling_benchmark.ipynb`

</details>

<details>
<summary>.debug_scratch/ — 7 inventoried files</summary>

- `.debug_scratch/lpp_alignment_full_check.py`
- `.debug_scratch/lpp_cache_risk_check.py`
- `.debug_scratch/lpp_coverage_report.py`
- `.debug_scratch/mamba2_pad_check.py`
- `.debug_scratch/mamba2_stride_check.py`
- `.debug_scratch/mamba2_unified.py`
- `.debug_scratch/moep_check.py`

</details>

<details>
<summary>paperwriting/ — 38 inventoried files</summary>

- `paperwriting/Figures LLM-Human Benchmark.pdf`
- `paperwriting/fancyhdr.sty`
- `paperwriting/figures/appendix_pareto_embedding_bestlayer.pdf`
- `paperwriting/figures/appendix_pareto_embedding_lastlayer.pdf`
- `paperwriting/figures/appendix_pareto_surprisal_bestlayer.pdf`
- `paperwriting/figures/figure1_overview.pdf`
- `paperwriting/figures/figure2_pareto_a.pdf`
- `paperwriting/figures/figure2_pareto_b.pdf`
- `paperwriting/figures/figure3_scaling_a.pdf`
- `paperwriting/figures/figure3_scaling_b.pdf`
- `paperwriting/figures/figure4_layerwise_a.pdf`
- `paperwriting/figures/figure4_layerwise_b.pdf`
- `paperwriting/figures/figure5_dlt_trend.pdf`
- `paperwriting/iclr2027_conference.bib`
- `paperwriting/iclr2027_conference.bst`
- `paperwriting/iclr2027_conference.sty`
- `paperwriting/main_iclr2027.aux`
- `paperwriting/main_iclr2027.bbl`
- `paperwriting/main_iclr2027.blg`
- `paperwriting/main_iclr2027.log`
- `paperwriting/main_iclr2027.out`
- `paperwriting/main_iclr2027.pdf`
- `paperwriting/main_iclr2027.tex`
- `paperwriting/math_commands.tex`
- `paperwriting/natbib.sty`
- `paperwriting/sections/abstract.tex`
- `paperwriting/sections/appendix.tex`
- `paperwriting/sections/corr_ceiling_table.tex`
- `paperwriting/sections/dataset_table.tex`
- `paperwriting/sections/discussion.tex`
- `paperwriting/sections/dll_ceiling_table.tex`
- `paperwriting/sections/experimental_setup.tex`
- `paperwriting/sections/intro.tex`
- `paperwriting/sections/lme_formulas.tex`
- `paperwriting/sections/model_table.tex`
- `paperwriting/sections/related_literature.tex`
- `paperwriting/sections/results.tex`
- `paperwriting/sections/ridge_config_table.tex`

</details>

<details>
<summary>reports/ — 235 inventoried files</summary>

- `reports/benchmark_report.md`
- `reports/benchmark_report.pdf`
- `reports/figures/behavioral_embed_average_scaling.png`
- `reports/figures/broderick_eeg_layer_profiles.png`
- `reports/figures/broderick_eeg_overview.png`
- `reports/figures/broderick_eeg_roi_profile.png`
- `reports/figures/broderick_eeg_scaling_by_group.png`
- `reports/figures/broderick_eeg_scaling_by_perplexity.png`
- `reports/figures/broderick_eeg_scaling_law.png`
- `reports/figures/broderick_eeg_timecourse.png`
- `reports/figures/cross_dataset_rank_correlation.png`
- `reports/figures/distance_to_ceiling.png`
- `reports/figures/dlt_hard_bars_behavioral_embedding.png`
- `reports/figures/dlt_hard_bars_behavioral_embedding_per_dataset.png`
- `reports/figures/dlt_hard_bars_behavioral_surprisal.png`
- `reports/figures/dlt_hard_bars_behavioral_surprisal_per_dataset.png`
- `reports/figures/dlt_hard_bars_best_layer_thr2.png`
- `reports/figures/dlt_hard_bars_last_layer_thr2.png`
- `reports/figures/dlt_hard_bars_neural.png`
- `reports/figures/dlt_hard_bars_neural_per_dataset.png`
- `reports/figures/executive_summary.png`
- `reports/figures/fedorenko2016_ecog_layer_profiles.png`
- `reports/figures/fedorenko2016_ecog_overview.png`
- `reports/figures/fedorenko2016_ecog_scaling_by_group.png`
- `reports/figures/fedorenko2016_ecog_scaling_by_perplexity.png`
- `reports/figures/fedorenko2016_ecog_scaling_law.png`
- `reports/figures/geco_overview.png`
- `reports/figures/geco_scaling_by_group.png`
- `reports/figures/global_behavioral_scaling_by_group.png`
- `reports/figures/global_neural_scaling.png`
- `reports/figures/global_neural_scaling_by_group.png`
- `reports/figures/global_scaling_laws.png`
- `reports/figures/group_effects_mixedlm_embedding_best.png`
- `reports/figures/group_effects_mixedlm_embedding_last.png`
- `reports/figures/group_effects_mixedlm_surprisal_best.png`
- `reports/figures/group_effects_mixedlm_surprisal_last.png`
- `reports/figures/lastlayer_dissociation_degradation_log_params.png`
- `reports/figures/lastlayer_dissociation_degradation_log_ppl.png`
- `reports/figures/lastlayer_dissociation_embedding_last_log_params.png`
- `reports/figures/lastlayer_dissociation_embedding_last_log_ppl.png`
- `reports/figures/lastlayer_dissociation_surprisal_last_log_params.png`
- `reports/figures/lastlayer_dissociation_surprisal_last_log_ppl.png`
- `reports/figures/llm_vs_mc_cohens_d.png`
- `reports/figures/lpp_fmri_layer_profiles.png`
- `reports/figures/lpp_fmri_overview.png`
- `reports/figures/lpp_fmri_roi_profile.png`
- `reports/figures/lpp_fmri_scaling_by_group.png`
- `reports/figures/lpp_fmri_scaling_by_perplexity.png`
- `reports/figures/lpp_fmri_scaling_law.png`
- `reports/figures/meco_overview.png`
- `reports/figures/meco_scaling_by_group.png`
- `reports/figures/michaelov_eeg_layer_profiles.png`
- `reports/figures/michaelov_eeg_overview.png`
- `reports/figures/michaelov_eeg_roi_profile.png`
- `reports/figures/michaelov_eeg_scaling_by_group.png`
- `reports/figures/michaelov_eeg_scaling_by_perplexity.png`
- `reports/figures/michaelov_eeg_scaling_law.png`
- `reports/figures/modality_behavioral_group_summary.png`
- `reports/figures/modality_behavioral_scaling_law.png`
- `reports/figures/modality_eeg_group_summary.png`
- `reports/figures/modality_eeg_scaling_law.png`
- `reports/figures/modality_fmri_group_summary.png`
- `reports/figures/modality_fmri_scaling_law.png`
- `reports/figures/modality_ieeg_ecog_group_summary.png`
- `reports/figures/modality_ieeg_ecog_scaling_law.png`
- `reports/figures/naturalstories_spr_overview.png`
- `reports/figures/naturalstories_spr_scaling_by_group.png`
- `reports/figures/naturalstories_spr_scaling_by_perplexity.png`
- `reports/figures/naturalstories_spr_scaling_law.png`
- `reports/figures/neural_by_group.png`
- `reports/figures/neural_vs_behavioral.png`
- `reports/figures/neural_vs_behavioral_by_group.png`
- `reports/figures/neural_vs_behavioral_by_group_embedding_best.png`
- `reports/figures/neural_vs_behavioral_by_group_embedding_best_n_params.png`
- `reports/figures/neural_vs_behavioral_by_group_embedding_best_perplexity.png`
- `reports/figures/neural_vs_behavioral_by_group_embedding_last.png`
- `reports/figures/neural_vs_behavioral_by_group_embedding_last_n_params.png`
- `reports/figures/neural_vs_behavioral_by_group_embedding_last_perplexity.png`
- `reports/figures/neural_vs_behavioral_by_group_surprisal_best.png`
- `reports/figures/neural_vs_behavioral_by_group_surprisal_best_n_params.png`
- `reports/figures/neural_vs_behavioral_by_group_surprisal_best_perplexity.png`
- `reports/figures/neural_vs_behavioral_by_group_surprisal_last.png`
- `reports/figures/neural_vs_behavioral_by_group_surprisal_last_n_params.png`
- `reports/figures/neural_vs_behavioral_by_group_surprisal_last_perplexity.png`
- `reports/figures/neural_vs_behavioral_by_range_embedding_best.png`
- `reports/figures/neural_vs_behavioral_by_range_embedding_last.png`
- `reports/figures/neural_vs_behavioral_by_range_surprisal_best.png`
- `reports/figures/neural_vs_behavioral_by_range_surprisal_last.png`
- `reports/figures/nieuwland_eeg_layer_profiles.png`
- `reports/figures/nieuwland_eeg_overview.png`
- `reports/figures/nieuwland_eeg_roi_profile.png`
- `reports/figures/nieuwland_eeg_scaling_by_group.png`
- `reports/figures/nieuwland_eeg_scaling_by_perplexity.png`
- `reports/figures/nieuwland_eeg_scaling_law.png`
- `reports/figures/ns_fmri_layer_profiles.png`
- `reports/figures/ns_fmri_overview.png`
- `reports/figures/ns_fmri_roi_profile.png`
- `reports/figures/ns_fmri_scaling_by_group.png`
- `reports/figures/ns_fmri_scaling_by_perplexity.png`
- `reports/figures/ns_fmri_scaling_law.png`
- `reports/figures/pareto_hard_case_n_params.png`
- `reports/figures/pareto_hard_case_thresholds_n_params.png`
- `reports/figures/pareto_lpp_naturalstories_embedding_best.png`
- `reports/figures/pareto_lpp_naturalstories_embedding_best_n_params.png`
- `reports/figures/pareto_lpp_naturalstories_embedding_last.png`
- `reports/figures/pareto_lpp_naturalstories_embedding_last_n_params.png`
- `reports/figures/pareto_lpp_naturalstories_grid_n_params.png`
- `reports/figures/pareto_lpp_naturalstories_surprisal_best.png`
- `reports/figures/pareto_lpp_naturalstories_surprisal_best_n_params.png`
- `reports/figures/pareto_lpp_naturalstories_surprisal_last.png`
- `reports/figures/pareto_lpp_naturalstories_surprisal_last_n_params.png`
- `reports/figures/pareto_neural_behavioral.png`
- `reports/figures/pareto_neural_behavioral_embedding_best.png`
- `reports/figures/pareto_neural_behavioral_embedding_best_n_params.png`
- `reports/figures/pareto_neural_behavioral_embedding_best_perplexity.png`
- `reports/figures/pareto_neural_behavioral_embedding_last.png`
- `reports/figures/pareto_neural_behavioral_embedding_last_n_params.png`
- `reports/figures/pareto_neural_behavioral_embedding_last_perplexity.png`
- `reports/figures/pareto_neural_behavioral_grid_n_params.png`
- `reports/figures/pareto_neural_behavioral_grid_perplexity.png`
- `reports/figures/pareto_neural_behavioral_grid_zscore.png`
- `reports/figures/pareto_neural_behavioral_surprisal_best.png`
- `reports/figures/pareto_neural_behavioral_surprisal_best_n_params.png`
- `reports/figures/pareto_neural_behavioral_surprisal_best_perplexity.png`
- `reports/figures/pareto_neural_behavioral_surprisal_last.png`
- `reports/figures/pareto_neural_behavioral_surprisal_last_n_params.png`
- `reports/figures/pareto_neural_behavioral_surprisal_last_perplexity.png`
- `reports/figures/per_dataset_group_bars_embedding_best.png`
- `reports/figures/per_dataset_group_bars_embedding_last.png`
- `reports/figures/per_dataset_group_bars_surprisal_best.png`
- `reports/figures/per_dataset_group_bars_surprisal_last.png`
- `reports/figures/perplexity_binned_trend.png`
- `reports/figures/podcast_ieeg_layer_profiles.png`
- `reports/figures/podcast_ieeg_overview.png`
- `reports/figures/podcast_ieeg_roi_profile.png`
- `reports/figures/podcast_ieeg_scaling_by_group.png`
- `reports/figures/podcast_ieeg_scaling_by_perplexity.png`
- `reports/figures/podcast_ieeg_scaling_law.png`
- `reports/figures/podcast_ieeg_timecourse.png`
- `reports/figures/provo_overview.png`
- `reports/figures/provo_scaling_by_group.png`
- `reports/figures/pythia_ckpt_meco_by_perplexity.png`
- `reports/figures/pythia_ckpt_meco_by_size.png`
- `reports/figures/pythia_ckpt_provo_by_perplexity.png`
- `reports/figures/pythia_ckpt_provo_by_size.png`
- `reports/figures/roi_profile_lpp_fmri.png`
- `reports/figures/roi_profile_ns_fmri.png`
- `reports/figures/roi_profile_podcast_ieeg.png`
- `reports/figures/roi_profile_tuckute_fmri.png`
- `reports/figures/ryskin_eeg_layer_profiles.png`
- `reports/figures/ryskin_eeg_overview.png`
- `reports/figures/ryskin_eeg_roi_profile.png`
- `reports/figures/ryskin_eeg_scaling_by_group.png`
- `reports/figures/ryskin_eeg_scaling_by_perplexity.png`
- `reports/figures/ryskin_eeg_scaling_law.png`
- `reports/figures/scaling_by_dataset_family.png`
- `reports/figures/scaling_by_dataset_family_embedding_best.png`
- `reports/figures/scaling_by_dataset_family_embedding_best_n_params.png`
- `reports/figures/scaling_by_dataset_family_embedding_best_perplexity.png`
- `reports/figures/scaling_by_dataset_family_embedding_last.png`
- `reports/figures/scaling_by_dataset_family_embedding_last_n_params.png`
- `reports/figures/scaling_by_dataset_family_embedding_last_perplexity.png`
- `reports/figures/scaling_by_dataset_family_surprisal_best.png`
- `reports/figures/scaling_by_dataset_family_surprisal_best_n_params.png`
- `reports/figures/scaling_by_dataset_family_surprisal_best_perplexity.png`
- `reports/figures/scaling_by_dataset_family_surprisal_last.png`
- `reports/figures/scaling_by_dataset_family_surprisal_last_n_params.png`
- `reports/figures/scaling_by_dataset_family_surprisal_last_perplexity.png`
- `reports/figures/scaling_saturation.png`
- `reports/figures/sparrkulee_eeg_layer_profiles.png`
- `reports/figures/sparrkulee_eeg_overview.png`
- `reports/figures/sparrkulee_eeg_scaling_by_group.png`
- `reports/figures/sparrkulee_eeg_scaling_by_perplexity.png`
- `reports/figures/sparrkulee_eeg_scaling_law.png`
- `reports/figures/sparrkulee_eeg_timecourse.png`
- `reports/figures/tuckute_fmri_layer_profiles.png`
- `reports/figures/tuckute_fmri_overview.png`
- `reports/figures/tuckute_fmri_roi_profile.png`
- `reports/figures/tuckute_fmri_scaling_by_group.png`
- `reports/figures/tuckute_fmri_scaling_by_perplexity.png`
- `reports/figures/tuckute_fmri_scaling_law.png`
- `reports/figures/ucl_reading_overview.png`
- `reports/figures/ucl_reading_scaling_by_group.png`
- `reports/figures/zuco_eeg_layer_profiles.png`
- `reports/figures/zuco_eeg_overview.png`
- `reports/figures/zuco_eeg_scaling_by_group.png`
- `reports/figures/zuco_eeg_scaling_by_perplexity.png`
- `reports/figures/zuco_eeg_scaling_law.png`
- `reports/figures/zuco_reading_overview.png`
- `reports/figures/zuco_reading_scaling_by_group.png`
- `reports/matched_scale_regression_report.pdf`
- `reports/matched_scale_regression_report.tex`
- `reports/methodology_report.md`
- `reports/methodology_report.pdf`
- `reports/model_report.md`
- `reports/model_report.pdf`
- `reports/neural_trends_findings.md`
- `reports/paper/babylm_checkpoint_scaling.pdf`
- `reports/paper/backups/20260826_pre_ohschuler_formula/perplexity_matched_scaling.pdf`
- `reports/paper/backups/20260826_pre_ohschuler_formula/pythia_checkpoint_scaling.pdf`
- `reports/paper/backups/20260826_pre_ohschuler_formula/scale_matched_comparison.pdf`
- `reports/paper/behavioral_surprisal_layerwise_by_model.pdf`
- `reports/paper/dlt_threshold_trend.pdf`
- `reports/paper/embedding_bestlayer_pareto.pdf`
- `reports/paper/embedding_bestlayer_pareto_robustz.pdf`
- `reports/paper/embedding_lastlayer_pareto.pdf`
- `reports/paper/embedding_lastlayer_pareto_robustz.pdf`
- `reports/paper/family_scaling_lastlayer.pdf`
- `reports/paper/fmri_surprisal_scaling_all.pdf`
- `reports/paper/layerwise_depth_profiles.pdf`
- `reports/paper/layerwise_depth_profiles_loglik.pdf`
- `reports/paper/lpp_fmri_surprisal_scaling.pdf`
- `reports/paper/matched_scale_regression copy.pdf`
- `reports/paper/matched_scale_regression_by_method.pdf`
- `reports/paper/matched_scale_report.pdf`
- `reports/paper/matched_scale_report_summary.md`
- `reports/paper/matched_scale_simple_regression.pdf`
- `reports/paper/noise_ceiling_summary.pdf`
- `reports/paper/pereira_fmri_scaling.pdf`
- `reports/paper/perplexity_matched_scaling.pdf`
- `reports/paper/provo_formula_ablation.pdf`
- `reports/paper/provo_ohschuler_checkpoint_scaling.pdf`
- `reports/paper/pythia_checkpoint_scaling.pdf`
- `reports/paper/roi_comparison_embedding.pdf`
- `reports/paper/roi_comparison_surprisal.pdf`
- `reports/paper/roi_scaling_lastlayer.pdf`
- `reports/paper/scale_matched_comparison.pdf`
- `reports/paper/surprisal_bestlayer_pareto.pdf`
- `reports/paper/surprisal_bestlayer_pareto_robustz.pdf`
- `reports/paper/surprisal_dlt_threshold_grid.pdf`
- `reports/paper/surprisal_hard_dlt_pareto.pdf`
- `reports/paper/surprisal_lastlayer_pareto.pdf`
- `reports/paper/surprisal_lastlayer_pareto_no_pereira.pdf`
- `reports/paper/surprisal_lastlayer_pareto_robustz.pdf`
- `reports/trends.png`

</details>

### F. How Claude should use this inventory

Implement the phased recommendations, not a filename-based cleanup script. Build an explicit migration manifest from these entries as each phase is executed: `old_path`, `action`, `new_path`, `reason`, `callers_updated`, `validation`, `rollback_path`. For a retained item, the destination is its current path. For unresolved evidence, mark `deferred` rather than inventing an unused-file verdict. A file not listed inside a third-party/data/cache directory remains retained under its directory policy, not implicitly approved for deletion.
