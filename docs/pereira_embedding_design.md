# Pereira fMRI — voxelwise embedding encoding design

Companion pipeline to `neuroai/benchmarks/pereira_lme.py` (the pooled-LME
surprisal benchmark). That one asks "does surprisal explain variance in
language-network BOLD, on top of nuisances?" This one asks the encoding-style
question instead: "how well do an LM's *hidden-state embeddings* predict
voxelwise BOLD?" — the same question `neuroai/lpp_encoding.py` answers for
LPP, `neuroai/benchmarks/tuckute_checkpoint_sweep` answers for Tuckute, and
`ns_fmri/embedding_sweep` answers for Natural Stories fMRI.

Restricted to the **language-network atlas only** (12 ROIs, 13,553 voxels,
present in every one of the 10 subjects — see `neuroai/datasets/pereira.py`
module docstring), exactly like the LME pipeline's `atlas="language"` default
and the LANG-only fix applied to Natural Stories fMRI's `LANG_MD` fROI table.

## 1. Why per-subject, not average-subject (key difference from LPP)

LPP's encoder has an explicit `fmri_unit: "single_subject" | "average_subject"`
mode, backed by a **precomputed** `average_subject_precomputed_dir` (l-bg's
`average_subject_run-<0..8>.gz` files) — a shared voxel space that different
subjects' BOLD has already been resampled/aligned into, so one ridge fit
predicts the *mean* subject's response directly.

**Pereira has no such precomputed shared voxel space.** Each subject's
`neuroid` axis in `Pereira2018_merged.nc` is that subject's own AAL-multimask
voxel set (own MNI coordinates, own voxel count per ROI — e.g. subject `018`
has a different number of `LH_AntTemp` voxels than subject `199`). There is no
existing "average subject" resource to fit into directly, and building one
from scratch (a genuine cross-subject voxel/vertex correspondence, e.g. via
cortical surface registration) is a much bigger undertaking than this
benchmark warrants.

**Design choice: fit one ridge encoding model PER SUBJECT** (10 independent
fits per layer), then **average the resulting per-voxel/per-ROI correlation
scores across subjects post-hoc** — the standard convention in this
literature when no shared voxel space exists (also what Pereira et al. 2018
and most brain-encoding papers using this dataset do). This trades "one fit
predicts everyone" for "ten fits, each subject scored on its own held-out
data, scores pooled at the end" — statistically equivalent for the group-level
question, just 10x the fitting cost per layer.

## 2. Feature extraction — one embedding vector per sentence

Pereira sentences are isolated, sentence-level BOLD (like Tuckute, unlike
LPP/NS-fMRI's continuous per-TR narrative) — no HRF convolution needed, one
feature vector per (model, layer, sentence).

**Settled — mean-pooled across words, per de Varda, Berzak, Fedorenko & Levy
(2026.08.21.746238v1, "Behavioral and Brain Responses to Language Reflect
Different Levels of Linguistic Representation")**, which runs this exact
analysis on Pereira2018 (and Tuckute2024) and states their method plainly:

> "EMB: the contextual representation of the word $w_i$, i.e., the model's
> activation in its final layer in response to $w_i$ (we chose the final
> layer... see Appendix A.5 for analyses considering all layers)."
> "For the fMRI datasets including responses to sentences, sentence-aggregated
> SFL and EMB regressors were obtained by **averaging over word-level
> predictors**."

So: extract each word's own per-layer hidden state (one forward pass over the
sentence, contextualized on everything to its left — same forward pass the
LME pipeline already does for surprisal), then **mean-pool across all words
in the sentence** to get one (n_layers, d_model) vector per sentence. This is
the literature precedent for this exact dataset, not last-token-only, and
directly contradicts what I'd tentatively guessed last turn — worth having
checked rather than assumed.

(2506.11338v2.pdf — Lin & Schuler — was checked too and turns out to be
irrelevant here: it's surprisal-only, no embedding/encoding analysis at all,
so it has nothing to say about pooling.)

Reuses the SAME per-model, per-layer hidden-state extraction infrastructure
already built for the LME pipeline (`neuroai/features/logit_lens.py`'s
hidden-state readout, not the surprisal head) — one GPU pass over all 627
sentences per model, all layers at once, mean-pooled over words before
caching, cached to disk exactly like `pereira_lme_pooled_surprisal_*.npy`
(this would be `pereira_embed_pooled_hidden_*.npy`, shape
`(627, n_layers, d_model)`).

## 3. Regression: himalaya GPU multi-output ridge

Reuses `neuroai/regressors/ridge.py::cv_ridge_score_multioutput_gpu` (the
existing SVD-based-GCV himalaya/torch backend already used by LPP,
Tuckute-checkpoint-sweep, and NS-fMRI's own embedding pipelines) rather than
writing a new ridge implementation:

- **Multi-output**: one ridge solve predicts every voxel in a subject's
  language-atlas mask at once (`Y` is `(n_sentences, n_voxels)`) — himalaya's
  SVD-GCV trick evaluates every alpha in the grid in O(k) amortized ops
  across all voxel columns simultaneously, so this is already "free"
  parallelism across voxels, not something to add separately.
- **CV**: `GroupKFold`, **5 splits** (matching the codebase-wide default in
  `cv_ridge_score_multioutput_gpu`/`cv_ridge_predict`, and what you asked to
  fix), grouped by **`passage_index`** (Pereira's 3–4-sentence passages —
  sentences from the same passage share topical/lexical content, so must
  stay together on one side of the split to avoid leakage; this is the
  direct Pereira analogue of LPP's leave-one-*section*-out grouping).
- **Nested alpha selection**: GCV is *itself* the nested step (it selects
  alpha using only the training fold's SVD, not the held-out fold) — no
  separate inner CV loop needed, unlike the plain-CPU `cv_ridge_score`'s
  explicit inner/outer split. This is the main reason to route everything
  through the GPU path rather than CPU `Ridge`/`RidgeCV`: correctness (no
  leakage) *and* speed together, not a speed-only upgrade.

## 4. Alpha range calibration (your requested first step)

Before committing to a production alpha grid, run **one model** (recommend
`openai-community/gpt2` — cheapest, already fully cached from the LME
pipeline's GPU stage) across **all 10 subjects × all layers** with a
deliberately wide grid, **`np.logspace(-3, 9, 12)`** (12 orders of magnitude,
12 points), and record the GCV-selected `best_alpha` per (subject, layer,
fold) — 10 × n_layers × 5 = up to 600 selections for GPT-2's 12 layers.

Report **min / median / max of the selected alphas** across that whole grid.
If the distribution clusters well inside the grid's interior (not pinned at
either edge), narrow to a denser grid centered on the median for the
production sweep across all 101 models — cheaper per-fit and a more precise
alpha estimate. If it's pinned at an edge, widen further before trusting any
model's score.

## 5. GPU parallelization plan

Everything data-parallel that can go on GPU, does:

- **Voxels**: free (himalaya's multi-output solve handles all voxels in one
  matrix op per alpha, per fold — see §3).
- **Layers**: `torch.cuda.stream()`-based concurrency, same pattern already
  used in `neuroai/benchmarks/zuco_eeg.py` (`ZuCoEegEncoder`) and referenced
  in `cv_ridge_score_multioutput_gpu`'s own docstring ("call from within a
  `torch.cuda.stream()` context for layer-parallel sweeps") — launch each
  layer's ridge fit on its own CUDA stream so the GPU pipelines independent
  layers' matmuls instead of running them serially.
- **Subjects**: independent problems (different voxel counts, different
  design matrices) — same CUDA-stream-per-task approach, or simply looped if
  a single fit is already GPU-saturated (10 subjects × up to ~40 layers is
  400 independent small-to-medium ridge problems; stream-parallelism matters
  most for the smaller subjects/layers where a single fit doesn't fill the
  GPU alone).
- **Models**: NOT parallelized within one job (same one-job-per-model
  convention as the LME sweep) — 101 models still means 101 SLURM tasks, each
  internally using the above three layers of GPU parallelism.

## 6. Output & aggregation

Per model, per subject: `pereira_embed_by_layer.csv` — one row per
(layer, fold), columns `alpha, pearsonr_mean, spearmanr_mean, n_voxels,
n_obs_train, n_obs_test` (mirroring `LppLayerwiseEncodingResult`'s schema),
plus a per-voxel `.npy`/`.npz` correlation map (voxel → (subject, roi,
mni-coords) already available from `PereiraDataset`) for ROI-level breakdown.

Aggregation ladder, same shape as the LME pipeline's final-checkpoint
analysis:
1. Per subject: mean correlation across that subject's language-atlas voxels
   (optionally broken down by the 12 named ROIs, same as the LME's per-ROI
   diagnostic).
2. **Across subjects: plain mean** of the 10 per-subject scores (this
   replaces LPP's single average-subject fit — see §1) — this is the
   headline "language-network encoding score" per (model, layer).
3. Best-layer vs. last-layer picking done exactly like the rest of this
   session's diagnostics (given the demonstrated last-layer bias for LPP,
   default to reporting **best-layer-per-subject**, not last-layer, unless
   you want last-layer for direct comparability with the LME sweep).

## 7. Compute plan

1. **Alpha-calibration job** (GPU, `qos_gpu_h100-dev`, single model): confirm
   the wide-grid `best_alpha` distribution, narrow the grid.
2. **Full sweep**: one SLURM job per model (101 total, same registry-family
   array-job convention as `pereira/surprisal_sweep`), each job doing all
   subjects × all layers × 5-fold GroupKFold-GCV on GPU with the narrowed
   grid.

## Decisions (settled)

1. **Sentence embedding = mean-pooled word-level, per-layer hidden states**
   (§2), per de Varda/Berzak/Fedorenko/Levy's own Pereira2018 methodology —
   not last-token.
2. **All layers** (matches the LME full sweep, lets best-layer be picked
   after the fact rather than baking in the last-layer bias found for LPP).
3. **Alpha-calibration grid: `np.logspace(-3, 9, 12)`**, on
   `openai-community/gpt2` (cheapest, already fully cached from the LME
   pipeline's GPU stage).

Ready to move to implementation whenever you want — next step would be the
feature-extraction code (mean-pooled per-layer hidden states, reusing
`logit_lens.py`'s hidden-state readout) plus the himalaya GPU ridge wiring.
