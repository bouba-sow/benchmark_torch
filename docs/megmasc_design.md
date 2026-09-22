# MEG-MASC benchmark — design & methods

Reference implementation notes for the **MEG-MASC** (Gwilliams et al. 2023)
MEG encoding + surprisal-LME benchmark. Written to be citable from the paper.

## 1. Dataset

**Source.** Gwilliams, L., Flick, G., Marantz, A., Pylkkänen, L., Poeppel, D.,
King, J.-R. (2023). *Introducing MEG-MASC: a high-quality magneto-encephalography
dataset for evaluating natural speech processing.* Scientific Data 10, 862.
OSF `ag3kj`; code `github.com/kingjr/meg-masc`.

**Task.** Participants listened to four fictional stories from the Manually
Annotated Sub-Corpus (`lw1`, `cable_spool_fort`, `easy_money`,
`The_Black_Willow` — the dataset's `events.tsv` spell it *Willow*; the README
typo's *Widow*), synthesized with macOS text-to-speech, interspersed with
random word lists and comprehension questions. Word/phoneme onsets come from
Gentle forced alignment and are stored in the `trial_type` column of each
`events.tsv` as a stringified Python dict.

**Acquisition.** KIT/Yokogawa whole-head system, **208 axial gradiometers**,
1000 Hz. The 50 Hz power-line peak (verified in the raw PSD) places acquisition
at **NYU Abu Dhabi**, not New York.

**Local subset.** The full release has 27 participants; this benchmark uses the
**11 (`sub-01` … `sub-11`)** currently mirrored locally. All 11 have complete
data; every one of their 208 MEG channels is flagged `good` in `channels.tsv`.
10 subjects did two identical sessions; `sub-03` did one.

**Session repeats.** The two sessions present the **identical stimulus** — same
four stories in the same task order, the *same* rendered `.wav` files, same
voice, same per-word speech rate; within-sound-file inter-word gaps correlate
r ≈ 0.994 across sessions (verified). The only session difference is the pause
between the ~3-min sound-file chunks, which lies outside every word-locked
epoch. This makes the two sessions a genuine same-stimulus repeat, used for
the noise ceiling (§5).

## 2. Preprocessing

One recipe, applied to the raw `.con` per (subject, session, story), adapted
from the LittlePrince M/EEG paper (Bel et al., arXiv:2412.17829) with the
post-onset window stretched from 0.6 to **0.8 s**:

| step | value |
|---|---|
| read | `mne.io.read_raw_kit`, MEG channels only (no head digitization — not needed for sensor space) |
| band-pass | **[0.1, 40] Hz** zero-phase Butterworth, applied at 1000 Hz |
| notch | **none** — 50 Hz mains is outside the pass-band (holds for both MEG datasets) |
| resample | **50 Hz** (polyphase) |
| scaling | per-channel `RobustScaler` (median / IQR) on the continuous signal, then clamp to **[−5, 5]** |
| epoch | **[−0.2, 0.8] s** relative to word onset (50 bins) |
| baseline | subtract the mean over **[−0.2, 0] s** |
| word filter | keep `kind == "word"`, `condition == "sentence"`, `pronounced == 1`; drop words shorter than 50 ms |

Word onsets are **recording-relative** (each recording has a variable silent
lead-in), so the canonical word list — extracted once for LLM features and
shared across subjects — carries no onsets; per-recording onsets are joined
back to it by within-recording word index.

**Epoch cache.** The bandpass + resample + RobustScaler + epoching is
model-independent and is the slow step (~8 KIT reads per subject at ~30 s
each). It is precomputed once per subject (`scripts/slurm/torch/utils/data_prep/megmasc_prime_cache.sh`,
CPU only) into one `.npz` keyed by an MD5 of the preprocessing parameters;
all model sweeps read the warm cache and do no raw-MEG I/O.

## 3. Encoding benchmark

Predicts word-locked MEG from LLM hidden states, per channel, with a
lag-matrix ridge (predicts all 50 lags jointly) and **leave-one-story-out**
cross-validation (4 folds; the two sessions of a story share a fold).
Per-layer sweep; brain score = mean over lags of the median-over-folds
Pearson r; the reported `mean_best_pearsonr` takes each channel's best layer,
averaged over channels and the 11 subjects.

Single GPU per model, sequential subjects, all 208 channels batched per layer
(`cv_ridge_score_lag_matrix_gpu_batched`) — this is the Podcast-iEEG pattern,
*not* SparrKULee's 4-GPU MPS fan-out, which is only justified by its 85
subjects. **Layer threads set to 1**: running ≥2 concurrent CUDA
`torch.linalg.eigh` calls SIGSEGVs for `d_model ∈ {2560, 4096}` on the Jean
Zay H100 stack.

**Result (45 checkpoints, 10 model families).** Best-layer r ranges
0.016 (Pythia-14M) → 0.023 (Mamba2-2.7B), mean 0.021 over all 208 sensors.
There is a clear jump from the smallest models to ~100 M parameters and then
**no scaling** past that. Averaged over lags the number is small because most
of the 208 sensors carry little language signal and the pre-/post-plateau
lags are near zero; at the peak lag the grand-mean r reaches ~0.034 (§6).

## 4. Surprisal LME (Δ log-likelihood)

Mirrors the two continuous-listening EEG datasets in this codebase
(`broderick_lme`, `sparrkulee_lme`).

**Outcome.** One scalar per (word, sensor region): the **mean MEG amplitude
over 250–450 ms** post word onset (see §6 for why this window), on the
preprocessed [0.1, 40] Hz / 50 Hz / [−0.2, 0] s-baselined epochs. Sensors are
averaged within six coarse KIT groups derived from sensor position
(`megmasc_sensor_region`): left/right hemisphere × anterior / central /
posterior. This grouping is a random-effect factor, not an anatomical claim
(that needs source localization, §7).

**Feature.** Per-layer logit-lens surprisal (Kuribayashi et al. 2025) — a
scalar per (word, layer) from the model's own final-norm + unembedding at each
layer, one forward pass per story. `prev_surprisal` is that array shifted one
word within a story.

**Model — `pooling = "subject"` (default): ONE pooled fit per layer, all
subjects together:**

```
nuisances = worddur_z + wlen_z + freq_z + prev_worddur_z

baseline:  y ~ nuisances + (1 | subject) + (1 | region)
full:      y ~ nuisances + surprisal_z + prev_surprisal_z + (1 | subject) + (1 + surprisal_z | region)
```

- **Pooled, not two-stage.** `region` is the *same* fixed six-way sensor
  grouping for every subject (unlike per-subject anatomy in an ROI design),
  so pooling — crossing `subject` with `region` — is well-posed here, where
  it was not for Broderick / SparrKULee (which went two-stage: fit per
  subject, then average). Validated: the pooled fit converges in ~6 min on
  the full ~250k-row table (11 subjects × 6 regions × words). Set
  `lme.pooling: "none"` for the two-stage per-subject-then-average fallback
  (formula then `y ~ nuisances [+ surprisal terms] + (1 + surprisal_z | region)`,
  no `subject` term, β and Δ log-lik averaged across subjects, mean ± SEM).
- **`subject` gets a random intercept only, not a random slope.** pymer4's
  held-out `predict` rejects a random-*slope* grouping factor ("Invalid
  grouping factor specification, subject"), which would kill the
  `delta_loglik_test` metric; `(1 | subject)` also matches the codebase's
  working N400 pooled LMEs. `region` keeps its random slope on
  `surprisal_z` — the scientifically informative one (sensor clusters
  differ in surprisal sensitivity).
- **`worddur_z` and `prev_worddur_z`** are added to the Broderick/SparrKULee
  nuisance set (`wlen_z + freq_z`) because MEG-MASC is *spoken*: the
  post-onset amplitude is strongly acoustically driven, `wlen_z`
  (characters) is a weak proxy, and acoustic word duration is available
  directly. `prev_worddur_z` is the acoustic analogue of `prev_surprisal_z`
  — with ~550 ms mean word spacing and a 250–450 ms window the window
  catches the tail of the previous word, so both linguistic and acoustic
  spillover need a control.
- A **parity** baseline (`wlen_z + freq_z` only) is also run so the number is
  directly comparable to Broderick / SparrKULee; the gap between the two
  estimates is the acoustic-confound magnitude.
- **Barr et al. (2013) fallback ladder** on non-convergence: drop the
  `region` slope-intercept correlation (`||`), then drop the `region` random
  slope entirely — applied identically to both models so the LRT stays
  valid.
- **Split** by story (`train_frac = 0.75` → 3 train / 1 held-out), so a story
  is never split across the train/test boundary; Δ log-likelihood is
  evaluated on the held-out story.

Primary metric: **Δ log-likelihood (test)** = full − baseline, per (subject,
layer), averaged over subjects; the summary reports the best layer.

**Perplexity.** The Stage-1 GPU job also computes each model's stride-based
sliding-window perplexity on the MEG-MASC corpus (the four story word lists,
English) — same recipe as every other `*_perplexity/` in the codebase — and
writes `outputs_jeanzay/megmasc/perplexity/<model>/perplexity.json`. Folded
into the surprisal-extraction job so it is not a separate pass;
`scripts/perplexity/megmasc.py` also runs standalone.

## 5. Noise ceiling

**Within-subject session test-retest** — the two identical sessions are two
noisy draws of the same word-locked response. Per channel: the mean over the
50 lags of the across-words Pearson r between session A and session B
(matched to the encoder's `mean_best_pearsonr`). 10 of the 11 subjects
contribute (`sub-03` is single-session).

Reported three ways (mean over 208 sensors × 10 subjects):

| quantity | value | interpretation |
|---|---|---|
| `r_tt` — raw single-session reliability | **0.012** | how repeatable one session's sensor response is |
| `sqrt(r_tt)` | 0.097 | ceiling for predicting a single trial (what the encoder does) |
| **Spearman-Brown** `2r/(1+r)` | **0.024** | ceiling for predicting the two-session average |

`p95` over sensors: `r_tt` 0.037, `sqrt` 0.19. Two subjects (`sub-02`,
`sub-06`) have near-zero / negative cross-session agreement — likely a
head-position shift — and are candidates for exclusion.

**Reading.** The best model's whole-lag `mean_best_pearsonr` (~0.023) is ≈ the
Spearman-Brown ceiling and ~24 % of the single-trial ceiling: at the sensor
level the models are close to the reliability limit. MEG single-word sensor
responses are near-unrepeatable session to session (`r_tt` ≈ 0.01); the small
absolute encoding r is the data, not the models. Real head-room for a
scaling / layer story requires source space + a language ROI (§7) and/or the
Δ log-likelihood framing (§4), which is designed to detect a small consistent
effect of one regressor across many noisy trials.

## 6. Encoding time course → the 250–450 ms window

`megmasc_mean_timecourse.csv` gives each model's best-layer r as a function
of lag (mean over 208 sensors × 11 subjects). Across all 45 completed
checkpoints in 10 families the shape is the same: flat during the pre-onset
baseline, a sharp rise from ~50–100 ms, a **broad plateau ~200–500 ms**, a
**peak at ~280 ms** (grand-mean r ≈ 0.034), then a slow decline (state-space
models — Mamba, Mamba2 — are flatter and appear to peak ~440 ms, i.e. no
sharp peak on the plateau). An early window (~80–180 ms) sits barely above
the pre-onset floor (grand-mean r ≈ 0.009 at 100 ms) — there is no early
auditory-lexical component for LLM features at the sensor level.

The **250–450 ms** LME window is therefore centred on the peak/plateau and
carries ~97 % of the peak signal at 350–400 ms. It is ~50 ms earlier than the
classic 300–500 ms EEG N400 window used for Broderick / SparrKULee; the
300–500 ms window is also run as a secondary for cross-dataset comparability
and still captures ~95 % of the plateau.

Plot: `outputs_jeanzay/megmasc/embedding_sweep/summary/megmasc_timecourse_by_family.png`
(`scripts/figures/megmasc_timecourse.py`).

## 7. Known limitations / roadmap

- **Sensor space.** All results are over the 208 KIT sensors. There is no
  anatomical ROI at the sensor level; the six `megmasc_sensor_region` groups
  are position bins, not brain regions.
- **Source-space v2.** The data support template (fsaverage) source
  localization: every subject has `.mrk` markers + ELP/HSP headshape +
  `coordsystem.json`. No individual MRIs are distributed (22/27 subjects
  never had one), so all 11 local subjects would use the fsaverage template
  brain — the standard MEG-MASC source pipeline (as in Gwilliams et al. and
  Défossez et al.). A volumetric source estimate morphed to MNI152 lets the
  benchmark's existing Fedorenko language mask
  (`data/lpp_fmri_extra/masks/allParcels-language-SN220-hgwm.nii`; 10 parcels, LH/RH ×
  {IFGorb, IFG, MFG, AntTemp, PostTemp}) be applied exactly as for the fMRI
  datasets, giving directly comparable per-region scores. The LLM feature
  caches are representation-agnostic, so source space reuses them — no model
  resubmission, only an added projection stage + a fast per-parcel ridge
  re-run. MEG is insensitive to radial/deep sources, so `LH_IFGorb` and
  `LH_AntTemp` will be the shaky parcels and `LH_IFG` / `LH_MFG` /
  `LH_PostTemp` the reliable ones.

## 8. Files

```
neuroai/datasets/megmasc.py               dataset adapter (KIT read, dict-events parser, sensor regions)
neuroai/benchmarks/megmasc.py             MegMascMegEncoder (encoding)
neuroai/benchmarks/megmasc_lme.py         MegMascLme  (surprisal Δ log-lik, pooled per layer, or per-subject)
neuroai/regressors/megmasc_lme.py         formulas (pooled + per-subject) + one-layer fits
scripts/run/embedding/neural/megmasc.py   encoding sweep CLI
scripts/run/surprisal/neural/megmasc_fit.py          LME sweep CLI (CPU, reads a surprisal cache)
scripts/run/surprisal/neural/megmasc_extract.py   logit-lens surprisal extraction (GPU) → cache; also computes perplexity
scripts/perplexity/megmasc.py     corpus perplexity → outputs_jeanzay/megmasc/perplexity/ (standalone or via Stage 1)
scripts/data/megmasc_epoch_cache.py      one-subject epoch-cache builder (CPU)
scripts/ceilings/correlation_megmasc.py  session test-retest ceiling (CPU)
scripts/figures/megmasc_timecourse.py        the §6 figure
configs/embedding/neural/megmasc.yaml     encoding config
configs/surprisal/neural/megmasc.yaml            LME config
configs/subjects/megmasc_subjects.json             the frozen 11-subject list
scripts/slurm/torch/megmasc_*.sh             per-family SLURM launchers (1 GPU, -C h100, 6 h)
scripts/slurm/torch/utils/data_prep/megmasc_prime_cache.sh   CPU epoch-cache prime array
scripts/slurm/torch/utils/ceilings/correlation.sh CPU noise-ceiling array
```
