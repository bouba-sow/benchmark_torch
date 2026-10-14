# RDM regression track (held-out delta log-likelihood)

Complementary track. The pooled-LME surprisal results and the ridge embedding
results remain the benchmark headline; this track does not replace them.

Code: `neuroai/analysis/rdm_regression.py` (tests: `tests/test_rdm_regression.py`).

## What is computed

For one dataset (and measure, and subject for neural data), one model, one layer:

```
base:  response_RDM ~ 1 [+ confound RDMs]
full:  response_RDM ~ 1 [+ confound RDMs] + model_RDM
delta_ll_per_obs = (LL_full - LL_base) / n_test_pairs        (held-out, Gaussian OLS)
```

* Two variants per model: `confounds = none` and `confounds = word length + frequency`
  (RDMs from `scripts/analysis/save_lexical_confounds_rdms.py` and
  `save_neural_lexical_confounds_rdms.py`; absolute differences of the scalar feature).
* Model RDM: embedding distance (one per layer, best layer chosen by held-out delta LL,
  as in the other tracks) or |delta surprisal| (surprisal track, single "layer").
* Response RDM: grand-average-subject RDM for behavioral datasets (matches the
  averaged-subject embedding sweep); per-subject RDM for neural datasets, then averaged
  across subjects.
* Noise variance is the ML estimate on training pairs; log-likelihood is evaluated on
  test pairs. Base and full models are scored on identical test pairs.

## Cross-validation over items, not pairs

RDM entries are pairs of items and are not independent (each item is in n-1 pairs).
Splitting pairs at random leaks: a test pair shares an item with many training pairs.
So we split items:

1. Assign items to K folds (K from the schedule below), at random.
2. For fold f: train on pairs with **both** items outside f; test on pairs with **both**
   items inside f. Pairs with one item in f are discarded.
3. Sum test log-likelihoods over folds; repeat with new random partitions
   (`n_repeats`, default 10) and average per-observation values.

Only about 1/K of all pairs are scored per repeat (`frac_pairs_tested` is reported).
Sampled-pair datasets (n(n-1)/2 > 250k, see `MAX_PAIRS` in `surprisal_rsa.py`) use the
same rule on the cached `pair_indices`.

The independence assumption inside the Gaussian likelihood is still violated (pairs
sharing an item within the training or test set remain correlated). This inflates
absolute delta LL and any naive standard errors roughly equally for all models, so the
track is used for **ranking models**, not for calibrated p-values or absolute effect
sizes. Uncertainty should come from resampling stimuli, not pairs (see below).

## Aggregation

Per dataset only. Pairs are never pooled across datasets (different units, noise
levels and RDM sizes). Across-dataset composites use the median of per-dataset scores,
as elsewhere in the benchmark. No RDM-level noise ceiling exists yet, so scores are raw
delta LL per pair until a split-half RDM ceiling is added.

## Literature and what is ours (for the paper)

Verified against the PDFs in `papers/`.

**Schutt, Kipnis, Diedrichsen & Kriegeskorte (2023). Statistical inference on
representational geometries.** (`papers/Schütt, Kriegeskorte et al.pdf`; the PDF is the
July 2023 preprint. Published in eLife, article 82566, per the eLife URL; confirm
volume/year before citing. Section and fold numbers below are from the preprint.)

* Adopted: fit on dissimilarities among fitting stimuli, evaluate on dissimilarities
  among held-out test stimuli (2-factor cross-validation, condition factor only).
* Adopted: fold count by number of conditions: 2 folds for few conditions, 3 for >=12,
  4 for >=24, 5 for >=40 (`n_folds_for`). The authors call these values ad hoc; minimum
  is 6 conditions.
* Not adopted (possible extension): held-out *subjects*, the 2-factor bootstrap over
  subjects and stimuli, and the correction for excess cross-validation variance.
  Confidence intervals for this track should be a bootstrap over stimuli following
  that paper.

**Kaniuth & Hebart. Feature-reweighted representational similarity analysis: a method
for improving the fit between computational models, brains, and behavior.**
(`papers/FR_RSA.pdf`; the PDF is the bioRxiv preprint of 28 Sep 2021,
doi 10.1101/2021.09.27.462005. Look up the published journal reference and year before
citing; the code comments say "Kaniuth & Hebart" without a year for this reason.)

* Adopted: data split along the condition axis, so training and test sets contain
  non-overlapping condition pairs; whole objects are left out rather than individual
  pair similarities, to avoid leakage because pair similarities are not independent
  (they contrast this with Peterson et al., 2016, who left out individual pairs, and
  cite Jozwik et al., 2017 for the object-level scheme).
* Adopted: repeat the outer split with different random partitions (they use 5-fold
  repeated 10 times) so that many pairs are eventually test pairs (`n_repeats=10`).
* Not adopted: ridge regression with nested cross-validation for the regularisation
  parameter (we use plain OLS with a few regressors, no tuned hyperparameter).

**Our own choices (not from either paper):** scoring with held-out Gaussian
log-likelihood and delta LL per observation (chosen to mirror the benchmark's
`delta_loglik_test / n_obs_test` convention; both papers score with correlation-type
measures), the with/without-confound base models, the grand-average behavioral RDM,
and using only within-fold test pairs for the likelihood.

## Running it

Driver: `scripts/analysis/rdm_regression_behavioral.py <dataset>` for provo, meco, geco,
onestop, naturalstories, ucl_reading, zuco_reading (embedding side of the last five in
`rdm_regression_behavioral_other.py`). SLURM wrapper:
`scripts/slurm/jz/analysis/rdm_regression_behavioral.sh <dataset> <both|embedding|surprisal> [summarize]`,
submitted as an array (`SHARD_COUNT` defaults to the array size; each task scores every N-th
model). Output root `$RDM_REGRESSION_OUT` (default `$SCRATCH/rdm_regression`); the driver
refuses to write under `outputs/`. Runs are resumable; per-model CSVs are skipped if present.
Item set, pairs and confound RDMs are those of the shared behavior RDM
(`outputs/<ds>/pearson_rsa/surprisal/rdm_cache/<measure>/shared/behavior_rdm.npz`);
model RDMs are rebuilt from per-item values on that pair set (embedding: correlation
distance per layer; surprisal: |delta surprisal|). Datasets whose shared RDM has 250k
sampled pairs use those pairs for the CV as well.

Outputs: `<root>/<dataset>/<track>/<measure>/<family>/<model>/rdm_regression_<rev>.csv`
(one row per layer x confound variant), `<root>/<dataset>/<track>/model_summary.csv`
(best and last layer per model), and `scripts/analysis/rdm_regression_combine.py --root <root>`
writes `all_model_summaries.csv`. `rsa_pearson` in every row is the plain fit-free RSA
correlation (matches the existing RSA results, e.g. Provo FP GPT-2 last layer 0.016487).
The `best_layer` columns of `model_summary.csv` choose the layer on the same held-out delta LL that is
reported. That is the optimistic rule the surprisal and embedding tracks dropped on 2026-10-01 (they now
select the layer on TRAIN delta LL; see `docs/architecture.md`). It is NOT corrected here because this
track stores no training-side delta LL (re-selecting would need `RdmDeltaLL` to also return the in-sample
value and a rerun of all layers). Every reported result of this track uses the LAST LAYER only
(`last_layer_delta_ll_per_obs`), which involves no layer selection, so the headline numbers are unaffected;
treat the `best_*` columns as exploratory.

Known difference from the earlier embedding RSA: for OneStop the item set is the shared
behavior-RDM one (words with >= 5 subjects, ~24k items per measure), while the older
`additional_behavioral_embedding_rsa.py` RSA used all ~31k items and its own pair sample,
so its OneStop `rsa_pearson` differs. For GECO, UCL, Natural Stories, ZuCo, Provo and MECO the
last-layer `rsa_pearson` reproduces the existing RSA values to ~1e-8.

## Neural datasets

Driver: `scripts/analysis/rdm_regression_neural.py <dataset>` (pereira, tuckute, fedorenko2016,
broderick, megmasc, nieuwland, szewczyk, podcast); the same sbatch wrapper dispatches to it by
dataset name. LPP and Natural Stories fMRI are not covered (no response RDM exists for them).

* Response unit = one subject: the shared per-subject neural RDM
  (`outputs/<ds>/pearson_rsa/surprisal/rdm_cache/shared/<ds>_<subject>.npz`, full pair set for
  small stimulus sets, fixed 250k-pair sample otherwise) with that subject's confound RDMs.
  Each subject is cross-validated on its own items; the summary averages delta LL over subjects
  at the layer that is best for the subject mean (`n_subjects` is reported).
* Both tracks use the stimuli that have a surprisal value (the existing surprisal RSA used the
  same intersection). This only bites for Fedorenko (333 of 416 words kept per subject).
* Embedding rows: id -> cache row per dataset (word_idx for Broderick/MEG-MASC/Podcast, item id for
  Nieuwland, per-subject chunk -> item for Szewczyk, sorted stimulus id for Pereira, condition
  order for Tuckute). One layer-normalised Gram matrix over the union of items per layer serves
  all subjects.
* Validation (last-layer `rsa_pearson`, per subject, vs the existing RSA outputs, smallest model):
  identical to ~1e-6 or better for MEG-MASC, Podcast, Szewczyk, Nieuwland (embedding), Pereira
  and Fedorenko (surprisal). Expected differences: Tuckute (existing RSA used full RDMs, ours the
  250k-pair sample, ~1e-3); Fedorenko embedding (existing used 416 words, ours 333);
  Broderick embedding (existing used 11,389 words from a newer table, the shared RDM has 11,386);
  Nieuwland surprisal (the surprisal cache was regenerated on 2026-09-29 20:05, after the existing
  RSA file, so the existing per-subject numbers there are stale).
* Pereira embedding has no like-for-like existing RSA (the old one is per experiment).

### Grand-average N400 variant (`nieuwland_avg`, `szewczyk_avg`)

With one RDM per subject every model scored below the no-model baseline on both N400 datasets
(single-subject N400 amplitudes are too noisy; the existing RSA is also ~0). These two datasets are
therefore also run with one grand-average response RDM: the mean N400 of each item over the subjects who
saw it (items need >= 5 subjects; Nieuwland 160 items, ~162 subjects/item; Szewczyk 1,325 items, ~21
subjects/item, where items are identified by sentence text, not by the counterbalanced chunk ID), RDM =
|difference of item means|, with the items' word-length/frequency confounds and full (unsampled) pair
sets. Results are kept under the separate dataset names `nieuwland_avg` / `szewczyk_avg`; the per-subject
runs remain available. Stimuli are restricted to those with surprisal for every model
(`--build-eligible`). The per-subject and averaged variants answer different questions
(single-subject vs group-level RDM) and should be reported as such.

## Stimulus bootstrap (last layer)

Code: `neuroai/analysis/rdm_bootstrap.py` (method), `scripts/analysis/rdm_bootstrap_run.py` (production run),
`scripts/slurm/jz/analysis/rdm_bootstrap.sh`; tests `tests/test_rdm_bootstrap.py`; simulations
`scripts/analysis/rdm_bootstrap_calibration.py`.

* **From Schutt et al. (2023, "Bootstrapping conditions")**: resample the stimuli with replacement, rebuild the
  data and model RDMs for the resample, rescore; the variance over resamples estimates the variance over
  experiments with different stimuli. Repeated stimuli have dissimilarity 0 with their own copy in the data
  and in every model, which would bias all scores upward, so these self-pairs are excluded. They recommend 2
  cross-validation cycles per bootstrap sample and correct the variance for the extra cross-validation noise.
* **Ours**: a resample is a vector of item counts `c`; pair (i, j) gets weight `c_i * c_j` (self-pairs are not in
  the list), fits are weighted least squares, test log-likelihoods weight-summed. This equals fitting on the
  explicitly duplicated stimuli (tested), with copies of an item always in the same fold. The partition-noise
  correction is `Var(resample scores) - mean within-resample partition variance / n_partitions`
  (n_partitions = 2, 200 resamples). The 95% interval is the percentile interval of the resample scores (it still
  contains some partition noise, so it is slightly conservative). The corrected SE is 0 when the estimated
  partition noise exceeds the total spread (near-null models); read the interval then. `boot_sd_uncorrected` is
  the uncorrected spread.
* **Shared resamples**: resamples are drawn once per dataset over the union of its stimuli, so subjects and
  measures that share stimuli share the resample (averaging over subjects does not shrink the stimulus-sampling
  variance); fold partitions are fixed per unit, so every model and both tracks are paired.
* **Validation (simulation, known population)**: bootstrap SE / true sampling SD = 0.90 to 1.16 over
  40 to 3,000 items, with full or sampled pairs; 95% interval coverage 92 to 100%; paired model differences
  behave the same. It is slightly conservative, never too small; at 40 items it is clearly conservative.
  The uncorrected SE is 1.0 to 3x too large, so the correction matters.

## Noise ceilings (last layer)

Code: `neuroai/analysis/rdm_ceiling.py`, `scripts/analysis/rdm_ceiling_run.py` (split-half),
`scripts/analysis/rdm_ceiling_neural.py` (leave-one-subject-out), `scripts/analysis/rdm_report.py` (join);
tests `tests/test_rdm_ceiling.py`.

* **Behavioral measures and the averaged N400 datasets: split-half, like-for-like.** This is the variance-ceiling
  recipe of Timkey et al. (2026, PNAS supplement; already used for the LME ceilings in
  `scripts/ceilings/dll_behavioral.py`) transposed to RDM space: split the participants into two equal random
  halves, use one half's item-mean RDM as the predictor of the other half's item-mean RDM (same item-fold
  cross-validated pair likelihood, same confound variants), sum both directions (total delta LL / total test
  pairs), 15 random splits, 5 fold partitions each. Items need an observation in both halves (same pairs in both
  directions). **The models are scored on the same half targets** (half-target delta LL), because a half-sample
  target RDM is noisier than the grand-average RDM: ratio = half-target model score / ceiling is then
  like-for-like. The full-target delta LL stays the main number. The ceiling is conservative by construction
  (predictor from half the participants); no Spearman-Brown-type correction exists on the delta-LL scale, so
  none is applied (see van Bree et al., clarifying note: split-half reliability is a variance, not a
  correlation, ceiling).
* **Per-subject neural datasets: leave-one-subject-out.** Each subject's own RDM is the target (as for the
  models); the predictor is the mean of the other subjects' RDMs on the subject's pairs (Nili et al. 2014 lower
  noise ceiling, as described in Kaniuth and Hebart). Models and ceiling share the target, so no rescoring.
  The upper bound (subject included in the mean) is not computed.
* **Reading the neural ceilings**: they are very small (about 5e-3 for Pereira, 1e-5 to 3e-4 for Tuckute,
  MEG-MASC, Fedorenko). A diagnosis on 4 subjects per dataset showed this is a property of the RDMs, not a
  bug: delta LL ceiling = rho^2/2 with rho the cross-subject correlation of the RDM entries (checked: Pereira
  subject 018 rho = 0.139 gives 0.0097, measured 0.0099); the correlation-distance RDM over only 4 to 12
  channels per item (6 ROIs Tuckute, 4 regions Broderick, 6 MEG-MASC, 12 Pereira) is almost noise and discards
  response amplitude, and word-level EEG/MEG responses (each word heard once) have between-subject item
  reliability around 0.01. Euclidean distance on z-scored channels would be 10 to 100 times more reliable for
  Tuckute, Broderick, Fedorenko, Podcast; we kept the original definition because the benchmark ranks models,
  and report the ceilings next to the scores.
* **Reporting**: always give the raw delta LL and the ceiling next to the ratio. The ratio is unstable when the
  ceiling is ~0. (A blog post by J. Carlin on RSA noise ceilings vs Spearman-Brown, not peer reviewed, makes the
  same point: dividing by the ceiling hides what the ceiling was.)

### Results overview (last layer; `$SCRATCH/rdm_report/`, made by `scripts/analysis/rdm_report.py`)

* Neural per-subject datasets, leave-one-subject-out bracket (none-variant), delta LL per pair:
  lower bound / upper bound = Broderick ~0 / 0.029, Fedorenko 0.0003 / 0.120, MEG-MASC 0.00002 / 0.049,
  Pereira 0.0045 / 0.097, Podcast 0.0017 / 0.072, Tuckute 0.000002 / 0.054. No model row (0 of 236 per dataset)
  exceeds the upper bound, so nothing indicates leakage or overfitting; models above the lower bound (Fedorenko 77,
  Tuckute 92 of 236) only beat a noisy mean of 4 to 9 other subjects. Best models reach 29% (Pereira, embedding)
  and 26% (Podcast, embedding) of the lower bound.
* Split-half (behavioral, averaged N400): best model / half-split ceiling is 1 to 9% for Provo, MECO, Natural
  Stories, ZuCo; 10 to 26% for the best OneStop embedding; 31% (GECO FP), 33% (UCL) and 69% (Szewczyk averaged)
  for the best surprisal model.
* Bootstrap: number of the 118 models whose 95% interval excludes 0 (no confounds), e.g. Provo FP 108 surprisal /
  52 embedding, MECO SP 95 / 85, Natural Stories 67 / 78; with word length and frequency added the counts drop
  (Provo FP surprisal 58). Pereira surprisal: 64 models are significantly BELOW 0 (the extra predictor hurts
  held-out fit). Neural datasets with almost no signal (MEG-MASC, Tuckute, Broderick, Nieuwland averaged) have no
  model with an interval above 0.

**Definition used for all reported results and figures (decision 2026-10-01):** ΔLL is always against the
confound-only model, ΔLL = LL(1 + length RDM + frequency RDM + model RDM) - LL(1 + length RDM + frequency RDM),
held out (the `len_freq` rows). The `none` rows (baseline = intercept only) stay in the result tables for
reference; the difference ΔLL(none) - ΔLL(len_freq) is the part of the model's explanatory power shared with the
confounds (commonality analysis), and is not shown in the figures.
