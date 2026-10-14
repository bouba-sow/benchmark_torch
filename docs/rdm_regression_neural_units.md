# RDM regression for the neural datasets: one regression per response unit

Replaces the earlier neural design (one correlation-distance RDM per subject over all regions/electrodes). A correlation
distance across regions measures the *pattern* across regions and ignores overall amplitude, while surprisal mostly changes
the response amplitude, so that target had almost nothing to predict. Here every response unit (subject x region/channel/
electrode ...) gets its own scalar response RDM, as in the surprisal LME. Plain OLS, no random effects. The behavioral
track (docs/rdm_regression.md) is unchanged.

## Common design

For every dataset, track (surprisal, embedding), model (118 registry models) and unit:

```
response RDM : |z(y_i) - z(y_j)|            y = scalar response of the unit to item i, z-scored across the unit's items
confounds    : |d covariate_k|               the LME nuisances (word/sentence length, frequency); see table
base         : response ~ 1 + confounds
full         : response ~ 1 + confounds + model RDM
delta_ll_per_obs = (LL_full - LL_base) / n_test_pairs     held-out, Gaussian OLS (neuroai/analysis/rdm_regression.py)
```

* **Item-fold cross-validation**: folds over items (a test pair has both items held out), repeated 10 times; the folds and
  the sampled pairs are fixed per (dataset, subject) so all models are scored on identical splits.
* **Surprisal track**: model RDM = |d surprisal| (last layer, the same value the LME uses).
* **Embedding track**: model RDM = 1 - Pearson between last-layer item vectors, each feature z-scored across the items
  (Timkey & van Schijndel 2021; see the RSA standardization notes). Last layer only.
* **Pairs**: all pairs when <= 250k (<= 100k for Podcast); otherwise a deterministic sample per subject.
* **Items**: for datasets where models differ in coverage, the items covered by all 118 models (`eligible_ids.json`).
* **Units** are stored with their labels (subject, atlas/ROI/channel/electrode, window) so any subset (e.g. language ROIs
  only) can be aggregated afterwards. Headline aggregation (`rdm_regression_neural_summary.py`): subject-averaged GROUP
  units for fMRI (Pereira, Tuckute), Broderick and MEG-MASC (channel level), the N400 sets and the behavioral datasets
  (grand-average RDM); Podcast (ROI-grouped, GROUP units); per-subject units only for Fedorenko, whose electrodes have no
  location information in the data we have. The per-subject aggregation of every dataset is saved in
  `reports/rsa_summary/rdm_regression/neural_rdm_summary_per_subject.csv` (per-subject EEG/MEG units are almost all
  negative: single channels are too noisy; the Broderick surprisal signal appears only after subject averaging).

## Per dataset

| Dataset | Script | Units | Items | Response | Confounds |
|---|---|---|---|---|---|
| Pereira | `rdm_regression_pereira_units.py` | (subject, passage set, ROI): 54 ROIs in 5 atlases (language 12, MD 20, DMN 12, visual 6, auditory 4) + GROUP unit per (set, ROI) | sentences; 243-set and 384-set kept separate (sessions) | ROI-mean BOLD | sentlen, mean word length, mean frequency |
| Tuckute | `rdm_regression_tuckute_units.py` | (subject, ROI): lang_*, md_*, dmn_* ROIs (47-53) + GROUP; Glasser parcels not included | ~988-1000 B sentences | ROI response (supplementary all-ROI file) | mean word length, mean frequency |
| Broderick | `rdm_regression_epoch_units.py broderick` | (subject, channel) ~114 per subject + (subject, region) 4 LME regions + GROUP (channels with >= 10 subjects) | 11,385 eligible words, 250k sampled pairs | 300-500 ms mean | word length, frequency |
| MEG-MASC | `rdm_regression_epoch_units.py megmasc` | (subject, sensor) 208 + (subject, region) 6 LME regions + GROUP | 7,678 eligible words, 250k sampled pairs | 250-450 ms mean (repeated presentations averaged) | word length, frequency |
| Fedorenko | `rdm_regression_fedorenko_units.py` | (subject, electrode): 97 electrodes, 5 subjects; no GROUP (electrodes are subject specific) | 331 eligible words (all pairs) | word response | word length, frequency |
| Podcast | `rdm_regression_podcast_roi_units.py` (headline), `rdm_regression_podcast_units.py` (electrodes) | ROI units: (subject, ROI) and GROUP per ROI for the left-hemisphere language-responsive electrodes (285 in the benchmark) assigned to the Fedorenko parcels IFGorb, IFG, MFG, AntTemp, PostTemp (+ `netw` = all of them) with `outputs/podcast/electrode_roi_map_language_responsive.csv`; GROUP needs >= 3 subjects (MFG has 2 and gets none); all 32 lags. Electrode-level units (285 x 32 lags) are kept for reference | 5,134 words, 100k sampled pairs | high-gamma at each lag; electrode z-scores averaged within the ROI | word length, frequency |
| Nieuwland, Federmeier 2007, Hubbard 2019 | `rdm_regression_n400_units.py` | one GROUP unit: subject-averaged item-mean N400 (the ridge target) | 160 / 564 / ... items (all pairs) | item-mean N400 | critical-word length, Zipf frequency; Federmeier 2007 / Hubbard 2019 also word position, OLD20, concreteness (paper formula `calculate_aics_szewczyk.R`; baseline amplitude already removed from the target); Nieuwland: none beyond length/frequency (its formula has only lab) |
| LPP, Natural Stories fMRI | - | not covered (no stimulus-level response; time-chunk design) | | | |

Run (SLURM, CPU, scratch outputs; `outputs/` is never written): `scripts/slurm/jz/analysis/rdm_regression_<name>_units.sh`.
Resumable per model; shard over subjects with `--shard I/N` (SLURM arrays pass it automatically).

## Choices to keep in mind

* Equal weights over units and subjects in the default aggregation; the LME pools observations, so it implicitly weights by
  observation count. The data are nearly balanced, but this has not been compared.
* N400 confounds: critical-word length and Zipf frequency, plus (Federmeier 2007, Hubbard 2019) position, OLD20 and
  concreteness. The LME's random slopes have no counterpart on a subject-averaged item target. Old len+freq-only results are in
  `$SCRATCH/rdm_n400_units_old_len_freq_only`.
* Podcast is scored lag by lag (32 lags). The per-ROI LME delta LL (`outputs_jeanzay/podcast/surprisal_roi_lastlayer`) allows an ROI-by-ROI comparison; the ROI map is the one used by the earlier ROI figures.
* Absolute delta LL values are small (1e-6 to 1e-3 per pair); read model rankings and units with a positive delta LL, not
  the absolute size. Gaussian likelihood on |difference| distances is misspecified and pairs sharing an item are
  correlated.
* Noise ceilings are not part of this first pass. Region/channel datasets can use a leave-one-subject-out ceiling per unit
  (units are shared across subjects); electrode datasets (Fedorenko, Podcast) have no equivalent.
