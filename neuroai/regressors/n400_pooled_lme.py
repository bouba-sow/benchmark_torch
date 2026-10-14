"""Linear mixed-effects helpers shared by the N400 EEG datasets (Nieuwland,
Szewczyk), fit independently per LM layer (per-layer logit-lens surprisal,
not embeddings).

Unlike the per-subject multi-output ridge encoders (``benchmarks/
n400_nieuwland.py``, ``benchmarks/n400_szewczyk.py``), this pools every subject
and electrode into ONE mixed-effects model per layer. These N400 datasets'
electrodes are standard 10-20 scalp locations — the same physical site across
subjects — so electrode is a CROSSED random effect (``1 | electrode``), unlike
Podcast's ECoG electrodes which must be nested within subject. See
``neuroai/regressors/lme.py::n400_lme_formulas`` for the formula and
``neuroai/features/logit_lens.py::logit_lens_surprisal_trials`` for how the
per-layer surprisal predictor is computed.

    baseline: n400 ~ wlen_z + freq_z +
                      (1 + wlen_z + freq_z | subject) +
                      (1 | electrode) [+ (1 | lab)]
    full:     n400 ~ wlen_z + freq_z + surprisal_z + prev_surprisal_z +
                      (1 + wlen_z + freq_z + surprisal_z + prev_surprisal_z | subject) +
                      (1 | electrode) [+ (1 | lab)]

``prev_surprisal_z`` (spillover control) is standard now: absent from the
baseline, added only in the full model alongside ``surprisal_z`` — matching
Natural Stories' ``clark_spr_lme_formulas`` rather than treating it as a
nuisance shared by both models.

Fit once per layer (the outcome ``n400`` is the same across layers; only
``surprisal_z`` changes). Train/held-out split is by sentence frame
(``frame_id``, renamed to ``chunk`` to reuse ``fmri_lme.split_train_test_chunks``
unmodified) — a frame's conditions never cross the train/test boundary,
mirroring the GroupKFold-by-frame convention already used by these datasets'
ridge encoders.

Dataset-agnostic: every column here (``subject, electrode, frame_id, wlen,
freq, prev_surprisal, surprisal, n400``, optionally ``lab``) is generic —
each dataset's benchmark orchestrator (``benchmarks/n400_nieuwland_lme.py``,
``benchmarks/n400_szewczyk_lme.py``) is responsible only for building a
DataFrame with these columns; the fitting logic itself never changes.

Reuses the outcome-agnostic low-level helpers from ``regressors/lme.py``
(``_fit_stats``, ``_extract_fixed_effect``, ``_partition_loglik_detailed``,
``_corrs``, ``_predict_or_nan``, ``zscore_series``) and the chunk-split logic
from ``regressors/fmri_lme.py`` rather than duplicating them — same pattern as
``regressors/podcast_lme.py``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl

from .fmri_lme import split_train_test_chunks
from .lme import (
    LmeFitResult,
    _corrs,
    _extract_fixed_effect,
    _fit_stats,
    _partition_loglik_detailed,
    _predict_or_nan,
    n400_lme_formulas,
    zscore_series,
)

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {
    "subject", "electrode", "frame_id", "wlen", "freq", "prev_surprisal", "surprisal", "n400",
}


def prepare_n400_lme_frame(df: pd.DataFrame, *, include_lab: bool = False) -> pl.DataFrame:
    """Factor columns + z-scored predictors for a pooled N400 LME.

    ``electrode`` values ARE globally comparable across subjects (standard
    10-20 scalp names) — unlike Podcast, no ``subject:electrode`` nesting is
    used; see ``n400_lme_formulas``.
    """
    required = set(_REQUIRED_COLS)
    if include_lab:
        required.add("lab")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for N400 pooled LME: {sorted(missing)}")

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["electrode"] = out["electrode"].astype(str)
    out["chunk"] = out["frame_id"].astype(str)  # reuse split_train_test_chunks as-is
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["prev_surprisal_z"] = zscore_series(out["prev_surprisal"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["n400"] = out["n400"].astype(float)

    cols = ["subject", "electrode", "chunk", "wlen_z", "freq_z", "prev_surprisal_z", "surprisal_z", "n400"]
    if include_lab:
        out["lab"] = out["lab"].astype(str)
        cols.append("lab")
    return pl.from_pandas(out[cols].copy())


def fit_n400_lme_one_layer(
    df: pd.DataFrame,
    *,
    include_lab: bool = False,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    allow_slope_fallback: bool = True,
    save_per_word: bool = False,
) -> LmeFitResult:
    """Fit the pooled N400 LME for one LM layer.

    ``df`` must have one row per (subject, electrode, item) with columns
    ``subject, electrode, frame_id, wlen, freq, prev_surprisal, surprisal,
    n400`` (plus ``lab`` if ``include_lab=True``) — ``surprisal`` is this
    specific layer's logit-lens surprisal at the target/critical word.

    If the full model (with the 4-term correlated random slope on subject)
    fails to converge, and ``allow_slope_fallback=True``, retries without the
    slope-intercept correlation and finally without the random slope
    entirely — same Barr et al. (2013)-style ladder as
    ``regressors.podcast_lme.fit_podcast_lme_one_lag``, applied identically to
    both models being compared so the LRT stays valid.
    """
    from pymer4.models import compare, lmer

    data = prepare_n400_lme_frame(df, include_lab=include_lab)
    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )

    baseline_formula, full_formula = n400_lme_formulas(include_lab=include_lab)
    outcome = "n400"
    beta_term = "surprisal_z"

    def _fit(formula: str):
        m = lmer(formula, data=train_data)
        m.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        return m

    def _converged(model) -> bool:
        _, _, _, ok = _fit_stats(model)
        return ok

    baseline_model = _fit(baseline_formula)
    base_ok = _converged(baseline_model)

    full_model = _fit(full_formula)
    converged = _converged(full_model)

    nuisances = "wlen_z + freq_z + prev_surprisal_z"
    slope_dropped = False
    if allow_slope_fallback and not (base_ok and converged):
        logger.warning(
            "N400 pooled LME: full/baseline model failed to converge — retrying "
            "without the subject slope-intercept correlation."
        )
        full_formula_nocorr = full_formula.replace(
            f"(1 + {nuisances} + surprisal_z | subject)",
            f"(1 + {nuisances} + surprisal_z || subject)",
        )
        full_model = _fit(full_formula_nocorr)
        converged = _converged(full_model)
        if not (base_ok and converged):
            logger.warning(
                "N400 pooled LME: still not converged — dropping the random "
                "slope on subject entirely (intercept-only)."
            )
            re_electrode = "(1 | electrode)"
            re_lab = " + (1 | lab)" if include_lab else ""
            baseline_formula = f"n400 ~ {nuisances} + (1 | subject) + {re_electrode}{re_lab}"
            full_formula = f"n400 ~ {nuisances} + surprisal_z + (1 | subject) + {re_electrode}{re_lab}"
            baseline_model = _fit(baseline_formula)
            base_ok = _converged(baseline_model)
            full_model = _fit(full_formula)
            converged = _converged(full_model)
            slope_dropped = True

    base_loglik, _, _, _ = _fit_stats(baseline_model)
    loglik, aic, bic, _ = _fit_stats(full_model)
    beta, se, t_stat, p_value = _extract_fixed_effect(full_model, beta_term)

    delta_loglik_train: float | None = None
    lrt_p: float | None = None
    try:
        cmp = compare(baseline_model, full_model, as_dataframe=True, test="LRT")
        if cmp.height >= 2 and "Pr(>Chisq)" in cmp.columns:
            lrt_p = float(cmp["Pr(>Chisq)"][1])
        delta_loglik_train = loglik - base_loglik
    except Exception as exc:
        logger.warning("LRT comparison failed: %s", exc)
        delta_loglik_train = loglik - base_loglik

    delta_loglik_test: float | None = None
    baseline_loglik_test: float | None = None
    loglik_test: float | None = None
    base_per_obs: np.ndarray | None = None
    full_per_obs: np.ndarray | None = None
    if test_data.height >= 10:
        baseline_loglik_test, base_per_obs = _partition_loglik_detailed(
            baseline_model, test_data, outcome=outcome
        )
        loglik_test, full_per_obs = _partition_loglik_detailed(
            full_model, test_data, outcome=outcome
        )
        if np.isfinite(baseline_loglik_test) and np.isfinite(loglik_test):
            delta_loglik_test = loglik_test - baseline_loglik_test

    y_true = np.asarray(test_data[outcome], dtype=np.float64)
    try:
        yhat_base = _predict_or_nan(baseline_model, test_data)
        yhat_full = _predict_or_nan(full_model, test_data)
        base_p, base_s = _corrs(y_true, yhat_base)
        full_p, full_s = _corrs(y_true, yhat_full)
    except Exception as exc:
        logger.warning("Prediction failed for OOS correlation: %s", exc)
        yhat_base = np.full(test_data.height, np.nan, dtype=np.float64)
        yhat_full = np.full(test_data.height, np.nan, dtype=np.float64)
        base_p, base_s, full_p, full_s = np.nan, np.nan, np.nan, np.nan

    per_word: pd.DataFrame | None = None
    if save_per_word and base_per_obs is not None and full_per_obs is not None:
        id_cols = [c for c in ("subject", "electrode", "chunk", "lab") if c in test_data.columns]
        per_word = test_data.select(id_cols).to_pandas()
        per_word[outcome] = y_true
        per_word["yhat_baseline"] = yhat_base
        per_word["yhat_full"] = yhat_full
        per_word["baseline_loglik"] = base_per_obs
        per_word["full_loglik"] = full_per_obs
        per_word["delta_loglik"] = full_per_obs - base_per_obs

    primary_delta = (
        delta_loglik_test if delta_loglik_test is not None else delta_loglik_train
    )

    return LmeFitResult(
        predictor="surprisal",
        beta=beta,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        loglik=loglik,
        aic=aic,
        bic=bic,
        baseline_loglik=base_loglik,
        delta_loglik=primary_delta,
        lrt_p_value=lrt_p,
        test_pearsonr_full_model=full_p,
        test_spearmanr_full_model=full_s,
        test_pearsonr_baseline_lme=base_p,
        test_spearmanr_baseline_lme=base_s,
        test_delta_pearsonr=(
            (full_p - base_p) if np.isfinite(full_p) and np.isfinite(base_p) else np.nan
        ),
        test_delta_spearmanr=(
            (full_s - base_s) if np.isfinite(full_s) and np.isfinite(base_s) else np.nan
        ),
        converged=converged and base_ok,
        n_obs=int(train_data.height),
        delta_loglik_train=delta_loglik_train,
        delta_loglik_test=delta_loglik_test,
        baseline_loglik_test=baseline_loglik_test,
        loglik_test=loglik_test,
        n_obs_test=int(test_data.height),
        slope_dropped=slope_dropped,
        per_word=per_word,
    )


def fit_n400_lme_one_layer_paper(
    df: pd.DataFrame,
    *,
    dataset: str,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    allow_slope_fallback: bool = True,
    save_per_word: bool = False,
) -> LmeFitResult:
    """Fit one LM layer's N400 LME using each dataset's PAPER-EXACT formula
    (see this module's ``_n400_ceiling_formulas_{dataset}_paper`` and
    ``prepare_{dataset}_paper_frame``) instead of the generic correlated-
    random-slope structure ``fit_n400_lme_one_layer`` uses. Diagnosed via the
    split-half noise ceiling: the generic formula's correlated multi-
    predictor random slope produced a NEGATIVE (unusable) ceiling for
    Nieuwland, while each dataset's own literature-validated
    formula did not (see the noise-ceiling comparison in this session's
    conversation history / paper appendix for the numbers).

    ``dataset`` selects the formula/frame-prep pair:

    - ``"nieuwland"``: ``prepare_nieuwland_paper_frame`` /
      ``_n400_ceiling_formulas_nieuwland_paper`` — pure random intercepts
      (subject, item), ``lab`` as a FIXED effect, no frequency covariate.
    - ``"szewczyk"``: ``prepare_szewczyk_paper_frame`` /
      ``_n400_ceiling_formulas_szewczyk_paper`` — uncorrelated random slopes;
      falls back to intercept-only on non-convergence (Barr et al. 2013-style
      ladder, as elsewhere in this module).

    ``df`` must already have the columns each ``prepare_*_paper_frame``
    requires (see their docstrings) plus ``surprisal`` (this layer's
    logit-lens surprisal). Train/held-out split is by ``frame_id`` (renamed
    ``chunk`` inside each ``prepare_*_paper_frame``), same convention as
    ``fit_n400_lme_one_layer``.
    """
    from pymer4.models import compare, lmer

    if dataset == "nieuwland":
        data = prepare_nieuwland_paper_frame(df)
        baseline_formula, full_formula = _n400_ceiling_formulas_nieuwland_paper()
    elif dataset == "szewczyk":
        data = prepare_szewczyk_paper_frame(df)
        baseline_formula, full_formula = _n400_ceiling_formulas_szewczyk_paper()
    else:
        raise ValueError(f"Unknown dataset {dataset!r}; expected nieuwland/szewczyk")

    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )
    outcome = "n400"
    beta_term = "surprisal_z"

    def _fit(formula: str):
        m = lmer(formula, data=train_data)
        m.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        return m

    def _converged(model) -> bool:
        _, _, _, ok = _fit_stats(model)
        return ok

    baseline_model = _fit(baseline_formula)
    base_ok = _converged(baseline_model)

    full_model = _fit(full_formula)
    converged = _converged(full_model)

    slope_dropped = False
    if dataset == "szewczyk" and allow_slope_fallback and not (base_ok and converged):
        logger.warning(
            "N400 paper-formula LME (%s): full/baseline model failed to converge — "
            "dropping the bline/pos_start random slopes entirely (intercept-only).",
            dataset,
        )
        nuisances = "bline_z + freq_z + pos_start_z + old20_z + concr_z"
        res = "(1 | subject) + (1 | chunk)"
        baseline_formula = f"n400 ~ {nuisances} + {res}"
        full_formula = f"n400 ~ {nuisances} + surprisal_z + {res}"
        baseline_model = _fit(baseline_formula)
        base_ok = _converged(baseline_model)
        full_model = _fit(full_formula)
        converged = _converged(full_model)
        slope_dropped = True

    base_loglik, _, _, _ = _fit_stats(baseline_model)
    loglik, aic, bic, _ = _fit_stats(full_model)
    beta, se, t_stat, p_value = _extract_fixed_effect(full_model, beta_term)

    delta_loglik_train: float | None = None
    lrt_p: float | None = None
    try:
        cmp = compare(baseline_model, full_model, as_dataframe=True, test="LRT")
        if cmp.height >= 2 and "Pr(>Chisq)" in cmp.columns:
            lrt_p = float(cmp["Pr(>Chisq)"][1])
        delta_loglik_train = loglik - base_loglik
    except Exception as exc:
        logger.warning("LRT comparison failed: %s", exc)
        delta_loglik_train = loglik - base_loglik

    delta_loglik_test: float | None = None
    baseline_loglik_test: float | None = None
    loglik_test: float | None = None
    base_per_obs: np.ndarray | None = None
    full_per_obs: np.ndarray | None = None
    if test_data.height >= 10:
        baseline_loglik_test, base_per_obs = _partition_loglik_detailed(
            baseline_model, test_data, outcome=outcome
        )
        loglik_test, full_per_obs = _partition_loglik_detailed(
            full_model, test_data, outcome=outcome
        )
        if np.isfinite(baseline_loglik_test) and np.isfinite(loglik_test):
            delta_loglik_test = loglik_test - baseline_loglik_test

    y_true = np.asarray(test_data[outcome], dtype=np.float64)
    try:
        yhat_base = _predict_or_nan(baseline_model, test_data)
        yhat_full = _predict_or_nan(full_model, test_data)
        base_p, base_s = _corrs(y_true, yhat_base)
        full_p, full_s = _corrs(y_true, yhat_full)
    except Exception as exc:
        logger.warning("Prediction failed for OOS correlation: %s", exc)
        yhat_base = np.full(test_data.height, np.nan, dtype=np.float64)
        yhat_full = np.full(test_data.height, np.nan, dtype=np.float64)
        base_p, base_s, full_p, full_s = np.nan, np.nan, np.nan, np.nan

    per_word: pd.DataFrame | None = None
    if save_per_word and base_per_obs is not None and full_per_obs is not None:
        id_cols = [c for c in ("subject", "electrode", "lab", "item_id", "chunk") if c in test_data.columns]
        per_word = test_data.select(id_cols).to_pandas()
        per_word[outcome] = y_true
        per_word["yhat_baseline"] = yhat_base
        per_word["yhat_full"] = yhat_full
        per_word["baseline_loglik"] = base_per_obs
        per_word["full_loglik"] = full_per_obs
        per_word["delta_loglik"] = full_per_obs - base_per_obs

    primary_delta = delta_loglik_test if delta_loglik_test is not None else delta_loglik_train

    return LmeFitResult(
        predictor="surprisal",
        beta=beta,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        loglik=loglik,
        aic=aic,
        bic=bic,
        baseline_loglik=base_loglik,
        delta_loglik=primary_delta,
        lrt_p_value=lrt_p,
        test_pearsonr_full_model=full_p,
        test_spearmanr_full_model=full_s,
        test_pearsonr_baseline_lme=base_p,
        test_spearmanr_baseline_lme=base_s,
        test_delta_pearsonr=(
            (full_p - base_p) if np.isfinite(full_p) and np.isfinite(base_p) else np.nan
        ),
        test_delta_spearmanr=(
            (full_s - base_s) if np.isfinite(full_s) and np.isfinite(base_s) else np.nan
        ),
        converged=converged and base_ok,
        n_obs=int(train_data.height),
        delta_loglik_train=delta_loglik_train,
        delta_loglik_test=delta_loglik_test,
        baseline_loglik_test=baseline_loglik_test,
        loglik_test=loglik_test,
        n_obs_test=int(test_data.height),
        slope_dropped=slope_dropped,
        per_word=per_word,
    )


# ---------------------------------------------------------------------------
# Split-half delta-log-likelihood noise ceiling (Timkey et al. 2026)
# ---------------------------------------------------------------------------
# Same split-half logic as neuroai.regressors.lme.compute_reading_time_
# ceiling_delta_loglik, adapted to the pooled N400 schema: "item" is
# (context_code, condition) = item_id (the granularity surprisal is actually
# assigned at), NOT frame_id (which only groups an item's 2-4 conditions for
# the train/test chunk split, so several items legitimately share one
# frame_id). The item-mean N400 is broadcast across every row (subject,
# electrode) matching that item_id, mirroring how a real LM's surprisal is a
# single per-item scalar broadcast across every subject/electrode row too.
#
# prev_surprisal_z is dropped entirely from the ceiling's full formula (not
# just set aside as in the behavioral ceiling): here it is the surprisal of
# the word immediately BEFORE the critical word within the same trial, not a
# separate response-bearing item — N400 is only measured at the critical
# word, so there is no N400 value at that preceding position to build a
# split-half mean from at all, unlike Natural Stories' spillover word. The
# ceiling therefore bounds the single-predictor (surprisal-only) contribution
# to ΔLL; the real model's ΔLL additionally includes prev_surprisal_z's own
# contribution, which this ceiling does not bound.


def _n400_ceiling_formulas(*, include_lab: bool = False) -> tuple[str, str]:
    """Baseline/full formulas for the ceiling fit -- same as
    ``n400_lme_formulas`` but with ``prev_surprisal_z`` dropped from the full
    model (see module-level note above for why)."""
    nuisances = "wlen_z + freq_z"
    re_electrode = "(1 | electrode)"
    re_lab = " + (1 | lab)" if include_lab else ""
    baseline = f"n400 ~ {nuisances} + (1 + {nuisances} | subject) + {re_electrode}{re_lab}"
    full = f"n400 ~ {nuisances} + surprisal_z + (1 + {nuisances} + surprisal_z | subject) + {re_electrode}{re_lab}"
    return baseline, full


def _n400_ceiling_formulas_nieuwland_paper() -> tuple[str, str]:
    """Baseline/full formulas matching
    ``data/n400_paper_formula/osfstorage/calculate_fits/calculate_aics_nieuwland.R``
    exactly: fixed effects surprisal + lab (a CATEGORICAL FIXED effect, not
    ``(1|lab)`` -- with only 9 labs, a random-effect variance for so few
    groups is poorly estimated, so the original analysis treats it as
    fixed), random intercepts for subject and item only. No electrode term:
    this dataset's ``n400`` is already pre-averaged across the 6
    centro-parietal electrodes per trial in the source data
    (data/n400_paper_formula/osfstorage/datasets/nieuwland_2018.tsv has no electrode
    column at all), unlike per-electrode long tables. No
    frequency/word-length/ON covariate either -- the original analysis uses
    none for this dataset. ``chunk`` here is the BASE item number (shared by
    both cloze conditions of the same item), matching their ``(1|item)``
    term and used as the train/test split boundary too, so both conditions
    of one item always land on the same side of the split."""
    res = "(1 | subject) + (1 | chunk)"
    baseline = f"n400 ~ lab + {res}"
    full = f"n400 ~ lab + surprisal_z + {res}"
    return baseline, full


def prepare_nieuwland_paper_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Factor columns + z-scored predictor for the paper-matching Nieuwland
    formula. ``df`` needs subject, lab, item_id (item x cloze-condition,
    where surprisal actually varies), frame_id (base item, shared across
    both conditions), surprisal, n400."""
    required = {"subject", "lab", "item_id", "frame_id", "surprisal", "n400"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Nieuwland-paper N400 frame: {sorted(missing)}")
    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["lab"] = out["lab"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    out["chunk"] = out["frame_id"].astype(str)  # reuse split_train_test_chunks as-is
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["n400"] = out["n400"].astype(float)
    cols = ["subject", "lab", "item_id", "chunk", "surprisal_z", "n400"]
    return pl.from_pandas(out[cols].copy())


def _n400_ceiling_formulas_szewczyk_paper() -> tuple[str, str]:
    """Baseline/full formulas matching
    ``data/n400_paper_formula/osfstorage/calculate_fits/calculate_aics_szewczyk.R``
    exactly: fixed effects surprisal + bline (pre-stimulus baseline
    amplitude) + logfreq + pos_start (word position in sentence) + old20
    (orthographic Levenshtein distance) + concr (concreteness), UNCORRELATED
    (``||``) random slopes for bline + pos_start on subject and for bline on
    item, no electrode term (this dataset's n400 is already pre-averaged
    over 4 centro-parietal ROI clusters into one scalar per trial -- see
    neuroai/datasets/szewczyk.py).

    Deviation from the source script: it fits ONE MODEL PER SUB-EXPERIMENT
    (5 separate ``lmer`` calls, looping over ``unique(current_dataset$dataset)``)
    rather than pooling. This benchmark pools across all 5 sub-experiments
    into a single model instead, same simplification this codebase already
    makes for Nieuwland relative to its own original
    per-study analyses -- subject IDs are already globally unique across
    sub-experiments (verified in szewczyk.py's module docstring), so pooling
    doesn't create any spurious subject-identity collisions. This baseline
    formula (surprisal_z dropped) is this benchmark's own addition on top of
    their validated structure, needed to get a ΔLL rather than a single AIC
    -- their script only ever fits the full model."""
    res = "(1 + bline_z + pos_start_z || subject) + (1 + bline_z || chunk)"
    nuisances = "bline_z + freq_z + pos_start_z + old20_z + concr_z"
    baseline = f"n400 ~ {nuisances} + {res}"
    full = f"n400 ~ {nuisances} + surprisal_z + {res}"
    return baseline, full


def prepare_szewczyk_paper_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Factor columns + z-scored predictors for the paper-matching Szewczyk
    formula. ``df`` needs subject, chunk (subexperiment-disambiguated item
    id -- Item numbers repeat across the 5 sub-experiments, see
    szewczyk.py), bline, freq (logfreq), pos_start, old20, concr, surprisal,
    n400 (bline/freq/pos_start/old20/concr straight from
    ``data/n400_paper_formula/osfstorage/datasets/szewczyk_2022.tsv``'s own precomputed
    columns, not recomputed)."""
    required = {"subject", "chunk", "bline", "freq", "pos_start", "old20", "concr", "surprisal", "n400"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Szewczyk-paper N400 frame: {sorted(missing)}")
    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["chunk"] = out["chunk"].astype(str)
    out["bline_z"] = zscore_series(out["bline"])
    out["freq_z"] = zscore_series(out["freq"])
    out["pos_start_z"] = zscore_series(out["pos_start"])
    out["old20_z"] = zscore_series(out["old20"])
    out["concr_z"] = zscore_series(out["concr"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["n400"] = out["n400"].astype(float)
    cols = ["subject", "chunk", "bline_z", "freq_z", "pos_start_z", "old20_z", "concr_z", "surprisal_z", "n400"]
    return pl.from_pandas(out[cols].copy())


def _n400_ceiling_formulas_michaelov_paper() -> tuple[str, str]:
    """Baseline/full formulas matching
    ``data/n400_paper_formula/osfstorage/calculate_fits/calculate_aics_michaelov.R``:
    fixed effects surprisal + ZipfFrequency + ON (orthographic neighbourhood), random
    intercepts for Subject, ContextCode (``chunk``), TargetWords (``target_word``) and Electrode.
    The baseline (surprisal_z dropped) is this benchmark's own addition, needed for a delta-LL."""
    res = "(1 | subject) + (1 | chunk) + (1 | target_word) + (1 | electrode)"
    return f"n400 ~ freq_z + on_z + {res}", f"n400 ~ freq_z + on_z + surprisal_z + {res}"


def prepare_michaelov_paper_frame(df: pd.DataFrame) -> pl.DataFrame:
    """``df`` needs subject, electrode, chunk (ContextCode), target_word, freq (ZipfFrequency),
    on (ON), surprisal, n400 -- the columns of ``michaelov_2024.tsv``."""
    required = {"subject", "electrode", "chunk", "target_word", "freq", "on", "surprisal", "n400"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Michaelov-paper N400 frame: {sorted(missing)}")
    out = df.copy()
    for c in ("subject", "electrode", "chunk", "target_word"):
        out[c] = out[c].astype(str)
    out["freq_z"] = zscore_series(out["freq"])
    out["on_z"] = zscore_series(out["on"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["n400"] = out["n400"].astype(float)
    cols = ["subject", "electrode", "chunk", "target_word", "freq_z", "on_z", "surprisal_z", "n400"]
    return pl.from_pandas(out[cols].copy())


def _n400_ceiling_formulas_ryskin_paper() -> tuple[str, str]:
    """Baseline/full formulas matching
    ``data/n400_paper_formula/osfstorage/calculate_fits/calculate_aics_ryskin.R``:
    ``n400 ~ surprisal + (1 + surprisal || Subject) + (1 + surprisal || Item) + (1 | Electrode)``.
    The paper only fits the full model. The baseline drops surprisal everywhere (fixed effect and
    the two by-surprisal random slopes, which cannot exist without it), so this delta-LL also
    contains the slopes' contribution."""
    full = "n400 ~ surprisal_z + (1 + surprisal_z || subject) + (1 + surprisal_z || chunk) + (1 | electrode)"
    base = "n400 ~ (1 | subject) + (1 | chunk) + (1 | electrode)"
    return base, full


def prepare_ryskin_paper_frame(df: pd.DataFrame) -> pl.DataFrame:
    """``df`` needs subject, electrode, chunk (base Item), surprisal, n400 (``ryskin_2021.tsv``)."""
    required = {"subject", "electrode", "chunk", "surprisal", "n400"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Ryskin-paper N400 frame: {sorted(missing)}")
    out = df.copy()
    for c in ("subject", "electrode", "chunk"):
        out[c] = out[c].astype(str)
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["n400"] = out["n400"].astype(float)
    return pl.from_pandas(out[["subject", "electrode", "chunk", "surprisal_z", "n400"]].copy())


def _n400_ceiling_direction_task(
    df: pd.DataFrame, source_ids: set, target_ids: set, item_col: str,
) -> pd.DataFrame | None:
    """Build one direction's target-half frame with the split-half item-mean
    N400 merged in as a column literally named ``"surprisal"`` (required so
    ``prepare_n400_lme_frame`` z-scores it into ``surprisal_z``, the term
    ``_n400_ceiling_formulas`` references)."""
    src = df[df["subject"].astype(str).isin(source_ids)]
    tgt = df[df["subject"].astype(str).isin(target_ids)].copy()
    item_mean = src.groupby(item_col)["n400"].mean().rename("surprisal").reset_index()
    tgt = tgt.drop(columns=["surprisal"], errors="ignore").merge(item_mean, on=item_col, how="left")
    tgt = tgt.dropna(subset=["surprisal"])
    return tgt if not tgt.empty else None


def _n400_ceiling_direction_worker(
    iteration: int,
    direction: int,
    tgt: pd.DataFrame,
    *,
    include_lab: bool,
    train_frac: float,
    random_state: int,
    verbose: bool,
    out_of_sample: bool = True,
    re_structure: str = "slopes",
) -> tuple[int, int, float | None, int, bool]:
    """Process-pool worker: fit one (iteration, direction)'s N400 ceiling
    model. Module-level (not a closure) so it can be pickled for spawn/fork.
    Returns (iteration, direction, delta_loglik or None on failure, n_obs, converged).

    ``out_of_sample=False`` fits and scores on ALL of ``tgt`` (matching
    Timkey et al.'s literal in-sample description) instead of splitting into
    train/held-out chunks. ``re_structure`` picks the formula: ``"nieuwland_paper"`` or this
    codebase's default correlated-random-slope structure (``"slopes"``)."""
    from pymer4.models import lmer

    try:
        if re_structure == "nieuwland_paper":
            data = prepare_nieuwland_paper_frame(tgt)
        elif re_structure == "michaelov_paper":
            data = prepare_michaelov_paper_frame(tgt)
        elif re_structure == "ryskin_paper":
            data = prepare_ryskin_paper_frame(tgt)
        elif re_structure == "szewczyk_paper":
            data = prepare_szewczyk_paper_frame(tgt)
        else:
            # prepare_n400_lme_frame requires a prev_surprisal column to
            # exist (it z-scores it into prev_surprisal_z), even though the
            # ceiling formula never references that term -- a constant
            # placeholder z-scores to all-zero (see zscore_series) and is
            # simply unused.
            frame_in = tgt.assign(prev_surprisal=0.0)
            data = prepare_n400_lme_frame(frame_in, include_lab=include_lab)

        if out_of_sample:
            train_data, test_data = split_train_test_chunks(
                data, train_frac=train_frac, random_state=random_state,
            )
        else:
            train_data = test_data = data
        if test_data.height < 10:
            return iteration, direction, None, 0, False

        if re_structure == "nieuwland_paper":
            baseline_formula, full_formula = _n400_ceiling_formulas_nieuwland_paper()
        elif re_structure == "michaelov_paper":
            baseline_formula, full_formula = _n400_ceiling_formulas_michaelov_paper()
        elif re_structure == "ryskin_paper":
            baseline_formula, full_formula = _n400_ceiling_formulas_ryskin_paper()
        elif re_structure == "szewczyk_paper":
            baseline_formula, full_formula = _n400_ceiling_formulas_szewczyk_paper()
        else:
            baseline_formula, full_formula = _n400_ceiling_formulas(include_lab=include_lab)

        baseline_model = lmer(baseline_formula, data=train_data)
        baseline_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, base_ok = _fit_stats(baseline_model)

        full_model = lmer(full_formula, data=train_data)
        full_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, full_ok = _fit_stats(full_model)

        base_ll, _ = _partition_loglik_detailed(baseline_model, test_data, outcome="n400")
        full_ll, _ = _partition_loglik_detailed(full_model, test_data, outcome="n400")
    except Exception as exc:
        logger.warning("N400 ceiling fit failed on split %d direction %d: %s", iteration, direction, exc)
        return iteration, direction, None, 0, False

    converged = bool(base_ok and full_ok)
    delta = full_ll - base_ll if np.isfinite(base_ll) and np.isfinite(full_ll) else None
    ok = delta is not None and np.isfinite(delta) and converged
    return iteration, direction, (delta if ok else None), int(test_data.height), converged


@dataclass
class N400DeltaLoglikCeilingResult:
    """Split-half theoretical upper bound for a pooled N400 LME's delta
    log-likelihood (Timkey et al. 2026 method, adapted -- see module note)."""

    ceiling_delta_loglik_per_obs: float
    ceiling_delta_loglik_per_obs_sd: float
    n_iterations_used: int
    n_iterations_requested: int
    per_iteration_delta_loglik: list[float] = field(default_factory=list)
    per_iteration_n_obs: list[int] = field(default_factory=list)
    per_direction_records: list[dict] = field(default_factory=list)


def compute_n400_ceiling_delta_loglik(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    include_lab: bool = False,
    n_splits: int = 15,
    train_frac: float = 0.75,
    seed: int = 0,
    verbose: bool = False,
    n_workers: int = 1,
    out_of_sample: bool = True,
    re_structure: str = "slopes",
    mode: str = "split_half",
    n_repeats: int = 1,
) -> N400DeltaLoglikCeilingResult:
    """Split-half noise ceiling for a pooled N400 LME's delta log-likelihood.

    ``df`` must have one row per (subject, electrode, item) with columns
    ``subject, electrode, wlen, freq, n400`` plus ``item_col`` (default
    ``"item_id"``) and ``"chunk"``-compatible ``"frame_id"`` if that differs
    from ``item_col`` (Nieuwland: several conditions share
    one frame_id, so pass ``item_col="item_id"`` and keep ``frame_id`` in
    ``df`` for the train/test chunk split). See module docstring above for
    the split-half method and why ``prev_surprisal_z`` is excluded.
    """
    required = {"subject", "n400", "frame_id", item_col}
    if re_structure == "nieuwland_paper":
        required |= {"lab"}  # no electrode column: n400 is already pre-averaged across electrodes
    elif re_structure == "michaelov_paper":
        required |= {"electrode", "target_word", "freq", "on"}
    elif re_structure == "ryskin_paper":
        required |= {"electrode"}
    elif re_structure == "szewczyk_paper":
        required |= {"chunk", "bline", "freq", "pos_start", "old20", "concr"}
    else:
        required |= {"electrode", "wlen", "freq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for N400 delta-LL ceiling: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(df["subject"].astype(str).unique()))

    tasks: list[tuple[int, int, pd.DataFrame, int]] = []   # (iteration, direction, frame, split seed)
    if mode == "loso":
        # ONE pooled fit per repeat; each row's oracle = item mean over all OTHER subjects.
        n_splits = n_repeats
        keys = [item_col]
        g = df.groupby(keys)["n400"]
        tot, cnt = g.transform("sum"), g.transform("count")
        gs = df.groupby(keys + [df["subject"].astype(str)])["n400"]
        own_sum, own_cnt = gs.transform("sum"), gs.transform("count")
        oracle = (tot - own_sum) / (cnt - own_cnt).where(cnt - own_cnt > 0)
        tgt = df.assign(surprisal=oracle).dropna(subset=["surprisal"])
        for rep in range(n_repeats):
            tasks.append((rep, 0, tgt, seed + rep))
    else:
        for it in range(n_splits):
            perm = rng.permutation(subjects)
            half = len(perm) // 2
            halves = (set(perm[:half].tolist()), set(perm[half:].tolist()))
            for direction, (source_ids, target_ids) in enumerate((halves, halves[::-1])):
                tgt = _n400_ceiling_direction_task(df, source_ids, target_ids, item_col)
                if tgt is not None:
                    tasks.append((it, direction, tgt, seed))
    iter_delta: dict[int, float] = {}
    iter_nobs: dict[int, int] = {}
    iter_failed: set[int] = set()
    per_direction_records: list[dict] = []

    def _record(iteration: int, direction: int, delta: float | None, n_obs: int, converged: bool) -> None:
        per_direction_records.append({
            "iteration": iteration, "direction": direction,
            "delta_loglik": delta, "n_obs": n_obs, "converged": converged,
        })
        if iteration in iter_failed:
            return
        if delta is None:
            iter_failed.add(iteration)
            iter_delta.pop(iteration, None)
            iter_nobs.pop(iteration, None)
            return
        iter_delta[iteration] = iter_delta.get(iteration, 0.0) + delta
        iter_nobs[iteration] = iter_nobs.get(iteration, 0) + n_obs

    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _n400_ceiling_direction_worker, it, direction, tgt,
                    include_lab=include_lab, train_frac=train_frac,
                    random_state=rs, verbose=verbose,
                    out_of_sample=out_of_sample, re_structure=re_structure,
                ): (it, direction)
                for it, direction, tgt, rs in tasks
            }
            for fut in as_completed(futures):
                iteration, direction, delta, n_obs, converged = fut.result()
                _record(iteration, direction, delta, n_obs, converged)
    else:
        for it, direction, tgt, rs in tasks:
            iteration, direction, delta, n_obs, converged = _n400_ceiling_direction_worker(
                it, direction, tgt, include_lab=include_lab, train_frac=train_frac,
                random_state=rs, verbose=verbose,
                out_of_sample=out_of_sample, re_structure=re_structure,
            )
            _record(iteration, direction, delta, n_obs, converged)

    per_direction_records.sort(key=lambda r: (r["iteration"], r["direction"]))
    per_iter_delta = [iter_delta[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]
    per_iter_nobs = [iter_nobs[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]

    if not per_iter_delta:
        return N400DeltaLoglikCeilingResult(
            ceiling_delta_loglik_per_obs=float("nan"),
            ceiling_delta_loglik_per_obs_sd=float("nan"),
            n_iterations_used=0,
            n_iterations_requested=n_splits,
            per_direction_records=per_direction_records,
        )

    per_obs = [d / n for d, n in zip(per_iter_delta, per_iter_nobs)]
    return N400DeltaLoglikCeilingResult(
        ceiling_delta_loglik_per_obs=float(np.mean(per_obs)),
        ceiling_delta_loglik_per_obs_sd=float(np.std(per_obs, ddof=1)) if len(per_obs) > 1 else float("nan"),
        n_iterations_used=len(per_obs),
        n_iterations_requested=n_splits,
        per_iteration_delta_loglik=per_iter_delta,
        per_iteration_n_obs=per_iter_nobs,
        per_direction_records=per_direction_records,
    )


# ---------------------------------------------------------------------------
# Extrapolated ceiling (Schrimpf et al. 2021, SI sec. 7)
# ---------------------------------------------------------------------------

@dataclass
class N400ExtrapolatedCeilingResult:
    points: list[dict]                # one per fit: size (source subjects), draw, delta, n_obs, per_obs
    v0: float                         # median asymptote over bootstraps
    v0_ci: tuple[float, float]        # 2.5 / 97.5 percentile
    tau0_median: float
    n_bootstraps: int
    finite_n_per_obs: dict = field(default_factory=dict)   # mean per-obs value at each source size


def _exp_saturation(x, v0, tau0):
    return v0 * (1.0 - np.exp(-x / tau0))


def fit_exponential_ceiling(x: np.ndarray, v: np.ndarray, n_bootstraps: int = 100, seed: int = 0):
    """Schrimpf et al. 2021 SI-7: fit ``v = v0 * (1 - exp(-x / tau0))`` to (x, v) points, with
    ``n_bootstraps`` bootstraps that each resample the (x, v) points with replacement; the
    ceiling is the median ``v0``. Returns (v0_median, (lo, hi), tau0_median, n_ok)."""
    from scipy.optimize import curve_fit

    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    v0s, taus = [], []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(x), len(x))
        try:
            p, _ = curve_fit(
                _exp_saturation, x[idx], v[idx], p0=[max(float(v.max()), 1e-4), float(np.median(x))],
                bounds=([-1.0, 1e-3], [10.0, 1e5]), maxfev=20000,
            )
        except Exception:  # noqa: BLE001 -- degenerate resample; skip like a failed bootstrap
            continue
        v0s.append(p[0])
        taus.append(p[1])
    if not v0s:
        return float("nan"), (float("nan"), float("nan")), float("nan"), 0
    lo, hi = np.percentile(v0s, [2.5, 97.5])
    return float(np.median(v0s)), (float(lo), float(hi)), float(np.median(taus)), len(v0s)


def compute_n400_extrapolated_ceiling(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    include_lab: bool = False,
    re_structure: str = "szewczyk_paper",
    target_size: int = 8,
    source_sizes: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24),
    n_draws: int = 20,
    train_frac: float = 0.75,
    split_seed: int = 0,
    seed: int = 0,
    n_workers: int = 1,
    n_bootstraps: int = 100,
    verbose: bool = False,
    out_of_sample: bool = True,
) -> N400ExtrapolatedCeilingResult:
    """Ceiling extrapolated to infinitely many subjects (Schrimpf et al. 2021, SI-7), on the
    real mixed model.

    For each source size ``s`` and draw: pick ``target_size`` target subjects and ``s`` DISJOINT
    source subjects; the oracle = mean N400 of the source subjects per item (constant per item
    within a fit, so no leave-one-out artefact); fit the paper formula on the target subjects'
    training chunks and score delta-logLik on their held-out chunks (``split_seed`` fixed, so the
    held-out chunks are the same ones the model sweeps use). Then fit ``v0 (1 - exp(-x/tau0))``
    over all (s, delta-LL-per-held-out-obs) points with bootstrapped points; ceiling = median
    ``v0``. ``target_size`` subjects instead of Schrimpf's single target so the formula's
    by-subject random effects stay estimable."""
    subjects = np.asarray(sorted(df["subject"].astype(str).unique()))
    max_src = len(subjects) - target_size
    sizes = [s for s in source_sizes if 1 <= s <= max_src]
    rng = np.random.default_rng(seed)

    tasks = []   # (size, draw, frame)
    for s in sizes:
        for d in range(n_draws):
            perm = rng.permutation(subjects)
            tgt_ids = set(perm[:target_size].tolist())
            src_ids = set(perm[target_size:target_size + s].tolist())
            tgt = _n400_ceiling_direction_task(df, src_ids, tgt_ids, item_col)
            if tgt is not None:
                tasks.append((s, d, tgt))

    kw = dict(include_lab=include_lab, train_frac=train_frac, random_state=split_seed,
              verbose=verbose, out_of_sample=out_of_sample, re_structure=re_structure)
    points: list[dict] = []

    def _add(size, draw, delta, n_obs, conv):
        if delta is None or n_obs <= 0:
            return
        points.append({"size": int(size), "draw": int(draw), "delta": float(delta),
                       "n_obs": int(n_obs), "per_obs": float(delta) / int(n_obs), "converged": bool(conv)})

    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_n400_ceiling_direction_worker, s, d, tgt, **kw): (s, d) for s, d, tgt in tasks}
            for fut in as_completed(futs):
                s, d, delta, n_obs, conv = fut.result()
                _add(s, d, delta, n_obs, conv)
    else:
        for s, d, tgt in tasks:
            _, _, delta, n_obs, conv = _n400_ceiling_direction_worker(s, d, tgt, **kw)
            _add(s, d, delta, n_obs, conv)

    points.sort(key=lambda p: (p["size"], p["draw"]))
    if len(points) < 4:
        return N400ExtrapolatedCeilingResult(points, float("nan"), (float("nan"),) * 2, float("nan"), 0)
    x = np.array([p["size"] for p in points], float)
    v = np.array([p["per_obs"] for p in points], float)
    v0, ci, tau, n_ok = fit_exponential_ceiling(x, v, n_bootstraps=n_bootstraps, seed=seed)
    finite = {int(s): float(np.mean([p["per_obs"] for p in points if p["size"] == s])) for s in sizes
              if any(p["size"] == s for p in points)}
    return N400ExtrapolatedCeilingResult(points, v0, ci, tau, n_ok, finite)
