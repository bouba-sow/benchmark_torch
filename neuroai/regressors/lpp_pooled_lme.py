"""Linear mixed-effects helpers for Le Petit Prince fMRI, pooled across
Fedorenko-parcel regions and sections, fit independently per LM layer
(per-layer logit-lens surprisal, average-subject BOLD).

Design rationale
-----------------
Structurally the closest existing analogue is Natural Stories fMRI
(``regressors/fmri_lme.py``): continuous naturalistic listening, HRF-convolved
word-level predictors resampled onto the TR grid. Two simplifications apply
here that Natural Stories fMRI doesn't get:

1. **Average-subject BOLD, not per-subject** — ``neuroai/lpp_encoding.py``'s
   ``fmri_unit="average_subject"`` path (``_load_averaged_masked_bold``) is
   already run in production. This removes ``subject`` as a random-effect
   grouping factor entirely; there is no ``(1 | subject)`` term anywhere in
   this module.
2. **Reliable-voxel selection already exists** (``_load_reliable_voxel_scores``
   / ``_top_reliable_voxel_indices`` in ``lpp_encoding.py``) — reused as-is
   by the benchmark orchestration, not rebuilt here.

In their place, the genuinely new grouping factor is ``region`` — 10 Fedorenko
language-network parcels (5 bilateral parcels × L/R hemisphere:
IFGorb/IFG/MFG/AntTemp/PostTemp, each split by hemisphere; see
``neuroai/roi/lpp.py::load_fedorenko_parcelization``). Region gets a random
SLOPE on ``surprisal_z`` (not just an intercept) — the one genuinely new
statistical idea here: different language-network regions may not respond to
surprisal the same way, and pooling shrinks each region's estimated
sensitivity toward the group mean rather than estimating 10 independent,
unregularized slopes (the alternative would be a fixed ``surprisal_z:region``
interaction — worth trying as an explicit Plan B if the random slope proves
empirically unstable, but not implemented here).

    baseline: bold ~ wlen_z + freq_z + sentpos_z +
                     (1 | region) +
                     (1 | section)
    full:     bold ~ wlen_z + freq_z + sentpos_z + surprisal_z +
                     (1 + surprisal_z | region) +
                     (1 | section)

Notes on scope, mirroring every other pooled LME in this codebase:

- ``section`` (9 LPP chapters) is a crossed random effect, intercept-only —
  9 levels is thin even for an intercept, a slope there would be pushing it
  (same treatment as Tuckute's ``roi``).
- ``region``'s random slope is scoped to ``surprisal_z`` only, not
  ``wlen_z``/``freq_z``/``sentpos_z`` too — a full-maximal slope on all four
  covariates would need a 10×10 covariance matrix estimated from only 10
  groups, clearly too rich. Richness is scoped to the one term of interest.
- No ``(1 | subject)`` — average-subject BOLD; subject-level variance is
  already gone before the LME ever sees the data.
- No ``prev_surprisal_z`` — same rationale as Tuckute/Natural Stories fMRI:
  this is HRF-convolved, region-averaged BOLD, not a word-locked ERP; there's
  no discrete "previous word" spillover mechanism to correct for at this
  temporal/spatial resolution (if anything more washed out here than Natural
  Stories fMRI, which is at least per-fROI rather than multi-voxel-averaged
  then parcel-averaged).

Train/held-out split reuses Natural Stories fMRI's chunked design
(``regressors/fmri_lme.py::split_train_test_chunks``) — BOLD is
autocorrelated within a section, so held-out partitions must respect
contiguous TR-chunk boundaries, never leave-region-out or leave-section-out.
Both ``region`` and ``section`` levels appear in both partitions (only
time-chunks are held out).

Reuses the outcome-agnostic low-level helpers from ``regressors/lme.py``
(``_fit_stats``, ``_extract_fixed_effect``, ``_partition_loglik_detailed``,
``_corrs``, ``_predict_or_nan``, ``zscore_series``) unchanged.
"""

from __future__ import annotations

import logging

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
    zscore_series,
)

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {
    "region", "section", "chunk", "wlen", "freq", "sentpos", "surprisal", "bold",
}


def prepare_lpp_lme_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Factor columns + z-scored predictors for the pooled LPP fMRI LME.

    ``wlen``/``freq``/``sentpos``/``surprisal`` are expected to already be
    HRF-convolved (one value per TR, not per word) — z-scoring happens here,
    after convolution, same order as ``fmri_lme.prepare_fmri_lme_frame``.
    ``region`` is a fixed canonical set of 10 Fedorenko-parcel × hemisphere
    labels shared across every row (crossed, not nested).
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for LPP pooled LME: {sorted(missing)}")

    out = df.copy()
    out["region"] = out["region"].astype(str)
    out["section"] = out["section"].astype(str)
    out["chunk"] = out["chunk"].astype(str)
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["sentpos_z"] = zscore_series(out["sentpos"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["bold"] = out["bold"].astype(float)

    cols = [
        "region", "section", "chunk",
        "wlen_z", "freq_z", "sentpos_z", "surprisal_z", "bold",
    ]
    return pl.from_pandas(out[cols].copy())


def lpp_lme_formulas(*, per_region: bool = False) -> tuple[str, str]:
    """Baseline/full formulas for pooled or one-region LPP fMRI fits."""
    nuisances = "wlen_z + freq_z + sentpos_z"
    re_section = "(1 | section)"

    if per_region:
        baseline = f"bold ~ {nuisances} + {re_section}"
        full = f"bold ~ {nuisances} + surprisal_z + {re_section}"
        return baseline, full

    re_region_base = "(1 | region)"
    re_region_full = "(1 + surprisal_z | region)"

    baseline = f"bold ~ {nuisances} + {re_region_base} + {re_section}"
    full = f"bold ~ {nuisances} + surprisal_z + {re_region_full} + {re_section}"
    return baseline, full


_REQUIRED_COLS_PERSUBJECT = _REQUIRED_COLS | {"subject"}


def prepare_lpp_lme_frame_persubject(df: pd.DataFrame) -> pl.DataFrame:
    """Per-subject analogue of ``prepare_lpp_lme_frame`` — same HRF-convolved
    per-TR predictors, plus a ``subject`` column (English LPP subjects, one
    row per (subject, region, section, TR) instead of one averaged row per
    (region, section, TR)). See ``neuroai/regressors/lpp_persubject.py`` for
    how the per-subject BOLD this requires is built."""
    missing = _REQUIRED_COLS_PERSUBJECT - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for per-subject LPP pooled LME: {sorted(missing)}")

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["region"] = out["region"].astype(str)
    out["section"] = out["section"].astype(str)
    out["chunk"] = out["chunk"].astype(str)
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["sentpos_z"] = zscore_series(out["sentpos"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["bold"] = out["bold"].astype(float)

    cols = [
        "subject", "region", "section", "chunk",
        "wlen_z", "freq_z", "sentpos_z", "surprisal_z", "bold",
    ]
    return pl.from_pandas(out[cols].copy())


def lpp_lme_formulas_persubject(*, per_region: bool = False) -> tuple[str, str]:
    """Baseline/full formulas for the per-subject pooled LPP fMRI LME.

    Adds ``(1 | subject)`` (intercept-only -- not a random slope, matching
    the split-half noise-ceiling's choice in
    ``neuroai/regressors/lpp_ceiling.py`` -- 48 subjects is plenty for an
    intercept but stacking a second untested random slope alongside
    region's existing one is not worth the added convergence risk) on top
    of the average-subject formula's region/section structure, otherwise
    unchanged."""
    nuisances = "wlen_z + freq_z + sentpos_z"
    re_section = "(1 | section)"
    re_subject = "(1 | subject)"

    if per_region:
        baseline = f"bold ~ {nuisances} + {re_section} + {re_subject}"
        full = f"bold ~ {nuisances} + surprisal_z + {re_section} + {re_subject}"
        return baseline, full

    re_region_base = "(1 | region)"
    re_region_full = "(1 + surprisal_z | region)"

    baseline = f"bold ~ {nuisances} + {re_region_base} + {re_section} + {re_subject}"
    full = f"bold ~ {nuisances} + surprisal_z + {re_region_full} + {re_section} + {re_subject}"
    return baseline, full


def fit_lpp_lme_one_layer(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    allow_slope_fallback: bool = True,
    save_per_word: bool = False,
    persubject: bool = False,
    per_region: bool = False,
) -> LmeFitResult:
    """Fit the pooled LPP fMRI LME for one LM layer.

    ``df`` must have one row per (region, section, TR) with columns
    ``region, section, chunk, wlen, freq, sentpos, surprisal, bold`` (plus
    ``subject`` when ``persubject=True``) — ``surprisal`` is this specific
    layer's HRF-convolved logit-lens surprisal.

    If the full model fails to converge and ``allow_slope_fallback=True``,
    retries first without the ``region`` slope-intercept correlation, then
    with the ``region`` random slope on ``surprisal_z`` dropped entirely
    (intercept-only for both ``region`` and ``section`` -- ``subject``, when
    present, stays intercept-only throughout the ladder) — the standard
    Barr et al. (2013)-style fallback ladder used by every other pooled LME
    in this codebase. Only 10 ``region`` levels back this slope (vs. e.g.
    Podcast's 285 subject:electrode groups), so non-convergence here is a
    real possibility, not just a formality.
    """
    from pymer4.models import compare, lmer

    data = prepare_lpp_lme_frame_persubject(df) if persubject else prepare_lpp_lme_frame(df)
    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )

    baseline_formula, full_formula = (
        lpp_lme_formulas_persubject(per_region=per_region)
        if persubject else lpp_lme_formulas(per_region=per_region)
    )
    outcome = "bold"
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
    if allow_slope_fallback and not (base_ok and converged):
        logger.warning(
            "LPP pooled LME: full/baseline model failed to converge — retrying "
            "without the region slope-intercept correlation."
        )
        full_formula_nocorr = full_formula.replace(
            "(1 + surprisal_z | region)", "(1 + surprisal_z || region)"
        ) if not per_region else full_formula
        full_model = _fit(full_formula_nocorr)
        converged = _converged(full_model)
        if not (base_ok and converged):
            logger.warning(
                "LPP pooled LME: still not converged — dropping the region "
                "random slope on surprisal_z entirely (intercept-only for "
                "region and section)."
            )
            nuisances = "wlen_z + freq_z + sentpos_z"
            re_subject = " + (1 | subject)" if persubject else ""
            region_re = " + (1 | region)" if not per_region else ""
            baseline_formula = f"bold ~ {nuisances}{region_re} + (1 | section){re_subject}"
            full_formula = (
                f"bold ~ {nuisances} + surprisal_z{region_re} + (1 | section){re_subject}"
            )
            baseline_model = _fit(baseline_formula)
            base_ok = _converged(baseline_model)
            full_model = _fit(full_formula)
            converged = _converged(full_model)
            slope_dropped = not per_region

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
    if (
        save_per_word and test_data.height >= 10
        and base_per_obs is not None and full_per_obs is not None
    ):
        id_cols = [c for c in ("subject", "region", "section", "chunk") if c in test_data.columns]
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
