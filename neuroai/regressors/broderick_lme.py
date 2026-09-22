"""Linear mixed-effects helpers for Broderick 2018 EEG, fit SEPARATELY PER
SUBJECT (then averaged across subjects downstream), independently per LM
layer, region-averaged (per-layer logit-lens surprisal, not embeddings).

Design rationale
-----------------
An earlier version of this module pooled all 19 subjects into one model per
layer, with ``subject`` as a random effect carrying a 4-term correlated
slope (``wlen_z + freq_z + surprisal_z + prev_surprisal_z``, a 5x5
covariance matrix = 15 free parameters) alongside ``region``'s own slope (3
more parameters). Region-averaging Broderick's 128 BioSemi channels into 4
anatomical groups (left/right/posterior/occipital, via
``neuroai/datasets/broderick.py::channel_region``) had already cut the
pooled table from ~28M to ~870K rows, but in production this STILL failed
to complete even a single layer fit within a 1-4h walltime. Since ZuCo's
pooled LME hit the identical wall at ~5M rows (~6x more data than
Broderick's ~870K), the bottleneck is the COVARIANCE SEARCH DIMENSIONALITY
from the rich correlated ``subject`` slope, not data volume — consistent
with LPP's pooled LME (``regressors/lpp_pooled_lme.py``) being fast
precisely because it has no ``subject`` random effect at all (average-
subject BOLD).

This version applies that lesson directly: fit one LME PER SUBJECT (so
``subject`` is never a grouping factor — with only one subject in the data,
a between-subject variance isn't even identifiable), keeping only
``region`` as a random effect with its slope on ``surprisal_z``. This is
structurally identical to LPP's one-random-slope-group formula, and mirrors
this codebase's own per-subject ridge/TRF encoder (``benchmarks/
broderick.py``), which likewise fits independently per subject and averages
results afterward — the mixed-effects analogue of that same pattern, not a
new one. Region-averaging is still applied (see
``benchmarks/broderick_lme.py``), just now per-subject rather than pooled.

    baseline: eeg ~ wlen_z + freq_z + (1 | region)
    full:     eeg ~ wlen_z + freq_z + surprisal_z + prev_surprisal_z +
                     (1 + surprisal_z | region)

``prev_surprisal_z`` spillover control and ``region``'s random slope
(scoped to ``surprisal_z`` only) are unchanged from the pooled version.
What changes is only that between-subject variability is no longer modeled
explicitly inside the LME — it's captured by fitting per subject and
averaging the resulting per-subject estimates (beta, delta_loglik_test, ...)
downstream, the same two-stage pattern (within-subject fit -> across-
subject average) used throughout neuroimaging and by this codebase's own
ridge/TRF encoders.

Barr et al. (2013) fallback ladder is now just 2 steps (only one random
slope left to simplify): drop the ``region`` slope-intercept correlation,
then drop the ``region`` random slope entirely (intercept-only) — same
pattern as ``regressors/podcast_lme.py``'s single-grouping-factor ladder.
With only 4 region levels, expect this to trigger more than LPP's 10-region
case, but each fit is now cheap regardless.

Train/held-out split reuses ``regressors/fmri_lme.py::split_train_test_chunks``
with ``chunk = segment_id`` (Broderick's 20 audiobook segments) — the
natural continuous-stimulus unit, never split across the train/test
boundary.

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
    "region", "chunk", "word_idx", "wlen", "freq", "prev_surprisal", "surprisal", "eeg",
}


def broderick_lme_formulas() -> tuple[str, str]:
    """Baseline/full formulas for the per-subject Broderick EEG LME (one LM layer).

    No ``subject`` term — this formula is fit separately per subject (see
    module docstring), so there is exactly one random-effect grouping factor
    (``region``), matching LPP's one-random-slope-group structure.
    """
    nuisances = "wlen_z + freq_z"
    re_region_base = "(1 | region)"
    re_region_full = "(1 + surprisal_z | region)"

    baseline = f"eeg ~ {nuisances} + {re_region_base}"
    full = f"eeg ~ {nuisances} + surprisal_z + prev_surprisal_z + {re_region_full}"
    return baseline, full


def prepare_broderick_lme_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Factor columns + z-scored predictors for the per-subject Broderick EEG LME.

    ``df`` is expected to already be restricted to ONE subject's rows — no
    ``subject`` column is required or used. ``region`` values (left/right/
    posterior/occipital) are fixed canonical labels derived from the same
    BioSemi A/B/C/D quadrant convention for every participant.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Broderick per-subject LME: {sorted(missing)}")

    out = df.copy()
    out["region"] = out["region"].astype(str)
    out["chunk"] = out["chunk"].astype(str)  # reuse split_train_test_chunks as-is
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["prev_surprisal_z"] = zscore_series(out["prev_surprisal"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["eeg"] = out["eeg"].astype(float)

    cols = [
        "region", "chunk", "word_idx",
        "wlen_z", "freq_z", "prev_surprisal_z", "surprisal_z", "eeg",
    ]
    return pl.from_pandas(out[cols].copy())


def fit_broderick_lme_one_layer(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    allow_slope_fallback: bool = True,
    save_per_word: bool = False,
) -> LmeFitResult:
    """Fit the per-subject Broderick EEG LME for one LM layer.

    ``df`` must have one row per (region, word) for a SINGLE subject, with
    columns ``region, chunk, word_idx, wlen, freq, prev_surprisal, surprisal,
    eeg`` — ``eeg`` is the region-averaged, window-averaged EEG amplitude;
    ``surprisal`` is this specific layer's logit-lens surprisal.

    If the full model fails to converge, and ``allow_slope_fallback=True``,
    retries without the ``region`` slope-intercept correlation, then drops
    the ``region`` random slope on ``surprisal_z`` entirely (intercept-only)
    — standard Barr et al. (2013)-style simplification, applied identically
    to both models being compared so the LRT stays valid.

    Pass ``save_per_word=True`` to additionally return a per-held-out-row
    DataFrame (``LmeFitResult.per_word``: region, chunk, word_idx, observed
    eeg, baseline/full predictions, per-observation log-likelihoods) — same
    DLT-difficulty-analysis convention as the behavioral LME datasets
    (``regressors/ucl_reading_lme.py``).
    """
    from pymer4.models import compare, lmer

    data = prepare_broderick_lme_frame(df)
    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )

    baseline_formula, full_formula = broderick_lme_formulas()
    outcome = "eeg"
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

    nuisances = "wlen_z + freq_z"
    slope_dropped = False
    if allow_slope_fallback and not (base_ok and converged):
        logger.warning(
            "Broderick per-subject LME: full/baseline model failed to "
            "converge — retrying without the region slope-intercept correlation."
        )
        full_formula_step1 = full_formula.replace(
            "(1 + surprisal_z | region)",
            "(1 + surprisal_z || region)",
        )
        full_model = _fit(full_formula_step1)
        converged = _converged(full_model)
        if not (base_ok and converged):
            logger.warning(
                "Broderick per-subject LME: still not converged — dropping "
                "the region random slope on surprisal_z entirely."
            )
            baseline_formula = f"eeg ~ {nuisances} + (1 | region)"
            full_formula = f"eeg ~ {nuisances} + surprisal_z + prev_surprisal_z + (1 | region)"
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
    per_word: pd.DataFrame | None = None
    if test_data.height >= 10:
        baseline_loglik_test, base_per_obs = _partition_loglik_detailed(
            baseline_model, test_data, outcome=outcome
        )
        loglik_test, full_per_obs = _partition_loglik_detailed(
            full_model, test_data, outcome=outcome
        )
        if np.isfinite(baseline_loglik_test) and np.isfinite(loglik_test):
            delta_loglik_test = loglik_test - baseline_loglik_test
        if save_per_word:
            id_cols = [c for c in ("region", "chunk", "word_idx") if c in test_data.columns]
            per_word = test_data.select(id_cols).to_pandas()
            per_word["baseline_loglik"] = base_per_obs
            per_word["full_loglik"] = full_per_obs
            per_word["delta_loglik"] = full_per_obs - base_per_obs

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

    if per_word is not None:
        per_word[outcome] = y_true
        per_word["yhat_baseline"] = yhat_base
        per_word["yhat_full"] = yhat_full

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
