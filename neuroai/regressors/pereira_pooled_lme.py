"""Linear mixed-effects helpers for Pereira et al. (2018) fMRI, pooled across
subjects, ROIs, and sentences, fit independently per LM layer (per-layer
logit-lens surprisal, not embeddings).

Design rationale
-----------------
Nearly identical shape to Tuckute (``regressors/tuckute_pooled_lme.py``):
per-SENTENCE BOLD, isolated sentences, one response per (subject, roi,
sentence), no HRF needed. In fact Pereira is the EXACT dataset Lin & Schuler
(2506.11338v2) fit their per-sentence fMRI LME to (their Experiment 2), so
this formula follows their design directly rather than Tuckute's adaptation
of it:

    baseline: y ~ sentlen_z + mean_wlen_z + mean_freq_z +
                  (1 + sentlen_z + mean_wlen_z + mean_freq_z | subject) +
                  (1 | roi) +
                  (1 | sentence_id)
    full:     y ~ sentlen_z + mean_wlen_z + mean_freq_z + surprisal_z +
                  (1 + sentlen_z + mean_wlen_z + mean_freq_z + surprisal_z | subject) +
                  (1 | roi) +
                  (1 + surprisal_z | sentence_id)

Three points of departure from Tuckute's copy of this formula, all because
Pereira's sentences (unlike Tuckute's fixed-6-word sentences) actually vary
in length:

1. **``sentlen_z`` (word count) is included as a real nuisance** -- this is
   Lin & Schuler's own ``sentlen`` covariate; Tuckute had to drop it (zero
   variance, all sentences exactly 6 words) and substitute mean wlen/freq
   instead. Here both are kept: ``sentlen_z`` captures the trial-level
   "how much text was there" effect, ``mean_wlen_z``/``mean_freq_z`` still
   control for lexical difficulty distributed across the sentence.
2. **Predictor = last-word surprisal**, exactly Lin & Schuler's own
   convention for this exact dataset ("its preceding context has already
   been taken into consideration... its the last words of the sentences") --
   not a stand-in borrowed from Tuckute, but literally what they did.
3. **``roi`` is a crossed random effect with 12 levels** (the language-atlas
   ROI labels, present in all 10 subjects -- see ``neuroai/datasets/
   pereira.py`` module docstring), not Tuckute's 6-level ``LANG_ROIS``; the
   formula shape is unaffected (still intercept-only, no random slope, same
   reasoning as Tuckute's crossed-electrode term).

No ``prev_surprisal`` spillover term, same reasoning as Tuckute: a
sentence-integrated BOLD signal already reflects every word's contribution,
and Lin & Schuler include no such term for Pereira either.
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
    "subject", "roi", "sentence_id", "sentlen", "mean_wlen", "mean_freq", "surprisal", "y",
}
_NUISANCES = "sentlen_z + mean_wlen_z + mean_freq_z"


def pereira_lme_formulas(*, per_roi: bool = False) -> tuple[str, str]:
    """Baseline/full formulas for pooled or one-ROI Pereira fits.

    ``y`` is the per-(subject, roi, sentence) BOLD value (averaged across
    voxels within the ROI); ``surprisal_z`` is that layer's last-word
    logit-lens surprisal for the sentence.
    """
    re_subject_base = f"(1 + {_NUISANCES} | subject)"
    re_subject_full = f"(1 + {_NUISANCES} + surprisal_z | subject)"
    re_roi = "" if per_roi else " + (1 | roi)"
    re_sentence_base = "(1 | sentence_id)"
    re_sentence_full = "(1 + surprisal_z | sentence_id)"

    baseline = f"y ~ {_NUISANCES} + {re_subject_base}{re_roi} + {re_sentence_base}"
    full = f"y ~ {_NUISANCES} + surprisal_z + {re_subject_full}{re_roi} + {re_sentence_full}"
    return baseline, full


def prepare_pereira_lme_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Factor columns + z-scored predictors for the pooled Pereira LME.

    ``roi`` values ARE globally comparable across subjects (fixed canonical
    12-label language-atlas set, same functional labels for every
    participant) -- a genuinely crossed random effect, like Tuckute's
    ``LANG_ROIS``.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Pereira pooled LME: {sorted(missing)}")

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["roi"] = out["roi"].astype(str)
    out["sentence_id"] = out["sentence_id"].astype(str)
    out["chunk"] = out["sentence_id"]  # reuse split_train_test_chunks as-is
    out["sentlen_z"] = zscore_series(out["sentlen"])
    out["mean_wlen_z"] = zscore_series(out["mean_wlen"])
    out["mean_freq_z"] = zscore_series(out["mean_freq"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["y"] = out["y"].astype(float)

    cols = [
        "subject", "roi", "sentence_id", "chunk",
        "sentlen_z", "mean_wlen_z", "mean_freq_z", "surprisal_z", "y",
    ]
    return pl.from_pandas(out[cols].copy())


def fit_pereira_lme_one_layer(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    allow_slope_fallback: bool = True,
    per_roi: bool = False,
) -> LmeFitResult:
    """Fit the pooled Pereira fMRI LME for one LM layer.

    ``df`` must have one row per (subject, roi, sentence) with columns
    ``subject, roi, sentence_id, sentlen, mean_wlen, mean_freq, surprisal,
    y`` -- ``surprisal`` is this specific layer's last-word logit-lens
    surprisal for the sentence.

    Same Barr et al. (2013)-style convergence fallback ladder as Tuckute's
    ``fit_tuckute_lme_one_layer`` (see its docstring): drop the
    ``sentence_id`` slope-intercept correlation, then the slope entirely,
    then the ``subject`` slope-intercept correlation, then all random slopes
    -- applied identically to both models being compared so the LRT stays
    valid throughout.
    """
    from pymer4.models import compare, lmer

    data = prepare_pereira_lme_frame(df)
    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )

    baseline_formula, full_formula = pereira_lme_formulas(per_roi=per_roi)
    outcome = "y"
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
            "Pereira pooled LME: full/baseline model failed to converge — retrying "
            "without the sentence_id slope-intercept correlation."
        )
        full_formula_step1 = full_formula.replace(
            "(1 + surprisal_z | sentence_id)",
            "(1 + surprisal_z || sentence_id)",
        )
        full_model = _fit(full_formula_step1)
        converged = _converged(full_model)
        if not (base_ok and converged):
            logger.warning(
                "Pereira pooled LME: still not converged — dropping the "
                "sentence_id random slope on surprisal_z entirely."
            )
            full_formula_step2 = (
                f"y ~ {_NUISANCES} + surprisal_z + "
                f"(1 + {_NUISANCES} + surprisal_z | subject)"
                f"{' + (1 | roi)' if not per_roi else ''} + (1 | sentence_id)"
            )
            full_model = _fit(full_formula_step2)
            converged = _converged(full_model)
            if not (base_ok and converged):
                logger.warning(
                    "Pereira pooled LME: still not converged — retrying without "
                    "the subject slope-intercept correlation."
                )
                full_formula_step3 = full_formula_step2.replace(
                    f"(1 + {_NUISANCES} + surprisal_z | subject)",
                    f"(1 + {_NUISANCES} + surprisal_z || subject)",
                )
                full_model = _fit(full_formula_step3)
                converged = _converged(full_model)
                if not (base_ok and converged):
                    logger.warning(
                        "Pereira pooled LME: still not converged — dropping all "
                        "random slopes entirely (intercept-only for subject, "
                        "roi, and sentence_id)."
                    )
                    baseline_formula = (
                        f"y ~ {_NUISANCES} + (1 | subject)"
                        f"{' + (1 | roi)' if not per_roi else ''} + (1 | sentence_id)"
                    )
                    full_formula = (
                        f"y ~ {_NUISANCES} + surprisal_z + "
                        f"(1 | subject)"
                        f"{' + (1 | roi)' if not per_roi else ''} + (1 | sentence_id)"
                    )
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
    if test_data.height >= 10:
        baseline_loglik_test, _ = _partition_loglik_detailed(
            baseline_model, test_data, outcome=outcome
        )
        loglik_test, _ = _partition_loglik_detailed(
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
        base_p, base_s, full_p, full_s = np.nan, np.nan, np.nan, np.nan

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
    )
