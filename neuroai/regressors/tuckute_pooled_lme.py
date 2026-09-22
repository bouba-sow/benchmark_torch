"""Linear mixed-effects helpers for Tuckute et al. (2023) fMRI, pooled across
subjects, ROIs, and sentences, fit independently per LM layer (per-layer
logit-lens surprisal, not embeddings).

Design rationale
-----------------
Tuckute is per-SENTENCE BOLD (2,000 isolated 6-word sentences × up to 10
subjects × 6 language ROIs; "BOLD responses are sentence-level, no HRF
needed" — ``neuroai/datasets/tuckute.py``), unlike the word-level iEEG/EEG
datasets in this family (Fedorenko2016, ...). Three design
choices follow directly from Lin & Schuler (2506.11338v2), who fit an LME to
Pereira et al. (2018)'s structurally-identical per-sentence fMRI data
(isolated sentences, one BOLD scalar per sentence):

1. **Predictor = last-word surprisal, not a sentence-mean.** Their stated
   rationale: "its preceding context has already been taken into
   consideration when calculating the surprisal estimates of the last words
   of the sentences" — a single forward pass over the whole sentence already
   conditions the last word's surprisal on every earlier word, so it is a
   valid summary of the sentence's cumulative predictability, following
   Hosseini et al. (2024)'s use of the same convention on this same dataset
   family.
2. **No ``prev_surprisal`` spillover term.** Unlike word-level ERP/iEEG data
   (where spillover from the immediately-preceding word into the current
   word's time-locked response is a well-established, specific confound —
   see ``regressors/ieeg_pooled_lme.py``), a sentence-integrated BOLD signal
   already reflects every word's contribution; there is no analogous
   "leakage from the previous trial" mechanism to correct for. Lin & Schuler
   include no such term either.
3. **Nuisances are sentence-level, not last-word-level**: mean word length
   and mean wordfreq Zipf frequency ACROSS the sentence, not the last word's
   own. Lin & Schuler use ``sentlen`` (sentence length in words) — a
   trial-level, not readout-word-level, covariate — but Tuckute's sentences
   are all fixed at 6 words, so ``sentlen`` has zero variance and can't be
   reused directly; mean wlen/freq is the natural trial-level substitute that
   still varies across sentences AND controls for lexical confounds
   distributed across every word of the sentence (not just the last one,
   which last-word wlen/freq alone would miss).

One further departure from Lin & Schuler's Pereira formula, forced by
Tuckute's data (not a stylistic choice): Pereira's BOLD is already
"aggregated across regions" into one scalar per sentence, so their formula
only has ``subject`` and ``sentid`` as crossed random effects. Tuckute keeps
6 distinct ROI-level responses per sentence, so a THIRD crossed grouping
factor, ``roi``, is required — ROI names are a fixed canonical set shared
across all participants (``LANG_ROIS``), so, as with a scalp
electrode (``regressors/lme.py::n400_lme_formulas``), this is genuinely
crossed, not nested like Fedorenko2016/Podcast's patient-specific electrode
grids. Given only 6 ROI levels, ``roi`` gets an intercept only (no random
slope) — same reasoning as the N400 datasets' crossed-electrode term.

    baseline: y ~ mean_wlen_z + mean_freq_z +
                  (1 + mean_wlen_z + mean_freq_z | subject) +
                  (1 | roi) +
                  (1 | sentence_id)
    full:     y ~ mean_wlen_z + mean_freq_z + surprisal_z +
                  (1 + mean_wlen_z + mean_freq_z + surprisal_z | subject) +
                  (1 | roi) +
                  (1 + surprisal_z | sentence_id)

The ``sentence_id`` random slope on ``surprisal_z`` mirrors Lin & Schuler's
``sentid`` slope on ``lastwordsurp`` in their Pereira formula (Appendix B) —
note ``surprisal_z`` is CONSTANT within a given ``sentence_id`` group (it's a
property of the sentence, shared by every subject/ROI row of that sentence),
so this slope is not separately identifiable from the ``sentence_id``
intercept in the strict sense; lme4 can still sometimes fit it (as Lin &
Schuler's Appendix B shows it surviving their fallback ladder), but if it
doesn't converge here, the standard Barr et al. (2013) fallback (drop
correlation, then drop the slope) — already used by every other pooled LME
in this codebase — degrades it automatically.
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

_REQUIRED_COLS = {"subject", "roi", "sentence_id", "mean_wlen", "mean_freq", "surprisal", "y"}


def tuckute_lme_formulas() -> tuple[str, str]:
    """Baseline/full formulas for the pooled Tuckute fMRI LME (one LM layer).

    ``y`` is the per-(subject, roi, sentence) BOLD value; ``surprisal_z`` is
    that layer's last-word logit-lens surprisal for the sentence.
    """
    nuisances = "mean_wlen_z + mean_freq_z"
    re_subject_base = f"(1 + {nuisances} | subject)"
    re_subject_full = f"(1 + {nuisances} + surprisal_z | subject)"
    re_roi = "(1 | roi)"
    re_sentence_base = "(1 | sentence_id)"
    re_sentence_full = "(1 + surprisal_z | sentence_id)"

    baseline = f"y ~ {nuisances} + {re_subject_base} + {re_roi} + {re_sentence_base}"
    full = f"y ~ {nuisances} + surprisal_z + {re_subject_full} + {re_roi} + {re_sentence_full}"
    return baseline, full


def prepare_tuckute_lme_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Factor columns + z-scored predictors for the pooled Tuckute LME.

    ``roi`` values ARE globally comparable across subjects (fixed canonical
    ``LANG_ROIS`` set, same functional labels for every participant) — a
    genuinely crossed random effect, like a scalp electrode, unlike
    Fedorenko2016/Podcast's per-subject electrode grids.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Tuckute pooled LME: {sorted(missing)}")

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["roi"] = out["roi"].astype(str)
    out["sentence_id"] = out["sentence_id"].astype(str)
    out["chunk"] = out["sentence_id"]  # reuse split_train_test_chunks as-is
    out["mean_wlen_z"] = zscore_series(out["mean_wlen"])
    out["mean_freq_z"] = zscore_series(out["mean_freq"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["y"] = out["y"].astype(float)

    cols = ["subject", "roi", "sentence_id", "chunk", "mean_wlen_z", "mean_freq_z", "surprisal_z", "y"]
    return pl.from_pandas(out[cols].copy())


def fit_tuckute_lme_one_layer(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    allow_slope_fallback: bool = True,
) -> LmeFitResult:
    """Fit the pooled Tuckute fMRI LME for one LM layer.

    ``df`` must have one row per (subject, roi, sentence) with columns
    ``subject, roi, sentence_id, mean_wlen, mean_freq, surprisal, y`` —
    ``surprisal`` is this specific layer's last-word logit-lens surprisal for
    the sentence.

    If the full model fails to converge, and ``allow_slope_fallback=True``,
    retries without the ``subject`` slope-intercept correlation, then without
    the ``sentence_id`` slope-intercept correlation, and finally drops both
    random slopes entirely (intercept-only for ``subject``, ``roi``, and
    ``sentence_id``) — same Barr et al. (2013)-style ladder as every other
    pooled LME in this codebase, applied identically to both models being
    compared so the LRT stays valid. The ``sentence_id`` slope on
    ``surprisal_z`` is the most likely non-convergence point (see module
    docstring: it's a slope on a within-group-constant predictor), so it is
    dropped before the ``subject`` slope.
    """
    from pymer4.models import compare, lmer

    data = prepare_tuckute_lme_frame(df)
    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )

    baseline_formula, full_formula = tuckute_lme_formulas()
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

    nuisances = "mean_wlen_z + mean_freq_z"
    slope_dropped = False
    if allow_slope_fallback and not (base_ok and converged):
        logger.warning(
            "Tuckute pooled LME: full/baseline model failed to converge — retrying "
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
                "Tuckute pooled LME: still not converged — dropping the "
                "sentence_id random slope on surprisal_z entirely."
            )
            full_formula_step2 = (
                f"y ~ {nuisances} + surprisal_z + "
                f"(1 + {nuisances} + surprisal_z | subject) + (1 | roi) + (1 | sentence_id)"
            )
            full_model = _fit(full_formula_step2)
            converged = _converged(full_model)
            if not (base_ok and converged):
                logger.warning(
                    "Tuckute pooled LME: still not converged — retrying without "
                    "the subject slope-intercept correlation."
                )
                full_formula_step3 = full_formula_step2.replace(
                    f"(1 + {nuisances} + surprisal_z | subject)",
                    f"(1 + {nuisances} + surprisal_z || subject)",
                )
                full_model = _fit(full_formula_step3)
                converged = _converged(full_model)
                if not (base_ok and converged):
                    logger.warning(
                        "Tuckute pooled LME: still not converged — dropping all "
                        "random slopes entirely (intercept-only for subject, "
                        "roi, and sentence_id)."
                    )
                    baseline_formula = (
                        f"y ~ {nuisances} + (1 | subject) + (1 | roi) + (1 | sentence_id)"
                    )
                    full_formula = (
                        f"y ~ {nuisances} + surprisal_z + "
                        f"(1 | subject) + (1 | roi) + (1 | sentence_id)"
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
