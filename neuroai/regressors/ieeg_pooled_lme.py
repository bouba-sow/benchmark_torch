"""Linear mixed-effects helpers for Fedorenko2016 word-level iEEG, pooled
across subjects and electrodes, fit independently per LM layer (per-layer
logit-lens surprisal, not embeddings).

Fedorenko2016 is intracranial (SEEG/ECoG-style, patient-specific electrode
grids) — electrode 12 on subject 1 and electrode 12 on subject 2 refer to
unrelated physical sites, unlike the N400 datasets' standard 10-20 scalp
locations (crossed electrode, see ``regressors/lme.py::n400_lme_formulas``).
This is architecturally identical to Podcast's ECoG situation
(``regressors/podcast_lme.py``): electrode must be a NESTED random effect,
``subject:electrode``, not crossed.

Unlike Podcast (which fits one model per lag), Fedorenko2016 has no time axis
— "Neural targets are scalar responses per electrode column (no time axis)"
(``neuroai/datasets/fedorenko2016.py``) — so, like the N400 datasets, this
fits exactly once per LM layer, no per-lag outer loop.

    baseline: y ~ wlen_z + freq_z +
                  (1 + wlen_z + freq_z | subject) +
                  (1 | subject:electrode)
    full:     y ~ wlen_z + freq_z + surprisal_z + prev_surprisal_z +
                  (1 + wlen_z + freq_z + surprisal_z + prev_surprisal_z | subject) +
                  (1 + surprisal_z + prev_surprisal_z | subject:electrode)

``prev_surprisal_z`` (spillover control) is standard now: absent from the
baseline, added to the full model alongside ``surprisal_z`` — both as fixed
effects and as a joint random slope on EVERY grouping factor that already
carried a slope on ``surprisal_z`` (both ``subject`` and the nested
``subject:electrode``, "option B" — full maximal extension, not just
``subject``), since Fedorenko2016's per-layer cost is far below Podcast's
(fewer electrodes/subjects, no per-lag outer loop), so the extra 2x2->3x3
covariance dimensionality on ``subject:electrode`` is not the same concern it
would be for Podcast (left untouched, see ``regressors/podcast_lme.py``).

Train/held-out split is by sentence (``sentence_id``, renamed to ``chunk`` to
reuse ``fmri_lme.split_train_test_chunks`` unmodified) — a sentence's words
never cross the train/test boundary, mirroring the sentence-grouped CV
already used by ``benchmarks/fedorenko2016.py``'s ridge encoder.

A parallel module to ``regressors/n400_pooled_lme.py`` rather than a
generalization of it: Nieuwland LME sweeps are actively
running against ``n400_pooled_lme.py`` on Jean Zay, so this duplicates
(instead of parameterizing) the fitting logic to avoid touching that shared
code. Reuses the outcome-agnostic low-level helpers from ``regressors/lme.py``
and the chunk-split logic from ``regressors/fmri_lme.py``, same as
``n400_pooled_lme.py`` and ``podcast_lme.py``.
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
    "subject", "electrode", "sentence_id", "wlen", "freq", "prev_surprisal", "surprisal", "y",
}
# Carried through when present (not required) so a save_per_word=True fit can
# identify which word/electrode each held-out prediction belongs to — see
# fit_ieeg_lme_one_layer's save_per_word docstring.
_OPTIONAL_ID_COLS = ("word", "word_num", "stimulus_id")


def prepare_ieeg_lme_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Factor columns + z-scored predictors for a pooled iEEG LME.

    ``electrode`` values need not be globally unique across subjects — lme4's
    ``subject:electrode`` interaction term handles the nesting regardless of
    whether names repeat between subjects (same as ``prepare_podcast_lme_frame``).
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for iEEG pooled LME: {sorted(missing)}")

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["electrode"] = out["electrode"].astype(str)
    out["chunk"] = out["sentence_id"].astype(str)  # reuse split_train_test_chunks as-is
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["prev_surprisal_z"] = zscore_series(out["prev_surprisal"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["y"] = out["y"].astype(float)

    cols = ["subject", "electrode", "chunk", "wlen_z", "freq_z", "prev_surprisal_z", "surprisal_z", "y"]
    cols += [c for c in _OPTIONAL_ID_COLS if c in out.columns]
    return pl.from_pandas(out[cols].copy())


def ieeg_lme_formulas() -> tuple[str, str]:
    """Nested subject/subject:electrode formulas, generalizing
    ``podcast_lme_formulas`` to per-layer form (with the N400-style
    ``prev_surprisal_z`` spillover control, absent from Podcast's formula).
    ``prev_surprisal_z`` is added only in the full model, alongside
    ``surprisal_z`` — see module docstring.
    """
    nuisances = "wlen_z + freq_z"
    re_subject_base = f"(1 + {nuisances} | subject)"
    re_subject_full = f"(1 + {nuisances} + surprisal_z + prev_surprisal_z | subject)"
    re_electrode_base = "(1 | subject:electrode)"
    re_electrode_full = "(1 + surprisal_z + prev_surprisal_z | subject:electrode)"

    baseline = f"y ~ {nuisances} + {re_subject_base} + {re_electrode_base}"
    full = f"y ~ {nuisances} + surprisal_z + prev_surprisal_z + {re_subject_full} + {re_electrode_full}"
    return baseline, full


def fit_ieeg_lme_one_layer(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    allow_slope_fallback: bool = True,
    save_per_word: bool = False,
) -> LmeFitResult:
    """Fit the pooled Fedorenko2016 iEEG LME for one LM layer.

    ``df`` must have one row per (subject, electrode, word) with columns
    ``subject, electrode, sentence_id, wlen, freq, prev_surprisal, surprisal,
    y`` — ``surprisal`` is this specific layer's logit-lens surprisal.
    Optionally also ``word``, ``word_num``, ``stimulus_id`` (carried through
    verbatim by ``prepare_ieeg_lme_frame`` when present) to identify each
    held-out prediction's word when ``save_per_word=True``.

    If the full model (with the correlated random slope on
    ``subject:electrode``) fails to converge, and ``allow_slope_fallback=True``,
    retries without the slope-intercept correlation and finally without the
    random slope entirely — same Barr et al. (2013)-style ladder as
    ``regressors.podcast_lme.fit_podcast_lme_one_lag`` and
    ``regressors.n400_pooled_lme.fit_n400_lme_one_layer``, applied identically
    to both models being compared so the LRT stays valid.

    ``save_per_word=True`` populates the returned result's ``per_word`` with
    one row per held-out (test-set) word x electrode: observed ``y``,
    baseline/full model predictions, and per-observation log-likelihoods —
    mirrors ``regressors.podcast_lme.fit_podcast_lme_one_lag``'s convention
    exactly (same column names), for post-hoc per-word difficulty splits
    (e.g. DLT cost). These values are already computed internally for the
    held-out correlation metrics below, so this costs nothing extra.
    """
    from pymer4.models import compare, lmer

    data = prepare_ieeg_lme_frame(df)
    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )

    baseline_formula, full_formula = ieeg_lme_formulas()
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

    nuisances = "wlen_z + freq_z + prev_surprisal_z"
    slope_dropped = False
    if allow_slope_fallback and not (base_ok and converged):
        logger.warning(
            "iEEG pooled LME: full/baseline model failed to converge — retrying "
            "without the subject:electrode slope-intercept correlation."
        )
        full_formula_nocorr = full_formula.replace(
            "(1 + surprisal_z | subject:electrode)",
            "(1 + surprisal_z || subject:electrode)",
        )
        full_model = _fit(full_formula_nocorr)
        converged = _converged(full_model)
        if not (base_ok and converged):
            logger.warning(
                "iEEG pooled LME: still not converged — dropping the random "
                "slope on subject:electrode entirely (intercept-only)."
            )
            baseline_formula = f"y ~ {nuisances} + (1 | subject) + (1 | subject:electrode)"
            full_formula = f"y ~ {nuisances} + surprisal_z + (1 | subject) + (1 | subject:electrode)"
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
    yhat_base = yhat_full = None
    try:
        yhat_base = _predict_or_nan(baseline_model, test_data)
        yhat_full = _predict_or_nan(full_model, test_data)
        base_p, base_s = _corrs(y_true, yhat_base)
        full_p, full_s = _corrs(y_true, yhat_full)
    except Exception as exc:
        logger.warning("Prediction failed for OOS correlation: %s", exc)
        base_p, base_s, full_p, full_s = np.nan, np.nan, np.nan, np.nan

    per_word: pd.DataFrame | None = None
    if (
        save_per_word and test_data.height >= 10
        and base_per_obs is not None and full_per_obs is not None
        and yhat_base is not None and yhat_full is not None
    ):
        id_cols = [
            c for c in ("subject", "electrode", "chunk", "word", "word_num", "stimulus_id")
            if c in test_data.columns
        ]
        per_word = test_data.select(id_cols).to_pandas()
        per_word["y_true"] = y_true
        per_word["yhat_base"] = np.asarray(yhat_base, dtype=np.float64).reshape(-1)
        per_word["yhat_full"] = np.asarray(yhat_full, dtype=np.float64).reshape(-1)
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
