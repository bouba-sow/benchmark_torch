"""Linear mixed-effects fitting for UCL self-paced reading (pymer4 / lme4).

Trial-level LME on raw RT with by-subject random slopes, KenLM unigram +
current-word + spillover (previous-word) surprisal as fixed effects, and an
out-of-sample train/test split for held-out ΔLogLik and correlation metrics.

This mirrors the design used for Natural Stories (``neuroai.regressors.lme``,
``clark2025`` style) but is a standalone implementation against UCL Reading's
own column names (``story_id`` = sentence, ``zone`` = word position within
sentence, ``participant_id`` = subject) so the two datasets can evolve
independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LmeFitResult:
    """Fixed-effect estimate for one predictor in the UCL Reading SPR LME."""

    predictor: str
    beta: float
    se: float
    t_stat: float
    p_value: float
    loglik: float
    aic: float
    bic: float
    baseline_loglik: float | None = None
    delta_loglik: float | None = None
    lrt_p_value: float | None = None
    test_pearsonr_full_model: float | None = None
    test_spearmanr_full_model: float | None = None
    test_pearsonr_baseline_lme: float | None = None
    test_spearmanr_baseline_lme: float | None = None
    test_delta_pearsonr: float | None = None
    test_delta_spearmanr: float | None = None
    converged: bool = True
    n_obs: int = 0
    layer: int | None = None
    delta_loglik_train: float | None = None
    delta_loglik_test: float | None = None
    baseline_loglik_test: float | None = None
    loglik_test: float | None = None
    n_obs_test: int = 0
    prev_surprisal_beta: float | None = None
    # Held-out per-word delta log-likelihood (one row per test-set word), used
    # to split the DLT-cost analysis into easy/hard bins post-hoc. None unless
    # ``fit_ucl_reading_lme(..., save_per_word=True)``.
    per_word: pd.DataFrame | None = None


def zscore_series(s: pd.Series) -> pd.Series:
    """Column-wise z-score; constant columns → zeros."""
    v = s.astype(float)
    mu = v.mean()
    sd = v.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=v.index)
    return (v - mu) / sd


def word_zipf_freq(words: pd.Series, *, language: str = "en") -> pd.Series:
    """Zipf lexical frequency per word (``wordfreq.zipf_frequency``).

    Same source/convention as the neural LME datasets' frequency nuisance
    (``neuroai.features.word_frequency.compute_wlen_wordfreq_covariates``) —
    a fast in-memory lookup, no model load or disk cache needed. Replaces
    the earlier KenLM-based lexical-surprisal nuisance.
    """
    from wordfreq import zipf_frequency

    w = words.astype(str).str.lower()
    freq = w.map(lambda x: zipf_frequency(x, language))
    return pd.Series(freq, index=words.index, dtype=float)


def prepare_ucl_reading_lme_frame(
    trials: pd.DataFrame,
    *,
    predictor_col: str,
    include_freq: bool = True,
    include_prev_surprisal: bool = True,
    filter_sentence_edges: bool = True,
    wordfreq_language: str = "en",
    paper_replication: bool = False,
) -> pl.DataFrame:
    """Build a polars frame for the UCL Reading LME.

    ``paper_replication=True`` additionally computes word-length and
    frequency at lags 1-2, plus surprisal at lag 2 (lag 1 is already
    computed unconditionally via ``include_prev_surprisal``) -- see
    ``ucl_reading_lme_formulas``'s docstring for why (matching Kuribayashi
    et al. 2025's exact baseline/full-model predictor set for their UCL/SPR
    comparison, arXiv:2502.01615)."""
    import polars as pl

    required = {"participant_id", "story_id", "zone", "rt", "word", predictor_col}
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"Missing columns for UCL Reading LME: {sorted(missing)}")

    df = trials.dropna(subset=["rt", predictor_col, "zone", "story_id", "participant_id"]).copy()
    if filter_sentence_edges and "is_edge" in df.columns:
        df = df.loc[~df["is_edge"].astype(bool)].reset_index(drop=True)

    df["subject"] = df["participant_id"].astype(str)
    df["sentid"] = df["story_id"].astype(str)
    df["sentpos"] = df["zone"].astype(int)
    df["wlen"] = df["word"].astype(str).str.len().astype(int)
    df["fdur"] = df["rt"].astype(float)

    df["sentpos_z"] = zscore_series(df["sentpos"])
    df["wlen_z"] = zscore_series(df["wlen"])
    pred_z_name = f"{predictor_col}_z"
    df[pred_z_name] = zscore_series(df[predictor_col])

    cols = ["subject", "word", "sentid", "sentpos", "wlen", "fdur", "sentpos_z", "wlen_z", pred_z_name]

    if include_freq:
        df["freq"] = word_zipf_freq(df["word"], language=wordfreq_language)
        df["freq_z"] = zscore_series(df["freq"])
        cols.append("freq_z")

    if include_prev_surprisal or paper_replication:
        prev_col = f"prev_{predictor_col}"
        if prev_col not in df.columns:
            df = df.sort_values(["participant_id", "story_id", "zone"])
            df[prev_col] = df.groupby(["participant_id", "story_id"], sort=False)[predictor_col].shift(1)
        df = df.dropna(subset=[prev_col])
        df["prev_surprisal_z"] = zscore_series(df[prev_col])
        cols.append("prev_surprisal_z")

    if paper_replication:
        df = df.sort_values(["participant_id", "story_id", "zone"])
        grp = df.groupby(["participant_id", "story_id"], sort=False)
        # Lag-2 surprisal (lag-1 already computed above).
        prev2_col = f"prev2_{predictor_col}"
        df[prev2_col] = grp[predictor_col].shift(2)
        # Lag-1/lag-2 word length and frequency -- Kuribayashi et al.'s
        # baseline includes length(w_{t-1}), length(w_{t-2}), freq(w_{t-1}),
        # freq(w_{t-2}) alongside the current word's length/freq; this
        # pipeline's non-replication baseline only has the current word's.
        df["wlen_prev1"] = grp["wlen"].shift(1)
        df["wlen_prev2"] = grp["wlen"].shift(2)
        df = df.dropna(subset=[prev2_col, "wlen_prev1", "wlen_prev2"])
        df["prev2_surprisal_z"] = zscore_series(df[prev2_col])
        df["wlen_prev1_z"] = zscore_series(df["wlen_prev1"])
        df["wlen_prev2_z"] = zscore_series(df["wlen_prev2"])
        cols.extend(["prev2_surprisal_z", "wlen_prev1_z", "wlen_prev2_z"])
        if include_freq:
            # Shift the already-computed raw `freq` column (not re-lookup
            # per row) -- same pattern as wlen_prev1/2 above.
            df["freq_prev1"] = grp["freq"].shift(1)
            df["freq_prev2"] = grp["freq"].shift(2)
            df = df.dropna(subset=["freq_prev1", "freq_prev2"])
            df["freq_prev1_z"] = zscore_series(df["freq_prev1"])
            df["freq_prev2_z"] = zscore_series(df["freq_prev2"])
            cols.extend(["freq_prev1_z", "freq_prev2_z"])

    return pl.from_pandas(df[cols].copy())


def ucl_reading_lme_formulas(
    *,
    include_freq: bool = True,
    include_prev_surprisal: bool = True,
    paper_replication: bool = False,
    add_sentence_intercept: bool = False,
) -> tuple[str, str]:
    """Baseline and full-model formulas (raw RT, by-subject random slopes).

    ``freq_z`` (Zipf lexical frequency, ``wordfreq``) is the nuisance term —
    same convention as the neural LME datasets, replacing the earlier
    KenLM-based lexical-surprisal nuisance.

    ``paper_replication=True`` switches to Kuribayashi et al. 2025's exact
    UCL/SPR predictor set (arXiv:2502.01615, Section 3.2, Eq. 3): baseline =
    length(w_t)/freq(w_t)/length(w_{t-1})/freq(w_{t-1})/length(w_{t-2})/
    freq(w_{t-2})/s(w_{t-1})/s(w_{t-2}); full = baseline + s(w_t). This
    isolates the CURRENT word's surprisal as the sole delta between the two
    nested models -- unlike the default formula, where the full model adds
    BOTH current and lag-1 surprisal on top of a baseline with no surprisal
    terms at all, so the "delta" there is the combined contribution of two
    surprisal lags, not one. ``sentpos_z`` is kept as an extra nuisance in
    both variants even though the paper doesn't have it (harmless, and
    dropping the one non-paper term isn't part of what was being matched).
    Random-effects structure is deliberately kept smaller than the fixed
    effects here (intercept + current-word terms only) to keep the by-
    subject mixed model tractable -- the paper itself uses plain OLS with no
    random effects at all, so there's no "matching" random-effects target;
    a maximal (9-term) random-slope structure would just invite convergence
    failures with no principled paper-fidelity benefit.

    ``add_sentence_intercept=True`` adds a crossed ``(1 | sentid)`` random
    intercept (only meaningful with ``paper_replication=False``) -- Oh &
    Schuler (2023, arXiv:2304.11389) Section 2.3 uses "by-subject random
    slopes for all fixed effects and random intercepts for each subject",
    plus "a random intercept for each sentence" specifically for eye-gaze
    data (their Dundee corpus; provo/geco/meco/zuco_reading here are also
    eye-tracking). Combine with include_freq=False, include_prev_surprisal=False
    to match their fixed-effect set too (their baseline is word length +
    word position only; no frequency, no spillover -- they "avoid potential
    confounds introduced by the corpus used for frequency estimation by
    evaluating surprisal estimates on their own")."""
    if paper_replication:
        parts = ["sentpos_z", "wlen_z", "wlen_prev1_z", "wlen_prev2_z"]
        if include_freq:
            parts.extend(["freq_z", "freq_prev1_z", "freq_prev2_z"])
        parts.extend(["prev_surprisal_z", "prev2_surprisal_z"])
        nuisances = " + ".join(parts)
        re_base = "(1 + wlen_z + freq_z | subject)" if include_freq else "(1 + wlen_z | subject)"
        baseline = f"fdur ~ {nuisances} + {re_base}"

        lm_terms = parts + ["surprisal_z"]
        re_slopes = ["wlen_z", "freq_z", "surprisal_z"] if include_freq else ["wlen_z", "surprisal_z"]
        re_full = f"(1 + {' + '.join(re_slopes)} | subject)"
        full = f"fdur ~ {' + '.join(lm_terms)} + {re_full}"
        return baseline, full

    parts = ["sentpos_z", "wlen_z"]
    if include_freq:
        parts.append("freq_z")
    nuisances = " + ".join(parts)
    re_base = "(1 + sentpos_z + wlen_z | subject)"
    sent_re = " + (1 | sentid)" if add_sentence_intercept else ""
    baseline = f"fdur ~ {nuisances} + {re_base}{sent_re}"

    lm_terms = list(parts)
    re_slopes = ["sentpos_z", "wlen_z"]
    if include_prev_surprisal:
        lm_terms.extend(["surprisal_z", "prev_surprisal_z"])
        re_slopes.extend(["surprisal_z", "prev_surprisal_z"])
    else:
        lm_terms.append("surprisal_z")
        re_slopes.append("surprisal_z")
    re_full = f"(1 + {' + '.join(re_slopes)} | subject)"
    full = f"fdur ~ {' + '.join(lm_terms)} + {re_full}{sent_re}"
    return baseline, full


def _extract_fixed_effect(model, term: str) -> tuple[float, float, float, float]:
    import polars as pl

    fit = model.result_fit
    if fit is None:
        raise RuntimeError("pymer4 model has no result_fit")
    row = fit.filter(pl.col("term") == term)
    if row.height == 0:
        raise KeyError(f"Term {term!r} not in fixed effects: {fit['term'].to_list()}")
    r = row.row(0, named=True)
    return (
        float(r["estimate"]),
        float(r.get("std_error", r.get("SE", np.nan))),
        float(r.get("statistic", r.get("t_stat", np.nan))),
        float(r.get("p_value", r.get("p", np.nan))),
    )


def _fit_stats(model) -> tuple[float, float, float, bool]:
    stats = model.result_fit_stats
    loglik = float(stats["logLik"].item())
    aic = float(stats["AIC"].item())
    bic = float(stats["BIC"].item())
    from .lme import parse_converged

    converged = parse_converged(getattr(model, "convergence_status", None))
    return loglik, aic, bic, converged


def _residual_sigma(model) -> float:
    """Residual SD from fitted LMM (for Gaussian log-likelihood on new data)."""
    try:
        from rpy2.robjects.packages import importr

        lme4 = importr("lme4")
        sigma = float(np.asarray(lme4.sigma(model.r_model)).reshape(-1)[0])
        if np.isfinite(sigma) and sigma > 0:
            return sigma
    except Exception:
        pass
    stats = model.result_fit_stats
    for key in ("sigma", "residual_std", "RMSE"):
        if key in stats.columns:
            val = float(stats[key].item())
            if np.isfinite(val) and val > 0:
                return val
    return np.nan


def _gaussian_loglik_per_obs(y: np.ndarray, yhat: np.ndarray, sigma: float) -> np.ndarray:
    """Per-observation Gaussian log-density, NaN where invalid (same length as ``y``)."""
    out = np.full(y.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(yhat)
    if valid.sum() < 3 or not np.isfinite(sigma) or sigma <= 0:
        return out
    resid = y[valid] - yhat[valid]
    out[valid] = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (resid / sigma) ** 2
    return out


def _gaussian_loglik(y: np.ndarray, yhat: np.ndarray, sigma: float) -> float:
    per_obs = _gaussian_loglik_per_obs(y, yhat, sigma)
    if np.isfinite(per_obs).sum() < 3:
        return np.nan
    return float(np.nansum(per_obs))


def _predict_or_nan(model, data: pl.DataFrame) -> np.ndarray:
    # np.asarray(...) is a zero-copy VIEW into R's memory when the source is
    # already float64-compatible -- if R's garbage collector reclaims that
    # buffer before this array is consumed (e.g. during later predict()/
    # compare() calls in the same worker), the eventual read is garbage
    # (denormalized near-zero floats), not an error. np.array(..., copy=True)
    # forces an immediate, safe, numpy-owned copy.
    try:
        return np.array(model.predict(data, allow_new_levels=True), dtype=np.float64, copy=True).reshape(-1)
    except TypeError:
        return np.array(model.predict(data), dtype=np.float64, copy=True).reshape(-1)
    except Exception as exc:
        logger.warning("Partition log-likelihood predict failed: %s", exc)
        return np.full(data.height, np.nan, dtype=np.float64)


def _partition_loglik_detailed(
    model, data: pl.DataFrame, *, outcome: str
) -> tuple[float, np.ndarray]:
    """Held-out log-likelihood: (summed scalar, per-observation array)."""
    y = np.asarray(data[outcome], dtype=np.float64)
    yhat = _predict_or_nan(model, data)
    per_obs = _gaussian_loglik_per_obs(y, yhat, _residual_sigma(model))
    n_fin = int(np.isfinite(per_obs).sum())
    # Guard (2026-10-01): a held-out sum that silently covers only a fraction of the test rows
    # (non-finite terms are skipped by nansum) is not comparable to the other model's sum, and
    # produced absurd delta-loglik values (+-2e4) in a few fits. Require >= 99% finite terms.
    if n_fin < max(3, int(0.99 * per_obs.size)):
        if n_fin:
            logger.warning("Held-out loglik: only %d/%d finite per-observation terms -> NaN", n_fin, per_obs.size)
        return np.nan, per_obs
    return float(np.nansum(per_obs)), per_obs


def _partition_loglik(model, data: pl.DataFrame, *, outcome: str) -> float:
    total, _ = _partition_loglik_detailed(model, data, outcome=outcome)
    return total


def _corrs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return np.nan, np.nan
    yt = y_true[valid]
    yp = y_pred[valid]
    pearson = float(np.corrcoef(yt, yp)[0, 1])
    yr = pd.Series(yt).rank(method="average").to_numpy()
    pr = pd.Series(yp).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(yr, pr)[0, 1])
    return pearson, spearman


def _assign_split_ids(data: pl.DataFrame, *, partition_mod: int) -> np.ndarray:
    pdf = data.to_pandas()
    subj_codes, _ = pd.factorize(pdf["subject"], sort=False)
    sent_codes, _ = pd.factorize(pdf["sentid"], sort=False)
    return (subj_codes + sent_codes) % partition_mod


def _split_train_test(
    data: pl.DataFrame,
    *,
    partition_mod: int,
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split on ``factor(subject) + factor(sentid)`` mod ``partition_mod``."""
    import polars as pl

    if partition_mod <= 1:
        return data, data

    split_id = _assign_split_ids(data, partition_mod=partition_mod)
    pdf = data.to_pandas()
    pdf["_split_id"] = split_id

    train_idx = np.asarray(sorted(set(int(i) for i in train_indices)), dtype=int)
    train_mask = np.isin(split_id, train_idx)
    if train_mask.sum() < 10:
        return data, data

    test_idx = np.asarray(sorted(set(int(i) for i in test_indices)), dtype=int)
    test_mask = np.isin(split_id, test_idx)
    if test_mask.sum() < 10:
        # Too few held-out rows: EMPTY test set (held-out metrics -> NaN) instead of
        # silently scoring on the full data, training rows included.
        logger.warning("held-out partition has only %d rows -> no held-out score", int(test_mask.sum()))
        train_df = pl.from_pandas(pdf.loc[train_mask].drop(columns="_split_id").reset_index(drop=True))
        return train_df, data.head(0)

    train_df = pl.from_pandas(pdf.loc[train_mask].drop(columns="_split_id").reset_index(drop=True))
    test_df = pl.from_pandas(pdf.loc[test_mask].drop(columns="_split_id").reset_index(drop=True))
    return train_df, test_df


def fit_ucl_reading_lme(
    trials: pd.DataFrame,
    *,
    predictor_col: str = "surprisal",
    compare_baseline: bool = True,
    out_of_sample: bool = True,
    partition_mod: int = 4,
    partition_fit_indices: tuple[int, ...] = (0, 1),
    partition_test_indices: tuple[int, ...] = (2, 3),
    include_freq: bool = True,
    include_prev_surprisal: bool = True,
    filter_sentence_edges: bool = True,
    wordfreq_language: str = "en",
    verbose: bool = False,
    layer: int | None = None,
    save_per_word: bool = False,
    paper_replication: bool = False,
    add_sentence_intercept: bool = False,
) -> LmeFitResult:
    """Fit the UCL Reading SPR LME for one scalar LM surprisal predictor.

    ``paper_replication=True`` uses Kuribayashi et al. 2025's exact UCL/SPR
    predictor set instead of this pipeline's default -- see
    ``ucl_reading_lme_formulas``'s docstring."""
    from pymer4.models import compare, lmer

    pred_z = f"{predictor_col}_z"

    data = prepare_ucl_reading_lme_frame(
        trials,
        predictor_col=predictor_col,
        include_freq=include_freq,
        include_prev_surprisal=include_prev_surprisal,
        filter_sentence_edges=filter_sentence_edges,
        wordfreq_language=wordfreq_language,
        paper_replication=paper_replication,
    )

    train_data = data
    test_data = data
    if out_of_sample:
        train_data, test_data = _split_train_test(
            data,
            partition_mod=partition_mod,
            train_indices=partition_fit_indices,
            test_indices=partition_test_indices,
        )

    baseline_formula, full_formula = ucl_reading_lme_formulas(
        include_freq=include_freq,
        include_prev_surprisal=include_prev_surprisal,
        paper_replication=paper_replication,
        add_sentence_intercept=add_sentence_intercept,
    )
    outcome = "fdur"
    beta_term = "surprisal_z"

    baseline_model = lmer(baseline_formula, data=train_data)
    baseline_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
    base_loglik, _, _, base_ok = _fit_stats(baseline_model)

    full_model = lmer(full_formula, data=train_data)
    full_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
    loglik, aic, bic, converged = _fit_stats(full_model)
    beta, se, t_stat, p_value = _extract_fixed_effect(full_model, beta_term)

    prev_beta: float | None = None
    if include_prev_surprisal:
        try:
            prev_beta, _, _, _ = _extract_fixed_effect(full_model, "prev_surprisal_z")
        except KeyError:
            prev_beta = np.nan

    delta_loglik_train: float | None = None
    lrt_p: float | None = None
    if compare_baseline:
        try:
            cmp = compare(baseline_model, full_model, as_dataframe=True, test="LRT")
            if cmp.height >= 2 and "Pr(>Chisq)" in cmp.columns:
                lrt_p = float(cmp["Pr(>Chisq)"][1])
            delta_loglik_train = loglik - base_loglik
        except Exception as exc:
            logger.warning("LRT comparison failed for %s: %s", predictor_col, exc)
            delta_loglik_train = loglik - base_loglik

    delta_loglik_test: float | None = None
    baseline_loglik_test: float | None = None
    loglik_test: float | None = None
    per_word: pd.DataFrame | None = None

    y_true = np.asarray(test_data[outcome], dtype=np.float64)
    try:
        # copy=True: np.asarray would be a zero-copy view into R's memory,
        # vulnerable to R's GC reclaiming it before this array is eventually
        # written to per_word (much later, after more R calls) -- see
        # _predict_or_nan's docstring for the full explanation.
        yhat_base = np.array(
            baseline_model.predict(test_data, allow_new_levels=True), dtype=np.float64, copy=True
        ).reshape(-1)
        yhat_full = np.array(
            full_model.predict(test_data, allow_new_levels=True), dtype=np.float64, copy=True
        ).reshape(-1)
        base_p, base_s = _corrs(y_true, yhat_base)
        full_p, full_s = _corrs(y_true, yhat_full)
    except Exception as exc:
        logger.warning("Prediction failed for OOS correlation (%s): %s", predictor_col, exc)
        yhat_base = np.full(test_data.height, np.nan, dtype=np.float64)
        yhat_full = np.full(test_data.height, np.nan, dtype=np.float64)
        base_p, base_s, full_p, full_s = np.nan, np.nan, np.nan, np.nan

    if out_of_sample and test_data.height >= 10:
        baseline_loglik_test, base_per_obs = _partition_loglik_detailed(
            baseline_model, test_data, outcome=outcome
        )
        loglik_test, full_per_obs = _partition_loglik_detailed(
            full_model, test_data, outcome=outcome
        )
        if np.isfinite(baseline_loglik_test) and np.isfinite(loglik_test):
            delta_loglik_test = loglik_test - baseline_loglik_test
        if save_per_word:
            id_cols = [c for c in ("subject", "sentid", "sentpos", "word") if c in test_data.columns]
            per_word = test_data.select(id_cols).to_pandas()
            per_word[outcome] = y_true
            per_word["yhat_baseline"] = yhat_base
            per_word["yhat_full"] = yhat_full
            per_word["baseline_loglik"] = base_per_obs
            per_word["full_loglik"] = full_per_obs
            per_word["delta_loglik"] = full_per_obs - base_per_obs

    primary_delta = delta_loglik_test if out_of_sample and delta_loglik_test is not None else delta_loglik_train

    return LmeFitResult(
        predictor=predictor_col,
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
        test_delta_pearsonr=(full_p - base_p) if np.isfinite(full_p) and np.isfinite(base_p) else np.nan,
        test_delta_spearmanr=(full_s - base_s) if np.isfinite(full_s) and np.isfinite(base_s) else np.nan,
        converged=converged and base_ok,
        n_obs=int(train_data.height),
        layer=layer,
        delta_loglik_train=delta_loglik_train,
        delta_loglik_test=delta_loglik_test,
        baseline_loglik_test=baseline_loglik_test,
        loglik_test=loglik_test,
        n_obs_test=int(test_data.height),
        prev_surprisal_beta=prev_beta,
        per_word=per_word,
    )
