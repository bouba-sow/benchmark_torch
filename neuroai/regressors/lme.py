"""Linear mixed-effects helpers for Natural Stories SPR (pymer4 / lme4).

Supports two analysis styles:

- ``clark2025``: Clark et al. (COLING 2025) — raw RT, unigram + spillover surprisal,
  subject-only random effects, train/test split, OOS metrics on test.
- ``modelblocks``: ModelBlocks ``spr.lmerform`` — log RT, word + sentid:subject RE.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

LmeStyle = Literal["clark2025", "modelblocks"]


@dataclass
class LmeFitResult:
    """Fixed-effect estimate for one predictor in an SPR LME."""

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
    # ``fit_spr_lme(..., save_per_word=True)``.
    per_word: pd.DataFrame | None = None
    # True if a richer random-effect term (e.g. a correlated random slope)
    # had to be dropped to reach convergence — set by callers that implement
    # a Barr et al. (2013)-style simplification ladder (e.g.
    # ``regressors.podcast_lme.fit_podcast_lme_one_lag``). False elsewhere.
    slope_dropped: bool = False


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


def _resolve_word_column(df: pd.DataFrame) -> pd.DataFrame:
    if "word" in df.columns:
        return df
    word_candidates = [c for c in ("word_x", "word_y") if c in df.columns]
    if not word_candidates:
        raise ValueError("Missing columns for LME: ['word']")
    df = df.copy()
    df["word"] = df[word_candidates[0]]
    for c in word_candidates[1:]:
        df["word"] = df["word"].where(df["word"].notna(), df[c])
    return df


def _filter_sentence_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sentence-initial and -final words (Clark SPR preprocessing)."""
    if "sentid" not in df.columns or "sentpos" not in df.columns:
        return df
    max_pos = df.groupby("sentid", observed=True)["sentpos"].transform("max")
    keep = (df["sentpos"] > 1) & (df["sentpos"] < max_pos)
    return df.loc[keep].reset_index(drop=True)


def prepare_spr_lme_frame(
    trials: pd.DataFrame,
    *,
    predictor_col: str,
    log_rt: bool = True,
    style: LmeStyle = "modelblocks",
    include_freq: bool = True,
    include_prev_surprisal: bool = True,
    filter_sentence_edges: bool = False,
    wordfreq_language: str = "en",
) -> pl.DataFrame:
    """Build a polars frame for SPR LME."""
    required = {"participant_id", "story_id", "zone", "rt", predictor_col}
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"Missing columns for LME: {sorted(missing)}")

    df = _resolve_word_column(trials.copy())
    df = df.dropna(subset=["rt", predictor_col, "zone", "story_id", "participant_id"])
    if filter_sentence_edges:
        df = _filter_sentence_edges(df)

    df["subject"] = df["participant_id"].astype(str)
    if "sentid" in df.columns:
        df["sentid"] = df["sentid"].astype(str)
    else:
        df["sentid"] = df["story_id"].astype(str)
    if "sentpos" in df.columns:
        df["sentpos"] = df["sentpos"].astype(int)
    else:
        df["sentpos"] = df["zone"].astype(int)
    df["wlen"] = df["word"].astype(str).str.len().astype(int)
    df["fdur"] = df["rt"].astype(float)
    use_log = log_rt and style != "clark2025"
    if use_log:
        df["log_fdur"] = np.log(df["fdur"].clip(lower=1.0))
    else:
        df["log_fdur"] = df["fdur"]

    df["sentpos_z"] = zscore_series(df["sentpos"])
    df["wlen_z"] = zscore_series(df["wlen"])
    pred_z_name = f"{predictor_col}_z"
    df[pred_z_name] = zscore_series(df[predictor_col])

    cols = [
        "subject",
        "word",
        "sentid",
        "sentpos",
        "wlen",
        "fdur",
        "log_fdur",
        "sentpos_z",
        "wlen_z",
        pred_z_name,
    ]

    if style == "clark2025":
        if include_freq:
            df["freq"] = word_zipf_freq(df["word"], language=wordfreq_language)
            df["freq_z"] = zscore_series(df["freq"])
            cols.append("freq_z")
        if include_prev_surprisal:
            prev_col = f"prev_{predictor_col}"
            if prev_col not in df.columns:
                sort_keys = ["story_id", "zone"]
                if "participant_id" in df.columns:
                    sort_keys = ["participant_id"] + sort_keys
                df = df.sort_values(sort_keys)
                df[prev_col] = df.groupby(
                    ["participant_id", "story_id"] if "participant_id" in df.columns else ["story_id"],
                    sort=False,
                )[predictor_col].shift(1)
            df = df.dropna(subset=[prev_col])
            df["prev_surprisal_z"] = zscore_series(df[prev_col])
            cols.append("prev_surprisal_z")

    return pl.from_pandas(df[cols].copy())


def spr_lme_formulas(
    predictor_z: str,
    *,
    main_random_slope: bool = False,
) -> tuple[str, str]:
    """ModelBlocks ``spr.lmerform`` baseline and full formulas (log RT)."""
    nuisances = "sentpos_z + wlen_z"
    if main_random_slope:
        re_subject = f"(1 + sentpos_z + wlen_z + {predictor_z} | subject)"
    else:
        re_subject = f"(1 + sentpos_z + wlen_z | subject)"
    random_other = "(1 | word) + (1 | sentid:subject)"
    baseline = f"log_fdur ~ {nuisances} + {re_subject} + {random_other}"
    full = f"log_fdur ~ {nuisances} + {predictor_z} + {re_subject} + {random_other}"
    return baseline, full


def clark_spr_lme_formulas(
    *,
    include_freq: bool = True,
    include_prev_surprisal: bool = True,
) -> tuple[str, str]:
    """Clark et al. (2025) Natural Stories SPR formulas (raw ``fdur``).

    ``freq_z`` (Zipf lexical frequency, ``wordfreq``) is the nuisance term —
    same convention as the neural LME datasets' ``n400_lme_formulas`` /
    ``fmri_lme_formulas`` (``wlen_z + freq_z``), replacing the earlier
    KenLM-based lexical-surprisal nuisance.
    """
    parts = ["sentpos_z", "wlen_z"]
    if include_freq:
        parts.append("freq_z")
    nuisances = " + ".join(parts)
    re_base = "(1 + sentpos_z + wlen_z | subject)"
    baseline = f"fdur ~ {nuisances} + {re_base}"

    lm_terms = list(parts)
    re_slopes = ["sentpos_z", "wlen_z"]
    if include_prev_surprisal:
        lm_terms.extend(["surprisal_z", "prev_surprisal_z"])
        re_slopes.extend(["surprisal_z", "prev_surprisal_z"])
    else:
        lm_terms.append("surprisal_z")
        re_slopes.append("surprisal_z")
    re_full = f"(1 + {' + '.join(re_slopes)} | subject)"
    full = f"fdur ~ {' + '.join(lm_terms)} + {re_full}"
    return baseline, full


def n400_lme_formulas(*, include_lab: bool = False) -> tuple[str, str]:
    """N400 EEG formulas: subject random slopes + a CROSSED electrode intercept.

    Unlike Podcast's ECoG (``podcast_lme_formulas``), where electrode identity
    is subject-specific and must be nested (``subject:electrode``), N400
    datasets here (Nieuwland, Szewczyk) use standard 10-20 scalp
    electrode names that refer to the same physical location across subjects
    — so electrode is a genuinely crossed random effect, ``(1 | electrode)``,
    shared across every subject rather than nested within one.

    ``include_lab=True`` adds ``(1 | lab)`` for multi-site studies (Nieuwland)
    where subject IDs are lab-prefixed and effectively nested within lab.

    ``prev_surprisal_z`` (spillover control) is standard now: absent from the
    baseline entirely, added to the full model alongside ``surprisal_z`` (both
    as fixed effects and as a joint random slope on ``subject``) — the LRT
    then tests their combined contribution, matching Natural Stories'
    ``clark_spr_lme_formulas`` rather than the old "shared nuisance in both
    models" treatment.
    """
    nuisances = "wlen_z + freq_z"
    re_subject_base = f"(1 + {nuisances} | subject)"
    re_subject_full = f"(1 + {nuisances} + surprisal_z + prev_surprisal_z | subject)"
    re_electrode = "(1 | electrode)"
    re_lab = " + (1 | lab)" if include_lab else ""

    baseline = f"n400 ~ {nuisances} + {re_subject_base} + {re_electrode}{re_lab}"
    full = f"n400 ~ {nuisances} + surprisal_z + prev_surprisal_z + {re_subject_full} + {re_electrode}{re_lab}"
    return baseline, full


def _extract_fixed_effect(model, term: str) -> tuple[float, float, float, float]:
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
    converged = True
    if getattr(model, "convergence_status", None) is not None:
        try:
            converged = bool(model.convergence_status[0])
        except (TypeError, IndexError, KeyError):
            converged = True
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
        return np.array(
            model.predict(data, allow_new_levels=True),
            dtype=np.float64, copy=True,
        ).reshape(-1)
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
    total = float(np.nansum(per_obs)) if np.isfinite(per_obs).sum() >= 3 else np.nan
    return total, per_obs


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
    test_indices: tuple[int, ...] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split on ``factor(subject) + factor(sentid)`` mod ``partition_mod``."""
    if partition_mod <= 1:
        return data, data

    split_id = _assign_split_ids(data, partition_mod=partition_mod)
    pdf = data.to_pandas()
    pdf["_split_id"] = split_id

    train_idx = np.asarray(sorted(set(int(i) for i in train_indices)), dtype=int)
    train_mask = np.isin(split_id, train_idx)
    if train_mask.sum() < 10:
        return data, data

    if test_indices is None:
        test_mask = ~train_mask
    else:
        test_idx = np.asarray(sorted(set(int(i) for i in test_indices)), dtype=int)
        test_mask = np.isin(split_id, test_idx)
    if test_mask.sum() < 10:
        train_df = pl.from_pandas(
            pdf.loc[train_mask].drop(columns="_split_id").reset_index(drop=True)
        )
        return train_df, data

    train_df = pl.from_pandas(
        pdf.loc[train_mask].drop(columns="_split_id").reset_index(drop=True)
    )
    test_df = pl.from_pandas(
        pdf.loc[test_mask].drop(columns="_split_id").reset_index(drop=True)
    )
    return train_df, test_df


def fit_spr_lme(
    trials: pd.DataFrame,
    *,
    predictor_col: str = "surprisal",
    style: LmeStyle = "clark2025",
    log_rt: bool = True,
    main_random_slope: bool = False,
    compare_baseline: bool = True,
    out_of_sample: bool = True,
    partition_mod: int = 4,
    partition_dev_indices: tuple[int, ...] = (0, 1),
    partition_fit_indices: tuple[int, ...] | None = None,
    partition_test_indices: tuple[int, ...] = (2, 3),
    partition_eval_indices: tuple[int, ...] | None = None,
    include_freq: bool = True,
    include_prev_surprisal: bool = True,
    filter_sentence_edges: bool = True,
    wordfreq_language: str = "en",
    verbose: bool = False,
    layer: int | None = None,
    save_per_word: bool = False,
) -> LmeFitResult:
    """Fit SPR LME for one scalar LM surprisal predictor."""
    from pymer4.models import compare, lmer

    pred_z = f"{predictor_col}_z"
    if style == "clark2025":
        log_rt = False
        filter_edges = filter_sentence_edges
    else:
        filter_edges = False

    data = prepare_spr_lme_frame(
        trials,
        predictor_col=predictor_col,
        log_rt=log_rt,
        style=style,
        include_freq=include_freq,
        include_prev_surprisal=include_prev_surprisal,
        filter_sentence_edges=filter_edges,
        wordfreq_language=wordfreq_language,
    )

    train_indices = (
        partition_fit_indices if partition_fit_indices is not None else partition_dev_indices
    )
    test_indices = partition_test_indices
    if partition_eval_indices is not None:
        test_indices = partition_eval_indices

    train_data = data
    test_data = data
    if out_of_sample:
        train_data, test_data = _split_train_test(
            data,
            partition_mod=partition_mod,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    if style == "clark2025":
        baseline_formula, full_formula = clark_spr_lme_formulas(
            include_freq=include_freq,
            include_prev_surprisal=include_prev_surprisal,
        )
        outcome = "fdur"
        beta_term = "surprisal_z"
    else:
        baseline_formula, full_formula = spr_lme_formulas(
            pred_z, main_random_slope=main_random_slope
        )
        outcome = "log_fdur"
        beta_term = pred_z

    baseline_model = lmer(baseline_formula, data=train_data)
    baseline_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
    base_loglik, _, _, base_ok = _fit_stats(baseline_model)

    full_model = lmer(full_formula, data=train_data)
    full_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
    loglik, aic, bic, converged = _fit_stats(full_model)
    beta, se, t_stat, p_value = _extract_fixed_effect(full_model, beta_term)

    prev_beta: float | None = None
    if style == "clark2025" and include_prev_surprisal:
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
        pred_kw = {"allow_new_levels": True} if style == "clark2025" else {}
        yhat_base = np.asarray(
            baseline_model.predict(test_data, **pred_kw),
            dtype=np.float64,
        ).reshape(-1)
        yhat_full = np.asarray(
            full_model.predict(test_data, **pred_kw),
            dtype=np.float64,
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

    primary_delta = (
        delta_loglik_test if out_of_sample and delta_loglik_test is not None else delta_loglik_train
    )

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


@dataclass
class DeltaLoglikCeilingResult:
    """Timkey et al. (2026, PNAS supplement) split-half theoretical upper
    bound for a reading-time LME's delta log-likelihood."""

    ceiling_delta_loglik_per_obs: float
    ceiling_delta_loglik_per_obs_sd: float
    n_iterations_used: int
    n_iterations_requested: int
    per_iteration_delta_loglik: list[float] = field(default_factory=list)
    per_iteration_n_obs: list[int] = field(default_factory=list)
    # Every individual (split, direction) fit, including failed/non-converged
    # ones (delta_loglik=None there) -- appendix-level detail, not just the
    # post-sum-over-directions per-split totals above.
    per_direction_records: list[dict] = field(default_factory=list)


def _ceiling_direction_task(
    trials: pd.DataFrame, source_ids: set, target_ids: set, item_cols: list[str],
) -> pd.DataFrame | None:
    """Build one direction's target-half trial frame with the split-half
    item-mean RT merged in as a column literally named ``"surprisal"``."""
    src = trials[trials["participant_id"].astype(str).isin(source_ids)]
    tgt = trials[trials["participant_id"].astype(str).isin(target_ids)].copy()
    item_mean = src.groupby(item_cols)["rt"].mean().rename("surprisal").reset_index()
    tgt = tgt.drop(columns=["surprisal"], errors="ignore").merge(item_mean, on=item_cols, how="left")
    tgt = tgt.dropna(subset=["surprisal"])
    return tgt if not tgt.empty else None


def _ceiling_direction_worker(
    iteration: int,
    direction: int,
    tgt: pd.DataFrame,
    *,
    style: LmeStyle,
    include_freq: bool,
    filter_sentence_edges: bool,
    verbose: bool,
    out_of_sample: bool = False,
    partition_mod: int = 4,
    partition_fit_indices: tuple[int, ...] = (0, 1),
    partition_test_indices: tuple[int, ...] = (2, 3),
) -> tuple[int, int, float | None, int, bool]:
    """Process-pool worker: fit one (iteration, direction)'s ceiling model.
    Module-level (not a closure) so it can be pickled for spawn/fork.
    Returns (iteration, direction, delta_loglik or None on failure, n_obs, converged).

    ``out_of_sample`` matches this to the SAME convention as the real models'
    headline ``delta_loglik_test`` (fit on a train partition, evaluated on a
    held-out partition of that same target half) rather than the in-sample
    fit-and-score-on-everything convention Timkey et al.'s supplement
    literally describes. In-sample ΔLL is known to be optimistic here
    specifically because the full model's random-slope term
    (``(1 + ... + surprisal_z + prev_surprisal_z | subject)``) is flexible
    enough to partly fit subject-level noise in the very data it's scored on
    -- which is exactly why this codebase's own real-model sweeps prefer
    ``delta_loglik_test`` over the legacy in-sample ``delta_loglik``. Mixing
    an in-sample ceiling with an out-of-sample real-model score would bias
    fraction-of-ceiling downward; matching conventions removes that bias.
    When ``out_of_sample=True``, uses ``delta_loglik_test``/``n_obs_test``
    (the held-out partition) rather than ``delta_loglik``/``n_obs`` (which
    would be the train-partition size under ``out_of_sample=True``)."""
    try:
        result = fit_spr_lme(
            tgt,
            predictor_col="surprisal",
            style=style,
            compare_baseline=True,
            out_of_sample=out_of_sample,
            partition_mod=partition_mod,
            partition_fit_indices=partition_fit_indices,
            partition_test_indices=partition_test_indices,
            include_freq=include_freq,
            include_prev_surprisal=False,
            filter_sentence_edges=filter_sentence_edges,
            verbose=verbose,
        )
    except Exception as exc:
        logger.warning("Delta-LL ceiling fit failed on split %d direction %d: %s", iteration, direction, exc)
        return iteration, direction, None, 0, False
    if out_of_sample:
        delta, n_obs = result.delta_loglik_test, result.n_obs_test
    else:
        delta, n_obs = result.delta_loglik, result.n_obs
    ok = delta is not None and np.isfinite(delta) and result.converged and n_obs >= 10
    return iteration, direction, (delta if ok else None), (n_obs or 0), bool(result.converged)


def compute_reading_time_ceiling_delta_loglik(
    trials: pd.DataFrame,
    *,
    style: LmeStyle = "clark2025",
    n_splits: int = 15,
    include_freq: bool = True,
    filter_sentence_edges: bool = True,
    seed: int = 0,
    verbose: bool = False,
    n_workers: int = 1,
    out_of_sample: bool = False,
    partition_mod: int = 4,
    partition_fit_indices: tuple[int, ...] = (0, 1),
    partition_test_indices: tuple[int, ...] = (2, 3),
) -> DeltaLoglikCeilingResult:
    """Split-half noise ceiling for a reading-time LME's delta log-likelihood
    (Timkey et al. 2026, PNAS supplement, Section A/B).

    For each of ``n_splits`` random equal splits of subjects into halves A/B:
    compute the per-item ``(story_id, zone)`` mean RT within one half, use it
    AS THE PREDICTOR (in place of LM surprisal) in a filler model fit to the
    OTHER half's raw trials, and take that model's in-sample delta
    log-likelihood (full vs baseline, same formula family as the real
    surprisal fit) -- then repeat in the opposite direction and sum both.
    Averaging the resulting per-observation delta-LL across splits gives the
    theoretical upper bound any single word-level predictor could achieve,
    given the intrinsic noise of the measure and of a single scalar per-word
    estimate (a stand-in for what a "perfect" surprisal would score).

    Reuses ``fit_spr_lme`` unmodified rather than re-implementing model
    fitting: ``clark_spr_lme_formulas`` hardcodes the fixed-effect term name
    ``surprisal_z`` regardless of ``predictor_col``, so the split-half
    item-mean is written into a column literally named ``"surprisal"``
    before calling ``fit_spr_lme`` -- required to hit the right formula term,
    not a shortcut. ``include_prev_surprisal=False`` throughout: the ceiling
    bounds a single scalar per-word predictor (surprisal itself), matching
    the paper's description of "a single word-level difficulty estimate";
    there is no natural spillover analogue for a split-half item mean.

    ``n_workers`` > 1 fits the ``2 * n_splits`` (iteration, direction) tasks
    in a process pool instead of sequentially: each is a fully independent
    lme4 fit (rpy2 embeds one R interpreter per process, so this must be
    process- not thread-based, same convention as
    ``natural_stories_lme.py``'s per-layer worker pool). A single direction's
    fit measured ~1786s (~30 min) in testing, so n_splits=15 (30 tasks) is
    ~15h sequential but only ~3 rounds (~90 min) at n_workers=10.

    ``out_of_sample`` (default False, matching the paper's literal in-sample
    description) switches each direction's fit to the SAME held-out
    train/test convention as the real models' headline ``delta_loglik_test``
    (see ``_ceiling_direction_worker`` for why the in-sample version is
    optimistic here and why matching conventions matters for a fair
    fraction-of-ceiling comparison).
    """
    # "word" is deliberately not required here: fit_spr_lme's own
    # _resolve_word_column already reconciles a word_x/word_y merge-suffix
    # split (which is what Natural Stories' raw trials actually carry, since
    # its RT table and word-list table both have a "word" column) -- an
    # extra check here would just duplicate that logic and risk disagreeing
    # with it.
    required = {"participant_id", "story_id", "zone", "rt"}
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"Missing columns for delta-LL ceiling: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(trials["participant_id"].astype(str).unique()))
    item_cols = ["story_id", "zone"]

    # Build every (iteration, direction) task frame upfront (cheap, no R
    # involved) so both the sequential and parallel paths share one task list.
    tasks: list[tuple[int, int, pd.DataFrame]] = []
    for it in range(n_splits):
        perm = rng.permutation(subjects)
        half = len(perm) // 2
        halves = (set(perm[:half].tolist()), set(perm[half:].tolist()))
        for direction, (source_ids, target_ids) in enumerate((halves, halves[::-1])):
            tgt = _ceiling_direction_task(trials, source_ids, target_ids, item_cols)
            if tgt is not None:
                tasks.append((it, direction, tgt))

    iter_delta: dict[int, float] = {}
    iter_nobs: dict[int, int] = {}
    iter_failed: set[int] = set()
    # Every individual (split, direction) fit, successes and failures alike --
    # appendix-level detail, kept independently of the headline aggregation.
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
                    _ceiling_direction_worker, it, direction, tgt,
                    style=style, include_freq=include_freq,
                    filter_sentence_edges=filter_sentence_edges, verbose=verbose,
                    out_of_sample=out_of_sample, partition_mod=partition_mod,
                    partition_fit_indices=partition_fit_indices,
                    partition_test_indices=partition_test_indices,
                ): (it, direction)
                for it, direction, tgt in tasks
            }
            for fut in as_completed(futures):
                iteration, direction, delta, n_obs, converged = fut.result()
                _record(iteration, direction, delta, n_obs, converged)
    else:
        for it, direction, tgt in tasks:
            iteration, direction, delta, n_obs, converged = _ceiling_direction_worker(
                it, direction, tgt, style=style, include_freq=include_freq,
                filter_sentence_edges=filter_sentence_edges, verbose=verbose,
                out_of_sample=out_of_sample, partition_mod=partition_mod,
                partition_fit_indices=partition_fit_indices,
                partition_test_indices=partition_test_indices,
            )
            _record(iteration, direction, delta, n_obs, converged)

    per_direction_records.sort(key=lambda r: (r["iteration"], r["direction"]))
    per_iter_delta = [iter_delta[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]
    per_iter_nobs = [iter_nobs[i] for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]

    if not per_iter_delta:
        return DeltaLoglikCeilingResult(
            ceiling_delta_loglik_per_obs=float("nan"),
            ceiling_delta_loglik_per_obs_sd=float("nan"),
            n_iterations_used=0,
            n_iterations_requested=n_splits,
            per_direction_records=per_direction_records,
        )

    per_obs = [d / n for d, n in zip(per_iter_delta, per_iter_nobs)]
    return DeltaLoglikCeilingResult(
        ceiling_delta_loglik_per_obs=float(np.mean(per_obs)),
        ceiling_delta_loglik_per_obs_sd=float(np.std(per_obs, ddof=1)) if len(per_obs) > 1 else float("nan"),
        n_iterations_used=len(per_obs),
        per_direction_records=per_direction_records,
        n_iterations_requested=n_splits,
        per_iteration_delta_loglik=per_iter_delta,
        per_iteration_n_obs=per_iter_nobs,
    )
