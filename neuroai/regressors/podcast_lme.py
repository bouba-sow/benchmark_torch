"""Linear mixed-effects helpers for Podcast ECoG, pooled across subjects and
electrodes, fit independently per lag (Design B from the surprisal-per-lag
discussion).

Unlike the per-electrode ridge encoder (``benchmarks/podcast.py``), which
can't pool subjects because electrode identity isn't comparable across
subjects (variable counts, different anatomical coverage), a NESTED random
effect resolves this cleanly: electrode is nested *within* subject
(``subject:electrode``), so no correspondence between e.g. electrode 12 of
subject A and electrode 12 of subject B is ever assumed — each
``(subject, electrode)`` pair gets its own random intercept/slope.

    baseline: y ~ wlen_z + freq_z +
                  (1 | subject) + (1 | subject:electrode)
    full:     y ~ wlen_z + freq_z + surprisal_z +
                  (1 | subject) + (1 + surprisal_z | subject:electrode)

Fit once per lag (the outcome ``y`` is the ECoG signal at that lag; the
predictors are identical across lags). Train/held-out split is by contiguous
word-index chunk (same mechanism as ``regressors/fmri_lme.py``'s TR-chunk
split, reused directly here) — words are inherently sequential in one
continuous podcast, so this is the direct word-index analogue of that
TR-index chunking.

No pre-onset baseline correction: the reference Hasson-lab tutorial for this
dataset (``hassonlab.github.io/podcast-ecog-tutorials``) epochs with
``baseline=None`` and instead z-scores the outcome with a train-fold-fit
``StandardScaler`` before ridge fitting. ``fit_podcast_lme_one_lag`` follows
the same approach: ``y`` is z-scored per ``(subject, electrode)`` group using
TRAIN-partition statistics only (fit after ``split_train_test_chunks``, then
applied to both train and test — never fit on data the held-out evaluation
later scores against, mirroring their ``scaler.fit_transform(Y_train)`` /
``scaler.transform(Y_test)``). Per-group (not global) because different
electrodes have wildly different high-gamma power scales that a single
pooled z-score wouldn't remove, and the LME's ``(1 | subject:electrode)``
random intercept is additive-only — it can't fix a scale difference.

Reuses the outcome-agnostic low-level helpers from ``regressors/lme.py``
(``_fit_stats``, ``_extract_fixed_effect``, ``_partition_loglik_detailed``,
``_corrs``, ``_predict_or_nan``, ``zscore_series``) and the chunk-split logic
from ``regressors/fmri_lme.py`` rather than duplicating them.
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

_REQUIRED_COLS = {"subject", "electrode", "chunk", "word_idx", "wlen", "freq", "surprisal", "y"}


def prepare_podcast_lme_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Factor columns + z-scored predictors for the pooled Podcast LME.

    ``electrode`` values need not be globally unique across subjects — lme4's
    ``subject:electrode`` interaction term handles the nesting regardless of
    whether names repeat between subjects. ``word_idx`` (position in the
    transcript, shared identity across every subject/electrode row of the
    same word) is carried through only for per-word prediction export
    (``fit_podcast_lme_one_lag(..., save_per_word=True)``) — never used as a
    predictor or grouping factor in the formula itself.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Podcast LME: {sorted(missing)}")

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["electrode"] = out["electrode"].astype(str)
    out["chunk"] = out["chunk"].astype(str)
    out["word_idx"] = out["word_idx"].astype(np.int64)
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["y"] = out["y"].astype(float)

    cols = ["subject", "electrode", "chunk", "word_idx", "wlen_z", "freq_z", "surprisal_z", "y"]
    return pl.from_pandas(out[cols].copy())


def _zscore_y_by_group_train_test(
    train_data: pl.DataFrame, test_data: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Z-score ``y`` per ``(subject, electrode)`` group, TRAIN statistics only.

    Replaces the pre-onset baseline correction this pipeline no longer
    applies (see module docstring) — the LME analogue of the reference
    tutorial's ``StandardScaler().fit_transform(Y_train)`` /
    ``.transform(Y_test)``. Group mean/std are computed from ``train_data``
    only and applied to both partitions, so no test-fold information leaks
    into the normalization. Zero-variance train groups (e.g. a near-constant
    electrode, or too few train words in that group) get ``std=NaN`` rather
    than a division by zero; the resulting NaN rows are dropped by lmer's
    default listwise deletion, same as any other non-finite predictor row
    elsewhere in this pipeline.
    """
    train_pdf = train_data.to_pandas()
    test_pdf = test_data.to_pandas()

    group = train_pdf["subject"] + ":" + train_pdf["electrode"]
    stats = train_pdf.groupby(group)["y"].agg(["mean", "std"])
    stats.loc[stats["std"] == 0, "std"] = np.nan

    def _apply(pdf: pd.DataFrame) -> pd.DataFrame:
        key = pdf["subject"] + ":" + pdf["electrode"]
        mu = key.map(stats["mean"])
        sigma = key.map(stats["std"])
        out = pdf.copy()
        out["y"] = (pdf["y"] - mu) / sigma
        return out

    return pl.from_pandas(_apply(train_pdf)), pl.from_pandas(_apply(test_pdf))


def podcast_lme_formulas() -> tuple[str, str]:
    """Nested subject/subject:electrode formulas for one lag."""
    baseline = "y ~ wlen_z + freq_z + (1 | subject) + (1 | subject:electrode)"
    full = (
        "y ~ wlen_z + freq_z + surprisal_z + "
        "(1 | subject) + (1 + surprisal_z | subject:electrode)"
    )
    return baseline, full


def fit_podcast_lme_one_lag(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    allow_slope_fallback: bool = True,
    save_per_word: bool = False,
) -> LmeFitResult:
    """Fit the pooled Podcast LME for one lag.

    ``df`` must have one row per (subject, electrode, word) with columns
    ``subject, electrode, chunk, wlen, freq, surprisal, y`` — ``y`` is the
    ECoG signal at this specific lag.

    If the full model with the random slope on ``subject:electrode`` fails to
    converge (the richer covariance structure is the most likely failure
    point — see Barr et al. 2013's "keep it maximal, simplify if needed"),
    and ``allow_slope_fallback=True``, retries without the correlation term
    and finally without the random slope entirely, always applying the same
    fallback ladder to both models being compared so the comparison stays
    valid across every LM in the sweep.

    ``save_per_word=True`` populates the returned result's ``per_word``
    DataFrame — one row per HELD-OUT test-set observation (subject,
    electrode, word_idx, chunk, y_true, yhat_base, yhat_full,
    baseline_loglik, full_loglik, delta_loglik) — same convention as
    ``fit_spr_lme(..., save_per_word=True)``, for DLT-difficulty-style
    post-hoc analysis. Train-partition rows are never included: in-sample
    predictions would be optimistic/overfit, so only the genuinely held-out
    25% (``train_frac``) is exported, same scope the aggregate held-out
    correlation/delta-loglik metrics already use.
    """
    from pymer4.models import compare, lmer

    data = prepare_podcast_lme_frame(df)
    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )
    train_data, test_data = _zscore_y_by_group_train_test(train_data, test_data)

    n_train_before = train_data.height
    train_data = train_data.filter(pl.col("y").is_finite())
    if train_data.height < n_train_before:
        logger.warning(
            "Dropped %d/%d train row(s) with non-finite z-scored y "
            "(zero-variance subject:electrode group in this train split)",
            n_train_before - train_data.height, n_train_before,
        )
    n_test_before = test_data.height
    test_data = test_data.filter(pl.col("y").is_finite())
    if test_data.height < n_test_before:
        logger.warning(
            "Dropped %d/%d test row(s) with non-finite z-scored y "
            "(zero-variance subject:electrode group in this train split)",
            n_test_before - test_data.height, n_test_before,
        )

    baseline_formula, full_formula = podcast_lme_formulas()
    outcome = "y"
    beta_term = "surprisal_z"

    def _fit(formula: str):
        m = lmer(formula, data=train_data)
        # Wald instead of Satterthwaite: the primary metrics here
        # (loglik/AIC/BIC, delta_loglik, LRT p-value, held-out correlations)
        # come from the REML fit / LRT comparison directly and never touch
        # conf_method; only the secondary beta/se/t_stat/p_value diagnostics
        # do. Satterthwaite's per-coefficient df approximation is the known
        # bottleneck for models with many random-effect groups (285 nested
        # subject:electrode groups here) and buys nothing at this sample size
        # (~1.4M rows) — Wald's asymptotic SE is already exact in that
        # regime. Empirically verified (single-lag A/B + same-conf_method
        # repeat control, see outputs_jeanzay/_scratch_conf_method_test/
        # conf_method_timing_results.csv): ~25% faster per lag (wald
        # ~298s vs satterthwaite ~396s mean), with no beta/loglik difference
        # beyond ordinary fit-to-fit REML optimizer noise (~4e-5 on beta
        # between two independent same-conf_method fits of the same lag).
        m.fit(verbose=verbose, summary=False, conf_method="wald")
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
            "Podcast LME: full/baseline model failed to converge — retrying "
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
                "Podcast LME: still not converged — dropping the random "
                "slope on subject:electrode entirely (intercept-only)."
            )
            baseline_formula = "y ~ wlen_z + freq_z + (1 | subject) + (1 | subject:electrode)"
            full_formula = (
                "y ~ wlen_z + freq_z + surprisal_z + "
                "(1 | subject) + (1 | subject:electrode)"
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
        id_cols = [c for c in ("subject", "electrode", "word_idx", "chunk") if c in test_data.columns]
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
