"""Linear mixed-effects helpers for MEG-MASC (Gwilliams et al. 2023) MEG,
fit SEPARATELY PER SUBJECT (then averaged across subjects), independently per
LM layer, region-averaged (per-layer logit-lens surprisal, not embeddings).

MEG-MASC is a continuous-audiobook-listening dataset, so the same per-subject
design as ``broderick_lme`` applies — see those module
docstrings for the full rationale (a pooled LME with a correlated ``subject``
random slope never finished a single layer within a 1-4 h walltime; the fix
is one small LME per subject with ``region`` the only grouping factor, then
average the per-subject estimates). With n = 1 subject per fit a between-
subject variance is not identifiable, so there is no ``subject`` term.

Default formula (MEG-appropriate — see ``docs/megmasc_design.md`` §4):

    nuisances = worddur_z + wlen_z + freq_z + prev_worddur_z
    baseline: y ~ nuisances + (1 | region)
    full:     y ~ nuisances + surprisal_z + prev_surprisal_z + (1 + surprisal_z | region)

``worddur_z`` / ``prev_worddur_z`` (acoustic word duration and its one-word
lag) are added to the ``wlen_z + freq_z`` set the two EEG datasets use,
because MEG-MASC is *spoken*: the post-onset amplitude is strongly acoustic
and ``wlen_z`` (characters) is a weak proxy. Pass ``nuisances="wlen_z +
freq_z"`` for a parity run directly comparable to Broderick.

``region`` is one of six coarse KIT sensor-position groups
(``L|R`` × ``A|C|P``) from ``datasets/megmasc.py::megmasc_sensor_region`` —
a sensor cluster, not an anatomical ROI.

Barr et al. (2013) fallback ladder (identical to Broderick): drop
the ``region`` slope-intercept correlation, then drop the ``region`` random
slope entirely. Train/held-out split reuses ``fmri_lme.split_train_test_chunks``
with ``chunk = story`` (one of the four MASC stories).
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
    zscore_series,
)

logger = logging.getLogger(__name__)

DEFAULT_NUISANCES = "worddur_z + wlen_z + freq_z + prev_worddur_z"
PARITY_NUISANCES = "wlen_z + freq_z"

_REQUIRED_COLS = {
    "region", "chunk", "word_idx", "wlen", "freq", "worddur", "prev_worddur",
    "prev_surprisal", "surprisal", "y",
}
_REQUIRED_COLS_POOLED = _REQUIRED_COLS | {"subject"}


def megmasc_lme_formulas(nuisances: str = DEFAULT_NUISANCES) -> tuple[str, str]:
    """Baseline / full formulas for the per-subject MEG-MASC LME (one LM layer).

    No ``subject`` term — fit separately per subject, so ``region`` is the
    only random-effect grouping factor.
    """
    baseline = f"y ~ {nuisances} + (1 | region)"
    full = f"y ~ {nuisances} + surprisal_z + prev_surprisal_z + (1 + surprisal_z | region)"
    return baseline, full


def prepare_megmasc_lme_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Factor columns + z-scored predictors for the per-subject MEG-MASC LME.

    ``df`` must already be restricted to ONE subject's rows.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for MEG-MASC per-subject LME: {sorted(missing)}")

    out = df.copy()
    out["region"] = out["region"].astype(str)
    out["chunk"] = out["chunk"].astype(str)
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["worddur_z"] = zscore_series(out["worddur"])
    out["prev_worddur_z"] = zscore_series(out["prev_worddur"])
    out["prev_surprisal_z"] = zscore_series(out["prev_surprisal"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["y"] = out["y"].astype(float)

    cols = [
        "region", "chunk", "word_idx",
        "wlen_z", "freq_z", "worddur_z", "prev_worddur_z",
        "prev_surprisal_z", "surprisal_z", "y",
    ]
    return pl.from_pandas(out[cols].copy())


def megmasc_lme_pooled_formulas(nuisances: str = DEFAULT_NUISANCES) -> tuple[str, str]:
    """Baseline / full formulas for the POOLED (all-subjects, one fit per
    layer) MEG-MASC LME.

    ``region`` is the same fixed six-way sensor grouping for every subject
    (unlike per-subject anatomy in an ROI-based design), so pooling subjects
    together, crossed with ``region``, is statistically well-posed.

    ``subject`` gets a random **intercept only** (not a random slope on
    ``surprisal_z``): pymer4's ``predict`` on held-out rows raises
    "Invalid grouping factor specification, subject" for a random-*slope*
    grouping factor, which would kill the primary ``delta_loglik_test``
    metric — and this matches the working pooled pattern of the other
    pooled LMEs in this codebase (``n400_pooled_lme`` etc., which also use
    ``(1 | subject)``). ``region`` keeps its random slope on ``surprisal_z``
    — that is the scientifically informative one (different sensor clusters
    express different surprisal sensitivity).
    """
    return (
        f"y ~ {nuisances} + (1 | subject) + (1 | region)",
        f"y ~ {nuisances} + surprisal_z + prev_surprisal_z + "
        f"(1 | subject) + (1 + surprisal_z | region)",
    )


def prepare_megmasc_lme_pooled_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Factor columns + z-scored predictors for the POOLED MEG-MASC LME.
    ``df`` spans ALL subjects (must include a ``subject`` column)."""
    missing = _REQUIRED_COLS_POOLED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for pooled MEG-MASC LME: {sorted(missing)}")

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["region"] = out["region"].astype(str)
    out["chunk"] = out["chunk"].astype(str)
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["worddur_z"] = zscore_series(out["worddur"])
    out["prev_worddur_z"] = zscore_series(out["prev_worddur"])
    out["prev_surprisal_z"] = zscore_series(out["prev_surprisal"])
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["y"] = out["y"].astype(float)

    cols = [
        "subject", "region", "chunk", "word_idx",
        "wlen_z", "freq_z", "worddur_z", "prev_worddur_z",
        "prev_surprisal_z", "surprisal_z", "y",
    ]
    return pl.from_pandas(out[cols].copy())


def fit_megmasc_lme_pooled_one_layer(
    df: pd.DataFrame,
    *,
    nuisances: str = DEFAULT_NUISANCES,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    allow_slope_fallback: bool = True,
    save_per_word: bool = False,
) -> LmeFitResult:
    """Fit ONE pooled LME (all subjects, all six regions) for one LM layer.

    ``df`` must have one row per (subject, region, word), with columns
    ``subject, region, chunk, word_idx, wlen, freq, worddur, prev_worddur,
    prev_surprisal, surprisal, y``.

    Fallback ladder (Barr et al. 2013-style), applied identically to
    baseline and full so the LRT stays valid — 4 steps since there are now
    two crossed grouping factors:
      1. full formula, correlated slopes on both subject and region
      2. drop both slope-intercept correlations (``||``)
      3. drop the region random slope (subject keeps its slope)
      4. drop the subject random slope too (both grouping factors intercept-only)
    """
    from pymer4.models import compare, lmer

    data = prepare_megmasc_lme_pooled_frame(df)
    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )
    baseline_formula, full_formula = megmasc_lme_pooled_formulas(nuisances)
    outcome, beta_term = "y", "surprisal_z"

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
        logger.warning("Pooled MEG-MASC LME: step 1 failed to converge — "
                        "dropping the region slope-intercept correlation.")
        step2 = full_formula.replace(
            "(1 + surprisal_z | region)", "(1 + surprisal_z || region)"
        )
        full_model = _fit(step2)
        converged = _converged(full_model)
        if not (base_ok and converged):
            logger.warning("Pooled MEG-MASC LME: step 2 failed — dropping the region random slope too.")
            baseline_formula = f"y ~ {nuisances} + (1 | subject) + (1 | region)"
            full_formula = (
                f"y ~ {nuisances} + surprisal_z + prev_surprisal_z + "
                f"(1 | subject) + (1 | region)"
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
            id_cols = [c for c in ("subject", "region", "chunk", "word_idx") if c in test_data.columns]
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

    primary_delta = delta_loglik_test if delta_loglik_test is not None else delta_loglik_train

    return LmeFitResult(
        predictor="surprisal",
        beta=beta, se=se, t_stat=t_stat, p_value=p_value,
        loglik=loglik, aic=aic, bic=bic,
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


def fit_megmasc_lme_one_layer(
    df: pd.DataFrame,
    *,
    nuisances: str = DEFAULT_NUISANCES,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    allow_slope_fallback: bool = True,
    save_per_word: bool = False,
) -> LmeFitResult:
    """Fit the per-subject MEG-MASC LME for one LM layer.

    ``df`` must have one row per (region, word) for a SINGLE subject, with
    columns ``region, chunk, word_idx, wlen, freq, worddur, prev_worddur,
    prev_surprisal, surprisal, y`` — ``y`` is the region-averaged, 250-450 ms
    window-averaged MEG amplitude; ``surprisal`` is this layer's logit-lens
    surprisal; ``chunk`` is the story.
    """
    from pymer4.models import compare, lmer

    data = prepare_megmasc_lme_frame(df)
    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )

    baseline_formula, full_formula = megmasc_lme_formulas(nuisances)
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
            "MEG-MASC per-subject LME: full/baseline failed to converge — "
            "retrying without the region slope-intercept correlation."
        )
        full_model = _fit(
            full_formula.replace("(1 + surprisal_z | region)", "(1 + surprisal_z || region)")
        )
        converged = _converged(full_model)
        if not (base_ok and converged):
            logger.warning(
                "MEG-MASC per-subject LME: still not converged — dropping the "
                "region random slope on surprisal_z entirely."
            )
            baseline_formula = f"y ~ {nuisances} + (1 | region)"
            full_formula = (
                f"y ~ {nuisances} + surprisal_z + prev_surprisal_z + (1 | region)"
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
        beta=beta, se=se, t_stat=t_stat, p_value=p_value,
        loglik=loglik, aic=aic, bic=bic,
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
# Delta-log-likelihood noise ceiling
# ---------------------------------------------------------------------------
# Theoretical upper bound on the pooled MEG-MASC LME's delta_loglik_test:
# how much held-out log-likelihood any *word-level* regressor could add on
# top of the nuisance model, given the reliable word-locked MEG signal. The
# "oracle" predictor is the per-word MEAN amplitude measured on an
# independent set of subjects — the best any single per-word number could
# do. Its delta_loglik is what a perfect surprisal predictor would asymptote
# to. Two ways to form the independent set:
#
#   mode="split_half"  Timkey et al. (2026) — permute subjects into two
#                      halves; oracle for the target half is the source
#                      half's per-word mean. Matches
#                      scripts/ceilings/dll_n400.py.
#                      With only 11 subjects a 5-6-subject per-word mean is
#                      almost pure noise here (word-locked SNR is tiny:
#                      correlation ceiling SB r ~= 0.024), so this comes out
#                      at ~0 and is NOT a usable bound for MEG-MASC.
#
#   mode="loso"        leave-one-subject-out — oracle for each held-out
#                      subject is the mean of the OTHER 10. A 10-subject
#                      per-word mean is a much better estimate of the
#                      reliable signal, so this is the usable ceiling for a
#                      low-N low-SNR dataset. The target has one subject, so
#                      the (1|subject) term drops (formulas become the
#                      per-subject megmasc formulas minus prev_surprisal_z).
#
# The train/held-out story split and the nuisances match
# fit_megmasc_lme_pooled_one_layer so the ceiling is directly comparable to
# the sweep's per-model delta_loglik_test. The oracle has no "previous word"
# analogue, so prev_surprisal_z is dropped from both ceiling formulas (same
# choice as the N400 ceiling).

_CEILING_REQUIRED_COLS = {
    "subject", "region", "chunk", "word_idx", "y",
    "wlen", "freq", "worddur", "prev_worddur",
}


@dataclass
class MegMascDeltaLoglikCeilingResult:
    """Theoretical upper bound for the pooled MEG-MASC LME's delta
    log-likelihood (per held-out observation)."""

    ceiling_delta_loglik_per_obs: float
    ceiling_delta_loglik_per_obs_sd: float
    n_iterations_used: int
    n_iterations_requested: int
    mode: str = "split_half"
    per_iteration_delta_loglik: list[float] = field(default_factory=list)
    per_iteration_n_obs: list[int] = field(default_factory=list)
    per_direction_records: list[dict] = field(default_factory=list)


def megmasc_ceiling_formulas(
    nuisances: str = DEFAULT_NUISANCES, *, with_subject: bool = True,
) -> tuple[str, str]:
    """Baseline / full formulas for the MEG-MASC delta-LL ceiling — the
    pooled formulas minus ``prev_surprisal_z`` (the oracle is a single
    per-word value with no lagged counterpart). ``with_subject=False`` (the
    LOSO target has one subject) drops the ``(1 | subject)`` term."""
    subj = " + (1 | subject)" if with_subject else ""
    return (
        f"y ~ {nuisances}{subj} + (1 | region)",
        f"y ~ {nuisances} + surprisal_z{subj} + (1 + surprisal_z | region)",
    )


def _megmasc_ceiling_direction_task(
    df: pd.DataFrame, source_ids: set, target_ids: set,
) -> pd.DataFrame | None:
    """Target rows with the source subjects' per-word mean amplitude merged
    in as a column literally named ``surprisal`` (so
    ``prepare_megmasc_lme_pooled_frame`` z-scores it into ``surprisal_z``)."""
    src = df[df["subject"].astype(str).isin(source_ids)]
    tgt = df[df["subject"].astype(str).isin(target_ids)].copy()
    oracle = src.groupby("word_idx")["y"].mean().rename("surprisal").reset_index()
    tgt = tgt.drop(columns=["surprisal"], errors="ignore").merge(
        oracle, on="word_idx", how="left"
    )
    tgt = tgt.dropna(subset=["surprisal"])
    # prepare_megmasc_lme_pooled_frame requires a prev_surprisal column even
    # though megmasc_ceiling_formulas never references prev_surprisal_z; a
    # constant z-scores to all-zero and is simply unused.
    tgt["prev_surprisal"] = 0.0
    return tgt if not tgt.empty else None


def _megmasc_ceiling_direction_worker(
    iteration: int,
    direction: int,
    tgt: pd.DataFrame,
    *,
    nuisances: str,
    train_frac: float,
    random_state: int,
    verbose: bool,
    with_subject: bool = True,
) -> tuple[int, int, float | None, int, bool]:
    """Process-pool worker: fit one (iteration, direction) ceiling model.
    Module-level so it pickles for spawn/fork. Returns
    (iteration, direction, delta_loglik or None, n_obs_test, converged)."""
    from pymer4.models import lmer

    try:
        data = prepare_megmasc_lme_pooled_frame(tgt)
        train_data, test_data = split_train_test_chunks(
            data, train_frac=train_frac, random_state=random_state,
        )
        if test_data.height < 10:
            return iteration, direction, None, 0, False

        baseline_formula, full_formula = megmasc_ceiling_formulas(
            nuisances, with_subject=with_subject,
        )

        baseline_model = lmer(baseline_formula, data=train_data)
        baseline_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, base_ok = _fit_stats(baseline_model)

        full_model = lmer(full_formula, data=train_data)
        full_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
        _, _, _, full_ok = _fit_stats(full_model)

        base_ll, _ = _partition_loglik_detailed(baseline_model, test_data, outcome="y")
        full_ll, _ = _partition_loglik_detailed(full_model, test_data, outcome="y")
    except Exception as exc:  # noqa: BLE001 — a bad split must not kill the sweep
        logger.warning(
            "MEG-MASC ceiling fit failed on split %d direction %d: %s",
            iteration, direction, exc,
        )
        return iteration, direction, None, 0, False

    converged = bool(base_ok and full_ok)
    delta = (
        full_ll - base_ll
        if np.isfinite(base_ll) and np.isfinite(full_ll)
        else None
    )
    ok = delta is not None and np.isfinite(delta) and converged
    return iteration, direction, (delta if ok else None), int(test_data.height), converged


def compute_megmasc_ceiling_delta_loglik(
    df: pd.DataFrame,
    *,
    mode: str = "split_half",
    n_splits: int = 15,
    train_frac: float = 0.75,
    seed: int = 0,
    nuisances: str = DEFAULT_NUISANCES,
    n_workers: int = 1,
    verbose: bool = False,
) -> MegMascDeltaLoglikCeilingResult:
    """Delta-log-likelihood ceiling for the pooled MEG-MASC LME.

    ``df`` is one model's ``megmasc_lme_pooled_table.parquet`` (the table is
    model-independent — MEG only). Columns required:
    ``subject, region, chunk, word_idx, y, wlen, freq, worddur, prev_worddur``.

    ``mode="split_half"`` (default) — ``n_splits`` random subject halves ×
    2 directions. ``mode="loso"`` — leave-one-subject-out (``n_splits`` is
    ignored; one fit per subject, oracle = mean of the other N-1).
    """
    missing = _CEILING_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns for MEG-MASC delta-LL ceiling: {sorted(missing)}"
        )
    if mode not in ("split_half", "loso"):
        raise ValueError(f"mode must be 'split_half' or 'loso', got {mode!r}")

    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(df["subject"].astype(str).unique()))
    if len(subjects) < 4:
        raise ValueError(
            f"Need >=4 subjects for a delta-LL ceiling, got {len(subjects)}"
        )

    with_subject = mode == "split_half"
    tasks: list[tuple[int, int, pd.DataFrame]] = []
    if mode == "loso":
        n_splits = len(subjects)
        for it, held_out in enumerate(subjects):
            source_ids = set(s for s in subjects.tolist() if s != held_out)
            tgt = _megmasc_ceiling_direction_task(df, source_ids, {held_out})
            if tgt is not None:
                tasks.append((it, 0, tgt))
    else:
        for it in range(n_splits):
            perm = rng.permutation(subjects)
            half = len(perm) // 2
            halves = (set(perm[:half].tolist()), set(perm[half:].tolist()))
            for direction, (source_ids, target_ids) in enumerate((halves, halves[::-1])):
                tgt = _megmasc_ceiling_direction_task(df, source_ids, target_ids)
                if tgt is not None:
                    tasks.append((it, direction, tgt))

    iter_delta: dict[int, float] = {}
    iter_nobs: dict[int, int] = {}
    iter_failed: set[int] = set()
    per_direction_records: list[dict] = []

    def _record(iteration: int, direction: int, delta: float | None,
                n_obs: int, converged: bool) -> None:
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
                    _megmasc_ceiling_direction_worker, it, direction, tgt,
                    nuisances=nuisances, train_frac=train_frac,
                    random_state=seed, verbose=verbose, with_subject=with_subject,
                ): (it, direction)
                for it, direction, tgt in tasks
            }
            for fut in as_completed(futures):
                iteration, direction, delta, n_obs, converged = fut.result()
                _record(iteration, direction, delta, n_obs, converged)
    else:
        for it, direction, tgt in tasks:
            iteration, direction, delta, n_obs, converged = _megmasc_ceiling_direction_worker(
                it, direction, tgt, nuisances=nuisances, train_frac=train_frac,
                random_state=seed, verbose=verbose, with_subject=with_subject,
            )
            _record(iteration, direction, delta, n_obs, converged)

    per_direction_records.sort(key=lambda r: (r["iteration"], r["direction"]))
    used = [i for i in sorted(iter_delta) if iter_nobs.get(i, 0) > 0]
    per_iter_delta = [iter_delta[i] for i in used]
    per_iter_nobs = [iter_nobs[i] for i in used]

    if not per_iter_delta:
        return MegMascDeltaLoglikCeilingResult(
            ceiling_delta_loglik_per_obs=float("nan"),
            ceiling_delta_loglik_per_obs_sd=float("nan"),
            n_iterations_used=0,
            n_iterations_requested=n_splits,
            mode=mode,
            per_direction_records=per_direction_records,
        )

    per_obs = [d / n for d, n in zip(per_iter_delta, per_iter_nobs)]
    return MegMascDeltaLoglikCeilingResult(
        ceiling_delta_loglik_per_obs=float(np.mean(per_obs)),
        ceiling_delta_loglik_per_obs_sd=(
            float(np.std(per_obs, ddof=1)) if len(per_obs) > 1 else float("nan")
        ),
        n_iterations_used=len(per_obs),
        n_iterations_requested=n_splits,
        mode=mode,
        per_iteration_delta_loglik=per_iter_delta,
        per_iteration_n_obs=per_iter_nobs,
        per_direction_records=per_direction_records,
    )
