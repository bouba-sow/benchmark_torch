"""Linear mixed-effects helpers for Natural Stories fMRI (pymer4 / lme4).

Extends Lin & Schuler (2025, arXiv:2506.11338)'s BOLD ~ HRF-convolved
surprisal/word-length/sentence-position formula with an optional ``freq_z``
(word frequency) nuisance term, for parity with every other pooled LME in
this codebase (N400, Fedorenko2016, Podcast, Tuckute all control for word
frequency; Lin & Schuler's original formula didn't). ``include_freq`` in
``fmri_lme_formulas``/``fit_fmri_lme`` switches between the two:

    include_freq=True (default):
      baseline: bold ~ wlen_z + freq_z + sentpos_z + (1 | subject) + (1 | story)
      full:     bold ~ wlen_z + freq_z + sentpos_z + surprisal_z +
                       (1 + surprisal_z | subject) + (1 | story)

    include_freq=False (drops freq_z from both — see Oh et al. 2024 on
    frequency as a known confound for surprisal's inverse-scaling effect):
      baseline: bold ~ wlen_z + sentpos_z + (1 | subject) + (1 | story)
      full:     bold ~ wlen_z + sentpos_z + surprisal_z +
                       (1 + surprisal_z | subject) + (1 | story)

RANDOM-EFFECTS STRUCTURE (changed 2026-08-27 -- see git history for the prior
version, Lin & Schuler's own correlated-multi-slope spec). The original
formula gave every nuisance covariate its own random slope on `subject`
*and* a random slope on `story` (only 8 story groups) -- four correlated
random slopes in total. Empirically this overfit train chunks without
generalizing: `pythia_checkpoints` sweep, `delta_loglik_train` was positive
for 99.6% of (model, fROI) fits (mean +36) while `delta_loglik_test` was
positive for only 28% (mean -6), and the two were essentially uncorrelated
(r=-0.06) -- the textbook signature of random-effects flexibility fitting
noise, not signal. LPP's pooled fMRI LME (`lpp_pooled_lme.py`) uses the
much more conservative pattern this now matches: exactly ONE random slope,
on the grouping factor with the most data (`subject`, 78 levels here, vs.
LPP's `region`, 10 levels), with every other grouping factor (`story`, only
8 levels -- too few to support its own slope) as a plain random intercept.
Verified directly before adopting: on 5 fROIs from one already-fit
checkpoint, this simplification turned delta_loglik_test positive for 2 of 3
LANG fROIs that were negative under the old formula (e.g. LangLIFG: -21.9 ->
+1.0) and shrank the remaining negative fROI's magnitude by >4x (MD network,
-37.0 -> -8.4 -- expected to stay small/negative since MD isn't strongly
surprisal-driven).

Deliberately NOT extended with a ``prev_surprisal`` spillover term (present
in the N400/Fedorenko2016 formulas): those datasets time-lock the DV to one
discrete word (e.g. a 300-500ms ERP window), where "current word" and
"previous word" are genuinely separable signals. Here, predictors are
HRF-convolved with a kernel that's effectively ~20s wide (Glover HRF), which
at natural narration speed (~3 words/sec) already blends dozens of
consecutive words' surprisal into one TR's convolved value — convolving a
one-word-shifted copy of surprisal separately would produce a regressor
almost perfectly collinear with ``surprisal_z`` post-convolution, hurting
convergence rather than adding separable signal.

Unlike the Natural Stories SPR LME (``regressors/lme.py``), which splits
train/test by a ``(subject + sentid) % mod`` hash (fine for roughly
independent per-word trials), BOLD is strongly autocorrelated within a story.
So here the split unit is a contiguous ~30 s TR chunk (never split across
partitions), shuffled and assigned to train/held-out by fraction — matching
Shain et al. (2020) / Lin & Schuler's original chunked fit/held-out design.
Both ``subject`` and ``story`` levels appear in both partitions (only
time-chunks are held out), so held-out predictions use the model's actual
fitted subject/story slopes rather than a new-level fallback.

Reuses the outcome-agnostic low-level helpers from ``regressors/lme.py``
(``_fit_stats``, ``_extract_fixed_effect``, ``_partition_loglik_detailed``,
``_corrs``, ``_predict_or_nan``, ``zscore_series``) rather than duplicating them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl

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

_REQUIRED_COLS = {"subject", "story", "chunk", "surprisal", "wlen", "freq", "sentpos", "bold"}


def prepare_fmri_lme_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Build a polars frame for the fMRI LME: factor columns + z-scored predictors.

    ``surprisal``/``wlen``/``freq``/``sentpos`` are expected to already be
    HRF-convolved (i.e. one value per TR, not per word) — z-scoring happens
    here, after convolution, matching Lin & Schuler's "convolved with the
    HRF ... then centered, scaled" order.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for fMRI LME: {sorted(missing)}")

    out = df.copy()
    out["subject"] = out["subject"].astype(str)
    out["story"] = out["story"].astype(str)
    out["chunk"] = out["chunk"].astype(str)
    out["surprisal_z"] = zscore_series(out["surprisal"])
    out["wlen_z"] = zscore_series(out["wlen"])
    out["freq_z"] = zscore_series(out["freq"])
    out["sentpos_z"] = zscore_series(out["sentpos"])
    out["bold"] = out["bold"].astype(float)

    cols = ["subject", "story", "chunk", "surprisal_z", "wlen_z", "freq_z", "sentpos_z", "bold"]
    return pl.from_pandas(out[cols].copy())


def fmri_lme_formulas(include_freq: bool = True) -> tuple[str, str]:
    """Natural Stories fMRI baseline / full formulas.

    ``include_freq=True`` (default) is the freq_z-controlled extension of
    Lin & Schuler (2025). ``include_freq=False`` drops ``freq_z`` from BOTH
    baseline and full formulas — freq_z must be removed from both models,
    not just one, or baseline is no longer nested inside full and
    delta_loglik no longer isolates surprisal's marginal contribution.

    Random effects: ONE random slope only (surprisal_z on subject, 78
    levels) — see module docstring for why (overfitting fix, 2026-08-27).
    story (8 levels) is a plain random intercept, matching LPP's convention
    of restricting the one random slope to whichever grouping factor has
    enough levels to support it.
    """
    nuisances = "wlen_z + freq_z" if include_freq else "wlen_z"
    baseline = f"bold ~ {nuisances} + sentpos_z + (1 | subject) + (1 | story)"
    full = (
        f"bold ~ {nuisances} + sentpos_z + surprisal_z + "
        f"(1 + surprisal_z | subject) + (1 | story)"
    )
    return baseline, full


def split_train_test_chunks(
    data: pl.DataFrame,
    *,
    train_frac: float = 0.75,
    random_state: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split by contiguous chunk id — a chunk is never split across partitions.

    Chunks are shuffled (fixed seed) before assignment so train/held-out both
    sample the full temporal range of every story, rather than e.g. all of
    train being the first half of each recording.
    """
    # sorted(...) — polars' .unique() gives no ordering guarantee (unlike
    # pandas' pd.unique(), which is first-appearance order), and every lag
    # here runs in its own freshly-spawned process (ProcessPoolExecutor(...,
    # max_tasks_per_child=1), to dodge an rpy2/lme4 memory leak). Rust's
    # hash-flood-protection seed is randomized per process, so .unique()'s
    # output order was effectively arbitrary per lag — making the
    # rng.shuffle() below depend on that arbitrary order, not just
    # random_state, so the train/test split silently reshuffled almost
    # independently at every lag instead of staying stable. Sorting first
    # makes the split a pure function of random_state again, as the
    # rng.shuffle() call already implies it should be.
    chunks = sorted(data["chunk"].unique().to_list())
    rng = np.random.default_rng(random_state)
    rng.shuffle(chunks)
    n_train = max(1, int(round(len(chunks) * train_frac)))
    train_chunks = set(chunks[:n_train])

    pdf = data.to_pandas()
    train_mask = pdf["chunk"].isin(train_chunks)
    train_df = pl.from_pandas(pdf.loc[train_mask].reset_index(drop=True))
    test_df = pl.from_pandas(pdf.loc[~train_mask].reset_index(drop=True))
    return train_df, test_df


def fit_fmri_lme(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.75,
    random_state: int = 0,
    verbose: bool = False,
    include_freq: bool = True,
    save_per_word: bool = False,
) -> LmeFitResult:
    """Fit BOLD ~ HRF-convolved surprisal LME (see module docstring for the formula).

    ``df`` must have one row per (subject, story, TR) with columns:
    ``subject, story, chunk, surprisal, wlen, freq, sentpos, bold`` — the
    predictors already HRF-convolved (raw, not yet z-scored). ``freq`` is
    still required even when ``include_freq=False`` (``prepare_fmri_lme_frame``
    always computes ``freq_z``); it's simply left out of both formulas.
    """
    from pymer4.models import compare, lmer

    data = prepare_fmri_lme_frame(df)
    train_data, test_data = split_train_test_chunks(
        data, train_frac=train_frac, random_state=random_state,
    )

    baseline_formula, full_formula = fmri_lme_formulas(include_freq=include_freq)
    outcome = "bold"
    beta_term = "surprisal_z"

    baseline_model = lmer(baseline_formula, data=train_data)
    baseline_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
    base_loglik, _, _, base_ok = _fit_stats(baseline_model)

    full_model = lmer(full_formula, data=train_data)
    full_model.fit(verbose=verbose, summary=False, conf_method="satterthwaite")
    loglik, aic, bic, converged = _fit_stats(full_model)
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
        id_cols = [c for c in ("subject", "story", "chunk") if c in test_data.columns]
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
        per_word=per_word,
    )
