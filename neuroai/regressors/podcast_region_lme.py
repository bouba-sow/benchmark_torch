"""Podcast ECoG LME, ONE MODEL PER REGION on region-averaged electrodes.

Replaces the pooled subject:electrode model for the Podcast surprisal headline (decided
2026-10-08): fewer parameters (no electrode random effects, no random slope -> no
convergence ladder), ~5x cheaper, and the per-region fits are the ROI scores directly.

    baseline: y ~ wlen_z + freq_z + (1 | subject)
    full:     y ~ wlen_z + freq_z + surprisal_z + (1 | subject)

Data: left-hemisphere ``assigned`` electrodes of the language-responsive ROI map, regions
AntTemp / IFG / PostTemp (IFGorb merged into IFG: it has only 4 subjects and split-half fits
lost 2/10 splits; MFG, 3 electrodes in 2 subjects, is dropped by ``min_subjects``).

Standardisation, ALL statistics from TRAIN chunks only (applied to train and test alike, so
no held-out information enters the scaling):
  1. chunk split first (``split_train_test_chunks``, random_state = split seed -- the same
     split the pooled sweep used);
  2. y z-scored per (subject, electrode) with train-chunk mean/std;
  3. electrodes averaged within (subject, region, word);
  4. the average z-scored again per (subject, region) with train-chunk mean/std.
Predictors (wlen/freq/surprisal) are z-scored over the region frame, label-free, exactly as in
``prepare_podcast_lme_frame`` for the pooled sweep.

Train/test membership is carried in an ``is_train`` column from step 1 and reused by the fit,
never re-derived (a re-derived split could differ if a chunk were missing from a subset and
would then disagree with the chunks whose statistics scaled y).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .fmri_lme import split_train_test_chunks
from .lme import _corrs, _extract_fixed_effect, _fit_stats, _partition_loglik_detailed, _predict_or_nan
from .podcast_lme import prepare_podcast_lme_frame

logger = logging.getLogger(__name__)

REGION_MERGE = {"IFGorb": "IFG"}
N_LAGS = 32
BASELINE = "y ~ wlen_z + freq_z + (1 | subject)"
FULL = "y ~ wlen_z + freq_z + surprisal_z + (1 | subject)"

ROW_COLUMNS = (
    "region", "lag", "lag_time_s", "beta", "se", "t_stat", "p_value", "loglik", "aic", "bic",
    "baseline_loglik", "delta_loglik_train", "delta_loglik_test", "baseline_loglik_test",
    "loglik_test", "lrt_p_value", "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme", "test_delta_pearsonr",
    "test_delta_spearmanr", "converged", "slope_dropped", "n_obs", "n_obs_test", "error",
    "fit_wall_time_s",
)


def lag_times(n_lags: int = N_LAGS) -> np.ndarray:
    """Analysis time of each lag index (same grid as configs/surprisal/neural/podcast*.yaml)."""
    return np.linspace(-0.2, 0.8, n_lags, endpoint=False)


def lags_in_window(lo_s: float = 0.0, hi_s: float = 0.6, n_lags: int = N_LAGS) -> list[int]:
    t = lag_times(n_lags)
    return [int(k) for k in np.where((t >= lo_s) & (t <= hi_s))[0]]


def _train_stats_zscore(df: pd.DataFrame, keys: list[str], cols: list[str], train_mask: np.ndarray) -> pd.DataFrame:
    """z-score ``cols`` per ``keys`` group with TRAIN-row statistics; zero-variance groups -> NaN."""
    tr = df.loc[train_mask].groupby(keys)[cols]
    mean, std = tr.mean(), tr.std()
    std = std.where(std != 0)
    idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
    mu = mean.reindex(idx).to_numpy()
    sd = std.reindex(idx).to_numpy()
    return pd.DataFrame((df[cols].to_numpy() - mu) / sd, index=df.index, columns=cols)


def build_region_table(
    table_path: Path, y_path: Path, roi_map_path: Path, lags: list[int], *,
    seed: int = 0, train_frac: float = 0.75, min_subjects: int = 4,
) -> tuple[pd.DataFrame, list[str]]:
    """(subject, region, word) rows with standardised region-averaged ``y_<lag>`` columns.

    ``table_path``/``y_path`` are the cached pooled table / (rows, lags) signal of the real
    sweep (model-independent apart from surprisal, which is dropped here).
    """
    df = pd.read_parquet(table_path).drop(columns=["surprisal"], errors="ignore")
    y = np.load(y_path, mmap_mode="r")
    ycols = [f"y_{k}" for k in lags]
    for k, c in zip(lags, ycols):
        df[c] = np.asarray(y[:, k], dtype=np.float64)
    roi = pd.read_csv(roi_map_path)
    roi = roi[(roi["assigned"] == True) & (roi["hemisphere"] == "L")][["subject", "channel", "region"]]  # noqa: E712
    roi["region"] = roi["region"].replace(REGION_MERGE)
    df = df.merge(roi.rename(columns={"channel": "electrode"}), on=["subject", "electrode"], how="inner")

    chunks = pl.DataFrame({"chunk": sorted(df["chunk"].astype(str).unique().tolist())})
    train_chunks, _ = split_train_test_chunks(chunks, train_frac=train_frac, random_state=seed)
    train_mask = df["chunk"].astype(str).isin(set(train_chunks["chunk"].to_list())).to_numpy()
    logger.info("chunk split: %d train / %d total chunks", len(train_chunks), len(chunks))

    df[ycols] = _train_stats_zscore(df, ["subject", "electrode"], ycols, train_mask)       # step 2
    df["is_train"] = train_mask
    agg_spec = {c: (c, "mean") for c in ycols} | {"is_train": ("is_train", "first")}
    agg = df.groupby(["subject", "region", "word_idx", "chunk", "wlen", "freq"], as_index=False).agg(**agg_spec)  # step 3
    agg[ycols] = _train_stats_zscore(agg, ["subject", "region"], ycols, agg["is_train"].to_numpy())             # step 4

    n_subj = agg.groupby("region")["subject"].nunique()
    keep = sorted(n_subj[n_subj >= min_subjects].index.tolist())
    for r, n in n_subj.items():
        logger.info("region %-9s %d subjects%s", r, n, "" if r in keep else "  -> skipped (< min_subjects)")
    return agg[agg["region"].isin(keep)].reset_index(drop=True), keep


def fit_region_lme_one(reg: pd.DataFrame, *, region: str, lag: int, y_col: str | None = None,
                       return_predictions: bool = False) -> dict:
    """Fit baseline/full on one region's TRAIN rows, score the held-out rows. ``reg`` needs
    subject, word_idx, chunk, wlen, freq, surprisal, is_train and ``y_<lag>`` (or ``y_col``).
    With ``return_predictions`` the row gets a ``_predictions`` DataFrame: one line per held-out
    (subject, word) with y, both models' predictions and per-observation Gaussian log-likelihoods."""
    from pymer4.models import compare, lmer

    t0 = time.time()
    row: dict = {c: None for c in ROW_COLUMNS}
    row.update(region=region, lag=lag, lag_time_s=float(lag_times()[lag]))
    try:
        frame = reg.assign(electrode=region, y=reg[y_col or f"y_{lag}"])
        data = prepare_podcast_lme_frame(frame)
        mask = frame["is_train"].to_numpy()
        train_data, test_data = data.filter(pl.Series(mask)), data.filter(pl.Series(~mask))
        train_data = train_data.filter(pl.col("y").is_finite())
        test_data = test_data.filter(pl.col("y").is_finite())

        bm = lmer(BASELINE, data=train_data)
        bm.fit(verbose=False, summary=False, conf_method="wald")
        fm = lmer(FULL, data=train_data)
        fm.fit(verbose=False, summary=False, conf_method="wald")
        base_ll, _, _, base_ok = _fit_stats(bm)
        ll, aic, bic, full_ok = _fit_stats(fm)
        beta, se, t_stat, p_value = _extract_fixed_effect(fm, "surprisal_z")
        lrt_p = None
        try:
            cmp = compare(bm, fm, as_dataframe=True, test="LRT")
            if cmp.height >= 2 and "Pr(>Chisq)" in cmp.columns:
                lrt_p = float(cmp["Pr(>Chisq)"][1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("LRT comparison failed: %s", exc)

        d_test = base_test = full_test = None
        base_obs = full_obs = None
        if test_data.height >= 10:
            base_test, base_obs = _partition_loglik_detailed(bm, test_data, outcome="y")
            full_test, full_obs = _partition_loglik_detailed(fm, test_data, outcome="y")
            if np.isfinite(base_test) and np.isfinite(full_test):
                d_test = full_test - base_test
        y_true = np.asarray(test_data["y"], dtype=np.float64)
        base_pred, full_pred = _predict_or_nan(bm, test_data), _predict_or_nan(fm, test_data)
        bp, bs = _corrs(y_true, base_pred)
        fp, fs = _corrs(y_true, full_pred)
        if return_predictions:
            row["_predictions"] = pd.DataFrame({
                "subject": test_data["subject"].to_numpy(), "word_idx": test_data["word_idx"].to_numpy(),
                "chunk": test_data["chunk"].to_numpy(), "y": y_true.astype(np.float32),
                "pred_base": base_pred.astype(np.float32), "pred_full": full_pred.astype(np.float32),
                "ll_base": (np.full(len(y_true), np.nan) if base_obs is None else base_obs).astype(np.float32),
                "ll_full": (np.full(len(y_true), np.nan) if full_obs is None else full_obs).astype(np.float32),
            })
        row.update(
            beta=beta, se=se, t_stat=t_stat, p_value=p_value, loglik=ll, aic=aic, bic=bic,
            baseline_loglik=base_ll, delta_loglik_train=ll - base_ll, delta_loglik_test=d_test,
            baseline_loglik_test=base_test, loglik_test=full_test, lrt_p_value=lrt_p,
            test_pearsonr_full_model=fp, test_spearmanr_full_model=fs,
            test_pearsonr_baseline_lme=bp, test_spearmanr_baseline_lme=bs,
            test_delta_pearsonr=fp - bp, test_delta_spearmanr=fs - bs,
            converged=bool(base_ok and full_ok), slope_dropped=False,
            n_obs=int(train_data.height + test_data.height), n_obs_test=int(test_data.height),
        )
    except Exception as exc:  # noqa: BLE001 -- one bad fit must not kill the pool
        logger.error("region %s lag %d failed", region, lag, exc_info=True)
        row["error"] = str(exc)
    row["fit_wall_time_s"] = time.time() - t0
    return row
