#!/usr/bin/env python3
"""Mean per-observation delta-log-likelihood (full LME - baseline LME) split
by DLT (Gibson 2000) integration-cost bin, for the Podcast neural
surprisal/LME track -- the neural-axis counterpart of
dlt/delta_loglik.py's behavioral cache (same "native metric, not the
derived held-out correlation" convention; see that script's docstring).

Podcast's per-lag predictions already carry a per-observation ``delta_loglik``
column (fit.per_word in neuroai/regressors/podcast_lme.py, saved to
predictions/lag<NNN>.parquet by neuroai/benchmarks/podcast_lme.py) -- unlike
the aggregate per-lag delta_loglik_test in podcast_lme_by_lag.csv, this is
already split per (subject, electrode, word_idx), so no new model fitting is
needed, just pooling + a groupby-mean, mirroring
dlt/surprisal_corr_podcast.py's lag-window-pooling convention exactly
(same [-0.2s, +0.8s] word-onset-relative window, same "pool every lag first,
then aggregate once" logic) but averaging delta_loglik directly instead of
computing a held-out Pearson correlation.

SWITCHED 2026-10-09 to the per-REGION LME (headline since 2026-10-08): reads the held-out
per-(subject, region, word) log-likelihoods saved by scripts/run/surprisal/neural/podcast_region_fit.py,
``<word_predictions_root>/podcast/surprisal_region_sweep/<family>/<model>/region_predictions/<region>_lag<NN>.parquet``
(columns ll_base, ll_full), lags 7-25 (0-600 ms, the headline window), joined against
outputs/dlt/podcast_dlt_cost.csv on word_idx. Score = pooled mean of (ll_full - ll_base) over those
rows, which equals the headline (n_r-weighted mean over regions of the mean over lags) restricted to
the DLT bin. Since 2026-10-09 each REGION is normalised by its own ceiling first (n_r = rows of the
bin in region r as weights); ``score`` stores sum_r n_r (m_r / C_r) / sum_r n_r times the dataset ceiling C
(ceiling_summary.json), so the loaders' score / C equals the per-region-normalised score. The old pooled-electrode cache is backed up at
$SCRATCH/benchmark_backups/dlt_podcast_lme_pre_region_20261009.csv.

Crash-safe incremental cache -- safe to re-run.

Usage:
  .venv/bin/python3 scripts/dlt/delta_loglik_podcast.py --n_jobs 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd
from joblib import Parallel, delayed

from neuroai.benchmarks.base import word_predictions_root
from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, OUTPUTS_JEANZAY, REPO_ROOT
from scripts.summarize.benchmark import _iter_model_score_files, _model_id_from_dir, _read_csv_safe, lookup_model_meta

SWEEP_ROOT = OUTPUTS_JEANZAY / "podcast/surprisal_region_sweep"
SCORES_FILENAME = "podcast_region_lme_by_lag.csv"
REGION_SUMMARY = OUTPUTS_JEANZAY / "podcast/summary_region/ceiling_summary.json"
DLT_COST_CSV = REPO_ROOT / "outputs" / "dlt" / "podcast_dlt_cost.csv"
CACHE_DIR = REPO_ROOT / "outputs" / "dlt_loglik_cache"
CACHE_PATH = CACHE_DIR / "podcast_lme.csv"
THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0)
# Headline window of the region LME: 0-600 ms (lags 7-25 of the 32-lag grid).
WINDOW_LAG_TIME_S = (0.0, 0.6)


def _lags_in_window(scores: pd.DataFrame) -> list[int]:
    if "lag" not in scores.columns or "lag_time_s" not in scores.columns:
        return []
    lo, hi = WINDOW_LAG_TIME_S
    t = pd.to_numeric(scores["lag_time_s"], errors="coerce")
    lag = pd.to_numeric(scores["lag"], errors="coerce")
    mask = t.notna() & lag.notna() & (t >= lo) & (t <= hi)
    return sorted(int(x) for x in lag.loc[mask].unique())


def _region_norm_mean(sub: pd.DataFrame, c_r: dict, c_ref: float) -> float:
    """sum_r n_r (mean_r / C_r) / sum_r n_r * C_ref -- NaN if the bin is empty."""
    if sub.empty:
        return float("nan")
    g = sub.groupby("region")["delta_loglik"].agg(["mean", "size"])
    return float(((g["mean"] / g.index.map(c_r)) * g["size"]).sum() / g["size"].sum() * c_ref)


def _process_one_model(family: str, model_dir: Path, dlt_key: pd.DataFrame) -> list[dict] | None:
    try:
        model_id = _model_id_from_dir(model_dir)
        if model_id in EXCLUDED_MODEL_IDS:  # broken upstream checkpoints, out of benchmark scope
            return None
        scores = _read_csv_safe(model_dir / SCORES_FILENAME)
        if scores is None or scores.empty:
            return None
        window_lags = _lags_in_window(scores)
        if not window_lags:
            return None

        need = ["word_idx", "ll_base", "ll_full"]
        pred_dir = word_predictions_root(model_dir) / "region_predictions"
        pred_frames = []
        for (region, lag), _ in scores[scores["lag"].isin(window_lags)].groupby(["region", "lag"]):
            pred_path = pred_dir / f"{region}_lag{int(lag):02d}.parquet"
            if not pred_path.exists():
                continue
            df = pd.read_parquet(pred_path, columns=need)
            df["delta_loglik"] = df["ll_full"].astype("float64") - df["ll_base"].astype("float64")
            df = df[["word_idx", "delta_loglik"]].assign(region=region)
            df = df.merge(dlt_key, on="word_idx", how="inner")
            df = df.dropna(subset=["total", "delta_loglik"])
            if df.empty:
                continue
            df["delta_loglik"] = df["delta_loglik"].astype("float32")
            pred_frames.append(df)
        if not pred_frames:
            return None
        pw = pd.concat(pred_frames, ignore_index=True)
        del pred_frames
        if pw.empty:
            return None

        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        rows: list[dict] = []
        easy_mask = pw["total"] <= 0
        easy_sub = pw.loc[easy_mask]
        n_easy = len(easy_sub)
        # n_words (row count) is inflated by subject x electrode x pooled-lag
        # multiplicity -- one row per (lag, subject, electrode, word_idx),
        # not one row per word. n_words_unique (nunique of word_idx) is the
        # real word count -- same distinction compute_dlt_podcast_surprisal_
        # corr.py already draws, reintroduced here (found 2026-09-08).
        n_easy_words = int(easy_sub["word_idx"].nunique())
        summ = json.loads(REGION_SUMMARY.read_text())
        c_r, c_ref = summ["per_region_ceiling"], summ["ceiling_delta_loglik_per_obs"]
        easy_score = _region_norm_mean(easy_sub, c_r, c_ref)
        for thr in THRESHOLDS:
            hard_sub = pw.loc[pw["total"] >= thr]
            n_hard = len(hard_sub)
            n_hard_words = int(hard_sub["word_idx"].nunique())
            hard_score = _region_norm_mean(hard_sub, c_r, c_ref)
            rows.append({
                "model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "n_lags_pooled": len(window_lags),
                "dlt_bin": "easy", "score": easy_score, "n_words": n_easy,
                "n_words_unique": n_easy_words, "hard_gt_threshold": thr,
            })
            rows.append({
                "model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "n_lags_pooled": len(window_lags),
                "dlt_bin": "hard", "score": hard_score, "n_words": n_hard,
                "n_words_unique": n_hard_words, "hard_gt_threshold": thr,
            })
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"[podcast_lme_dll] FAILED model {model_dir}: {e}", file=sys.stderr)
        return None


def compute(n_jobs: int, force: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dlt_key = pd.read_csv(DLT_COST_CSV)[["word_idx", "total"]].drop_duplicates("word_idx")

    done_ids: set[str] = set()
    existing_rows: list[pd.DataFrame] = []
    if CACHE_PATH.exists() and not force:
        cached = pd.read_csv(CACHE_PATH)
        existing_rows.append(cached)
        counts = cached.groupby("model")["hard_gt_threshold"].nunique()
        done_ids = set(counts[counts >= len(THRESHOLDS)].index)
        print(f"[podcast_lme_dll] resuming: {len(done_ids)} models already cached")

    todo = [
        (family, model_dir) for family, model_dir, _files in _iter_model_score_files(SWEEP_ROOT, SCORES_FILENAME)
        if _model_id_from_dir(model_dir) not in done_ids
    ]
    print(f"[podcast_lme_dll] {len(todo)} models to process (n_jobs={n_jobs})")

    BATCH = max(n_jobs, 8)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_process_one_model)(family, model_dir, dlt_key) for family, model_dir in batch
            )
        except Exception as e:  # noqa: BLE001
            print(f"[podcast_lme_dll] batch {i}-{i+len(batch)} CRASHED: {e} -- skipping batch", file=sys.stderr)
            continue
        new_rows = [r for res in results if res for r in res]
        if new_rows:
            existing_rows.append(pd.DataFrame(new_rows))
            pd.concat(existing_rows, ignore_index=True).to_csv(CACHE_PATH, index=False)
        print(f"[podcast_lme_dll] batch {i}-{i+len(batch)} done, {len(new_rows)} new rows")

    print(f"[podcast_lme_dll] COMPLETE -- wrote {CACHE_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    compute(args.n_jobs, args.force)


if __name__ == "__main__":
    main()
