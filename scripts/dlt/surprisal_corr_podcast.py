#!/usr/bin/env python3
"""Held-out Pearson correlation (full LME model's predictions vs true iEEG
response), split by DLT (Gibson 2000) integration-cost bin, for the Podcast
neural surprisal/LME track -- currently the only neural surprisal-track
dataset in the hard-DLT Pareto figures (fedorenko2016_lme was removed per
explicit request; see dlt/neural_surprisal_corr.py's docstring for
the other 6 neural surprisal datasets ruled out as infeasible).

Podcast's LME fit sweeps LAG (word-onset-relative time), not model depth --
"peak lag" (a per-model argmax over held-out test_delta_pearsonr) was the
original convention here, but that's a noisy best-of-N selection (the same
class of problem as the "last layer, not best layer" fix applied elsewhere
in this pipeline -- see dlt/surprisal_corr.py's docstring). Per
explicit request, this instead POOLS every lag whose lag_time_s falls in the
word-onset-relative window [-0.2s, +0.8s] into one combined table before
computing each DLT bin's Pearson correlation -- same "pool everything, then
correlate once" convention already used within a single lag (pooling across
subject x electrode x word), just extended to also pool across lags. The
window is read per-model from that model's own podcast_lme_by_lag.csv (lag
spacing/count can vary slightly by model), not hard-coded to a fixed lag
index range.

Reads per-lag held-out predictions from
outputs_jeanzay/podcast/surprisal_sweep/<family>/<model>/predictions/lag<NNN>.parquet
(subject, electrode, word_idx, y_true, yhat_base, yhat_full), joined against
outputs/dlt/podcast_dlt_cost.csv on word_idx. Podcast's DLT cost tops out at
80, so this dataset alone supports every threshold figure6 requests (1-5)
without falling back to a behavioral-only panel.

Crash-safe incremental cache -- safe to re-run.

Usage:
  .venv/bin/python3 scripts/dlt/surprisal_corr_podcast.py --n_jobs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, OUTPUTS_JEANZAY, REPO_ROOT
from scripts.summarize.benchmark import (
    _iter_model_score_files, _model_id_from_dir, _read_csv_safe, lookup_model_meta,
)

SWEEP_ROOT = OUTPUTS_JEANZAY / "podcast/surprisal_sweep"
SCORES_FILENAME = "podcast_lme_by_lag.csv"
DLT_COST_CSV = REPO_ROOT / "outputs" / "dlt" / "podcast_dlt_cost.csv"
CACHE_DIR = REPO_ROOT / "outputs" / "dlt_corr_cache"
CACHE_PATH = CACHE_DIR / "podcast_lme.csv"
THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
MIN_N = 5
WINDOW_LAG_TIME_S = (-0.2, 0.8)


def _pearson_or_nan(a, b) -> float:
    if len(a) < MIN_N:
        return float("nan")
    r = pearsonr(a, b).statistic
    return float(r) if r == r else float("nan")  # NaN-check without numpy import


def _lags_in_window(scores: pd.DataFrame) -> list[int]:
    if "lag" not in scores.columns or "lag_time_s" not in scores.columns:
        return []
    lo, hi = WINDOW_LAG_TIME_S
    t = pd.to_numeric(scores["lag_time_s"], errors="coerce")
    lag = pd.to_numeric(scores["lag"], errors="coerce")
    mask = t.notna() & lag.notna() & (t >= lo) & (t <= hi)
    return sorted(int(x) for x in lag.loc[mask].unique())


def _process_one_model(family: str, model_dir: Path, dlt_key: pd.DataFrame) -> list[dict] | None:
    try:
        model_id = _model_id_from_dir(model_dir)
        if model_id in EXCLUDED_MODEL_IDS:  # broken upstream checkpoints, out of benchmark scope
            return None
        scores_path = model_dir / SCORES_FILENAME
        scores = _read_csv_safe(scores_path)
        if scores is None or scores.empty:
            return None
        window_lags = _lags_in_window(scores)
        if not window_lags:
            return None

        # Memory: pooling ~32 lags x ~250K (subject x electrode x word) rows
        # is ~8M rows/model -- with 16 models in flight at once this OOM-killed
        # the node. Cut per-lag frames down before concatenating: read only
        # the 4 needed columns (not the full ~11-column parquet), inner-merge
        # + dropna PER LAG (not once on the full pooled table, so unmatched/
        # invalid rows never enter the running concatenation), and downcast
        # to float32.
        need = {"word_idx", "y_true", "yhat_base", "yhat_full"}
        pred_frames = []
        for lag in window_lags:
            pred_path = model_dir / "predictions" / f"lag{lag:03d}.parquet"
            if not pred_path.exists():
                continue
            df = pd.read_parquet(pred_path, columns=list(need))
            if not need <= set(df.columns):
                continue
            df = df.merge(dlt_key, on="word_idx", how="inner")
            df = df.dropna(subset=["total", "y_true", "yhat_full", "yhat_base"])
            if df.empty:
                continue
            for col in ("y_true", "yhat_base", "yhat_full", "total"):
                df[col] = df[col].astype("float32")
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
        n_easy_words = int(easy_sub["word_idx"].nunique())
        easy_full = _pearson_or_nan(easy_sub["yhat_full"], easy_sub["y_true"])
        easy_base = _pearson_or_nan(easy_sub["yhat_base"], easy_sub["y_true"])
        for thr in THRESHOLDS:
            hard_sub = pw.loc[pw["total"] >= thr]
            n_hard = len(hard_sub)
            n_hard_words = int(hard_sub["word_idx"].nunique())
            hard_full = _pearson_or_nan(hard_sub["yhat_full"], hard_sub["y_true"])
            hard_base = _pearson_or_nan(hard_sub["yhat_base"], hard_sub["y_true"])
            rows.append({
                "model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "n_lags_pooled": len(window_lags),
                "dlt_bin": "easy", "score": easy_full, "score_baseline": easy_base,
                "score_delta": easy_full - easy_base, "n_words": n_easy,
                "n_words_unique": n_easy_words, "hard_gt_threshold": thr,
            })
            rows.append({
                "model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "n_lags_pooled": len(window_lags),
                "dlt_bin": "hard", "score": hard_full, "score_baseline": hard_base,
                "score_delta": hard_full - hard_base, "n_words": n_hard,
                "n_words_unique": n_hard_words, "hard_gt_threshold": thr,
            })
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"[podcast_lme] FAILED model {model_dir}: {e}", file=sys.stderr)
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
        print(f"[podcast_lme] resuming: {len(done_ids)} models already cached")

    todo = [
        (family, model_dir) for family, model_dir, _files in _iter_model_score_files(SWEEP_ROOT, SCORES_FILENAME)
        if _model_id_from_dir(model_dir) not in done_ids
    ]
    print(f"[podcast_lme] {len(todo)} models to process (n_jobs={n_jobs})")

    BATCH = max(n_jobs, 8)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_process_one_model)(family, model_dir, dlt_key) for family, model_dir in batch
            )
        except Exception as e:  # noqa: BLE001
            print(f"[podcast_lme] batch {i}-{i+len(batch)} CRASHED: {e} -- skipping batch", file=sys.stderr)
            continue
        new_rows = [r for res in results if res for r in res]
        if new_rows:
            existing_rows.append(pd.DataFrame(new_rows))
            pd.concat(existing_rows, ignore_index=True).to_csv(CACHE_PATH, index=False)
        print(f"[podcast_lme] batch {i}-{i+len(batch)} done, {len(new_rows)} new rows")

    print(f"[podcast_lme] COMPLETE -- wrote {CACHE_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    compute(args.n_jobs, args.force)


if __name__ == "__main__":
    main()
