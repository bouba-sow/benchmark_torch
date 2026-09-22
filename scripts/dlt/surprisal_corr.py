#!/usr/bin/env python3
"""Held-out Pearson correlation (full LME model's predictions vs true value),
split by DLT (Gibson 2000) integration-cost bin (easy: cost<=0, hard:
cost>=threshold, thresholds 1/2/3/4 computed together in one pass), for the
behavioral surprisal/LME track.

This is the bounded, ceiling-normalisable counterpart to
dlt/delta_loglik.py's mean-delta-loglik metric (which is an
unnormalised sum-derived quantity, not comparable on the same 0..1 scale as
the embedding/neural tracks' Spearman-r/ceiling scores). Here we instead
compute pearsonr(yhat_full, fdur) pooled over just the words in each DLT
bin -- the exact same convention used by the embedding-track and neural DLT
pipelines (scripts/dlt/difficulty_scores.py's _bin_score), just applied
to the LME per-word held-out predictions.

Only models whose per-word file has the newer schema (fdur, yhat_baseline,
yhat_full columns -- held-out predictions) can be scored this way; legacy
delta-loglik-only rows are skipped. Coverage (checked before writing this):
meco/provo/ucl_reading/zuco_reading = 102/102 models; geco = 29/102 (GECO has
a known inconsistent per-word schema across families).

Crash-safe incremental cache -- safe to re-run, skips model_ids already
cached for ALL thresholds.

Usage (one dataset per invocation, meant to be run under srun/sbatch since it
reads big per-word files):
  .venv/bin/python3 scripts/dlt/surprisal_corr.py --dataset geco --n_jobs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from scipy.stats import pearsonr

from neuroai.analysis.benchmark_common import REPO_ROOT
from scripts.dlt.delta_loglik import DATASET_SPEC, _final_revision
from scripts.summarize.benchmark import (
    _iter_model_score_files,
    _model_id_from_dir,
    _read_csv_safe,
    lookup_model_meta,
)

CACHE_DIR = REPO_ROOT / "outputs" / "dlt_corr_cache"
THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
MIN_N = 5


def _pick_last_layer(scores: pd.DataFrame) -> int | None:
    """Highest fitted layer index -- matches the "last layer" convention used
    by the old figure-1 script (since removed) / figure2's hard-DLT
    companion (this cache feeds figure2's behavioral axis). Was previously
    ``_pick_best_layer`` (max held-out test_delta_pearsonr over layers) --
    that silently gave deeper models (more layers to search) a much larger
    "multiple comparisons" advantage at finding a noisy peak, on top of not
    matching the last-layer convention the rest of the Pareto figures use."""
    if "layer" not in scores.columns:
        return None
    lay = pd.to_numeric(scores["layer"], errors="coerce").dropna()
    if lay.empty:
        return None
    return int(lay.max())


def _pearson_or_nan(a, b) -> float:
    if len(a) < MIN_N:
        return float("nan")
    r = pearsonr(a, b).statistic
    return float(r) if r == r else float("nan")  # NaN-check without numpy import


def _process_one_model(dataset: str, family: str, model_dir: Path, scores_filename: str,
                        per_word_filename: str, dlt_key: pd.DataFrame) -> list[dict] | None:
    try:
        model_id = _model_id_from_dir(model_dir)
        score_files = sorted(model_dir.rglob(scores_filename)) or [model_dir / scores_filename]
        frames = [d for f in score_files if f.exists() and (d := _read_csv_safe(f)) is not None]
        if not frames:
            return None
        scores = pd.concat(frames, ignore_index=True)
        revision = _final_revision(scores)
        scores_rev = scores[scores["revision"].astype(str) == revision] if "revision" in scores.columns else scores
        last_layer = _pick_last_layer(scores_rev)

        per_word_path = model_dir / per_word_filename
        if not per_word_path.exists():
            return None

        lf = pl.scan_csv(per_word_path, schema_overrides={"sentid": pl.Int64, "sentpos": pl.Int64})
        cols = lf.collect_schema().names()
        need = {"sentid", "sentpos", "fdur", "yhat_full"}
        if not need <= set(cols):
            return None  # legacy schema, no held-out predictions -- skip
        select_cols = ["sentid", "sentpos", "fdur", "yhat_full"]
        if "yhat_baseline" in cols:
            select_cols.append("yhat_baseline")
        if "revision" in cols:
            select_cols.append("revision")
        if "layer" in cols:
            select_cols.append("layer")
        pw = lf.select(select_cols).collect(engine="streaming").to_pandas()

        if "revision" in pw.columns:
            pw = pw[pw["revision"].astype(str) == revision]
        if "layer" in pw.columns:
            if last_layer is None:
                return None
            pw = pw[pd.to_numeric(pw["layer"], errors="coerce") == last_layer]
        if pw.empty:
            return None

        pw["story_id"] = pw["sentid"].astype(int)
        pw["zone"] = pw["sentpos"].astype(int)
        pw = pw.merge(dlt_key, on=["story_id", "zone"], how="left")
        has_baseline = "yhat_baseline" in pw.columns
        subset_cols = ["total", "fdur", "yhat_full"] + (["yhat_baseline"] if has_baseline else [])
        pw = pw.dropna(subset=subset_cols)
        if pw.empty:
            return None

        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        rows: list[dict] = []
        easy_mask = pw["total"] <= 0
        easy_sub = pw.loc[easy_mask]
        n_easy = len(easy_sub)
        n_easy_words = int(easy_sub[["story_id", "zone"]].drop_duplicates().shape[0])
        easy_full = _pearson_or_nan(easy_sub["yhat_full"], easy_sub["fdur"])
        easy_base = _pearson_or_nan(easy_sub["yhat_baseline"], easy_sub["fdur"]) if has_baseline else float("nan")
        for thr in THRESHOLDS:
            hard_sub = pw.loc[pw["total"] >= thr]
            n_hard = len(hard_sub)
            n_hard_words = int(hard_sub[["story_id", "zone"]].drop_duplicates().shape[0])
            hard_full = _pearson_or_nan(hard_sub["yhat_full"], hard_sub["fdur"])
            hard_base = _pearson_or_nan(hard_sub["yhat_baseline"], hard_sub["fdur"]) if has_baseline else float("nan")
            rows.append({
                "model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "revision": revision,
                "dlt_bin": "easy", "score": easy_full, "score_baseline": easy_base,
                "score_delta": easy_full - easy_base if has_baseline else float("nan"),
                "n_words": n_easy, "n_words_unique": n_easy_words, "hard_gt_threshold": thr,
            })
            rows.append({
                "model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "revision": revision,
                "dlt_bin": "hard", "score": hard_full, "score_baseline": hard_base,
                "score_delta": hard_full - hard_base if has_baseline else float("nan"),
                "n_words": n_hard, "n_words_unique": n_hard_words, "hard_gt_threshold": thr,
            })
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"[{dataset}] FAILED model {model_dir}: {e}", file=sys.stderr)
        return None


def compute_dataset(dataset: str, n_jobs: int, force: bool = False) -> None:
    sweep_root, scores_filename, per_word_filename, dlt_cost_csv = DATASET_SPEC[dataset]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{dataset}.csv"

    dlt_key = pd.read_csv(dlt_cost_csv)[["story_id", "zone", "total"]].drop_duplicates(["story_id", "zone"])

    done_ids: set[str] = set()
    existing_rows: list[pd.DataFrame] = []
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path)
        existing_rows.append(cached)
        counts = cached.groupby("model")["hard_gt_threshold"].nunique()
        done_ids = set(counts[counts >= len(THRESHOLDS)].index)
        print(f"[{dataset}] resuming: {len(done_ids)} models already cached")

    todo = [
        (family, model_dir) for family, model_dir, _files in _iter_model_score_files(sweep_root, scores_filename)
        if _model_id_from_dir(model_dir) not in done_ids
    ]
    print(f"[{dataset}] {len(todo)} models to process (n_jobs={n_jobs})")

    BATCH = max(n_jobs, 8)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_process_one_model)(dataset, family, model_dir, scores_filename, per_word_filename, dlt_key)
                for family, model_dir in batch
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{dataset}] batch {i}-{i+len(batch)} CRASHED: {e} -- skipping batch", file=sys.stderr)
            continue
        new_rows = [r for res in results if res for r in res]
        if new_rows:
            existing_rows.append(pd.DataFrame(new_rows))
            pd.concat(existing_rows, ignore_index=True).to_csv(cache_path, index=False)
        print(f"[{dataset}] batch {i}-{i+len(batch)} done, {len(new_rows)} new rows")

    print(f"[{dataset}] COMPLETE -- wrote {cache_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_SPEC))
    ap.add_argument("--n_jobs", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    compute_dataset(args.dataset, args.n_jobs, args.force)


if __name__ == "__main__":
    main()
