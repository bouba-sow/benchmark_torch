#!/usr/bin/env python3
"""Mean delta-log-likelihood (full LME model - baseline LME model) split by
DLT (Gibson 2000) integration-cost bin (easy: cost<=0, hard: cost>=threshold,
thresholds 1/2/3/4 computed together in one pass), for the behavioral
surprisal/LME track. This is the native metric for that track (per user
request -- the ceiling-normalised delta-Pearson-r used elsewhere is a
derived/held-out metric, not the primary one for LME fits).

Reads raw per-word LME predictions (large files, up to a few GB per model)
joined against outputs/dlt/{dataset}_dlt_cost.csv on (sentid->story_id,
sentpos->zone). Crash-safe incremental cache -- safe to re-run, skips
model_ids already cached for ALL thresholds.

Usage (one dataset per invocation, meant to be run under srun/sbatch since it
reads big per-word files):
  .venv/bin/python3 scripts/dlt/delta_loglik.py --dataset geco --n_jobs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd
import polars as pl
from joblib import Parallel, delayed

from neuroai.analysis.benchmark_common import REPO_ROOT, OUTPUTS_JEANZAY
from scripts.summarize.benchmark import (
    _iter_model_score_files,
    _model_id_from_dir,
    _pick_final_row,
    _read_csv_safe,
    lookup_model_meta,
)

DLT_DIR = REPO_ROOT / "outputs" / "dlt"
CACHE_DIR = REPO_ROOT / "outputs" / "dlt_loglik_cache"
# 5.0 added so this cache covers every threshold figures/figure6_dlt_threshold_grid.py
# requests (that grid's panels go up to "hard >= 5") -- was (1,2,3,4) only,
# which silently left that grid's last panel with no DLL-based data to plot.
THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0)

# dataset_key -> (sweep_root, scores_filename, per_word_filename, dlt_cost_csv)
DATASET_SPEC: dict[str, tuple[Path, str, str, Path]] = {
    "geco": (
        # Was geco/lme_sweep_dlt -- a stale legacy dir (last touched Jul 15,
        # effectively empty/single-row) that predates this dataset's 75/25
        # OOS resweep. geco/surprisal_sweep/fp is the current,
        # complete, 75/25 sweep (verified 2026-09-08).
        OUTPUTS_JEANZAY / "geco/surprisal_sweep" / "fp",
        "geco_lme_checkpoint_scores.csv", "geco_FP_lme_per_word.csv",
        DLT_DIR / "geco_dlt_cost.csv",
    ),
    "geco_gp": (
        OUTPUTS_JEANZAY / "geco/surprisal_sweep" / "gp",
        "geco_lme_checkpoint_scores.csv", "geco_GP_lme_per_word.csv",
        DLT_DIR / "geco_dlt_cost.csv",   # same corpus/text as FP, different duration measure
    ),
    "meco": (
        OUTPUTS_JEANZAY / "meco/surprisal_sweep" / "fp",
        "meco_lme_checkpoint_scores.csv", "meco_FP_lme_per_word.csv",
        DLT_DIR / "meco_dlt_cost.csv",
    ),
    "meco_gp": (
        OUTPUTS_JEANZAY / "meco/surprisal_sweep" / "gp",
        "meco_lme_checkpoint_scores.csv", "meco_GP_lme_per_word.csv",
        DLT_DIR / "meco_dlt_cost.csv",
    ),
    "meco_sp": (
        OUTPUTS_JEANZAY / "meco/surprisal_sweep" / "sp",
        "meco_lme_checkpoint_scores.csv", "meco_SP_lme_per_word.csv",
        DLT_DIR / "meco_dlt_cost.csv",
    ),
    "provo": (
        OUTPUTS_JEANZAY / "provo/surprisal_sweep" / "fp",
        "provo_lme_checkpoint_scores.csv", "provo_FP_lme_per_word.csv",
        DLT_DIR / "provo_dlt_cost.csv",
    ),
    "provo_gp": (
        OUTPUTS_JEANZAY / "provo/surprisal_sweep" / "gp",
        "provo_lme_checkpoint_scores.csv", "provo_GP_lme_per_word.csv",
        DLT_DIR / "provo_dlt_cost.csv",
    ),
    "provo_sp": (
        OUTPUTS_JEANZAY / "provo/surprisal_sweep" / "sp",
        "provo_lme_checkpoint_scores.csv", "provo_SP_lme_per_word.csv",
        DLT_DIR / "provo_dlt_cost.csv",
    ),
    "ucl_reading": (
        OUTPUTS_JEANZAY / "ucl_reading/surprisal_sweep",
        "ucl_reading_spr_lme_checkpoint_scores.csv", "ucl_reading_spr_lme_per_word.csv",
        DLT_DIR / "ucl_reading_dlt_cost.csv",
    ),
    "zuco_reading": (
        # Was zuco_reading/checkpoint_sweep_lme_dlt/nr -- the pre-refit,
        # 50/50-split tree. zuco_reading/surprisal_sweep/nr is
        # the corrected 75/25 resweep, word predictions backfilled 2026-09-08.
        OUTPUTS_JEANZAY / "zuco_reading/surprisal_sweep" / "nr",
        "zuco_reading_trt_lme_checkpoint_scores.csv", "zuco_reading_NR_lme_per_word.csv",
        DLT_DIR / "zuco_reading_nr_dlt_cost.csv",
    ),
    "zuco_reading_sr": (
        OUTPUTS_JEANZAY / "zuco_reading/surprisal_sweep" / "sr",
        "zuco_reading_trt_lme_checkpoint_scores.csv", "zuco_reading_SR_lme_per_word.csv",
        DLT_DIR / "zuco_reading_sr_dlt_cost.csv",
    ),
    "zuco_reading_tsr": (
        OUTPUTS_JEANZAY / "zuco_reading/surprisal_sweep" / "tsr",
        "zuco_reading_trt_lme_checkpoint_scores.csv", "zuco_reading_TSR_lme_per_word.csv",
        DLT_DIR / "zuco_reading_tsr_dlt_cost.csv",
    ),
}


def _pick_best_layer(scores: pd.DataFrame) -> int | None:
    if "layer" not in scores.columns:
        return None
    col = "test_delta_pearsonr" if "test_delta_pearsonr" in scores.columns else None
    if col is None:
        return None
    s = pd.to_numeric(scores[col], errors="coerce")
    lay = pd.to_numeric(scores["layer"], errors="coerce")
    valid = lay.notna() & s.notna() & (s.abs() <= 0.1)
    if not valid.any():
        return None
    return int(lay.loc[s[valid].idxmax()])


def _final_revision(scores: pd.DataFrame) -> str:
    key = "tokens_seen" if "tokens_seen" in scores.columns else ("step" if "step" in scores.columns else None)
    if key is not None and scores[key].notna().any():
        scores = scores[scores[key] == scores[key].max()]
    return str(_pick_final_row(scores).get("revision", "main"))


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
        best_layer = _pick_best_layer(scores_rev)

        per_word_path = model_dir / per_word_filename
        if not per_word_path.exists():
            return None

        lf = pl.scan_csv(per_word_path, schema_overrides={"sentid": pl.Int64, "sentpos": pl.Int64})
        cols = lf.collect_schema().names()
        if "delta_loglik" not in cols or "sentid" not in cols or "sentpos" not in cols:
            return None
        select_cols = ["sentid", "sentpos", "delta_loglik"]
        if "revision" in cols:
            select_cols.append("revision")
        if "layer" in cols:
            select_cols.append("layer")
        pw = lf.select(select_cols).collect(engine="streaming").to_pandas()

        if "revision" in pw.columns:
            pw = pw[pw["revision"].astype(str) == revision]
        if "layer" in pw.columns:
            if best_layer is None:
                return None
            pw = pw[pd.to_numeric(pw["layer"], errors="coerce") == best_layer]
        if pw.empty:
            return None

        pw["story_id"] = pw["sentid"].astype(int)
        pw["zone"] = pw["sentpos"].astype(int)
        pw = pw.merge(dlt_key, on=["story_id", "zone"], how="left")
        pw = pw.dropna(subset=["total", "delta_loglik"])
        if pw.empty:
            return None

        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        rows: list[dict] = []
        easy_mask = pw["total"] <= 0
        easy_sub = pw.loc[easy_mask]
        n_easy = len(easy_sub)
        # n_words (row count) is inflated by however many subjects/held-out
        # test folds a given word appears in -- this per-word file has one
        # row per (subject, sentid, sentpos), not one row per word (mean
        # ~3.9x, up to 8x on GECO). n_words_unique (nunique of the actual
        # word key) is the real word count -- same distinction
        # dlt/surprisal_corr.py already draws for its own n_words vs
        # n_words_unique, reintroduced here after this cache's n_words alone
        # was wired into the figures' word-count displays (found 2026-09-08).
        n_easy_words = int(easy_sub[["story_id", "zone"]].drop_duplicates().shape[0])
        easy_score = float(easy_sub["delta_loglik"].mean()) if n_easy > 0 else float("nan")
        for thr in THRESHOLDS:
            hard_sub = pw.loc[pw["total"] >= thr]
            n_hard = len(hard_sub)
            n_hard_words = int(hard_sub[["story_id", "zone"]].drop_duplicates().shape[0])
            hard_score = float(hard_sub["delta_loglik"].mean()) if n_hard > 0 else float("nan")
            rows.append({
                "model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "revision": revision,
                "dlt_bin": "easy", "score": easy_score, "n_words": n_easy,
                "n_words_unique": n_easy_words, "hard_gt_threshold": thr,
            })
            rows.append({
                "model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "revision": revision,
                "dlt_bin": "hard", "score": hard_score, "n_words": n_hard,
                "n_words_unique": n_hard_words, "hard_gt_threshold": thr,
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
