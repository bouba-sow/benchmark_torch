#!/usr/bin/env python3
"""Held-out Pearson correlation (full LME model's predictions vs true iEEG
response), split by DLT (Gibson 2000) integration-cost bin, for the NEURAL
surprisal/LME track -- the neural counterpart of dlt/surprisal_corr.py
(which does this for the behavioral/reading-time LME track).

Scope: fedorenko2016_ecog only. This is the ONLY neural surprisal-track
dataset with both (a) per-word held-out predictions saved to disk
(fedorenko2016_lme_per_word.csv: yhat_base, yhat_full, y_true, stimulus_id,
word_num) and (b) a word-level DLT cost file that shares a join key
(outputs/dlt/fedorenko2016_dlt_cost.csv: stimulus_id, word_num). Investigated
and ruled out for the other 4 neural surprisal-track datasets used by
the old figure-1 script (since removed)'s NEURAL_DELTA_SPEC:
  - lpp_fmri / ns_fmri / tuckute_fmri: fMRI is TR-locked (each TR spans
    several words), not word-aligned -- no clean per-word join is possible.
  - podcast_ieeg (the "podcast_lme" surprisal-track pipeline): only the
    per-lag AGGREGATE scores and the raw design matrix are saved to disk
    (podcast_lme_by_lag.csv, podcast_lme_pooled_table.parquet) -- no
    per-word/electrode yhat_full/yhat_base was ever cached, so this would
    need a full LME refit with prediction-storage enabled to compute.
  - nieuwland_eeg: nieuwland_lme_per_word/layer*.parquet has yhat_full/
    yhat_baseline but is indexed by (subject, electrode, chunk, lab) with NO
    item/word identifier recoverable from the saved file -- can't be joined
    to outputs/dlt/nieuwland_dlt_cost.csv's item_number without rerunning
    the fit with the join key additionally saved.

Computes pearsonr(yhat_full, y_true) and pearsonr(yhat_base, y_true) pooled
over (electrode, chunk) pairs within each DLT bin, at the model's best layer
(by held-out test_delta_pearsonr, from fedorenko2016_lme_by_layer.csv).
score_delta = full - base, the delta convention used throughout the paper's
Pareto figures. Crash-safe incremental cache -- safe to re-run.

Usage:
  .venv/bin/python3 scripts/dlt/neural_surprisal_corr.py --n_jobs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr

from neuroai.analysis.benchmark_common import OUTPUTS_JEANZAY, REPO_ROOT
from scripts.summarize.benchmark import (
    _iter_model_score_files, _model_id_from_dir, _read_csv_safe, lookup_model_meta,
)

SWEEP_ROOT = OUTPUTS_JEANZAY / "fedorenko2016/surprisal_sweep"
SCORES_FILENAME = "fedorenko2016_lme_by_layer.csv"
PER_WORD_FILENAME = "fedorenko2016_lme_per_word.csv"
DLT_COST_CSV = REPO_ROOT / "outputs" / "dlt" / "fedorenko2016_dlt_cost.csv"
CACHE_DIR = REPO_ROOT / "outputs" / "dlt_corr_cache"
CACHE_PATH = CACHE_DIR / "fedorenko2016_lme.csv"
THRESHOLDS = (1.0, 2.0, 3.0, 4.0)
MIN_N = 5


def _pearson_or_nan(a, b) -> float:
    if len(a) < MIN_N:
        return float("nan")
    r = pearsonr(a, b).statistic
    return float(r) if r == r else float("nan")  # NaN-check without numpy import


def _pick_last_layer(scores: pd.DataFrame) -> int | None:
    """Highest fitted layer index -- matches the "last layer" convention used
    by the old figure-1 script (since removed) / figure2's hard-DLT
    companion (this cache feeds figure2's neural axis). Was previously
    ``_pick_best_layer`` (max held-out test_delta_pearsonr over layers) --
    see the identical fix/rationale in dlt/surprisal_corr.py."""
    if "layer" not in scores.columns:
        return None
    lay = pd.to_numeric(scores["layer"], errors="coerce").dropna()
    if lay.empty:
        return None
    return int(lay.max())


def _process_one_model(family: str, model_dir: Path, dlt_key: pd.DataFrame) -> list[dict] | None:
    try:
        model_id = _model_id_from_dir(model_dir)
        scores_path = model_dir / SCORES_FILENAME
        scores = _read_csv_safe(scores_path)
        if scores is None or scores.empty:
            return None
        last_layer = _pick_last_layer(scores)
        if last_layer is None:
            return None

        per_word_path = model_dir / PER_WORD_FILENAME
        if not per_word_path.exists():
            return None
        pw = _read_csv_safe(per_word_path)
        if pw is None or pw.empty:
            return None
        need = {"layer", "stimulus_id", "word_num", "y_true", "yhat_base", "yhat_full"}
        if not need <= set(pw.columns):
            return None
        pw = pw[pd.to_numeric(pw["layer"], errors="coerce") == last_layer]
        if pw.empty:
            return None

        pw = pw.merge(dlt_key, on=["stimulus_id", "word_num"], how="left")
        pw = pw.dropna(subset=["total", "y_true", "yhat_full", "yhat_base"])
        if pw.empty:
            return None

        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        rows: list[dict] = []
        easy_mask = pw["total"] <= 0
        easy_sub = pw.loc[easy_mask]
        n_easy = len(easy_sub)
        easy_full = _pearson_or_nan(easy_sub["yhat_full"], easy_sub["y_true"])
        easy_base = _pearson_or_nan(easy_sub["yhat_base"], easy_sub["y_true"])
        for thr in THRESHOLDS:
            hard_sub = pw.loc[pw["total"] >= thr]
            n_hard = len(hard_sub)
            hard_full = _pearson_or_nan(hard_sub["yhat_full"], hard_sub["y_true"])
            hard_base = _pearson_or_nan(hard_sub["yhat_base"], hard_sub["y_true"])
            rows.append({
                "model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "last_layer": last_layer,
                "dlt_bin": "easy", "score": easy_full, "score_baseline": easy_base,
                "score_delta": easy_full - easy_base, "n_words": n_easy, "hard_gt_threshold": thr,
            })
            rows.append({
                "model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "last_layer": last_layer,
                "dlt_bin": "hard", "score": hard_full, "score_baseline": hard_base,
                "score_delta": hard_full - hard_base, "n_words": n_hard, "hard_gt_threshold": thr,
            })
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"[fedorenko2016_lme] FAILED model {model_dir}: {e}", file=sys.stderr)
        return None


def compute(n_jobs: int, force: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dlt_key = pd.read_csv(DLT_COST_CSV)[["stimulus_id", "word_num", "total"]].drop_duplicates(["stimulus_id", "word_num"])

    done_ids: set[str] = set()
    existing_rows: list[pd.DataFrame] = []
    if CACHE_PATH.exists() and not force:
        cached = pd.read_csv(CACHE_PATH)
        existing_rows.append(cached)
        counts = cached.groupby("model")["hard_gt_threshold"].nunique()
        done_ids = set(counts[counts >= len(THRESHOLDS)].index)
        print(f"[fedorenko2016_lme] resuming: {len(done_ids)} models already cached")

    todo = [
        (family, model_dir) for family, model_dir, _files in _iter_model_score_files(SWEEP_ROOT, SCORES_FILENAME)
        if _model_id_from_dir(model_dir) not in done_ids
    ]
    print(f"[fedorenko2016_lme] {len(todo)} models to process (n_jobs={n_jobs})")

    BATCH = max(n_jobs, 8)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_process_one_model)(family, model_dir, dlt_key) for family, model_dir in batch
            )
        except Exception as e:  # noqa: BLE001
            print(f"[fedorenko2016_lme] batch {i}-{i+len(batch)} CRASHED: {e} -- skipping batch", file=sys.stderr)
            continue
        new_rows = [r for res in results if res for r in res]
        if new_rows:
            existing_rows.append(pd.DataFrame(new_rows))
            pd.concat(existing_rows, ignore_index=True).to_csv(CACHE_PATH, index=False)
        print(f"[fedorenko2016_lme] batch {i}-{i+len(batch)} done, {len(new_rows)} new rows")

    print(f"[fedorenko2016_lme] COMPLETE -- wrote {CACHE_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    compute(args.n_jobs, args.force)


if __name__ == "__main__":
    main()
