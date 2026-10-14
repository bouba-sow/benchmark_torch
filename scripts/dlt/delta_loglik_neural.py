#!/usr/bin/env python3
"""Mean per-observation delta-log-likelihood (full - baseline LME) split by DLT (Gibson 2000) cost bin for
the neural surprisal datasets with saved per-word held-out log-likelihoods: Broderick (EEG), MEG-MASC (MEG),
Fedorenko2016 (ECoG). Counterpart of scripts/dlt/delta_loglik_podcast.py (Podcast) and delta_loglik.py
(behavioral); same output layout and bins (easy: cost == 0; hard: cost >= 1..5), so the cache can be loaded
with neuroai.analysis.dlt_difficulty._load_dll_bin.

Reads  <sweep>/<family>/<model>/<dataset>_lme_per_word.csv   (held-out rows only; checked: row count and summed
delta_loglik equal n_obs_test / delta_loglik_test of the by-layer CSV) at the LAST layer (highest layer index of
the by-layer CSV, as the *last-layer* figures), joined to outputs/dlt/<dataset>_dlt_cost.csv:
  broderick, megmasc : on word_idx (global index, unique in the cost table)
  fedorenko2016      : on stimulus_id (word_num repeats across sentences)
Score = mean delta_loglik over the held-out rows of the bin (rows = subject x region/electrode x word).
Written to outputs/dlt_loglik_cache/<dataset>_lme.csv. Crash-safe, resumable.

  .venv/bin/python scripts/dlt/delta_loglik_neural.py --dataset broderick --n_jobs 18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd
import polars as pl
from joblib import Parallel, delayed

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, OUTPUTS_JEANZAY, REPO_ROOT
from scripts.summarize.benchmark import _iter_model_score_files, _model_id_from_dir, lookup_model_meta

DLT_DIR = REPO_ROOT / "outputs" / "dlt"
CACHE_DIR = REPO_ROOT / "outputs" / "dlt_loglik_cache"
THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0)
# dataset -> (by-layer csv, per-word csv, join key, dlt cost csv)
SPEC = {
    "broderick": ("broderick_lme_by_layer.csv", "broderick_lme_per_word.csv", "word_idx", "broderick_dlt_cost.csv"),
    "megmasc": ("megmasc_lme_by_layer.csv", "megmasc_lme_per_word.csv", "word_idx", "megmasc_dlt_cost.csv"),
    "fedorenko2016": ("fedorenko2016_lme_by_layer.csv", "fedorenko2016_lme_per_word.csv", "stimulus_id",
                      "fedorenko2016_dlt_cost.csv"),
}


_BRODERICK_COLS = ["layer", "region", "chunk", "word_idx", "baseline_loglik", "full_loglik", "delta_loglik", "eeg",
                   "yhat_baseline", "yhat_full"]


def _broderick_new_block_start(path: Path) -> int | None:
    """15 Broderick per-word files (bloom, fox, rwkv4) hold an OLD run with a leading ``subject`` column (11
    fields, under the header) followed by the appended re-run without it (10 fields, no header). The by-layer
    CSV matches the second block only (checked on bloom-1b1: summed delta_loglik 43.574 vs by-layer 43.574, old
    block 39.18). Returns the number of lines to skip to reach it, or None for ordinary single-block files."""
    with open(path, "rb") as fh:
        header = fh.readline()
        if not header.startswith(b"subject"):
            return None
        n_old = header.count(b",")
        for i, line in enumerate(fh, start=1):
            if line.count(b",") != n_old:
                return i
    return None


def _process(dataset: str, family: str, model_dir: Path, cost: pl.DataFrame) -> list[dict] | None:
    by_layer_name, pw_name, key, _ = SPEC[dataset]
    a_dataset = dataset
    try:
        model_id = _model_id_from_dir(model_dir)
        if model_id in EXCLUDED_MODEL_IDS:
            return None
        bl = pd.read_csv(model_dir / by_layer_name, usecols=["layer", "delta_loglik_test", "n_obs_test"])
        bl = bl.dropna(subset=["delta_loglik_test", "n_obs_test"])
        if bl.empty:
            return None
        last = int(pd.to_numeric(bl["layer"], errors="coerce").max())
        skip = _broderick_new_block_start(model_dir / pw_name) if a_dataset == "broderick" else None
        if skip is not None:
            scan = pl.scan_csv(model_dir / pw_name, skip_rows=skip, has_header=False, new_columns=_BRODERICK_COLS)
        else:
            scan = pl.scan_csv(model_dir / pw_name)
        pw = scan.filter(pl.col("layer") == last).collect()
        if a_dataset == "broderick" and pw.height > int(bl.loc[pd.to_numeric(bl["layer"], errors="coerce") == last, "n_obs_test"].iloc[0]):
            # bloom-7b1 and rwkv-4-7b hold the re-run block twice (appended copies; rows carry no subject column, so
            # they cannot be de-duplicated by key): keep the first n_obs_test rows; the sum check below verifies it
            pw = pw.head(int(bl.loc[pd.to_numeric(bl["layer"], errors="coerce") == last, "n_obs_test"].iloc[0]))
        pw = pw.select([key, "delta_loglik"])
        # guard: the per-word rows must be the ones the by-layer score was computed on
        ref = bl[pd.to_numeric(bl["layer"], errors="coerce") == last].iloc[0]
        if pw.height != int(ref["n_obs_test"]) or abs(pw["delta_loglik"].sum() - ref["delta_loglik_test"]) > 1e-3 * max(1.0, abs(ref["delta_loglik_test"])):
            raise ValueError(f"per-word rows ({pw.height}, {pw['delta_loglik'].sum():.4f}) != by-layer "
                             f"({int(ref['n_obs_test'])}, {ref['delta_loglik_test']:.4f})")
        pw = pw.join(cost, on=key, how="inner").drop_nulls(["total", "delta_loglik"])
        if pw.is_empty():
            return None
        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        base = {"model": model_id, "family": family, "model_short": model_id.replace("/", "_"),
                "n_params": meta.n_params, "n_lags_pooled": 1, "layer": last}
        rows = []
        easy = pw.filter(pl.col("total") <= 0)
        for thr in THRESHOLDS:
            hard = pw.filter(pl.col("total") >= thr)
            for name, sub in (("easy", easy), ("hard", hard)):
                rows.append(base | {
                    "dlt_bin": name, "score": float(sub["delta_loglik"].mean()) if sub.height else float("nan"),
                    "n_words": sub.height, "n_words_unique": int(sub[key].n_unique()) if sub.height else 0,
                    "hard_gt_threshold": thr})
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"[{dataset}] FAILED {model_dir}: {e}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPEC))
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    by_layer_name, _, key, cost_name = SPEC[a.dataset]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{a.dataset}_lme.csv"
    cost = pl.read_csv(DLT_DIR / cost_name).select([key, "total"]).unique(subset=[key], keep="first")
    frames, done = [], set()
    if out.exists() and not a.force:
        old = pd.read_csv(out)
        frames.append(old)
        done = set(old.groupby("model")["hard_gt_threshold"].nunique().loc[lambda s: s >= len(THRESHOLDS)].index)
    todo = [(f, d) for f, d, _ in _iter_model_score_files(OUTPUTS_JEANZAY / a.dataset / "surprisal_sweep", by_layer_name)
            if _model_id_from_dir(d) not in done]
    print(f"[{a.dataset}] {len(todo)} models to process")
    B = max(a.n_jobs, 4)
    for i in range(0, len(todo), B):
        res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_process)(a.dataset, f, d, cost) for f, d in todo[i:i + B])
        new = [r for x in res if x for r in x]
        if new:
            frames.append(pd.DataFrame(new))
            pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"[{a.dataset}] {i + len(todo[i:i + B])}/{len(todo)}")
    print("wrote", out)


if __name__ == "__main__":
    main()
