#!/usr/bin/env python3
"""Podcast per-DLT-bin scores normalised by the BIN's own split-half ceiling, per region.

Per model, bin b and region r: m_{r,b} = mean over the held-out (subject, word) rows of the bin and lags 7-25 of
(ll_full - ll_base) (per-word predictions of podcast_region_fit.py); C_{r,b} = mean over lags 7-25 of the split-half
ceiling of the bin (outputs/noise_ceilings/dll_bins/podcast_region/, scripts/ceilings/dll_podcast_region_bins.py).
Score = sum_r n_{r,b} (m_{r,b} / C_{r,b}) / sum_r n_{r,b}  (n = rows of the bin in region r) -- the per-region
normalisation of the headline, with the bin's ceiling instead of the all-words one.
Writes outputs/dlt_loglik_cache/podcast_lme_binceil.csv (columns: model, family, n_params, dlt_bin,
hard_gt_threshold, norm, n_words_unique). Bins: easy (cost 0), hard cost >= 1..5.

    .venv/bin/python scripts/dlt/podcast_bin_ceiling_scores.py --n_jobs 18
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, OUTPUTS_JEANZAY, REPO_ROOT
from neuroai.benchmarks.base import word_predictions_root
from scripts.summarize.benchmark import _model_id_from_dir, lookup_model_meta

SWEEP = OUTPUTS_JEANZAY / "podcast/surprisal_region_sweep"
CEIL_DIR = REPO_ROOT / "outputs/noise_ceilings/dll_bins/podcast_region"
COST = REPO_ROOT / "outputs/dlt/podcast_dlt_cost.csv"
OUT = REPO_ROOT / "outputs/dlt_loglik_cache/podcast_lme_binceil.csv"
LAGS = range(7, 26)
BINS = {"easy": ("easy", 1.0, lambda c: c <= 0)} | {f"hard{k}": ("hard", float(k), (lambda k: lambda c: c >= k)(k))
                                                    for k in range(1, 6)}


def _ceilings() -> dict[tuple[str, str], float]:
    acc: dict[tuple[str, str], list[float]] = {}
    for f in sorted(CEIL_DIR.glob("*.json")):
        d = json.load(open(f))
        if d["lag_index"] not in LAGS:
            continue
        for r, v in d["per_region"].items():
            for b in BINS:
                acc.setdefault((r, b), []).append(v[b]["ceiling_delta_loglik_per_obs"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _one(model_dir: Path, family: str, cost: pd.Series, ceil: dict) -> list[dict] | None:
    pdir = word_predictions_root(model_dir) / "region_predictions"
    sums: dict[tuple[str, str], list] = {}
    words: dict[str, set] = {b: set() for b in BINS}
    for lag in LAGS:
        for f in glob.glob(str(pdir / f"*_lag{lag:02d}.parquet")):
            region = Path(f).name.split("_lag")[0]
            x = pd.read_parquet(f, columns=["word_idx", "ll_base", "ll_full"])
            d = (x["ll_full"].astype("float64") - x["ll_base"]).to_numpy()
            c = x["word_idx"].map(cost).to_numpy(float)
            for b, (_, _, fn) in BINS.items():
                m = fn(c) & np.isfinite(d)
                s = sums.setdefault((region, b), [0.0, 0])
                s[0] += float(d[m].sum()); s[1] += int(m.sum())
                words[b].update(x["word_idx"].to_numpy()[m].tolist())
    if not sums:
        return None
    model_id = _model_id_from_dir(model_dir)
    meta = lookup_model_meta(model_id, family, n_params_fallback=None)
    rows = []
    for b, (bn, thr, _) in BINS.items():
        parts = [(sums[(r, b)][0] / sums[(r, b)][1] / ceil[(r, b)], sums[(r, b)][1])
                 for r in {k[0] for k in sums} if sums[(r, b)][1] > 0 and ceil[(r, b)] > 0]
        norm = sum(v * n for v, n in parts) / sum(n for _, n in parts) if parts else float("nan")
        rows.append({"model": model_id, "family": family, "n_params": meta.n_params, "dlt_bin": bn,
                     "hard_gt_threshold": thr, "norm": norm, "n_words_unique": len(words[b])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=8)
    a = ap.parse_args()
    cost = pd.read_csv(COST).drop_duplicates("word_idx").set_index("word_idx")["total"]
    ceil = _ceilings()
    dirs = [(Path(d), Path(d).parent.name) for d in sorted(glob.glob(str(SWEEP / "*/*"))) if Path(d, "podcast_region_lme_by_lag.csv").exists()]
    res = Parallel(n_jobs=a.n_jobs)(delayed(_one)(d, f, cost, ceil) for d, f in dirs)
    out = pd.DataFrame([r for x in res if x for r in x])
    out = out[~out["model"].isin(EXCLUDED_MODEL_IDS)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {out['model'].nunique()} models")


if __name__ == "__main__":
    main()
