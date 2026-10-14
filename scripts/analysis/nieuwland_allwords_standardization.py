#!/usr/bin/env python3
"""Nieuwland: standardize the critical-noun states with mean/std over ALL words of the stimulus sentences.

Compares per model (embedding track, last layer) three geometries of the model RDM (1 - Pearson):
  raw       : noun states as extracted
  std_items : per-feature z-score over the 160 noun states (current production setting)
  std_words : (x - mu) / sigma with mu, sigma over every word of every sentence (scripts/features/sentence_word_stats.py)
Scores: Pearson RSA and held-out RDM-regression delta LL (same plan as embedding_geometry_comparison.py), plus the
rank agreement between geometries and the agreement with the ridge benchmark. Output: $SCRATCH/geometry_comparison/nieuwland_allwords.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/run/embedding/neural"))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from neuroai.benchmarks.eeg_erp import _feature_cache_path, _items_hash
from scripts.analysis import n400_item_mean_rsa as M
from scripts.analysis import n400_paper_rsa as P
from scripts.analysis.embedding_geometry_comparison import build

SCR = Path(os.environ["SCRATCH"])
STATS = SCR / "sentence_word_stats/nieuwland"


def main():
    src, plan, resp = build("nieuwland")
    ih = _items_hash(src.items["sentence_context"].tolist(), src.items["noun"].tolist())
    bench = pd.read_csv(ROOT / "reports/rsa_summary/rsa/model_scores_correlation.csv")
    bench = bench[(bench.dataset == "nieuwland") & (bench.track == "embedding")]
    rows = []
    for family, task in P.registry_tasks():
        rev = task.revision or "main"
        sp = STATS / family / task.model_dir_safe / f"{rev.replace('/', '_')}.npz"
        cache = _feature_cache_path(M.NIEUWLAND_CACHE, task.load_model_id, None if rev == "main" else rev, "all", ih)
        if not (sp.is_file() and cache.is_file()):
            continue
        f = np.load(cache)["features"]
        x = (f if f.ndim == 2 else f[:, -1, :]).astype(np.float64)
        z = np.load(sp)
        sd = np.where(z["sigma"] <= 1e-12, 1.0, z["sigma"])
        geoms = {"raw": x, "std_items": (x - x.mean(0)) / (x.std(0) + 1e-8), "std_words": (x - z["mu"]) / sd}
        for mode, g in geoms.items():
            d = pdist(g, metric="correlation")
            if not np.isfinite(d).all() or d.std() == 0:
                continue
            r = plan.delta_ll(d)
            rows.append(dict(family=family, model_id=task.model_id, revision=rev, mode=mode, rsa_pearson=plan.pearson(d),
                             delta_ll=r.delta_ll_per_obs, noun_rel_diff=float(z["noun_rel_diff"])))
    d = pd.DataFrame(rows)
    out = SCR / "geometry_comparison"
    d.to_csv(out / "nieuwland_allwords.csv", index=False)
    d = d.merge(bench[["family", "model_id", "revision", "benchmark_score"]], on=["family", "model_id", "revision"], how="left")
    print(f"{d.model_id.nunique()} models; max noun recompute rel diff {d.noun_rel_diff.max():.2e}")
    for mode in ("raw", "std_items", "std_words"):
        g = d[d["mode"] == mode]
        print(f"{mode:9s} median RSA {g.rsa_pearson.median():+.4f}  median dLL {g.delta_ll.median():+.2e}  frac dLL>0 {(g.delta_ll > 0).mean():.2f}  "
              f"rho(RSA, ridge) {spearmanr(g.rsa_pearson, g.benchmark_score, nan_policy='omit').statistic:+.2f}  "
              f"rho(dLL, ridge) {spearmanr(g.delta_ll, g.benchmark_score, nan_policy='omit').statistic:+.2f}")
    for col in ("rsa_pearson", "delta_ll"):
        w = d.pivot_table(index=["family", "model_id", "revision"], columns="mode", values=col)
        print(f"model-rank agreement ({col}):\n{w.corr(method='spearman').round(2).to_string()}")


if __name__ == "__main__":
    main()
