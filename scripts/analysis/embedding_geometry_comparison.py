#!/usr/bin/env python3
"""Compare embedding geometries for the RDM analyses on small N400 datasets (Nieuwland 160 items, Hubbard 2019 192 items).

Modes (model RDM = 1 - Pearson between item vectors after the transform):
  raw   : last-layer states as extracted
  std   : per-feature z-score across the scored items (the current production setting, Timkey & van Schijndel 2021 style)
  rank  : each item vector replaced by its ranks over the dimensions (= Spearman correlation between two embeddings;
          Zhelezniak et al. 2019, the rank option discussed by Timkey & van Schijndel)
Scores per model and mode: plain Pearson RSA and Spearman RSA on the item-mean N400 RDM, and held-out RDM-regression
delta LL with the same nuisance RDMs as scripts/analysis/rdm_regression_n400_units.py. Output: $SCRATCH/geometry_comparison.
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
from scipy.stats import rankdata, spearmanr

from neuroai.analysis.rdm_regression import RdmCvPlan
from neuroai.datasets.szewczyk_paper import TSV, load_study
from neuroai.features.word_frequency import compute_wlen_wordfreq_covariates
from scripts.analysis import n400_item_mean_rsa as M
from scripts.analysis import n400_paper_rsa as P

SCR = Path(os.environ["SCRATCH"])
MODES = {
    "raw": lambda s: s,
    "std": lambda s: (s - s.mean(0)) / (s.std(0) + 1e-8),
    "rank": lambda s: rankdata(s, axis=1).astype(np.float32),
}


def build(name):
    src = M.Nieuwland() if name == "nieuwland" else M.Szewczyk(name)
    words = (src.items.noun if name == "nieuwland" else src.items.critical_word).astype(str).tolist()
    lex = compute_wlen_wordfreq_covariates(words).astype(np.float64)
    if name != "nieuwland":
        extra = load_study(ROOT / TSV, name).groupby("item")[["pos_start", "old20", "concr"]].first()
        lex = np.column_stack([lex, extra.loc[src.items.item].to_numpy(np.float64)])
    n = len(src.y)
    iu = np.triu_indices(n, 1)
    conf = np.column_stack([np.abs(lex[iu[0], j] - lex[iu[1], j]) for j in range(lex.shape[1])])
    y = (src.y - src.y.mean()) / src.y.std()
    resp = np.abs(y[iu[0]] - y[iu[1]])
    return src, RdmCvPlan(resp, n, None, conf, 10, seed=0), resp


def main():
    out = SCR / "geometry_comparison"
    out.mkdir(exist_ok=True)
    bench = pd.read_csv(ROOT / "reports/rsa_summary/rsa/model_scores_correlation.csv")
    for name, ds in (("nieuwland", "nieuwland"), ("h2019", "szewczyk_h2019")):
        src, plan, resp = build(name)
        files = list((ROOT / "outputs_jeanzay" / src.dataset / "embedding_sweep").glob("*/*/*item_mean_scores.csv"))
        keys = set()
        for p in files:
            t = pd.read_csv(p, dtype={"revision": str})
            keys.update(t[["family", "model_id", "revision"]].itertuples(index=False, name=None))
        tasks = [(f, t) for f, t in P.registry_tasks() if (f, t.model_id, t.revision or "main") in keys]
        rows = []
        for family, task in tasks:
            for mode, fn in MODES.items():
                M._z = fn
                dist = src.embedding(family, task)
                if dist is None or not np.isfinite(dist).all() or np.std(dist) == 0:
                    continue
                r = plan.delta_ll(dist)
                rows.append(dict(dataset=ds, family=family, model_id=task.model_id, revision=task.revision or "main",
                                 mode=mode, rsa_pearson=plan.pearson(dist), rsa_spearman=float(spearmanr(dist, resp).statistic),
                                 delta_ll=r.delta_ll_per_obs))
        d = pd.DataFrame(rows)
        d.to_csv(out / f"{ds}_geometry.csv", index=False)
        b = bench[(bench.dataset == ds) & (bench.track == "embedding")][["family", "model_id", "revision", "benchmark_score"]]
        d = d.merge(b, on=["family", "model_id", "revision"], how="left")
        print(f"\n== {ds}: {d.model_id.nunique()} models")
        wide = d.pivot_table(index=["family", "model_id", "revision"], columns="mode", values=["rsa_pearson", "delta_ll"])
        for mode in MODES:
            g = d[d["mode"] == mode]
            print(f"{mode:5s} median pearson RSA {g.rsa_pearson.median():+.4f}  median spearman RSA {g.rsa_spearman.median():+.4f}  "
                  f"median dLL {g.delta_ll.median():+.2e}  frac dLL>0 {(g.delta_ll > 0).mean():.2f}  "
                  f"rho(pearson RSA, ridge) {spearmanr(g.rsa_pearson, g.benchmark_score, nan_policy='omit').statistic:+.2f}  "
                  f"rho(dLL, ridge) {spearmanr(g.delta_ll, g.benchmark_score, nan_policy='omit').statistic:+.2f}")
        for col in ("rsa_pearson", "delta_ll"):
            print(f"rank agreement between modes ({col}, Spearman across models):")
            print(wide[col].corr(method="spearman").round(2).to_string())


if __name__ == "__main__":
    main()
