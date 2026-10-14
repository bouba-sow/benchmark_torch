#!/usr/bin/env python3
"""Diagnostic: Tuckute surprisal RSA with different neural distances vs the surprisal (dLL) benchmark."""
from __future__ import annotations
import os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from scripts.analysis import surprisal_rsa as S
from scripts.analysis.tuckute_rsa_variants import neural_pairs
from neuroai.analysis.benchmark_common import lookup_model_meta

ids0, cube = neural_pairs(None)
valid = np.isfinite(cube).all(axis=(0, 2)); cube = cube[:, valid]; ids0 = np.array([i for i, v in zip(ids0, valid) if v])
nS, nN, nR = cube.shape
V = {"v0_corr6": [pdist(cube[s], "correlation") for s in range(nS)],
     "v2_scalar": [pdist(cube[s][:, [r]], "euclidean") for s in range(nS) for r in range(nR)],
     "v3_group_roi": [pdist(cube.mean(0)[:, [r]], "euclidean") for r in range(nR)],
     "v4_group_mean": [pdist(cube.mean((0, 2))[:, None], "euclidean")]}
bench = pd.read_csv(ROOT / "reports/rsa_summary/rsa/model_scores_dll.csv", dtype={"revision": str})
bench = bench[(bench.dataset == "tuckute") & (bench.track == "surprisal")][["family", "model_id", "revision", "benchmark_score", "rsa_pearson"]]
def one(r):
    try:
        v = S._load_neural_surprisal("tuckute", r.family, r.model_id, r.revision)
    except Exception as e:
        return None
    v = v.drop_duplicates("sentence_id").set_index("sentence_id").surprisal
    s = np.array([v.get(int(i), np.nan) for i in ids0])
    if not np.isfinite(s).all():
        return None
    md = pdist(s[:, None], "euclidean")
    row = dict(family=r.family, model_id=r.model_id, revision=r.revision, benchmark=r.benchmark_score, current=r.rsa_pearson)
    for k, lst in V.items():
        row[k] = float(np.mean([pearsonr(md, d).statistic for d in lst]))
    m = lookup_model_meta(r.model_id, r.family); row["n_params"] = m.n_params
    return row
with ThreadPoolExecutor(8) as ex:
    rows = [x for x in ex.map(one, bench.itertuples()) if x]
df = pd.DataFrame(rows); out = Path(os.environ.get("SCRATCH", "/tmp")) / "tuckute_rsa_variants"; out.mkdir(exist_ok=True, parents=True)
df.to_csv(out / "surprisal_variants.csv", index=False)
m = df.dropna(subset=["benchmark"])
for k in ["current", *V]:
    print(f"{k:14s} n={len(df)} mean={df[k].mean():.4f} r(bench)={pearsonr(m[k], m.benchmark)[0]:+.2f} r(size)={np.corrcoef(np.log10(df.n_params), df[k])[0,1]:+.2f}")
print("bench r(size)=", round(np.corrcoef(np.log10(m.n_params), m.benchmark)[0, 1], 2))
