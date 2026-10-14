#!/usr/bin/env python3
"""Diagnostic: does the response transform (raw RT vs log RT vs rank) change the surprisal RDM-regression ranking?

Surprisal track, last layer, confounds = word length + frequency (the 'len_freq' variant), item-fold CV (5 repeats).
Compares each model's held-out dLL per pair with the LME benchmark dLL (reports/rsa_summary/rsa/model_scores_dll.csv).
Output: $SCRATCH/rdm_target_transform.csv
"""
from __future__ import annotations
import os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, skew
from neuroai.analysis.rdm_regression import RdmCvPlan, all_pairs
from scripts.analysis import behavioral_rsa as B
from scripts.analysis.surprisal_rsa import _load_behavior_surprisal, _tasks as surprisal_tasks

DATASET, MEASURES = sys.argv[1], sys.argv[2].split(",")
S = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
_, responses = B._prepare(DATASET)
bench = pd.read_csv(ROOT / "reports/rsa_summary/rsa/model_scores_dll.csv", dtype={"revision": str})
bench = bench[(bench.dataset == DATASET) & (bench.track == "surprisal")]
rows = []
for measure in MEASURES:
    shared = np.load(ROOT / f"outputs/{DATASET}/pearson_rsa/surprisal/rdm_cache/{measure.lower()}/shared/behavior_rdm.npz")
    ids = np.asarray(shared["stimulus_ids"]); n = len(ids)
    items = responses[measure].set_index(["story_id", "zone"])
    rt = np.array([items.rt.loc[(int(s), int(z))] for s, z in ids], float)
    print(measure, "items", n, "rt: min", rt.min(), "median", np.median(rt), "max", rt.max(), "skew", round(float(skew(rt)), 2), "frac<=0", float((rt <= 0).mean()), flush=True)
    conf = np.column_stack([np.load(ROOT / f"outputs/{DATASET}/pearson_rsa/confounds/{measure.lower()}/{nm}_rdm.npz")["distances"] for nm in ("word_length", "frequency")])
    pairs = all_pairs(n)
    transforms = {"raw": rt, "log": np.log(np.maximum(rt, 1e-3)), "rank": rankdata(rt)}
    plans = {k: RdmCvPlan(np.abs(v[pairs[:, 0]] - v[pairs[:, 1]]), n, None, conf, 5, seed=0) for k, v in transforms.items()}
    sb = bench[bench.measure.str.upper() == measure][["family", "model_id", "revision", "benchmark_score"]]
    tasks = surprisal_tasks(DATASET, tuple(MEASURES))
    def one(t):
        fam, mid, rev = t
        v = _load_behavior_surprisal(DATASET, fam, mid, rev, measure)
        if v is None:
            return None
        m = pd.DataFrame(ids, columns=["story_id", "zone"]).merge(v, on=["story_id", "zone"], how="left")
        if m.surprisal.isna().any():
            return None
        s = m.surprisal.to_numpy(float)
        x = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
        r = dict(family=fam, model_id=mid, revision=rev)
        for k, p in plans.items():
            r[k] = p.delta_ll(x).delta_ll_per_obs
        return r
    with ThreadPoolExecutor(4) as ex:
        res = [r for r in ex.map(one, tasks) if r]
    d = pd.DataFrame(res).merge(sb, on=["family", "model_id", "revision"])
    for k in transforms:
        rows.append(dict(dataset=DATASET, measure=measure, transform=k, n=len(d), frac_neg=float((d[k] < 0).mean()),
                         mean_dll=float(d[k].mean()), spearman_vs_lme=float(spearmanr(d[k], d.benchmark_score).statistic)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(S / f"rdm_target_transform_{DATASET}.csv", index=False)
