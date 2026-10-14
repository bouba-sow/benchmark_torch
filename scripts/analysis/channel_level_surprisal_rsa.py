#!/usr/bin/env python3
"""Surprisal RSA for Broderick / MEG-MASC with per-channel neural patterns (see channel_level_word_rsa.py).

|d surprisal| (final layer, as in surprisal_rsa.py) vs 1-Pearson distance across all channels of the window-mean
response, per subject, averaged over subjects. Output: <out-root>/model_summary.csv
"""
from __future__ import annotations
import argparse, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from scripts.analysis import surprisal_rsa as S
from scripts.analysis.channel_level_word_rsa import patterns
from neuroai.analysis.benchmark_common import lookup_model_meta

def score(task, dataset, neural):
    family, model_id, revision = task
    v = S._load_neural_surprisal(dataset, family, model_id, revision)
    if v is None:
        return None
    v = v.copy(); v["w"] = v.stimulus_id.str.rsplit("::", n=1).str[1].astype(np.int64)
    sw = v.groupby("w").surprisal.mean()
    scores = []
    for subject, (ids, vals) in neural.items():
        ok = np.isin(ids, sw.index.to_numpy())
        ids2, vals2 = ids[ok], vals[ok]
        if len(ids2) < 3:
            continue
        s = sw.reindex(ids2).to_numpy(float)
        if len(ids2) * (len(ids2) - 1) // 2 > S.MAX_PAIRS:
            md, _, pairs = S._pair_distances(s, s, dataset, subject)
            nd = S._distances_for_pairs(vals2, pairs, "correlation")
        else:
            md, nd = pdist(s[:, None]), pdist(vals2, "correlation")
        scores.append(float(pearsonr(md, nd).statistic))
    m = lookup_model_meta(model_id, family)
    return dict(family=family, model_id=model_id, revision=revision, n_subjects=len(scores),
                rsa_pearson=float(np.mean(scores)), group=m.group, n_params=m.n_params)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["broderick", "megmasc"])
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    neural = patterns(a.dataset)
    tasks = S._tasks(a.dataset, ("",))
    print(a.dataset, len(tasks), "models", flush=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(score, t, a.dataset, neural) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            if r:
                rows.append(r); print(r["model_id"], round(r["rsa_pearson"], 4), flush=True)
    a.out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out_root / "model_summary.csv", index=False)
