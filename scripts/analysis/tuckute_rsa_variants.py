#!/usr/bin/env python3
"""Compare Tuckute neural-RDM definitions against the ridge benchmark (diagnostic).

Variants (neural distance between two B sentences; model RDM is always 1 - Pearson of last-layer states):
  v0_corr6      per subject, 1 - Pearson across the 6 ROIs (current RSA)
  v1_euc6       per subject, Euclidean across the 6 ROIs
  v2_scalar     per (subject, ROI) |delta|, scores averaged over the 60 targets (mirrors the ridge)
  v3_group_roi  subject-averaged per ROI |delta|, scores averaged over the 6 ROIs
  v4_group_mean subject- and ROI-averaged scalar |delta|
Writes to --out (default $SCRATCH/tuckute_rsa_variants), never to outputs/.
"""
from __future__ import annotations
import argparse, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from scripts.analysis import other_neural_rsa as O
from neuroai.analysis.benchmark_common import lookup_model_meta
from neuroai.datasets.tuckute import TuckuteDataset
from neuroai.sweep import safe_model_dir

def neural_pairs(selected_ids):
    ds = TuckuteDataset(ROOT / "data/tuckute_fmri/osfstorage")
    _, _, all_ids = ds.get_condition("all")
    sel = {int(i): k for k, i in enumerate(O.np.asarray(all_ids)[ds.condition_mask("B")])}
    per = ds.get_condition_per_subject("B")
    subs = sorted(per)
    ids0 = [int(i) for i in per[subs[0]][2]]
    cube = []
    for s in subs:
        _, bold, ids = per[s]
        order = {int(i): k for k, i in enumerate(ids)}
        cube.append(np.stack([bold[order[i]] for i in ids0]))
    cube = np.stack(cube)                      # (subj, sent, roi)
    return ids0, cube

def run(args):
    ids0, cube = neural_pairs(None)
    nS, nN, nR = cube.shape
    valid = np.isfinite(cube).all(axis=(0, 2))
    cube = cube[:, valid]; ids0 = [i for i, v in zip(ids0, valid) if v]
    print("neural cube", cube.shape, flush=True)
    tri = np.triu_indices(cube.shape[1], 1)
    def flat(x, metric):  # x: (sent, feat)
        return pdist(x, metric=metric)
    V = {"v0_corr6": [flat(cube[s], "correlation") for s in range(nS)],
         "v1_euc6": [flat(cube[s], "euclidean") for s in range(nS)],
         "v2_scalar": [flat(cube[s][:, [r]], "euclidean") for s in range(nS) for r in range(nR)],
         "v3_group_roi": [flat(cube.mean(0)[:, [r]], "euclidean") for r in range(nR)],
         "v4_group_mean": [flat(cube.mean((0, 2))[:, None], "euclidean")]}
    ds = TuckuteDataset(ROOT / "data/tuckute_fmri/osfstorage")
    sentences, selected, _ = O._prepare_tuckute(ROOT / "data/tuckute_fmri/osfstorage")
    _, _, all_ids = ds.get_condition("all")
    pos = {int(i): k for k, i in enumerate(all_ids)}
    rows_idx = np.array([pos[i] for i in ids0])
    enc_root = ROOT / "outputs_jeanzay/tuckute/embedding_sweep"
    cache_root = enc_root / ".embed_cache"
    tasks = O._tasks("tuckute", enc_root)
    def one(task):
        family, model_id, revision = task
        path = cache_root / safe_model_dir(model_id) / revision.replace("/", "_") / "embeddings.npz"
        if not path.exists():
            return None
        states = O._load_tuckute(path, sentences)[rows_idx]
        z = (states - states.mean(0)) / (states.std(0) + 1e-8)
        MD = {"m0": pdist(states, metric="correlation"), "m1z": pdist(z, metric="correlation")}
        row = dict(family=family, model_id=model_id, revision=revision)
        for mk, md in MD.items():
            for k, lst in V.items():
                row[f"{mk}_{k}"] = float(np.mean([pearsonr(md, d).statistic for d in lst]))
        try:
            row["encoding_pearson"] = O._encoding_score("tuckute", enc_root / family / safe_model_dir(model_id), revision)
        except Exception:
            row["encoding_pearson"] = np.nan
        m = lookup_model_meta(model_id, family); row["group"] = m.group; row["n_params"] = m.n_params
        return row
    with ThreadPoolExecutor(args.workers) as ex:
        out = [r for r in ex.map(one, tasks) if r]
    df = pd.DataFrame(out)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "variants.csv", index=False)
    m = df.dropna(subset=["encoding_pearson"])
    for k in [c for c in df.columns if c.startswith(("m0_", "m1z_"))]:
        print(f"{k:18s} n={len(df)} mean={df[k].mean():.4f} r(bench)={pearsonr(m[k], m.encoding_pearson)[0]:+.2f} "
              f"r(size)={np.corrcoef(np.log10(df.n_params), df[k])[0,1]:+.2f}", flush=True)
    print("bench r(size)=", round(np.corrcoef(np.log10(m.n_params), m.encoding_pearson)[0,1], 2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path(os.environ.get("SCRATCH", "/tmp")) / "tuckute_rsa_variants3")
    run(p.parse_args())
