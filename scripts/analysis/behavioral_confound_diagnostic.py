#!/usr/bin/env python3
"""Diagnostic: how much of behavioral embedding RSA is word length / frequency, raw vs standardized.

For each dataset/measure with a saved embedding rdm_cache, and every model, using the model's own sampled word pairs:
  rsa      = corr(model RDM, reading-time RDM)
  r_freq/r_len = corr(model RDM, |d log-frequency| / |d word length|)
  partial  = corr of residuals after regressing |d freq| and |d len| out of both RDMs
Raw RDMs come from the backup (rdm_cache_old), standardized from outputs/. Output: $SCRATCH/behavioral_confound_diag.csv
"""
from __future__ import annotations
import os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd

S = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
SETS = {"geco": ["fp", "gp"], "onestop": ["fp", "gp", "sp"], "naturalstories": ["spr"],
        "ucl_reading": ["spr"], "zuco_reading": ["nr", "sr", "tsr"]}

def key(ids):
    ids = np.asarray(ids, np.int64)
    return ids[:, 0] * 1_000_000 + ids[:, 1]

def feat_lookup(path):
    z = np.load(path, allow_pickle=True)
    k, v = key(z["stimulus_ids"]), np.asarray(z["feature_values"], float)
    o = np.argsort(k)
    return k[o], v[o]

def lookup(table, ids):
    k, v = table
    kk = key(ids)
    idx = np.searchsorted(k, kk)
    idx = np.clip(idx, 0, len(k) - 1)
    ok = k[idx] == kk
    return np.where(ok, v[idx], np.nan)

def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])

def resid(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta

def one(args):
    path, ft = args
    z = np.load(path, allow_pickle=True)
    ids, pairs = z["stimulus_ids"], z["pair_indices"]
    fa, fb = lookup(ft["freq"], ids[pairs[:, 0]]), lookup(ft["freq"], ids[pairs[:, 1]])
    la, lb = lookup(ft["len"], ids[pairs[:, 0]]), lookup(ft["len"], ids[pairs[:, 1]])
    df, dl = np.abs(fa - fb), np.abs(la - lb)
    m, r = np.asarray(z["model_distances"], float), np.asarray(z["response_distances"], float)
    ok = np.isfinite(df) & np.isfinite(dl) & np.isfinite(m) & np.isfinite(r)
    m, r, df, dl = m[ok], r[ok], df[ok], dl[ok]
    X = np.column_stack([np.ones_like(df), df, dl])
    return dict(rsa=corr(m, r), r_freq=corr(m, df), r_len=corr(m, dl), rt_freq=corr(r, df), rt_len=corr(r, dl),
                partial=corr(resid(m, X), resid(r, X)), frac_pairs=ok.mean())

rows = []
sc = pd.read_csv(ROOT / "reports/rsa_summary/rsa/model_scores_correlation.csv", dtype={"revision": str})
sc = sc[sc.track == "embedding"]
for ds, measures in SETS.items():
    for m in measures:
        conf = ROOT / "outputs" / ds / "pearson_rsa" / "confounds" / m
        ft = {"freq": feat_lookup(conf / "frequency_rdm.npz"), "len": feat_lookup(conf / "word_length_rdm.npz")}
        for variant, base in [("raw", S / "rsa_std_backup_old" / ds / "rdm_cache_old"),
                              ("std", ROOT / "outputs" / ds / "pearson_rsa" / "embedding" / "rdm_cache")]:
            files = [p for p in (base / m).rglob("*.npz") if "shared" not in p.parts and p.name != "reading_time_rdm.npz"]
            with ThreadPoolExecutor(8) as ex:
                res = list(ex.map(one, [(p, ft) for p in files]))
            for p, r in zip(files, res):
                rel = p.relative_to(base / m).parts
                rows.append(dict(dataset=ds, measure=m, variant=variant, family=rel[0], model_dir=rel[1], file=rel[2], **r))
            print(ds, m, variant, len(files), flush=True)
out = pd.DataFrame(rows); out.to_csv(S / "behavioral_confound_diag.csv", index=False)
