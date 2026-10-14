#!/usr/bin/env python3
"""Diagnostic: behavioral surprisal RSA vs word length / frequency (partial RSA), per model.

Uses the saved surprisal rdm_cache (|d surprisal| on the sampled/full word pairs), the shared reading-time RDM and the
saved confound features. Output: $SCRATCH/surprisal_confound_diag.csv
"""
from __future__ import annotations
import os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist
S = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
SETS = {"provo": ["fp", "gp", "sp"], "meco": ["fp", "gp", "sp"], "geco": ["fp", "gp"], "onestop": ["fp", "gp", "sp"],
        "naturalstories": ["spr"], "ucl_reading": ["spr"], "zuco_reading": ["nr", "sr", "tsr"]}
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
def resid(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); return y - X @ beta

def feats(conf, ids, pairs):
    out = []
    for name in ("frequency_rdm.npz", "word_length_rdm.npz"):
        z = np.load(conf / name, allow_pickle=True)
        zi, zv = z["stimulus_ids"], np.asarray(z["feature_values"], float)
        if pairs is None:
            if not np.array_equal(zi, ids):
                raise ValueError("id mismatch " + str(conf))
            out.append(pdist(zv[:, None], "euclidean"))
        else:
            kz = zi[:, 0].astype(np.int64) * 1_000_000 + zi[:, 1]; o = np.argsort(kz)
            ki = ids[:, 0].astype(np.int64) * 1_000_000 + ids[:, 1]
            idx = np.clip(np.searchsorted(kz[o], ki), 0, len(kz) - 1); v = np.where(kz[o][idx] == ki, zv[o][idx], np.nan)
            out.append(np.abs(v[pairs[:, 0]] - v[pairs[:, 1]]))
    return out

rows = []
for ds, measures in SETS.items():
    for m in measures:
        base = ROOT / "outputs" / ds / "pearson_rsa"
        shared = np.load(base / "surprisal/rdm_cache" / m / "shared/behavior_rdm.npz", allow_pickle=True)
        r_full = np.asarray(shared["distances"], float)
        files = [p for p in (base / "surprisal/rdm_cache" / m).rglob("*.npz") if "shared" not in p.parts]
        cache = {}
        def one(p):
            z = np.load(p, allow_pickle=True)
            ids = z["stimulus_ids"]; pairs = z["pair_indices"] if "pair_indices" in z.files else None
            key = (ids.shape, None if pairs is None else pairs[:3].tobytes())
            if key not in cache:
                cache[key] = feats(base / "confounds" / m, ids, pairs)
            df, dl = cache[key]
            x = np.asarray(z["distances"], float)
            if pairs is None:
                r = r_full
            elif len(r_full) == len(pairs):
                r = r_full
            else:  # full condensed reading-time RDM, model cache holds sampled pairs
                n = len(ids); i = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64); j = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
                r = r_full[i * n - i * (i + 1) // 2 + (j - i - 1)]
            ok = np.isfinite(x) & np.isfinite(r) & np.isfinite(df) & np.isfinite(dl)
            x, r, df, dl = x[ok], r[ok], df[ok], dl[ok]
            X = np.column_stack([np.ones_like(df), df, dl])
            return dict(rsa=corr(x, r), r_freq=corr(x, df), r_len=corr(x, dl), rt_freq=corr(r, df), rt_len=corr(r, dl),
                        partial=corr(resid(x, X), resid(r, X)))
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(one, files))
        for p, r in zip(files, res):
            rel = p.relative_to(base / "surprisal/rdm_cache" / m).parts
            rows.append(dict(dataset=ds, measure=m, family=rel[0], model_dir=rel[1], file=rel[2], **r))
        print(ds, m, len(files), flush=True)
pd.DataFrame(rows).to_csv(S / "surprisal_confound_diag.csv", index=False)
