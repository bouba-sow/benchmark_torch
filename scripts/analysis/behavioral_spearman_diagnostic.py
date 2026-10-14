#!/usr/bin/env python3
"""Diagnostic: Spearman (rank) behavioral embedding RSA, raw vs standardized, on the saved sampled pairs."""
from __future__ import annotations
import os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import rankdata
S = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
SETS = {"geco": ["fp", "gp"], "onestop": ["fp", "gp", "sp"], "naturalstories": ["spr"],
        "ucl_reading": ["spr"], "zuco_reading": ["nr", "sr", "tsr"]}
def one(p):
    z = np.load(p, allow_pickle=True)
    m, r = np.asarray(z["model_distances"], float), np.asarray(z["response_distances"], float)
    ok = np.isfinite(m) & np.isfinite(r)
    return float(np.corrcoef(rankdata(m[ok]), rankdata(r[ok]))[0, 1])
rows = []
for ds, ms in SETS.items():
    for m in ms:
        for variant, base in [("raw", S / "rsa_std_backup_old" / ds / "rdm_cache_old"), ("std", ROOT / "outputs" / ds / "pearson_rsa" / "embedding" / "rdm_cache")]:
            files = [p for p in (base / m).rglob("*.npz") if "shared" not in p.parts and p.name != "reading_time_rdm.npz"]
            with ThreadPoolExecutor(8) as ex:
                res = list(ex.map(one, files))
            for p, r in zip(files, res):
                rel = p.relative_to(base / m).parts
                rows.append(dict(dataset=ds, measure=m, variant=variant, family=rel[0], model_dir=rel[1], file=rel[2], spearman=r))
            print(ds, m, variant, len(files), flush=True)
pd.DataFrame(rows).to_csv(S / "behavioral_spearman_diag.csv", index=False)
