#!/usr/bin/env python3
"""Evolution table of embedding-RSA scores: original -> (N400 item-mean) -> standardized features.

Reads the backups in $SCRATCH/rsa_std_backup_old (and tuckute_rsa_backup_old) plus the current
outputs/. For every stage: mean RSA, r(RSA, ridge Pearson) and r(RSA, log10 params) over models.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
S = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
B, O = S / "rsa_std_backup_old", ROOT / "outputs"

# One fixed benchmark per dataset/measure: the comparator used by build_rsa_report.py (embedding track),
# so every stage (original / N400 item-mean / standardized) is compared with the same ridge score.
SC = pd.read_csv(ROOT / "reports/rsa_summary/rsa/model_scores_correlation.csv", dtype={"revision": str})
SC = SC[SC.track == "embedding"]
KEYS = ["family", "model_id", "revision"]
CUR = {}

def stat(path, bench=None):
    if not Path(path).is_file():
        return (np.nan,) * 3
    d = pd.read_csv(path, dtype={"revision": str})
    ref = SC[(SC.dataset == CUR["ds"]) & (SC.measure.fillna("") == CUR["measure"])][KEYS + ["benchmark_score"]]
    m = d.drop(columns=["benchmark_score"], errors="ignore").merge(ref, on=KEYS)
    m = m.dropna(subset=["benchmark_score"])
    bench = "benchmark_score"
    r = np.corrcoef(m.rsa_pearson, m[bench])[0, 1] if len(m) > 3 else np.nan
    sz = np.corrcoef(np.log10(d.n_params), d.rsa_pearson)[0, 1]
    return d.rsa_pearson.mean(), r, sz

rows = []
def add(name, original, final, mid=None, bench=None, ds=None, measure=""):
    CUR.update(ds=ds, measure=measure)
    o, f = stat(original, bench), stat(final, bench)
    m = stat(mid, bench) if mid else (np.nan,) * 3
    rows.append(dict(dataset=name, mean_orig=o[0], r_ridge_orig=o[1], r_size_orig=o[2],
                     mean_item=m[0], r_ridge_item=m[1], r_size_item=m[2],
                     mean_final=f[0], r_ridge_final=f[1], r_size_final=f[2]))

E_ = lambda ds: O / ds / "pearson_rsa" / "embedding"
add("Pereira fMRI", B/"pereira/model_summary.csv", O/"pereira/pearson_rsa/model_summary.csv", ds="pereira")
add("Tuckute B fMRI", S/"tuckute_rsa_backup_old/model_summary.csv", O/"tuckute/pearson_rsa/model_summary.csv", ds="tuckute")
add("Fedorenko ECoG", B/"fedorenko2016/model_summary.csv", O/"fedorenko2016/pearson_rsa/model_summary.csv", ds="fedorenko2016")
add("Broderick EEG", B/"broderick/model_summary.csv", E_("broderick")/"model_summary.csv", ds="broderick")
add("MEG-MASC MEG", B/"megmasc/model_summary.csv", E_("megmasc")/"model_summary.csv", ds="megmasc")
for name, ds in [("Nieuwland N400", "nieuwland"), ("Federmeier 2007 N400", "szewczyk_f2007"), ("Hubbard 2019 N400", "szewczyk_h2019")]:
    add(name, E_(ds)/"model_summary_per_subject.csv", E_(ds)/"model_summary.csv", mid=E_(ds)/"model_summary_unstandardized.csv", ds=ds)
add("Podcast ECoG (lag-mean)", B/"podcast/model_summary.csv", O/"podcast/pearson_rsa/embedding/lagwise/model_summary.csv", ds="podcast")
for ds, ms, sub in [("provo", ["fp", "gp", "sp"], ""), ("meco", ["fp", "gp", "sp"], ""),
                    ("geco", ["fp", "gp"], "embedding"), ("onestop", ["fp", "gp", "sp"], "embedding"),
                    ("naturalstories", ["spr"], "embedding"), ("ucl_reading", ["spr"], "embedding"),
                    ("zuco_reading", ["nr", "sr", "tsr"], "embedding")]:
    for m in ms:
        add(f"{ds} {m.upper()}", B/ds/m/"model_summary.csv", O/ds/"pearson_rsa"/sub/m/"model_summary.csv", ds=ds, measure=m.upper())
df = pd.DataFrame(rows)
out = ROOT / "reports/rsa_summary/rsa"
df.to_csv(out / "standardization_evolution.csv", index=False)
f = lambda x: "–" if pd.isna(x) else f"{x:+.2f}"
g = lambda x: "–" if pd.isna(x) else f"{x:.3f}"
lines = ["| Dataset | mean RSA orig → final | r vs ridge: orig | N400 item-mean | final | r vs size: orig | N400 item-mean | final |",
         "|---|---|---|---|---|---|---|---|"]
for r in df.itertuples():
    lines.append(f"| {r.dataset} | {g(r.mean_orig)} → {g(r.mean_final)} | {f(r.r_ridge_orig)} | {f(r.r_ridge_item)} | {f(r.r_ridge_final)} | "
                 f"{f(r.r_size_orig)} | {f(r.r_size_item)} | {f(r.r_size_final)} |")
(out / "standardization_evolution.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
