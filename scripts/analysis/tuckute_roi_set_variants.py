#!/usr/bin/env python3
"""Diagnostic: Tuckute embedding RSA with different ROI sets as the per-sentence neural pattern.

Standardized last-layer embeddings (model RDM = 1 - Pearson of z-scored states), per-subject neural RDM =
1 - Pearson across the chosen ROIs, averaged over the 10 subjects. Benchmark = the ridge Pearson r (6 LH language ROIs).
Source: data_SI brain-allroi-data (432 ROIs). Writes to $SCRATCH/tuckute_roi_sets.
"""
from __future__ import annotations
import os, sys
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

S = Path(os.environ.get("SCRATCH", "/tmp"))
df = pd.read_csv(S / "tuckute_si/data_SI/brain-allroi-data_participant_20231001.csv", usecols=["roi", "response_target", "item_id", "target_UID", "cond"])
print(df.cond.unique(), flush=True)
df = df[df.cond == "B"]
rois = sorted(df.roi.unique())
SETS = {"lang6_LH": ["lang_LH_netw", "lang_LH_IFGorb", "lang_LH_IFG", "lang_LH_MFG", "lang_LH_AntTemp", "lang_LH_PostTemp"],
        "lang_sub_LHRH": [r for r in rois if r.startswith("lang_") and not r.endswith("netw")],
        "lang_all": [r for r in rois if r.startswith("lang_")],
        "glasser180_LH": [r for r in rois if r.startswith("anatglasser_LH_")],
        "lang+md+dmn": [r for r in rois if r.startswith(("lang_", "md_", "dmn_")) and "LHRH" not in r]}
subjects = sorted(df.target_UID.unique())
ids = np.array(sorted(set.intersection(*[set(df[df.target_UID == s].item_id) for s in subjects])))
wide = {}
for s in subjects:
    w = df[df.target_UID == s].pivot_table(index="item_id", columns="roi", values="response_target", aggfunc="mean").reindex(ids)
    wide[s] = w
V = {}
common = set.intersection(*[set(w.columns[w.notna().any()]) for w in wide.values()])
for name, cols in list(SETS.items()):
    cols = [c for c in cols if c in common]; SETS[name] = cols
    lst = []
    for s in subjects:
        x = wide[s][cols].to_numpy(float)
        ok = np.isfinite(x).all(1)
        lst.append((ok, x))
    ok = np.all([o for o, _ in lst], axis=0)
    V[name] = [pdist(x[ok], "correlation") for _, x in lst]
    V[name + "_ok"] = ok
    print(name, len(cols), "rois", int(ok.sum()), "sentences", flush=True)
ds = TuckuteDataset(ROOT / "data/tuckute_fmri/osfstorage")
sentences, selected, _ = O._prepare_tuckute(ROOT / "data/tuckute_fmri/osfstorage")
_, _, all_ids = ds.get_condition("all")
pos = {int(i): k for k, i in enumerate(all_ids)}
enc_root = ROOT / "outputs_jeanzay/tuckute/embedding_sweep"; cache_root = enc_root / ".embed_cache"
def one(task):
    family, model_id, revision = task
    path = cache_root / safe_model_dir(model_id) / revision.replace("/", "_") / "embeddings.npz"
    if not path.exists():
        return None
    states = O._load_tuckute(path, sentences)
    row = dict(family=family, model_id=model_id, revision=revision)
    for name in SETS:
        sel = np.array([pos[int(i)] for i in ids[V[name + "_ok"]]])
        z = states[sel]; z = (z - z.mean(0)) / (z.std(0) + 1e-8)
        md = pdist(z, "correlation")
        row[name] = float(np.mean([pearsonr(md, d).statistic for d in V[name]]))
    try:
        row["encoding_pearson"] = O._encoding_score("tuckute", enc_root / family / safe_model_dir(model_id), revision)
    except Exception:
        row["encoding_pearson"] = np.nan
    m = lookup_model_meta(model_id, family); row["n_params"] = m.n_params
    return row
tasks = O._tasks("tuckute", enc_root)
with ThreadPoolExecutor(8) as ex:
    rows = [r for r in ex.map(one, tasks) if r]
out = pd.DataFrame(rows); (S / "tuckute_roi_sets").mkdir(exist_ok=True, parents=True); out.to_csv(S / "tuckute_roi_sets/roi_sets.csv", index=False)
m = out.dropna(subset=["encoding_pearson"])
for name in SETS:
    print(f"{name:16s} n={len(out)} mean={out[name].mean():.4f} r(ridge)={pearsonr(m[name], m.encoding_pearson)[0]:+.2f} r(size)={np.corrcoef(np.log10(out.n_params), out[name])[0,1]:+.2f}")
print("ridge r(size)=", round(np.corrcoef(np.log10(m.n_params), m.encoding_pearson)[0, 1], 2))
