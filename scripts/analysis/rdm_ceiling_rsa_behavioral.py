#!/usr/bin/env python3
"""Pearson-RSA noise ceilings of the behavioral measures (split-half reliability of the item-mean response RDM).

Same definition as the neural ceilings of rdm_ceiling_units.py: 15 random participant splits into halves A/B; r_half = Pearson
correlation between the two half item-mean RDMs (|m_i - m_j|, over the pairs of the shared response RDM with both items observed
in both halves); rsa_ceiling = sqrt(2 r_half / (1 + r_half)) (Spearman-Brown corrected reliability of the full-sample RDM, square-
rooted: the highest RSA any model RDM could reach against the full-sample target). Trials, items and pairs are those of
rdm_ceiling_run.py / the shared behavior RDM.  Output: $SCRATCH/rdm_ceiling_rsa/behavioral_rsa_ceiling.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_ceiling import split_half_targets
from neuroai.analysis.rdm_regression import all_pairs
from scripts.analysis import rdm_ceiling_run as R

SCR = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
RSA_DIR = R.base.RSA_DIR


def one(dataset: str, measure: str, seed: int = 0) -> dict:
    shared = RSA_DIR / dataset / "pearson_rsa" / "surprisal" / "rdm_cache" / measure.lower() / "shared" / "behavior_rdm.npz"
    ids, response, pairs = R.base._load_rdm(shared)
    n = len(ids)
    pairs = all_pairs(n) if pairs is None else np.asarray(pairs)
    values, subj, item = R._arrays_for_unit(dataset, measure, R.behavioral_trials(dataset, measure), ids)
    full = np.bincount(item, weights=values, minlength=n) / np.maximum(np.bincount(item, minlength=n), 1)
    probe = slice(0, min(3000, len(pairs)))
    err = float(np.abs(np.abs(full[pairs[probe, 0]] - full[pairs[probe, 1]]) - response[probe]).max())
    if err > 1e-3 * max(1.0, float(np.abs(response[probe]).max())):
        raise ValueError(f"{dataset}/{measure}: trials do not reproduce the shared RDM (max err {err:.3g})")
    means, valid, halves = split_half_targets(values, subj, item, n, n_splits=R.N_SPLITS, seed=seed)
    i, j = pairs[:, 0], pairs[:, 1]
    rs = []
    for ta, tb in halves:
        ok = (valid[ta][i] * valid[ta][j] > 0) & (valid[tb][i] * valid[tb][j] > 0)
        a = np.abs(means[ta][i[ok]] - means[ta][j[ok]])
        b = np.abs(means[tb][i[ok]] - means[tb][j[ok]])
        if ok.sum() > 10 and a.std() > 0 and b.std() > 0:
            rs.append(np.corrcoef(a, b)[0, 1])
    r_half = float(np.mean(rs))
    sb = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan
    return dict(dataset=dataset, unit=measure, r_half=r_half, r_half_sd=float(np.std(rs, ddof=1)),
                rsa_ceiling=float(np.sqrt(sb)) if sb > 0 else np.nan, n_items=n, n_pairs=len(pairs),
                n_participants=int(subj.max()) + 1, align_err=err)


def main():
    rows = []
    for dataset, measures in R.base.DATASET_MEASURES.items():
        for m in measures:
            rows.append(one(dataset, m))
            print(rows[-1], flush=True)
    out = SCR / "rdm_ceiling_rsa"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "behavioral_rsa_ceiling.csv", index=False)


if __name__ == "__main__":
    main()
