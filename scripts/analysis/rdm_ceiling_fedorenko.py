#!/usr/bin/env python3
"""Split-half noise ceiling of the Fedorenko 2016 per-electrode RDM regression.

Electrodes are subject specific, so the split-half of the other datasets (same unit in both halves) is not possible. As for the
Fedorenko LME ceiling (neuroai/regressors/ecog_ceiling.py, Timkey et al. 2026 adapted) the predictor is an ITEM-LEVEL signal
shared by all target electrodes, exactly like surprisal in the models:

  for each of N_SPLITS random splits of the 5 subjects into halves A / B (2 + 3 subjects, both assignments are drawn):
    m_A(word) = mean over the electrodes of the subjects in A of the electrode's z-scored word response   (331 eligible words)
    predictor RDM = |m_A,i - m_A,j|
    target        = each electrode's response RDM |z(y_i) - z(y_j)| of the subjects in B (the same unit RDM as the models)
    score         = held-out delta LL per pair beyond |d word length| + |d word frequency| (RdmCvPlan: same item folds per
                    subject, same OLS Gaussian likelihood as the models); mean over the electrodes of a subject, then over subjects
  both directions (A->B and B->A) are combined so that every subject is a target once per split; ceiling = mean over splits.
Like the other ceilings it is conservative (predictor from 2-3 subjects) and compared with the full-sample model scores.

Output: <out-root>/ceilings_fedorenko2016.csv (dll_ceiling, dll_ceiling_sd, r_pred = mean Pearson between predictor and target RDMs).
    .venv/bin/python scripts/analysis/rdm_ceiling_fedorenko.py --out-root $SCRATCH/rdm_ceiling_units
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_regression import RdmCvPlan, all_pairs
from scripts.analysis import surprisal_rsa as S

SCR = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
N_SPLITS = 15


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=SCR / "rdm_ceiling_units")
    ap.add_argument("--n-repeats", type=int, default=10)
    a = ap.parse_args()
    if ROOT / "outputs" in a.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    elig = sorted(json.load(open(SCR / "rdm_regression_r30/fedorenko2016/eligible_ids.json"))["ids"]["*"], key=int)
    fam, mid, rev = S._tasks("fedorenko2016", ("",))[0]
    table = pd.read_parquet(S._neural_cache_paths("fedorenko2016", fam, mid, rev)[1],
                            columns=["subject", "electrode", "stimulus_id", "wlen", "freq", "y"])
    table["stimulus_id"] = table.stimulus_id.astype(str)
    cov = table.drop_duplicates("stimulus_id").set_index("stimulus_id")[["wlen", "freq"]].loc[elig]
    n = len(elig)
    pairs = all_pairs(n)
    conf = np.column_stack([np.abs(cov[c].to_numpy(float)[pairs[:, 0]] - cov[c].to_numpy(float)[pairs[:, 1]]) for c in ("wlen", "freq")])
    units = []  # (subject, electrode, z-scored response, plan): same units, folds and confounds as rdm_regression_fedorenko_units.py
    for (sub, el), part in table.groupby(["subject", "electrode"]):
        y = part.set_index("stimulus_id").y.reindex(elig).to_numpy(float)
        if not np.isfinite(y).all() or y.std() == 0:
            continue
        y = (y - y.mean()) / y.std()
        seed = int.from_bytes(hashlib.sha256(f"fedorenko2016:{sub}".encode()).digest()[:4], "little")
        units.append((str(sub), str(el), y, RdmCvPlan(np.abs(y[pairs[:, 0]] - y[pairs[:, 1]]), n, None, conf, a.n_repeats, seed=seed)))
    subjects = sorted({u[0] for u in units})
    print(f"{n} words, {len(units)} electrodes, {len(subjects)} subjects", flush=True)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(b"fedorenko2016:ceiling").digest()[:4], "little"))
    rows = []
    for k in range(N_SPLITS):
        perm = rng.permutation(len(subjects))
        halves = [{subjects[i] for i in perm[: len(subjects) // 2]}, {subjects[i] for i in perm[len(subjects) // 2:]}]
        per_sub, per_r = {}, []
        for src, tgt in (halves, halves[::-1]):
            m = np.mean([u[2] for u in units if u[0] in src], axis=0)
            pred = np.abs(m[pairs[:, 0]] - m[pairs[:, 1]])
            for sub, el, y, plan in units:
                if sub in tgt:
                    per_sub.setdefault(sub, []).append(plan.delta_ll(pred).delta_ll_per_obs)
                    per_r.append(plan.pearson(pred))
        score = float(np.mean([np.mean(v) for v in per_sub.values()]))
        rows.append(dict(split=k, dll=score, r_pred=float(np.mean(per_r)), src_a="|".join(sorted(halves[0])), src_b="|".join(sorted(halves[1]))))
        print(f"split {k}: dLL {score:+.3e}", flush=True)
    d = pd.DataFrame(rows)
    out = pd.DataFrame([dict(dataset="fedorenko2016", unit="word_response_mean", dll_ceiling=d.dll.mean(),
                             dll_ceiling_sd=d.dll.std(ddof=1), r_pred=d.r_pred.mean(), rsa_ceiling=np.nan,
                             n_subjects=len(subjects), n_items=n, n_pairs=len(pairs), n_splits=N_SPLITS)])
    a.out_root.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_root / "ceilings_fedorenko2016.csv", index=False)
    d.to_csv(a.out_root / "ceilings_fedorenko2016_splits.csv", index=False)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
