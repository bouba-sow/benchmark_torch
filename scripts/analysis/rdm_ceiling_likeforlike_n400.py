#!/usr/bin/env python3
"""Like-for-like noise ceiling experiment on one N400 dataset (default: Federmeier 2007).

The split-half ceiling (rdm_ceiling_units.py) predicts one half-sample RDM from the other half's. Models are normally scored on
the FULL-sample item-mean RDM, which is less noisy, so model / ceiling can exceed 1. Here the models are scored on the SAME half
targets as the ceiling: for each of the 15 participant splits and each half, the model RDM predicts that half's RDM beyond the
confound RDMs (same item folds / partitions), the two halves are summed and divided by the total test pairs
(neuroai/analysis/rdm_ceiling.py, Timkey et al. 2026). Reports, per track, the full-target score, the half-target score, the
ceiling, the ratios and how many models exceed the ceiling in each case.

    .venv/bin/python scripts/analysis/rdm_ceiling_likeforlike_n400.py f2007 --out $SCRATCH/rdm_ceiling_likeforlike
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_ceiling import RdmTargetReplicates, combine_halves, split_half_targets
from scripts.analysis import n400_item_mean_rsa as M
from scripts.analysis import n400_paper_rsa as P
from scripts.analysis.rdm_ceiling_units import N_SPLITS, n400_inputs

SCR = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", choices=["nieuwland", "f2007", "h2019"])
    ap.add_argument("--out", type=Path, default=SCR / "rdm_ceiling_likeforlike")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    M.STANDARDIZE = True

    val, subj, item, n, pairs, conf, seed = n400_inputs(a.study)
    means, valid, half_pairs = split_half_targets(val, subj, item, n, N_SPLITS, seed)
    i, j = pairs[:, 0], pairs[:, 1]
    resp = np.abs(means[:, i] - means[:, j])
    rt = RdmTargetReplicates(resp, n, pairs, conf, item_valid=valid, n_partitions=5, seed=seed)
    v = rt.variants.index("len_freq")
    delta, n_test = rt.score_partners(np.arange(len(resp)) ^ 1)
    ceiling = float(np.nanmean(combine_halves(delta, n_test, half_pairs)[0, v]))
    print(f"{a.study}: {n} items, ceiling (partner halves) = {ceiling:.4e}", flush=True)

    src = M.Nieuwland() if a.study == "nieuwland" else M.Szewczyk(a.study)
    files = list((ROOT / "outputs_jeanzay" / src.dataset / "embedding_sweep").glob("*/*/*item_mean_scores.csv"))
    keys = set()
    for f in files:
        t = pd.read_csv(f, dtype={"revision": str})
        keys.update(t[["family", "model_id", "revision"]].itertuples(index=False, name=None))
    tasks = [(f, t) for f, t in P.registry_tasks() if (f, t.model_id, t.revision or "main") in keys]
    if a.limit:
        tasks = tasks[:a.limit]

    # full-target reference (same plan as the headline regression: plain item-mean target, 10 repeats)
    from neuroai.analysis.rdm_regression import RdmCvPlan
    y = (src.y - src.y.mean()) / src.y.std()
    full_plan = RdmCvPlan(np.abs(y[pairs[:, 0]] - y[pairs[:, 1]]), n, pairs, conf, 10, seed=seed)

    rows = []
    for track in ("embedding", "surprisal"):
        ids, rdms = [], []
        for family, task in tasks:
            d = getattr(src, track)(family, task)
            if d is None or not np.isfinite(d).all() or np.std(d) == 0:
                continue
            ids.append((family, task.model_id, task.revision or "main"))
            rdms.append(np.asarray(d, dtype=np.float64))
        rdms = np.stack(rdms)
        dm, nt = rt.score_models(rdms)                        # (models, variants, targets, partitions)
        half = combine_halves(dm, nt, half_pairs)             # (models, variants, splits, partitions)
        half_score = np.nanmean(half[:, v], axis=(-2, -1))
        for k, (family, model_id, revision) in enumerate(ids):
            rows.append(dict(track=track, family=family, model_id=model_id, revision=revision,
                             full_target_dll=full_plan.delta_ll(rdms[k]).delta_ll_per_obs, half_target_dll=float(half_score[k]),
                             ceiling=ceiling))
        print(f"{track}: scored {len(ids)} models", flush=True)
    r = pd.DataFrame(rows)
    r["ratio_full"] = r.full_target_dll / ceiling
    r["ratio_half"] = r.half_target_dll / ceiling
    r.to_csv(a.out / f"likeforlike_{a.study}.csv", index=False)
    summ = []
    for track, g in r.groupby("track"):
        summ.append(dict(track=track, n_models=len(g), ceiling=ceiling,
                         full_median=g.full_target_dll.median(), full_best=g.full_target_dll.max(),
                         full_best_over_ceiling=g.ratio_full.max(), full_n_above=int((g.ratio_full > 1).sum()),
                         half_median=g.half_target_dll.median(), half_best=g.half_target_dll.max(),
                         half_best_over_ceiling=g.ratio_half.max(), half_n_above=int((g.ratio_half > 1).sum()),
                         spearman_full_vs_half=g.full_target_dll.corr(g.half_target_dll, method="spearman")))
    s = pd.DataFrame(summ)
    s.to_csv(a.out / f"likeforlike_{a.study}_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(5).T.to_string())


if __name__ == "__main__":
    main()
