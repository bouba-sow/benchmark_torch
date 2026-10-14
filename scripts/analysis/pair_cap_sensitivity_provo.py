#!/usr/bin/env python3
"""Pair-count sensitivity of the RDM-regression scores on a behavioral dataset with ALL pairs available (Provo).

Provo's shared response / confound RDMs hold every pair (2,394 items, 2,864,421 pairs), so the all-pairs score is the ground truth
and a random subset of ``cap`` pairs is exactly what the sampled datasets (GECO, OneStop, ...) and the neural datasets use.
For every cap and 3 random pair samples, the surprisal-track last-layer score of each model (same models as the neural runs,
``models.txt``) is computed with the production machinery (RdmCvPlan: item folds, Gaussian held-out dLL beyond the length and
frequency RDMs, 10 repeats; the item partitions are the same for every cap, so only the pair sample changes).

    .venv/bin/python scripts/analysis/pair_cap_sensitivity_provo.py --out $SCRATCH/pair_cap_sensitivity/provo.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_regression import RdmCvPlan, all_pairs
from scripts.analysis import rdm_regression_behavioral as B

CAPS = (50_000, 100_000, 250_000, 500_000, 1_000_000)
SEEDS = (0, 1, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="provo")
    ap.add_argument("--measure", default="FP")
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-repeats", type=int, default=10)
    a = ap.parse_args()
    keep = {l.strip() for l in open(a.models) if l.strip()}

    shared = B.RSA_DIR / a.dataset / "pearson_rsa" / "surprisal" / "rdm_cache" / a.measure.lower() / "shared" / "behavior_rdm.npz"
    ids, response, pairs = B._load_rdm(shared)
    assert pairs is None, "this test needs a dataset whose shared RDM holds all pairs"
    conf = []
    for name in ("word_length", "frequency"):
        cids, dist, _ = B._load_rdm(B.RSA_DIR / a.dataset / "pearson_rsa" / "confounds" / a.measure.lower() / f"{name}_rdm.npz")
        assert np.array_equal(cids, ids)
        conf.append(dist)
    conf = np.column_stack(conf)
    n = len(ids)
    pair_idx = all_pairs(n)
    total = len(pair_idx)
    assert total == len(response)
    print(f"{a.dataset}/{a.measure}: {n} items, {total} pairs", flush=True)

    # per-item surprisal of the selected models (same merge as rdm_regression_behavioral.score_surprisal)
    item_ids = pd.DataFrame(ids, columns=["story_id", "zone"])
    s_by_model = {}
    for family, model_id, revision in B.surprisal_tasks(a.dataset, (a.measure,)):
        if model_id not in keep and model_id.split("/")[-1] not in keep:
            continue
        v = B._load_behavior_surprisal(a.dataset, family, model_id, revision, a.measure)
        if v is None:
            continue
        m = item_ids.merge(v, on=["story_id", "zone"], how="left", validate="one_to_one")
        if m.surprisal.isna().any():
            continue
        s_by_model[(family, model_id, revision)] = m.surprisal.to_numpy(np.float64)
    print(f"{len(s_by_model)} models", flush=True)

    rows = []
    settings = [("all", 0, np.arange(total))] + [(cap, seed, np.sort(np.random.default_rng(1000 * seed + cap).choice(total, cap, replace=False)))
                                                for cap in CAPS for seed in SEEDS]
    for cap, seed, idx in settings:
        pr = pair_idx[idx]
        plan = RdmCvPlan(response[idx], n, pr, conf[idx], a.n_repeats, seed=0)  # same item partitions for every setting
        for (family, model_id, revision), s in s_by_model.items():
            rdm = np.abs(s[pr[:, 0]] - s[pr[:, 1]])
            r = plan.delta_ll(rdm)
            rows.append(dict(cap=cap, pair_seed=seed, family=family, model_id=model_id, revision=revision,
                             delta_ll_per_obs=r.delta_ll_per_obs, rsa_pearson=plan.pearson(rdm), n_pairs=len(idx)))
        print(f"cap={cap} seed={seed}: {len(s_by_model)} models", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
