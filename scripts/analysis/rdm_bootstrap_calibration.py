#!/usr/bin/env python3
"""Does the item bootstrap (neuroai/analysis/rdm_bootstrap.py) estimate the real sampling
variability of the held-out RDM delta LL? Simulation with a known population.

Population: items with latent z ~ N(0,1). Response value r = z + noise, model A value
s_A = z + small noise, model B value s_B = z + large noise, confound c = z + noise; every RDM is
the absolute difference of item values (like the word-level RDMs in the benchmark), so entries
share items exactly as real RDM entries do.

For each n: S independent experiments (fresh items each time) give the point estimate
(production CV, RdmCvPlan, 10 repeats); their spread is the TRUE sampling SD. On a subset of
experiments the bootstrap (200 resamples x 2 partitions) gives SE, 95% percentile intervals and
paired A-B differences. Reported: SE ratio (bootstrap / true), naive (uncorrected) SE ratio,
coverage of the 95% interval for E[estimate], same for the A-B difference.
Usage: rdm_bootstrap_calibration.py N_ITEMS [--experiments 400] [--boot-experiments 120]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from neuroai.analysis.rdm_bootstrap import RdmBootstrap
from neuroai.analysis.rdm_regression import RdmCvPlan

NOISE = {"resp": 1.0, "A": 0.4, "B": 1.6, "conf": 1.0}


def draw(n, rng, n_pairs=None):
    """RDMs over all pairs, or over ``n_pairs`` randomly sampled pairs (like the 250k-pair
    samples used for the large datasets). Returns (rdms, pairs or None)."""
    z = rng.normal(size=n)
    vals = {k: z + s * rng.normal(size=n) for k, s in NOISE.items()}
    if n_pairs is None:
        return {k: pdist(v[:, None]) for k, v in vals.items()}, None
    a, b = rng.integers(0, n, size=3 * n_pairs), rng.integers(0, n, size=3 * n_pairs)
    cand = np.unique(np.column_stack((np.minimum(a, b), np.maximum(a, b)))[a != b], axis=0)
    pairs = cand[rng.permutation(len(cand))[:n_pairs]]
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))].astype(np.int32)
    return {k: np.abs(v[pairs[:, 0]] - v[pairs[:, 1]]) for k, v in vals.items()}, pairs


def point(n, rdm, seed, pairs=None):
    """Production estimate: 10-repeat item-fold CV, both variants -> (model, variant) array."""
    conf = rdm["conf"][:, None]
    out = np.empty((2, 2))
    for vi, c in enumerate((None, conf)):
        plan = RdmCvPlan(rdm["resp"], n, pairs, c, n_repeats=10, seed=seed)
        for mi, key in enumerate(("A", "B")):
            out[mi, vi] = plan.delta_ll(rdm[key]).delta_ll_per_obs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_items", type=int)
    ap.add_argument("--experiments", type=int, default=400)
    ap.add_argument("--boot-experiments", type=int, default=120)
    ap.add_argument("--n-pairs", type=int, default=None, help="sample this many pairs per experiment")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    n = a.n_items
    rng = np.random.default_rng(12345 + n)
    pts, se, naive, lo, hi, dse, dlo, dhi = [], [], [], [], [], [], [], []
    for s in range(a.experiments):
        rdm, pairs = draw(n, rng, a.n_pairs)
        pts.append(point(n, rdm, seed=s, pairs=pairs))
        if s < a.boot_experiments:
            boot = RdmBootstrap(rdm["resp"], n, pairs, rdm["conf"][:, None], n_boot=200,
                                n_partitions=2, seed=1000 + s)
            res = boot.delta_ll_batch(np.stack([rdm["A"], rdm["B"]]))
            se.append(res.se())
            naive.append(np.sqrt(np.nanvar(res.theta(), axis=-1, ddof=1)))
            ci = res.ci()
            lo.append(ci[0]); hi.append(ci[1])
            d = res.diff(0, 1)
            dse.append(d["se"]); dlo.append(d["ci_low"]); dhi.append(d["ci_high"])
    pts = np.array(pts)                                   # (S, model, variant)
    truth = pts.mean(axis=0)
    emp_sd = pts.std(axis=0, ddof=1)
    diff_pts = pts[:, 0] - pts[:, 1]
    diff_truth, diff_sd = diff_pts.mean(axis=0), diff_pts.std(axis=0, ddof=1)
    nb = a.boot_experiments
    se, naive, lo, hi = (np.array(x) for x in (se, naive, lo, hi))
    dse, dlo, dhi = (np.array(x) for x in (dse, dlo, dhi))
    rows = []
    for mi, mname in enumerate("AB"):
        for vi, vname in enumerate(("none", "len_freq")):
            cover = ((lo[:, mi, vi] <= truth[mi, vi]) & (truth[mi, vi] <= hi[:, mi, vi])).mean()
            rows.append({"n_items": n, "quantity": f"model {mname}", "variant": vname,
                         "true_mean": truth[mi, vi], "true_sd": emp_sd[mi, vi],
                         "boot_se_mean": se[:, mi, vi].mean(),
                         "se_ratio": se[:, mi, vi].mean() / emp_sd[mi, vi],
                         "naive_se_ratio": naive[:, mi, vi].mean() / emp_sd[mi, vi],
                         "coverage95": cover})
    for vi, vname in enumerate(("none", "len_freq")):
        cover = ((dlo[:, vi] <= diff_truth[vi]) & (diff_truth[vi] <= dhi[:, vi])).mean()
        rows.append({"n_items": n, "quantity": "A - B", "variant": vname,
                     "true_mean": diff_truth[vi], "true_sd": diff_sd[vi],
                     "boot_se_mean": dse[:, vi].mean(), "se_ratio": dse[:, vi].mean() / diff_sd[vi],
                     "naive_se_ratio": np.nan, "coverage95": cover})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"n_items={n} n_pairs={a.n_pairs or n*(n-1)//2}: {a.experiments} experiments, bootstrap on {nb}")
    print(table.round(5).to_string(index=False), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
