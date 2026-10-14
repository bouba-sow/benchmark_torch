#!/usr/bin/env python3
"""All-pairs split-half noise ceilings of the behavioral reading measures (delta LL and RSA-correlation reliability), matrix-product engine.

Same recipe, seeds and outputs as rdm_ceiling_run.py (ceilings only, no model scoring) and rdm_ceiling_rsa_behavioral.py: 15 random
participant splits, 5 fold partitions per half-target, the item means of one half predict the other half beyond |d length| + |d frequency|,
both directions summed; the seed is crc32("<dataset>|<measure>"). Items = those of the existing shared response RDM (read-only under
outputs/), lexical covariates = the stored feature values of the confound archives. neuroai/analysis/rdm_ceiling_dense.py computes every
pair of the items at once (tests/test_rdm_ceiling_dense.py: equal to RdmTargetReplicates).

Output <out-root>/<dataset>/ceiling_<measure>.npz (ceiling: variants x splits x partitions, same fields as rdm_ceiling_run.py) and
<out-root>/<dataset>/rsa_ceiling_<measures|all>.csv (r_half, rsa_ceiling = sqrt(Spearman-Brown r_half) per measure).
    python rdm_ceiling_behavioral_dense.py ucl_reading --out-root $SCRATCH/rdm_ceiling_allpairs
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_ceiling import split_half_targets
from neuroai.analysis.rdm_ceiling_dense import DenseCeilingDesign
from scripts.analysis import rdm_ceiling_run as R

N_SPLITS = 15
MEASURES = {"geco": ("FP", "GP"), "onestop": ("FP", "GP", "SP"), "naturalstories": ("SPR",), "ucl_reading": ("SPR",),
            "zuco_reading": ("NR", "SR", "TSR"), "provo": ("FP", "GP", "SP"), "meco": ("FP", "GP", "SP")}


def pair_abs_diff(v: np.ndarray) -> np.ndarray:
    return np.concatenate([np.abs(v[i] - v[i + 1:]) for i in range(len(v) - 1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(MEASURES))
    ap.add_argument("--measures", nargs="+")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if ROOT / "outputs" in a.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    old = ROOT / "outputs" / a.dataset / "pearson_rsa"
    rsa_rows = []
    for measure in (a.measures or MEASURES[a.dataset]):
        z = np.load(old / "surprisal" / "rdm_cache" / measure.lower() / "shared" / "behavior_rdm.npz")
        ids, old_d = z["stimulus_ids"], z["distances"].astype(np.float64)
        old_pairs = z["pair_indices"] if "pair_indices" in z.files else None
        n = len(ids)
        feats = []
        for name in ("word_length", "frequency"):
            c = np.load(old / "confounds" / measure.lower() / f"{name}_rdm.npz")
            assert np.array_equal(c["stimulus_ids"], ids), f"{a.dataset}/{measure}/{name}: item order differs"
            feats.append(c["feature_values"].astype(np.float64))
        values, subj, item = R._arrays_for_unit(a.dataset, measure, R.behavioral_trials(a.dataset, measure), ids)
        full = np.bincount(item, weights=values, minlength=n) / np.maximum(np.bincount(item, minlength=n), 1)
        if old_pairs is not None:
            probe = slice(0, min(3000, len(old_pairs)))
            got = np.abs(full[old_pairs[probe, 0]] - full[old_pairs[probe, 1]])
        else:
            probe = slice(0, min(3000, len(old_d)))
            got = pair_abs_diff(full)[probe]
        err = float(np.abs(got - old_d[probe]).max())
        if err > 1e-3 * max(1.0, float(np.abs(old_d[probe]).max())):
            raise ValueError(f"{a.dataset}/{measure}: subject-level trials do not reproduce the shared RDM (max err {err:.3g})")
        seed = (zlib.crc32(f"{a.dataset}|{measure}".encode()) + a.seed) % (2 ** 31)
        design = DenseCeilingDesign(n, [pair_abs_diff(f) for f in feats])
        means, valid, halves = split_half_targets(values, subj, item, n, n_splits=N_SPLITS, seed=seed)
        comb, n_test, rs = design.run(means, valid, halves, seed)
        out = a.out_root / a.dataset / f"ceiling_{measure}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, ceiling=comb, variants=np.array(design.variants), valid_item_fraction=valid[::2].mean(axis=1),
                            n_test=n_test, align_err=err, n_participants=int(subj.max()) + 1)
        r_half = float(np.mean(rs))
        sb = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan
        rsa_rows.append(dict(dataset=a.dataset, unit=measure, r_half=r_half, r_half_sd=float(np.std(rs, ddof=1)),
                             rsa_ceiling=float(np.sqrt(sb)) if sb > 0 else np.nan, n_items=n, n_pairs=n * (n - 1) // 2,
                             n_participants=int(subj.max()) + 1, align_err=err))
        print(f"{a.dataset}/{measure}: {n} items, dll ceiling {np.nanmean(comb[design.variants.index('len_freq')]):.4e}, "
              f"r_half {r_half:.3f} rsa ceiling {rsa_rows[-1]['rsa_ceiling']:.3f}", flush=True)
    tag = "_".join(a.measures) if a.measures else "all"
    pd.DataFrame(rsa_rows).to_csv(a.out_root / a.dataset / f"rsa_ceiling_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
