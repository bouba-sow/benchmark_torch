#!/usr/bin/env python3
"""Rebuild the shared response RDM and the word-length / frequency confound RDMs of the SAMPLED behavioral datasets (GECO, OneStop,
Natural Stories, UCL Reading, ZuCo Reading) on a random sample of ``--cap`` pairs (default 1,000,000; they were 250,000).

The items (stimulus_ids, same order), the per-item reading time and the per-item lexical covariates are those of the existing runs:
ids and covariate values are read from the existing shared / confound RDM archives under outputs/ (read-only), the item means from
surprisal_rsa._prepare_behavior. The pair sample is drawn by the same function as before (surprisal_rsa._pair_distances, seed from the
dataset and measure) with the larger cap. The archives are written to a SCRATCH root with the same layout and fields, so that
rdm_regression_behavioral.py can read them by setting RDM_RSA_DIR to that root:

  <root>/<dataset>/pearson_rsa/surprisal/rdm_cache/<measure>/shared/behavior_rdm.npz
  <root>/<dataset>/pearson_rsa/confounds/<measure>/{word_length,frequency}_rdm.npz

Before writing, the item means are checked against the stored response distances on the OLD pairs (max difference must be float32 rounding).
    python rdm_behavior_pairs_build.py <dataset> --root $SCRATCH/rdm_behavior_1m
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

from scripts.analysis import surprisal_rsa as S

def pair_abs_diff(v: np.ndarray) -> np.ndarray:
    """|v_i - v_j| for all pairs i < j, in all_pairs() order."""
    return np.concatenate([np.abs(v[i] - v[i + 1:]) for i in range(len(v) - 1)])


MEASURES = {"geco": ("FP", "GP"), "onestop": ("FP", "GP", "SP"), "naturalstories": ("SPR",), "ucl_reading": ("SPR",),
            "zuco_reading": ("NR", "SR", "TSR")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(MEASURES))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=1_000_000)
    ap.add_argument("--all-pairs", action="store_true", help="every pair (no pair_indices in the archives): response / confound distances in all_pairs order")
    a = ap.parse_args()
    if ROOT / "outputs" in a.root.resolve().parents or a.root.resolve() == ROOT / "outputs":
        ap.error("refusing to write under outputs/")
    S.MAX_PAIRS = a.cap
    responses = S._prepare_behavior(a.dataset)
    old = ROOT / "outputs" / a.dataset / "pearson_rsa"
    for measure in MEASURES[a.dataset]:
        z = np.load(old / "surprisal" / "rdm_cache" / measure.lower() / "shared" / "behavior_rdm.npz")
        ids, old_d, old_pairs = z["stimulus_ids"], z["distances"].astype(np.float64), z["pair_indices"]
        n = len(ids)
        merged = pd.DataFrame(ids, columns=["story_id", "zone"]).merge(
            responses[measure], on=["story_id", "zone"], how="left", validate="one_to_one")
        rt = merged.rt.to_numpy(np.float64)
        if not np.isfinite(rt).all():
            raise ValueError(f"{a.dataset}/{measure}: items without a reading time")
        rebuilt = np.abs(rt[old_pairs[:, 0]] - rt[old_pairs[:, 1]])
        err = float(np.abs(rebuilt - old_d).max())
        if err > 1e-5 * max(1.0, float(np.abs(old_d).max())):
            raise ValueError(f"{a.dataset}/{measure}: item means do not reproduce the stored response RDM (max err {err:.3g})")
        if a.all_pairs:
            pairs = None
            dist = lambda v: pair_abs_diff(v).astype(np.float32)  # noqa: E731
            extra = {}
        else:
            _, _, pairs = S._pair_distances(np.zeros(n), np.zeros(n), a.dataset, measure)
            if pairs is None:
                raise ValueError(f"{a.dataset}/{measure}: all {n * (n - 1) // 2} pairs fit under the cap; nothing to sample")
            pairs = np.asarray(pairs, dtype=np.int32)
            dist = lambda v: np.abs(v[pairs[:, 0]] - v[pairs[:, 1]]).astype(np.float32)  # noqa: E731
            extra = {"pair_indices": pairs}
        out = a.root / a.dataset / "pearson_rsa"
        shared = out / "surprisal" / "rdm_cache" / measure.lower() / "shared" / "behavior_rdm.npz"
        shared.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shared, stimulus_ids=ids, distances=dist(rt), **extra, dataset=np.asarray(a.dataset),
                            measure=np.asarray(measure), rdm_type=np.asarray("absolute_reading_time_difference"))
        for name in ("word_length", "frequency"):
            c = np.load(old / "confounds" / measure.lower() / f"{name}_rdm.npz")
            if not np.array_equal(c["stimulus_ids"], ids):
                raise ValueError(f"{a.dataset}/{measure}/{name}: stimulus order differs from the shared RDM")
            v = c["feature_values"].astype(np.float64)
            dest = out / "confounds" / measure.lower() / f"{name}_rdm.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, stimulus_ids=ids, feature_values=c["feature_values"],
                                distances=dist(v), **extra,
                                dataset=np.asarray(a.dataset), measure=np.asarray(measure), rdm_type=np.asarray("absolute_difference"),
                                feature=c["feature"])
        print(f"{a.dataset}/{measure}: {n} items, {n * (n - 1) // 2 if pairs is None else len(pairs)} pairs (was {len(old_pairs)}); item-mean check max err {err:.2e}", flush=True)


if __name__ == "__main__":
    main()
