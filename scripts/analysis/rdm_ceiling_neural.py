#!/usr/bin/env python3
"""Leave-one-subject-out RDM noise ceiling for the per-subject neural datasets.

Each subject's own neural RDM is the target (exactly what the models are scored on). The ceiling
predictor is the mean of the OTHER subjects' RDMs on that subject's pairs (average of their correlation
distances; the standard RSA lower noise-ceiling construction of Nili et al. 2014, as used by
Kaniuth & Hebart), scored with the same item-fold cross-validated pair likelihood and confound
variants as the models (neuroai/analysis/rdm_ceiling.py). Models and ceiling share the target, so
no half-target rescoring is needed here. Pairs that no other subject observed (Pereira pairs across
the two experiments) are dropped from the ceiling; their share is reported.

Datasets: pereira, tuckute, fedorenko2016, broderick, megmasc, podcast. Needs the raw neural patterns
(surprisal_rsa._prepare_neural_patterns) and verifies, per subject, that recomputing the subject's own
RDM from them reproduces the shared response RDM.
Output: <out-root>/<dataset>/loso_ceiling.npz and loso_ceiling_summary.csv (subject x variant).
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

from neuroai.analysis.rdm_ceiling import RdmTargetReplicates
from scripts.analysis import rdm_regression_neural as neural
from scripts.analysis.surprisal_rsa import _distances_for_pairs, _prepare_neural_patterns

DATASETS = ("pereira", "tuckute", "fedorenko2016", "broderick", "megmasc", "podcast")
N_PARTITIONS = 10


class PlanStub:
    def __init__(self, response, n_items, pairs=None, confounds=None, n_repeats=30, n_folds=None, seed=0):
        self.response, self.n_items, self.pairs, self.confounds = response, n_items, pairs, confounds


neural.RdmCvPlan = PlanStub


def _rows_in(pattern_ids: dict, ids: np.ndarray) -> np.ndarray:
    return np.fromiter((pattern_ids.get(s, -1) for s in ids.tolist()), dtype=np.int64, count=len(ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--elig-root", type=Path, default=None)
    ap.add_argument("--limit-subjects", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    out_root = args.out_root or (scratch / "rdm_ceiling" if scratch else None)
    if out_root is None:
        ap.error("--out-root is required when $SCRATCH is unset")
    if ROOT / "outputs" in out_root.resolve().parents or out_root.resolve() == ROOT / "outputs":
        ap.error("refusing to write under outputs/")
    elig_root = args.elig_root or (scratch / "rdm_regression_r30" if scratch else None)
    dataset = args.dataset

    prepared = neural.prepare(dataset, 1, 0, elig_root)
    units = prepared["units"][:args.limit_subjects] if args.limit_subjects else prepared["units"]
    raw = _prepare_neural_patterns(dataset)
    pat = {}
    for subject, (ids, values) in raw.items():
        canon = neural.canon_ids(ids)
        pat[str(subject)] = ({s: i for i, s in enumerate(canon.tolist())}, np.asarray(values))
    print(f"{dataset}: {len(units)} target subjects, patterns for {len(pat)} subjects", flush=True)

    subjects, ceilings, uppers, fractions, n_obs = [], [], [], [], []
    for unit in units:
        s = unit["subject"]
        pairs = neural._unit_pairs(unit)
        stub = unit["plans"]["len_freq"]
        a, b = pairs[:, 0], pairs[:, 1]
        # 1) the subject's own RDM recomputed from raw patterns must equal the shared response RDM
        own_map, own_pat = pat[s]
        own_rows = _rows_in(own_map, unit["ids"])
        if (own_rows < 0).any():
            raise ValueError(f"{dataset}/{s}: {(own_rows < 0).sum()} unit stimuli missing from the raw patterns")
        own = _distances_for_pairs(own_pat, np.column_stack((own_rows[a], own_rows[b])), "correlation")
        err = float(np.abs(own - np.asarray(stub.response)).max())
        if err > 1e-3:
            raise ValueError(f"{dataset}/{s}: raw patterns do not reproduce the shared RDM (max err {err:.3g})")
        # 2) mean of the other subjects' RDMs on this subject's pairs
        total, count = np.zeros(len(pairs)), np.zeros(len(pairs))
        for t, (t_map, t_pat) in pat.items():
            if t == s:
                continue
            rows = _rows_in(t_map, unit["ids"])
            ok = (rows[a] >= 0) & (rows[b] >= 0)
            if not ok.any():
                continue
            d = _distances_for_pairs(t_pat, np.column_stack((rows[a][ok], rows[b][ok])), "correlation")
            total[ok] += d
            count[ok] += 1
        valid = count > 0
        if valid.sum() < 100:
            print(f"{dataset}/{s}: only {int(valid.sum())} pairs observed by other subjects; skipped", flush=True)
            continue
        predictor = np.where(valid, total / np.maximum(count, 1), 0.0)
        # Nili et al. (2014) UPPER bound: the subject is part of the mean it is predicted from (optimistic)
        upper_pred = np.where(valid, (total + own) / (count + 1), 0.0)
        seed = (zlib.crc32(f"{dataset}|{s}".encode()) + args.seed) % (2**31)
        rep = RdmTargetReplicates(np.asarray(stub.response)[None], stub.n_items, stub.pairs, stub.confounds,
                                  None, n_partitions=N_PARTITIONS, seed=seed, pair_valid=valid[None])
        delta2, _ = rep.score_models(np.stack([predictor, upper_pred]))     # (2, V, 1, P)
        delta = delta2[:1]
        subjects.append(s)
        ceilings.append(delta[0, :, 0, :])
        uppers.append(delta2[1, :, 0, :])
        fractions.append(valid.mean())
        n_obs.append(count[valid].mean())
        print(f"{dataset}/{s}: ceiling {np.nanmean(delta[0, :, 0, :], axis=-1).round(5)}, "
              f"{valid.mean():.3f} of pairs observed by others (mean {count[valid].mean():.1f} others), "
              f"own-RDM check err {err:.1e}", flush=True)
    ceilings, uppers = np.stack(ceilings), np.stack(uppers)                 # (S, V, P)
    out = out_root / dataset
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "loso_ceiling.npz", subjects=np.array(subjects), ceiling=ceilings, upper=uppers,
                        variants=np.array(rep.variants), valid_pair_fraction=np.array(fractions),
                        mean_other_subjects=np.array(n_obs))
    rows = [{"dataset": dataset, "subject": s, "confounds": v, "ceiling_delta_ll_per_obs": float(np.nanmean(c[i])),
             "upper_delta_ll_per_obs": float(np.nanmean(u[i])),
             "valid_pair_fraction": f, "mean_other_subjects": k}
            for s, c, u, f, k in zip(subjects, ceilings, uppers, fractions, n_obs) for i, v in enumerate(rep.variants)]
    pd.DataFrame(rows).to_csv(out / "loso_ceiling_summary.csv", index=False)
    mean = np.nanmean(ceilings, axis=(0, 2))
    mean_up = np.nanmean(uppers, axis=(0, 2))
    print(f"{dataset}: subject-mean LOSO lower {dict(zip(rep.variants, mean.round(5)))}, "
          f"upper {dict(zip(rep.variants, mean_up.round(5)))} over {len(subjects)} subjects", flush=True)


if __name__ == "__main__":
    main()
