#!/usr/bin/env python3
"""Per-unit RDM regression (held-out delta log-likelihood) for Pereira 2018, all 54 ROIs.

Unit = (subject, passage set, ROI), 5 atlases (language 12, MD 20, DMN 12, visual 6, auditory 4).
Passage sets ('243sentences', '384sentences') are separate sessions, so pairs stay within a set.

  response RDM : |z(y_i) - z(y_j)|, y = ROI-mean BOLD, z-scored across the unit's sentences
  confounds    : |d sentlen|, |d mean word length|, |d mean frequency| (the LME nuisances, pooled-table columns)
  model RDM    : surprisal track  -> |d sentence surprisal| (last layer, as in the LME)
                 embedding track  -> 1 - Pearson between last-layer mean-pooled sentence states, each feature
                                     z-scored across the 627 sentences (Timkey & van Schijndel 2021)
  regression   : plain OLS, no random effects. base: y ~ 1 + confounds; full: + model RDM. Item-fold CV
                 (neuroai/analysis/rdm_regression.py), delta LL per held-out pair. Same folds for all models.

Output (one CSV per model, under --out-root, never outputs/): one row per unit with delta_ll_per_obs and
rsa_pearson. Aggregation (ROI subsets, subjects) is done afterwards by rdm_regression_pereira_summary.py.

    .venv/bin/python scripts/analysis/rdm_regression_pereira_units.py --track both --out-root $SCRATCH/rdm_pereira_units
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_regression import RdmCvPlan, all_pairs, standardize_features
from neuroai.datasets.pereira import PereiraDataset
from neuroai.sweep import safe_model_dir
from scripts.analysis import surprisal_rsa as S
from scripts.analysis.pereira_rsa import _tasks as embedding_tasks, load_last_layer

ATLASES = ("language", "MD", "DMN", "visual", "auditory")
SETS = ("243sentences", "384sentences")
CONF_COLS = ("sentlen", "mean_wlen", "mean_freq")
CACHE = Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/pereira_embed_features")


def seed_for(subject: str, pset: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{subject}:{pset}".encode()).digest()[:4], "little")


class Prepared:
    """Units, pairs, confounds and CV plans (model independent), grouped by (subject, set)."""

    def __init__(self, n_repeats: int):
        ds = PereiraDataset(ROOT / "data/pereira_fmri/Pereira2018_merged.nc")
        _, self.all_ids = ds.get_sentences()
        self.all_ids = np.asarray(self.all_ids).astype(str)
        self.row = {s: i for i, s in enumerate(self.all_ids)}
        fam, mid, rev = S._tasks("pereira", ("",))[0]
        _, table_path = S._neural_cache_paths("pereira", fam, mid, rev)
        sent = (pd.read_parquet(table_path, columns=["sentence_id", *CONF_COLS])
                .drop_duplicates("sentence_id").set_index("sentence_id"))
        by_group: dict[tuple[str, str], dict] = {}
        for atlas in ATLASES:
            for subject, (bold, ids) in ds.get_condition_per_subject(atlas).items():
                ids = np.asarray(ids).astype(str)
                rois = ds.rois(atlas)
                for pset in SETS:
                    sel = np.flatnonzero(np.char.startswith(ids, pset))
                    if len(sel) < 6:
                        continue
                    order = sel[np.argsort(ids[sel])]
                    g = by_group.setdefault((subject, pset), {"ids": ids[order], "units": []})
                    if not np.array_equal(g["ids"], ids[order]):
                        raise ValueError(f"Sentence sets differ across atlases for {subject}/{pset}")
                    for j, roi in enumerate(rois):
                        y = bold[order, j]
                        if np.isfinite(y).all() and y.std() > 0:
                            g["units"].append((atlas, roi, y))
        # Group units: per (set, ROI) mean of the z-scored responses over the subjects who saw the set
        # (regions are shared across subjects). Much higher SNR than a single subject's ROI-mean BOLD.
        for pset in SETS:
            members = [(k, g) for k, g in by_group.items() if k[1] == pset]
            if len(members) < 3:
                continue
            ids0 = members[0][1]["ids"]
            acc: dict[tuple[str, str], list] = {}
            for _, g in members:
                if not np.array_equal(g["ids"], ids0):
                    raise ValueError(f"Subjects differ in sentences within {pset}")
                for atlas, roi, y in g["units"]:
                    acc.setdefault((atlas, roi), []).append((y - y.mean()) / y.std())
            by_group[("GROUP", pset)] = {"ids": ids0, "units": [
                (a, r, np.mean(v, axis=0)) for (a, r), v in sorted(acc.items()) if len(v) >= 3]}
        self.groups = {}
        for (subject, pset), g in sorted(by_group.items()):
            ids, n = g["ids"], len(g["ids"])
            pairs = all_pairs(n)
            conf = np.column_stack([np.abs(sent.loc[ids, c].to_numpy(float)[pairs[:, 0]] -
                                           sent.loc[ids, c].to_numpy(float)[pairs[:, 1]]) for c in CONF_COLS])
            plans = []
            for atlas, roi, y in g["units"]:
                resp = np.abs(y[pairs[:, 0]] - y[pairs[:, 1]])
                plans.append((atlas, roi, RdmCvPlan(resp, n, None, conf, n_repeats, seed=seed_for(subject, pset))))
            self.groups[(subject, pset)] = {"ids": ids, "n": n, "pairs": pairs, "plans": plans,
                                            "rows": np.array([self.row[s] for s in ids])}
            print(f"{subject} {pset}: {n} sentences, {len(plans)} units", flush=True)


def corr_rdm(states: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    x = states.astype(np.float32).copy()
    x -= x.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    if (nrm <= np.finfo(np.float32).tiny).any():
        raise ValueError("Constant hidden-state vector")
    x /= nrm
    return 1.0 - (x @ x.T)[pairs[:, 0], pairs[:, 1]].astype(np.float64)


def out_path(root: Path, track: str, family: str, model_id: str, revision: str) -> Path:
    return root / track / family / safe_model_dir(model_id) / f"rdm_units_{revision.replace('/', '_')}.csv"


def score(prep: Prepared, track: str, task, root: Path) -> str:
    family, model_id, revision, cache_id = task
    path = out_path(root, track, family, model_id, revision)
    if path.is_file():
        return f"existing {model_id}@{revision}"
    if track == "surprisal":
        v = S._load_neural_surprisal("pereira", family, model_id, revision)
        if v is None:
            return f"missing surprisal {model_id}@{revision}"
        sv = v.drop_duplicates("sentence_id").set_index("sentence_id").surprisal
        states = None
    else:
        try:
            states = standardize_features(load_last_layer(CACHE, cache_id, revision, len(prep.all_ids)))
        except FileNotFoundError:
            return f"missing embedding {model_id}@{revision}"
    rows = []
    for (subject, pset), g in prep.groups.items():
        if track == "surprisal":
            s = sv.reindex(g["ids"]).to_numpy(np.float64)
            if not np.isfinite(s).all():
                return f"missing surprisal for some sentences {model_id}@{revision}"
            rdm = np.abs(s[g["pairs"][:, 0]] - s[g["pairs"][:, 1]])
        else:
            rdm = corr_rdm(states[g["rows"]], g["pairs"])
        if not np.isfinite(rdm).all() or rdm.std() == 0:
            raise ValueError(f"Bad model RDM {model_id}@{revision} {subject}/{pset}")
        for atlas, roi, plan in g["plans"]:
            r = plan.delta_ll(rdm)
            rows.append(dict(track=track, family=family, model_id=model_id, revision=revision,
                             subject=subject, passage_set=pset, atlas=atlas, roi=roi, n_items=g["n"],
                             n_pairs=plan.n_pairs, delta_ll_per_obs=r.delta_ll_per_obs,
                             delta_ll_per_obs_sd=r.delta_ll_per_obs_sd, rsa_pearson=plan.pearson(rdm),
                             frac_pairs_tested=r.frac_pairs_tested))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return f"scored {model_id}@{revision}: {len(rows)} units"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", metavar="I/N")
    args = ap.parse_args()
    if ROOT / "outputs" in args.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    prep = Prepared(args.n_repeats)
    enc_root = ROOT / "outputs_jeanzay/pereira/embedding_sweep"
    emb = {(f, m, r): c for f, m, c, r in embedding_tasks(enc_root)}
    sur = S._tasks("pereira", ("",))
    tracks = ["embedding", "surprisal"] if args.track == "both" else [args.track]
    for track in tracks:
        if track == "embedding":
            tasks = [(f, m, r, c) for (f, m, r), c in emb.items()]
        else:
            tasks = [(f, m, r, m) for f, m, r in sur]
        if args.limit:
            tasks = tasks[:args.limit]
        if args.shard:
            i, n = (int(x) for x in args.shard.split("/"))
            tasks = tasks[i::n]
        print(f"{track}: {len(tasks)} models", flush=True)
        fails = 0
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(score, prep, track, t, args.out_root): t for t in tasks}
            for f in as_completed(futs):
                try:
                    print(f.result(), flush=True)
                except Exception:
                    fails += 1
                    print(f"ERROR {futs[f][1]}@{futs[f][2]}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        if fails:
            sys.exit(f"{fails} model(s) failed")


if __name__ == "__main__":
    main()
