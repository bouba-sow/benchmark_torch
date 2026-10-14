#!/usr/bin/env python3
"""RDM-regression (held-out delta log-likelihood) track for Provo and English MECO.

For each measure (FP/GP/SP), predict the grand-average reading-time RDM
(|difference of item mean RT|, from the shared cached behavior RDM) from a model
RDM, with and without word-length + frequency confound RDMs, and score the
item-fold cross-validated delta log-likelihood (neuroai/analysis/rdm_regression.py,
docs/rdm_regression.md).

* embedding track: one model RDM per layer (correlation distance between the
  cached per-word hidden states, as in behavioral_rsa.py, but for every layer).
* surprisal track: one model RDM, |delta surprisal| from the pooled surprisal caches.

Model RDMs are rebuilt from per-item values on the response's own pair set (the
cached surprisal RDMs use a 250k-pair sample; the shared response and confound RDMs
use all pairs), so nothing is read from the sampled RDM caches. Inputs are read-only.
Output goes under --out-root; use a temporary/scratch root, not outputs/, until the
results are reviewed. Resumable: models with an existing CSV are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rdm-regression-matplotlib")
# Parallelism is over models (--workers); multi-threaded BLAS on top of that
# oversubscribes shared nodes and was ~4x slower in a login-node test.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

from neuroai.analysis.benchmark_common import lookup_model_meta
from neuroai.analysis.rdm_regression import RdmCvPlan, standardize_features
from neuroai.sweep import safe_model_dir
from scripts.analysis.behavioral_rsa import _cache_paths, _tasks as embedding_tasks
from scripts.analysis.surprisal_rsa import (
    _load_behavior_surprisal, _tasks as surprisal_tasks)

MEASURES = ("FP", "GP", "SP")
# Measures per dataset (same as surprisal_rsa.MEASURES). provo/meco embeddings are read here;
# the other datasets go through scripts/analysis/rdm_regression_behavioral_other.py.
DATASET_MEASURES = {
    "provo": MEASURES, "meco": MEASURES, "geco": ("FP", "GP"), "onestop": MEASURES,
    "naturalstories": ("SPR",), "ucl_reading": ("SPR",), "zuco_reading": ("NR", "SR", "TSR"),
}
VARIANTS = ("none", "len_freq")
RSA_DIR = Path(os.environ["RDM_RSA_DIR"]) if os.environ.get("RDM_RSA_DIR") else ROOT / "outputs"  # RDM_RSA_DIR: 1M-pair shared RDMs (scratch)
LAYER_SURPRISAL = -1  # `layer` value used by the (single-RDM) surprisal track
# RDM_LAST_LAYER_ONLY=1 scores only the final layer (used by the bootstrap / ceiling runs)
LAST_LAYER_ONLY = os.environ.get("RDM_LAST_LAYER_ONLY") == "1"
# RDM_EMBED_STANDARDIZE=1: z-score every hidden-state feature across the items before the correlation distance
EMBED_STANDARDIZE = os.environ.get("RDM_EMBED_STANDARDIZE") == "1"


def _load_rdm(path: Path):
    with np.load(path) as a:
        pairs = np.asarray(a["pair_indices"]) if "pair_indices" in a.files else None
        return np.asarray(a["stimulus_ids"]), np.asarray(a["distances"], dtype=np.float64), pairs


def _drop_items(ids, response, pairs, confounds, keep):
    """Restrict a response RDM (+ aligned confound RDMs) to the items in ``keep``. ``pairs is None`` means the
    distances are the condensed upper triangle over all items (scipy pdist order); otherwise ``pairs`` lists the
    (i, j) item indices of each distance."""
    n = len(ids)
    if pairs is None:
        i, j = np.triu_indices(n, 1)
        sel = keep[i] & keep[j]
    else:
        sel = keep[pairs[:, 0]] & keep[pairs[:, 1]]
        remap = np.cumsum(keep) - 1
        pairs = np.column_stack([remap[pairs[sel, 0]], remap[pairs[sel, 1]]])
    return np.asarray(ids)[keep], response[sel], pairs, [c[sel] for c in confounds]


def load_measure(dataset: str, measure: str, n_repeats: int, seed: int) -> dict:
    """Response RDM, confounds and one CV plan per confound variant (same partitions)."""
    shared = RSA_DIR / dataset / "pearson_rsa" / "surprisal" / "rdm_cache" / measure.lower() / "shared" / "behavior_rdm.npz"
    ids, response, pairs = _load_rdm(shared)
    confounds = []
    for name in ("word_length", "frequency"):
        cids, dist, cpairs = _load_rdm(RSA_DIR / dataset / "pearson_rsa" / "confounds" / measure.lower() / f"{name}_rdm.npz")
        if not np.array_equal(cids, ids) or (pairs is None) != (cpairs is None) or (
                pairs is not None and not np.array_equal(pairs, cpairs)):
            raise ValueError(f"Confound '{name}' is not aligned with the shared response RDM: {dataset}/{measure}")
        confounds.append(dist)
    if dataset == "naturalstories":
        # v2 surprisal (2026-10-12 audit) is NaN for each story's first word (no left context; the LME fits drop
        # these rows too) and, for OLMo-2-7B only, for 42 further words. Drop every item that lacks surprisal in ANY
        # model (RDM_DROP_ITEMS json of [story_id, zone]; default: story-initial words only) for every model and both
        # tracks, so all models are scored on the same items (52 of 10,256).
        drop_path = os.environ.get("RDM_DROP_ITEMS")
        drop = {tuple(x) for x in json.load(open(drop_path))} if drop_path else set()
        sid = np.asarray(ids).astype(int)
        keep = np.array([(s, z) not in drop and z != 1 for s, z in sid])
        if not keep.all():
            ids, response, pairs, confounds = _drop_items(ids, response, pairs, confounds, keep)
    n = len(ids)
    conf = np.column_stack(confounds)
    if os.environ.get("RDM_DENSE") == "1":  # all pairs: matrix-product engine (neuroai/analysis/rdm_regression_dense.py)
        if pairs is not None:
            raise ValueError("RDM_DENSE=1 needs shared RDMs that hold every pair")
        from neuroai.analysis.rdm_regression_dense import DenseDesign
        d_none = DenseDesign(n, [], n_repeats, seed=seed)
        d_conf = DenseDesign(n, [conf[:, 0], conf[:, 1]], n_repeats, seed=seed)
        return {"ids": ids, "n": n, "pairs": None, "n_pairs": len(response),
                "plans": {"none": d_none.plan(response), "len_freq": d_conf.plan(response)}}
    return {
        "ids": ids, "n": n, "pairs": pairs, "n_pairs": len(response),
        "plans": {"none": RdmCvPlan(response, n, pairs, None, n_repeats, seed=seed),
                  "len_freq": RdmCvPlan(response, n, pairs, conf, n_repeats, seed=seed)},
    }


def _pair_index(meas: dict) -> np.ndarray:
    if meas["pairs"] is not None:
        return meas["pairs"]
    from neuroai.analysis.rdm_regression import all_pairs
    if "_pair_index" not in meas:
        meas["_pair_index"] = all_pairs(meas["n"])
    return meas["_pair_index"]


def _rows(meas, measure, dataset, track, task, layer, rdm):
    if not np.isfinite(rdm).all():
        raise ValueError(f"Non-finite model RDM ({dataset}/{measure}/{task[1]}@{task[2]} layer {layer})")
    out = []
    for variant in VARIANTS:
        plan = meas["plans"][variant]
        r = plan.delta_ll(rdm)
        out.append({"dataset": dataset, "measure": measure, "track": track,
                    "family": task[0], "model_id": task[1], "revision": task[2],
                    "layer": layer, "confounds": variant, "n_items": meas["n"],
                    "n_pairs": meas["n_pairs"], "rsa_pearson": plan.pearson(rdm),
                    "delta_ll_per_obs": r.delta_ll_per_obs,
                    "delta_ll_per_obs_sd": r.delta_ll_per_obs_sd,
                    "ll_full_per_obs": r.ll_full_per_obs, "ll_base_per_obs": r.ll_base_per_obs,
                    "n_test_pairs": r.n_test_pairs, "frac_pairs_tested": r.frac_pairs_tested,
                    "n_folds": r.n_folds, "n_repeats": r.n_repeats})
    return out


def _out_path(root, dataset, track, measure, task):
    return (root / dataset / track / measure.lower() / task[0] / safe_model_dir(task[1]) /
            f"rdm_regression_{task[2].replace('/', '_')}.csv")


def _write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _layer_rdm(states: np.ndarray, pairs: np.ndarray, gram_max_items: int = int(os.environ.get("RDM_GRAM_MAX_ITEMS", 5000)),
               block: int = 16384) -> np.ndarray:
    """1 - Pearson r between item vectors, on ``pairs`` (matches pdist 'correlation').

    Up to ``gram_max_items`` items use one Gram matrix; above that (sampled-pair
    datasets with tens of thousands of items) pairs are processed in blocks so no
    n x n matrix is materialised."""
    x = standardize_features(states) if EMBED_STANDARDIZE else states.astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if (norm <= np.finfo(np.float32).tiny).any():
        raise ValueError("Constant hidden-state vector: correlation distance undefined")
    x /= norm
    if len(x) <= gram_max_items:
        return 1.0 - (x @ x.T)[pairs[:, 0], pairs[:, 1]].astype(np.float64)
    out = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), block):
        b = pairs[start:start + block]
        out[start:start + len(b)] = 1.0 - np.clip(np.einsum("ij,ij->i", x[b[:, 0]], x[b[:, 1]]), -1.0, 1.0)
    return out


def score_embedding(task, dataset, measures, prepared, out_root):
    todo = [m for m in measures if not _out_path(out_root, dataset, "embedding", m, task).is_file()]
    if not todo:
        return f"existing {task[1]}@{task[2]}"
    family, model_id, revision = task
    stories = sorted({int(s) for m in todo for s in prepared[m]["ids"][:, 0]})
    paths = _cache_paths(dataset, model_id, revision, {s: None for s in stories})
    if paths is None:
        return f"missing cache {model_id}@{revision}"
    feats = {}
    for s in stories:
        with np.load(paths[s]) as archive:
            f = archive["features"]
        if f.ndim != 3:
            raise ValueError(f"Unexpected feature shape {f.shape}: {paths[s]}")
        feats[s] = f
    n_layers = {f.shape[1] for f in feats.values()}
    if len(n_layers) != 1:
        raise ValueError(f"Inconsistent layer count across sections for {model_id}@{revision}")
    L = n_layers.pop()
    for measure in todo:
        meas = prepared[measure]
        ids = meas["ids"]
        for s, z in ids:
            if not 1 <= z <= len(feats[int(s)]):
                raise ValueError(f"{dataset} item {(s, z)} outside cached section of {model_id}")
        pairs = _pair_index(meas)
        rows = []
        for layer in (range(L - 1, L) if LAST_LAYER_ONLY else range(L)):
            states = np.stack([feats[int(s)][int(z) - 1, layer, :] for s, z in ids])
            rows += _rows(meas, measure, dataset, "embedding", task, layer, _layer_rdm(states, pairs))
        _write(_out_path(out_root, dataset, "embedding", measure, task), rows)
    return f"scored {model_id}@{revision} ({L} layers)"


def score_surprisal(task, dataset, measures, prepared, out_root):
    todo = [m for m in measures if not _out_path(out_root, dataset, "surprisal", m, task).is_file()]
    if not todo:
        return f"existing {task[1]}@{task[2]}"
    family, model_id, revision = task
    done = 0
    for measure in todo:
        values = _load_behavior_surprisal(dataset, family, model_id, revision, measure)
        if values is None:
            continue
        meas = prepared[measure]
        ids = pd.DataFrame(meas["ids"], columns=["story_id", "zone"])
        merged = ids.merge(values, on=["story_id", "zone"], how="left", validate="one_to_one")
        if merged.surprisal.isna().any():
            raise ValueError(f"{dataset}/{measure}: {int(merged.surprisal.isna().sum())} items lack "
                             f"surprisal for {model_id}@{revision}")
        s = merged.surprisal.to_numpy(dtype=np.float64)
        pairs = _pair_index(meas)
        rdm = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
        _write(_out_path(out_root, dataset, "surprisal", measure, task),
               _rows(meas, measure, dataset, "surprisal", task, LAYER_SURPRISAL, rdm))
        done += 1
    return f"scored {model_id}@{revision}" if done else f"missing cache {model_id}@{revision}"


def summarize(dataset: str, track: str, out_root: Path):
    frames = [pd.read_csv(p) for p in (out_root / dataset / track).glob("*/*/*/rdm_regression_*.csv")]
    if not frames:
        print(f"{dataset}/{track}: no results to summarize", flush=True)
        return
    df = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "measure", "track", "family", "model_id", "revision", "confounds"]
    best = df.loc[df.groupby(keys).delta_ll_per_obs.idxmax()].rename(
        columns={"layer": "best_layer", "delta_ll_per_obs": "best_delta_ll_per_obs",
                 "rsa_pearson": "best_layer_rsa_pearson"})
    last = df.loc[df.groupby(keys).layer.idxmax(), keys + ["delta_ll_per_obs"]].rename(
        columns={"delta_ll_per_obs": "last_layer_delta_ll_per_obs"})
    summary = best[keys + ["best_layer", "best_delta_ll_per_obs", "best_layer_rsa_pearson",
                           "frac_pairs_tested", "n_items"]].merge(last, on=keys)
    meta = [lookup_model_meta(m, f) for m, f in zip(summary.model_id, summary.family)]
    summary["group"] = [x.group for x in meta]
    summary["n_params"] = [x.n_params for x in meta]
    path = out_root / dataset / track / "model_summary.csv"
    summary.sort_values(keys).to_csv(path, index=False)
    print(f"{dataset}/{track}: {len(summary)} rows -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(DATASET_MEASURES))
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--measures", nargs="+", default=None, help="default: all measures of the dataset")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="default: $SCRATCH/rdm_regression (required if SCRATCH is unset)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, help="score only the first N models per track")
    ap.add_argument("--shard", metavar="I/N", help="score every N-th model starting at I (SLURM array "
                    "task I of N); a sharded run does not summarize, run --summarize-only afterwards")
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out_root = args.out_root or (Path(os.environ["SCRATCH"]) / "rdm_regression" if os.environ.get("SCRATCH") else None)
    if out_root is None:
        ap.error("--out-root is required when $SCRATCH is unset")
    if ROOT / "outputs" in out_root.resolve().parents or out_root.resolve() == ROOT / "outputs":
        ap.error("refusing to write under outputs/; use a scratch/temporary root")
    shard = None
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            ap.error("--shard I/N needs 0 <= I < N")
        shard = (i, n)
    tracks = ["embedding", "surprisal"] if args.track == "both" else [args.track]
    measures = tuple(args.measures or DATASET_MEASURES[args.dataset])
    if not set(measures) <= set(DATASET_MEASURES[args.dataset]):
        ap.error(f"{args.dataset} measures are {DATASET_MEASURES[args.dataset]}")
    failures = 0
    if not args.summarize_only:
        prepared = {m: load_measure(args.dataset, m, args.n_repeats, args.seed) for m in measures}
        print(args.dataset + ": " + ", ".join(f"{m}={prepared[m]['n']} items/{prepared[m]['n_pairs']} pairs"
                                              for m in measures), flush=True)
        embed_fn = score_embedding
        embed_tasks = lambda: embedding_tasks(args.dataset)  # noqa: E731
        if args.dataset not in ("provo", "meco"):
            from scripts.analysis import rdm_regression_behavioral_other as other
            prepared["_ctx"] = other.prepare_context(args.dataset, measures)
            embed_fn = other.score_embedding
            embed_tasks = lambda: other.embedding_tasks(args.dataset, measures)  # noqa: E731
        for track in tracks:
            tasks = (embed_tasks() if track == "embedding"
                     else surprisal_tasks(args.dataset, measures))
            if args.limit:
                tasks = tasks[:args.limit]
            if shard:
                tasks = tasks[shard[0]::shard[1]]
            fn = embed_fn if track == "embedding" else score_surprisal
            print(f"{track}: {len(tasks)} models", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(fn, t, args.dataset, measures, prepared, out_root): t for t in tasks}
                for fut in as_completed(futures):
                    try:
                        print(fut.result(), flush=True)
                    except Exception:
                        failures += 1
                        print(f"ERROR {futures[fut][1]}@{futures[fut][2]}:\n{traceback.format_exc()}",
                              file=sys.stderr, flush=True)
    if not shard:
        for track in tracks:
            summarize(args.dataset, track, out_root)
    if failures:
        sys.exit(f"{failures} model(s) failed; see ERROR lines above")


if __name__ == "__main__":
    main()
