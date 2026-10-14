#!/usr/bin/env python3
"""Cache-only Pearson RDM RSA for the Federmeier 2007 and Hubbard 2019 N400 studies."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

from neuroai.analysis.benchmark_common import EXCLUDED_FAMILY_DIRS, lookup_model_meta
from neuroai.benchmarks.eeg_erp import _feature_cache_path, _items_hash
from neuroai.datasets.szewczyk_paper import STUDIES, TSV, load_study, study_items
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
from neuroai.sweep import safe_model_dir
from scripts.run.embedding.neural.szewczyk_item_mean import OLD_RDS

CACHE = Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/szewczyk_features")


def registry_tasks():
    return [(family, task) for family in LPP_FAMILY_NAMES if family not in EXCLUDED_FAMILY_DIRS
            for task in flatten_family_tasks(family)]


def response_patterns(study):
    df = load_study(ROOT / TSV, study)
    items = study_items(df)
    item_key_by_subject_chunk = (df[["Subject", "chunk", "item"]].drop_duplicates()
                                 .set_index(["Subject", "chunk"])["item"].to_dict())
    patterns = {}
    for subject, part in df.groupby("Subject", sort=True):
        vals = part.groupby("item", sort=False).n400_resid.mean()
        patterns[str(subject)] = (vals.index.astype(str).to_numpy(), vals.to_numpy(float))
    return df, items, patterns, item_key_by_subject_chunk


def feature_array(family, task, items, study):
    rev = task.revision or "main"
    hf_rev = None if rev == "main" else rev
    feature_path = _feature_cache_path(CACHE, task.load_model_id, hf_rev, "all",
                                       _items_hash(items.context.tolist(), items.critical_word.tolist()))
    if feature_path.is_file():
        with np.load(feature_path) as z:
            states = np.asarray(z["features"], dtype=np.float32)
    else:
        # The two legacy pooled caches can be very large. Select the final layer
        # directly (memory-mapped where possible) instead of loading every layer.
        from neuroai.datasets.szewczyk import build_items_df, load_data
        from scripts.analysis.word_neural_embedding_rsa import _last_layer_states
        pooled = build_items_df(load_data(ROOT / OLD_RDS))
        ctx, wrd = pooled.context.tolist(), pooled.critical_word.tolist()
        pooled_path = _feature_cache_path(CACHE, task.load_model_id, hf_rev, "all",
                                          _items_hash(ctx, wrd))
        if not pooled_path.is_file():
            return None
        row_by_item = {}
        for i, key in enumerate(zip(ctx, wrd)):
            row_by_item.setdefault(key, i)
        rows = [row_by_item.get((c, w)) for c, w in zip(items.context, items.critical_word)]
        if any(i is None for i in rows):
            raise ValueError(f"Pooled embedding cache does not cover all items for {task.model_id}")
        states = _last_layer_states(pooled_path)[np.asarray(rows, dtype=np.int64)]
    if states.ndim == 2:
        states = states[:, None, :]
    if states.shape[0] != len(items) or not np.isfinite(states).all():
        raise ValueError(f"Invalid feature cache for {task.model_id}@{rev}: {states.shape}")
    return states[:, -1, :]


def item_index(items):
    return {str(item): i for i, item in enumerate(items.item.astype(str))}


def score_embedding(study, family, task, items, patterns):
    states = feature_array(family, task, items, study)
    if states is None:
        return None
    # Compute the embedding RDM once per model. Recomputing a high-dimensional
    # correlation RDM separately for every participant is needlessly expensive.
    embedding_rdm = squareform(pdist(states, metric="correlation"))
    positions = item_index(items)
    rows = []
    for subject, (ids, neural) in patterns.items():
        keep = [(i, positions[str(item)]) for i, item in enumerate(ids) if str(item) in positions]
        if len(keep) < 3:
            continue
        rrows, srows = zip(*keep)
        ids, neural = ids[list(rrows)], neural[list(rrows)]
        model_dist = embedding_rdm[np.ix_(srows, srows)][np.triu_indices(len(srows), k=1)]
        neural_dist = pdist(neural[:, None], metric="euclidean")
        score = float(pearsonr(model_dist, neural_dist).statistic)
        rows.append(dict(family=family, model_id=task.model_id, revision=task.revision or "main",
                         subject=subject, n_stimuli=len(ids), n_pairs=len(model_dist),
                         rsa_pearson=score))
    return pd.DataFrame(rows)


def surprisal_patterns(study, family, task, patterns, item_key_by_subject_chunk):
    root = ROOT / "outputs_jeanzay" / f"szewczyk_{study}" / "surprisal_sweep" / family / safe_model_dir(task.model_id)
    rev = (task.revision or "main").replace("/", "_")
    table_path = root / f"szewczyk_lme_pooled_table_{rev}.parquet"
    npy_path = root / f"szewczyk_lme_pooled_surprisal_{rev}.npy"
    if not table_path.is_file() or not npy_path.is_file():
        return None
    table = pd.read_parquet(table_path, columns=["subject", "chunk"])
    vals = np.load(npy_path, mmap_mode="r")
    if vals.ndim != 2 or vals.shape[0] != len(table):
        raise ValueError(f"Surprisal/table row mismatch: {npy_path}")
    table = table.copy()
    table["surprisal"] = np.asarray(vals[:, -1], dtype=float)
    # The extracted value can vary by subject when repeated item IDs occur in
    # different sentence contexts/conditions. Keep the exact subject × item
    # alignment used by the neural responses.
    table["subject"] = table.subject.astype(str)
    rows = []
    for subject, (ids, neural) in patterns.items():
        part = table.loc[table.subject == subject].copy()
        part["item"] = [item_key_by_subject_chunk[(subject, str(chunk))]
                        for chunk in part.chunk.astype(str)]
        part = part.groupby("item", as_index=False).surprisal.mean().set_index("item")
        surp = part.surprisal
        neural_by_id = {str(item): float(value) for item, value in zip(ids, neural)}
        shared = [str(i) for i in ids if str(i) in surp.index]
        if len(shared) < 3:
            continue
        nv = np.asarray([neural_by_id[i] for i in shared], dtype=float)
        sv = surp.loc[shared].to_numpy(dtype=float)
        model_dist = pdist(sv[:, None], metric="euclidean")
        neural_dist = pdist(nv[:, None], metric="euclidean")
        score = float(pearsonr(model_dist, neural_dist).statistic)
        rows.append(dict(family=family, model_id=task.model_id, revision=task.revision or "main",
                         subject=subject, n_stimuli=len(shared), n_pairs=len(model_dist),
                         rsa_pearson=score))
    return pd.DataFrame(rows)


def run_study(study):
    dataset = f"szewczyk_{study}"
    _, items, patterns, item_key_by_subject_chunk = response_patterns(study)
    score_files = list((ROOT / "outputs_jeanzay" / dataset / "embedding_sweep").glob(
        "*/*/*_item_mean_scores.csv"))
    score_keys = set()
    for path in score_files:
        table = pd.read_csv(path, dtype={"revision": str})
        score_keys.update(table[["family", "model_id", "revision"]].itertuples(index=False, name=None))
    expected = [(f, t) for f, t in registry_tasks()
                if (f, t.model_id, t.revision or "main") in score_keys]
    # Iterate only the exact registered model/revision keys in the saved embedding sweep.
    emb_rows, surp_rows = [], []
    for n, (family, task) in enumerate(expected, 1):
        revision = task.revision or "main"
        print(f"{dataset}: {n}/{len(expected)} {task.model_id}@{revision}", flush=True)
        emb = score_embedding(study, family, task, items, patterns)
        if emb is not None and not emb.empty:
            emb_rows.extend(emb.to_dict("records"))
        surp = surprisal_patterns(study, family, task, patterns, item_key_by_subject_chunk)
        if surp is not None and not surp.empty:
            surp_rows.extend(surp.to_dict("records"))
    for track, rows in (("embedding", emb_rows), ("surprisal", surp_rows)):
        out = ROOT / "outputs" / dataset / "pearson_rsa" / track
        out.mkdir(parents=True, exist_ok=True)
        by_model = pd.DataFrame(rows)
        if by_model.empty:
            raise RuntimeError(f"No {track} RSA scores for {dataset}")
        by_model.to_csv(out / "by_subject_scores.csv", index=False)
        summary = by_model.groupby(["family", "model_id", "revision"], as_index=False).agg(
            n_subjects=("subject", "nunique"), rsa_pearson=("rsa_pearson", "mean"))
        meta = [lookup_model_meta(row.model_id, row.family) for row in summary.itertuples()]
        summary["group"] = [m.group for m in meta]
        summary["n_params"] = [m.n_params for m in meta]
        if track == "embedding":
            enc = pd.concat([pd.read_csv(p, dtype={"revision": str}) for p in score_files])
            enc = enc[["family", "model_id", "revision", "last_layer_pearsonr"]].rename(
                columns={"last_layer_pearsonr": "encoding_pearson"})
            summary = summary.merge(enc, on=["family", "model_id", "revision"], how="left", validate="one_to_one")
        summary.to_csv(out / "model_summary.csv", index=False)
        print(f"{dataset}/{track}: {len(summary)} models", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("study", nargs="?", choices=sorted(STUDIES), default=None)
    args = parser.parse_args()
    for study in ([args.study] if args.study else sorted(STUDIES)):
        run_study(study)
