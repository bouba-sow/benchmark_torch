#!/usr/bin/env python3
"""Compute last-layer RDM regression and split-half ceilings for the two new N400 studies."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from neuroai.analysis.rdm_ceiling import RdmTargetReplicates, combine_halves, split_half_targets
from neuroai.analysis.rdm_regression import RdmCvPlan, all_pairs
from neuroai.benchmarks.eeg_erp import _items_hash
from neuroai.datasets.szewczyk_paper import STUDIES, TSV, load_study, study_items, item_means
from neuroai.features.word_frequency import compute_wlen_wordfreq_covariates
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
from neuroai.sweep import safe_model_dir
from scripts.analysis.n400_paper_rsa import CACHE, feature_array

N_REPEATS, N_SPLITS, N_PARTITIONS = 30, 15, 5
OUT = ROOT / "reports/rsa_summary/rdm_regression"


def tasks_with_scores(study):
    files = list((ROOT / "outputs_jeanzay" / f"szewczyk_{study}" / "embedding_sweep").glob(
        "*/*/*_item_mean_scores.csv"))
    keys = set()
    for p in files:
        d = pd.read_csv(p, dtype={"revision": str})
        keys.update(d[["family", "model_id", "revision"]].itertuples(index=False, name=None))
    return [(f, t) for f in LPP_FAMILY_NAMES for t in flatten_family_tasks(f)
            if (f, t.model_id, t.revision or "main") in keys]


def prepare(study):
    trial = load_study(ROOT / TSV, study)
    items = study_items(trial)
    items["item_idx"] = np.arange(len(items))
    item_idx = dict(zip(items.item.astype(str), items.item_idx))
    trial["item_idx"] = trial.item.astype(str).map(item_idx)
    # Subject/item cells: baseline-corrected response used by the paper-specific item-mean track.
    trial = trial.groupby(["Subject", "item_idx"], as_index=False).n400_resid.mean()
    counts = trial.groupby("item_idx").Subject.nunique()
    keep = set(counts[counts >= 5].index.astype(int))
    # Lexical controls are computed once for each critical word in exact feature-cache order.
    lexical = compute_wlen_wordfreq_covariates(items.critical_word.astype(str).tolist())
    items["wlen"] = [x[0] for x in lexical]
    items["freq"] = [x[1] for x in lexical]
    tasks = tasks_with_scores(study)
    # Same eligible items for all model RDMs: intersection of per-model surprisal coverage.
    shared = None
    source_map = (load_study(ROOT / TSV, study)[["Subject", "chunk", "item"]].drop_duplicates()
                  .set_index(["Subject", "chunk"]).item.to_dict())
    for family, task in tasks:
        root = ROOT / "outputs_jeanzay" / f"szewczyk_{study}" / "surprisal_sweep" / family / safe_model_dir(task.model_id)
        rev = (task.revision or "main").replace("/", "_")
        table_path = root / f"szewczyk_lme_pooled_table_{rev}.parquet"
        array_path = root / f"szewczyk_lme_pooled_surprisal_{rev}.npy"
        table = pd.read_parquet(table_path, columns=["subject", "chunk"])
        arr = np.load(array_path, mmap_mode="r")
        if arr.ndim != 2 or len(arr) != len(table):
            raise ValueError(f"Surprisal/table row mismatch: {array_path}")
        table["surprisal"] = np.asarray(arr[:, -1], dtype=float)
        table["item_idx"] = [item_idx[str(source_map[(str(s), str(c))])]
                             for s, c in zip(table.subject, table.chunk)]
        available = set(table.loc[np.isfinite(table.surprisal), "item_idx"].astype(int)) & keep
        shared = available if shared is None else shared & available
    if shared is None:
        raise FileNotFoundError(f"No surprisal cache coverage for {study}")
    selected = np.asarray(sorted(shared), dtype=int)
    items = items.set_index("item_idx").loc[selected].reset_index()
    lookup = {int(old): new for new, old in enumerate(selected)}
    trial = trial.loc[trial.item_idx.isin(lookup)].copy()
    trial["item_idx"] = trial.item_idx.map(lookup).astype(int)
    values = trial.n400_resid.to_numpy(float)
    subj = pd.factorize(trial.Subject.astype(str))[0]
    item = trial.item_idx.to_numpy(int)
    y = items.y.to_numpy(float) if "y" in items else np.asarray([
        trial.loc[trial.item_idx == i, "n400_resid"].mean() for i in range(len(items))])
    pairs = all_pairs(len(items))
    response = pdist(y[:, None], metric="euclidean")
    conf = np.column_stack([pdist(items[c].to_numpy(float)[:, None], metric="euclidean")
                            for c in ("wlen", "freq")])
    plans = {"none": RdmCvPlan(response, len(items), pairs, None, N_REPEATS, seed=0),
             "len_freq": RdmCvPlan(response, len(items), pairs, conf, N_REPEATS, seed=0)}
    means, valid, halves = split_half_targets(values, subj, item, len(items), n_splits=N_SPLITS,
                                             seed=study_seed(study))
    responses = np.abs(np.nan_to_num(means)[:, pairs[:, 0]] - np.nan_to_num(means)[:, pairs[:, 1]])
    rep = RdmTargetReplicates(responses, len(items), pairs, conf, valid,
                              n_partitions=N_PARTITIONS, seed=study_seed(study))
    partner = np.empty(2 * N_SPLITS, dtype=int)
    for a, b in halves:
        partner[a], partner[b] = b, a
    delta, n_test = rep.score_partners(partner)
    ceiling = combine_halves(delta, n_test, halves)[0]
    return trial, items, item_idx, selected, lookup, source_map, tasks, pairs, plans, rep, halves, ceiling


def study_seed(study):
    return 2007 if study == "f2007" else 2019


def model_rdms(study, items, item_idx, selected, lookup, source_map, tasks):
    model_rdm = {"embedding": [], "surprisal": []}
    identifiers = []
    all_items = study_items(load_study(ROOT / TSV, study))
    for family, task in tasks:
        ids = (family, task.model_id, task.revision or "main")
        states = feature_array(family, task, all_items, study)
        if states is None:
            raise FileNotFoundError(f"Missing embedding cache for {ids}")
        edist = pdist(states[selected], metric="correlation").astype(np.float32)
        root = ROOT / "outputs_jeanzay" / f"szewczyk_{study}" / "surprisal_sweep" / family / safe_model_dir(task.model_id)
        rev = (task.revision or "main").replace("/", "_")
        table_path = root / f"szewczyk_lme_pooled_table_{rev}.parquet"
        array_path = root / f"szewczyk_lme_pooled_surprisal_{rev}.npy"
        table = pd.read_parquet(table_path, columns=["subject", "chunk"])
        a = np.load(array_path, mmap_mode="r")
        table["surprisal"] = np.asarray(a[:, -1], dtype=float)
        table["item_idx"] = [item_idx[str(source_map[(str(s), str(c))])]
                             for s, c in zip(table.subject, table.chunk)]
        by_item = table.groupby("item_idx").surprisal.mean()
        sdist = pdist(np.asarray([by_item.loc[int(i)] for i in selected])[:, None],
                      metric="euclidean").astype(np.float32)
        identifiers.append(ids)
        model_rdm["embedding"].append(edist)
        model_rdm["surprisal"].append(sdist)
        print(f"{study}: cached RDMs {len(identifiers)}/{len(tasks)} {task.model_id}", flush=True)
    return identifiers, model_rdm


def run(study, limit=None):
    dataset = f"szewczyk_{study}_avg"
    (trial, items, item_idx, selected, lookup, source_map, tasks, pairs,
     plans, rep, halves, ceiling) = prepare(study)
    if limit:
        tasks = tasks[:limit]
    identifiers, rdms = model_rdms(study, items, item_idx, selected, lookup, source_map, tasks)
    records, ceiling_records = [], []
    for track in ("embedding", "surprisal"):
        print(f"{study}: scoring {track} ({len(rdms[track])} model RDMs)", flush=True)
        matrices = np.asarray(rdms[track], dtype=np.float32)
        del rdms[track]
        # Bound temporary predictor arrays in RdmTargetReplicates for the large
        # split-half target set by scoring models in small batches.
        score_batches = []
        for start in range(0, len(matrices), 4):
            half_delta, n_test = rep.score_models(matrices[start:start + 4])
            score_batches.append(combine_halves(half_delta, n_test, halves))
        half_scores = np.concatenate(score_batches, axis=0)
        print(f"{study}: half-target scoring finished for {track}", flush=True)
        for i, (family, model_id, revision) in enumerate(identifiers):
            for variant in ("none", "len_freq"):
                result = plans[variant].delta_ll(matrices[i])
                records.append(dict(dataset=dataset, track=track, family=family, model_id=model_id,
                                    revision=revision, unit="all", confounds=variant,
                                    last_layer_delta_ll_per_obs=result.delta_ll_per_obs,
                                    group="", n_params=np.nan, point_delta_ll_per_obs=np.nan,
                                    boot_se=np.nan, boot_sd_uncorrected=np.nan, ci_low=np.nan, ci_high=np.nan,
                                    ci_excludes_zero=np.nan,
                                    ceiling_delta_ll_per_obs=float(np.nanmean(ceiling[1])),
                                    half_target_delta_ll_per_obs=float(np.nanmean(half_scores[i, 1])),
                                    ratio_to_ceiling=float(np.nanmean(half_scores[i, 1]) / np.nanmean(ceiling[1])),
                                    ceiling_type="split_half_half_target",
                                    ceiling_upper_delta_ll_per_obs=np.nan, ratio_to_upper=np.nan))
    # Replace model metadata from the registry records used by the RSA summaries.
    meta = pd.read_csv(ROOT / "outputs" / f"szewczyk_{study}" / "pearson_rsa" / "embedding" / "model_summary.csv",
                       dtype={"revision": str})
    meta = meta[["family", "model_id", "revision", "group", "n_params"]]
    out = pd.DataFrame(records).drop(columns=["group", "n_params"]).merge(
        meta, on=["family", "model_id", "revision"], validate="many_to_one")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("study", nargs="?", choices=sorted(STUDIES))
    parser.add_argument("--out", type=Path, default=OUT / "rdm_n400_split_results.csv")
    parser.add_argument("--limit", type=int, default=None, help="limit models for a smoke run")
    args = parser.parse_args()
    frames = [run(s, args.limit) for s in ([args.study] if args.study else sorted(STUDIES))]
    pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
    print(f"Wrote {args.out}", flush=True)
