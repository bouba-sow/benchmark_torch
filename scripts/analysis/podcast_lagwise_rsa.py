#!/usr/bin/env python3
"""Podcast RSA at each scored lag, with track-specific neural epoch grids.

Prepare each subject once, then score registry models independently. Embedding
neural epochs reproduce the encoder's [-1, 1) preprocessing before selecting
[-.2, .8); surprisal uses the saved pooled-LME epochs on its native grid.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

OUTPUT = ROOT / "outputs/podcast/pearson_rsa"
# RSA_STANDARDIZE=1: z-score each embedding feature across words (Timkey & van Schijndel 2021);
# results go to by_model_std / model_summary*_std.csv so they never mix with the raw-state scores.
STD = os.environ.get("RSA_STANDARDIZE") == "1"
BY_MODEL = "by_model_std" if STD else "by_model"
SUFFIX = "_std" if STD else ""
LME_BASE = ROOT / "outputs_jeanzay/podcast/surprisal_sweep/gpt2/openai-community_gpt2"
EPOCH_SOURCE = ROOT / "outputs_jeanzay/podcast/electrode_selection"
N_LAGS = 32


def tasks():
    from neuroai.analysis.benchmark_common import EXCLUDED_FAMILY_DIRS
    from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
    return [(family, task.model_id, task.revision)
            for family in LPP_FAMILY_NAMES if family not in EXCLUDED_FAMILY_DIRS
            for task in flatten_family_tasks(family)]


def response_path(track, subject):
    return OUTPUT / track / "lagwise" / "response" / f"{subject}.npz"


def _common_word_ids(subject):
    table = pd.read_parquet(LME_BASE / "podcast_lme_pooled_table.parquet",
                            columns=["subject", "word_idx"])
    return np.sort(table.loc[table.subject == subject, "word_idx"].unique().astype(int))


def prepare(track, subject):
    out = response_path(track, subject)
    if out.is_file():
        with np.load(out) as z:
            assert z["distances"].shape == (N_LAGS, len(z["pairs"]))
        return print(f"validated {out}", flush=True)
    if track == "embedding":
        from neuroai.benchmarks.podcast import load_subject_epochs
        from neuroai.datasets.podcast import PodcastDataset
        ds = PodcastDataset(ROOT / "data/podcast")
        words = ds.load_words()
        loaded = load_subject_epochs(
            ds, subject, words.onset.to_numpy(dtype=float), tmin=-1.0,
            tmax=1.0, tmin_extract=-1.0, downsample_to=32.0,
            clip_iqr=5.0, baseline=None, baseline_mode="logratio",
            electrode_selection_dir=str(EPOCH_SOURCE))
        if loaded is None:
            raise RuntimeError(f"No encoder epochs for {subject}")
        epochs, valid, full_times, _ = loaded
        time_mask = (full_times >= -0.2 - 1e-9) & (full_times < 0.8 - 1e-9)
        times = full_times[time_mask]
        ids = words.word_idx.to_numpy(dtype=int)[valid]
        keep = np.isin(ids, _common_word_ids(subject))
        ids, patterns = ids[keep], epochs[keep][:, :, time_mask]
    else:
        table = pd.read_parquet(LME_BASE / "podcast_lme_pooled_table.parquet",
                                columns=["subject", "electrode", "word_idx"])
        y = np.load(LME_BASE / "podcast_lme_pooled_y.npy", mmap_mode="r")
        if y.shape != (len(table), N_LAGS):
            raise ValueError(f"LME response shape {y.shape} disagrees with table/lags")
        table = table.loc[table.subject == subject].copy()
        table["row"] = table.index
        wide = table.pivot(index="word_idx", columns="electrode", values="row")
        wide = wide.dropna(axis=0, how="any").sort_index()
        ids = wide.index.to_numpy(dtype=int)
        patterns = np.asarray(y[wide.to_numpy(dtype=int)], dtype=np.float32)
        times = -0.2 + np.arange(N_LAGS) / 32.0
    if len(times) != N_LAGS or patterns.shape[-1] != N_LAGS:
        raise ValueError(f"Expected 32 scored lags, got {len(times)} / {patterns.shape}")
    if len(ids) < 3 or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Invalid word IDs for {track}/{subject}")
    valid_words = np.isfinite(patterns).all(axis=(1, 2))
    valid_words &= np.std(patterns, axis=1).min(axis=1) > 1e-12
    ids, patterns = ids[valid_words], patterns[valid_words]
    from scripts.analysis.surprisal_rsa import _distances_for_pairs, _pair_distances
    _, _, pairs = _pair_distances(np.arange(len(ids), dtype=float),
                                  np.arange(len(ids), dtype=float), "podcast", subject)
    if pairs is None:
        raise ValueError("Podcast pair sampling unexpectedly returned a full RDM")
    distances = np.empty((N_LAGS, len(pairs)), dtype=np.float32)
    for lag in range(N_LAGS):
        distances[lag] = _distances_for_pairs(patterns[:, :, lag], pairs, "correlation")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, word_idx=ids, times=times, pairs=pairs,
                            distances=distances, patterns=patterns)
    os.replace(tmp, out)
    print(f"prepared {track}/{subject}: {len(ids)} words, {len(pairs)} pairs, {len(times)} lags", flush=True)


def _comparators(track, family, model_id, revision):
    from neuroai.sweep import safe_model_dir
    base = ROOT / "outputs_jeanzay/podcast" / ("embedding_sweep" if track == "embedding" else "surprisal_sweep")
    path = base / family / safe_model_dir(model_id) / (
        "podcast_electrode_timecourse.csv" if track == "embedding" else "podcast_lme_by_lag.csv")
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if track == "embedding":
        df = df.loc[df.revision == revision].copy()
        df["lag"] = np.rint((df.time_s + 1.0) * 32).astype(int)
        df = df.loc[df.lag.between(26, 57)]
        return df.groupby(["subject", "lag"]).pearsonr.mean().to_dict()
    df = df.loc[df.lag.between(0, 31)].copy()
    df["delta_ll_per_obs"] = df.delta_loglik_test / df.n_obs_test
    return df.set_index("lag").delta_ll_per_obs.to_dict()


def score(track, index):
    from neuroai.sweep import safe_model_dir
    from scripts.analysis.podcast_embedding_rsa import _cache_for, _embedding_distances
    from scripts.analysis.surprisal_rsa import _distances_for_pairs, _neural_cache_paths, _pair_distances
    family, model_id, revision = tasks()[index]
    out = OUTPUT / track / "lagwise" / BY_MODEL / family / safe_model_dir(model_id) / f"rsa_{revision.replace('/', '_')}.csv"
    if out.is_file():
        existing = pd.read_csv(out)
        if existing.groupby("subject").lag.nunique().eq(N_LAGS).all() and existing.subject.nunique() == 9:
            return print(f"validated {out}", flush=True)
    comparator = _comparators(track, family, model_id, revision)
    if track == "embedding":
        cache = _cache_for(model_id, revision)
        if cache is None:
            raise FileNotFoundError(f"No embedding cache for {model_id}@{revision}")
        with np.load(cache) as z:
            features = z["features"]
            if features.ndim == 2:
                features = features[:, None, :]
            states = np.asarray(features[:, -1, :], dtype=np.float32)
        if STD:
            states = (states - states.mean(0)) / (states.std(0) + 1e-8)
        model_map = None
    else:
        _, table_path = _neural_cache_paths("podcast", family, model_id, revision)
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        tab = pd.read_parquet(table_path, columns=["word_idx", "surprisal"])
        spread = tab.groupby("word_idx").surprisal.agg(lambda x: x.max() - x.min())
        if (spread > 1e-5).any():
            raise ValueError(f"Surprisal differs across electrodes: {table_path}")
        model_map = tab.drop_duplicates("word_idx").set_index("word_idx").surprisal
    rows = []
    for subject in [f"sub-{i:02d}" for i in range(1, 10)]:
        with np.load(response_path(track, subject)) as z:
            ids, times, pairs = z["word_idx"], z["times"], z["pairs"]
            neural = z["distances"]
            if track == "embedding":
                good = (ids >= 0) & (ids < len(states))
                good &= np.isfinite(states[np.clip(ids, 0, len(states)-1)]).all(axis=1)
            else:
                good = np.isin(ids, model_map.index.to_numpy())
                good &= np.isfinite(model_map.reindex(ids).to_numpy())
            if not good.all():
                # Preserve incomplete models: make a model-specific pair set.
                ids = ids[good]
                _, _, pairs = _pair_distances(np.arange(len(ids), dtype=float),
                                              np.arange(len(ids), dtype=float), "podcast", subject)
                patterns = z["patterns"][good]
                neural = np.stack([_distances_for_pairs(patterns[:, :, k], pairs, "correlation")
                                   for k in range(N_LAGS)])
            if len(ids) < 3:
                raise ValueError(f"Too few matched words for {model_id}/{subject}")
            if track == "embedding":
                model_dist = _embedding_distances(states, ids, pairs)
            else:
                values = model_map.reindex(ids).to_numpy(dtype=float)
                model_dist = np.abs(values[pairs[:, 0]] - values[pairs[:, 1]])
            for lag, time in enumerate(times):
                comp = comparator.get((subject, lag + 26) if track == "embedding" else lag, np.nan)
                rows.append(dict(family=family, model_id=model_id, revision=revision,
                                 subject=subject, lag=lag, time_s=float(time),
                                 n_stimuli=len(ids), n_pairs=len(pairs),
                                 rsa_pearson=float(pearsonr(model_dist, neural[lag]).statistic),
                                 benchmark_score=comp))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"scored {track} {model_id}@{revision}: {len(rows)} subject-lags", flush=True)


def summarize(track):
    from neuroai.analysis.benchmark_common import lookup_model_meta
    from neuroai.sweep import safe_model_dir
    root = OUTPUT / track / "lagwise"
    if track == "embedding":
        ridge_root = root / "ridge_last_layer" / "by_model"
        ridge_files = list(ridge_root.glob("*/*/ridge_*.csv"))
        if len(ridge_files) != len(tasks()):
            raise RuntimeError(f"Expected final-layer ridge results for {len(tasks())} models; found {len(ridge_files)}")
        for ridge_path in ridge_files:
            ridge = pd.read_csv(ridge_path)
            comp = ridge.groupby(["subject", "lag"], as_index=False).pearsonr.mean()
            if comp.subject.nunique() != 9 or not comp.groupby("subject").lag.nunique().eq(N_LAGS).all():
                raise RuntimeError(f"Incomplete last-layer/window ridge results: {ridge_path}")
            first = ridge.iloc[0]
            rsa_path = (root / BY_MODEL / first.family / safe_model_dir(first.model_id) /
                        f"rsa_{str(first.revision).replace('/', '_')}.csv")
            rsa = pd.read_csv(rsa_path)
            if "benchmark_score_best_layer_full_timeline" not in rsa:
                rsa = rsa.rename(columns={"benchmark_score": "benchmark_score_best_layer_full_timeline"})
            keyed = comp.set_index(["subject", "lag"]).pearsonr
            rsa["benchmark_score"] = [keyed.get((row.subject, row.lag), np.nan)
                                      for row in rsa.itertuples()]
            if not np.isfinite(rsa.benchmark_score).all():
                raise RuntimeError(f"Missing final-layer ridge comparator values for {rsa_path}")
            rsa.to_csv(rsa_path, index=False)
    rows = []
    for path in root.glob(f"{BY_MODEL}/*/*/rsa_*.csv"):
        df = pd.read_csv(path)
        if df.subject.nunique() != 9 or not df.groupby("subject").lag.nunique().eq(N_LAGS).all():
            continue
        first = df.iloc[0]
        meta = lookup_model_meta(first.model_id, first.family)
        for lag, part in df.groupby("lag"):
            first = part.iloc[0]
            rows.append(dict(family=first.family, model_id=first.model_id,
                             revision=first.revision, lag=lag, time_s=first.time_s,
                             n_subjects=part.subject.nunique(),
                             rsa_pearson=part.rsa_pearson.mean(),
                             benchmark_score=part.benchmark_score.mean(),
                             group=meta.group, n_params=meta.n_params))
    if not rows:
        raise RuntimeError(f"No complete {track} lagwise scores")
    by_lag = pd.DataFrame(rows).sort_values(["lag", "family", "model_id"])
    by_lag.to_csv(root / f"model_summary_by_lag{SUFFIX}.csv", index=False)
    averaged = (by_lag.groupby(["family", "model_id", "revision"], as_index=False)
                .agg(rsa_pearson=("rsa_pearson", "mean"),
                     benchmark_score=("benchmark_score", "mean"),
                     n_lags_rsa=("rsa_pearson", "count"),
                     n_lags_benchmark=("benchmark_score", "count"),
                     n_subjects=("n_subjects", "min"),
                     n_params=("n_params", "first"), group=("group", "first")))
    averaged.to_csv(root / f"model_summary{SUFFIX}.csv", index=False)
    print(f"summarized {track}: {len(averaged)} models / {len(rows)} model-lags", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("track", choices=["embedding", "surprisal"])
    p.add_argument("action", choices=["prepare", "score", "summarize"])
    p.add_argument("--subject")
    p.add_argument("--model-index", type=int)
    a = p.parse_args()
    if a.action == "prepare":
        if not a.subject:
            p.error("--subject is required for prepare")
        prepare(a.track, a.subject)
    elif a.action == "score":
        if a.model_index is None:
            p.error("--model-index is required for score")
        score(a.track, a.model_index)
    else:
        summarize(a.track)


if __name__ == "__main__":
    main()
