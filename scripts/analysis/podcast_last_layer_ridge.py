#!/usr/bin/env python3
"""Recompute Podcast ridge scores from the final layer on the benchmark window.

This matches the final-layer embedding RDM and keeps the original epoch
preprocessing on [-1, 1) s while fitting/scoring only the 32 bins in [-.2, .8).
One invocation processes one registry model across all nine subjects.
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

OUT = ROOT / "outputs/podcast/pearson_rsa/embedding/lagwise/ridge_last_layer"
CACHE_ROOT = Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/podcast_features")


def tasks():
    from neuroai.analysis.benchmark_common import EXCLUDED_FAMILY_DIRS
    from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
    return [(family, task.model_id, task.revision)
            for family in LPP_FAMILY_NAMES if family not in EXCLUDED_FAMILY_DIRS
            for task in flatten_family_tasks(family)]


def cache_for(model_id, revision):
    from neuroai.sweep import safe_model_dir
    rev = revision.replace("/", "_").replace("-", "_")
    if model_id.startswith("cclark/cbr-rnn") and revision == "main":
        path = CACHE_ROOT / f"{safe_model_dir(model_id)}__en__sec00__ctx48__cbrrnn.npz"
        return path if path.is_file() else None
    prefix = f"{safe_model_dir(model_id)}__{rev}__en__sec00__fulltext_"
    paths = [p for p in CACHE_ROOT.glob(f"{prefix}*__all.npz") if "logitlens" not in p.name]
    if len(paths) > 1:
        raise ValueError(f"Ambiguous feature cache for {model_id}@{revision}: {paths}")
    return paths[0] if paths else None


def run_model(index):
    import torch
    from neuroai.benchmarks.podcast import load_subject_epochs
    from neuroai.datasets.podcast import PodcastDataset
    from neuroai.regressors.ridge import cv_ridge_score_lag_matrix_gpu_batched
    from neuroai.sweep import safe_model_dir

    family, model_id, revision = tasks()[index]
    out = OUT / "by_model" / family / safe_model_dir(model_id) / f"ridge_{revision.replace('/', '_')}.csv"
    if out.is_file():
        old = pd.read_csv(out)
        if old.subject.nunique() == 9 and old.groupby("subject").time_s.nunique().eq(32).all():
            print(f"already complete {model_id}@{revision}", flush=True)
            return

    cache = cache_for(model_id, revision)
    if cache is None:
        raise FileNotFoundError(f"No Podcast embedding cache for {model_id}@{revision}")
    with np.load(cache) as z:
        features = z["features"]
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3:
        raise ValueError(f"Invalid embedding cache {cache}: shape={features.shape}")
    n_cache_words = features.shape[0]
    # Match PodcastBenchmark's finite-feature row mask across all cached layers.
    final_layer = features[:, -1, :]
    keep = np.empty(features.shape[0], dtype=bool)
    for start in range(0, len(keep), 64):
        stop = min(start + 64, len(keep))
        keep[start:stop] = np.isfinite(features[start:stop]).all(axis=(1, 2))
    states = np.asarray(final_layer[keep], dtype=np.float32)
    ds = PodcastDataset(ROOT / "data/podcast")
    words = ds.load_words()
    if len(words) != n_cache_words:
        raise ValueError(f"Transcript/cache row mismatch: {len(words)} != {n_cache_words}")
    onsets = words.onset.to_numpy(dtype=np.float64)[keep]
    word_ids = words.word_idx.to_numpy(dtype=np.int64)[keep]
    if not len(states) or not np.isfinite(states).all():
        raise ValueError(f"No finite final-layer embeddings in {cache}")
    # Large checkpoints have tens of gigabytes of all-layer activations in
    # their compressed cache. Keep only the selected final-layer matrix.
    del features, final_layer, keep
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Final-layer Podcast ridge requires the requested V100 GPU")
    try:
        from himalaya.backend import set_backend
        set_backend("torch_cuda")
    except Exception as exc:
        raise RuntimeError(f"Himalaya CUDA backend unavailable: {exc}") from exc

    rows = []
    for subject in [f"sub-{i:02d}" for i in range(1, 10)]:
        loaded = load_subject_epochs(
            ds, subject, onsets, tmin=-1.0, tmax=1.0, tmin_extract=-1.0,
            downsample_to=32.0, clip_iqr=5.0, baseline=None,
            baseline_mode="logratio",
            electrode_selection_dir=str(ROOT / "outputs_jeanzay/podcast/electrode_selection"),
        )
        if loaded is None:
            raise RuntimeError(f"No Podcast response epochs for {subject}")
        epochs, valid_mask, full_times, electrode_meta = loaded
        window = (full_times >= -0.2 - 1e-9) & (full_times < 0.8 - 1e-9)
        times = full_times[window]
        if len(times) != 32:
            raise ValueError(f"Expected 32 scored lags, got {len(times)}")
        X = states[valid_mask]
        # load_subject_epochs has already applied valid_mask to the epoch tensor;
        # it is retained separately only to select matching feature rows.
        Y = epochs[:, :, window]
        ids = word_ids[valid_mask]
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError(f"Nonfinite regression inputs for {model_id}/{subject}")
        n = len(X)
        fold_size = n // 3
        groups = np.zeros(n, dtype=np.int64)
        groups[fold_size:] = 1
        groups[2 * fold_size:] = 2
        result = cv_ridge_score_lag_matrix_gpu_batched(
            X, Y, times, groups=groups, n_splits=3,
            alpha=np.logspace(3.5, 8, 5), scale=True,
            standardize_target=True, random_state=0,
        )
        channels = list(electrode_meta)
        if len(result) != len(channels):
            raise ValueError(f"Electrode/result mismatch: {len(channels)} != {len(result)}")
        for channel, score in zip(channels, result):
            for lag, time_s in enumerate(times):
                rows.append(dict(family=family, model_id=model_id, revision=revision,
                                 subject=subject, electrode=channel,
                                 lag=lag, time_s=float(time_s),
                                 n_words=len(ids), pearsonr=float(score.pearsonr[lag]),
                                 ridge_alpha=float(score.ridge_alpha[lag])))
        print(f"{model_id}@{revision} {subject}: {len(channels)} channels × 32 lags", flush=True)
        del epochs, loaded, X, Y, result, valid_mask, full_times, electrode_meta
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    frame = pd.DataFrame(rows)
    if frame.subject.nunique() != 9 or not frame.groupby("subject").lag.nunique().eq(32).all():
        raise RuntimeError(f"Incomplete final-layer regression output: {model_id}@{revision}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"saved {model_id}@{revision}: {len(frame)} electrode-lags", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-index", type=int, required=True)
    a = p.parse_args()
    run_model(a.model_index)


if __name__ == "__main__":
    main()
