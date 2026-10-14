#!/usr/bin/env python3
"""Last-layer embedding RSA for Podcast, using the 50–250 ms response window.

The Podcast embedding caches are word-indexed. Neural RDMs use each subject's
electrode pattern after averaging each word's response over the requested time
window. Large RDMs use the same deterministic 250,000-pair sampling as the
Podcast surprisal RSA and retain the sampled indices in the cache.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = "/tmp/podcast-embedding-rsa-matplotlib"

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

from neuroai.analysis.benchmark_common import lookup_model_meta
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
from neuroai.sweep import safe_model_dir
from scripts.analysis.surprisal_rsa import (
    MAX_PAIRS, _distances_for_pairs, _pair_distances, _prepare_neural_patterns,
)

CACHE_ROOT = Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/podcast_features")


def _cache_for(model_id: str, revision: str) -> Path | None:
    rev = revision.replace('/', '_').replace('-', '_')
    prefix = f"{safe_model_dir(model_id)}__{rev}__en__sec00__fulltext_"
    pattern = (f"{safe_model_dir(model_id)}__en__sec00__ctx48__cbrrnn.npz"
               if model_id.startswith("cclark/cbr-rnn") and revision == "main"
               else f"{prefix}*__all.npz")
    paths = [p for p in CACHE_ROOT.glob(pattern) if "logitlens" not in p.name]
    if len(paths) > 1:
        raise ValueError(f"Ambiguous Podcast embedding cache for {model_id}@{revision}: {paths}")
    return paths[0] if paths else None


def _tasks():
    root = ROOT / "outputs_jeanzay/podcast/embedding_sweep"
    tasks = []
    for family in LPP_FAMILY_NAMES:
        for task in flatten_family_tasks(family):
            score_path = root / family / safe_model_dir(task.model_id) / "podcast_electrode_timecourse.csv"
            cache = _cache_for(task.model_id, task.revision)
            if score_path.is_file() and cache is not None:
                tasks.append((family, task.model_id, task.revision, cache, score_path))
    return sorted(tasks, key=lambda task: task[3].stat().st_size)


def _encoding_by_subject(path: Path, revision: str) -> pd.Series:
    scores = pd.read_csv(path)
    scores = scores.loc[(scores.revision == revision) &
                        scores.time_s.between(0.05 - 1e-9, 0.25 + 1e-9)]
    if scores.empty:
        raise ValueError(f"No Podcast encoding scores in 50–250 ms window: {path}")
    return scores.groupby("subject").pearsonr.mean()


def _embedding_distances(states: np.ndarray, rows: np.ndarray,
                         pair_indices: np.ndarray) -> np.ndarray:
    """Correlation distances for sampled row pairs without allocating an RDM."""
    selected = np.asarray(states[rows], dtype=np.float32)
    selected -= selected.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(selected, axis=1)
    if (norms <= np.finfo(np.float32).tiny).any():
        raise ValueError("A Podcast embedding has zero centered norm")
    selected /= norms[:, None]
    distances = np.empty(len(pair_indices), dtype=np.float32)
    chunk = 4096
    for start in range(0, len(pair_indices), chunk):
        part = pair_indices[start:start + chunk]
        similarity = np.einsum("ij,ij->i", selected[part[:, 0]], selected[part[:, 1]])
        distances[start:start + len(part)] = 1 - np.clip(similarity, -1, 1)
    return distances


def _score(task, neural_patterns, output_root: Path):
    family, model_id, revision, cache, encoding_path = task
    existing = (output_root / "by_model" / family / safe_model_dir(model_id) /
                f"rsa_{revision.replace('/', '_')}.csv")
    if existing.is_file():
        return f"existing {model_id}@{revision}"
    with np.load(cache) as archive:
        features = archive["features"]
        if features.ndim == 2 and cache.name.endswith("__cbrrnn.npz"):
            features = features[:, None, :]
        if features.ndim != 3 or features.shape[-1] < 2:
            return f"skipped unusable embedding cache {model_id}@{revision}: {features.shape}"
        states = np.asarray(features[:, -1, :], dtype=np.float32)
    if not np.isfinite(states).all():
        raise ValueError(f"Nonfinite Podcast embeddings in {cache}")

    encoding = _encoding_by_subject(encoding_path, revision)
    rows = []
    for subject, (neural_ids, neural_values) in neural_patterns.items():
        # `word_idx` is the original transcript index, which is also the row
        # index in the full-transcript embedding cache.
        cache_ids = np.asarray([int(str(item).rsplit("::", 1)[1]) for item in neural_ids])
        valid = (cache_ids >= 0) & (cache_ids < len(states))
        cache_ids, neural_ids, neural_values = cache_ids[valid], neural_ids[valid], neural_values[valid]
        if len(cache_ids) < 3 or len(np.unique(cache_ids)) != len(cache_ids):
            continue
        _, _, pair_indices = _pair_distances(
            np.arange(len(cache_ids), dtype=float), np.arange(len(cache_ids), dtype=float),
            "podcast", subject,
        )
        if pair_indices is None:
            continue
        model_dist = _embedding_distances(states, cache_ids, pair_indices)
        neural_dist = _distances_for_pairs(neural_values, pair_indices, "correlation")
        score = float(pearsonr(model_dist, neural_dist).statistic)
        rows.append({"family": family, "model_id": model_id, "revision": revision,
                     "subject": subject, "n_stimuli": len(cache_ids),
                     "n_pairs": len(pair_indices), "rsa_pearson": score,
                     "encoding_pearson_50_250ms": float(encoding.get(subject, np.nan))})

        out = output_root / "rdm_cache" / family / safe_model_dir(model_id) / revision.replace('/', '_') / f"{subject}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, stimulus_ids=neural_ids.astype(str),
                                pair_indices=pair_indices, distances=model_dist,
                                rdm_type="pearson_embedding_distance", dataset="podcast",
                                model_id=model_id, revision=revision, subject=subject,
                                response_window_s=np.array([0.05, 0.25]))
        os.replace(temporary, out)
    out = output_root / "by_model" / family / safe_model_dir(model_id) / f"rsa_{revision.replace('/', '_')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return f"scored {model_id}@{revision}: {np.mean([r['rsa_pearson'] for r in rows]):.4f}"


def _summarize(output_root: Path):
    rows = []
    for path in output_root.glob("by_model/*/*/rsa_*.csv"):
        scores = pd.read_csv(path)
        if scores.empty:
            continue
        first = scores.iloc[0]
        meta = lookup_model_meta(first.model_id, first.family)
        rows.append({"family": first.family, "model_id": first.model_id,
                     "revision": first.revision, "n_subjects": scores.subject.nunique(),
                     "rsa_pearson": scores.rsa_pearson.mean(),
                     "encoding_pearson_50_250ms": scores.encoding_pearson_50_250ms.mean(),
                     "group": meta.group, "n_params": meta.n_params})
    if not rows:
        print("No Podcast embedding RSA scores yet", flush=True)
        return
    summary = pd.DataFrame(rows).sort_values(["family", "model_id", "revision"])
    summary.to_csv(output_root / "model_summary.csv", index=False)
    if len(summary) >= 3:
        print(f"podcast: {len(summary)} models; encoding-window vs RSA r="
              f"{pearsonr(summary.encoding_pearson_50_250ms, summary.rsa_pearson).statistic:.4f}", flush=True)
    valid = summary.loc[np.isfinite(summary.n_params) & (summary.n_params > 0)].copy()
    valid["log10_params"] = np.log10(valid.n_params)
    trends = []
    if len(valid) >= 3 and valid.log10_params.nunique() >= 3:
        for score in ("rsa_pearson", "encoding_pearson_50_250ms"):
            fit = linregress(valid.log10_params, valid[score])
            trends.append({"scope": "all", "score": score, "n_models": len(valid),
                           "slope_per_log10_params": fit.slope, "pearson_r": fit.rvalue})
    pd.DataFrame(trends).to_csv(output_root / "scaling_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output_root = ROOT / "outputs/podcast/pearson_rsa/embedding"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        _summarize(output_root)
        return
    tasks = _tasks()
    if args.limit:
        tasks = tasks[:args.limit]
    neural_patterns = _prepare_neural_patterns("podcast")
    print(f"podcast: {len(tasks)} models, {len(neural_patterns)} subject response patterns", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_score, task, neural_patterns, output_root) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    _summarize(output_root)


if __name__ == "__main__":
    main()
