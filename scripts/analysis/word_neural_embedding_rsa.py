#!/usr/bin/env python3
"""Last-layer embedding RSA for word-level EEG/MEG response datasets.

This is a cache-only companion to ``podcast_embedding_rsa.py``. Broderick and
MEG-MASC embeddings align by the original global ``word_idx``; Nieuwland uses
the 0-based item-condition ID; Szewczyk maps subject/frame records to the
feature cache's unique critical-word item order. Subject-specific neural
response patterns and sampled RDM pairs are retained with each score.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = "/tmp/word-neural-embedding-rsa"

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import linregress, pearsonr

from neuroai.analysis.benchmark_common import lookup_model_meta
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
from neuroai.sweep import safe_model_dir
from scripts.analysis.surprisal_rsa import (
    MAX_PAIRS, _distances_for_pairs, _pair_distances, _prepare_neural_patterns,
)

STANDARDIZE = False

CACHE_ROOTS = {
    "broderick": Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/broderick_features"),
    "megmasc": Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/megmasc_features"),
    "nieuwland": Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/nieuwland_features"),
    "szewczyk": Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/szewczyk_features"),
}


def _cache_for(dataset: str, model_id: str, revision: str) -> Path | None:
    if dataset in {"nieuwland", "szewczyk"}:
        prefix = f"{safe_model_dir(model_id)}_{revision.replace('/', '_')}_all_"
        paths = list(CACHE_ROOTS[dataset].glob(f"{prefix}*.npz"))
        if len(paths) > 1:
            raise ValueError(f"Ambiguous {dataset} embedding cache for {model_id}@{revision}: {paths}")
        return paths[0] if paths else None
    rev = revision.replace('/', '_').replace('-', '_')
    prefix = f"{safe_model_dir(model_id)}__{rev}__en__sec00__fulltext_"
    pattern = (f"{safe_model_dir(model_id)}__en__sec00__ctx48__cbrrnn.npz"
               if model_id.startswith("cclark/cbr-rnn") and revision == "main"
               else f"{prefix}*__all.npz")
    paths = [p for p in CACHE_ROOTS[dataset].glob(pattern)
             if "logitlens" not in p.name]
    if not paths and model_id.startswith("cclark/cbr-rnn"):
        # MEG-MASC caches the CBR-RNN models in the regular full-text format.
        paths = [p for p in CACHE_ROOTS[dataset].glob(f"{prefix}*__all.npz") if "logitlens" not in p.name]
    if len(paths) > 1:
        raise ValueError(f"Ambiguous {dataset} embedding cache for {model_id}@{revision}: {paths}")
    return paths[0] if paths else None


def _tasks(dataset: str):
    root = ROOT / "outputs_jeanzay" / dataset / "embedding_sweep"
    score_name = "nieuwland_checkpoint_layerwise.csv" if dataset == "nieuwland" else f"{dataset}_by_subject.csv"
    tasks = []
    for family in LPP_FAMILY_NAMES:
        for task in flatten_family_tasks(family):
            score_path = root / family / safe_model_dir(task.model_id) / score_name
            if dataset == "szewczyk" and not score_path.is_file():
                score_path = ROOT / "outputs" / dataset / "embedding_sweep" / family / safe_model_dir(task.model_id) / score_name
            cache = _cache_for(dataset, task.model_id, task.revision)
            if score_path.is_file() and cache is not None:
                tasks.append((family, task.model_id, task.revision, cache, score_path))
    return sorted(tasks, key=lambda task: task[3].stat().st_size)


def _encoding_by_subject(path: Path, revision: str):
    frame = pd.read_csv(path, dtype={"subject": str})
    if "layer" in frame and "mean_pearsonr" in frame:
        frame = frame.loc[frame.revision.astype(str) == revision]
        if frame.empty:
            raise ValueError(f"No Nieuwland layer scores for revision {revision}: {path}")
        last = frame.loc[frame.layer == frame.layer.max()]
        return float(last.mean_pearsonr.iloc[0])
    frame = frame.loc[frame.revision.astype(str) == revision]
    if "last_layer_mean_pearsonr" in frame:
        return frame.groupby("subject").last_layer_mean_pearsonr.mean()
    if "pearsonr_mean" not in frame or frame.empty:
        raise ValueError(f"Missing subject-level last-layer encoding scores: {path}")
    return frame.groupby("subject").pearsonr_mean.mean()


def _embedding_distances(states: np.ndarray, indices: np.ndarray,
                         pair_indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(states[indices], dtype=np.float32)
    selected -= selected.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(selected, axis=1)
    if (norms <= np.finfo(np.float32).tiny).any():
        raise ValueError("A cached embedding has zero centered norm")
    selected /= norms[:, None]
    distances = np.empty(len(pair_indices), dtype=np.float32)
    for start in range(0, len(pair_indices), 4096):
        pairs = pair_indices[start:start + 4096]
        sim = np.einsum("ij,ij->i", selected[pairs[:, 0]], selected[pairs[:, 1]])
        distances[start:start + len(pairs)] = 1 - np.clip(sim, -1, 1)
    return distances


def _last_layer_states(cache: Path) -> np.ndarray:
    """Load the final layer, memory-mapping very large stored NPZ members.

    Some full-text caches are tens of gigabytes. Their ``features.npy`` member
    is ZIP_STORED, so it can be memory-mapped directly inside the NPZ. This
    avoids materializing every layer and a second float32 copy just to select
    the last layer.
    """
    with zipfile.ZipFile(cache) as archive:
        info = archive.getinfo("features.npy")
        if info.compress_type == zipfile.ZIP_STORED and info.file_size >= 2_000_000_000:
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    shape, fortran_order, dtype = np.lib.format._read_array_header(member, version)
                header_bytes = member.tell()
            if len(shape) == 3 and not fortran_order:
                with cache.open("rb") as raw:
                    raw.seek(info.header_offset)
                    fields = struct.unpack("<IHHHHHIIIHH", raw.read(30))
                name_length, extra_length = fields[-2:]
                data_offset = info.header_offset + 30 + name_length + extra_length + header_bytes
                mapped = np.memmap(cache, mode="r", dtype=dtype, offset=data_offset,
                                   shape=shape, order="C")
                return mapped[:, -1, :]
    with np.load(cache) as archive:
        features = archive["features"]
        if features.ndim == 2 and cache.name.endswith("__cbrrnn.npz"):
            return np.asarray(features, dtype=np.float32)
        if features.ndim != 3:
            raise ValueError(f"Expected (word, layer, feature) cache, got {features.shape}: {cache}")
        return np.asarray(features[:, -1, :], dtype=np.float32)


def _szewczyk_neural_patterns(table_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Join subject N400 values to the exact item order used by the embedding cache."""
    table = pd.read_parquet(table_path, columns=["subject", "chunk", "n400"])
    source_path = ROOT / "data/n400_paper_formula/osfstorage/datasets/szewczyk_2022.tsv"
    source = pd.read_csv(source_path, sep="\t", dtype={"Subject": str, "Item": str})
    source["chunk"] = source.dataset.astype(str) + "_" + source.Item.astype(str)
    items = source[["FullText", "TargetWords"]].drop_duplicates().reset_index(drop=True)
    items["item_id"] = np.arange(len(items), dtype=np.int64)
    source = source[["Subject", "chunk", "FullText", "TargetWords"]].drop_duplicates()
    source.rename(columns={"Subject": "subject"}, inplace=True)
    source = source.merge(items, on=["FullText", "TargetWords"], validate="many_to_one")
    aligned = table.merge(source[["subject", "chunk", "item_id"]],
                          on=["subject", "chunk"], how="inner", validate="one_to_one")
    if len(aligned) != len(table):
        raise ValueError(f"Szewczyk N400/embedding item alignment lost {len(table)-len(aligned)} rows")
    patterns = {}
    for subject, part in aligned.groupby("subject", sort=True):
        part = part.groupby("item_id", as_index=False).n400.mean().sort_values("item_id")
        patterns[str(subject)] = (part.item_id.to_numpy(dtype=np.int64),
                                  part.n400.to_numpy(dtype=np.float32)[:, None])
    return patterns


def _score(task, dataset, neural_patterns, output_root: Path):
    family, model_id, revision, cache, encoding_path = task
    existing = (output_root / "by_model" / family / safe_model_dir(model_id) /
                f"rsa_{revision.replace('/', '_')}.csv")
    if existing.is_file():
        return f"existing {model_id}@{revision}"
    states = _last_layer_states(cache)
    if not np.isfinite(states).all():
        raise ValueError(f"Nonfinite embeddings in {cache}")
    if STANDARDIZE:
        states = np.asarray(states, dtype=np.float32)
        states = (states - states.mean(0)) / (states.std(0) + 1e-8)

    encoding = _encoding_by_subject(encoding_path, revision)
    rows = []
    for subject, (stimulus_ids, neural_values) in neural_patterns.items():
        if np.issubdtype(np.asarray(stimulus_ids).dtype, np.number):
            word_indices = np.asarray(stimulus_ids, dtype=np.int64)
        else:
            word_indices = np.asarray([int(str(item).rsplit("::", 1)[1]) for item in stimulus_ids])
        valid = (word_indices >= 0) & (word_indices < len(states))
        word_indices, stimulus_ids, neural_values = (
            word_indices[valid], stimulus_ids[valid], neural_values[valid]
        )
        if len(word_indices) < 3 or len(np.unique(word_indices)) != len(word_indices):
            continue
        _, _, pair_indices = _pair_distances(
            np.arange(len(word_indices), dtype=float), np.arange(len(word_indices), dtype=float),
            dataset, subject,
        )
        neural_metric = "correlation" if neural_values.shape[1] > 1 else "euclidean"
        if pair_indices is None:
            model_dist = pdist(states[word_indices], metric="correlation").astype(np.float32)
            neural_dist = pdist(neural_values, metric=neural_metric).astype(np.float32)
        else:
            model_dist = _embedding_distances(states, word_indices, pair_indices)
            neural_dist = _distances_for_pairs(neural_values, pair_indices, neural_metric)
        rsa = float(pearsonr(model_dist, neural_dist).statistic)
        enc = float(encoding.get(subject, np.nan)) if isinstance(encoding, pd.Series) else float(encoding)
        rows.append({"family": family, "model_id": model_id, "revision": revision,
                     "subject": subject, "n_stimuli": len(word_indices),
                     "n_pairs": len(model_dist), "rsa_pearson": rsa,
                     "encoding_pearson": enc})

        out = (output_root / "rdm_cache" / family / safe_model_dir(model_id)
               / revision.replace('/', '_') / f"{subject}.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        metadata = {"stimulus_ids": stimulus_ids.astype(str), "distances": model_dist,
                    "neural_distances": neural_dist,
                    "rdm_type": "pearson_embedding_distance", "dataset": dataset,
                    "model_id": model_id, "revision": revision, "subject": subject}
        if pair_indices is not None:
            metadata["pair_indices"] = pair_indices
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **metadata)
        os.replace(temporary, out)
    out = output_root / "by_model" / family / safe_model_dir(model_id) / f"rsa_{revision.replace('/', '_')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return f"scored {model_id}@{revision}: {np.mean([r['rsa_pearson'] for r in rows]):.4f}"


def _summarize(dataset: str, output_root: Path):
    rows = []
    for path in output_root.glob("by_model/*/*/rsa_*.csv"):
        df = pd.read_csv(path)
        if df.empty:
            continue
        first = df.iloc[0]
        meta = lookup_model_meta(first.model_id, first.family)
        rows.append({"family": first.family, "model_id": first.model_id,
                     "revision": first.revision, "n_subjects": df.subject.nunique(),
                     "rsa_pearson": df.rsa_pearson.mean(),
                     "encoding_pearson": df.encoding_pearson.mean(),
                     "group": meta.group, "n_params": meta.n_params})
    if not rows:
        print(f"{dataset}: no embedding RSA scores", flush=True)
        return
    summary = pd.DataFrame(rows).sort_values(["family", "model_id", "revision"])
    summary.to_csv(output_root / "model_summary.csv", index=False)
    if len(summary) >= 3:
        r = pearsonr(summary.rsa_pearson, summary.encoding_pearson).statistic
        print(f"{dataset}: {len(summary)} models; encoding vs RSA r={r:.4f}", flush=True)
    valid = summary.loc[np.isfinite(summary.n_params) & (summary.n_params > 0)].copy()
    valid["log10_params"] = np.log10(valid.n_params)
    trends = []
    if len(valid) >= 3 and valid.log10_params.nunique() >= 3:
        for metric in ("rsa_pearson", "encoding_pearson"):
            fit = linregress(valid.log10_params, valid[metric])
            trends.append({"scope": "all", "metric": metric, "n_models": len(valid),
                           "slope_per_log10_params": fit.slope, "pearson_r": fit.rvalue})
    pd.DataFrame(trends).to_csv(output_root / "scaling_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(CACHE_ROOTS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start", type=int, default=0,
                        help="Start at this sorted model-cache task index")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--standardize-features", action="store_true",
                        help="z-score each embedding feature across words before the model RDM")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()
    global STANDARDIZE
    STANDARDIZE = args.standardize_features
    output_root = args.out_root or ROOT / "outputs" / args.dataset / "pearson_rsa" / "embedding"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        _summarize(args.dataset, output_root)
        return
    tasks = _tasks(args.dataset)
    if args.start:
        tasks = tasks[args.start:]
    if args.limit:
        tasks = tasks[:args.limit]
    if args.dataset == "szewczyk":
        family, model_id, revision, _, _ = tasks[0]
        from scripts.analysis.surprisal_rsa import _neural_cache_paths
        _, table_path = _neural_cache_paths(args.dataset, family, model_id, revision)
        neural_patterns = _szewczyk_neural_patterns(table_path)
    else:
        neural_patterns = _prepare_neural_patterns(args.dataset)
    print(f"{args.dataset}: {len(tasks)} models; {len(neural_patterns)} subject response patterns", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_score, task, args.dataset, neural_patterns, output_root) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    _summarize(args.dataset, output_root)


if __name__ == "__main__":
    main()
