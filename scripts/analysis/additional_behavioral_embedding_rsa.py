#!/usr/bin/env python3
"""Last-layer embedding RSA for cached word-reading behavioral datasets.

The script reads the existing contextual embedding caches, averages each
reading-time measure across subjects at the word level, and correlates model
and reading-time RDMs. No encoding model is fitted. For large datasets it uses
the same deterministic 250,000 item pairs for both RDMs and saves those pair
indices with the ordered word IDs for later RSA regression.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/additional-behavioral-rsa-mpl")

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import linregress, pearsonr

from neuroai.analysis.benchmark_common import filter_registry_models, EXCLUDED_FAMILY_DIRS
from neuroai.analysis.benchmark_common import lookup_model_meta
from neuroai.benchmarks.subject_average import average_items_across_subjects
from neuroai.datasets.geco import GecoDataset
from neuroai.datasets.geco import geco_word_list
from neuroai.datasets.natural_stories import NaturalStoriesDataset
from neuroai.datasets.onestop import OnestopDataset
from neuroai.datasets.onestop import onestop_word_list
from neuroai.datasets.ucl_reading import UCLReadingDataset
from neuroai.datasets.ucl_reading import sentence_word_lists
from neuroai.datasets.zuco import load_canonical_words
from neuroai.datasets.zuco_reading import ZuCoReadingDataset
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
from neuroai.sweep import safe_model_dir

CACHE_ROOT = Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache")
MEASURES = {
    "geco": ("FP", "GP"),
    "onestop": ("FP", "GP", "SP"),
    "naturalstories": ("SPR",),
    "ucl_reading": ("SPR",),
    "zuco_reading": ("NR", "SR", "TSR"),
}
MAX_PAIRS = 250_000
STREAM_FEATURE_THRESHOLD = 4 * 1024**3


def _last_layer_from_npz(path: Path, scratch_dir: Path) -> np.ndarray:
    """Load only the last hidden layer, streaming huge compressed caches.

    `np.load()` decompresses an NPZ member into one dense array. That can use
    more memory than the final-layer representation needs for very large
    models. For members larger than 4 GiB, stream the NPY payload to local
    scratch, memory-map it, copy the last layer, and remove the temporary file.
    """
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("features.npy")
        if info.file_size > STREAM_FEATURE_THRESHOLD:
            scratch_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix="embedding_rsa_", suffix=".npy",
                                            dir=scratch_dir)
            os.close(fd)
            try:
                with archive.open(info) as source, open(tmp_path, "wb") as target:
                    shutil.copyfileobj(source, target, length=16 * 1024**2)
                features = np.load(tmp_path, mmap_mode="r")
                if features.ndim != 3:
                    raise ValueError(f"Invalid embedding cache shape: {path}")
                last = np.array(features[:, -1, :], dtype=np.float32, copy=True)
                del features
                return last
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    with np.load(path) as archive:
        features = archive["features"]
        if features.ndim == 2 and path.name.endswith("__cbrrnn.npz"):
            return np.array(features, dtype=np.float32, copy=True)
        if features.ndim != 3:
            raise ValueError(f"Invalid embedding cache shape: {path}")
        return np.array(features[:, -1, :], dtype=np.float32, copy=True)


def _feature_roots(dataset: str, measure: str) -> tuple[Path, ...]:
    names = {
        "geco": "geco_embed_features",
        "onestop": "onestop_embed_features",
        "naturalstories": "naturalstories_features",
        "ucl_reading": "ucl_reading_features",
        "zuco_reading": f"zuco_reading_features/{measure.lower()}",
    }
    roots = [CACHE_ROOT / names[dataset]]
    if dataset == "naturalstories":
        # Natural Stories' completed contextual embeddings live in the repo
        # cache on fswork, while older logit-lens artifacts live under $SCRATCH.
        # Search both; keep the repo cache first when a model/section exists in
        # both locations, and use the scratch cache to fill any missing sections.
        roots.insert(0, ROOT / ".cache" / names[dataset])
    return tuple(roots)


def _feature_paths(dataset: str, measure: str, model_id: str, revision: str) -> dict[int, Path] | None:
    safe = safe_model_dir(model_id)
    rev = revision.replace("/", "_").replace("-", "_")
    paths = {}
    for root in _feature_roots(dataset, measure):
        pattern = (f"{safe}__en__sec*__ctx48__cbrrnn.npz"
                   if model_id.startswith("cclark/cbr-rnn") and revision == "main"
                   else f"{safe}__{rev}__en__sec*__all.npz")
        for path in root.glob(pattern):
            match = re.search(r"__sec(\d+)__", path.name)
            if match:
                paths.setdefault(int(match.group(1)), path)
    return paths or None


def _response_items(dataset: str, measure: str) -> pd.DataFrame:
    cache = ROOT / "outputs" / dataset / "pearson_rsa" / "embedding" / "response_table" / f"{measure.lower()}.parquet"
    if cache.is_file():
        return pd.read_parquet(cache)
    root = {
        "geco": ROOT / "data/geco",
        "onestop": ROOT / "data/onestop",
        "naturalstories": ROOT / "data/naturalstories",
        "ucl_reading": ROOT / "data/ucl_reading",
        "zuco_reading": ROOT / "data/zuco_reading/osfstorage",
    }[dataset]
    if dataset == "geco":
        raw = GecoDataset(root, measure, aggregate=False).load_words()
    elif dataset == "onestop":
        subject_path = ROOT / "configs/subjects/onestop_subjects_120.txt"
        subjects = subject_path.read_text().split()
        raw = OnestopDataset(root, measure, subjects=subjects, aggregate=False).load_words()
    elif dataset == "naturalstories":
        raw = NaturalStoriesDataset(root, aggregate=False).load_words()
    elif dataset == "ucl_reading":
        raw = UCLReadingDataset(root, aggregate=False).load_words()
    else:
        raw = ZuCoReadingDataset(root, measure, aggregate=False).load_words()
    raw = raw.dropna(subset=["rt"])
    raw["rt"] = pd.to_numeric(raw.rt, errors="coerce")
    raw = raw.dropna(subset=["rt"])
    if "is_edge" in raw:
        raw = raw.loc[~raw.is_edge.astype(bool)]
    items = average_items_across_subjects(raw, item_key_cols=("story_id", "zone"))
    items = items.sort_values(["story_id", "zone"]).reset_index(drop=True)
    if items.duplicated(["story_id", "zone"]).any() or not np.isfinite(items.rt).all():
        raise ValueError(f"Invalid word response table for {dataset}/{measure}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(cache, index=False)
    return items


def _feature_index_map(dataset: str, items_by_measure: dict[str, pd.DataFrame]):
    """Map each dataset word ID to its row in the cached feature sequence.

    Some embedding caches omit sentence/document edge words or flatten all
    ZuCo sentences into one sequence. In those cases `zone - 1` is not the
    cache row, so mirror the benchmark's ordered word list explicitly.
    """
    items = next(iter(items_by_measure.values()))
    if dataset == "geco":
        words = geco_word_list(ROOT / "data/geco", exclude_doc_edges=True)
    elif dataset == "onestop":
        words = onestop_word_list(ROOT / "data/onestop", exclude_doc_edges=True)
    elif dataset == "zuco_reading":
        return {
            measure: {(int(row.sent_id), int(row.word_id) + 1): int(row.global_word_idx)
                      for row in load_canonical_words(
                          ROOT / "data/zuco_reading/osfstorage", measure).itertuples()}
            for measure in items_by_measure
        }
    elif dataset == "ucl_reading":
        sentences = sentence_word_lists(ROOT / "data/ucl_reading")
        mapping = {(int(story), zone): zone - 1 for story, tokens in sentences.items()
                   for zone in range(1, len(tokens) + 1)}
        return {measure: mapping for measure in items_by_measure}
    else:
        # Natural Stories embeddings are extracted from the already
        # aggregated, sorted item table used as the ridge target.
        mapping = {(int(story), int(row.zone)): i
                for story, group in items.sort_values(["story_id", "zone"]).groupby("story_id", sort=True)
                for i, row in enumerate(group.itertuples(index=False))}
        return {measure: mapping for measure in items_by_measure}
    words = words.sort_values(["story_id", "zone"])
    mapping = {(int(story), int(row.zone)): i
            for story, group in words.groupby("story_id", sort=True)
            for i, row in enumerate(group.itertuples(index=False))}
    return {measure: mapping for measure in items_by_measure}


def _encoding_score(dataset: str, measure: str, model_id: str) -> float:
    summary_key = {
        ("geco", "FP"): "geco", ("geco", "GP"): "geco_gp",
        ("onestop", "FP"): "onestop", ("onestop", "GP"): "onestop_gp",
        ("onestop", "SP"): "onestop_sp",
        ("naturalstories", "SPR"): "naturalstories_spr",
        ("ucl_reading", "SPR"): "ucl_reading",
        ("zuco_reading", "NR"): "zuco_reading",
        ("zuco_reading", "SR"): "zuco_reading_sr",
        ("zuco_reading", "TSR"): "zuco_reading_tsr",
    }[(dataset, measure)]
    path = ROOT / "outputs/benchmark_summary" / f"{summary_key}__embed/all_models_scores.csv"
    if not path.exists():
        return float("nan")
    frame = pd.read_csv(path)
    rows = frame.loc[frame.model_id.astype(str) == model_id]
    return float(rows.score_last.iloc[0]) if not rows.empty else float("nan")


def _tasks(dataset: str, measures: tuple[str, ...]):
    indexes = {}
    for measure in measures:
        index = {}
        # Require the exact hidden-state cache tag. A broader `*all.npz`
        # pattern also matches `logitlens_all.npz`, which stores scalar
        # logit-lens values and is not an embedding representation.
        for root in _feature_roots(dataset, measure):
            for path in root.glob("*__en__sec*__all.npz"):
                match = re.search(r"__sec(\d+)__", path.name)
                if match:
                    prefix = path.name.split("__en__sec", 1)[0]
                    index.setdefault(prefix, {}).setdefault(int(match.group(1)), path)
            for path in root.glob("cclark_cbr-rnn*__en__sec*__ctx48__cbrrnn.npz"):
                match = re.search(r"__sec(\d+)__", path.name)
                if match:
                    prefix = path.name.split("__en__sec", 1)[0] + "__main"
                    index.setdefault(prefix, {}).setdefault(int(match.group(1)), path)
        indexes[measure] = index
    tasks = []
    for family in LPP_FAMILY_NAMES:
        if family in EXCLUDED_FAMILY_DIRS:
            continue
        for task in flatten_family_tasks(family):
            available = {}
            for measure in measures:
                prefix = (safe_model_dir(task.model_id) + "__" + task.revision
                          .replace("/", "_").replace("-", "_"))
                paths = indexes[measure].get(prefix)
                if paths is not None:
                    available[measure] = paths
            if available:
                tasks.append((family, task.model_id, task.revision, available))
    # Run narrower representations first. Each per-word cache task uses only
    # its final layer, but the largest models still have wider hidden states.
    tasks.sort(key=lambda task: (lookup_model_meta(task[1], task[0]).n_params or float("inf"),
                                task[1], task[2]))
    return tasks


def _sample_pairs(n: int, dataset: str, measure: str) -> np.ndarray | None:
    total = n * (n - 1) // 2
    if total <= MAX_PAIRS:
        return None
    seed = int.from_bytes(hashlib.sha256(f"{dataset}:{measure}:rsa-pairs-v1".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    pairs = np.empty((0, 2), dtype=np.int32)
    while len(pairs) < MAX_PAIRS:
        batch = max((MAX_PAIRS - len(pairs)) * 3, 10_000)
        left, right = rng.integers(0, n, size=batch), rng.integers(0, n, size=batch)
        candidate = np.column_stack((np.minimum(left, right), np.maximum(left, right)))
        candidate = candidate[candidate[:, 0] != candidate[:, 1]].astype(np.int32)
        pairs = np.unique(np.concatenate((pairs, candidate), axis=0), axis=0)
    if len(pairs) > MAX_PAIRS:
        pairs = pairs[rng.choice(len(pairs), MAX_PAIRS, replace=False)]
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def _sampled_distances(states: np.ndarray, rt: np.ndarray, pairs: np.ndarray):
    selected = np.asarray(states, dtype=np.float32)
    selected -= selected.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(selected, axis=1)
    if (norms <= np.finfo(np.float32).tiny).any():
        raise ValueError("A model embedding has zero centered norm")
    selected /= norms[:, None]
    model_dist = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), 4096):
        block = pairs[start:start + 4096]
        sim = np.einsum("ij,ij->i", selected[block[:, 0]], selected[block[:, 1]])
        model_dist[start:start + len(block)] = 1 - np.clip(sim, -1, 1)
    rt_dist = np.abs(rt[pairs[:, 0]] - rt[pairs[:, 1]]).astype(np.float32)
    return model_dist, rt_dist


STANDARDIZE = False


def _score_task(task, dataset, response_items, feature_index, output_root):
    family, model_id, revision, paths_by_measure = task
    todo = [measure for measure in response_items
            if measure in paths_by_measure and not (
                output_root / measure.lower() / family / safe_model_dir(model_id)
                / f"rsa_{revision.replace('/', '_')}.csv").exists()]
    if not todo:
        return f"existing {model_id}@{revision}"
    results = []
    for measure in todo:
        items = response_items[measure]
        cache_paths = paths_by_measure.get(measure)
        item_index = feature_index[measure]
        if not cache_paths:
            continue
        # Cross-model RSA comparisons require the same stimulus set. A cache
        # may contain only a subset of sections (for example, a truncated
        # long-context run); scoring whatever happens to be present would make
        # the model's n_items and pair sample differ silently.
        if dataset != "zuco_reading":
            required_sections = set(response_items[measure].story_id.astype(int))
            missing_sections = required_sections.difference(cache_paths)
            if missing_sections:
                return (f"skipped {model_id}@{revision}: missing {len(missing_sections)} "
                        f"of {len(required_sections)} response sections for {measure}")
        row_indices = np.asarray([item_index.get((int(story), int(zone)), -1)
                                  for story, zone in zip(items.story_id, items.zone)], dtype=np.int64)
        if dataset == "zuco_reading":
            path = cache_paths.get(0)
            if path is None:
                continue
            flat_states = _last_layer_from_npz(path, output_root / "temporary")
            valid = (row_indices >= 0) & (row_indices < len(flat_states))
        else:
            present = items.story_id.isin(cache_paths)
            if not present.any():
                continue
            items = items.loc[present].reset_index(drop=True)
            row_indices = row_indices[present.to_numpy()]
            stories = np.unique(items.story_id.to_numpy(dtype=np.int64))
            states_by_story = {}
            for story in stories:
                path = cache_paths[int(story)]
                states_by_story[int(story)] = _last_layer_from_npz(path, output_root / "temporary")
            valid = np.asarray([0 <= index < len(states_by_story[int(story)])
                                for story, index in zip(items.story_id, row_indices)])
        if not valid.all():
            print(f"{dataset}/{measure} {model_id}: excluding {(~valid).sum()} response words "
                  "outside cached section lengths", flush=True)
            items = items.loc[valid].reset_index(drop=True)
            row_indices = row_indices[valid]
        if len(items) < 3:
            continue
        if dataset == "zuco_reading":
            matrix = flat_states[row_indices]
        else:
            matrix = np.stack([states_by_story[int(story)][index]
                               for story, index in zip(items.story_id, row_indices)])
        if STANDARDIZE:  # per-feature z-score over the scored words (Timkey & van Schijndel 2021)
            matrix = np.asarray(matrix, dtype=np.float32)
            matrix = (matrix - matrix.mean(0)) / (matrix.std(0) + 1e-8)
        rt = items.rt.to_numpy(dtype=np.float32)
        pairs = _sample_pairs(len(items), dataset, measure)
        if pairs is None:
            model_dist = pdist(matrix, metric="correlation").astype(np.float32)
            rt_dist = pdist(rt[:, None], metric="euclidean").astype(np.float32)
        else:
            model_dist, rt_dist = _sampled_distances(matrix, rt, pairs)
        score = float(pearsonr(model_dist, rt_dist).statistic)
        cache = (output_root / "rdm_cache" / measure.lower() / family
                 / safe_model_dir(model_id) / f"{revision.replace('/', '_')}.npz")
        cache.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"stimulus_ids": items[["story_id", "zone"]].to_numpy(dtype=np.int64),
                    "model_distances": model_dist, "response_distances": rt_dist,
                    "rdm_type": "pearson_embedding_distance_and_absolute_rt_difference",
                    "dataset": dataset, "measure": measure, "model_id": model_id,
                    "revision": revision}
        if pairs is not None:
            metadata["pair_indices"] = pairs
        temporary = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **metadata)
        os.replace(temporary, cache)
        meta = lookup_model_meta(model_id, family)
        results.append({"family": family, "model_id": model_id, "revision": revision,
                        "measure": measure, "n_items": len(items), "n_pairs": len(model_dist),
                        "rsa_pearson": score, "encoding_pearson": _encoding_score(dataset, measure, model_id),
                        "group": meta.group, "n_params": meta.n_params})
    for row in results:
        out = (output_root / row["measure"].lower() / family / safe_model_dir(model_id)
               / f"rsa_{revision.replace('/', '_')}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out, index=False)
    return f"scored {model_id}@{revision}" if results else f"no matched word/cache rows for {model_id}@{revision}"


def _summarize(dataset: str, output_root: Path, measures: tuple[str, ...]):
    for measure in measures:
        rows = []
        for path in (output_root / measure.lower()).glob("*/*/rsa_*.csv"):
            rows.append(pd.read_csv(path).iloc[0].to_dict())
        if not rows:
            continue
        summary = filter_registry_models(pd.DataFrame(rows)).sort_values(["family", "model_id", "revision"])
        target = output_root / measure.lower()
        summary.to_csv(target / "model_summary.csv", index=False)
        valid = summary.loc[np.isfinite(summary.encoding_pearson)]
        if len(valid) > 2:
            print(f"{dataset}/{measure}: {len(summary)} models; encoding vs RSA r="
                  f"{pearsonr(valid.encoding_pearson, valid.rsa_pearson).statistic:.4f}", flush=True)
        meta = summary.dropna(subset=["n_params"]).copy() if "n_params" in summary else summary
        trends = []
        if len(meta) and meta.n_params.notna().any():
            meta = meta.loc[meta.n_params > 0].copy()
            meta["log10_params"] = np.log10(meta.n_params)
            for scope, name, part in [("all", "all models", meta), *[("family", str(name), group)
                       for name, group in meta.groupby("family")]]:
                if len(part) < 3 or part.log10_params.nunique() < 3:
                    continue
                for metric in ("rsa_pearson", "encoding_pearson"):
                    fit = linregress(part.log10_params, part[metric].astype(float))
                    trends.append({"scope": scope, "name": name, "metric": metric,
                                   "n_models": len(part), "slope_per_log10_param": fit.slope,
                                   "pearson_r": fit.rvalue})
        pd.DataFrame(trends).to_csv(target / "scaling_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(MEASURES))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summarize-only", action="store_true",
                         help="Rebuild summaries from saved per-model result files")
    parser.add_argument("--standardize-features", action="store_true")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()
    global STANDARDIZE
    STANDARDIZE = args.standardize_features
    measures = MEASURES[args.dataset]
    output_root = args.out_root or ROOT / "outputs" / args.dataset / "pearson_rsa" / "embedding"
    if args.summarize_only:
        _summarize(args.dataset, output_root, measures)
        return
    response_items = {measure: _response_items(args.dataset, measure) for measure in measures}
    feature_index = _feature_index_map(args.dataset, response_items)
    tasks = _tasks(args.dataset, measures)
    if args.limit:
        tasks = tasks[:args.limit]
    output_root.mkdir(parents=True, exist_ok=True)
    for measure, items in response_items.items():
        pairs = _sample_pairs(len(items), args.dataset, measure)
        rt = items.rt.to_numpy(dtype=np.float32)
        distances = (pdist(rt[:, None], metric="euclidean").astype(np.float32) if pairs is None
                     else np.abs(rt[pairs[:, 0]] - rt[pairs[:, 1]]).astype(np.float32))
        shared = output_root / "rdm_cache" / measure.lower() / "shared" / "reading_time_rdm.npz"
        shared.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stimulus_ids": items[["story_id", "zone"]].to_numpy(dtype=np.int64),
                   "distances": distances, "rdm_type": "absolute_reading_time_difference",
                   "dataset": args.dataset, "measure": measure}
        if pairs is not None:
            payload["pair_indices"] = pairs
        temporary = shared.with_name(f"{shared.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, shared)
    print(f"{args.dataset}: {len(tasks)} cached model tasks; " + ", ".join(
        f"{m}={len(x)} words" for m, x in response_items.items()), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_score_task, task, args.dataset, response_items, feature_index, output_root)
                   for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    _summarize(args.dataset, output_root, measures)


if __name__ == "__main__":
    main()
