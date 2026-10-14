#!/usr/bin/env python3
"""Fit-free last-layer surprisal RSA from existing pooled surprisal caches.

Behavioral datasets compare absolute surprisal differences with absolute
reading-time differences. Tuckute and Fedorenko compare surprisal distances
with subject-specific neural RDMs. Saved RDM vectors retain stimulus IDs for
the later variance-partitioning RSA.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
PODCAST_V2_CACHE = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug")) / ".cache" / "podcast_features"
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = "/tmp/surprisal-rsa-matplotlib"

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import linregress, pearsonr

from neuroai.analysis.benchmark_common import filter_registry_models
from neuroai.analysis.benchmark_common import lookup_model_meta, EXCLUDED_FAMILY_DIRS
from neuroai.datasets.fedorenko2016 import Fedorenko2016Dataset
from neuroai.datasets.tuckute import TuckuteDataset
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
from neuroai.sweep import safe_model_dir

BEHAVIORAL = {"provo", "meco", "geco", "onestop", "naturalstories", "ucl_reading", "zuco_reading"}
NEURAL = {"pereira", "tuckute", "fedorenko2016", "broderick", "megmasc",
          "nieuwland", "szewczyk", "podcast"}
MEASURES = {
    "provo": ("FP", "GP", "SP"), "meco": ("FP", "GP", "SP"),
    "geco": ("FP", "GP"), "onestop": ("FP", "GP", "SP"),
    "naturalstories": ("SPR",), "ucl_reading": ("SPR",),
    "zuco_reading": ("NR", "SR", "TSR"),
}
MAX_PAIRS = 250_000


def _tasks(dataset: str, measures: tuple[str, ...]):
    tasks = []
    for family in LPP_FAMILY_NAMES:
        if family in EXCLUDED_FAMILY_DIRS:
            continue
        for task in flatten_family_tasks(family):
            model_dir = ROOT / "outputs_jeanzay" / dataset / "surprisal_sweep"
            found = False
            for measure in measures:
                if dataset in BEHAVIORAL:
                    cache, _ = _behavior_cache_paths(dataset, family, task.model_id, task.revision, measure)
                else:
                    cache, _ = _neural_cache_paths(dataset, family, task.model_id, task.revision)
                found |= cache.exists()
            if found:
                if dataset == "podcast":
                    signal_path, _ = _neural_cache_paths(dataset, family, task.model_id, task.revision)
                    # The 50–250 ms average needs the complete word-locked
                    # response; checkpoint caches that contain only one lag
                    # cannot implement the requested window.
                    try:
                        found = np.load(signal_path, mmap_mode="r").shape[1] >= 32
                    except (OSError, ValueError, IndexError):
                        found = False
            if found:
                tasks.append((family, task.model_id, task.revision))
    return list(dict.fromkeys(tasks))


def _behavior_cache_paths(dataset, family, model_id, revision, measure):
    safe = safe_model_dir(model_id)
    measure_dir = measure.lower() if dataset in {"provo", "meco", "geco", "onestop", "zuco_reading"} else ""
    root = ROOT / "outputs_jeanzay" / dataset / "surprisal_sweep" / measure_dir / family / safe
    prefix = {"naturalstories": "ns", "ucl_reading": "ucl_reading"}.get(dataset, dataset)
    tag = f"{prefix}_{measure}_lme" if dataset in {"provo", "meco", "geco", "onestop", "zuco_reading"} else f"{prefix}_lme"
    if dataset == "meco":
        tag = f"meco_en_{measure}_lme"
    rev = revision.replace("/", "_")
    if dataset in {"ucl_reading", "naturalstories"}:
        # Pooled-LME rewrite dropped the per-revision filename suffix for
        # most families in these two (unlike provo/meco/geco/onestop/
        # zuco_reading, which kept it) -- but the "direct load"
        # special-cased architectures (bloom/rwkv4/fox/molm) still write
        # the suffixed filename here.
        return _first_existing_pair(
            (root / f"{tag}_surprisal.npy", root / f"{tag}_word_base.parquet"),
            (root / f"{tag}_surprisal_{rev}.npy", root / f"{tag}_word_base_{rev}.parquet"),
        )
    return root / f"{tag}_surprisal_{rev}.npy", root / f"{tag}_word_base_{rev}.parquet"



def _first_existing_pair(*candidates):
    """Return the first (npy, table) candidate pair where both files exist,
    else the last candidate (so downstream .is_file() checks still fail
    gracefully). Needed because the pooled-LME rewrite left some model
    families (bloom/rwkv4/fox/molm -- the "direct load" special-cased
    architectures in neuroai/features/llm.py) writing a revision-suffixed
    filename while standard families write unsuffixed -- both conventions
    coexist on disk for the same dataset."""
    for pair in candidates[:-1]:
        if all(p.is_file() for p in pair):
            return pair
    return candidates[-1]


def _neural_cache_paths(dataset, family, model_id, revision):
    safe = safe_model_dir(model_id)
    root = ROOT / "outputs_jeanzay" / dataset / "surprisal_sweep" / family / safe
    rev = revision.replace("/", "_")
    if dataset == "nieuwland":
        return _first_existing_pair(
            (root / "nieuwland_lme_pooled_surprisal.npy", root / "nieuwland_lme_pooled_table.parquet"),
            (root / f"nieuwland_lme_pooled_surprisal_{rev}.npy", root / f"nieuwland_lme_pooled_table_{rev}.parquet"),
        )
    if dataset == "szewczyk":
        return root / f"szewczyk_lme_pooled_surprisal_{rev}.npy", root / f"szewczyk_lme_pooled_table_{rev}.parquet"
    if dataset == "podcast":
        return _first_existing_pair(
            (root / "podcast_lme_pooled_y.npy", root / "podcast_lme_pooled_table.parquet"),
            (root / f"podcast_lme_pooled_y_{rev}.npy", root / f"podcast_lme_pooled_table_{rev}.parquet"),
        )
    if dataset == "megmasc":
        return root / "megmasc_lme_pooled_surprisal.npy", root / "megmasc_lme_pooled_table.parquet"
    if dataset == "broderick":
        return root / "broderick_lme_surprisal.npy", root / "broderick_lme_table.parquet"
    if dataset in {"fedorenko2016", "tuckute"}:
        # Pooled-LME rewrite dropped the per-revision filename suffix for
        # most families in these two datasets (unlike pereira/lpp, which
        # kept it) -- but the "direct load" special-cased architectures
        # (bloom/rwkv4/fox/molm) still write the suffixed filename here.
        return _first_existing_pair(
            (root / f"{dataset}_lme_pooled_surprisal.npy", root / f"{dataset}_lme_pooled_table.parquet"),
            (root / f"{dataset}_lme_pooled_surprisal_{rev}.npy", root / f"{dataset}_lme_pooled_table_{rev}.parquet"),
        )
    return root / f"{dataset}_lme_pooled_surprisal_{rev}.npy", root / f"{dataset}_lme_pooled_table_{rev}.parquet"


def _naturalstories_trial_fallback(base, model_id, revision):
    """Recover stale LME exports from complete extraction caches, with ID checks."""
    from neuroai.datasets.natural_stories import NaturalStoriesDataset

    from neuroai.features.logit_lens import _trial_surprisal_cache_path

    filename = _trial_surprisal_cache_path(".", model_id, revision, "natural_stories", False, 0).name
    roots = [ROOT / ".cache" / "naturalstories_features",
             Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
             / ".cache" / "naturalstories_features"]
    if os.environ.get("NS_FEATURE_CACHE_DIR"):
        roots.insert(0, Path(os.environ["NS_FEATURE_CACHE_DIR"]))
    for root in dict.fromkeys(roots):
        path = root / filename
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            values = archive["features"]
        if values.ndim != 2 or not np.isfinite(values[:, -1]).all():
            continue
        # Trial caches contain no IDs. Reconstruct the exact extractor ordering,
        # then require equality with the exported word table; shape alone is unsafe.
        words = NaturalStoriesDataset(ROOT / "data/naturalstories", min_correct=0,
                                      aggregate=True).load_words().reset_index(drop=True)
        if len(values) != len(words) or not np.array_equal(
                words[["story_id", "zone"]].to_numpy(), base[["story_id", "zone"]].to_numpy()):
            raise ValueError(f"Natural Stories trial cache word ordering mismatch: {path}")
        print(f"Using complete trial cache instead of nonfinite LME export: {path}", flush=True)
        return values
    return None


def _load_behavior_surprisal(dataset, family, model_id, revision, measure):
    npy, table = _behavior_cache_paths(dataset, family, model_id, revision, measure)
    if not npy.is_file() or not table.is_file():
        return None
    base = pd.read_parquet(table)
    values = np.load(npy, mmap_mode="r")
    if values.ndim != 2 or values.shape[0] != len(base):
        raise ValueError(f"Surprisal/table row mismatch: {npy}")
    if not {"story_id", "zone"}.issubset(base.columns):
        raise ValueError(f"Missing behavioral word IDs in {table}")
    if dataset == "naturalstories" and not np.isfinite(values[:, -1]).all():
        replacement = _naturalstories_trial_fallback(base, model_id, revision)
        if replacement is not None:
            values = replacement
    base = base[["story_id", "zone"]].copy()
    base["surprisal"] = np.asarray(values[:, -1], dtype=np.float32)
    if base.duplicated(["story_id", "zone"]).any():
        raise ValueError(f"Duplicate behavioral word IDs in {table}")
    return base


def _load_neural_surprisal(dataset, family, model_id, revision):
    npy, table = _neural_cache_paths(dataset, family, model_id, revision)
    if dataset == "broderick":
        # Pooled-LME rewrite: one shared table+surprisal file per model,
        # already covering every subject (crossed subject x region design,
        # "subject" is a native column) -- mirrors the megmasc branch below,
        # not the old per-subject-file scheme this used to expect.
        if not npy.is_file() or not table.is_file():
            return None
        base = pd.read_parquet(table, columns=["subject", "chunk", "word_idx", "region"])
        base["stimulus_id"] = base.chunk.astype(str) + "::" + base.word_idx.astype(str)
        values = np.load(npy, mmap_mode="r")
        if values.ndim != 2 or values.shape[0] != len(base):
            raise ValueError(f"Surprisal/table row mismatch: {npy}")
        base["surprisal"] = np.asarray(values[:, -1], dtype=np.float32)
        spread = base.groupby(["subject", "stimulus_id"]).surprisal.agg(lambda x: float(x.max() - x.min()))
        if (spread > 1e-5).any():
            raise ValueError(f"Repeated Broderick surprisal differs across regions: {npy}")
        return (base.drop_duplicates(["subject", "stimulus_id"])
                [["subject", "stimulus_id", "surprisal"]]
                .sort_values(["subject", "stimulus_id"]).reset_index(drop=True))
    if not npy.is_file() or not table.is_file():
        return None
    base = pd.read_parquet(table, columns=(
        ["subject", "chunk", "word_idx", "surprisal"] if dataset == "podcast" else
        ["subject", "chunk", "word_idx", "region"] if dataset == "megmasc" else
        None
    ))
    if dataset == "podcast":
        # The pooled table predates the 2026-10-12 surprisal fixes (Podcast moved to the region sweep, which writes
        # no per-word table). Keep it only as the stimulus index; take the surprisal itself from the v2 logit-lens
        # cache (last layer, summed subwords), whose rows are the transcript word positions (= word_idx).
        # main revision, else the one checkpoint cached (OLMo-2 caches are named by stage/step). CBR-RNN has no logit-lens
        # cache: its surprisal is word-level by construction (untouched by the summed-subword fix), so the table value is kept.
        stem = model_id.replace('/', '_')
        v2 = (sorted(PODCAST_V2_CACHE.glob(f"{stem}__main__*__logitlens_all_v2_sum.npz"))
              or sorted(PODCAST_V2_CACHE.glob(f"{stem}__*__logitlens_all_v2_sum.npz")))
        if v2:
            surp = np.load(v2[0])["features"][:, -1, 0]
            base["surprisal"] = surp[base.word_idx.to_numpy(dtype=int)]
        elif family != "cbr_rnn":
            return None
        base["stimulus_id"] = base.chunk.astype(str) + "::" + base.word_idx.astype(str)
        values = base[["subject", "stimulus_id", "surprisal"]].copy()
        values["subject"] = values.subject.astype(str)
        if values.duplicated(["subject", "stimulus_id"]).any():
            spread = values.groupby(["subject", "stimulus_id"]).surprisal.agg(lambda x: float(x.max() - x.min()))
            if (spread > 1e-5).any():
                raise ValueError(f"Repeated Podcast surprisal differs across electrodes in {table}")
            values = values.drop_duplicates(["subject", "stimulus_id"])
        return values.sort_values(["subject", "stimulus_id"]).reset_index(drop=True)
    if dataset == "megmasc":
        values = base.copy()
        values["stimulus_id"] = values.chunk.astype(str) + "::" + values.word_idx.astype(str)
        surprisal = np.load(npy, mmap_mode="r")
        if surprisal.ndim != 2 or surprisal.shape[0] != len(values):
            raise ValueError(f"Surprisal/table row mismatch: {npy}")
        values["surprisal"] = np.asarray(surprisal[:, -1], dtype=np.float32)
        spread = values.groupby(["subject", "stimulus_id"]).surprisal.agg(lambda x: float(x.max() - x.min()))
        if (spread > 1e-5).any():
            raise ValueError(f"Repeated MEG-MASC surprisal differs across regions: {npy}")
        return (values.drop_duplicates(["subject", "stimulus_id"])
                [["subject", "stimulus_id", "surprisal"]]
                .sort_values(["subject", "stimulus_id"]).reset_index(drop=True))
    values = np.load(npy, mmap_mode="r")
    if values.ndim != 2 or values.shape[0] != len(base):
        raise ValueError(f"Surprisal/table row mismatch: {npy}")
    ids = {"pereira": "sentence_id", "tuckute": "sentence_id", "fedorenko2016": "stimulus_id",
           "nieuwland": "item_id", "szewczyk": "chunk"}[dataset]
    keep = ["subject", ids] if dataset == "szewczyk" else [ids]
    base = base[keep].copy()
    if dataset not in {"szewczyk", "pereira"}:
        base[ids] = pd.to_numeric(base[ids], errors="raise").astype(int)
    base["surprisal"] = np.asarray(values[:, -1], dtype=np.float32)
    if dataset == "szewczyk":
        base["subject"] = base.subject.astype(str)
        if base.duplicated(["subject", ids]).any():
            spread = base.groupby(["subject", ids]).surprisal.agg(lambda x: float(x.max() - x.min()))
            if (spread > 1e-5).any():
                raise ValueError(f"Repeated surprisal rows disagree within subject/item in {npy}")
            base = base.drop_duplicates(["subject", ids])
        return base.sort_values(["subject", ids]).reset_index(drop=True)
    # The pooled LME cache repeats the same item surprisal across subject/ROI
    # or subject/electrode rows. Check it, then retain one value per stimulus.
    spread = base.groupby(ids).surprisal.agg(lambda x: float(x.max() - x.min()))
    if (spread > 1e-5).any():
        raise ValueError(f"Repeated surprisal rows disagree in {npy}")
    return base.drop_duplicates(ids).sort_values(ids).reset_index(drop=True)


def _save_rdm(path: Path, stimulus_ids: np.ndarray, distances: np.ndarray, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    np.savez_compressed(tmp, stimulus_ids=stimulus_ids,
                        distances=np.asarray(distances, dtype=np.float32), **metadata)
    # np.savez appends .npz if the supplied name lacks that suffix.
    actual = tmp if tmp.exists() else Path(f"{tmp}.npz")
    os.replace(actual, path)


def _prepare_behavior(dataset: str):
    if dataset not in {"provo", "meco"}:
        measures = MEASURES[dataset]
        family, model_id, revision = _tasks(dataset, measures)[0]
        result = {}
        for measure in measures:
            _, word_path = _behavior_cache_paths(dataset, family, model_id, revision, measure)
            trials_path = word_path.with_name(word_path.name.replace("word_base", "trials_base"))
            trials = pd.read_parquet(trials_path)
            trials = trials.loc[np.isfinite(pd.to_numeric(trials.rt, errors="coerce"))].copy()
            if "is_edge" in trials:
                trials = trials.loc[~trials.is_edge.fillna(False).astype(bool)]
            grouped = trials.groupby(["story_id", "zone"], as_index=False).agg(
                rt=("rt", "mean"), n_subjects=("participant_id", "nunique") if "participant_id" in trials else ("rt", "size"))
            if dataset == "onestop":
                grouped = grouped.loc[grouped.n_subjects >= 5]
            result[measure] = grouped[["story_id", "zone", "rt"]].sort_values(["story_id", "zone"]).reset_index(drop=True)
        return result
    from scripts.analysis.behavioral_rsa import _prepare
    _, responses = _prepare(dataset)
    return responses


def _score_behavior_model(task, dataset, responses, output_root, measures):
    family, model_id, revision = task
    summary = []
    existing_count = 0
    for measure in measures:
        out = output_root / measure.lower() / family / safe_model_dir(model_id) / f"rsa_{revision.replace('/', '_')}.csv"
        if out.is_file():
            previous = pd.read_csv(out)
            if len(previous) == 1 and np.isfinite(previous.rsa_pearson.iloc[0]):
                existing_count += 1
                continue
        values = _load_behavior_surprisal(dataset, family, model_id, revision, measure)
        if values is None:
            continue
        items = responses[measure]
        merged = items[["story_id", "zone", "rt"]].merge(
            values, on=["story_id", "zone"], how="inner", validate="one_to_one",
        ).sort_values(["story_id", "zone"]).reset_index(drop=True)
        if dataset in {"provo", "meco"} and len(merged) != len(items):
            raise ValueError(f"{dataset}/{measure}: only {len(merged)}/{len(items)} aligned words for {model_id}")
        if not np.isfinite(merged[["surprisal", "rt"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{dataset}/{measure}: nonfinite aligned surprisal/response for {model_id}; repair source cache before RSA")
        keys = merged[["story_id", "zone"]].to_numpy(dtype=np.int32)
        model_dist, response_dist, pair_indices = _pair_distances(
            merged.surprisal.to_numpy(dtype=float), merged.rt.to_numpy(dtype=float), dataset, measure)
        score = float(pearsonr(model_dist, response_dist).statistic)
        if not np.isfinite(score):
            raise ValueError(f"{dataset}/{measure}: undefined RSA for {model_id}; check constant RDMs")
        cache = output_root / "rdm_cache" / measure.lower() / family / safe_model_dir(model_id) / f"{revision.replace('/', '_')}.npz"
        metadata = {"pair_indices": pair_indices} if pair_indices is not None else {}
        _save_rdm(cache, keys, model_dist, rdm_type="absolute_surprisal_difference",
                  **metadata,
                  measure=measure, model_id=model_id, revision=revision)
        shared = output_root / "rdm_cache" / measure.lower() / "shared" / "behavior_rdm.npz"
        if not shared.exists():
            _save_rdm(shared, keys, response_dist, rdm_type="absolute_reading_time_difference",
                      **metadata,
                      dataset=dataset, measure=measure)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"family": family, "model_id": model_id, "revision": revision,
                       "measure": measure, "n_items": len(merged), "n_pairs": len(model_dist),
                       "rsa_pearson": score}]).to_csv(out, index=False)
        summary.append(score)
    if summary:
        return f"scored {model_id}@{revision}"
    if existing_count:
        return f"existing {model_id}@{revision}"
    return f"missing cache {model_id}@{revision}"


def _pair_distances(x, y, dataset, measure):
    n = len(x)
    total = n * (n - 1) // 2
    if total <= MAX_PAIRS:
        return pdist(x[:, None]), pdist(y[:, None]), None
    seed = int.from_bytes(__import__("hashlib").sha256(f"{dataset}:{measure}:rsa-pairs-v1{__import__('os').environ.get('RDM_PAIR_SEED_TAG', '')}".encode()).digest()[:8], "little")  # tag only for the pair-count sensitivity runs
    rng = np.random.default_rng(seed)
    pairs = np.empty((0, 2), dtype=np.int32)
    while len(pairs) < MAX_PAIRS:
        batch = max((MAX_PAIRS - len(pairs)) * 3, 10_000)
        a, b = rng.integers(0, n, size=batch), rng.integers(0, n, size=batch)
        candidates = np.column_stack((np.minimum(a, b), np.maximum(a, b)))
        candidates = candidates[candidates[:, 0] != candidates[:, 1]].astype(np.int32)
        pairs = np.unique(np.concatenate((pairs, candidates), axis=0), axis=0)
    if len(pairs) > MAX_PAIRS:
        pairs = pairs[rng.choice(len(pairs), MAX_PAIRS, replace=False)]
    pair_indices = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return np.abs(x[pair_indices[:, 0]] - x[pair_indices[:, 1]]), np.abs(y[pair_indices[:, 0]] - y[pair_indices[:, 1]]), pair_indices


def _distances_for_pairs(patterns: np.ndarray, pair_indices: np.ndarray, metric: str) -> np.ndarray:
    left = np.asarray(patterns[pair_indices[:, 0]], dtype=np.float64)
    right = np.asarray(patterns[pair_indices[:, 1]], dtype=np.float64)
    if metric == "euclidean":
        return np.sqrt(np.square(left - right).sum(axis=1)).astype(np.float32)
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    if (denominator <= np.finfo(np.float64).tiny).any():
        raise ValueError("Correlation RDM contains a constant neural response pattern")
    correlation = np.clip(numerator / denominator, -1.0, 1.0)
    return (1.0 - correlation).astype(np.float32)


def _prepare_neural_patterns(dataset: str):
    """Load full neural patterns once; model-specific stimulus masks are applied later."""
    patterns = {}
    if dataset == "tuckute":
        ds = TuckuteDataset(ROOT / "data/tuckute_fmri/osfstorage")
        for subject, (_, bold, ids) in ds.get_condition_per_subject("B").items():
            valid = np.isfinite(bold).all(axis=1)
            ids = ids[valid].astype(int)
            bold = bold[valid]
            patterns[str(subject)] = (ids, bold)
        return patterns
    if dataset == "pereira":
        tasks = _tasks(dataset, ("",))
        if not tasks:
            raise FileNotFoundError("No cached Pereira surprisal tables")
        family, model_id, revision = tasks[0]
        _, table_path = _neural_cache_paths(dataset, family, model_id, revision)
        table = pd.read_parquet(table_path, columns=["subject", "roi", "sentence_id", "y"])
        for subject, part in table.groupby("subject", sort=True):
            wide = part.pivot_table(index="sentence_id", columns="roi", values="y", aggfunc="mean")
            wide = wide.dropna(axis=0, how="any").sort_index()
            if len(wide) >= 3 and wide.shape[1] >= 2:
                patterns[str(subject)] = (wide.index.to_numpy(dtype=str), wide.to_numpy(dtype=np.float32))
        return patterns
    if dataset == "podcast":
        tasks = _tasks(dataset, ("",))
        if not tasks:
            raise FileNotFoundError("No Podcast response caches found")
        family, model_id, revision = tasks[0]
        npy, table_path = _neural_cache_paths(dataset, family, model_id, revision)
        table = pd.read_parquet(table_path)
        y = np.load(npy, mmap_mode="r")
        if y.ndim != 2 or y.shape[0] != len(table):
            raise ValueError(f"Podcast response/table row mismatch: {npy}")
        times = np.linspace(-0.2, 0.8, y.shape[1], endpoint=False)
        window = (times >= 0.05 - 1e-9) & (times <= 0.25 + 1e-9)
        if not window.any():
            raise ValueError("Podcast response cache has no samples in 50–250 ms")
        table["neural_response"] = np.asarray(y[:, window].mean(axis=1), dtype=np.float32)
        table["stimulus_id"] = table.chunk.astype(str) + "::" + table.word_idx.astype(str)
        for subject, part in table.groupby("subject", sort=True):
            # One response vector per word, with electrodes as features.
            wide = part.pivot_table(index="stimulus_id", columns="electrode",
                                    values="neural_response", aggfunc="mean")
            wide = wide.dropna(axis=0, how="any").sort_index()
            if len(wide) < 3 or wide.shape[1] < 2:
                continue
            values = wide.to_numpy(dtype=np.float32)
            valid = np.isfinite(values).all(axis=1) & (np.std(values, axis=1) > 1e-12)
            if valid.sum() >= 3:
                patterns[str(subject)] = (wide.index.to_numpy(dtype=str)[valid], values[valid])
        return patterns
    if dataset in {"broderick", "megmasc"}:
        tasks = _tasks(dataset, ("",))
        if not tasks:
            raise FileNotFoundError(f"No cached surprisal tables for {dataset}")
        family, model_id, revision = tasks[0]
        _, table_path = _neural_cache_paths(dataset, family, model_id, revision)
        if dataset == "broderick":
            # Pooled-LME rewrite: one shared table per model, covering every
            # subject natively (crossed subject x region design) -- mirrors
            # the megmasc branch below, not the old per-subject-file scheme.
            table = pd.read_parquet(table_path, columns=["subject", "region", "chunk", "word_idx", "eeg"])
            table["stimulus_id"] = table.chunk.astype(str) + "::" + table.word_idx.astype(str)
            for subject, part in table.groupby("subject", sort=True):
                wide = part.pivot_table(index="stimulus_id", columns="region", values="eeg", aggfunc="mean")
                wide = wide.dropna(axis=0, how="any").sort_index()
                if len(wide) >= 3 and wide.shape[1] >= 2:
                    patterns[str(subject)] = (wide.index.to_numpy(dtype=str), wide.to_numpy(dtype=np.float32))
        else:
            table = pd.read_parquet(table_path, columns=["subject", "region", "chunk", "word_idx", "y"])
            table["stimulus_id"] = table.chunk.astype(str) + "::" + table.word_idx.astype(str)
            for subject, part in table.groupby("subject", sort=True):
                wide = part.pivot_table(index="stimulus_id", columns="region", values="y", aggfunc="mean")
                wide = wide.dropna(axis=0, how="any").sort_index()
                if len(wide) >= 3 and wide.shape[1] >= 2:
                    patterns[str(subject)] = (wide.index.to_numpy(dtype=str), wide.to_numpy(dtype=np.float32))
        if not patterns:
            raise ValueError(f"No multichannel word response patterns could be built for {dataset}")
        return patterns
    if dataset in {"nieuwland", "szewczyk"}:
        tasks = _tasks(dataset, ("",))
        if not tasks:
            raise FileNotFoundError(f"No cached surprisal tables for {dataset}")
        family, model_id, revision = tasks[0]
        _, table_path = _neural_cache_paths(dataset, family, model_id, revision)
        table = pd.read_parquet(table_path)
        id_col = "item_id" if dataset == "nieuwland" else "chunk"
        response_col = "n400"
        table[id_col] = table[id_col].astype(int) if dataset == "nieuwland" else table[id_col].astype(str)
        for subject, part in table.groupby("subject", sort=True):
            items = (part.groupby(id_col, as_index=False)[response_col].mean()
                     .sort_values(id_col).reset_index(drop=True))
            ids = items[id_col].to_numpy()
            values = items[response_col].to_numpy(dtype=float)[:, None]
            valid = np.isfinite(values[:, 0])
            if valid.sum() >= 3:
                patterns[str(subject)] = (ids[valid], values[valid])
        return patterns
    ds = Fedorenko2016Dataset(ROOT / "data/fedorenko2016")
    words = ds.load_word_table().sort_values("stimulus_id").reset_index(drop=True)
    electrodes = ds.load_electrodes()
    for subject, group in electrodes.groupby("subject_UID"):
        cols = [f"electrode_{int(e)}" for e in group.electrode if f"electrode_{int(e)}" in words]
        values = words[cols].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        values = values[valid]
        ids = words.loc[valid, "stimulus_id"].to_numpy(dtype=int)
        keep_electrode = np.std(values, axis=0) > 1e-8
        values = values[:, keep_electrode]
        patterns[str(subject)] = (ids, values)
    return patterns


def _save_neural_response_rdms(dataset: str, patterns: dict, output_root: Path):
    for subject, (ids, values) in patterns.items():
        if len(ids) * (len(ids) - 1) // 2 > MAX_PAIRS:
            _, _, pair_indices = _pair_distances(
                np.arange(len(ids), dtype=float), np.arange(len(ids), dtype=float),
                dataset, subject)
            distances = _distances_for_pairs(values, pair_indices, "correlation")
            shared = output_root / "rdm_cache" / "shared" / f"{dataset}_{subject}.npz"
            _save_rdm(shared, ids, distances, rdm_type="pearson_neural_distance",
                      pair_indices=pair_indices, dataset=dataset, subject=subject,
                      **({"response_window_s": np.array([0.05, 0.25])} if dataset == "podcast" else {}))
            continue
        metric = "correlation" if values.shape[1] > 1 else "euclidean"
        distances = pdist(values, metric=metric)
        shared = output_root / "rdm_cache" / "shared" / f"{dataset}_{subject}.npz"
        _save_rdm(shared, ids, distances, rdm_type=f"{metric}_neural_distance",
                  dataset=dataset, subject=subject)


def _score_neural_model(task, dataset, output_root, neural_patterns):
    family, model_id, revision = task
    existing = (output_root / "by_subject" / family / safe_model_dir(model_id) /
                f"rsa_{revision.replace('/', '_')}.csv")
    if existing.is_file():
        return f"existing {model_id}@{revision}"
    values = _load_neural_surprisal(dataset, family, model_id, revision)
    if values is None:
        return f"missing cache {model_id}@{revision}"
    if dataset == "podcast":
        rows = []
        for subject, (neural_ids, patterns) in neural_patterns.items():
            part = values.loc[values.subject == subject]
            model_by_id = part.set_index("stimulus_id").surprisal.to_dict()
            shared = np.intersect1d(neural_ids, np.asarray(list(model_by_id), dtype=str))
            if len(shared) < 3:
                continue
            model_values = np.asarray([model_by_id[item] for item in shared], dtype=float)
            neural_index = {item: i for i, item in enumerate(neural_ids)}
            neural_values = patterns[np.asarray([neural_index[item] for item in shared])]
            if len(shared) * (len(shared) - 1) // 2 > MAX_PAIRS:
                _, _, pair_indices = _pair_distances(
                    np.arange(len(shared), dtype=float), np.arange(len(shared), dtype=float),
                    dataset, subject)
                model_dist = np.abs(model_values[pair_indices[:, 0]] - model_values[pair_indices[:, 1]])
                neural_dist = _distances_for_pairs(neural_values, pair_indices, "correlation")
            else:
                pair_indices = None
                model_dist = pdist(model_values[:, None], metric="euclidean")
                neural_dist = pdist(neural_values, metric="correlation")
            score = float(pearsonr(model_dist, neural_dist).statistic)
            cache = (output_root / "rdm_cache" / "model" / family / safe_model_dir(model_id)
                     / revision.replace('/', '_') / f"{subject}.npz")
            metadata = {"pair_indices": pair_indices} if pair_indices is not None else {}
            _save_rdm(cache, shared, model_dist, rdm_type="absolute_surprisal_difference",
                      **metadata, neural_distances=neural_dist,
                      dataset=dataset, model_id=model_id, revision=revision,
                      subject=subject, response_window_s=np.array([0.05, 0.25]))
            rows.append({"family": family, "model_id": model_id, "revision": revision,
                         "subject": subject, "n_stimuli": len(shared),
                         "n_pairs": len(model_dist), "rsa_pearson": score})
        out = output_root / "by_subject" / family / safe_model_dir(model_id) / f"rsa_{revision.replace('/', '_')}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        return f"scored {model_id}@{revision}: {np.mean([r['rsa_pearson'] for r in rows]):.4f}"
    if dataset == "szewczyk":
        rows = []
        for subject, (neural_ids, patterns) in neural_patterns.items():
            part = values.loc[values.subject == subject]
            model_by_id = part.set_index("chunk").surprisal.to_dict()
            shared = np.intersect1d(neural_ids, np.asarray(list(model_by_id), dtype=str))
            if len(shared) < 3:
                continue
            model_values = np.asarray([model_by_id[item] for item in shared], dtype=float)
            neural_index = {item: i for i, item in enumerate(neural_ids)}
            neural_values = patterns[np.asarray([neural_index[item] for item in shared])]
            model_dist = pdist(model_values[:, None], metric="euclidean")
            neural_dist = pdist(neural_values, metric="euclidean")
            score = float(pearsonr(model_dist, neural_dist).statistic)
            cache = (output_root / "rdm_cache" / "model" / family / safe_model_dir(model_id)
                     / revision.replace('/', '_') / f"{subject}.npz")
            _save_rdm(cache, shared, model_dist, rdm_type="absolute_surprisal_difference",
                      neural_distances=neural_dist,
                      dataset=dataset, model_id=model_id, revision=revision, subject=subject)
            rows.append({"family": family, "model_id": model_id, "revision": revision,
                         "subject": subject, "n_stimuli": len(shared),
                         "n_pairs": len(model_dist), "rsa_pearson": score})
        out = output_root / "by_subject" / family / safe_model_dir(model_id) / f"rsa_{revision.replace('/', '_')}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        return f"scored {model_id}@{revision}: {np.mean([r['rsa_pearson'] for r in rows]):.4f}"
    if dataset in {"broderick", "megmasc"}:
        rows = []
        for subject, (neural_ids, patterns) in neural_patterns.items():
            part = values.loc[values.subject == subject]
            model_by_id = part.set_index("stimulus_id").surprisal.to_dict()
            shared = np.intersect1d(neural_ids, np.asarray(list(model_by_id), dtype=str))
            if len(shared) < 3:
                continue
            model_values = np.asarray([model_by_id[item] for item in shared], dtype=float)
            neural_index = {item: i for i, item in enumerate(neural_ids)}
            neural_values = patterns[np.asarray([neural_index[item] for item in shared])]
            if len(shared) * (len(shared) - 1) // 2 > MAX_PAIRS:
                _, _, pair_indices = _pair_distances(
                    np.arange(len(shared), dtype=float), np.arange(len(shared), dtype=float),
                    dataset, subject)
                model_dist = np.abs(model_values[pair_indices[:, 0]] - model_values[pair_indices[:, 1]])
                neural_dist = _distances_for_pairs(neural_values, pair_indices, "correlation")
            else:
                pair_indices = None
                model_dist = pdist(model_values[:, None], metric="euclidean")
                neural_dist = pdist(neural_values, metric="correlation")
            score = float(pearsonr(model_dist, neural_dist).statistic)
            cache = (output_root / "rdm_cache" / "model" / family / safe_model_dir(model_id)
                     / revision.replace('/', '_') / f"{subject}.npz")
            metadata = {"pair_indices": pair_indices} if pair_indices is not None else {}
            _save_rdm(cache, shared, model_dist, rdm_type="absolute_surprisal_difference",
                      **metadata, neural_distances=neural_dist,
                      dataset=dataset, model_id=model_id, revision=revision, subject=subject)
            rows.append({"family": family, "model_id": model_id, "revision": revision,
                         "subject": subject, "n_stimuli": len(shared),
                         "n_pairs": len(model_dist), "rsa_pearson": score})
        out = output_root / "by_subject" / family / safe_model_dir(model_id) / f"rsa_{revision.replace('/', '_')}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        return f"scored {model_id}@{revision}: {np.mean([r['rsa_pearson'] for r in rows]):.4f}"
    id_col = {"pereira": "sentence_id", "tuckute": "sentence_id", "fedorenko2016": "stimulus_id",
              "nieuwland": "item_id", "szewczyk": "chunk"}[dataset]
    ids = values[id_col].to_numpy()
    surp = values.surprisal.to_numpy(dtype=float)
    if not np.isfinite(surp).all():
        raise ValueError(f"Nonfinite surprisal values for {model_id}")
    model_dist = pdist(surp[:, None], metric="euclidean")
    rows = []
    surp_index = {item: i for i, item in enumerate(ids)}
    for subject, (neural_ids_all, patterns) in neural_patterns.items():
        shared = np.intersect1d(ids, neural_ids_all, assume_unique=True)
        if len(shared) < 3:
            continue
        model_rows = np.array([surp_index[item] for item in shared])
        neural_index = {item: i for i, item in enumerate(neural_ids_all)}
        neural_rows = np.array([neural_index[item] for item in shared])
        use_ids = shared
        use_model_dist = pdist(surp[model_rows, None], metric="euclidean")
        metric = "correlation" if patterns.shape[1] > 1 else "euclidean"
        neural_dist = pdist(patterns[neural_rows], metric=metric)
        score = float(pearsonr(use_model_dist, neural_dist).statistic)
        rows.append({"family": family, "model_id": model_id, "revision": revision,
                     "subject": subject, "n_stimuli": len(use_ids),
                     "n_pairs": len(use_model_dist), "rsa_pearson": score})
    cache = output_root / "rdm_cache" / "model" / family / safe_model_dir(model_id) / f"{revision.replace('/', '_')}.npz"
    _save_rdm(cache, ids, model_dist, rdm_type="absolute_surprisal_difference",
              dataset=dataset, model_id=model_id, revision=revision)
    out = output_root / "by_subject" / family / safe_model_dir(model_id) / f"rsa_{revision.replace('/', '_')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return f"scored {model_id}@{revision}: {np.mean([r['rsa_pearson'] for r in rows]):.4f}"


def _summarize(dataset: str, output_root: Path):
    if dataset in BEHAVIORAL:
        for measure in MEASURES[dataset]:
            root = output_root / measure.lower()
            rows = []
            for path in root.glob("*/*/rsa_*.csv"):
                df = pd.read_csv(path)
                if len(df):
                    rows.append(df.iloc[0].to_dict())
            if not rows:
                continue
            summary = filter_registry_models(pd.DataFrame(rows))
            embedding_summary_path = ROOT / "outputs" / dataset / "pearson_rsa" / measure.lower() / "model_summary.csv"
            if embedding_summary_path.exists():
                embedding = pd.read_csv(embedding_summary_path)
                embedding = embedding[["family", "model_id", "revision", "rsa_pearson", "encoding_pearson"]]
                embedding.rename(columns={"rsa_pearson": "embedding_rsa"}, inplace=True)
                summary = summary.merge(embedding, on=["family", "model_id", "revision"], how="left")
            metadata = [lookup_model_meta(row.model_id, row.family) for row in summary.itertuples()]
            summary["group"] = [item.group for item in metadata]
            summary["n_params"] = [item.n_params for item in metadata]
            out = root / "model_summary.csv"
            summary.to_csv(out, index=False)
            _save_scaling(summary, root)
            print(f"{dataset}/{measure}: {len(summary)} models, mean surprisal RSA={summary.rsa_pearson.mean():.4f}", flush=True)
            for reference in ("embedding_rsa", "encoding_pearson"):
                if reference in summary and summary[reference].notna().sum() >= 3:
                    paired = summary[["rsa_pearson", reference]].dropna()
                    correlation = pearsonr(paired.rsa_pearson, paired[reference]).statistic
                    print(f"  surprisal RSA vs {reference}: r={correlation:.4f} across {len(paired)} models", flush=True)
    else:
        rows = []
        for path in (output_root / "by_subject").glob("*/*/rsa_*.csv"):
            df = pd.read_csv(path, dtype={"subject": str})
            if df.empty:
                continue
            first = df.iloc[0]
            rows.append({"family": first.family, "model_id": first.model_id,
                         "revision": first.revision, "n_subjects": df.subject.nunique(),
                         "rsa_pearson": df.rsa_pearson.mean()})
        if rows:
            summary = filter_registry_models(pd.DataFrame(rows))
            embedding_path = ROOT / "outputs" / dataset / "pearson_rsa" / "model_summary.csv"
            if embedding_path.exists():
                embedding = pd.read_csv(embedding_path)
                embedding = embedding[["family", "model_id", "revision", "rsa_pearson", "encoding_pearson"]]
                embedding.rename(columns={"rsa_pearson": "embedding_rsa"}, inplace=True)
                summary = summary.merge(embedding, on=["family", "model_id", "revision"], how="left")
            metadata = [lookup_model_meta(row.model_id, row.family) for row in summary.itertuples()]
            summary["group"] = [item.group for item in metadata]
            summary["n_params"] = [item.n_params for item in metadata]
            summary.to_csv(output_root / "model_summary.csv", index=False)
            _save_scaling(summary, output_root)
            print(f"{dataset}: {len(summary)} models, mean surprisal RSA={summary.rsa_pearson.mean():.4f}", flush=True)
            for reference in ("embedding_rsa", "encoding_pearson"):
                if reference in summary and summary[reference].notna().sum() >= 3:
                    paired = summary[["rsa_pearson", reference]].dropna()
                    correlation = pearsonr(paired.rsa_pearson, paired[reference]).statistic
                    print(f"  surprisal RSA vs {reference}: r={correlation:.4f} across {len(paired)} models", flush=True)


def _save_scaling(summary: pd.DataFrame, root: Path):
    valid = summary.loc[np.isfinite(summary.n_params) & (summary.n_params > 0)].copy()
    valid["log10_params"] = np.log10(valid.n_params)
    rows = []
    scopes = [("all", "all models", valid)]
    scopes += [("group", name, part) for name, part in valid.groupby("group")]
    scopes += [("family", name, part) for name, part in valid.groupby("family")]
    for scope, name, part in scopes:
        if len(part) < 3 or part.log10_params.nunique() < 3:
            continue
        for metric in ("rsa_pearson", "encoding_pearson"):
            if metric not in part:
                continue
            fit = linregress(part.log10_params, part[metric])
            rows.append({"scope": scope, "name": name, "metric": metric,
                         "n_models": len(part), "slope_per_log10_param": fit.slope,
                         "pearson_r": fit.rvalue})
    repeated = valid.groupby("family").filter(lambda part: part.log10_params.nunique() > 1)
    if len(repeated) >= 3:
        x = repeated.log10_params - repeated.groupby("family").log10_params.transform("mean")
        for metric in ("rsa_pearson", "encoding_pearson"):
            if metric not in repeated:
                continue
            y = repeated[metric] - repeated.groupby("family")[metric].transform("mean")
            rows.append({"scope": "within_family", "name": "family-centered", "metric": metric,
                         "n_models": len(repeated), "slope_per_log10_param": float((x*y).sum()/(x*x).sum()),
                         "pearson_r": float(pearsonr(x, y).statistic)})
    pd.DataFrame(rows).to_csv(root / "scaling_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(BEHAVIORAL | NEURAL))
    parser.add_argument("--measure", choices=sorted({m for ms in MEASURES.values() for m in ms}), action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output_root = ROOT / "outputs" / args.dataset / "pearson_rsa" / "surprisal"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        _summarize(args.dataset, output_root)
        return
    measures = tuple(dict.fromkeys(args.measure or MEASURES[args.dataset])) if args.dataset in BEHAVIORAL else ("",)
    tasks = _tasks(args.dataset, measures)
    if args.limit:
        tasks = tasks[:args.limit]
    if args.dataset in BEHAVIORAL:
        responses = _prepare_behavior(args.dataset)
        print(f"{args.dataset}: {len(tasks)} models across {len(measures)} reading measures", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_score_behavior_model, task, args.dataset, responses,
                                   output_root, measures) for task in tasks]
            for future in as_completed(futures):
                print(future.result(), flush=True)
    else:
        print(f"{args.dataset}: {len(tasks)} cached model tasks", flush=True)
        neural_patterns = _prepare_neural_patterns(args.dataset)
        _save_neural_response_rdms(args.dataset, neural_patterns, output_root)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_score_neural_model, task, args.dataset, output_root, neural_patterns)
                       for task in tasks]
            for future in as_completed(futures):
                print(future.result(), flush=True)
    _summarize(args.dataset, output_root)


if __name__ == "__main__":
    main()
