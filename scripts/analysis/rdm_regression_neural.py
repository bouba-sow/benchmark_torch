#!/usr/bin/env python3
"""RDM-regression (held-out delta log-likelihood) track for the neural datasets.

Datasets: pereira, tuckute, fedorenko2016, broderick, megmasc, nieuwland, szewczyk,
podcast (LPP and Natural Stories fMRI have no response RDM and are not covered).

One response unit per subject: the shared per-subject neural RDM written by
surprisal_rsa.py (``outputs/<ds>/pearson_rsa/surprisal/rdm_cache/shared/<ds>_<subject>.npz``,
full pdist order or a fixed 250k-pair sample) plus the aligned word-length/frequency
confound RDMs. Per subject and model, predict the neural RDM from the model RDM with and
without the confounds and score the item-fold cross-validated delta log-likelihood
(neuroai/analysis/rdm_regression.py, docs/rdm_regression.md). Subject scores are then
averaged (summary) at the layer that is best for the subject mean.

* embedding track: correlation distance between per-item hidden states, every layer;
* surprisal track: |delta surprisal|.

Model RDMs are rebuilt from per-item values on each unit's own pairs. Inputs are read-only;
output goes under --out-root (refuses outputs/). Resumable per model.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import struct
import sys
import tempfile
import shutil
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rdm-regression-matplotlib")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

from neuroai.analysis.benchmark_common import EXCLUDED_FAMILY_DIRS, lookup_model_meta
from neuroai.analysis.rdm_regression import RdmCvPlan, all_pairs, standardize_features
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks  # noqa: F401
from neuroai.sweep import safe_model_dir

NEURAL = ("pereira", "tuckute", "fedorenko2016", "broderick", "megmasc", "nieuwland", "szewczyk", "podcast",
          "nieuwland_avg", "szewczyk_avg")
# N400 datasets with one grand-average item RDM (mean N400 per item over the subjects who saw it)
# instead of one very noisy RDM per subject. Outputs are kept apart under the *_avg dataset name.
AVG = {"nieuwland_avg": "nieuwland", "szewczyk_avg": "szewczyk"}
MIN_SUBJECTS_PER_ITEM = 5


def base_of(dataset: str) -> str:
    return AVG.get(dataset, dataset)
VARIANTS = ("none", "len_freq")
LAYER_SURPRISAL = -1
LAST_LAYER_ONLY = os.environ.get("RDM_LAST_LAYER_ONLY") == "1"   # last layer only (bootstrap / ceiling runs)
EMBED_STANDARDIZE = os.environ.get("RDM_EMBED_STANDARDIZE") == "1"  # z-score features across the item universe
CACHE = Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache")
CHUNK_BYTES = int(float(os.environ.get("RDM_CHUNK_GIB", "10")) * 1024**3)
STREAM_BYTES = 4 * 1024**3
MAX_GRAM_ITEMS = 20000


# ----------------------------------------------------------------------------- response units
def canon_ids(ids) -> np.ndarray:
    """Stimulus IDs as plain strings ('12' for ints, unchanged for string IDs)."""
    a = np.asarray(ids)
    if a.dtype.kind in "iu":
        return a.astype(np.int64).astype(str)
    return a.astype(str)


def _load_rdm(path: Path):
    with np.load(path, allow_pickle=True) as a:
        pairs = np.asarray(a["pair_indices"], dtype=np.int32) if "pair_indices" in a.files else None
        subject = str(a["subject"].item()) if "subject" in a.files else None
        return canon_ids(a["stimulus_ids"]), np.asarray(a["distances"], dtype=np.float64), pairs, subject


def _restrict(ids, response, conf, pairs, keep):
    """Keep only stimuli in ``keep`` (bool mask); pairs become explicit and re-indexed."""
    n = len(ids)
    full = pairs if pairs is not None else all_pairs(n)
    sel = keep[full[:, 0]] & keep[full[:, 1]]
    new_index = np.cumsum(keep) - 1
    return (ids[keep], response[sel], [c[sel] for c in conf],
            new_index[full[sel]].astype(np.int32))


def load_units(dataset: str, n_repeats: int, seed: int, eligible: dict | None = None) -> list[dict]:
    """One response unit per subject. ``eligible`` (subject or '*' -> set of stimulus IDs)
    restricts every unit to stimuli that have a surprisal value, so both tracks use the
    same items (the existing surprisal RSA used exactly this intersection)."""
    shared_dir = ROOT / "outputs" / dataset / "pearson_rsa" / "surprisal" / "rdm_cache" / "shared"
    units, dropped_items, dropped_units = [], 0, 0
    for path in sorted(shared_dir.glob("*.npz")):
        ids, response, pairs, subject = _load_rdm(path)
        subject = subject or path.stem
        conf = []
        for name in ("word_length", "frequency"):
            cpath = ROOT / "outputs" / dataset / "pearson_rsa" / "confounds" / subject / f"{name}_rdm.npz"
            cids, dist, cpairs, _ = _load_rdm(cpath)
            if not np.array_equal(cids, ids) or (pairs is None) != (cpairs is None) or (
                    pairs is not None and not np.array_equal(pairs, cpairs)):
                raise ValueError(f"Confound '{name}' not aligned with response RDM: {dataset}/{subject}")
            conf.append(dist)
        if eligible is not None:
            ok = eligible.get(subject, eligible.get("*", set()))
            keep = np.fromiter((s in ok for s in ids.tolist()), dtype=bool, count=len(ids))
            dropped_items += int((~keep).sum())
            if keep.sum() < 6:
                dropped_units += 1
                continue
            if not keep.all():
                ids, response, conf, pairs = _restrict(ids, response, conf, pairs, keep)
        n = len(ids)
        if n < 6:
            dropped_units += 1
            continue
        units.append({
            "subject": subject, "ids": ids, "n": n, "pairs": pairs, "n_pairs": len(response),
            "plans": {"none": RdmCvPlan(response, n, pairs, None, n_repeats, seed=seed),
                      "len_freq": RdmCvPlan(response, n, pairs, np.column_stack(conf), n_repeats, seed=seed)},
        })
    if dropped_items or dropped_units:
        print(f"{dataset}: dropped {dropped_items} stimulus entries without surprisal, "
              f"{dropped_units} subjects with too few items left", flush=True)
    if not units:
        raise FileNotFoundError(f"No usable shared response RDMs in {shared_dir}")
    return units


def _id_col(dataset: str) -> str:
    return {"pereira": "sentence_id", "tuckute": "sentence_id", "fedorenko2016": "stimulus_id",
            "nieuwland": "item_id", "szewczyk": "chunk"}.get(dataset, "stimulus_id")


def _surprisal_id_sets(dataset: str, family: str, model_id: str, revision: str) -> dict | None:
    """Stimulus IDs with a finite surprisal for one model, read with light column reads
    (no duplicate-consistency checks; subject -> set, or '*' -> set when the table has no subject)."""
    from scripts.analysis.surprisal_rsa import _neural_cache_paths
    npy, table = _neural_cache_paths(dataset, family, model_id, revision)
    if not Path(npy).is_file() or not Path(table).is_file():
        return None
    if dataset in ("broderick", "megmasc", "podcast"):
        cols = ["subject", "chunk", "word_idx"] + (["surprisal"] if dataset == "podcast" else [])
    elif dataset == "szewczyk":
        cols = ["subject", "chunk"]
    else:
        cols = [_id_col(dataset)]
    df = pd.read_parquet(table, columns=cols)
    if dataset == "podcast":
        finite = np.isfinite(df["surprisal"].to_numpy(dtype=float))
    else:
        values = np.load(npy, mmap_mode="r")
        if values.ndim != 2 or values.shape[0] != len(df):
            raise ValueError(f"Surprisal/table row mismatch: {npy}")
        finite = np.isfinite(np.asarray(values[:, -1], dtype=float))
    df = df.loc[finite]
    if dataset in ("broderick", "megmasc", "podcast"):
        ids = canon_ids((df.chunk.astype(str) + "::" + df.word_idx.astype(str)).to_numpy())
    else:
        ids = canon_ids(df[_id_col(dataset)].to_numpy())
    if "subject" in df.columns:
        frame = pd.DataFrame({"s": df.subject.astype(str).to_numpy(), "i": ids})
        return {s: set(g.i) for s, g in frame.groupby("s")}
    return {"*": set(ids)}


def build_eligible(dataset: str, out_root: Path) -> Path:
    """Intersect the surprisal-covered stimulus sets over EVERY model, so all models (and
    both tracks) are scored on identical stimuli. Written to <out_root>/<dataset>/eligible_ids.json."""
    import json
    inter, n_models = None, 0
    chunk_to_item = _szewczyk_chunk_to_item() if dataset == "szewczyk_avg" else None
    for family, model_id, revision, _ in surprisal_tasks(dataset):
        sets = _surprisal_id_sets(base_of(dataset), family, model_id, revision)
        if sets is None:
            continue
        if dataset in AVG:  # item-level coverage: an item counts if any subject's trial has a value
            items = set()
            for subj, ids_ in sets.items():
                items |= ({str(chunk_to_item[(subj, c)]) for c in ids_ if (subj, c) in chunk_to_item}
                          if chunk_to_item else set(ids_))
            sets = {"*": items}
        n_models += 1
        if inter is None:
            inter = {k: set(v) for k, v in sets.items()}
            continue
        for k in list(inter):
            inter[k] &= sets.get(k, set())
    if inter is None:
        raise FileNotFoundError(f"No surprisal caches found for {dataset}")
    path = out_root / dataset / "eligible_ids.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        json.dump({"n_models": n_models, "ids": {k: sorted(v) for k, v in inter.items()}}, fh)
    os.replace(tmp, path)
    print(f"{dataset}: eligible stimuli over {n_models} models: "
          + ", ".join(f"{k}={len(v)}" for k, v in list(inter.items())[:6]) + (" ..." if len(inter) > 6 else ""), flush=True)
    return path


def eligible_ids(dataset: str, out_root: Path | None = None) -> dict | None:
    """Stimulus IDs that have a surprisal value: the all-model intersection if
    ``--build-eligible`` has been run for this dataset, else the first model with a cache."""
    import json
    if out_root is not None:
        path = out_root / dataset / "eligible_ids.json"
        if path.is_file():
            with open(path) as fh:
                return {k: set(v) for k, v in json.load(fh)["ids"].items()}
    from scripts.analysis.surprisal_rsa import _load_neural_surprisal
    if dataset in AVG:
        raise FileNotFoundError(f"Run --build-eligible for {dataset} first (item-level coverage file missing)")
    id_col = {"pereira": "sentence_id", "tuckute": "sentence_id", "fedorenko2016": "stimulus_id",
              "nieuwland": "item_id", "szewczyk": "chunk"}.get(dataset, "stimulus_id")
    for family, model_id, revision, _ in surprisal_tasks(dataset):
        v = _load_neural_surprisal(dataset, family, model_id, revision)
        if v is None:
            continue
        v = v.copy()
        v["_id"] = canon_ids(v[id_col].to_numpy())
        if "subject" in v.columns:
            return {str(s): set(g._id) for s, g in v.groupby(v.subject.astype(str))}
        return {"*": set(v._id)}
    return None


def load_average_unit(dataset: str, n_repeats: int, seed: int, eligible: dict | None) -> list[dict]:
    """One response unit: the RDM of item-mean N400 amplitudes (|difference|), with the item's
    word-length/frequency confound RDMs (taken from the per-subject confound archives)."""
    from scipy.spatial.distance import pdist
    from scripts.analysis.surprisal_rsa import _neural_cache_paths
    base = AVG[dataset]
    family, model_id, revision, _ = next(t for t in surprisal_tasks(base)
                                         if Path(_neural_cache_paths(base, *t[:3])[1]).is_file())
    table_path = _neural_cache_paths(base, family, model_id, revision)[1]
    chunk_to_item = _szewczyk_chunk_to_item() if base == "szewczyk" else None

    def item_key(subject: str, raw) -> str:
        return str(chunk_to_item[(subject, str(raw))]) if chunk_to_item else canon_ids([raw])[0]

    cols = ["subject", "chunk", "n400"] if base == "szewczyk" else ["subject", "item_id", "n400"]
    df = pd.read_parquet(table_path, columns=cols)
    raw_col = "chunk" if base == "szewczyk" else "item_id"
    df["subject"] = df.subject.astype(str)
    df["item"] = [item_key(s, r) for s, r in zip(df.subject, df[raw_col])]
    df = df.loc[np.isfinite(df.n400.to_numpy(dtype=float))]
    agg = df.groupby("item").n400.agg(["mean", "count"]).reset_index()
    agg = agg.loc[agg["count"] >= MIN_SUBJECTS_PER_ITEM]
    agg = agg.sort_values("item", key=lambda s: s.astype(int)).reset_index(drop=True)
    # confounds: per-item word length / frequency, identical across the subjects who saw the item
    feats = {"word_length": {}, "frequency": {}}
    for d in sorted((ROOT / "outputs" / base / "pearson_rsa" / "confounds").iterdir()):
        for name in feats:
            ids, _, _, _ = _load_rdm(d / f"{name}_rdm.npz")
            with np.load(d / f"{name}_rdm.npz", allow_pickle=True) as a:
                vals = np.asarray(a["feature_values"], dtype=float)
            for raw, v in zip(ids.tolist(), vals):
                key = item_key(d.name, raw)
                old = feats[name].setdefault(key, v)
                if abs(old - v) > 1e-4:
                    raise ValueError(f"{base}: item {key} has different {name} across subjects")
    ids = agg["item"].to_numpy(dtype=str)
    if eligible is not None:
        keep = np.fromiter((s in eligible.get("*", set()) for s in ids.tolist()), dtype=bool, count=len(ids))
        print(f"{dataset}: {int((~keep).sum())} of {len(ids)} items dropped (no surprisal for some model)", flush=True)
        agg, ids = agg.loc[keep].reset_index(drop=True), ids[keep]
    missing = [s for s in ids.tolist() if s not in feats["word_length"]]
    if missing:
        raise ValueError(f"{dataset}: {len(missing)} items without confound values, e.g. {missing[:3]}")
    n = len(ids)
    response = pdist(agg["mean"].to_numpy(dtype=float)[:, None])
    conf = np.column_stack([pdist(np.asarray([feats[name][s] for s in ids.tolist()])[:, None])
                            for name in ("word_length", "frequency")])
    print(f"{dataset}: {n} items, mean {agg['count'].mean():.0f} subjects/item (min {MIN_SUBJECTS_PER_ITEM})", flush=True)
    return [{"subject": "grand_average", "ids": ids, "n": n, "pairs": None, "n_pairs": len(response),
             "plans": {"none": RdmCvPlan(response, n, None, None, n_repeats, seed=seed),
                       "len_freq": RdmCvPlan(response, n, None, conf, n_repeats, seed=seed)}}]


def _unit_pairs(unit: dict) -> np.ndarray:
    return unit["pairs"] if unit["pairs"] is not None else all_pairs(unit["n"])


# ----------------------------------------------------------------------------- id -> cache row
def _szewczyk_chunk_to_item() -> dict[tuple[str, str], int]:
    """(subject, chunk) -> embedding item row, as in word_neural_embedding_rsa: the same chunk
    ID can carry different sentences for different subjects."""
    src = pd.read_csv(ROOT / "data/n400_paper_formula/osfstorage/datasets/szewczyk_2022.tsv", sep="\t",
                      dtype={"Subject": str, "Item": str})
    src["chunk"] = src.dataset.astype(str) + "_" + src.Item.astype(str)
    items = src[["FullText", "TargetWords"]].drop_duplicates().reset_index(drop=True)
    items["item_id"] = np.arange(len(items), dtype=np.int64)
    m = src[["Subject", "chunk", "FullText", "TargetWords"]].drop_duplicates().merge(
        items, on=["FullText", "TargetWords"], validate="many_to_one")
    if m.duplicated(["Subject", "chunk"]).any():
        raise ValueError("Szewczyk (subject, chunk) maps to several embedding items")
    return {(str(s), str(c)): int(i) for s, c, i in zip(m.Subject, m.chunk, m.item_id)}


def _all_shared_ids(dataset: str) -> np.ndarray:
    """Every stimulus ID in the (unrestricted) shared response RDMs."""
    shared_dir = ROOT / "outputs" / dataset / "pearson_rsa" / "surprisal" / "rdm_cache" / "shared"
    return np.unique(np.concatenate([_load_rdm(p)[0] for p in sorted(shared_dir.glob("*.npz"))]))


def rows_for(dataset: str, ids_by_unit: list[np.ndarray], subjects: list[str]) -> list[np.ndarray]:
    """Cache row of every stimulus of every unit (model-independent). Position-based
    mappings (Pereira, Fedorenko) use the full stimulus set, not the surprisal-restricted one."""
    union = _all_shared_ids(dataset) if dataset in ("pereira", "fedorenko2016") else np.unique(np.concatenate(ids_by_unit))
    if dataset in AVG:  # item ids are already embedding-row ids
        return [np.asarray([int(s) for s in ids.tolist()], dtype=np.int64) for ids in ids_by_unit]
    if dataset == "pereira":
        # get_sentences() order is sorted(stimulus_id); subject 018 has all 627 sentences.
        order = {s: i for i, s in enumerate(sorted(union.tolist()))}
        if len(order) != 627:
            raise ValueError(f"Expected 627 Pereira sentences, found {len(order)}")
        table = order
    elif dataset == "tuckute":
        from neuroai.datasets.tuckute import TuckuteDataset
        all_ids = TuckuteDataset(ROOT / "data/tuckute_fmri/osfstorage").get_condition("all")[2]
        table = {str(int(x)): i for i, x in enumerate(all_ids)}
    elif dataset == "fedorenko2016":
        nums = sorted(int(x) for x in union)
        if nums != list(range(len(nums))):
            raise ValueError("Fedorenko stimulus_ids are not 0..n-1; row mapping unknown")
        table = {str(i): i for i in nums}
    elif dataset in ("broderick", "megmasc", "podcast"):
        table = {s: int(s.rsplit("::", 1)[1]) for s in union.tolist()}
    elif dataset == "nieuwland":
        table = {s: int(s) for s in union.tolist()}
    elif dataset == "szewczyk":
        by_subject = _szewczyk_chunk_to_item()
        table = None
    else:
        raise KeyError(dataset)
    out = []
    for ids, subject in zip(ids_by_unit, subjects):
        if table is None:
            lookup = {s: by_subject.get((subject, s)) for s in ids.tolist()}
        else:
            lookup = {s: table.get(s) for s in ids.tolist()}
        missing = [s for s, r in lookup.items() if r is None]
        if missing:
            raise ValueError(f"{dataset}/{subject}: {len(missing)} stimuli without a cache row, e.g. {missing[:3]}")
        out.append(np.asarray([lookup[s] for s in ids.tolist()], dtype=np.int64))
    return out


def prepare(dataset: str, n_repeats: int, seed: int, out_root: Path | None = None) -> dict:
    elig = eligible_ids(dataset, out_root)
    units = (load_average_unit(dataset, n_repeats, seed, elig) if dataset in AVG
             else load_units(dataset, n_repeats, seed, elig))
    unit_rows = rows_for(dataset, [u["ids"] for u in units], [u["subject"] for u in units])
    universe = np.unique(np.concatenate(unit_rows))
    for u, rows in zip(units, unit_rows):
        pos = np.searchsorted(universe, rows)
        pairs = _unit_pairs(u)
        u["up0"], u["up1"] = pos[pairs[:, 0]].astype(np.int32), pos[pairs[:, 1]].astype(np.int32)
        u["rows"] = rows
    return {"units": units, "universe": universe}


# ----------------------------------------------------------------------------- unit rows
def _score_unit(unit: dict, dataset: str, track: str, task: tuple, layer: int, rdm: np.ndarray) -> list[dict]:
    if not np.isfinite(rdm).all():
        raise ValueError(f"Non-finite model RDM ({dataset}/{unit['subject']}/{task[1]} layer {layer})")
    out = []
    for variant in VARIANTS:
        plan = unit["plans"][variant]
        r = plan.delta_ll(rdm)
        out.append({"dataset": dataset, "track": track, "family": task[0], "model_id": task[1],
                    "revision": task[2], "subject": unit["subject"], "layer": layer,
                    "confounds": variant, "n_items": unit["n"], "n_pairs": unit["n_pairs"],
                    "rsa_pearson": plan.pearson(rdm), "delta_ll_per_obs": r.delta_ll_per_obs,
                    "delta_ll_per_obs_sd": r.delta_ll_per_obs_sd, "ll_full_per_obs": r.ll_full_per_obs,
                    "ll_base_per_obs": r.ll_base_per_obs, "n_test_pairs": r.n_test_pairs,
                    "frac_pairs_tested": r.frac_pairs_tested, "n_folds": r.n_folds,
                    "n_repeats": r.n_repeats})
    return out


def _out_path(root: Path, dataset: str, track: str, task: tuple) -> Path:
    return root / dataset / track / task[0] / safe_model_dir(task[1]) / f"rdm_regression_{task[2].replace('/', '_')}.csv"


def _write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- embedding sources
@contextmanager
def open_array(path: Path, key: str):
    """Yield an (n, ...) array from an .npz member: memory-mapped when stored uncompressed,
    streamed to $TMPDIR and mapped when compressed and huge, plain load otherwise."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(f"{key}.npy")
        stored = info.compress_type == zipfile.ZIP_STORED
        mapped = None
        if stored:
            with z.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    shape, fortran, dtype = np.lib.format._read_array_header(member, version)
                header_bytes = member.tell()
            if not fortran and dtype.kind == "f":
                with path.open("rb") as raw:
                    raw.seek(info.header_offset)
                    fields = struct.unpack("<IHHHHHIIIHH", raw.read(30))
                data_offset = info.header_offset + 30 + fields[-2] + fields[-1] + header_bytes
                mapped = np.memmap(path, mode="r", dtype=dtype, offset=data_offset, shape=shape, order="C")
        big = (not stored or mapped is None) and info.file_size > STREAM_BYTES
        tmp = None
        if big:
            fd, tmp = tempfile.mkstemp(prefix="rdm_regression_", suffix=".npy")
            os.close(fd)
            with z.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024**2)
    try:
        if mapped is not None:
            yield mapped
        elif tmp is not None:
            yield np.load(tmp, mmap_mode="r")
        else:
            with np.load(path, allow_pickle=True) as a:
                yield a[key]
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)


class Source:
    """(n_rows, n_layers, dim) hidden states behind ``read(rows, l0, l1)``."""

    def __init__(self, array=None, files=None, pereira_cbr=None):
        self.array, self.files = array, files
        if array is not None:
            a = array if array.ndim == 3 else array[:, None, :]
            self.n_rows, self.n_layers, self.dim = a.shape
        else:
            first = np.load(files[0])["features"]
            self.n_rows, (self.n_layers, self.dim) = len(files), first.shape

    def read(self, rows: np.ndarray, l0: int, l1: int) -> np.ndarray:
        if self.array is not None:
            a = self.array if self.array.ndim == 3 else self.array[:, None, :]
            return np.asarray(a[rows, l0:l1, :], dtype=np.float32)
        out = np.empty((len(rows), l1 - l0, self.dim), dtype=np.float32)
        for k, r in enumerate(rows):
            with np.load(self.files[int(r)]) as z:
                f = z["features"]
            if f.shape != (self.n_layers, self.dim):
                raise ValueError(f"Inconsistent sentence cache shape {f.shape}: {self.files[int(r)]}")
            out[k] = f[l0:l1]
        return out


def _pereira_files(cache_model_id: str, revision: str, n: int):
    d = CACHE / "pereira_embed_features"
    safe, rev = cache_model_id.replace("/", "_"), revision.replace("/", "_").replace("-", "_")
    if cache_model_id in {"cclark/cbr-rnn", "cclark/cbr-rnn-m"}:
        return ("cbr", d / f"{safe}__{rev}__sentence_mean_cbrrnn.npz")
    prefix = f"{safe}__{rev}__en__sec"
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)__fulltext_ml\d+__sdelta\d+__hidden_mean_all\.npz$")
    paths = {}
    for p in d.glob(f"{prefix}*__hidden_mean_all.npz"):
        m = pat.match(p.name)
        if m:
            paths[int(m.group(1))] = p
    missing = sorted(set(range(n)) - paths.keys())
    if missing:
        raise FileNotFoundError(f"{len(missing)} sentence caches missing (first {missing[:3]})")
    return ("files", [paths[i] for i in range(n)])


@contextmanager
def open_source(dataset: str, task: tuple, ctx: dict):
    family, model_id, revision, extra = task
    if dataset == "pereira":
        kind, what = _pereira_files(extra["load_model_id"], revision, 627)
        if kind == "cbr":
            with open_array(what, "features") as arr:
                yield Source(array=arr)
        else:
            yield Source(files=what)
        return
    if dataset == "tuckute":
        path = (ROOT / "outputs_jeanzay/tuckute/embedding_sweep/.embed_cache" / safe_model_dir(model_id)
                / revision.replace("/", "_") / "embeddings.npz")
        key = "embeddings"
    elif dataset == "fedorenko2016":
        paths = list((CACHE / "fedorenko2016_features" / safe_model_dir(model_id) / revision).glob("fedorenko2016_*.npz"))
        if len(paths) != 1:
            raise FileNotFoundError("missing/ambiguous Fedorenko cache")
        path, key = paths[0], "word_embeddings"
    else:
        path, key = extra["path"], "features"
    if not Path(path).is_file():
        raise FileNotFoundError(str(path))
    with open_array(Path(path), key) as arr:
        yield Source(array=arr)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    x -= x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if (norm <= np.finfo(np.float32).tiny).any() or not np.isfinite(norm).all():
        raise ValueError("Constant or non-finite hidden-state vector: correlation distance undefined")
    x /= norm
    return x


def score_embedding(task, dataset, prepared, out_root):
    family, model_id, revision, extra = task
    out = _out_path(out_root, dataset, "embedding", task[:3])
    if out.is_file():
        return f"existing {model_id}@{revision}"
    units, universe = prepared["units"], prepared["universe"]
    if len(universe) > MAX_GRAM_ITEMS:
        raise ValueError(f"{len(universe)} universe items exceed the Gram-matrix limit")
    try:
        cm = open_source(dataset, task, prepared)
        src = cm.__enter__()
    except FileNotFoundError as e:
        return f"missing cache {model_id}@{revision}: {e}"
    try:
        if universe.max() >= src.n_rows:
            return f"skipped {model_id}@{revision}: cache has {src.n_rows} rows, need row {universe.max()}"
        L, d = src.n_layers, src.dim
        chunk = max(1, min(L, CHUNK_BYTES // (len(universe) * d * 4)))
        rows_out = []
        for l0 in range(L - 1 if LAST_LAYER_ONLY else 0, L, chunk):
            l1 = min(L, l0 + chunk)
            block = src.read(universe, l0, l1)
            for j in range(l1 - l0):
                x = _normalize(standardize_features(block[:, j, :]) if EMBED_STANDARDIZE else block[:, j, :])
                gram = x @ x.T
                for unit in units:
                    rdm = 1.0 - gram[unit["up0"], unit["up1"]].astype(np.float64)
                    rows_out += _score_unit(unit, dataset, "embedding", task[:3], l0 + j, rdm)
            del block
    finally:
        cm.__exit__(None, None, None)
    _write(out, rows_out)
    return f"scored {model_id}@{revision} ({L} layers, {len(units)} subjects)"


# ----------------------------------------------------------------------------- surprisal
def score_surprisal(task, dataset, prepared, out_root):
    from scripts.analysis.surprisal_rsa import _load_neural_surprisal
    family, model_id, revision, _ = task
    out = _out_path(out_root, dataset, "surprisal", task[:3])
    if out.is_file():
        return f"existing {model_id}@{revision}"
    base = base_of(dataset)
    values = _load_neural_surprisal(base, family, model_id, revision)
    if values is None:
        return f"missing cache {model_id}@{revision}"
    id_col = {"pereira": "sentence_id", "tuckute": "sentence_id", "fedorenko2016": "stimulus_id",
              "nieuwland": "item_id", "szewczyk": "chunk"}.get(base, "stimulus_id")
    values = values.copy()
    if dataset == "szewczyk_avg":
        # surprisal belongs to the item text: map (subject, chunk) -> item and average
        # over the subjects' trials of the same item (they must agree).
        m = _szewczyk_chunk_to_item()
        values["_id"] = [str(m[(str(s), str(c))]) for s, c in zip(values.subject, values.chunk)]
        spread = values.groupby("_id").surprisal.agg(lambda x: float(x.max() - x.min()))
        if (spread > 1e-3).any():
            raise ValueError(f"Surprisal of the same Szewczyk item differs across trials ({model_id})")
        values = values.groupby("_id", as_index=False).surprisal.mean()
    else:
        values["_id"] = canon_ids(values[id_col].to_numpy())
    per_subject = "subject" in values.columns
    if per_subject:
        values["subject"] = values.subject.astype(str)
        lookup = {s: dict(zip(g._id, g.surprisal.astype(float))) for s, g in values.groupby("subject")}
    else:
        common = dict(zip(values._id, values.surprisal.astype(float)))
    rows_out = []
    for unit in prepared["units"]:
        table = lookup.get(unit["subject"], {}) if per_subject else common
        missing = [s for s in unit["ids"].tolist() if s not in table]
        if missing:
            return (f"skipped {model_id}@{revision}: {len(missing)} of {unit['n']} stimuli of "
                    f"subject {unit['subject']} lack surprisal (e.g. {missing[:2]})")
        s = np.asarray([table[i] for i in unit["ids"].tolist()], dtype=np.float64)
        if not np.isfinite(s).all():
            raise ValueError(f"Non-finite surprisal for {model_id}@{revision} subject {unit['subject']}")
        pairs = _unit_pairs(unit)
        rdm = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
        rows_out += _score_unit(unit, dataset, "surprisal", task[:3], LAYER_SURPRISAL, rdm)
    _write(out, rows_out)
    return f"scored {model_id}@{revision} ({len(prepared['units'])} subjects)"


# ----------------------------------------------------------------------------- tasks
def _keep(tasks):
    tasks = [t for t in tasks if t[0] not in EXCLUDED_FAMILY_DIRS]
    tasks = list(dict.fromkeys((t[0], t[1], t[2], tuple(sorted(t[3].items()))) for t in tasks))
    tasks = [(f, m, r, dict(e)) for f, m, r, e in tasks]
    tasks.sort(key=lambda t: (lookup_model_meta(t[1], t[0]).n_params or float("inf"), t[1], t[2]))
    return tasks


def embedding_tasks(dataset: str):
    dataset = base_of(dataset)
    if dataset in ("broderick", "megmasc", "nieuwland", "szewczyk", "podcast"):
        # Every registry model with a hidden-state cache (encoding-score files are not needed).
        if dataset == "podcast":
            from scripts.analysis.podcast_embedding_rsa import _cache_for as cache_for_podcast
            cache_for = lambda ds, m, r: cache_for_podcast(m, r)  # noqa: E731
        else:
            from scripts.analysis.word_neural_embedding_rsa import _cache_for as cache_for
        found = []
        for family in LPP_FAMILY_NAMES:
            for task in flatten_family_tasks(family):
                try:
                    cache = cache_for(dataset, task.model_id, task.revision)
                except ValueError as e:
                    print(f"WARNING ambiguous cache, model skipped: {e}", flush=True)
                    continue
                if cache is not None:
                    found.append((family, task.model_id, task.revision, {"path": cache}))
        return _keep(found)
    if dataset == "pereira":
        from scripts.analysis import pereira_rsa as pr
        return _keep([(f, m, r, {"load_model_id": lm})
                      for f, m, lm, r in pr._tasks(ROOT / "outputs_jeanzay/pereira/embedding_sweep")])
    from scripts.analysis import other_neural_rsa as o
    return _keep([(f, m, r, {}) for f, m, r in o._tasks(dataset, ROOT / "outputs_jeanzay" / dataset / "embedding_sweep")])


def surprisal_tasks(dataset: str):
    from scripts.analysis.surprisal_rsa import _tasks
    dataset = base_of(dataset)
    return _keep([(f, m, r, {}) for f, m, r in _tasks(dataset, ("",))])


# ----------------------------------------------------------------------------- summary
def summarize(dataset: str, track: str, out_root: Path):
    paths = list((out_root / dataset / track).glob("*/*/rdm_regression_*.csv"))
    if not paths:
        print(f"{dataset}/{track}: no results to summarize", flush=True)
        return
    df = pd.concat([pd.read_csv(p, dtype={"subject": str}) for p in paths], ignore_index=True)
    keys = ["dataset", "track", "family", "model_id", "revision", "confounds"]
    per_layer = (df.groupby(keys + ["layer"], as_index=False)
                 .agg(delta_ll_per_obs=("delta_ll_per_obs", "mean"), rsa_pearson=("rsa_pearson", "mean"),
                      n_subjects=("subject", "nunique"), frac_pairs_tested=("frac_pairs_tested", "mean")))
    best = per_layer.loc[per_layer.groupby(keys).delta_ll_per_obs.idxmax()].rename(
        columns={"layer": "best_layer", "delta_ll_per_obs": "best_delta_ll_per_obs",
                 "rsa_pearson": "best_layer_rsa_pearson"})
    last = per_layer.loc[per_layer.groupby(keys).layer.idxmax(), keys + ["delta_ll_per_obs"]].rename(
        columns={"delta_ll_per_obs": "last_layer_delta_ll_per_obs"})
    summary = best.merge(last, on=keys)
    meta = [lookup_model_meta(m, f) for m, f in zip(summary.model_id, summary.family)]
    summary["group"] = [x.group for x in meta]
    summary["n_params"] = [x.n_params for x in meta]
    path = out_root / dataset / track / "model_summary.csv"
    summary.sort_values(keys).to_csv(path, index=False)
    print(f"{dataset}/{track}: {len(summary)} rows -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=NEURAL)
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", metavar="I/N")
    ap.add_argument("--n-repeats", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--build-eligible", action="store_true",
                    help="write the all-model surprisal-covered stimulus set and exit")
    args = ap.parse_args()
    out_root = args.out_root or (Path(os.environ["SCRATCH"]) / "rdm_regression" if os.environ.get("SCRATCH") else None)
    if out_root is None:
        ap.error("--out-root is required when $SCRATCH is unset")
    if ROOT / "outputs" in out_root.resolve().parents or out_root.resolve() == ROOT / "outputs":
        ap.error("refusing to write under outputs/; use a scratch/temporary root")
    if args.build_eligible:
        build_eligible(args.dataset, out_root)
        return
    shard = None
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            ap.error("--shard I/N needs 0 <= I < N")
        shard = (i, n)
    tracks = ["embedding", "surprisal"] if args.track == "both" else [args.track]
    failures = 0
    if not args.summarize_only:
        prepared = prepare(args.dataset, args.n_repeats, args.seed, out_root)
        print(f"{args.dataset}: {len(prepared['units'])} subjects, {len(prepared['universe'])} distinct items, "
              f"{sum(u['n_pairs'] for u in prepared['units'])} pairs in total", flush=True)
        for track in tracks:
            tasks = embedding_tasks(args.dataset) if track == "embedding" else surprisal_tasks(args.dataset)
            if args.limit:
                tasks = tasks[:args.limit]
            if shard:
                tasks = tasks[shard[0]::shard[1]]
            fn = score_embedding if track == "embedding" else score_surprisal
            print(f"{track}: {len(tasks)} models", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(fn, t, args.dataset, prepared, out_root): t for t in tasks}
                for fut in as_completed(futures):
                    try:
                        print(fut.result(), flush=True)
                    except Exception:
                        failures += 1
                        t = futures[fut]
                        print(f"ERROR {t[1]}@{t[2]}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    if not shard:
        for track in tracks:
            summarize(args.dataset, track, out_root)
    if failures:
        sys.exit(f"{failures} model(s) failed; see ERROR lines above")


if __name__ == "__main__":
    main()
