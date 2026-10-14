"""Embedding side of the RDM-regression track for GECO, OneStop, Natural Stories,
UCL Reading and ZuCo Reading (Provo/MECO live in rdm_regression_behavioral.py).

Called from rdm_regression_behavioral.py; not a CLI. Reuses the cache discovery,
word-to-row mapping and response tables of additional_behavioral_embedding_rsa.py so
the stimulus alignment is the one the existing benchmark uses. Differences:

* every layer is scored, not just the last one;
* the item set and pair sample come from the shared behavior RDM (the same one the
  surprisal track and the confound RDMs use), not from the embedding RSA cache;
* hidden states are gathered a chunk of layers at a time under a memory budget
  (RDM_CHUNK_GIB, default 10 GiB per worker), since n_items x n_layers x dim does
  not fit in memory for the largest models on the 40k-word datasets.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from scripts.analysis import additional_behavioral_embedding_rsa as add
from scripts.analysis import rdm_regression_behavioral as base

STREAM_MEMBER_BYTES = 4 * 1024**3
CHUNK_BYTES = int(float(os.environ.get("RDM_CHUNK_GIB", "10")) * 1024**3)


def prepare_context(dataset: str, measures: tuple[str, ...]) -> dict:
    """Word -> cache-row maps. Refuses to rebuild missing response tables (that would
    write under outputs/); they already exist for every dataset handled here."""
    for m in measures:
        table = add.ROOT / "outputs" / dataset / "pearson_rsa" / "embedding" / "response_table" / f"{m.lower()}.parquet"
        if not table.is_file():
            raise FileNotFoundError(f"Missing response table {table}; create it with "
                                    "additional_behavioral_embedding_rsa.py first")
    items = {m: add._response_items(dataset, m) for m in measures}
    return {"index_map": add._feature_index_map(dataset, items)}


def embedding_tasks(dataset: str, measures: tuple[str, ...]):
    return add._tasks(dataset, tuple(measures))


@contextmanager
def _open_features(path: Path):
    """Yield the (n_words, n_layers, dim) array; huge members are streamed to local
    disk and memory-mapped instead of decompressed into RAM."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("features.npy")
        big = info.file_size > STREAM_MEMBER_BYTES
        if big:
            fd, tmp = tempfile.mkstemp(prefix="rdm_regression_", suffix=".npy")
            os.close(fd)
            with archive.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024**2)
    if not big:
        with np.load(path) as archive:
            yield archive["features"]
        return
    try:
        yield np.load(tmp, mmap_mode="r")
    finally:
        Path(tmp).unlink(missing_ok=True)


def _as_3d(features: np.ndarray, path: Path) -> np.ndarray:
    if features.ndim == 2 and path.name.endswith("__cbrrnn.npz"):
        return features[:, None, :]
    if features.ndim != 3:
        raise ValueError(f"Invalid embedding cache shape {features.shape}: {path}")
    return features


def _gather(paths: dict, u_sec: np.ndarray, u_row: np.ndarray, l0: int, l1: int,
            n_layers: int, dim: int) -> np.ndarray:
    out = np.empty((len(u_sec), l1 - l0, dim), dtype=np.float32)
    for sec in np.unique(u_sec):
        sel = np.flatnonzero(u_sec == sec)
        path = paths[int(sec)]
        with _open_features(path) as raw:
            feats = _as_3d(raw, path)
            if feats.shape[1] != n_layers or feats.shape[2] != dim:
                raise ValueError(f"Inconsistent feature shape {feats.shape} in {path}")
            rows = u_row[sel]
            if rows.max() >= feats.shape[0]:
                raise ValueError(f"Item row {rows.max()} outside cached section of {path}")
            out[sel] = feats[rows, l0:l1, :]
    return out


def score_embedding(task, dataset, measures, prepared, out_root):
    family, model_id, revision, available = task
    out_task = (family, model_id, revision)
    todo = [m for m in measures if m in available
            and not base._out_path(out_root, dataset, "embedding", m, out_task).is_file()]
    if not todo:
        return f"existing {model_id}@{revision}"
    index_map = prepared["_ctx"]["index_map"]
    layout, skipped = {}, []
    for m in todo:
        ids = prepared[m]["ids"]
        rows = np.asarray([index_map[m].get((int(s), int(z)), -1) for s, z in ids], dtype=np.int64)
        if (rows < 0).any():
            raise ValueError(f"{dataset}/{m}: {int((rows < 0).sum())} response words have no cache row "
                             f"(e.g. {tuple(ids[int(np.flatnonzero(rows < 0)[0])])})")
        sec = np.zeros(len(ids), dtype=np.int64) if dataset == "zuco_reading" else ids[:, 0].astype(np.int64)
        missing = set(sec.tolist()) - set(available[m])
        if missing:
            # Different item sets across models would make the scores incomparable.
            skipped.append(f"{m}: missing {len(missing)} of {len(set(sec.tolist()))} sections")
            continue
        layout[m] = (sec, rows)
    if not layout:
        return f"skipped {model_id}@{revision}: " + "; ".join(skipped)
    # Measures that read the very same cache files share one gather pass.
    groups: dict[tuple, list[str]] = {}
    for m in layout:
        used = sorted(set(layout[m][0].tolist()))
        groups.setdefault(tuple((s, str(available[m][s])) for s in used), []).append(m)
    n_layers_seen = None
    for key, group in groups.items():
        paths = {s: Path(p) for s, p in key}
        keys = [layout[m][0] * (1 << 40) + layout[m][1] for m in group]
        uniq, inverse = np.unique(np.concatenate(keys), return_inverse=True)
        u_sec, u_row = uniq >> 40, uniq & ((1 << 40) - 1)
        splits = np.split(inverse, np.cumsum([len(k) for k in keys])[:-1])
        first = paths[int(u_sec[0])]
        with _open_features(first) as raw:
            shape = _as_3d(raw, first).shape
        n_layers, dim = shape[1], shape[2]
        n_layers_seen = n_layers
        chunk = max(1, min(n_layers, CHUNK_BYTES // (len(uniq) * dim * 4)))
        rows_by_measure = {m: [] for m in group}
        for l0 in range(n_layers - 1 if base.LAST_LAYER_ONLY else 0, n_layers, chunk):
            l1 = min(n_layers, l0 + chunk)
            block = _gather(paths, u_sec, u_row, l0, l1, n_layers, dim)
            for m, u_m in zip(group, splits):
                meas = prepared[m]
                pairs = base._pair_index(meas)
                for j in range(l1 - l0):
                    rdm = base._layer_rdm(block[u_m, j, :], pairs)
                    rows_by_measure[m] += base._rows(meas, m, dataset, "embedding", out_task, l0 + j, rdm)
            del block
        for m in group:
            base._write(base._out_path(out_root, dataset, "embedding", m, out_task), rows_by_measure[m])
    note = f"; skipped {'; '.join(skipped)}" if skipped else ""
    return f"scored {model_id}@{revision} ({n_layers_seen} layers){note}"
