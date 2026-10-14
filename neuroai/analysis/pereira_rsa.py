"""Fit-free, last-layer Pearson RSA for Pereira2018."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


def load_last_layer(cache_dir: Path, model_id: str, revision: str, n_sentences: int) -> np.ndarray:
    """Read the existing sentence caches without loading or running a model."""
    safe_model = model_id.replace("/", "_")
    safe_rev = revision.replace("/", "_").replace("-", "_")
    if model_id in {"cclark/cbr-rnn", "cclark/cbr-rnn-m"}:
        path = cache_dir / f"{safe_model}__{safe_rev}__sentence_mean_cbrrnn.npz"
        with np.load(path) as archive:
            features = archive["features"]
            if features.ndim != 3 or features.shape[:2] != (n_sentences, 1) or not np.isfinite(features).all():
                raise ValueError(f"Invalid CBR-RNN sentence cache: {path}")
            return np.asarray(features[:, -1, :], dtype=np.float32)
    prefix = f"{safe_model}__{safe_rev}__en__sec"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)__fulltext_ml\d+__sdelta\d+__hidden_mean_all\.npz$")
    paths: dict[int, Path] = {}
    for path in cache_dir.glob(f"{prefix}*__hidden_mean_all.npz"):
        match = pattern.match(path.name)
        if match:
            index = int(match.group(1))
            if index in paths:
                raise ValueError(f"Duplicate cache for sentence {index}: {paths[index]} and {path}")
            paths[index] = path
    missing = sorted(set(range(n_sentences)) - paths.keys())
    if missing:
        raise FileNotFoundError(f"{model_id}@{revision}: {len(missing)} sentence caches missing; first: {missing[:5]}")
    rows = []
    for index in range(n_sentences):
        with np.load(paths[index]) as archive:
            features = archive["features"]
            if features.ndim != 2 or features.shape[0] == 0:
                raise ValueError(f"Invalid sentence features in {paths[index]}")
            rows.append(np.asarray(features[-1], dtype=np.float32))
    if len({row.shape for row in rows}) != 1:
        raise ValueError(f"Inconsistent last-layer widths for {model_id}@{revision}")
    states = np.stack(rows)
    if not np.isfinite(states).all():
        raise ValueError(f"Non-finite cached states for {model_id}@{revision}")
    return states


def pearson_rdm(patterns: np.ndarray) -> np.ndarray:
    """Square RDM: one minus Pearson correlation between stimulus patterns."""
    distances = pdist(patterns, metric="correlation")
    if not np.isfinite(distances).all():
        raise ValueError("Undefined Pearson distance: a stimulus pattern is constant or non-finite")
    return squareform(distances)


@dataclass(frozen=True)
class NeuralRdm:
    subject: str
    experiment: str
    stimulus_indices: np.ndarray
    distances: np.ndarray


def prepare_neural_rdms(
    all_item_ids: np.ndarray,
    subjects_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[NeuralRdm]:
    """Align each subject's voxel patterns to the cached sentence order."""
    item_to_index = {item: index for index, item in enumerate(all_item_ids)}
    prepared = []
    for subject, (bold, item_ids, _) in subjects_data.items():
        experiments = np.array([item.split(".")[0] for item in item_ids])
        for experiment in sorted(set(experiments)):
            selected = np.flatnonzero(experiments == experiment)
            if len(selected) < 3:
                continue
            prepared.append(NeuralRdm(
                subject=subject, experiment=experiment,
                stimulus_indices=np.array([item_to_index[item] for item in item_ids[selected]]),
                distances=pearson_rdm(bold[selected]),
            ))
    return prepared


def score_last_layer(
    states: np.ndarray,
    neural_rdms: list[NeuralRdm],
    *,
    model_id: str,
    revision: str,
) -> pd.DataFrame:
    """Pearson correlation between model and neural RDM upper triangles."""
    model_rdm = pearson_rdm(states)
    rows = []
    for neural in neural_rdms:
        indices = neural.stimulus_indices
        model_subset = model_rdm[np.ix_(indices, indices)]
        triangle = np.triu_indices(len(indices), k=1)
        rows.append({
            "model_id": model_id, "revision": revision,
            "subject": neural.subject, "experiment": neural.experiment,
            "n_stimuli": len(indices), "n_pairs": len(triangle[0]),
            "rsa_pearson": float(pearsonr(model_subset[triangle], neural.distances[triangle]).statistic),
        })
    return pd.DataFrame(rows)
