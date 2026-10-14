"""Corpus-based feature standardization (Timkey & van Schijndel 2021, arXiv 2109.04404).

z = (x - mu) / sigma with mu, sigma computed once per model over a fixed English-Wikipedia reference corpus
(``scripts/features/build_reference_corpus.py``, ``scripts/features/reference_stats.py``), instead of over the scored
stimuli of each dataset. Stats live in ``$SCRATCH/reference_stats/<family>/<model_dir>/<revision>.npz``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("REFERENCE_STATS_DIR", Path(os.environ.get("SCRATCH", ".")) / "reference_stats"))


def stats_path(family: str, model_id: str, revision: str | None = None) -> Path:
    rev = (revision or "main").replace("/", "_")
    return ROOT / family / model_id.replace("/", "_") / f"{rev}.npz"


def load_reference_stats(family: str, model_id: str, revision: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    p = stats_path(family, model_id, revision)
    if not p.is_file():
        raise FileNotFoundError(f"no reference-corpus stats for {family}/{model_id}@{revision or 'main'}: {p}")
    with np.load(p) as z:
        return z["mu"].astype(np.float32), z["sigma"].astype(np.float32)


def standardize_reference(x: np.ndarray, ref: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """(x - mu) / sigma over the last axis, with the reference-corpus mu/sigma. Constant features (sigma ~ 0) become 0."""
    mu, sd = ref
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != mu.shape[0]:
        raise ValueError(f"feature dim {x.shape[-1]} != reference dim {mu.shape[0]}")
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (x - mu) / sd
