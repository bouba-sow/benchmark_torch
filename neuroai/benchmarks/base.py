"""Shared configuration dataclasses and BaseBenchmark ABC.

All benchmark classes (BehavioralBenchmark, IeegBenchmark, FmriBenchmark) share
the same DataConfig / ModelConfig / RegressionConfig building blocks so that a
single YAML config drives any modality.
"""

from __future__ import annotations

import abc
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..regressors.ridge import AlphaInput


def default_ridge_alphas() -> list[float]:
    """Default ridge penalty grid: 10^-1 … 10^4 (10 log-spaced candidates)."""
    return np.logspace(3, 9, 6).tolist()


def word_predictions_root(output_dir: "str | Path") -> Path:
    """Resolve the per-model root a word-predictions subdirectory should
    live under.

    Redirected under ``$SCRATCH`` when that env var is set, mirroring the
    ``outputs_jeanzay/...`` relative structure — these directories hold raw
    per-word/per-electrode OOF prediction arrays (large, uncompressible) and
    are the dominant disk consumer across all sweeps, unlike the tiny score
    CSVs that stay under ``output_dir`` on ``$WORK``/fswork. Falls back to
    ``output_dir`` directly (old behavior) if ``$SCRATCH`` isn't set.

    Jean Zay's fswork quota filled up entirely from this session's word-
    prediction backfills (2026-08-18) — this keeps the same failure from
    recurring, since $SCRATCH has ~100x the headroom.
    """
    import os

    output_dir = Path(output_dir)
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        return output_dir

    parts = output_dir.parts
    if "outputs_jeanzay" in parts:
        idx = parts.index("outputs_jeanzay")
        rel = Path(*parts[idx + 1:])
    else:
        rel = Path(output_dir.name)
    return Path(scratch) / "benchmark_word_predictions" / rel


def word_predictions_dir(output_dir: "str | Path", dirname: str) -> Path:
    """``word_predictions_root(output_dir) / dirname`` — the common case of
    wanting one named subdirectory (e.g. ``"podcast_word_predictions"``)
    under the resolved (possibly $SCRATCH-redirected) root."""
    return word_predictions_root(output_dir) / dirname


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset location and loading options."""
    bids_root: str
    subjects: list[str] | None = None
    tasks: list[str] | None = None
    cache_folder: str = ".cache/neuralset"


@dataclass
class ModelConfig:
    """LLM feature extractor settings."""
    model_name: str
    device: str = "auto"
    token_aggregation: str = "mean"
    batch_size: int = 16
    cache_n_layers: int | None = None
    contextualized: bool = True
    cache_folder: str | None = None
    # HuggingFace revision for training-checkpoint sweeps (e.g. Pythia "step8000").
    # None → load the default (latest) weights.
    model_revision: str | None = None
    # Approximate parameter count (enables device_map for large models, e.g. OPT-66B).
    n_params: float | None = None
    # Required for models with custom HuggingFace code (e.g. stickbreaking, alibi, forgetting).
    trust_remote_code: bool = False
    # If True, skip pretrained weight loading and use a randomly initialised
    # model of the same architecture (untrained baseline for brain scores).
    untrained: bool = False
    untrained_seed: int = 0


@dataclass
class RegressionConfig:
    """Ridge regression and cross-validation settings."""
    n_splits: int = 5
    alpha: AlphaInput = field(default_factory=default_ridge_alphas)
    scale: bool = True
    random_state: int = 0
    print_chosen_ridge_alpha: bool = False
    # Force the efficient closed-form GCV alpha search (cv=None, one SVD,
    # every candidate alpha evaluated ~for free) instead of the default
    # nested k-fold grid search (a full refit per (alpha, inner-fold) pair --
    # n_alphas * inner_splits separate solves). Opt-in, default False, so
    # every existing caller keeps its current alpha-selection behavior;
    # meant for wide-hidden-dim checkpoints (e.g. opt-66b, hidden=9216)
    # where the nested-CV cost made a single layer's fit take hours -- see
    # neuroai.regressors.ridge._fit_ridge_for_fold's fast_gcv docstring.
    fast_gcv: bool = False


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseBenchmark(abc.ABC):
    """Minimal contract every benchmark must satisfy.

    Concrete subclasses implement :meth:`run` which returns a modality-specific
    result object.  The :meth:`score` helper is optional — subclasses may expose
    richer result types that already carry all metrics.
    """

    @abc.abstractmethod
    def run(self) -> tp.Any:
        """Execute the full benchmark pipeline and return a result object."""

    def score(self) -> dict[str, float]:
        """Return a flat ``{metric: value}`` summary.

        Default implementation calls ``run()`` and delegates to ``result.score.to_dict()``.
        Override in subclasses that need a different summary format.
        """
        result = self.run()
        if hasattr(result, "score") and hasattr(result.score, "to_dict"):
            return result.score.to_dict()
        raise NotImplementedError(
            f"{type(self).__name__}.score() is not implemented; call run() directly."
        )
