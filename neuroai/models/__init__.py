"""``neuroai.models`` public API, loaded lazily.

``.embedding`` pulls in NeuralSet's audio extractors -> torch/mne/sklearn (a
multi-second import chain) purely as a side effect of module structure, not
because anything here needs it. Since Python always runs a package's
``__init__.py`` before importing any of its submodules, this used to make
even ``from neuroai.models.registry import flatten_family_tasks`` -- a pure
metadata lookup needed by scripts/submit.py for every array-size computation
-- pay that cost on every single CLI invocation (confirmed via `-X
importtime`: ~5s in `torch._C` alone, ~20-28s total, worse under login-node
load). Fixed 2026-09-23 by deferring the actual import to first attribute
access (PEP 562 module ``__getattr__``) -- the public names below still work
exactly as before, just imported on demand instead of eagerly.
"""

from __future__ import annotations

__all__ = [
    "LayerwiseEmbeddingConfig",
    "make_embedding_extractor",
    "CheckpointHuggingFaceText",
    "PYTHIA_MODELS",
    "PYTHIA_CHECKPOINT_STEPS",
    "GPT2_MODELS",
    "pythia_checkpoint_revision",
    "compute_surprisal",
    "batch_surprisal",
    "batch_surprisal_mean_two_words",
    "batch_surprisal_one_model",
]

_FROM_EMBEDDING = {
    "LayerwiseEmbeddingConfig",
    "make_embedding_extractor",
    "CheckpointHuggingFaceText",
    "PYTHIA_MODELS",
    "PYTHIA_CHECKPOINT_STEPS",
    "GPT2_MODELS",
    "pythia_checkpoint_revision",
}
_FROM_SURPRISAL = {
    "batch_surprisal",
    "batch_surprisal_mean_two_words",
    "batch_surprisal_one_model",
    "compute_surprisal",
}


def __getattr__(name: str):
    if name in _FROM_EMBEDDING:
        from . import embedding
        return getattr(embedding, name)
    if name in _FROM_SURPRISAL:
        from . import surprisal
        return getattr(surprisal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
