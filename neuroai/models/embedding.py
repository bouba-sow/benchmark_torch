"""Embedding configuration helpers for layerwise LLM analysis.

Wraps NeuralSet's ``HuggingFaceText`` extractor with a simpler interface
focused on scaling-law analyses: layer-wise hidden states, multiple model
families, and YAML-driven configuration.

For training-checkpoint sweeps (e.g. Pythia step512, step8000 …) use
``CheckpointHuggingFaceText`` which adds ``revision`` support on top of the
base ``HuggingFaceText`` extractor.
"""

from __future__ import annotations

import re
import typing as tp

from neuralset.extractors.text import HuggingFaceText

# OLMo 2 pretraining checkpoints on Hugging Face are git branches like
# ``stage1-step20000-tokens42B`` (not Pythia-style ``step512``). The
# ``tokens{T}B`` suffix is **billions of tokens**; the optimizer step ``N``
# differs by model size, so the same nominal token budget may use different
# branch names on 1B vs 7B vs 13B repos.
OLMO2_STAGE1_REVISION_RE = re.compile(r"^stage1-step(\d+)-tokens(\d+)B$")


def olmo2_stage1_metrics(revision: str) -> tp.Optional[tuple[int, int]]:
    """Parse a stage-1 OLMo2 HF branch name.

    Returns
    -------
    (optimizer_step, tokens_seen) or None if the string does not match.

    ``tokens_seen`` is ``tokens_b * 10**9`` (integer, approximate).
    """
    m = OLMO2_STAGE1_REVISION_RE.match(revision.strip())
    if not m:
        return None
    opt = int(m.group(1))
    tokens_b = int(m.group(2))
    return opt, int(tokens_b * 1_000_000_000)


# ---------------------------------------------------------------------------
# Checkpoint-aware extractor
# ---------------------------------------------------------------------------

class CheckpointHuggingFaceText(HuggingFaceText):
    """``HuggingFaceText`` extended with a ``revision`` field.

    Allows loading any HuggingFace model at a specific training checkpoint
    by passing a ``revision`` string (branch / tag / commit SHA) to
    ``from_pretrained``.  Useful for Pythia, whose 154 training-step
    checkpoints are published under ``revision="stepXXXXX"`` at
    ``EleutherAI/pythia-*``.

    Parameters
    ----------
    revision : str
        HuggingFace branch, tag, or commit SHA.
        Examples: ``"step512"``, ``"step16000"``, ``"main"`` (default).

    Notes
    -----
    * The **tokenizer** is always loaded from the ``"main"`` revision because
      the Pythia tokenizer is identical across all training checkpoints.
    * ``revision`` is automatically included in the NeuralSet cache key so
      different checkpoints never collide in the embedding cache.
    * ``repo_exists`` is overridden to always return True so NeuralSet does not
      reject ``*-deduped`` / OLMo-style ids missing from its bundled whitelist or
      call the Hub on offline GPU nodes. Loading is still validated by
      ``transformers`` against ``HF_HUB_CACHE``.
    """

    revision: str = "main"

    def repo_exists(self) -> bool:  # type: ignore[override]
        return True

    def _load_model(self, **kwargs: tp.Any):  # type: ignore[override]
        """Forward ``revision`` to ``from_pretrained``."""
        return super()._load_model(revision=self.revision, **kwargs)


# ---------------------------------------------------------------------------
# Model family registries
# ---------------------------------------------------------------------------

# Pythia size-scaling family (full model releases)
PYTHIA_MODELS = [
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]

# Pythia training-step checkpoints published at EleutherAI/pythia-*
# (log-spaced subset — 20 checkpoints from step0 to step143000)
PYTHIA_CHECKPOINT_STEPS: list[int] = [
    0, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000,
]

def pythia_checkpoint_revision(step: int) -> str:
    """Return the HF revision string for a Pythia training step."""
    return f"step{step}"


# ---------------------------------------------------------------------------
# OLMo 2 — example revision lists (verify on HF; names vary by checkpoint repo)
# ---------------------------------------------------------------------------

# Log-spaced **stage-1** milestones + final stage-1 branch (see model cards / ``list_repo_refs``).
OLMO2_0425_1B_DEFAULT_REVISIONS: list[str] = [
    "stage1-step0-tokens0B",
    "stage1-step300-tokens1B",
    "stage1-step10000-tokens21B",
    "stage1-step20000-tokens42B",
    "stage1-step40000-tokens84B",
    "stage1-step80000-tokens168B",
    "stage1-step200000-tokens420B",
    "stage1-step400000-tokens839B",
    "stage1-step1000000-tokens2098B",
    "stage1-step1907359-tokens4001B",
]

# 7B repo has no ``step0``; first public checkpoint is ``stage1-step150-tokens1B``.
OLMO2_1124_7B_DEFAULT_REVISIONS: list[str] = [
    "stage1-step150-tokens1B",
    "stage1-step2000-tokens9B",
    "stage1-step10000-tokens42B",
    "stage1-step20000-tokens84B",
    "stage1-step40000-tokens168B",
    "stage1-step99000-tokens416B",
    "stage1-step200000-tokens839B",
    "stage1-step500000-tokens2098B",
    "stage1-step928646-tokens3896B",
]

OLMO2_1124_13B_DEFAULT_REVISIONS: list[str] = [
    "stage1-step0-tokens0B",
    "stage1-step1000-tokens9B",
    "stage1-step5000-tokens42B",
    "stage1-step10000-tokens84B",
    "stage1-step20000-tokens168B",
    "stage1-step50000-tokens420B",
    "stage1-step100000-tokens839B",
    "stage1-step250000-tokens2098B",
    "stage1-step596057-tokens5001B",
]

# GPT-2 size-scaling family
GPT2_MODELS = ["openai-community/gpt2", "openai-community/gpt2-medium",
               "openai-community/gpt2-large", "openai-community/gpt2-xl"]


def make_embedding_extractor(
    model_name: str,
    *,
    cache_dir: str | None = None,
    device: str = "auto",
    token_aggregation: str = "mean",
    layer_aggregation: tp.Literal["mean", "sum"] | None = None,
    cache_n_layers: int | None = None,
    contextualized: bool = True,
    batch_size: int = 16,
) -> HuggingFaceText:
    """Return a configured ``HuggingFaceText`` extractor.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"EleutherAI/pythia-160m"``).
    cache_dir : str, optional
        Folder for caching embeddings. Defaults to ``.cache/embeddings/`` if
        the caller sets it, or NeuralSet's default (no caching) otherwise.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    token_aggregation : str
        How to pool across tokens within a word: ``"mean"``, ``"last"``, etc.
    layer_aggregation : str or None
        ``None`` → return per-layer activations of shape ``(n_layers, emb_dim)``.
        ``"mean"`` → average across all layers → ``(emb_dim,)``.
    cache_n_layers : int or None
        Number of layers to cache. Set to the model's total layer count to
        preserve all layers for post-hoc layer selection.
        ``None`` → cache only the aggregated output.
    contextualized : bool
        Use preceding word context (requires ``context`` column on events).
    batch_size : int
        Token batch size for HuggingFace inference.
    """
    infra_kwargs: dict[str, tp.Any] = {}
    if cache_dir:
        infra_kwargs["folder"] = cache_dir

    from exca import MapInfra

    return HuggingFaceText(
        model_name=model_name,
        device=device,
        contextualized=contextualized,
        token_aggregation=token_aggregation,
        layer_aggregation=layer_aggregation,
        cache_n_layers=cache_n_layers,
        batch_size=batch_size,
        infra=MapInfra(**infra_kwargs) if infra_kwargs else MapInfra(),
    )


class LayerwiseEmbeddingConfig(tp.NamedTuple):
    """Minimal config bundle for one model's layerwise embedding extraction."""

    model_name: str
    n_layers: int  # total transformer layers (including embedding)
    emb_dim: int
    family: str  # e.g. "pythia", "gpt2"

    def make_extractor(
        self,
        cache_dir: str | None = None,
        device: str = "auto",
        token_aggregation: str = "mean",
        batch_size: int = 16,
        contextualized: bool = True,
    ) -> HuggingFaceText:
        """Create extractor that caches all layers for layer-wise regression."""
        return make_embedding_extractor(
            self.model_name,
            cache_dir=cache_dir,
            device=device,
            token_aggregation=token_aggregation,
            layer_aggregation=None,  # keep all layers
            cache_n_layers=self.n_layers,
            contextualized=contextualized,
            batch_size=batch_size,
        )
