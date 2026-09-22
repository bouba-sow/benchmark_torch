"""Per-layer, mean-pooled-over-words hidden states for sentence-level fMRI
encoding (Pereira — see ``neuroai/benchmarks/pereira_embed.py``).

Unlike ``logit_lens.py`` (which reuses the model's own final-norm + unembed
to reconstruct a next-token distribution from EACH layer's hidden state),
this module only needs the raw per-layer hidden states themselves — no
final_norm resolution, no unembed matmul, no logits/softmax — so it is
simpler and applies uniformly to every architecture ``_forward_hidden_states``
already supports (native ``output_hidden_states=True`` path, or the
forward-hook fallback for ``trust_remote_code`` families that ignore it),
with no ``native_only``/GPT-BERT/moep special-casing needed.

Pooling convention: mean-pooled per-layer hidden state ACROSS ALL WORDS in
the sentence, following de Varda, Berzak, Fedorenko & Levy (2026,
bioRxiv 2026.08.21.746238v1) — the paper that runs this exact
embedding-encoding analysis on Pereira2018: "sentence-aggregated ... EMB
regressors were obtained by averaging over word-level predictors." Each
word's own hidden state is itself the MEAN across that word's constituent
sub-word tokens (same convention as ``logit_lens.py``'s surprisal
aggregation, and ``neuroai.features.surprisal``'s).

Same full-text sliding-window chunking / char-span word-alignment scheme as
``logit_lens_surprisal_all_layers`` — for Pereira's short (<20-word)
sentences this always resolves to exactly one chunk, but the general
machinery is reused as-is (not reimplemented) to stay correct for any
sentence length and match this repo's established convention.
"""

from __future__ import annotations

import logging
import typing as tp
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .llm import (
    _ensure_pad_token_for_batches,
    _forward_hidden_states,
    _fulltext_feature_cache_path,
    _get_cached_causal_lm,
    _load_cached_features,
    _model_input_device,
    _save_cached_features,
)
from .logit_lens import _exclude_embedding_layer

logger = logging.getLogger(__name__)


def mean_pooled_hidden_states_all_layers(
    words: list[str],
    model_name: str,
    *,
    revision: str | None = None,
    device: str = "cpu",
    max_length: int | None = None,
    stride_delta: int = 64,
    tokenizer_no_revision: bool = True,
    cache_dir: str | Path | None = None,
    section_id: int = 0,
    lang: str = "en",
    n_params: float | None = None,
    trust_remote_code: bool = False,
    untrained: bool = False,
    untrained_seed: int = 0,
    require_cache_hit: bool = False,
    dtype: "torch.dtype | str | None" = None,
) -> np.ndarray:
    """Per-layer hidden state, mean-pooled across every word in ``words``.

    Returns
    -------
    np.ndarray
        Shape ``(n_layers, d_model)``, float32. Layer 0 is the first
        transformer block's output — the embedding layer (``hidden_states[0]``)
        is excluded, same convention as ``logit_lens_surprisal_all_layers``.
    """
    from transformers import AutoConfig

    rev_norm = revision if revision not in (None, "", "main") else None
    cf_kw: dict[str, tp.Any] = {"revision": rev_norm} if rev_norm is not None else {}
    if trust_remote_code:
        cf_kw["trust_remote_code"] = True
    try:
        lm_cfg = AutoConfig.from_pretrained(model_name, **cf_kw)
    except Exception:
        lm_cfg = None

    if max_length is not None:
        eff_max_length: int = max_length
    else:
        cfg_max = getattr(lm_cfg, "max_position_embeddings", None) if lm_cfg is not None else None
        candidates = [v for v in (cfg_max,) if isinstance(v, int) and 0 < v < 1_000_000]
        eff_max_length = min(candidates) if candidates else 2048
        eff_max_length = min(eff_max_length, 32768)

    stride = max(1, eff_max_length - stride_delta)

    # --- cache check (before loading model/tokenizer) ---
    # Distinct layer_index tag ("hidden_mean_all") from logit-lens's
    # ("logitlens_all") and the plain embeddings cache ("all"/int) so none
    # of the three ever collide even sharing cache_dir.
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _fulltext_feature_cache_path(
            cache_dir, model_name, revision, lang, section_id,
            eff_max_length, stride_delta, "hidden_mean_all",
            untrained=untrained, untrained_seed=untrained_seed,
        )
        cached = _load_cached_features(cache_path)
        if cached is not None:
            logger.info(
                "Mean-pooled hidden states: loaded from cache %s (shape %s)",
                cache_path.resolve(), cached.shape,
            )
            return cached

    if require_cache_hit:
        raise RuntimeError(
            f"hidden-states: cache miss for {model_name}@{revision or 'main'} "
            f"section={section_id} (cache_dir={cache_dir!r}, expected path={cache_path}) — "
            "refusing to load the model with require_cache_hit=True. Run the GPU "
            "prefetch stage for this model/checkpoint first."
        )

    load_kw: dict[str, tp.Any] = {"output_hidden_states": True, "trust_remote_code": trust_remote_code}
    if dtype is not None:
        load_kw["torch_dtype"] = dtype
    elif trust_remote_code and torch.cuda.is_available():
        # Same sm_80+ bf16-Triton-kernel guard as logit_lens.py — see that
        # module's identical block for the full rationale.
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            load_kw["torch_dtype"] = torch.float32
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=revision,
        device=device,
        load_kw=load_kw,
        tokenizer_no_revision=tokenizer_no_revision,
        n_params=n_params,
        untrained=untrained,
        untrained_seed=untrained_seed,
    )
    input_device = _model_input_device(model)

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "mean_pooled_hidden_states_all_layers requires a fast tokenizer "
            f"(return_offsets_mapping); got {type(tokenizer).__name__}."
        )
    _ensure_pad_token_for_batches(tokenizer)

    # --- build full text and exact char spans for each word (same as logit_lens.py) ---
    full_text = " ".join(words)
    word_char_spans: list[tuple[int, int]] = []
    pos = 0
    for w in words:
        word_char_spans.append((pos, pos + len(w)))
        pos += len(w) + 1

    overflow_enc = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=eff_max_length,
        return_overflowing_tokens=True,
        stride=stride,
        padding=True,
    )
    om_np = overflow_enc["offset_mapping"].numpy()
    n_chunks = int(overflow_enc["input_ids"].shape[0])
    seq_len = int(overflow_enc["input_ids"].shape[1])

    logger.info(
        "Hidden-states fulltext [%s%s]: %d words → %d chunks × %d tokens (stride_delta=%d)",
        model_name, f"@{revision}" if revision else "",
        len(words), n_chunks, eff_max_length, stride_delta,
    )

    # --- assign each word to the first chunk where it enters the non-overlap region ---
    new_region_start = [0] + [stride] * (n_chunks - 1)
    word_assignments: list[tuple[int, list[int]]] = []
    for wi, (w_start, w_end) in enumerate(word_char_spans):
        assigned = False
        for k in range(n_chunks):
            ns = new_region_start[k]
            t_starts = om_np[k, ns:, 0]
            t_ends = om_np[k, ns:, 1]
            mask = (t_starts < w_end) & (t_ends > w_start)
            idx = np.where(mask)[0]
            if idx.size > 0:
                word_assignments.append((k, (idx + ns).tolist()))
                assigned = True
                break
        if not assigned:
            logger.warning(
                "hidden-states: no token found for word %d %r — using last chunk end",
                wi, words[wi],
            )
            word_assignments.append((n_chunks - 1, [seq_len - 1]))

    chunk_to_words: dict[int, list[int]] = defaultdict(list)
    for wi, (k, _) in enumerate(word_assignments):
        chunk_to_words[k].append(wi)

    n_layers: int | None = None
    d_model: int | None = None
    word_vecs: list[np.ndarray | None] = [None] * len(words)
    # Set once (first chunk) and reused for every subsequent chunk of this
    # sentence — some architectures' residual-stream width changes
    # mid-network (e.g. the BabyLM MoE model Jtapsa/moep: 384 -> 192 -> 384;
    # OPT-350m has an analogous embedding/hidden-size mismatch), so
    # np.stack-ing every layer's hidden state raises "all input arrays must
    # have the same shape". Same fallback semantics as logit_lens.py's
    # ``native_only``: degrade to the LAST layer only (guaranteed present
    # and correctly-shaped — it's literally ``last_hidden_state``) rather
    # than hard-failing the whole model.
    native_only: bool | None = None

    with torch.no_grad():
        for k in sorted(chunk_to_words.keys()):
            ids_k = overflow_enc["input_ids"][k : k + 1].to(input_device)
            am_k = overflow_enc["attention_mask"][k : k + 1].to(input_device)

            hs = _forward_hidden_states(model, ids_k, am_k, model_name=model_name)
            # Exclude the embedding layer (index 0) — same convention as
            # logit_lens.py's per-layer surprisal.
            layer_hs = _exclude_embedding_layer(hs, model)

            if native_only is None:
                shapes = {tuple(h.shape) for h in layer_hs}
                native_only = len(shapes) > 1
                if native_only:
                    logger.warning(
                        "%s: per-layer hidden states have inconsistent shapes "
                        "(%s) — falling back to the final layer only (no "
                        "per-earlier-layer embeddings for this architecture).",
                        model_name, sorted(shapes),
                    )
            if native_only:
                layer_hs = layer_hs[-1:]

            if n_layers is None:
                n_layers = len(layer_hs)

            # (n_layers, seq_len, D) for this chunk, float32.
            chunk_arr = np.stack(
                [h[0].float().cpu().numpy() for h in layer_hs], axis=0,
            )
            if d_model is None:
                d_model = chunk_arr.shape[-1]

            for wi in chunk_to_words[k]:
                _, tok_idx = word_assignments[wi]
                # Mean across this word's constituent sub-word tokens, per
                # layer — same "mean across sub-word tokens" convention as
                # logit_lens.py / neuroai.features.surprisal.
                word_vecs[wi] = chunk_arr[:, tok_idx, :].mean(axis=1)  # (n_layers, D)

            del hs, layer_hs, chunk_arr

    assert n_layers is not None and d_model is not None
    missing = [wi for wi, v in enumerate(word_vecs) if v is None]
    if missing:
        raise RuntimeError(f"hidden-states: {len(missing)} word(s) never assigned a vector: {missing}")

    # Mean across ALL WORDS in the sentence — de Varda et al.'s convention
    # (see module docstring).
    stacked = np.stack(word_vecs, axis=0)  # (n_words, n_layers, D)
    pooled = stacked.mean(axis=0).astype(np.float32)  # (n_layers, D)

    if cache_path is not None:
        _save_cached_features(cache_path, pooled)

    return pooled
