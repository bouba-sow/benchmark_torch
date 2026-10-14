"""Logit-lens per-layer surprisal (Kuribayashi et al. 2025, arXiv:2502.01615).

For each transformer layer ``l`` (excluding the embedding layer), reuses the
model's OWN final LayerNorm and unembedding matrix to read out a next-token
distribution from that layer's hidden state — i.e. "pretend layer l's output
was already the last layer":

    p(w_t | w_<t; h_{l,t}) = softmax(W_U · LayerNorm(h_{l,t})) [id(w_t)]

No extra training is involved (that would be tuned-lens, not implemented
here — Kuribayashi et al.'s paper's simpler variant). Sub-word tokens are
aggregated per word via the SUM of their surprisals (the word's surprisal by the
chain rule; Kuribayashi et al. 2021, Eq. 2). Until the 2026-10-12 audit this module
used the MEAN, which divides by a tokenizer-dependent token count; ``subword_agg="mean"``
reproduces it as a sensitivity variant. The LAST layer is always the model's own output
logits (see ``SURPRISAL_CACHE_VERSION``).

Scope: standard HuggingFace causal-LM architectures (GPT-2/GPT-Neo,
GPT-NeoX/Pythia, Llama/Mistral/Qwen2-3/OLMo2-style, OPT, Mamba/Mamba2) plus
the FLA-based ``trust_remote_code`` families (stickbreaking, alibi,
forgetting, hard n-gram) — these need a CUDA-capable node just to import
(``fla``'s Triton kernels probe for an active GPU driver at import time), and
their custom ``forward()`` implementations ignore
``output_hidden_states=True``, so hidden states come from the
``_forward_hidden_states`` hook-based fallback instead of the native path
(see ``_exclude_embedding_layer`` below for why that distinction matters).
CBR-RNN (bypasses ``AutoModelForCausalLM``/``_get_cached_causal_lm`` entirely,
and is single-layer so per-layer surprisal isn't a meaningful concept for it)
remains fully out of scope.

Architectures with no resolvable final-norm path at all — e.g. the
BabyLM-community ``gpt-bert-*`` models (BERT-style MLM head: LayerNorm ->
Linear -> GELU -> LayerNorm -> Linear, nothing like a single pre-unembedding
norm+matmul) — degrade automatically to *native* final-layer-only surprisal:
the model's own ``.logits`` output for the last position, not a logit-lens
reconstruction. :func:`_find_final_norm` raising ValueError is what triggers
this fallback (see ``native_only`` in :func:`logit_lens_surprisal_all_layers`)
rather than a hard failure.

The BabyLM MoE model ``Jtapsa/moep`` is a different case: it resolves a
final-norm path (``model.norm``) just fine, so ``_find_final_norm`` never
raises — but its residual-stream width changes mid-network (384 → 192 → 384),
so applying that norm to an *intermediate* layer's differently-shaped hidden
state crashes during reconstruction (``RuntimeError`` from ``F.layer_norm``,
not the clean ``_find_final_norm`` ``ValueError`` GPT-BERT hits). Since this
failure is architectural (deterministic, not data-dependent), ``"moep" in
model_name.lower()`` forces the same native final-layer-only fallback as
GPT-BERT up front, rather than trying and reactively catching a RuntimeError
mid-reconstruction.
"""

from __future__ import annotations

import hashlib
import logging
import time
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
    config_dim_attr,
)

# Bump whenever the numerical definition of cached surprisal changes, so stale caches are
# never read back. v2 (2026-10-12 audit): word surprisal is the SUM over subword tokens,
# target tokens are located by character offsets (BOS/EOS-safe), the last layer is the
# model's own logits, and batches are checked against unbatched forwards.
SURPRISAL_CACHE_VERSION = "v2"
SubwordAgg = tp.Literal["sum", "mean"]


def _aggregate_subwords(vals: np.ndarray, how: str) -> np.ndarray:
    """Aggregate token surprisals (..., n_tokens) into word surprisal (...,).

    ``sum`` is the word's surprisal -log p(word | context) (chain rule); ``mean`` is kept
    only as a sensitivity variant. A word with any undefined token (no left context) is NaN.
    """
    if how == "sum":
        return vals.sum(axis=-1)
    if how == "mean":
        return vals.mean(axis=-1)
    raise ValueError(f"subword_agg must be 'sum' or 'mean', got {how!r}")


def _valid_span(attention_mask_row: torch.Tensor) -> tuple[int, int]:
    """[start, end) of the non-padded positions of one sequence (left or right padding)."""
    nz = torch.nonzero(attention_mask_row, as_tuple=False).flatten()
    if nz.numel() == 0:
        return 0, 0
    return int(nz[0]), int(nz[-1]) + 1

logger = logging.getLogger(__name__)

# Tried in order; the first fully-resolvable dotted path wins. Covers the
# "standard LLM" families in the registry (gpt2, gptneo, opt, pythia, olmo2,
# llama32, mistral, qwen3, mamba, mamba2, and the standard-architecture
# BabyLM models CLASS-IT/blm-gpt2s) — GPT-2/GPT-Neo, Llama/Mistral/Qwen2-3/
# OLMo2, and Mamba/Mamba2 (`MambaForCausalLM`/`Mamba2ForCausalLM`, verified
# against `state-spaces/mamba-130m-hf` / `AntonV/mamba2-130m-hf`) each share
# the same attribute layout as their respective HF base classes. Also covers
# the FLA-based trust_remote_code families (stickbreaking/alibi/forgetting/
# hard-n-gram share the Llama-style `model.norm`, verified on Jean Zay H100 — see
# outputs_jeanzay/_scratch_conf_method_test/logit_lens_constrained_models_handoff.md).
# The BabyLM MoE model (Jtapsa/moep) is NOT covered here despite also having
# a `model.norm`: its residual-stream width changes mid-network
# (384 -> 192 -> 384), so no single final-norm/unembedding pair is valid for
# every layer — out of scope, see `_find_final_norm`'s docstring above.
_FINAL_NORM_PATHS: tuple[str, ...] = (
    "transformer.ln_f",             # GPT-2, GPT-Neo
    "gpt_neox.final_layer_norm",    # GPT-NeoX / Pythia
    "model.norm",                   # Llama, Mistral, Qwen2/3, OLMo2, stickbreaking/alibi/hard-n-gram
    "model.decoder.final_layer_norm",  # OPT
    "backbone.norm_f",              # Mamba, Mamba2
)


def _resolve_attr_path(obj: tp.Any, path: str) -> tp.Any | None:
    for attr in path.split("."):
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def _exclude_embedding_layer(hs: tuple, model: tp.Any) -> tuple:
    """Drop the embedding-layer entry (index 0), only when it's actually there.

    The native ``output_hidden_states=True`` path returns embeddings + one
    entry per transformer block (``num_hidden_layers + 1`` total). The
    hook-based fallback in ``_forward_hidden_states`` (used for
    trust_remote_code architectures that ignore ``output_hidden_states``)
    returns only the per-block captures — no embedding entry to strip.
    Blindly slicing ``hs[1:]`` in that case would silently discard a real
    transformer layer instead of the embedding.
    """
    n_hidden_layers = config_dim_attr(
        model.config, ("num_hidden_layers", "n_layer", "n_layers"),
    )
    return hs[1:] if len(hs) == n_hidden_layers + 1 else hs


def _find_final_norm(model: tp.Any) -> torch.nn.Module:
    """Locate the model's final (pre-unembedding) LayerNorm/RMSNorm module."""
    for path in _FINAL_NORM_PATHS:
        found = _resolve_attr_path(model, path)
        if isinstance(found, torch.nn.Module):
            return found
    raise ValueError(
        f"logit-lens: could not find a final-norm module for {type(model).__name__} "
        f"(tried {_FINAL_NORM_PATHS}). This architecture isn't in the supported "
        "standard-LLM list for logit-lens surprisal."
    )


def logit_lens_surprisal_all_layers(
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
    native_only: bool = False,
    subword_agg: SubwordAgg = "sum",
) -> np.ndarray:
    """Per-layer logit-lens surprisal, one scalar per (word, layer).

    Word surprisal is the SUM of its subword-token surprisals (``subword_agg="mean"`` is
    a sensitivity variant). The last layer is read from the model's own logits.

    ``native_only=True`` forces the native final-layer-only path for ANY
    architecture (skips the per-earlier-layer ``final_norm`` + full-vocab
    unembed reconstruction entirely — the dominant cost for large corpora):
    the model's own ``.logits`` output is read directly, giving an
    ``(n_words, 1, 1)`` result whose single layer is the true final layer.
    The cache path is unchanged, so a downstream ``last_layer_only`` LME fit
    (which selects ``n_layers - 1``) reads it identically to a full cache's
    last column. Do NOT reuse a ``native_only`` cache for a full per-layer
    sweep — clear it first.

    Full-text sliding-window extraction (one forward pass per chunk, not one
    per word) — same chunking/word-alignment scheme as
    :func:`~neuroai.features.llm.contextual_word_embeddings_hf_fulltext`, so
    results are directly comparable across the two feature modes.

    Pass ``require_cache_hit=True`` to raise instead of silently falling
    back to a model load on a cache miss — the CPU-only stage of every
    two-stage GPU/CPU pooled-LME pipeline in this codebase needs this: a
    cache miss there isn't just "slow", it can be a hard crash for the
    FLA-based ``trust_remote_code`` families (stickbreaking/alibi/
    forgetting/hard-n-gram), whose Triton backend needs a live CUDA driver
    just to *import* — something a CPU-only SLURM node doesn't have. Without
    this check, a caller only testing "is a cache_dir configured" (not "is
    THIS item actually cached") would let that import crash reach the user
    as an inscrutable ``RuntimeError: 0 active drivers`` deep inside
    ``transformers``' dynamic module loader, instead of a clear message here.

    Returns
    -------
    np.ndarray
        Shape ``(n_words, n_layers, 1)``, float32. Layer 0 is the first
        transformer block's output — the embedding layer (``hidden_states[0]``)
        is excluded, per Kuribayashi et al. (2025).
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
    # Distinct layer_index tag ("logitlens_all") from the embeddings cache
    # ("all"/int) so the two never collide even sharing cache_dir.
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _fulltext_feature_cache_path(
            cache_dir, model_name, revision, lang, section_id,
            eff_max_length, stride_delta, f"logitlens_all_{SURPRISAL_CACHE_VERSION}_{subword_agg}",
            untrained=untrained, untrained_seed=untrained_seed,
        )
        cached = _load_cached_features(cache_path)
        if cached is not None:
            logger.info(
                "Logit-lens fulltext surprisal: loaded from cache %s (shape %s)",
                cache_path.resolve(), cached.shape,
            )
            return cached

    if require_cache_hit:
        raise RuntimeError(
            f"logit-lens: cache miss for {model_name}@{revision or 'main'} "
            f"section={section_id} (cache_dir={cache_dir!r}, expected path={cache_path}) — "
            "refusing to load the model with require_cache_hit=True. Run the GPU "
            "prefetch stage for this model/checkpoint first."
        )

    # CBR-RNN has a custom architecture not loadable via AutoModelForCausalLM
    # (see neuroai.features.llm.contextual_word_embeddings_hf_fulltext's
    # identically-gated branch for the embedding side of this same model).
    # It was never routed through this function at all -- meaning cbr-rnn
    # never got a standard "*_logitlens_all.npz" RSA surprisal cache, only
    # its own dataset-specific caches used internally by the LME fit (see
    # neuroai.features.cbr_rnn.contextual_word_surprisal_cbr_rnn). Reuse the
    # already-tested per-(context, target) surprisal helper from that module,
    # one call per word with a preceding-word sliding window (same
    # context_words convention as this model's embedding extraction), to
    # populate the SAME standard cache format every other architecture uses.
    if "cbr-rnn" in model_name.lower() or "cbr_rnn" in model_name.lower():
        from .cbr_rnn import _cbr_surprisal_one_pair, _get_cached_cbr_rnn

        tokenizer, model = _get_cached_cbr_rnn(model_name, device)
        context_words = 48
        surps = np.empty(len(words), dtype=np.float32)
        for i, word in enumerate(words):
            context = " ".join(words[max(0, i - context_words):i])
            surps[i] = _cbr_surprisal_one_pair(model, tokenizer, torch.device(device), context, word)[0]
        result = surps[:, np.newaxis, np.newaxis]  # (n_words, 1, 1) -- CBR-RNN has no layer dim
        if cache_path is not None:
            _save_cached_features(cache_path, result)
            logger.info("CBR-RNN logit-lens surprisal: cached → %s (shape %s)", cache_path.resolve(), result.shape)
        return result

    load_kw: dict[str, tp.Any] = {"output_hidden_states": True, "trust_remote_code": trust_remote_code}
    if dtype is not None:
        # Some trust_remote_code architectures' custom forward() isn't
        # bf16-safe (e.g. BabyLM-community's gpt-bert-* models: an in-place
        # buffer .set_() inside their custom DWA accumulator hard-codes
        # float32, so loading the model's own default bf16 weights crashes
        # with "Could not set tensor of type BFloat16 to a tensor of type
        # float" the first time the model actually runs). Caller opts in
        # explicitly per-model — this never changes behavior for the
        # standard families' default bf16 loading.
        load_kw["torch_dtype"] = dtype
    elif trust_remote_code and torch.cuda.is_available():
        # FLA-based families (stickbreaking/alibi/forgetting/hard-n-gram)
        # JIT-compile their own Triton kernels, some of which emit hardware
        # bf16 PTX instructions (`.bf16`/`cvt.bf16.f32`) that require
        # sm_80+ (Ampere or newer). On an older GPU (e.g. Jean Zay's V100,
        # sm_70) ptxas hard-fails with "Feature '.bf16' requires .target
        # sm_80 or higher" the first time such a kernel actually gets
        # compiled — which model/config combination triggers it is kernel-
        # specific (alibi/forgetting's smaller 2-layer configs didn't hit
        # it; stickbreaking's did), so this checks the GPU's actual compute
        # capability rather than allow-listing specific model names.
        # float32 sidesteps the bf16 codegen path entirely; a no-op on
        # sm_80+ (H100 etc.), where dtype stays the model's own default.
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
    import copy

    # Architectures without a resolvable final-norm path (moep's mid-network
    # residual-width change; GPT-BERT's BERT-style MLM head — LayerNorm ->
    # Linear -> GELU -> LayerNorm -> Linear, nothing like a single
    # pre-unembedding norm+matmul) can't support the full per-layer logit-lens
    # reconstruction. Rather than hard-failing, degrade to native
    # final-layer-only surprisal for those: skip the hidden-states/final_norm/
    # unembed reconstruction entirely and just read the model's own `.logits`
    # output for the last position — correct for ANY architecture by
    # construction, since it's literally what the model itself predicts
    # (unlike reconstructing via get_output_embeddings(), which would be
    # wrong for a head that does more than a single norm+matmul).
    # Jtapsa/moep resolves a final-norm path fine (model.norm) but its
    # residual-stream width changes mid-network (384 -> 192 -> 384), so
    # applying that norm to an intermediate layer's differently-shaped
    # hidden state crashes with a RuntimeError from F.layer_norm — not the
    # clean ValueError _find_final_norm raises for GPT-BERT. Forced into the
    # native_only fallback up front rather than caught reactively, since the
    # failure is architectural (always happens, not data-dependent).
    force_native_only = native_only or "moep" in model_name.lower()
    try:
        if force_native_only:
            raise ValueError(
                "native_only=True requested — final-layer-only surprisal"
                if native_only
                else "moep: mid-network residual-width change, see module docstring"
            )
        # final_norm DOES need a float32 copy (unlike unembed below): under
        # the model's native bf16/fp16, applying the LAST layer's norm to an
        # INTERMEDIATE layer's much-smaller-scale hidden state can overflow
        # inside the normalization itself (e.g. the variance computation),
        # not just in the matmul afterwards — casting only the matmul input
        # too late can't fix that, it needs the norm computed in float32 too.
        # Cheap to copy (hidden_size params only, not vocab-sized). Copying
        # (rather than converting the live module in place) keeps the
        # model's own forward pass — used just above to get hidden states —
        # completely unaffected.
        final_norm = copy.deepcopy(_find_final_norm(model)).float()
        native_only = False
    except ValueError as exc:
        (logger.info if native_only else logger.warning)(
            "%s: %s — using native final-layer-only surprisal "
            "(model's own logits output, no per-earlier-layer logit-lens "
            "reconstruction).",
            model_name, exc,
        )
        final_norm = None
        native_only = True

    if native_only:
        unembed = None
    else:
        unembed_native = model.get_output_embeddings()
        if unembed_native is None:
            raise ValueError(
                f"logit-lens: {model_name} has no output embeddings "
                "(get_output_embeddings() returned None)."
            )
        # unembed stays the model's OWN live module (no copy, no dtype
        # change) — for a large multi-GPU-sharded model (e.g. OPT-66B across
        # 2 H100s), the vocab-sized lm_head weight is already the single
        # biggest tensor in the model; a deep-copied float32 duplicate of it
        # (observed: 1.73 GiB for OPT-66B's ~50k-vocab head) was enough on
        # its own to OOM a GPU already ~1 GiB from its ceiling
        # ("torch.OutOfMemoryError: ... Tried to allocate 1.73 GiB").
        # Unlike final_norm (see below), the unembed matmul itself doesn't
        # need float32 WEIGHTS for stability — only the resulting LOGITS
        # need to be float32 before softmax (done in the loop below), which
        # is a much smaller tensor (seq_len x vocab, not hidden x vocab).
        unembed = unembed_native
        unembed_dtype = next(unembed_native.parameters()).dtype

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "logit_lens_surprisal_all_layers requires a fast tokenizer "
            f"(return_offsets_mapping); got {type(tokenizer).__name__}."
        )
    _ensure_pad_token_for_batches(tokenizer)

    # --- build full text and exact char spans for each word (same as fulltext embeddings) ---
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
        "Logit-lens fulltext [%s%s]: %d words → %d chunks × %d tokens (stride_delta=%d)",
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
                "logit-lens: no token found for word %d %r — using last chunk end",
                wi, words[wi],
            )
            word_assignments.append((n_chunks - 1, [seq_len - 1]))

    chunk_to_words: dict[int, list[int]] = defaultdict(list)
    for wi, (k, _) in enumerate(word_assignments):
        chunk_to_words[k].append(wi)

    n_layers: int | None = None
    out: list[np.ndarray | None] = [None] * len(words)

    with torch.no_grad():
        for k in sorted(chunk_to_words.keys()):
            # Strip padding before the forward pass: some kernels (FoX/forgetting FLA)
            # mis-handle padded sequences even with an attention mask. Surprisals are
            # written back at their original (padded) positions.
            v0, v1 = _valid_span(overflow_enc["attention_mask"][k])
            ids_k = overflow_enc["input_ids"][k : k + 1, v0:v1].to(input_device)
            am_k = torch.ones_like(ids_k)

            ids_seq = ids_k[0]  # (valid_len,)
            target_ids = ids_seq[1:]  # token at position p predicted from position p-1

            if native_only:
                # No per-layer reconstruction at all — one real forward pass,
                # read the model's own final logits directly. n_layers is
                # always 1 here (see module docstring / fallback comment above).
                # return_dict=True forced explicitly — some trust_remote_code
                # architectures (e.g. GPT-BERT) default to a plain tuple
                # otherwise, whose first element isn't reliably `.logits`
                # across every custom forward() implementation.
                out_fwd = model(
                    input_ids=ids_k, attention_mask=am_k, use_cache=False, return_dict=True,
                )
                if n_layers is None:
                    n_layers = 1
                surp_chunk = np.full((1, seq_len), np.nan, dtype=np.float32)
                logits_l = out_fwd.logits[0].float()  # (valid_len, vocab)
                log_probs_l = torch.log_softmax(logits_l, dim=-1)
                lp = log_probs_l[:-1, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)
                surp_chunk[0, v0 + 1 : v1] = (-lp).cpu().numpy()
                if not torch.isfinite(lp).all():
                    n_bad = int((~torch.isfinite(lp)).sum().item())
                    logger.warning(
                        "logit-lens (native fallback): chunk %d has %d/%d "
                        "non-finite log-probs (logits range [%.3g, %.3g])",
                        k, n_bad, lp.numel(),
                        float(logits_l.min()), float(logits_l.max()),
                    )
                del out_fwd, logits_l, log_probs_l, lp
            else:
                hs, native_logits = _forward_hidden_states(
                    model, ids_k, am_k, model_name=model_name, return_logits=True,
                )
                # Exclude the embedding layer (index 0) — logit-lens starts at
                # the first transformer block, per Kuribayashi et al. (2025).
                # Only strips it when actually present (native path); the
                # hook-based fallback for trust_remote_code architectures
                # already returns per-block-only captures with no embedding
                # entry to strip.
                layer_hs = _exclude_embedding_layer(hs, model)
                if n_layers is None:
                    n_layers = len(layer_hs)

                # surp_chunk[l, p] = surprisal of the token AT position p, from
                # layer l's readout; column 0 stays NaN (no left context).
                surp_chunk = np.full((n_layers, seq_len), np.nan, dtype=np.float32)
                for li, h in enumerate(layer_hs):
                    # Last layer: the model's own logits. Hook-captured block outputs
                    # (stickbreaking, hard n-gram) are PRE-final-norm, so reconstructing
                    # the last layer from hidden states is wrong for them; native logits
                    # are correct for every architecture. Earlier layers: final_norm
                    # (float32 copy) + the model's unembedding, i.e. the logit lens.
                    if li == n_layers - 1:
                        logits_l = native_logits[0].float()
                        normed = None
                    else:
                        normed = final_norm(h[0].float())
                        logits_l = unembed(normed.to(unembed_dtype)).float()  # (valid_len, vocab)
                    log_probs_l = torch.log_softmax(logits_l, dim=-1)
                    lp = log_probs_l[:-1, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)
                    surp_chunk[li, v0 + 1 : v1] = (-lp).cpu().numpy()
                    if not torch.isfinite(lp).all():
                        n_bad = int((~torch.isfinite(lp)).sum().item())
                        logger.warning(
                            "logit-lens: chunk %d layer %d has %d/%d non-finite "
                            "log-probs (logits range [%.3g, %.3g])",
                            k, li, n_bad, lp.numel(),
                            float(logits_l.min()), float(logits_l.max()),
                        )
                    del logits_l, log_probs_l, lp, normed

            for wi in chunk_to_words[k]:
                tok_idx = word_assignments[wi][1]
                vals = surp_chunk[:, tok_idx]  # (n_layers, n_subtoks)
                # A word containing the very first token of the text (no left context)
                # has undefined surprisal -> NaN (no partial sums).
                out[wi] = _aggregate_subwords(vals, subword_agg).astype(np.float32)

    missing = [wi for wi, v in enumerate(out) if v is None]
    if missing:
        raise RuntimeError(f"logit-lens: {len(missing)} words never assigned a chunk")

    result = np.stack(out, axis=0)[:, :, None].astype(np.float32)  # (n_words, n_layers, 1)

    non_finite = ~np.isfinite(result[:, :, 0])
    n_non_finite_per_layer = non_finite.sum(axis=0)
    # Word 0 (no left context) is NaN by construction in every layer — that's
    # the expected baseline, not a problem. Anything beyond that is real.
    if n_non_finite_per_layer.max() > 1:
        logger.warning(
            "logit-lens: non-finite surprisal per layer (word 0's expected "
            "NaN included): %s / %d words",
            n_non_finite_per_layer.tolist(), len(words),
        )
    else:
        logger.info(
            "logit-lens: surprisal is finite for all words/layers "
            "(only the expected word-0 NaN present)."
        )

    if cache_path is not None:
        try:
            _save_cached_features(cache_path, result)
        except Exception as exc:
            logger.warning("Could not save logit-lens surprisal cache: %s", exc)

    return result


def _trial_surprisal_cache_path(
    cache_dir: str | Path,
    model_name: str,
    revision: str | None,
    cache_key: str,
    untrained: bool,
    untrained_seed: int,
    subword_agg: str = "sum",
) -> Path:
    """Cache key for trial-based (cross-sectional) logit-lens surprisal.

    The name carries :data:`SURPRISAL_CACHE_VERSION` and the subword aggregation, so caches
    written by the pre-audit code (mean, BOS-shifted) are never read back.

    Unlike :func:`_fulltext_feature_cache_path` (keyed by section/maxlen/
    stride for one continuous document), trial data has no natural "section"
    — ``cache_key`` is caller-supplied instead (e.g. "target" vs "prev" word,
    or a dataset name) to distinguish different trial sets sharing a
    ``cache_dir``.
    """
    safe_model = model_name.replace("/", "_")
    safe_rev = (revision or "main").replace("/", "_").replace("-", "_")
    safe_key = cache_key.replace("/", "_")
    tag = f"untrained{untrained_seed}" if untrained else "trained"
    fname = (
        f"logitlens_trials_{SURPRISAL_CACHE_VERSION}_{subword_agg}_"
        f"{safe_model}_{safe_rev}_{safe_key}_{tag}.npz"
    )
    return Path(cache_dir) / fname


def logit_lens_surprisal_trials(
    contexts: list[str],
    targets: list[str],
    model_name: str,
    *,
    revision: str | None = None,
    device: str = "cpu",
    tokenizer_no_revision: bool = True,
    n_params: float | None = None,
    trust_remote_code: bool = False,
    untrained: bool = False,
    untrained_seed: int = 0,
    cache_dir: str | Path | None = None,
    cache_key: str = "trials",
    require_cache_hit: bool = False,
    dtype: "torch.dtype | str | None" = None,
    max_length: int | None = None,
    native_only: bool = False,
    native_batch_size: int = 16,
    layer_batch_size: int = 8,
    subword_agg: SubwordAgg = "sum",
) -> np.ndarray:
    """Per-layer logit-lens surprisal for isolated ``(context, target)`` trial pairs.

    ``native_only=True`` forces the native final-layer-only path for ANY
    architecture (skips the per-earlier-layer reconstruction — the dominant
    cost): returns ``(n_items, 1)`` whose single column is the true final
    layer. Cache path unchanged; a downstream ``last_layer_only`` LME fit
    reads it identically to a full cache's last column. Do NOT reuse such a
    cache for a full per-layer sweep.

    The ``native_only`` path batches ``native_batch_size`` trials per forward
    call (right-padded to the batch's longest sequence, via ``attention_mask``)
    instead of one call per trial — added 2026-09-12 after RWKV4 (no compiled
    WKV CUDA kernel available for this project's pinned torch 2.6, so HF falls
    back to a slow pure-PyTorch sequential recurrence) failed to finish even
    its smallest (169M) checkpoint's ~10k-trial Natural Stories sweep within a
    2-hour job. Numerically exact vs. the unbatched loop for any causal
    (uni-directional) architecture, RWKV's recurrence included: right-padding
    only appends tokens *after* the position range being read out
    (``n_ctx_eff .. n_ctx_eff + len(target_ids) - 1``), and causal/recurrent
    models cannot let a later position influence an earlier one's hidden
    state, so trailing padding cannot change any value this function reads.
    The full per-layer path (used by everything else — BLOOM, FoX, …) is
    ALSO batched (``layer_batch_size`` trials per forward call, same
    right-padding argument as above — the norm/unembed readout is per-token,
    so it is likewise unaffected by trailing padding) — added 2026-09-12
    after MoLM (its sparse-MoE expert routing runs mostly CPU-bound, not
    GPU-accelerated — see ``neuroai/models/registry.py``'s ``"molm"`` family
    comment) also failed to finish a checkpoint's Natural Stories sweep
    within a 2-hour job. Default batch size is smaller than
    ``native_batch_size`` (8 vs. 16) because this path additionally holds a
    ``(batch, seq, vocab)`` float32 logits tensor per layer at once, so the
    memory cost per batch element is higher.

    Unlike :func:`logit_lens_surprisal_all_layers` (continuous narrative,
    sliding-window chunking — designed for one long document), this is for
    cross-sectional trial data such as N400 experiments: one independent
    forward pass per ``(context, target)`` pair. ``context`` may be either a
    strict prefix (text strictly before ``target`` — the N400 datasets'
    convention) or cumulative text ending with ``target`` as a substring (the
    SPR/eye-tracking LME datasets' convention — Natural Stories, Provo, GECO,
    MECO, OneStop, UCL Reading, ZuCo Reading), auto-detected per item exactly
    like :func:`neuroai.models.surprisal._surprisal_one_pair`.

    Pass ``require_cache_hit=True`` to raise instead of silently falling back
    to a model load on a cache miss — same rationale as
    :func:`logit_lens_surprisal_all_layers`'s identically-named parameter: the
    CPU-only stage of a two-stage GPU-extract/CPU-fit pipeline must never
    load a model.

    The model/tokenizer are loaded once (LRU-cached by
    :func:`~neuroai.features.llm._get_cached_causal_lm`) and the float32
    final-norm + unembed copies used for logit-lens readout are built once,
    outside the per-item loop — calling
    :func:`logit_lens_surprisal_all_layers` in a loop instead would redo that
    (possibly large) deepcopy on every single item.

    Sub-word tokens of ``target`` are aggregated via their SUM (the word's surprisal,
    -log p(target | context)); ``subword_agg="mean"`` is a sensitivity variant only.
    The last layer is the model's own logits; earlier layers use the logit lens.

    Pass ``cache_dir``/``cache_key`` to cache to (and read from) disk — the
    cache is checked BEFORE the model is loaded, so a cache hit never touches
    the LLM at all. This is what makes the GPU-extraction / CPU-only-LME
    split possible (same pattern as Podcast's pooled LME): run once on GPU to
    populate the cache, then every CPU-only LME job reusing the same
    ``cache_dir``/``cache_key``/model/revision gets a fast cache hit.

    Returns
    -------
    np.ndarray
        Shape ``(n_items, n_layers)``, float32. Layer 0 is the first
        transformer block's output (embedding layer excluded), matching
        :func:`logit_lens_surprisal_all_layers`.
    """
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets must have equal length")

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _trial_surprisal_cache_path(
            cache_dir, model_name, revision, cache_key, untrained, untrained_seed,
            subword_agg=subword_agg,
        )
        cached = _load_cached_features(cache_path)
        if cached is not None:
            logger.info(
                "Logit-lens trial surprisal: loaded from cache %s (shape %s)",
                cache_path.resolve(), cached.shape,
            )
            return cached

    if require_cache_hit:
        raise RuntimeError(
            f"logit-lens trials: cache miss for {model_name}@{revision or 'main'} "
            f"cache_key={cache_key!r} (cache_dir={cache_dir!r}, expected path={cache_path}) — "
            "refusing to load the model with require_cache_hit=True. Run the GPU "
            "prefetch stage for this model/checkpoint first."
        )

    # CBR-RNN special case -- see the identically-gated branch in
    # logit_lens_surprisal_all_layers for the full rationale. Each (context,
    # target) pair maps directly onto the already-tested per-pair helper.
    if "cbr-rnn" in model_name.lower() or "cbr_rnn" in model_name.lower():
        from .cbr_rnn import _cbr_surprisal_one_pair, _get_cached_cbr_rnn

        tokenizer, model = _get_cached_cbr_rnn(model_name, device)
        dev = torch.device(device)
        result = np.empty((len(contexts), 1), dtype=np.float32)
        for i, (ctx, tgt) in enumerate(zip(contexts, targets)):
            result[i, 0] = _cbr_surprisal_one_pair(model, tokenizer, dev, ctx, tgt)[0]
        if cache_path is not None:
            _save_cached_features(cache_path, result)
            logger.info("CBR-RNN logit-lens trial surprisal: cached → %s (shape %s)", cache_path.resolve(), result.shape)
        return result

    import copy

    load_kw: dict[str, tp.Any] = {"output_hidden_states": True, "trust_remote_code": trust_remote_code}
    if dtype is not None:
        # Same gpt-bert DWA-accumulator bf16 workaround as
        # logit_lens_surprisal_all_layers — see that function's setup comment
        # for the full rationale (in-place buffer .set_() hard-codes float32).
        load_kw["torch_dtype"] = dtype
    elif trust_remote_code and torch.cuda.is_available():
        # Same V100 bf16-PTX-codegen workaround as
        # logit_lens_surprisal_all_layers — see that function's setup comment.
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

    # Clamp to the model's own position-embedding capacity, same as
    # logit_lens_surprisal_all_layers -- a caller-agnostic hardcoded 2048
    # default silently overflows small-context architectures (e.g.
    # Jtapsa/moep's max_position_embeddings=1024, a separate learned table
    # from its 16384-row token embedding) with an out-of-bounds embedding
    # lookup that surfaces as an opaque "CUDA error: device-side assert
    # triggered" many frames away from the real cause.
    if max_length is not None:
        eff_max_length = max_length
    else:
        cfg_max = getattr(model.config, "max_position_embeddings", None)
        eff_max_length = (
            cfg_max if isinstance(cfg_max, int) and 0 < cfg_max < 1_000_000 else 2048
        )
        eff_max_length = min(eff_max_length, 32768)

    # Same architecture-scoped native fallback as logit_lens_surprisal_all_layers
    # (see that function's setup comment and the module docstring for the full
    # rationale): GPT-BERT-style heads have no resolvable single final-norm
    # path at all (_find_final_norm raises ValueError), and Jtapsa/moep
    # resolves one (model.norm) but its residual-stream width changes
    # mid-network (384 -> 192 -> 384), so applying that norm to an
    # intermediate layer's differently-shaped hidden state crashes instead of
    # raising a clean ValueError — forced into the same fallback up front.
    force_native_only = native_only or "moep" in model_name.lower()
    try:
        if force_native_only:
            raise ValueError(
                "native_only=True requested — final-layer-only surprisal"
                if native_only
                else "moep: mid-network residual-width change, see module docstring"
            )
        # final_norm: float32 copy, built ONCE for the whole trial set — see
        # logit_lens_surprisal_all_layers' setup comment for why (norm
        # overflow risk on intermediate-layer hidden states, cheap since
        # hidden_size-only).
        final_norm = copy.deepcopy(_find_final_norm(model)).float()
        native_only = False
    except ValueError as exc:
        (logger.info if native_only else logger.warning)(
            "%s: %s — using native final-layer-only surprisal "
            "(model's own logits output, no per-earlier-layer logit-lens "
            "reconstruction).",
            model_name, exc,
        )
        final_norm = None
        native_only = True

    if native_only:
        unembed = None
        unembed_dtype = None
    else:
        unembed_native = model.get_output_embeddings()
        if unembed_native is None:
            raise ValueError(
                f"logit-lens: {model_name} has no output embeddings "
                "(get_output_embeddings() returned None)."
            )
        # unembed: stays the model's OWN live module, no copy/dtype change —
        # see logit_lens_surprisal_all_layers' setup comment (a deep-copied
        # float32 vocab-sized lm_head is enough on its own to OOM a large
        # multi-GPU model already near its memory ceiling, e.g. OPT-66B).
        # Only the resulting LOGITS get upcast to float32 in the loop below,
        # not the weight matrix.
        unembed = unembed_native
        unembed_dtype = next(unembed_native.parameters()).dtype

    _ensure_pad_token_for_batches(tokenizer)
    # Truncate from the LEFT (drop the earliest context), not the default
    # right/end truncation — the target word always sits at the very end of
    # ``full`` (both the cumulative-context and prefix+target conventions
    # below), so right-truncation on an over-length item would silently drop
    # the target itself instead of just some early context, producing a
    # spurious zero-target-token item (same symptom as an actual bug, just
    # for legitimately-long trials instead of a real tokenizer break).
    tokenizer.truncation_side = "left"

    # None placeholders for zero-target-token items are backfilled with a
    # correctly-shaped NaN array once n_layers is known (see below) — it
    # isn't necessarily known yet for the first item(s), since n_layers is
    # only set from a real forward pass in the non-native branch.
    out: list[np.ndarray | None] = [None] * len(contexts)
    n_layers: int | None = 1 if native_only else None
    special_ids = set(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or []))
    is_fast = bool(getattr(tokenizer, "is_fast", False))

    def _target_positions(ctx_s: str, tgt_s: str) -> tuple[list[int], list[int]]:
        """Input token ids and the positions of the target word's tokens.

        Two context conventions coexist across callers: N400-style trial data passes a
        PREFIX (text strictly before ``tgt``), the SPR/eye-tracking datasets pass
        CUMULATIVE context ending with ``tgt``; auto-detected via the suffix check.
        Target tokens are located by character offsets in the tokenization actually fed to
        the model (special tokens included), so BOS/CLS prefixes and EOS/SEP suffixes can
        never shift the target window (the pre-2026-10-12 code counted the context without
        special tokens and was off by one for every BOS-adding tokenizer). Trailing special
        tokens are dropped from the input. Positions whose token has no left context
        (position 0) make the whole word undefined (empty list -> NaN).
        """
        cumulative = bool(tgt_s and ctx_s.endswith(tgt_s))
        if cumulative:
            full = ctx_s
        else:
            full = (ctx_s + " " + tgt_s).strip() if ctx_s else tgt_s
        tgt_start = len(full) - len(tgt_s)
        if is_fast:
            enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True)
            ids = [int(i) for i in enc["input_ids"]]
            pos = [
                i for i, (a, b) in enumerate(enc["offset_mapping"])
                if b > a and b > tgt_start and ids[i] not in special_ids
            ]
        else:
            ids = [int(i) for i in tokenizer(full, add_special_tokens=True)["input_ids"]]
            plain = [int(i) for i in tokenizer(full, add_special_tokens=False)["input_ids"]]
            n_pre = next(
                (k for k in range(len(ids) - len(plain) + 1) if ids[k : k + len(plain)] == plain), 0,
            )
            prefix = full[:tgt_start].rstrip()
            n_ctx = len(tokenizer(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
            pos = list(range(n_pre + n_ctx, n_pre + len(plain)))
        if not pos:
            return ids, []
        ids = ids[: pos[-1] + 1]
        if len(ids) > eff_max_length:  # left-truncate, keep the target at the end
            drop = len(ids) - eff_max_length
            ids = ids[drop:]
            pos = [p - drop for p in pos]
        if pos[0] < 1:
            return ids, []
        return ids, pos

    def _pad_batch(pending: list[tuple[int, list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(ids) for _, ids, _ in pending)
        pad_id = tokenizer.pad_token_id
        batch_ids = torch.full((len(pending), max_len), pad_id, dtype=torch.long, device=input_device)
        batch_am = torch.zeros((len(pending), max_len), dtype=torch.long, device=input_device)
        for bi, (_, ids, _) in enumerate(pending):
            batch_ids[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=input_device)
            batch_am[bi, : len(ids)] = 1
        return batch_ids, batch_am

    def _readout(log_probs_b: torch.Tensor, pending) -> list[float]:
        vals = []
        for bi, (_, ids, pos) in enumerate(pending):
            tok = [-float(log_probs_b[bi, p - 1, ids[p]]) for p in pos]
            vals.append(float(_aggregate_subwords(np.asarray(tok, dtype=np.float64), subword_agg)))
        return vals

    def _compute(pending) -> list[np.ndarray]:
        """Per-item (n_layers,) surprisal for one right-padded batch."""
        nonlocal n_layers
        batch_ids, batch_am = _pad_batch(pending)
        if native_only:
            out_fwd = model(input_ids=batch_ids, attention_mask=batch_am, use_cache=False, return_dict=True)
            log_probs_b = torch.log_softmax(out_fwd.logits.float(), dim=-1)
            res = [np.array([v], dtype=np.float32) for v in _readout(log_probs_b, pending)]
            del out_fwd, log_probs_b
            return res
        hs, native_logits = _forward_hidden_states(
            model, batch_ids, batch_am, model_name=model_name, return_logits=True,
        )
        layer_hs = _exclude_embedding_layer(hs, model)
        if n_layers is None:
            n_layers = len(layer_hs)
        per_layer = np.empty((len(pending), n_layers), dtype=np.float32)
        for li, h in enumerate(layer_hs):
            # Last layer = the model's own logits (correct for every architecture,
            # including hook-captured pre-norm block outputs); earlier layers = logit lens.
            if li == n_layers - 1:
                logits_b = native_logits.float()
            else:
                logits_b = unembed(final_norm(h.float()).to(unembed_dtype)).float()
            log_probs_b = torch.log_softmax(logits_b, dim=-1)
            per_layer[:, li] = _readout(log_probs_b, pending)
            del logits_b, log_probs_b
        return [per_layer[bi].copy() for bi in range(len(pending))]

    # Right-padded batching is only exact if the model honours the attention mask for a
    # causal readout. FoX/forgetting FLA kernels do not (corr 0.38-0.81 with unbatched on
    # Provo), so the first multi-item batch is also run item by item; on disagreement the
    # rest of the run falls back to batch size 1.
    batch_safe: bool | None = None

    def _flush(pending) -> None:
        nonlocal batch_safe
        if not pending:
            return
        if batch_safe is False or len(pending) == 1:
            res = [_compute([p])[0] for p in pending]
        elif batch_safe is None:
            batched = _compute(pending)
            single = [_compute([p])[0] for p in pending]
            diffs = [np.abs(a - b) for a, b in zip(batched, single)]
            max_diff = max((float(np.nanmax(d)) for d in diffs if np.isfinite(d).any()), default=0.0)
            batch_safe = max_diff <= 0.5
            (logger.info if batch_safe else logger.warning)(
                "Logit-lens trials [%s]: batched vs unbatched max |diff| = %.3g nats -> %s",
                model_name, max_diff, "batching enabled" if batch_safe else "batch size 1 for this model",
            )
            res = batched if batch_safe else single
        else:
            res = _compute(pending)
        for (out_idx, _, _), v in zip(pending, res):
            out[out_idx] = v

    # Resumable partial checkpoint: a job killed mid-loop (e.g. 2h dev-QoS
    # walltime) keeps every finished trial and the next run skips them. The
    # file name embeds a fingerprint of the trial set + readout mode, so a
    # changed trial list or native/full switch can never load stale values.
    partial_path: Path | None = None
    done = np.zeros(len(contexts), dtype=bool)
    if cache_path is not None:
        fp = hashlib.md5()
        for c_, t_ in zip(contexts, targets):
            fp.update(c_.encode()); fp.update(b"\x00"); fp.update(t_.encode()); fp.update(b"\x01")
        mode = "nat" if native_only else "full"
        partial_path = cache_path.with_name(
            f"{cache_path.stem}.partial_{fp.hexdigest()[:12]}_{len(contexts)}_{mode}.npz"
        )
        prev = _load_cached_features(partial_path)
        if prev is not None and prev.ndim == 2 and prev.shape[0] == len(contexts):
            done = prev[:, -1] > 0.5
            n_layers = prev.shape[1] - 1
            for i in np.flatnonzero(done):
                out[i] = prev[i, :-1].copy()
            logger.info(
                "Logit-lens trials: resuming from partial %s (%d/%d trials done)",
                partial_path.name, int(done.sum()), len(contexts),
            )
    last_partial_save = time.monotonic()

    def _save_partial(force: bool = False) -> None:
        nonlocal last_partial_save
        if partial_path is None or n_layers is None:
            return
        if not force and time.monotonic() - last_partial_save < 120:
            return
        arr = np.zeros((len(out), n_layers + 1), dtype=np.float32)
        for i, v in enumerate(out):
            if v is not None:
                arr[i, :-1] = v
                arr[i, -1] = 1.0
        try:
            _save_cached_features(partial_path, arr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save partial trial-surprisal checkpoint: %s", exc)
        last_partial_save = time.monotonic()

    batch_size = native_batch_size if native_only else layer_batch_size
    pending: list[tuple[int, list[int], list[int]]] = []
    with torch.no_grad():
        for item_idx, (ctx, tgt) in enumerate(zip(contexts, targets)):
            if done[item_idx]:
                continue
            ids, pos = _target_positions(ctx.strip(), tgt.strip())
            if not pos:
                # No scoreable target tokens (empty target, or target at position 0
                # with no left context): left as None, backfilled with NaN below.
                continue
            pending.append((item_idx, ids, pos))
            if len(pending) >= batch_size:
                _flush(pending)
                pending = []
                _save_partial()
        _flush(pending)
        pending = []

    if any(item is None for item in out):
        if n_layers is None:
            raise RuntimeError(
                "logit-lens trials: every item had zero target tokens — "
                "n_layers could never be determined from a real forward pass."
            )
        out = [
            np.full(n_layers, np.nan, dtype=np.float32) if item is None else item
            for item in out
        ]

    result = np.stack(out, axis=0).astype(np.float32)

    if cache_path is not None:
        try:
            _save_cached_features(cache_path, result)
        except Exception as exc:
            logger.warning("Could not save logit-lens trial surprisal cache: %s", exc)
        else:
            if partial_path is not None:
                partial_path.unlink(missing_ok=True)

    return result
