"""Word-level hidden-state extraction for the CBR-RNN model (cclark/cbr-rnn).

The CBR-RNN (Cued-By-Retrieval RNN, Clark et al.) is a non-transformer RNN
that uses attention over past key-value caches.  It loads via
``PyTorchModelHubMixin``, not ``AutoModelForCausalLM``, so it needs its own
extraction path.

Unlike transformer models there is a **single hidden-state vector** per token
(no layer dimension).  Returned shape is always ``(n_words, d_hid)``.

Reference
---------
  cclark/cbr-rnn on HuggingFace Hub
  https://github.com/Neuro-Cog-AI/HumanWMBenchmark
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# LRU-1 in-process model cache — same pattern as features/llm.py
_cbr_slot_key: tuple[str, str] | None = None
_cbr_slot_tokenizer: tp.Any = None
_cbr_slot_model: tp.Any = None


def _get_cached_cbr_rnn(model_name: str, device: str) -> tuple[tp.Any, tp.Any]:
    """Load and cache (LRU-1) a CbrRnn model + tokenizer pair."""
    global _cbr_slot_key, _cbr_slot_tokenizer, _cbr_slot_model

    slot = (model_name, device)
    if _cbr_slot_key == slot and _cbr_slot_model is not None:
        return _cbr_slot_tokenizer, _cbr_slot_model

    # Evict previous
    if _cbr_slot_model is not None:
        del _cbr_slot_model
        _cbr_slot_model = None
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    from ..models.cbr_rnn import CbrRnn

    # Use AutoTokenizer exactly as the original authors do in surprisal.py.
    # Requires tokenizer.json to be in the HF cache (prefetch on a login node
    # if missing: python -c "from transformers import AutoTokenizer;
    # AutoTokenizer.from_pretrained('cclark/cbr-rnn')")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # The CBR-RNN tokenizer has no pad_token; set it to eos_token so that
    # sliding-window encoding (padding=True) can stack chunks into a tensor.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = CbrRnn.from_pretrained(model_name)
    model.eval()
    model.to(device)

    _cbr_slot_key = slot
    _cbr_slot_tokenizer = tokenizer
    _cbr_slot_model = model
    logger.info("CBR-RNN: loaded & cached %s on %s (d_hid=%d)", model_name, device, model.d_hid)
    return tokenizer, model


def _cbr_feature_cache_path(
    cache_dir: Path | str,
    model_name: str,
    lang: str,
    section: int,
    context_words: int,
) -> Path:
    safe_model = model_name.replace("/", "_")
    filename = (
        f"{safe_model}__{lang.lower()}"
        f"__sec{section:02d}"
        f"__ctx{context_words}"
        f"__cbrrnn.npz"
    )
    return Path(cache_dir) / filename


def contextual_word_embeddings_cbr_rnn(
    words: list[str],
    model_name: str,
    *,
    device: str = "cpu",
    context_words: int = 48,
    cache_dir: str | Path | None = None,
    section_id: int | None = None,
    lang: str = "en",
) -> np.ndarray:
    """Return CBR-RNN hidden states for a list of surface words.

    For each word, builds a context window of up to ``context_words`` preceding
    words, prepends the model's BOS token, runs the CBR-RNN, and takes the
    hidden state at the last token position.  This mirrors the moving-window
    approach used by :func:`~neuroai.features.llm.contextual_word_embeddings_hf`.

    Parameters
    ----------
    words : list[str]
        Surface tokens in section-local order.
    model_name : str
        HuggingFace model id (e.g. ``"cclark/cbr-rnn"``).
    device : str
        ``"cuda"`` or ``"cpu"``.
    context_words : int
        Maximum number of preceding words to include in the context window.
    cache_dir : path, optional
        Directory for ``.npz`` disk cache.
    section_id : int, optional
        Section index (used for the cache key).
    lang : str
        Language code — used for the cache key only.

    Returns
    -------
    np.ndarray, shape ``(n_words, d_hid)``
        Float32 array of CBR-RNN hidden states, one row per word.
    """
    if not words:
        raise ValueError("words must be non-empty")

    # Cache look-up
    cache_path: Path | None = None
    if cache_dir is not None and section_id is not None:
        cache_path = _cbr_feature_cache_path(
            cache_dir, model_name, lang, section_id, context_words
        )
        if cache_path.exists():
            try:
                arr = np.load(str(cache_path))["features"]
                logger.info(
                    "CBR-RNN features: loaded from cache %s (shape %s)",
                    cache_path.resolve(),
                    arr.shape,
                )
                return arr
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load CBR-RNN cache %s: %s", cache_path, exc)

    tokenizer, model = _get_cached_cbr_rnn(model_name, device)

    bos_id: int | None = getattr(tokenizer, "bos_token_id", None)
    out_vecs: list[np.ndarray] = []

    for i, word in enumerate(words):
        context = words[max(0, i - context_words) : i + 1]
        text = " ".join(context)
        token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)

        # Prepend BOS to match the model's training regime
        if bos_id is not None:
            input_ids = [bos_id] + token_ids
        else:
            input_ids = token_ids

        input_tensor = torch.tensor([input_ids], device=device)  # [1, seq]

        with torch.no_grad():
            _logits, hiddens, _mem = model(input_tensor)
        # hiddens: [seq_len, batch=1, d_hid]
        # Take the hidden state at the LAST token (which corresponds to the
        # current word's last sub-token in the context sequence).
        vec = hiddens[-1, 0, :].float().cpu().numpy()  # (d_hid,)
        out_vecs.append(vec)

    emb = np.stack(out_vecs, axis=0).astype(np.float32, copy=False)  # (n_words, d_hid)
    logger.info(
        "CBR-RNN features: model=%s words=%d shape=%s device=%s",
        model_name,
        len(words),
        "×".join(str(d) for d in emb.shape),
        device,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.parent / f".{cache_path.stem}.part.npz"
        try:
            np.savez_compressed(str(tmp), features=emb)
            tmp.replace(cache_path)
        finally:
            tmp.unlink(missing_ok=True)
        logger.info(
            "CBR-RNN features: cached → %s (%d MB)",
            cache_path.resolve(),
            cache_path.stat().st_size // (1024 * 1024),
        )

    return emb


# ---------------------------------------------------------------------------
# Surprisal extraction (l-bg full-text sliding window)
# ---------------------------------------------------------------------------

def contextual_word_surprisal_cbr_rnn(
    words_df: "pd.DataFrame",
    model_name: str,
    *,
    bids_root: "Path | str | None" = None,
    lbg_story_text_zip: "Path | str | None" = None,
    lang: str = "en",
    section_id: "int | str | None" = None,
    device: str = "cpu",
    hf_lbg_stride_delta: int = 64,
    hf_lbg_max_length: "int | None" = None,
    cache_dir: "Path | str | None" = None,
) -> "np.ndarray":
    """Per-word surprisal (-log P) from CBR-RNN using the l-bg sliding window.

    Uses the same full-text tokenisation and word alignment as
    :func:`~neuroai.features.llm.contextual_word_surprisal_hf_lbg`, but
    forwards tokens through the CBR-RNN instead of a HuggingFace causal LM.
    The CBR-RNN logits have the same ``[batch, seq_len, vocab_size]`` shape
    as a standard causal LM, so the surprisal formula is identical.

    Returns
    -------
    np.ndarray, shape ``(n_words,)``
        Surprisal in nats per word.
    """
    from ..features.llm import (
        _lbg_feature_cache_path,
        _load_cached_features,
        _save_cached_features,
        build_lbg_word_stream,
        do_word_match,
        load_story_text,
        simplify_word,
    )

    lg = lang.lower()
    max_seq_len = hf_lbg_max_length or 256   # CBR-RNN default max_seq_len
    stride = max_seq_len - hf_lbg_stride_delta
    if stride <= 0:
        raise ValueError(f"Invalid stride: max_seq_len={max_seq_len}, delta={hf_lbg_stride_delta}")

    cache_path: "Path | None" = None
    if cache_dir is not None:
        cache_path = _lbg_feature_cache_path(
            cache_dir, model_name, None, lg, section_id, max_seq_len, stride, "surprisal"
        )
        cached = _load_cached_features(cache_path)
        if cached is not None:
            arr = cached.squeeze()
            if arr.shape[0] == len(words_df):
                logger.info(
                    "CBR-RNN surprisal: loaded from cache %s", cache_path.resolve()
                )
                return arr
            logger.warning(
                "CBR-RNN surprisal: stale cache (shape %s, expected %d) — recomputing",
                arr.shape,
                len(words_df),
            )

    tokenizer, model = _get_cached_cbr_rnn(model_name, device)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "CBR-RNN surprisal requires a fast tokenizer with return_offsets_mapping. "
            f"Got {type(tokenizer).__name__}."
        )

    fulltext = load_story_text(
        bids_root=bids_root,
        lang=lg,
        section=section_id,
        lbg_story_text_zip=Path(lbg_story_text_zip) if lbg_story_text_zip else None,
    ).replace("\n", " ").replace("\r", " ")

    word_list, char_to_word_idx, n_words_after_cn = build_lbg_word_stream(words_df, lg)
    if not word_list:
        raise ValueError("Empty word stream for CBR-RNN surprisal extraction")

    overflow_enc = tokenizer(
        fulltext,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=max_seq_len,
        return_overflowing_tokens=True,
        stride=stride,
    )
    om = overflow_enc["offset_mapping"]
    seq_len = int(overflow_enc["input_ids"].shape[1])

    # ── word → (batch, token) alignment ─────────────────────────────────────
    idx_batch = 0
    idx_token = 0
    idx_word_to_idx_token: list[tuple[int, int]] = []
    for idx_word, raw_w in enumerate(word_list):
        w_key = simplify_word(raw_w, lg)
        if idx_token == seq_len:
            idx_batch += 1
            idx_token = stride
        i_start = int(om[idx_batch, idx_token, 0].item())
        i_stop = int(om[idx_batch, idx_token, 1].item())
        n_try = 0
        while n_try < 20 and not do_word_match(w_key, fulltext[i_start:i_stop], lg):
            idx_token += 1
            if idx_token == seq_len:
                idx_batch += 1
                idx_token = stride
            i_start = int(om[idx_batch, idx_token, 0].item())
            i_stop = int(om[idx_batch, idx_token, 1].item())
            n_try += 1
        if n_try == 20:
            raise ValueError(
                f"CBR-RNN surprisal alignment failed "
                f"(word {idx_word}, model={model_name}, section={section_id})"
            )
        idx_word_to_idx_token.append((idx_batch, idx_token))
        if lg != "cn":
            idx_token += 1
        elif idx_word < len(word_list) - 1 and not do_word_match(
            str(word_list[idx_word + 1]), fulltext[i_start:i_stop], lg
        ):
            idx_token += 1

    # ── forward passes: collect per-token log-probs per batch ───────────────
    nb = overflow_enc["input_ids"].shape[0]
    logp_batches: list[np.ndarray] = []
    for k in range(int(nb)):
        ids_k = overflow_enc["input_ids"][k : k + 1].to(device)
        with torch.no_grad():
            logits, _, _ = model(ids_k)
        logits_cpu = logits[0].float().cpu()          # [seq_len, vocab]
        log_probs = torch.log_softmax(logits_cpu, dim=-1).numpy()
        ids_cpu = overflow_enc["input_ids"][k].numpy()
        surp = np.zeros(seq_len, dtype=np.float32)
        surp[1:] = -log_probs[np.arange(seq_len - 1), ids_cpu[1:]]
        logp_batches.append(surp)

    # ── aggregate to word level ──────────────────────────────────────────────
    def _tok_surp(b: int, t: int) -> float:
        return float(logp_batches[b][t])

    word_surprisals: list[float] = []
    for idx_word in range(len(word_list) - 1):
        b_a, t_a = idx_word_to_idx_token[idx_word]
        b_b, t_b = idx_word_to_idx_token[idx_word + 1]
        s = 0.0
        if b_a == b_b:
            for t in range(t_a, t_b):
                s += _tok_surp(b_a, t)
        else:
            for t in range(t_a, seq_len):
                s += _tok_surp(b_a, t)
            for t in range(stride, t_b):
                s += _tok_surp(b_b, t)
        word_surprisals.append(s)

    # last word
    b_l, t_l = idx_word_to_idx_token[-1]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    row_ids = overflow_enc["input_ids"][b_l].numpy()
    row_attn = overflow_enc["attention_mask"][b_l].numpy()
    valid_len = int(row_attn.sum())
    s = 0.0
    for t in range(t_l, min(valid_len, seq_len)):
        if eos_id is not None and int(row_ids[t]) == eos_id:
            break
        s += _tok_surp(b_l, t)
    word_surprisals.append(s)

    result = np.array(word_surprisals, dtype=np.float32)
    if result.shape[0] != len(words_df):
        raise ValueError(
            f"CBR-RNN surprisal shape mismatch: {result.shape[0]} values, "
            f"{len(words_df)} words (model={model_name}, section={section_id})"
        )

    if cache_path is not None:
        _save_cached_features(cache_path, result[:, np.newaxis])
        logger.info(
            "CBR-RNN surprisal: cached → %s (shape %s)", cache_path.resolve(), result.shape
        )

    return result


# ---------------------------------------------------------------------------
# Natural Stories surprisal (moving-window context, no BIDS required)
# ---------------------------------------------------------------------------

def _cbr_surprisal_one_pair(
    model: tp.Any,
    tokenizer: tp.Any,
    device: torch.device,
    context: str,
    target: str,
) -> np.ndarray:
    """CBR-RNN surprisal for one (context, target) pair.

    Mirrors :func:`~neuroai.features.surprisal._surprisal_one_pair` but calls
    the CBR-RNN forward (returns a ``(logits, hiddens, mem)`` tuple with
    ``logits`` shape ``[batch, seq_len, vocab]``).

    The input is left-truncated to the model's max sequence length (the size
    of the precomputed ``cos`` positional buffer) so that long contexts never
    cause an out-of-range index in the TorchScript RNN kernel.
    """
    # CBR-RNN max sequence length is the size of the precomputed cos buffer.
    max_seq = int(model.cos.shape[0])

    ctx_s = context.strip()
    tgt_s = target.strip()

    cumulative = bool(tgt_s and ctx_s.endswith(tgt_s))
    if cumulative:
        prefix_text = ctx_s[: -len(tgt_s)].rstrip()
    else:
        prefix_text = ctx_s

    # Tokenize prefix and target separately to know the boundary, then
    # left-truncate the combined sequence to fit within max_seq.
    prefix_ids: list[int] = (
        tokenizer.encode(prefix_text, add_special_tokens=False) if prefix_text else []
    )
    if cumulative:
        tgt_ids: list[int] = tokenizer.encode(
            tgt_s, add_special_tokens=False
        )
        full_ids = prefix_ids + tgt_ids
    else:
        full_text = (ctx_s + " " + tgt_s).strip() if ctx_s else tgt_s
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        # Recount prefix boundary after joint tokenisation
        prefix_ids = tokenizer.encode(ctx_s, add_special_tokens=False) if ctx_s else []

    n_ctx_full = len(prefix_ids)

    # Left-truncate to max_seq, keeping the target tokens at the end.
    if len(full_ids) > max_seq:
        drop = len(full_ids) - max_seq
        full_ids = full_ids[drop:]
        n_ctx = max(0, n_ctx_full - drop)
    else:
        n_ctx = n_ctx_full

    input_ids = torch.tensor([full_ids], device=device)  # [1, seq_len]
    with torch.no_grad():
        logits, _, _ = model(input_ids)  # [1, seq_len, vocab] or [seq_len, 1, vocab]

    # Normalise to [seq_len, vocab] regardless of batch dimension order.
    logits_cpu = logits[0].float().cpu()
    log_probs = torch.log_softmax(logits_cpu, dim=-1)
    ids_t = torch.tensor(full_ids)

    target_ids = ids_t[n_ctx:]
    surp = (
        np.array(
            [
                -log_probs[n_ctx + j - 1, target_ids[j]].item()
                for j in range(len(target_ids))
            ],
            dtype=np.float64,
        )
        if len(target_ids) > 0
        else np.array([np.nan], dtype=np.float64)
    )
    # Word surprisal = SUM over the target's tokens (was the mean before the 2026-10-12 audit).
    return np.array([float(surp.sum())], dtype=np.float32)


def _cbr_batch_surprisal_cache_path(
    cache_dir: Path | str,
    model_name: str,
    section: int,
) -> Path:
    safe_model = model_name.replace("/", "_")
    return Path(cache_dir) / f"{safe_model}__sec{section:02d}__cbrrnn_batch_surprisal_v2sum.npz"


def batch_surprisal_cbr_rnn(
    model_name: str,
    contexts: list[str],
    targets: list[str],
    *,
    device: str = "auto",
    n_params: float | None = None,  # unused; kept for API parity with batch_surprisal_one_model
    cache_dir: str | Path | None = None,
    section_id: int | None = None,
    require_cache_hit: bool = False,
) -> np.ndarray:
    """Word-level surprisal for Natural Stories using CBR-RNN.

    Uses the same moving-window context as
    :func:`~neuroai.features.surprisal.batch_surprisal_one_model` so the
    pipeline is plug-compatible.  Does **not** require BIDS data. CBR-RNN has
    a single hidden-state vector per token (no layer dimension), so this
    returns a single "layer" of surprisal — used as the (n_words, 1) result
    for the per-layer pooled LME pipelines, which treat n_layers generically.

    Pass ``cache_dir``/``section_id`` to cache to (and read from) disk, same
    two-stage GPU/CPU contract as :func:`~neuroai.features.logit_lens.
    logit_lens_surprisal_all_layers` — the cache is checked BEFORE loading
    the model, so a cache hit never touches CBR-RNN at all. Pass
    ``require_cache_hit=True`` to raise instead of silently falling back to
    a model load on a miss (same rationale as ``logit_lens_surprisal_all_layers``'s
    parameter of the same name — enforces the CPU-only stage's "fail fast on
    cache miss" contract instead of relying on the caller to separately
    guess whether a configured cache_dir means the item is actually cached).

    Returns
    -------
    np.ndarray, shape ``(n_words, 1)``
        Surprisal in nats, matching the output shape of
        :func:`~neuroai.features.surprisal.batch_surprisal_one_model`.
    """
    from .llm import _load_cached_features, _save_cached_features

    cache_path: Path | None = None
    if cache_dir is not None and section_id is not None:
        cache_path = _cbr_batch_surprisal_cache_path(cache_dir, model_name, section_id)
        cached = _load_cached_features(cache_path)
        if cached is not None and cached.shape[0] == len(targets):
            logger.info("CBR-RNN batch surprisal: loaded from cache %s", cache_path.resolve())
            return cached

    if require_cache_hit:
        raise RuntimeError(
            f"CBR-RNN batch surprisal: cache miss for {model_name} section={section_id} "
            f"(cache_dir={cache_dir!r}, expected path={cache_path}) — refusing to load "
            "the model with require_cache_hit=True. Run the GPU prefetch stage first."
        )

    dev_str = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    device_t = torch.device(dev_str)

    tokenizer, model = _get_cached_cbr_rnn(model_name, dev_str)
    rows: list[np.ndarray] = []
    for ctx, tgt in zip(contexts, targets):
        rows.append(_cbr_surprisal_one_pair(model, tokenizer, device_t, ctx, tgt))
    result = np.vstack(rows)

    if cache_path is not None:
        _save_cached_features(cache_path, result)
        logger.info("CBR-RNN batch surprisal: cached → %s", cache_path.resolve())

    return result
