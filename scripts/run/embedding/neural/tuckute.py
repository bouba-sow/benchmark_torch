"""Tuckute et al. (2023) fMRI encoding checkpoint sweep.

Extracts LLM sentence embeddings (last-token, all layers) and predicts
participant-averaged BOLD responses via ridge regression (k=4 fold CV).
Evaluated separately for conditions B (baseline), D (drive), S (suppress),
and all (combined).

Single-stage pipeline — no HRF, no two-step prefetch.  Embeddings are
cached to disk per model/revision so restarts are fast.

Output CSVs (under ``<out_dir>/<family>/<model_safe>/``):
  tuckute_scores.csv        — one row per checkpoint (summary across ROIs)
  tuckute_by_roi.csv        — one row per checkpoint × condition × ROI × layer
  tuckute_layerwise.csv     — one row per checkpoint × condition × layer (mean over ROIs)

Usage
-----
  # list models for a family
  python scripts/run/embedding/neural/tuckute.py --family gpt2 --list

  # run one SLURM array task
  python scripts/run/embedding/neural/tuckute.py \\
      --family gpt2 --array-index 0 --device cuda

  # run summary/plots only
  python scripts/run/embedding/neural/tuckute.py \\
      --family gpt2 --summary-only

  # full family in one go (local testing)
  python scripts/run/embedding/neural/tuckute.py \\
      --family gpt2 --all --device cuda

Jean Zay::

  python scripts/submit.py tuckute embedding fit --family gpt2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np
import pandas as pd
import torch
import yaml

from neuroai.datasets.tuckute import LANG_ROIS, CONDITIONS, TuckuteDataset
from neuroai.features.llm import clear_hf_model_cache
from neuroai.models.registry import (
    build_checkpoint_items,
    registry_model_configs,
    resolve_family_task,
    resolve_load_model_id,
)
from neuroai.regressors.ridge import cv_ridge_score_multioutput, cv_ridge_score_multioutput_gpu
from neuroai.features.llm import embedding_layer_selection  # noqa: E402
from neuroai.sweep import (
    CheckpointItem,
    append_row,
    already_computed,
    load_existing_csv,
    safe_model_dir,
    free_gpu,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)

EVAL_CONDITIONS: list[str] = ["B", "D", "S"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(os.path.expandvars(f.read()))


def _resolve_alphas(raw: Any) -> list[float]:
    if raw in (None, "null", []):
        alphas = np.logspace(2, 7, 5).tolist()  # [100, 1778, 31623, 562341, 1e7]
    else:
        alphas = [float(a) for a in raw]
    logger.info("Ridge alpha grid: %s", [f"{a:.3g}" for a in alphas])
    return alphas


def _model_output_dir(out_root: Path, model_id: str, family: str) -> Path:
    return out_root / family / safe_model_dir(model_id)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _embed_cache_path(
    cache_dir: Path,
    model_id: str,
    revision: str,
) -> Path:
    safe = safe_model_dir(model_id)
    rev  = revision.replace("/", "_")
    from neuroai.features.llm import embedding_cache_tag

    return cache_dir / safe / rev / f"embeddings{embedding_cache_tag(model_id)}.npz"


def _find_transformer_layers(model: Any) -> list:
    """Return the list of transformer decoder layers for hook-based extraction.

    Tries common attribute paths used by HuggingFace and custom architectures.
    """
    candidates = [
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
        "decoder.layers",
        "layers",
        "model.model.layers",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(
        f"Cannot find transformer layers in {type(model).__name__} "
        f"for hook-based hidden-state extraction."
    )


def _resolve_transformer_hidden_states(
    all_hidden_states: tuple,
    *,
    n_layers: int,
    model_name: str,
    warned: list[bool],
) -> tuple:
    """Drop the token-embedding layer, returning exactly ``n_layers`` tensors.

    Normally ``all_hidden_states[0]`` is the embedding layer and the rest are
    the ``n_layers`` decoder outputs, all sharing ``hidden_size``. Some models
    (OPT-350M's ``word_embed_proj_dim != hidden_size``; MoE "babylm"
    checkpoints) instead emit one tensor whose last dim differs from the rest
    — at the *end* of the tuple in current transformers versions, not
    necessarily at index 0 — so a positional ``[1:]`` slice can keep the
    mismatched tensor and drop a real layer instead. Filter by dimension
    instead of position to be robust to either layout (same fix as
    ``neuroai.benchmarks.fedorenko2016._resolve_transformer_hidden_states``).
    """
    dims = [int(t.shape[-1]) for t in all_hidden_states]
    if len(set(dims)) <= 1:
        return all_hidden_states[1:]

    from collections import Counter

    modal_dim = Counter(dims).most_common(1)[0][0]
    kept = tuple(t for t in all_hidden_states if int(t.shape[-1]) == modal_dim)
    if not warned[0]:
        logger.warning(
            "%s: non-uniform hidden dims across %d hidden-state tensors — "
            "keeping %d tensor(s) with dim=%d (embedding/output projection "
            "mismatch, e.g. OPT-350M or MoE babylm checkpoints)",
            model_name, len(all_hidden_states), len(kept), modal_dim,
        )
        warned[0] = True
    if len(kept) != n_layers:
        raise ValueError(
            f"{model_name}: expected {n_layers} uniform-dim hidden states after "
            f"filtering, got {len(kept)} (dims={sorted(set(dims))})"
        )
    return kept


@embedding_layer_selection
def extract_embeddings(
    model_id: str,
    sentences: list[str],
    *,
    revision: str = "main",
    load_model_id: str | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    trust_remote_code: bool = False,
    cache_dir: Path | None = None,
    n_params: float = 0.0,
) -> np.ndarray:
    """Return last-token embeddings for all sentences from every layer.

    Returns
    -------
    np.ndarray of shape (n_sentences, n_layers, hidden_size)
    """
    load_id = load_model_id or model_id
    rev_kwarg = revision if revision not in ("", "main") else None

    if cache_dir is not None:
        cache_path = _embed_cache_path(cache_dir, model_id, revision)
        if cache_path.is_file():
            data = np.load(cache_path, allow_pickle=True)
            cached_sents = data["sentences"].tolist()
            if cached_sents == sentences:
                logger.info("Embeddings cache hit: %s", cache_path)
                nlls  = data["sentence_nll"]   if "sentence_nll"   in data else np.full(len(sentences), np.nan)
                ntoks = data["sentence_ntoks"] if "sentence_ntoks" in data else np.zeros(len(sentences), dtype=np.int32)
                return data["embeddings"], nlls, ntoks
            logger.warning("Cache sentences mismatch — recomputing: %s", cache_path)

    if "cbr-rnn" in load_id.lower() or "cbr_rnn" in load_id.lower():
        # CBR-RNN is not an AutoModelForCausalLM (no `model_type` in its
        # config.json, can't load via _get_cached_causal_lm) and has a single
        # hidden-state layer — no layer dimension. Same special-casing as
        # neuroai.benchmarks.fedorenko2016's cbr-rnn branch: extract per-word
        # via the native CBR-RNN extractor, take the last word as the sentence
        # representation (matching this script's last-token pooling for every
        # other model), and skip perplexity/NLL (not wired up for CBR-RNN
        # anywhere in this codebase — NaN, same as fedorenko2016).
        from neuroai.features.cbr_rnn import contextual_word_embeddings_cbr_rnn

        logger.info("CBR-RNN: extracting %d sentence embeddings (last word, no NLL)", len(sentences))
        sent_embeds = [
            contextual_word_embeddings_cbr_rnn(
                sent.split(),
                load_id,
                device=device,
                context_words=48,
                cache_dir=None,
                section_id=0,
                lang="en",
            )[-1]
            for sent in sentences
        ]
        all_embeds = np.stack(sent_embeds, axis=0)[:, None, :].astype(np.float32)  # (n_sent, 1, d_hid)
        all_nlls = np.full(len(sentences), np.nan, dtype=np.float64)
        all_ntoks = np.zeros(len(sentences), dtype=np.int32)

        if cache_dir is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                embeddings=all_embeds,
                sentences=np.array(sentences, dtype=object),
                sentence_nll=all_nlls,
                sentence_ntoks=all_ntoks,
            )
            logger.info("Embeddings cached → %s", cache_path)

        return all_embeds, all_nlls, all_ntoks

    from neuroai.features.llm import (
        _get_cached_causal_lm,
        _model_input_device,
        config_dim_attr,
    )

    logger.info("Loading tokenizer: %s", load_id)
    load_kw: dict[str, Any] = {"output_hidden_states": True}
    if trust_remote_code:
        load_kw["trust_remote_code"] = True
    tokenizer, model = _get_cached_causal_lm(
        load_id,
        revision=rev_kwarg,
        device=device,
        load_kw=load_kw,
        tokenizer_no_revision=False,
        n_params=n_params,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # RIGHT-padding, not left: with left-padding, causal/recurrent models (Mamba,
    # Mamba2) process the padding tokens BEFORE any real content, and the padded
    # positions being zeroed (rather than truly absent) still perturbs the SSM's
    # recurrent state on the way in -- verified empirically (batched-vs-unbatched
    # NLL for a short sentence padded to a longer batch's length differed by up
    # to +56 nats/token for Mamba-130M, blowing corpus perplexity up to 1e14-1e27,
    # scaling with the amount of padding). Right-padding is immune by causality:
    # trailing padding can never influence the loss computed at real (earlier)
    # positions, for ANY causal architecture. This also makes the `last_tok`
    # formula below (attention_mask.sum(dim=1) - 1) correct -- it was written for
    # right-padding semantics but the tokenizer was left at "left", silently
    # extracting the WRONG (padding-adjacent) hidden state as "last token" for
    # every sentence shorter than its batch's longest one.
    tokenizer.padding_side = "right"
    input_device = _model_input_device(model)

    # Custom architectures (e.g. MoE "babylm" checkpoints' MoEPLMConfig) don't
    # always use the standard field names.
    n_layers = config_dim_attr(
        model.config, ("num_hidden_layers", "n_layer", "n_layers", "num_layers")
    )
    hidden_size = config_dim_attr(
        model.config, ("hidden_size", "n_embd", "d_model", "dim", "hidden_dim")
    )
    if "moep" in load_id.lower():
        # Per-layer decomposition isn't safe for this architecture (dynamic,
        # content-dependent expert width -- see the native_only hidden_states
        # override below); n_layers must match what that override actually
        # produces (1), not the config's real transformer depth, or
        # all_embeds below is allocated too large and every "layer" past the
        # first is silently left as zeros instead of real data.
        n_layers = 1
    n_sentences = len(sentences)
    logger.info(
        "Model loaded: %d layers × %d hidden — extracting %d sentences (batch=%d)",
        n_layers, hidden_size, n_sentences, batch_size,
    )

    import torch.nn.functional as F

    # `all_embeds` is allocated lazily below, from the first batch's actually
    # resolved hidden-state width — not from config `hidden_size` here. Some
    # custom architectures (e.g. MoE "babylm" checkpoints) report a config
    # hidden_size that doesn't match the real transformer layer output width
    # (the inverse of OPT-350M's embedding/hidden_size split), which would
    # otherwise silently mis-size this array.
    all_embeds: np.ndarray | None = None
    all_nlls   = np.full(n_sentences, np.nan, dtype=np.float64)   # mean NLL per token
    all_ntoks  = np.zeros(n_sentences, dtype=np.int32)            # non-padding tokens scored
    dim_warned = [False]

    hook_captures: list[torch.Tensor] = []
    hooks: list = []

    def _make_hook(captures: list) -> Any:
        def _hook(module: Any, inp: Any, output: Any) -> None:
            h = output[0] if isinstance(output, tuple) else output
            captures.append(h.detach())
        return _hook

    for start in range(0, n_sentences, batch_size):
        end = min(start + batch_size, n_sentences)
        batch_sents = sentences[start:end]

        enc = tokenizer(
            batch_sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,   # 6-word sentences need at most ~12 tokens; 64 is safe
        )
        input_ids      = enc["input_ids"].to(input_device)
        attention_mask = enc["attention_mask"].to(input_ids.device)

        with torch.no_grad():
            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            except TypeError:
                # Some custom models don't accept use_cache kwarg
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

        # Some trust_remote_code architectures (e.g. GPT-BERT) ignore/default
        # differently on return_dict and hand back a plain tuple instead of a
        # ModelOutput — same rationale as neuroai.features.llm's
        # _forward_hidden_states. Fall through to the hook-based path below
        # rather than crashing on `.hidden_states` access.
        if "moep" in load_id.lower():
            # MoE "babylm" checkpoints route tokens through experts of
            # DIFFERENT width depending on content, not just depending on
            # layer index -- individual per-token hidden-state slices within
            # one tensor can therefore differ in width even after
            # _resolve_transformer_hidden_states' per-layer filtering (that
            # only catches whole-tensor mismatches), which crashed here with
            # "could not broadcast input array from shape (192,) into shape
            # (384,)". No per-layer decomposition is safe for this
            # architecture, so — same rationale as logit_lens.py's
            # native_only fallback for this exact model — skip it entirely
            # and use only the model's own final hidden state, hooked
            # directly off the last transformer layer (guaranteed to match
            # hidden_size, since it's what feeds the LM head) as a single
            # "layer". out.hidden_states is None for this architecture (it
            # ignores output_hidden_states=True the same way FLA/
            # stickbreaking do below), so there's no attribute to read this
            # from either -- has to be a hook.
            if start == 0:
                last_layer = _find_transformer_layers(model)[-1:]
                for layer in last_layer:
                    hooks.append(layer.register_forward_hook(_make_hook(hook_captures)))
            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=False, use_cache=False, return_dict=True)
                except TypeError:
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=False, return_dict=True)
            hidden_states = tuple(hook_captures[:1])
            hook_captures.clear()
        elif isinstance(out, tuple) or out.hidden_states is None:
            # Custom architectures (e.g. FLA / stickbreaking) ignore
            # output_hidden_states=True — fall back to forward hooks.
            # Register hooks on the first batch only; reuse for subsequent batches.
            if start == 0:
                layers = _find_transformer_layers(model)[:n_layers]
                for layer in layers:
                    hooks.append(layer.register_forward_hook(_make_hook(hook_captures)))
            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=False, use_cache=False, return_dict=True)
                except TypeError:
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=False, return_dict=True)
            hidden_states = tuple(hook_captures[:n_layers])
            hook_captures.clear()
        else:
            # Standard path: tuple (embed, layer1, …, layerN) → drop embed
            hidden_states = _resolve_transformer_hidden_states(
                out.hidden_states, n_layers=n_layers, model_name=model_id, warned=dim_warned,
            )

        if all_embeds is None:
            actual_hidden_size = int(hidden_states[0].shape[-1])
            if actual_hidden_size != hidden_size:
                logger.warning(
                    "%s: config hidden_size=%d but resolved hidden-state width=%d — "
                    "allocating embeddings with the resolved width.",
                    model_id, hidden_size, actual_hidden_size,
                )
            all_embeds = np.zeros((n_sentences, n_layers, actual_hidden_size), dtype=np.float32)

        # Last non-padding token per sentence (correct for right-padding: the
        # real tokens are always the first attention_mask.sum() positions).
        last_tok = attention_mask.sum(dim=1) - 1  # (batch,)

        for layer_idx, hs in enumerate(hidden_states):
            # hs: (batch, seq_len, hidden)
            for bi in range(end - start):
                tok_idx = int(last_tok[bi].item())
                all_embeds[start + bi, layer_idx] = (
                    hs[bi, tok_idx].float().cpu().numpy()
                )

        # Per-sentence mean NLL from logits (causal LM: predict token t+1 from t)
        # out.logits is always returned by CausalLM — no extra compute needed.
        # Tuple-safe: architectures whose forward() hands back a plain tuple
        # even with return_dict=True (see the isinstance(out, tuple) guard
        # above) put logits first — labels are never passed at this call
        # site, so there's no leading loss element to skip.
        logits = (out[0] if isinstance(out, tuple) else out.logits).float()  # (batch, seq, vocab)
        shift_logits = logits[:, :-1, :]              # (batch, seq-1, vocab)
        shift_ids    = input_ids[:, 1:]               # (batch, seq-1)
        shift_mask   = attention_mask[:, 1:].float()  # (batch, seq-1)

        nll_per_tok = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_ids.reshape(-1),
            reduction="none",
        ).reshape(shift_logits.size(0), -1)            # (batch, seq-1)

        nll_per_tok = nll_per_tok * shift_mask
        n_tok = shift_mask.sum(dim=1).clamp(min=1)    # (batch,)
        mean_nll = (nll_per_tok.sum(dim=1) / n_tok).cpu().numpy()  # (batch,)

        for bi in range(end - start):
            all_nlls[start + bi]  = float(mean_nll[bi])
            all_ntoks[start + bi] = int(n_tok[bi].item())

        del logits, shift_logits, shift_ids, shift_mask, nll_per_tok

        if (start // batch_size) % 5 == 0:
            logger.info("  Sentences %d–%d / %d", start, end, n_sentences)

    for h in hooks:   # remove forward hooks registered for FLA-style models
        h.remove()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Some custom-code models (e.g. ALiBi) produce NaN/Inf activations due to
    # numerical instability. Replace them with 0 before caching so downstream
    # ridge regression does not crash (same handling as fedorenko2016.py /
    # eeg_erp.py's extract_sentence_features).
    n_bad = int(np.sum(~np.isfinite(all_embeds)))
    if n_bad > 0:
        logger.warning(
            "extract_embeddings: %d non-finite values in embeddings "
            "(%s, rev=%s) — replacing with 0",
            n_bad, model_id, revision or "main",
        )
        all_embeds = np.nan_to_num(all_embeds, nan=0.0, posinf=0.0, neginf=0.0)

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=all_embeds,
            sentences=np.array(sentences, dtype=object),
            sentence_nll=all_nlls,
            sentence_ntoks=all_ntoks,
        )
        logger.info("Embeddings cached → %s", cache_path)

    return all_embeds, all_nlls, all_ntoks  # embeddings + per-sentence NLL + token counts


# ---------------------------------------------------------------------------
# Ridge regression per condition
# ---------------------------------------------------------------------------

def _regress_layer_cpu(
    X_all: np.ndarray,   # (n_cond, n_layers, hidden_size)
    Y: np.ndarray,       # (n_cond, n_rois)
    groups: np.ndarray,
    n_splits: int,
    alphas: list[float],
) -> list[list]:
    """CPU fallback: sequential layer loop using sklearn RidgeCV (inner 5-fold CV)."""
    n_layers = X_all.shape[1]
    layer_scores = []
    for layer_idx in range(n_layers):
        scores = cv_ridge_score_multioutput(
            X_all[:, layer_idx, :], Y,
            groups=groups,
            n_splits=n_splits,
            alpha=alphas,
            scale=True,
            standardize_target=True,
        )
        layer_scores.append(scores)
    return layer_scores


def _regress_layer_gpu(
    X_all: np.ndarray,   # (n_cond, n_layers, hidden_size)
    Y: np.ndarray,       # (n_cond, n_rois)
    groups: np.ndarray,
    n_splits: int,
    alphas: list[float],
    n_layer_workers: int = 32,
) -> list[list]:
    """GPU path: himalaya torch_cuda backend + one CUDA stream per layer batch."""
    from himalaya.backend import set_backend as _set_backend
    _set_backend("torch_cuda")

    n_layers = X_all.shape[1]
    n_workers = min(n_layer_workers, n_layers)
    layer_scores: list[list | None] = [None] * n_layers

    def _run_one(layer_idx: int) -> list:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            scores = cv_ridge_score_multioutput_gpu(
                X_all[:, layer_idx, :], Y,
                groups=groups,
                n_splits=n_splits,
                alpha=alphas,
                scale=True,
                standardize_target=True,
            )
        stream.synchronize()
        return scores

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, li): li for li in range(n_layers)}
        for fut in as_completed(futures):
            layer_idx = futures[fut]
            layer_scores[layer_idx] = fut.result()

    return layer_scores  # type: ignore[return-value]


def _run_regression(
    embeddings: np.ndarray,     # (n_sentences_all, n_layers, hidden_size)
    dataset: TuckuteDataset,
    all_item_ids: np.ndarray,
    alphas: list[float],
    k_folds: int,
    device: str = "cpu",
    n_layer_workers: int = 32,
) -> list[dict]:
    """k-fold ridge CV for each condition × layer, averaged over subjects.

    Uses a stacked-subject strategy when all subjects share the same sentences
    (B condition): Y is stacked to (n_sentences, n_subjects × n_rois) and one
    regression call replaces N sequential per-subject calls (~10× speedup).

    Falls back to sequential per-subject for conditions where subjects see
    different sentence subsets (D / S: each sentence seen by only 2-3
    participants, so the intersection across all subjects is empty).

    GPU (himalaya torch_cuda + CUDA streams) when device starts with 'cuda'.
    """
    rows: list[dict] = []
    n_layers = embeddings.shape[1]
    n_rois   = len(dataset.rois)
    use_gpu  = device.startswith("cuda") and torch.cuda.is_available()

    id_to_idx = {int(iid): i for i, iid in enumerate(all_item_ids)}

    for cond in EVAL_CONDITIONS:
        subj_data  = dataset.get_condition_per_subject(cond)
        subj_ids   = list(subj_data.keys())
        n_subjects = len(subj_ids)

        # Check whether all subjects share the same sentence set
        id_sets    = [set(int(i) for i in subj_data[s][2]) for s in subj_ids]
        common_ids = sorted(set.intersection(*id_sets)) if id_sets else []
        use_stacked = len(common_ids) >= k_folds

        if use_stacked:
            mean_pr, mean_sr, mean_alpha = _run_regression_stacked(
                cond, subj_ids, subj_data, common_ids,
                embeddings, id_to_idx, alphas, k_folds,
                n_layers, n_rois, use_gpu, n_layer_workers,
            )
        else:
            mean_pr, mean_sr, mean_alpha = _run_regression_per_subject(
                cond, subj_ids, subj_data,
                embeddings, id_to_idx, alphas, k_folds,
                n_layers, n_rois, use_gpu, n_layer_workers,
            )

        # Log alpha stats at last layer
        last_alphas = mean_alpha[-1]
        fin = last_alphas[np.isfinite(last_alphas)]
        if fin.size:
            logger.info(
                "  cond=%s chosen alpha (last layer, %d ROIs, %d subjects): "
                "median=%.3g  mean=%.3g  max=%.3g",
                cond, n_rois, n_subjects,
                float(np.median(fin)), float(fin.mean()), float(fin.max()),
            )

        if use_gpu:
            torch.cuda.empty_cache()

        for layer_idx in range(n_layers):
            for roi_idx, roi in enumerate(dataset.rois):
                rows.append({
                    "cond":        cond,
                    "roi":         roi,
                    "layer":       layer_idx,
                    "pearsonr":    float(mean_pr[layer_idx, roi_idx]),
                    "spearmanr":   float(mean_sr[layer_idx, roi_idx]),
                    "ridge_alpha": float(mean_alpha[layer_idx, roi_idx]),
                })

    return rows


def _run_regression_stacked(
    cond: str,
    subj_ids: list[str],
    subj_data: dict,
    common_ids: list[int],
    embeddings: np.ndarray,
    id_to_idx: dict,
    alphas: list[float],
    k_folds: int,
    n_layers: int,
    n_rois: int,
    use_gpu: bool,
    n_layer_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One regression call with Y stacked across subjects → (n_layers, n_rois)."""
    n_subjects = len(subj_ids)

    # Align each subject's BOLD to common_ids and stack as extra columns
    Y_list: list[np.ndarray] = []
    for s in subj_ids:
        _, bold_s, item_ids_s = subj_data[s]
        row_of = {int(iid): r for r, iid in enumerate(item_ids_s)}
        Y_list.append(bold_s[[row_of[iid] for iid in common_ids]])  # (n_common, n_rois)

    Y_stacked = np.hstack(Y_list)                            # (n_common, n_subj * n_rois)
    x_idx     = np.array([id_to_idx[iid] for iid in common_ids], dtype=np.intp)
    X_all     = embeddings[x_idx]                            # (n_common, n_layers, hidden)

    # Drop rows where ANY subject has NaN in ANY ROI
    valid_mask = ~np.any(np.isnan(Y_stacked), axis=1)
    if not valid_mask.all():
        n_dropped   = int((~valid_mask).sum())
        nan_subjids = [
            subj_ids[si] for si in range(n_subjects)
            if np.any(np.isnan(Y_list[si][~valid_mask]))
        ]
        logger.warning(
            "  cond=%s: dropping %d/%d sentences with NaN BOLD (subjects: %s)",
            cond, n_dropped, len(valid_mask), nan_subjids,
        )
        Y_stacked = Y_stacked[valid_mask]
        X_all     = X_all[valid_mask]

    n        = len(Y_stacked)
    n_splits = min(k_folds, n)
    groups   = np.arange(n) % k_folds

    logger.info(
        "  cond=%s  n_subjects=%d  n=%d  Y_stacked=%s  n_splits=%d  backend=%s  [stacked]",
        cond, n_subjects, n, Y_stacked.shape, n_splits,
        "gpu" if use_gpu else "cpu",
    )

    if use_gpu:
        layer_scores = _regress_layer_gpu(X_all, Y_stacked, groups, n_splits, alphas, n_layer_workers)
    else:
        layer_scores = _regress_layer_cpu(X_all, Y_stacked, groups, n_splits, alphas)

    # Reshape (n_layers, n_subj * n_rois) → (n_layers, n_subj, n_rois) → mean over subjects
    pr_flat    = np.array([[s.pearsonr    for s in ls] for ls in layer_scores])
    sr_flat    = np.array([[s.spearmanr   for s in ls] for ls in layer_scores])
    alpha_flat = np.array([[s.ridge_alpha for s in ls] for ls in layer_scores])

    mean_pr    = np.nanmean(pr_flat.reshape(n_layers, n_subjects, n_rois),    axis=1)
    mean_sr    = np.nanmean(sr_flat.reshape(n_layers, n_subjects, n_rois),    axis=1)
    mean_alpha = np.nanmedian(alpha_flat.reshape(n_layers, n_subjects, n_rois), axis=1)
    return mean_pr, mean_sr, mean_alpha


def _run_regression_per_subject(
    cond: str,
    subj_ids: list[str],
    subj_data: dict,
    embeddings: np.ndarray,
    id_to_idx: dict,
    alphas: list[float],
    k_folds: int,
    n_layers: int,
    n_rois: int,
    use_gpu: bool,
    n_layer_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequential per-subject regression, averaged at the end → (n_layers, n_rois)."""
    logger.info(
        "  cond=%s  n_subjects=%d  backend=%s  [per-subject]",
        cond, len(subj_ids), "gpu" if use_gpu else "cpu",
    )

    all_pr: list[np.ndarray] = []
    all_sr: list[np.ndarray] = []
    all_alpha: list[np.ndarray] = []

    for subj_id in subj_ids:
        _, bold_s, item_ids_s = subj_data[subj_id]
        indices = np.array([id_to_idx[int(iid)] for iid in item_ids_s], dtype=np.intp)
        X_s = embeddings[indices]
        Y_s = bold_s

        valid = ~np.any(np.isnan(Y_s), axis=1)
        if not valid.all():
            logger.warning("    subj=%s: dropping %d/%d sentences with NaN BOLD",
                           subj_id, int((~valid).sum()), len(valid))
            X_s = X_s[valid]
            Y_s = Y_s[valid]

        n_s = len(Y_s)
        if n_s < k_folds:
            logger.warning("    subj=%s: only %d valid sentences — skipping", subj_id, n_s)
            continue

        n_splits = min(k_folds, n_s)
        groups   = np.arange(n_s) % k_folds
        logger.info("    subj=%s  n=%d  Y.shape=%s  n_splits=%d", subj_id, n_s, Y_s.shape, n_splits)

        if use_gpu:
            layer_scores = _regress_layer_gpu(X_s, Y_s, groups, n_splits, alphas, n_layer_workers)
        else:
            layer_scores = _regress_layer_cpu(X_s, Y_s, groups, n_splits, alphas)

        all_pr.append(np.array([[s.pearsonr    for s in ls] for ls in layer_scores]))
        all_sr.append(np.array([[s.spearmanr   for s in ls] for ls in layer_scores]))
        all_alpha.append(np.array([[s.ridge_alpha for s in ls] for ls in layer_scores]))

    mean_pr    = np.nanmean(all_pr,    axis=0)
    mean_sr    = np.nanmean(all_sr,    axis=0)
    mean_alpha = np.nanmedian(all_alpha, axis=0)
    return mean_pr, mean_sr, mean_alpha


# ---------------------------------------------------------------------------
# Per-checkpoint runner
# ---------------------------------------------------------------------------

def _summary_row(
    ckpt: CheckpointItem,
    model_id: str,
    n_params: float,
    family: str,
    roi_rows: list[dict],
) -> dict:
    """Collapse per-roi/layer rows into a single summary dict for the scores CSV."""
    row: dict[str, Any] = {
        "model_id":  model_id,
        "revision":  ckpt.revision,
        "step":      ckpt.step,
        "n_params":  n_params,
        "family":    family,
    }
    df = pd.DataFrame(roi_rows)

    for cond in EVAL_CONDITIONS:
        cdf = df[df["cond"] == cond]
        if cdf.empty:
            continue
        # Best layer per ROI (by spearmanr — primary metric) → mean over ROIs
        best_layer_per_roi = cdf.groupby("roi")["spearmanr"].idxmax()
        best_sr = cdf.loc[best_layer_per_roi.values].groupby("roi")["spearmanr"].max()
        best_pr = cdf.loc[best_layer_per_roi.values].groupby("roi")["pearsonr"].max()

        row[f"{cond}_mean_best_spearmanr"] = float(best_sr.mean())
        row[f"{cond}_mean_best_pearsonr"]  = float(best_pr.mean())   # secondary

        for roi in LANG_ROIS:
            if roi in best_sr.index:
                row[f"{cond}_{roi}_best_spearmanr"] = float(best_sr[roi])
                row[f"{cond}_{roi}_best_pearsonr"]  = float(best_pr[roi])

    return row


# ---------------------------------------------------------------------------
# Per-sentence perplexity saving
# ---------------------------------------------------------------------------

def _save_sentence_ppl(
    output_dir: Path,
    model_id: str,
    revision: str,
    dataset: TuckuteDataset,
    all_item_ids: np.ndarray,
    sentence_nlls: np.ndarray,
    sentence_ntoks: np.ndarray,
) -> None:
    """Append per-sentence surprisal to CSV and update corpus-level perplexity.json."""
    import json

    # Build item_id → (cond, sentence) lookup
    sent_df = dataset._sentences_df.set_index("item_id")

    ppl_csv = output_dir / "tuckute_sentence_ppl.csv"
    ppl_rows = []
    for i, iid in enumerate(all_item_ids):
        nll  = float(sentence_nlls[i])
        ntok = int(sentence_ntoks[i])
        row_info = sent_df.loc[int(iid)] if int(iid) in sent_df.index else None
        ppl_rows.append({
            "model_id":       model_id,
            "revision":       revision,
            "item_id":        int(iid),
            "cond":           row_info["cond"] if row_info is not None else "",
            "sentence":       row_info["sentence"] if row_info is not None else "",
            "mean_nll":       round(nll, 6),
            "n_tokens":       ntok,
            "perplexity":     round(float(np.exp(nll)), 4) if np.isfinite(nll) else float("nan"),
        })

    new_df = pd.DataFrame(ppl_rows)
    if ppl_csv.is_file():
        existing = pd.read_csv(ppl_csv)
        # Drop rows for this revision if rerunning
        if "revision" in existing.columns:
            existing = existing[existing["revision"] != revision]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(ppl_csv, index=False)
    logger.info("Sentence PPL saved → %s  (n=%d)", ppl_csv, len(new_df))

    # Corpus-level perplexity: exp(total_nll / total_tokens) — properly token-weighted
    valid = np.isfinite(sentence_nlls)
    if valid.any():
        total_nll  = float((sentence_nlls[valid] * sentence_ntoks[valid]).sum())
        total_toks = int(sentence_ntoks[valid].sum())
        corpus_ppl = float(np.exp(total_nll / max(total_toks, 1)))
        record = {
            "model_id":    model_id,
            "revision":    revision,
            "corpus":      "tuckute",
            "perplexity":  round(corpus_ppl, 4),
            "n_sentences": int(valid.sum()),
            "n_tokens":    total_toks,
        }
        ppl_json = output_dir / "perplexity.json"
        ppl_json.write_text(json.dumps(record, indent=2))
        logger.info(
            "Corpus PPL saved → %s  (ppl=%.3f  n_tok=%d)",
            ppl_json, corpus_ppl, total_toks,
        )


def run_one_checkpoint(
    ckpt: CheckpointItem,
    *,
    model_id: str,
    n_params: float,
    family: str,
    model_cfg: dict,
    dataset: TuckuteDataset,
    alphas: list[float],
    k_folds: int,
    batch_size: int,
    device: str,
    n_layer_workers: int,
    output_dir: Path,
    embed_cache_dir: Path | None,
) -> dict | None:
    load_model_id = resolve_load_model_id(model_cfg, ckpt.revision)
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    logger.info(
        "[%s] Tuckute ── %s (step=%d) ──",
        model_id, ckpt.revision, ckpt.step,
    )

    # 1. Extract embeddings + per-sentence NLL for all 2000 sentences
    all_sentences, _, all_item_ids = dataset.get_condition("all")
    try:
        embeddings, sentence_nlls, sentence_ntoks = extract_embeddings(
            model_id=model_id,
            sentences=all_sentences,
            revision=ckpt.revision,
            load_model_id=load_model_id,
            batch_size=batch_size,
            device=device,
            trust_remote_code=trust_remote_code,
            cache_dir=embed_cache_dir,
            n_params=n_params,
        )
    except Exception:
        logger.error("[%s] embedding extraction failed", model_id, exc_info=True)
        traceback.print_exc()
        return None
    finally:
        clear_hf_model_cache()
        free_gpu()

    # 1b. Save per-sentence perplexity CSV + corpus-level perplexity.json
    _save_sentence_ppl(
        output_dir=output_dir,
        model_id=model_id,
        revision=ckpt.revision,
        dataset=dataset,
        all_item_ids=all_item_ids,
        sentence_nlls=sentence_nlls,
        sentence_ntoks=sentence_ntoks,
    )

    # 2. Ridge regression per condition × layer
    logger.info("[%s] Running ridge regression (all conditions × %d layers)…",
                model_id, embeddings.shape[1])
    try:
        roi_rows = _run_regression(
            embeddings, dataset, all_item_ids, alphas, k_folds,
            device=device, n_layer_workers=n_layer_workers,
        )
    except Exception:
        logger.error("[%s] regression failed", model_id, exc_info=True)
        traceback.print_exc()
        return None

    # 3. Annotate rows with checkpoint metadata and append to CSVs
    meta = {
        "model_id": model_id,
        "revision": ckpt.revision,
        "step":     ckpt.step,
        "n_params": n_params,
        "family":   family,
    }
    roi_csv    = output_dir / "tuckute_by_roi.csv"
    layer_csv  = output_dir / "tuckute_layerwise.csv"

    for r in roi_rows:
        append_row(roi_csv, {**meta, **r})

    # Layerwise: mean pearsonr over ROIs per (cond, layer)
    df_roi = pd.DataFrame(roi_rows)
    for (cond, layer), grp in df_roi.groupby(["cond", "layer"]):
        append_row(layer_csv, {
            **meta,
            "cond":   cond,
            "layer":  layer,
            "pearsonr_mean": float(grp["pearsonr"].mean()),
            "spearmanr_mean": float(grp["spearmanr"].mean()),
        })

    summary = _summary_row(ckpt, model_id, n_params, family, roi_rows)
    return summary


# ---------------------------------------------------------------------------
# Family sweep
# ---------------------------------------------------------------------------

def run_model_sweep(
    checkpoints: list[CheckpointItem],
    *,
    model_id: str,
    n_params: float,
    family: str,
    model_cfg: dict,
    dataset: TuckuteDataset,
    alphas: list[float],
    k_folds: int,
    batch_size: int,
    device: str,
    n_layer_workers: int,
    output_dir: Path,
    embed_cache_dir: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / "tuckute_scores.csv"
    existing   = load_existing_csv(scores_csv)

    for ckpt in checkpoints:
        if already_computed(existing, ckpt.revision):
            logger.info("[%s] %s already done — skipping", model_id, ckpt.revision)
            continue

        summary = run_one_checkpoint(
            ckpt,
            model_id=model_id,
            n_params=n_params,
            family=family,
            model_cfg=model_cfg,
            dataset=dataset,
            alphas=alphas,
            k_folds=k_folds,
            batch_size=batch_size,
            device=device,
            n_layer_workers=n_layer_workers,
            output_dir=output_dir,
            embed_cache_dir=embed_cache_dir,
        )
        if summary is not None:
            append_row(scores_csv, summary)
            existing = load_existing_csv(scores_csv)


# ---------------------------------------------------------------------------
# Summary / plots
# ---------------------------------------------------------------------------

def run_summary(output_dir: Path, family: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    search_root = output_dir / family if family else output_dir
    rows: list[dict] = []
    for csv_path in sorted(search_root.glob("**/tuckute_scores.csv")):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                rows.append(df.tail(1).iloc[0].to_dict())  # final checkpoint
        except Exception:
            pass

    if not rows:
        logger.warning("No tuckute_scores.csv found under %s", search_root)
        return

    df = pd.DataFrame(rows)
    summary_dir = (output_dir / family / "summary") if family else (output_dir / "summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    cond_cols = {c: f"{c}_mean_best_pearsonr" for c in EVAL_CONDITIONS}
    available = {c: col for c, col in cond_cols.items() if col in df.columns}
    if not available:
        logger.warning("No condition score columns found in summary data")
        return

    # Training curves (if checkpoint sweep with >1 step)
    for scores_csv in sorted(search_root.glob("**/tuckute_scores.csv")):
        try:
            full_df = pd.read_csv(scores_csv)
        except Exception:
            continue
        if "step" not in full_df.columns or full_df["step"].nunique() < 2:
            continue
        if "model_id" not in full_df.columns:
            continue
        model_id = str(full_df["model_id"].iloc[0])
        model_dir = scores_csv.parent
        n = len(available)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (cond, col) in zip(axes, available.items()):
            if col not in full_df.columns:
                continue
            g = full_df.sort_values("step")
            ax.plot(g["step"], g[col], "o-", lw=1.5)
            ax.set_title(f"cond={cond}", fontsize=10)
            ax.set_xlabel("Training step", fontsize=8)
            ax.set_ylabel("Mean best-layer Pearson r", fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Tuckute — Training curves ({model_id})", fontsize=12)
        fig.tight_layout()
        fig.savefig(model_dir / "tuckute_training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Summary done → %s", summary_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/embedding/neural/tuckute.yaml")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None,
                   help="SLURM array task index within the family")
    p.add_argument("--all", action="store_true",
                   help="Run all checkpoints for the family sequentially")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--list", action="store_true",
                   help="List models/checkpoints for the family and exit")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_layer_workers", type=int, default=32,
                   help="CUDA stream workers for layer-parallel GPU ridge (default 8)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    ds_cfg  = cfg.get("data", {})
    reg_cfg = cfg.get("regression", {})
    mod_cfg = cfg.get("model", {})

    data_root = Path(os.path.expandvars(str(ds_cfg.get("data_root",
        "data/tuckute_fmri/osfstorage"))))
    out_root = Path(args.out_dir or os.path.expandvars(
        str(cfg.get("output", {}).get("save_dir", "outputs/tuckute/embedding_sweep"))
    ))
    embed_cache_dir_raw = ds_cfg.get("embed_cache_dir")
    embed_cache_dir = (
        Path(os.path.expandvars(str(embed_cache_dir_raw)))
        if embed_cache_dir_raw else out_root / ".embed_cache"
    )

    alphas          = _resolve_alphas(reg_cfg.get("alpha"))
    k_folds         = int(ds_cfg.get("k_folds", 4))
    batch_size      = args.batch_size or int(mod_cfg.get("batch_size", 64))
    device          = args.device
    n_layer_workers = args.n_layer_workers

    logger.info(
        "Hyperparameters — alpha_grid=%s  k_folds=%d  batch_size=%d  "
        "device=%s  n_layer_workers=%d  backend=%s",
        [f"{a:.3g}" for a in alphas], k_folds, batch_size, device, n_layer_workers,
        "himalaya/gpu" if device.startswith("cuda") else "sklearn/cpu",
    )

    if args.summary_only:
        run_summary(out_root, family=args.family)
        return

    if args.family is None:
        sys.exit("--family is required (or --summary-only)")

    family = args.family

    # ── list mode ─────────────────────────────────────────────────────────
    if args.list:
        from neuroai.models.registry import flatten_family_tasks
        tasks = flatten_family_tasks(family)
        print(f"{'idx':>4}  {'model_id':<50}  {'revision':<35}  step")
        print("-" * 100)
        for i, t in enumerate(tasks):
            print(f"{i:>4}  {t.model_id:<50}  {t.revision:<35}  {t.step}")
        print(f"\nTotal: {len(tasks)} tasks  →  --array-index 0–{len(tasks)-1}")
        return

    # ── resolve which model(s) to run ─────────────────────────────────────
    if args.array_index is not None:
        task = resolve_family_task(family, args.array_index)
        model_configs = [c for c in registry_model_configs(task.registry_family)
                         if c["model_id"] == task.model_id]
        if not model_configs:
            sys.exit(f"model_id {task.model_id!r} not found in registry family {family!r}")
        model_cfg = model_configs[0]
        model_id  = task.model_id
        n_params  = float(model_cfg.get("n_params", 0))
        checkpoints = [
            CheckpointItem(rev, step, tokens)
            for rev, step, tokens in build_checkpoint_items(model_cfg)
        ]
        run_groups = [(model_id, n_params, model_cfg, checkpoints)]

    elif args.all:
        run_groups = []
        for model_cfg in registry_model_configs(family):
            mid = model_cfg["model_id"]
            run_groups.append((
                mid,
                float(model_cfg.get("n_params", 0)),
                model_cfg,
                [
                    CheckpointItem(rev, step, tokens)
                    for rev, step, tokens in build_checkpoint_items(model_cfg)
                ],
            ))
    else:
        sys.exit("Provide --array-index N, --all, --list, or --summary-only")

    # ── load dataset once (shared across all models in this job) ──────────
    logger.info("Loading TuckuteDataset from %s", data_root)
    dataset = TuckuteDataset(data_root)

    # ── run ───────────────────────────────────────────────────────────────
    for model_id, n_params, model_cfg, checkpoints in run_groups:
        output_dir = _model_output_dir(out_root, model_id, family)
        logger.info("=== %s | family=%s | checkpoints=%d ===",
                    model_id, family, len(checkpoints))
        run_model_sweep(
            checkpoints,
            model_id=model_id,
            n_params=n_params,
            family=family,
            model_cfg=model_cfg,
            dataset=dataset,
            alphas=alphas,
            k_folds=k_folds,
            batch_size=batch_size,
            device=device,
            n_layer_workers=n_layer_workers,
            output_dir=output_dir,
            embed_cache_dir=embed_cache_dir,
        )

    run_summary(out_root, family=family)
    logger.info("Done.")


if __name__ == "__main__":
    main()
