"""Shared corpus-perplexity computation for the embedding-ridge sweep scripts.

``run_<dataset>_embed_sweep.py`` (geco/meco/provo/ucl_reading/zuco_reading)
extract hidden-state *embeddings* per checkpoint for ridge regression — they
never compute logits/loss, so perplexity isn't a free byproduct there (unlike
the LME pipeline's ``prefetch_<dataset>_surprisal.py``, which derives it from
logit-lens surprisal already being computed). This module gives each embed
sweep a small, self-contained way to also save perplexity for every
checkpoint it processes, using the exact same sliding-window recipe as
``scripts/perplexity/lpp.py`` (stride = max_length // 2, only the
non-overlapping suffix of each window scores).

Output layout matches the existing ``prefetch_<dataset>_surprisal.py``
convention so results merge into the same tree:
    <perplexity_dir>/<model_safe>/<revision>_<dataset_label>_perplexity.json
    <perplexity_dir>/<dataset_label>_perplexity_summary.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 4096


def _compute_corpus_perplexity(
    model, tokenizer, text: str, device: str, max_length: int, stride: int | None = None,
) -> dict:
    """Stride-based sliding-window perplexity on ``text``.

    Only the non-overlapping suffix of each window contributes to the loss —
    same recipe as ``compute_lpp_perplexity.compute_perplexity``.
    """
    import torch

    if stride is None:
        stride = max_length // 2
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"][0]
    n_tokens = int(input_ids.size(0))

    nlls = []
    n_scored = 0
    prev_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + max_length, n_tokens)
        trg_len = end - prev_end
        ids = input_ids[begin:end].unsqueeze(0).to(device)
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(ids, labels=target_ids)
            neg_log_likelihood = out.loss.float() * trg_len
        nlls.append(neg_log_likelihood.cpu())
        n_scored += trg_len
        prev_end = end
        if end == n_tokens:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / n_scored).item()
    return {"perplexity": ppl, "n_tokens": n_scored}


def _max_length_for_model(model, max_length_override: int | None) -> int:
    if max_length_override is not None:
        return int(max_length_override)
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"):
        val = getattr(cfg, attr, None)
        if val and isinstance(val, int) and val > 0:
            return min(int(val), DEFAULT_MAX_LENGTH)
    return DEFAULT_MAX_LENGTH


def save_checkpoint_perplexity(
    *,
    model_id: str,
    revision: str | None,
    label: str,
    n_params: float | None,
    family: str | None,
    device: str,
    text: str,
    perplexity_dir: str | Path,
    dataset_label: str,
    max_length: int | None = None,
    trust_remote_code: bool = False,
    force: bool = False,
) -> None:
    """Compute and save corpus perplexity for one (model_id, revision).

    Self-contained model load (independent of whatever the embedding
    extraction cached) — safe/idempotent: skips checkpoints already saved
    successfully, recomputes ones whose saved JSON has an ``error`` field.

    Loads via ``neuroai.features.llm._get_cached_causal_lm`` rather than a bare
    ``AutoModelForCausalLM.from_pretrained`` — required for checkpoints that
    bypass the standard HF auto-loading path entirely (MoLM/ModuleFormer,
    forgetting-transformer, FoX; see that function's ``_MOLM_DIRECT_PREFIX`` /
    ``_FORGETTING_DIRECT_PREFIX`` / ``_FOX_DIRECT_PREFIX`` dispatch and their
    loader docstrings for why: none of these publish a ``config.json`` with an
    ``auto_map``, so ``trust_remote_code=True`` alone has nothing to resolve
    to, regardless of the installed ``transformers`` version). Using the same
    loader as the embedding-extraction path also means every other special
    case handled there (VLM prefix-strip, Pythia-2.8B sharding, device_map
    sizing, gpt-bert's bf16 workaround, ...) is covered here too, rather than
    reimplemented ad hoc.
    """
    import torch

    from neuroai.features.llm import _get_cached_causal_lm, _model_input_device, clear_hf_model_cache

    perplexity_dir = Path(perplexity_dir)
    safe = model_id.replace("/", "_")
    rev_safe = (revision or "main").replace("/", "_")
    out_dir = perplexity_dir / safe
    out_path = out_dir / f"{rev_safe}_{dataset_label}_perplexity.json"

    if out_path.is_file() and not force:
        existing = json.loads(out_path.read_text())
        if "error" not in existing:
            logger.info("perplexity already done: %s @ %s — skipping", model_id, revision)
            return
        logger.warning("Previous perplexity run had error — recomputing: %s", out_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        load_kw: dict[str, object] = {"torch_dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32}
        if trust_remote_code:
            load_kw["trust_remote_code"] = True
        tokenizer, model = _get_cached_causal_lm(
            model_id, revision=revision, device=device, load_kw=load_kw, n_params=n_params,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        input_device = _model_input_device(model)
        eff_max_length = _max_length_for_model(model, max_length)
        result = _compute_corpus_perplexity(model, tokenizer, text, input_device, eff_max_length)
    except Exception as exc:
        logger.error("perplexity FAILED for %s @ %s: %s", model_id, revision, exc, exc_info=True)
        out_path.write_text(json.dumps({
            "model_id": model_id, "revision": revision, "label": label,
            "n_params": n_params, "family": family, "dataset": dataset_label,
            "error": str(exc),
        }, indent=2))
        return
    finally:
        if model is not None:
            clear_hf_model_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = {
        "model_id": model_id, "revision": revision, "label": label,
        "n_params": n_params, "family": family, "dataset": dataset_label,
        "perplexity": result["perplexity"], "n_tokens": result["n_tokens"],
        "max_length": eff_max_length,
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved perplexity %.4f for %s @ %s -> %s",
                result["perplexity"], model_id, revision, out_path)

    summary_csv = perplexity_dir / f"{dataset_label}_perplexity_summary.csv"
    existing_df = pd.read_csv(summary_csv) if summary_csv.is_file() else pd.DataFrame()
    if not existing_df.empty:
        keep = ~(
            (existing_df["model_id"] == model_id)
            & (existing_df["revision"].astype(str) == str(revision))
        )
        existing_df = existing_df.loc[keep]
    out_df = pd.concat([existing_df, pd.DataFrame([record])], ignore_index=True)
    out_df.to_csv(summary_csv, index=False)
