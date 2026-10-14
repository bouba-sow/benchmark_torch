#!/usr/bin/env python3
"""Compute perplexity of all LPP registry models on the LPP English story text.

Perplexity is computed with a stride-based sliding window over the concatenation
of all 9 LPP English story sections.  Results are saved as JSON under:

    <out_dir>/<model_safe>/perplexity.json

Usage
-----
    # List all models and their array indices
    python scripts/perplexity/lpp.py --list

    # Run one model by SLURM array index (primary usage)
    python scripts/perplexity/lpp.py --array_index 3 --device cuda

    # Run one model directly
    python scripts/perplexity/lpp.py --model_id facebook/opt-125m --device cuda

    # Collect existing JSON files into a summary CSV
    python scripts/perplexity/lpp.py --summarize
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import torch

from neuroai.models.registry import LPP_FAMILIES
from neuroai.lpp_llm_features import load_story_text

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

N_SECTIONS = 9
LANG = "en"

# Default stride-based window; Mamba has no positional limit so we use 4096.
DEFAULT_MAX_LENGTH = 2048
MAMBA_MAX_LENGTH = 4096


# ---------------------------------------------------------------------------
# Flat model list (one entry per model_id, main/last revision only)
# ---------------------------------------------------------------------------

def flat_model_list() -> list[dict]:
    """All registry models, one entry per unique model_id (final checkpoint only)."""
    seen: set[str] = set()
    models: list[dict] = []
    for family, specs in LPP_FAMILIES.items():
        for spec in specs:
            mid = spec.model_id
            if mid in seen:
                continue
            seen.add(mid)
            models.append(
                {
                    "model_id": mid,
                    "label": spec.label,
                    "n_params": spec.n_params,
                    "family": family,
                }
            )
    return models


def _safe_dir(model_id: str) -> str:
    return model_id.replace("/", "_")


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def load_lpp_text(bids_root: Path, lang: str = LANG) -> list[str]:
    """Return list of section texts (9 sections)."""
    texts: list[str] = []
    for sec in range(1, N_SECTIONS + 1):
        try:
            text = load_story_text(bids_root=bids_root, lang=lang, section=sec)
            texts.append(text)
        except FileNotFoundError as e:
            logger.warning("Section %d not found: %s", sec, e)
    if not texts:
        raise RuntimeError(f"No LPP text sections found under {bids_root}")
    logger.info("Loaded %d/%d LPP sections (%s)", len(texts), N_SECTIONS, lang)
    return texts


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_id: str,
    n_params: float,
    device: str,
):
    """Load a causal LM with automatic multi-GPU sharding for large models.

    Uses :func:`neuroai.features.llm._get_cached_causal_lm` so Ministral-3 (VLM
  checkpoint layout) and other architecture-specific loaders work correctly.
    """
    from neuroai.features.llm import _get_cached_causal_lm

    load_kw: dict = {
        "trust_remote_code": True,
        # forgetting-transformer checkpoints may have non-fatal shape mismatches
        "ignore_mismatched_sizes": True,
    }

    logger.info("Loading tokenizer + model: %s", model_id)
    tokenizer, model = _get_cached_causal_lm(
        model_id,
        revision=None,
        device=device,
        load_kw=load_kw,
        tokenizer_no_revision=True,
        n_params=n_params,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── DEBUG: weight checksums — temporary, to confirm different repos really
    # load different weights (remove once the hard-n-gram/cbr-rnn investigation
    # is resolved). ──────────────────────────────────────────────────────────
    try:
        emb = model.get_input_embeddings().weight
        emb_f = emb.detach().float()
        logger.warning(
            "[DEBUG-WEIGHTS] %s: embed sum=%.6f mean=%.8f std=%.8f shape=%s",
            model_id, emb_f.sum().item(), emb_f.mean().item(), emb_f.std().item(), tuple(emb.shape),
        )
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            tied = lm_head.weight.data_ptr() == emb.data_ptr()
            lmh_f = lm_head.weight.detach().float()
            logger.warning(
                "[DEBUG-WEIGHTS] %s: lm_head tied_to_embed=%s sum=%.6f mean=%.8f std=%.8f",
                model_id, tied, lmh_f.sum().item(), lmh_f.mean().item(), lmh_f.std().item(),
            )
        # First transformer block's first Linear-like weight, whatever it's called.
        for name, param in model.named_parameters():
            if param.dim() == 2 and "embed" not in name.lower() and "lm_head" not in name.lower():
                p_f = param.detach().float()
                logger.warning(
                    "[DEBUG-WEIGHTS] %s: first_2d_param=%s sum=%.6f mean=%.8f std=%.8f shape=%s",
                    model_id, name, p_f.sum().item(), p_f.mean().item(), p_f.std().item(), tuple(param.shape),
                )
                break
    except Exception as exc:  # noqa: BLE001 - debug only, never fail the run
        logger.warning("[DEBUG-WEIGHTS] checksum failed for %s: %s", model_id, exc)

    logger.info("Model loaded: %s (dtype=%s)", model_id, next(model.parameters()).dtype)
    return tokenizer, model


def _input_device(model) -> str:
    """First device for the model (handles single-GPU and device_map cases)."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first = next(iter(model.hf_device_map.values()))
        val = first[0] if isinstance(first, (list, tuple)) else first
        # device_map values can be bare ints (0, 1) — normalise to "cuda:N"
        if isinstance(val, int):
            return f"cuda:{val}"
        s = str(val)
        return f"cuda:{s}" if s.isdigit() else s
    return str(next(model.parameters()).device)


def _max_length_for_model(model, model_id: str) -> int:
    """Return the context window to use for perplexity sliding window."""
    cfg = getattr(model, "config", None)
    # Mamba-style models have no positional limit
    is_mamba = "mamba" in model_id.lower()
    if is_mamba:
        return MAMBA_MAX_LENGTH
    for attr in ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"):
        val = getattr(cfg, attr, None)
        if val and isinstance(val, int) and val > 0:
            return min(val, DEFAULT_MAX_LENGTH)
    return DEFAULT_MAX_LENGTH


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_length: int,
    stride: int | None = None,
) -> dict:
    """Stride-based sliding-window perplexity on concatenated text.

    Only the non-overlapping suffix of each window contributes to the loss,
    matching the standard HuggingFace perplexity recipe.
    """
    if stride is None:
        stride = max_length // 2

    full_text = " ".join(texts)
    import logging as _logging
    logger.info("Tokenising %d chars of text …", len(full_text))
    # Suppress the "sequence longer than max_length" warning: it fires because we
    # tokenize the full text in one call, but we never run it through the model
    # at full length — the sliding window handles that below.
    _tok_logger = _logging.getLogger("transformers.tokenization_utils_base")
    _prev_level = _tok_logger.level
    _tok_logger.setLevel(_logging.ERROR)
    try:
        enc = tokenizer(full_text, return_tensors="pt")
    finally:
        _tok_logger.setLevel(_prev_level)
    input_ids: torch.Tensor = enc["input_ids"]
    seq_len = input_ids.size(1)
    logger.info("Total tokens: %d  |  max_length=%d  stride=%d", seq_len, max_length, stride)

    inp_device = _input_device(model)
    nlls: list[torch.Tensor] = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # tokens to evaluate (non-overlapping part)

        ids = input_ids[:, begin:end].to(inp_device)
        labels = ids.clone()
        # Mask the overlapping prefix — use actual window length, not max_length,
        # so the last (shorter) window doesn't get fully masked and produce nan loss.
        window_len = end - begin
        labels[:, : window_len - trg_len] = -100

        with torch.no_grad():
            if type(model).__name__ == "GPTBERTForCausalLM":
                # BabyLM-community's gpt-bert-* forward(labels=...) is built
                # for a genuine masked-LM batch (predictions gathered at
                # masked positions only): passed a full-sequence labels
                # tensor, subword_prediction keeps its size-1 leading batch
                # dim un-gathered, so their own F.cross_entropy(subword_
                # prediction, gold_labels) fails with "batch_size (1) !=
                # target batch_size (seq_len)". Bypass their labels path
                # entirely: forward without labels, then do the same
                # *shifted* next-token cross-entropy every other
                # architecture gets internally via labels= — this exact
                # shift convention (position p's logits predict the token at
                # p+1) is the one already verified correct for this model in
                # neuroai/features/logit_lens.py's native_only surprisal
                # fallback.
                out = model(ids, use_cache=False, return_dict=True)
                logits = out.logits
                if logits.dim() == 3:
                    logits = logits[0]  # (seq_len, vocab) — drop batch dim
                shift_logits = logits[:-1, :].float()
                shift_labels = labels[0, 1:]  # overlap-prefix mask already applied above
                loss = torch.nn.functional.cross_entropy(
                    shift_logits, shift_labels, ignore_index=-100,
                ).cpu()
            else:
                # return_dict=True forced explicitly — some trust_remote_code
                # architectures default to a plain tuple otherwise (loss is
                # the tuple's first element, not `.loss`), and every standard
                # HF forward() already accepts this kwarg, so it's a no-op
                # for the rest.
                try:
                    out = model(ids, labels=labels, use_cache=False, return_dict=True)
                except TypeError:
                    out = model(ids, labels=labels, return_dict=True)
                # loss is mean NLL over unmasked tokens; multiply back to get
                # sum. Some custom models (e.g. stickbreaking) return
                # per-token loss tensors instead of a scalar — flatten to a
                # scalar mean first.
                loss = out.loss.cpu()
                if loss.dim() > 0:
                    loss = loss.reshape(-1).mean()
            nlls.append(loss * trg_len)

            # ── DEBUG: first-window logit stats — temporary, to confirm the
            # forward pass actually depends on the model's weights/input rather
            # than collapsing to a fixed/degenerate output. ────────────────────
            if begin == 0:
                try:
                    logits = getattr(out, "logits", None)
                    if logits is not None:
                        lf = logits.detach().float()
                        row0 = lf[0, -1]
                        logger.warning(
                            "[DEBUG-LOGITS] first window: shape=%s mean=%.6f std=%.6f "
                            "last_tok_argmax=%d last_tok_top1_logit=%.4f loss=%.6f",
                            tuple(logits.shape), lf.mean().item(), lf.std().item(),
                            int(row0.argmax().item()), row0.max().item(), loss.item(),
                        )
                except Exception as exc:  # noqa: BLE001 - debug only
                    logger.warning("[DEBUG-LOGITS] failed: %s", exc)

        prev_end = end
        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = math.exp(total_nll.item() / seq_len)
    logger.info("Perplexity: %.4f  (total_nll=%.4f, n_tokens=%d)", ppl, total_nll.item(), seq_len)
    return {"perplexity": ppl, "n_tokens": int(seq_len)}


# ---------------------------------------------------------------------------
# Main computation entry point
# ---------------------------------------------------------------------------

def run_model(
    model_entry: dict,
    bids_root: Path,
    out_dir: Path,
    device: str,
    lang: str = LANG,
    force: bool = False,
    max_length_override: int | None = None,
) -> dict | None:
    model_id = model_entry["model_id"]
    safe = _safe_dir(model_id)
    out_path = out_dir / safe / "perplexity.json"

    if out_path.is_file() and not force:
        existing = json.loads(out_path.read_text())
        if "error" not in existing:
            logger.info("Already done: %s — skipping (use --force to recompute)", model_id)
            return existing
        logger.warning("Previous run had error — recomputing: %s", out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    texts = load_lpp_text(bids_root, lang)

    try:
        tokenizer, model = load_model_and_tokenizer(model_id, model_entry["n_params"], device)
    except Exception as exc:
        logger.error("FAILED %s (model load): %s", model_id, exc, exc_info=True)
        out_path.write_text(json.dumps({
            "model_id": model_id, "label": model_entry["label"],
            "n_params": model_entry["n_params"], "family": model_entry["family"],
            "lang": lang, "error": f"load: {exc}",
        }, indent=2))
        return None

    max_length = _max_length_for_model(model, model_id)
    if max_length_override is not None:
        max_length = min(max_length_override, max_length)
        logger.info("max_length capped to %d (override=%d)", max_length, max_length_override)

    # If accelerate CPU-offloaded some layers, reduce max_length to limit activation memory
    if getattr(model, "hf_device_map", None) and any(
        v == "cpu" for v in model.hf_device_map.values()
    ):
        max_length = min(max_length, 512)
        logger.warning(
            "%s has CPU-offloaded layers — reducing max_length to %d to avoid OOM",
            model_id, max_length,
        )

    torch_dtype = str(next(model.parameters()).dtype)

    try:
        result = compute_perplexity(model, tokenizer, texts, device, max_length)
    except Exception as exc:
        logger.error("FAILED %s: %s", model_id, exc, exc_info=True)
        error_record = {
            "model_id": model_id,
            "label": model_entry["label"],
            "n_params": model_entry["n_params"],
            "family": model_entry["family"],
            "lang": lang,
            "error": str(exc),
        }
        out_path.write_text(json.dumps(error_record, indent=2))
        return None
    finally:
        # Explicitly free GPU memory before loading the next model
        del model
        del tokenizer
        from neuroai.features.llm import clear_hf_model_cache

        clear_hf_model_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = {
        "model_id": model_id,
        "label": model_entry["label"],
        "n_params": model_entry["n_params"],
        "family": model_entry["family"],
        "lang": lang,
        "perplexity": result["perplexity"],
        "n_tokens": result["n_tokens"],
        "max_length": max_length,
        "stride": max_length // 2,
        "device": device,
        "torch_dtype": torch_dtype,
        "n_sections": len(texts),
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved → %s", out_path)
    return record


# ---------------------------------------------------------------------------
# Summary collection
# ---------------------------------------------------------------------------

def collect_summary(out_dir: Path) -> None:
    """Gather all perplexity.json files into a summary CSV."""
    import pandas as pd

    rows = []
    for p in sorted(out_dir.rglob("perplexity.json")):
        d = json.loads(p.read_text())
        if "error" not in d:
            rows.append(d)
        else:
            logger.warning("Skipping failed model: %s (%s)", d.get("model_id"), d.get("error"))

    if not rows:
        print("No completed perplexity results found.")
        return

    df = pd.DataFrame(rows).sort_values("n_params")
    out_csv = out_dir / "lpp_perplexity_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Summary: {len(df)} models → {out_csv}")
    print(df[["label", "family", "perplexity", "n_tokens"]].to_string(index=False))


# ---------------------------------------------------------------------------
# CBR-RNN perplexity (custom loader — not AutoModelForCausalLM)
# ---------------------------------------------------------------------------

def compute_cbr_rnn_perplexity_over_texts(
    model_id: str, texts: list[str], device: str,
) -> dict:
    """Dataset-agnostic core of compute_cbr_rnn_perplexity below: CBR-RNN
    (``neuroai.models.cbr_rnn.CbrRnn``) is a custom PyTorchModelHubMixin
    recurrent architecture, not ``AutoModelForCausalLM`` -- ``compute_
    perplexity``'s HF-generate-style sliding window doesn't apply, so this
    reimplements the same "prepend BOS, slide at stride = max_seq_len // 2,
    score only the non-overlapping suffix of each window" recipe directly
    against the model's own forward signature, taking ``texts`` (already
    loaded by the caller) instead of a bids_root -- lets every dataset's own
    perplexity script (Tuckute/Fedorenko2016/NS-fMRI included, which lack
    their own compute_cbr_rnn_perplexity) reuse this one core instead of
    reimplementing it per dataset. Returns the same keys as
    ``compute_perplexity`` (perplexity, n_tokens, max_length, stride) so
    callers can build their own dataset-specific JSON record uniformly with
    both code paths.
    """
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from neuroai.models.cbr_rnn import CbrRnn

    full_text = " ".join(texts)

    logger.info("Loading CBR-RNN tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading CBR-RNN model: %s", model_id)
    model = CbrRnn.from_pretrained(model_id)
    model.eval()
    model.to(device)

    bos_id: int = tokenizer.bos_token_id or 0
    max_seq_len: int = model.max_seq_len        # 256 from config
    stride: int = max_seq_len // 2

    logger.info("Tokenising %d chars …", len(full_text))
    token_ids: list[int] = tokenizer.encode(full_text, add_special_tokens=False)
    inputs: list[int] = [bos_id] + token_ids[:-1]
    targets: list[int] = token_ids
    n_tokens = len(targets)
    logger.info("Total tokens: %d  |  max_seq_len=%d  stride=%d", n_tokens, max_seq_len, stride)

    total_nll = 0.0
    n_scored = 0

    for start in range(0, len(inputs), stride):
        end = min(start + max_seq_len, len(inputs))
        win_inputs = inputs[start:end]
        win_targets = targets[start:end]

        ids = torch.tensor([win_inputs], device=device)
        tgt = torch.tensor(win_targets, device=device)

        with torch.no_grad():
            logits, _, _ = model(ids)

        logits = logits.squeeze(0)  # [window_len, vocab_size]
        nll_per_tok = F.cross_entropy(logits, tgt, reduction="none")

        score_start_in_window = 0 if start == 0 else stride
        for j in range(score_start_in_window, len(win_targets)):
            total_nll += nll_per_tok[j].item()
            n_scored += 1

        if end == len(inputs):
            break

    ppl = math.exp(total_nll / n_scored)
    logger.info("CBR-RNN perplexity: %.4f  (n_scored=%d)", ppl, n_scored)

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "perplexity": ppl, "n_tokens": n_scored, "max_length": max_seq_len, "stride": stride,
    }


def compute_cbr_rnn_perplexity(
    model_id: str,
    bids_root: Path,
    out_dir: Path,
    device: str,
    lang: str = LANG,
    force: bool = False,
    label: str = "CBR-RNN",
) -> dict | None:
    """Compute perplexity for the CBR-RNN model.

    Uses the same sliding-window recipe as the original HumanWMBenchmark
    surprisal.py: prepend BOS, slide at stride = max_seq_len // 2, score
    only the non-overlapping suffix of each window.
    """
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from neuroai.models.cbr_rnn import CbrRnn

    safe = _safe_dir(model_id)
    out_path = out_dir / safe / "perplexity.json"
    if out_path.is_file() and not force:
        existing = json.loads(out_path.read_text())
        if "error" not in existing:
            logger.info("Already done: %s — skipping (use --force to recompute)", model_id)
            return existing
        logger.warning("Previous run had error — recomputing: %s", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    texts = load_lpp_text(bids_root, lang)
    full_text = " ".join(texts)

    logger.info("Loading CBR-RNN tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading CBR-RNN model: %s", model_id)
    model = CbrRnn.from_pretrained(model_id)
    model.eval()
    model.to(device)

    bos_id: int = tokenizer.bos_token_id or 0
    max_seq_len: int = model.max_seq_len        # 256 from config
    stride: int = max_seq_len // 2

    logger.info("Tokenising %d chars …", len(full_text))
    token_ids: list[int] = tokenizer.encode(full_text, add_special_tokens=False)
    # Build causal-LM style: input = [BOS, tok_0, ..., tok_{n-2}]
    #                         target = [tok_0, ..., tok_{n-1}]
    inputs: list[int] = [bos_id] + token_ids[:-1]
    targets: list[int] = token_ids
    n_tokens = len(targets)
    logger.info("Total tokens: %d  |  max_seq_len=%d  stride=%d", n_tokens, max_seq_len, stride)

    total_nll = 0.0
    n_scored = 0

    for start in range(0, len(inputs), stride):
        end = min(start + max_seq_len, len(inputs))
        win_inputs = inputs[start:end]
        win_targets = targets[start:end]

        ids = torch.tensor([win_inputs], device=device)
        tgt = torch.tensor(win_targets, device=device)

        with torch.no_grad():
            logits, _, _ = model(ids)

        logits = logits.squeeze(0)  # [window_len, vocab_size]
        # per-token NLL (no reduction)
        nll_per_tok = F.cross_entropy(logits, tgt, reduction="none")

        # Score only the non-overlapping suffix of each window
        score_start_in_window = 0 if start == 0 else stride
        for j in range(score_start_in_window, len(win_targets)):
            total_nll += nll_per_tok[j].item()
            n_scored += 1

        if end == len(inputs):
            break

    ppl = math.exp(total_nll / n_scored)
    logger.info("CBR-RNN perplexity: %.4f  (n_scored=%d)", ppl, n_scored)

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    record = {
        "model_id": model_id,
        "label": label,
        "n_params": 17_360_688,
        "family": "cbr_rnn",
        "lang": lang,
        "perplexity": ppl,
        "n_tokens": n_scored,
        "max_length": max_seq_len,
        "stride": stride,
        "device": device,
        "torch_dtype": "float32",
        "n_sections": len(texts),
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved → %s", out_path)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--array_index", type=int, default=None,
                   help="SLURM array task index (0-based)")
    p.add_argument("--model_id", default=None,
                   help="Run a specific model by HuggingFace ID")
    p.add_argument("--family", default=None,
                   help="Restrict to one registry family (e.g. stickbreaking, alibi, forgetting)")
    p.add_argument("--architecture", default=None, choices=("cbr_rnn",),
                   help="Custom architecture for non-AutoModelForCausalLM models")
    p.add_argument("--device", default="cuda",
                   help="Torch device (default: cuda)")
    p.add_argument("--max_length", type=int, default=None,
                   help="Override sliding-window max_length (default: model-specific, up to 2048). "
                        "Set to 4096 on H100 for faster throughput.")
    p.add_argument("--lang", default=LANG,
                   help="Language for LPP text (default: en)")
    p.add_argument("--bids_root", type=Path,
                   default=Path("data/lpp_fmri"),
                   help="LPP BIDS root (needs annotation/ for TextGrids)")
    p.add_argument("--out_dir", type=Path,
                   default=Path("outputs_torch/lpp/perplexity"),
                   help="Output directory (default: outputs_torch/lpp/perplexity)")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if output already exists")
    p.add_argument("--all", action="store_true",
                   help="Run all registry models sequentially (skip already-done)")
    p.add_argument("--list", action="store_true",
                   help="Print the flat model list with indices and exit")
    p.add_argument("--summarize", action="store_true",
                   help="Collect existing JSON files into a summary CSV and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_models = flat_model_list()
    if args.family == "cbr_rnn":
        args.architecture = "cbr_rnn"
    models = [
        m for m in all_models
        if m.get("family") != "cbr_rnn"
        and (args.family is None or m.get("family") == args.family)
    ]

    if args.list:
        fam_label = f" (family={args.family})" if args.family else ""
        print(f"{'idx':>4}  {'family':<10}  {'label':<30}  {'n_params':>15}  model_id")
        print("-" * 90)
        for i, m in enumerate(models):
            print(f"{i:>4}  {m['family']:<10}  {m['label']:<30}  {m['n_params']:>15.3e}  {m['model_id']}")
        print(f"\nTotal{fam_label}: {len(models)} models  →  --array 0-{len(models)-1}")
        return

    out_dir = args.out_dir.resolve()

    if args.summarize:
        collect_summary(out_dir)
        return

    bids_root = args.bids_root.resolve()
    if not (args.list or args.summarize) and not bids_root.is_dir():
        sys.exit(f"LPP BIDS root not found: {bids_root}")

    # ── CBR-RNN shortcut (uses custom loader, not AutoModelForCausalLM) ──────
    if args.architecture == "cbr_rnn":
        model_id = args.model_id or "cclark/cbr-rnn"
        cbr_entry = next(
            (m for m in all_models if m.get("model_id") == model_id and m.get("family") == "cbr_rnn"),
            None,
        )
        label = cbr_entry["label"] if cbr_entry else "CBR-RNN"
        logger.info("=== CBR-RNN perplexity: %s ===", model_id)
        compute_cbr_rnn_perplexity(
            model_id, bids_root, out_dir, args.device, args.lang, force=args.force, label=label
        )
        return

    if args.all:
        fam_label = f" (family={args.family})" if args.family else ""
        logger.info("Running %d models%s sequentially", len(models), fam_label)
        for i, entry in enumerate(models):
            logger.info("── [%d/%d] %s ──", i + 1, len(models), entry["label"])
            run_model(entry, bids_root, out_dir, args.device, args.lang, force=args.force,
                      max_length_override=args.max_length)
        collect_summary(out_dir)
        return

    # Resolve which model to run
    if args.array_index is not None:
        if args.array_index < 0 or args.array_index >= len(models):
            sys.exit(f"array_index {args.array_index} out of range [0, {len(models)-1}]")
        entry = models[args.array_index]
    elif args.model_id is not None:
        matches = [m for m in models if m["model_id"] == args.model_id]
        if not matches:
            sys.exit(f"Model {args.model_id!r} not found in registry")
        entry = matches[0]
    else:
        sys.exit("Provide --all, --array_index, --model_id, --architecture, --list, or --summarize")

    logger.info("=== LPP perplexity: %s (%s) ===", entry["label"], entry["model_id"])
    run_model(entry, bids_root, out_dir, args.device, args.lang, force=args.force,
              max_length_override=args.max_length)


if __name__ == "__main__":
    main()
