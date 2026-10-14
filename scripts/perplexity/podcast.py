#!/usr/bin/env python3
"""Compute perplexity of all LPP registry models on the Podcast transcript.

Uses the same stride-based sliding-window approach as
``scripts/perplexity/lpp.py``.  The corpus is the concatenation of
all words in ``stimuli/podcast_transcript.csv``.  Results are saved as JSON:

    <out_dir>/<model_safe>/perplexity.json

Usage
-----
    # List all models
    python scripts/perplexity/podcast.py --list

    # Run all registry models sequentially (skip already-done)
    python scripts/perplexity/podcast.py --all --device cuda \\
        --bids_root data/podcast

    # Run one model directly
    python scripts/perplexity/podcast.py \\
        --model_id facebook/opt-125m --device cuda --bids_root data/podcast

    # Collect results into a summary CSV
    python scripts/perplexity/podcast.py --summarize

Jean Zay (all models, one job)::

    sbatch scripts/slurm/jz/perplexity/podcast.sh
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import torch

# BabyLM-community's gpt-bert-* models trip three separate transformers
# 5.13.1 incompatibilities in their vendor trust_remote_code (same ones hit
# and fixed in scripts/run/surprisal/neural/podcast_extract.py — see that file's
# comments for the full story): (1) a raw torch.dtype in their config dict
# breaks transformers' own unconditional `logger.info(f"Model config {config}")`
# during AutoConfig.from_pretrained; (2) their custom __init__ never calls
# self.post_init(), so `all_tied_weights_keys` is never set; (3) their
# `_keys_to_ignore_on_load_unexpected` is an old-style list where this
# transformers version expects a set. All three patches are no-ops for every
# other already-working architecture.
import json as _json

_orig_json_default = _json.JSONEncoder.default


def _json_default_stringify_unknown(self, o):
    try:
        return _orig_json_default(self, o)
    except TypeError:
        return str(o)


_json.JSONEncoder.default = _json_default_stringify_unknown

import transformers.modeling_utils as _modeling_utils

_orig_pretrained_model_init = _modeling_utils.PreTrainedModel.__init__


def _pretrained_model_init_with_tied_weights_fallback(self, *args, **kwargs):
    _orig_pretrained_model_init(self, *args, **kwargs)
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = {}
    if isinstance(getattr(self, "_keys_to_ignore_on_load_unexpected", None), list):
        self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected)
    if isinstance(getattr(self, "_keys_to_ignore_on_load_missing", None), list):
        self._keys_to_ignore_on_load_missing = set(self._keys_to_ignore_on_load_missing)


_modeling_utils.PreTrainedModel.__init__ = _pretrained_model_init_with_tied_weights_fallback

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUMMARY_CSV = "podcast_perplexity_summary.csv"


# ---------------------------------------------------------------------------
# Shared helpers re-used from perplexity/lpp.py
# ---------------------------------------------------------------------------

def _import_lpp_perplexity():
    path = Path(__file__).resolve().with_name("lpp.py")
    spec = importlib.util.spec_from_file_location("_lpp_perplexity", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import shared helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lpp = _import_lpp_perplexity()
flat_model_list = _lpp.flat_model_list
_safe_dir = _lpp._safe_dir
load_model_and_tokenizer = _lpp.load_model_and_tokenizer
compute_perplexity = _lpp.compute_perplexity
_max_length_for_model = _lpp._max_length_for_model


def flat_checkpoint_list() -> list[dict]:
    """All registry models × all checkpoints (one entry per model × revision).

    For Pythia: expands every step in ``LPP_PYTHIA_STEPS`` → revision ``step{N}``.
    For OLMo-2: expands every revision string.
    For all other families: single entry with revision ``"main"``.

    Each entry has keys: model_id, load_model_id, revision, label, n_params, family.
    ``load_model_id`` may differ from ``model_id`` for OLMo-2 early-training repos.
    """
    from neuroai.models.registry import (
        LPP_FAMILIES,
        LPP_PYTHIA_STEPS,
        resolve_load_model_id,
        model_spec_to_cfg,
    )
    from neuroai.models._checkpoints import pythia_checkpoint_revision

    entries: list[dict] = []
    for family, specs in LPP_FAMILIES.items():
        for spec in specs:
            cfg = model_spec_to_cfg(spec, family)
            ck = cfg.get("checkpoints", {})
            ck_type = ck.get("type", "main")

            if ck_type == "steps":
                steps = ck.get("steps") or list(LPP_PYTHIA_STEPS)
                for step in steps:
                    rev = pythia_checkpoint_revision(step)
                    entries.append({
                        "model_id": spec.model_id,
                        "load_model_id": spec.model_id,
                        "revision": rev,
                        "label": spec.label,
                        "n_params": spec.n_params,
                        "family": family,
                        "step": step,
                    })
            elif ck_type == "revisions":
                revisions = ck.get("revisions") or []
                for rev in revisions:
                    load_id = resolve_load_model_id(cfg, rev)
                    entries.append({
                        "model_id": spec.model_id,
                        "load_model_id": load_id,
                        "revision": rev,
                        "label": spec.label,
                        "n_params": spec.n_params,
                        "family": family,
                    })
            else:
                entries.append({
                    "model_id": spec.model_id,
                    "load_model_id": spec.model_id,
                    "revision": "main",
                    "label": spec.label,
                    "n_params": spec.n_params,
                    "family": family,
                })
    return entries


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_podcast_text(bids_root: Path) -> str:
    """Return the full podcast transcript as a single space-joined string."""
    import pandas as pd
    transcript_path = bids_root / "stimuli" / "podcast_transcript.csv"
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Podcast transcript not found: {transcript_path}")
    df = pd.read_csv(transcript_path)
    words = df["word"].dropna().astype(str).tolist()
    text = " ".join(words)
    logger.info("Podcast corpus: %d words, %d chars", len(words), len(text))
    return text


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_entry: dict,
    bids_root: Path,
    out_dir: Path,
    device: str,
    force: bool = False,
    max_length_override: int | None = None,
) -> dict | None:
    model_id = model_entry["model_id"]
    load_model_id = model_entry.get("load_model_id", model_id)
    revision = model_entry.get("revision", "main")
    safe = _safe_dir(model_id)
    # Each revision gets its own subdirectory to avoid collisions across steps.
    if revision and revision != "main":
        out_path = out_dir / safe / revision / "perplexity.json"
    else:
        out_path = out_dir / safe / "perplexity.json"

    if out_path.is_file() and not force:
        existing = json.loads(out_path.read_text())
        if "error" not in existing:
            logger.info("Already done: %s@%s — skipping", model_id, revision)
            return existing
        logger.warning("Previous run had error — recomputing: %s", out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = load_podcast_text(bids_root)

    rev_kwarg = revision if revision and revision != "main" else None

    try:
        # load_model_and_tokenizer doesn't accept revision; call from_pretrained directly
        # for checkpoint steps by patching the model load inline.
        if rev_kwarg is None and "gpt-bert" in load_model_id.lower():
            # BabyLM-community's gpt-bert-* models aren't bf16-safe (custom
            # DWA accumulator does an in-place float32 buffer .set_() —
            # see run/surprisal/neural/podcast_extract.py's `dtype` handling for
            # the same issue in the LME pipeline), so load_model_and_tokenizer's
            # bf16 default crashes on the first real forward pass. Force
            # float32 explicitly for just this family.
            from neuroai.features.llm import _get_cached_causal_lm

            tokenizer, model = _get_cached_causal_lm(
                load_model_id,
                revision=None,
                device=device,
                load_kw={"trust_remote_code": True, "torch_dtype": torch.float32},
                tokenizer_no_revision=True,
                n_params=model_entry["n_params"],
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif rev_kwarg is None:
            tokenizer, model = load_model_and_tokenizer(load_model_id, model_entry["n_params"], device)
        else:
            from neuroai.features.llm import (
                _apply_device_map_if_needed,
                _load_hf_tokenizer,
                _resolve_causal_lm_class,
            )
            tokenizer = _load_hf_tokenizer(load_model_id, {})
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            load_kw: dict = {}
            if device.startswith("cuda") and torch.cuda.is_available():
                _apply_device_map_if_needed(load_kw, device, n_params=model_entry["n_params"])
                if "device_map" not in load_kw:
                    load_kw["torch_dtype"] = torch.float32
            model_cls = _resolve_causal_lm_class(load_model_id, {})
            model = model_cls.from_pretrained(
                load_model_id, revision=rev_kwarg, trust_remote_code=True, **load_kw
            )
            model.eval()
            if "device_map" not in load_kw:
                model = model.to(device)
    except Exception as exc:
        logger.error("FAILED %s@%s (model load): %s", model_id, revision, exc, exc_info=True)
        out_path.write_text(json.dumps({
            "model_id": model_id, "revision": revision,
            "label": model_entry["label"],
            "n_params": model_entry["n_params"], "family": model_entry["family"],
            "corpus": "podcast", "error": f"load: {exc}",
        }, indent=2))
        return None

    max_length = _max_length_for_model(model, model_id)
    if max_length_override is not None:
        max_length = min(max_length_override, max_length)
        logger.info("max_length capped to %d (override=%d)", max_length, max_length_override)
    if getattr(model, "hf_device_map", None) and any(
        v == "cpu" for v in model.hf_device_map.values()
    ):
        max_length = min(max_length, 512)
        logger.warning("%s has CPU-offloaded layers — reducing max_length to %d",
                       model_id, max_length)

    torch_dtype = str(next(model.parameters()).dtype)

    try:
        result = compute_perplexity(model, tokenizer, [text], device, max_length)
    except Exception as exc:
        logger.error("FAILED %s@%s: %s", model_id, revision, exc, exc_info=True)
        out_path.write_text(json.dumps({
            "model_id": model_id, "revision": revision,
            "label": model_entry["label"],
            "n_params": model_entry["n_params"], "family": model_entry["family"],
            "corpus": "podcast", "error": str(exc),
        }, indent=2))
        return None
    finally:
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = {
        "model_id": model_id,
        "revision": revision,
        "label": model_entry["label"],
        "n_params": model_entry["n_params"],
        "family": model_entry["family"],
        "corpus": "podcast",
        "perplexity": result["perplexity"],
        "n_tokens": result["n_tokens"],
        "max_length": max_length,
        "stride": max_length // 2,
        "device": device,
        "torch_dtype": torch_dtype,
    }
    if "step" in model_entry:
        record["step"] = model_entry["step"]
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved → %s  (ppl=%.2f)", out_path, record["perplexity"])
    return record


# ---------------------------------------------------------------------------
# CBR-RNN
# ---------------------------------------------------------------------------

def compute_cbr_rnn_perplexity(
    model_id: str,
    bids_root: Path,
    out_dir: Path,
    device: str,
    force: bool = False,
    label: str = "CBR-RNN",
) -> dict | None:
    import math
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from neuroai.models.cbr_rnn import CbrRnn

    safe = _safe_dir(model_id)
    out_path = out_dir / safe / "perplexity.json"
    if out_path.is_file() and not force:
        existing = json.loads(out_path.read_text())
        if "error" not in existing:
            logger.info("Already done: %s — skipping", model_id)
            return existing
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = load_podcast_text(bids_root)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = CbrRnn.from_pretrained(model_id)
    model.eval()
    model.to(device)

    bos_id: int = tokenizer.bos_token_id or 0
    max_seq_len: int = model.max_seq_len
    stride: int = max_seq_len // 2

    token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    inputs: list[int] = [bos_id] + token_ids[:-1]
    targets: list[int] = token_ids
    logger.info("CBR-RNN: %d tokens, max_seq_len=%d, stride=%d",
                len(targets), max_seq_len, stride)

    total_nll = 0.0
    n_scored = 0
    for start in range(0, len(inputs), stride):
        end = min(start + max_seq_len, len(inputs))
        ids = torch.tensor([inputs[start:end]], device=device)
        tgt = torch.tensor(targets[start:end], device=device)
        with torch.no_grad():
            logits, _, _ = model(ids)
        logits = logits.squeeze(0)
        nll_per_tok = F.cross_entropy(logits, tgt, reduction="none")
        score_start = 0 if start == 0 else stride
        for j in range(score_start, len(targets[start:end])):
            total_nll += nll_per_tok[j].item()
            n_scored += 1
        if end == len(inputs):
            break

    ppl = math.exp(total_nll / n_scored)
    logger.info("CBR-RNN podcast perplexity: %.4f  (n_scored=%d)", ppl, n_scored)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    record = {
        "model_id": model_id, "label": label,
        "n_params": 17_360_688, "family": "cbr_rnn",
        "corpus": "podcast", "perplexity": ppl,
        "n_tokens": n_scored, "max_length": max_seq_len,
        "stride": stride, "device": device,
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved → %s", out_path)
    return record


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def collect_summary(out_dir: Path) -> None:
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
    out_csv = out_dir / SUMMARY_CSV
    df.to_csv(out_csv, index=False)
    print(f"Summary: {len(df)} models → {out_csv}")
    print(df[["label", "family", "perplexity", "n_tokens"]].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_id", default=None, help="Run a specific model by HuggingFace ID")
    p.add_argument("--family", default=None,
                   help="Restrict to one registry family (e.g. stickbreaking, alibi, forgetting)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_length", type=int, default=None,
                   help="Override sliding-window max_length (default: model-specific, up to 2048). "
                        "Set to 4096 on H100 for faster throughput.")
    p.add_argument("--bids_root", type=Path, default=Path("data/podcast"),
                   help="Podcast BIDS root containing stimuli/podcast_transcript.csv")
    p.add_argument("--out_dir", type=Path,
                   default=Path("outputs_torch/podcast/perplexity"),
                   help="Output directory (default: outputs_torch/podcast/perplexity)")
    p.add_argument("--architecture", default=None, choices=("cbr_rnn",))
    p.add_argument("--force", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="Run all registry models sequentially (skip already-done)")
    p.add_argument("--checkpoints", action="store_true",
                   help="Expand all training checkpoints (Pythia steps, OLMo-2 revisions). "
                        "Implies --all. Use with --all or standalone.")
    p.add_argument("--list", action="store_true")
    p.add_argument("--summarize", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_checkpoints = args.checkpoints or False

    if args.family == "cbr_rnn":
        args.architecture = "cbr_rnn"

    if use_checkpoints:
        entries = [
            e for e in flat_checkpoint_list()
            if e.get("family") != "cbr_rnn"
            and (args.family is None or e.get("family") == args.family)
        ]
    else:
        entries = [
            m for m in flat_model_list()
            if m.get("family") != "cbr_rnn"
            and (args.family is None or m.get("family") == args.family)
        ]

    if args.list:
        if use_checkpoints:
            print(f"{'idx':>5}  {'family':<10}  {'label':<30}  {'revision':<35}  model_id")
            print("-" * 110)
            for i, m in enumerate(entries):
                print(f"{i:>5}  {m['family']:<10}  {m['label']:<30}  "
                      f"{m.get('revision','main'):<35}  {m['model_id']}")
        else:
            print(f"{'idx':>4}  {'family':<10}  {'label':<30}  {'n_params':>15}  model_id")
            print("-" * 90)
            for i, m in enumerate(entries):
                print(f"{i:>4}  {m['family']:<10}  {m['label']:<30}  "
                      f"{m['n_params']:>15.3e}  {m['model_id']}")
        print(f"\nTotal: {len(entries)} {'checkpoints' if use_checkpoints else 'models'}")
        return

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        collect_summary(out_dir)
        return

    bids_root = args.bids_root.resolve()
    transcript = bids_root / "stimuli" / "podcast_transcript.csv"
    if not transcript.is_file():
        sys.exit(f"Podcast transcript not found: {transcript}")

    if args.architecture == "cbr_rnn":
        model_id = args.model_id or "cclark/cbr-rnn"
        cbr_entry = next(
            (m for m in flat_model_list() if m.get("model_id") == model_id and m.get("family") == "cbr_rnn"),
            None,
        )
        label = cbr_entry["label"] if cbr_entry else "CBR-RNN"
        logger.info("=== CBR-RNN podcast perplexity: %s ===", model_id)
        compute_cbr_rnn_perplexity(model_id, bids_root, out_dir, args.device, force=args.force, label=label)
        collect_summary(out_dir)
        return

    if args.all or use_checkpoints:
        label = "checkpoints" if use_checkpoints else "models"
        fam_label = f" (family={args.family})" if args.family else ""
        logger.info("Running %d %s%s sequentially on Podcast", len(entries), label, fam_label)
        for i, entry in enumerate(entries):
            rev = entry.get("revision", "main")
            logger.info("── [%d/%d] %s @ %s ──", i + 1, len(entries), entry["label"], rev)
            run_model(entry, bids_root, out_dir, args.device, force=args.force,
                      max_length_override=args.max_length)
        collect_summary(out_dir)
        return

    if args.model_id:
        matches = [m for m in entries if m["model_id"] == args.model_id]
        if not matches:
            sys.exit(f"Model {args.model_id!r} not found in registry")
        run_model(matches[0], bids_root, out_dir, args.device, force=args.force,
                  max_length_override=args.max_length)
        return

    sys.exit("Provide --all, --checkpoints, --model_id, --list, or --summarize")


if __name__ == "__main__":
    main()
