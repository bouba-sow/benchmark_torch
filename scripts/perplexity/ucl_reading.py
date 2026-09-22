#!/usr/bin/env python3
"""Compute standard (token-level) perplexity of all LPP registry models on UCL Reading text.

Perplexity uses the same stride-based sliding window as
``scripts/perplexity/lpp.py`` / ``scripts/perplexity/naturalstories.py``,
but the corpus is the concatenation of all 361 UCL Reading sentences (from
``stimuli.txt``). This is the standard HF perplexity recipe (every subword token
scored once, non-overlapping windows) — unlike the word-pooled perplexity derived
inline during ``scripts/run/surprisal/behavioral/ucl_reading_extract.py``, this reloads the model
in a dedicated pass.

Results are saved as JSON under:

    <out_dir>/<model_safe>/perplexity.json

Usage
-----
    # List all models and their array indices
    python scripts/perplexity/ucl_reading.py --list

    # Run one model by SLURM array index
    python scripts/perplexity/ucl_reading.py --array_index 3 --device cuda

    # Run one model directly
    python scripts/perplexity/ucl_reading.py --model_id facebook/opt-125m --device cuda

    # Run all registry models sequentially (skip already-done)
    python scripts/perplexity/ucl_reading.py --all --device cuda

    # Collect existing JSON files into a summary CSV
    python scripts/perplexity/ucl_reading.py --summarize

Jean Zay::

    sbatch scripts/slurm/jz/perplexity/ucl_reading.sh
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

from neuroai.datasets.ucl_reading import load_stimuli

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUMMARY_CSV = "ucl_reading_perplexity_summary.csv"


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


def load_ucl_reading_text(dataset_root: Path, sentences: list[int] | None = None) -> list[str]:
    """Return one string per UCL Reading sentence (361 by default)."""
    stim = load_stimuli(dataset_root)
    if sentences is not None:
        stim = stim[stim["sent_nr"].isin(sentences)]
    texts = stim.sort_values("sent_nr")["sentence"].astype(str).tolist()
    if not texts:
        raise RuntimeError(f"No UCL Reading text found under {dataset_root}")
    logger.info("Loaded %d UCL Reading sentences", len(texts))
    return texts


def run_model(
    model_entry: dict,
    dataset_root: Path,
    out_dir: Path,
    device: str,
    sentences: list[int] | None = None,
    force: bool = False,
    max_length_override: int | None = None,
) -> dict | None:
    model_id = model_entry["model_id"]
    safe = _safe_dir(model_id)
    out_path = out_dir / safe / "perplexity.json"

    if out_path.is_file() and not force:
        logger.info("Already done: %s — skipping (use --force to recompute)", model_id)
        return json.loads(out_path.read_text())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    texts = load_ucl_reading_text(dataset_root, sentences=sentences)

    try:
        tokenizer, model = load_model_and_tokenizer(model_id, model_entry["n_params"], device)
    except Exception as exc:
        logger.error("FAILED %s (model load): %s", model_id, exc, exc_info=True)
        out_path.write_text(json.dumps({
            "model_id": model_id, "label": model_entry["label"],
            "n_params": model_entry["n_params"], "family": model_entry["family"],
            "corpus": "ucl_reading", "error": f"load: {exc}",
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
        logger.warning(
            "%s has CPU-offloaded layers — reducing max_length to %d to avoid OOM",
            model_id, max_length,
        )

    torch_dtype = str(next(model.parameters()).dtype)

    try:
        result = compute_perplexity(model, tokenizer, texts, device, max_length)
    except Exception as exc:
        logger.error("FAILED %s: %s", model_id, exc, exc_info=True)
        out_path.write_text(json.dumps({
            "model_id": model_id, "label": model_entry["label"],
            "n_params": model_entry["n_params"], "family": model_entry["family"],
            "corpus": "ucl_reading", "error": str(exc),
        }, indent=2))
        return None
    finally:
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
        "corpus": "ucl_reading",
        "perplexity": result["perplexity"],
        "n_tokens": result["n_tokens"],
        "max_length": max_length,
        "stride": max_length // 2,
        "device": device,
        "torch_dtype": torch_dtype,
        "n_sentences": len(texts),
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved → %s", out_path)
    return record


def compute_cbr_rnn_perplexity(
    model_id: str,
    dataset_root: Path,
    out_dir: Path,
    device: str,
    sentences: list[int] | None = None,
    force: bool = False,
    max_length_override: int | None = None,
    label: str = "CBR-RNN",
) -> dict | None:
    """Perplexity for the CBR-RNN model on UCL Reading (custom loader, not AutoModelForCausalLM)."""
    import math

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
        logger.warning("CBR-RNN output has error field — recomputing: %s", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    texts = load_ucl_reading_text(dataset_root, sentences=sentences)
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
    max_seq_len: int = model.max_seq_len
    if max_length_override is not None:
        max_seq_len = min(max_length_override, max_seq_len)
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
    logger.info("CBR-RNN UCL Reading perplexity: %.4f  (n_scored=%d)", ppl, n_scored)

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    record = {
        "model_id": model_id,
        "label": label,
        "n_params": 17_360_688,
        "family": "cbr_rnn",
        "corpus": "ucl_reading",
        "perplexity": ppl,
        "n_tokens": n_scored,
        "max_length": max_seq_len,
        "stride": stride,
        "device": device,
        "n_sentences": len(texts),
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved → %s", out_path)
    return record


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--array_index", type=int, default=None, help="SLURM array task index (0-based)")
    p.add_argument("--model_id", default=None, help="Run a specific model by HuggingFace ID")
    p.add_argument("--family", default=None, help="Restrict to one registry family (e.g. gpt2, pythia, mamba)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    p.add_argument("--max_length", type=int, default=None,
                   help="Override sliding-window max_length (default: model-specific, up to 2048).")
    p.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("data/ucl_reading"),
        help="UCL Reading dataset root (needs stimuli.txt)",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/ucl_reading/perplexity"),
        help="Output directory (default: outputs/ucl_reading/perplexity)",
    )
    p.add_argument("--sentences", nargs="*", type=int, default=None, help="Optional sentence numbers (1-based)")
    p.add_argument(
        "--architecture",
        default=None,
        choices=("cbr_rnn",),
        help="Custom architecture (not AutoModelForCausalLM). Currently: cbr_rnn",
    )
    p.add_argument("--force", action="store_true", help="Recompute even if output already exists")
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all registry models sequentially (skip already-done)",
    )
    p.add_argument("--list", action="store_true", help="Print the flat model list with indices and exit")
    p.add_argument(
        "--summarize",
        action="store_true",
        help="Collect existing JSON files into a summary CSV and exit",
    )
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
            print(
                f"{i:>4}  {m['family']:<10}  {m['label']:<30}  {m['n_params']:>15.3e}  {m['model_id']}"
            )
        print(f"\nTotal{fam_label}: {len(models)} models  →  --array 0-{len(models) - 1}")
        return

    out_dir = args.out_dir.resolve()

    if args.summarize:
        collect_summary(out_dir)
        return

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        sys.exit(f"UCL Reading dataset root not found: {dataset_root}")
    if not (dataset_root / "stimuli.txt").is_file():
        sys.exit(f"stimuli.txt not found under {dataset_root}")

    if args.architecture == "cbr_rnn":
        model_id = args.model_id or "cclark/cbr-rnn"
        cbr_entry = next(
            (m for m in all_models if m.get("model_id") == model_id and m.get("family") == "cbr_rnn"),
            None,
        )
        label = cbr_entry["label"] if cbr_entry else "CBR-RNN"
        logger.info("=== CBR-RNN UCL Reading perplexity: %s ===", model_id)
        compute_cbr_rnn_perplexity(
            model_id, dataset_root, out_dir, args.device,
            sentences=args.sentences, force=args.force,
            max_length_override=args.max_length,
            label=label,
        )
        collect_summary(out_dir)
        return

    if args.all:
        fam_label = f" (family={args.family})" if args.family else ""
        logger.info("Running %d models%s sequentially on UCL Reading", len(models), fam_label)
        for i, entry in enumerate(models):
            logger.info("── [%d/%d] %s ──", i + 1, len(models), entry["label"])
            run_model(
                entry, dataset_root, out_dir, args.device,
                sentences=args.sentences, force=args.force,
                max_length_override=args.max_length,
            )
        collect_summary(out_dir)
        return

    if args.array_index is not None:
        if args.array_index < 0 or args.array_index >= len(models):
            sys.exit(f"array_index {args.array_index} out of range [0, {len(models) - 1}]")
        entry = models[args.array_index]
    elif args.model_id is not None:
        matches = [m for m in models if m["model_id"] == args.model_id]
        if not matches:
            sys.exit(f"Model {args.model_id!r} not found in registry")
        entry = matches[0]
    else:
        sys.exit("Provide --all, --array_index, --model_id, --family, --list, or --summarize")

    logger.info("=== UCL Reading perplexity: %s (%s) ===", entry["label"], entry["model_id"])
    run_model(
        entry, dataset_root, out_dir, args.device,
        sentences=args.sentences, force=args.force,
        max_length_override=args.max_length,
    )
    collect_summary(out_dir)


if __name__ == "__main__":
    main()
