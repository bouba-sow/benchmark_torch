#!/usr/bin/env python3
"""Compute perplexity of all LPP registry models on the Nieuwland N400 stimuli.

The corpus is reconstructed from the Nieuwland et al. (2018) BIDS dataset
(data/nieuwland).  Each item produces two sentences (expected and unexpected
article + noun).  Both variants are included.  Results are saved as JSON under:

    <out_dir>/<model_safe>/perplexity.json

Usage
-----
    python scripts/perplexity/nieuwland.py --all --device cuda
    python scripts/perplexity/nieuwland.py --summarize

Jean Zay::

    sbatch scripts/slurm/jz/perplexity/nieuwland.sh
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

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUMMARY_CSV = "nieuwland_perplexity_summary.csv"
CORPUS_LABEL = "nieuwland"


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


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_nieuwland_text(bids_root: Path) -> list[str]:
    """Return unique Nieuwland sentences from BIDS events.tsv files.

    Each event row for a noun trial has ``sentence_context``, ``expected_noun``,
    and ``unexpected_noun``.  We generate two sentences per item
    (expected: context + expected_noun, unexpected: context + unexpected_noun)
    and de-duplicate across subjects (all subjects share the same stimuli).
    """
    import pandas as pd

    seen: set[str] = set()
    sentences: list[str] = []

    # Iterate over one subject per lab (stimuli are shared across subjects)
    evts_files = sorted(bids_root.glob("sub-*/eeg/*_task-delong_events.tsv"))
    if not evts_files:
        raise FileNotFoundError(f"No Nieuwland events.tsv found under {bids_root}")

    # One pass through enough subjects to cover all items (typically one subject covers all)
    for evts_path in evts_files:
        df = pd.read_csv(evts_path, sep="\t", dtype=str, na_values=["n/a"])
        # Keep only noun trial rows that have sentence_context
        noun_mask = df["trial_type"].isin(["noun_expected", "noun_unexpected"])
        noun_df = df[noun_mask].dropna(subset=["sentence_context"])
        if noun_df.empty:
            continue

        for _, row in noun_df.iterrows():
            ctx = str(row.get("sentence_context", "")).strip()
            exp_noun = str(row.get("expected_noun", "")).strip()
            unexp_noun = str(row.get("unexpected_noun", "")).strip()
            for noun in (exp_noun, unexp_noun):
                if noun and noun.lower() not in ("n/a", "nan", ""):
                    s = f"{ctx} {noun}".strip()
                    if s and s not in seen:
                        seen.add(s)
                        sentences.append(s)

        if len(sentences) >= 200:
            # All unique items have been collected (240 items × 2 variants)
            break

    if not sentences:
        raise RuntimeError(f"No sentences loaded from Nieuwland BIDS at {bids_root}")
    logger.info("Nieuwland corpus: %d unique sentences", len(sentences))
    return sentences


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_entry: dict,
    bids_root: Path,
    out_dir: Path,
    device: str,
    force: bool = False,
) -> dict | None:
    model_id = model_entry["model_id"]
    safe = _safe_dir(model_id)
    out_path = out_dir / safe / "perplexity.json"

    if out_path.is_file() and not force:
        existing = json.loads(out_path.read_text())
        if "error" not in existing:
            logger.info("Already done: %s — skipping", model_id)
            return existing
        logger.warning("Previous run had error — recomputing: %s", out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    texts = load_nieuwland_text(bids_root)

    try:
        tokenizer, model = load_model_and_tokenizer(model_id, model_entry["n_params"], device)
    except Exception as exc:
        logger.error("FAILED %s (model load): %s", model_id, exc, exc_info=True)
        out_path.write_text(json.dumps({
            "model_id": model_id, "label": model_entry["label"],
            "n_params": model_entry["n_params"], "family": model_entry["family"],
            "corpus": CORPUS_LABEL, "error": f"load: {exc}",
        }, indent=2))
        return None

    max_length = _max_length_for_model(model, model_id)
    if getattr(model, "hf_device_map", None) and any(
        v == "cpu" for v in model.hf_device_map.values()
    ):
        max_length = min(max_length, 512)
        logger.warning("%s has CPU-offloaded layers — reducing max_length to %d", model_id, max_length)

    torch_dtype = str(next(model.parameters()).dtype)

    try:
        result = compute_perplexity(model, tokenizer, texts, device, max_length)
    except Exception as exc:
        logger.error("FAILED %s: %s", model_id, exc, exc_info=True)
        out_path.write_text(json.dumps({
            "model_id": model_id, "label": model_entry["label"],
            "n_params": model_entry["n_params"], "family": model_entry["family"],
            "corpus": CORPUS_LABEL, "error": str(exc),
        }, indent=2))
        return None
    finally:
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = {
        "model_id": model_id,
        "label": model_entry["label"],
        "n_params": model_entry["n_params"],
        "family": model_entry["family"],
        "corpus": CORPUS_LABEL,
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
    bids_root: Path,
    out_dir: Path,
    device: str,
    force: bool = False,
) -> dict | None:
    """Perplexity for a CBR-RNN family model (custom PyTorchModelHubMixin
    architecture, not AutoModelForCausalLM -- uses its own tokenizer/model
    loader and the same sliding-window recipe as the original surprisal.py)."""
    import math

    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from neuroai.models.cbr_rnn import CbrRnn

    safe = _safe_dir(model_id)
    out_path = out_dir / safe / "perplexity.json"
    if out_path.is_file() and not force:
        logger.info("Already done: %s — skipping (use --force to recompute)", model_id)
        return json.loads(out_path.read_text())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    texts = load_nieuwland_text(bids_root)
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

        logits = logits.squeeze(0)
        nll_per_tok = F.cross_entropy(logits, tgt, reduction="none")

        score_start_in_window = 0 if start == 0 else stride
        for j in range(score_start_in_window, len(win_targets)):
            total_nll += nll_per_tok[j].item()
            n_scored += 1

        if end == len(inputs):
            break

    ppl = math.exp(total_nll / n_scored)
    logger.info("CBR-RNN perplexity: %.4f  (n_scored=%d)", ppl, n_scored)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    record = {
        "model_id": model_id,
        "label": "CBR-RNN-M" if model_id.endswith("-m") else "CBR-RNN",
        "n_params": 17_360_688,
        "family": "cbr_rnn",
        "corpus": CORPUS_LABEL,
        "perplexity": ppl,
        "n_tokens": n_scored,
        "max_length": max_seq_len,
        "stride": stride,
        "device": device,
        "torch_dtype": "float32",
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
    p.add_argument("--array_index", type=int, default=None)
    p.add_argument("--model_id", default=None)
    p.add_argument("--family", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bids_root", type=Path,
                   default=Path("data/nieuwland"),
                   help="Nieuwland BIDS root (default: data/nieuwland)")
    p.add_argument("--out_dir", type=Path,
                   default=Path("outputs_jeanzay/nieuwland/perplexity"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--architecture", default=None, choices=("cbr_rnn",),
                   help="Custom architecture for non-AutoModelForCausalLM models")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_models = flat_model_list()
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
    if not bids_root.is_dir():
        sys.exit(f"Nieuwland BIDS root not found: {bids_root}")

    # ── CBR-RNN shortcut (excluded from the registry loop above -- custom
    # PyTorchModelHubMixin architecture, not AutoModelForCausalLM) ──────────
    if args.architecture == "cbr_rnn":
        model_id = args.model_id or "cclark/cbr-rnn"
        logger.info("=== CBR-RNN perplexity: %s ===", model_id)
        compute_cbr_rnn_perplexity(model_id, bids_root, out_dir, args.device, force=args.force)
        return

    if args.all:
        fam_label = f" (family={args.family})" if args.family else ""
        logger.info("Running %d models%s sequentially on Nieuwland", len(models), fam_label)
        for i, entry in enumerate(models):
            logger.info("── [%d/%d] %s ──", i + 1, len(models), entry["label"])
            run_model(entry, bids_root, out_dir, args.device, force=args.force)
        collect_summary(out_dir)
        return

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
        sys.exit("Provide --all, --array_index, --model_id, --list, or --summarize")

    logger.info("=== Nieuwland perplexity: %s (%s) ===", entry["label"], entry["model_id"])
    run_model(entry, bids_root, out_dir, args.device, force=args.force)


if __name__ == "__main__":
    main()
