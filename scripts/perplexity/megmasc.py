#!/usr/bin/env python3
"""Perplexity of LPP-registry models on the MEG-MASC (Gwilliams 2023) corpus.

Corpus = the four MASC story word lists (English, from each recording's
``events.tsv`` — same canonical list the surprisal / encoding pipelines use),
one space-joined string per story. Stride-based sliding-window perplexity,
identical recipe to ``perplexity/nieuwland.py`` (shared helpers from
``perplexity/lpp.py``).

Output: ``<out_dir>/<model_safe>/perplexity.json`` (default out_dir
``outputs_torch/megmasc/perplexity``), the layout ``load_perplexity_dir``
in ``neuroai/analysis/benchmark_common.py`` expects.

Runs standalone, and is also invoked automatically by
``scripts/run/surprisal/neural/megmasc_extract.py`` (Stage 1) so perplexity is
produced in the same GPU job as the logit-lens surprisal cache.

Usage
-----
    python scripts/perplexity/megmasc.py --all --device cuda
    python scripts/perplexity/megmasc.py --from-registry --family gpt2 --array_index 0 --device cuda
    python scripts/perplexity/megmasc.py --model_id EleutherAI/pythia-1b --device cuda
    python scripts/perplexity/megmasc.py --summarize
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import torch

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUMMARY_CSV = "megmasc_perplexity_summary.csv"
CORPUS_LABEL = "megmasc"
DEFAULT_OUT = Path("outputs_torch/megmasc/perplexity")


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
# Corpus
# ---------------------------------------------------------------------------

def _default_bids_root() -> Path:
    return Path(
        os.environ.get("MEGMASC_BIDS_ROOT")
        or (Path(os.environ["SCRATCH"]) / "data" / "gwilliams" if os.environ.get("SCRATCH") else "data/gwilliams")
    )


def load_megmasc_text(bids_root: str | Path) -> list[str]:
    """One space-joined string per MASC story (English)."""
    from neuroai.datasets.megmasc import MegMascDataset, load_story_words

    ds = MegMascDataset(bids_root)
    texts: list[str] = []
    for story in ds.available_stories():
        words = load_story_words(bids_root, story).sort_values("word_index")["word"].astype(str).tolist()
        if words:
            texts.append(" ".join(words))
    if not texts:
        raise RuntimeError(f"No MEG-MASC story words found under {bids_root}")
    logger.info("MEG-MASC corpus: %d stories, %d words total",
                len(texts), sum(len(t.split()) for t in texts))
    return texts


# ---------------------------------------------------------------------------
# Per-model
# ---------------------------------------------------------------------------

def run_model(
    model_entry: dict,
    bids_root: str | Path,
    out_dir: Path,
    device: str,
    force: bool = False,
    texts: list[str] | None = None,
) -> dict | None:
    """Compute + save perplexity for one model. ``model_entry`` needs
    ``model_id`` and ``n_params``; ``label`` / ``family`` optional."""
    model_id = model_entry["model_id"]
    label = model_entry.get("label", model_id)
    family = model_entry.get("family", "adhoc")
    n_params = model_entry.get("n_params")
    safe = _safe_dir(model_id)
    out_path = out_dir / safe / "perplexity.json"

    if out_path.is_file() and not force:
        existing = json.loads(out_path.read_text())
        if "error" not in existing:
            logger.info("Already done: %s — skipping", model_id)
            return existing
        logger.warning("Previous run had error — recomputing: %s", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if texts is None:
        texts = load_megmasc_text(bids_root)

    try:
        tokenizer, model = load_model_and_tokenizer(model_id, n_params, device)
    except Exception as exc:  # noqa: BLE001
        logger.error("FAILED %s (model load): %s", model_id, exc, exc_info=True)
        out_path.write_text(json.dumps({
            "model_id": model_id, "label": label, "n_params": n_params,
            "family": family, "corpus": CORPUS_LABEL, "error": f"load: {exc}",
        }, indent=2))
        return None

    max_length = _max_length_for_model(model, model_id)
    if getattr(model, "hf_device_map", None) and any(
        v == "cpu" for v in model.hf_device_map.values()
    ):
        max_length = min(max_length, 512)
        logger.warning("%s has CPU-offloaded layers — max_length -> %d", model_id, max_length)
    torch_dtype = str(next(model.parameters()).dtype)

    try:
        result = compute_perplexity(model, tokenizer, texts, device, max_length)
    except Exception as exc:  # noqa: BLE001
        logger.error("FAILED %s: %s", model_id, exc, exc_info=True)
        out_path.write_text(json.dumps({
            "model_id": model_id, "label": label, "n_params": n_params,
            "family": family, "corpus": CORPUS_LABEL, "error": str(exc),
        }, indent=2))
        return None
    finally:
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = {
        "model_id": model_id, "label": label, "n_params": n_params, "family": family,
        "corpus": CORPUS_LABEL, "lang": "en",
        "perplexity": result["perplexity"], "n_tokens": result["n_tokens"],
        "max_length": max_length, "stride": max_length // 2,
        "device": device, "torch_dtype": torch_dtype, "n_stories": len(texts),
    }
    if model_entry.get("revision"):
        record["revision"] = model_entry["revision"]
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved → %s  (ppl=%.2f)", out_path, record["perplexity"])
    return record


def collect_summary(out_dir: Path) -> None:
    import pandas as pd
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.rglob("perplexity.json"))]
    rows = [r for r in rows if "error" not in r]
    if not rows:
        print("No completed perplexity results.")
        return
    df = pd.DataFrame(rows).sort_values("n_params")
    df.to_csv(out_dir / SUMMARY_CSV, index=False)
    print(f"Summary: {len(df)} models → {out_dir / SUMMARY_CSV}")
    print(df[["label", "family", "perplexity", "n_tokens"]].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="every LPP-registry model")
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array_index", type=int, default=None)
    p.add_argument("--model_id", default=None)
    p.add_argument("--n_params", type=float, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bids_root", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()

    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        collect_summary(out_dir)
        return

    bids_root = args.bids_root or _default_bids_root()
    texts = load_megmasc_text(bids_root)

    if args.from_registry:
        from neuroai.models.registry import (
            registry_model_configs, resolve_family_task, resolve_load_model_id,
        )
        if not args.family or args.array_index is None:
            p.error("--from-registry requires --family and --array_index")
        task = resolve_family_task(args.family, args.array_index)
        mcfgs = [m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id]
        entry = {
            "model_id": resolve_load_model_id(mcfgs[0], task.revision) if mcfgs else task.model_id,
            "label": getattr(task, "label", task.model_id),
            "n_params": mcfgs[0].get("n_params") if mcfgs else None,
            "family": task.registry_family,
            "revision": task.revision if task.revision not in (None, "", "main") else None,
        }
        run_model(entry, bids_root, out_dir, args.device, force=args.force, texts=texts)
    elif args.model_id:
        run_model({"model_id": args.model_id, "n_params": args.n_params, "family": "adhoc"},
                  bids_root, out_dir, args.device, force=args.force, texts=texts)
    elif args.all:
        for entry in flat_model_list():
            run_model(entry, bids_root, out_dir, args.device, force=args.force, texts=texts)
        collect_summary(out_dir)
    else:
        p.error("pass --all, --from-registry, --model_id, or --summarize")


if __name__ == "__main__":
    main()
