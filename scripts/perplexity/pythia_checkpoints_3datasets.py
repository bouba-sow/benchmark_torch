#!/usr/bin/env python3
"""Per-checkpoint perplexity for ONE Pythia size, on all 3 behavioral
datasets that don't have per-checkpoint perplexity yet (naturalstories_spr,
ucl_reading, zuco_reading -- provo/geco/meco already have it, computed
earlier alongside their embedding checkpoint sweep).

For each of the 11 LPP_PYTHIA_STEPS training steps, the model is loaded
ONCE and perplexity is computed against all 3 datasets' corpora before
moving to the next checkpoint -- avoids reloading the same ~GB checkpoint
3 times (once per dataset), which is what a naive per-dataset script would
do. Meant to be launched as 10 separate jobs, one per Pythia size, each
computing 11 checkpoints x 3 datasets = 33 perplexity values.

Output (same convention as neuroai.perplexity.save_checkpoint_perplexity,
used by the provo/geco/meco embed-sweep-era per-checkpoint perplexity):
    outputs_jeanzay/<dataset>_perplexity/<model_safe>/<revision>_<dataset>_perplexity.json
    outputs_jeanzay/<dataset>_perplexity/<dataset>_perplexity_summary.csv

Usage:
    .venv/bin/python scripts/perplexity/pythia_checkpoints_3datasets.py \\
        --model_id EleutherAI/pythia-410m-deduped --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd
import torch

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from neuroai.models._checkpoints import pythia_checkpoint_revision
from neuroai.models.registry import LPP_PYTHIA_STEPS, LPP_FAMILIES

DATASETS = ("naturalstories_spr", "ucl_reading", "zuco_reading")


def _n_params_for(model_id: str) -> float:
    for spec in LPP_FAMILIES["pythia"]:
        if spec.model_id == model_id:
            return spec.n_params
    raise ValueError(f"{model_id} not found in LPP_FAMILIES['pythia']")


def _load_texts() -> dict[str, list[str]]:
    """Load each dataset's corpus ONCE (text doesn't change across checkpoints)."""
    texts: dict[str, list[str]] = {}

    from scripts.perplexity.naturalstories import load_naturalstories_text
    texts["naturalstories_spr"] = load_naturalstories_text(Path("data/naturalstories"))
    logger.info("naturalstories_spr corpus: %d stories", len(texts["naturalstories_spr"]))

    from scripts.perplexity.ucl_reading import load_ucl_reading_text
    texts["ucl_reading"] = load_ucl_reading_text(Path("data/ucl_reading"))
    logger.info("ucl_reading corpus: %d sentences", len(texts["ucl_reading"]))

    from scripts.perplexity.zuco_reading import load_zuco_text
    texts["zuco_reading"] = load_zuco_text(Path("data/zuco_reading/osfstorage"))
    logger.info("zuco_reading corpus: %d sentences", len(texts["zuco_reading"]))

    return texts


def _out_paths(dataset: str, model_id: str, revision: str) -> tuple[Path, Path]:
    perplexity_dir = Path("outputs_jeanzay") / f"{dataset}_perplexity"
    safe = model_id.replace("/", "_")
    rev_safe = (revision or "main").replace("/", "_")
    out_dir = perplexity_dir / safe
    out_path = out_dir / f"{rev_safe}_{dataset}_perplexity.json"
    summary_csv = perplexity_dir / f"{dataset}_perplexity_summary.csv"
    return out_path, summary_csv


def _save(dataset: str, model_id: str, revision: str, label: str, n_params: float,
          perplexity: float, n_tokens: int, max_length: int, force: bool) -> None:
    out_path, _summary_csv = _out_paths(dataset, model_id, revision)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file() and not force:
        existing = json.loads(out_path.read_text())
        if "error" not in existing:
            logger.info("[%s] already done: %s @ %s — skipping", dataset, model_id, revision)
            return

    record = {
        "model_id": model_id, "revision": revision, "label": label,
        "n_params": n_params, "family": "pythia", "dataset": dataset,
        "perplexity": perplexity, "n_tokens": n_tokens, "max_length": max_length,
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("[%s] Saved perplexity %.4f for %s @ %s", dataset, perplexity, model_id, revision)
    # NOTE: deliberately NOT touching summary_csv here. Up to 20 of these
    # jobs (10 model sizes x 2 queues) run concurrently and would all
    # read-modify-write the SAME shared summary CSV -- a real race (verified:
    # crashed with "KeyError: 'revision'" reading a summary CSV another
    # process's concurrent write had left with a mismatched header/data
    # schema). The per-checkpoint JSON write just above is the only
    # source of truth during the sweep; run this script's --collect_summary
    # mode ONCE, after all jobs finish, to rebuild <dataset>_perplexity_summary.csv
    # from every JSON file on disk (no concurrent-write risk, since it's a
    # single pass run once).


def _already_done(model_id: str, revision: str, force: bool) -> bool:
    if force:
        return False
    for ds in DATASETS:
        out_path, _ = _out_paths(ds, model_id, revision)
        if not out_path.is_file():
            return False
        existing = json.loads(out_path.read_text())
        if "error" in existing:
            return False
    return True


def collect_summary(dataset: str) -> None:
    """Rebuild <dataset>_perplexity_summary.csv from EVERY *_<dataset>_perplexity.json
    on disk (Pythia checkpoints from this script, plus any other source using
    the same naming convention -- e.g. provo/geco/meco's existing per-checkpoint
    perplexity). Single pass, no concurrent-write risk -- run this once after
    all sweep jobs finish, not during them."""
    perplexity_dir = Path("outputs_jeanzay") / f"{dataset}_perplexity"
    rows = []
    for p in sorted(perplexity_dir.glob(f"*/*_{dataset}_perplexity.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "error" in d:
            continue
        rows.append(d)
    if not rows:
        logger.warning("[%s] no *_%s_perplexity.json files found under %s", dataset, dataset, perplexity_dir)
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["model_id", "revision"], keep="last")
    out_csv = perplexity_dir / f"{dataset}_perplexity_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info("[%s] wrote %d rows -> %s", dataset, len(df), out_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_id", help="e.g. EleutherAI/pythia-410m-deduped")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--collect_summary", action="store_true",
                     help="Rebuild all 3 datasets' summary CSVs from JSON files and exit "
                          "(run once after all sweep jobs finish, not concurrently with them).")
    args = ap.parse_args()

    if args.collect_summary:
        for ds in DATASETS:
            collect_summary(ds)
        return

    if not args.model_id:
        ap.error("--model_id is required unless --collect_summary is given")

    from scripts.perplexity.lpp import compute_perplexity, _max_length_for_model
    from neuroai.features.llm import _get_cached_causal_lm, clear_hf_model_cache

    n_params = _n_params_for(args.model_id)
    label = args.model_id.split("/")[-1]
    texts = _load_texts()

    for step in LPP_PYTHIA_STEPS:
        revision = pythia_checkpoint_revision(step)
        if _already_done(args.model_id, revision, args.force):
            logger.info("step=%d (%s) — all 3 datasets already done, skipping", step, revision)
            continue

        logger.info("=== %s @ %s (step=%d) — loading model ONCE for all 3 datasets ===",
                     args.model_id, revision, step)
        try:
            tokenizer, model = _get_cached_causal_lm(
                args.model_id,
                revision=revision,
                device=args.device,
                load_kw={"trust_remote_code": False},
                tokenizer_no_revision=True,
                n_params=n_params,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            eff_max_length = _max_length_for_model(model, args.model_id)
            if args.max_length is not None:
                eff_max_length = min(args.max_length, eff_max_length)
        except Exception as exc:
            logger.error("FAILED to load %s @ %s: %s", args.model_id, revision, exc, exc_info=True)
            for ds in DATASETS:
                _save(ds, args.model_id, revision, label, n_params, float("nan"), 0, 0, args.force)
            continue

        for ds in DATASETS:
            out_path, _ = _out_paths(ds, args.model_id, revision)
            if out_path.is_file() and not args.force:
                existing = json.loads(out_path.read_text())
                if "error" not in existing:
                    logger.info("[%s] already done: %s @ %s — skipping", ds, args.model_id, revision)
                    continue
            try:
                result = compute_perplexity(model, tokenizer, texts[ds], args.device, eff_max_length)
                _save(ds, args.model_id, revision, label, n_params,
                      result["perplexity"], result["n_tokens"], eff_max_length, args.force)
            except Exception as exc:
                logger.error("[%s] FAILED %s @ %s: %s", ds, args.model_id, revision, exc, exc_info=True)

        del model
        del tokenizer
        clear_hf_model_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("All checkpoints done for %s", args.model_id)


if __name__ == "__main__":
    main()
