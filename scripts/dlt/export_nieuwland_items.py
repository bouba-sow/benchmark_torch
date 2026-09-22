#!/usr/bin/env python3
"""Rebuild the Nieuwland (item_number, sentence_context, noun) table WITHOUT
the raw BIDS EEG data, by reading it back out of an already-computed
``nieuwland_word_predictions/*.npz`` file from a checkpoint-sweep run.

Why this works: ``neuroai.benchmarks.eeg_erp._save_word_predictions`` stores,
for each item_number, the "first occurrence" sentence_context/noun pair
(see its docstring) — DLT integration cost at the critical noun is structural,
so it doesn't depend on which of the two conditions' noun happens to be
picked. This is exactly the same representative-noun-per-item table that
``dlt/costs_nieuwland.py export-items`` produces from the real BIDS
data (which then gets averaged over condition anyway in ``parse``), just
without needing to load ~300 subjects of raw EEG — useful on a machine (e.g.
habilis/oberon) that doesn't have the Nieuwland BIDS dataset (``data/nieuwland``)
but does have predictions copied over from Jean Zay.

Usage
-----
    .venv/bin/python scripts/dlt/export_nieuwland_items.py \\
        --predictions_npz outputs_jeanzay/nieuwland/embedding_sweep/gpt2/openai-community_gpt2-large/nieuwland_word_predictions/openai-community_gpt2-large_main.npz \\
        --out_dir outputs/dlt/_items_export

If ``--predictions_npz`` is omitted, the first ``*_main.npz`` found under
``--sweep_root`` (default ``outputs_jeanzay/nieuwland/embedding_sweep``)
is used — any model works since the item table is identical across models.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
ITEMS_CSV_NAME = "nieuwland_items.csv"
# The predictions npz has no per-condition breakdown (see module docstring) —
# a constant placeholder is used so downstream `parse` (which expects a
# `condition` column) still works; it only affects the *_by_condition.csv
# detail table, not the final item-level `nieuwland_dlt_cost.csv` (which
# averages over condition regardless).
PLACEHOLDER_CONDITION = "pooled"


def find_first_predictions_npz(sweep_root: Path) -> Path:
    candidates = sorted(sweep_root.glob("*/*/nieuwland_word_predictions/*_main.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No nieuwland_word_predictions/*_main.npz found under {sweep_root}"
        )
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions_npz", default=None,
                     help="Path to a nieuwland_word_predictions/*_main.npz file. "
                          "If omitted, the first one found under --sweep_root is used.")
    ap.add_argument("--sweep_root", default=str(REPO_ROOT / "outputs_jeanzay" / "nieuwland/embedding_sweep"),
                     help="Root dir to search for a predictions npz when --predictions_npz is not given.")
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "dlt" / "_items_export"))
    args = ap.parse_args()

    npz_path = Path(args.predictions_npz) if args.predictions_npz else find_first_predictions_npz(Path(args.sweep_root))
    logger.info("Reading item table from %s", npz_path)
    data = np.load(npz_path, allow_pickle=True)

    items = pd.DataFrame({
        "item_number": data["item_number"],
        "condition": PLACEHOLDER_CONDITION,
        "sentence_context": data["sentence_context"],
        "noun": data["noun"],
    })
    logger.info("Found %d items", len(items))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ITEMS_CSV_NAME
    items.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
