#!/usr/bin/env python3
"""Join precomputed DLT (Gibson 2000) integration costs (outputs/dlt/*.csv,
see dlt/costs.py) onto already-
cached per-word/per-TR tables from the pooled-LME benchmark pipelines.

Pure reconstruction — no model/embedding reruns. Each ``join_*`` function
reads one representative already-computed pooled table (the word/TR indexing
in these tables is model-independent; only the ``surprisal``/``bold`` columns
differ across models in a checkpoint sweep) and writes a DLT-augmented copy
under ``outputs/dlt/joined/``.

Usage
-----
    .venv/bin/python scripts/dlt/join_costs.py podcast \\
        --pooled_table outputs_jeanzay/podcast/surprisal_sweep/gpt2/openai-community_gpt2-xl/podcast_lme_pooled_table.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DLT_DIR = REPO_ROOT / "outputs" / "dlt"
JOINED_DIR = DLT_DIR / "joined"


def join_podcast(pooled_table_path: str) -> None:
    pooled = pd.read_parquet(pooled_table_path)
    dlt = pd.read_csv(DLT_DIR / "podcast_dlt_cost.csv")[
        ["word_idx", "storage", "integration", "total"]
    ].rename(columns={"storage": "dlt_storage", "integration": "dlt_integration", "total": "dlt_total"})

    merged = pooled.merge(dlt, on="word_idx", how="left")
    n_missing = int(merged["dlt_total"].isna().sum())
    logger.info(
        "[podcast] pooled=%d rows, dlt=%d words, merged=%d rows, %d rows with no DLT match",
        len(pooled), len(dlt), len(merged), n_missing,
    )

    JOINED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = JOINED_DIR / "podcast_with_dlt.parquet"
    merged.to_parquet(out_path, index=False)
    logger.info("[podcast] wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_podcast = sub.add_parser("podcast")
    p_podcast.add_argument("--pooled_table", required=True)

    args = ap.parse_args()
    if args.cmd == "podcast":
        join_podcast(args.pooled_table)


if __name__ == "__main__":
    main()
