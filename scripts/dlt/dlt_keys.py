#!/usr/bin/env python3
"""Join keys between per-word LME CSVs and ``outputs/dlt/{dataset}_dlt_cost.csv``.

The cost files are keyed on ``(story_id, zone)``. For every behavioral dataset except
Natural Stories the per-word CSVs carry ``sentid == story_id`` and ``sentpos == zone``.
Natural Stories' LME frame keeps the corpus' own sentence numbering instead (``sentid``
0..475, ``sentpos`` = position inside the sentence), so a direct cast matched ~4% of rows
and ~1% of those to the right word. ``outputs/dlt/naturalstories_sent_map.csv`` maps
(sentid, sentpos) -> (story_id, zone); build it once with

    .venv/bin/python scripts/dlt/dlt_keys.py build-ns-map \\
        --trials outputs_jeanzay/naturalstories/surprisal_sweep/gpt2/openai-community_gpt2/ns_lme_trials_base_main.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NS_SENT_MAP = REPO_ROOT / "outputs" / "dlt" / "naturalstories_sent_map.csv"


def attach_story_zone(per_word: pd.DataFrame, *, natural_stories: bool = False) -> pd.DataFrame:
    """Add integer ``story_id`` / ``zone`` columns to a per-word LME frame."""
    if not natural_stories:
        per_word["story_id"] = per_word["sentid"].astype(int)
        per_word["zone"] = per_word["sentpos"].astype(int)
        return per_word
    if not NS_SENT_MAP.is_file():
        raise FileNotFoundError(f"{NS_SENT_MAP} missing — run scripts/dlt/dlt_keys.py build-ns-map")
    sent_map = pd.read_csv(NS_SENT_MAP)
    out = per_word.assign(sentid=per_word["sentid"].astype(int), sentpos=per_word["sentpos"].astype(int))
    return out.merge(sent_map, on=["sentid", "sentpos"], how="left", validate="m:1")


def _build_ns_map(trials_path: str) -> None:
    t = pd.read_parquet(trials_path, columns=["story_id", "zone", "sentid", "sentpos"])
    m = t.drop_duplicates(["story_id", "zone"]).astype(int)
    assert not m.duplicated(["sentid", "sentpos"]).any(), "(sentid, sentpos) is not unique per word"
    NS_SENT_MAP.parent.mkdir(parents=True, exist_ok=True)
    m.sort_values(["story_id", "zone"]).to_csv(NS_SENT_MAP, index=False)
    print(f"wrote {len(m)} rows -> {NS_SENT_MAP}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-ns-map")
    p.add_argument("--trials", required=True)
    args = ap.parse_args()
    _build_ns_map(args.trials)
