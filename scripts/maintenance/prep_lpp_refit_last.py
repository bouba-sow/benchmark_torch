#!/usr/bin/env python3
"""Invalidate each main LPP-LME model's LAST-layer row in lpp_lme_by_layer.csv
so a --last_layer_only --save_word_predictions rerun actually refits it.

Why this is needed: run/surprisal/neural/lpp_fit.py's resume logic skips any layer
already marked "done" (no error) in lpp_lme_by_layer.csv -- so simply
re-running with --save_word_predictions would see the last layer already
fit and never re-enter the fit code path that writes lpp_lme_per_word.csv.
Removing just that one row (revision="main" only, every other layer left
untouched) forces the next run to refit exactly the last layer, reusing the
already-cached surprisal (.npy) and pooled table (.parquet) -- no GPU/LLM
work is repeated.

Run ONCE on the login node before submitting the last-layer refit array
jobs (python scripts/submit.py lpp surprisal refit --family <family> --env REFIT_LAYERS=last).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, OUTPUTS_JEANZAY, model_dir_to_model_id

SWEEP = OUTPUTS_JEANZAY / "lpp/surprisal_sweep"
# Training-checkpoint side-sweeps, not main models -- leave untouched.
SKIP_FAMILY_DIRS = {"pythia_checkpoints", "babylm_gpt2_checkpoints"}


def main() -> None:
    n_touched = 0
    n_skipped_excluded = 0
    for family_dir in sorted(SWEEP.iterdir()):
        if not family_dir.is_dir() or family_dir.name in SKIP_FAMILY_DIRS:
            continue
        for model_dir in sorted(family_dir.iterdir()):
            model_id = model_dir_to_model_id(model_dir.name)
            if model_id in EXCLUDED_MODEL_IDS:
                n_skipped_excluded += 1
                continue
            csv = model_dir / "lpp_lme_by_layer.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if df.empty or "layer" not in df.columns:
                continue
            rev = df["revision"].fillna("main") if "revision" in df.columns else pd.Series(["main"] * len(df))
            main_rows = df[rev == "main"]
            if main_rows.empty:
                continue
            last_layer = main_rows["layer"].max()
            before = len(df)
            keep = df[~((rev == "main") & (df["layer"] == last_layer))]
            if len(keep) == before:
                continue
            keep.to_csv(csv, index=False)
            n_touched += 1
            print(f"  {family_dir.name}/{model_dir.name}: invalidated layer {int(last_layer)}")
    print(f"\ndone -- {n_touched} model(s) prepped for last-layer refit "
          f"({n_skipped_excluded} excluded model(s) left untouched)")


if __name__ == "__main__":
    main()
