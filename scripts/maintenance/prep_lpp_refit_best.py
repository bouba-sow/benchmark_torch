#!/usr/bin/env python3
"""Second prep pass, alongside maintenance/prep_lpp_refit_last.py: also
invalidate each main LPP-LME model's BEST-layer row (by held-out
delta_loglik_test, falling back to test_delta_pearsonr) so the next sweep
run refits last AND best layer -- exactly two layers per model, both with
--save_word_predictions on.

"Best" is computed from whatever rows are still present (the last-layer row
was already removed by the first prep pass) -- if the true best happened to
BE the last layer, this picks the runner-up instead, which is harmless: the
real last layer gets refit anyway by the other invalidated row, so the
overall refit set still covers the genuinely-best layer (or is one layer
short of it in the rare case the runner-up isn't close -- acceptable given
how cheap this is meant to be).

Run ONCE, after maintenance/prep_lpp_refit_last.py, before submitting the
last+best refit jobs (python scripts/submit.py lpp surprisal refit --family <family>).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, OUTPUTS_JEANZAY, model_dir_to_model_id

SWEEP = OUTPUTS_JEANZAY / "lpp/surprisal_sweep"
SKIP_FAMILY_DIRS = {"pythia_checkpoints", "babylm_gpt2_checkpoints"}


def _score_col(df: pd.DataFrame) -> str | None:
    for c in ("delta_loglik_test", "test_delta_pearsonr"):
        if c in df.columns and df[c].notna().any():
            return c
    return None


def main() -> None:
    n_touched = 0
    n_already_gone = 0
    for family_dir in sorted(SWEEP.iterdir()):
        if not family_dir.is_dir() or family_dir.name in SKIP_FAMILY_DIRS:
            continue
        for model_dir in sorted(family_dir.iterdir()):
            model_id = model_dir_to_model_id(model_dir.name)
            if model_id in EXCLUDED_MODEL_IDS:
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
                n_already_gone += 1
                continue
            col = _score_col(main_rows)
            if col is None:
                continue
            best_layer = main_rows.loc[main_rows[col].idxmax(), "layer"]
            before = len(df)
            keep = df[~((rev == "main") & (df["layer"] == best_layer))]
            if len(keep) == before:
                continue
            keep.to_csv(csv, index=False)
            n_touched += 1
            print(f"  {family_dir.name}/{model_dir.name}: invalidated best layer "
                  f"{int(best_layer)} (by {col})")
    print(f"\ndone -- {n_touched} model(s) prepped for best-layer refit "
          f"({n_already_gone} model(s) already had no 'main' rows left, e.g. best==last)")


if __name__ == "__main__":
    main()
