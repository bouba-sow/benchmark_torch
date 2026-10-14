#!/usr/bin/env python3
"""Concatenate every <dataset>/<track>/model_summary.csv under an RDM-regression output root
into <root>/all_model_summaries.csv (long format, one row per dataset/measure/track/model/
confound variant). No cross-dataset normalisation: per-dataset scores stay raw delta LL per
pair until an RDM-level noise ceiling exists (docs/rdm_regression.md)."""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True)
root = ap.parse_args().root
paths = sorted(root.glob("*/*/model_summary.csv"))
if not paths:
    raise SystemExit(f"no model_summary.csv under {root}")
frames = [pd.read_csv(p) for p in paths]
out = pd.concat(frames, ignore_index=True)
# Neural summaries are per dataset (subject-averaged) and have no reading `measure`.
if "measure" not in out:
    out["measure"] = ""
out["measure"] = out["measure"].fillna("neural")
out.to_csv(root / "all_model_summaries.csv", index=False)
print(out.groupby(["dataset", "track", "measure"]).model_id.nunique().rename("n_models").to_string())
