#!/usr/bin/env python3
"""Split-half theoretical-upper-bound delta log-likelihood ceiling for the
LPP fMRI pooled LME, split by SUBJECT (English subjects only). See
neuroai.regressors.lpp_ceiling.compute_lpp_ceiling_delta_loglik for the
method and neuroai.regressors.lpp_persubject for how the per-subject BOLD
table (built by scripts/data/lpp_persubject_bold_table.py -- run that
first) is produced.

wlen/freq/sentpos are pulled from an existing cached
lpp_lme_pooled_table.parquet (any model works -- identical across every
model, only surprisal differs) by (region, section, tr_idx) -- they depend
only on words/HRF design, not on which subject's BOLD is used.

Usage:
    .venv/bin/python scripts/ceilings/dll_lpp.py \\
        --n_splits 15 --n_workers 20 --out_of_sample

Output: outputs/noise_ceilings/delta_loglik_ceiling_lpp_fmri_persubject_<insample|oos>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.regressors.lpp_ceiling import compute_lpp_ceiling_delta_loglik

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT_DIR = REPO_ROOT / "outputs" / "noise_ceilings"

_PERSUBJECT_BOLD = OUT_DIR / "lpp_persubject_bold.parquet"
_NUISANCE_TABLE = (
    REPO_ROOT / "outputs_jeanzay" / "lpp/surprisal_sweep" / "babylm"
    / "colinglab_CLASS_IT-140M" / "lpp_lme_pooled_table.parquet"
)


def _load_lpp_persubject_pooled() -> pd.DataFrame:
    bold_df = pd.read_parquet(_PERSUBJECT_BOLD)
    bold_df["tr_idx"] = bold_df.groupby(["subject", "region", "section"]).cumcount()
    bold_df["region"] = bold_df["region"].astype(str)
    bold_df["section"] = bold_df["section"].astype(str)

    nuis = pd.read_parquet(_NUISANCE_TABLE, columns=["region", "section", "wlen", "freq", "sentpos"])
    nuis["region"] = nuis["region"].astype(str)
    nuis["section"] = nuis["section"].astype(str)
    nuis["tr_idx"] = nuis.groupby(["region", "section"]).cumcount()

    df = bold_df.merge(nuis, on=["region", "section", "tr_idx"], how="inner")
    dropped = len(bold_df) - len(df)
    if dropped:
        logger.warning(
            "%d/%d per-subject rows had no matching (region, section, tr_idx) in the "
            "cached nuisance table (TR-count mismatch) -- dropped.", dropped, len(bold_df),
        )
    df["item_id"] = df["section"] + "_" + df["tr_idx"].astype(str)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_splits", type=int, default=15)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_of_sample", dest="out_of_sample", action="store_true", default=True)
    ap.add_argument("--in_sample", dest="out_of_sample", action="store_false")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    df = _load_lpp_persubject_pooled()
    if df.empty:
        raise ValueError("No pooled rows for lpp_fmri_persubject")
    logger.info(
        "[lpp_fmri_persubject] %d rows, %d subjects, %d regions, %d items",
        len(df), df["subject"].nunique(), df["region"].nunique(), df["item_id"].nunique(),
    )

    t0 = time.time()
    result = compute_lpp_ceiling_delta_loglik(
        df, item_col="item_id", n_splits=args.n_splits, seed=args.seed,
        n_workers=args.n_workers, out_of_sample=args.out_of_sample,
    )
    elapsed = time.time() - t0

    logger.info(
        "[lpp_fmri_persubject] ceiling delta-LL/obs = %.5f (sd=%.5f, %d/%d iterations converged, %.1fs)",
        result.ceiling_delta_loglik_per_obs, result.ceiling_delta_loglik_per_obs_sd,
        result.n_iterations_used, result.n_iterations_requested, elapsed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "oos" if args.out_of_sample else "insample"
    out_path = out_dir / f"delta_loglik_ceiling_lpp_fmri_persubject_{suffix}.json"
    out_path.write_text(json.dumps({
        "dataset": "lpp_fmri_persubject",
        "method": "timkey_et_al_2026_split_half_lpp_persubject_adapted",
        "out_of_sample": args.out_of_sample,
        "item_col": "item_id",
        "n_splits_requested": result.n_iterations_requested,
        "n_splits_converged": result.n_iterations_used,
        "seed": args.seed,
        "ceiling_delta_loglik_per_obs": result.ceiling_delta_loglik_per_obs,
        "ceiling_delta_loglik_per_obs_sd": result.ceiling_delta_loglik_per_obs_sd,
        "per_iteration_delta_loglik": result.per_iteration_delta_loglik,
        "per_iteration_n_obs": result.per_iteration_n_obs,
        "per_direction_records": result.per_direction_records,
        "wall_time_s": elapsed,
    }, indent=2))
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
