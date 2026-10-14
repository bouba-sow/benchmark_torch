#!/usr/bin/env python3
"""Split-half theoretical-upper-bound delta log-likelihood ceiling for
Broderick's pooled, crossed subject x region LME. See
neuroai.regressors.broderick_ceiling.compute_broderick_ceiling_delta_loglik
for the method (adapted from scripts/ceilings/dll_ecog.py's nested
subject:electrode template).

Reuses the ALREADY-CACHED pooled base table the real sweep builds
(broderick_lme_table.parquet -- identical across every model/layer, only
surprisal, cached separately, differs), so no raw EEG epoching is needed
here.

Usage:
    .venv/bin/python scripts/ceilings/dll_broderick.py --n_splits 15 --n_workers 30

Output: outputs/noise_ceilings/delta_loglik_ceiling_broderick_<suffix>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd

from neuroai.regressors.broderick_ceiling import compute_broderick_ceiling_delta_loglik

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT_DIR = REPO_ROOT / "outputs" / "noise_ceilings"

# Any model works -- the pooled base table is identical across every
# model/layer in a sweep (only surprisal, cached separately, differs).
_TABLE = (REPO_ROOT / "outputs_jeanzay" / "broderick/surprisal_sweep" / "mamba2"
          / "AntonV_mamba2-370m-hf" / "broderick_lme_table.parquet")


def _load_broderick_pooled(table: str | None = None) -> pd.DataFrame:
    """subject, region, chunk, word_idx, wlen, freq, eeg already in the
    cached table -- word_idx (unique per canonical word) is the split-half
    item key."""
    path = Path(table) if table else _TABLE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- run the broderick.surprisal.fit stage for "
            "at least one model first (any family/model works)."
        )
    return pd.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_splits", type=int, default=15)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("split_half", "loso"), default="split_half")
    ap.add_argument("--n_repeats", type=int, default=1, help="loso: repeats over chunk splits (seed+rep)")
    ap.add_argument("--out_of_sample", dest="out_of_sample", action="store_true", default=True)
    ap.add_argument("--in_sample", dest="out_of_sample", action="store_false")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--table", default=None, help="pooled target parquet (default: the production mamba2 table)")
    ap.add_argument("--tag", default="", help="suffix added to the output JSON name")
    args = ap.parse_args()

    base_df = _load_broderick_pooled(args.table)
    if base_df.empty:
        raise ValueError("No pooled rows for broderick")
    logger.info(
        "[broderick] %d rows, %d subjects, %d regions, %d items (item_col=word_idx)",
        len(base_df), base_df["subject"].nunique(), base_df["region"].nunique(),
        base_df["word_idx"].nunique(),
    )

    t0 = time.time()
    result = compute_broderick_ceiling_delta_loglik(
        base_df, item_col="word_idx",
        n_splits=args.n_splits, seed=args.seed, n_workers=args.n_workers,
        out_of_sample=args.out_of_sample, mode=args.mode, n_repeats=args.n_repeats,
    )
    elapsed = time.time() - t0

    logger.info(
        "[broderick] ceiling delta-LL/obs = %.5f (sd=%.5f, %d/%d iterations converged, %.1fs)",
        result.ceiling_delta_loglik_per_obs, result.ceiling_delta_loglik_per_obs_sd,
        result.n_iterations_used, result.n_iterations_requested, elapsed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("oos" if args.out_of_sample else "insample") + ("" if args.mode == "split_half" else "_" + args.mode)
    out_path = out_dir / f"delta_loglik_ceiling_broderick_{suffix}{('_' + args.tag) if args.tag else ''}.json"
    out_path.write_text(json.dumps({
        "dataset": "broderick",
        "ceiling_delta_loglik_per_obs": result.ceiling_delta_loglik_per_obs,
        "ceiling_delta_loglik_per_obs_sd": result.ceiling_delta_loglik_per_obs_sd,
        "n_iterations_used": result.n_iterations_used,
        "n_iterations_requested": result.n_iterations_requested,
        "n_splits": args.n_splits,
        "out_of_sample": args.out_of_sample,
        "mode": args.mode,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "elapsed_s": elapsed,
        "per_iteration_delta_loglik": result.per_iteration_delta_loglik,
        "per_iteration_n_obs": result.per_iteration_n_obs,
    }, indent=2))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
