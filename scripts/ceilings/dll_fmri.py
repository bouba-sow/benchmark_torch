#!/usr/bin/env python3
"""Split-half theoretical-upper-bound delta log-likelihood ceiling for the
three per-subject fMRI LME datasets -- Natural Stories fMRI, Tuckute, and
Pereira. See neuroai.regressors.fmri_ceiling.compute_fmri_ceiling_delta_loglik
for the method, and that module's docstring for the Natural Stories fMRI
item-key caveat (row-order-within-story alignment, not an explicit TR
index). LPP fMRI is NOT covered -- its real pipeline has no per-subject
structure at all (average-subject BOLD target); see fmri_ceiling.py's
module docstring.

Both loaders reuse the ALREADY-CACHED tables the real sweeps build
(identical across every model/layer -- only surprisal, stored separately,
differs), so no raw fMRI/nilearn processing is needed here.

Usage:
    .venv/bin/python scripts/ceilings/dll_fmri.py \\
        --dataset tuckute --n_splits 15 --n_workers 30

Output: outputs/noise_ceilings/delta_loglik_ceiling_<dataset>_fmri.json
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

from neuroai.regressors.fmri_ceiling import compute_fmri_ceiling_delta_loglik

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT_DIR = REPO_ROOT / "outputs" / "noise_ceilings"

_TUCKUTE_TABLE = (REPO_ROOT / "outputs_jeanzay" / "tuckute/surprisal_sweep" / "mamba2"
                  / "AntonV_mamba2-370m-hf" / "tuckute_lme_pooled_table.parquet")
_PEREIRA_TABLE = (REPO_ROOT / "outputs_jeanzay" / "pereira/surprisal_sweep" / "mamba2"
                  / "AntonV_mamba2-130m-hf" / "pereira_lme_pooled_table_main.parquet")
_NS_FMRI_DIR = (REPO_ROOT / "outputs_jeanzay" / "ns_fmri/surprisal_sweep" / "mamba"
                / "state-spaces_mamba-370m-hf")
_NS_FMRI_NUISANCE = _NS_FMRI_DIR / "ns_fmri_lme_nuisance_table.parquet"
_NS_FMRI_BOLD = _NS_FMRI_DIR / "ns_fmri_lme_bold.npy"
_NS_FMRI_DATA_ROOT = REPO_ROOT / "data" / "naturalstories_fmri" / "osfstorage"

# A canonical LANG-network left-IFG fROI -- a standard, non-cherry-picked
# language-network region (not selected by looking at any particular
# model's best-scoring fROI).
_NS_FMRI_FROI = "LangLIFG"


def _load_tuckute_pooled() -> tuple[pd.DataFrame, str, str]:
    """subject, roi, sentence_id, mean_wlen, mean_freq, y already in the
    cached table -- sentence_id (unambiguous) is the split-half item key."""
    df = pd.read_parquet(_TUCKUTE_TABLE).drop(columns=["surprisal"], errors="ignore")
    return df, "sentence_id", "tuckute"


def _load_pereira_pooled() -> tuple[pd.DataFrame, str, str]:
    """subject, roi, sentence_id, sentlen, mean_wlen, mean_freq, y already
    in the cached table (identical across every model/layer, revision-
    namespaced as "_main" since Pereira's LME sweep is per-checkpoint-
    capable in principle even though only "main" was ever run) --
    sentence_id (unambiguous, 627 unique sentences) is the split-half item
    key, same convention as Tuckute."""
    df = pd.read_parquet(_PEREIRA_TABLE)
    return df, "sentence_id", "pereira"


def _load_ns_fmri_pooled() -> tuple[pd.DataFrame, str, str]:
    """Merge one fROI's BOLD column onto the cached nuisance table. See
    fmri_ceiling.py's module docstring for the row-order-within-(subject,
    story) item-key caveat -- there is no explicit TR-index column."""
    from neuroai.datasets.natural_stories_fmri import available_frois

    df = pd.read_parquet(_NS_FMRI_NUISANCE)
    bold_all = np.load(_NS_FMRI_BOLD)
    frois = available_frois(str(_NS_FMRI_DATA_ROOT), "LANG_MD")
    if _NS_FMRI_FROI not in frois:
        raise ValueError(f"{_NS_FMRI_FROI!r} not in available fROIs: {frois}")
    froi_idx = frois.index(_NS_FMRI_FROI)
    df = df.copy()
    df["bold"] = bold_all[:, froi_idx].astype(np.float64)
    df = df.dropna(subset=["bold"]).reset_index(drop=True)
    # Row-order-within-(subject, story) as the item key: same TR position
    # within a story is assumed to line up across subjects (see caveat).
    df["tr_pos"] = df.groupby(["subject", "story"]).cumcount()
    df["item_id"] = df["story"].astype(str) + "_" + df["tr_pos"].astype(str)
    return df, "item_id", "ns_fmri"


DATASET_LOADERS = {
    "tuckute": _load_tuckute_pooled,
    "ns_fmri": _load_ns_fmri_pooled,
    "pereira": _load_pereira_pooled,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_LOADERS), required=True)
    ap.add_argument("--n_splits", type=int, default=15)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_of_sample", dest="out_of_sample", action="store_true", default=True)
    ap.add_argument("--in_sample", dest="out_of_sample", action="store_false")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    base_df, item_col, dataset = DATASET_LOADERS[args.dataset]()
    if base_df.empty:
        raise ValueError(f"No pooled rows for {args.dataset!r}")
    logger.info(
        "[%s] %d rows, %d subjects, %d items (item_col=%s)",
        args.dataset, len(base_df), base_df["subject"].nunique(),
        base_df[item_col].nunique(), item_col,
    )

    t0 = time.time()
    result = compute_fmri_ceiling_delta_loglik(
        base_df, dataset=dataset, item_col=item_col,
        n_splits=args.n_splits, seed=args.seed, n_workers=args.n_workers,
        out_of_sample=args.out_of_sample,
    )
    elapsed = time.time() - t0

    logger.info(
        "[%s] ceiling delta-LL/obs = %.5f (sd=%.5f, %d/%d iterations converged, %.1fs)",
        args.dataset, result.ceiling_delta_loglik_per_obs, result.ceiling_delta_loglik_per_obs_sd,
        result.n_iterations_used, result.n_iterations_requested, elapsed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "oos" if args.out_of_sample else "insample"
    out_path = out_dir / f"delta_loglik_ceiling_{args.dataset}_fmri_{suffix}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "method": "timkey_et_al_2026_split_half_fmri_adapted",
        "out_of_sample": args.out_of_sample,
        "item_col": item_col,
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
