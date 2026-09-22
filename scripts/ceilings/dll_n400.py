#!/usr/bin/env python3
"""Split-half theoretical-upper-bound delta log-likelihood ceiling for the
pooled N400 EEG LME datasets (Nieuwland, Szewczyk) -- see
neuroai.regressors.n400_pooled_lme.compute_n400_ceiling_delta_loglik for the
method (adapted from Timkey et al. 2026's behavioral split-half design; see
that module's docstring for why prev_surprisal_z is excluded here).

"nieuwland_paper" builds from the OSF project's own
validated dataset file (data/n400_paper_formula/osfstorage/datasets/) which already have
every covariate the original published analysis used precomputed -- no need
for this codebase's heavier raw-EEG MNE epoching pipeline for the ceiling
specifically (Nieuwland's own n400 column is already pre-averaged across
electrodes per trial in that source file).

Usage:
    .venv/bin/python scripts/ceilings/dll_n400.py \\
        --dataset nieuwland_paper --n_splits 15 --n_workers 30

Output: outputs/noise_ceilings/delta_loglik_ceiling_<dataset>_n400.json
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

from neuroai.regressors.n400_pooled_lme import compute_n400_ceiling_delta_loglik

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT_DIR = REPO_ROOT / "outputs" / "noise_ceilings"


def _load_nieuwland_paper_pooled() -> tuple[pd.DataFrame, bool]:
    """Build Nieuwland's pooled table directly from the OSF project's own
    validated dataset file -- n400 is already pre-averaged across the 6
    centro-parietal electrodes per trial there (no electrode column at all),
    unlike per-electrode long tables. See
    data/n400_paper_formula/osfstorage/calculate_fits/calculate_aics_nieuwland.R for the
    exact formula this is built to match (surprisal + lab fixed effect, pure
    random intercepts for subject/item, no slopes, no frequency covariate).

    ``item`` (1-80) is shared by both cloze conditions of the same base
    sentence -- item_id = item x cloze (where surprisal actually varies) is
    built with an item_id/frame_id split, with
    ``frame_id`` = the base item number reused as the train/test chunk
    boundary and the ``(1|item)`` random-intercept grouping."""
    path = REPO_ROOT / "data" / "87gmv" / "osfstorage" / "datasets" / "nieuwland_2018.tsv"
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"item": "base_item"})
    df = df.dropna(subset=["n400", "lab", "base_item"]).reset_index(drop=True)

    items = (
        df[["base_item", "cloze"]]
        .drop_duplicates()
        .sort_values(["base_item", "cloze"])
        .reset_index(drop=True)
    )
    items["item_id"] = items.index
    base_items = sorted(items["base_item"].unique())
    base_to_int = {b: i for i, b in enumerate(base_items)}
    items["frame_id"] = items["base_item"].map(base_to_int).astype(int)

    df = df.merge(items, on=["base_item", "cloze"], how="left")
    base_df = df[["subject", "lab", "frame_id", "item_id", "n400"]].copy()
    base_df["subject"] = base_df["subject"].astype(str)
    base_df["lab"] = base_df["lab"].astype(str)
    return base_df, False  # include_lab param is unused for nieuwland_paper (lab is a fixed effect in its formula)


def _load_szewczyk_pooled() -> tuple[pd.DataFrame, bool]:
    """Build Szewczyk's pooled table for the 'nieuwland_paper' re_structure:
    n400 is pre-averaged over the 4 centro-parietal ROI clusters and the
    300-500ms window (no electrode column at all), matching that structure's
    formula (surprisal + lab fixed effect + intercepts for subject/item, no
    frequency covariate). ``subexperiment`` plays the role Nieuwland's
    multi-lab ``lab`` column plays: 5 merged sub-experiments pooled via the
    same lab fixed effect.

    item_id = (subexperiment, item, critical_word) -- unique per cloze
    completion; frame_id = (subexperiment, item) -- shared by every cloze
    completion of the same sentence, used as the train/test chunk boundary
    and (1|item) random-intercept grouping. Mirrors
    neuroai/datasets/szewczyk.py's build_items_df/build_subject_items."""
    from neuroai.datasets.szewczyk import build_items_df, load_data
    path = REPO_ROOT / "data" / "Szewczyk_Federmeir" / "osfstorage" / "bind-5-dset_300-500.rds"
    df = load_data(path)
    items = build_items_df(df)
    df = df.merge(
        items[["subexperiment", "item", "critical_word", "item_id", "frame_id"]],
        on=["subexperiment", "item", "critical_word"], how="left",
    )
    base_df = df.rename(columns={"subexperiment": "lab"})[
        ["subject", "lab", "frame_id", "item_id", "n400"]
    ].copy()
    base_df["subject"] = base_df["subject"].astype(str)
    base_df["lab"] = base_df["lab"].astype(str)
    base_df = base_df.dropna(subset=["n400", "lab", "frame_id", "item_id"]).reset_index(drop=True)
    return base_df, False  # include_lab param is unused for nieuwland_paper (lab is a fixed effect in its formula)


def _load_nieuwland_real_pooled() -> tuple[pd.DataFrame, bool]:
    """Load the ALREADY-CACHED pooled table this codebase's real Nieuwland
    sweep builds (identical across every model/layer -- only surprisal,
    stored separately, differs). Used to test this codebase's DEFAULT
    formula (``re_structure='slopes'``, i.e. ``n400_lme_formulas``) rather
    than the paper-exact one, on Nieuwland's much larger sample (334
    subjects vs far fewer in the earlier N400 datasets).

    CAVEAT: this cached table's ``frame_id`` is the BASE item (80 values,
    shared by both cloze conditions) -- the real sweep only ever uses it for
    train/test chunk splitting, never as a fixed/random-effect grouping at
    finer granularity, so no condition-level (item x cloze) id was retained.
    Passing this as ``item_col`` to the ceiling means the split-half
    substitute predictor is a BASE-ITEM mean (constant across both
    conditions of an item), coarser than how real surprisal actually varies
    (per condition) -- this likely UNDERSTATES the true ceiling for this
    formula, but faithfully reflects the granularity our real pipeline's
    formula actually operates at."""
    path = (REPO_ROOT / "outputs_jeanzay" / "nieuwland/surprisal_sweep" / "babylm"
            / "colinglab_CLASS_IT-140M" / "nieuwland_lme_pooled_table.parquet")
    base_df = pd.read_parquet(path)
    return base_df, True  # include_lab=True (matches the real Nieuwland sweep's n400_lme_formulas call)


DATASET_LOADERS = {
    "nieuwland_paper": _load_nieuwland_paper_pooled,
    "szewczyk": _load_szewczyk_pooled,
    "nieuwland_real": _load_nieuwland_real_pooled,
}

# item_col override per dataset -- default is "item_id"; the "_real" loaders
# (cached tables from the actual sweep, which never retained a condition-level
# id) must use the coarser "frame_id" instead. See their loader docstrings.
DATASET_ITEM_COL = {
    "nieuwland_real": "frame_id",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_LOADERS), default="nieuwland_paper")
    ap.add_argument("--n_splits", type=int, default=15)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_of_sample", dest="out_of_sample", action="store_true", default=True,
                     help="Fit on a train chunk-split, evaluate on held-out chunks (default).")
    ap.add_argument("--in_sample", dest="out_of_sample", action="store_false",
                     help="Fit and score on ALL data (Timkey et al.'s literal in-sample description).")
    ap.add_argument("--re_structure", choices=["slopes", "nieuwland_paper"], default="slopes",
                     help="'slopes' = this codebase's default correlated-random-slope structure; "
                          "'nieuwland_paper' = surprisal+lab(fixed)+intercepts for subject/item, no "
                          "electrode/frequency term (use with --dataset nieuwland_paper or szewczyk).")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    base_df, include_lab = DATASET_LOADERS[args.dataset]()
    if base_df.empty:
        raise ValueError(f"No pooled rows after filtering for {args.dataset!r}")
    item_col = DATASET_ITEM_COL.get(args.dataset, "item_id")
    logger.info(
        "[%s] %d rows, %d subjects, %d items (item_col=%s, %d frames)",
        args.dataset, len(base_df), base_df["subject"].nunique(),
        base_df[item_col].nunique(), item_col, base_df["frame_id"].nunique(),
    )

    t0 = time.time()
    result = compute_n400_ceiling_delta_loglik(
        base_df, item_col=item_col, include_lab=include_lab,
        n_splits=args.n_splits, seed=args.seed, n_workers=args.n_workers,
        out_of_sample=args.out_of_sample, re_structure=args.re_structure,
    )
    elapsed = time.time() - t0

    logger.info(
        "[%s] ceiling delta-LL/obs = %.4f (sd=%.4f, %d/%d iterations converged, %.1fs)",
        args.dataset, result.ceiling_delta_loglik_per_obs, result.ceiling_delta_loglik_per_obs_sd,
        result.n_iterations_used, result.n_iterations_requested, elapsed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{'oos' if args.out_of_sample else 'insample'}_{args.re_structure}"
    out_path = out_dir / f"delta_loglik_ceiling_{args.dataset}_n400_{suffix}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "method": "timkey_et_al_2026_split_half_n400_adapted",
        "out_of_sample": args.out_of_sample,
        "re_structure": args.re_structure,
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
