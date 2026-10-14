#!/usr/bin/env python3
"""Split-half delta log-likelihood ceiling for Podcast with ONE MODEL PER REGION on
region-averaged electrodes (no electrode random effect).

Same split-half oracle idea as ``scripts/ceilings/dll_ecog.py`` (Timkey et al. 2026, adapted),
but the unit is subject x region instead of subject:electrode:

    baseline: y ~ wlen_z + freq_z + (1 | subject)
    full:     y ~ wlen_z + freq_z + surprisal_z + (1 | subject)       # surprisal_z = split-half oracle

Standardisation (all statistics from TRAIN chunks only, applied to train and test, so no
held-out information enters the scaling):
  1. chunk split first (``split_train_test_chunks``, random_state = seed, i.e. the sweep's split);
  2. y z-scored per (subject, electrode) with train-chunk mean/std;
  3. electrodes averaged within (subject, region, word);
  4. the averaged signal z-scored again per (subject, region) with train-chunk mean/std.
The oracle for a target subject's word = mean over the SOURCE-half subjects' standardised
region-averaged y for that word (other subjects only, by construction of the split-half design).
Predictors wlen/freq/oracle are z-scored over the region frame (label-free, same convention as
``prepare_podcast_lme_frame`` in the real sweep).

Uses the cached pooled table of the real sweep (identical across models), one lag column.
Regions: ``assigned`` left-hemisphere electrodes of outputs/podcast/electrode_roi_map_language_responsive.csv;
regions (IFGorb merged into IFG, see neuroai/regressors/podcast_region_lme.py) with fewer than --min_subjects subjects are skipped (a split-half needs >= 2 per half).

Usage:
    .venv/bin/python scripts/ceilings/dll_podcast_region.py --lag 16 --n_splits 10 --n_workers 20 \\
        --out_dir outputs_jeanzay/noise_ceilings/podcast_lags_region
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.regressors.lme import _fit_stats, _partition_loglik_detailed
from neuroai.regressors.podcast_lme import prepare_podcast_lme_frame
from neuroai.regressors.podcast_region_lme import BASELINE, FULL, build_region_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_SWEEP_DIR = (REPO_ROOT / "outputs_jeanzay" / "podcast/surprisal_sweep" / "mamba2" / "AntonV_mamba2-370m-hf")
_TABLE = _SWEEP_DIR / "podcast_lme_pooled_table.parquet"
_Y = _SWEEP_DIR / "podcast_lme_pooled_y.npy"
_ROI_MAP = REPO_ROOT / "outputs" / "podcast" / "electrode_roi_map_language_responsive.csv"

def _worker(region: str, iteration: int, direction: int, tgt: pd.DataFrame) -> tuple[str, int, int, float | None, int, bool]:
    """Fit baseline/full on the target half's TRAIN chunks, score the held-out chunks."""
    from pymer4.models import lmer

    try:
        tgt = tgt.assign(electrode=region)
        data = prepare_podcast_lme_frame(tgt)
        # train/test membership is the one fixed in build_region_table (the chunks whose statistics
        # standardised y) -- never re-derived here, so the scaling and the fit see the same split.
        mask = tgt["is_train"].to_numpy()
        train_data, test_data = data.filter(pl.Series(mask)), data.filter(pl.Series(~mask))
        if test_data.height < 10:
            return region, iteration, direction, None, 0, False
        bm = lmer(BASELINE, data=train_data)
        bm.fit(verbose=False, summary=False, conf_method="wald")
        _, _, _, base_ok = _fit_stats(bm)
        fm = lmer(FULL, data=train_data)
        fm.fit(verbose=False, summary=False, conf_method="wald")
        _, _, _, full_ok = _fit_stats(fm)
        base_ll, _ = _partition_loglik_detailed(bm, test_data, outcome="y")
        full_ll, _ = _partition_loglik_detailed(fm, test_data, outcome="y")
    except Exception as exc:  # noqa: BLE001
        logger.warning("fit failed region=%s split=%d dir=%d: %s", region, iteration, direction, exc)
        return region, iteration, direction, None, 0, False
    conv = bool(base_ok and full_ok)
    delta = full_ll - base_ll if np.isfinite(base_ll) and np.isfinite(full_ll) else None
    ok = delta is not None and np.isfinite(delta) and conv
    return region, iteration, direction, (delta if ok else None), int(test_data.height), conv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lag", type=int, required=True, help="lag index 0-31 (time = -0.2 + lag/32 s)")
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.75)
    ap.add_argument("--min_subjects", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    t0 = time.time()
    tab, regions = build_region_table(_TABLE, _Y, _ROI_MAP, [args.lag], seed=args.seed,
                                      train_frac=args.train_frac, min_subjects=args.min_subjects)
    tab = tab.rename(columns={f"y_{args.lag}": "y"})
    subjects = np.asarray(sorted(tab["subject"].unique()))
    logger.info("regions=%s, %d rows, %d subjects", regions, len(tab), len(subjects))

    rng = np.random.default_rng(args.seed)
    tasks = []
    for it in range(args.n_splits):
        perm = rng.permutation(subjects)
        half = len(perm) // 2
        halves = (set(perm[:half].tolist()), set(perm[half:].tolist()))
        for direction, (src_ids, tgt_ids) in enumerate((halves, halves[::-1])):
            for region in regions:
                reg = tab[tab["region"] == region]
                src = reg[reg["subject"].isin(src_ids)]
                tgt = reg[reg["subject"].isin(tgt_ids)]
                if src.empty or tgt["subject"].nunique() < 1:
                    continue
                oracle = src.groupby("word_idx")["y"].mean().rename("surprisal").reset_index()
                tgt = tgt.drop(columns=["surprisal"], errors="ignore").merge(oracle, on="word_idx", how="left")
                tgt = tgt.dropna(subset=["surprisal", "y"])
                if not tgt.empty:
                    tasks.append((region, it, direction, tgt))
    logger.info("%d tasks (regions x splits x directions)", len(tasks))

    results = []
    kw: dict = {}
    if args.n_workers > 1:
        # spawn, not fork: the parent has already run polars (thread pool) and loaded R/rpy2; forked
        # children of such a parent hang (observed: 20 workers idle at <1% CPU for 38 min).
        with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=mp.get_context("spawn")) as pool:
            futs = [pool.submit(_worker, r, it, d, tgt, **kw) for r, it, d, tgt in tasks]
            for f in as_completed(futs):
                results.append(f.result())
    else:
        results = [_worker(r, it, d, tgt, **kw) for r, it, d, tgt in tasks]
    logger.info("fits done: %d/%d tasks returned", len(results), len(tasks))

    # per (region, split): sum both directions, an iteration fails if either direction failed
    per_region: dict[str, dict] = {}
    for region in regions:
        iters = []
        for it in range(args.n_splits):
            rs = [x for x in results if x[0] == region and x[1] == it]
            if len(rs) == 2 and all(x[3] is not None for x in rs):
                iters.append((sum(x[3] for x in rs), sum(x[4] for x in rs)))
        per_obs = [d / n for d, n in iters]
        per_region[region] = {
            "ceiling_delta_loglik_per_obs": float(np.mean(per_obs)) if per_obs else float("nan"),
            "ceiling_delta_loglik_per_obs_sd": float(np.std(per_obs, ddof=1)) if len(per_obs) > 1 else float("nan"),
            "n_iterations_used": len(iters), "n_iterations_requested": args.n_splits,
            "per_iteration_delta_loglik": [d for d, _ in iters], "per_iteration_n_obs": [n for _, n in iters],
        }
        logger.info("region %-9s ceiling dLL/obs = %.5f (sd %.5f, %d/%d iters)", region,
                    per_region[region]["ceiling_delta_loglik_per_obs"],
                    per_region[region]["ceiling_delta_loglik_per_obs_sd"], len(iters), args.n_splits)

    # combined across regions: per split sum(dLL)/sum(n_obs), then mean over splits
    comb = []
    for it in range(args.n_splits):
        parts = [(per_region[r]["per_iteration_delta_loglik"], per_region[r]["per_iteration_n_obs"]) for r in regions]
        if all(len(p[0]) == args.n_splits for p in parts):  # only when every region kept all splits
            comb.append(sum(p[0][it] for p in parts) / sum(p[1][it] for p in parts))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"delta_loglik_ceiling_podcast_region_oos_lag{args.lag:02d}.json"
    out.write_text(json.dumps({
        "dataset": "podcast", "method": "split_half_region_averaged_per_region_lme", "out_of_sample": True,
        "lag_index": args.lag, "lag_time_s": -0.2 + args.lag / 32, "seed": args.seed,
        "baseline_formula": BASELINE, "full_formula": FULL, "regions": regions,
        "per_region": per_region,
        "combined_ceiling_delta_loglik_per_obs": float(np.mean(comb)) if comb else float("nan"),
        "combined_ceiling_sd": float(np.std(comb, ddof=1)) if len(comb) > 1 else float("nan"),
        "wall_time_s": time.time() - t0,
    }, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
