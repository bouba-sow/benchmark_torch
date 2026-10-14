#!/usr/bin/env python3
"""Split-half delta-LL ceiling of the Podcast per-region LME, BROKEN DOWN BY DLT COST BIN.

Identical procedure to ``scripts/ceilings/dll_podcast_region.py`` (same table, split seed, train-only
standardisation, split-half oracle surprisal, baseline/full LME fitted on the target half's TRAIN chunks, scored on
its held-out chunks); the only difference is that the held-out per-observation log-likelihood gain
(ll_full - ll_base) is summed per DLT bin (outputs/dlt/podcast_dlt_cost.csv: easy = cost 0, hard = cost >= 1..5,
plus ``all``) instead of over all words. The ``all`` bin reproduces the original ceiling exactly.
Output: <out_dir>/delta_loglik_ceiling_podcast_region_bins_lagNN.json
  per_region[region][bin] = {ceiling_delta_loglik_per_obs, _sd, n_iterations_used, n_obs_mean, n_words_unique}

    .venv/bin/python scripts/ceilings/dll_podcast_region_bins.py --lag 16 --n_workers 20 \\
        --out_dir outputs_jeanzay/noise_ceilings/podcast_lags_region_bins
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
_SWEEP_DIR = REPO_ROOT / "outputs_jeanzay" / "podcast/surprisal_sweep" / "mamba2" / "AntonV_mamba2-370m-hf"
_TABLE, _Y = _SWEEP_DIR / "podcast_lme_pooled_table.parquet", _SWEEP_DIR / "podcast_lme_pooled_y.npy"
_ROI_MAP = REPO_ROOT / "outputs" / "podcast" / "electrode_roi_map_language_responsive.csv"
_COST = REPO_ROOT / "outputs" / "dlt" / "podcast_dlt_cost.csv"
BINS = ["all", "easy", "hard1", "hard2", "hard3", "hard4", "hard5"]


def _bin_masks(cost: np.ndarray) -> dict[str, np.ndarray]:
    return {"all": np.isfinite(cost), "easy": cost <= 0} | {f"hard{k}": cost >= k for k in range(1, 6)}


def _worker(region: str, iteration: int, direction: int, tgt: pd.DataFrame, cost_by_word: dict):
    from pymer4.models import lmer

    try:
        data = prepare_podcast_lme_frame(tgt.assign(electrode=region))
        mask = tgt["is_train"].to_numpy()
        train_data, test_data = data.filter(pl.Series(mask)), data.filter(pl.Series(~mask))
        if test_data.height < 10:
            return region, iteration, direction, None
        bm = lmer(BASELINE, data=train_data); bm.fit(verbose=False, summary=False, conf_method="wald")
        _, _, _, base_ok = _fit_stats(bm)
        fm = lmer(FULL, data=train_data); fm.fit(verbose=False, summary=False, conf_method="wald")
        _, _, _, full_ok = _fit_stats(fm)
        _, base_obs = _partition_loglik_detailed(bm, test_data, outcome="y")
        _, full_obs = _partition_loglik_detailed(fm, test_data, outcome="y")
        if not (base_ok and full_ok):
            return region, iteration, direction, None
        d = np.asarray(full_obs) - np.asarray(base_obs)
        widx = test_data["word_idx"].to_numpy()
        cost = np.array([cost_by_word.get(int(w), np.nan) for w in widx])
        out = {}
        for b, m in _bin_masks(cost).items():
            ok = m & np.isfinite(d)
            out[b] = (float(d[ok].sum()), int(ok.sum()), int(len(np.unique(widx[ok]))))
        return region, iteration, direction, out
    except Exception as exc:  # noqa: BLE001
        logger.warning("fit failed region=%s split=%d dir=%d: %s", region, iteration, direction, exc)
        return region, iteration, direction, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lag", type=int, required=True)
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.75)
    ap.add_argument("--min_subjects", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    t0 = time.time()
    cost_by_word = pd.read_csv(_COST).drop_duplicates("word_idx").set_index("word_idx")["total"].to_dict()
    tab, regions = build_region_table(_TABLE, _Y, _ROI_MAP, [args.lag], seed=args.seed,
                                      train_frac=args.train_frac, min_subjects=args.min_subjects)
    tab = tab.rename(columns={f"y_{args.lag}": "y"})
    subjects = np.asarray(sorted(tab["subject"].unique()))
    rng = np.random.default_rng(args.seed)
    tasks = []
    for it in range(args.n_splits):               # same RNG stream / split construction as dll_podcast_region.py
        perm = rng.permutation(subjects)
        half = len(perm) // 2
        halves = (set(perm[:half].tolist()), set(perm[half:].tolist()))
        for direction, (src_ids, tgt_ids) in enumerate((halves, halves[::-1])):
            for region in regions:
                reg = tab[tab["region"] == region]
                src, tgt = reg[reg["subject"].isin(src_ids)], reg[reg["subject"].isin(tgt_ids)]
                if src.empty or tgt["subject"].nunique() < 1:
                    continue
                oracle = src.groupby("word_idx")["y"].mean().rename("surprisal").reset_index()
                tgt = tgt.drop(columns=["surprisal"], errors="ignore").merge(oracle, on="word_idx", how="left")
                tgt = tgt.dropna(subset=["surprisal", "y"])
                if not tgt.empty:
                    tasks.append((region, it, direction, tgt))
    logger.info("%d tasks", len(tasks))
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(_worker, r, it, d, tgt, cost_by_word) for r, it, d, tgt in tasks]
        for f in as_completed(futs):
            results.append(f.result())

    per_region: dict = {}
    for region in regions:
        per_region[region] = {}
        for b in BINS:
            per_obs, ns, uw = [], [], []
            for it in range(args.n_splits):
                rs = [x for x in results if x[0] == region and x[1] == it]
                if len(rs) == 2 and all(x[3] is not None for x in rs):
                    s, n = sum(x[3][b][0] for x in rs), sum(x[3][b][1] for x in rs)
                    if n > 0:
                        per_obs.append(s / n); ns.append(n); uw.append(max(x[3][b][2] for x in rs))
            per_region[region][b] = {
                "ceiling_delta_loglik_per_obs": float(np.mean(per_obs)) if per_obs else float("nan"),
                "ceiling_delta_loglik_per_obs_sd": float(np.std(per_obs, ddof=1)) if len(per_obs) > 1 else float("nan"),
                "n_iterations_used": len(per_obs), "n_obs_mean": float(np.mean(ns)) if ns else 0.0,
                "n_words_unique_max": int(max(uw)) if uw else 0}
        logger.info("region %-9s all=%.5f easy=%.5f hard1=%.5f hard2=%.5f", region,
                    *(per_region[region][b]["ceiling_delta_loglik_per_obs"] for b in ("all", "easy", "hard1", "hard2")))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"delta_loglik_ceiling_podcast_region_bins_lag{args.lag:02d}.json"
    out.write_text(json.dumps({"dataset": "podcast", "method": "split_half_region_per_region_lme_by_dlt_bin",
                               "lag_index": args.lag, "seed": args.seed, "bins": BINS, "regions": regions,
                               "per_region": per_region, "wall_time_s": time.time() - t0}, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
