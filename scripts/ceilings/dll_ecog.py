#!/usr/bin/env python3
"""Split-half theoretical-upper-bound delta log-likelihood ceiling for the
two nested subject:electrode ECoG/iEEG LME datasets -- Fedorenko2016 (5
subjects) and Podcast (9 subjects). See
neuroai.regressors.ecog_ceiling.compute_ecog_ceiling_delta_loglik for the
method. Both datasets are much smaller than any already-ceiling-tested
dataset (smaller EEG datasets were already noisy) -- expect a very noisy
estimate, possibly indistinguishable from zero.

Both loaders reuse the ALREADY-CACHED pooled table this codebase's real
sweeps build (identical across every model/layer -- only surprisal, stored
separately, differs), so no raw ECoG epoching is needed here.

Usage:
    .venv/bin/python scripts/ceilings/dll_ecog.py \\
        --dataset fedorenko2016 --n_splits 15 --n_workers 30

Output: outputs/noise_ceilings/delta_loglik_ceiling_<dataset>_ecog.json
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

from neuroai.regressors.ecog_ceiling import compute_ecog_ceiling_delta_loglik

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT_DIR = REPO_ROOT / "outputs" / "noise_ceilings"

# Any model works -- the pooled base table is identical across every
# model/layer in a sweep (only surprisal, cached separately, differs).
_FED_TABLE = (REPO_ROOT / "outputs_jeanzay" / "fedorenko2016/surprisal_sweep" / "mamba2"
              / "AntonV_mamba2-370m-hf" / "fedorenko2016_lme_pooled_table.parquet")
_PODCAST_TABLE = (REPO_ROOT / "outputs_jeanzay" / "podcast/surprisal_sweep" / "mamba2"
                  / "AntonV_mamba2-370m-hf" / "podcast_lme_pooled_table.parquet")
_PODCAST_Y = (REPO_ROOT / "outputs_jeanzay" / "podcast/surprisal_sweep" / "mamba2"
              / "AntonV_mamba2-370m-hf" / "podcast_lme_pooled_y.npy")

# FIXED 2026-08-29: was a single lag (index 16 == lag_time_s 0.300, i.e.
# 300ms post-word-onset) picked from the full 32-lag grid
# (np.linspace(-0.2, 0.8, 32, endpoint=False), 31.25ms/lag) -- but every
# checkpoint-trajectory score this ceiling is meant to normalise (Pythia/
# BabyLM, see scripts/submit.py (see docs/history/launcher_migration.csv)) uses
# MEAN-WINDOW mode over lags 0-200ms instead (--window_ms 0 200,
# PodcastLmeConfig.window_s), collapsing 32 lags into ONE fit. A single lag
# at 300ms is neither that window nor the main (non-checkpoint) models'
# best-of-32-lags convention -- a real three-way mismatch, caught via user
# question. Now matches the checkpoint convention exactly: mean of the same
# 6 lags (indices 7-12, times 0.019-0.175s) that window_s=(0, 0.2) selects.
_PODCAST_LAG_WINDOW_S = (0.0, 0.2)


def _load_fedorenko2016_pooled() -> tuple[pd.DataFrame, str, str]:
    """subject, electrode, sentence_id, wlen, freq, y already in the cached
    table -- stimulus_id (unique per word token) is the split-half item key."""
    df = pd.read_parquet(_FED_TABLE)
    return df, "stimulus_id", "fedorenko2016"


def _load_podcast_pooled(lag: int | None = None) -> tuple[pd.DataFrame, str, str]:
    """subject, electrode, chunk, word_idx, wlen, freq in the cached table;
    y is a separate (n_rows, n_lags) array -- average the lag columns
    falling in _PODCAST_LAG_WINDOW_S (matches the checkpoint sweeps'
    mean-window convention, see that constant's comment). word_idx (unique
    per transcript word position) is the split-half item key."""
    df = pd.read_parquet(_PODCAST_TABLE).drop(columns=["surprisal"], errors="ignore")
    y_all = np.load(_PODCAST_Y)
    n_lags = y_all.shape[1]
    # Same grid as configs/surprisal/neural/podcast.yaml: tmin=-0.2, tmax=0.8, n_lags points,
    # endpoint=False -- reconstructed here (not imported) since this script
    # only ever reads the already-cached y array, never the raw config.
    analysis_times = np.linspace(-0.2, 0.8, n_lags, endpoint=False)
    lo, hi = _PODCAST_LAG_WINDOW_S
    win_mask = (analysis_times >= lo) & (analysis_times <= hi)
    if lag is not None:
        if not 0 <= lag < n_lags:
            raise ValueError(f"--lag {lag} outside [0, {n_lags})")
        win_mask = np.zeros(n_lags, dtype=bool)
        win_mask[lag] = True
    if not win_mask.any():
        raise ValueError(f"window {_PODCAST_LAG_WINDOW_S} matches no lags in "
                          f"[{analysis_times.min():.3f}, {analysis_times.max():.3f}]s")
    logger.info("Podcast ceiling: averaging %d lag(s) in %ss (indices %s, times %s)",
                int(win_mask.sum()), _PODCAST_LAG_WINDOW_S, np.where(win_mask)[0].tolist(),
                np.round(analysis_times[win_mask], 4).tolist())
    df["y"] = y_all[:, win_mask].mean(axis=1).astype(np.float64)
    return df, "word_idx", "podcast"


DATASET_LOADERS = {
    "fedorenko2016": _load_fedorenko2016_pooled,
    "podcast": _load_podcast_pooled,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_LOADERS), required=True)
    ap.add_argument("--n_splits", type=int, default=15)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("split_half", "loso"), default="split_half")
    ap.add_argument("--n_repeats", type=int, default=1,
                    help="loso only: repeats over different train/held-out chunk splits (seed+rep)")
    ap.add_argument("--out_of_sample", dest="out_of_sample", action="store_true", default=True)
    ap.add_argument("--in_sample", dest="out_of_sample", action="store_false")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--lag", type=int, default=None,
                    help="podcast only: use this single lag index (0-31; time = -0.2 + lag/32 s) "
                         "as y instead of the 0-200 ms window mean; output gets a _lagNN tag")
    args = ap.parse_args()
    if args.lag is not None and args.dataset != "podcast":
        ap.error("--lag is only supported for --dataset podcast")

    loader = DATASET_LOADERS[args.dataset]
    base_df, item_col, dataset = loader(args.lag) if args.lag is not None else loader()
    if base_df.empty:
        raise ValueError(f"No pooled rows for {args.dataset!r}")
    logger.info(
        "[%s] %d rows, %d subjects, %d electrodes, %d items (item_col=%s)",
        args.dataset, len(base_df), base_df["subject"].nunique(),
        base_df["electrode"].nunique(), base_df[item_col].nunique(), item_col,
    )

    t0 = time.time()
    result = compute_ecog_ceiling_delta_loglik(
        base_df, dataset=dataset, item_col=item_col,
        n_splits=args.n_splits, seed=args.seed, n_workers=args.n_workers,
        out_of_sample=args.out_of_sample, mode=args.mode, n_repeats=args.n_repeats,
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
    mtag = "" if args.mode == "split_half" else f"_{args.mode}"
    if args.lag is not None:
        mtag += f"_lag{args.lag:02d}"
    out_path = out_dir / f"delta_loglik_ceiling_{args.dataset}_ecog_{suffix}{mtag}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "method": "timkey_et_al_2026_split_half_ecog_adapted",
        "out_of_sample": args.out_of_sample,
        "mode": args.mode,
        "n_repeats": args.n_repeats,
        "lag_index": args.lag,
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
