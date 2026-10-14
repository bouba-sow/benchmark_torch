#!/usr/bin/env python3
"""Podcast ECoG surprisal LME, one model per REGION on region-averaged electrodes (last layer).

For one model: final-layer logit-lens surprisal is read from the existing cache (no LLM is
loaded), then ``y ~ wlen_z + freq_z (+ surprisal_z) + (1 | subject)`` is fit per (region, lag)
for the lags in the 0-600 ms post-onset window (lags 7-25 of the 32-lag, 31.25 ms grid). See
``neuroai/regressors/podcast_region_lme.py`` for the model, regions and the train-only
standardisation.

Output: <out_dir>/<family>/<model>/podcast_region_lme_by_lag.csv, one row per (region, lag),
written incrementally and resumable (rows without an error are skipped on re-run).

    python scripts/run/surprisal/neural/podcast_region_fit.py --from-registry --family gpt2 \\
        --array-index 0 --out_dir outputs_jeanzay/podcast/surprisal_region_sweep --n_workers 20
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from neuroai.benchmarks.base import word_predictions_root
from neuroai.benchmarks.podcast_lme import PodcastLmeBenchmark
from neuroai.datasets.podcast import load_transcript
from neuroai.models.registry import registry_model_configs, resolve_family_task, resolve_load_model_id
from neuroai.regressors.podcast_region_lme import ROW_COLUMNS, fit_region_lme_one, lags_in_window, build_region_table
from neuroai.sweep import safe_model_dir, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Reuse the pooled sweep's config plumbing (config loading + benchmark config builder).
_spec = importlib.util.spec_from_file_location("podcast_fit_cfg", Path(__file__).with_name("podcast_fit.py"))
_pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pf)

# Model-independent pooled table/signal of the real sweep, copied once under the new root so the
# region sweep does not depend on the (backed-up, possibly moved) old sweep directory.
_BASE_SRC = REPO_ROOT / "outputs_jeanzay/podcast/surprisal_sweep/mamba2/AntonV_mamba2-370m-hf"
ROI_MAP = REPO_ROOT / "outputs/podcast/electrode_roi_map_language_responsive.csv"


def _ensure_base(out_root: Path) -> tuple[Path, Path]:
    base = out_root / "_base"
    base.mkdir(parents=True, exist_ok=True)
    out = []
    for name in ("podcast_lme_pooled_table.parquet", "podcast_lme_pooled_y.npy"):
        dst = base / name
        if not dst.exists():
            tmp = dst.with_suffix(dst.suffix + f".tmp{os.getpid()}")
            shutil.copy2(_BASE_SRC / name, tmp)
            os.replace(tmp, dst)  # atomic: concurrent array tasks never see a partial file
        out.append(dst)
    return out[0], out[1]


def _task(reg: pd.DataFrame, region: str, lag: int) -> dict:
    return fit_region_lme_one(reg, region=region, lag=lag, return_predictions=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/podcast.yaml")
    p.add_argument("--out_dir", default="outputs_jeanzay/podcast/surprisal_region_sweep")
    p.add_argument("--model_name", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--n_workers", type=int, default=20)
    p.add_argument("--window_s", type=float, nargs=2, default=(0.0, 0.6), metavar=("LO", "HI"))
    p.add_argument("--feature_cache_dir", default=None)
    args = p.parse_args()

    cfg = _pf.load_config(args.config)
    out_root = Path(args.out_dir)
    feature_cache_dir = Path(args.feature_cache_dir or os.path.expandvars(
        str(cfg.get("cache", {}).get("features_dir", ".cache/podcast_features"))))

    if args.from_registry:
        if not args.family or args.array_index is None:
            p.error("--from-registry requires --family and --array-index")
        task = resolve_family_task(args.family, args.array_index)
        model_cfgs = [m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id]
        model_id = resolve_load_model_id(model_cfgs[0], task.revision) if model_cfgs else task.model_id
        revision = task.revision
        n_params = model_cfgs[0]["n_params"] if model_cfgs else None
        trust = bool(model_cfgs[0].get("trust_remote_code", False)) if model_cfgs else None
        output_dir = out_root / args.family / safe_model_dir(task.model_id)
    elif args.model_name:
        model_id, revision, n_params, trust = args.model_name, args.revision, None, None
        output_dir = out_root / safe_model_dir(model_id)
    else:
        p.error("Specify --model_name, or --from-registry --family F --array-index I")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_cfg = _pf._build_benchmark_config(
        cfg, model_id=model_id, revision=revision, n_params=n_params, output_dir=output_dir,
        feature_cache_dir=feature_cache_dir, n_workers=1, lag_stride=1, trust_remote_code=trust,
        epoch_cache_dir=None, save_word_predictions=False, window_s=None,
    )

    t0 = time.time()
    words = load_transcript(bench_cfg.data.bids_root)["word"].astype(str).tolist()
    surprisal = PodcastLmeBenchmark(bench_cfg)._load_surprisal(words)       # final-layer logit lens, cache only
    surp = pd.Series(surprisal, index=np.arange(len(words)), name="surprisal")
    surp = surp[np.isfinite(surp)]
    logger.info("model=%s: %d/%d words with finite surprisal", model_id, len(surp), len(words))

    table_path, y_path = _ensure_base(out_root)
    lags = lags_in_window(*args.window_s)
    tab, regions = build_region_table(table_path, y_path, ROI_MAP, lags)
    tab = tab.merge(surp.rename_axis("word_idx").reset_index(), on="word_idx", how="inner")
    logger.info("lags %s (%.3f-%.3f s); regions %s; %d rows", lags[0], args.window_s[0], args.window_s[1], regions, len(tab))

    out_csv = output_dir / "podcast_region_lme_by_lag.csv"
    pred_dir = word_predictions_root(output_dir) / "region_predictions"   # $SCRATCH when set
    pred_dir.mkdir(parents=True, exist_ok=True)

    def _pred_path(region: str, lag: int) -> Path:
        return pred_dir / f"{region}_lag{lag:02d}.parquet"
    done: set[tuple[str, int]] = set()
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        ok = old[old["error"].isna()]
        done = {(r, k) for r, k in zip(ok["region"].astype(str), ok["lag"].astype(int)) if _pred_path(r, k).exists()}
        ok = ok[[(r, k) in done for r, k in zip(ok["region"].astype(str), ok["lag"].astype(int))]]   # refit fits lacking predictions
        if len(ok) != len(old):
            ok.reindex(columns=list(ROW_COLUMNS)).to_csv(out_csv, index=False)
        logger.info("resuming: %d (region, lag) fits already done", len(done))

    todo = [(r, k) for r in regions for k in lags if (r, k) not in done]
    logger.info("%d fits to run on %d workers", len(todo), args.n_workers)
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0

    def _append(row: dict) -> None:
        nonlocal write_header
        preds = row.pop("_predictions", None)
        if preds is not None and row.get("error") is None:
            tmp = _pred_path(row["region"], row["lag"]).with_suffix(f".tmp{os.getpid()}")
            preds.to_parquet(tmp, index=False)
            os.replace(tmp, _pred_path(row["region"], row["lag"]))
        pd.DataFrame([row]).reindex(columns=list(ROW_COLUMNS)).to_csv(out_csv, mode="a", header=write_header, index=False)
        write_header = False

    regs = {r: tab[tab["region"] == r].reset_index(drop=True) for r in regions}
    # spawn, not fork: this parent has run polars and loaded R/rpy2, and forked children of such a
    # parent hang (seen on the ceiling script: workers idle at <1% CPU).
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(_task, regs[r], r, k) for r, k in todo]
        for i, f in enumerate(as_completed(futs), 1):
            _append(f.result())
            if i % 10 == 0 or i == len(futs):
                logger.info("%d/%d fits done (%.0fs)", i, len(futs), time.time() - t0)
    logger.info("Done: %s", out_csv)


if __name__ == "__main__":
    main()
