#!/usr/bin/env python3
"""Split-half delta-LL ceiling of the behavioral reading-time LMEs, BROKEN DOWN BY DLT COST BIN.

Same procedure as ``scripts/ceilings/dll_behavioral.py --out_of_sample`` (same trial loaders, split seed, item-mean
oracle surprisal, ``fit_spr_lme`` settings) -- see ``neuroai.regressors.lme.compute_reading_time_ceiling_delta_loglik``
-- but each direction's held-out per-observation gain (full_loglik - baseline_loglik, ``save_per_word``) is kept and
summed per DLT bin (outputs/dlt/<dataset>_dlt_cost.csv joined on story_id = sentid, zone = sentpos):
easy = cost 0, hard = cost >= 1..5, ``all``. The ``all`` bin must reproduce the published ceiling (checked in the log).

    .venv/bin/python scripts/ceilings/dll_behavioral_bins.py --entry geco_gp --n_workers 30
Output: outputs_jeanzay/noise_ceilings/behavioral_bins/delta_loglik_ceiling_<entry>_bins.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.regressors.lme import _ceiling_direction_task, fit_spr_lme
from scripts.ceilings.dll_behavioral import DATASET_LOADERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
BINS = ["all", "easy", "hard1", "hard2", "hard3", "hard4", "hard5"]
# entry -> (dataset, measure, dlt cost csv, published ceiling file stem)
ENTRIES = {
    "geco": ("geco", "FP", "geco_dlt_cost.csv", "geco"), "geco_gp": ("geco", "GP", "geco_dlt_cost.csv", "geco_gp"),
    "meco": ("meco", "FP", "meco_dlt_cost.csv", "meco"), "meco_gp": ("meco", "GP", "meco_dlt_cost.csv", "meco_gp"),
    "meco_sp": ("meco", "SP", "meco_dlt_cost.csv", "meco_sp"),
    "provo": ("provo", "FP", "provo_dlt_cost.csv", "provo"), "provo_gp": ("provo", "GP", "provo_dlt_cost.csv", "provo_gp"),
    "provo_sp": ("provo", "SP", "provo_dlt_cost.csv", "provo_sp"),
    "ucl_reading": ("ucl_reading", None, "ucl_reading_dlt_cost.csv", "ucl_reading"),
    "zuco_reading": ("zuco_reading", "NR", "zuco_reading_nr_dlt_cost.csv", "zuco_reading"),
    "zuco_reading_sr": ("zuco_reading", "SR", "zuco_reading_sr_dlt_cost.csv", "zuco_reading_sr"),
    "zuco_reading_tsr": ("zuco_reading", "TSR", "zuco_reading_tsr_dlt_cost.csv", "zuco_reading_tsr"),
}


def _worker(it: int, direction: int, tgt: pd.DataFrame, cost: pd.DataFrame):
    try:
        r = fit_spr_lme(tgt, predictor_col="surprisal", style="clark2025", compare_baseline=True, out_of_sample=True,
                        partition_mod=4, partition_fit_indices=(0, 1), partition_test_indices=(2, 3),
                        include_freq=True, include_prev_surprisal=False, filter_sentence_edges=True,
                        verbose=False, save_per_word=True)
        if not r.converged or r.per_word is None or r.n_obs_test is None or r.n_obs_test < 10:
            return it, direction, None
        pw = r.per_word.assign(story_id=lambda d: d["sentid"].astype(int), zone=lambda d: d["sentpos"].astype(int))
        pw = pw.merge(cost, on=["story_id", "zone"], how="left")
        d, c = pw["delta_loglik"].to_numpy(float), pw["total"].to_numpy(float)
        masks = {"all": np.isfinite(d), "easy": c <= 0} | {f"hard{k}": c >= k for k in range(1, 6)}
        out = {}
        for b, m in masks.items():
            ok = m & np.isfinite(d)
            out[b] = (float(d[ok].sum()), int(ok.sum()),
                      int(pw.loc[ok, ["story_id", "zone"]].drop_duplicates().shape[0]))
        return it, direction, out
    except Exception as exc:  # noqa: BLE001
        logger.warning("fit failed split=%d dir=%d: %s", it, direction, exc)
        return it, direction, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", required=True, choices=sorted(ENTRIES))
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "outputs_jeanzay/noise_ceilings/behavioral_bins"))
    a = ap.parse_args()
    dataset, measure, cost_csv, stem = ENTRIES[a.entry]
    loader = DATASET_LOADERS[dataset][0]
    trials = loader(measure) if measure is not None else loader()
    cost = (pd.read_csv(REPO_ROOT / "outputs/dlt" / cost_csv)[["story_id", "zone", "total"]]
            .drop_duplicates(["story_id", "zone"]).astype({"story_id": int, "zone": int}))
    t0 = time.time()
    rng = np.random.default_rng(a.seed)                      # same stream as compute_reading_time_ceiling_delta_loglik
    subjects = np.asarray(sorted(trials["participant_id"].astype(str).unique()))
    tasks = []
    for it in range(a.n_splits):
        perm = rng.permutation(subjects)
        half = len(perm) // 2
        halves = (set(perm[:half].tolist()), set(perm[half:].tolist()))
        for direction, (src, tgt_ids) in enumerate((halves, halves[::-1])):
            tgt = _ceiling_direction_task(trials, src, tgt_ids, ["story_id", "zone"])
            if tgt is not None:
                tasks.append((it, direction, tgt))
    logger.info("[%s] %d trials, %d subjects, %d tasks", a.entry, len(trials), len(subjects), len(tasks))
    res = []
    if a.n_workers > 1:
        with ProcessPoolExecutor(max_workers=a.n_workers) as pool:
            for f in as_completed([pool.submit(_worker, it, d, t, cost) for it, d, t in tasks]):
                res.append(f.result())
    else:
        res = [_worker(it, d, t, cost) for it, d, t in tasks]

    summary = {}
    for b in BINS:
        per_obs, ns, uw = [], [], []
        for it in range(a.n_splits):
            rs = [x for x in res if x[0] == it]
            if len(rs) == 2 and all(x[2] is not None for x in rs):
                s, n = sum(x[2][b][0] for x in rs), sum(x[2][b][1] for x in rs)
                if n > 0:
                    per_obs.append(s / n); ns.append(n); uw.append(max(x[2][b][2] for x in rs))
        summary[b] = {"ceiling_delta_loglik_per_obs": float(np.mean(per_obs)) if per_obs else float("nan"),
                      "ceiling_delta_loglik_per_obs_sd": float(np.std(per_obs, ddof=1)) if len(per_obs) > 1 else float("nan"),
                      "n_iterations_used": len(per_obs), "n_obs_mean": float(np.mean(ns)) if ns else 0.0,
                      "n_words_unique_max": int(max(uw)) if uw else 0}
    pub = json.load(open(REPO_ROOT / f"outputs/noise_ceilings/delta_loglik_ceiling_{stem}_oos.json"))["ceiling_delta_loglik_per_obs"]
    logger.info("[%s] all-bin ceiling %.6f vs published %.6f", a.entry, summary["all"]["ceiling_delta_loglik_per_obs"], pub)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"delta_loglik_ceiling_{a.entry}_bins.json").write_text(json.dumps({
        "entry": a.entry, "dataset": dataset, "measure": measure, "seed": a.seed, "n_splits": a.n_splits,
        "published_all_words_ceiling": pub, "per_bin": summary, "wall_time_s": time.time() - t0}, indent=2))
    logger.info("[%s] done in %.0fs", a.entry, time.time() - t0)


if __name__ == "__main__":
    main()
