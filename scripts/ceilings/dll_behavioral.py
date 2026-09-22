#!/usr/bin/env python3
"""Split-half theoretical-upper-bound delta log-likelihood ceiling for
reading-time LME benchmarks (Timkey et al. 2026, PNAS supplement, Section
A/B) -- see neuroai.regressors.lme.compute_reading_time_ceiling_delta_loglik
for the method itself.

Covers all 7 behavioral reading-time datasets (naturalstories_spr, geco,
meco, provo, ucl_reading, zuco_reading, onestop). Loader configs (dataset
root, measure, filtering thresholds) mirror each dataset's real production
sweep config exactly (configs/surprisal/behavioral/{geco,meco,provo,onestop}_fp.yaml,
configs/surprisal/behavioral/ucl_reading.yaml, configs/surprisal/behavioral/zuco_reading_nr.yaml)
so the trials fed into the ceiling are preprocessed identically to the real
surprisal-based fits -- including OneStop's fixed 120-of-360 subject subset
(configs/subjects/onestop_subjects_120.txt), so its ceiling is a fair upper bound for
the actual reported (120-subject) scores, not a different population's.
GECO (14 subj) and ZuCo Reading (12 subj) will have much noisier split-half
estimates than the others given so few subjects to split into two halves --
treat their ceiling SD accordingly.

Usage:
    .venv/bin/python scripts/ceilings/dll_behavioral.py \\
        --dataset naturalstories_spr --n_splits 15

Output: outputs/noise_ceilings/delta_loglik_ceiling_<dataset>_<insample|oos>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.regressors.lme import compute_reading_time_ceiling_delta_loglik

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT_DIR = REPO_ROOT / "outputs" / "noise_ceilings"


def _load_naturalstories_trials() -> "pd.DataFrame":  # noqa: F821
    from neuroai.datasets.natural_stories import NaturalStoriesDataset
    root = REPO_ROOT / "data" / "naturalstories"
    ds = NaturalStoriesDataset(str(root), stories=None, min_correct=5, aggregate=False)
    return ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)


def _load_geco_trials(measure: str = "FP") -> "pd.DataFrame":  # noqa: F821
    from neuroai.datasets.geco import GecoDataset
    root = REPO_ROOT / "data" / "geco"
    ds = GecoDataset(
        str(root), measure, trials=None, min_dur=50.0, max_dur=5000.0,
        exclude_doc_edges=True, aggregate=False,
    )
    return ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)


def _load_meco_trials(measure: str = "FP") -> "pd.DataFrame":  # noqa: F821
    from neuroai.datasets.meco import MecoDataset
    root = (REPO_ROOT / "data" / "meco" / "osfstorage" / "release 2.0" / "version 2.1"
            / "wave 1" / "primary data" / "eye tracking data")
    ds = MecoDataset(
        str(root), measure, language="en", passages=None, min_dur=50.0, max_dur=5000.0,
        exclude_doc_edges=True, aggregate=False,
        rda_filename="joint_l1_data_trimmed_version2.1.rda",
    )
    return ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)


def _load_provo_trials(measure: str = "FP") -> "pd.DataFrame":  # noqa: F821
    from neuroai.datasets.provo import ProvoDataset
    root = REPO_ROOT / "data" / "Provo" / "osfstorage"
    ds = ProvoDataset(
        str(root), measure, texts=None, min_dur=50.0, max_dur=5000.0,
        exclude_doc_edges=True, aggregate=False,
    )
    return ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)


def _load_ucl_reading_trials() -> "pd.DataFrame":  # noqa: F821
    from neuroai.datasets.ucl_reading import UCLReadingDataset
    root = REPO_ROOT / "data" / "UCL_reading"
    ds = UCLReadingDataset(
        str(root), sentences=None, min_accuracy=0.0, min_rt=100.0, max_rt=3000.0,
        aggregate=False,
    )
    return ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)


def _load_zuco_reading_trials(task: str = "NR") -> "pd.DataFrame":  # noqa: F821
    from neuroai.datasets.zuco_reading import ZuCoReadingDataset
    root = REPO_ROOT / "data" / "q3zws" / "osfstorage"
    ds = ZuCoReadingDataset(
        str(root), task, subjects=None, min_fixations=1, min_trt=50.0, max_trt=5000.0,
        aggregate=False,
    )
    return ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)


def _onestop_subjects_120() -> list[str]:
    """Same fixed 120-of-360 subject subset the real production sweep uses
    (configs/subjects/onestop_subjects_120.txt) -- see
    neuroai.benchmarks.onestop_lme.OneStopLmeBenchmark._resolve_subjects for
    the original parsing this mirrors."""
    path = REPO_ROOT / "configs" / "configs/subjects/onestop_subjects_120.txt"
    subs = [
        ln.strip() for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not subs:
        raise ValueError(f"OneStop subjects_file is empty: {path}")
    return subs


def _load_onestop_trials(measure: str = "FP") -> "pd.DataFrame":  # noqa: F821
    from neuroai.datasets.onestop import OnestopDataset
    root = REPO_ROOT / "data" / "OneStop"
    ds = OnestopDataset(
        str(root), measure, stories=None, min_dur=50.0, max_dur=5000.0,
        exclude_doc_edges=True, aggregate=False, subjects=_onestop_subjects_120(),
    )
    return ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)


# Each entry: (loader, default measure/task, valid choices). Datasets with a
# single fixed measure (naturalstories_spr, ucl_reading) get ``None`` -- no
# --measure override is meaningful for them. GECO/MECO/Provo's "measure" is
# a duration measure (FP/GP/[SP]); ZuCo Reading's is a distinct READING TASK
# (NR/SR/TSR -- different stimuli entirely, not a duration measure of the
# same trials, see zuco_reading.py's module docstring) but is threaded
# through the same --measure flag here for a uniform CLI.
DATASET_LOADERS = {
    "naturalstories_spr": (_load_naturalstories_trials, None, None),
    "geco": (_load_geco_trials, "FP", ("FP", "GP")),
    "meco": (_load_meco_trials, "FP", ("FP", "GP", "SP")),
    "provo": (_load_provo_trials, "FP", ("FP", "GP", "SP")),
    "ucl_reading": (_load_ucl_reading_trials, None, None),
    "zuco_reading": (_load_zuco_reading_trials, "NR", ("NR", "SR", "TSR")),
    "onestop": (_load_onestop_trials, "FP", ("FP", "GP", "SP")),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_LOADERS), default="naturalstories_spr")
    ap.add_argument("--measure", default=None,
                     help="Duration measure for geco/meco/provo (FP/GP/SP) or reading TASK for "
                          "zuco_reading (NR/SR/TSR, different stimuli, not a duration measure -- "
                          "see zuco_reading.py). Defaults to each dataset's real production sweep "
                          "measure (FP / NR). Not applicable to naturalstories_spr/ucl_reading.")
    ap.add_argument("--n_splits", type=int, default=15)
    ap.add_argument("--n_workers", type=int, default=1,
                     help="Process-pool size for fitting (iteration, direction) tasks in "
                          "parallel (each is an independent lme4 fit -- process-based since "
                          "rpy2 embeds one R interpreter per process).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_of_sample", action="store_true",
                     help="Fit on a train partition, evaluate delta-loglik on a held-out "
                          "partition of the same target half (partition_mod/fit/test indices "
                          "matching the real sweeps' delta_loglik_test convention), instead of "
                          "the paper's literal in-sample fit-and-score-on-everything.")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    loader, default_measure, valid_measures = DATASET_LOADERS[args.dataset]
    measure = args.measure.upper() if args.measure else default_measure
    if measure is not None and valid_measures is not None and measure not in valid_measures:
        ap.error(f"--measure {measure!r} invalid for {args.dataset!r}; expected one of {valid_measures}")
    trials = loader(measure) if measure is not None else loader()
    label = f"{args.dataset}/{measure}" if measure is not None else args.dataset
    if trials.empty:
        raise ValueError(f"No trial-level RT observations after filtering for {label!r}")
    logger.info(
        "[%s] %d trials, %d subjects, %d unique (story_id, zone) items",
        label, len(trials), trials["participant_id"].nunique(),
        trials[["story_id", "zone"]].drop_duplicates().shape[0],
    )

    t0 = time.time()
    result = compute_reading_time_ceiling_delta_loglik(
        trials, n_splits=args.n_splits, seed=args.seed, n_workers=args.n_workers,
        out_of_sample=args.out_of_sample,
    )
    elapsed = time.time() - t0

    logger.info(
        "[%s] ceiling delta-LL/obs = %.4f (sd=%.4f, %d/%d iterations converged, %.1fs)",
        label, result.ceiling_delta_loglik_per_obs, result.ceiling_delta_loglik_per_obs_sd,
        result.n_iterations_used, result.n_iterations_requested, elapsed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "oos" if args.out_of_sample else "insample"
    # Keep the original filename (no measure tag) when using each dataset's
    # default measure, so the already-published FP/NR ceiling files this
    # replaces stay at the same path; only non-default measures (GP/SP,
    # SR/TSR) get a filename suffix.
    measure_tag = f"_{measure.lower()}" if (measure is not None and measure != default_measure) else ""
    out_path = out_dir / f"delta_loglik_ceiling_{args.dataset}{measure_tag}_{suffix}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "measure": measure,
        "method": "timkey_et_al_2026_split_half",
        "out_of_sample": args.out_of_sample,
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
