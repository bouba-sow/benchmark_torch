#!/usr/bin/env python3
"""Split-half theoretical-upper-bound delta log-likelihood ceiling for the
pooled MEG-MASC (Gwilliams et al. 2023) surprisal LME.

Method = Timkey et al. (2026) split-half, mirroring
scripts/ceilings/dll_n400.py — see
neuroai.regressors.megmasc_lme.compute_megmasc_ceiling_delta_loglik for the
details. Permute the 11 subjects into two halves; the "oracle" word-level
predictor for the target half is the per-word mean MEG amplitude measured on
the *source* half (independent subjects), merged in as ``surprisal``. Its
held-out delta_loglik (nuisance+oracle vs nuisance) is what a perfect
surprisal predictor would asymptote to. The train/held-out story split, the
``(1|subject) + (1+surprisal_z|region)`` structure and the nuisances all
match the sweep's ``fit_megmasc_lme_pooled_one_layer`` so the number is
directly comparable to each model's ``delta_loglik_test`` in
``outputs_jeanzay/megmasc/surprisal_sweep/<family>/<model>/megmasc_lme_by_layer.csv``.

No LLM / raw MEG needed: the pooled MEG table is model-independent, so this
reuses any one model's ``megmasc_lme_pooled_table.parquet`` that the sweep
already wrote.

Usage:
    .venv/bin/python scripts/ceilings/dll_megmasc.py \\
        --n_splits 15 --n_workers 12

Output: outputs_jeanzay/noise_ceilings/megmasc/delta_loglik_ceiling_megmasc.json
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

from neuroai.regressors.megmasc_lme import (  # noqa: E402
    DEFAULT_NUISANCES,
    PARITY_NUISANCES,
    compute_megmasc_ceiling_delta_loglik,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SWEEP_ROOT = REPO_ROOT / "outputs_jeanzay" / "megmasc/surprisal_sweep"
OUT_DIR = REPO_ROOT / "outputs_jeanzay" / "noise_ceilings" / "megmasc"


def _find_pooled_table(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"--pooled_table not found: {p}")
        return p
    cands = sorted(SWEEP_ROOT.glob("*/*/megmasc_lme_pooled_table.parquet"))
    if not cands:
        raise FileNotFoundError(
            f"No megmasc_lme_pooled_table.parquet under {SWEEP_ROOT} — run the "
            "pooled LME sweep first, or pass --pooled_table."
        )
    return cands[0]


def _sweep_delta_loglik_per_obs() -> pd.DataFrame:
    """Best-layer delta_loglik_test per model, converted to per-held-out-obs,
    for a quick 'fraction of ceiling' readout. Best-effort — returns empty on
    any hiccup."""
    rows = []
    for csv in SWEEP_ROOT.glob("*/*/megmasc_lme_by_layer.csv"):
        try:
            d = pd.read_csv(csv)
            d = d.loc[d["error"].isna() & d["delta_loglik_test"].notna()]
            if d.empty:
                continue
            best = d.loc[d["delta_loglik_test"].astype(float).idxmax()]
            n_test = float(best.get("n_obs_test", np.nan))
            if not np.isfinite(n_test) or n_test <= 0:
                continue
            rows.append({
                "family": csv.parent.parent.name,
                "model": csv.parent.name,
                "best_layer": int(best["layer"]),
                "delta_loglik_test": float(best["delta_loglik_test"]),
                "n_obs_test": int(n_test),
                "delta_loglik_test_per_obs": float(best["delta_loglik_test"]) / n_test,
            })
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pooled_table", default=None,
                    help="megmasc_lme_pooled_table.parquet (default: auto-find one under the sweep root)")
    ap.add_argument("--mode", choices=("split_half", "loso"), default="loso",
                    help="'loso' (default, leave-one-subject-out oracle — the usable ceiling for "
                         "this low-N/low-SNR dataset) or 'split_half' (Timkey et al. 2026; comes "
                         "out ~0 here)")
    ap.add_argument("--n_splits", type=int, default=15,
                    help="split_half only; ignored for loso (one fit per subject)")
    ap.add_argument("--train_frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=12)
    ap.add_argument("--nuisances", default=DEFAULT_NUISANCES,
                    help=f'default {DEFAULT_NUISANCES!r}; pass {PARITY_NUISANCES!r} for a Broderick-parity ceiling')
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--tag", default="megmasc", help="output filename tag")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pt = _find_pooled_table(args.pooled_table)
    logger.info("pooled MEG table: %s", pt)
    df = pd.read_parquet(pt)
    logger.info("table: %d rows, %d subjects, %d regions, %d stories",
                len(df), df["subject"].nunique(), df["region"].nunique(), df["chunk"].nunique())

    t0 = time.time()
    res = compute_megmasc_ceiling_delta_loglik(
        df,
        mode=args.mode,
        n_splits=args.n_splits,
        train_frac=args.train_frac,
        seed=args.seed,
        nuisances=args.nuisances,
        n_workers=args.n_workers,
        verbose=args.verbose,
    )
    dt = time.time() - t0
    logger.info("ceiling [%s]: delta_loglik_per_obs = %.6g  (sd %.3g, %d/%d iters, %.0fs)",
                res.mode, res.ceiling_delta_loglik_per_obs, res.ceiling_delta_loglik_per_obs_sd,
                res.n_iterations_used, res.n_iterations_requested, dt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _subj = " + (1 | subject)" if args.mode == "split_half" else ""
    _method = {
        "loso": "leave-one-subject-out; oracle = mean per-word amplitude of the other N-1 subjects",
        "split_half": "split-half subject permutation; oracle = source-half per-word mean amplitude (Timkey et al. 2026)",
    }[args.mode]
    tag = f"{args.tag}_{args.mode}"
    payload = {
        "dataset": "megmasc",
        "mode": res.mode,
        "metric": "delta_loglik_per_held_out_obs",
        "method": _method,
        "ceiling_delta_loglik_per_obs": res.ceiling_delta_loglik_per_obs,
        "ceiling_delta_loglik_per_obs_sd": res.ceiling_delta_loglik_per_obs_sd,
        "n_iterations_used": res.n_iterations_used,
        "n_iterations_requested": res.n_iterations_requested,
        "per_iteration_delta_loglik": res.per_iteration_delta_loglik,
        "per_iteration_n_obs": res.per_iteration_n_obs,
        "config": {
            "pooled_table": str(pt),
            "n_splits": args.n_splits,
            "train_frac": args.train_frac,
            "seed": args.seed,
            "nuisances": args.nuisances,
            "window_ms": [250, 450],
            "formulas": {
                "baseline": f"y ~ {args.nuisances}{_subj} + (1 | region)",
                "full": f"y ~ {args.nuisances} + surprisal_z{_subj} + (1 + surprisal_z | region)",
            },
        },
        "per_direction_records": res.per_direction_records,
    }
    out_json = out_dir / f"delta_loglik_ceiling_{tag}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", out_json)

    # Optional: fraction-of-ceiling readout against the finished sweep.
    ceil = res.ceiling_delta_loglik_per_obs
    if np.isfinite(ceil) and ceil > 0:
        sweep = _sweep_delta_loglik_per_obs()
        if not sweep.empty:
            sweep["frac_of_ceiling"] = sweep["delta_loglik_test_per_obs"] / ceil
            sweep = sweep.sort_values("frac_of_ceiling", ascending=False)
            csv_out = out_dir / f"delta_loglik_ceiling_{tag}__vs_sweep.csv"
            sweep.to_csv(csv_out, index=False)
            logger.info("wrote %s", csv_out)
            top = sweep.head(10)
            logger.info("top models by fraction of delta-LL ceiling:")
            for _, r in top.iterrows():
                logger.info("  %-40s L%-2d  dLL/obs=%.3g  = %.1f%% of ceiling",
                            r["model"], r["best_layer"], r["delta_loglik_test_per_obs"],
                            100 * r["frac_of_ceiling"])


if __name__ == "__main__":
    main()
