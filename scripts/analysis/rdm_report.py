#!/usr/bin/env python3
"""Join the RDM-regression results into one last-layer table and a per-dataset ceiling table.

Inputs (all read-only, produced by the SLURM chains documented in docs/rdm_regression.md):
  <regression>/<dataset>/<track>/model_summary.csv   held-out delta LL, all layers scored (30 partitions)
  <bootstrap>/<dataset>/bootstrap_summary.csv         last-layer stimulus bootstrap (point, SE, 95% interval)
  <ceiling>/<dataset>/ceiling_summary.csv             split-half ceiling + half-target model scores
                                                      (behavioral measures and the *_avg N400 datasets)
  <ceiling>/<dataset>/loso_ceiling_summary.csv        leave-one-subject-out ceiling (per-subject neural datasets)
Output: <out>/rdm_last_layer_results.csv (one row per dataset, unit, track, model, confound variant),
        <out>/rdm_ceilings.csv (one row per dataset, unit, confound variant) and a printed overview.

Units: the reading measure (FP/GP/SP/SPR/NR/SR/TSR) for behavioral datasets, 'all' for the neural datasets
(subject-averaged delta LL; the *_avg datasets have one grand-average RDM).
Ceiling ratios: split-half datasets use the half-target model score / split-half ceiling (like-for-like,
Timkey et al. 2026); per-subject neural datasets use last-layer delta LL / LOSO ceiling (same target).
The raw delta LL and the ceiling are always reported next to the ratio; the ratio is unstable when the
ceiling is ~0 (see docs/rdm_regression.md).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["dataset", "track", "family", "model_id", "revision", "unit", "confounds"]
BEHAVIORAL = {"provo", "meco", "geco", "onestop", "naturalstories", "ucl_reading", "zuco_reading"}


def _read_all(pattern_root: Path, pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype={"subject": str}) for p in sorted(pattern_root.glob(pattern))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    scratch = Path(os.environ["SCRATCH"])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regression", type=Path, default=scratch / "rdm_regression_r30")
    ap.add_argument("--bootstrap", type=Path, default=scratch / "rdm_bootstrap_r200")
    ap.add_argument("--ceiling", type=Path, default=scratch / "rdm_ceiling")
    ap.add_argument("--out", type=Path, default=scratch / "rdm_report")
    a = ap.parse_args()

    main_df = _read_all(a.regression, "*/*/model_summary.csv")
    if main_df.empty:
        raise SystemExit(f"no model_summary.csv under {a.regression}")
    main_df["unit"] = np.where(main_df.dataset.isin(BEHAVIORAL), main_df.get("measure", "all"), "all")
    main_df = main_df.rename(columns={"last_layer_delta_ll_per_obs": "last_layer_delta_ll_per_obs"})
    keep = KEYS + ["last_layer_delta_ll_per_obs", "group", "n_params"]
    table = main_df[keep].copy()

    boot = _read_all(a.bootstrap, "*/bootstrap_summary.csv")
    if not boot.empty:
        boot["unit"] = np.where(boot.dataset.isin(BEHAVIORAL), boot.unit, "all")
        table = table.merge(boot[KEYS + ["point_delta_ll_per_obs", "boot_se", "boot_sd_uncorrected", "ci_low", "ci_high"]],
                            on=KEYS, how="left")
        table["ci_excludes_zero"] = np.where(table.ci_low.isna(), np.nan,
                                             ((table.ci_low > 0) | (table.ci_high < 0)).astype(float))
    else:
        print("bootstrap: no summaries found yet")

    ceil = _read_all(a.ceiling, "*/ceiling_summary.csv")
    table["ceiling_delta_ll_per_obs"] = np.nan
    table["half_target_delta_ll_per_obs"] = np.nan
    table["ratio_to_ceiling"] = np.nan
    table["ceiling_type"] = ""
    if not ceil.empty:
        ceil["unit"] = np.where(ceil.dataset.isin(BEHAVIORAL), ceil.unit, "all")
        m = table.merge(ceil[KEYS + ["half_target_delta_ll_per_obs", "ceiling_delta_ll_per_obs", "ratio_to_ceiling"]],
                        on=KEYS, how="left", suffixes=("", "_c"))
        for col in ("half_target_delta_ll_per_obs", "ceiling_delta_ll_per_obs", "ratio_to_ceiling"):
            table[col] = m[col + "_c"].to_numpy()
        table.loc[table.ceiling_delta_ll_per_obs.notna(), "ceiling_type"] = "split_half_half_target"
    loso = _read_all(a.ceiling, "*/loso_ceiling_summary.csv")
    if not loso.empty:
        by = (loso.groupby(["dataset", "confounds"], as_index=False)
              .agg(loso=("ceiling_delta_ll_per_obs", "mean"), upper=("upper_delta_ll_per_obs", "mean"),
                   n_subjects=("subject", "nunique"),
                   valid_pair_fraction=("valid_pair_fraction", "mean")))
        m = table.merge(by, on=["dataset", "confounds"], how="left")
        use = m.loso.notna().to_numpy()
        table.loc[use, "ceiling_delta_ll_per_obs"] = m.loso.to_numpy()[use]
        table.loc[use, "ratio_to_ceiling"] = (m.last_layer_delta_ll_per_obs / m.loso).to_numpy()[use]
        table.loc[use, "ceiling_type"] = "loso_same_target"
        table["ceiling_upper_delta_ll_per_obs"] = np.nan
        table.loc[use, "ceiling_upper_delta_ll_per_obs"] = m.upper.to_numpy()[use]
        table["ratio_to_upper"] = np.nan
        table.loc[use, "ratio_to_upper"] = (m.last_layer_delta_ll_per_obs / m.upper).to_numpy()[use]
    if ceil.empty and loso.empty:
        print("ceilings: none found yet")

    a.out.mkdir(parents=True, exist_ok=True)
    table.sort_values(KEYS).to_csv(a.out / "rdm_last_layer_results.csv", index=False)

    rows = []
    for (ds, unit, conf), g in table.groupby(["dataset", "unit", "confounds"]):
        row = {"dataset": ds, "unit": unit, "confounds": conf,
               "ceiling_delta_ll_per_obs": g.ceiling_delta_ll_per_obs.dropna().mean() if g.ceiling_delta_ll_per_obs.notna().any() else np.nan,
               "ceiling_type": next((t for t in g.ceiling_type if t), ""), "n_model_rows": len(g)}
        for track in ("embedding", "surprisal"):
            t = g[g.track == track]
            if len(t):
                best = t.loc[t.last_layer_delta_ll_per_obs.idxmax()]
                row[f"best_{track}_last_layer_delta_ll"] = best.last_layer_delta_ll_per_obs
                row[f"best_{track}_model"] = best.model_id
                row[f"best_{track}_ratio"] = best.ratio_to_ceiling
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(a.out / "rdm_ceilings.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summary.drop(columns=[c for c in summary if c.endswith("_model")]).round(5).to_string(index=False))
    print(f"\nwrote {a.out / 'rdm_last_layer_results.csv'} ({len(table)} rows) and {a.out / 'rdm_ceilings.csv'}")


if __name__ == "__main__":
    main()
