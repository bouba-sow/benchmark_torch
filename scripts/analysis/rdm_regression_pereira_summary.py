#!/usr/bin/env python3
"""Aggregate the Pereira per-unit RDM-regression results (rdm_regression_pereira_units.py).

  per-subject units : mean over the selected ROIs within (subject, passage set), then over a subject's sets,
                      then over subjects
  group units       : mean over the selected ROIs within each passage set, then over sets

Prints, per track and variant, the share of models with delta LL > 0, and the Spearman correlation across models
with the benchmark (surprisal: LME test dLL per obs; embedding: ridge Pearson r, both from
reports/rsa_summary/model_scores_*.csv). Writes <root>/summary_<atlas>.csv.

    .venv/bin/python scripts/analysis/rdm_regression_pereira_summary.py --root $SCRATCH/rdm_pereira_units [--atlas language]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
KEYS = ["track", "family", "model_id", "revision"]


def load(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype={"subject": str, "revision": str}) for p in (root).glob("*/*/*/rdm_units_*.csv")]
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame, atlas: list[str] | None) -> pd.DataFrame:
    if atlas:
        df = df[df.atlas.isin(atlas)]
    per_set = df.groupby(KEYS + ["subject", "passage_set"], as_index=False).delta_ll_per_obs.mean()
    group = per_set[per_set.subject == "GROUP"].groupby(KEYS, as_index=False).delta_ll_per_obs.mean().assign(variant="group")
    persub = per_set[per_set.subject != "GROUP"].groupby(KEYS + ["subject"], as_index=False).delta_ll_per_obs.mean()
    persub = persub.groupby(KEYS, as_index=False).delta_ll_per_obs.mean().assign(variant="per_subject")
    out = pd.concat([group, persub], ignore_index=True)
    n_units = df.groupby(KEYS + [df.subject.eq("GROUP").map({True: "group", False: "per_subject"}).rename("variant")]).size().rename("n_units").reset_index()
    return out.merge(n_units, on=KEYS + ["variant"], how="left")


def benchmark(track: str) -> pd.DataFrame:
    name = "model_scores_dll.csv" if track == "surprisal" else "model_scores_correlation.csv"
    sc = pd.read_csv(ROOT / "reports/rsa_summary/rsa" / name, dtype={"revision": str})
    return sc[(sc.dataset == "pereira") & (sc.track == track)][["family", "model_id", "revision", "benchmark_score"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("SCRATCH", ".")) / "rdm_pereira_units")
    ap.add_argument("--atlas", nargs="*", default=None, help="subset of language MD DMN visual auditory (default all ROIs)")
    args = ap.parse_args()
    df = load(args.root)
    summ = aggregate(df, args.atlas)
    tag = "_".join(args.atlas) if args.atlas else "all"
    summ.to_csv(args.root / f"summary_{tag}.csv", index=False)
    for (track, variant), g in summ.groupby(["track", "variant"]):
        m = g.merge(benchmark(track), on=["family", "model_id", "revision"])
        rho = spearmanr(m.delta_ll_per_obs, m.benchmark_score).statistic if len(m) > 3 else np.nan
        print(f"{tag:10s} {track:9s} {variant:12s} n={len(g)} frac_pos={float((g.delta_ll_per_obs > 0).mean()):.2f} "
              f"mean={g.delta_ll_per_obs.mean():.2e} median={g.delta_ll_per_obs.median():.2e} spearman_vs_benchmark={rho:+.2f}")


if __name__ == "__main__":
    main()
