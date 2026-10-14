#!/usr/bin/env python3
"""Summarize the per-unit neural RDM-regression runs (docs/rdm_regression_neural_units.md).

Headline aggregation per dataset:
  subject-averaged ("GROUP") units : Pereira, Tuckute, Broderick (channels), MEG-MASC (channels), Nieuwland,
                                     Federmeier 2007, Hubbard 2019
  per-subject units                : Fedorenko, Podcast (electrodes are subject specific)
                                     (mean over units within a subject, then over subjects; GROUP units excluded)
The per-subject aggregation of every dataset that has per-subject units is also written, to
neural_rdm_summary_per_subject.csv (Broderick / MEG-MASC split by unit kind, channel and region).
Pereira: mean over ROIs per passage set, then over the two sets. Optional filters pick ROI subsets (--atlas) or Podcast
lags (--lags).

Per dataset and track it prints the share of models with delta LL > 0 and the Spearman correlation across models with the
benchmark (surprisal: LME test dLL per obs; embedding: ridge Pearson r; reports/rsa_summary/model_scores_*.csv) and writes
<out>/neural_rdm_summary.csv (one row per dataset/track/model).

    .venv/bin/python scripts/analysis/rdm_regression_neural_summary.py [--scratch $SCRATCH] [--atlas language MD]
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
HEADLINE = {"pereira": "group", "tuckute": "group", "nieuwland": "group", "szewczyk_f2007": "group",
            "szewczyk_h2019": "group", "broderick": "group", "megmasc": "group", "fedorenko2016": "subject",
            "podcast": "group", "podcast_electrode": "subject"}
# Broderick / MEG-MASC: subject-averaged channel units (regions are available with --kind region).
# Fedorenko / Podcast electrodes are subject specific, so there is no subject average.
DEFAULT_KIND = {"broderick": "channel", "megmasc": "channel"}


def read(paths):
    frames = [pd.read_csv(p, dtype={"subject": str, "revision": str}) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load(ds: str, S: Path, atlas, kind, lags) -> pd.DataFrame:
    if ds == "pereira":
        d = read((S / "rdm_pereira_units").glob("*/*/*/rdm_units_*.csv"))
        if atlas:
            d = d[d.atlas.isin(atlas)]
        d["cell"] = d.subject + "|" + d.passage_set
    elif ds == "tuckute":
        d = read((S / "rdm_tuckute_units").glob("*/units_*.csv"))
        if atlas:
            d = d[d.atlas.isin([a.lower() for a in atlas])]
        d["cell"] = d.subject
    elif ds in ("broderick", "megmasc"):
        d = read((S / f"rdm_{ds}_units").glob("*/units_*.csv"))
        if kind:
            d = d[d.kind == kind]
        d["cell"] = d.subject
    elif ds == "fedorenko2016":
        d = read((S / "rdm_fedorenko_units").glob("*/units.csv"))
        d["cell"] = d.subject
    elif ds == "podcast":  # ROI units (5 Fedorenko language parcels + netw), subject-averaged GROUP units are the headline
        d = read((S / "rdm_podcast_roi_units").glob("*/units_roi_c*.csv"))
        if len(d):  # default: the five parcels; --atlas netw (or any parcel names) selects other subsets
            d = d[d.roi.isin(atlas)] if atlas else d[d.roi != "netw"]
        if lags and len(d):
            d = d[(d.lag >= lags[0]) & (d.lag <= lags[1])]
        d["cell"] = d.subject + "|" + d.roi
    elif ds == "podcast_electrode":  # electrode-level units, per subject (kept for reference)
        d = read((S / "rdm_podcast_units").glob("*/units_*.csv"))
        if lags and len(d):
            k = d.window.str.extract(r"lags(\d+)-")[0].astype(int)
            d = d[(k >= lags[0]) & (k <= lags[1])]
        d["cell"] = d.subject
    else:
        d = read((S / "rdm_n400_units" / ds).glob("*/units.csv"))
        d["cell"] = d.subject
    return d


def aggregate(d: pd.DataFrame, mode: str) -> pd.DataFrame:
    if d.empty:
        return d
    is_group = d.subject.str.startswith("GROUP")
    d = d[is_group] if mode == "group" else d[~is_group]
    cols = ["delta_ll_per_obs", "rsa_pearson"]
    cell = d.groupby(KEYS + ["subject", "cell"], as_index=False)[cols].mean()
    sub = cell.groupby(KEYS + ["subject"], as_index=False)[cols].mean()
    out = sub.groupby(KEYS, as_index=False).agg(delta_ll_per_obs=("delta_ll_per_obs", "mean"), rsa_pearson=("rsa_pearson", "mean"),
                                                n_subjects=("subject", "nunique"))
    n_units = d.groupby(KEYS).size().rename("n_units").reset_index()
    return out.merge(n_units, on=KEYS)


def bench(ds: str, track: str) -> pd.DataFrame:
    f = "model_scores_dll.csv" if track == "surprisal" else "model_scores_correlation.csv"
    sc = pd.read_csv(ROOT / "reports/rsa_summary/rsa" / f, dtype={"revision": str})
    return sc[(sc.dataset == ds) & (sc.track == track)][["family", "model_id", "revision", "benchmark_score"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug")))
    ap.add_argument("--datasets", nargs="*", default=list(HEADLINE))
    ap.add_argument("--atlas", nargs="*", default=None, help="Pereira: language MD DMN visual auditory; Tuckute: lang md dmn; Podcast: ROI names (IFG IFGorb MFG AntTemp PostTemp netw)")
    ap.add_argument("--kind", choices=["channel", "region"], default=None, help="Broderick / MEG-MASC unit kind")
    ap.add_argument("--lags", nargs=2, type=int, default=None, help="Podcast lag index range, e.g. 8 31")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.scratch / "rdm_neural_summary"
    out.mkdir(parents=True, exist_ok=True)
    allrows, persub = [], []
    for ds in a.datasets:
        kind = a.kind or DEFAULT_KIND.get(ds)
        d = load(ds, a.scratch, a.atlas, kind, a.lags)
        if d.empty:
            print(f"{ds}: no results yet")
            continue
        s = aggregate(d, HEADLINE[ds]).assign(dataset=ds, headline=HEADLINE[ds])
        if ds != "podcast_electrode":
            allrows.append(s)
        if ds not in ("nieuwland", "szewczyk_f2007", "szewczyk_h2019"):
            for k in (["channel", "region"] if ds in DEFAULT_KIND else [None]):
                dk = load(ds, a.scratch, a.atlas, k, a.lags)
                ps = aggregate(dk, "subject")
                if not ps.empty:
                    persub.append(ps.assign(dataset=ds, unit_kind=k or "all"))
        for track, g in s.groupby("track"):
            m = g.merge(bench("podcast" if ds == "podcast_electrode" else ds, track), on=["family", "model_id", "revision"])
            rho = spearmanr(m.delta_ll_per_obs, m.benchmark_score).statistic if len(m) > 3 else np.nan
            print(f"{ds:15s} {track:9s} {HEADLINE[ds]:7s} n={len(g):3d} frac_pos={(g.delta_ll_per_obs > 0).mean():.2f} "
                  f"median={g.delta_ll_per_obs.median():+.2e} spearman_vs_benchmark={rho:+.2f} units/model={int(g.n_units.median())}")
    if allrows:
        pd.concat(allrows).to_csv(out / "neural_rdm_summary.csv", index=False)
    if persub:
        pd.concat(persub).to_csv(out / "neural_rdm_summary_per_subject.csv", index=False)


if __name__ == "__main__":
    main()
