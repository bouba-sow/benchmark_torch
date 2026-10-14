#!/usr/bin/env python3
"""Summarize the pair-count sensitivity runs (pair_cap_sensitivity_provo.py; rdm_regression_{epoch,podcast_roi}_units.py with
RDM_MAX_PAIRS / RDM_PAIR_SEED_TAG, --group-only, the models of models.txt).

For every dataset and number of pairs, over the units of the dataset and the models of the subset:
  rho_seeds : Spearman correlation ACROSS MODELS of the unit scores of two independent pair samples (median over units and pairs of samples)
  rho_ref   : Spearman across models between the score at this cap (mean of the pair samples) and the reference setting
              (Provo: all 2.86M pairs; Broderick / Podcast: 1,000,000 pairs, the largest run)
  rho_comp  : same, for the dataset composite used in the figures (per-model MEDIAN over the units)
  dev_sd    : median over (unit, model) cells of |score(cap) - score(ref)| in units of the across-model SD of the reference score
  sign      : share of (unit, model) cells whose sign of the score agrees with the reference
for the held-out dLL (delta_ll_per_obs) and the RSA correlation (rsa_pearson).

    .venv/bin/python scripts/analysis/pair_cap_sensitivity_summary.py $SCRATCH/pair_cap_sensitivity
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRICS = ("delta_ll_per_obs", "rsa_pearson")


def load_neural(root: Path, ds: str) -> pd.DataFrame:
    frames = []
    for run in sorted((root / "runs").glob(f"{ds}_cap*_seed*")):
        cap, seed = run.name.split("_cap")[1].split("_seed")
        for f in (run / "surprisal").glob("units*.csv"):
            d = pd.read_csv(f, dtype={"revision": str, "subject": str})
            d["cap"], d["pair_seed"] = int(cap), seed
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.subject.str.startswith("GROUP")]
    d["unit_key"] = d.unit.astype(str) if "unit" in d else d.roi.astype(str) + "|" + d.lag.astype(str)
    if "roi" in d and "unit" in d:
        d["unit_key"] = d.roi.astype(str) + "|" + d.lag.astype(str)
    return d[["cap", "pair_seed", "model_id", "unit_key", *METRICS]]


def load_provo(root: Path, name: str = "provo_FP.csv") -> pd.DataFrame:
    d = pd.read_csv(root / name, dtype={"revision": str})
    d["pair_seed"] = d.pair_seed.astype(str)
    d["unit_key"] = "FP"
    d["cap"] = d.cap.astype(str)
    return d[["cap", "pair_seed", "model_id", "unit_key", *METRICS]]


def wide(d: pd.DataFrame, cap, seed, metric) -> pd.DataFrame:
    s = d[(d.cap == cap) & (d.pair_seed == seed)]
    return s.pivot_table(index="model_id", columns="unit_key", values=metric)


def median_rho(a: pd.DataFrame, b: pd.DataFrame) -> float:
    cols = [c for c in a.columns if c in b.columns]
    r = []
    for c in cols:
        x, y = a[c].align(b[c], join="inner")
        ok = x.notna() & y.notna()
        if ok.sum() >= 5 and x[ok].std() > 0 and y[ok].std() > 0:
            r.append(spearmanr(x[ok], y[ok]).statistic)
    return float(np.median(r)) if r else np.nan


def summarize(d: pd.DataFrame, caps: list, ref_cap, ref_seeds: list[str], label: str) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        ref = pd.concat([wide(d, ref_cap, s, metric) for s in ref_seeds]).groupby(level=0).mean()
        for cap in caps:
            seeds = sorted(d[d.cap == cap].pair_seed.unique())
            ws = {s: wide(d, cap, s, metric) for s in seeds}
            rho_seed = np.nanmedian([median_rho(ws[a], ws[b]) for a, b in combinations(seeds, 2)]) if len(seeds) > 1 else np.nan
            mean = pd.concat(ws.values()).groupby(level=0).mean()
            cols = [c for c in mean.columns if c in ref.columns]
            m, r = mean[cols].align(ref[cols], join="inner")
            rho_ref = median_rho(m, r)
            comp_m, comp_r = m.median(axis=1), r.median(axis=1)
            rho_comp = spearmanr(comp_m, comp_r).statistic if len(comp_m) > 4 else np.nan
            sd = r.std(axis=0).replace(0, np.nan)
            dev = ((m - r).abs() / sd).to_numpy().ravel()
            same = (np.sign(m.to_numpy()) == np.sign(r.to_numpy())).ravel()
            rows.append(dict(dataset=label, metric=metric, cap=cap, rho_seeds=rho_seed, rho_ref=rho_ref, rho_comp=rho_comp,
                             dev_sd=float(np.nanmedian(dev)), sign=float(np.mean(same)), n_units=m.shape[1], n_models=m.shape[0]))
    return pd.DataFrame(rows)


def main():
    root = Path(sys.argv[1])
    out = []
    for ds in ("broderick", "podcast"):
        try:
            d = load_neural(root, ds)
        except Exception as e:  # noqa: BLE001
            print(f"{ds}: not available ({e})")
            continue
        caps = sorted(d.cap.unique())
        out.append(summarize(d, caps, caps[-1], sorted(d[d.cap == caps[-1]].pair_seed.unique()), ds))
    if (root / "provo_FP.csv").exists():
        d = load_provo(root)
        caps = sorted((c for c in d.cap.unique() if c != "all"), key=int)
        out.append(summarize(d, caps, "all", ["0"], "provo_FP"))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(root / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    for metric in METRICS:
        print(f"\n== {metric}")
        print(res[res.metric == metric].drop(columns="metric").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
