#!/usr/bin/env python3
"""Plot held-out full-model likelihood against size for embedding RDMs.

Reads existing per-model RDM-regression CSVs; it does not recompute embeddings,
RDMs, or regression scores. Uses the no-confounds (intercept-only baseline)
variant and each model's last layer. Pythia checkpoints are excluded to match
the revised RDM scaling plots.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullLocator
import numpy as np
import pandas as pd
from scipy import stats

from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, lookup_model_meta
from neuroai.analysis.paper_style import use_paper_style
from scripts.figures.rdm_regression_figures import LAYOUT, panel_title

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT_DEFAULT = REPO_ROOT / "reports" / "rsa_summary" / "rdm_regression"


def load_last_layer(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/embedding/**/rdm_regression_*.csv"))
    if not paths:
        raise SystemExit(f"No embedding RDM-regression CSVs found under {root}")
    needed = {"dataset", "track", "family", "model_id", "revision", "subject", "measure",
              "layer", "confounds", "ll_full_per_obs"}
    frames = [pd.read_csv(p, dtype={"subject": str}, usecols=lambda col: col in needed) for p in paths]
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.confounds == "none") & (d.family != "pythia")].copy()
    d["layer"] = pd.to_numeric(d.layer, errors="coerce")
    d["ll_full_per_obs"] = pd.to_numeric(d.ll_full_per_obs, errors="coerce")
    d = d[np.isfinite(d.layer) & np.isfinite(d.ll_full_per_obs)]
    d["unit"] = d["measure"].fillna("all") if "measure" in d else "all"

    # Neural outputs have one row per subject. Match the scaling summaries by
    # averaging subjects at each layer, then selecting the model's final layer.
    keys = ["dataset", "unit", "family", "model_id", "revision", "layer"]
    d = d.groupby(keys, as_index=False).agg(ll_full_per_obs=("ll_full_per_obs", "mean"))
    id_keys = keys[:-1]
    last = d.loc[d.groupby(id_keys).layer.idxmax()].copy()

    metadata = [lookup_model_meta(mid, fam) for mid, fam in zip(last.model_id, last.family)]
    last["group"] = [m.group for m in metadata]
    last["n_params"] = [m.n_params for m in metadata]
    last = last[np.isfinite(last.n_params) & (last.n_params > 0)]
    return last


def draw(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    ncols, nrows = 4, len(LAYOUT)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.4, nrows * 3.4), squeeze=False)
    records = []
    for ri, row in enumerate(LAYOUT):
        for ci, panel in enumerate(row):
            ax = axes[ri][ci]
            if panel is None:
                ax.set_visible(False)
                continue
            ds, unit = panel
            sub = d[(d.dataset == ds) & (d.unit == unit)]
            ax.set_title(panel_title(ds, unit), fontsize=11, fontweight="bold", loc="left", pad=8)
            if sub.empty:
                ax.text(.5, .5, "no results", ha="center", va="center", transform=ax.transAxes, color="#777777")
                continue
            for group in GROUP_ORDER:
                s = sub[sub.group == group]
                if len(s):
                    ax.scatter(s.n_params, s.ll_full_per_obs, s=18, alpha=.7,
                               color=GROUP_COLORS[group], linewidths=0, rasterized=True)
            x = np.log10(sub.n_params.to_numpy(float))
            y = sub.ll_full_per_obs.to_numpy(float)
            slope = corr = pvalue = np.nan
            if len(np.unique(x)) >= 3:
                fit = stats.linregress(x, y)
                slope, corr, pvalue = fit.slope, fit.rvalue, fit.pvalue
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(10 ** xs, fit.intercept + fit.slope * xs, color="black", lw=1.4)
            lo, hi = np.percentile(y, [1, 99])
            pad = .3 * (hi - lo) if hi > lo else 1e-6
            ax.set_ylim(lo - pad, hi + pad)
            hidden = int(((y < lo - pad) | (y > hi + pad)).sum())
            note = f"slope/10×: {slope:+.2e}, r={corr:+.2f}, n={len(sub)}" if np.isfinite(slope) else f"n={len(sub)}"
            ax.text(.02, .98, note, transform=ax.transAxes, va="top", fontsize=8.5,
                    bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
            if hidden:
                ax.text(.02, .02, f"{hidden} point(s) outside view", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=7.5, color="#b00020")
            ax.set_xscale("log")
            ax.grid(alpha=.13)
            ax.set_xlabel("Model parameters (log scale)")
            ax.set_ylabel("Held-out Gaussian log likelihood per pair")
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
            records.append({"dataset": ds, "unit": unit, "n_models": len(sub),
                            "slope_per_10x_params": slope, "pearson_r_log10_params": corr,
                            "p_value": pvalue, "median_ll_full_per_obs": float(np.median(y)),
                            "min_ll_full_per_obs": float(np.min(y)), "max_ll_full_per_obs": float(np.max(y))})

    handles = [Line2D([], [], marker="o", linestyle="None", color=GROUP_COLORS[g], markersize=6,
                      label=GROUP_DISPLAY.get(g, g)) for g in GROUP_ORDER]
    handles.append(Line2D([], [], color="black", lw=1.4, label="linear fit, non-Pythia models"))
    fig.suptitle("Embedding RDM held-out likelihood by model size · no confounds · last layer", fontsize=18, y=.995)
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(.5, .975), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .025, 1, .955), h_pad=2.0, w_pad=2.2)
    fig.text(.5, .008, "Full-model held-out Gaussian log likelihood per pair (intercept + embedding RDM); response RDM is z-scored. "
             "Black line: OLS on log10(parameters), excluding Pythia checkpoints. Descriptive fit.",
             ha="center", fontsize=9, color="#555555")
    out.mkdir(parents=True, exist_ok=True)
    base = out / "rdm_embedding_likelihood_noconf"
    for suffix in ("pdf", "png"):
        fig.savefig(base.with_suffix(f".{suffix}"), dpi=180 if suffix == "png" else None, facecolor="white")
    plt.close(fig)
    summary = pd.DataFrame(records)
    summary.to_csv(out / "rdm_embedding_likelihood_noconf_summary.csv", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    scratch = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
    ap.add_argument("--regression", type=Path, default=scratch / "rdm_regression_r30")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    use_paper_style()
    data = load_last_layer(args.regression)
    summary = draw(data, args.out)
    print(f"wrote likelihood plots and summary to {args.out} ({len(data)} model-unit rows, {len(summary)} panels)")


if __name__ == "__main__":
    main()
