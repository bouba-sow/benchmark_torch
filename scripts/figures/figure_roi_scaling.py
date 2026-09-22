#!/usr/bin/env python3
"""Paper figure: ROI scaling laws, LAST layer -- same visual convention as
figures/figure3_family_scaling.py (family_scaling_lastlayer.pdf: scatter
of score vs model size, log-x, one per-group log-linear fit line), but with
one panel per Fedorenko ROI (IFGorb, IFG, MFG, AntTemp, PostTemp) instead of
one panel per dataset family.

Two pages:

  Page 1 (embedding, last layer): one row per dataset -- LPP fMRI, Natural
    Stories fMRI, Podcast iEEG -- x5 ROI columns.
  Page 2 (surprisal, last layer): Natural Stories fMRI only. LPP's LME fit
    (lpp_lme_by_layer.csv) and Podcast's (podcast_lme_by_lag.csv) pool every
    region into ONE model per layer/lag with region as a random effect --
    there is no per-ROI held-out score to plot for them (same limitation
    documented in figures/figure5_roi_comparison.py).

Reuses the exact ROI/layer loaders from figures/figure5_roi_comparison.py
(load_lpp/load_ns_embed/load_ns_surprisal/load_podcast -- same sweeps, same
layer_frac convention), but selects each model's LAST layer instead of its
best layer, and joins each model's n_params from the registry.

Score: same raw Pearson r (embedding) / held-out delta-Pearson-r (surprisal)
convention as figure 5. There is no per-ROI noise-ceiling estimate yet, so
(unlike family_scaling_lastlayer.pdf's ceiling-normalised "Fraction of noise
ceiling") this is NOT ceiling-normalised -- compare slopes/patterns across
ROIs and classes, not absolute magnitude against the other paper figures.

Output: reports/paper/roi_scaling_lastlayer.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from neuroai.analysis.benchmark_common import GROUP_DISPLAY, GROUP_ORDER, REGISTRY_INDEX, REPO_ROOT
from neuroai.analysis.paper_style import use_paper_style
from scripts.figures.figure3_family_scaling import GROUP_COLOR
from scripts.figures.figure5_roi_comparison import (
    ROI_ORDER, load_lpp, load_ns_embed, load_ns_surprisal, load_podcast,
)

OUT_DIR = REPO_ROOT / "reports" / "paper"


def _last_layer(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model_id, roi): the row at that model's highest layer
    index (matches family_scaling_lastlayer.pdf's score_last_norm, i.e. the
    final layer rather than the best one)."""
    if df.empty:
        return df
    idx = df.groupby(["model_id", "roi"])["layer"].idxmax()
    out = df.loc[idx].copy()
    out["n_params"] = out["model_id"].map(
        lambda m: REGISTRY_INDEX[m].n_params if m in REGISTRY_INDEX else np.nan)
    return out.dropna(subset=["n_params"])


def _panel(ax, d: pd.DataFrame) -> None:
    d = d[d["n_params"] > 1e6]  # same sanity floor as figure3
    if d.empty:
        ax.set_axis_off()
        return
    for g in GROUP_ORDER:
        sub = d[d["group"] == g]
        if sub.empty:
            continue
        ax.scatter(sub["n_params"], sub["score"], s=26, color=GROUP_COLOR[g],
                   alpha=0.65, edgecolors="black", linewidths=0.4, zorder=2)
        if len(sub) >= 3:
            lx = np.log10(sub["n_params"].to_numpy(dtype=float))
            ly = sub["score"].to_numpy(dtype=float)
            b, a = np.polyfit(lx, ly, 1)
            xs = np.linspace(lx.min(), lx.max(), 50)
            ax.plot(10 ** xs, b * xs + a, color=GROUP_COLOR[g],
                    lw=1.4, ls="--", alpha=0.9, zorder=3)
    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=0.6, alpha=0.4, zorder=1)
    ax.grid(True, alpha=0.2)


def _page(fig, rows: list[tuple[str, pd.DataFrame]], score_label: str) -> None:
    """rows: list of (dataset_title, last-layer long_df) pairs; each becomes
    one row of len(ROI_ORDER) panels."""
    n_rows = len(rows)
    axes = fig.subplots(n_rows, len(ROI_ORDER), squeeze=False)
    for ri, (title, d) in enumerate(rows):
        for ci, roi in enumerate(ROI_ORDER):
            ax = axes[ri][ci]
            _panel(ax, d[d["roi"] == roi])
            if ri == 0:
                ax.set_title(roi, fontweight="semibold", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{title}\n{score_label}", fontsize=8.5)
            if ri == n_rows - 1:
                ax.set_xlabel("Parameters (log)")

    handles = [Line2D([], [], marker="o", ls="None", color=GROUP_COLOR[g],
                      markeredgecolor="black", markersize=7, label=GROUP_DISPLAY.get(g, g))
               for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(GROUP_ORDER), frameon=False,
               bbox_to_anchor=(0.5, -0.02 / n_rows))


def main() -> None:
    use_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "roi_scaling_lastlayer.pdf"

    with PdfPages(out) as pdf:
        # Page 1: embedding track, last layer -- LPP / NS / Podcast.
        rows = [
            ("LPP fMRI", _last_layer(load_lpp())),
            ("NS fMRI", _last_layer(load_ns_embed())),
            ("Podcast iEEG", _last_layer(load_podcast())),
        ]
        fig = plt.figure(figsize=(3.4 * len(ROI_ORDER), 3.0 * len(rows) + 0.6))
        _page(fig, rows, "Pearson r")
        fig.suptitle("ROI scaling laws -- embedding, last layer", fontsize=12,
                     fontweight="semibold", x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        pdf.savefig(fig, dpi=600)
        plt.close(fig)
        print(f"wrote page: embedding, last layer ({', '.join(t for t, _ in rows)})")

        # Page 2: surprisal track, last layer -- NS fMRI only (see docstring).
        rows_s = [("NS fMRI", _last_layer(load_ns_surprisal()))]
        fig = plt.figure(figsize=(3.4 * len(ROI_ORDER), 3.4))
        _page(fig, rows_s, "held-out $\\Delta$ Pearson r")
        fig.suptitle("ROI scaling laws -- surprisal, last layer", fontsize=12,
                     fontweight="semibold", x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        pdf.savefig(fig, dpi=600)
        plt.close(fig)
        print("wrote page: surprisal, last layer (NS fMRI only)")

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
