#!/usr/bin/env python3
"""Ad hoc diagnostic figure: scaling law for a single neural dataset's
surprisal track, last layer -- raw (unnormalized) score alongside the
ceiling-normalized score, side by side, so the effect of dividing by the
dataset's DLL ceiling on the model ranking/scaling picture is visible
directly. Same per-group scatter + log-linear-fit style as
figures/figure3_family_scaling.py / figures/figure_roi_scaling.py.

Context: 2026-09-14 investigation into why RWKV4-169M appeared on the
surprisal Pareto front (reports/paper/surprisal_lastlayer_pareto.pdf) --
Pereira's tiny DLL ceiling (~0.0029) amplifies already-noisy raw scores.
RWKV4-169M is in fact rank #1/116 by RAW score too (not purely a
normalization artifact), but raw scores are tightly clustered around zero
(mean=-0.0004, std=0.001) with most models below zero, consistent with a
dataset whose surprisal signal is barely distinguishable from noise.
LPP fMRI (ceiling ~0.0101, ~3.5x Pereira's) is plotted alongside it for
comparison.

Usage: scripts/figures/figure_pereira_scaling.py [--dataset SLUG ...]
Defaults to both pereira_fmri and lpp_fmri__surprisal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from neuroai.analysis.benchmark_common import GROUP_DISPLAY, GROUP_ORDER, REPO_ROOT, dll_ceiling_for
from neuroai.analysis.paper_style import use_paper_style
from scripts.figures.figure3_family_scaling import GROUP_COLOR
from scripts.summarize.report import _load_model_df

DEFAULT_DATASETS = ["pereira_fmri", "lpp_fmri__surprisal"]
ALL_FMRI_DATASETS = ["lpp_fmri__surprisal", "ns_fmri__surprisal", "pereira_fmri", "tuckute_fmri__surprisal"]
OUT_DIR = REPO_ROOT / "reports" / "paper"


def _panel(ax, d: pd.DataFrame, y_col: str, title: str, ylabel: str) -> None:
    d = d.dropna(subset=[y_col, "n_params"])
    d = d[d["n_params"] > 1e6]
    for g in GROUP_ORDER:
        sub = d[d["group"] == g]
        if sub.empty:
            continue
        ax.scatter(sub["n_params"], sub[y_col], s=30, color=GROUP_COLOR[g],
                   alpha=0.65, edgecolors="black", linewidths=0.4, zorder=2,
                   label=GROUP_DISPLAY.get(g, g))
        if len(sub) >= 3:
            lx = np.log10(sub["n_params"].to_numpy(dtype=float))
            ly = sub[y_col].to_numpy(dtype=float)
            b, a = np.polyfit(lx, ly, 1)
            xs = np.linspace(lx.min(), lx.max(), 50)
            ax.plot(10 ** xs, b * xs + a, color=GROUP_COLOR[g], lw=1.6, ls="--", alpha=0.9, zorder=3)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5, zorder=1)
    ax.set_xscale("log")
    ax.set_title(title, fontweight="semibold", fontsize=10)
    ax.set_xlabel("Parameters (log)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)


def _build_one(slug: str) -> Path:
    df = _load_model_df(slug)
    if df is None or df.empty:
        raise SystemExit(f"{slug} summary not found -- run summarize/benchmark.py first")
    ceiling = dll_ceiling_for(slug)
    ceiling_str = f"{ceiling:.4f}" if ceiling is not None else "n/a"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _panel(axes[0], df, "score_last", "Raw (unnormalized)", "held-out $\\Delta$LL / word")
    _panel(axes[1], df, "score_last_norm", "Ceiling-normalized", "BrainScore (fraction of DLL ceiling)")

    handles = [Line2D([], [], marker="o", ls="None", color=GROUP_COLOR[g],
                      markeredgecolor="black", markersize=7, label=GROUP_DISPLAY.get(g, g))
               for g in GROUP_ORDER if (df["group"] == g).any()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{slug} -- surprisal, last layer  (n={df['model_id'].nunique()} models, "
                 f"DLL ceiling $\\approx$ {ceiling_str})", fontsize=12, fontweight="semibold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out = OUT_DIR / f"{slug.replace('__', '_')}_scaling.pdf"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def _build_combined(slugs: list[str], out_name: str) -> Path:
    n = len(slugs)
    fig, axes = plt.subplots(n, 2, figsize=(14, 6 * n))
    all_groups: set[str] = set()
    for row, slug in enumerate(slugs):
        df = _load_model_df(slug)
        if df is None or df.empty:
            raise SystemExit(f"{slug} summary not found -- run summarize/benchmark.py first")
        ceiling = dll_ceiling_for(slug)
        ceiling_str = f"{ceiling:.4f}" if ceiling is not None else "n/a"
        all_groups |= set(df["group"].unique())

        row_label = f"{slug}  (n={df['model_id'].nunique()}, ceiling$\\approx${ceiling_str})"
        ax_raw, ax_norm = axes[row]
        _panel(ax_raw, df, "score_last", f"{row_label}\nRaw (unnormalized)", "held-out $\\Delta$LL / word")
        _panel(ax_norm, df, "score_last_norm", f"{row_label}\nCeiling-normalized", "BrainScore (fraction of DLL ceiling)")

    handles = [Line2D([], [], marker="o", ls="None", color=GROUP_COLOR[g],
                      markeredgecolor="black", markersize=7, label=GROUP_DISPLAY.get(g, g))
               for g in GROUP_ORDER if g in all_groups]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.005 / n))
    fig.suptitle("fMRI datasets -- surprisal scaling law, last layer", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.03 / n, 1, 0.97), h_pad=3.0)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", dest="datasets", default=None)
    parser.add_argument("--all_fmri", action="store_true", help="combine all 4 fMRI datasets into one PDF")
    parser.add_argument("--out_name", default="fmri_surprisal_scaling_all.pdf")
    args = parser.parse_args()

    use_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_fmri:
        _build_combined(args.datasets or ALL_FMRI_DATASETS, args.out_name)
        return

    datasets = args.datasets or DEFAULT_DATASETS
    for slug in datasets:
        _build_one(slug)


if __name__ == "__main__":
    main()
