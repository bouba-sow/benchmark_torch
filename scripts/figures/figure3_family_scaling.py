#!/usr/bin/env python3
"""Paper figure 3: per-dataset-family scaling laws, LAST layer, two separate
pages in one PDF -- page 1 = surprisal features, page 2 = embedding features.
No title (caption goes in the paper). Fixed marker size everywhere (x-axis is
already model size, so an additional size encoding would be redundant).

Both pages use summarize/report.py's canonical ``_family_model_df``
(mode="surprisal"/"embedding") -- i.e. the SAME DLL-ceiling-normalised
BehavScore/NeuralScore convention (score_last_norm, read live off
outputs/benchmark_summary/<slug>/all_models_scores.csv) used everywhere else
in the current paper. Page 1 previously reused
the old figure-1 script (since removed)'s BEHAVIORAL_DELTA_SPEC/
NEURAL_DELTA_SPEC machinery -- the OLD, superseded held-out
delta-Pearson-r/ceiling metric (see figures/pareto_figures.py's docstring
for why that was replaced report-wide on 2026-09-08) -- AND read directly
from stale sweep dirs (geco/lme_sweep_dlt, zuco_reading_..._dlt/nr -- the
same stale/pre-refit paths already fixed in dlt/delta_loglik.py's
DATASET_SPEC) that predate the corrected 75/25 OOS resweeps, plus included
datasets since excluded from the canonical DATASETS list. Switched to `_family_model_df` 2026-09-08 (a user request to confirm
this figure used "the good normalization and the good scores" surfaced all
of this).

Output: reports/paper/family_scaling_lastlayer.pdf
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

from neuroai.analysis.benchmark_common import DATASET_FAMILY_ORDER, GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, REPO_ROOT
from neuroai.analysis.paper_style import use_paper_style
from scripts.summarize.report import _family_model_df, _load_model_df

OUT_DIR = REPO_ROOT / "reports" / "paper"

GROUP_COLOR = dict(GROUP_COLORS)
GROUP_COLOR["Constrained-BabyLM"] = "#ff7f0e"  # orange, was purple (#9467bd)


def _megmasc_family_df() -> pd.DataFrame:
    """MEG-MASC (Gwilliams), embedding track only -- direct pull, bypassing
    DATASETS/MODALITY_BUCKET (megmasc_meg is excluded from those, see
    neuroai/analysis/benchmark_common.py's DATASETS list note: its DLL/surprisal-track
    ceiling isn't settled yet). The embedding track doesn't touch DLL at
    all -- its correlation-path ceiling (r_tt_sqrt_mean, from a separate,
    already-computed test-retest reliability estimate) is fine -- so this
    one family/column is safe to include here per explicit request, even
    though the dataset stays out of every DLL-normalized comparison."""
    df = _load_model_df("megmasc_meg")
    if df is None or df.empty or "score_last_norm" not in df.columns:
        return pd.DataFrame()
    out = df[["model_id", "family", "group", "label", "n_params"]].copy()
    out["score"] = df["score_last_norm"]
    return out


def _draw_page(fig, dfs: dict[str, pd.DataFrame], ylabel: str, fam_order=DATASET_FAMILY_ORDER) -> None:
    fams = [f for f in fam_order if f in dfs and not dfs[f].empty]
    axes = fig.subplots(1, len(fams))
    axes = np.atleast_1d(axes)
    for ci, fam in enumerate(fams):
        ax = axes[ci]
        d = dfs[fam].dropna(subset=["score", "n_params"])
        # Sanity floor: a couple of unregistered BabyLM variants carry a
        # bogus n_params=98 baked into ns_fmri/lme_summary's last-layer file
        # specifically (the equivalent LPP/Tuckute files correctly leave it
        # blank for the same models) -- no real model in this benchmark has
        # under ~1M params, so treat anything below that as missing data.
        d = d[d["n_params"] > 1e6]
        if d.empty:
            ax.set_axis_off()
            continue
        for g in GROUP_ORDER:
            sub = d[d["group"] == g]
            if sub.empty:
                continue
            ax.scatter(sub["n_params"], sub["score"], s=30, color=GROUP_COLOR[g],
                       alpha=0.65, edgecolors="black", linewidths=0.4, zorder=2)
            if len(sub) >= 3:
                lx = np.log10(sub["n_params"].to_numpy(dtype=float))
                ly = sub["score"].to_numpy(dtype=float)
                b, a = np.polyfit(lx, ly, 1)
                xs = np.linspace(lx.min(), lx.max(), 50)
                ax.plot(10 ** xs, b * xs + a, color=GROUP_COLOR[g],
                        lw=1.6, ls="--", alpha=0.9, zorder=3)
        ax.set_xscale("log")
        ax.set_title(fam, fontweight="semibold", fontsize=10)
        ax.set_xlabel("Parameters (log)")
        if ci == 0:
            ax.set_ylabel(ylabel)

    handles = [Line2D([], [], marker="o", ls="None", color=GROUP_COLOR[g],
                      markeredgecolor="black", markersize=7, label=GROUP_DISPLAY.get(g, g))
               for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(GROUP_ORDER), frameon=False,
               bbox_to_anchor=(0.5, -0.04))


def main() -> None:
    use_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "family_scaling_lastlayer.pdf"
    n_fams = len(DATASET_FAMILY_ORDER)
    with PdfPages(out) as pdf:
        # Page 1: surprisal, DLL-ceiling-normalised score (score_last_norm),
        # the same convention as every other current paper figure.
        dfs_surprisal = {f: _family_model_df(f, "surprisal", "last") for f in DATASET_FAMILY_ORDER}
        fig = plt.figure(figsize=(3.6 * n_fams, 4.6))
        _draw_page(fig, dfs_surprisal, "Fraction of noise ceiling")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        pdf.savefig(fig, dpi=600)
        plt.close(fig)
        print("wrote page: surprisal, last layer")

        # Page 2: embedding, unchanged (full-model score / ceiling) -- plus
        # an extra MEG column (Gwilliams/megmasc_meg), per explicit request:
        # the embedding track doesn't need the DLL ceiling that's still
        # unsettled for this dataset, so it's safe here even though megmasc
        # stays excluded from every DLL-normalized comparison elsewhere.
        dfs_embed = {f: _family_model_df(f, "embedding", "last") for f in DATASET_FAMILY_ORDER}
        dfs_embed["MEG"] = _megmasc_family_df()
        fam_order_embed = list(DATASET_FAMILY_ORDER) + ["MEG"]
        fig = plt.figure(figsize=(3.6 * (n_fams + 1), 4.6))
        _draw_page(fig, dfs_embed, "Fraction of noise ceiling", fam_order=fam_order_embed)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        pdf.savefig(fig, dpi=600)
        plt.close(fig)
        print("wrote page: embedding, last layer (+MEG)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
