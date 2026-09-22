#!/usr/bin/env python3
"""Paper figure 2: Neural-behavioral Pareto front, surprisal features,
HARD-WORD subset only (DLT integration cost >= 2, Gibson 2000). Same
Pareto-figure convention/styling as figures/pareto_figures.py (same
colors, single circle marker, log-size-by-params dots) but restricted to the
syntactically hardest words in each dataset. BOTH axes are DLL-ceiling-
normalised mean per-observation delta-log-likelihood (full LME minus
baseline-only LME) -- the SAME BehavScore/NeuralScore convention used by the
main (all-words) surprisal Pareto figures now, not a different (correlation)
scale. Was previously scored as a held-out delta-Pearson-r via
outputs/dlt_corr_cache/*.csv -- switched 2026-09-08 (a reader noticed this
figure's axis looked inconsistent with the rest of the paper) once the
DLL-per-bin caches below were backfilled/wired in; neuroai/analysis/dlt_difficulty.py
still keeps the old correlation-based CORR_CACHE_FILES/_load_corr_one for
summarize/report.py's own hard-case Pareto section, untouched here.

Neural axis: DLL-ceiling-normalised mean delta-loglik on the hard-word
subset, from podcast_ieeg only (outputs/dlt_loglik_cache/podcast_lme.csv,
written by scripts/dlt/delta_loglik_podcast.py -- pools every lag in
the word-onset-relative window [-0.2s, +0.8s], same convention as the old
correlation script). fedorenko2016_ecog was removed from this axis per
explicit request; see dlt/delta_loglik_podcast.py's docstring for why
the other 6 neural surprisal datasets in this report aren't computable this
way at all (no per-observation delta-loglik cached).

Behavioral axis: DLL-ceiling-normalised mean delta-loglik, pooled over just
the hard-word subset, averaged per model across every dataset/measure entry
in ``neuroai.analysis.dlt_difficulty.LOGLIK_CACHE_FILES`` (as of 2026-09-08: geco
FP/GP, meco FP/GP/SP, provo FP/GP/SP, ucl_reading, zuco_reading NR/SR/TSR --
onestop excluded, no DLT-cost reference file computed for it yet) -- the
exact same DLL convention as the main surprisal Pareto's behavioral axis,
just restricted to DLT cost >= 2 words instead of all words. Deliberately
excludes the embedding-track behavioral datasets used by the main report's
hard-case Pareto section (no DLL concept for the ridge-regression embedding
track).

Output: reports/paper/surprisal_hard_dlt_pareto.pdf
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
from matplotlib.lines import Line2D

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, REPO_ROOT
from neuroai.analysis.dlt_difficulty import LOGLIK_CACHE_FILES, load_behavioral_dll, load_podcast_dll
from neuroai.analysis.paper_style import log_size_map, use_paper_style
from scripts.summarize.report import _log_range, _pareto_front

OUT_DIR = REPO_ROOT / "reports" / "paper"
DLT_THRESHOLD = 2.0


def main() -> None:
    use_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    behav = load_behavioral_dll("hard", DLT_THRESHOLD)
    behav = behav[~behav["model_id"].isin(EXCLUDED_MODEL_IDS)]
    print(f"[behavioral, DLL] {behav['model_id'].nunique()} models, "
          f"mean score/ceiling={behav['behav'].mean():.4f}")
    neural = load_podcast_dll("hard", DLT_THRESHOLD)
    neural = neural[~neural["model_id"].isin(EXCLUDED_MODEL_IDS)]
    print(f"[neural, DLL, podcast_ieeg only] {neural['model_id'].nunique()} models, "
          f"mean score/ceiling={neural['neural'].mean():.4f}")
    neural = neural[["model_id", "neural"]]

    m = behav.merge(neural, on="model_id", how="inner").dropna(subset=["neural", "behav"])
    n_behav_sources = len(LOGLIK_CACHE_FILES)
    print(f"n={len(m)} models, averaged across {n_behav_sources} surprisal-track behavioral "
          f"dataset/measure entries (DLT cost >= {DLT_THRESHOLD:g})")

    # Single circle marker for every group (color already encodes group) --
    # matches the canonical convention in figures/pareto_figures.py
    # (surprisal_lastlayer_pareto.pdf etc.), which replaced the old
    # per-group-marker-shape style this figure used to copy from figure1.
    lo, hi = _log_range([m], "n_params")
    sizes_all = log_size_map(m["n_params"], lo, hi, s_min=25, s_max=400)

    front = _pareto_front(m, "neural", "behav")
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for g in GROUP_ORDER:
        mask = (m["group"] == g).to_numpy()
        sub = m[mask]
        if sub.empty:
            continue
        ax.scatter(sub["neural"], sub["behav"], s=sizes_all[mask], color=GROUP_COLORS[g],
                   marker="o", alpha=0.6, edgecolors="white", linewidths=0.5, zorder=2,
                   label=GROUP_DISPLAY.get(g, g))
    if len(front) > 1:
        ax.plot(front["neural"], front["behav"], color="black", lw=1.8, alpha=0.9, zorder=3)
    for _, r in front.iterrows():
        ax.scatter([r["neural"]], [r["behav"]], s=90, facecolors="none",
                   edgecolors="black", linewidths=1.8, zorder=4, marker="o")
        ax.annotate(r["label"], (r["neural"], r["behav"]), fontsize=7, color="black",
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("BrainScore", fontsize=13, color="black")
    ax.set_ylabel("BehavScore", fontsize=13, color="black")
    ax.tick_params(axis="both", labelsize=9.5, colors="black", width=1.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("black")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.6)

    group_handles = [Line2D([], [], marker="o", linestyle="None", color=GROUP_COLORS[g],
                            markeredgecolor="white", markersize=8, label=GROUP_DISPLAY.get(g, g))
                     for g in GROUP_ORDER if (m["group"] == g).any()]
    group_legend = ax.legend(handles=group_handles, fontsize=8, title_fontsize=8,
                              loc="upper left", frameon=True, title="Model group")
    ax.add_artist(group_legend)

    size_refs = [r for r in (1e7, 1e8, 1e9, 1e10, 7e10) if lo <= np.log10(r) <= hi]
    size_handles = [
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="grey",
               markeredgecolor="white", alpha=0.65,
               markersize=float(np.sqrt(log_size_map(pd.Series([r]), lo, hi, s_min=25, s_max=400)[0])),
               label=(f"{r/1e9:.0f}B" if r >= 1e9 else f"{r/1e6:.0f}M"))
        for r in size_refs
    ]
    if size_handles:
        ax.legend(handles=size_handles, title="model size", fontsize=8, title_fontsize=8,
                  loc="lower right", frameon=True, labelspacing=1.2, borderpad=0.9)
        ax.add_artist(group_legend)

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = OUT_DIR / "surprisal_hard_dlt_pareto.pdf"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
