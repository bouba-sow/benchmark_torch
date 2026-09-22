#!/usr/bin/env python3
"""Paper figures: Neural-behavioral Pareto front, one PDF per (feature mode,
layer selection) combination -- surprisal/embedding x best/last layer.

2026-09-08: REPLACES the old the old figure-1 script (since removed),
whose docstring explicitly forced BOTH axes onto delta-Pearson-r ("per
explicit request" at the time) -- wrong for the behavioral axis, which
should use the same DLL-noise-ceiling-normalized score (score_best_norm /
score_last_norm) as every other figure in this report. Reuses
summarize/report.py's already-fixed, already-audited
_neural_behavioral_merged()/ceiling_for() pipeline directly instead of a
second parallel one, so this figure can't drift out of sync with the rest
of the report again: for mode="surprisal" both axes are DLL-ceiling-
normalized (per-observation delta log-likelihood / split-half DLL ceiling);
for mode="embedding" both axes are correlation-ceiling-normalized (ridge/
BrainScore Pearson r / split-half correlation ceiling).

Markers: circle only (color already encodes group) -- no per-group marker
shape.

Output: reports/paper/{surprisal,embedding}_{best,last}layer_pareto.pdf
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

from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, REPO_ROOT
from neuroai.analysis.paper_style import log_size_map, use_paper_style
from scripts.summarize.report import (
    _neural_behavioral_merged, _neural_behavioral_merged_zscore, _pareto_front,
)

OUT_DIR = REPO_ROOT / "reports" / "paper"

_SIZE_REFS = (1e7, 1e8, 1e9, 1e10, 7e10)


def _fmt_params(n: float) -> str:
    return f"{n/1e9:.0f}B" if n >= 1e9 else f"{n/1e6:.0f}M"


def _draw_one(mode: str, layer_sel: str, out_name: str, *, robust: bool = False, raw_z: bool = False,
              exclude_neural_bases: frozenset[str] = frozenset()) -> Path | None:
    if raw_z:
        m = _neural_behavioral_merged_zscore(mode, layer_sel, raw=True)
    elif robust:
        m = _neural_behavioral_merged_zscore(mode, layer_sel)
    else:
        m = _neural_behavioral_merged(mode, layer_sel, exclude_neural_bases)
    m = m.dropna(subset=["neural", "behav", "n_params"])
    if m.empty:
        print(f"  {out_name}: no data -- skipping")
        return None
    front = _pareto_front(m, "neural", "behav")

    v = m["n_params"].astype(float)
    v = v[v > 0]
    lo, hi = (float(np.log10(v.min())), float(np.log10(v.max()))) if len(v) else (0.0, 1.0)
    sizes = log_size_map(m["n_params"], lo, hi, s_min=25, s_max=400)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    for g in GROUP_ORDER:
        mask = (m["group"] == g).to_numpy()
        if not mask.any():
            continue
        ax.scatter(m.loc[mask, "neural"], m.loc[mask, "behav"], s=sizes[mask],
                   color=GROUP_COLORS[g], alpha=0.6, edgecolors="white",
                   linewidths=0.5, zorder=2, marker="o", label=g)

    if len(front) > 1:
        ax.plot(front["neural"], front["behav"], color="black", lw=1.8, zorder=3, alpha=0.9)
    for _, r in front.iterrows():
        ax.scatter([r["neural"]], [r["behav"]], s=90, facecolors="none",
                   edgecolors="black", linewidths=1.8, zorder=4, marker="o")
        ax.annotate(r["label"], (r["neural"], r["behav"]), fontsize=7,
                    color="black", xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("BrainScore", fontsize=13, color="black")
    ax.set_ylabel("BehavScore", fontsize=13, color="black")
    ax.tick_params(axis="both", labelsize=9.5, colors="black", width=1.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("black")
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.6)

    # Model-group legend (color; circle marker for every group -- see module
    # docstring) -- upper-left, matching the old figure's two-legend layout.
    group_handles = [
        Line2D([], [], marker="o", linestyle="None", color=GROUP_COLORS[g],
               markeredgecolor="white", markersize=8, label=GROUP_DISPLAY.get(g, g))
        for g in GROUP_ORDER if (m["group"] == g).any()
    ]
    group_legend = ax.legend(handles=group_handles, fontsize=8, title_fontsize=8,
                              loc="upper left", frameon=True, title="Model group")
    ax.add_artist(group_legend)

    # Model-size legend -- lower-right, as before.
    refs = [r for r in _SIZE_REFS if lo <= np.log10(r) <= hi]
    handles = [
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="grey",
               markeredgecolor="white", alpha=0.65,
               markersize=float(np.sqrt(log_size_map(pd.Series([r]), lo, hi)[0])),
               label=_fmt_params(r))
        for r in refs
    ]
    if handles:
        ax.legend(handles=handles, title="model size", fontsize=8, title_fontsize=8,
                  loc="lower right", frameon=True, labelspacing=1.2, borderpad=0.9)
        ax.add_artist(group_legend)  # re-assert -- the second legend() call above replaces ax's default legend

    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"  Wrote {out}")
    return out


def main() -> None:
    use_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for robust in (False, True):
        suffix = "_robustz" if robust else ""
        for mode in ("surprisal", "embedding"):
            for layer_sel in ("last", "best"):
                print(f"=== {mode}, {layer_sel} layer{' (robust)' if robust else ''} ===")
                _draw_one(mode, layer_sel, f"{mode}_{layer_sel}layer_pareto{suffix}.pdf", robust=robust)


if __name__ == "__main__":
    main()
