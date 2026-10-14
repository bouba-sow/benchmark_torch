#!/usr/bin/env python3
"""Paper figure 6: Neural-behavioral Pareto front, surprisal features, LAST
layer, across DLT (Gibson 2000) integration-cost bins: easy (cost==0), then
hard cost >= 1/2/3/4/5 (capped at 5 -- beyond that, several behavioral
datasets thin out to single-digit word counts, too sparse to be meaningful).
Same Pareto convention as figures/figure2_hard_dlt_pareto.py (which this
reuses/generalizes) -- one 6-panel row in a single figure instead of 6
separate files. Both axes are DLL-ceiling-normalised mean per-observation
delta-log-likelihood (BehavScore/NeuralScore, same convention as the main
surprisal Pareto figures) -- was a held-out delta-Pearson-r via
outputs/dlt_corr_cache/*.csv until 2026-09-08 (a reader noticed this grid's
axis looked inconsistent with the rest of the paper); see
figures/figure2_hard_dlt_pareto.py's docstring for the full history.

Neural axis: podcast_ieeg only (outputs/dlt_loglik_cache/podcast_lme.csv,
written by scripts/dlt/delta_loglik_podcast.py -- pools every lag
in the word-onset-relative window [-0.2s, +0.8s], not a "peak lag"
selection). fedorenko2016_ecog was removed from this axis per explicit
request. Podcast's DLT cost goes up to 80 with thousands of hard words even
at cost>=5, so every panel gets a real 2D Pareto scatter -- no fallback to a
behavioral-only strip plot is needed at any threshold this figure requests.

The behavioral axis (12 dataset/measure entries pooled, see
neuroai.analysis.dlt_difficulty.LOGLIK_CACHE_FILES) still thins out fast at higher
thresholds -- so some models may have NaN/dropped behavioral scores from the
sparser datasets at the cost>=4 and cost>=5 panels.

Output: reports/paper/surprisal_dlt_threshold_grid.pdf
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
from matplotlib.ticker import MaxNLocator

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, REPO_ROOT
from neuroai.analysis.dlt_difficulty import (
    LOGLIK_CACHE_DIR, LOGLIK_CACHE_FILES, MAX_BIN_CEILING_THRESHOLD, PODCAST_LOGLIK_CACHE, load_behavioral_dll,
    load_behavioral_dll_bincal, load_podcast_dll, load_podcast_dll_bincal,
)
from neuroai.analysis.paper_style import log_size_map, use_paper_style
from scripts.summarize.report import _log_range, _pareto_front

OUT_DIR = REPO_ROOT / "reports" / "paper"

# (panel title, dlt_bin, hard_gt_threshold used to select the row -- "easy"
# rows carry the same value at every threshold, so any threshold selects the
# same easy set; 1.0 is used arbitrarily for the easy panel).
PANELS: list[tuple[str, str, float]] = [
    ("Easy (cost = 0)", "easy", 1.0),
    ("Hard (cost ≥ 1)", "hard", 1.0),
    ("Hard (cost ≥ 2)", "hard", 2.0),
    ("Hard (cost ≥ 3)", "hard", 3.0),
    ("Hard (cost ≥ 4)", "hard", 4.0),
    ("Hard (cost ≥ 5)", "hard", 5.0),
]
PANEL_NCOLS = len(PANELS)  # single row

# Single circle marker for every group (color already encodes group) --
# matches the canonical convention in figures/pareto_figures.py /
# figures/figure2_hard_dlt_pareto.py.
GROUP_COLOR = dict(GROUP_COLORS)
GROUP_COLOR["Constrained-BabyLM"] = "#ff7f0e"

# Models excluded by id. AntonV/mamba2-130m-hf was the single highest-scoring
# model on the "easy" behavioral panel by a wide margin under the old
# correlation-delta metric (0.00736 vs. the next-best 0.00401 -- nearly 2x --
# and ~5 std above the mean of 0.0021+-0.0010 across the other 99 models), an
# isolated outlier; kept excluded from this figure only (not from
# EXCLUDED_MODEL_IDS globally) after the switch to the DLL metric too, since
# re-checking it there didn't change the case for including it back in.
EXCLUDED_MODEL_IDS_FIGURE6: frozenset[str] = frozenset({"AntonV/mamba2-130m-hf"})


def _word_counts(dlt_bin: str, hard_gt: float) -> tuple[int, int, dict[str, int]]:
    """(behavioral_total, neural_total) word counts actually used at this
    (dlt_bin, hard_gt) filter -- the mode of ``n_words_unique`` per dataset
    (nearly always a single constant value per dataset; GECO is the one
    exception, where a handful of models get matched to a smaller word set
    due to a known per-model sentid/zone merge gap, so the mode -- the value
    the vast majority of models actually used -- is the representative
    count, not the mean/median which would be pulled toward the minority
    value), summed across every dataset pooled into that axis (12 behavioral
    dataset/measure entries; 1 neural dataset, podcast).

    ``n_words`` (row count) is NOT the real word count here -- the
    dlt_loglik_cache per-word files have one row per (subject, word) (and,
    for podcast, also per lag x electrode), so ``n_words`` is inflated by
    that multiplicity (e.g. ~4x on GECO, ~285x on podcast). ``n_words_unique``
    (nunique of the actual word key) is the real count -- reintroduced here
    after this cache's plain ``n_words`` was wired into these word-count
    displays without it (found 2026-09-08, the same inflation issue
    previously fixed for the correlation-cache-based version of this figure).
    Also returns the per-dataset breakdown for the behavioral axis (dataset
    -> word count) so the figure can show it."""
    def _mode_n_words_unique(df: pd.DataFrame) -> int:
        sub = df[np.isclose(df["hard_gt_threshold"].astype(float), hard_gt) & (df["dlt_bin"] == dlt_bin)]
        col = "n_words_unique" if "n_words_unique" in sub.columns else "n_words"
        vals = sub[col].dropna() if col in sub.columns else pd.Series(dtype=float)
        return int(vals.mode().iloc[0]) if not vals.empty else 0

    behav_total = 0
    behav_breakdown: dict[str, int] = {}
    for report_slug, (cache_stem, ceiling_slug) in LOGLIK_CACHE_FILES.items():
        path = LOGLIK_CACHE_DIR / f"{cache_stem}.csv"
        if not path.is_file():
            continue
        n = _mode_n_words_unique(pd.read_csv(path))
        behav_breakdown[report_slug] = n
        behav_total += n

    neural_total = 0
    if PODCAST_LOGLIK_CACHE.is_file():
        neural_total = _mode_n_words_unique(pd.read_csv(PODCAST_LOGLIK_CACHE))

    return behav_total, neural_total, behav_breakdown


_BEHAV_DATASET_LABEL = {
    "geco": "GECO/FP", "geco_gp": "GECO/GP",
    "meco": "MECO/FP", "meco_gp": "MECO/GP", "meco_sp": "MECO/SP",
    "provo": "Provo/FP", "provo_gp": "Provo/GP", "provo_sp": "Provo/SP",
    "ucl_reading": "UCL",
    "zuco_reading_nr": "ZuCo/NR", "zuco_reading_sr": "ZuCo/SR", "zuco_reading_tsr": "ZuCo/TSR",
}


def _format_breakdown(behav_breakdown: dict[str, int]) -> str:
    return "  ".join(f"{_BEHAV_DATASET_LABEL.get(k, k)}={v:,}" for k, v in behav_breakdown.items())


def _draw_pareto_panel(ax, m: pd.DataFrame, title: str, n_words_behav: int = 0, n_words_neural: int = 0,
                       behav_breakdown: dict[str, int] | None = None) -> None:
    lo, hi = _log_range([m], "n_params")
    sizes_all = log_size_map(m["n_params"], lo, hi, s_min=12, s_max=160)
    front = _pareto_front(m, "neural", "behav")
    for g in GROUP_ORDER:
        mask = (m["group"] == g).to_numpy()
        sub = m[mask]
        if sub.empty:
            continue
        ax.scatter(sub["neural"], sub["behav"], s=sizes_all[mask], color=GROUP_COLOR[g],
                   marker="o", alpha=0.6, edgecolors="white",
                   linewidths=0.4, zorder=2)
    ax.plot(front["neural"], front["behav"], color="#333333", lw=1.0, alpha=0.7, zorder=3)
    ax.scatter(front["neural"], front["behav"], s=70, facecolors="none",
               edgecolors="#333333", linewidths=1.2, zorder=4)
    for _, r in front.iterrows():
        ax.annotate(r["label"], (r["neural"], r["behav"]), fontsize=5.5,
                    xytext=(3, 3), textcoords="offset points", zorder=5)
    ax.set_title(f"{title}\n(n={len(m)} models; words: behav={n_words_behav:,}, neural={n_words_neural:,})",
                fontsize=8, fontweight="semibold")
    ax.set_xlabel("BrainScore", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)
    # Narrow panels + matplotlib's default tick count collided into
    # overlapping, unreadable x-tick labels (e.g. "0.0025 0.050") -- cap it
    # to a handful of well-spaced ticks instead.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))


def _draw_behavioral_only_panel(ax, behav: pd.DataFrame, title: str, n_words_behav: int = 0,
                                behav_breakdown: dict[str, int] | None = None) -> None:
    """Fallback for panels with no neural data: 1D strip plot of the
    behavioral score, one dot per model, grouped/colored/jittered by group."""
    if behav.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)", fontsize=9)
        return
    rng = np.random.default_rng(0)
    y_base = {g: i for i, g in enumerate(GROUP_ORDER)}
    for g in GROUP_ORDER:
        sub = behav[behav["group"] == g]
        if sub.empty:
            continue
        jitter = rng.uniform(-0.3, 0.3, size=len(sub))
        ax.scatter(sub["behav"], y_base[g] + jitter, color=GROUP_COLOR[g],
                   marker="o", s=22, alpha=0.6,
                   edgecolors="white", linewidths=0.4, zorder=2)
    ax.set_yticks(list(y_base.values()))
    ax.set_yticklabels([GROUP_DISPLAY.get(g, g) for g in GROUP_ORDER], fontsize=7)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(f"{title}\n(n={behav['model_id'].nunique()} models; words: behav={n_words_behav:,}, no neural data)",
                fontsize=8, fontweight="semibold")
    ax.set_xlabel("BehavScore", fontsize=8)
    ax.grid(True, axis="x", alpha=0.2)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-ceiling", action="store_true",
                    help="divide each DLT bin by the ceiling computed on that bin's own words; panels restricted to "
                         "easy, cost >= 1 and >= 2 (thin bins at higher cost have unstable ceilings); output "
                         "surprisal_dlt_threshold_grid_binceiling.pdf")
    bin_ceiling = ap.parse_args().bin_ceiling
    load_behav, load_neural = ((load_behavioral_dll_bincal, load_podcast_dll_bincal) if bin_ceiling
                               else (load_behavioral_dll, load_podcast_dll))
    panels = [p for p in PANELS if not (bin_ceiling and p[1] == "hard" and p[2] > MAX_BIN_CEILING_THRESHOLD)]
    use_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ncols = len(panels)
    nrows = -(-len(panels) // ncols)  # ceil
    # constrained_layout instead of a trailing tight_layout(): on this very
    # wide (6 panels, single row), short figure, tight_layout mis-sized the
    # per-panel margins -- each axes box ended up squeezed into roughly the
    # right 40% of its panel, with a large dead-space gap on the left (worst
    # on the leftmost panel, which also carries the y-axis label/ticks).
    # constrained_layout solves the same per-axes spacing problem
    # incrementally as artists are added and accounts for text bounding
    # boxes (tick labels, the outlier annotations) directly, which is what
    # tight_layout's single end-of-figure pass was getting wrong here.
    fig, axes = plt.subplots(nrows, ncols, figsize=((4.6 if not bin_ceiling else 5.4) * ncols, 4.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (title, dlt_bin, hard_gt) in zip(axes, panels):
        behav = load_behav(dlt_bin, hard_gt)
        behav = behav[~behav["model_id"].isin(EXCLUDED_MODEL_IDS) & ~behav["model_id"].isin(EXCLUDED_MODEL_IDS_FIGURE6)]
        neural = load_neural(dlt_bin, hard_gt)
        neural = neural[~neural["model_id"].isin(EXCLUDED_MODEL_IDS) & ~neural["model_id"].isin(EXCLUDED_MODEL_IDS_FIGURE6)]
        n_words_behav, n_words_neural, behav_breakdown = _word_counts(dlt_bin, hard_gt)
        print(f"[{title}] behav n={behav['model_id'].nunique() if not behav.empty else 0}, "
              f"neural n={neural['model_id'].nunique() if not neural.empty else 0}, "
              f"words: behav={n_words_behav} ({behav_breakdown}), neural={n_words_neural}")
        if neural.empty:
            _draw_behavioral_only_panel(ax, behav, title, n_words_behav, behav_breakdown)
            continue
        neural_s = neural[["model_id", "neural"]]
        m = behav.merge(neural_s, on="model_id", how="inner").dropna(subset=["neural", "behav"])
        if m.empty:
            _draw_behavioral_only_panel(ax, behav, title, n_words_behav, behav_breakdown)
            continue
        _draw_pareto_panel(ax, m, title, n_words_behav, n_words_neural, behav_breakdown)
    for j in range(len(panels), len(axes)):
        axes[j].set_axis_off()
    for row in range(nrows):
        axes[row * ncols].set_ylabel("BehavScore", fontsize=8)

    handles = [Line2D([], [], marker="o", linestyle="None", color=GROUP_COLOR[g],
                      markeredgecolor="white", markersize=7, label=GROUP_DISPLAY.get(g, g))
               for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(GROUP_ORDER), frameon=False)

    out = OUT_DIR / ("surprisal_dlt_threshold_grid_binceiling.pdf" if bin_ceiling else "surprisal_dlt_threshold_grid.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
