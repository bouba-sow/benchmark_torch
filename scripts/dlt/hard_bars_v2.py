#!/usr/bin/env python3
"""DLT (Gibson 2000) hard-word (integration cost >= 2) vs regular-word bar
plots, for behavioral-surprisal, behavioral-embedding, and neural tracks --
built directly from the precomputed outputs/dlt_difficulty/*_multi_threshold.csv
files (fast, no recomputation needed). Supersedes dlt/hard_word_bars.py
(which computed the same thing the hard way, from raw per-word predictions,
for behavioral-surprisal only).

Usage: .venv/bin/python3 scripts/dlt/hard_bars_v2.py
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

from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, REPO_ROOT, sem
from neuroai.analysis.dlt_difficulty import DLT_DIR, DLT_GROUPS, _load_one, aggregate_group
from scripts.summarize.benchmark import lookup_model_meta

HARD_THRESHOLD = 2.0
OUT_DIR = DLT_DIR
LOGLIK_CACHE_DIR = REPO_ROOT / "outputs" / "dlt_loglik_cache"

# behavioral-surprisal file stem (as used in DLT_GROUPS) -> delta-loglik cache filename stem
LOGLIK_FILE_MAP: dict[str, str] = {
    "geco": "geco", "provo": "provo", "ucl_reading": "ucl_reading",
    "meco": "meco", "zuco_reading_nr": "zuco_reading",
}


def _load_loglik(file_stem: str, hard_gt: float = 2.0) -> pd.DataFrame | None:
    """Mean delta-log-likelihood (native LME metric, NOT ceiling-normalised --
    log-lik has no fixed ceiling) split by DLT easy/hard bin, from the
    outputs/dlt_loglik_cache/{dataset}.csv computed by
    scripts/dlt/delta_loglik.py."""
    cache_stem = LOGLIK_FILE_MAP.get(file_stem, file_stem)
    path = LOGLIK_CACHE_DIR / f"{cache_stem}.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df = df[df["hard_gt_threshold"] == hard_gt]
    if df.empty:
        return None
    rows = []
    for model_id, g in df.groupby("model"):
        family = g["family"].iloc[0]
        n_params_fallback = g["n_params"].iloc[0] if "n_params" in g.columns else None
        meta = lookup_model_meta(model_id, family, n_params_fallback=n_params_fallback)
        easy = g.loc[g["dlt_bin"] == "easy", "score"]
        hard = g.loc[g["dlt_bin"] == "hard", "score"]
        if easy.empty or hard.empty:
            continue
        rows.append({
            "model_id": model_id, "family": meta.family, "group": meta.group,
            "label": meta.label, "n_params": meta.n_params,
            "score_easy": float(easy.iloc[0]), "score_hard": float(hard.iloc[0]),
        })
    return pd.DataFrame(rows) if rows else None


def _draw_group_bar_axis(ax, d, score_col: str, title: str) -> list[tuple[float, float]]:
    """Mean +/- SEM only (diamond + whiskers) per group -- no bar rectangles
    or per-model scatter, so a handful of outlier models can't force the
    y-axis wide enough to flatten the mean markers into invisibility."""
    groups = [g for g in GROUP_ORDER if (d["group"] == g).any()]
    mean_sems: list[tuple[float, float]] = []
    for gi, g in enumerate(groups):
        sub = d[d["group"] == g]
        vals = sub[score_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        m, s = float(vals.mean()), sem(vals)
        mean_sems.append((m, s))
        ax.errorbar(gi, m, yerr=s, fmt="D", color=GROUP_COLORS[g], markersize=7,
                    markerfacecolor=GROUP_COLORS[g], markeredgecolor="black", markeredgewidth=0.8,
                    ecolor=GROUP_COLORS[g], capsize=5, elinewidth=1.8, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{GROUP_DISPLAY.get(g, g)}\n(n={int((d['group'] == g).sum())})" for g in groups], fontsize=8.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.axhline(0, color="grey", linewidth=0.8, zorder=0)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    return mean_sems


EASY_LABEL = "Easy words (dlt=0)"
HARD_LABEL = f"Hard words (dlt≥{HARD_THRESHOLD:g})"


def _plot_group_per_dataset(group_key: str, group_title: str, use_loglik: bool = False) -> Path | None:
    """One row per dataset file in the group, 2 panels (easy | hard) each --
    NOT averaged across datasets, per explicit feedback that cross-dataset
    averaging hides which specific datasets show the effect. If
    ``use_loglik``, the behavioral-surprisal track's native metric (mean
    delta log-likelihood, NOT ceiling-normalised) is used instead of the
    ceiling-normalised delta-Pearson-r used everywhere else, per explicit
    request that the LME track's headline metric is delta-loglik."""
    file_stems = DLT_GROUPS[group_key]
    loader = _load_loglik if use_loglik else _load_one
    per_dataset = {f: d for f in file_stems if (d := loader(f, HARD_THRESHOLD)) is not None}
    if not per_dataset:
        print(f"[{group_key}] no data -- skipping")
        return None

    ylabel = "mean delta log-likelihood\n(full LME - baseline LME)" if use_loglik else "score / noise ceiling"
    nrows = len(per_dataset)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.4 * nrows), squeeze=False)
    for row, (file_stem, d) in enumerate(per_dataset.items()):
        ms_easy = _draw_group_bar_axis(axes[row][0], d, "score_easy", f"{file_stem} -- {EASY_LABEL}")
        ms_hard = _draw_group_bar_axis(axes[row][1], d, "score_hard", f"{file_stem} -- {HARD_LABEL}")
        lo = min((m - s for m, s in ms_easy + ms_hard), default=0.0)
        hi = max((m + s for m, s in ms_easy + ms_hard), default=1.0)
        pad = (hi - lo) * 0.25 or 0.01
        for ax in axes[row]:
            ax.set_ylim(lo - pad, hi + pad)
        axes[row][0].set_ylabel(ylabel)

    handles = [plt.Line2D([0], [0], marker="D", color=GROUP_COLORS[g], markersize=7,
                          markerfacecolor=GROUP_COLORS[g], markeredgecolor="black",
                          markeredgewidth=0.8, linestyle="none")
               for g in GROUP_ORDER]
    fig.legend(handles, [GROUP_DISPLAY.get(g, g) for g in GROUP_ORDER], loc="upper center",
               ncol=len(GROUP_ORDER), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"{group_title}: {EASY_LABEL} vs {HARD_LABEL}, per dataset (not averaged)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / f"dlt_hard_bars_{group_key}_per_dataset.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (datasets={list(per_dataset)})")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_group_per_dataset("behavioral_surprisal", "Behavioral, surprisal features", use_loglik=True)
    _plot_group_per_dataset("behavioral_embedding", "Behavioral, embedding features")
    _plot_group_per_dataset("neural", "Neural (embedding + surprisal tracks)")


if __name__ == "__main__":
    main()
