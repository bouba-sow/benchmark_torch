"""Two supporting figures for the neural-trends findings writeup: the
cross-dataset rank-correlation heatmap and the LLM-vs-Memory-Constrained
effect-size bar chart. Run after scripts/analysis/neural_trends.py (reads its
saved CSVs)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

ANALYSIS_DIR = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "outputs" / "benchmark_summary" / "_analysis"
NICE_NAMES = {
    "lpp_fmri": "LPP fMRI", "podcast_ieeg": "Podcast iEEG", "nieuwland_eeg": "Nieuwland EEG",
    "broderick_eeg": "Broderick EEG", "ns_fmri": "NS fMRI",
    "fedorenko2016_ecog": "Fedorenko2016 ECoG", "tuckute_fmri": "Tuckute fMRI",
}


def plot_correlation_heatmap() -> None:
    corr = pd.read_csv(ANALYSIS_DIR / "cross_dataset_rank_correlation.csv", index_col=0)
    labels = [NICE_NAMES.get(c, c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.to_numpy()[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if abs(v) > 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman rho (model score rank)", shrink=0.8)
    ax.set_title("Cross-dataset consistency: do models that rank well on one\nneural dataset also rank well on others?")
    fig.tight_layout()
    out = ANALYSIS_DIR / "cross_dataset_rank_correlation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_cohens_d_bar() -> None:
    ge = pd.read_csv(ANALYSIS_DIR / "group_effects.csv", index_col=0)
    ge = ge.dropna(subset=["llm_vs_mc_cohens_d"]).sort_values("llm_vs_mc_cohens_d")
    labels = [NICE_NAMES.get(d, d) for d in ge.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, ge["llm_vs_mc_cohens_d"], color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axvline(0.8, color="grey", linestyle="--", linewidth=1, label="conventional \"large effect\" (d=0.8)")
    ax.set_xlabel("Cohen's d (LLM mean score - Memory-Constrained mean score)")
    ax.set_title("Standard LLMs outperform Memory-Constrained architectures\nby a large, consistent margin across every neural dataset")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = ANALYSIS_DIR / "llm_vs_mc_cohens_d.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_saturation_bar() -> None:
    sat = pd.read_csv(ANALYSIS_DIR / "scaling_saturation.csv", index_col=0)
    sat = sat.dropna(subset=["saturation_ratio"]).sort_values("saturation_ratio")
    labels = [NICE_NAMES.get(d, d) for d in sat.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d62728" if v > 1 else "#1f77b4" for v in sat["saturation_ratio"]]
    ax.barh(labels, sat["saturation_ratio"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1, label="no saturation (upper-half slope = lower-half slope)")
    ax.set_xlabel("Saturation ratio (upper-half slope / lower-half slope, log-params)")
    ax.set_title("Scaling-law saturation: most datasets show diminishing returns\nfrom the largest LLMs (ratio << 1); ZuCo is a clear outlier")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = ANALYSIS_DIR / "scaling_saturation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_correlation_heatmap()
    plot_cohens_d_bar()
    plot_saturation_bar()
