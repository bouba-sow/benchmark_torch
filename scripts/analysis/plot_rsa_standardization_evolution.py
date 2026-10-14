#!/usr/bin/env python3
"""Dumbbell figure: r(RSA, ridge) and r(RSA, log params) before vs after feature standardization."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
R = Path(__file__).resolve().parents[2] / "reports/rsa_summary/rsa"
d = pd.read_csv(R / "standardization_evolution.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), sharey=True)
y = np.arange(len(d))[::-1]
for ax, (a, b, title) in zip(axes, [("r_ridge_orig", "r_ridge_final", "Agreement with ridge score\n(r across models)"),
                                    ("r_size_orig", "r_size_final", "Trend with model size\n(r with log10 params)")]):
    for yi, o, f in zip(y, d[a], d[b]):
        ax.plot([o, f], [yi, yi], color="#bbbbbb", lw=1.5, zorder=1)
    ax.scatter(d[a], y, s=28, color="#888888", label="original", zorder=2)
    ax.scatter(d[b], y, s=28, color="#2166ac", label="standardized", zorder=3)
    ax.axvline(0, color="#dddddd", lw=.8)
    ax.set_title(title, fontsize=11); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.15)
axes[0].set_yticks(y); axes[0].set_yticklabels(d.dataset, fontsize=8)
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("Embedding RSA before vs after feature standardization", fontsize=13)
fig.tight_layout()
for sfx in ("png", "pdf"):
    fig.savefig(R / f"standardization_evolution.{sfx}", dpi=180, facecolor="white")
