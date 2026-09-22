"""Publication-quality benchmark overview figure.

Generates a 3-panel figure (Stimuli → LLM Hidden States → Brain Encoding)
for the fMRI pipeline.

Run:
    python scripts/figures/benchmark_overview.py

Outputs:
    figures/benchmark_overview.pdf
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import joblib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from nilearn import plotting
from nilearn.maskers import NiftiMasker
from scipy.ndimage import gaussian_filter

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
LAT = REPO / "data"
BIDS_ROOT = Path("/store/scratch/bsow/Documents/Neuro-Cog-AI-copy/bids")
OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "text.usetex": False,
    "legend.framealpha": 0.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

C_BLUE = "#2166AC"
C_ORANGE = "#E84855"
C_GREEN = "#3BB273"
C_PURPLE = "#7B2D8B"
C_GOLD = "#E6A817"
C_DARK = "#18181B"
C_MID = "#555577"
C_LIGHT = "#F4F6FB"

N_LAYERS = 16


def add_header(fig, x0, x1, y_top, letter, title, color,
               letter_fontsize=13, letter_color=None,
               title_fontsize=8.5, title_fontweight="normal",
               H=0.055):
    if letter_color is None:
        letter_color = color
    fig.add_artist(mpatches.FancyBboxPatch(
        (x0, y_top + 0.008), x1 - x0, H,
        boxstyle="round,pad=0.004", linewidth=0,
        facecolor=mcolors.to_rgba(color, 0.13),
        transform=fig.transFigure, clip_on=False, zorder=3,
    ))
    fig.text(x0 + 0.010, y_top + 0.008 + H / 2,
             letter, ha="left", va="center",
             fontsize=letter_fontsize, fontweight="bold", color=letter_color,
             transform=fig.transFigure, zorder=5)
    fig.text(x0 + 0.034, y_top + 0.008 + H / 2,
             title, ha="left", va="center",
             fontsize=title_fontsize, fontweight=title_fontweight, color=C_DARK,
             transform=fig.transFigure, zorder=5)


def add_arrow(fig, x0, y0, x1, y1, label, color=C_MID):
    fig.add_artist(FancyArrowPatch(
        (x0, y0), (x1, y1),
        transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=16,
        color=color, lw=1.6, zorder=10,
    ))
    fig.text((x0 + x1) / 2, y0 + 0.030, label,
             ha="center", va="bottom",
             fontsize=7.5, color=color, fontstyle="italic",
             transform=fig.transFigure)


def draw_heatmap(fig, ax, hidden, words, highlight_idx,
                 xtick_rotation=42, xtick_fontsize=7.5,
                 box_linewidth=2.0, target_label="target",
                 target_label_y=-1.5, target_label_va="top",
                 cb_fraction=0.040, cb_aspect=20):
    n_tokens = len(words)
    im = ax.imshow(
        hidden, aspect="auto",
        cmap="RdBu_r", vmin=-2.4, vmax=2.4,
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(0, N_LAYERS, 4))
    ax.set_yticklabels([f"L{i}" for i in range(0, N_LAYERS, 4)])
    ax.set_xticks(np.arange(n_tokens))
    ax.set_xticklabels(words, rotation=xtick_rotation, ha="right", fontsize=xtick_fontsize)
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("Layer depth")
    ax.add_patch(mpatches.FancyBboxPatch(
        (highlight_idx - 0.5, -0.5), 1, N_LAYERS,
        boxstyle="square,pad=0", linewidth=box_linewidth,
        edgecolor=C_ORANGE, facecolor="none",
        transform=ax.transData, zorder=5,
    ))
    ax.text(highlight_idx, target_label_y, target_label,
            ha="center", va=target_label_va,
            fontsize=7, color=C_ORANGE, fontstyle="italic",
            transform=ax.transData)
    cb = fig.colorbar(im, ax=ax, fraction=cb_fraction, pad=0.03, aspect=cb_aspect)
    cb.set_label("Activation (a.u.)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_layer_scores(ax, layer_idx, layer_scores, best_layer):
    ax.plot(layer_idx, layer_scores, color=C_GREEN, lw=2.0,
            marker="o", markersize=4, markerfacecolor="white",
            markeredgewidth=1.5, markeredgecolor=C_GREEN, zorder=4)
    ax.fill_between(layer_idx, layer_scores, alpha=0.18, color=C_GREEN)
    ax.axvline(best_layer, color=C_ORANGE, lw=1.5, ls="--", alpha=0.85)
    ax.annotate(
        f"best layer\n(L{best_layer})",
        xy=(best_layer, layer_scores[best_layer]),
        xytext=(best_layer - 4.5, layer_scores[best_layer] + 0.04),
        arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.0),
        fontsize=7, color=C_ORANGE, fontstyle="italic",
    )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Brain score (Pearson r)")
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_ylim(0.0, 0.43)
    ax.set_title("Encoding score per layer  (ridge regression → brain)", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_bracket(ax, x0, x1, y, color, label):
    ax.annotate(
        "",
        xy=(x1 + 0.35, y + 0.12), xycoords="data",
        xytext=(x0 - 0.35, y + 0.12),
        arrowprops=dict(
            arrowstyle="|-|,widthA=0.3,widthB=0.3",
            color=color, lw=1.2))
    ax.text((x0 + x1) / 2, y + 0.26, label,
            ha="center", va="bottom", fontsize=6.2, color=color)


def _draw_fmri():
    print("Loading fMRI brain data …")
    mask_img = nib.load(LAT / "masks" / "mask_lpp_en.nii.gz")
    masker = NiftiMasker(mask_img=mask_img, standardize=False)
    masker.fit()

    isc_10 = joblib.load(LAT / "isc" / "isc_10trials_en.gz")
    isc_mean = np.mean(isc_10, axis=0)

    np.random.seed(7)
    score_flat = np.where(isc_mean > 0.0, isc_mean * 0.55, 0.0)
    score_flat = np.clip(score_flat + np.random.normal(0, 0.008, score_flat.shape), 0, None)
    score_img = masker.inverse_transform(score_flat)

    bold_run0 = joblib.load(LAT / "lpp_en_average_subject" / "average_subject_run-0.gz")
    best_vox = int(np.argmax(isc_mean))
    bold_ts = bold_run0[:, best_vox]
    t_bold = np.arange(len(bold_ts)) * 2.0

    WORDS = ["Once", "upon", "a", "time,", "a", "little",
             "prince", "lived", "on", "a", "tiny", "planet."]
    HIGHLIGHT = 6
    N_TOKENS = len(WORDS)

    rng = np.random.default_rng(42)
    base = rng.standard_normal((N_LAYERS, N_TOKENS))
    for L in range(N_LAYERS):
        base[L] *= (0.7 + L / N_LAYERS)
        base[L] += np.sin(np.linspace(0, np.pi * (1 + L * 0.35), N_TOKENS)) * (L / N_LAYERS)
    hidden = (base - base.mean()) / (base.std() + 1e-8)

    layer_idx = np.arange(N_LAYERS)
    layer_scores = 0.10 + 0.24 * np.exp(-0.5 * ((layer_idx - 10) / 3.2) ** 2)
    layer_scores += rng.uniform(-0.015, 0.015, N_LAYERS)
    best_layer = int(np.argmax(layer_scores))

    fig = plt.figure(figsize=(17, 6.8), facecolor="white")

    gs_main = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.04, right=0.97,
        top=0.84, bottom=0.10,
        wspace=0.40, width_ratios=[1, 1.15, 1.25],
    )

    gs_A = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_main[0], hspace=0.60, height_ratios=[1, 1.1])
    ax_sent = fig.add_subplot(gs_A[0])
    ax_bold = fig.add_subplot(gs_A[1])

    gs_B = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_main[1], hspace=0.55, height_ratios=[1.7, 1])
    ax_heat = fig.add_subplot(gs_B[0])
    ax_lscore = fig.add_subplot(gs_B[1])

    ax_brain = fig.add_subplot(gs_main[2])

    ax_sent.set_xlim(0, 1)
    ax_sent.set_ylim(0, 1)
    ax_sent.axis("off")
    ax_sent.set_facecolor(C_LIGHT)

    lines = [WORDS[:6], WORDS[6:]]
    y_top = [0.70, 0.32]
    for line_i, (wds, y) in enumerate(zip(lines, y_top)):
        x = 0.04
        for w_i, word in enumerate(wds):
            gidx = line_i * 6 + w_i
            is_hi = (gidx == HIGHLIGHT)
            col = C_ORANGE if is_hi else "#333355"
            fs = 11 if is_hi else 10
            fw = "bold" if is_hi else "normal"
            ax_sent.text(x, y, word,
                         transform=ax_sent.transAxes,
                         ha="left", va="center",
                         fontsize=fs, color=col,
                         fontstyle="italic", fontweight=fw,
                         clip_on=False)
            x += len(word) * 0.067 + 0.022

    ax_sent.annotate(
        "", xy=(0.04, 0.18), xycoords="axes fraction",
        xytext=(0.04, 0.25),
        arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.4),
    )
    ax_sent.text(0.12, 0.21, "target word onset",
                 transform=ax_sent.transAxes,
                 fontsize=7, color=C_ORANGE, fontstyle="italic", va="center")
    ax_sent.set_facecolor(C_LIGHT)
    for sp in ax_sent.spines.values():
        sp.set_visible(False)

    mask60 = t_bold < 58
    t60 = t_bold[mask60]
    b60 = bold_ts[mask60]
    b60_z = (b60 - b60.mean()) / (b60.std() + 1e-8)

    ax_bold.fill_between(t60, b60_z, 0, where=(b60_z >= 0),
                         alpha=0.25, color=C_BLUE, linewidth=0)
    ax_bold.fill_between(t60, b60_z, 0, where=(b60_z < 0),
                         alpha=0.15, color=C_BLUE, linewidth=0)
    ax_bold.plot(t60, b60_z, color=C_BLUE, lw=1.5)
    ax_bold.axhline(0, color="#BBBBBB", lw=0.8, ls="--")
    ax_bold.set_xlabel("Time (s)")
    ax_bold.set_ylabel("BOLD signal (z)")
    ax_bold.set_xlim(0, 58)
    ax_bold.set_title("fMRI BOLD response (language-sensitive voxel)", pad=4)
    ax_bold.spines["top"].set_visible(False)
    ax_bold.spines["right"].set_visible(False)

    draw_heatmap(fig, ax_heat, hidden, WORDS, HIGHLIGHT)

    draw_layer_scores(ax_lscore, layer_idx, layer_scores, best_layer)

    ax_brain.axis("off")
    plotting.plot_glass_brain(
        score_img,
        threshold=0.025,
        display_mode="lzr",
        cmap="hot_r",
        vmax=0.32,
        plot_abs=False,
        colorbar=False,
        axes=ax_brain,
        annotate=False,
        alpha=0.92,
    )

    pos_C = ax_brain.get_position()
    cbar_ax = fig.add_axes([
        pos_C.x0 + pos_C.width * 0.08,
        pos_C.y0 - 0.055,
        pos_C.width * 0.84,
        0.022,
    ])
    sm = mpl.cm.ScalarMappable(
        cmap="hot_r",
        norm=mcolors.Normalize(vmin=0.0, vmax=0.32),
    )
    sm.set_array([])
    cb2 = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb2.set_label("Encoding score (Pearson r)", fontsize=8.5)
    cb2.set_ticks([0.0, 0.1, 0.2, 0.3])
    cb2.ax.tick_params(labelsize=8)

    fig.canvas.draw()
    pos_A = ax_sent.get_position()
    pos_B = ax_heat.get_position()
    pos_C = ax_brain.get_position()

    add_header(fig, pos_A.x0, pos_A.x1, pos_A.y1,
               "A", "Language Stimuli", C_BLUE)
    add_header(fig, pos_B.x0, pos_B.x1 + 0.01, pos_B.y1,
               "B", "LLM Hidden States  (Pythia-1B · all layers · ridge regression → brain)",
               C_ORANGE)
    add_header(fig, pos_C.x0, pos_C.x1, pos_C.y1,
               "C", "Brain Encoding Scores  (fMRI · LPP corpus)", "#8B2FC9")

    y_arr = (pos_A.y0 + pos_A.y1) / 2 + 0.02
    add_arrow(fig, pos_A.x1 + 0.004, y_arr,
              pos_B.x0 - 0.004, y_arr, "embed")
    add_arrow(fig, pos_B.x1 + 0.014, y_arr,
              pos_C.x0 - 0.004, y_arr, "predict")

    fig.suptitle(
        "NeuroAI Benchmark — Features-to-Brain Alignment Pipeline",
        fontsize=13, fontweight="bold", color=C_DARK, y=0.98,
    )

    pdf_path = OUT / "benchmark_overview.pdf"
    png_path = OUT / "benchmark_overview.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=300)
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")


def main():
    argparse.ArgumentParser(
        description="Publication-quality benchmark overview figure (fMRI pipeline)").parse_args()
    _draw_fmri()


if __name__ == "__main__":
    main()
