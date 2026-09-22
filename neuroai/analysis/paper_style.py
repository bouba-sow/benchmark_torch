"""Shared paper-quality figure style.

One place to control typography, spines, grids and colour so every figure in the
report reads as one system. Import and call :func:`use_paper_style` before
plotting.

Design rules (why, not just what):
  * No chartjunk: top/right spines off, light y-grid only, grid behind the data.
  * Colour encodes MODEL GROUP only; it never encodes anything else, so a colour
    means the same thing in every figure of the paper.
  * Group colours are colourblind-safe (Okabe-Ito): blue / vermillion / green
    are distinguishable under deuteranopia and protanopia.
  * Size encodes a *quantitative* variable (params or perplexity) and is always
    mapped on log10, because both span 3-4 orders of magnitude.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette, reused for the model groups.
GROUP_COLORS_PAPER: dict[str, str] = {
    "LLM": "#0072B2",                 # blue
    "Memory-Constrained": "#D55E00",  # vermillion
    "Constrained-BabyLM": "#CC79A7",  # reddish purple
    "BabyLM": "#009E73",              # bluish green
}

# Neutral accents.
INK = "#222222"
MUTED = "#666666"
FAINT = "#BBBBBB"


def use_paper_style(base_font: float = 10.0) -> None:
    """Apply the shared rcParams. Idempotent; call once per figure script."""
    mpl.rcParams.update({
        # typography
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": base_font,
        "axes.titlesize": base_font + 1,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 1.5,
        "ytick.labelsize": base_font - 1.5,
        "legend.fontsize": base_font - 1.5,
        "figure.titlesize": base_font + 3,
        # axes / spines: drop the box, keep it airy
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,   # was MUTED (#666666) -- darker spines per request
        "axes.linewidth": 0.9,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,      # was MUTED
        "ytick.color": INK,      # was MUTED
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        # grid: light, horizontal, behind the data
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": FAINT,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "axes.axisbelow": True,
        # legend
        "legend.frameon": False,
        "legend.handletextpad": 0.6,
        "legend.borderaxespad": 0.4,
        # output
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def despine(ax) -> None:
    """Extra tidy-up for axes created outside rcParams control."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_label(ax, letter: str, dx: float = -0.08, dy: float = 1.06) -> None:
    """Bold panel letter (a, b, c ...) in figure-panel convention."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right", color=INK)


def log_size_map(values, lo: float, hi: float,
                 s_min: float = 16.0, s_max: float = 330.0):
    """Map values -> marker AREA on a log10 scale (params & perplexity both span
    orders of magnitude, so a linear map would hide almost every model)."""
    import numpy as np
    import pandas as pd

    v = pd.to_numeric(values, errors="coerce")
    v = v.fillna(10 ** lo).clip(lower=1.0)
    lp = np.log10(v)
    norm = (lp - lo) / (hi - lo) if hi > lo else np.full(len(lp), 0.5)
    return s_min + np.clip(np.asarray(norm, dtype=float), 0.0, 1.0) * (s_max - s_min)
