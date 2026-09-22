"""Publication-quality plots for LPP fMRI checkpoint sweep experiments.

Functions
---------
plot_lpp_training_curve
    Brain score vs training step / tokens seen for one model.
plot_lpp_lateralization_curve
    Left–right hemisphere asymmetry (brain-score difference) vs training.
plot_lpp_layer_heatmap
    Heatmap of per-layer brain scores across training checkpoints.
plot_lpp_all_models_panel
    Multi-model comparison panel (brain score vs tokens seen).
plot_lpp_lateralization_panel
    Multi-model lateralization panel.
"""

from __future__ import annotations

import typing as tp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Colour palette (matching run_checkpoint_sweep.py style)
# ---------------------------------------------------------------------------

# Colorblind-friendly, high contrast across Pythia sizes (1B / 1.4B / 6.9B / …)
MODEL_SCALE_COLORS = (
    "#0072B2",  # blue — smallest
    "#E69F00",  # amber
    "#009E73",  # green
    "#CC79A7",  # pink
    "#D55E00",  # vermillion
    "#56B4E9",  # sky
)

_PALETTE = sns.color_palette("tab10")
_MODEL_COLORS: dict[str, tuple] = {}


def model_color_map(
    df: pd.DataFrame,
    *,
    model_col: str = "model_label",
    size_col: str = "n_params_b",
) -> dict[str, tuple]:
    """Stable model → color mapping sorted by parameter count (smallest first)."""
    if model_col not in df.columns:
        labels = sorted(df.index.unique()) if hasattr(df.index, "unique") else []
        return {lbl: MODEL_SCALE_COLORS[i % len(MODEL_SCALE_COLORS)] for i, lbl in enumerate(labels)}
    order_df = df[[model_col, size_col]].drop_duplicates() if size_col in df.columns else df[[model_col]]
    if size_col in order_df.columns:
        order_df = order_df.sort_values(size_col)
    else:
        order_df = order_df.sort_values(model_col)
    labels = order_df[model_col].tolist()
    return {
        lbl: MODEL_SCALE_COLORS[i % len(MODEL_SCALE_COLORS)]
        for i, lbl in enumerate(labels)
    }


def _model_color(model_label: str, all_labels: list[str]) -> tuple:
    if model_label not in _MODEL_COLORS:
        idx = all_labels.index(model_label) % len(_PALETTE)
        _MODEL_COLORS[model_label] = _PALETTE[idx]
    return _MODEL_COLORS[model_label]


# ---------------------------------------------------------------------------
# Axis formatters
# ---------------------------------------------------------------------------

def _tokens_formatter(x: float, _: object) -> str:
    """Human-readable token count: 1B, 42B, etc."""
    if x == 0:
        return "0"
    b = x / 1e9
    if b >= 1:
        return f"{b:.0f}B" if b == int(b) else f"{b:.1f}B"
    m = x / 1e6
    return f"{m:.0f}M" if m >= 1 else f"{x:.0f}"


def _step_formatter(x: float, _: object) -> str:
    if x == 0:
        return "step0"
    if x >= 1000:
        return f"{int(x // 1000)}k"
    return str(int(x))


# ---------------------------------------------------------------------------
# Training curve (single model)
# ---------------------------------------------------------------------------

def plot_lpp_training_curve(
    df: pd.DataFrame,
    *,
    x_col: str = "tokens_seen",
    y_col: str = "mean_pearson",
    hue_col: str | None = "lang",
    model_label: str = "",
    xlabel: str | None = None,
    ylabel: str = "Mean Pearson r (voxelwise)",
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Plot brain score (mean Pearson r) vs training progress for one model.

    Parameters
    ----------
    df
        DataFrame with columns including ``x_col``, ``y_col``, and optionally
        ``hue_col`` (e.g. ``"lang"``).
    x_col
        Column to use as x-axis.  Typical values: ``"tokens_seen"``,
        ``"step"``, ``"tokens_seen_b"`` (billions).
    y_col
        Column with the brain score.
    hue_col
        Optional column for colour grouping (e.g. language).
    """
    fig, ax = plt.subplots(figsize=figsize)

    groups = df[hue_col].unique().tolist() if hue_col and hue_col in df.columns else [None]
    colors = sns.color_palette("tab10", n_colors=max(len(groups), 1))

    for color, grp_val in zip(colors, groups):
        subset = df[df[hue_col] == grp_val] if grp_val is not None else df
        subset = subset.dropna(subset=[x_col, y_col]).sort_values(x_col)
        label = str(grp_val) if grp_val is not None else model_label
        ax.plot(
            subset[x_col], subset[y_col],
            marker="o", markersize=5, linewidth=1.8,
            color=color, label=label,
        )
        if "std_pearson" in subset.columns:
            ax.fill_between(
                subset[x_col],
                subset[y_col] - subset["std_pearson"],
                subset[y_col] + subset["std_pearson"],
                alpha=0.15, color=color,
            )

    if x_col in ("tokens_seen", "tokens_seen_b"):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_formatter))
    elif "step" in x_col:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_step_formatter))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"LPP brain score — {model_label}")
    if len(groups) > 1 or (hue_col is not None):
        ax.legend(framealpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Lateralization curve (single model)
# ---------------------------------------------------------------------------

def plot_lpp_lateralization_curve(
    df: pd.DataFrame,
    *,
    x_col: str = "tokens_seen",
    lat_col: str = "lateralization_index",
    left_col: str = "mean_pearson_left",
    right_col: str = "mean_pearson_right",
    model_label: str = "",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 4),
) -> plt.Figure:
    """Two-panel figure: brain scores (left/right hemispheres) + asymmetry index.

    Parameters
    ----------
    df
        DataFrame with columns ``x_col``, ``left_col``, ``right_col``,
        ``lat_col``.
    """
    has_lat = lat_col in df.columns and not df[lat_col].isna().all()
    fig, axes = plt.subplots(1, 2 if has_lat else 1, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    df_sorted = df.dropna(subset=[x_col]).sort_values(x_col)

    # Left panel: L and R hemisphere scores
    ax = axes[0]
    for col, lbl, color in [
        (left_col, "Left", "#2166ac"),
        (right_col, "Right", "#d6604d"),
    ]:
        if col in df_sorted.columns:
            sub = df_sorted.dropna(subset=[col])
            ax.plot(sub[x_col], sub[col], marker="o", markersize=5,
                    linewidth=1.8, color=color, label=lbl)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Mean Pearson r")
    ax.set_title(f"L/R brain scores — {model_label}")
    ax.legend(framealpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")

    if x_col in ("tokens_seen",):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_formatter))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Right panel: asymmetry
    if has_lat:
        ax2 = axes[1]
        sub = df_sorted.dropna(subset=[lat_col])
        ax2.plot(sub[x_col], sub[lat_col], marker="o", markersize=5,
                 linewidth=1.8, color="#4d9221", label="L − R")
        ax2.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax2.set_xlabel(x_col)
        ax2.set_ylabel("Lateralization index (L − R)")
        ax2.set_title(f"Asymmetry — {model_label}")
        if x_col in ("tokens_seen",):
            ax2.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_formatter))
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Layer heatmap (single model, layers × checkpoints)
# ---------------------------------------------------------------------------

def plot_lpp_layer_heatmap(
    df: pd.DataFrame,
    *,
    layer_col: str = "layer",
    x_col: str = "tokens_seen",
    score_col: str = "mean_pearson",
    model_label: str = "",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 5),
    cmap: str = "RdYlBu_r",
) -> plt.Figure:
    """Heatmap of brain scores across layers (y) and training steps (x).

    Parameters
    ----------
    df
        Tidy DataFrame with one row per (layer, checkpoint) combination.
        Must contain ``layer_col``, ``x_col``, ``score_col``.
    """
    pivot = df.pivot_table(index=layer_col, columns=x_col, values=score_col)
    pivot = pivot.sort_index(ascending=False)

    # Relabel columns to human-readable token counts
    col_labels = [_tokens_formatter(c, None) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=figsize)
    vmax = float(np.nanpercentile(pivot.values, 98)) if np.any(np.isfinite(pivot.values)) else 0.1
    vmin = 0.0
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=col_labels,
        cbar_kws={"label": "Mean Pearson r"},
    )
    ax.set_xlabel("Tokens seen (training)")
    ax.set_ylabel("Layer index")
    ax.set_title(f"LPP layer × checkpoint heatmap — {model_label}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Multi-model comparison panels
# ---------------------------------------------------------------------------

def plot_lpp_all_models_panel(
    summary_df: pd.DataFrame,
    *,
    x_col: str = "tokens_seen",
    y_col: str = "mean_pearson",
    model_col: str = "model_label",
    hue_col: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 5),
    title: str = "LPP brain scores — all models",
) -> plt.Figure:
    """Overlay training curves for all models on a single axes.

    Parameters
    ----------
    summary_df
        Combined DataFrame (all models / checkpoints / optional languages).
        Must contain ``x_col``, ``y_col``, ``model_col``.
    """
    fig, ax = plt.subplots(figsize=figsize)

    color_map = model_color_map(summary_df, model_col=model_col)
    order_df = summary_df[[model_col, "n_params_b"]].drop_duplicates()
    if "n_params_b" in order_df.columns:
        all_models = order_df.sort_values("n_params_b")[model_col].tolist()
    else:
        all_models = sorted(summary_df[model_col].dropna().unique().tolist())

    for model_lbl in all_models:
        sub = summary_df[summary_df[model_col] == model_lbl].sort_values(x_col)
        if sub.empty:
            continue
        if hue_col and hue_col in sub.columns:
            for lang_val in sub[hue_col].unique():
                subsub = sub[sub[hue_col] == lang_val]
                ax.plot(
                    subsub[x_col], subsub[y_col],
                    marker="o", markersize=5, linewidth=2.0,
                    color=color_map[model_lbl],
                    linestyle="--" if str(lang_val) == "fr" else "-",
                    label=f"{model_lbl} ({lang_val})",
                )
        else:
            ax.plot(
                sub[x_col], sub[y_col],
                marker="o", markersize=6, linewidth=2.2,
                color=color_map[model_lbl], label=model_lbl,
            )

    if x_col in ("tokens_seen", "tokens_seen_b"):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_formatter))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Tokens seen (pretraining, log scale)" if x_col in ("tokens_seen",) else x_col)
    ax.set_ylabel("Mean Pearson r (voxelwise)")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.85, fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_lpp_lateralization_panel(
    summary_df: pd.DataFrame,
    *,
    x_col: str = "tokens_seen",
    lat_col: str = "lateralization_index",
    model_col: str = "model_label",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 5),
    title: str = "LPP lateralization index — all models",
) -> plt.Figure:
    """Plot L−R hemisphere asymmetry vs training for all models."""
    fig, ax = plt.subplots(figsize=figsize)

    color_map = model_color_map(summary_df, model_col=model_col)
    order_df = summary_df[[model_col, "n_params_b"]].drop_duplicates()
    if "n_params_b" in order_df.columns:
        all_models = order_df.sort_values("n_params_b")[model_col].tolist()
    else:
        all_models = sorted(summary_df[model_col].dropna().unique().tolist())

    for model_lbl in all_models:
        sub = summary_df[summary_df[model_col] == model_lbl].dropna(subset=[lat_col]).sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(
            sub[x_col], sub[lat_col],
            marker="o", markersize=6, linewidth=2.2,
            color=color_map[model_lbl], label=model_lbl,
        )

    if x_col in ("tokens_seen", "tokens_seen_b"):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_formatter))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Tokens seen (pretraining, log scale)" if x_col in ("tokens_seen",) else x_col)
    ax.set_ylabel("Lateralization index (L − R)")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.85, fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_lpp_layer_heatmap_grid(
    heatmap_data: dict[str, pd.DataFrame],
    *,
    layer_col: str = "layer",
    x_col: str = "tokens_seen",
    score_col: str = "mean_pearson",
    save_path: str | Path | None = None,
    figsize_per: tuple[float, float] = (5, 3),
    cmap: str = "RdYlBu_r",
) -> plt.Figure:
    """Grid of layer heatmaps, one subplot per model in ``heatmap_data``."""
    n = len(heatmap_data)
    if n == 0:
        fig, ax = plt.subplots(1, 1)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False,
    )

    items = list(heatmap_data.items())
    for idx, (model_lbl, df) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        try:
            pivot = df.pivot_table(index=layer_col, columns=x_col, values=score_col)
            pivot = pivot.sort_index(ascending=False)
            col_labels = [_tokens_formatter(c, None) for c in pivot.columns]
            vmax = float(np.nanpercentile(pivot.values, 98)) if np.any(np.isfinite(pivot.values)) else 0.1
            sns.heatmap(
                pivot, ax=ax, cmap=cmap, vmin=0.0, vmax=vmax,
                xticklabels=col_labels, cbar=True,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(model_lbl, fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("Layer")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
        except Exception:  # noqa: BLE001
            ax.text(0.5, 0.5, f"{model_lbl}\n(no data)", ha="center", va="center")

    # Hide unused axes
    for idx in range(len(items), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


__all__ = [
    "plot_lpp_all_models_panel",
    "plot_lpp_lateralization_curve",
    "plot_lpp_lateralization_panel",
    "plot_lpp_layer_heatmap",
    "plot_lpp_layer_heatmap_grid",
    "plot_lpp_training_curve",
]
