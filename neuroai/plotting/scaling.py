"""Publication-quality plots for NeuroAI benchmarking.

RT / scaling plots
------------------
plot_scaling_curve      — score vs model size (scaling law)
plot_layerwise_heatmap  — models × layers → score heatmap
plot_inverse_u          — relative-layer score profile

Neural / electrode plots
------------------------
plot_electrode_brain_scores  — electrodes projected on 2-D brain views, colored by score
plot_region_scores           — bar chart of mean score per brain region
plot_hemisphere_comparison   — boxplot comparing left vs right hemispheres
plot_neural_layerwise        — mean neural score vs layer depth (+ std band)
"""

from __future__ import annotations

import typing as tp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEMI_PALETTE = {"L": "#4878d0", "R": "#ee854a", "l": "#4878d0", "r": "#ee854a"}


def _despine_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# RT / scaling plots
# ---------------------------------------------------------------------------

def plot_scaling_curve(
    df: pd.DataFrame,
    *,
    x: str = "n_params",
    y: str = "pearsonr",
    hue: str = "family",
    ci_lo: str | None = None,
    ci_hi: str | None = None,
    title: str = "Scaling law: model size vs brain/RT prediction score",
    xlog: bool = True,
    save_path: str | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot prediction score vs model size (scaling curve).

    Parameters
    ----------
    df : DataFrame
        One row per model. Must contain ``x`` and ``y`` columns.
    x : str
        Column for x-axis (e.g. number of parameters).
    y : str
        Column for y-axis (e.g. Pearson r).
    hue : str
        Column for colour grouping (e.g. model family).
    ci_lo, ci_hi : str, optional
        Columns for lower / upper confidence interval bounds.
    xlog : bool
        Log-scale x-axis.
    save_path : str, optional
        Save figure to this path (PNG/PDF auto-detected).
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hue in df.columns:
        palette = sns.color_palette("tab10", n_colors=df[hue].nunique())
        for (family, group), color in zip(df.groupby(hue), palette):
            group = group.sort_values(x)
            ax.plot(group[x], group[y], marker="o", label=str(family), color=color)
            if ci_lo and ci_hi and ci_lo in group.columns and ci_hi in group.columns:
                ax.fill_between(group[x], group[ci_lo], group[ci_hi],
                                alpha=0.15, color=color)
    else:
        df_sorted = df.sort_values(x)
        ax.plot(df_sorted[x], df_sorted[y], marker="o", color="#4878d0")
        if ci_lo and ci_hi and ci_lo in df_sorted.columns and ci_hi in df_sorted.columns:
            ax.fill_between(df_sorted[x], df_sorted[ci_lo], df_sorted[ci_hi],
                            alpha=0.15, color="#4878d0")

    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel("Model size (parameters)" if x == "n_params" else x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.set_title(title, fontsize=13)
    if hue in df.columns and df[hue].nunique() > 1:
        ax.legend(frameon=False, fontsize=10)
    _despine_ax(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_layerwise_heatmap(
    scores_matrix: np.ndarray,
    model_names: list[str],
    *,
    metric: str = "Pearson r",
    title: str = "Layer-wise prediction score",
    cmap: str = "viridis",
    annotate: bool = False,
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap: models (rows) × layers (columns) → score.

    Parameters
    ----------
    scores_matrix : (n_models, n_layers)
        Score matrix.  NaN = model doesn't have that many layers.
    model_names : list of str
        Row labels.
    annotate : bool
        Write score values inside cells.
    """
    n_models, n_layers = scores_matrix.shape
    if figsize is None:
        figsize = (max(8, n_layers * 0.45), max(3, n_models * 0.65))

    fig, ax = plt.subplots(figsize=figsize)
    vmin = np.nanmin(scores_matrix)
    vmax = np.nanmax(scores_matrix)
    im = ax.imshow(scores_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xticks(range(0, n_layers, max(1, n_layers // 10)))
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, label=metric, fraction=0.03)

    if annotate and n_models * n_layers <= 200:
        for r in range(n_models):
            for c in range(n_layers):
                val = scores_matrix[r, c]
                if np.isfinite(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if val < (vmin + vmax) / 2 else "black")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_inverse_u(
    df: pd.DataFrame,
    *,
    x: str = "layer_rel",
    y: str = "pearsonr",
    hue: str = "model_name",
    title: str = "Layer-relative prediction score",
    save_path: str | None = None,
    figsize: tuple[float, float] = (9, 5),
) -> plt.Figure:
    """Line plot of score vs relative layer depth (0 = embedding, 1 = last layer).

    Parameters
    ----------
    df : DataFrame
        Long format: one row per (model, layer).
    """
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab10", n_colors=df[hue].nunique())
    for (model, group), color in zip(df.groupby(hue), palette):
        group = group.sort_values(x)
        ax.plot(group[x], group[y], label=str(model), color=color)
    ax.set_xlabel("Relative layer depth (0 = embedding, 1 = last layer)", fontsize=11)
    ax.set_ylabel(y, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc="best")
    _despine_ax(ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Neural / electrode plots
# ---------------------------------------------------------------------------

def plot_electrode_brain_scores(
    df: pd.DataFrame,
    *,
    score_col: str = "pearsonr",
    views: tp.Sequence[str] = ("axial", "sagittal"),
    title: str = "Electrode brain scores",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.85,
    point_size: float = 50.0,
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Scatter plot of electrode positions coloured by prediction score.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``x``, ``y``, ``z``, and ``score_col``.
        Rows with missing coordinates are dropped.
    views : sequence of str
        Which projections to show: 'axial' (x–y), 'sagittal' (y–z),
        'coronal' (x–z).  Each view becomes one subplot.
    score_col : str
        Column used to colour the dots.
    cmap : str
        Diverging colourmap for scores (positive = good prediction).
    vmin, vmax : float, optional
        Colour scale limits.  Defaults to symmetric ±max(|score|).
    point_size : float
        Dot size in scatter units.
    """
    df = df.dropna(subset=["x", "y", "z", score_col]).copy()
    if df.empty:
        raise ValueError("No electrodes with complete coordinates + scores.")

    scores = df[score_col].values
    if vmin is None or vmax is None:
        lim = max(abs(np.nanmin(scores)), abs(np.nanmax(scores)))
        vmin = vmin if vmin is not None else -lim
        vmax = vmax if vmax is not None else lim
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    view_axes = {
        "axial":    ("x", "y", "Axial (x–y)"),
        "sagittal": ("y", "z", "Sagittal (y–z)"),
        "coronal":  ("x", "z", "Coronal (x–z)"),
    }
    views = [v for v in views if v in view_axes]
    n_views = len(views)
    if figsize is None:
        figsize = (5 * n_views, 5)

    fig, axes = plt.subplots(1, n_views, figsize=figsize, squeeze=False)
    axes = axes[0]

    sc = None
    for ax, view in zip(axes, views):
        xc, yc, view_title = view_axes[view]
        sc = ax.scatter(
            df[xc], df[yc],
            c=scores, cmap=cmap, norm=norm,
            s=point_size, alpha=alpha, linewidths=0.3, edgecolors="k",
        )
        ax.set_xlabel(xc + " (mm)", fontsize=10)
        ax.set_ylabel(yc + " (mm)", fontsize=10)
        ax.set_title(view_title, fontsize=11)
        ax.set_aspect("equal")
        _despine_ax(ax)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes[-1], fraction=0.04, pad=0.04)
        cbar.set_label(score_col, fontsize=10)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_region_scores(
    df: pd.DataFrame,
    *,
    region_col: str = "region",
    score_col: str = "pearsonr",
    top_n: int | None = 20,
    sort_by: str = "mean",
    title: str = "Brain score by region",
    palette: str | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Horizontal bar chart of mean prediction score per brain region.

    Parameters
    ----------
    df : DataFrame
        One row per electrode. Must contain ``region_col`` and ``score_col``.
    top_n : int, optional
        Only show the top N regions (by mean score).
    sort_by : str
        'mean' = sort by mean score, 'count' = sort by electrode count.
    """
    df = df.dropna(subset=[region_col, score_col]).copy()
    if df.empty:
        raise ValueError("No rows with region + score data.")

    agg = (
        df.groupby(region_col)[score_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={region_col: "region"})
    )
    agg = agg.sort_values(sort_by, ascending=False)
    if top_n:
        agg = agg.head(top_n)

    n_regions = len(agg)
    if figsize is None:
        figsize = (8, max(4, n_regions * 0.35))

    fig, ax = plt.subplots(figsize=figsize)
    pal = palette or sns.color_palette("mako_r", n_colors=n_regions)
    bars = ax.barh(
        agg["region"],
        agg["mean"],
        xerr=agg["std"],
        color=pal if not isinstance(pal, str) else sns.color_palette(pal, n_regions),
        capsize=3,
        height=0.6,
        error_kw={"elinewidth": 1, "alpha": 0.7},
    )
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel(score_col, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.invert_yaxis()
    _despine_ax(ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_hemisphere_comparison(
    df: pd.DataFrame,
    *,
    hemi_col: str = "hemisphere",
    score_col: str = "pearsonr",
    title: str = "Brain score: left vs right hemisphere",
    palette: dict | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Boxplot + strip comparing electrode scores between hemispheres.

    Parameters
    ----------
    df : DataFrame
        One row per electrode. Must contain ``hemi_col`` and ``score_col``.
    palette : dict, optional
        Colour map {hemisphere_label: colour}.
    """
    df = df.dropna(subset=[hemi_col, score_col]).copy()
    if df.empty:
        raise ValueError("No rows with hemisphere + score data.")

    pal = palette or _HEMI_PALETTE
    hemis = sorted(df[hemi_col].unique())

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df, x=hemi_col, y=score_col,
        hue=hemi_col, order=hemis,
        palette={h: pal.get(h, "#999999") for h in hemis},
        width=0.45, linewidth=1.2, fliersize=0, legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df, x=hemi_col, y=score_col,
        hue=hemi_col, order=hemis,
        palette={h: pal.get(h, "#999999") for h in hemis},
        size=4, alpha=0.5, jitter=True, dodge=False, legend=False,
        ax=ax,
    )
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Hemisphere", fontsize=11)
    ax.set_ylabel(score_col, fontsize=11)
    ax.set_title(title, fontsize=12)
    _despine_ax(ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_neural_layerwise(
    layerwise_matrix: np.ndarray,
    *,
    model_name: str = "",
    metric: str = "Pearson r",
    title: str | None = None,
    show_std: bool = True,
    electrode_traces: bool = False,
    save_path: str | None = None,
    figsize: tuple[float, float] = (9, 5),
) -> plt.Figure:
    """Mean (± std) neural brain score across electrodes vs layer index.

    Parameters
    ----------
    layerwise_matrix : (n_electrodes, n_layers) ndarray
        Pearson r per electrode × layer.  NaN values are ignored.
    show_std : bool
        Shade ±1 SD band around the mean.
    electrode_traces : bool
        Draw individual electrode lines (semi-transparent).
    """
    if title is None:
        title = f"Neural brain score by layer — {model_name}" if model_name else \
                "Neural brain score by layer"

    n_electrodes, n_layers = layerwise_matrix.shape
    layers = np.arange(n_layers)
    mean_scores = np.nanmean(layerwise_matrix, axis=0)
    std_scores = np.nanstd(layerwise_matrix, axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    if electrode_traces:
        for row in layerwise_matrix:
            ax.plot(layers, row, color="#4878d0", alpha=0.08, linewidth=0.7)

    ax.plot(layers, mean_scores, color="#4878d0", linewidth=2.0,
            label="Mean across electrodes")
    if show_std:
        ax.fill_between(layers, mean_scores - std_scores, mean_scores + std_scores,
                        color="#4878d0", alpha=0.18, label="±1 SD")

    best_layer = int(np.nanargmax(mean_scores))
    ax.axvline(best_layer, color="crimson", linestyle="--", linewidth=1.2,
               label=f"Best layer ({best_layer}): r={mean_scores[best_layer]:.3f}")

    ax.axhline(0, color="k", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _despine_ax(ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_neural_time_course(
    time_courses: "list | np.ndarray",
    window_centers: np.ndarray | None = None,
    *,
    model_name: str = "",
    metric: str = "spearmanr",
    metric_label: str | None = None,
    show_std: bool = True,
    electrode_traces: bool = False,
    highlight_peak: bool = True,
    onset_line: bool = True,
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Plot mean (± std) neural encoding score across time windows.

    Visualises the temporal profile of encoding scores produced by
    ``cv_ridge_score_windowed``, averaged across electrodes.

    Parameters
    ----------
    time_courses : list of ``TimeResolvedScore`` or (n_electrodes, n_windows) ndarray
        Temporal encoding profiles to average.  Pass ``result.electrode_scores``
        time_course objects, e.g.::

            tcs = [e.time_course for e in result.electrode_scores
                   if e.time_course is not None]
            plot_neural_time_course(tcs)

    window_centers : (n_windows,) ndarray
        Time axis (seconds relative to event onset).  Required when
        ``time_courses`` is an ndarray; inferred automatically from
        ``TimeResolvedScore`` objects.
    metric : str
        Score to plot: "spearmanr" (default), "pearsonr", or "r2".
    metric_label : str | None
        Y-axis label.  Defaults to ``metric``.
    show_std : bool
        Shade ±1 SD band across electrodes.
    electrode_traces : bool
        Draw individual electrode traces (semi-transparent).
    highlight_peak : bool
        Add a vertical dashed line at the mean-curve peak.
    onset_line : bool
        Add a vertical line at t=0 (event onset).
    """
    if hasattr(time_courses[0], metric):
        scores_mat = np.stack(
            [getattr(tc, metric) for tc in time_courses], axis=0
        )
        if window_centers is None:
            window_centers = time_courses[0].window_centers
    else:
        scores_mat = np.asarray(time_courses)
        if window_centers is None:
            raise ValueError(
                "window_centers must be provided when time_courses is a plain ndarray"
            )

    mean_scores = np.nanmean(scores_mat, axis=0)
    std_scores = np.nanstd(scores_mat, axis=0)

    if title is None:
        title = (
            f"Neural time-resolved encoding — {model_name}"
            if model_name else "Neural time-resolved encoding score"
        )
    if metric_label is None:
        metric_label = metric

    fig, ax = plt.subplots(figsize=figsize)

    if electrode_traces:
        for row in scores_mat:
            ax.plot(window_centers, row, color="#4878d0", alpha=0.07, linewidth=0.6)

    ax.plot(window_centers, mean_scores, color="#4878d0", linewidth=2.0,
            label="Mean across electrodes")
    if show_std:
        ax.fill_between(
            window_centers,
            mean_scores - std_scores,
            mean_scores + std_scores,
            color="#4878d0", alpha=0.18, label="±1 SD",
        )

    if onset_line:
        ax.axvline(0, color="k", linestyle="-", linewidth=1.0, alpha=0.6,
                   label="Word onset (t=0)")

    if highlight_peak and np.any(np.isfinite(mean_scores)):
        peak_idx = int(np.nanargmax(mean_scores))
        peak_t = float(window_centers[peak_idx])
        peak_v = float(mean_scores[peak_idx])
        ax.axvline(peak_t, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Peak t={peak_t:.2f}s ({metric_label}={peak_v:.3f})")

    ax.axhline(0, color="k", linewidth=0.7, linestyle=":", alpha=0.4)
    ax.set_xlabel("Time relative to word onset (s)", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _despine_ax(ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
