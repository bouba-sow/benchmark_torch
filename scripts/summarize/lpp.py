#!/usr/bin/env python3
"""Summarize LPP checkpoint sweep results.

Subcommands:
  roi      -- step-wise + layer-wise plots for one model's ROI CSV
  parcels  -- Harvard-Oxford parcel L/R plots for one model
  models   -- cross-model scaling comparison across a sweep directory

Usage::

    python scripts/summarize/lpp.py roi \\
        --model_dir outputs/lpp/embedding_sweep/EleutherAI_pythia-1b

    python scripts/summarize/lpp.py parcels \\
        --model_dir outputs/lpp/embedding_sweep/EleutherAI_pythia-1b

    python scripts/summarize/lpp.py models \\
        --sweep_dir outputs/lpp/embedding_sweep
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats

from neuroai.roi.lpp import load_harvard_oxford_parcelization, parcel_summary_from_correlations
from neuroai.plotting.lpp import (
    model_color_map,
    plot_lpp_all_models_panel,
    plot_lpp_lateralization_panel,
    plot_lpp_layer_heatmap_grid,
)

# ---------------------------------------------------------------------------
# ROI constants
# ---------------------------------------------------------------------------

ROI_ORDER = ["TP", "aSTS", "pSTS", "BA44", "BA45", "BA47", "AG_TPJ"]
ROI_SOURCE = (
    "llms_brain_lateralization/roi_masks/*.nii.gz intersected with mask_lpp_en.nii.gz; "
    "coordinates from Pallier et al. 2011 / Zaccarella et al. 2017 / Price et al. 2015"
)

LANG_PARCEL_KEYWORDS = (
    "Temporal", "Frontal", "Opercular", "Planum", "Heschl",
    "Insular", "Fusiform", "Parahippocampal", "Angular",
    "Supramarginal", "Parietal Opercular",
)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _checkpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    """Unique checkpoints with step, revision, tokens (sorted by step)."""
    return (
        df[["step", "revision", "tokens_seen"]]
        .drop_duplicates()
        .sort_values("step")
        .reset_index(drop=True)
    )


def _format_step_label(step: float) -> str:
    s = int(step)
    if s == 0:
        return "0"
    if s >= 1000:
        return f"{s // 1000}k" if s % 1000 == 0 else f"{s}"
    return str(s)


def _format_tokens_label(tokens: float) -> str:
    if tokens <= 0:
        return "0"
    b = tokens / 1e9
    if b >= 1:
        return f"{b:.0f}B" if abs(b - round(b)) < 0.05 else f"{b:.1f}B"
    m = tokens / 1e6
    return f"{m:.0f}M" if m >= 1 else f"{int(tokens)}"


def _step_x_for_log(steps: np.ndarray | pd.Series) -> np.ndarray:
    """Positive x coordinates for log-scale training-step plots (step 0 → 1)."""
    return np.maximum(np.asarray(steps, dtype=float), 1.0)


def _step_xedges_log_spaced(steps: np.ndarray) -> np.ndarray:
    """Cell edges for heatmaps so columns look evenly spaced on a log step axis."""
    x = np.sort(_step_x_for_log(steps))
    if len(x) == 1:
        return np.array([x[0] * 0.5, x[0] * 2.0])
    edges = np.zeros(len(x) + 1, dtype=float)
    edges[0] = x[0] ** 2 / x[1]
    edges[-1] = x[-1] ** 2 / x[-2]
    edges[1:-1] = np.sqrt(x[:-1] * x[1:])
    return np.maximum(edges, 1.0)


def _set_step_axis(ax: plt.Axes, ckpt: pd.DataFrame) -> None:
    steps = ckpt["step"].to_numpy(dtype=float)
    x = _step_x_for_log(steps)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_format_step_label(s) for s in steps],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_xlabel("Training step (log scale)")
    ax.set_xlim(x.min() * 0.7, x.max() * 1.3)


def _set_tokens_axis(ax: plt.Axes, ckpt: pd.DataFrame) -> None:
    """Log-spaced ticks only at observed checkpoints (no symlog auto labels)."""
    tokens = ckpt["tokens_seen"].to_numpy(dtype=float)
    ax.set_xscale("log")
    ax.set_xticks(tokens)
    ax.set_xticklabels(
        [_format_tokens_label(t) for t in tokens],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_xlabel("Tokens seen (pretraining, log scale)")
    lo, hi = tokens.min(), tokens.max()
    ax.set_xlim(lo * 0.7, hi * 1.3)


def _plot_heatmap_vs_log_step(
    ax: plt.Axes,
    wide: pd.DataFrame,
    *,
    cmap: str = "viridis",
    cbar_label: str,
    center: float | None = None,
    annot: bool = False,
    fmt: str = ".2f",
) -> None:
    steps = np.sort(np.asarray(wide.columns, dtype=float))
    wide = wide.reindex(columns=steps)
    xedges = _step_xedges_log_spaced(steps)
    yedges = np.arange(len(wide) + 1, dtype=float)
    mesh_kw: dict = {"cmap": cmap, "shading": "flat"}
    if center is not None:
        vals = wide.to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(vals)))
        mesh_kw["vmin"], mesh_kw["vmax"] = -vmax, vmax
    pcm = ax.pcolormesh(xedges, yedges, wide.to_numpy(dtype=float), **mesh_kw)
    if annot:
        x = _step_x_for_log(steps)
        for j, xv in enumerate(x):
            for i, name in enumerate(wide.index):
                val = wide.iloc[i, j]
                if np.isfinite(val):
                    ax.text(
                        xv,
                        i + 0.5,
                        format(val, fmt),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="0.15",
                    )
    ax.set_xscale("log")
    ax.set_xticks(_step_x_for_log(steps))
    ax.set_xticklabels(
        [_format_step_label(s) for s in steps],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(wide)) + 0.5)
    ax.set_yticklabels(wide.index, fontsize=8)
    ax.set_xlabel("Training step (log scale)")
    plt.colorbar(pcm, ax=ax, label=cbar_label)


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------


def _roi_list(df: pd.DataFrame) -> list[str]:
    ordered = [r for r in ROI_ORDER if r in df["roi"].unique()]
    extra = sorted(set(df["roi"].unique()) - set(ordered))
    return ordered + extra


def _best_layer_per_roi_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["revision", "step", "tokens_seen", "roi"])["mean_pearson"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def write_roi_inventory(df: pd.DataFrame, out: Path) -> None:
    """Document which ROIs appear in this run (and voxel counts)."""
    inv = (
        df.groupby("roi", as_index=False)
        .agg(
            n_voxels=("n_voxels", "first"),
            n_rows=("roi", "size"),
            n_checkpoints=("step", "nunique"),
            n_layers=("layer", "nunique"),
        )
        .sort_values("roi", key=lambda s: s.map({r: i for i, r in enumerate(ROI_ORDER)}))
    )
    inv["in_lbg_roi_set"] = inv["roi"].isin(ROI_ORDER)
    inv["mask_source"] = ROI_SOURCE
    _write_csv(inv, out / "roi_inventory.csv")
    meta = {
        "roi_names": sorted(df["roi"].unique().tolist()),
        "expected_lbg_rois": ROI_ORDER,
        "only_lbg_rois_present": sorted(df["roi"].unique().tolist()) == sorted(
            [r for r in ROI_ORDER if r in df["roi"].unique()]
        ),
        "checkpoints_in_roi_file": _checkpoint_table(df).to_dict(orient="records"),
        "note": (
            "ROI scores are mean Pearson r within each sphere ROI (reliable voxels only). "
            "Per checkpoint×layer, the best layer per ROI is used in step_analysis plots."
        ),
    }
    (out / "roi_inventory.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def plot_roi_training_curves_by_step(
    best: pd.DataFrame, ckpt: pd.DataFrame, out: Path, model_label: str
) -> None:
    rois = _roi_list(best)
    palette = sns.color_palette("tab10", n_colors=len(rois))

    fig, ax = plt.subplots(figsize=(9, 5))
    for color, roi in zip(palette, rois):
        sub = best[best["roi"] == roi].sort_values("step")
        ax.plot(
            _step_x_for_log(sub["step"]),
            sub["mean_pearson"],
            marker="o",
            ms=6,
            lw=2,
            color=color,
            label=roi,
        )
    _set_step_axis(ax, ckpt)
    ax.set_ylabel("Mean Pearson r (best layer in ROI)")
    ax.set_title(f"{model_label} — ROI encoding vs training step")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roi_training_curves_by_tokens(
    best: pd.DataFrame, ckpt: pd.DataFrame, out: Path, model_label: str
) -> None:
    rois = _roi_list(best)
    palette = sns.color_palette("tab10", n_colors=len(rois))

    fig, ax = plt.subplots(figsize=(9, 5))
    for color, roi in zip(palette, rois):
        sub = best[best["roi"] == roi].sort_values("tokens_seen")
        ax.plot(
            sub["tokens_seen"],
            sub["mean_pearson"],
            marker="o",
            ms=6,
            lw=2,
            color=color,
            label=roi,
        )
    _set_tokens_axis(ax, ckpt)
    ax.set_ylabel("Mean Pearson r (best layer in ROI)")
    ax.set_title(f"{model_label} — ROI encoding vs tokens seen")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roi_step_faceted(
    best: pd.DataFrame, ckpt: pd.DataFrame, out: Path, model_label: str
) -> None:
    """One subplot per ROI: training-step curve (best layer per checkpoint)."""
    rois = _roi_list(best)
    n = len(rois)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, roi in zip(axes, rois):
        sub = best[best["roi"] == roi].sort_values("step")
        ax.plot(
            _step_x_for_log(sub["step"]),
            sub["mean_pearson"],
            "o-",
            color="#2a6fbb",
            lw=2,
            ms=5,
        )
        ax.set_xscale("log")
        ax.set_title(roi, fontsize=11, fontweight="bold")
        ax.set_ylabel("Pearson r")
        ax.grid(True, alpha=0.25, which="both")
        if sub["layer"].notna().any():
            best_li = int(sub.loc[sub["mean_pearson"].idxmax(), "layer"])
            ax.text(
                0.03, 0.97, f"best layer @ max step: {best_li}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7, color="0.35",
            )

    for ax in axes[len(rois):]:
        ax.set_visible(False)

    for ax in axes[: len(rois)]:
        _set_step_axis(ax, ckpt)

    fig.suptitle(f"{model_label} — per-ROI training curves (best layer)", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roi_step_heatmap(best: pd.DataFrame, out: Path, model_label: str) -> None:
    wide = best.pivot(index="roi", columns="step", values="mean_pearson")
    wide = wide.reindex(_roi_list(best))
    fig, ax = plt.subplots(figsize=(max(9, wide.shape[1] * 0.55), 4.5))
    _plot_heatmap_vs_log_step(
        ax, wide, cmap="viridis", cbar_label="Mean Pearson r (best layer)", annot=True, fmt=".2f",
    )
    ax.set_ylabel("ROI")
    ax.set_title(f"{model_label} — ROI × step (best layer per cell)")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roi_layer_profiles_all_rois(
    df: pd.DataFrame, out: Path, model_label: str, *, step: int | None = None
) -> None:
    sub = df if step is None else df[df["step"] == step]
    if sub.empty:
        return
    rois = _roi_list(sub)
    g = sns.relplot(
        data=sub,
        x="layer",
        y="mean_pearson",
        hue="roi",
        hue_order=rois,
        kind="line",
        marker="o",
        height=5,
        aspect=1.6,
        palette="tab10",
    )
    g.set(xlabel="Layer index", ylabel="Mean Pearson r")
    title_step = f"step {step}" if step is not None else f"step {sub['step'].max()} (final)"
    g.fig.suptitle(f"{model_label} — layer depth by ROI ({title_step})", y=1.02, fontsize=12)
    g.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(g.fig)


def plot_roi_layer_heatmaps(df: pd.DataFrame, out_dir: Path, model_label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _checkpoint_table(df)
    for roi in _roi_list(df):
        sub = df[df["roi"] == roi]
        wide = sub.pivot_table(index="layer", columns="step", values="mean_pearson", aggfunc="mean")
        wide = wide.sort_index()
        wide = wide.reindex(columns=ckpt["step"].tolist())
        fig, ax = plt.subplots(
            figsize=(max(8, wide.shape[1] * 0.48), max(4, wide.shape[0] * 0.22))
        )
        _plot_heatmap_vs_log_step(
            ax, wide, cmap="mako", cbar_label="Mean Pearson r", annot=True, fmt=".2f",
        )
        ax.set_title(f"{model_label} — {roi}")
        ax.set_ylabel("Layer")
        fig.tight_layout()
        fig.savefig(out_dir / f"roi_layer_heatmap_{roi}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def plot_roi_layer_profiles_by_step(df: pd.DataFrame, out_dir: Path, model_label: str) -> None:
    """One layer×depth curve per ROI at a few reference steps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = sorted(df["step"].unique())
    picks = [steps[0], steps[len(steps) // 2], steps[-1]]
    for st in picks:
        plot_roi_layer_profiles_all_rois(
            df, out_dir / f"roi_layer_profile_step{st}.png", model_label, step=int(st),
        )


# ---------------------------------------------------------------------------
# Parcel helpers
# ---------------------------------------------------------------------------


def _pvalue_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _best_layer_per_parcel_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Best layer per parcel × checkpoint (max mean L/R encoding)."""
    score = df[["mean_pearson_left", "mean_pearson_right"]].mean(axis=1)
    if "mean_pearson" in df.columns:
        score = score.fillna(df["mean_pearson"])
    tmp = df.copy()
    tmp["_layer_score"] = score
    keys = ["revision", "step", "tokens_seen", "parcel"]
    picks: list[int] = []
    for _, grp in tmp.groupby(keys, sort=False):
        valid = grp["_layer_score"].notna()
        if not valid.any():
            continue
        picks.append(int(grp.loc[valid, "_layer_score"].idxmax()))
    return tmp.loc[picks].drop(columns=["_layer_score"]).reset_index(drop=True)


def _encoding_score(df: pd.DataFrame) -> pd.Series:
    """Scalar encoding score per row (mean L/R; best-layer rows)."""
    s = df[["mean_pearson_left", "mean_pearson_right"]].mean(axis=1)
    if "mean_pearson" in df.columns:
        s = s.fillna(df["mean_pearson"])
    return s


def _parcel_order(df: pd.DataFrame) -> list[str]:
    if "parcel" not in df.columns:
        return []
    tmp = df.copy()
    tmp["_score"] = tmp[["mean_pearson_left", "mean_pearson_right"]].mean(axis=1)
    sub = tmp.groupby("parcel", as_index=False)["_score"].mean()
    return sub.sort_values("_score", ascending=True)["parcel"].tolist()


def _all_parcels_ordered(best: pd.DataFrame) -> list[str]:
    """All parcels, low → high mean encoding (readable grid order)."""
    return _parcel_order(best) or sorted(best["parcel"].unique())


def plot_lateralization_bars(
    sub: pd.DataFrame,
    out: Path,
    title: str,
    *,
    value_col: str = "lateralization_index",
    p_col: str = "lateralization_pvalue",
) -> None:
    """Horizontal bar chart of L−R means (Fig. B.1c layout)."""
    order = _parcel_order(sub) if value_col == "lateralization_index" else sorted(sub["parcel"].unique())
    if not order:
        return
    plot_df = sub.set_index("parcel").reindex(order).reset_index()
    y = np.arange(len(plot_df))
    vals = plot_df[value_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5, max(8, len(plot_df) * 0.22)))
    colors = sns.color_palette("Spectral", n_colors=len(plot_df))
    ax.barh(y, vals, color=colors, edgecolor="none")
    for i, (v, p) in enumerate(zip(vals, plot_df[p_col].to_numpy(dtype=float), strict=False)):
        stars = _pvalue_stars(float(p))
        if not stars:
            continue
        if v >= 0:
            ax.text(v + 0.002, i, stars, va="center", ha="left", fontsize=7)
        else:
            ax.text(v - 0.002, i, stars, va="center", ha="right", fontsize=7)
    ax.axvline(0.0, ls="--", color="0.25", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["parcel"], fontsize=7)
    ax.set_xlabel("Left − right mean Pearson r")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lr_scatter(sub: pd.DataFrame, out: Path, title: str) -> None:
    order = _parcel_order(sub)
    colors = sns.color_palette("Spectral", n_colors=max(len(order), 1))
    name_to_color = {n: colors[i] for i, n in enumerate(order)}

    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in sub.iterrows():
        ax.plot(
            row["mean_pearson_right"],
            row["mean_pearson_left"],
            "s",
            color=name_to_color.get(row["parcel"], "0.5"),
            ms=5,
        )
    lims = [
        np.nanmin(sub[["mean_pearson_left", "mean_pearson_right"]].to_numpy()),
        np.nanmax(sub[["mean_pearson_left", "mean_pearson_right"]].to_numpy()),
    ]
    if np.all(np.isfinite(lims)):
        ax.plot(lims, lims, "--", color="0.3", zorder=0)
    ax.set_xlabel("Right hemisphere")
    ax.set_ylabel("Left hemisphere")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _classify_step_curves(best: pd.DataFrame) -> pd.DataFrame:
    """Per parcel: peak step, monotonic trend, inverse-U flag (interior peak)."""
    rows: list[dict] = []
    for parcel, grp in best.groupby("parcel", sort=False):
        g = grp.sort_values("step").copy()
        steps = g["step"].to_numpy(dtype=float)
        y = _encoding_score(g).to_numpy(dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 3:
            continue
        xs, ys = steps[valid], y[valid]
        peak_i = int(np.argmax(ys))
        peak_step = float(xs[peak_i])
        at_first = peak_i == 0
        at_last = peak_i == len(xs) - 1
        rho, p_lin = stats.spearmanr(xs, ys)
        xn = (xs - xs.mean()) / (xs.std() + 1e-12)
        coef = np.polyfit(xn, ys, deg=2)
        a2 = float(coef[0])
        interior_peak = (not at_first) and (not at_last)
        inverse_u = interior_peak and a2 < 0
        rows.append(
            {
                "parcel": parcel,
                "peak_step": peak_step,
                "peak_encoding": float(ys[peak_i]),
                "first_step_encoding": float(ys[0]),
                "last_step_encoding": float(ys[-1]),
                "interior_peak": interior_peak,
                "inverse_u_candidate": inverse_u,
                "quad_coef_a": a2,
                "spearman_rho_step": float(rho),
                "spearman_p": float(p_lin),
                "monotonic_increase": bool(at_last and rho > 0.3),
                "monotonic_decrease": bool(at_first and rho < -0.3),
            }
        )
    return pd.DataFrame(rows)


def plot_global_encoding_vs_step(
    best: pd.DataFrame, ckpt: pd.DataFrame, out: Path, model_label: str
) -> None:
    """Mean encoding across all parcels vs training step (best layer each)."""
    tmp = best.copy()
    tmp["encoding"] = _encoding_score(tmp)
    agg = (
        tmp.groupby("step", as_index=False)
        .agg(
            mean_encoding=("encoding", "mean"),
            std_encoding=("encoding", "std"),
            mean_lateralization=("lateralization_index", "mean"),
        )
        .merge(ckpt[["step", "tokens_seen"]].drop_duplicates(), on="step")
        .sort_values("step")
    )
    agg.to_csv(out.with_suffix(".csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    x_step = _step_x_for_log(agg["step"])
    ax.plot(x_step, agg["mean_encoding"], "o-", lw=2, ms=7, color="C0")
    ax.fill_between(
        x_step,
        agg["mean_encoding"] - agg["std_encoding"],
        agg["mean_encoding"] + agg["std_encoding"],
        alpha=0.2,
        color="C0",
    )
    _set_step_axis(ax, ckpt)
    ax.set_ylabel("Mean Pearson r (avg over 48 parcels)")
    ax.set_title(f"{model_label} — encoding vs step")
    ax.grid(True, alpha=0.25, which="both")

    ax2 = axes[1]
    ax2.plot(agg["tokens_seen"], agg["mean_encoding"], "o-", lw=2, ms=7, color="C1")
    _set_tokens_axis(ax2, ckpt)
    ax2.set_ylabel("Mean Pearson r (avg over 48 parcels)")
    ax2.set_title(f"{model_label} — encoding vs tokens")
    ax2.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_encoding_heatmap(best: pd.DataFrame, out: Path, model_label: str) -> None:
    """Parcel × step heatmap of encoding (best layer per cell)."""
    tmp = best.copy()
    tmp["encoding"] = _encoding_score(tmp)
    wide = tmp.pivot(index="parcel", columns="step", values="encoding")
    wide = wide.reindex(_parcel_order(best))
    fig, ax = plt.subplots(figsize=(max(9, wide.shape[1] * 0.55), max(8, wide.shape[0] * 0.18)))
    _plot_heatmap_vs_log_step(ax, wide, cmap="viridis", cbar_label="Mean Pearson r (best layer)")
    ax.set_ylabel("Parcel")
    ax.set_title(f"{model_label} — parcel encoding × step")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_selected_parcel_curves(
    best: pd.DataFrame,
    ckpt: pd.DataFrame,
    out: Path,
    model_label: str,
    *,
    parcels: list[str],
) -> None:
    tmp = best.copy()
    tmp["encoding"] = _encoding_score(tmp)
    palette = sns.color_palette("husl", n_colors=len(parcels))
    fig, ax = plt.subplots(figsize=(9, 5))
    for color, parcel in zip(palette, parcels):
        sub = tmp[tmp["parcel"] == parcel].sort_values("step")
        if sub.empty:
            continue
        ax.plot(_step_x_for_log(sub["step"]), sub["encoding"], "o-", lw=2, ms=5, color=color, label=parcel)
    _set_step_axis(ax, ckpt)
    ax.set_ylabel("Mean Pearson r (best layer)")
    ax.set_title(f"{model_label} — selected parcels vs step")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_parcel_curves_grid(
    best: pd.DataFrame,
    ckpt: pd.DataFrame,
    out: Path,
    model_label: str,
    parcels: list[str],
    *,
    title: str,
    ncols: int = 4,
    mark_peak: bool = True,
) -> None:
    """One small panel per parcel: encoding vs training step."""
    if not parcels:
        return
    tmp = best.copy()
    tmp["encoding"] = _encoding_score(tmp)
    nrows = int(np.ceil(len(parcels) / ncols))
    fig_w = 2.8 * ncols if len(parcels) > 16 else 3.2 * ncols
    fig_h = 2.1 * nrows if len(parcels) > 16 else 2.4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=False)
    axes = np.atleast_2d(axes)
    for ax, parcel in zip(axes.ravel(), parcels):
        sub = tmp[tmp["parcel"] == parcel].sort_values("step")
        if sub.empty:
            ax.axis("off")
            continue
        ax.plot(_step_x_for_log(sub["step"]), sub["encoding"], "o-", lw=1.6, ms=3, color="C0")
        if mark_peak:
            peak = sub.loc[sub["encoding"].idxmax()]
            ax.axvline(_step_x_for_log(peak["step"]), ls=":", color="0.5", lw=0.9)
        ax.set_xscale("log")
        ax.set_title(parcel, fontsize=7 if len(parcels) > 20 else 8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, which="both")
    for ax in axes.ravel()[len(parcels):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            _set_step_axis(ax, ckpt)
    fig.suptitle(title, y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160 if len(parcels) > 20 else 150, bbox_inches="tight")
    plt.close(fig)


def plot_all_parcel_curves_overlay(
    best: pd.DataFrame,
    ckpt: pd.DataFrame,
    out: Path,
    model_label: str,
    parcels: list[str],
) -> None:
    """All parcels on one axes."""
    if not parcels:
        return
    tmp = best.copy()
    tmp["encoding"] = _encoding_score(tmp)
    palette = sns.color_palette("husl", n_colors=len(parcels))
    fig, ax = plt.subplots(figsize=(11, 6))
    for color, parcel in zip(palette, parcels):
        sub = tmp[tmp["parcel"] == parcel].sort_values("step")
        if sub.empty:
            continue
        ax.plot(_step_x_for_log(sub["step"]), sub["encoding"], "o-", lw=1.4, ms=3, color=color, label=parcel)
    _set_step_axis(ax, ckpt)
    ax.set_ylabel("Mean Pearson r (best layer)")
    ax.set_title(f"{model_label} — all {len(parcels)} parcels vs training step")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=5, ncol=1, frameon=False)
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_top_parcel_curves_grid(
    best: pd.DataFrame, ckpt: pd.DataFrame, out: Path, model_label: str, *, n_top: int = 12,
) -> None:
    tmp = best.copy()
    tmp["encoding"] = _encoding_score(tmp)
    final_step = tmp["step"].max()
    ranking = (
        tmp[tmp["step"] == final_step].groupby("parcel")["encoding"].mean().sort_values(ascending=False)
    )
    top = ranking.head(n_top).index.tolist()
    plot_parcel_curves_grid(
        best, ckpt, out, model_label, top,
        title=f"{model_label} — top {n_top} parcels by final-step encoding (dotted = peak step)",
        ncols=4,
    )


def plot_lateralization_by_step(best: pd.DataFrame, out: Path, model_label: str) -> None:
    """Heatmap: parcel × step of L−R (best layer per cell)."""
    wide = best.pivot(index="parcel", columns="step", values="lateralization_index")
    wide = wide.reindex(_parcel_order(best))
    fig, ax = plt.subplots(figsize=(max(9, wide.shape[1] * 0.5), max(8, wide.shape[0] * 0.18)))
    _plot_heatmap_vs_log_step(
        ax, wide, cmap="RdBu_r", cbar_label="L − R mean Pearson r", center=0.0,
    )
    ax.set_title(f"{model_label} — parcel lateralization × training step")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_step_analysis(
    best: pd.DataFrame, step_dir: Path, model_label: str, *, snapshot_step: int | None
) -> None:
    """All plots keyed by training step (encoding scaling / inverse-U)."""
    step_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _checkpoint_table(best)

    plot_global_encoding_vs_step(best, ckpt, step_dir / "parcel_mean_encoding_vs_step.png", model_label)
    plot_encoding_heatmap(best, step_dir / "parcel_encoding_heatmap.png", model_label)
    plot_lateralization_by_step(best, step_dir / "parcel_lateralization_heatmap.png", model_label)

    shape_df = _classify_step_curves(best)
    shape_df.to_csv(step_dir / "parcel_curve_shape_by_step.csv", index=False)

    all_parcels = _all_parcels_ordered(best)
    n_all = len(all_parcels)

    plot_parcel_curves_grid(
        best, ckpt, step_dir / "parcel_encoding_curves_all_grid.png", model_label, all_parcels,
        title=f"{model_label} — all {n_all} parcels vs step (dotted = peak step)",
        ncols=8 if n_all > 24 else 4,
    )
    plot_all_parcel_curves_overlay(
        best, ckpt, step_dir / "parcel_encoding_curves_all_overlay.png", model_label, all_parcels,
    )
    plot_top_parcel_curves_grid(best, ckpt, step_dir / "parcel_encoding_curves_top12.png", model_label)

    lang_parcels = [p for p in all_parcels if any(kw in p for kw in LANG_PARCEL_KEYWORDS)]
    if lang_parcels:
        plot_selected_parcel_curves(
            best, ckpt, step_dir / "parcel_encoding_curves_language.png", model_label,
            parcels=lang_parcels,
        )

    if snapshot_step is not None:
        sub = best[best["step"] == snapshot_step].copy()
        if not sub.empty:
            sub.to_csv(step_dir / f"parcel_best_layer_step{snapshot_step}.csv", index=False)
            plot_lateralization_bars(
                sub, step_dir / f"parcel_l_minus_r_step{snapshot_step}.png",
                f"{model_label} — L−R at step {snapshot_step}",
            )
            plot_lr_scatter(
                sub, step_dir / f"parcel_left_vs_right_step{snapshot_step}.png",
                f"{model_label} — L vs R at step {snapshot_step}",
            )


def write_isc_panels(
    isc_path: Path, brain_mask_path: Path, summary_dir: Path, lang: str = "en",
) -> None:
    isc_trials = np.asarray(joblib.load(isc_path))
    isc = np.mean(isc_trials, axis=0)
    parcelization = load_harvard_oxford_parcelization(brain_mask_path)
    gix = np.arange(isc.shape[0], dtype=np.int64)
    rows = parcel_summary_from_correlations(isc, gix, parcelization)
    isc_df = pd.DataFrame(rows)
    isc_df.to_csv(summary_dir / "isc_by_parcel.csv", index=False)
    plot_lateralization_bars(
        isc_df, summary_dir / "isc_parcel_l_minus_r.png",
        f"Inter-subject correlation — L−R by parcel ({lang})",
    )
    plot_lr_scatter(isc_df, summary_dir / "isc_parcel_left_vs_right.png", f"ISC — L vs R ({lang})")


# ---------------------------------------------------------------------------
# Models helpers
# ---------------------------------------------------------------------------


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "_")


def _model_id_from_dir(dirname: str) -> str:
    return dirname.replace("_", "/")


def _load_model_metadata(config_path: Path | None) -> dict[str, dict]:
    """Map model_id and safe_dir → {label, n_params, n_layers, model_id}."""
    meta: dict[str, dict] = {}
    if config_path is None or not config_path.is_file():
        return meta
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for m in raw.get("models") or []:
        mid = str(m.get("model_id", "")).strip()
        if not mid:
            continue
        row = {
            "model_id": mid,
            "label": str(m.get("label", mid.split("/")[-1])),
            "n_params": float(m["n_params"]) if m.get("n_params") is not None else np.nan,
            "n_layers": int(m["n_layers"]) if m.get("n_layers") is not None else np.nan,
        }
        meta[mid] = row
        meta[_safe_model_dir(mid)] = row
    return meta


def _discover_model_dirs(sweep_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "lpp_checkpoint_scores.csv").is_file():
            out.append(p)
    return out


def _attach_metadata(scores: pd.DataFrame, meta: dict[str, dict]) -> pd.DataFrame:
    df = scores.copy()
    if "model_dir" not in df.columns:
        df["model_dir"] = df.get("model_folder", "")

    def lookup(row: pd.Series) -> pd.Series:
        mid = str(row.get("model_id", ""))
        folder = str(row.get("model_dir", ""))
        m = meta.get(mid) or meta.get(folder) or meta.get(_model_id_from_dir(folder), {})
        label = m.get("label") or row.get("model_label") or folder
        n_params = m.get("n_params", np.nan)
        if not np.isfinite(n_params):
            msize = re.search(r"pythia-([\d.]+)b", folder, re.I)
            if msize:
                n_params = float(msize.group(1)) * 1e9
        return pd.Series(
            {
                "model_label": label,
                "n_params": n_params,
                "n_params_b": n_params / 1e9 if np.isfinite(n_params) else np.nan,
                "log10_n_params": np.log10(n_params) if np.isfinite(n_params) and n_params > 0 else np.nan,
            }
        )

    extra = df.apply(lookup, axis=1)
    out = df.loc[:, ~df.columns.duplicated()].copy()
    for col in ("model_label", "n_params", "n_params_b", "log10_n_params"):
        if col in extra.columns:
            out[col] = extra[col].to_numpy()
    return out


def _fit_scaling(x: np.ndarray, y: np.ndarray) -> dict:
    """Linear and quadratic fits in log10(n_params); flag inverse-U if vertex in range."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    out: dict = {"n": int(len(x)), "linear_r2": np.nan, "quad_r2": np.nan}
    if len(x) < 2:
        return out
    slope, intercept, r, p, _ = stats.linregress(x, y)
    out.update(
        linear_slope=slope, linear_intercept=intercept, linear_r=float(r),
        linear_r2=float(r**2), linear_pvalue=float(p),
    )
    if len(x) >= 3:
        coef = np.polyfit(x, y, 2)
        pred = np.polyval(coef, x)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        quad_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        a, b, c = coef
        vertex_x = -b / (2 * a) if abs(a) > 1e-12 else np.nan
        vertex_y = np.polyval(coef, vertex_x) if np.isfinite(vertex_x) else np.nan
        order = np.argsort(x)
        y_sorted = y[order]
        mid_higher = len(y) >= 3 and y_sorted[1] > y_sorted[0] and y_sorted[1] > y_sorted[-1]
        inverse_u = (
            len(x) >= 4
            and np.isfinite(vertex_x)
            and (x.min() < vertex_x < x.max())
            and a < 0
            and mid_higher
        )
        if len(x) < 4:
            preferred = (
                "scaling_up (linear; n<4 models for quad U-test)" if slope > 0
                else ("scaling_down" if slope < 0 else "flat")
            )
        else:
            preferred = "inverse_u" if inverse_u else ("scaling_up" if slope > 0 else "scaling_down")
        out.update(
            quad_a=a, quad_b=b, quad_c=c, quad_r2=quad_r2,
            vertex_log10_n_params=vertex_x, vertex_mean_pearson=vertex_y,
            inverse_u_in_range=bool(inverse_u), preferred_model=preferred,
        )
    return out


def _params_x_for_log(params_b: np.ndarray | pd.Series) -> np.ndarray:
    return np.maximum(np.asarray(params_b, dtype=float), 0.05)


def _set_models_axis(ax: plt.Axes, models: pd.DataFrame) -> None:
    """Log x at observed parameter counts; tick labels = model names."""
    tab = (
        models[["model_label", "n_params_b"]].drop_duplicates().sort_values("n_params_b")
    )
    x = _params_x_for_log(tab["n_params_b"])
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tab["model_label"], rotation=25, ha="right", fontsize=9)
    ax.set_xlabel("Model size (parameters, log scale)")
    ax.set_xlim(x.min() * 0.75, x.max() * 1.25)


def _pick_global_best_step_layer(df: pd.DataFrame) -> tuple[int, int]:
    """Single (step, layer) maximizing mean parcel encoding for one model."""
    tmp = df.copy()
    tmp["encoding"] = _encoding_score(tmp)
    agg = tmp.groupby(["step", "layer"], as_index=False)["encoding"].mean()
    row = agg.loc[agg["encoding"].idxmax()]
    return int(row["step"]), int(row["layer"])


def _load_parcel_best_snapshots(
    model_dirs: list[Path], meta: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per model: global best checkpoint+layer → one row per parcel."""
    snaps: list[pd.DataFrame] = []
    meta_rows: list[dict] = []
    for mdir in model_dirs:
        parcel_csv = mdir / "lpp_checkpoint_by_parcel.csv"
        if not parcel_csv.is_file():
            continue
        raw = pd.read_csv(parcel_csv)
        if raw.empty:
            continue
        raw = raw.assign(model_dir=mdir.name)
        df = _attach_metadata(raw, meta)
        step, layer = _pick_global_best_step_layer(df)
        snap = df[(df["step"] == step) & (df["layer"] == layer)].copy()
        snap["encoding"] = _encoding_score(snap)
        snap["best_step"] = step
        snap["best_layer"] = layer
        snaps.append(snap)
        meta_rows.append(
            {
                "model_label": str(snap["model_label"].iloc[0]),
                "model_dir": mdir.name,
                "n_params_b": float(snap["n_params_b"].iloc[0]),
                "best_step": step,
                "best_layer": layer,
                "mean_encoding": float(snap["encoding"].mean()),
            }
        )
    if not snaps:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(snaps, ignore_index=True), pd.DataFrame(meta_rows)


def plot_scaling_at_steps(
    df: pd.DataFrame, steps: list[int], out: Path, *, metric: str = "mean_pearson",
) -> None:
    """Model size (log x) vs brain score at a few training steps."""
    sub = df[df["step"].isin(steps)].copy()
    if sub.empty:
        return
    colors = model_color_map(sub)
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, step in enumerate(steps):
        snap = sub[sub["step"] == step].sort_values("n_params_b")
        if snap.empty:
            continue
        mk = markers[i % len(markers)]
        ax.plot(
            snap["n_params_b"], snap[metric], linestyle="-", lw=1.4, color="0.45",
            zorder=1, label=f"step {_format_step_label(step)}",
        )
        for _, row in snap.iterrows():
            ax.plot(row["n_params_b"], row[metric], marker=mk, ms=10, linestyle="none",
                    color=colors[row["model_label"]], zorder=3)
            ax.annotate(row["model_label"], (row["n_params_b"], row[metric]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7,
                        color=colors[row["model_label"]])
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions)")
    ax.set_ylabel("Mean Pearson r (voxelwise, best layer)")
    ax.set_title("LPP encoding vs model size at reference checkpoints")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_final_with_fits(
    final: pd.DataFrame, fit: dict, out: Path, *, metric: str = "mean_pearson",
) -> None:
    """Final-checkpoint scaling with linear + quadratic overlays."""
    sub = final.sort_values("n_params_b")
    if sub.empty:
        return
    colors = model_color_map(sub)
    x = sub["log10_n_params"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)
    x_line = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in sub.iterrows():
        ax.scatter(row["n_params_b"], row[metric], s=140, zorder=3,
                   color=colors[row["model_label"]], edgecolors="k", linewidths=0.6,
                   label=row["model_label"])
        ax.annotate(row["model_label"], (row["n_params_b"], row[metric]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9,
                    color=colors[row["model_label"]])
    if np.isfinite(fit.get("linear_slope", np.nan)):
        ax.plot(10 ** x_line, fit["linear_slope"] * x_line + fit["linear_intercept"],
                "--", color="#d6604d", lw=2, label=f"linear fit (R²={fit.get('linear_r2', np.nan):.3f})")
    if np.isfinite(fit.get("quad_a", np.nan)):
        ax.plot(10 ** x_line, np.polyval([fit["quad_a"], fit["quad_b"], fit["quad_c"]], x_line),
                "-", color="#4d9221", lw=2, label=f"quadratic fit (R²={fit.get('quad_r2', np.nan):.3f})")
        vx = fit.get("vertex_log10_n_params")
        if fit.get("inverse_u_in_range") and np.isfinite(vx) and fit.get("n", 0) >= 4:
            ax.axvline(10**vx, color="0.45", ls=":", lw=1.5)
            ax.scatter([10**vx], [fit["vertex_mean_pearson"]], s=160, facecolors="none",
                       edgecolors="#4d9221", linewidths=2, zorder=4,
                       label=f"quad peak @ {10**vx:.2f}B params")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions)")
    ax.set_ylabel("Mean Pearson r (final checkpoint)")
    step_final = int(sub["step"].iloc[0]) if "step" in sub.columns else -1
    ax.set_title(f"Model scaling at final step ({_format_step_label(step_final)})")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    note = fit.get("preferred_model", "")
    if note:
        ax.text(0.02, 0.02, f"Diagnostic: {note.replace('_', ' ')}", transform=ax.transAxes,
                fontsize=8, color="0.35")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_models_step_heatmap(df: pd.DataFrame, out: Path, *, metric: str = "mean_pearson") -> None:
    wide = (
        df.pivot_table(index="model_label", columns="step", values=metric, aggfunc="first")
        .sort_index(key=lambda s: df.groupby("model_label")["n_params_b"].first().loc[s])
    )
    fig, ax = plt.subplots(figsize=(max(9, wide.shape[1] * 0.55), max(3.5, wide.shape[0] * 0.55)))
    sns.heatmap(wide, ax=ax, cmap="viridis", annot=True, fmt=".3f", linewidths=0.4,
                cbar_kws={"label": metric})
    ax.set_xlabel("Training step")
    ax.set_ylabel("Model")
    ax.set_title("LPP encoding — models × checkpoint")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_per_model_training_facets(df: pd.DataFrame, out: Path, *, metric: str = "mean_pearson") -> None:
    colors = model_color_map(df)
    models = (
        df[["model_label", "n_params_b"]].drop_duplicates().sort_values("n_params_b")["model_label"].tolist()
    )
    n = len(models)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4 * nrows), squeeze=False)
    for ax, model_lbl in zip(axes.ravel(), models):
        sub = df[df["model_label"] == model_lbl].sort_values("tokens_seen")
        ax.plot(sub["tokens_seen"], sub[metric], "o-", color=colors[model_lbl], lw=2.2, ms=5)
        ax.set_title(model_lbl, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Pearson r")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _format_tokens_label(x)))
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        ax.grid(True, alpha=0.25, which="both")
    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)
    fig.supxlabel("Tokens seen (pretraining)")
    fig.suptitle("Per-model training curves (voxelwise score)", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roi_scaling_at_final(roi_df: pd.DataFrame, final_step: int, out: Path) -> None:
    """Mean L-bg ROI score vs model size at final checkpoint."""
    best = _best_layer_per_roi_checkpoint(roi_df)
    snap = best[best["step"] == final_step].copy()
    if snap.empty:
        snap = best.groupby("model_label", as_index=False).apply(
            lambda g: g.loc[g["step"].idxmax()], include_groups=False
        )
    if "n_params_b" not in snap.columns:
        return

    per_model = (
        snap.groupby(["model_label", "n_params_b", "log10_n_params"], as_index=False)
        .agg(mean_roi_pearson=("mean_pearson", "mean"))
        .sort_values("n_params_b")
    )
    colors = model_color_map(per_model.assign(mean_pearson=per_model["mean_roi_pearson"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in per_model.iterrows():
        ax.plot(row["n_params_b"], row["mean_roi_pearson"], "o", color=colors[row["model_label"]], ms=10)
    ax.plot(per_model["n_params_b"], per_model["mean_roi_pearson"], "-", color="0.35", lw=1.5, zorder=0)
    for _, row in per_model.iterrows():
        ax.annotate(row["model_label"], (row["n_params_b"], row["mean_roi_pearson"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9,
                    color=colors[row["model_label"]])
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions)")
    ax.set_ylabel("Mean Pearson r (L-bg ROIs, best layer)")
    ax.set_title(f"ROI-mean encoding vs model size @ step {_format_step_label(final_step)}")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    rois = [r for r in ROI_ORDER if r in snap["roi"].unique()]
    if not rois:
        rois = _roi_list(snap)[:7]
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10", n_colors=len(rois))
    for color, roi in zip(palette, rois):
        sub = snap[snap["roi"] == roi].sort_values("n_params_b")
        if sub.empty:
            continue
        ax.plot(sub["n_params_b"], sub["mean_pearson"], "o-", ms=6, lw=1.8, color=color, label=roi)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions)")
    ax.set_ylabel("Mean Pearson r (best layer in ROI)")
    ax.set_title(f"L-bg ROIs vs model size @ step {_format_step_label(final_step)}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out.with_name(out.stem + "_by_roi.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    _write_csv(per_model, out.with_suffix(".csv"))


def plot_parcel_encoding_heatmap_models(snap: pd.DataFrame, out: Path) -> None:
    wide = snap.pivot(index="parcel", columns="model_label", values="encoding")
    order = (
        snap[["model_label", "n_params_b"]].drop_duplicates().sort_values("n_params_b")["model_label"].tolist()
    )
    wide = wide.reindex(_parcel_order(snap), columns=order)
    fig, ax = plt.subplots(figsize=(max(6, wide.shape[1] * 1.8), max(8, wide.shape[0] * 0.18)))
    sns.heatmap(wide, ax=ax, cmap="viridis", linewidths=0.15, annot=True, fmt=".2f",
                cbar_kws={"label": "Mean Pearson r (best step & layer)"})
    ax.set_xlabel("Model")
    ax.set_ylabel("Parcel")
    ax.set_title("Parcel encoding — models at each model's best checkpoint/layer")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_parcel_lateralization_heatmap_models(snap: pd.DataFrame, out: Path) -> None:
    wide = snap.pivot(index="parcel", columns="model_label", values="lateralization_index")
    order = (
        snap[["model_label", "n_params_b"]].drop_duplicates().sort_values("n_params_b")["model_label"].tolist()
    )
    wide = wide.reindex(_parcel_order(snap), columns=order)
    fig, ax = plt.subplots(figsize=(max(6, wide.shape[1] * 1.8), max(8, wide.shape[0] * 0.18)))
    sns.heatmap(wide, ax=ax, cmap="RdBu_r", center=0, linewidths=0.2, annot=True, fmt=".2f",
                cbar_kws={"label": "L − R mean Pearson r"})
    ax.set_title("Parcel lateralization — models at best checkpoint/layer")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_parcel_mean_encoding_vs_models(snap: pd.DataFrame, out: Path) -> None:
    agg = (
        snap.groupby(["model_label", "n_params_b"], as_index=False)
        .agg(mean_encoding=("encoding", "mean"), std_encoding=("encoding", "std"),
             mean_lateralization=("lateralization_index", "mean"))
        .sort_values("n_params_b")
    )
    agg.to_csv(out.with_suffix(".csv"), index=False)
    colors = model_color_map(agg.rename(columns={"mean_encoding": "mean_pearson"}))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for _, row in agg.iterrows():
        ax.errorbar(row["n_params_b"], row["mean_encoding"], yerr=row["std_encoding"],
                    fmt="o", ms=9, capsize=4, color=colors[row["model_label"]], label=row["model_label"])
    _set_models_axis(ax, agg)
    ax.set_ylabel("Mean Pearson r (avg over parcels)")
    ax.set_title("Mean parcel encoding vs model size")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.25, which="both")

    ax2 = axes[1]
    for _, row in agg.iterrows():
        ax2.plot(row["n_params_b"], row["mean_lateralization"], "o", ms=9,
                 color=colors[row["model_label"]], label=row["model_label"])
    _set_models_axis(ax2, agg)
    ax2.axhline(0, color="0.4", ls="--", lw=0.8)
    ax2.set_ylabel("Mean L − R (parcels)")
    ax2.set_title("Mean lateralization vs model size")
    ax2.legend(framealpha=0.9, fontsize=8)
    ax2.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_parcel_curves_grid_models(
    snap: pd.DataFrame, out: Path, parcels: list[str], *, title: str, ncols: int = 4,
) -> None:
    if not parcels:
        return
    colors = model_color_map(snap)
    nrows = int(np.ceil(len(parcels) / ncols))
    fig_w = 2.8 * ncols if len(parcels) > 16 else 3.2 * ncols
    fig_h = 2.1 * nrows if len(parcels) > 16 else 2.4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=False)
    axes = np.atleast_2d(axes)
    for ax, parcel in zip(axes.ravel(), parcels):
        sub = snap[snap["parcel"] == parcel].sort_values("n_params_b")
        if sub.empty:
            ax.axis("off")
            continue
        for _, row in sub.iterrows():
            ax.plot(_params_x_for_log(row["n_params_b"]), row["encoding"], "o", ms=5,
                    color=colors[row["model_label"]])
        if len(sub) > 1:
            ax.plot(_params_x_for_log(sub["n_params_b"]), sub["encoding"], "-", lw=1, color="0.55", zorder=0)
        ax.set_xscale("log")
        ax.set_title(parcel, fontsize=7 if len(parcels) > 20 else 8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, which="both")
    for ax in axes.ravel()[len(parcels):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            _set_models_axis(ax, snap)
    fig.suptitle(title, y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160 if len(parcels) > 20 else 150, bbox_inches="tight")
    plt.close(fig)


def plot_parcel_overlay_models(snap: pd.DataFrame, out: Path, parcels: list[str]) -> None:
    if not parcels:
        return
    colors = model_color_map(snap)
    palette = sns.color_palette("husl", n_colors=len(parcels))
    fig, ax = plt.subplots(figsize=(11, 6))
    for pcolor, parcel in zip(palette, parcels):
        sub = snap[snap["parcel"] == parcel].sort_values("n_params_b")
        if sub.empty:
            continue
        ax.plot(_params_x_for_log(sub["n_params_b"]), sub["encoding"], "o-",
                lw=1.4, ms=3, color=pcolor, label=parcel)
    _set_models_axis(ax, snap)
    ax.set_ylabel("Mean Pearson r")
    ax.set_title(f"All {len(parcels)} parcels vs model size (best checkpoint/layer each)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m], markersize=8, label=m)
        for m in colors
    ]
    leg1 = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8,
                     title="Model", frameon=True)
    ax.add_artist(leg1)
    ax.legend(bbox_to_anchor=(1.02, 0.35), loc="upper left", fontsize=4, ncol=1,
              frameon=False, title="Parcel")
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_selected_parcel_curves_models(snap: pd.DataFrame, out: Path, parcels: list[str]) -> None:
    colors = model_color_map(snap)
    palette = sns.color_palette("husl", n_colors=len(parcels))
    fig, ax = plt.subplots(figsize=(9, 5))
    for pcolor, parcel in zip(palette, parcels):
        sub = snap[snap["parcel"] == parcel].sort_values("n_params_b")
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            ax.plot(_params_x_for_log(row["n_params_b"]), row["encoding"], "o", ms=7,
                    color=colors[row["model_label"]])
        ax.plot(_params_x_for_log(sub["n_params_b"]), sub["encoding"], "-", lw=1.5, color=pcolor, label=parcel)
    _set_models_axis(ax, snap)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m], markersize=8, label=m)
        for m in colors
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Model")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("Language-adjacent parcels vs model size")
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_cross_model_parcel_analysis(
    model_dirs: list[Path], meta: dict[str, dict], out_dir: Path,
) -> None:
    """Mirror per-model step_analysis/ but x-axis = model size at best snapshot."""
    snap, best_meta = _load_parcel_best_snapshots(model_dirs, meta)
    if snap.empty:
        print("  (skip parcel cross-model: no lpp_checkpoint_by_parcel.csv found)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    snap.to_csv(out_dir / "parcel_at_best_checkpoint_all_models.csv", index=False)
    best_meta.to_csv(out_dir / "best_checkpoint_per_model.csv", index=False)
    (out_dir / "best_checkpoint_per_model.json").write_text(
        json.dumps(best_meta.to_dict(orient="records"), indent=2), encoding="utf-8",
    )

    plot_parcel_mean_encoding_vs_models(snap, out_dir / "parcel_mean_encoding_vs_model_size.png")
    plot_parcel_encoding_heatmap_models(snap, out_dir / "parcel_encoding_heatmap.png")
    plot_parcel_lateralization_heatmap_models(snap, out_dir / "parcel_lateralization_heatmap.png")

    all_parcels = _all_parcels_ordered(snap)
    plot_parcel_curves_grid_models(
        snap, out_dir / "parcel_encoding_curves_all_grid.png", all_parcels,
        title=f"All {len(all_parcels)} parcels vs model size",
        ncols=8 if len(all_parcels) > 24 else 4,
    )
    plot_parcel_overlay_models(snap, out_dir / "parcel_encoding_curves_all_overlay.png", all_parcels)

    top12 = (
        snap.groupby("parcel")["encoding"].mean().sort_values(ascending=False).head(12).index.tolist()
    )
    plot_parcel_curves_grid_models(
        snap, out_dir / "parcel_encoding_curves_top12.png", top12,
        title="Top 12 parcels (mean across models) vs model size", ncols=4,
    )

    lang_parcels = [p for p in all_parcels if any(kw in p for kw in LANG_PARCEL_KEYWORDS)]
    if lang_parcels:
        plot_selected_parcel_curves_models(snap, out_dir / "parcel_encoding_curves_language.png", lang_parcels)

    for _, row in best_meta.iterrows():
        lbl = row["model_label"]
        sub = snap[snap["model_label"] == lbl]
        step, layer = int(row["best_step"]), int(row["best_layer"])
        plot_lateralization_bars(
            sub, out_dir / f"parcel_l_minus_r_{lbl.replace(' ', '_')}.png",
            f"{lbl} — L−R @ step {_format_step_label(step)}, layer {layer}",
        )
        plot_lr_scatter(
            sub, out_dir / f"parcel_left_vs_right_{lbl.replace(' ', '_')}.png",
            f"{lbl} — L vs R @ step {_format_step_label(step)}, layer {layer}",
        )

    print(f"  Parcel cross-model panels → {out_dir}")
    for _, row in best_meta.iterrows():
        print(f"    {row['model_label']}: step {_format_step_label(row['best_step'])}, "
              f"layer {row['best_layer']}, mean r={row['mean_encoding']:.3f}")


# ---------------------------------------------------------------------------
# Subcommand entrypoints
# ---------------------------------------------------------------------------


def main_roi(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.resolve()
    roi_csv = model_dir / "lpp_checkpoint_by_roi.csv"
    if not roi_csv.is_file():
        sys.exit(f"Missing {roi_csv}")

    df = pd.read_csv(roi_csv)
    if df.empty:
        sys.exit(f"Empty {roi_csv}")

    model_label = str(df.get("model_label", pd.Series([model_dir.name])).iloc[0])
    summary_dir = model_dir / "summary"
    step_dir = summary_dir / "step_analysis"
    layer_dir = summary_dir / "layer_analysis"
    for d in (summary_dir, step_dir, layer_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_roi_inventory(df, summary_dir)

    best = _best_layer_per_roi_checkpoint(df).sort_values(["roi", "step"])
    ckpt = _checkpoint_table(df)

    _write_csv(best, step_dir / "roi_best_layer_per_checkpoint.csv")
    _write_csv(
        best.pivot(index="roi", columns="revision", values="mean_pearson").reset_index(),
        step_dir / "roi_pearson_by_revision_wide.csv",
    )
    plot_roi_training_curves_by_step(best, ckpt, step_dir / "roi_training_curves_by_step.png", model_label)
    plot_roi_training_curves_by_tokens(best, ckpt, step_dir / "roi_training_curves_by_tokens.png", model_label)
    plot_roi_step_faceted(best, ckpt, step_dir / "roi_training_curves_faceted.png", model_label)
    plot_roi_step_heatmap(best, step_dir / "roi_step_heatmap.png", model_label)
    plot_roi_training_curves_by_step(best, ckpt, summary_dir / "roi_training_curves.png", model_label)

    plot_roi_layer_profiles_all_rois(df, layer_dir / "roi_layer_profiles_final_checkpoint.png", model_label)
    plot_roi_layer_profiles_by_step(df, layer_dir / "roi_layer_profiles_by_step", model_label)
    plot_roi_layer_heatmaps(df, layer_dir / "roi_layer_heatmaps", model_label)

    max_layer = (
        df.groupby(["revision", "step", "tokens_seen", "roi"], as_index=False)["mean_pearson"].max()
    )
    plot_roi_step_heatmap(max_layer, summary_dir / "roi_checkpoint_heatmap.png", model_label)

    print(f"Wrote ROI summary → {summary_dir}")
    print(f"  ROIs ({df['roi'].nunique()}): {', '.join(_roi_list(df))}")
    print(f"  Checkpoints in ROI file: {len(ckpt)}  layers: {df['layer'].nunique()}")
    if 0 not in df["step"].values:
        print("  Note: step0 is not in lpp_checkpoint_by_roi.csv.")


def main_parcels(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.resolve()
    parcel_csv = model_dir / "lpp_checkpoint_by_parcel.csv"
    if not parcel_csv.is_file():
        sys.exit(
            f"Missing {parcel_csv}\n"
            "Re-run the sweep with analysis.harvard_oxford_parcels: true."
        )

    df = pd.read_csv(parcel_csv)
    if df.empty:
        sys.exit(f"Empty {parcel_csv}")

    model_label = str(df.get("model_label", pd.Series([model_dir.name])).iloc[0])
    summary_dir = model_dir / "summary" / "parcel_analysis"
    summary_dir.mkdir(parents=True, exist_ok=True)

    best = _best_layer_per_parcel_checkpoint(df)
    best.to_csv(summary_dir / "parcel_best_layer_all_checkpoints.csv", index=False)

    step_dir = summary_dir / "step_analysis"
    write_step_analysis(best, step_dir, model_label, snapshot_step=args.step)

    if not args.step_only:
        plot_lateralization_by_step(best, summary_dir / "parcel_l_minus_r_heatmap.png", model_label)
        if args.step is not None:
            sub = best[best["step"] == args.step].copy()
            if not sub.empty:
                plot_lateralization_bars(
                    sub, summary_dir / f"parcel_l_minus_r_step{args.step}.png",
                    f"{model_label} — L−R at step {args.step}",
                )

    shape_df = pd.read_csv(step_dir / "parcel_curve_shape_by_step.csv")
    meta = {
        "n_parcels": int(df["parcel"].nunique()),
        "checkpoints": sorted(df["step"].unique().tolist()),
        "parcel_names": sorted(df["parcel"].unique().tolist()),
        "inverse_u_candidates": int(shape_df["inverse_u_candidate"].sum()) if not shape_df.empty else 0,
        "monotonic_increase_parcels": int(shape_df["monotonic_increase"].sum()) if not shape_df.empty else 0,
        "source": "nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split=True)",
        "step_analysis_dir": str(step_dir),
    }
    (summary_dir / "parcel_inventory.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.isc_path is not None:
        isc_path = args.isc_path.resolve()
        if not isc_path.is_file():
            print(f"ISC file not found: {isc_path}", file=sys.stderr)
        else:
            write_isc_panels(isc_path, args.brain_mask_path.resolve(), summary_dir)

    print(f"Wrote parcel summary → {summary_dir}")
    if not shape_df.empty:
        print(f"  Curve shapes: {meta['inverse_u_candidates']} inverse-U candidates, "
              f"{meta['monotonic_increase_parcels']} monotonic ↑")


def main_models(args: argparse.Namespace) -> None:
    sweep_dir = args.sweep_dir.resolve()
    out_dir = sweep_dir / "summary" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_model_metadata(args.config.resolve() if args.config else None)
    model_dirs = _discover_model_dirs(sweep_dir)
    if not model_dirs:
        sys.exit(f"No model folders with lpp_checkpoint_scores.csv under {sweep_dir}")

    score_frames: list[pd.DataFrame] = []
    roi_frames: list[pd.DataFrame] = []
    lw_by_label: dict[str, pd.DataFrame] = {}

    for mdir in model_dirs:
        scores = pd.read_csv(mdir / "lpp_checkpoint_scores.csv")
        if scores.empty:
            continue
        if "step" in scores.columns:
            scores = scores.sort_values("step", kind="mergesort").reset_index(drop=True)
        scores = scores.assign(model_dir=mdir.name, model_folder=mdir.name)
        score_frames.append(scores)

        roi_csv = mdir / "lpp_checkpoint_by_roi.csv"
        if roi_csv.is_file():
            roi = pd.read_csv(roi_csv)
            if not roi.empty:
                roi_frames.append(roi.assign(model_dir=mdir.name))

        lw_csv = mdir / "lpp_checkpoint_layerwise.csv"
        if lw_csv.is_file():
            lw = pd.read_csv(lw_csv)
            if not lw.empty and {"step", "layer"}.issubset(lw.columns):
                lw = lw.sort_values(["step", "layer"], kind="mergesort").reset_index(drop=True)
            if not lw.empty:
                lbl = str(scores["model_label"].iloc[0]) if "model_label" in scores.columns else mdir.name
                lw_by_label[lbl] = lw

    combined = _attach_metadata(pd.concat(score_frames, ignore_index=True), meta)
    sort_cols = [c for c in ("n_params_b", "model_label", "step") if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    _write_csv(combined, out_dir / "all_models_lpp_scores.csv")
    _write_csv(combined, sweep_dir / "summary" / "all_models_lpp_scores.csv")

    x_col = "tokens_seen" if "tokens_seen" in combined.columns else "step"
    plot_lpp_all_models_panel(combined, x_col=x_col, save_path=out_dir / "all_models_training_curves.png",
                              title="LPP brain scores — all models")
    if "lateralization_index" in combined.columns and combined["lateralization_index"].notna().any():
        plot_lpp_lateralization_panel(combined, x_col=x_col, save_path=out_dir / "all_models_lateralization.png")
    if lw_by_label:
        plot_lpp_layer_heatmap_grid(lw_by_label, x_col=x_col,
                                    save_path=out_dir / "all_models_layer_heatmaps_grid.png")

    plot_models_step_heatmap(combined, out_dir / "models_x_step_heatmap.png")
    plot_per_model_training_facets(combined, out_dir / "models_training_facets.png")

    ref_steps = [s for s in args.reference_steps if s in set(combined["step"].tolist())]
    if not ref_steps:
        ref_steps = sorted(combined["step"].unique().astype(int).tolist())

    plot_scaling_at_steps(combined, ref_steps, out_dir / "model_scaling_at_reference_steps.png")

    final_step = int(combined["step"].max())
    final = combined[combined["step"] == final_step].copy()
    _write_csv(final.sort_values("n_params_b"), out_dir / f"model_scores_final_step{final_step}.csv")

    fit_final = _fit_scaling(final["log10_n_params"].to_numpy(), final["mean_pearson"].to_numpy())
    fit_final["step"] = final_step
    fit_final["metric"] = "mean_pearson"

    fits_by_step: list[dict] = []
    for step in ref_steps:
        snap = combined[combined["step"] == step]
        fit = _fit_scaling(snap["log10_n_params"].to_numpy(), snap["mean_pearson"].to_numpy())
        fit["step"] = int(step)
        fit["metric"] = "mean_pearson"
        fits_by_step.append(fit)

    (out_dir / "scaling_fit_results.json").write_text(
        json.dumps({"final_checkpoint": fit_final, "reference_steps": fits_by_step}, indent=2),
        encoding="utf-8",
    )
    _write_csv(pd.DataFrame(fits_by_step), out_dir / "scaling_fit_by_step.csv")
    plot_scaling_final_with_fits(final, fit_final, out_dir / "model_scaling_final_checkpoint.png")

    long_rows = combined[combined["step"].isin(ref_steps)].copy()
    _write_csv(long_rows.sort_values(["step", "n_params_b"]), out_dir / "model_scores_at_reference_steps.csv")

    if roi_frames:
        roi_all = _attach_metadata(pd.concat(roi_frames, ignore_index=True), meta)
        _write_csv(roi_all, out_dir / "all_models_roi_long.csv")
        plot_roi_scaling_at_final(roi_all, final_step, out_dir / "roi_scaling_final.png")

    write_cross_model_parcel_analysis(model_dirs, meta, out_dir / "parcel_at_best_checkpoint")

    print(f"Wrote model comparison → {out_dir}")
    print(f"  Models: {', '.join(sorted(combined['model_label'].unique()))}")
    pref = str(fit_final.get("preferred_model", ""))
    if fit_final.get("inverse_u_in_range"):
        print(f"  Possible inverse-U: quad peak near {10 ** fit_final['vertex_log10_n_params']:.2f}B params")
    elif "scaling_up" in pref:
        print("  Scores increase with model size at final checkpoint (scaling-up trend)")
    if fit_final.get("n", 0) < 4:
        print("  Note: only 3 models — add more sizes before trusting a quadratic / inverse-U fit")


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    roi_p = sub.add_parser("roi", help="ROI summary for one model folder")
    roi_p.add_argument("--model_dir", type=Path, required=True,
                       help="Directory containing lpp_checkpoint_by_roi.csv")

    parcel_p = sub.add_parser("parcels", help="Parcel L/R summary for one model folder")
    parcel_p.add_argument("--model_dir", type=Path, required=True)
    parcel_p.add_argument("--brain_mask_path", type=Path,
                          default=Path("llms_brain_lateralization/mask_lpp_en.nii.gz"))
    parcel_p.add_argument("--isc_path", type=Path, default=None)
    parcel_p.add_argument("--step", type=int, default=None)
    parcel_p.add_argument("--step_only", action="store_true")

    models_p = sub.add_parser("models", help="Cross-model scaling comparison")
    models_p.add_argument("--sweep_dir", type=Path, default=Path("outputs/lpp/embedding_sweep"))
    models_p.add_argument("--config", type=Path, default=Path("configs/embedding/neural/lpp.yaml"))
    models_p.add_argument("--reference_steps", type=int, nargs="+", default=[0, 8000, 32000, 143000])
    models_p.add_argument("--out_subdir", type=str, default="model_scaling")

    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.subcommand == "roi":
        main_roi(args)
    elif args.subcommand == "parcels":
        main_parcels(args)
    elif args.subcommand == "models":
        main_models(args)


if __name__ == "__main__":
    main()
