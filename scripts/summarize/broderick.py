#!/usr/bin/env python3
"""Summarize Broderick 2018 EEG encoding results across model families.

Reads the per-model output directories produced by
``scripts/run/embedding/neural/broderick.py`` and generates:

  all_models_broderick_scores.csv     — one row per model (final checkpoint)
  scaling_law.png                     — n_params vs mean best-layer Pearson r
  layer_profiles.png                  — layer_rel vs mean Pearson r per model
  timecourse.png                      — mean TRF curve per model
  region_bars.png                     — score by scalp region (left/right/frontal/occipital)
  training_curves/<family>.png        — training dynamics (checkpoint sweep families)

Usage
-----
  python scripts/summarize/broderick.py
  python scripts/summarize/broderick.py \\
      --sweep_dir outputs_jeanzay/broderick_checkpoint_sweep \\
      --out_dir   outputs/broderick/summary
  python scripts/summarize/broderick.py --families gpt2 pythia mamba2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TRF window for timecourse summary plots
TC_TMIN = -0.2
TC_TMAX =  1.0

# Primary metric used everywhere
SCORE_METRIC   = "mean_best_pearsonr"    # best-layer mean Pearson r (checkpoint_scores.csv)
FLAT_METRIC    = "mean_flat_pearsonr"    # last-layer mean Pearson r
LAYER_METRIC   = "mean_pearsonr"         # per-layer metric (checkpoint_layerwise.csv)
TC_METRIC      = "pearsonr"              # timecourse metric (mean_timecourse.csv)
REGION_METRIC  = "best_pearsonr_mean"    # per-region metric (by_region.csv)

SCALP_REGIONS = ["left", "right", "posterior", "occipital"]

FAMILY_COLORS = {
    "gpt2":    "#1f77b4",
    "gptneo":  "#ff7f0e",
    "opt":     "#2ca02c",
    "llama32": "#d62728",
    "mamba":   "#9467bd",
    "mamba2":  "#8c564b",
    "mistral": "#e377c2",
    "qwen3":   "#7f7f7f",
    "pythia":  "#bcbd22",
    "olmo2":   "#17becf",
    "cbr_rnn": "#e41a1c",
}

FAMILY_MARKERS = {
    "gpt2": "o", "gptneo": "s", "opt": "^", "llama32": "D",
    "mamba": "v", "mamba2": "p", "mistral": "h", "qwen3": "*",
    "pythia": "P", "olmo2": "X", "cbr_rnn": "H",
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _registry_meta() -> dict[str, dict]:
    reg_path = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "neuroai" / "models" / "registry.py"
    text = reg_path.read_text(encoding="utf-8")

    var_map: dict[str, str] = {}
    for m in re.finditer(r'^([A-Z][A-Z0-9_]+)\s*=\s*"([^"]+)"', text, flags=re.M):
        var_map[m.group(1)] = m.group(2)

    meta: dict[str, dict] = {}
    fam_blocks = re.findall(r'"([^"]+)":\s*\[(.*?)\],', text, flags=re.S)

    main_re    = re.compile(r'_main_only\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
    spec_lit_re = re.compile(r'LppModelSpec\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
    spec_var_re = re.compile(r'LppModelSpec\(\s*([A-Z][A-Z0-9_]+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)

    def _add(model_id: str, label: str, n_layers: str, n_params: str, fam: str) -> None:
        entry = {"family": fam, "label": label, "n_params": float(n_params), "n_layers": int(n_layers)}
        meta[model_id] = entry
        meta[model_id.replace("/", "_")] = entry

    for fam, block in fam_blocks:
        for pattern in (main_re, spec_lit_re):
            for m in pattern.finditer(block):
                _add(*m.groups()[:4], fam)
        for m in spec_var_re.finditer(block):
            var_name, label, n_layers, n_params = m.groups()[:4]
            model_id = var_map.get(var_name)
            if model_id:
                _add(model_id, label, n_layers, n_params, fam)
    return meta


# ---------------------------------------------------------------------------
# Revision helpers
# ---------------------------------------------------------------------------

def _revision_step(rev: str) -> int:
    m = re.search(r"step(\d+)", rev)
    return int(m.group(1)) if m else (0 if rev in ("main", "None", "") else -1)


def _pick_final_revision(revisions: list[str]) -> str:
    return max(revisions, key=_revision_step)


# ---------------------------------------------------------------------------
# Data loading — final checkpoint scores
# ---------------------------------------------------------------------------

def load_all_models(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """One row per model (final checkpoint). Reads broderick_checkpoint_scores.csv."""
    rows: list[dict] = []

    for scores_csv in sorted(sweep_dir.glob("*/*/broderick_checkpoint_scores.csv")):
        family = scores_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = scores_csv.parent.name
        model_id  = model_dir.replace("_", "/", 1)

        try:
            df = pd.read_csv(scores_csv)
        except Exception:
            continue
        if df.empty:
            continue

        # Pick final checkpoint row
        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().astype(str).unique().tolist())
            row_df = df[df["revision"].astype(str) == rev]
        else:
            rev = str(df["revision"].iloc[0]) if "revision" in df.columns else "main"
            row_df = df

        if row_df.empty:
            continue
        row_df = row_df.iloc[-1]  # last row for this revision

        m = meta.get(model_id, meta.get(model_dir, {}))
        entry: dict = {
            "family":    family,
            "model_id":  model_id,
            "label":     m.get("label", model_id.split("/")[-1]),
            "n_params":  m.get("n_params", np.nan),
            "n_layers":  m.get("n_layers", np.nan),
            "revision":  rev,
        }
        for col in (SCORE_METRIC, FLAT_METRIC, "mean_best_spearmanr", "best_layer",
                    "n_subjects", "n_channels", "n_words"):
            entry[col] = float(row_df[col]) if col in row_df.index and pd.notna(row_df[col]) else np.nan
        rows.append(entry)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["family", "n_params"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Data loading — per-layer profiles
# ---------------------------------------------------------------------------

def load_layerwise(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """Reads broderick_checkpoint_layerwise.csv — one row per (model × layer)."""
    rows: list[dict] = []

    for lw_csv in sorted(sweep_dir.glob("*/*/broderick_checkpoint_layerwise.csv")):
        family = lw_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = lw_csv.parent.name
        model_id  = model_dir.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir, {}))

        try:
            df = pd.read_csv(lw_csv)
        except Exception:
            continue
        if df.empty or LAYER_METRIC not in df.columns:
            continue

        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().astype(str).unique().tolist())
            df = df[df["revision"].astype(str) == rev]
        else:
            rev = str(df["revision"].iloc[0]) if "revision" in df.columns else "main"

        if "layer_rel" not in df.columns and "layer" in df.columns:
            n = int(df["layer"].max()) + 1
            df = df.copy()
            df["layer_rel"] = df["layer"] / max(1, n - 1)

        df = df.copy()
        df["family"]   = family
        df["model_id"] = model_id
        df["label"]    = m.get("label", model_id.split("/")[-1])
        df["n_params"] = m.get("n_params", np.nan)
        df["revision"] = rev
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Data loading — mean TRF timecourses
# ---------------------------------------------------------------------------

def load_timecourses(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """Reads broderick_mean_timecourse.csv — one row per (model × time_bin)."""
    rows: list[dict] = []

    for tc_csv in sorted(sweep_dir.glob("*/*/broderick_mean_timecourse.csv")):
        family = tc_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = tc_csv.parent.name
        model_id  = model_dir.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir, {}))

        try:
            df = pd.read_csv(tc_csv)
        except Exception:
            continue
        if df.empty or TC_METRIC not in df.columns:
            continue

        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().astype(str).unique().tolist())
            df = df[df["revision"].astype(str) == rev]
        else:
            rev = str(df["revision"].iloc[0]) if "revision" in df.columns else "main"

        df = df.copy()
        df["family"]   = family
        df["model_id"] = model_id
        df["label"]    = m.get("label", model_id.split("/")[-1])
        df["n_params"] = m.get("n_params", np.nan)
        df["revision"] = rev
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Data loading — per-region scores
# ---------------------------------------------------------------------------

def load_region_scores(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """Reads broderick_by_region.csv — aggregated per (model × region)."""
    rows: list[dict] = []

    for reg_csv in sorted(sweep_dir.glob("*/*/broderick_by_region.csv")):
        family = reg_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = reg_csv.parent.name
        model_id  = model_dir.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir, {}))

        try:
            df = pd.read_csv(reg_csv)
        except Exception:
            continue
        if df.empty or REGION_METRIC not in df.columns:
            continue

        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().astype(str).unique().tolist())
            df = df[df["revision"].astype(str) == rev]
        else:
            rev = str(df["revision"].iloc[0]) if "revision" in df.columns else "main"

        if "region" in df.columns:
            agg = df.groupby("region")[REGION_METRIC].mean().reset_index()
        else:
            continue

        for _, r in agg.iterrows():
            rows.append({
                "family":         family,
                "model_id":       model_id,
                "label":          m.get("label", model_id.split("/")[-1]),
                "n_params":       m.get("n_params", np.nan),
                "revision":       rev,
                "region":         r["region"],
                REGION_METRIC:    r[REGION_METRIC],
            })

    if not rows:
        return pd.DataFrame()
    return pd.concat([pd.DataFrame([r]) for r in rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Data loading — checkpoint training curves
# ---------------------------------------------------------------------------

def load_training_curves(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """Reads broderick_checkpoint_scores.csv for all checkpoints (all steps)."""
    rows: list[dict] = []

    for scores_csv in sorted(sweep_dir.glob("*/*/broderick_checkpoint_scores.csv")):
        family = scores_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = scores_csv.parent.name
        model_id  = model_dir.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir, {}))

        try:
            df = pd.read_csv(scores_csv)
        except Exception:
            continue
        if df.empty or "step" not in df.columns or SCORE_METRIC not in df.columns:
            continue

        df = df.copy()
        df["family"]   = family
        df["model_id"] = model_id
        df["label"]    = m.get("label", model_id.split("/")[-1])
        df["n_params"] = m.get("n_params", np.nan)
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Scaling-law fit
# ---------------------------------------------------------------------------

def _fit_log_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    lx = np.log10(x[mask])
    yy = y[mask]
    coeffs = np.polyfit(lx, yy, 1)
    y_hat = coeffs[0] * lx + coeffs[1]
    ss_res = float(np.sum((yy - y_hat) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coeffs[0]), float(coeffs[1]), float(r2)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_scaling_law(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in [
        (axes[0], SCORE_METRIC,  "Mean best-layer Pearson r (all channels)", "Broderick EEG — scaling law (best layer)"),
        (axes[1], FLAT_METRIC,   "Mean last-layer Pearson r (all channels)", "Broderick EEG — scaling law (last layer)"),
    ]:
        plot_df = df.dropna(subset=["n_params", metric]).copy()
        if plot_df.empty:
            ax.set_title(f"{title} (no data)")
            continue

        for family, grp in plot_df.groupby("family"):
            color  = FAMILY_COLORS.get(family, "gray")
            marker = FAMILY_MARKERS.get(family, "o")
            grp_s  = grp.sort_values("n_params")
            # Single plot() call so markers and connecting line share the same points
            ax.plot(grp_s["n_params"], grp_s[metric],
                    marker=marker, color=color, lw=1.5, ms=7,
                    label=family, zorder=3, alpha=0.9)
            if len(grp_s) > 1:
                slope, intercept, r2 = _fit_log_linear(grp_s["n_params"].values, grp_s[metric].values)
                if np.isfinite(slope):
                    xs = np.logspace(np.log10(grp_s["n_params"].min()),
                                     np.log10(grp_s["n_params"].max()), 100)
                    ax.plot(xs, slope * np.log10(xs) + intercept,
                            color=color, lw=1.0, ls="--", alpha=0.5)

            for _, r in grp_s.iterrows():
                ax.annotate(r["label"], (r["n_params"], r[metric]),
                            fontsize=6, alpha=0.8, xytext=(3, 3), textcoords="offset points")

        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_layer_profiles(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    families = df["family"].unique()
    n_fam = len(families)
    ncols = min(4, n_fam)
    nrows = int(np.ceil(n_fam / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for ax_flat, family in zip(axes.flat, sorted(families)):
        grp = df[df["family"] == family].sort_values("n_params")
        color = FAMILY_COLORS.get(family, "gray")
        for model_id, mgrp in grp.groupby("model_id", sort=False):
            label   = mgrp["label"].iloc[0]
            n_param = mgrp["n_params"].iloc[0]
            alpha   = 0.4 + 0.6 * (np.log10(max(n_param, 1e6)) - 6) / 6 if np.isfinite(n_param) else 0.7
            alpha   = float(np.clip(alpha, 0.3, 1.0))
            mgrp_s  = mgrp.sort_values("layer_rel")
            ax_flat.plot(mgrp_s["layer_rel"], mgrp_s[LAYER_METRIC],
                         color=color, alpha=alpha, lw=1.5, label=label)
        ax_flat.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax_flat.set_title(family, fontsize=10)
        ax_flat.set_xlabel("Relative layer depth")
        ax_flat.set_ylabel("Mean Pearson r")
        ax_flat.legend(fontsize=6, ncol=1)
        ax_flat.grid(True, alpha=0.3)

    for ax_flat in axes.flat[n_fam:]:
        ax_flat.set_visible(False)

    fig.suptitle("Broderick EEG — layer profiles (best-of-all-layers analysis)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_timecourses(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    families = df["family"].unique()
    n_fam = len(families)
    ncols = min(4, n_fam)
    nrows = int(np.ceil(n_fam / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for ax_flat, family in zip(axes.flat, sorted(families)):
        grp = df[df["family"] == family]
        color = FAMILY_COLORS.get(family, "gray")
        for model_id, mgrp in grp.groupby("model_id"):
            label   = mgrp["label"].iloc[0]
            n_param = mgrp["n_params"].iloc[0]
            alpha   = 0.4 + 0.6 * (np.log10(max(n_param, 1e6)) - 6) / 6 if np.isfinite(n_param) else 0.7
            alpha   = float(np.clip(alpha, 0.3, 1.0))
            mgrp_s  = mgrp.sort_values("time_s")
            ax_flat.plot(mgrp_s["time_s"], mgrp_s[TC_METRIC],
                         color=color, alpha=alpha, lw=1.5, label=label)
        ax_flat.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax_flat.axvline(0, color="k", lw=0.5, ls=":", alpha=0.4)
        ax_flat.set_title(family, fontsize=10)
        ax_flat.set_xlabel("Time (s)")
        ax_flat.set_ylabel("Mean Pearson r")
        ax_flat.legend(fontsize=6, ncol=1)
        ax_flat.grid(True, alpha=0.3)

    for ax_flat in axes.flat[n_fam:]:
        ax_flat.set_visible(False)

    fig.suptitle("Broderick EEG — mean TRF timecourse (all channels, best layer)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_region_bars(df: pd.DataFrame, out_path: Path, summary_df: pd.DataFrame) -> None:
    if df.empty:
        return
    regions = [r for r in SCALP_REGIONS if r in df["region"].unique()]
    if not regions:
        return

    all_models = (
        summary_df.dropna(subset=["n_params"])
        .sort_values(["family", "n_params"])
        [["model_id", "label", "family", "n_params"]]
        .drop_duplicates("model_id")
    )
    if all_models.empty:
        return

    n_models = len(all_models)
    x = np.arange(n_models)
    width = 0.8 / len(regions)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 0.6), 5))
    for i, region in enumerate(regions):
        reg_df = df[df["region"] == region].set_index("model_id")
        vals = [reg_df.loc[mid, REGION_METRIC] if mid in reg_df.index else np.nan
                for mid in all_models["model_id"]]
        ax.bar(x + i * width, vals, width, label=region, alpha=0.8)

    ax.set_xticks(x + width * (len(regions) - 1) / 2)
    ax.set_xticklabels(all_models["label"].tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"Mean best-layer Pearson r")
    ax.set_title("Broderick EEG — brain score by scalp region")
    ax.legend(title="Region")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_training_curves(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, grp in df.groupby("family"):
        if grp["step"].nunique() <= 1:
            continue
        color = FAMILY_COLORS.get(family, "gray")
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_id, mgrp in grp.groupby("model_id"):
            label   = mgrp["label"].iloc[0]
            n_param = mgrp["n_params"].iloc[0]
            alpha   = 0.4 + 0.6 * (np.log10(max(n_param, 1e6)) - 6) / 6 if np.isfinite(n_param) else 0.7
            alpha   = float(np.clip(alpha, 0.3, 1.0))
            mgrp_s  = mgrp.sort_values("step")
            ax.plot(mgrp_s["step"], mgrp_s[SCORE_METRIC],
                    "o-", color=color, alpha=alpha, lw=1.5, label=label)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean best-layer Pearson r")
        ax.set_title(f"Broderick EEG — {family} training dynamics")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"{family}_training_curves.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (no data)")
        return
    cols = ["family", "label", "n_params", SCORE_METRIC, FLAT_METRIC, "best_layer", "n_subjects", "n_channels"]
    cols = [c for c in cols if c in df.columns]
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 160)
    print(df[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep_dir", default="outputs_jeanzay/broderick/embedding_sweep",
                   help="Root directory of the Broderick checkpoint sweep outputs")
    p.add_argument("--out_dir", default="outputs/broderick/summary",
                   help="Where to write summary CSV and plots")
    p.add_argument("--families", nargs="+", default=None,
                   help="Restrict to specific model families")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sweep_dir.exists():
        print(f"ERROR: sweep_dir not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Summarizing Broderick results from: {sweep_dir}")
    print(f"Output directory: {out_dir}")

    meta = _registry_meta()

    # --- Final checkpoint scores ---
    print("\n[1/5] Loading final checkpoint scores ...")
    summary_df = load_all_models(sweep_dir, args.families, meta)
    if summary_df.empty:
        print("  No broderick_checkpoint_scores.csv files found.")
    else:
        out_csv = out_dir / "all_models_broderick_scores.csv"
        summary_df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}  ({len(summary_df)} models)")
        print_summary_table(summary_df)

    # --- Scaling law ---
    print("\n[2/5] Plotting scaling law ...")
    if not summary_df.empty:
        plot_scaling_law(summary_df, out_dir / "scaling_law.png")

    # --- Layer profiles ---
    print("\n[3/5] Loading and plotting layer profiles ...")
    layer_df = load_layerwise(sweep_dir, args.families, meta)
    if not layer_df.empty:
        plot_layer_profiles(layer_df, out_dir / "layer_profiles.png")
    else:
        print("  No layerwise data found.")

    # --- TRF timecourses ---
    print("\n[4/5] Loading and plotting TRF timecourses ...")
    tc_df = load_timecourses(sweep_dir, args.families, meta)
    if not tc_df.empty:
        plot_timecourses(tc_df, out_dir / "timecourse.png")
    else:
        print("  No timecourse data found.")

    # --- Region bars ---
    print("\n[5/5] Loading and plotting region scores ...")
    region_df = load_region_scores(sweep_dir, args.families, meta)
    if not region_df.empty and not summary_df.empty:
        plot_region_bars(region_df, out_dir / "region_bars.png", summary_df)
    else:
        print("  No region data found.")

    # --- Training curves (checkpoint sweep) ---
    print("\nTraining curves ...")
    curve_df = load_training_curves(sweep_dir, args.families, meta)
    if not curve_df.empty and curve_df["step"].nunique() > 1:
        plot_training_curves(curve_df, out_dir / "training_curves")
    else:
        print("  No multi-checkpoint families found (only final checkpoints).")

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
