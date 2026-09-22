#!/usr/bin/env python3
"""Summarize Tuckute et al. (2023) fMRI encoding results across model families.

Reads ``tuckute_scores.csv`` (summary) and ``tuckute_by_roi.csv`` (per-ROI/layer)
produced by ``scripts/run/embedding/neural/tuckute.py``.

Conditions
----------
  B   — 1,000 naturalistic baseline sentences (NC-normalizable)
  D   — 500 sentences that drive   LH language activity
  S   — 500 sentences that suppress LH language activity
  all — all 2,000 sentences combined

Primary metric: mean best-layer Spearman r across 6 language ROIs.

Outputs (default ``outputs/tuckute/summary/``):
  all_models_scores.csv
  scaling_law.png            — n_params vs brain score (one panel per condition)
  scaling_by_perplexity.png  — perplexity vs brain score
  roi_profile.png            — per-ROI bar chart (one panel per condition)
  layer_profiles.png         — per-family layer curves (LANG ROIs mean)

Usage
-----
  python scripts/summarize/tuckute.py
  python scripts/summarize/tuckute.py \\
      --sweep_dir outputs_jeanzay/tuckute/embedding_sweep \\
      --out_dir   outputs/tuckute/summary
  python scripts/summarize/tuckute.py --families gpt2 pythia
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from neuroai.datasets.tuckute import LANG_ROIS, CONDITIONS

EVAL_CONDITIONS: list[str] = ["B", "D", "S"]
SUMMARY_METRIC  = "mean_best_spearmanr"  # primary; full col = f"{cond}_{SUMMARY_METRIC}"

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
    "alibi":   "#aec7e8",
    "stickbreaking": "#ffbb78",
    "forgetting":    "#98df8a",
    "babylm":        "#c5b0d5",
}

FAMILY_MARKERS = {
    "gpt2": "o", "gptneo": "s", "opt": "^", "llama32": "D",
    "mamba": "v", "mamba2": "p", "mistral": "h", "qwen3": "*",
    "pythia": "P", "olmo2": "X", "cbr_rnn": "H",
    "alibi": "<", "stickbreaking": ">", "forgetting": "d", "babylm": "8",
}

COND_LABELS = {"B": "Baseline (B)", "D": "Drive (D)", "S": "Suppress (S)"}


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

    main_re   = re.compile(r'_main_only\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
    spec_re   = re.compile(r'LppModelSpec\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
    var_re    = re.compile(r'LppModelSpec\(\s*([A-Z][A-Z0-9_]+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)

    def _add(mid: str, label: str, n_layers: str, n_params: str, fam: str) -> None:
        e = {"family": fam, "label": label, "n_params": float(n_params), "n_layers": int(n_layers)}
        meta[mid] = e
        meta[mid.replace("/", "_")] = e

    for fam, block in fam_blocks:
        for pat in (main_re, spec_re):
            for m in pat.finditer(block):
                _add(m.group(1), m.group(2), m.group(3), m.group(4), fam)
        for m in var_re.finditer(block):
            mid = var_map.get(m.group(1))
            if mid:
                _add(mid, m.group(2), m.group(3), m.group(4), fam)
    return meta


def _revision_step(rev: str) -> int:
    m = re.search(r"step(\d+)", str(rev))
    return int(m.group(1)) if m else (0 if str(rev) == "main" else -1)


def _pick_final_revision(revisions: list[str]) -> str:
    return max(revisions, key=_revision_step)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_models(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """One row per model (final checkpoint): condition scores + metadata."""
    rows: list[dict] = []

    for scores_csv in sorted(sweep_dir.glob("*/*/tuckute_scores.csv")):
        family   = scores_csv.parent.parent.name
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

        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df  = df[df["revision"] == rev]
        else:
            rev = str(df.get("revision", pd.Series(["main"])).iloc[0])

        m = meta.get(model_id, meta.get(model_dir, {}))
        row: dict = {
            "family":   family,
            "model_id": model_id,
            "label":    m.get("label", model_id.split("/")[-1]),
            "n_params": m.get("n_params", np.nan),
            "n_layers": m.get("n_layers", np.nan),
            "revision": rev,
        }
        for cond in EVAL_CONDITIONS:
            col = f"{cond}_{SUMMARY_METRIC}"
            if col in df.columns:
                row[col] = float(df[col].iloc[0])

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["family", "n_params"]).reset_index(drop=True)


def load_layerwise(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """Per-(model, cond, layer) mean Pearson r at the final revision."""
    rows: list[pd.DataFrame] = []

    for lw_csv in sorted(sweep_dir.glob("*/*/tuckute_layerwise.csv")):
        family   = lw_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = lw_csv.parent.name
        model_id  = model_dir.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir, {}))

        try:
            df = pd.read_csv(lw_csv)
        except Exception:
            continue
        if df.empty or "spearmanr_mean" not in df.columns:
            continue

        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df  = df[df["revision"] == rev]

        df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
        df = df.dropna(subset=["layer"])
        df["layer"] = df["layer"].astype(int)

        n_layers_total = int(df["layer"].max()) + 1
        df["layer_rel"] = df["layer"] / max(1, n_layers_total - 1)
        df["family"]    = family
        df["model_id"]  = model_id
        df["label"]     = m.get("label", model_id.split("/")[-1])
        df["n_params"]  = m.get("n_params", np.nan)
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_perplexity(sweep_dir: Path) -> pd.DataFrame:
    """Load corpus-level perplexity from perplexity.json files co-located with model results.

    The sweep script saves <sweep_dir>/<family>/<model_safe>/perplexity.json
    with corpus='tuckute' so we can distinguish it from naturalstories JSONs.
    """
    import json
    rows = []
    for p in sorted(sweep_dir.rglob("perplexity.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "error" in d or "perplexity" not in d:
            continue
        if d.get("corpus", "tuckute") not in ("tuckute", ""):
            continue
        rows.append({"model_id": d.get("model_id", ""), "perplexity": float(d["perplexity"])})
    if not rows:
        return pd.DataFrame(columns=["model_id", "perplexity"])
    return pd.DataFrame(rows).drop_duplicates("model_id")


# ---------------------------------------------------------------------------
# Fit helper
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

def _scatter_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_log: bool = True,
    x_invert: bool = False,
    x_label: str = "",
    y_label: str = "Mean best-layer Spearman r",
    title: str = "",
) -> None:
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        ax.set_title(f"{title} (no data)")
        return

    for family, grp in plot_df.groupby("family"):
        color  = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        grp_s  = grp.sort_values(x_col)
        ax.scatter(grp_s[x_col], grp_s[y_col],
                   label=family, color=color, marker=marker, s=70, zorder=3, alpha=0.9)
        if len(grp_s) > 1:
            ax.plot(grp_s[x_col], grp_s[y_col], color=color, lw=0.8, alpha=0.4)
        for _, row in grp_s.iterrows():
            ax.annotate(row["label"], (row[x_col], row[y_col]),
                        fontsize=5, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points")

    a, b, r2 = _fit_log_linear(plot_df[x_col].to_numpy(float), plot_df[y_col].to_numpy(float))
    if np.isfinite(a):
        x_min, x_max = plot_df[x_col].min(), plot_df[x_col].max()
        x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 200) if x_log else np.linspace(x_min, x_max, 200)
        y_fit = a * np.log10(x_fit) + b if x_log else a * x_fit + b
        ax.plot(x_fit, y_fit, "k--", lw=1.2, alpha=0.5, label=f"fit R²={r2:.3f}")

    if x_log:
        ax.set_xscale("log")
    if x_invert:
        ax.invert_xaxis()
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.25)


def plot_scaling_law(df: pd.DataFrame, out_dir: Path) -> None:
    """4 panels: B / D / S / all — brain score vs n_params."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, cond in zip(axes, EVAL_CONDITIONS):
        col = f"{cond}_{SUMMARY_METRIC}"
        _scatter_ax(ax, df, "n_params", col,
                    x_label="Number of parameters",
                    title=COND_LABELS[cond])
    fig.suptitle("Tuckute fMRI — Brain score vs model size", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "scaling_law.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_scaling_by_perplexity(df: pd.DataFrame, out_dir: Path) -> None:
    """4 panels: B / D / S / all — brain score vs perplexity (inverted x)."""
    if "perplexity" not in df.columns or df["perplexity"].isna().all():
        print("  [skip] no perplexity data for perplexity scaling plot")
        return

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, cond in zip(axes, EVAL_CONDITIONS):
        col = f"{cond}_{SUMMARY_METRIC}"
        _scatter_ax(ax, df, "perplexity", col,
                    x_log=True, x_invert=True,
                    x_label="Perplexity on Natural Stories  ← better LM",
                    title=COND_LABELS[cond])
    fig.suptitle("Tuckute fMRI — Brain score vs perplexity", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "scaling_by_perplexity.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_roi_profile(df_roi: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart per condition: mean best-layer Spearman r per ROI."""
    if df_roi.empty or "spearmanr" not in df_roi.columns:
        print("  [skip] no per-ROI data")
        return

    # best layer per (model, cond, roi) by spearmanr, then mean over models
    best = (
        df_roi.groupby(["model_id", "cond", "roi"])["spearmanr"]
        .max()
        .reset_index()
        .groupby(["cond", "roi"])["spearmanr"]
        .mean()
        .reset_index()
    )

    n_conds = len(EVAL_CONDITIONS)
    fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 4), sharey=False)
    roi_order = LANG_ROIS

    for ax, cond in zip(axes, EVAL_CONDITIONS):
        cdf = best[best["cond"] == cond].set_index("roi").reindex(roi_order)
        ax.bar(range(len(roi_order)), cdf["spearmanr"].fillna(0),
               color="#2166ac", alpha=0.85)
        ax.set_xticks(range(len(roi_order)))
        ax.set_xticklabels([r.replace("lang_LH_", "") for r in roi_order],
                            rotation=40, ha="right", fontsize=8)
        ax.set_title(COND_LABELS[cond], fontsize=10)
        ax.set_ylabel("Mean best-layer Spearman r", fontsize=8)
        ax.axhline(0, color="k", lw=0.7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Tuckute fMRI — Per-ROI profile (mean over models)", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "roi_profile.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_layer_profiles(lw_df: pd.DataFrame, out_dir: Path) -> None:
    """Per-family subplots; one line per model × condition."""
    if lw_df.empty:
        print("  [skip] no layerwise data")
        return

    COND_COLORS = {"B": "#1f77b4", "D": "#d62728", "S": "#2ca02c"}

    families = sorted(lw_df["family"].unique())
    ncols = min(3, len(families))
    nrows = (len(families) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4),
                              squeeze=False, sharey=False)

    for i, fam in enumerate(families):
        ax = axes[i // ncols][i % ncols]
        fam_df = lw_df[lw_df["family"] == fam]
        model_ids = (
            fam_df.drop_duplicates("model_id")
            .sort_values("n_params")["model_id"].tolist()
        )
        for mi, mid in enumerate(model_ids):
            mdf = fam_df[fam_df["model_id"] == mid]
            linestyle = ["-", "--", ":", "-."][mi % 4]
            for cond in ["B", "D", "S"]:
                cdf = mdf[mdf["cond"] == cond].sort_values("layer_rel")
                if cdf.empty:
                    continue
                ax.plot(cdf["layer_rel"], cdf["spearmanr_mean"],
                        color=COND_COLORS[cond], lw=1.4, linestyle=linestyle, alpha=0.85)

        legend_handles = [
            plt.Line2D([0], [0], color=COND_COLORS[c], lw=2, label=c)
            for c in ["B", "D", "S"]
        ]
        ax.legend(handles=legend_handles, fontsize=7, loc="upper left", framealpha=0.7)
        ax.set_title(fam, fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative layer depth", fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel("Mean Spearman r", fontsize=8)
        ax.grid(True, alpha=0.25)

    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Tuckute fMRI — Layer profiles by family (colors = condition)", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "layer_profiles.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_b_vs_ds(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: B score vs D score and B score vs S score (one panel each)."""
    b_col = f"B_{SUMMARY_METRIC}"
    d_col = f"D_{SUMMARY_METRIC}"
    s_col = f"S_{SUMMARY_METRIC}"
    if b_col not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (other_col, other_label) in zip(
        axes, [(d_col, "Drive (D)"), (s_col, "Suppress (S)")]
    ):
        plot_df = df.dropna(subset=[b_col, other_col])
        if plot_df.empty:
            ax.set_visible(False)
            continue
        for family, grp in plot_df.groupby("family"):
            color  = FAMILY_COLORS.get(family, "gray")
            marker = FAMILY_MARKERS.get(family, "o")
            ax.scatter(grp[b_col], grp[other_col],
                       color=color, marker=marker, s=70, alpha=0.9, label=family)
            for _, row in grp.iterrows():
                ax.annotate(row["label"], (row[b_col], row[other_col]),
                            fontsize=5, ha="left", va="bottom",
                            xytext=(2, 2), textcoords="offset points")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.5, label="B = other")
        ax.set_xlabel("Baseline (B) Spearman r", fontsize=9)
        ax.set_ylabel(f"{other_label} Spearman r", fontsize=9)
        ax.set_title(f"B vs {other_label}", fontsize=10)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Tuckute fMRI — Baseline vs drive/suppress", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "b_vs_ds.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep_dir", type=Path,
                   default=Path("outputs_jeanzay/tuckute/embedding_sweep"))
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--families", nargs="+", default=None)
    args = p.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    out_dir   = (args.out_dir or Path("outputs/tuckute/summary")).resolve()

    if not sweep_dir.is_dir():
        sys.exit(f"sweep_dir not found: {sweep_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from : {sweep_dir}")
    print(f"Writing to   : {out_dir}\n")

    meta   = _registry_meta()
    df     = load_all_models(sweep_dir, args.families, meta)
    lw_df  = load_layerwise(sweep_dir, args.families, meta)

    if df.empty:
        sys.exit("No tuckute_scores.csv files found — nothing to summarize.")

    # ── perplexity merge (loaded from sweep_dir itself — co-located with results)
    ppl_df = load_perplexity(sweep_dir)
    if not ppl_df.empty:
        df = df.merge(ppl_df[["model_id", "perplexity"]], on="model_id", how="left")
        print(f"  Perplexity: {df['perplexity'].notna().sum()}/{len(df)} models")

    # ── print table ────────────────────────────────────────────────────────
    print(f"Found {len(df)} models across {df['family'].nunique()} families:\n")
    header = f"  {'family':<13} {'label':<28} {'B':>7}  {'D':>7}  {'S':>7}"
    print(header)
    print("  " + "-" * len(header))
    for _, row in df.iterrows():
        def _f(v: float) -> str:
            return f"{v:>7.4f}" if np.isfinite(v) else f"{'n/a':>7}"
        print(
            f"  {row['family']:<13} {row['label']:<28} "
            f"{_f(row.get(f'B_{SUMMARY_METRIC}', np.nan))}  "
            f"{_f(row.get(f'D_{SUMMARY_METRIC}', np.nan))}  "
            f"{_f(row.get(f'S_{SUMMARY_METRIC}', np.nan))}"
        )
    print()

    out_csv = out_dir / "all_models_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}\n")

    # ── load per-ROI CSV for roi_profile plot ─────────────────────────────
    roi_rows: list[pd.DataFrame] = []
    for roi_csv in sorted(sweep_dir.glob("*/*/tuckute_by_roi.csv")):
        family = roi_csv.parent.parent.name
        if args.families and family not in args.families:
            continue
        try:
            rdf = pd.read_csv(roi_csv)
            if not rdf.empty:
                roi_rows.append(rdf)
        except Exception:
            pass
    df_roi = pd.concat(roi_rows, ignore_index=True) if roi_rows else pd.DataFrame()

    # ── plots ─────────────────────────────────────────────────────────────
    plot_scaling_law(df, out_dir)
    plot_scaling_by_perplexity(df, out_dir)
    plot_roi_profile(df_roi, out_dir)
    plot_layer_profiles(lw_df, out_dir)
    plot_b_vs_ds(df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
