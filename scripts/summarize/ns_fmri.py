#!/usr/bin/env python3
"""Summarize Natural Stories fMRI encoding results across model families.

Two modes (``--checkpoints`` selects checkpoint-sweep summarization):

**Final checkpoint** (default) — brain score from the last training step:
  Reads ``ns_fmri_by_roi.csv`` from each model directory.

**Checkpoint sweep** (``--checkpoints``) — training dynamics for staged sweeps:
  Reads ``ns_fmri_checkpoint_scores.csv`` from
  ``<sweep_dir>/<family>/<model>/``.

Networks
--------
  LANG = 6 left-hemisphere language parcels (Fedorenko lab)
  MD   = 20 bilateral multiple-demand parcels

Primary metric: Spearman r (best-layer).
Key contrast  : LANG − MD (language specificity).

Outputs (default ``outputs/ns_fmri/summary/``):
  all_models_scores.csv, scaling_law.png, layer_profiles.png,
  froi_profile.png, lang_vs_md.png

Usage
-----
  python scripts/summarize/ns_fmri.py
  python scripts/summarize/ns_fmri.py \\
      --sweep_dir outputs_jeanzay/ns_fmri/embedding_sweep \\
      --out_dir   outputs/ns_fmri/summary
  python scripts/summarize/ns_fmri.py --families gpt2 pythia

  python scripts/summarize/ns_fmri.py --checkpoints \\
      --sweep_dir outputs_jeanzay/ns_fmri/embedding_sweep
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NETWORKS = ["LANG", "MD"]
NETWORK_COLORS = {"LANG": "#2166ac", "MD": "#d6604d"}
NETWORK_MARKERS = {"LANG": "o", "MD": "s"}

LANG_FROIS = [
    "LangLAngG", "LangLAntTemp", "LangLIFG",
    "LangLIFGorb", "LangLMFG", "LangLPostTemp",
]

SUMMARY_METRIC = "mean_best_spearmanr"
ELEC_METRIC = "best_spearmanr"

MIN_CHECKPOINT_STEPS = 2

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
}

FAMILY_MARKERS = {
    "gpt2": "o", "gptneo": "s", "opt": "^", "llama32": "D",
    "mamba": "v", "mamba2": "p", "mistral": "h", "qwen3": "*",
    "pythia": "P", "olmo2": "X", "cbr_rnn": "H",
    "alibi": "<", "stickbreaking": ">", "forgetting": "d",
}


# ---------------------------------------------------------------------------
# Registry helpers (shared with other summarize scripts)
# ---------------------------------------------------------------------------

def _registry_meta() -> dict[str, dict]:
    reg_path = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "neuroai" / "models" / "registry.py"
    text = reg_path.read_text(encoding="utf-8")

    var_map: dict[str, str] = {}
    for m in re.finditer(r'^([A-Z][A-Z0-9_]+)\s*=\s*"([^"]+)"', text, flags=re.M):
        var_map[m.group(1)] = m.group(2)

    meta: dict[str, dict] = {}
    fam_blocks = re.findall(r'"([^"]+)":\s*\[(.*?)\],', text, flags=re.S)

    main_re = re.compile(
        r'_main_only\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)',
        flags=re.S,
    )
    spec_lit_re = re.compile(
        r'LppModelSpec\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)',
        flags=re.S,
    )
    spec_var_re = re.compile(
        r'LppModelSpec\(\s*([A-Z][A-Z0-9_]+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)',
        flags=re.S,
    )

    def _add(model_id: str, label: str, n_layers: str, n_params: str, fam: str) -> None:
        entry = {"family": fam, "label": label,
                 "n_params": float(n_params), "n_layers": int(n_layers)}
        meta[model_id] = entry
        meta[model_id.replace("/", "_")] = entry

    for fam, block in fam_blocks:
        for pattern in (main_re, spec_lit_re):
            for m in pattern.finditer(block):
                _add(m.group(1), m.group(2), m.group(3), m.group(4), fam)
        for m in spec_var_re.finditer(block):
            model_id = var_map.get(m.group(1))
            if model_id:
                _add(model_id, m.group(2), m.group(3), m.group(4), fam)
    return meta


# ---------------------------------------------------------------------------
# Revision helpers
# ---------------------------------------------------------------------------

def _revision_step(rev: str) -> int:
    m = re.search(r"step(\d+)", str(rev))
    return int(m.group(1)) if m else (0 if str(rev) == "main" else -1)


def _pick_final_revision(revisions: list[str]) -> str:
    return max(revisions, key=_revision_step)


# ---------------------------------------------------------------------------
# Data loading — final checkpoint mode
# ---------------------------------------------------------------------------

def load_all_models(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """One row per model: overall / LANG / MD / LANG−MD best-layer Spearman r."""
    rows: list[dict] = []

    for roi_csv in sorted(sweep_dir.glob("*/*/ns_fmri_by_roi.csv")):
        family = roi_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = roi_csv.parent.name
        model_id = model_dir.replace("_", "/", 1)

        try:
            df = pd.read_csv(roi_csv)
        except Exception:
            continue
        if df.empty or ELEC_METRIC not in df.columns:
            continue

        if "revision" in df.columns and df["revision"].nunique() > 1:
            revision = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == revision]
        else:
            revision = str(df["revision"].iloc[0]) if "revision" in df.columns else "main"

        m = meta.get(model_id, meta.get(model_dir, {}))
        row: dict = {
            "family": family,
            "model_id": model_id,
            "label": m.get("label", model_id.split("/")[-1]),
            "n_params": m.get("n_params", np.nan),
            "n_layers": m.get("n_layers", np.nan),
            "revision": revision,
            "n_subjects": int(df["subject"].nunique()) if "subject" in df.columns else 0,
            "n_rois": int(df["froi"].nunique()) if "froi" in df.columns else 0,
            SUMMARY_METRIC: float(df[ELEC_METRIC].mean()),
        }

        if "network" in df.columns:
            for net in NETWORKS:
                ndf = df[df["network"] == net]
                row[f"{net}_{SUMMARY_METRIC}"] = float(ndf[ELEC_METRIC].mean()) if not ndf.empty else np.nan
            lang_v = row.get(f"LANG_{SUMMARY_METRIC}", np.nan)
            md_v   = row.get(f"MD_{SUMMARY_METRIC}",   np.nan)
            row["lang_minus_md_spearmanr"] = (
                lang_v - md_v if np.isfinite(lang_v) and np.isfinite(md_v) else np.nan
            )

        if "froi" in df.columns:
            for froi, gdf in df.groupby("froi"):
                row[f"{froi}_spearmanr"] = float(gdf[ELEC_METRIC].mean())

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["family", "n_params"]).reset_index(drop=True)


def load_layerwise(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """Per-(model, network, layer) mean Spearman r at the final revision."""
    rows: list[pd.DataFrame] = []

    for lw_csv in sorted(sweep_dir.glob("*/*/ns_fmri_checkpoint_layerwise.csv")):
        family = lw_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = lw_csv.parent.name
        model_id = model_dir.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir, {}))

        try:
            df = pd.read_csv(lw_csv)
        except Exception:
            continue
        if df.empty or "spearmanr" not in df.columns:
            continue

        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == rev]
        else:
            rev = str(df["revision"].iloc[0]) if "revision" in df.columns else "main"

        group_cols = [c for c in ["network", "layer"] if c in df.columns]
        if not group_cols:
            continue

        df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
        df = df.dropna(subset=["layer"])
        df["layer"] = df["layer"].astype(int)

        agg = df.groupby(group_cols)["spearmanr"].mean().reset_index()
        agg.rename(columns={"spearmanr": "mean_spearmanr"}, inplace=True)
        n_layers_total = int(agg["layer"].max()) + 1
        agg["layer_rel"] = agg["layer"] / max(1, n_layers_total - 1)
        agg["family"]    = family
        agg["model_id"]  = model_id
        agg["label"]     = m.get("label", model_id.split("/")[-1])
        agg["n_params"]  = m.get("n_params", np.nan)
        agg["revision"]  = rev
        rows.append(agg)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Perplexity loading
# ---------------------------------------------------------------------------

def load_perplexity(ppl_dir: Path) -> pd.DataFrame:
    """Load naturalstories perplexity.json files → DataFrame with model_id + perplexity."""
    rows: list[dict] = []
    for p in sorted(ppl_dir.rglob("perplexity.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "error" in d or "perplexity" not in d:
            continue
        rows.append({
            "model_id":   d.get("model_id", ""),
            "perplexity": float(d["perplexity"]),
            "n_tokens":   d.get("n_tokens", np.nan),
        })
    if not rows:
        return pd.DataFrame(columns=["model_id", "perplexity"])
    return pd.DataFrame(rows).drop_duplicates("model_id")


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
# Plot helpers
# ---------------------------------------------------------------------------

def _scaling_law_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    plot_df = df.dropna(subset=["n_params", metric]).copy()
    if plot_df.empty:
        ax.set_title(f"{title} (no data)")
        return

    for family, grp in plot_df.groupby("family"):
        color = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        grp_s = grp.sort_values("n_params")
        ax.scatter(grp_s["n_params"], grp_s[metric],
                   label=family, color=color, marker=marker, s=70, zorder=3, alpha=0.9)
        if len(grp_s) > 1:
            ax.plot(grp_s["n_params"], grp_s[metric], color=color, lw=0.8, alpha=0.4)
        for _, row in grp_s.iterrows():
            ax.annotate(row["label"], (row["n_params"], row[metric]),
                        fontsize=5, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points")

    a, b, r2 = _fit_log_linear(
        plot_df["n_params"].to_numpy(float),
        plot_df[metric].to_numpy(float),
    )
    if np.isfinite(a):
        x_fit = np.logspace(
            np.log10(plot_df["n_params"].min()),
            np.log10(plot_df["n_params"].max()), 200,
        )
        ax.plot(x_fit, a * np.log10(x_fit) + b, "k--", lw=1.2, alpha=0.5,
                label=f"fit R²={r2:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters", fontsize=9)
    ax.set_ylabel("Mean best-layer Spearman ρ", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.25)


def plot_scaling_law(df: pd.DataFrame, out_dir: Path) -> None:
    """1×3: Overall / LANG / MD best-layer Spearman r vs model size."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_titles = [
        (SUMMARY_METRIC,            "Overall (LANG + MD)"),
        (f"LANG_{SUMMARY_METRIC}",  "LANG network"),
        (f"MD_{SUMMARY_METRIC}",    "MD network"),
    ]
    for ax, (metric, title) in zip(axes, metrics_titles):
        _scaling_law_ax(ax, df, metric, title)
    fig.suptitle("NS fMRI — Brain score vs model size", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "scaling_law.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def _scaling_ppl_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    plot_df = df.dropna(subset=["perplexity", metric]).copy()
    if plot_df.empty:
        ax.set_title(f"{title} (no data)")
        return

    for family, grp in plot_df.groupby("family"):
        color  = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        grp_s  = grp.sort_values("perplexity")
        ax.scatter(grp_s["perplexity"], grp_s[metric],
                   label=family, color=color, marker=marker, s=70, zorder=3, alpha=0.9)
        if len(grp_s) > 1:
            ax.plot(grp_s["perplexity"], grp_s[metric], color=color, lw=0.8, alpha=0.4)
        for _, row in grp_s.iterrows():
            ax.annotate(row["label"], (row["perplexity"], row[metric]),
                        fontsize=5, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points")

    a, b, r2 = _fit_log_linear(
        plot_df["perplexity"].to_numpy(float),
        plot_df[metric].to_numpy(float),
    )
    if np.isfinite(a):
        x_fit = np.logspace(
            np.log10(plot_df["perplexity"].min()),
            np.log10(plot_df["perplexity"].max()), 200,
        )
        ax.plot(x_fit, a * np.log10(x_fit) + b, "k--", lw=1.2, alpha=0.5,
                label=f"fit R²={r2:.3f}")

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Perplexity on Natural Stories  ← better LM", fontsize=9)
    ax.set_ylabel("Mean best-layer Spearman ρ", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.25)


def plot_scaling_by_perplexity(df: pd.DataFrame, out_dir: Path) -> None:
    """1×3: Overall / LANG / MD brain score vs naturalstories perplexity."""
    ppl_df = df.dropna(subset=["perplexity"])
    if ppl_df.empty:
        print("  [skip] no perplexity data — run ns_perplexity_*.sh first")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_titles = [
        (SUMMARY_METRIC,           "Overall (LANG + MD)"),
        (f"LANG_{SUMMARY_METRIC}", "LANG network"),
        (f"MD_{SUMMARY_METRIC}",   "MD network"),
    ]
    for ax, (metric, title) in zip(axes, metrics_titles):
        _scaling_ppl_ax(ax, ppl_df, metric, title)
    fig.suptitle("NS fMRI — Brain score vs perplexity (Natural Stories)", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "scaling_by_perplexity.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_lang_vs_md(df: pd.DataFrame, out_dir: Path) -> None:
    """LANG vs MD scatter (one dot per model) with identity diagonal."""
    lang_col = f"LANG_{SUMMARY_METRIC}"
    md_col   = f"MD_{SUMMARY_METRIC}"
    plot_df  = df.dropna(subset=[lang_col, md_col])
    if plot_df.empty:
        print("  [skip] no LANG/MD data for lang_vs_md plot")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for family, grp in plot_df.groupby("family"):
        color  = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        ax.scatter(grp[md_col], grp[lang_col],
                   color=color, marker=marker, s=70, zorder=3, alpha=0.9, label=family)
        for _, row in grp.iterrows():
            ax.annotate(row["label"], (row[md_col], row[lang_col]),
                        fontsize=5, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points")

    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="LANG = MD")
    ax.set_xlabel("MD best-layer Spearman ρ", fontsize=9)
    ax.set_ylabel("LANG best-layer Spearman ρ", fontsize=9)
    ax.set_title("NS fMRI — LANG vs MD specificity", fontsize=11)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fname = out_dir / "lang_vs_md.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_froi_profile(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: mean best-layer Spearman r per fROI (LANG in blue, MD in red)."""
    froi_cols = sorted([
        c for c in df.columns
        if c.endswith("_spearmanr")
        and c not in {SUMMARY_METRIC, "LANG_mean_best_spearmanr", "MD_mean_best_spearmanr",
                      "lang_minus_md_spearmanr", f"LANG_{SUMMARY_METRIC}", f"MD_{SUMMARY_METRIC}"}
        and not c.startswith("mean_")
    ])
    if not froi_cols:
        print("  [skip] no per-fROI columns found")
        return

    froi_means = df[froi_cols].mean().sort_values(ascending=False)
    froi_labels = [c.replace("_spearmanr", "") for c in froi_means.index]
    bar_colors  = [
        NETWORK_COLORS["LANG"] if any(f == label for f in LANG_FROIS) else NETWORK_COLORS["MD"]
        for label in froi_labels
    ]

    fig, ax = plt.subplots(figsize=(max(9, len(froi_cols) * 0.55), 4))
    ax.bar(range(len(froi_means)), froi_means.values, color=bar_colors)
    ax.set_xticks(range(len(froi_means)))
    ax.set_xticklabels(froi_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean best-layer Spearman ρ", fontsize=9)
    ax.set_title("NS fMRI — per-fROI profile (mean over models)", fontsize=11)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.legend(handles=[Patch(color=NETWORK_COLORS["LANG"], label="LANG"),
                        Patch(color=NETWORK_COLORS["MD"],   label="MD")],
              fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "froi_profile.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_layer_profiles(lw_df: pd.DataFrame, out_dir: Path) -> None:
    """Per-family subplots; within each: one line per model × network (LANG/MD)."""
    if lw_df.empty:
        print("  [skip] no layerwise data found")
        return

    families = sorted(lw_df["family"].unique())
    ncols = min(3, len(families))
    nrows = (len(families) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4),
                             sharey=False, squeeze=False)

    for i, fam in enumerate(families):
        ax = axes[i // ncols][i % ncols]
        fam_df = lw_df[lw_df["family"] == fam]
        model_ids = (
            fam_df.drop_duplicates("model_id")
            .sort_values("n_params")["model_id"]
            .tolist()
        )
        n_models = max(len(model_ids), 1)

        for mi, mid in enumerate(model_ids):
            mdf = fam_df[fam_df["model_id"] == mid]
            label_base = mdf["label"].iloc[0]
            linestyle = ["-", "--", ":", "-."][mi % 4]

            if "network" in mdf.columns:
                for net in NETWORKS:
                    ndf = mdf[mdf["network"] == net].sort_values("layer_rel")
                    if ndf.empty:
                        continue
                    ax.plot(ndf["layer_rel"], ndf["mean_spearmanr"],
                            color=NETWORK_COLORS[net],
                            lw=1.5 + 0.4 * mi / max(n_models - 1, 1),
                            linestyle=linestyle, alpha=0.85,
                            label=f"{label_base} {net}" if mi == 0 else None)
            else:
                overall = mdf.sort_values("layer_rel")
                ax.plot(overall["layer_rel"], overall["mean_spearmanr"],
                        lw=1.5, linestyle=linestyle, alpha=0.85, label=label_base)

        net_handles = [Line2D([0], [0], color=NETWORK_COLORS[n], lw=2, label=n)
                       for n in NETWORKS]
        ax.legend(handles=net_handles, fontsize=7, loc="upper left", framealpha=0.7)
        ax.set_title(fam, fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative layer depth", fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel("Mean Spearman ρ", fontsize=8)
        ax.grid(True, alpha=0.25)

    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("NS fMRI — Layer profiles by family (colors = network)", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "layer_profiles.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Checkpoint sweep summarization
# ---------------------------------------------------------------------------

def _collect_checkpoint_scores(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for ck_csv in sorted(sweep_dir.glob("*/*/ns_fmri_checkpoint_scores.csv")):
        family = ck_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = ck_csv.parent.name
        model_id = model_dir.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir, {}))

        try:
            df = pd.read_csv(ck_csv)
        except Exception:
            continue
        if df.empty or "revision" not in df.columns or df["revision"].nunique() < MIN_CHECKPOINT_STEPS:
            continue

        df = df.copy()
        df["step"]     = df["revision"].apply(_revision_step)
        df["model_id"] = model_id
        df["label"]    = m.get("label", model_id.split("/")[-1])
        df["family"]   = family
        df["n_params"] = m.get("n_params", np.nan)
        df["n_layers"] = m.get("n_layers", np.nan)

        # Normalise network column names to uppercase
        for src, dst in [("lang_best_spearmanr", "LANG_best_spearmanr"),
                         ("md_best_spearmanr",   "MD_best_spearmanr")]:
            if src in df.columns and dst not in df.columns:
                df.rename(columns={src: dst}, inplace=True)

        rows.append(df)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    sort_cols = [c for c in ("family", "n_params", "model_id", "step") if c in out.columns]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _plot_checkpoint_training_curves(df: pd.DataFrame, out_dir: Path, family: str) -> None:
    """4-panel training curves: Overall / LANG / MD / LANG−MD."""
    panels = [
        ("mean_best_spearmanr",    "Overall (LANG + MD)"),
        ("LANG_best_spearmanr",    "LANG network"),
        ("MD_best_spearmanr",      "MD network"),
        ("lang_minus_md_spearmanr","LANG − MD (specificity)"),
    ]
    available = [(col, title) for col, title in panels if col in df.columns]
    if not available:
        print("  [skip] no Spearman columns in checkpoint scores")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    model_ids = df["model_id"].unique()
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(model_ids), 1)))
    color_map = dict(zip(model_ids, cmap))

    for ax, (col, title) in zip(axes, available):
        for model_id, grp in df.groupby("model_id"):
            g = grp.sort_values("step")
            if g[col].isna().all():
                continue
            label = g["label"].iloc[0]
            final = float(g[col].dropna().iloc[-1])
            ax.plot(g["step"], g[col], "o-", color=color_map[model_id],
                    lw=1.6, alpha=0.9, label=f"{label} ({final:.4f})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training step", fontsize=8)
        ax.set_ylabel("Mean best-layer Spearman ρ", fontsize=8)
        ax.legend(fontsize=6, loc="lower right", framealpha=0.7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"NS fMRI — Training curves ({family})", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "training_curves.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def _plot_checkpoint_scaling(df: pd.DataFrame, out_dir: Path, family: str) -> dict:
    """1×3 scaling-law at the final training step."""
    final_step = int(df["step"].max())
    final_df = df[df["step"] == final_step].copy()

    fit_results: dict[str, dict] = {}
    metrics_titles = [
        ("mean_best_spearmanr",   "Overall"),
        ("LANG_best_spearmanr",   "LANG network"),
        ("MD_best_spearmanr",     "MD network"),
    ]
    available = [(col, title) for col, title in metrics_titles if col in final_df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    color  = FAMILY_COLORS.get(family, "gray")
    marker = FAMILY_MARKERS.get(family, "o")

    for ax, (col, title) in zip(axes, available):
        plot_df = final_df.dropna(subset=["n_params", col])
        for _, row in plot_df.sort_values("n_params").iterrows():
            ax.scatter(row["n_params"], row[col], s=80,
                       color=color, marker=marker, alpha=0.9, zorder=3)
            ax.annotate(row["label"], (row["n_params"], row[col]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 2), textcoords="offset points")

        if len(plot_df) > 1:
            a, b, r2 = _fit_log_linear(plot_df["n_params"].to_numpy(float),
                                        plot_df[col].to_numpy(float))
            if np.isfinite(a):
                x_fit = np.logspace(np.log10(plot_df["n_params"].min()),
                                    np.log10(plot_df["n_params"].max()), 200)
                ax.plot(x_fit, a * np.log10(x_fit) + b, "--", color=color,
                        lw=1.3, alpha=0.6, label=f"R²={r2:.3f}")
                ax.legend(fontsize=7)
                fit_results[col] = {"slope": a, "intercept": b, "r2": r2}

        ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Number of parameters", fontsize=8)
        ax.set_ylabel("Mean best-layer Spearman ρ", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(f"NS fMRI — Scaling law at final step ({family}, step {final_step})", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "scaling_law_final.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fit_results


def _collect_checkpoint_layerwise_final(
    sweep_dir: Path,
    family: str,
    meta: dict[str, dict],
    all_ck_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for lw_csv in sorted(sweep_dir.glob(f"{family}/*/ns_fmri_checkpoint_layerwise.csv")):
        model_dir = lw_csv.parent.name
        model_id = model_dir.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir, {}))
        sub_ck = all_ck_df[all_ck_df["model_id"] == model_id]
        if sub_ck.empty:
            continue
        final_step = int(sub_ck["step"].max())
        final_rev  = str(sub_ck.loc[sub_ck["step"] == final_step, "revision"].iloc[0])

        try:
            df = pd.read_csv(lw_csv)
        except Exception:
            continue
        if df.empty or "spearmanr" not in df.columns:
            continue
        if "revision" in df.columns:
            df = df[df["revision"] == final_rev]
        if df.empty:
            continue

        group_cols = [c for c in ["network", "layer"] if c in df.columns]
        agg = df.groupby(group_cols)["spearmanr"].mean().reset_index()
        agg.rename(columns={"spearmanr": "mean_spearmanr"}, inplace=True)
        n_layers_total = int(agg["layer"].max()) + 1
        agg["layer_rel"] = agg["layer"] / max(1, n_layers_total - 1)
        agg["model_id"]  = model_id
        agg["label"]     = m.get("label", model_id.split("/")[-1])
        agg["n_params"]  = m.get("n_params", np.nan)
        rows.append(agg)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _plot_checkpoint_layer_profiles(lw_df: pd.DataFrame, out_dir: Path) -> None:
    if lw_df.empty:
        print("  [skip] no checkpoint layerwise data")
        return

    model_ids = lw_df["model_id"].unique()
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(model_ids), 1)))
    color_map = dict(zip(model_ids, cmap))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, net in zip(axes, NETWORKS):
        ndf = lw_df[lw_df["network"] == net] if "network" in lw_df.columns else lw_df
        for model_id, grp in ndf.groupby("model_id"):
            g = grp.sort_values("layer_rel")
            ax.plot(g["layer_rel"], g["mean_spearmanr"],
                    color=color_map[model_id], lw=1.6, alpha=0.85,
                    label=g["label"].iloc[0])
        ax.set_title(net, fontsize=10)
        ax.set_xlabel("Relative layer depth", fontsize=8)
        ax.set_ylabel("Mean Spearman ρ", fontsize=8)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
        ax.grid(True, alpha=0.25)

    fig.suptitle("NS fMRI — Layer profiles at final checkpoint (LANG / MD)", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "layer_profiles_final_checkpoint.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def summarize_checkpoint_family(
    sweep_dir: Path,
    family: str,
    out_root: Path,
    meta: dict[str, dict],
) -> pd.DataFrame:
    out_dir = out_root / family
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_checkpoint_scores(sweep_dir, [family], meta)
    if df.empty:
        print(f"[{family}] no multi-checkpoint scores found")
        return df

    csv_path = out_dir / "all_models_ns_fmri_checkpoint_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    _plot_checkpoint_training_curves(df, out_dir, family)
    fit = _plot_checkpoint_scaling(df, out_dir, family)
    (out_dir / "scaling_fit_results.json").write_text(
        json.dumps(fit, indent=2), encoding="utf-8",
    )

    lw_df = _collect_checkpoint_layerwise_final(sweep_dir, family, meta, df)
    _plot_checkpoint_layer_profiles(lw_df, out_dir)

    n_models = df["model_id"].nunique()
    n_steps  = df["step"].nunique()
    print(f"[{family}] {n_models} models, {n_steps} checkpoints")
    return df


def run_checkpoint_summary(
    sweep_dir: Path,
    out_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> None:
    all_df = _collect_checkpoint_scores(sweep_dir, families, meta)
    if all_df.empty:
        sys.exit(
            f"No multi-checkpoint ns_fmri_checkpoint_scores.csv found under {sweep_dir} "
            f"(need ≥{MIN_CHECKPOINT_STEPS} distinct revisions per model)."
        )

    discovered = sorted(all_df["family"].unique())
    print(f"Sweep root   : {sweep_dir}")
    print(f"Found {all_df['model_id'].nunique()} models across {len(discovered)} families:")
    print(f"  {', '.join(discovered)}\n")

    for fam in discovered:
        print(f"[{fam}]")
        summarize_checkpoint_family(sweep_dir, fam, out_dir, meta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoints", action="store_true",
        help="Summarize checkpoint sweeps (ns_fmri_checkpoint_scores.csv) instead of final scores",
    )
    p.add_argument(
        "--sweep_dir", type=Path,
        default=Path("outputs_jeanzay/ns_fmri/embedding_sweep"),
        help="Sweep results root (default: outputs_jeanzay/ns_fmri/embedding_sweep)",
    )
    p.add_argument(
        "--ppl_dir", type=Path,
        default=Path("outputs_jeanzay/naturalstories/perplexity"),
        help="Naturalstories perplexity results dir (default: outputs_jeanzay/naturalstories/perplexity)",
    )
    p.add_argument(
        "--out_dir", type=Path, default=None,
        help="Output directory (default: outputs/ns_fmri/summary[/checkpoint])",
    )
    p.add_argument(
        "--families", nargs="+", default=None,
        help="Restrict to these families (default: all)",
    )
    args = p.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    out_dir = (args.out_dir or (
        Path("outputs/ns_fmri/summary/checkpoint") if args.checkpoints
        else Path("outputs/ns_fmri/summary")
    )).resolve()

    if not sweep_dir.is_dir():
        sys.exit(f"sweep_dir not found: {sweep_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from : {sweep_dir}")
    print(f"Writing to   : {out_dir}")

    meta = _registry_meta()

    if args.checkpoints:
        print("Mode         : checkpoint sweep\n")
        run_checkpoint_summary(sweep_dir, out_dir, args.families, meta)
        print("\nDone.")
        return

    print("Mode         : final checkpoint\n")

    df = load_all_models(sweep_dir, args.families, meta)
    if df.empty:
        sys.exit("No ns_fmri_by_roi.csv files found — nothing to summarize.")

    print(f"Found {len(df)} models across {df['family'].nunique()} families:\n")
    lang_col = f"LANG_{SUMMARY_METRIC}"
    md_col   = f"MD_{SUMMARY_METRIC}"
    header = (
        f"  {'family':<12} {'model':<28} {'subj':>4} {'rois':>4} "
        f"{'overall':>8}  {'LANG':>8}  {'MD':>8}  {'LANG-MD':>8}"
    )
    print(header)
    print("  " + "-" * len(header))
    for _, row in df.iterrows():
        def _fmt(v: float) -> str:
            return f"{v:>8.4f}" if np.isfinite(v) else f"{'n/a':>8}"
        print(
            f"  {row['family']:<12} {row['label']:<28} "
            f"{row['n_subjects']:>4} {row['n_rois']:>4} "
            f"{_fmt(row[SUMMARY_METRIC])}  "
            f"{_fmt(row.get(lang_col, np.nan))}  "
            f"{_fmt(row.get(md_col, np.nan))}  "
            f"{_fmt(row.get('lang_minus_md_spearmanr', np.nan))}"
        )
    print()

    # ── perplexity ──────────────────────────────────────────────────────────
    ppl_dir = args.ppl_dir.resolve()
    if ppl_dir.is_dir():
        ppl_df = load_perplexity(ppl_dir)
        if not ppl_df.empty:
            df = df.merge(ppl_df[["model_id", "perplexity"]], on="model_id", how="left")
            n_ppl = df["perplexity"].notna().sum()
            print(f"  Perplexity loaded for {n_ppl}/{len(df)} models from {ppl_dir}")
        else:
            print(f"  [info] no perplexity.json files found under {ppl_dir}")
    else:
        print(f"  [info] --ppl_dir not found ({ppl_dir}), skipping perplexity plot")

    out_csv = out_dir / "all_models_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    plot_scaling_law(df, out_dir)
    plot_scaling_by_perplexity(df, out_dir)
    plot_lang_vs_md(df, out_dir)
    plot_froi_profile(df, out_dir)

    print("\nLoading layerwise data...")
    lw_df = load_layerwise(sweep_dir, args.families, meta)
    plot_layer_profiles(lw_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
