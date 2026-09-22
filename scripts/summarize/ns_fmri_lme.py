#!/usr/bin/env python3
"""Summarize Natural Stories fMRI pooled-LME encoding results across model families.

Companion to ``summarize/ns_fmri.py`` (the ridge/embeddings summarizer, which
reads ``ns_fmri_by_roi.csv`` / ``ns_fmri_checkpoint_scores.csv``): this reads
the LME variant instead — ``natural_stories_fmri_lme.py``, built "following
Lin & Schuler (2025, arXiv:2506.11338)" (Natural Stories fMRI IS that paper's
Experiment 1 dataset, Shain et al. 2020) — which fits

    BOLD ~ wlen_z + sentpos_z + surprisal_z +
           (1 + wlen_z + surprisal_z | subject) + (1 + surprisal_z | story)

per (fROI, layer) pair, using PER-LAYER logit-lens surprisal (Kuribayashi
et al. 2025) — same per-layer sweep as the Fedorenko2016/Tuckute
pooled LMEs, generalized to a second grouping dimension (26 fROIs).
``ns_fmri_lme_by_roi.csv`` has one row per (fROI, layer, revision); the
"final checkpoint" mode below selects each model's LAST layer before
averaging across fROIs — averaging across every layer blindly would mix
early-layer noise into the real signal, so this is not optional.

Two modes (``--checkpoints`` selects checkpoint-sweep summarization):

**Final checkpoint** (default) — brain score from the last training step:
  Reads ``ns_fmri_lme_by_roi.csv`` from each model directory. This is the
  direct analog of the paper's Figure 1 (log-likelihood vs parameter count,
  replicated across every model family in the registry, not just the
  paper's own GPT-2/GPT-Neo/OPT triplet).

**Checkpoint sweep** (``--checkpoints``) — training dynamics for staged sweeps:
  Reads ``ns_fmri_lme_checkpoint_scores.csv`` from
  ``<sweep_dir>/<family>/<model>/`` (already has LANG/MD/overall columns
  precomputed by the benchmark, unlike the ridge version's checkpoint CSV).

Networks
--------
  LANG = 6 left-hemisphere language parcels (Fedorenko lab)
  MD   = 20 bilateral multiple-demand parcels

Primary metric: held-out ΔLogLik (``delta_loglik_test``) — matches the
paper's own metric (model log-likelihood on held-out data), not a
correlation. Secondary: held-out ΔPearson/ΔSpearman r.
Key contrast  : LANG − MD (language specificity).

Outputs (default ``outputs/ns_fmri/lme_summary/``), each written twice — once
per layer-selection mode ("last" = highest layer index, "best" = layer
maximizing mean ΔLogLik across fROIs — see ``_select_layer_rows``), suffixed
``_last_layer`` / ``_best_layer`` so both coexist:
  all_models_scores_{last,best}_layer.csv,
  scaling_law_{last,best}_layer.png (1x3: Overall/LANG/MD),
  scaling_law_secondary_{last,best}_layer.png, scaling_by_perplexity_{last,best}_layer.png,
  lang_vs_md_{last,best}_layer.png, froi_profile_{last,best}_layer.png
  (checkpoint mode: <family>/all_models_ns_fmri_lme_checkpoint_scores.csv,
   <family>/training_curves.png, <family>/scaling_law_final.png)

Usage
-----
  python scripts/summarize/ns_fmri_lme.py
  python scripts/summarize/ns_fmri_lme.py \\
      --sweep_dir outputs_jeanzay/ns_fmri/surprisal_sweep \\
      --out_dir   outputs/ns_fmri/lme_summary
  python scripts/summarize/ns_fmri_lme.py --families gpt2 pythia

  python scripts/summarize/ns_fmri_lme.py --checkpoints \\
      --sweep_dir outputs_jeanzay/ns_fmri/surprisal_sweep
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
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from neuroai.analysis.benchmark_common import family_group, plot_scaling_law_by_group

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

# Primary: matches the paper's own metric (held-out log-likelihood), not a
# correlation — see module docstring.
SUMMARY_METRIC = "mean_delta_loglik_test"
FROI_METRIC = "delta_loglik_test"
SECONDARY_METRICS = [
    ("mean_test_spearmanr", "test_spearmanr_full_model", "ΔSpearman ρ (test, full model)"),
    ("mean_test_pearsonr", "test_pearsonr_full_model", "ΔPearson r (test, full model)"),
    # The delta itself (full-model minus baseline-only correlation) -- what the
    # report's ceiling-normalised surprisal headline actually needs, distinct
    # from the two full-model-only rows above.
    ("mean_test_delta_pearsonr", "test_delta_pearsonr", "Δ Pearson r (full - baseline)"),
]

MIN_CHECKPOINT_STEPS = 2

FAMILY_COLORS = {
    "gpt2": "#1f77b4",
    "gptneo": "#ff7f0e",
    "opt": "#2ca02c",
    "llama32": "#d62728",
    "mamba": "#9467bd",
    "mamba2": "#8c564b",
    "mistral": "#e377c2",
    "qwen3": "#7f7f7f",
    "pythia": "#bcbd22",
    "olmo2": "#17becf",
    "cbr_rnn": "#e41a1c",
    "alibi": "#aec7e8",
    "stickbreaking": "#ffbb78",
    "forgetting": "#98df8a",
    "babylm": "#1a9850",
    "hard_2gram": "#999999",
    "hard_3gram": "#666666",
    "hard_5gram": "#333333",
}

FAMILY_MARKERS = {
    "gpt2": "o", "gptneo": "s", "opt": "^", "llama32": "D",
    "mamba": "v", "mamba2": "p", "mistral": "h", "qwen3": "*",
    "pythia": "P", "olmo2": "X", "cbr_rnn": "H",
    "alibi": "<", "stickbreaking": ">", "forgetting": "d",
    "babylm": "8", "hard_2gram": "1", "hard_3gram": "2", "hard_5gram": "3",
}


# ---------------------------------------------------------------------------
# Registry helpers (shared with summarize/ns_fmri.py)
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

def _select_layer_rows(df: pd.DataFrame, layer_mode: str) -> tuple[pd.DataFrame, float]:
    """Collapse a (fROI, layer) df down to a single layer's rows.

    ``layer_mode="last"``: the highest layer index (matches the paper's
    late-layer intuition — this was this script's only behavior previously).
    ``layer_mode="best"``: the layer whose across-fROI (converged-only) mean
    ΔLogLik is highest. Picked from the *overall* mean across fROIs, not a
    separate best layer per fROI — the latter would double-dip (each fROI
    gets to pick whichever layer flatters it) and inflate every fROI's score.
    """
    if "layer" not in df.columns or not df["layer"].notna().any():
        return df, float("nan")
    if layer_mode == "last":
        layer = df["layer"].max()
    else:
        if "converged" in df.columns:
            ok = df[df["converged"].fillna(False).astype(bool)]
        else:
            ok = df
        means = ok.groupby("layer")[FROI_METRIC].mean()
        layer = means.idxmax() if not means.empty else df["layer"].max()
    return df[df["layer"] == layer], float(layer)


def load_all_models(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
    *,
    layer_mode: str = "last",
) -> pd.DataFrame:
    """One row per model: overall / LANG / MD / LANG−MD ΔLogLik (+ secondary metrics).

    ``layer_mode``: "last" (highest layer index) or "best" (layer maximizing
    mean ΔLogLik across fROIs) — see ``_select_layer_rows``.
    """
    rows: list[dict] = []

    for roi_csv in sorted(sweep_dir.glob("*/*/ns_fmri_lme_by_roi.csv")):
        family = roi_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = roi_csv.parent.name
        model_id = model_dir.replace("_", "/", 1)

        try:
            df = pd.read_csv(roi_csv)
        except Exception:
            continue
        if df.empty or FROI_METRIC not in df.columns:
            continue

        if "revision" in df.columns and df["revision"].nunique() > 1:
            revision = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == revision]
        else:
            revision = str(df["revision"].iloc[0]) if "revision" in df.columns else "main"

        # One row per (fROI, layer) — collapse to a single layer before
        # averaging across fROIs. Averaging across every layer blindly would
        # mix early-layer noise into the real signal.
        n_layers_fit = np.nan
        layer_used = float("nan")
        if "layer" in df.columns and df["layer"].notna().any():
            n_layers_fit = int(df["layer"].nunique())
            df, layer_used = _select_layer_rows(df, layer_mode)

        # Crashed/unconverged fROI fits leave delta_loglik_test as NaN or
        # converged=False — drop those from the mean, but report the rate.
        n_total = len(df)
        if "converged" in df.columns:
            df_ok = df[df["converged"].fillna(False).astype(bool)]
        else:
            df_ok = df
        n_converged = len(df_ok)

        m = meta.get(model_id, meta.get(model_dir, {}))
        row: dict = {
            "family": family,
            "model_id": model_id,
            "label": m.get("label", model_id.split("/")[-1]),
            "n_params": m.get("n_params", np.nan),
            "n_layers": m.get("n_layers", np.nan),
            "n_layers_fit": n_layers_fit,
            "layer_used": layer_used,
            "layer_mode": layer_mode,
            "revision": revision,
            "n_frois": int(df["froi"].nunique()) if "froi" in df.columns else 0,
            "n_converged": n_converged,
            "convergence_rate": n_converged / n_total if n_total else np.nan,
            SUMMARY_METRIC: float(df_ok[FROI_METRIC].mean()) if not df_ok.empty else np.nan,
        }
        for mean_col, per_row_col, _ in SECONDARY_METRICS:
            row[mean_col] = float(df_ok[per_row_col].mean()) if per_row_col in df_ok.columns and not df_ok.empty else np.nan

        if "network" in df.columns:
            for net in NETWORKS:
                ndf = df_ok[df_ok["network"] == net]
                row[f"{net}_{SUMMARY_METRIC}"] = float(ndf[FROI_METRIC].mean()) if not ndf.empty else np.nan
            lang_v = row.get(f"LANG_{SUMMARY_METRIC}", np.nan)
            md_v = row.get(f"MD_{SUMMARY_METRIC}", np.nan)
            row["lang_minus_md_delta_loglik_test"] = (
                lang_v - md_v if np.isfinite(lang_v) and np.isfinite(md_v) else np.nan
            )

        if "froi" in df.columns:
            for froi, gdf in df_ok.groupby("froi"):
                row[f"{froi}_{FROI_METRIC}"] = float(gdf[FROI_METRIC].mean())

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["family", "n_params"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Perplexity loading (reused verbatim from summarize/ns_fmri.py)
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
            "model_id": d.get("model_id", ""),
            "perplexity": float(d["perplexity"]),
            "n_tokens": d.get("n_tokens", np.nan),
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

def _scaling_law_ax(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
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
                label=f"fit R²={r2:.3f} slope={a:+.3f}")

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.25)


def plot_scaling_law(df: pd.DataFrame, out_dir: Path, *, suffix: str = "", title_extra: str = "") -> None:
    """1×3: Overall / LANG / MD ΔLogLik vs model size — the paper's Fig 1 analog."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_titles = [
        (SUMMARY_METRIC, "Overall (LANG + MD)"),
        (f"LANG_{SUMMARY_METRIC}", "LANG network"),
        (f"MD_{SUMMARY_METRIC}", "MD network"),
    ]
    for ax, (metric, title) in zip(axes, metrics_titles):
        _scaling_law_ax(ax, df, metric, "Mean held-out ΔLogLik", title)
    fig.suptitle(f"NS fMRI (pooled LME) — Brain score vs model size{title_extra}", fontsize=13)
    fig.tight_layout()
    fname = out_dir / f"scaling_law{suffix}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_scaling_law_secondary(df: pd.DataFrame, out_dir: Path, *, suffix: str = "", title_extra: str = "") -> None:
    """1×N secondary-metric scaling plots (ΔSpearman/ΔPearson), overall only."""
    available = [(col, ylabel) for col, _, ylabel in SECONDARY_METRICS if col in df.columns]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    for ax, (metric, ylabel) in zip(axes, available):
        _scaling_law_ax(ax, df, metric, ylabel, ylabel)
    fig.suptitle(f"NS fMRI (pooled LME) — Secondary metrics vs model size{title_extra}", fontsize=13)
    fig.tight_layout()
    fname = out_dir / f"scaling_law_secondary{suffix}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def _scaling_ppl_ax(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str) -> None:
    plot_df = df.dropna(subset=["perplexity", metric]).copy()
    if plot_df.empty:
        ax.set_title(f"{title} (no data)")
        return

    for family, grp in plot_df.groupby("family"):
        color = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        grp_s = grp.sort_values("perplexity")
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

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Perplexity on Natural Stories  ← better LM", fontsize=9)
    ax.set_ylabel("Mean held-out ΔLogLik", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.25)


def plot_scaling_by_perplexity(df: pd.DataFrame, out_dir: Path, *, suffix: str = "", title_extra: str = "") -> None:
    """1×3: Overall / LANG / MD brain score vs naturalstories perplexity."""
    ppl_df = df.dropna(subset=["perplexity"])
    if ppl_df.empty:
        print("  [skip] no perplexity data — run ns_perplexity_*.sh first")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_titles = [
        (SUMMARY_METRIC, "Overall (LANG + MD)"),
        (f"LANG_{SUMMARY_METRIC}", "LANG network"),
        (f"MD_{SUMMARY_METRIC}", "MD network"),
    ]
    for ax, (metric, title) in zip(axes, metrics_titles):
        _scaling_ppl_ax(ax, ppl_df, metric, title)
    fig.suptitle(f"NS fMRI (pooled LME) — Brain score vs perplexity (Natural Stories){title_extra}", fontsize=13)
    fig.tight_layout()
    fname = out_dir / f"scaling_by_perplexity{suffix}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_scaling_by_group_regions(df: pd.DataFrame, out_dir: Path, *, suffix: str = "", title_extra: str = "") -> None:
    """2x2 grids (All groups / LLM / Memory-Constrained / BabyLM) — one figure
    each for Overall (LANG+MD), LANG only, and MD only. Each panel is its own
    Axes, so a category's own scaling trend isn't crushed by the others.
    Mirrors the embed scripts' by-group scaling grids
    (neuroai/analysis/benchmark_common.py's plot_scaling_law_by_group)."""
    df_grp = df.copy()
    df_grp["group"] = df_grp["family"].map(family_group)

    panels = [
        (SUMMARY_METRIC, "overall", "Overall (LANG + MD)"),
        (f"LANG_{SUMMARY_METRIC}", "lang", "LANG network"),
        (f"MD_{SUMMARY_METRIC}", "md", "MD network"),
    ]
    for value_col, slug, panel_label in panels:
        if value_col not in df_grp.columns:
            continue
        fname = out_dir / f"scaling_by_group_{slug}{suffix}.png"
        plot_scaling_law_by_group(
            df_grp, value_col=value_col, out_path=fname,
            ylabel="Mean held-out ΔLogLik",
            title=f"NS fMRI (pooled LME) — {panel_label} scaling by model category{title_extra}",
        )
        print(f"  Saved: {fname}")


def plot_lang_vs_md(df: pd.DataFrame, out_dir: Path, *, suffix: str = "", title_extra: str = "") -> None:
    """LANG vs MD scatter (one dot per model) with identity diagonal."""
    lang_col = f"LANG_{SUMMARY_METRIC}"
    md_col = f"MD_{SUMMARY_METRIC}"
    plot_df = df.dropna(subset=[lang_col, md_col])
    if plot_df.empty:
        print("  [skip] no LANG/MD data for lang_vs_md plot")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for family, grp in plot_df.groupby("family"):
        color = FAMILY_COLORS.get(family, "gray")
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
    ax.set_xlabel("MD ΔLogLik", fontsize=9)
    ax.set_ylabel("LANG ΔLogLik", fontsize=9)
    ax.set_title(f"NS fMRI (pooled LME) — LANG vs MD specificity{title_extra}", fontsize=11)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fname = out_dir / f"lang_vs_md{suffix}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_froi_profile(df: pd.DataFrame, out_dir: Path, *, suffix: str = "", title_extra: str = "") -> None:
    """Bar chart: mean ΔLogLik per fROI (LANG in blue, MD in red)."""
    froi_cols = sorted([
        c for c in df.columns
        if c.endswith(f"_{FROI_METRIC}")
        and c not in {SUMMARY_METRIC, f"LANG_{SUMMARY_METRIC}", f"MD_{SUMMARY_METRIC}"}
    ])
    if not froi_cols:
        print("  [skip] no per-fROI columns found")
        return

    froi_means = df[froi_cols].mean().sort_values(ascending=False)
    froi_labels = [c.replace(f"_{FROI_METRIC}", "") for c in froi_means.index]
    bar_colors = [
        NETWORK_COLORS["LANG"] if any(f == label for f in LANG_FROIS) else NETWORK_COLORS["MD"]
        for label in froi_labels
    ]

    fig, ax = plt.subplots(figsize=(max(9, len(froi_cols) * 0.55), 4))
    ax.bar(range(len(froi_means)), froi_means.values, color=bar_colors)
    ax.set_xticks(range(len(froi_means)))
    ax.set_xticklabels(froi_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean held-out ΔLogLik", fontsize=9)
    ax.set_title(f"NS fMRI (pooled LME) — per-fROI profile (mean over models){title_extra}", fontsize=11)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.legend(handles=[Patch(color=NETWORK_COLORS["LANG"], label="LANG"),
                        Patch(color=NETWORK_COLORS["MD"], label="MD")],
              fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / f"froi_profile{suffix}.png"
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

    for ck_csv in sorted(sweep_dir.glob("*/*/ns_fmri_lme_checkpoint_scores.csv")):
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
        if "step" not in df.columns:
            df["step"] = df["revision"].apply(_revision_step)
        df["model_id"] = model_id
        df["label"] = m.get("label", model_id.split("/")[-1])
        df["family"] = family
        df["n_params"] = m.get("n_params", np.nan)
        df["n_layers"] = m.get("n_layers", np.nan)
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    sort_cols = [c for c in ("family", "n_params", "model_id", "step") if c in out.columns]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _plot_checkpoint_training_curves(df: pd.DataFrame, out_dir: Path, family: str) -> None:
    """4-panel training curves: Overall / LANG / MD / LANG−MD ΔLogLik."""
    panels = [
        (SUMMARY_METRIC, "Overall (LANG + MD)"),
        (f"LANG_{FROI_METRIC}", "LANG network"),
        (f"MD_{FROI_METRIC}", "MD network"),
        ("lang_minus_md_delta_loglik_test", "LANG − MD (specificity)"),
    ]
    available = [(col, title) for col, title in panels if col in df.columns]
    if not available:
        print("  [skip] no ΔLogLik columns in checkpoint scores")
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
                    lw=1.6, alpha=0.9, label=f"{label} ({final:.3f})")
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training step", fontsize=8)
        ax.set_ylabel("Mean held-out ΔLogLik", fontsize=8)
        ax.legend(fontsize=6, loc="lower right", framealpha=0.7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"NS fMRI (pooled LME) — Training curves ({family})", fontsize=13)
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
        (SUMMARY_METRIC, "Overall"),
        (f"LANG_{FROI_METRIC}", "LANG network"),
        (f"MD_{FROI_METRIC}", "MD network"),
    ]
    available = [(col, title) for col, title in metrics_titles if col in final_df.columns]
    if not available:
        return fit_results

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    color = FAMILY_COLORS.get(family, "gray")
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

        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Number of parameters", fontsize=8)
        ax.set_ylabel("Mean held-out ΔLogLik", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(f"NS fMRI (pooled LME) — Scaling law at final step ({family}, step {final_step})", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "scaling_law_final.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fit_results


def _print_model_table(df: pd.DataFrame) -> None:
    lang_col = f"LANG_{SUMMARY_METRIC}"
    md_col = f"MD_{SUMMARY_METRIC}"
    header = (
        f"  {'family':<14} {'model':<28} {'frois':>5} {'conv%':>6} "
        f"{'overall':>9}  {'LANG':>9}  {'MD':>9}  {'LANG-MD':>9}"
    )
    print(header)
    print("  " + "-" * len(header))
    for _, row in df.iterrows():
        def _fmt(v: float) -> str:
            return f"{v:>9.3f}" if np.isfinite(v) else f"{'n/a':>9}"
        conv_pct = row["convergence_rate"] * 100 if np.isfinite(row["convergence_rate"]) else np.nan
        print(
            f"  {row['family']:<14} {row['label']:<28} "
            f"{row['n_frois']:>5} {conv_pct:>5.0f}% "
            f"{_fmt(row[SUMMARY_METRIC])}  "
            f"{_fmt(row.get(lang_col, np.nan))}  "
            f"{_fmt(row.get(md_col, np.nan))}  "
            f"{_fmt(row.get('lang_minus_md_delta_loglik_test', np.nan))}"
        )
    print()


def run_final_checkpoint_summary(
    sweep_dir: Path,
    out_dir: Path,
    ppl_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
    *,
    layer_mode: str,
) -> pd.DataFrame:
    """Final-checkpoint summary for one layer-selection mode ("last" or
    "best" — see ``_select_layer_rows``). Writes the CSV + all five plots,
    suffixed ``_last_layer``/``_best_layer`` so both modes coexist side by
    side in the same output directory."""
    suffix = f"_{layer_mode}_layer"
    title_extra = f" ({layer_mode} layer)"

    df = load_all_models(sweep_dir, families, meta, layer_mode=layer_mode)
    if df.empty:
        print(f"[{layer_mode} layer] no ns_fmri_lme_by_roi.csv files found — skipping")
        return df

    print(f"[{layer_mode} layer] Found {len(df)} models across {df['family'].nunique()} families:\n")
    _print_model_table(df)

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

    out_csv = out_dir / f"all_models_scores{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    plot_scaling_law(df, out_dir, suffix=suffix, title_extra=title_extra)
    plot_scaling_law_secondary(df, out_dir, suffix=suffix, title_extra=title_extra)
    plot_scaling_by_perplexity(df, out_dir, suffix=suffix, title_extra=title_extra)
    plot_lang_vs_md(df, out_dir, suffix=suffix, title_extra=title_extra)
    plot_froi_profile(df, out_dir, suffix=suffix, title_extra=title_extra)
    print()
    return df


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

    csv_path = out_dir / "all_models_ns_fmri_lme_checkpoint_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    _plot_checkpoint_training_curves(df, out_dir, family)
    fit = _plot_checkpoint_scaling(df, out_dir, family)
    (out_dir / "scaling_fit_results.json").write_text(
        json.dumps(fit, indent=2), encoding="utf-8",
    )

    n_models = df["model_id"].nunique()
    n_steps = df["step"].nunique()
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
            f"No multi-checkpoint ns_fmri_lme_checkpoint_scores.csv found under {sweep_dir} "
            f"(need >={MIN_CHECKPOINT_STEPS} distinct revisions per model)."
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoints", action="store_true",
        help="Summarize checkpoint sweeps (ns_fmri_lme_checkpoint_scores.csv) instead of final scores",
    )
    p.add_argument(
        "--sweep_dir", type=Path,
        default=Path("outputs_jeanzay/ns_fmri/surprisal_sweep"),
        help="Sweep results root (default: outputs_jeanzay/ns_fmri/surprisal_sweep)",
    )
    p.add_argument(
        "--ppl_dir", type=Path,
        default=Path("outputs_jeanzay/naturalstories/perplexity"),
        help="Naturalstories perplexity results dir (default: outputs_jeanzay/naturalstories/perplexity)",
    )
    p.add_argument(
        "--out_dir", type=Path, default=None,
        help="Output directory (default: outputs/ns_fmri/lme_summary[/checkpoint])",
    )
    p.add_argument(
        "--families", nargs="+", default=None,
        help="Restrict to these families (default: all)",
    )
    args = p.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    out_dir = (args.out_dir or (
        Path("outputs/ns_fmri/lme_summary/checkpoint") if args.checkpoints
        else Path("outputs/ns_fmri/lme_summary")
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

    print("Mode         : final checkpoint (last layer + best layer)\n")

    ppl_dir = args.ppl_dir.resolve()
    df_last = run_final_checkpoint_summary(
        sweep_dir, out_dir, ppl_dir, args.families, meta, layer_mode="last",
    )
    if df_last.empty:
        sys.exit("No ns_fmri_lme_by_roi.csv files found — nothing to summarize.")
    run_final_checkpoint_summary(
        sweep_dir, out_dir, ppl_dir, args.families, meta, layer_mode="best",
    )

    print("Done.")


if __name__ == "__main__":
    main()
