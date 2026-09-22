#!/usr/bin/env python3
"""Summarize Podcast ECoG encoding results across model families.

Two modes (``--checkpoints`` selects checkpoint-sweep summarization):

**Final checkpoint** (default) — brain score from the last training step:
  Reads ``podcast_electrode_timecourse.csv`` (window mean) or falls back to
  ``podcast_by_electrode.csv``.

**Checkpoint sweep** (``--checkpoints``) — training dynamics for staged sweeps only:
  Reads ``podcast_checkpoint_scores.csv`` from
  ``<sweep_dir>/checkpoints/<family>/<model>/`` (or directly from
  ``.../checkpoints`` if passed as ``--sweep_dir``).  Does **not** touch the
  final-checkpoint summary plots.

Electrode groups (final mode only):
  G  = grid  |  D = depth  |  S = strip  |  EG = edge grid

Outputs
-------
Final mode (default ``outputs/podcast/summary/``):
  all_models_scores.csv, scaling_law*.png, layer_profiles.png, timecourse*.png
  Uses top-level ``<family>/<model>/`` results (final checkpoint only).

Checkpoint mode (default ``outputs/podcast/summary/checkpoint/``):
  ``<family>/all_models_podcast_checkpoint_scores.csv``
  ``<family>/all_models_podcast_training_curves.png``
  ``<family>/scaling_law_final.png``
  ``<family>/layer_profiles_final_checkpoint.png``

Usage
-----
  python scripts/summarize/podcast.py
  python scripts/summarize/podcast.py \\
      --sweep_dir outputs_jeanzay/podcast/checkpoint_sweep \\
      --out_dir   outputs/podcast/summary
  python scripts/summarize/podcast.py --families gpt2 qwen3 mamba

  python scripts/summarize/podcast.py --checkpoints \\
      --sweep_dir outputs_jeanzay/podcast/checkpoint_sweep
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
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TC_TMIN = -1.0   # analysis window start (s)
TC_TMAX =  1.0   # analysis window end   (s)

# Correlation metric for all summary plots
TC_CORR = "spearmanr"
ELECTRODE_BEST_CORR = "best_spearmanr"
ELECTRODE_FLAT_CORR = "spearmanr"
LAYER_CORR = "spearmanr"
LAYER_PROFILE_METRIC = "mean_spearmanr"

SUMMARY_METRIC = "mean_spearmanr"
CHECKPOINT_METRIC = "mean_window_spearmanr"
CHECKPOINT_LAST_LAYER_METRIC = "mean_flat_spearmanr"


def _window_label() -> str:
    return f"[{TC_TMIN:+.1f},{TC_TMAX:+.1f}]s"


def _ylabel_mean_spearman(*, window: bool = False) -> str:
    suffix = f" {_window_label()}" if window else ""
    return f"Mean Spearman ρ{suffix}"

ELECTRODE_GROUPS = {
    "G":  "Grid",
    "D":  "Depth",
    "S":  "Strip",
    "EG": "Edge Grid",
}

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
# Registry helpers (no torch import — parse registry source text directly)
# ---------------------------------------------------------------------------

def _registry_meta() -> dict[str, dict]:
    reg_path = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "neuroai" / "models" / "registry.py"
    text = reg_path.read_text(encoding="utf-8")

    # Resolve top-level string variable assignments, e.g. OLMO2_1B_REPO = "allenai/OLMo-2-0425-1B"
    var_map: dict[str, str] = {}
    for m in re.finditer(r'^([A-Z][A-Z0-9_]+)\s*=\s*"([^"]+)"', text, flags=re.M):
        var_map[m.group(1)] = m.group(2)

    meta: dict[str, dict] = {}
    fam_blocks = re.findall(r'"([^"]+)":\s*\[(.*?)\],', text, flags=re.S)

    # First arg is a string literal
    main_re = re.compile(
        r'_main_only\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)',
        flags=re.S,
    )
    spec_lit_re = re.compile(
        r'LppModelSpec\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)',
        flags=re.S,
    )
    # First arg is a variable reference, e.g. LppModelSpec(OLMO2_1B_REPO, "OLMo2-1B", ...)
    spec_var_re = re.compile(
        r'LppModelSpec\(\s*([A-Z][A-Z0-9_]+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)',
        flags=re.S,
    )

    def _add(model_id: str, label: str, n_layers: str, n_params: str, fam: str) -> None:
        entry = {
            "family": fam,
            "label": label,
            "n_params": float(n_params),
            "n_layers": int(n_layers),
        }
        meta[model_id] = entry
        meta[model_id.replace("/", "_")] = entry

    for fam, block in fam_blocks:
        for pattern in (main_re, spec_lit_re):
            for m in pattern.finditer(block):
                model_id, label, n_layers, n_params = m.groups()[:4]
                _add(model_id, label, n_layers, n_params, fam)
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
    return int(m.group(1)) if m else (0 if rev == "main" else -1)


def _pick_final_revision(revisions: list[str]) -> str:
    return max(revisions, key=_revision_step)


def _is_checkpoint_staging_path(path: Path) -> bool:
    """True for files under a checkpoints/ staging directory."""
    return "checkpoints" in path.parts


def _checkpoint_search_root(sweep_dir: Path) -> Path:
    """Return the checkpoints/ subtree when summarizing training dynamics."""
    staged = sweep_dir / "checkpoints"
    if staged.is_dir() and sweep_dir.name != "checkpoints":
        return staged
    return sweep_dir


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _mean_score_from_timecourse(tc_csv: Path, revision: str) -> dict[str, float]:
    """
    Read podcast_electrode_timecourse.csv, filter to [TC_TMIN, TC_TMAX] and
    the given revision, then return mean Spearman ρ per electrode group and overall.

    Returns dict with keys: 'all', 'G', 'D', 'S', 'EG'  (NaN if no data).
    """
    result = {g: np.nan for g in ["all"] + list(ELECTRODE_GROUPS)}
    try:
        df = pd.read_csv(tc_csv)
        if "revision" in df.columns:
            df = df[df["revision"] == revision]
        df = df[(df["time_s"] >= TC_TMIN) & (df["time_s"] <= TC_TMAX)]
        if df.empty:
            return result

        if TC_CORR not in df.columns:
            return result

        # Mean over time window per electrode, then grand mean
        per_elec = (
            df.groupby(["subject", "electrode", "region"])[TC_CORR]
            .mean()
            .reset_index()
        )
        result["all"] = float(per_elec[TC_CORR].mean())

        if "region" in per_elec.columns:
            for grp in ELECTRODE_GROUPS:
                sub = per_elec[per_elec["region"] == grp]
                result[grp] = float(sub[TC_CORR].mean()) if not sub.empty else np.nan
    except Exception as exc:
        print(f"    [warn] timecourse read failed: {exc}", file=sys.stderr)
    return result


def _mean_score_from_electrode_csv(elec_csv: Path, revision: str) -> dict[str, float]:
    """
    Fallback: read podcast_by_electrode.csv and return mean best_spearmanr
    per electrode group and overall.
    """
    result = {g: np.nan for g in ["all"] + list(ELECTRODE_GROUPS)}
    try:
        df = pd.read_csv(elec_csv)
        if "revision" in df.columns:
            df = df[df["revision"] == revision]
        if df.empty or ELECTRODE_BEST_CORR not in df.columns:
            return result
        result["all"] = float(df[ELECTRODE_BEST_CORR].mean())
        if "region" in df.columns:
            for grp in ELECTRODE_GROUPS:
                sub = df[df["region"] == grp]
                result[grp] = float(sub[ELECTRODE_BEST_CORR].mean()) if not sub.empty else np.nan
    except Exception as exc:
        print(f"    [warn] electrode CSV read failed: {exc}", file=sys.stderr)
    return result


def load_all_models(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """One row per model. Uses timecourse CSV (window) when available."""
    rows: list[dict] = []

    for elec_csv in sorted(sweep_dir.glob("*/*/podcast_by_electrode.csv")):
        if _is_checkpoint_staging_path(elec_csv):
            continue
        family = elec_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = elec_csv.parent.name
        model_id = model_dir.replace("_", "/", 1)

        # Determine final revision from electrode CSV
        try:
            elec_df = pd.read_csv(elec_csv)
        except Exception:
            continue
        if elec_df.empty:
            continue

        if "revision" in elec_df.columns and elec_df["revision"].nunique() > 1:
            revision = _pick_final_revision(elec_df["revision"].dropna().unique().tolist())
        elif "revision" in elec_df.columns:
            revision = str(elec_df["revision"].iloc[0])
        else:
            revision = "main"

        n_electrodes = len(elec_df[elec_df.get("revision", revision) == revision]) if "revision" in elec_df.columns else len(elec_df)
        n_subjects = int(elec_df["subject"].nunique()) if "subject" in elec_df.columns else 0

        # Scores: prefer timecourse window, fall back to electrode CSV
        tc_csv = elec_csv.parent / "podcast_electrode_timecourse.csv"
        if tc_csv.exists():
            scores = _mean_score_from_timecourse(tc_csv, revision)
            score_source = f"timecourse [{TC_TMIN:+.1f},{TC_TMAX:+.1f}]s"
        else:
            scores = _mean_score_from_electrode_csv(elec_csv, revision)
            score_source = f"{ELECTRODE_BEST_CORR} (fallback)"

        m = meta.get(model_id, meta.get(model_dir, {}))
        row = {
            "family": family,
            "model_id": model_id,
            "label": m.get("label", model_id.split("/")[-1]),
            "n_params": m.get("n_params", np.nan),
            "n_layers": m.get("n_layers", np.nan),
            "revision": revision,
            "n_electrodes": n_electrodes,
            "n_subjects": n_subjects,
            "score_source": score_source,
            SUMMARY_METRIC: scores["all"],
        }
        for grp in ELECTRODE_GROUPS:
            row[f"{SUMMARY_METRIC}_{grp}"] = scores[grp]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["family", "n_params"]).reset_index(drop=True)
    return df


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
# Timecourse loading
# ---------------------------------------------------------------------------

def load_layerwise(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """
    Read podcast_by_electrode_layerwise.csv for each model.
    Returns long DataFrame with columns:
      family, model_id, label, n_params, revision, layer, layer_rel,
      mean_spearmanr
    One row per (model × layer).
    """
    rows: list[dict] = []
    for lw_csv in sorted(sweep_dir.glob("*/*/podcast_by_electrode_layerwise.csv")):
        if _is_checkpoint_staging_path(lw_csv):
            continue
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
        if df.empty:
            continue
        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == rev]
        else:
            rev = str(df["revision"].iloc[0]) if "revision" in df.columns else "main"

        if LAYER_CORR not in df.columns:
            continue

        agg = df.groupby("layer").agg(**{LAYER_PROFILE_METRIC: (LAYER_CORR, "mean")}).reset_index()
        n_layers = int(agg["layer"].max()) + 1
        agg["layer_rel"] = agg["layer"] / max(1, n_layers - 1)
        agg["family"] = family
        agg["model_id"] = model_id
        agg["label"] = m.get("label", model_id.split("/")[-1])
        agg["n_params"] = m.get("n_params", np.nan)
        agg["revision"] = rev
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_timecourses(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """
    Collect per-time-bin mean Pearson r across electrodes × subjects for every
    model.  Returns a long DataFrame with columns:
      family, model_id, label, n_params, revision, time_s, spearmanr, sem, n
    One row per (model × time_bin).
    """
    rows: list[dict] = []

    for tc_csv in sorted(sweep_dir.glob("*/*/podcast_electrode_timecourse.csv")):
        if _is_checkpoint_staging_path(tc_csv):
            continue
        family = tc_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = tc_csv.parent.name
        model_id = model_dir.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir, {}))

        try:
            df = pd.read_csv(tc_csv)
        except Exception:
            continue
        if df.empty:
            continue

        # Keep only the final revision
        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == rev]
        else:
            rev = str(df["revision"].iloc[0]) if "revision" in df.columns else "main"

        if TC_CORR not in df.columns:
            continue

        # Average over electrodes × subjects at each time bin
        agg = (
            df.groupby("time_s")[TC_CORR]
            .agg(spearmanr="mean", sem=lambda x: x.sem(), n="count")
            .reset_index()
        )
        agg["family"] = family
        agg["model_id"] = model_id
        agg["label"] = m.get("label", model_id.split("/")[-1])
        agg["n_params"] = m.get("n_params", np.nan)
        agg["revision"] = rev
        rows.append(agg)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _scaling_law_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
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
                   label=family, color=color, marker=marker, s=80, zorder=3, alpha=0.9)
        if len(grp_s) > 1:
            ax.plot(grp_s["n_params"], grp_s[metric], color=color, lw=0.8, alpha=0.45)
        for _, row in grp_s.iterrows():
            ax.annotate(row["label"], (row["n_params"], row[metric]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    a, b, r2 = _fit_log_linear(
        plot_df["n_params"].to_numpy(float),
        plot_df[metric].to_numpy(float),
    )
    if np.isfinite(a):
        x_fit = np.logspace(
            np.log10(plot_df["n_params"].min()),
            np.log10(plot_df["n_params"].max()), 200,
        )
        ax.plot(x_fit, a * np.log10(x_fit) + b, "k--", lw=1.3, alpha=0.55,
                label=f"fit: {a:+.4f}·log₁₀(N) {b:+.4f}  R²={r2:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)


def plot_scaling_law(df: pd.DataFrame, out_dir: Path, metric: str = SUMMARY_METRIC,
                     suffix: str = "", title_extra: str = "") -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ylabel = f"{_ylabel_mean_spearman(window=True)} (all electrodes × subjects)"
    title = f"Podcast ECoG — Brain score vs model size{title_extra}"
    _scaling_law_ax(ax, df, metric, title, ylabel)
    fig.tight_layout()
    fname = out_dir / f"scaling_law{suffix}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_scaling_by_family(df: pd.DataFrame, out_dir: Path,
                           metric: str = SUMMARY_METRIC) -> None:
    """One subplot per family: brain score vs model size, with log-linear fit."""
    families = sorted(df["family"].unique())
    if not families:
        return

    ncols = min(3, len(families))
    nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5, nrows * 4),
        squeeze=False,
    )
    window_str = _window_label()

    for i, fam in enumerate(families):
        ax = axes[i // ncols][i % ncols]
        fam_df = df[df["family"] == fam].dropna(subset=["n_params", metric]).copy()
        fam_df = fam_df.sort_values("n_params")

        color = FAMILY_COLORS.get(fam, "gray")
        marker = FAMILY_MARKERS.get(fam, "o")

        ax.scatter(fam_df["n_params"], fam_df[metric],
                   color=color, marker=marker, s=80, zorder=3, alpha=0.9)
        if len(fam_df) > 1:
            ax.plot(fam_df["n_params"], fam_df[metric],
                    color=color, lw=1.0, alpha=0.5)

        for _, row in fam_df.iterrows():
            ax.annotate(row["label"], (row["n_params"], row[metric]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

        a, b, r2 = _fit_log_linear(
            fam_df["n_params"].to_numpy(float),
            fam_df[metric].to_numpy(float),
        )
        if np.isfinite(a) and len(fam_df) > 1:
            x_fit = np.logspace(
                np.log10(fam_df["n_params"].min()),
                np.log10(fam_df["n_params"].max()), 200,
            )
            ax.plot(x_fit, a * np.log10(x_fit) + b, "--",
                    color=color, lw=1.5, alpha=0.7,
                    label=f"slope={a:+.4f}  R²={r2:.2f}")
            ax.legend(fontsize=6, loc="upper left", framealpha=0.7)

        ax.set_xscale("log")
        ax.set_title(fam, fontsize=10, fontweight="bold",
                     color=FAMILY_COLORS.get(fam, "gray"))
        if i % ncols == 0:
            ax.set_ylabel(_ylabel_mean_spearman(window=True), fontsize=8)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Number of parameters", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Podcast ECoG — Scaling law by family", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "scaling_law_by_family.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_scaling_by_group(df: pd.DataFrame, out_dir: Path) -> None:
    """One scaling-law plot per electrode group (G, D, S, EG)."""
    available = [g for g in ELECTRODE_GROUPS if f"{SUMMARY_METRIC}_{g}" in df.columns
                 and df[f"{SUMMARY_METRIC}_{g}"].notna().any()]
    if not available:
        print("  [skip] no per-group scores found")
        return

    # Combined multi-panel figure
    ncols = min(2, len(available))
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 9, nrows * 6), squeeze=False)
    window_str = _window_label()

    for i, grp in enumerate(available):
        ax = axes[i // ncols][i % ncols]
        long_name = ELECTRODE_GROUPS[grp]
        metric = f"{SUMMARY_METRIC}_{grp}"
        _scaling_law_ax(
            ax, df, metric,
            title=f"{long_name} electrodes ({grp})",
            ylabel=_ylabel_mean_spearman(window=True),
        )

    # Hide unused panels
    for j in range(len(available), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Podcast ECoG — Scaling law by electrode group", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "scaling_law_by_group.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

    # Also save individual plots
    for grp in available:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        long_name = ELECTRODE_GROUPS[grp]
        _scaling_law_ax(
            ax2, df, f"{SUMMARY_METRIC}_{grp}",
            title=f"Podcast ECoG — {long_name} ({grp}) — Scaling law",
            ylabel=_ylabel_mean_spearman(window=True),
        )
        fig2.tight_layout()
        fname2 = out_dir / f"scaling_law_{grp}.png"
        fig2.savefig(fname2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {fname2}")


def plot_layer_profiles(lw_df: pd.DataFrame, out_dir: Path) -> None:
    """One subplot per family, all in a single figure. Each curve = one model."""
    if lw_df.empty:
        print("  [skip] no layerwise data found")
        return

    families = sorted(lw_df["family"].unique())
    ncols = min(3, len(families))
    nrows = (len(families) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5, nrows * 4),
        sharey=False, squeeze=False,
    )

    for i, fam in enumerate(families):
        ax = axes[i // ncols][i % ncols]
        fam_df = lw_df[lw_df["family"] == fam]
        model_ids = (
            fam_df.drop_duplicates("model_id")
            .sort_values("n_params")["model_id"]
            .tolist()
        )
        base_colors = plt.cm.tab20(np.linspace(0, 1, max(len(model_ids), 1)))
        rng = np.random.default_rng(seed=42)
        colors = rng.permutation(base_colors)

        for color, mid in zip(colors, model_ids):
            g = fam_df[fam_df["model_id"] == mid].sort_values("layer_rel")
            if g.empty:
                continue
            label = g["label"].iloc[0]
            ax.plot(g["layer_rel"], g[LAYER_PROFILE_METRIC],
                    lw=1.6, alpha=0.85, color=color, label=label)
            # Label at peak
            peak_idx = g[LAYER_PROFILE_METRIC].idxmax()
            ax.annotate(
                label,
                xy=(g.loc[peak_idx, "layer_rel"], g.loc[peak_idx, LAYER_PROFILE_METRIC]),
                fontsize=6, color=color,
                xytext=(3, 2), textcoords="offset points",
            )

        ax.set_title(fam, fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative layer depth", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel(_ylabel_mean_spearman(), fontsize=9)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
        ax.grid(True, alpha=0.25)

    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Podcast ECoG — Layer profiles by family", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "layer_profiles.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_timecourse(tc_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot mean Spearman ρ over time, one curve per family
    (mean across models in that family), plus a grand average.
    Shaded area = ±1 SEM across models within the family.
    The analysis window [TC_TMIN, TC_TMAX] is highlighted.
    """
    if tc_df.empty:
        print("  [skip] no timecourse data found")
        return

    families = sorted(tc_df["family"].unique())

    # ── per-family mean over models at each time bin ──────────────────────
    # First collapse to one curve per model, then average over models per family
    model_tc = (
        tc_df.groupby(["family", "model_id", "time_s"])["spearmanr"]
        .mean()
        .reset_index()
    )
    fam_tc = (
        model_tc.groupby(["family", "time_s"])["spearmanr"]
        .agg(mean="mean", sem=lambda x: x.sem())
        .reset_index()
    )
    grand = (
        model_tc.groupby("time_s")["spearmanr"]
        .agg(mean="mean", sem=lambda x: x.sem())
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    # Highlight analysis window
    ax.axvspan(TC_TMIN, TC_TMAX, color="#d4e8ff", alpha=0.5, zorder=0,
               label=f"analysis window [{TC_TMIN:+.1f},{TC_TMAX:+.1f}]s")
    ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=1)

    # Per-family curves
    for fam in families:
        sub = fam_tc[fam_tc["family"] == fam].sort_values("time_s")
        if sub.empty:
            continue
        color = FAMILY_COLORS.get(fam, "gray")
        ax.plot(sub["time_s"], sub["mean"], color=color, lw=1.4, label=fam, alpha=0.8)
        ax.fill_between(sub["time_s"],
                        sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"],
                        color=color, alpha=0.12)

    # Grand average
    grand = grand.sort_values("time_s")
    ax.plot(grand["time_s"], grand["mean"], color="black", lw=2.2,
            label="grand average", zorder=5)
    ax.fill_between(grand["time_s"],
                    grand["mean"] - grand["sem"],
                    grand["mean"] + grand["sem"],
                    color="black", alpha=0.12, zorder=4)

    ax.set_xlabel("Time relative to word onset (s)", fontsize=11)
    ax.set_ylabel("Mean Spearman ρ (across electrodes × subjects)", fontsize=11)
    ax.set_title("Podcast ECoG — Encoding score time course (mean ± SEM across models)", fontsize=12)
    ax.set_xlim(-2.0, 2.0)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fname = out_dir / "timecourse.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

    # ── per-family individual panels ──────────────────────────────────────
    ncols = min(3, len(families))
    nrows = (len(families) + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 6, nrows * 4),
                              sharey=True, squeeze=False)

    for i, fam in enumerate(families):
        ax2 = axes[i // ncols][i % ncols]
        fam_models = model_tc[model_tc["family"] == fam]

        # Sort models by n_params for consistent color ordering
        model_ids = (
            tc_df[tc_df["family"] == fam]
            .drop_duplicates("model_id")
            .sort_values("n_params")["model_id"]
            .tolist()
        )
        base_colors = plt.cm.tab20(np.linspace(0, 1, max(len(model_ids), 1)))
        rng = np.random.default_rng(seed=42)
        cmap = rng.permutation(base_colors)

        for color, mid in zip(cmap, model_ids):
            g = fam_models[fam_models["model_id"] == mid].sort_values("time_s")
            if g.empty:
                continue
            label = tc_df.loc[tc_df["model_id"] == mid, "label"].iloc[0]
            line, = ax2.plot(g["time_s"], g["spearmanr"], lw=1.5,
                             alpha=0.85, color=color, label=label)
            # Label at the peak of the curve
            peak_idx = g["spearmanr"].idxmax()
            ax2.annotate(
                label,
                xy=(g.loc[peak_idx, "time_s"], g.loc[peak_idx, "spearmanr"]),
                fontsize=6, color=color,
                xytext=(3, 2), textcoords="offset points",
            )

        ax2.axvspan(TC_TMIN, TC_TMAX, color="#d4e8ff", alpha=0.4, zorder=0)
        ax2.axvline(0, color="gray", lw=0.6, ls="--")
        ax2.set_title(fam, fontsize=10, fontweight="bold")
        ax2.set_xlim(-2.0, 2.0)
        ax2.grid(True, alpha=0.2)
        ax2.legend(fontsize=6, loc="upper right", framealpha=0.7)
        if i % ncols == 0:
            ax2.set_ylabel(_ylabel_mean_spearman(), fontsize=9)
        if i // ncols == nrows - 1:
            ax2.set_xlabel("Time (s)", fontsize=9)

    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig2.suptitle("Podcast ECoG — Time courses by family", fontsize=12, y=1.01)
    fig2.tight_layout()
    fname2 = out_dir / "timecourse_by_family.png"
    fig2.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {fname2}")


# ---------------------------------------------------------------------------
# Checkpoint sweep summarization
# ---------------------------------------------------------------------------

MIN_CHECKPOINT_STEPS = 2  # ignore single-checkpoint stubs in staging area


def _window_scores_by_revision(tc_csv: Path) -> dict[str, float]:
    """Mean Spearman ρ in [TC_TMIN, TC_TMAX] per revision from electrode timecourse."""
    scores: dict[str, float] = {}
    try:
        df = pd.read_csv(tc_csv)
    except Exception:
        return scores
    if df.empty or "revision" not in df.columns or TC_CORR not in df.columns:
        return scores
    for revision, rev_df in df.groupby("revision"):
        windowed = rev_df[(rev_df["time_s"] >= TC_TMIN) & (rev_df["time_s"] <= TC_TMAX)]
        if windowed.empty:
            continue
        per_elec = (
            windowed.groupby(["subject", "electrode"])[TC_CORR]
            .mean()
        )
        scores[str(revision)] = float(per_elec.mean())
    return scores


def _flat_scores_by_revision(elec_csv: Path) -> dict[str, float]:
    """Mean last-layer Spearman ρ per revision from podcast_by_electrode.csv."""
    scores: dict[str, float] = {}
    try:
        df = pd.read_csv(elec_csv)
    except Exception:
        return scores
    if df.empty or "revision" not in df.columns or ELECTRODE_FLAT_CORR not in df.columns:
        return scores
    for revision, rev_df in df.groupby("revision"):
        scores[str(revision)] = float(rev_df[ELECTRODE_FLAT_CORR].mean())
    return scores


def _attach_window_scores(df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """Add mean_window_spearmanr from podcast_electrode_timecourse.csv when available."""
    tc_csv = model_dir / "podcast_electrode_timecourse.csv"
    if not tc_csv.is_file():
        df[CHECKPOINT_METRIC] = np.nan
        return df
    by_rev = _window_scores_by_revision(tc_csv)
    if not by_rev:
        df[CHECKPOINT_METRIC] = np.nan
        return df
    rev_col = "revision" if "revision" in df.columns else None
    if rev_col is None:
        df[CHECKPOINT_METRIC] = np.nan
        return df
    df[CHECKPOINT_METRIC] = df[rev_col].astype(str).map(by_rev)
    return df


def _attach_flat_scores(df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """Add mean_flat_spearmanr (last layer) from podcast_by_electrode.csv."""
    elec_csv = model_dir / "podcast_by_electrode.csv"
    if not elec_csv.is_file():
        df[CHECKPOINT_LAST_LAYER_METRIC] = np.nan
        return df
    by_rev = _flat_scores_by_revision(elec_csv)
    if not by_rev:
        df[CHECKPOINT_LAST_LAYER_METRIC] = np.nan
        return df
    rev_col = "revision" if "revision" in df.columns else None
    if rev_col is None:
        df[CHECKPOINT_LAST_LAYER_METRIC] = np.nan
        return df
    df[CHECKPOINT_LAST_LAYER_METRIC] = df[rev_col].astype(str).map(by_rev)
    return df


def _family_model_from_scores_csv(search_root: Path, csv_path: Path) -> tuple[str, str, str]:
    """Return (family, model_dir, model_id) relative to the checkpoint search root."""
    rel = csv_path.parent.relative_to(search_root)
    parts = rel.parts
    if len(parts) >= 2:
        family, model_dir = parts[0], parts[1]
    else:
        family, model_dir = "unknown", parts[-1]
    model_id = model_dir.replace("_", "/", 1)
    return family, model_dir, model_id


def _discover_checkpoint_score_csvs(
    search_root: Path,
    families: list[str] | None,
) -> list[Path]:
    """Find multi-checkpoint score CSVs under the checkpoints staging area."""
    found: list[Path] = []
    for csv_path in sorted(search_root.glob("**/podcast_checkpoint_scores.csv")):
        family, model_dir, _ = _family_model_from_scores_csv(search_root, csv_path)
        if families and family not in families:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty or "step" not in df.columns or df["step"].nunique() < MIN_CHECKPOINT_STEPS:
            continue
        found.append(csv_path)
    return found


def _collect_checkpoint_scores(
    search_root: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for csv_path in _discover_checkpoint_score_csvs(search_root, families):
        family, model_dir, model_id = _family_model_from_scores_csv(search_root, csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty or "step" not in df.columns:
            continue
        m = meta.get(model_id, meta.get(model_dir, {}))
        df = df.copy()
        df = _attach_window_scores(df, csv_path.parent)
        df = _attach_flat_scores(df, csv_path.parent)
        if df[CHECKPOINT_METRIC].isna().all() and "mean_best_spearmanr" in df.columns:
            df[CHECKPOINT_METRIC] = df["mean_best_spearmanr"]
        df.insert(0, "model_id", model_id)
        df.insert(1, "label", m.get("label", model_id.split("/")[-1]))
        df.insert(2, "family", family)
        df["n_params"] = m.get("n_params", np.nan)
        df["n_layers"] = m.get("n_layers", np.nan)
        df["n_params_b"] = df["n_params"] / 1e9
        df["log10_n_params"] = np.log10(df["n_params"].where(df["n_params"] > 0))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    sort_cols = [c for c in ("family", "n_params", "label", "step") if c in out.columns]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _fit_scaling_log10(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"ok": False, "r2": np.nan, "slope": np.nan, "intercept": np.nan, "n": int(x.size)}
    coeffs = np.polyfit(x, y, deg=1)
    y_hat = coeffs[0] * x + coeffs[1]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {
        "ok": True,
        "r2": float(r2),
        "slope": float(coeffs[0]),
        "intercept": float(coeffs[1]),
        "n": int(x.size),
    }


def _plot_checkpoint_training_curves(df: pd.DataFrame, out_path: Path, family: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    model_handles: list = []
    model_labels: list[str] = []
    has_last_layer = CHECKPOINT_LAST_LAYER_METRIC in df.columns

    for model_id, grp in df.groupby("model_id"):
        g = grp.sort_values("step")
        label = g["label"].iloc[0]
        final_best = float(g[CHECKPOINT_METRIC].iloc[-1])
        (line,) = ax.plot(
            g["step"], g[CHECKPOINT_METRIC],
            "o-", label=f"{label} ({final_best:.4f})",
            alpha=0.9, linewidth=1.6,
        )
        model_handles.append(line)
        model_labels.append(f"{label} ({final_best:.4f})")
        if has_last_layer and g[CHECKPOINT_LAST_LAYER_METRIC].notna().any():
            ax.plot(
                g["step"], g[CHECKPOINT_LAST_LAYER_METRIC],
                "--", color=line.get_color(), alpha=0.75, linewidth=1.2,
            )

    ax.set_xlabel("Training step")
    ax.set_ylabel(_ylabel_mean_spearman(window=True))
    ax.set_title(
        f"Podcast ECoG checkpoint sweep — {family} "
        f"(best layer, window {_window_label()})"
    )
    ax.grid(True, alpha=0.3)

    style_handles = [
        Line2D(
            [0], [0], color="k", linestyle="-", marker="o", linewidth=1.6,
            label=f"Best layer, mean ρ {_window_label()}",
        ),
    ]
    style_labels = [f"Best layer, mean ρ {_window_label()}"]
    if has_last_layer:
        style_handles.append(
            Line2D(
                [0], [0], color="k", linestyle="--", linewidth=1.2,
                label="Last layer (full TRF mean)",
            )
        )
        style_labels.append("Last layer (full TRF mean)")

    ax.legend(
        style_handles + model_handles,
        style_labels + model_labels,
        fontsize=7,
        ncol=2,
        title="Line style / model (final best-layer ρ)",
        title_fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_checkpoint_scaling(
    df: pd.DataFrame,
    out_path: Path,
    family: str,
    *,
    title: str,
) -> dict:
    final_step = int(df["step"].max())
    final = df[df["step"] == final_step].copy()
    final = final[np.isfinite(final["n_params"]) & np.isfinite(final[CHECKPOINT_METRIC])]
    fit = _fit_scaling_log10(
        final["log10_n_params"].to_numpy(float),
        final[CHECKPOINT_METRIC].to_numpy(float),
    )
    fit["family"] = family
    fit["metric"] = CHECKPOINT_METRIC
    fit["step"] = final_step

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in final.sort_values("n_params").iterrows():
        color = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        ax.scatter(row["n_params"], row[CHECKPOINT_METRIC], s=80, color=color, marker=marker, alpha=0.9)
        ax.annotate(
            row["label"],
            (row["n_params"], row[CHECKPOINT_METRIC]),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=7,
        )
    if fit.get("ok") and len(final) > 1:
        x_line = np.linspace(final["log10_n_params"].min(), final["log10_n_params"].max(), 100)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(10 ** x_line, y_line, "--", color="#444444", lw=1.2,
                label=f"fit R²={fit['r2']:.3f}")
        ax.legend(loc="best", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel(_ylabel_mean_spearman(window=True))
    ax.set_title(f"{title} — {family} (final step {final_step})")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fit


def _collect_checkpoint_layerwise_final(
    search_root: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """Layer profiles at the final training step per model."""
    rows: list[pd.DataFrame] = []
    for scores_csv in _discover_checkpoint_score_csvs(search_root, families):
        family, model_dir, model_id = _family_model_from_scores_csv(search_root, scores_csv)
        layer_csv = scores_csv.parent / "podcast_checkpoint_layerwise.csv"
        if not layer_csv.is_file():
            continue
        try:
            scores = pd.read_csv(scores_csv)
            layers = pd.read_csv(layer_csv)
        except Exception:
            continue
        if scores.empty or layers.empty:
            continue
        final_step = int(scores["step"].max())
        rev = scores.loc[scores["step"] == final_step, "revision"]
        revision = str(rev.iloc[0]) if not rev.empty else None
        sub = layers[layers["step"] == final_step]
        if revision is not None and "revision" in sub.columns:
            sub = sub[sub["revision"] == revision]
        if sub.empty:
            continue
        m = meta.get(model_id, meta.get(model_dir, {}))
        sub = sub.copy()
        if "mean_spearmanr" in sub.columns:
            sub[LAYER_PROFILE_METRIC] = sub["mean_spearmanr"]
        elif "mean_pearsonr" in sub.columns:
            sub[LAYER_PROFILE_METRIC] = sub["mean_pearsonr"]
        sub["family"] = family
        sub["model_id"] = model_id
        sub["label"] = m.get("label", model_id.split("/")[-1])
        sub["n_params"] = m.get("n_params", np.nan)
        sub["final_step"] = final_step
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _plot_checkpoint_layer_profiles(lw_df: pd.DataFrame, out_dir: Path) -> None:
    if lw_df.empty:
        print("  [skip] no checkpoint layerwise data found")
        return
    families = sorted(lw_df["family"].unique())
    ncols = min(3, len(families))
    nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    for i, fam in enumerate(families):
        ax = axes[i // ncols][i % ncols]
        fam_df = lw_df[lw_df["family"] == fam]
        model_ids = (
            fam_df.drop_duplicates("model_id")
            .sort_values("n_params")["model_id"]
            .tolist()
        )
        base_colors = plt.cm.tab20(np.linspace(0, 1, max(len(model_ids), 1)))
        rng = np.random.default_rng(seed=42)
        colors = rng.permutation(base_colors)
        for color, mid in zip(colors, model_ids):
            g = fam_df[fam_df["model_id"] == mid].sort_values("layer_rel")
            if g.empty:
                continue
            label = g["label"].iloc[0]
            ax.plot(g["layer_rel"], g[LAYER_PROFILE_METRIC], lw=1.6, alpha=0.85, color=color, label=label)
            peak_idx = g[LAYER_PROFILE_METRIC].idxmax()
            ax.annotate(
                label,
                xy=(g.loc[peak_idx, "layer_rel"], g.loc[peak_idx, LAYER_PROFILE_METRIC]),
                fontsize=6, color=color,
                xytext=(3, 2), textcoords="offset points",
            )
        ax.set_title(fam, fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative layer depth", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel(_ylabel_mean_spearman(), fontsize=9)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
        ax.grid(True, alpha=0.25)
    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Podcast ECoG — Layer profiles at final checkpoint", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "layer_profiles_final_checkpoint.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def summarize_checkpoint_family(
    search_root: Path,
    family: str,
    out_root: Path,
    meta: dict[str, dict],
) -> pd.DataFrame:
    out_dir = out_root / family
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _collect_checkpoint_scores(search_root, [family], meta)
    if df.empty:
        print(f"[{family}] no multi-checkpoint scores found")
        return df

    csv_path = out_dir / "all_models_podcast_checkpoint_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    curves_path = out_dir / "all_models_podcast_training_curves.png"
    _plot_checkpoint_training_curves(df, curves_path, family)
    print(f"  Saved: {curves_path}")

    fit = _plot_checkpoint_scaling(
        df,
        out_dir / "scaling_law_final.png",
        family,
        title="Podcast ECoG scaling law (final checkpoint)",
    )
    (out_dir / "scaling_fit_results.json").write_text(
        json.dumps({"final_checkpoint": fit}, indent=2),
        encoding="utf-8",
    )

    lw_df = _collect_checkpoint_layerwise_final(search_root, [family], meta)
    _plot_checkpoint_layer_profiles(lw_df, out_dir)

    print(f"[{family}] {df['model_id'].nunique()} models, {df['step'].nunique()} checkpoints")
    return df


def run_checkpoint_summary(
    sweep_dir: Path,
    out_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> None:
    search_root = _checkpoint_search_root(sweep_dir)
    all_df = _collect_checkpoint_scores(search_root, families, meta)
    if all_df.empty:
        sys.exit(
            f"No multi-checkpoint podcast_checkpoint_scores.csv found under {search_root} "
            f"(need ≥{MIN_CHECKPOINT_STEPS} steps per model)."
        )

    discovered = sorted(all_df["family"].unique())
    print(f"Checkpoint root: {search_root}")
    print(f"Found {all_df['model_id'].nunique()} models across {len(discovered)} families:")
    print(f"  {', '.join(discovered)}\n")

    for fam in discovered:
        print(f"[{fam}]")
        summarize_checkpoint_family(search_root, fam, out_dir, meta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoints", action="store_true",
        help="Summarize checkpoint sweeps (podcast_checkpoint_scores.csv) instead of final scores",
    )
    p.add_argument(
        "--sweep_dir", type=Path,
        default=Path("outputs_jeanzay/podcast/checkpoint_sweep"),
        help="Read-only sweep results (default: outputs_jeanzay/podcast/checkpoint_sweep)",
    )
    p.add_argument(
        "--out_dir", type=Path,
        default=None,
        help="Output directory (default: outputs/podcast/summary or outputs/podcast/summary/checkpoint)",
    )
    p.add_argument(
        "--families", nargs="+", default=None,
        help="Restrict to these families (default: all)",
    )
    args = p.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    out_dir = (args.out_dir or (
        Path("outputs/podcast/summary/checkpoint") if args.checkpoints
        else Path("outputs/podcast/summary")
    )).resolve()

    if not sweep_dir.is_dir():
        sys.exit(f"sweep_dir not found: {sweep_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from : {sweep_dir}")
    print(f"Writing to   : {out_dir}")

    meta = _registry_meta()

    if args.checkpoints:
        print(f"Mode         : checkpoint sweep")
        print(f"Time window  : [{TC_TMIN:+.1f}, {TC_TMAX:+.1f}] s\n")
        run_checkpoint_summary(sweep_dir, out_dir, args.families, meta)
        print("\nDone.")
        return

    print(f"Time window  : [{TC_TMIN:+.1f}, {TC_TMAX:+.1f}] s\n")

    df = load_all_models(sweep_dir, args.families, meta)

    if df.empty:
        sys.exit("No podcast_by_electrode.csv files found — nothing to summarize.")

    # Print summary table
    print(f"Found {len(df)} models across {df['family'].nunique()} families:\n")
    hdr = f"  {'family':<10} {'model':<28} {'n_subj':>6} {'n_elec':>7} {'mean_ρ':>8}  "
    hdr += "  ".join(f"{g:>6}" for g in ELECTRODE_GROUPS)
    hdr += "  source"
    print(hdr)
    print("  " + "-" * len(hdr))
    for _, row in df.iterrows():
        group_vals = "  ".join(
            f"{row[f'{SUMMARY_METRIC}_{g}']:>6.4f}" if np.isfinite(row[f'{SUMMARY_METRIC}_{g}']) else f"{'n/a':>6}"
            for g in ELECTRODE_GROUPS
        )
        print(
            f"  {row['family']:<10} {row['label']:<28} "
            f"{row['n_subjects']:>6} {row['n_electrodes']:>7} "
            f"{row[SUMMARY_METRIC]:>8.4f}  {group_vals}  {row['score_source']}"
        )
    print()

    # Save summary CSV
    out_csv = out_dir / "all_models_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # Plots
    plot_scaling_law(df, out_dir)
    plot_scaling_by_family(df, out_dir)
    plot_scaling_by_group(df, out_dir)

    print("\nLoading layerwise data...")
    lw_df = load_layerwise(sweep_dir, args.families, meta)
    plot_layer_profiles(lw_df, out_dir)

    print("\nLoading timecourse data...")
    tc_df = load_timecourses(sweep_dir, args.families, meta)
    plot_timecourse(tc_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
