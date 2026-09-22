#!/usr/bin/env python3
"""Summarize Nieuwland et al. (2018) N400 EEG encoding results across model families.

Two modes:

**Final checkpoint** (default) — brain score from the last training step.
  Reads ``nieuwland_checkpoint_scores.csv`` from each model directory.

**Checkpoint sweep** (``--checkpoints``) — training dynamics.
  Same CSV, but all revisions kept, plotted as training curves per family.

EEG channels: Cz, C3, C4 (central), Pz, P3, P4 (parietal).
N400 window : 0.200 – 0.500 s post-word-onset.
Full window : −0.200 – 0.800 s.

Primary metric: pc_sr = mean_subject_perchannel_spearmanr (per-subject, metric 3).

Outputs (default ``outputs/nieuwland/summary/``):
  all_models_scores.csv
  scaling_law.png                    — all families, pc_sr vs model size
  scaling_law_by_family.png          — one panel per family
  scaling_law_by_channel.png         — one panel per EEG channel (pc_sr)
  layer_profiles.png                 — per-layer profile per family
  waveform_timecourse.png            — ERP correlation over full window, all families
  waveform_timecourse_by_family.png  — individual family panels
  waveform_per_channel.png           — one panel per EEG channel

Checkpoint mode (default ``outputs/nieuwland/summary/checkpoint/``):
  <family>/all_models_training_curves.png
  <family>/scaling_law_final.png
  <family>/layer_profiles_final.png

Usage
-----
  python scripts/summarize/nieuwland.py
  python scripts/summarize/nieuwland.py \\
      --sweep_dir outputs_jeanzay/nieuwland/checkpoint_sweep \\
      --out_dir   outputs/nieuwland/summary
  python scripts/summarize/nieuwland.py --families gpt2 pythia
  python scripts/summarize/nieuwland.py --checkpoints \\
      --sweep_dir outputs_jeanzay/nieuwland/checkpoint_sweep
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

N400_TMIN = 0.200
N400_TMAX = 0.500
FULL_TMIN = -0.200
FULL_TMAX =  0.800

CHANNELS      = ["Cz", "C3", "C4", "Pz", "P3", "P4"]
CHAN_GROUPS   = {"central": ["Cz", "C3", "C4"], "parietal": ["Pz", "P3", "P4"]}
CHAN_COLORS   = {"Cz": "#1f77b4", "C3": "#aec7e8", "C4": "#ff7f0e",
                 "Pz": "#2ca02c", "P3": "#98df8a", "P4": "#d62728"}

PC_SR_COL = "mean_subject_perchannel_spearmanr"   # primary metric
WF_SR_COL = "mean_subject_waveform_spearmanr"

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
MIN_CHECKPOINT_STEPS = 2


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
    spec_lit   = re.compile(r'LppModelSpec\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
    spec_var   = re.compile(r'LppModelSpec\(\s*([A-Z][A-Z0-9_]+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
    def _add(model_id, label, n_layers, n_params, fam):
        e = {"family": fam, "label": label, "n_params": float(n_params), "n_layers": int(n_layers)}
        meta[model_id] = e; meta[model_id.replace("/", "_")] = e
    for fam, block in fam_blocks:
        for pattern in (main_re, spec_lit):
            for m in pattern.finditer(block):
                _add(*m.groups()[:4], fam)
        for m in spec_var.finditer(block):
            var, label, nl, np_ = m.groups()[:4]
            mid = var_map.get(var)
            if mid: _add(mid, label, nl, np_, fam)
    return meta


def _revision_step(rev: str) -> int:
    m = re.search(r"step(\d+)", rev)
    return int(m.group(1)) if m else (0 if rev == "main" else -1)


def _pick_final_revision(revisions: list[str]) -> str:
    return max(revisions, key=_revision_step)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _scores_csv_paths(sweep_dir: Path, families: list[str] | None) -> list[Path]:
    return [
        p for p in sorted(sweep_dir.glob("*/*/nieuwland_checkpoint_scores.csv"))
        if families is None or p.parent.parent.name in families
    ]


def load_all_models(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
    *,
    all_revisions: bool = False,
) -> pd.DataFrame:
    """One row per model (final revision) or per (model × revision) if all_revisions."""
    rows: list[pd.DataFrame] = []
    for csv_path in _scores_csv_paths(sweep_dir, families):
        family   = csv_path.parent.parent.name
        model_dir = csv_path.parent.name
        model_id  = model_dir.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir, {}))
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "family",   family)
        df.insert(1, "model_id", model_id)
        df.insert(2, "label",    m.get("label", model_id.split("/")[-1]))
        df["n_params"]  = df.get("n_params",  m.get("n_params",  np.nan))
        df["n_layers"]  = df.get("n_layers",  m.get("n_layers",  np.nan))
        if not all_revisions and "revision" in df.columns and df["revision"].nunique() > 1:
            final_rev = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == final_rev]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["family", "n_params"]).reset_index(drop=True)


def load_layerwise(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """Per-layer scores at the final checkpoint per model."""
    rows: list[pd.DataFrame] = []
    for lw_csv in sorted(sweep_dir.glob("*/*/nieuwland_checkpoint_layerwise.csv")):
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
        if df.empty or "layer" not in df.columns:
            continue
        # Keep final revision only
        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == rev]
        n_layers  = int(df["layer"].max()) + 1
        df = df.copy()
        df["layer_rel"] = df["layer"] / max(1, n_layers - 1)
        df["family"]    = family
        df["model_id"]  = model_id
        df["label"]     = m.get("label", model_id.split("/")[-1])
        df["n_params"]  = m.get("n_params", np.nan)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_waveform_npz(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> list[dict]:
    """
    Load per-model ERP correlation time courses from nieuwland_time_course_*.npz.

    Returns list of dicts, each with:
      family, model_id, label, n_params, revision,
      spearmanr  (n_layers, n_ch, n_t),
      times      (n_t,),
      channel_names (n_ch,)
    """
    records = []
    for model_dir in sorted(sweep_dir.glob("*/*")):
        family = model_dir.parent.name
        if families and family not in families:
            continue
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir.name, {}))
        # Pick latest revision npz
        npz_files = sorted(model_dir.glob("nieuwland_time_course_*.npz"))
        if not npz_files:
            continue
        # Choose the one with the highest step
        def _npz_step(p: Path) -> int:
            s = re.search(r"step(\d+)", p.stem)
            return int(s.group(1)) if s else 0
        npz_path = max(npz_files, key=_npz_step)
        try:
            d = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue
        if "spearmanr" not in d or "times" not in d:
            continue
        records.append({
            "family":        family,
            "model_id":      model_id,
            "label":         m.get("label", model_id.split("/")[-1]),
            "n_params":      m.get("n_params", np.nan),
            "revision":      str(d.get("revision", "main")),
            "spearmanr":     d["spearmanr"],        # (n_layers, n_ch, n_t)
            "pearsonr":      d.get("pearsonr"),
            "times":         d["times"],            # (n_t,)
            "channel_names": list(d["channel_names"]),
        })
    return sorted(records, key=lambda r: (r["family"], r.get("n_params", 0) or 0))


# ---------------------------------------------------------------------------
# Scaling-law fit
# ---------------------------------------------------------------------------

def _fit_log_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    lx = np.log10(x[mask]); yy = y[mask]
    coeffs = np.polyfit(lx, yy, 1)
    y_hat  = coeffs[0] * lx + coeffs[1]
    ss_res = float(np.sum((yy - y_hat) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coeffs[0]), float(coeffs[1]), float(r2)


def _scaling_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    *,
    annotate: bool = True,
) -> None:
    plot_df = df.dropna(subset=["n_params", metric]).copy()
    if plot_df.empty:
        ax.set_title(f"{title} (no data)"); return
    for family, grp in plot_df.groupby("family"):
        color  = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        grp_s  = grp.sort_values("n_params")
        ax.scatter(grp_s["n_params"], grp_s[metric],
                   label=family, color=color, marker=marker, s=80, zorder=3, alpha=0.9)
        if len(grp_s) > 1:
            ax.plot(grp_s["n_params"], grp_s[metric], color=color, lw=0.8, alpha=0.45)
        if annotate:
            for _, row in grp_s.iterrows():
                ax.annotate(row["label"], (row["n_params"], row[metric]),
                            fontsize=6, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points")
    a, b, r2 = _fit_log_linear(plot_df["n_params"].to_numpy(float), plot_df[metric].to_numpy(float))
    if np.isfinite(a):
        x_fit = np.logspace(np.log10(plot_df["n_params"].min()), np.log10(plot_df["n_params"].max()), 200)
        ax.plot(x_fit, a * np.log10(x_fit) + b, "k--", lw=1.3, alpha=0.55,
                label=f"fit: {a:+.4f}·log₁₀(N) {b:+.4f}  R²={r2:.3f}")
    ax.set_xscale("log"); ax.set_xlabel("Number of parameters", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, which="both", alpha=0.25)


# ---------------------------------------------------------------------------
# Plot: scaling law (all families, pc_sr)
# ---------------------------------------------------------------------------

def plot_scaling_law(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    _scaling_ax(ax, df, PC_SR_COL,
                "N400 EEG — Brain score (pc_sr) vs model size",
                "Mean Spearman ρ  [per-subject, pc_sr]")
    fig.tight_layout()
    fname = out_dir / "scaling_law.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot: scaling law per family
# ---------------------------------------------------------------------------

def plot_scaling_by_family(df: pd.DataFrame, out_dir: Path) -> None:
    families = sorted(df["family"].unique())
    if not families: return
    metric = PC_SR_COL
    ylabel = "Mean Spearman ρ  [pc_sr]"
    ncols = min(3, len(families)); nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    for i, fam in enumerate(families):
        ax     = axes[i // ncols][i % ncols]
        fam_df = df[df["family"] == fam].dropna(subset=["n_params", metric]).sort_values("n_params")
        color  = FAMILY_COLORS.get(fam, "gray"); marker = FAMILY_MARKERS.get(fam, "o")
        ax.scatter(fam_df["n_params"], fam_df[metric], color=color, marker=marker, s=80, zorder=3, alpha=0.9)
        if len(fam_df) > 1:
            ax.plot(fam_df["n_params"], fam_df[metric], color=color, lw=1.0, alpha=0.5)
        for _, row in fam_df.iterrows():
            ax.annotate(row["label"], (row["n_params"], row[metric]),
                        fontsize=6, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
        a, b, r2 = _fit_log_linear(fam_df["n_params"].to_numpy(float), fam_df[metric].to_numpy(float))
        if np.isfinite(a) and len(fam_df) > 1:
            x_fit = np.logspace(np.log10(fam_df["n_params"].min()), np.log10(fam_df["n_params"].max()), 200)
            ax.plot(x_fit, a * np.log10(x_fit) + b, "--", color=color, lw=1.5, alpha=0.7,
                    label=f"slope={a:+.4f}  R²={r2:.2f}")
            ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
        ax.set_xscale("log"); ax.set_title(fam, fontsize=10, fontweight="bold", color=color)
        if i % ncols == 0: ax.set_ylabel(ylabel, fontsize=8)
        if i // ncols == nrows - 1: ax.set_xlabel("Number of parameters", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Nieuwland N400 — Scaling law by family (pc_sr)", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "scaling_law_by_family.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot: scaling law per EEG channel (item-mean)
# ---------------------------------------------------------------------------

def plot_scaling_by_channel(df: pd.DataFrame, out_dir: Path) -> None:
    # Prefer pc_sr per channel (primary metric); fall back to item_mean for old runs
    def _pick_col(ch: str) -> tuple[str, str] | None:
        for col, label in [
            (f"ch_{ch}_pc_sr",              "pc_sr [per-subject]"),
            (f"ch_{ch}_item_mean_spearmanr", "item-mean spearmanr"),
            (f"ch_{ch}_spearmanr",           "timecourse N400 mean"),
        ]:
            if col in df.columns and df[col].notna().any():
                return col, label
        return None

    ch_info = {ch: _pick_col(ch) for ch in CHANNELS}
    ch_info = {ch: v for ch, v in ch_info.items() if v is not None}
    if not ch_info:
        print("  [skip] no per-channel columns found"); return

    # Report which metric is used
    metrics_used = set(label for _, (_, label) in ch_info.items())
    print(f"  Per-channel scaling: using {metrics_used}")

    ncols = 3; nrows = (len(ch_info) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False)
    for i, (ch, (col, ylabel)) in enumerate(ch_info.items()):
        ax = axes[i // ncols][i % ncols]
        _scaling_ax(ax, df, col,
                    f"Channel {ch}",
                    f"Mean Spearman ρ  [{ylabel}]", annotate=False)
    for j in range(len(ch_info), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    # Title reflects whichever metric is actually shown
    primary_label = next(iter(ch_info.values()))[1]
    fig.suptitle(f"Nieuwland N400 — Scaling law per EEG channel ({primary_label})", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "scaling_law_by_channel.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot: layer profiles
# ---------------------------------------------------------------------------

def plot_layer_profiles(lw_df: pd.DataFrame, out_dir: Path) -> None:
    if lw_df.empty:
        print("  [skip] no layerwise data found"); return
    # Use pc_sr (main metric) when available, fall back to mean_spearmanr for old runs
    metric = "pc_sr" if ("pc_sr" in lw_df.columns and lw_df["pc_sr"].notna().any()) else "mean_spearmanr"
    ylabel = "Mean Spearman ρ  [pc_sr, per-subject]" if metric == "pc_sr" else "Mean Spearman ρ  [N400 window]"
    print(f"  Layer profiles: using column '{metric}'")
    families = sorted(lw_df["family"].unique())
    ncols = min(3, len(families)); nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    for i, fam in enumerate(families):
        ax     = axes[i // ncols][i % ncols]
        fam_df = lw_df[lw_df["family"] == fam]
        model_ids = (fam_df.drop_duplicates("model_id").sort_values("n_params")["model_id"].tolist())
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(model_ids), 1)))
        rng    = np.random.default_rng(seed=42); colors = rng.permutation(colors)
        for color, mid in zip(colors, model_ids):
            g = fam_df[fam_df["model_id"] == mid].sort_values("layer_rel")
            if g.empty or metric not in g.columns or g[metric].isna().all(): continue
            label = g["label"].iloc[0]
            ax.plot(g["layer_rel"], g[metric], lw=1.6, alpha=0.85, color=color, label=label)
            peak_idx = g[metric].idxmax()
            ax.annotate(label, xy=(g.loc[peak_idx, "layer_rel"], g.loc[peak_idx, metric]),
                        fontsize=6, color=color, xytext=(3, 2), textcoords="offset points")
        ax.set_title(fam, fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative layer depth", fontsize=9)
        if i % ncols == 0: ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
        ax.grid(True, alpha=0.25)
    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Nieuwland N400 — Layer profiles by family", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "layer_profiles.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot: ERP waveform (full window, correlation timecourse)
# ---------------------------------------------------------------------------

def _waveform_at_best_layer(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (times, mean_sr_over_channels) at the best layer.
    Best layer = argmax of mean sr in the N400 window averaged over channels.
    """
    sr    = rec["spearmanr"]      # (n_layers, n_ch, n_t)
    times = rec["times"]
    n400  = (times >= N400_TMIN) & (times <= N400_TMAX)
    # mean over channels, then mean over N400 timepoints, per layer
    layer_score = np.nanmean(sr[:, :, n400], axis=(1, 2))  # (n_layers,)
    best_li     = int(np.nanargmax(layer_score))
    mean_ch     = np.nanmean(sr[best_li], axis=0)           # (n_t,) mean over channels
    return times, mean_ch


def _waveform_per_channel(rec: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (times, {channel_name: sr_at_best_layer})."""
    sr    = rec["spearmanr"]
    times = rec["times"]
    n400  = (times >= N400_TMIN) & (times <= N400_TMAX)
    layer_score = np.nanmean(sr[:, :, n400], axis=(1, 2))
    best_li     = int(np.nanargmax(layer_score))
    ch_names    = rec["channel_names"]
    return times, {ch: sr[best_li, ci] for ci, ch in enumerate(ch_names)}


def plot_waveforms(records: list[dict], out_dir: Path) -> None:
    """Grand overview: one curve per family (mean across models), all channels averaged."""
    if not records:
        print("  [skip] no waveform .npz files found"); return

    # Collect per-model waveforms
    fam_model_waves: dict[str, dict[str, np.ndarray]] = {}  # family → model_id → (n_t,)
    times_ref: np.ndarray | None = None
    for rec in records:
        t, wave = _waveform_at_best_layer(rec)
        if times_ref is None: times_ref = t
        fam_model_waves.setdefault(rec["family"], {})[rec["model_id"]] = wave

    if times_ref is None: return
    families = sorted(fam_model_waves.keys())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axvspan(N400_TMIN, N400_TMAX, color="#ffe0e0", alpha=0.5, zorder=0,
               label=f"N400 window [{N400_TMIN:.2f}–{N400_TMAX:.2f}s]")
    ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color="gray", lw=0.4, ls=":", zorder=1)

    all_waves = []
    for fam in families:
        waves = np.stack(list(fam_model_waves[fam].values()), axis=0)  # (n_models, n_t)
        fam_mean = np.nanmean(waves, axis=0)
        fam_sem  = np.nanstd(waves, axis=0) / max(1, np.sqrt(waves.shape[0]))
        color = FAMILY_COLORS.get(fam, "gray")
        ax.plot(times_ref, fam_mean, color=color, lw=1.5, label=fam, alpha=0.85)
        ax.fill_between(times_ref, fam_mean - fam_sem, fam_mean + fam_sem,
                        color=color, alpha=0.12)
        all_waves.append(waves)

    # Grand average
    grand = np.nanmean(np.concatenate(all_waves, axis=0), axis=0)
    ax.plot(times_ref, grand, color="black", lw=2.5, label="grand average", zorder=5)

    ax.set_xlabel("Time relative to word onset (s)", fontsize=11)
    ax.set_ylabel("Mean Spearman ρ (channels × subjects)", fontsize=11)
    ax.set_title("Nieuwland N400 — ERP correlation timecourse (best layer per model, mean ± SEM across models)", fontsize=11)
    ax.set_xlim(FULL_TMIN - 0.05, FULL_TMAX + 0.05)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fname = out_dir / "waveform_timecourse.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


def plot_waveforms_by_family(records: list[dict], out_dir: Path) -> None:
    """One panel per family — each curve is one model, sorted by n_params."""
    if not records:
        return

    families = sorted({r["family"] for r in records})
    ncols = min(3, len(families)); nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5),
                             sharey=False, squeeze=False)

    for i, fam in enumerate(families):
        ax   = axes[i // ncols][i % ncols]
        recs = sorted([r for r in records if r["family"] == fam],
                      key=lambda r: r.get("n_params") or 0)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(recs), 1)))

        ax.axvspan(N400_TMIN, N400_TMAX, color="#ffe0e0", alpha=0.4, zorder=0)
        ax.axvline(0, color="gray", lw=0.7, ls="--", zorder=1)
        ax.axhline(0, color="gray", lw=0.4, ls=":", zorder=1)

        for color, rec in zip(colors, recs):
            t, wave = _waveform_at_best_layer(rec)
            ax.plot(t, wave, lw=1.5, alpha=0.85, color=color, label=rec["label"])
            # Label at N400 trough
            n400_mask = (t >= N400_TMIN) & (t <= N400_TMAX)
            if n400_mask.any():
                peak_ti = int(np.nanargmin(wave[n400_mask]))
                t_peak  = t[n400_mask][peak_ti]
                y_peak  = wave[n400_mask][peak_ti]
                ax.annotate(rec["label"], xy=(t_peak, y_peak), fontsize=6, color=color,
                            xytext=(3, -8), textcoords="offset points")

        ax.set_title(fam, fontsize=10, fontweight="bold", color=FAMILY_COLORS.get(fam, "gray"))
        ax.set_xlim(FULL_TMIN - 0.05, FULL_TMAX + 0.05)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
        if i % ncols == 0: ax.set_ylabel("Mean Spearman ρ", fontsize=9)
        if i // ncols == nrows - 1: ax.set_xlabel("Time (s)", fontsize=9)

    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Nieuwland N400 — ERP timecourse by family (all channels, best layer)", fontsize=12, y=1.01)
    fig.tight_layout()
    fname = out_dir / "waveform_timecourse_by_family.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


def plot_waveforms_by_channel(records: list[dict], out_dir: Path) -> None:
    """One panel per EEG channel — all models overlaid per channel."""
    if not records:
        return
    # Collect per-channel waveforms across all models
    ch_waves: dict[str, list[tuple[str, str, np.ndarray]]] = {ch: [] for ch in CHANNELS}
    times_ref: np.ndarray | None = None
    for rec in records:
        t, ch_dict = _waveform_per_channel(rec)
        if times_ref is None: times_ref = t
        for ch, wave in ch_dict.items():
            if ch in ch_waves:
                ch_waves[ch].append((rec["family"], rec["label"], wave))

    if times_ref is None: return
    ncols = 3; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5),
                             sharey=True, squeeze=False)
    for i, ch in enumerate(CHANNELS):
        ax = axes[i // ncols][i % ncols]
        ax.axvspan(N400_TMIN, N400_TMAX, color="#ffe0e0", alpha=0.4, zorder=0)
        ax.axvline(0, color="gray", lw=0.7, ls="--", zorder=1)
        ax.axhline(0, color="gray", lw=0.4, ls=":", zorder=1)
        for fam, label, wave in ch_waves[ch]:
            color = FAMILY_COLORS.get(fam, "gray")
            ax.plot(times_ref, wave, lw=1.2, alpha=0.6, color=color)
        # Family mean
        for fam in sorted({x[0] for x in ch_waves[ch]}):
            waves = np.stack([w for f, _, w in ch_waves[ch] if f == fam], axis=0)
            ax.plot(times_ref, np.nanmean(waves, axis=0), lw=2.2,
                    color=FAMILY_COLORS.get(fam, "gray"), alpha=0.9, label=fam)
        ax.set_title(f"Channel {ch}", fontsize=10, fontweight="bold", color=CHAN_COLORS.get(ch, "gray"))
        ax.set_xlim(FULL_TMIN - 0.05, FULL_TMAX + 0.05)
        ax.grid(True, alpha=0.2)
        if i % ncols == 0: ax.set_ylabel("Mean Spearman ρ", fontsize=9)
        if i // ncols == nrows - 1: ax.set_xlabel("Time (s)", fontsize=9)
        ax.legend(fontsize=6, framealpha=0.7)
    fig.suptitle("Nieuwland N400 — ERP timecourse per EEG channel (best layer)", fontsize=12, y=1.01)
    fig.tight_layout()
    fname = out_dir / "waveform_per_channel.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Checkpoint sweep mode
# ---------------------------------------------------------------------------

def _collect_checkpoint_df(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    return load_all_models(sweep_dir, families, meta, all_revisions=True)


def _plot_training_curves(df: pd.DataFrame, out_path: Path, family: str) -> None:
    metric = PC_SR_COL; ylabel = "Per-subject pc_sr"
    fig, ax = plt.subplots(figsize=(10, 6))
    if metric in df.columns:
        for model_id, grp in df.groupby("model_id"):
            g = grp.dropna(subset=["step", metric]).sort_values("step")
            if g.empty: continue
            label     = g["label"].iloc[0]
            final_val = float(g[metric].iloc[-1])
            ax.plot(g["step"], g[metric], "o-", lw=1.6, alpha=0.9,
                    label=f"{label} ({final_val:.4f})")
    ax.set_xlabel("Training step"); ax.set_ylabel(ylabel)
    ax.set_title(f"Nieuwland N400 — {family} — {ylabel}")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_checkpoint_scaling(df: pd.DataFrame, out_path: Path, family: str, metric: str, ylabel: str) -> None:
    if metric not in df.columns or "step" not in df.columns: return
    final_step = int(df["step"].max())
    final = df[df["step"] == final_step].dropna(subset=["n_params", metric])
    if final.empty: return
    fig, ax = plt.subplots(figsize=(9, 6))
    color = FAMILY_COLORS.get(family, "gray"); marker = FAMILY_MARKERS.get(family, "o")
    ax.scatter(final["n_params"], final[metric], s=80, color=color, marker=marker, alpha=0.9)
    for _, row in final.sort_values("n_params").iterrows():
        ax.annotate(row["label"], (row["n_params"], row[metric]),
                    textcoords="offset points", xytext=(4, 2), fontsize=7)
    a, b, r2 = _fit_log_linear(final["n_params"].to_numpy(float), final[metric].to_numpy(float))
    if np.isfinite(a) and len(final) > 1:
        x_fit = np.logspace(np.log10(final["n_params"].min()), np.log10(final["n_params"].max()), 200)
        ax.plot(x_fit, a * np.log10(x_fit) + b, "k--", lw=1.3,
                label=f"fit: {a:+.4f}·log₁₀(N) {b:+.4f}  R²={r2:.3f}")
        ax.legend(fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Number of parameters")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Nieuwland N400 — Scaling law — {family} (step {final_step})")
    ax.grid(True, which="both", alpha=0.25); fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_checkpoint_layer_profiles(
    lw_df: pd.DataFrame,
    out_path: Path,
    family: str,
) -> None:
    if lw_df.empty: return
    fam_df    = lw_df[lw_df["family"] == family]
    model_ids = fam_df.drop_duplicates("model_id").sort_values("n_params")["model_id"].tolist()
    if not model_ids: return
    metric = "pc_sr" if ("pc_sr" in fam_df.columns and fam_df["pc_sr"].notna().any()) else "mean_spearmanr"
    ylabel = "Mean Spearman ρ  [pc_sr, per-subject]" if metric == "pc_sr" else "Mean Spearman ρ  [N400]"
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(model_ids)))
    fig, ax = plt.subplots(figsize=(9, 6))
    for color, mid in zip(colors, model_ids):
        g = fam_df[fam_df["model_id"] == mid].sort_values("layer_rel")
        if g.empty or metric not in g.columns or g[metric].isna().all(): continue
        label = g["label"].iloc[0]
        ax.plot(g["layer_rel"], g[metric], lw=1.6, alpha=0.85, color=color, label=label)
    ax.set_xlabel("Relative layer depth"); ax.set_ylabel(ylabel)
    ax.set_title(f"Nieuwland N400 — Layer profiles at final checkpoint — {family}")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def summarize_checkpoint_family(
    sweep_dir: Path,
    family: str,
    out_root: Path,
    meta: dict[str, dict],
) -> pd.DataFrame:
    out_dir = out_root / family; out_dir.mkdir(parents=True, exist_ok=True)
    df = _collect_checkpoint_df(sweep_dir, [family], meta)
    if df.empty:
        print(f"[{family}] no checkpoint data found"); return df
    if "step" not in df.columns or df["step"].nunique() < MIN_CHECKPOINT_STEPS:
        print(f"[{family}] fewer than {MIN_CHECKPOINT_STEPS} checkpoints — skipping"); return df

    csv_path = out_dir / "all_models_training_curves.csv"
    df.to_csv(csv_path, index=False); print(f"  Saved: {csv_path}")

    _plot_training_curves(df, out_dir / "all_models_training_curves.png", family)
    print(f"  Saved: {out_dir}/all_models_training_curves.png")

    _plot_checkpoint_scaling(df, out_dir / "scaling_law_final.png",
                             family, PC_SR_COL, "Per-subject pc_sr")
    print(f"  Saved: {out_dir}/scaling_law_final.png")

    lw_df = load_layerwise(sweep_dir, [family], meta)
    _plot_checkpoint_layer_profiles(lw_df, out_dir / "layer_profiles_final.png", family)
    print(f"  Saved: {out_dir}/layer_profiles_final.png")

    print(f"[{family}] {df['model_id'].nunique()} models, {df['step'].nunique()} checkpoints")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", action="store_true",
                   help="Summarize all checkpoint steps (training curves) instead of final only")
    p.add_argument("--sweep_dir", type=Path,
                   default=Path("outputs_jeanzay/nieuwland/checkpoint_sweep"),
                   help="Root of the sweep results")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Output directory")
    p.add_argument("--families", nargs="+", default=None,
                   help="Restrict to these families")
    args = p.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    if not sweep_dir.is_dir():
        sys.exit(f"sweep_dir not found: {sweep_dir}")

    if args.out_dir:
        out_dir = args.out_dir.resolve()
    elif args.checkpoints:
        out_dir = Path("outputs/nieuwland/summary/checkpoint").resolve()
    else:
        out_dir = Path("outputs/nieuwland/summary").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from : {sweep_dir}")
    print(f"Writing to   : {out_dir}")

    meta = _registry_meta()

    if args.checkpoints:
        print("Mode: checkpoint sweep\n")
        all_df = _collect_checkpoint_df(sweep_dir, args.families, meta)
        if all_df.empty:
            sys.exit(f"No nieuwland_checkpoint_scores.csv found under {sweep_dir}")
        families = sorted(all_df["family"].unique())
        print(f"Families: {', '.join(families)}\n")
        for fam in families:
            print(f"[{fam}]")
            summarize_checkpoint_family(sweep_dir, fam, out_dir, meta)
        print("\nDone.")
        return

    # ── Final-checkpoint mode ──────────────────────────────────────────────
    df = load_all_models(sweep_dir, args.families, meta)
    if df.empty:
        sys.exit("No nieuwland_checkpoint_scores.csv files found — nothing to summarize.")

    # Print summary table
    print(f"\nFound {len(df)} models across {df['family'].nunique()} families:\n")
    hdr = f"  {'family':<10} {'model':<28} {'n_params':>12}  {'pc_sr':>8}  {'wf_sr':>8}"
    print(hdr); print("  " + "-" * len(hdr))
    for _, row in df.iterrows():
        def _fmt(col): return f"{row[col]:>8.4f}" if col in row and np.isfinite(row.get(col, np.nan)) else f"{'n/a':>8}"
        print(f"  {row['family']:<10} {row['label']:<28} "
              f"{row.get('n_params', np.nan):>12.3g}  "
              f"{_fmt(PC_SR_COL)}  {_fmt(WF_SR_COL)}")
    print()

    # Save summary CSV
    out_csv = out_dir / "all_models_scores.csv"
    df.to_csv(out_csv, index=False); print(f"  Saved: {out_csv}")

    # ── Scaling law plots ──────────────────────────────────────────────────
    print("\nScaling law plots...")
    plot_scaling_law(df, out_dir)
    plot_scaling_by_family(df, out_dir)
    plot_scaling_by_channel(df, out_dir)

    # ── Layer profiles ─────────────────────────────────────────────────────
    print("\nLayer profiles...")
    lw_df = load_layerwise(sweep_dir, args.families, meta)
    plot_layer_profiles(lw_df, out_dir)

    # ── ERP waveform timecourses ───────────────────────────────────────────
    print("\nERP waveform timecourses...")
    records = load_waveform_npz(sweep_dir, args.families, meta)
    print(f"  Found waveform data for {len(records)} models")
    plot_waveforms(records, out_dir)
    plot_waveforms_by_family(records, out_dir)
    plot_waveforms_by_channel(records, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
