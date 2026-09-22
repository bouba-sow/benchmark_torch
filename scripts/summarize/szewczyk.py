#!/usr/bin/env python3
"""Summarize Szewczyk & Federmeier (2021) N400 EEG encoding results across model families.

Two modes:

**Final checkpoint** (default) — brain score from the last training step.
  Reads ``szewczyk_checkpoint_scores.csv`` from each model directory.

**Checkpoint sweep** (``--checkpoints``) — training dynamics.
  Same CSV, but all revisions kept, plotted as training curves per family.

Dataset: ~2,300 unique (subexperiment, item, critical_word) trials, ~138
subjects. N400 is a single scalar per trial (already averaged over 4
centro-parietal ROI clusters in the OSF deposit) — there is no per-electrode
breakdown, unlike Nieuwland.
Primary metric: mean_spearmanr — per-subject Spearman r (predicted vs actual
                N400 at best layer), averaged over ~138 subjects.

No discrete conditions (continuous cloze-probability manipulation instead).
Secondary diagnostics, in place of condition/effect plots:
  expected_spearmanr_best / unexpected_spearmanr_best — pooled OOF Spearman r,
    split by cloze_p >= 0.5 ("expected") vs < 0.5 ("unexpected")
  cloze_corr_pred_best / cloze_corr_actual — pooled Spearman r between
    predicted/actual N400 and cloze_p itself (effect-validity sanity check)

Outputs (default ``outputs/szewczyk_summary/``):
  all_models_scores.csv
  scaling_law.png                 — all families, mean_spearmanr vs model size
  scaling_law_by_family.png       — one panel per family
  scaling_law_by_cloze.png        — expected vs unexpected panels
  cloze_tracking.png              — corr(predicted/actual N400, cloze_p) vs size
  layer_profiles.png              — per-layer profile per family (final checkpoint)

Checkpoint mode (default ``outputs/szewczyk_summary/checkpoint/``):
  <family>/all_models_training_curves.png
  <family>/scaling_law_final.png
  <family>/layer_profiles_final.png

Usage
-----
  python scripts/summarize/szewczyk.py
  python scripts/summarize/szewczyk.py \\
      --sweep_dir outputs_jeanzay/szewczyk/embedding_sweep \\
      --out_dir   outputs/szewczyk_summary
  python scripts/summarize/szewczyk.py --families gpt2 pythia
  python scripts/summarize/szewczyk.py --checkpoints \\
      --sweep_dir outputs_jeanzay/szewczyk/embedding_sweep
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

PRIMARY_METRIC = "mean_spearmanr"

FAMILY_COLORS = {
    "gpt2":         "#1f77b4",
    "gptneo":       "#ff7f0e",
    "opt":          "#2ca02c",
    "llama32":      "#d62728",
    "mamba":        "#9467bd",
    "mamba2":       "#8c564b",
    "mistral":      "#e377c2",
    "qwen3":        "#7f7f7f",
    "pythia":       "#bcbd22",
    "olmo2":        "#17becf",
    "cbr_rnn":      "#e41a1c",
    "stickbreaking":"#a65628",
    "alibi":        "#f781bf",
    "forgetting":   "#999999",
    "babylm":       "#66c2a5",
}
FAMILY_MARKERS = {
    "gpt2": "o", "gptneo": "s", "opt": "^", "llama32": "D",
    "mamba": "v", "mamba2": "p", "mistral": "h", "qwen3": "*",
    "pythia": "P", "olmo2": "X", "cbr_rnn": "H",
    "stickbreaking": "<", "alibi": ">", "forgetting": "8", "babylm": "d",
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
    main_re  = re.compile(r'_main_only\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
    spec_lit = re.compile(r'LppModelSpec\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
    spec_var = re.compile(r'LppModelSpec\(\s*([A-Z][A-Z0-9_]+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
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
            if mid:
                _add(mid, label, nl, np_, fam)
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
        p for p in sorted(sweep_dir.glob("*/*/szewczyk_checkpoint_scores.csv"))
        if families is None or p.parent.parent.name in families
    ]


def load_all_models(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
    *,
    all_revisions: bool = False,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for csv_path in _scores_csv_paths(sweep_dir, families):
        family    = csv_path.parent.parent.name
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
        df["n_params"] = df.get("n_params", m.get("n_params", np.nan))
        df["n_layers"] = df.get("n_layers", m.get("n_layers", np.nan))
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
    rows: list[pd.DataFrame] = []
    for lw_csv in sorted(sweep_dir.glob("*/*/szewczyk_checkpoint_layerwise.csv")):
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
        if df.empty or "layer" not in df.columns:
            continue
        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == rev]
        n_layers = int(df["layer"].max()) + 1
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


# ---------------------------------------------------------------------------
# Scaling-law helpers
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
    a, b, r2 = _fit_log_linear(
        plot_df["n_params"].to_numpy(float), plot_df[metric].to_numpy(float)
    )
    if np.isfinite(a):
        x_fit = np.logspace(
            np.log10(plot_df["n_params"].min()), np.log10(plot_df["n_params"].max()), 200
        )
        ax.plot(x_fit, a * np.log10(x_fit) + b, "k--", lw=1.3, alpha=0.55,
                label=f"fit: {a:+.4f}·log₁₀(N) {b:+.4f}  R²={r2:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)


# ---------------------------------------------------------------------------
# Plot: scaling law (all families, primary metric)
# ---------------------------------------------------------------------------

def plot_scaling_law(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    _scaling_ax(ax, df, PRIMARY_METRIC,
                "Szewczyk N400 — Brain score (mean Spearman ρ) vs model size",
                "Mean Spearman ρ  [per-subject, best layer]")
    fig.tight_layout()
    fname = out_dir / "scaling_law.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot: scaling law per family
# ---------------------------------------------------------------------------

def plot_scaling_by_family(df: pd.DataFrame, out_dir: Path) -> None:
    families = sorted(df["family"].unique())
    if not families:
        return
    ncols = min(3, len(families)); nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    for i, fam in enumerate(families):
        ax     = axes[i // ncols][i % ncols]
        fam_df = df[df["family"] == fam].dropna(subset=["n_params", PRIMARY_METRIC]).sort_values("n_params")
        color  = FAMILY_COLORS.get(fam, "gray")
        marker = FAMILY_MARKERS.get(fam, "o")
        ax.scatter(fam_df["n_params"], fam_df[PRIMARY_METRIC], color=color, marker=marker, s=80, zorder=3, alpha=0.9)
        if len(fam_df) > 1:
            ax.plot(fam_df["n_params"], fam_df[PRIMARY_METRIC], color=color, lw=1.0, alpha=0.5)
        for _, row in fam_df.iterrows():
            ax.annotate(row["label"], (row["n_params"], row[PRIMARY_METRIC]),
                        fontsize=6, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
        a, b, r2 = _fit_log_linear(fam_df["n_params"].to_numpy(float), fam_df[PRIMARY_METRIC].to_numpy(float))
        if np.isfinite(a) and len(fam_df) > 1:
            x_fit = np.logspace(np.log10(fam_df["n_params"].min()), np.log10(fam_df["n_params"].max()), 200)
            ax.plot(x_fit, a * np.log10(x_fit) + b, "--", color=color, lw=1.5, alpha=0.7,
                    label=f"slope={a:+.4f}  R²={r2:.2f}")
            ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
        ax.set_xscale("log")
        ax.set_title(fam, fontsize=10, fontweight="bold", color=color)
        if i % ncols == 0:
            ax.set_ylabel("Mean Spearman ρ", fontsize=8)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Number of parameters", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Szewczyk N400 — Scaling law by family", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "scaling_law_by_family.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot: scaling law split by cloze expectedness (replaces per-condition plot —
# Szewczyk has no discrete conditions, only continuous cloze probability)
# ---------------------------------------------------------------------------

def plot_scaling_by_cloze(df: pd.DataFrame, out_dir: Path) -> None:
    cols = {
        "expected_spearmanr_best":   "Expected (cloze_p ≥ 0.5)",
        "unexpected_spearmanr_best": "Unexpected (cloze_p < 0.5)",
    }
    cols = {c: title for c, title in cols.items() if c in df.columns and df[c].notna().any()}
    if not cols:
        print("  [skip] no expected/unexpected columns in scores CSV"); return

    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5), squeeze=False)
    for i, (col, title) in enumerate(cols.items()):
        ax = axes[0][i]
        _scaling_ax(ax, df, col, title,
                    "Pooled Spearman ρ (predicted vs actual N400)", annotate=False)
    fig.suptitle("Szewczyk N400 — Scaling law by cloze expectedness (pooled OOF, median split)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "scaling_law_by_cloze.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot: cloze-tracking diagnostics (replaces per-effect plot — no discrete
# conditions here, so the "effect validity" check is a continuous correlation
# between predicted/actual N400 and cloze probability itself)
# ---------------------------------------------------------------------------

def plot_cloze_tracking(df: pd.DataFrame, out_dir: Path) -> None:
    cols = {
        "cloze_corr_pred_best": "corr(predicted N400, cloze_p)",
        "cloze_corr_actual":    "corr(actual N400, cloze_p)",
    }
    cols = {c: title for c, title in cols.items() if c in df.columns and df[c].notna().any()}
    if not cols:
        print("  [skip] no cloze-correlation columns in scores CSV"); return

    fig, ax = plt.subplots(figsize=(9, 6))
    for col, label in cols.items():
        plot_df = df.dropna(subset=["n_params", col])
        for family, grp in plot_df.groupby("family"):
            grp_s = grp.sort_values("n_params")
            ax.plot(grp_s["n_params"], grp_s[col], "o-", lw=1.2, alpha=0.7,
                    label=f"{label} ({family})")
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Szewczyk N400 — Cloze-probability tracking vs model size")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fname = out_dir / "cloze_tracking.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot: layer profiles
# ---------------------------------------------------------------------------

def plot_layer_profiles(lw_df: pd.DataFrame, out_dir: Path) -> None:
    if lw_df.empty:
        print("  [skip] no layerwise data found"); return
    families = sorted(lw_df["family"].unique())
    ncols = min(3, len(families)); nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    for i, fam in enumerate(families):
        ax     = axes[i // ncols][i % ncols]
        fam_df = lw_df[lw_df["family"] == fam]
        model_ids = fam_df.drop_duplicates("model_id").sort_values("n_params")["model_id"].tolist()
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(model_ids), 1)))
        rng    = np.random.default_rng(seed=42); colors = rng.permutation(colors)
        for color, mid in zip(colors, model_ids):
            g = fam_df[fam_df["model_id"] == mid].sort_values("layer_rel")
            if g.empty or g[PRIMARY_METRIC].isna().all():
                continue
            label = g["label"].iloc[0]
            ax.plot(g["layer_rel"], g[PRIMARY_METRIC], lw=1.6, alpha=0.85, color=color, label=label)
            peak_idx = g[PRIMARY_METRIC].idxmax()
            ax.annotate(label, xy=(g.loc[peak_idx, "layer_rel"], g.loc[peak_idx, PRIMARY_METRIC]),
                        fontsize=6, color=color, xytext=(3, 2), textcoords="offset points")
        ax.set_title(fam, fontsize=10, fontweight="bold", color=FAMILY_COLORS.get(fam, "gray"))
        ax.set_xlabel("Relative layer depth", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("Mean Spearman ρ", fontsize=9)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
        ax.grid(True, alpha=0.25)
    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Szewczyk N400 — Layer profiles by family (final checkpoint)", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "layer_profiles.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Checkpoint sweep mode
# ---------------------------------------------------------------------------

def _plot_training_curves(df: pd.DataFrame, out_path: Path, family: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_id, grp in df.groupby("model_id"):
        g = grp.dropna(subset=["step", PRIMARY_METRIC]).sort_values("step")
        if g.empty:
            continue
        label     = g["label"].iloc[0]
        final_val = float(g[PRIMARY_METRIC].iloc[-1])
        ax.plot(g["step"], g[PRIMARY_METRIC], "o-", lw=1.6, alpha=0.9,
                label=f"{label} ({final_val:.4f})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean Spearman ρ")
    ax.set_title(f"Szewczyk N400 — {family} — training curves")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_checkpoint_scaling(df: pd.DataFrame, out_path: Path, family: str) -> None:
    if PRIMARY_METRIC not in df.columns or "step" not in df.columns:
        return
    final_step = int(df["step"].max())
    final = df[df["step"] == final_step].dropna(subset=["n_params", PRIMARY_METRIC])
    if final.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    color  = FAMILY_COLORS.get(family, "gray")
    marker = FAMILY_MARKERS.get(family, "o")
    ax.scatter(final["n_params"], final[PRIMARY_METRIC], s=80, color=color, marker=marker, alpha=0.9)
    for _, row in final.sort_values("n_params").iterrows():
        ax.annotate(row["label"], (row["n_params"], row[PRIMARY_METRIC]),
                    textcoords="offset points", xytext=(4, 2), fontsize=7)
    a, b, r2 = _fit_log_linear(final["n_params"].to_numpy(float), final[PRIMARY_METRIC].to_numpy(float))
    if np.isfinite(a) and len(final) > 1:
        x_fit = np.logspace(np.log10(final["n_params"].min()), np.log10(final["n_params"].max()), 200)
        ax.plot(x_fit, a * np.log10(x_fit) + b, "k--", lw=1.3,
                label=f"fit: {a:+.4f}·log₁₀(N) {b:+.4f}  R²={r2:.3f}")
        ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Mean Spearman ρ")
    ax.set_title(f"Szewczyk N400 — Scaling law — {family} (step {final_step})")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_checkpoint_layer_profiles(lw_df: pd.DataFrame, out_path: Path, family: str) -> None:
    if lw_df.empty:
        return
    fam_df    = lw_df[lw_df["family"] == family]
    model_ids = fam_df.drop_duplicates("model_id").sort_values("n_params")["model_id"].tolist()
    if not model_ids:
        return
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(model_ids)))
    fig, ax = plt.subplots(figsize=(9, 6))
    for color, mid in zip(colors, model_ids):
        g = fam_df[fam_df["model_id"] == mid].sort_values("layer_rel")
        if g.empty or g[PRIMARY_METRIC].isna().all():
            continue
        ax.plot(g["layer_rel"], g[PRIMARY_METRIC], lw=1.6, alpha=0.85, color=color,
                label=g["label"].iloc[0])
    ax.set_xlabel("Relative layer depth")
    ax.set_ylabel("Mean Spearman ρ")
    ax.set_title(f"Szewczyk N400 — Layer profiles at final checkpoint — {family}")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def summarize_checkpoint_family(
    sweep_dir: Path,
    family: str,
    out_root: Path,
    meta: dict[str, dict],
) -> pd.DataFrame:
    out_dir = out_root / family; out_dir.mkdir(parents=True, exist_ok=True)
    df = load_all_models(sweep_dir, [family], meta, all_revisions=True)
    if df.empty:
        print(f"[{family}] no checkpoint data found"); return df
    if "step" not in df.columns or df["step"].nunique() < MIN_CHECKPOINT_STEPS:
        print(f"[{family}] fewer than {MIN_CHECKPOINT_STEPS} checkpoints — skipping"); return df

    csv_path = out_dir / "all_models_training_curves.csv"
    df.to_csv(csv_path, index=False); print(f"  Saved: {csv_path}")

    _plot_training_curves(df, out_dir / "all_models_training_curves.png", family)
    print(f"  Saved: {out_dir}/all_models_training_curves.png")

    _plot_checkpoint_scaling(df, out_dir / "scaling_law_final.png", family)
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
                   default=Path("outputs_jeanzay/szewczyk/embedding_sweep"),
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
        out_dir = Path("outputs/szewczyk_summary/checkpoint").resolve()
    else:
        out_dir = Path("outputs/szewczyk_summary").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from : {sweep_dir}")
    print(f"Writing to   : {out_dir}")

    meta = _registry_meta()

    if args.checkpoints:
        print("Mode: checkpoint sweep\n")
        all_df = load_all_models(sweep_dir, args.families, meta, all_revisions=True)
        if all_df.empty:
            sys.exit(f"No szewczyk_checkpoint_scores.csv found under {sweep_dir}")
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
        sys.exit("No szewczyk_checkpoint_scores.csv files found — nothing to summarize.")

    # Print summary table
    print(f"\nFound {len(df)} models across {df['family'].nunique()} families:\n")
    hdr = (f"  {'family':<14} {'model':<30} {'n_params':>12}  {'sr':>8}  {'last_sr':>8}  "
           f"{'expect':>8}  {'unexp':>8}  {'cloze_pr':>9}  {'cloze_ac':>9}")
    print(hdr); print("  " + "-" * len(hdr))
    for _, row in df.iterrows():
        def _f(col): return f"{row[col]:>8.4f}" if col in row.index and np.isfinite(float(row.get(col, np.nan))) else f"{'n/a':>8}"
        print(
            f"  {row['family']:<14} {row['label']:<30} "
            f"{float(row.get('n_params', np.nan)):>12.3g}  "
            f"{_f('mean_spearmanr')}  {_f('last_layer_mean_spearmanr')}  "
            f"{_f('expected_spearmanr_best')}  {_f('unexpected_spearmanr_best')}  "
            f"{row.get('cloze_corr_pred_best', float('nan')):>9.4f}  "
            f"{row.get('cloze_corr_actual', float('nan')):>9.4f}"
        )
    print()

    # Save summary CSV
    out_csv = out_dir / "all_models_scores.csv"
    df.to_csv(out_csv, index=False); print(f"  Saved: {out_csv}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nScaling law plots...")
    plot_scaling_law(df, out_dir)
    plot_scaling_by_family(df, out_dir)
    plot_scaling_by_cloze(df, out_dir)
    plot_cloze_tracking(df, out_dir)

    print("\nLayer profiles...")
    lw_df = load_layerwise(sweep_dir, args.families, meta)
    plot_layer_profiles(lw_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
