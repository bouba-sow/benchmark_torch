#!/usr/bin/env python3
"""Summarize Fedorenko2016 checkpoint-sweep results across families.

Reads ``fedorenko2016_checkpoint_scores.csv`` and
``fedorenko2016_checkpoint_layerwise.csv`` from sweep outputs.
Primary metric: mean best-layer Spearman r.
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


SUMMARY_METRIC = "mean_best_spearmanr"

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
    "babylm": "#c5b0d5",
}

FAMILY_MARKERS = {
    "gpt2": "o",
    "gptneo": "s",
    "opt": "^",
    "llama32": "D",
    "mamba": "v",
    "mamba2": "p",
    "mistral": "h",
    "qwen3": "*",
    "pythia": "P",
    "olmo2": "X",
    "cbr_rnn": "H",
    "alibi": "<",
    "stickbreaking": ">",
    "forgetting": "d",
    "babylm": "8",
}


def _registry_meta() -> dict[str, dict]:
    reg_path = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "neuroai" / "models" / "registry.py"
    text = reg_path.read_text(encoding="utf-8")

    var_map: dict[str, str] = {}
    for m in re.finditer(r'^([A-Z][A-Z0-9_]+)\s*=\s*"([^"]+)"', text, flags=re.M):
        var_map[m.group(1)] = m.group(2)

    meta: dict[str, dict] = {}
    fam_blocks = re.findall(r'"([^"]+)":\s*\[(.*?)\],', text, flags=re.S)

    main_re = re.compile(r'_main_only\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
    spec_re = re.compile(r'LppModelSpec\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)
    var_re = re.compile(r'LppModelSpec\(\s*([A-Z][A-Z0-9_]+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([0-9.eE+-]+)', flags=re.S)

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


def load_all_models(sweep_dir: Path, families: list[str] | None, meta: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for scores_csv in sorted(sweep_dir.glob("*/*/fedorenko2016_checkpoint_scores.csv")):
        family = scores_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = scores_csv.parent.name
        model_id = model_dir.replace("_", "/", 1)

        try:
            df = pd.read_csv(scores_csv)
        except Exception:
            continue
        if df.empty:
            continue

        if "revision" in df.columns and df["revision"].nunique() > 1:
            rev = _pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == rev]
        else:
            rev = str(df.get("revision", pd.Series(["main"])).iloc[0])

        m = meta.get(model_id, meta.get(model_dir, {}))
        rows.append(
            {
                "family": family,
                "model_id": model_id,
                "label": m.get("label", model_id.split("/")[-1]),
                "n_params": m.get("n_params", np.nan),
                "n_layers": m.get("n_layers", np.nan),
                "revision": rev,
                SUMMARY_METRIC: float(df[SUMMARY_METRIC].iloc[0]) if SUMMARY_METRIC in df.columns else np.nan,
                "mean_best_pearsonr": float(df["mean_best_pearsonr"].iloc[0]) if "mean_best_pearsonr" in df.columns else np.nan,
                "perplexity": float(df["perplexity"].iloc[0]) if "perplexity" in df.columns else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["family", "n_params"]).reset_index(drop=True)


def load_layerwise(sweep_dir: Path, families: list[str] | None, meta: dict[str, dict]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for lw_csv in sorted(sweep_dir.glob("*/*/fedorenko2016_checkpoint_layerwise.csv")):
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
        df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
        df = df.dropna(subset=["layer"])
        if df.empty:
            continue
        n_layers_total = int(df["layer"].max()) + 1
        df["layer_rel"] = df["layer"] / max(1, n_layers_total - 1)
        df["family"] = family
        df["model_id"] = model_id
        df["label"] = m.get("label", model_id.split("/")[-1])
        df["n_params"] = m.get("n_params", np.nan)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


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


def _scaling_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    title: str,
    invert_x: bool = False,
) -> None:
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        ax.set_title(f"{title} (no data)")
        return

    for family, grp in plot_df.groupby("family"):
        color = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        grp_s = grp.sort_values(x_col)
        ax.scatter(grp_s[x_col], grp_s[y_col],
                   label=family, color=color, marker=marker, s=70, zorder=3, alpha=0.9)
        if len(grp_s) > 1:
            ax.plot(grp_s[x_col], grp_s[y_col], color=color, lw=0.8, alpha=0.4)
        for _, row in grp_s.iterrows():
            ax.annotate(
                row["label"],
                (row[x_col], row[y_col]),
                fontsize=5, ha="left", va="bottom",
                xytext=(2, 2), textcoords="offset points",
            )

    a, b, r2 = _fit_log_linear(plot_df[x_col].to_numpy(float), plot_df[y_col].to_numpy(float))
    if np.isfinite(a):
        x_fit = np.logspace(
            np.log10(plot_df[x_col].min()), np.log10(plot_df[x_col].max()), 200
        )
        ax.plot(x_fit, a * np.log10(x_fit) + b, "k--", lw=1.2, alpha=0.5,
                label=f"fit R²={r2:.3f}")

    ax.set_xscale("log")
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Mean best-layer Spearman ρ", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.25)


def plot_scaling_law(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.dropna(subset=["n_params", SUMMARY_METRIC])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    _scaling_ax(ax, plot_df, "n_params", SUMMARY_METRIC,
                xlabel="Number of parameters",
                title="Fedorenko2016 — Scaling law")
    fig.tight_layout()
    fname = out_dir / "scaling_law.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_perplexity_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.dropna(subset=["perplexity", SUMMARY_METRIC])
    if plot_df.empty:
        print("  [skip] no perplexity data")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    _scaling_ax(ax, plot_df, "perplexity", SUMMARY_METRIC,
                xlabel="Perplexity on Fedorenko2016  ← better LM",
                title="Fedorenko2016 — Brain score vs perplexity",
                invert_x=True)
    fig.tight_layout()
    fname = out_dir / "scaling_by_perplexity.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_layer_profiles(lw_df: pd.DataFrame, out_dir: Path) -> None:
    if lw_df.empty:
        print("  [skip] no layerwise data found")
        return
    families = sorted(lw_df["family"].unique())
    ncols = min(3, len(families))
    nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4),
                             squeeze=False)
    for i, fam in enumerate(families):
        ax = axes[i // ncols][i % ncols]
        fam_df = lw_df[lw_df["family"] == fam]
        model_ids = (
            fam_df.drop_duplicates("model_id")
            .sort_values("n_params")["model_id"]
            .tolist()
        )
        for mi, mid in enumerate(model_ids):
            grp = fam_df[fam_df["model_id"] == mid].sort_values("layer_rel")
            linestyle = ["-", "--", ":", "-."][mi % 4]
            label = grp["label"].iloc[0]
            color = FAMILY_COLORS.get(fam, "gray")
            ax.plot(grp["layer_rel"], grp["mean_spearmanr"],
                    lw=1.5, linestyle=linestyle, alpha=0.85,
                    color=color, label=label)
            # Annotate the last point of each curve
            ax.annotate(
                label,
                (float(grp["layer_rel"].iloc[-1]), float(grp["mean_spearmanr"].iloc[-1])),
                fontsize=5, ha="left", va="center",
                xytext=(3, 0), textcoords="offset points",
            )
        ax.set_title(fam, fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative layer depth", fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel("Mean Spearman ρ", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
    for j in range(len(families), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Fedorenko2016 — Layer profiles by family", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "layer_profiles.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep_dir", type=Path, default=Path("outputs_jeanzay/fedorenko2016/embedding_sweep"))
    p.add_argument("--out_dir", type=Path, default=Path("outputs/fedorenko2016/summary"))
    p.add_argument("--families", nargs="+", default=None)
    args = p.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not sweep_dir.is_dir():
        sys.exit(f"sweep_dir not found: {sweep_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _registry_meta()
    df = load_all_models(sweep_dir, args.families, meta)
    lw_df = load_layerwise(sweep_dir, args.families, meta)
    if df.empty:
        sys.exit("No fedorenko2016 results found.")

    out_csv = out_dir / "all_models_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")
    print()
    plot_scaling_law(df, out_dir)
    plot_perplexity_scaling(df, out_dir)
    plot_layer_profiles(lw_df, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()

