#!/usr/bin/env python3
"""Summarize Nieuwland N400 pooled-LME-per-layer sweeps.

Reads ``<sweep_dir>/<family>/<model>/nieuwland_lme_by_layer.csv`` — one row
per LM layer, pooling every subject/electrode/lab (electrode CROSSED, not
nested — Nieuwland's scalp electrodes are the same physical 10-20 locations
across subjects; see ``neuroai/regressors/n400_pooled_lme.py``). Unlike
the earlier N400 datasets, Nieuwland is multi-site (9 UK labs), so the LME formula
also includes a crossed ``(1 | lab)`` random intercept
(``n400_lme_formulas(include_lab=True)``, ``neuroai/regressors/lme.py``).
Main metrics: ``delta_loglik_test`` (primary — held-out log-likelihood gain
from adding per-layer logit-lens surprisal) and the held-out
``test_delta_pearsonr`` / ``test_delta_spearmanr``.

A crashed layer (OOM-killed worker, unconverged fit) leaves a row with
``error`` set and all score columns NaN — this script drops those from every
plot/summary stat, but reports the crash rate per model.

Outputs (default ``outputs/nieuwland/lme_summary/``):
  all_models_lme_by_layer.csv          — concatenated raw per-layer rows, all models
  peak_summary.csv                     — one row per model: peak/mean metrics,
                                          convergence + crash diagnostics
  layer_profiles/<family>_<metric>.png       — metric vs layer, one line/model
  scaling/<family>_<metric>_vs_params.png    — peak metric vs model size
  scaling/all_families_<metric>_vs_params.png — cross-family scaling

Usage
-----
  python scripts/summarize/nieuwland_lme.py
  python scripts/summarize/nieuwland_lme.py \\
      --sweep_dir outputs_jeanzay/nieuwland/surprisal_sweep \\
      --out_dir   outputs/nieuwland/lme_summary
  python scripts/summarize/nieuwland_lme.py --families gpt2 pythia
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuroai.models.registry import LPP_FAMILIES, model_spec_to_cfg
from neuroai.analysis.benchmark_common import family_group, plot_scaling_law_by_group, read_lme_by_layer_csv

DEFAULT_SWEEP_DIR = Path("outputs_jeanzay/nieuwland/surprisal_sweep")
DEFAULT_OUT_DIR = Path("outputs/nieuwland/lme_summary")

# (column, plot label) — the metrics worth a layer-profile / scaling-law plot.
LME_METRICS: list[tuple[str, str]] = [
    ("delta_loglik_test", "ΔLogLik (test, held-out)"),
    ("test_delta_pearsonr", "ΔPearson r (test)"),
    ("test_delta_spearmanr", "ΔSpearman r (test)"),
]

# Same palette as summarize/nieuwland.py (ridge/embeddings mode) so the two
# reports stay visually consistent for the same family across modes.
_FAMILY_COLORS = {
    "gpt2":          "#1f77b4",
    "gptneo":        "#ff7f0e",
    "opt":           "#2ca02c",
    "llama32":       "#d62728",
    "mamba":         "#9467bd",
    "mamba2":        "#8c564b",
    "mistral":       "#e377c2",
    "qwen3":         "#7f7f7f",
    "pythia":        "#bcbd22",
    "olmo2":         "#17becf",
    "babylm":        "#1a9850",
    "cbr_rnn":       "#e41a1c",
    "stickbreaking": "#a65628",
    "alibi":         "#f781bf",
    "forgetting":    "#999999",
}


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "_")


def _registry_meta() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for fam, specs in LPP_FAMILIES.items():
        for spec in specs:
            cfg = model_spec_to_cfg(spec, fam)
            row = {
                "model_id": spec.model_id,
                "model_label": spec.label,
                "n_params": float(spec.n_params),
                "family": fam,
            }
            meta[spec.model_id] = row
            meta[_safe_model_dir(spec.model_id)] = row
    return meta


def _metric_slug(metric: str) -> str:
    return metric.replace(".", "_")


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _discover_families(sweep_dir: Path) -> list[str]:
    families: list[str] = []
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_dir() or p.name == "summary":
            continue
        if list(p.glob("*/nieuwland_lme_by_layer.csv")):
            families.append(p.name)
    return families


def _load_family(sweep_dir: Path, family: str, meta: dict[str, dict]) -> pd.DataFrame:
    fam_dir = sweep_dir / family
    parts: list[pd.DataFrame] = []
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_path = model_dir / "nieuwland_lme_by_layer.csv"
        if not csv_path.is_file():
            continue
        df = read_lme_by_layer_csv(csv_path)
        if df.empty:
            continue
        model_id = model_dir.name.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir.name, {}))
        df.insert(0, "model_id", m.get("model_id", model_id))
        df.insert(1, "family", family)
        df["model_label"] = m.get("model_label", model_id)
        df["n_params"] = m.get("n_params", np.nan)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _peak_summary_row(model_id: str, model_label: str, family: str, n_params: float, df: pd.DataFrame) -> dict:
    n_total = len(df)
    crashed = df["error"].notna()
    n_crashed = int(crashed.sum())
    ok = df.loc[~crashed].copy()
    n_converged = int(ok["converged"].fillna(False).astype(bool).sum()) if "converged" in ok else 0
    n_slope_dropped = int(ok["slope_dropped"].fillna(False).astype(bool).sum()) if "slope_dropped" in ok else 0

    row: dict = {
        "model_id": model_id,
        "model_label": model_label,
        "family": family,
        "n_params": n_params,
        "n_params_b": n_params / 1e9 if np.isfinite(n_params) else np.nan,
        "n_layers_total": n_total,
        "n_layers_crashed": n_crashed,
        "n_layers_converged": n_converged,
        "n_layers_slope_dropped": n_slope_dropped,
        "crash_rate": n_crashed / n_total if n_total else np.nan,
    }
    last_layer_idx = int(ok["layer"].max()) if "layer" in ok and not ok.empty else None
    for metric, _ in LME_METRICS:
        if metric not in ok.columns or not np.isfinite(ok[metric]).any():
            row[f"peak_{metric}"] = np.nan
            row[f"peak_{metric}_layer"] = np.nan
            row[f"mean_{metric}"] = np.nan
            row[f"last_{metric}"] = np.nan
            continue
        finite = ok[np.isfinite(ok[metric])]
        peak_idx = finite[metric].idxmax()
        row[f"peak_{metric}"] = float(finite.loc[peak_idx, metric])
        row[f"peak_{metric}_layer"] = float(finite.loc[peak_idx, "layer"])
        row[f"mean_{metric}"] = float(finite[metric].mean())
        last_row = ok.loc[ok["layer"] == last_layer_idx] if last_layer_idx is not None else pd.DataFrame()
        last_val = last_row[metric].iloc[0] if not last_row.empty else np.nan
        row[f"last_{metric}"] = float(last_val) if np.isfinite(last_val) else np.nan
    return row


def plot_layer_profiles(df: pd.DataFrame, out_dir: Path, family: str) -> None:
    for metric, ylabel in LME_METRICS:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        any_plotted = False
        for model_id, grp in df.groupby("model_id"):
            g = grp[grp["error"].isna()].sort_values("layer")
            y = g[metric].astype(float)
            if not np.isfinite(y).any():
                continue
            any_plotted = True
            label = f"{g['model_label'].iloc[0]} ({g['n_params'].iloc[0] / 1e9:.2g}B)"
            ax.plot(g["layer"], y, "o-", label=label, alpha=0.85, markersize=3)
        if not any_plotted:
            plt.close(fig)
            continue
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("LM layer")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Nieuwland pooled LME — {ylabel} vs layer — {family}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        _savefig(fig, out_dir / f"{family}_{_metric_slug(metric)}.png")


def _fit_log_linear(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return {"ok": False, "r2": np.nan, "n": int(x.size)}
    coeffs = np.polyfit(x, y, deg=1)
    y_hat = coeffs[0] * x + coeffs[1]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"ok": True, "r2": float(r2), "slope": float(coeffs[0]), "intercept": float(coeffs[1]), "n": int(x.size)}


def _scaling_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    *,
    family: str | None,
) -> None:
    sub = summary[np.isfinite(summary["n_params_b"]) & np.isfinite(summary[col])].copy()
    if sub.empty:
        ax.set_title(f"{title} (no data)")
        return
    if family is None:
        for fam, grp in sub.groupby("family"):
            color = _FAMILY_COLORS.get(str(fam), "#555555")
            g = grp.sort_values("n_params_b")
            ax.scatter(g["n_params_b"], g[col], s=60, alpha=0.9, color=color, label=str(fam))
            if len(g) > 1:
                ax.plot(g["n_params_b"], g[col], color=color, alpha=0.4, linewidth=1)
    else:
        color = _FAMILY_COLORS.get(family, "#4878d0")
        ax.scatter(sub["n_params_b"], sub[col], s=60, alpha=0.9, color=color)
        for _, row in sub.iterrows():
            ax.annotate(row["model_label"], (row["n_params_b"], row[col]),
                        textcoords="offset points", xytext=(4, 2), fontsize=7)
    fit = _fit_log_linear(np.log10(sub["n_params_b"]).to_numpy(float), sub[col].to_numpy(float))
    if fit.get("ok"):
        x_line = np.linspace(np.log10(sub["n_params_b"]).min(), np.log10(sub["n_params_b"]).max(), 100)
        ax.plot(10 ** x_line, fit["slope"] * x_line + fit["intercept"], "--",
                 color="#222222", linewidth=1.2, label=f"fit R²={fit['r2']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, loc="best")


def plot_scaling(summary: pd.DataFrame, out_dir: Path, *, family: str | None = None) -> None:
    label_suffix = family if family else "all families"
    for metric, ylabel in LME_METRICS:
        peak_col = f"peak_{metric}"
        last_col = f"last_{metric}"
        if peak_col not in summary.columns and last_col not in summary.columns:
            continue
        fig, (ax_best, ax_last) = plt.subplots(1, 2, figsize=(15, 5.5))
        _scaling_panel(
            ax_best, summary, peak_col, f"Peak {ylabel}",
            f"Best layer — {ylabel} vs model size — {label_suffix}", family=family,
        )
        _scaling_panel(
            ax_last, summary, last_col, f"Last-layer {ylabel}",
            f"Last layer — {ylabel} vs model size — {label_suffix}", family=family,
        )
        fig.tight_layout()
        prefix = family if family else "all_families"
        _savefig(fig, out_dir / f"{prefix}_{_metric_slug(metric)}_vs_params.png")


def plot_scaling_by_group(summary: pd.DataFrame, out_dir: Path) -> None:
    """2x2 grids (All groups / LLM / Memory-Constrained / BabyLM) — one figure
    per metric, for peak (best layer) and last layer separately. Each panel is
    its own Axes, so a category's own scaling trend isn't crushed by the
    others. Mirrors summarize/ns_fmri_lme.py's plot_scaling_by_group_regions
    (neuroai/analysis/benchmark_common.py's plot_scaling_law_by_group)."""
    df_grp = summary.copy()
    df_grp["group"] = df_grp["family"].map(family_group)
    df_grp["label"] = df_grp["model_label"]

    for metric, ylabel in LME_METRICS:
        for kind, col_prefix, title_extra in [
            ("best_layer", "peak", " (best layer)"),
            ("last_layer", "last", " (last layer)"),
        ]:
            col = f"{col_prefix}_{metric}"
            if col not in df_grp.columns:
                continue
            fname = out_dir / f"{_metric_slug(metric)}_vs_params_by_group_{kind}.png"
            plot_scaling_law_by_group(
                df_grp, value_col=col, out_path=fname, ylabel=ylabel,
                title=f"Nieuwland pooled LME — {ylabel} scaling by model category{title_extra}",
            )
            print(f"  Saved: {fname}")


def summarize_family(sweep_dir: Path, family: str, out_root: Path, meta: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = out_root / family
    df = _load_family(sweep_dir, family, meta)
    if df.empty:
        print(f"[{family}] no LME layer results found")
        return df, pd.DataFrame()

    _write_csv(df, out_dir / "all_layers.csv")

    rows = []
    for model_id, grp in df.groupby("model_id"):
        rows.append(_peak_summary_row(
            model_id, grp["model_label"].iloc[0], family, grp["n_params"].iloc[0], grp,
        ))
    summary = pd.DataFrame(rows).sort_values("n_params_b", kind="mergesort").reset_index(drop=True)
    _write_csv(summary, out_dir / "peak_summary.csv")

    plot_layer_profiles(df, out_root / "layer_profiles", family)
    plot_scaling(summary, out_root / "scaling", family=family)

    n_crashed_total = int(summary["n_layers_crashed"].sum())
    if n_crashed_total:
        print(f"[{family}] {n_crashed_total} crashed layer(s) across {len(summary)} model(s) — see crash_rate in peak_summary.csv")
    print(f"[{family}] wrote LME layer summaries to {out_dir}")
    return df, summary


def run_lme_summary(sweep_dir: Path | str, out_dir: Path | str | None = None, *, families: list[str] | None = None) -> None:
    sweep_path = Path(sweep_dir).resolve()
    out_path = Path(out_dir).resolve() if out_dir else DEFAULT_OUT_DIR.resolve()

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Missing sweep dir: {sweep_path}")

    meta = _registry_meta()
    fams = families or _discover_families(sweep_path)
    if not fams:
        print(f"No nieuwland_lme_by_layer.csv found under {sweep_path}")
        return

    print(f"Summarizing Nieuwland LME families: {fams}")
    print(f"Reading from: {sweep_path}")
    print(f"Writing to:   {out_path}")

    all_layers: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    for fam in fams:
        df, summary = summarize_family(sweep_path, fam, out_path, meta)
        if not df.empty:
            all_layers.append(df)
        if not summary.empty:
            all_summaries.append(summary)

    if all_layers:
        _write_csv(pd.concat(all_layers, ignore_index=True), out_path / "all_models_lme_by_layer.csv")
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        _write_csv(combined_summary, out_path / "peak_summary.csv")
        plot_scaling(combined_summary, out_path / "scaling", family=None)
        plot_scaling_by_group(combined_summary, out_path / "scaling_by_group")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep_dir", type=Path, default=DEFAULT_SWEEP_DIR,
                   help="Read-only sweep results (e.g. synced outputs_jeanzay/)")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Writable summary/plot output root (default: outputs/nieuwland/lme_summary)")
    p.add_argument("--families", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_lme_summary(args.sweep_dir, args.out_dir, families=args.families)
    print("Done.")


if __name__ == "__main__":
    main()
