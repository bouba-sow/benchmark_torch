#!/usr/bin/env python3
"""Summarize Podcast pooled-LME-per-lag sweeps (Design B).

Reads ``<sweep_dir>/<family>/<model>/podcast_lme_by_lag.csv`` — one row per
lag, pooling every subject/electrode (electrode nested within subject; see
``neuroai/benchmarks/podcast_lme.py``). Main metrics: ``delta_loglik_test``
(primary — held-out log-likelihood gain from adding surprisal) and the
held-out ``test_delta_pearsonr`` / ``test_delta_spearmanr``.

A crashed lag (OOM-killed worker, pre-fix) leaves a row with ``error`` set
and all score columns NaN — this script drops those from every plot/summary
stat, but reports the crash rate per model since it's still new/unvalidated
territory for this codebase.

Outputs (default ``outputs/podcast/lme_summary/``):
  all_models_lme_by_lag.csv      — concatenated raw per-lag rows, all models
  peak_summary.csv               — one row per model: peak/mean metrics,
                                    convergence + crash diagnostics
  lag_profiles/<family>_<metric>.png   — metric vs lag time, one line/model
  scaling/<family>_<metric>_vs_params.png — peak metric vs model size
  scaling/all_families_<metric>_vs_params.png — cross-family scaling

Usage
-----
  python scripts/summarize/podcast_lme.py
  python scripts/summarize/podcast_lme.py \\
      --sweep_dir outputs_jeanzay/podcast/surprisal_sweep \\
      --out_dir   outputs/podcast/lme_summary
  python scripts/summarize/podcast_lme.py --families qwen3
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
from neuroai.analysis.benchmark_common import GROUP_ORDER, family_group, plot_scaling_law_by_group

DEFAULT_SWEEP_DIR = Path("outputs_jeanzay/podcast/surprisal_sweep")
DEFAULT_OUT_DIR = Path("outputs/podcast/lme_summary")

# (column, plot label) — the metrics worth a lag-profile / scaling-law plot.
LME_METRICS: list[tuple[str, str]] = [
    ("delta_loglik_test", "ΔLogLik (test, held-out)"),
    ("test_delta_pearsonr", "ΔPearson r (test)"),
    ("test_delta_spearmanr", "ΔSpearman r (test)"),
]

_FAMILY_COLORS = {
    "gpt2": "#4878d0",
    "gptneo": "#ee854a",
    "opt": "#6acc64",
    "pythia": "#d65f5f",
    "olmo2": "#956cb4",
    "qwen3": "#8c613c",
    "llama32": "#dc7ec0",
    "mistral": "#797979",
    "mamba": "#17becf",
    "mamba2": "#bcbd22",
    "cbr_rnn": "#aa5555",
    "stickbreaking": "#55aa99",
    "alibi": "#c49c94",
    "forgetting": "#7f7fbf",
    "hard_2gram": "#999999",
    "hard_3gram": "#777777",
    "hard_5gram": "#555555",
    "babylm": "#e377c2",
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
        if list(p.glob("*/podcast_lme_by_lag.csv")):
            families.append(p.name)
    return families


def _load_family(sweep_dir: Path, family: str, meta: dict[str, dict]) -> pd.DataFrame:
    fam_dir = sweep_dir / family
    parts: list[pd.DataFrame] = []
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_path = model_dir / "podcast_lme_by_lag.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
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
        "n_lags_total": n_total,
        "n_lags_crashed": n_crashed,
        "n_lags_converged": n_converged,
        "n_lags_slope_dropped": n_slope_dropped,
        "crash_rate": n_crashed / n_total if n_total else np.nan,
    }
    for metric, _ in LME_METRICS:
        if metric not in ok.columns or not np.isfinite(ok[metric]).any():
            row[f"peak_{metric}"] = np.nan
            row[f"peak_{metric}_lag_time_s"] = np.nan
            row[f"mean_{metric}"] = np.nan
            continue
        finite = ok[np.isfinite(ok[metric])]
        peak_idx = finite[metric].idxmax()
        row[f"peak_{metric}"] = float(finite.loc[peak_idx, metric])
        row[f"peak_{metric}_lag_time_s"] = float(finite.loc[peak_idx, "lag_time_s"])
        row[f"mean_{metric}"] = float(finite[metric].mean())
    return row


def plot_lag_profiles(df: pd.DataFrame, out_dir: Path, family: str) -> None:
    for metric, ylabel in LME_METRICS:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        any_plotted = False
        for model_id, grp in df.groupby("model_id"):
            g = grp[grp["error"].isna()].sort_values("lag_time_s")
            y = g[metric].astype(float)
            if not np.isfinite(y).any():
                continue
            any_plotted = True
            label = f"{g['model_label'].iloc[0]} ({g['n_params'].iloc[0] / 1e9:.2g}B)"
            ax.plot(g["lag_time_s"], y, "o-", label=label, alpha=0.85, markersize=3)
        if not any_plotted:
            plt.close(fig)
            continue
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Lag (s, word onset at 0)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Podcast pooled LME — {ylabel} vs lag — {family}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        _savefig(fig, out_dir / f"{family}_{_metric_slug(metric)}.png")


def _draw_lag_profile_ax(
    ax, df: pd.DataFrame, metric: str, ylabel: str, title: str, *, group_by_family: bool,
) -> None:
    """One line per model (or, colored by family with one legend entry per
    family when ``group_by_family=True``) — metric vs lag time. All models
    share the same lag grid (see run/surprisal/neural/podcast_fit.py), so no common-axis
    interpolation is needed here (unlike the embed scripts' per-layer
    profiles, where raw layer counts differ across model sizes)."""
    any_plotted = False
    cmap = plt.get_cmap("viridis")
    model_ids = sorted(df["model_id"].unique())
    seen_families: set[str] = set()
    for i, model_id in enumerate(model_ids):
        g = df[(df["model_id"] == model_id) & df["error"].isna()].sort_values("lag_time_s")
        y = g[metric].astype(float)
        if not np.isfinite(y).any():
            continue
        any_plotted = True
        fam = g["family"].iloc[0]
        if group_by_family:
            color = _FAMILY_COLORS.get(fam, "#4878d0")
            label = fam if fam not in seen_families else None
            seen_families.add(fam)
            ax.plot(g["lag_time_s"], y, color=color, alpha=0.6, linewidth=1.3, label=label)
        else:
            color = cmap(i / max(1, len(model_ids) - 1)) if len(model_ids) > 1 else "#4878d0"
            label = f"{g['model_label'].iloc[0]} ({g['n_params'].iloc[0] / 1e9:.2g}B)"
            ax.plot(g["lag_time_s"], y, "o-", color=color, alpha=0.9, linewidth=1.6,
                     markersize=3, label=label)

    if not any_plotted:
        ax.set_title(f"{title} (no data)")
        ax.set_axis_off()
        return
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Lag (s, word onset at 0)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")


def plot_lag_profiles_by_group(df: pd.DataFrame, out_dir: Path) -> None:
    """2x2 grid (All groups / LLM / Memory-Constrained / BabyLM) of lag
    time-course profiles, one figure per metric. Each panel is its own Axes,
    so a category's own time course isn't crushed by every other family's
    lines. Mirrors the embed scripts' by-group layer-profiles grid
    (neuroai/analysis/benchmark_common.py's plot_scaling_law_by_group pattern)."""
    df_grp = df.copy()
    df_grp["group"] = df_grp["family"].map(family_group)

    for metric, ylabel in LME_METRICS:
        if metric not in df_grp.columns:
            continue
        panels = [("All groups", df_grp)]
        panels += [(g, df_grp[df_grp["group"] == g]) for g in GROUP_ORDER]

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        for ax, (panel_title, sub) in zip(axes.flat, panels):
            n = sub["model_id"].nunique() if not sub.empty else 0
            _draw_lag_profile_ax(ax, sub, metric, ylabel, f"{panel_title} (n={n})", group_by_family=True)
        fig.suptitle(f"Podcast pooled LME — {ylabel} time course by model category", fontsize=14)
        fig.tight_layout()
        fname = out_dir / f"{_metric_slug(metric)}_by_group.png"
        _savefig(fig, fname)
        print(f"  Saved: {fname}")


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


def plot_scaling(summary: pd.DataFrame, out_dir: Path, *, family: str | None = None) -> None:
    label_suffix = family if family else "all families"
    for metric, ylabel in LME_METRICS:
        col = f"peak_{metric}"
        if col not in summary.columns:
            continue
        sub = summary[np.isfinite(summary["n_params_b"]) & np.isfinite(summary[col])].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
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
        ax.set_ylabel(f"Peak {ylabel}")
        ax.set_title(f"Peak {ylabel} vs model size — {label_suffix}")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        prefix = family if family else "all_families"
        _savefig(fig, out_dir / f"{prefix}_{_metric_slug(metric)}_vs_params.png")


def plot_scaling_by_group(summary: pd.DataFrame, out_dir: Path) -> None:
    """2x2 grids (All groups / LLM / Memory-Constrained / BabyLM) — one figure
    per metric, for both ``peak`` (max ΔLogLik across lags — the lag with the
    strongest effect) and ``mean`` (averaged across all lags, diluted by
    off-peak lags but immune to picking a lucky noise spike). Each panel is
    its own Axes, so a category's own scaling trend isn't crushed by the
    others. Mirrors summarize/ns_fmri_lme.py's plot_scaling_by_group_regions
    (neuroai/analysis/benchmark_common.py's plot_scaling_law_by_group)."""
    df_grp = summary.copy()
    df_grp["group"] = df_grp["family"].map(family_group)
    df_grp["label"] = df_grp["model_label"]

    for metric, ylabel in LME_METRICS:
        for col_prefix, kind_label in [("peak", "Peak"), ("mean", "Mean")]:
            col = f"{col_prefix}_{metric}"
            if col not in df_grp.columns:
                continue
            fname = out_dir / f"{_metric_slug(metric)}_vs_params_by_group_{col_prefix}.png"
            plot_scaling_law_by_group(
                df_grp, value_col=col, out_path=fname, ylabel=f"{kind_label} {ylabel}",
                title=f"Podcast pooled LME — {kind_label} {ylabel} scaling by model category",
            )
            print(f"  Saved: {fname}")


def summarize_family(sweep_dir: Path, family: str, out_root: Path, meta: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = out_root / family
    df = _load_family(sweep_dir, family, meta)
    if df.empty:
        print(f"[{family}] no LME lag results found")
        return df, pd.DataFrame()

    _write_csv(df, out_dir / "all_lags.csv")

    rows = []
    for model_id, grp in df.groupby("model_id"):
        rows.append(_peak_summary_row(
            model_id, grp["model_label"].iloc[0], family, grp["n_params"].iloc[0], grp,
        ))
    summary = pd.DataFrame(rows).sort_values("n_params_b", kind="mergesort").reset_index(drop=True)
    _write_csv(summary, out_dir / "peak_summary.csv")

    plot_lag_profiles(df, out_root / "lag_profiles", family)
    plot_scaling(summary, out_root / "scaling", family=family)

    n_crashed_total = int(summary["n_lags_crashed"].sum())
    if n_crashed_total:
        print(f"[{family}] {n_crashed_total} crashed lag(s) across {len(summary)} model(s) — see crash_rate in peak_summary.csv")
    print(f"[{family}] wrote LME lag summaries to {out_dir}")
    return df, summary


def run_lme_summary(sweep_dir: Path | str, out_dir: Path | str | None = None, *, families: list[str] | None = None) -> None:
    sweep_path = Path(sweep_dir).resolve()
    out_path = Path(out_dir).resolve() if out_dir else DEFAULT_OUT_DIR.resolve()

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Missing sweep dir: {sweep_path}")

    meta = _registry_meta()
    fams = families or _discover_families(sweep_path)
    if not fams:
        print(f"No podcast_lme_by_lag.csv found under {sweep_path}")
        return

    print(f"Summarizing Podcast LME families: {fams}")
    print(f"Reading from: {sweep_path}")
    print(f"Writing to:   {out_path}")

    all_lags: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    for fam in fams:
        df, summary = summarize_family(sweep_path, fam, out_path, meta)
        if not df.empty:
            all_lags.append(df)
        if not summary.empty:
            all_summaries.append(summary)

    if all_lags:
        combined_lags = pd.concat(all_lags, ignore_index=True)
        _write_csv(combined_lags, out_path / "all_models_lme_by_lag.csv")
        plot_lag_profiles_by_group(combined_lags, out_path / "lag_profiles_by_group")
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
                   help="Writable summary/plot output root (default: outputs/podcast/lme_summary)")
    p.add_argument("--families", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_lme_summary(args.sweep_dir, args.out_dir, families=args.families)
    print("Done.")


if __name__ == "__main__":
    main()
