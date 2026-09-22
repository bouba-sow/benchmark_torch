#!/usr/bin/env python3
"""Summarize and plot Natural Stories LME checkpoint sweeps.

Reads ``<sweep_dir>/<family>/<model>/spr_lme_checkpoint_scores.csv``,
deduplicates failed reruns, and writes per-family + cross-family plots:

- training curves (metric vs tokens seen)
- scaling vs parameter count (final checkpoint)
- scaling vs inverse perplexity (final checkpoint)

Usage::

    # Read synced Jean Zay results (read-only), write plots locally:
    python scripts/summarize/naturalstories_lme_families.py \\
        --sweep_dir outputs_jeanzay/naturalstories/surprisal_sweep \\
        --out_dir outputs/naturalstories/surprisal_sweep

    python scripts/summarize/naturalstories_lme_families.py \\
        --sweep_dir outputs_jeanzay/naturalstories/surprisal_sweep \\
        --perplexity_dir outputs_jeanzay/naturalstories/perplexity
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from neuroai.models.registry import LPP_FAMILIES, model_spec_to_cfg

# Primary LME metrics (column, plot label).
LME_METRICS: list[tuple[str, str]] = [
    ("beta", "Surprisal β (train)"),
    ("prev_beta", "Prev. surprisal β (train)"),
    ("delta_loglik_test", "ΔLogLik (test)"),
    ("delta_loglik_train", "ΔLogLik (train)"),
    ("test_spearmanr_full_model", "Spearman r (test, full LME)"),
    ("test_pearsonr_full_model", "Pearson r (test, full LME)"),
    ("test_spearmanr_baseline_lme", "Spearman r (test, baseline LME)"),
    ("test_delta_spearmanr", "ΔSpearman r (test)"),
    ("test_delta_pearsonr", "ΔPearson r (test)"),
]

# Old column names → canonical (pre-rename CSVs).
_COLUMN_ALIASES: dict[str, str] = {
    "surprisal_beta": "beta",
    "surprisal_prev_beta": "prev_beta",
    "surprisal_t": "t",
    "surprisal_p": "p",
    "surprisal_delta_loglik": "delta_loglik",
    "surprisal_delta_loglik_train": "delta_loglik_train",
    "surprisal_delta_loglik_test": "delta_loglik_test",
    "surprisal_lrt_p": "lrt_p",
    "surprisal_converged": "converged",
    "surprisal_test_pearsonr_full_model": "test_pearsonr_full_model",
    "surprisal_test_spearmanr_full_model": "test_spearmanr_full_model",
    "surprisal_test_pearsonr_baseline_lme": "test_pearsonr_baseline_lme",
    "surprisal_test_spearmanr_baseline_lme": "test_spearmanr_baseline_lme",
    "surprisal_test_delta_pearsonr": "test_delta_pearsonr",
    "surprisal_test_delta_spearmanr": "test_delta_spearmanr",
    "surprisal_pearsonr": "test_pearsonr_full_model",
    "surprisal_spearmanr": "test_spearmanr_full_model",
    "surprisal_baseline_pearsonr": "test_pearsonr_baseline_lme",
    "surprisal_baseline_spearmanr": "test_spearmanr_baseline_lme",
    "surprisal_delta_pearsonr": "test_delta_pearsonr",
    "surprisal_delta_spearmanr": "test_delta_spearmanr",
}

DEFAULT_SWEEP_DIR = Path("outputs_jeanzay/naturalstories/surprisal_sweep")
DEFAULT_SUMMARY_OUT_DIR = Path("outputs/naturalstories/surprisal_sweep")
DEFAULT_PPL_DIR = Path("outputs_jeanzay/naturalstories/perplexity")

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
}

# Distinct per-point colors for panels of unrelated individual models (e.g.
# BabyLM's three different architectures) rather than a size-scan family.
_FAMILY_COLORS_CYCLE = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4", "#8c613c"]


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
                "n_layers": int(spec.n_layers),
                "family": fam,
                "registry_family": fam,
            }
            meta[spec.model_id] = row
            meta[_safe_model_dir(spec.model_id)] = row
    return meta


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {k: v for k, v in _COLUMN_ALIASES.items() if k in out.columns and v not in out.columns}
    if rename:
        out = out.rename(columns=rename)
    if "converged" in out.columns:
        out["converged"] = out["converged"].fillna(False).astype(bool)
    else:
        out["converged"] = True
    return out


def _dedupe_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the best row per (model_id, revision): converged wins over failed."""
    if df.empty:
        return df
    tmp = df.copy()
    tmp["_conv_rank"] = tmp["converged"].astype(int)
    tmp = tmp.sort_values(["model_id", "revision", "_conv_rank"])
    out = tmp.groupby(["model_id", "revision"], as_index=False).tail(1)
    return out.drop(columns=["_conv_rank"], errors="ignore")


def _attach_metadata(df: pd.DataFrame, meta: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    labels, n_params, families = [], [], []
    for mid in out["model_id"]:
        m = meta.get(mid, meta.get(_safe_model_dir(mid), {}))
        labels.append(m.get("model_label", mid))
        n_params.append(m.get("n_params", np.nan))
        families.append(m.get("family", out.get("family", pd.Series([None])).iloc[0] if "family" in out.columns else None))
    out["model_label"] = labels
    out["n_params"] = n_params
    if "family" not in out.columns or out["family"].isna().all():
        out["family"] = families
    out["n_params_b"] = out["n_params"] / 1e9
    out["log10_n_params"] = np.log10(out["n_params"].where(out["n_params"] > 0))
    out["tokens_seen_b"] = out["tokens_seen"].astype(float) / 1e9
    return out


def _load_perplexity_map(perplexity_dir: Path) -> dict[str, float]:
    if not perplexity_dir.is_dir():
        return {}
    out: dict[str, float] = {}
    for summary_name in (
        "naturalstories_perplexity_summary.csv",
        "lpp_perplexity_summary.csv",
    ):
        summary_csv = perplexity_dir / summary_name
        if not summary_csv.is_file():
            continue
        summary = pd.read_csv(summary_csv)
        if {"model_id", "perplexity"}.issubset(summary.columns):
            for _, row in summary.iterrows():
                ppl = row["perplexity"]
                if pd.notna(ppl):
                    out[str(row["model_id"])] = float(ppl)
    for p in sorted(perplexity_dir.rglob("perplexity.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        mid = data.get("model_id")
        ppl = data.get("perplexity")
        if mid and ppl is not None and mid not in out:
            out[str(mid)] = float(ppl)
    return out


def _attach_perplexity(df: pd.DataFrame, ppl_map: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    if not ppl_map:
        return out
    out["perplexity"] = out["model_id"].map(ppl_map)
    mask = out["perplexity"].notna() & out["perplexity"].gt(0)
    out["inv_perplexity"] = np.where(mask, 1.0 / out["perplexity"], np.nan)
    out["log2_perplexity"] = np.where(mask, np.log2(out["perplexity"]), np.nan)
    return out


def _discover_families(sweep_dir: Path) -> list[str]:
    families: list[str] = []
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_dir() or p.name in {"summary", "logs"}:
            continue
        if list(p.glob("*/spr_lme_checkpoint_scores.csv")):
            families.append(p.name)
    return families


def _collect_family_scores(sweep_dir: Path, family: str, meta: dict[str, dict]) -> pd.DataFrame:
    fam_dir = sweep_dir / family
    parts: list[pd.DataFrame] = []
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "summary":
            continue
        csv_path = model_dir / "spr_lme_checkpoint_scores.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        model_id = model_dir.name.replace("_", "/", 1)
        df = _normalize_columns(df)
        df.insert(0, "model_id", model_id)
        df.insert(1, "family", family)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined = _dedupe_scores(combined)
    combined = _attach_metadata(combined, meta)
    sort_cols = [c for c in ("n_params_b", "model_label", "tokens_seen", "step") if c in combined.columns]
    return combined.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _final_checkpoint_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: latest converged checkpoint by tokens_seen."""
    if df.empty:
        return df
    use = df[df["converged"]].copy() if df["converged"].any() else df.copy()
    idx = use.groupby("model_id")["tokens_seen"].idxmax()
    return use.loc[idx].sort_values("n_params_b", kind="mergesort").reset_index(drop=True)


def _fit_scaling(x: np.ndarray, y: np.ndarray) -> dict:
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


def _savefig(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _metric_slug(metric: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", metric).strip("_").lower()
    return re.sub(r"[^a-zA-Z0-9]+", "_", metric).strip("_").lower()


def _size_label(n_params_b: float) -> str:
    if not np.isfinite(n_params_b) or n_params_b <= 0:
        return ""
    return f"{n_params_b:.1f}B" if n_params_b >= 1 else f"{n_params_b * 1000:.0f}M"


def _plot_training_curves(df: pd.DataFrame, out_path: Path, family: str, metric: str, ylabel: str) -> None:
    if metric not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_id, grp in df.groupby("model_id"):
        g = grp.sort_values("tokens_seen")
        y = g[metric].astype(float)
        if not np.isfinite(y).any():
            continue
        label = g["model_label"].iloc[0]
        ax.plot(g["tokens_seen_b"], y, "o-", label=label, alpha=0.9)
    ax.set_xlabel("Tokens seen (billions)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Natural Stories LME — {ylabel} — {family}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    _savefig(fig, out_path)


def _plot_scaling_params(
    final: pd.DataFrame, out_path: Path, family: str, metric: str, ylabel: str
) -> dict:
    sub = final[np.isfinite(final["n_params_b"]) & np.isfinite(final[metric])].copy()
    fit = _fit_scaling(sub["log10_n_params"].to_numpy(float), sub[metric].to_numpy(float))
    fit.update({"family": family, "metric": metric, "x": "n_params"})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sub["n_params_b"], sub[metric], s=70, alpha=0.9, color=_FAMILY_COLORS.get(family, "#4878d0"))
    for _, row in sub.iterrows():
        ax.annotate(
            row["model_label"],
            (row["n_params_b"], row[metric]),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=7,
        )
    if fit.get("ok"):
        x_line = np.linspace(sub["log10_n_params"].min(), sub["log10_n_params"].max(), 100)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(10 ** x_line / 1e9, y_line, "-", color="#444444", linewidth=1.2, label=f"fit R²={fit['r2']:.3f}")
        ax.legend(loc="best", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs model size — {family} (final checkpoint)")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    _savefig(fig, out_path)
    return fit


def _plot_scaling_perplexity(
    final: pd.DataFrame, out_path: Path, family: str, metric: str, ylabel: str
) -> dict:
    x_col = "inv_perplexity"
    sub = final[np.isfinite(final[x_col]) & np.isfinite(final[metric])].copy()
    fit = _fit_scaling(np.log2(sub[x_col]).to_numpy(float), sub[metric].to_numpy(float))
    fit.update({"family": family, "metric": metric, "x": "inv_perplexity"})

    if sub.empty:
        return fit

    fig, ax = plt.subplots(figsize=(8, 5))
    color = _FAMILY_COLORS.get(family, "#4878d0")
    ax.scatter(sub[x_col], sub[metric], s=70, alpha=0.9, color=color)
    if len(sub) > 1:
        ax.plot(sub.sort_values(x_col)[x_col], sub.sort_values(x_col)[metric], color=color, alpha=0.5)
    for _, row in sub.iterrows():
        ann = _size_label(row.get("n_params_b", np.nan)) or row["model_label"]
        ax.annotate(ann, (row[x_col], row[metric]), textcoords="offset points", xytext=(4, 2), fontsize=7)

    if fit.get("ok") and len(sub) >= 2:
        x_log = np.log2(sub[x_col])
        x_line = np.linspace(x_log.min(), x_log.max(), 100)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(2 ** x_line, y_line, "--", color="#333333", linewidth=1.2, label=f"fit R²={fit['r2']:.3f}")
        ax.legend(loc="best", fontsize=8)

    ax.set_xscale("log", base=2)
    x_vals = sub[x_col].dropna()
    if not x_vals.empty:
        pad = 0.15
        log2_min = math.log2(x_vals.min())
        log2_max = math.log2(x_vals.max())
        ax.set_xlim(2 ** (log2_min - pad), 2 ** (log2_max + pad))
        ticks = [2 ** k for k in range(math.ceil(log2_min - pad), math.floor(log2_max + pad) + 1)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"2^{int(round(math.log2(v)))}"))

    ax.set_xlabel("Inverse perplexity (log₂ scale, higher = better LM)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs inverse perplexity — {family} (final checkpoint)")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    _savefig(fig, out_path)
    return fit


# Family groupings for the by-group scaling panels (see registry.py's own
# comments: "Memory-constrained Lanni-ni models" for stickbreaking/alibi/
# forgetting, plus CBR-RNN which is also memory-constrained; "BabyLM 2025
# shared-task models" for babylm; everything else is a standard pretrained LLM
# family).
_MEMORY_CONSTRAINED_FAMILIES = {"stickbreaking", "alibi", "forgetting", "cbr_rnn"}
_BABYLM_FAMILIES = {"babylm"}


def _family_group_subsets(sub: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split a final-checkpoint frame into (panel_title, subset) for the 2x2 grid."""
    mem = sub[sub["family"].isin(_MEMORY_CONSTRAINED_FAMILIES)]
    baby = sub[sub["family"].isin(_BABYLM_FAMILIES)]
    llm = sub[~sub["family"].isin(_MEMORY_CONSTRAINED_FAMILIES | _BABYLM_FAMILIES)]
    return [
        ("All families", sub),
        ("Memory-constrained", mem),
        ("BabyLM", baby),
        ("LLMs", llm),
    ]


def _plot_scaling_panel(
    ax,
    sub: pd.DataFrame,
    metric: str,
    ylabel: str,
    panel_title: str,
    *,
    x_col: str,
    xlabel: str,
    log_x: bool,
    log_base: int | None,
    connect_lines: bool = True,
    label_individual_points: bool = False,
) -> dict:
    """Draw one scaling-law panel onto `ax` (one quadrant of the 2x2 grid).

    ``connect_lines=False`` / ``label_individual_points=True`` is for panels
    like BabyLM whose "family" is really a handful of unrelated architectures
    (not different sizes of the same model) — a connecting line there would
    misleadingly imply a size-scaling trend, so instead each point is scatter-
    plotted on its own and labeled with its model name.
    """
    sub = sub[np.isfinite(sub[metric])].copy()
    if x_col == "n_params_b":
        sub = sub[np.isfinite(sub["n_params_b"])]
        x_fit = sub["log10_n_params"].to_numpy(float)
    else:
        sub = sub[np.isfinite(sub[x_col])]
        x_fit = np.log2(sub[x_col]).to_numpy(float) if log_base == 2 else sub[x_col].to_numpy(float)
    y = sub[metric].to_numpy(float)
    fit = _fit_scaling(x_fit, y)

    if label_individual_points:
        for i, (_, row) in enumerate(sub.sort_values(x_col).iterrows()):
            color = _FAMILY_COLORS_CYCLE[i % len(_FAMILY_COLORS_CYCLE)]
            ax.scatter([row[x_col]], [row[metric]], s=45, alpha=0.9, color=color)
            label = str(row.get("model_label", row.get("model_id", "")))
            ax.annotate(
                label, (row[x_col], row[metric]),
                textcoords="offset points", xytext=(4, 3), fontsize=6,
            )
    else:
        for fam, grp in sub.groupby("family"):
            g = grp.sort_values(x_col)
            color = _FAMILY_COLORS.get(str(fam), "#555555")
            ax.scatter(g[x_col], g[metric], s=45, alpha=0.9, label=str(fam), color=color)
            if connect_lines and len(g) > 1:
                ax.plot(g[x_col], g[metric], linewidth=0.8, alpha=0.5, color=color)

    if fit.get("ok") and len(sub) >= 2:
        if x_col == "n_params_b":
            x_line = np.linspace(sub["log10_n_params"].min(), sub["log10_n_params"].max(), 100)
            ax.plot(10 ** x_line / 1e9, fit["slope"] * x_line + fit["intercept"], "--", color="#222222", linewidth=1.2, label=f"fit R²={fit['r2']:.3f}")
        elif log_base == 2:
            x_log = np.log2(sub[x_col])
            x_line = np.linspace(x_log.min(), x_log.max(), 100)
            ax.plot(2 ** x_line, fit["slope"] * x_line + fit["intercept"], "--", color="#222222", linewidth=1.2, label=f"fit R²={fit['r2']:.3f}")

    if log_x:
        ax.set_xscale("log", base=(log_base or 10))
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    n_fams = sub["family"].nunique()
    r2_str = f"R²={fit['r2']:.3f}" if fit.get("ok") else "no fit"
    ax.set_title(f"{panel_title} (n={len(sub)}, {n_fams} fam., {r2_str})", fontsize=10)
    ax.grid(True, which="both", alpha=0.2)
    if not sub.empty:
        ax.legend(fontsize=6, ncol=2, loc="best")
    return fit


def _plot_global_scaling(
    final_all: pd.DataFrame,
    out_path: Path,
    metric: str,
    ylabel: str,
    *,
    x_col: str = "n_params_b",
    xlabel: str = "Parameters (billions)",
    log_x: bool = True,
    log_base: int | None = None,
) -> dict:
    """2x2 grid: All families / Memory-constrained / BabyLM / LLMs."""
    sub_all = final_all[np.isfinite(final_all[metric])].copy()
    groups = _family_group_subsets(sub_all)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fits: dict[str, dict] = {}
    for ax, (panel_title, panel_df) in zip(axes.flat, groups):
        if panel_df.empty:
            ax.set_visible(False)
            fits[panel_title] = {"ok": False}
            continue
        fits[panel_title] = _plot_scaling_panel(
            ax, panel_df, metric, ylabel, panel_title,
            x_col=x_col, xlabel=xlabel, log_x=log_x, log_base=log_base,
            connect_lines=(panel_title != "BabyLM"),
            label_individual_points=(panel_title == "BabyLM"),
        )

    fig.suptitle(f"Natural Stories LME — {ylabel} (final checkpoint)", fontsize=13)
    fig.tight_layout()
    _savefig(fig, out_path)

    fit = dict(fits.get("All families", {"ok": False}))
    fit["metric"] = metric
    fit["x"] = x_col
    fit["by_group"] = fits
    return fit


def summarize_family(
    sweep_dir: Path,
    family: str,
    out_root: Path,
    meta: dict[str, dict],
    ppl_map: dict[str, float],
) -> pd.DataFrame:
    out_dir = out_root / family
    train_dir = out_dir / "training_curves"
    scale_params_dir = out_dir / "scaling_params"
    scale_ppl_dir = out_dir / "scaling_perplexity"
    for d in (out_dir, train_dir, scale_params_dir, scale_ppl_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = _collect_family_scores(sweep_dir, family, meta)
    if df.empty:
        print(f"[{family}] no LME scores found")
        return df

    df = _attach_perplexity(df, ppl_map)
    _write_csv(df, out_dir / "all_models_spr_lme_scores.csv")

    final = _final_checkpoint_rows(df)
    _write_csv(final, out_dir / "final_checkpoint_scores.csv")

    fit_results: dict[str, dict] = {"params": {}, "perplexity": {}}
    for metric, label in LME_METRICS:
        if metric not in df.columns:
            continue
        slug = _metric_slug(metric)
        _plot_training_curves(df, train_dir / f"{slug}_training.png", family, metric, label)
        fit_results["params"][metric] = _plot_scaling_params(
            final, scale_params_dir / f"{slug}_vs_params.png", family, metric, label
        )
        if ppl_map and final["perplexity"].notna().any():
            fit_results["perplexity"][metric] = _plot_scaling_perplexity(
                final, scale_ppl_dir / f"{slug}_vs_perplexity.png", family, metric, label
            )

    (out_dir / "scaling_fit_results.json").write_text(json.dumps(fit_results, indent=2), encoding="utf-8")
    print(f"[{family}] wrote LME summaries to {out_dir}")
    return df


def summarize_global(frames: dict[str, pd.DataFrame], out_root: Path, ppl_map: dict[str, float]) -> None:
    if not frames:
        return
    out_dir = out_root / "summary" / "all_families"
    out_dir.mkdir(parents=True, exist_ok=True)

    finals = [_final_checkpoint_rows(df) for df in frames.values() if not df.empty]
    if not finals:
        return
    final_all = pd.concat(finals, ignore_index=True)
    final_all = _attach_perplexity(final_all, ppl_map)
    _write_csv(final_all, out_dir / "all_families_final_checkpoint.csv")

    fit_all: dict[str, dict] = {"params": {}, "perplexity": {}}
    for metric, label in LME_METRICS:
        if metric not in final_all.columns:
            continue
        slug = _metric_slug(metric)
        fit_all["params"][metric] = _plot_global_scaling(
            final_all,
            out_dir / f"all_families_{slug}_vs_params.png",
            metric,
            label,
            x_col="n_params_b",
            xlabel="Parameters (billions)",
        )
        if ppl_map and final_all["perplexity"].notna().any():
            fit_all["perplexity"][metric] = _plot_global_scaling(
                final_all,
                out_dir / f"all_families_{slug}_vs_perplexity.png",
                metric,
                label,
                x_col="inv_perplexity",
                xlabel="Inverse perplexity (log₂ scale, higher = better LM)",
                log_base=2,
            )

    (out_dir / "scaling_fit_results.json").write_text(json.dumps(fit_all, indent=2), encoding="utf-8")
    print(f"Global LME summary -> {out_dir}")


def run_lme_summary(
    sweep_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    families: list[str] | None = None,
    perplexity_dir: Path | str | None = None,
) -> None:
    """Entry point for sweep scripts and CLI."""
    sweep_path = Path(sweep_dir).resolve()
    if out_dir is not None:
        out_path = Path(out_dir).resolve()
    elif "outputs_jeanzay" in sweep_path.parts:
        out_path = DEFAULT_SUMMARY_OUT_DIR.resolve()
    else:
        out_path = sweep_path
    ppl_dir = Path(perplexity_dir).resolve() if perplexity_dir else DEFAULT_PPL_DIR.resolve()

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Missing sweep dir: {sweep_path}")

    meta = _registry_meta()
    ppl_map = _load_perplexity_map(ppl_dir)
    fams = families or _discover_families(sweep_path)
    if not fams:
        print(f"No LME results under {sweep_path}")
        return

    print(f"Summarizing LME families: {fams}")
    print(f"Reading from: {sweep_path}")
    print(f"Writing to:   {out_path}")
    if ppl_map:
        print(f"Perplexity:   {len(ppl_map)} models from {ppl_dir}")
    else:
        print(f"No perplexity data under {ppl_dir} — skipping perplexity plots")

    frames: dict[str, pd.DataFrame] = {}
    for fam in fams:
        df = summarize_family(sweep_path, fam, out_path, meta, ppl_map)
        if not df.empty:
            frames[fam] = df
    summarize_global(frames, out_path, ppl_map)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep_dir",
        type=Path,
        default=DEFAULT_SWEEP_DIR,
        help="Read-only sweep results (e.g. synced outputs_jeanzay/)",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_SUMMARY_OUT_DIR,
        help="Writable summary/plot output root (default: outputs/naturalstories/surprisal_sweep)",
    )
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument(
        "--perplexity_dir",
        type=Path,
        default=DEFAULT_PPL_DIR,
        help="Perplexity JSON/CSV root (default: outputs_jeanzay/naturalstories/perplexity)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_lme_summary(
        args.sweep_dir,
        args.out_dir,
        families=args.families,
        perplexity_dir=args.perplexity_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
