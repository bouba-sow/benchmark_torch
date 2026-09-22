#!/usr/bin/env python3
"""Summarize and plot Provo Corpus embedding-ridge checkpoint sweeps.

Reads ``<sweep_dir>/<family>/<model>/provo_embed_checkpoint_scores.csv``
(mean-across-subjects, one row per (checkpoint, layer)) and
``provo_embed_by_subject.csv`` (one row per (checkpoint, subject, layer)) and
writes cross-family plots:

- per-family training curves (best-layer Spearman r vs tokens seen), shaded
  by the across-subject standard error at each checkpoint
- scaling law vs parameter count (final checkpoint, best layer), with
  across-subject SEM error bars
- layerwise heatmap grid across all models (relative-depth-binned, since raw
  layer counts differ across families/sizes)

Mirrors ``summarize/provo_lme_families.py``'s structure and CLI, adapted for
the embed pipeline's per-layer / per-subject shape (see
``neuroai.benchmarks.provo_embed`` module docstring for the per-subject
fitting design). One Provo measure (SP/FP/GP) per invocation.

Most registry families (all but pythia/olmo2) are "main-only": one final
checkpoint per model size, no training-step sweep — ``tokens_seen`` is 0 for
every model in those families, so a training-curve plot (x=tokens) degenerates
to a single point per model. Every family therefore also gets a
**scaling-vs-parameters** plot (x=n_params), which is the informative one for
main-only families; pythia/olmo2 additionally get a real training curve.

Usage::

    python scripts/summarize/provo_embed_families.py \\
        --measure fp \\
        --sweep_dir outputs_jeanzay/provo/embedding_sweep/fp \\
        --out_dir outputs/provo/embedding_sweep/fp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuroai.models.registry import LPP_FAMILIES, model_spec_to_cfg
from neuroai.plotting.scaling import plot_layerwise_heatmap
from neuroai.analysis.benchmark_common import GROUP_ORDER, family_group, plot_scaling_law_by_group

SCORES_FILENAME = "provo_embed_checkpoint_scores.csv"
BY_SUBJECT_FILENAME = "provo_embed_by_subject.csv"

DEFAULT_MEASURE = "fp"
DEFAULT_PPL_DIR = Path("outputs_jeanzay/provo/perplexity")

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


# ---------------------------------------------------------------------------
# Small helpers (mirrors summarize/provo_lme_families.py)
# ---------------------------------------------------------------------------

def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "_")


def _registry_meta() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for fam, specs in LPP_FAMILIES.items():
        for spec in specs:
            model_spec_to_cfg(spec, fam)
            row = {
                "model_id": spec.model_id,
                "model_label": spec.label,
                "n_params": float(spec.n_params),
                "family": fam,
            }
            meta[spec.model_id] = row
            meta[_safe_model_dir(spec.model_id)] = row
    return meta


def _savefig(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _discover_families(sweep_dir: Path) -> list[str]:
    families: list[str] = []
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_dir() or p.name in {"summary", "logs"}:
            continue
        if list(p.glob(f"*/{SCORES_FILENAME}")):
            families.append(p.name)
    return families


def _collect_family(
    sweep_dir: Path, family: str, meta: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (scores_df, by_subject_df) for every model in ``family``."""
    fam_dir = sweep_dir / family
    score_parts: list[pd.DataFrame] = []
    subj_parts: list[pd.DataFrame] = []
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "summary":
            continue
        model_id = model_dir.name.replace("_", "/", 1)

        scores_path = model_dir / SCORES_FILENAME
        if scores_path.is_file():
            df = pd.read_csv(scores_path)
            if not df.empty:
                df.insert(0, "model_id", model_id)
                df.insert(1, "family", family)
                score_parts.append(df)

        subj_path = model_dir / BY_SUBJECT_FILENAME
        if subj_path.is_file():
            sdf = pd.read_csv(subj_path)
            if not sdf.empty:
                sdf.insert(0, "model_id", model_id)
                sdf.insert(1, "family", family)
                subj_parts.append(sdf)

    scores_df = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
    subj_df = pd.concat(subj_parts, ignore_index=True) if subj_parts else pd.DataFrame()

    if not scores_df.empty:
        scores_df = _attach_metadata(scores_df, meta)
    return scores_df, subj_df


def _attach_metadata(df: pd.DataFrame, meta: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    labels, n_params = [], []
    for mid in out["model_id"]:
        m = meta.get(mid, meta.get(_safe_model_dir(mid), {}))
        labels.append(m.get("model_label", mid))
        n_params.append(m.get("n_params", np.nan))
    out["model_label"] = labels
    out["n_params"] = n_params
    out["n_params_b"] = out["n_params"] / 1e9
    out["log10_n_params"] = np.log10(out["n_params"].where(out["n_params"] > 0))
    out["tokens_seen_b"] = out["tokens_seen"].astype(float) / 1e9
    return out


def _load_perplexity_map(perplexity_dir: Path) -> dict[str, float]:
    """Provo perplexity is measure-agnostic — one summary file for all of
    SP/FP/GP (see provo_lme.py / run/surprisal/behavioral/provo_extract.py), keyed by
    model_id (the canonical HF repo id, same as this script's model_id)."""
    if not perplexity_dir.is_dir():
        return {}
    out: dict[str, float] = {}
    summary_csv = perplexity_dir / "provo_perplexity_summary.csv"
    if summary_csv.is_file():
        summary = pd.read_csv(summary_csv)
        if {"model_id", "perplexity"}.issubset(summary.columns):
            for _, row in summary.iterrows():
                ppl = row["perplexity"]
                if pd.notna(ppl):
                    out[str(row["model_id"])] = float(ppl)
    for p in sorted(perplexity_dir.rglob("*_provo_perplexity.json")):
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
        out["perplexity"] = np.nan
        out["inv_perplexity"] = np.nan
        return out
    out["perplexity"] = out["model_id"].map(ppl_map)
    mask = out["perplexity"].notna() & out["perplexity"].gt(0)
    out["inv_perplexity"] = np.where(mask, 1.0 / out["perplexity"], np.nan)
    return out


def _best_layer_rows(scores_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model_id, revision): the layer with the highest Spearman r."""
    if scores_df.empty:
        return scores_df
    idx = scores_df.groupby(["model_id", "revision"])["spearmanr"].idxmax()
    return scores_df.loc[idx].reset_index(drop=True)


def _final_checkpoint_rows(best_layer_df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: best-layer score at the latest (max tokens_seen) checkpoint."""
    if best_layer_df.empty:
        return best_layer_df
    idx = best_layer_df.groupby("model_id")["tokens_seen"].idxmax()
    return best_layer_df.loc[idx].sort_values("n_params_b", kind="mergesort").reset_index(drop=True)


def _last_layer_rows(scores_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model_id, revision): the final (highest-index) layer."""
    if scores_df.empty:
        return scores_df
    idx = scores_df.groupby(["model_id", "revision"])["layer"].idxmax()
    return scores_df.loc[idx].reset_index(drop=True)


def _subject_sem(
    subj_df: pd.DataFrame, model_id: str, revision: str, layer: int, metric: str,
) -> tuple[float, int]:
    """Across-subject mean SEM (std / sqrt(n)) for one (model, checkpoint, layer)."""
    if subj_df.empty:
        return float("nan"), 0
    sub = subj_df[
        (subj_df["model_id"] == model_id) & (subj_df["revision"] == revision) & (subj_df["layer"] == layer)
    ]
    vals = sub[metric].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n < 2:
        return float("nan"), n
    return float(np.std(vals, ddof=1) / np.sqrt(n)), n


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_family_training_curve(
    best_layer_df: pd.DataFrame, subj_df: pd.DataFrame, out_path: Path, family: str, measure: str,
    *, layer_label: str = "Best-layer",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    color = _FAMILY_COLORS.get(family, "#4878d0")
    cmap = plt.get_cmap("viridis")
    models = sorted(best_layer_df["model_id"].unique())
    for i, model_id in enumerate(models):
        g = best_layer_df[best_layer_df["model_id"] == model_id].sort_values("tokens_seen")
        label = g["model_label"].iloc[0]
        c = cmap(i / max(1, len(models) - 1)) if len(models) > 1 else color
        sems = np.array([
            _subject_sem(subj_df, model_id, rev, int(li), "spearmanr")[0]
            for rev, li in zip(g["revision"], g["layer"])
        ])
        y = g["spearmanr"].to_numpy(dtype=float)
        x = g["tokens_seen_b"].to_numpy(dtype=float)
        ax.plot(x, y, "o-", label=label, color=c, alpha=0.9)
        if np.isfinite(sems).any():
            lo = y - np.nan_to_num(sems)
            hi = y + np.nan_to_num(sems)
            ax.fill_between(x, lo, hi, color=c, alpha=0.15)
    ax.set_xlabel("Tokens seen (billions)")
    ax.set_ylabel(f"{layer_label} Spearman r (mean across subjects, ±SEM)")
    ax.set_title(f"Provo embed [{measure.upper()}] — {family} ({layer_label.lower()})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    _savefig(fig, out_path)


def _plot_scaling(
    final_df: pd.DataFrame,
    subj_df: pd.DataFrame,
    out_path: Path,
    *,
    x_col: str,
    xlabel: str,
    title: str,
    group_by_family: bool = True,
    single_color: str = "#4878d0",
    layer_label: str = "Best-layer",
) -> dict:
    """Shared scaling-plot renderer: score vs ``x_col`` (params or perplexity),
    with across-subject SEM error bars. ``group_by_family=True`` colors/legends
    by family (cross-family plot); ``False`` uses one color (per-family plot).
    """
    sub = final_df[np.isfinite(final_df[x_col]) & np.isfinite(final_df["spearmanr"])].copy()
    if sub.empty:
        return {"ok": False}

    fig, ax = plt.subplots(figsize=(9, 6) if group_by_family else (7, 5))
    groups = sub.groupby("family") if group_by_family else [(None, sub)]
    for family, grp in groups:
        color = _FAMILY_COLORS.get(family, single_color) if family else single_color
        yerr = np.array([
            _subject_sem(subj_df, mid, rev, int(li), "spearmanr")[0]
            for mid, rev, li in zip(grp["model_id"], grp["revision"], grp["layer"])
        ])
        ax.errorbar(
            grp[x_col], grp["spearmanr"], yerr=np.nan_to_num(yerr),
            fmt="o", color=color, alpha=0.85, capsize=3, label=family if family else None,
        )
        for _, row in grp.iterrows():
            ax.annotate(
                row["model_label"], (row[x_col], row["spearmanr"]),
                textcoords="offset points", xytext=(4, 2), fontsize=6,
            )

    x = np.log10(sub[x_col].to_numpy(float))
    y = sub["spearmanr"].to_numpy(float)
    fit = {"ok": False, "r2": np.nan, "slope": np.nan, "intercept": np.nan}
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2:
        coeffs = np.polyfit(x[mask], y[mask], deg=1)
        y_hat = coeffs[0] * x[mask] + coeffs[1]
        ss_res = float(np.sum((y[mask] - y_hat) ** 2))
        ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        fit = {"ok": True, "r2": float(r2), "slope": float(coeffs[0]), "intercept": float(coeffs[1])}
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(10 ** x_line, coeffs[0] * x_line + coeffs[1], "--", color="#444444",
                 linewidth=1.2, label=f"fit R²={r2:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{layer_label} Spearman r (final checkpoint, mean ± SEM across subjects)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    _savefig(fig, out_path)
    return fit


def _interp_relative_depth(layer_rel: np.ndarray, scores: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Interpolate a per-layer score curve onto ``n_bins`` common relative-depth
    points (0..1) so models with different layer counts can share one heatmap row."""
    bins = np.linspace(0.0, 1.0, n_bins)
    order = np.argsort(layer_rel)
    return np.interp(bins, layer_rel[order], scores[order])


def _draw_layer_profiles_ax(
    ax, scores_df: pd.DataFrame, title: str, *, group_by_family: bool,
) -> None:
    """Spearman r vs relative depth (0=input, 1=output), one line per model,
    interpolated onto a common 20-point depth axis so models with different
    layer counts overlay cleanly. ``group_by_family=True`` colors by family
    with one legend entry per family (cross-family plot, many models);
    ``False`` colors by model (viridis cycle, per-family plot)."""
    idx = scores_df.groupby("model_id")["tokens_seen"].transform("max") == scores_df["tokens_seen"]
    final_all_layers = scores_df[idx]
    if final_all_layers.empty:
        ax.set_title(f"{title} (no data)")
        ax.set_axis_off()
        return

    bins = np.linspace(0.0, 1.0, 20)
    model_ids = sorted(final_all_layers["model_id"].unique())
    cmap = plt.get_cmap("viridis")
    seen_families: set[str] = set()
    for i, model_id in enumerate(model_ids):
        g = final_all_layers[final_all_layers["model_id"] == model_id].sort_values("layer")
        if g.empty or g["layer_rel"].isna().all():
            continue
        y = _interp_relative_depth(g["layer_rel"].to_numpy(float), g["spearmanr"].to_numpy(float))
        fam = g["family"].iloc[0]
        if group_by_family:
            color = _FAMILY_COLORS.get(fam, "#4878d0")
            label = fam if fam not in seen_families else None
            seen_families.add(fam)
            ax.plot(bins, y, color=color, alpha=0.6, linewidth=1.3, label=label)
        else:
            color = cmap(i / max(1, len(model_ids) - 1)) if len(model_ids) > 1 else "#4878d0"
            label = g["model_label"].iloc[0]
            ax.plot(bins, y, color=color, alpha=0.9, linewidth=1.6, marker="o", markersize=3, label=label)

    ax.axhline(0, color="k", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_xlabel("Relative depth (0 = input embedding, 1 = final layer)")
    ax.set_ylabel("Spearman r (final checkpoint, mean across subjects)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")


def _plot_layer_profiles(
    scores_df: pd.DataFrame, out_path: Path, title: str, *, group_by_family: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_layer_profiles_ax(ax, scores_df, title, group_by_family=group_by_family)
    fig.tight_layout()
    _savefig(fig, out_path)


def _plot_layer_profiles_by_group(scores_df: pd.DataFrame, out_path: Path, title: str) -> None:
    """2x2 grid (All groups / LLM / Memory-Constrained / BabyLM): each panel
    its own Axes, colored by family, so a category's own depth profile is
    readable on its own instead of being crushed under every other family's
    lines. Mirrors ``plot_scaling_law_by_group`` (neuroai/analysis/benchmark_common.py)."""
    scores_with_group = scores_df.copy()
    scores_with_group["group"] = scores_with_group["family"].map(family_group)
    panels = [("All groups", scores_with_group)]
    panels += [(g, scores_with_group[scores_with_group["group"] == g]) for g in GROUP_ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, (panel_title, sub) in zip(axes.flat, panels):
        n = sub["model_id"].nunique() if not sub.empty else 0
        _draw_layer_profiles_ax(ax, sub, f"{panel_title} (n={n})", group_by_family=True)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _savefig(fig, out_path)


def _plot_layerwise_heatmap_grid(scores_df: pd.DataFrame, out_path: Path, measure: str) -> None:
    # Final checkpoint per model (all layers, not just best).
    idx = scores_df.groupby("model_id")["tokens_seen"].transform("max") == scores_df["tokens_seen"]
    final_all_layers = scores_df[idx]

    rows: list[np.ndarray] = []
    labels: list[str] = []
    order = final_all_layers[["model_id", "family", "n_params_b"]].drop_duplicates().sort_values(
        ["family", "n_params_b"], kind="mergesort",
    )
    for _, r in order.iterrows():
        g = final_all_layers[final_all_layers["model_id"] == r["model_id"]].sort_values("layer")
        if g.empty or g["layer_rel"].isna().all():
            continue
        rows.append(_interp_relative_depth(
            g["layer_rel"].to_numpy(float), g["spearmanr"].to_numpy(float),
        ))
        labels.append(f"{r['family']}/{g['model_label'].iloc[0]}")

    if not rows:
        return
    mat = np.stack(rows, axis=0)
    plot_layerwise_heatmap(
        mat, labels, metric="Spearman r",
        title=f"Provo embed layerwise (final checkpoint, relative depth) [{measure.upper()}]",
        save_path=str(out_path),
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_embed_summary(
    sweep_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    measure: str = DEFAULT_MEASURE,
    families: list[str] | None = None,
    perplexity_dir: Path | str | None = None,
) -> None:
    sweep_path = Path(sweep_dir).resolve()
    out_path = Path(out_dir).resolve() if out_dir else sweep_path

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Missing sweep dir: {sweep_path}")

    meta = _registry_meta()
    ppl_dir = Path(perplexity_dir).resolve() if perplexity_dir else DEFAULT_PPL_DIR.resolve()
    ppl_map = _load_perplexity_map(ppl_dir)
    fams = families or _discover_families(sweep_path)
    if not fams:
        print(f"No Provo embed results under {sweep_path}")
        return

    print(f"Summarizing Provo embed families [{measure}]: {fams}")
    print(f"Reading from: {sweep_path}")
    print(f"Writing to:   {out_path}")
    if ppl_map:
        print(f"Perplexity:   {len(ppl_map)} models from {ppl_dir}")
    else:
        print(f"No perplexity data under {ppl_dir} — skipping perplexity plots")

    all_scores: list[pd.DataFrame] = []
    all_subj: list[pd.DataFrame] = []
    fit_results: dict[str, dict] = {}

    for fam in fams:
        scores_df, subj_df = _collect_family(sweep_path, fam, meta)
        if scores_df.empty:
            continue
        all_scores.append(scores_df)
        if not subj_df.empty:
            all_subj.append(subj_df)

        best_layer_df = _best_layer_rows(scores_df)
        last_layer_df = _last_layer_rows(scores_df)
        # Training curve (x=tokens seen): only informative for pythia/olmo2,
        # which are the only families with a real checkpoint sweep — every
        # other (main-only) family has one checkpoint per model, so this plot
        # degenerates to a single point per model at tokens_seen=0. The
        # scaling-vs-params plot below is the informative one for those.
        _plot_family_training_curve(
            best_layer_df, subj_df, out_path / fam / "provo_embed_training_curve.png", fam, measure,
        )
        _plot_family_training_curve(
            last_layer_df, subj_df, out_path / fam / "provo_embed_training_curve_last_layer.png", fam, measure,
            layer_label="Last-layer",
        )
        _plot_layer_profiles(
            scores_df, out_path / fam / "provo_embed_layer_profiles.png",
            f"Provo embed [{measure.upper()}] — {fam} — layer profiles", group_by_family=False,
        )

        final_fam = _final_checkpoint_rows(best_layer_df)
        final_fam_last = _final_checkpoint_rows(last_layer_df)
        color = _FAMILY_COLORS.get(fam, "#4878d0")
        fit_results.setdefault(fam, {})["scaling_vs_params"] = _plot_scaling(
            final_fam, subj_df, out_path / fam / "provo_embed_scaling_vs_params.png",
            x_col="n_params_b", xlabel="Parameters (billions)",
            title=f"Provo embed [{measure.upper()}] — {fam} — scaling vs model size",
            group_by_family=False, single_color=color,
        )
        fit_results[fam]["scaling_vs_params_last_layer"] = _plot_scaling(
            final_fam_last, subj_df, out_path / fam / "provo_embed_scaling_vs_params_last_layer.png",
            x_col="n_params_b", xlabel="Parameters (billions)",
            title=f"Provo embed [{measure.upper()}] — {fam} — scaling vs model size (last layer)",
            group_by_family=False, single_color=color, layer_label="Last-layer",
        )
        if ppl_map:
            final_fam_ppl = _attach_perplexity(final_fam, ppl_map)
            fit_results[fam]["scaling_vs_perplexity"] = _plot_scaling(
                final_fam_ppl, subj_df, out_path / fam / "provo_embed_scaling_vs_perplexity.png",
                x_col="inv_perplexity", xlabel="Inverse perplexity (higher = better LM)",
                title=f"Provo embed [{measure.upper()}] — {fam} — scaling vs LM quality",
                group_by_family=False, single_color=color,
            )
            final_fam_last_ppl = _attach_perplexity(final_fam_last, ppl_map)
            fit_results[fam]["scaling_vs_perplexity_last_layer"] = _plot_scaling(
                final_fam_last_ppl, subj_df,
                out_path / fam / "provo_embed_scaling_vs_perplexity_last_layer.png",
                x_col="inv_perplexity", xlabel="Inverse perplexity (higher = better LM)",
                title=f"Provo embed [{measure.upper()}] — {fam} — scaling vs LM quality (last layer)",
                group_by_family=False, single_color=color, layer_label="Last-layer",
            )

    if not all_scores:
        print(f"No {SCORES_FILENAME} found under {sweep_path}")
        return

    combined_scores = pd.concat(all_scores, ignore_index=True)
    combined_subj = pd.concat(all_subj, ignore_index=True) if all_subj else pd.DataFrame()

    summary_dir = out_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    combined_scores.to_csv(summary_dir / "all_families_provo_embed_scores.csv", index=False)
    if not combined_subj.empty:
        combined_subj.to_csv(summary_dir / "all_families_provo_embed_by_subject.csv", index=False)

    best_layer_all = _best_layer_rows(combined_scores)
    final_all = _final_checkpoint_rows(best_layer_all)
    if ppl_map:
        final_all = _attach_perplexity(final_all, ppl_map)
    final_all.to_csv(summary_dir / "final_checkpoint_best_layer.csv", index=False)

    last_layer_all = _last_layer_rows(combined_scores)
    final_all_last = _final_checkpoint_rows(last_layer_all)
    if ppl_map:
        final_all_last = _attach_perplexity(final_all_last, ppl_map)
    final_all_last.to_csv(summary_dir / "final_checkpoint_last_layer.csv", index=False)

    fit_results["all_families_scaling_vs_params"] = _plot_scaling(
        final_all, combined_subj, summary_dir / "provo_embed_scaling_vs_params.png",
        x_col="n_params_b", xlabel="Parameters (billions)",
        title=f"Provo embed scaling law [{measure.upper()}]", group_by_family=True,
    )
    fit_results["all_families_scaling_vs_params_last_layer"] = _plot_scaling(
        final_all_last, combined_subj, summary_dir / "provo_embed_scaling_vs_params_last_layer.png",
        x_col="n_params_b", xlabel="Parameters (billions)",
        title=f"Provo embed scaling law [{measure.upper()}] (last layer)", group_by_family=True,
        layer_label="Last-layer",
    )
    if ppl_map:
        fit_results["all_families_scaling_vs_perplexity"] = _plot_scaling(
            final_all, combined_subj, summary_dir / "provo_embed_scaling_vs_perplexity.png",
            x_col="inv_perplexity", xlabel="Inverse perplexity (higher = better LM)",
            title=f"Provo embed scaling vs LM quality [{measure.upper()}]", group_by_family=True,
        )
        fit_results["all_families_scaling_vs_perplexity_last_layer"] = _plot_scaling(
            final_all_last, combined_subj,
            summary_dir / "provo_embed_scaling_vs_perplexity_last_layer.png",
            x_col="inv_perplexity", xlabel="Inverse perplexity (higher = better LM)",
            title=f"Provo embed scaling vs LM quality [{measure.upper()}] (last layer)", group_by_family=True,
            layer_label="Last-layer",
        )
    _plot_layer_profiles(
        combined_scores, summary_dir / "provo_embed_layer_profiles.png",
        f"Provo embed layer profiles (all families) [{measure.upper()}]", group_by_family=True,
    )
    _plot_layer_profiles_by_group(
        combined_scores, summary_dir / "provo_embed_layer_profiles_by_group.png",
        f"Provo embed layer profiles by model category [{measure.upper()}]",
    )
    _plot_layerwise_heatmap_grid(
        combined_scores, summary_dir / "provo_embed_layerwise_heatmap_grid.png", measure,
    )

    # 2x2 grids (All groups / LLM / Memory-Constrained / BabyLM), one figure
    # for parameters and one for perplexity. Each panel is its own Axes, so
    # matplotlib autoscales the x-range to that panel's own data — this is
    # what keeps Memory-Constrained models (whose perplexities often sit very
    # close together) readable instead of being crushed onto the full
    # cross-category range. Mirrors scripts/summarize/report.py's
    # plot_scaling_law_by_group (neuroai/analysis/benchmark_common.py).
    by_group_df = final_all.copy()
    by_group_df["group"] = by_group_df["family"].map(family_group)
    by_group_df["label"] = by_group_df["model_label"]
    plot_scaling_law_by_group(
        by_group_df, value_col="spearmanr",
        out_path=summary_dir / "provo_embed_scaling_vs_params_by_group.png",
        ylabel="Best-layer Spearman r (final checkpoint, mean across subjects)",
        title=f"Provo embed scaling law by model category [{measure.upper()}]",
    )
    if ppl_map:
        by_group_ppl_df = by_group_df.copy()
        by_group_ppl_df["perplexity"] = by_group_ppl_df["model_id"].map(ppl_map)
        plot_scaling_law_by_group(
            by_group_ppl_df, value_col="spearmanr",
            out_path=summary_dir / "provo_embed_scaling_vs_perplexity_by_group.png",
            ylabel="Best-layer Spearman r (final checkpoint, mean across subjects)",
            title=f"Provo embed scaling vs LM quality by model category [{measure.upper()}]",
            ppl_col="perplexity",
        )

    by_group_df_last = final_all_last.copy()
    by_group_df_last["group"] = by_group_df_last["family"].map(family_group)
    by_group_df_last["label"] = by_group_df_last["model_label"]
    plot_scaling_law_by_group(
        by_group_df_last, value_col="spearmanr",
        out_path=summary_dir / "provo_embed_scaling_vs_params_by_group_last_layer.png",
        ylabel="Last-layer Spearman r (final checkpoint, mean across subjects)",
        title=f"Provo embed scaling law by model category [{measure.upper()}] (last layer)",
    )
    if ppl_map:
        by_group_ppl_df_last = by_group_df_last.copy()
        by_group_ppl_df_last["perplexity"] = by_group_ppl_df_last["model_id"].map(ppl_map)
        plot_scaling_law_by_group(
            by_group_ppl_df_last, value_col="spearmanr",
            out_path=summary_dir / "provo_embed_scaling_vs_perplexity_by_group_last_layer.png",
            ylabel="Last-layer Spearman r (final checkpoint, mean across subjects)",
            title=f"Provo embed scaling vs LM quality by model category [{measure.upper()}] (last layer)",
            ppl_col="perplexity",
        )

    (summary_dir / "scaling_fit_results.json").write_text(json.dumps(fit_results, indent=2), encoding="utf-8")

    print(f"Provo embed summary -> {summary_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--measure", default=DEFAULT_MEASURE, choices=["sp", "fp", "gp", "SP", "FP", "GP"],
        help="Provo measure whose sweep to summarize (default: fp)",
    )
    p.add_argument(
        "--sweep_dir", type=Path, default=None,
        help="Read-only sweep results (default: outputs_jeanzay/provo/embedding_sweep/<measure>)",
    )
    p.add_argument(
        "--out_dir", type=Path, default=None,
        help="Writable summary/plot output root (default: same as --sweep_dir)",
    )
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument(
        "--perplexity_dir", type=Path, default=DEFAULT_PPL_DIR,
        help="Perplexity JSON/CSV root (default: outputs_jeanzay/provo/perplexity)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    measure = args.measure.lower()
    sweep_dir = args.sweep_dir or Path(f"outputs_jeanzay/provo/embedding_sweep/{measure}")
    run_embed_summary(
        sweep_dir, args.out_dir, measure=measure, families=args.families,
        perplexity_dir=args.perplexity_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
