#!/usr/bin/env python3
"""Perplexity-matched scaling analysis: does GROUP explain score once
perplexity is controlled for, or is the whole group effect just a proxy for
where a model sits on the (well-documented, inverse-U-shaped -- Oh & Schuler
2023) score-vs-perplexity curve?

Pools every model x checkpoint into one (dataset, group, n_params,
perplexity, score_norm) table per behavioral dataset with a delta-loglik
ceiling AND a per-checkpoint perplexity source:
  - "pythia_checkpoints" family (10 Pythia sizes x 11 training steps, added
    to neuroai/models/registry.py) supplies a dense, continuous perplexity
    range for the "LLM" group -- each (model, revision)'s own best layer by
    delta_loglik_test, exactly like build_paper_figure8's "dll" mode, but
    keeping every revision as its own row instead of collapsing to one
    best/final row per model.
  - Every other family (Memory-Constrained, Constrained-BabyLM, BabyLM, and
    the non-checkpoint-swept LLM families) contributes its single "main"
    checkpoint as one more point on the same curve.

Two outputs:
  1. reports/paper/perplexity_matched_scaling.pdf -- one panel per dataset,
     score_norm (y) vs perplexity (x, log scale), coloured by group, plus a
     pooled LOWESS/quadratic fit ignoring group. If every group's points
     hug the SAME curve, that's direct evidence architecture is fully
     mediated by perplexity (a sharper, more mechanistic version of the
     Platonic Representation Hypothesis claim already in the paper draft,
     not a failure of it). If a group sits systematically off the pooled
     curve at matched perplexity, that's evidence of a real residual
     architecture effect.
  2. A printed OLS ANCOVA per dataset: score_norm ~ poly(log10(perplexity), 2)
     + C(group), reporting the group term's significance AFTER the
     perplexity quadratic has absorbed the U-shape.

Data availability (2026-08-25): naturalstories_spr, geco, meco, provo,
ucl_reading, zuco_reading, fedorenko2016_ecog, lpp_fmri, tuckute_fmri,
podcast_ieeg, ns_fmri all have delta-loglik ceilings (dll mode). Per-CHECKPOINT
perplexity (not just one value per model) currently only exists for
provo/geco/meco (outputs_jeanzay/<dataset>_perplexity/<dataset>_perplexity_summary.csv,
with a "revision" column) -- naturalstories_spr/ucl_reading/zuco_reading/the
neural datasets only have a single main-checkpoint perplexity value each, so
they contribute only ONE point per model here (no Pythia perplexity curve)
until a dedicated per-checkpoint perplexity pass is run for them. The
pythia_checkpoints LME sweep itself (see scripts/slurm/jz/surprisal/*/
*_pythia_checkpoints_lme_cpu.sh) was still running/queued as of this
writing -- re-run this script once those jobs finish.

Usage:
    .venv/bin/python scripts/analysis/perplexity_matched_scaling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, REGISTRY_INDEX, REPO_ROOT
from neuroai.analysis.paper_style import use_paper_style
from scripts.figures.figure8_scale_matched_comparison import (
    DLL_DATASET_SPECS,
    NOISE_CEILING_DIR,
    _best_row_per_word,
    _model_id_from_dirname,
)
import json

OUT_PATH = REPO_ROOT / "reports" / "paper" / "perplexity_matched_scaling.pdf"

# Datasets with a per-CHECKPOINT perplexity summary (has a "revision" column
# with real step entries, not just one main-checkpoint value). Extend this
# list as more datasets get a dedicated per-checkpoint perplexity pass.
PERPLEXITY_STEP_DATASETS: dict[str, str] = {
    "provo": "outputs_jeanzay/provo/perplexity/provo_perplexity_summary.csv",
    "geco": "outputs_jeanzay/geco/perplexity/geco_perplexity_summary.csv",
    "meco": "outputs_jeanzay/meco/perplexity/meco_perplexity_summary.csv",
}


def _load_perplexity_per_checkpoint(slug: str) -> dict[tuple[str, str], float]:
    """(model_id, revision) -> perplexity, from a per-checkpoint summary CSV."""
    path = REPO_ROOT / PERPLEXITY_STEP_DATASETS[slug]
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "revision" not in df.columns:
        return {}
    df = df.dropna(subset=["model_id", "revision", "perplexity"])
    # Keep the first occurrence per (model_id, revision) -- some rows are
    # duplicated (e.g. revision="step143000" and revision="main" both
    # present for the same final checkpoint; harmless, just redundant).
    df = df.drop_duplicates(subset=["model_id", "revision"], keep="first")
    return {(r.model_id, r.revision): float(r.perplexity) for r in df.itertuples()}


# In-sample re-fit sweep roots (Oh & Schuler 2023 replication check: fit AND
# evaluate on the full exploratory set, no train/test partition -- see
# configs/surprisal/behavioral/<dataset>_*insample.yaml, out_of_sample: false).
# Written by the surprisal fit runners with CONFIG set to those files,
# which reuse the existing Stage-1 GPU surprisal cache (no GPU re-run needed).
# Kept in a SEPARATE tree from the out-of-sample sweep (DLL_DATASET_SPECS'
# "root"), not mixed in.
INSAMPLE_SWEEP_ROOTS: dict[str, str] = {
    "naturalstories_spr": "naturalstories/surprisal_sweep_insample",
    "geco": "geco/surprisal_sweep_insample/fp",
    "meco": "meco/surprisal_sweep_insample/fp",
    "provo": "provo_checkpoint_sweep_lme_dlt_insample/fp",
    "ucl_reading": "ucl_reading/surprisal_sweep_insample",
    "zuco_reading": "zuco_reading/surprisal_sweep_insample/nr",
}


def _best_row_insample(per_layer: pd.DataFrame) -> tuple[float, float] | None:
    """(delta_loglik_train, n_obs) at the row with max delta_loglik_train --
    the in-sample analogue of build_paper_figure8._best_row_per_word."""
    if per_layer.empty or "delta_loglik_train" not in per_layer.columns:
        return None
    valid = per_layer.dropna(subset=["delta_loglik_train", "n_obs"])
    if valid.empty:
        return None
    best = valid.loc[valid["delta_loglik_train"].idxmax()]
    return float(best["delta_loglik_train"]), float(best["n_obs"])


def _load_dll_per_checkpoint_insample(slug: str) -> pd.DataFrame:
    """Like _load_dll_per_checkpoint, but reads the in-sample (no OOS split)
    re-fit sweep instead -- matches Oh & Schuler (2023)'s own convention of
    fitting and evaluating on the full exploratory set. No noise-ceiling
    normalisation is applied here (score_raw/dll_total only) since the
    existing ceilings were computed OOS and aren't directly comparable."""
    root = INSAMPLE_SWEEP_ROOTS.get(slug)
    spec = DLL_DATASET_SPECS.get(slug)
    if root is None or spec is None:
        return pd.DataFrame()

    sweep_root = REPO_ROOT / "outputs_jeanzay" / root
    rows = []
    for fam_dir in sorted(sweep_root.iterdir()) if sweep_root.is_dir() else []:
        if not fam_dir.is_dir():
            continue
        for model_dir in sorted(fam_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            csv_paths = [p for p in model_dir.glob(spec["pattern"]) if not p.name.endswith(".lock")]
            if not csv_paths:
                continue
            mid = _model_id_from_dirname(model_dir.name)
            if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
                continue
            df = pd.read_csv(csv_paths[0])
            if df.empty or "delta_loglik_train" not in df.columns or "revision" not in df.columns:
                continue
            meta = REGISTRY_INDEX[mid]
            for revision, rev_df in df.groupby("revision"):
                picked = _best_row_insample(rev_df)
                if picked is None:
                    continue
                per_word = picked[0] / picked[1]
                rows.append(dict(
                    model_id=mid, revision=revision, family=fam_dir.name, group=meta.group,
                    n_params=meta.n_params, dll_total=picked[0], n_obs=picked[1],
                    score_raw=per_word,
                ))
    return pd.DataFrame(rows)


def _load_dll_per_checkpoint(slug: str) -> pd.DataFrame:
    """Like build_paper_figure8's _load_dll, but keeps EVERY (model_id,
    revision) as its own row instead of collapsing to one row per model."""
    spec = DLL_DATASET_SPECS.get(slug)
    if spec is None:
        return pd.DataFrame()
    ceiling_path = NOISE_CEILING_DIR / f"delta_loglik_ceiling_{spec['ceiling_tag']}_oos.json"
    if not ceiling_path.exists():
        return pd.DataFrame()
    with open(ceiling_path) as f:
        ceiling = json.load(f)["ceiling_delta_loglik_per_obs"]

    sweep_root = REPO_ROOT / "outputs_jeanzay" / spec["root"]
    rows = []
    for fam_dir in sorted(sweep_root.iterdir()) if sweep_root.is_dir() else []:
        if not fam_dir.is_dir():
            continue
        for model_dir in sorted(fam_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            csv_paths = [p for p in model_dir.glob(spec["pattern"]) if not p.name.endswith(".lock")]
            if not csv_paths:
                continue
            mid = _model_id_from_dirname(model_dir.name)
            if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
                continue
            df = pd.read_csv(csv_paths[0])
            if df.empty or "delta_loglik_test" not in df.columns or "revision" not in df.columns:
                continue
            meta = REGISTRY_INDEX[mid]
            for revision, rev_df in df.groupby("revision"):
                picked = _best_row_per_word(rev_df)
                if picked is None:
                    continue
                per_word = picked[0] / picked[1]
                rows.append(dict(
                    model_id=mid, revision=revision, family=fam_dir.name, group=meta.group,
                    n_params=meta.n_params, dll_total=picked[0], n_obs_test=picked[1],
                    score_raw=per_word, score_norm=per_word / ceiling,
                ))
    return pd.DataFrame(rows)


def _pooled_trend(x: np.ndarray, y: np.ndarray, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic fit of y on log10(x), ignoring group -- the "one curve
    explains everything" null model this whole analysis is testing against."""
    logx = np.log10(x)
    coefs = np.polyfit(logx, y, deg=2)
    grid = np.linspace(logx.min(), logx.max(), n_points)
    fit = np.polyval(coefs, grid)
    return 10 ** grid, fit


def _ancova(df: pd.DataFrame) -> str:
    """score_norm ~ poly(log10(perplexity), 2) + C(group); returns a short
    text summary of the group term's significance after the perplexity
    quadratic is fit. Uses statsmodels if available, else a manual
    F-test via numpy least squares (no extra dependency required)."""
    logp = np.log10(df["perplexity"].to_numpy())
    y = df["score_norm"].to_numpy()
    groups = df["group"].to_numpy()
    uniq_groups = sorted(set(groups))

    # Design matrix: [1, logp, logp^2, dummy(group)...] (drop first group as reference)
    X_ppl = np.column_stack([np.ones_like(logp), logp, logp ** 2])
    dummies = np.column_stack([(groups == g).astype(float) for g in uniq_groups[1:]]) if len(uniq_groups) > 1 else None
    X_full = np.column_stack([X_ppl, dummies]) if dummies is not None else X_ppl

    def _rss(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float(resid @ resid)

    rss_restricted = _rss(X_ppl)
    rss_full = _rss(X_full)
    n = len(y)
    p_full = X_full.shape[1]
    p_restricted = X_ppl.shape[1]
    df1 = p_full - p_restricted
    df2 = n - p_full
    if df1 <= 0 or df2 <= 0 or rss_full <= 0:
        return "  (not enough data / degenerate design for an F-test)"
    f_stat = ((rss_restricted - rss_full) / df1) / (rss_full / df2)
    try:
        from scipy import stats
        p_val = float(1 - stats.f.cdf(f_stat, df1, df2))
        p_str = f"p={p_val:.4g}"
    except ImportError:
        p_str = "(scipy unavailable for exact p-value)"
    r2_restricted = 1 - rss_restricted / np.sum((y - y.mean()) ** 2)
    r2_full = 1 - rss_full / np.sum((y - y.mean()) ** 2)
    return (
        f"  group term (after quadratic-in-log-perplexity): F({df1},{df2})={f_stat:.3f}, {p_str}\n"
        f"  R^2 perplexity-only={r2_restricted:.3f}  ->  R^2 + group={r2_full:.3f}"
        f"  (delta R^2={r2_full - r2_restricted:.3f})"
    )


def main() -> None:
    use_paper_style()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Perplexity-matched scaling analysis")
    print("=" * 60)

    panels: dict[str, pd.DataFrame] = {}
    for slug in PERPLEXITY_STEP_DATASETS:
        dll = _load_dll_per_checkpoint(slug)
        if dll.empty:
            print(f"[{slug}] no dll data yet (pythia_checkpoints sweep still running?) -- skipping")
            continue
        ppl_map = _load_perplexity_per_checkpoint(slug)

        rows = []
        for r in dll.itertuples():
            key = (r.model_id, r.revision)
            ppl = ppl_map.get(key)
            if ppl is None and r.revision == "main":
                # non-checkpoint-swept families: try their bare model_id with
                # any revision label the perplexity CSV happens to use for
                # the single main checkpoint.
                candidates = [v for (mid, rev), v in ppl_map.items() if mid == r.model_id]
                ppl = candidates[0] if candidates else None
            if ppl is None or ppl <= 0:
                continue
            rows.append(dict(
                model_id=r.model_id, revision=r.revision, group=r.group,
                n_params=r.n_params, perplexity=ppl, score_norm=r.score_norm,
            ))
        merged = pd.DataFrame(rows)
        if merged.empty:
            print(f"[{slug}] dll data present but no perplexity joined -- skipping")
            continue
        panels[slug] = merged
        n_pythia_steps = (merged["group"] == "LLM").sum()
        print(f"[{slug}] {len(merged)} (model, checkpoint) points, "
              f"{n_pythia_steps} on the LLM/Pythia perplexity curve")
        print(_ancova(merged))
        print()

    if not panels:
        print("Nothing to plot yet -- re-run once the pythia_checkpoints LME sweep "
              "(scripts/slurm/jz/surprisal/*/*_pythia_checkpoints_lme_cpu.sh) completes.")
        return

    with PdfPages(OUT_PATH) as pdf:
        ncols = min(3, len(panels))
        nrows = -(-len(panels) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
        for i, (slug, df) in enumerate(panels.items()):
            ax = axes[i // ncols][i % ncols]
            for g in GROUP_ORDER:
                sub = df[df["group"] == g]
                if sub.empty:
                    continue
                ax.scatter(sub["perplexity"], sub["score_norm"], s=14, color=GROUP_COLORS[g],
                           label=GROUP_DISPLAY[g], alpha=0.8, edgecolor="white", linewidth=0.3)
            gx, gy = _pooled_trend(df["perplexity"].to_numpy(), df["score_norm"].to_numpy())
            ax.plot(gx, gy, color="#333333", linewidth=1.4, linestyle="--", zorder=5,
                    label="pooled fit (no group)")
            ax.set_xscale("log")
            ax.set_xlabel("perplexity (log scale)", fontsize=8)
            ax.set_ylabel("score/ceiling", fontsize=8)
            ax.text(0.02, 0.98, slug, transform=ax.transAxes, fontsize=8, fontweight="bold",
                    va="top", ha="left", color="#333333")
            ax.tick_params(labelsize=7)
        for j in range(len(panels), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        handles = [Patch(facecolor=GROUP_COLORS[g], label=GROUP_DISPLAY[g]) for g in GROUP_ORDER]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.text(0.01, 0.995,
                  "score/ceiling (dll) vs perplexity, pooled across model x checkpoint -- "
                  "dashed line: quadratic-in-log(perplexity) fit ignoring group",
                  fontsize=8, color="#666666", ha="left", va="top")
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
