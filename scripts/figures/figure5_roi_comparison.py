#!/usr/bin/env python3
"""Paper figure 5: ROI comparison across the left-hemisphere Fedorenko
language-network parcels (IFGorb, IFG, MFG, AntTemp, PostTemp), broken down
by MODEL CLASS (Standard LLM / Memory-Constrained / Constrained-BabyLM /
BabyLM).

Two outputs:

  reports/paper/roi_comparison_embedding.pdf  -- embedding/ridge track.
      One page each for LPP fMRI, Natural Stories fMRI, Podcast iEEG.
  reports/paper/roi_comparison_surprisal.pdf  -- surprisal/LME track.
      Natural Stories fMRI only: the NS LME sweep was run per froi
      (ns_fmri_lme_by_roi.csv). LPP's LME fit (lpp_lme_by_layer.csv) and
      Podcast's (podcast_lme_by_lag.csv) pool every region / electrode into
      ONE model per layer/lag with region as a random effect -- there is no
      per-ROI held-out delta stored, and producing one means re-fitting the
      LME separately per parcel for every model (a cluster sweep, not a
      re-aggregation). Until that runs, LPP and Podcast have no surprisal
      ROI page.

Each page has two parts:

  * a GROUPED BAR CHART -- 5 ROIs x 4 model classes, each bar the mean
    best-layer score (+/- SEM) across that class's models;
  * a row of 5 RELATIVE-LAYER-DEPTH plots, one per ROI, with one mean +/- SEM
    curve per model class (draw_layer_profiles_interp, the same resample-onto-
    a-shared-grid method used by layerwise_depth_profiles.pdf) -- so the depth
    at which each class peaks in each region is visible.

Score: embedding track = raw Pearson r (best layer per model / electrode);
surprisal track = held-out delta Pearson r (full LME minus baseline LME).
There is no per-ROI noise-ceiling estimate, so compare ROI and depth
PATTERNS within a panel, not absolute magnitudes across datasets. Podcast's
MFG has only ~3 left-hemisphere language-responsive electrodes -- shown but
unreliable (small n).

Data sources (per (model, layer, ROI), aggregated across subjects /
electrodes):
  - LPP embedding:  lpp/embedding_sweep/*/lpp_checkpoint_by_fedorenko_parcel.csv
                    (mean_pearson_left, subject-averaged BOLD).
  - NS embedding:   ns_fmri/embedding_sweep/*/ns_fmri_checkpoint_layerwise.csv
                    (pearsonr, LANG network, L hemisphere).
  - Podcast embed.: podcast/embedding_sweep/*/podcast_by_electrode_layerwise.csv
                    (current canonical podcast sweep; the older
                    podcast/checkpoint_sweep_surprisal is a different scoring
                    window and lacks BabyLM/gpt2/qwen3) joined to
                    outputs/podcast/electrode_roi_map_language_responsive.csv
                    (L hemisphere, assigned electrodes, <=10mm cutoff).
  - NS surprisal:   ns_fmri/surprisal_sweep/*/ns_fmri_lme_by_roi.csv
                    (test_delta_pearsonr, LANG network, L hemisphere).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from neuroai.analysis.benchmark_common import (
    EXCLUDED_MODEL_IDS, GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, OUTPUTS_JEANZAY, REPO_ROOT,
    draw_layer_profiles_interp, lookup_model_meta,
)
from neuroai.analysis.paper_style import use_paper_style
from scripts.summarize.benchmark import (
    _iter_model_score_files, _lpp_flattened_memory_models, _lpp_flattened_registry_models,
    _model_id_from_dir, _pick_final_row, _read_csv_safe,
)

OUT_DIR = REPO_ROOT / "reports" / "paper"
ROI_ORDER = ("IFGorb", "IFG", "MFG", "AntTemp", "PostTemp")

LPP_SWEEP = OUTPUTS_JEANZAY / "lpp/embedding_sweep"
NS_SWEEP = OUTPUTS_JEANZAY / "ns_fmri/embedding_sweep"
NS_LME_SWEEP = OUTPUTS_JEANZAY / "ns_fmri/surprisal_sweep"
# podcast/embedding_sweep is the CURRENT canonical podcast embedding sweep
# (window [-1, 1] s, no baseline correction -- see configs/embedding/neural/podcast.yaml
# and summarize_benchmark.load_podcast_ieeg). The older
# podcast/checkpoint_sweep_surprisal used a different scoring window (~3x smaller
# Pearson r) and never got the BabyLM / gpt2 / qwen3 families -- do NOT mix them.
PODCAST_SWEEP = OUTPUTS_JEANZAY / "podcast/embedding_sweep"
PODCAST_ROI_MAP = REPO_ROOT / "outputs" / "podcast/electrode_roi_map_language_responsive.csv"


# ---------------------------------------------------------------------------
# Loaders -- each returns a long df: model_id, group, roi, layer, layer_frac, score
# ---------------------------------------------------------------------------

def _final_checkpoint_rows(scores: pd.DataFrame) -> pd.DataFrame:
    key = "tokens_seen" if "tokens_seen" in scores.columns else ("step" if "step" in scores.columns else None)
    if key is None or scores[key].isna().all():
        return scores
    final = _pick_final_row(scores)
    return scores[scores[key] == final[key]]


def _add_layer_frac(df: pd.DataFrame) -> pd.DataFrame:
    """layer_frac = layer / max(layer) per model (0 if the model has one layer)."""
    df = df.copy()
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df = df.dropna(subset=["layer"])
    maxl = df.groupby("model_id")["layer"].transform("max")
    df["layer_frac"] = np.where(maxl > 0, df["layer"] / maxl, 0.0)
    return df


def _iter_final(sweep: Path, filename: str):
    for family, model_dir, files in _iter_model_score_files(sweep, filename):
        model_id = _model_id_from_dir(model_dir)
        if model_id in EXCLUDED_MODEL_IDS:
            continue
        frames = [d for f in files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = _final_checkpoint_rows(pd.concat(frames, ignore_index=True))
        yield model_id, lookup_model_meta(model_id, family).group, scores


def _iter_lpp_final(filename: str):
    """LPP sweep has THREE layouts (see summarize_benchmark.load_lpp_fmri):
    nested <family>/<model>/, flat Lanni-ni_* Memory-Constrained dirs, and flat
    <org>_<repo>/ BabyLM / cbr-rnn dirs. Walk all three so the class breakdown
    is not silently LLM-only."""
    nested = list(_iter_model_score_files(LPP_SWEEP, filename))
    lanni = list(_lpp_flattened_memory_models(LPP_SWEEP, filename))
    seen = {d.name for _, d, _ in nested} | {d.name for _, d, _ in lanni}
    extra = list(_lpp_flattened_registry_models(LPP_SWEEP, filename, seen))
    for family, model_dir, files in nested + lanni + extra:
        model_id = _model_id_from_dir(model_dir)
        if model_id in EXCLUDED_MODEL_IDS:
            continue
        frames = [d for f in files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = _final_checkpoint_rows(pd.concat(frames, ignore_index=True))
        yield model_id, lookup_model_meta(model_id, family).group, scores


def load_lpp() -> pd.DataFrame:
    rows = []
    for model_id, group, scores in _iter_lpp_final("lpp_checkpoint_by_fedorenko_parcel.csv"):
        scores = scores.dropna(subset=["mean_pearson_left", "parcel", "layer"])
        scores = scores[scores["parcel"].isin(ROI_ORDER)]
        for (roi, layer), g in scores.groupby(["parcel", "layer"]):
            rows.append({"model_id": model_id, "group": group, "roi": roi,
                         "layer": layer, "score": float(g["mean_pearson_left"].mean())})
    return _add_layer_frac(pd.DataFrame(rows))


def load_ns_embed() -> pd.DataFrame:
    rows = []
    for model_id, group, scores in _iter_final(NS_SWEEP, "ns_fmri_checkpoint_layerwise.csv"):
        scores = scores[(scores["network"] == "LANG") & (scores["hemisphere"] == "L")]
        scores = scores.dropna(subset=["pearsonr", "froi", "layer"]).copy()
        scores["roi"] = scores["froi"].str.replace("LangL", "", regex=False)
        scores = scores[scores["roi"].isin(ROI_ORDER)]
        for (roi, layer), g in scores.groupby(["roi", "layer"]):
            rows.append({"model_id": model_id, "group": group, "roi": roi,
                         "layer": layer, "score": float(g["pearsonr"].mean())})
    return _add_layer_frac(pd.DataFrame(rows))


def load_ns_surprisal() -> pd.DataFrame:
    rows = []
    for model_id, group, scores in _iter_final(NS_LME_SWEEP, "ns_fmri_lme_by_roi.csv"):
        scores = scores[(scores["network"] == "LANG") & (scores["hemisphere"] == "L")]
        scores = scores.dropna(subset=["test_delta_pearsonr", "froi", "layer"]).copy()
        if "converged" in scores.columns:
            scores = scores[scores["converged"] == True]  # noqa: E712
        scores["roi"] = scores["froi"].str.replace("LangL", "", regex=False)
        scores = scores[scores["roi"].isin(ROI_ORDER)]
        for (roi, layer), g in scores.groupby(["roi", "layer"]):
            rows.append({"model_id": model_id, "group": group, "roi": roi,
                         "layer": layer, "score": float(g["test_delta_pearsonr"].mean())})
    return _add_layer_frac(pd.DataFrame(rows))


def load_podcast() -> pd.DataFrame:
    roi_map = pd.read_csv(PODCAST_ROI_MAP)
    roi_map = roi_map[roi_map["assigned"] & (roi_map["hemisphere"] == "L")]
    roi_map = roi_map[["subject", "channel", "region"]].rename(columns={"channel": "electrode"})
    rows = []
    for model_id, group, scores in _iter_final(PODCAST_SWEEP, "podcast_by_electrode_layerwise.csv"):
        scores = scores.dropna(subset=["pearsonr", "subject", "electrode", "layer"])
        m = scores.merge(roi_map, on=["subject", "electrode"], how="inner")
        m = m[m["region"].isin(ROI_ORDER)]
        for (roi, layer), g in m.groupby(["region", "layer"]):
            rows.append({"model_id": model_id, "group": group, "roi": roi,
                         "layer": layer, "score": float(g["pearsonr"].mean())})
    return _add_layer_frac(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _grouped_bar(ax, long_df: pd.DataFrame, score_label: str) -> None:
    """5 ROIs x 4 model classes; bar = mean best-layer score +/- SEM across
    that class's models."""
    x = np.arange(len(ROI_ORDER))
    w = 0.20
    groups_present = [g for g in GROUP_ORDER if (long_df["group"] == g).any()]
    for gi, grp in enumerate(groups_present):
        means, sems, ns = [], [], []
        for roi in ROI_ORDER:
            sub = long_df[(long_df["roi"] == roi) & (long_df["group"] == grp)]
            if sub.empty:
                means.append(np.nan); sems.append(0.0); ns.append(0); continue
            best = sub.groupby("model_id")["score"].max()
            means.append(float(best.mean()))
            sems.append(float(best.std(ddof=1) / np.sqrt(len(best))) if len(best) > 1 else 0.0)
            ns.append(len(best))
        offs = x + (gi - (len(groups_present) - 1) / 2) * w
        ax.bar(offs, means, w, yerr=sems, capsize=2, color=GROUP_COLORS[grp],
               edgecolor="black", linewidth=0.4, alpha=0.9,
               label=f"{GROUP_DISPLAY.get(grp, grp)} (n={max(ns) if ns else 0})")
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_ORDER)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_ylabel(f"{score_label}\n(best layer, mean $\\pm$ SEM over models)")
    ax.legend(fontsize=7.5, ncol=len(groups_present), loc="lower center",
              bbox_to_anchor=(0.5, 1.01), frameon=False)


def _page(pdf: PdfPages, long_df: pd.DataFrame, dataset_title: str, score_label: str) -> None:
    if long_df.empty:
        print(f"  [{dataset_title}] no data -- skipping page")
        return
    n_models = long_df["model_id"].nunique()
    fig = plt.figure(figsize=(15.5, 8.6))
    gs = gridspec.GridSpec(2, len(ROI_ORDER), height_ratios=[1.25, 1.0],
                           hspace=0.5, wspace=0.32, left=0.06, right=0.985, top=0.86, bottom=0.09)
    ax_bar = fig.add_subplot(gs[0, :])
    _grouped_bar(ax_bar, long_df, score_label)

    axes_lw = [fig.add_subplot(gs[1, i]) for i in range(len(ROI_ORDER))]
    for i, (ax, roi) in enumerate(zip(axes_lw, ROI_ORDER)):
        sub = long_df[long_df["roi"] == roi]
        draw_layer_profiles_interp(ax, sub, "score", score_label if i == 0 else "", roi)
        if i != 0:
            ax.set_ylabel("")
        if i != 0 and (leg := ax.get_legend()) is not None:
            leg.remove()
        ax.set_xlabel("Rel. layer depth" if i == 2 else "")

    fig.suptitle(f"{dataset_title}  --  ROI x model class  (n={n_models} models)",
                 fontsize=12, fontweight="semibold", x=0.06, ha="left")
    pdf.savefig(fig)
    plt.close(fig)
    print(f"  [{dataset_title}] page written ({n_models} models, "
          f"{long_df.groupby('group')['model_id'].nunique().to_dict()})")


def main() -> None:
    use_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== embedding track ===")
    emb_out = OUT_DIR / "roi_comparison_embedding.pdf"
    with PdfPages(emb_out) as pdf:
        _page(pdf, load_lpp(), "LPP fMRI (embedding)", "Pearson r")
        _page(pdf, load_ns_embed(), "Natural Stories fMRI (embedding)", "Pearson r")
        _page(pdf, load_podcast(), "Podcast iEEG (embedding)", "Pearson r")
    print(f"wrote {emb_out}")

    print("=== surprisal track ===")
    sur_out = OUT_DIR / "roi_comparison_surprisal.pdf"
    with PdfPages(sur_out) as pdf:
        _page(pdf, load_ns_surprisal(), "Natural Stories fMRI (surprisal / LME)",
              "held-out $\\Delta$ Pearson r")
    print(f"wrote {sur_out}")


if __name__ == "__main__":
    main()
