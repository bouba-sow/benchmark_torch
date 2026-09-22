#!/usr/bin/env python3
"""Paper figure 4: layer-wise (relative-depth) profiles, small multiples --
one panel per individual dataset/measure. Two pages: surprisal features,
embedding features. No title text baked in (caption goes in the paper).

2026-09-08 REWRITE: replaces the old per-dataset-hardcoded pipeline, which
(a) forced the SURPRISAL page onto delta-Pearson-r instead of the DLL-
normalized score every other current figure uses, (b) pointed geco at a
stale root with no n_obs_test, (c) never had the GP/SP/OneStop/Pereira/ZuCo-
SR-TSR additions, and (d) still included the earlier N400 datasets (now excluded
project-wide for degenerate ceilings). Reuses DLL_DATASET_SPECS (surprisal)
and SECONDARY_SPEC/NEURAL_EMBED_LOADERS-style direct pulls (embedding) --
the same already-audited source-of-truth every other current paper figure
uses -- instead of a third parallel pipeline.

y-axis: "BehavScore" or "BrainScore" only, per panel's own domain (matches
figures/pareto_figures.py's convention exactly).

Output: reports/paper/layerwise_depth_profiles.pdf
"""
from __future__ import annotations

import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from neuroai.analysis.benchmark_common import (
    EXCLUDED_MODEL_IDS, GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, OUTPUTS_JEANZAY,
    REPO_ROOT, ceiling_for, draw_layer_profiles_interp, lookup_model_meta, read_lme_by_layer_csv,
)
from neuroai.analysis.paper_style import use_paper_style
from scripts.analysis.matched_scale_report import BEHAVIORAL_DATASETS, NEURAL_DATASETS

# analysis/matched_scale_regression.py used to export this; it's now a compact
# formula+table report with private dataset lists, so rebuild the map here
# from the canonical source instead.
MODALITY_MAP = {d: "behavioral" for d in BEHAVIORAL_DATASETS}
MODALITY_MAP.update({d: "neural" for d in NEURAL_DATASETS})
from scripts.figures.figure8_scale_matched_comparison import DLL_DATASET_SPECS, _model_id_from_dirname
from scripts.summarize.benchmark import LOADERS, SECONDARY_SPEC, _iter_model_score_files, _model_id_from_dir, _pick_final_row

OUT_DIR = REPO_ROOT / "reports" / "paper"

DISPLAY_LABEL: dict[str, str] = {
    "naturalstories_spr": "Nat. Stories SPR", "geco": "GECO FP", "geco_gp": "GECO GP",
    "meco": "MECO FP", "meco_gp": "MECO GP", "meco_sp": "MECO SP",
    "provo": "Provo FP", "provo_gp": "Provo GP", "provo_sp": "Provo SP",
    "ucl_reading": "UCL Reading", "zuco_reading": "ZuCo NR",
    "zuco_reading_sr": "ZuCo SR", "zuco_reading_tsr": "ZuCo TSR",
    "onestop_fp": "OneStop FP", "onestop_gp": "OneStop GP", "onestop_sp": "OneStop SP",
    "lpp_fmri": "LPP fMRI", "tuckute_fmri": "Tuckute fMRI",
    "fedorenko2016_ecog": "Fedorenko2016 ECoG", "pereira_fmri": "Pereira fMRI",
    "ns_fmri": "Nat. Stories fMRI", "podcast_ieeg": "Podcast iEEG",
}

# Datasets with no continuous per-layer axis (per-ROI or per-lag group_col,
# not a simple per-layer table) -- excluded from this figure, same as the
# old script.
NO_LAYER_AXIS = {"ns_fmri", "podcast_ieeg"}

# Neural datasets whose embedding-track LOADER already returns layer_df
# directly (pulled the same way build_paper_figure3 / the old figure4 did).
NEURAL_EMBED_LOADERS = ("lpp_fmri", "ns_fmri", "tuckute_fmri", "podcast_ieeg", "fedorenko2016_ecog")


def _surprisal_layer_df(slug: str) -> pd.DataFrame:
    """Every (model, layer) row's DLL-normalized score for one DLL_DATASET_SPECS
    slug -- delta_loglik_test / n_obs_test / ceiling, at every fitted layer
    (not just the best one)."""
    spec = DLL_DATASET_SPECS[slug]
    root = OUTPUTS_JEANZAY / spec["root"]
    ceiling = ceiling_for(slug)
    if ceiling is None or not root.is_dir():
        return pd.DataFrame()
    rows = []
    for fam_dir in sorted(root.iterdir()):
        if not fam_dir.is_dir() or fam_dir.name.endswith("_checkpoints"):
            continue
        for model_dir in sorted(fam_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            csvs = [p for p in model_dir.glob(spec["pattern"]) if not p.name.endswith(".lock")]
            if not csvs:
                continue
            mid = _model_id_from_dirname(model_dir.name)
            if mid in EXCLUDED_MODEL_IDS:
                continue
            try:
                df = read_lme_by_layer_csv(csvs[0])  # plain read_csv mis-parses mixed 26/29-column sweep files
            except Exception:
                continue
            need = {"layer", "delta_loglik_test", "n_obs_test"}
            if not need <= set(df.columns):
                continue
            if "revision" in df.columns:
                df = df[df["revision"].isin(["main", "step143000"])]
            if "converged" in df.columns:
                conv = df[df["converged"] == True]  # noqa: E712
                if not conv.empty:
                    df = conv
            df = df.dropna(subset=["layer", "delta_loglik_test", "n_obs_test"])
            df = df[df["n_obs_test"] > 0]
            if spec.get("group_col"):  # per-ROI etc. -- sum across units per layer first
                df = df.groupby("layer", as_index=False)[["delta_loglik_test", "n_obs_test"]].sum()
            if df.empty:
                continue
            n_layers = int(df["layer"].max()) + 1
            meta = lookup_model_meta(mid, fam_dir.name, n_params_fallback=None)
            for _, r in df.iterrows():
                rows.append({
                    "model_id": mid, "family": meta.family, "group": meta.group,
                    "layer_frac": r["layer"] / max(1, n_layers - 1),
                    "score": float(r["delta_loglik_test"]) / float(r["n_obs_test"]) / ceiling,
                })
    return pd.DataFrame(rows)


def _embed_layer_df_from_loader(embed_slug: str) -> pd.DataFrame:
    """Neural embedding datasets whose LOADER already returns a populated
    layer_df directly (lpp_fmri/tuckute_fmri/fedorenko2016_ecog -- read raw
    per-model-per-layer files, unlike the SECONDARY_SPEC-based behavioral
    loaders below, which only ever populate model_df)."""
    if embed_slug not in LOADERS:
        return pd.DataFrame()
    data = LOADERS[embed_slug]()
    ld = data.get("layer_df", pd.DataFrame())
    if ld.empty or "score" not in ld.columns or "layer_frac" not in ld.columns:
        return pd.DataFrame()
    ceiling = ceiling_for(embed_slug)
    if ceiling is None:
        return pd.DataFrame()
    ld = ld[~ld["model_id"].isin(EXCLUDED_MODEL_IDS)] if "model_id" in ld.columns else ld
    ld = ld.dropna(subset=["score", "layer_frac"]).copy()
    ld["score"] = pd.to_numeric(ld["score"], errors="coerce") / ceiling
    keep = [c for c in ("model_id", "family", "group", "layer_frac", "score") if c in ld.columns]
    return ld[keep]


# DLL_DATASET_SPECS/SECONDARY_SPEC naming mismatch: the FP measure is the
# bare slug in SECONDARY_SPEC ("onestop__embed") but "onestop_fp" in
# DLL_DATASET_SPECS. naturalstories_spr's embed sweep isn't in SECONDARY_SPEC
# at all (bespoke loader, own filename) -- handled separately below.
_SECONDARY_ALIAS = {"onestop_fp": "onestop"}


def _embed_layer_df_from_secondary(slug: str) -> pd.DataFrame:
    """Behavioral embedding datasets: SECONDARY_SPEC's loader only keeps the
    collapsed model_df (best/last layer), so read the raw per-model files
    directly here instead, keeping every layer."""
    embed_slug = f"{_SECONDARY_ALIAS.get(slug, slug)}__embed"
    if embed_slug not in SECONDARY_SPEC:
        return pd.DataFrame()
    root_rel, fname, score_col, _pick, _ppl = SECONDARY_SPEC[embed_slug]
    ceiling = ceiling_for(embed_slug)
    if ceiling is None:
        return pd.DataFrame()
    rows = []
    for family, model_dir, score_files in _iter_model_score_files(OUTPUTS_JEANZAY / root_rel, fname):
        mid = _model_id_from_dir(model_dir)
        if mid in EXCLUDED_MODEL_IDS:
            continue
        frames = [pd.read_csv(f) for f in score_files if f.exists()]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        if score_col not in df.columns or "layer" not in df.columns:
            continue
        final = _pick_final_row(df)
        key = "tokens_seen" if "tokens_seen" in df.columns else ("step" if "step" in df.columns else None)
        if key is not None and pd.notna(final.get(key)):
            df = df[df[key] == final[key]]
        df = df.dropna(subset=[score_col, "layer"])
        if df.empty:
            continue
        n_layers = int(df["layer"].max()) + 1
        meta = lookup_model_meta(mid, family, n_params_fallback=None)
        for _, r in df.iterrows():
            v = pd.to_numeric(r[score_col], errors="coerce")
            if pd.isna(v):
                continue
            rows.append({"model_id": mid, "family": meta.family, "group": meta.group,
                        "layer_frac": r["layer"] / max(1, n_layers - 1), "score": float(v) / ceiling})
    return pd.DataFrame(rows)


def _embed_layer_df_pereira() -> pd.DataFrame:
    """Pereira embedding: per-(layer, subject) rows -- average across
    subjects within each layer first (matches load_pereira_embed's
    best/last-picking convention), keeping every layer this time."""
    root = OUTPUTS_JEANZAY / "pereira/embedding_sweep"
    ceiling = ceiling_for("pereira_fmri__embed")
    if ceiling is None or not root.is_dir():
        return pd.DataFrame()
    rows = []
    for family, model_dir, score_files in _iter_model_score_files(root, "pereira_embed_by_layer.csv"):
        mid = _model_id_from_dir(model_dir)
        if mid in EXCLUDED_MODEL_IDS:
            continue
        frames = [pd.read_csv(f) for f in score_files if f.exists()]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        if "layer" not in df.columns or "pearsonr_mean" not in df.columns:
            continue
        if "revision" in df.columns:
            df = df[df["revision"] == "main"]
        per_layer = df.groupby("layer")["pearsonr_mean"].mean().dropna()
        if per_layer.empty:
            continue
        n_layers = int(per_layer.index.max()) + 1
        meta = lookup_model_meta(mid, family, n_params_fallback=None)
        for layer, v in per_layer.items():
            rows.append({"model_id": mid, "family": meta.family, "group": meta.group,
                        "layer_frac": layer / max(1, n_layers - 1), "score": float(v) / ceiling})
    return pd.DataFrame(rows)


def _embed_layer_df_naturalstories() -> pd.DataFrame:
    """naturalstories_spr's embed sweep -- own root/filename, not in
    SECONDARY_SPEC (see load_naturalstories_embed's docstring)."""
    ceiling = ceiling_for("naturalstories_spr__embed")
    if ceiling is None:
        return pd.DataFrame()
    rows = []
    for family, model_dir, score_files in _iter_model_score_files(
            OUTPUTS_JEANZAY / "naturalstories/embedding_sweep", "spr_checkpoint_layerwise.csv"):
        mid = _model_id_from_dir(model_dir)
        if mid in EXCLUDED_MODEL_IDS:
            continue
        frames = [pd.read_csv(f) for f in score_files if f.exists()]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        if "spearmanr" not in df.columns or "layer" not in df.columns:
            continue
        final = _pick_final_row(df)
        key = "tokens_seen" if "tokens_seen" in df.columns else ("step" if "step" in df.columns else None)
        if key is not None and pd.notna(final.get(key)):
            df = df[df[key] == final[key]]
        df = df.dropna(subset=["spearmanr", "layer"])
        if df.empty:
            continue
        n_layers = int(df["layer"].max()) + 1
        meta = lookup_model_meta(mid, family, n_params_fallback=None)
        for _, r in df.iterrows():
            v = pd.to_numeric(r["spearmanr"], errors="coerce")
            if pd.isna(v):
                continue
            rows.append({"model_id": mid, "family": meta.family, "group": meta.group,
                        "layer_frac": r["layer"] / max(1, n_layers - 1), "score": float(v) / ceiling})
    return pd.DataFrame(rows)


def _embed_layer_df(slug: str) -> pd.DataFrame:
    if slug in NEURAL_EMBED_LOADERS:
        return _embed_layer_df_from_loader(slug)
    if slug == "pereira_fmri":
        return _embed_layer_df_pereira()
    if slug == "naturalstories_spr":
        return _embed_layer_df_naturalstories()
    return _embed_layer_df_from_secondary(slug)


def _panels(mode: str) -> list[tuple[str, str, pd.DataFrame]]:
    """List of (display_label, domain, layer_df)."""
    loader = _surprisal_layer_df if mode == "surprisal" else _embed_layer_df
    panels = []
    for slug in DLL_DATASET_SPECS:
        if slug in NO_LAYER_AXIS:
            continue
        d = loader(slug)
        if d.empty:
            continue
        domain = MODALITY_MAP.get(slug, "neural")
        panels.append((DISPLAY_LABEL.get(slug, slug), domain, d))
    return panels


def _draw_page(fig, panels: list[tuple[str, str, pd.DataFrame]], ncols: int) -> None:
    n = len(panels)
    nrows = ceil(n / ncols)
    axes = fig.subplots(nrows, ncols)
    axes = np.atleast_1d(axes).reshape(-1)
    for i, (label, domain, d) in enumerate(panels):
        ax = axes[i]
        ylabel = "BehavScore" if domain == "behavioral" else "BrainScore"
        draw_layer_profiles_interp(ax, d, "score", ylabel, label)
        ax.set_title(label, fontweight="semibold", fontsize=9)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    for j in range(n, nrows * ncols):
        axes[j].set_axis_off()

    handles = [Line2D([], [], color=GROUP_COLORS[g], lw=2, label=GROUP_DISPLAY.get(g, g)) for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(GROUP_ORDER), frameon=False,
               bbox_to_anchor=(0.5, -0.01))


def main() -> None:
    use_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "layerwise_depth_profiles.pdf"
    with PdfPages(out) as pdf:
        for mode in ("surprisal", "embedding"):
            panels = _panels(mode)
            n = len(panels)
            # Keep to <=3 rows: widen columns (6-8) instead of stacking more rows.
            ncols = min(8, max(6, ceil(n / 3)))
            nrows = ceil(n / ncols)
            fig = plt.figure(figsize=(3.3 * ncols, 3.2 * nrows + 0.6))
            _draw_page(fig, panels, ncols)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            pdf.savefig(fig, dpi=600)
            plt.close(fig)
            print(f"wrote page: {mode} ({n} datasets, {nrows}x{ncols} grid)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
