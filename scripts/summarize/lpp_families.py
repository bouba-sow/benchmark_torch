#!/usr/bin/env python3
"""Summarize LPP checkpoint sweeps grouped by model family.

Reads ``<sweep_dir>/<family>/<model>/checkpoints/<revision>/lpp_checkpoint_scores.csv``
(layout produced when using ``--from-registry --family``) and writes per-family
plots/CSVs plus a global cross-family overview with all models on one plot ordered
by parameter count.

Usage::

    # All families with results on disk (auto-discovered):
    python scripts/summarize/lpp_families.py \\
        --sweep_dir outputs_jeanzay/lpp/embedding_sweep

    # Specific families only:
    python scripts/summarize/lpp_families.py \\
        --sweep_dir outputs_jeanzay/lpp/embedding_sweep --families gpt2 qwen3
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
import numpy as np
import pandas as pd

from neuroai.models.registry import LPP_FAMILIES, LPP_FAMILY_NAMES, model_spec_to_cfg
from neuroai.plotting.lpp import (
    plot_lpp_all_models_panel,
    plot_lpp_lateralization_panel,
    plot_lpp_layer_heatmap_grid,
    plot_lpp_layer_profiles,
    plot_lpp_layer_profiles_all_models,
    plot_lpp_score_by_model,
)

import importlib.util

_summ_mod = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "summarize_lpp",
        Path(__file__).resolve().parent / "lpp.py",
    )
)
importlib.util.spec_from_file_location(
    "summarize_lpp",
    Path(__file__).resolve().parent / "lpp.py",
).loader.exec_module(_summ_mod)  # type: ignore[union-attr]

_attach_metadata = _summ_mod._attach_metadata
_fit_scaling = _summ_mod._fit_scaling
_safe_model_dir = _summ_mod._safe_model_dir
_write_csv = _summ_mod._write_csv
_best_layer_per_roi_checkpoint = _summ_mod._best_layer_per_roi_checkpoint
_best_layer_per_parcel_checkpoint = _summ_mod._best_layer_per_parcel_checkpoint
_all_parcels_ordered = _summ_mod._all_parcels_ordered
_encoding_score = _summ_mod._encoding_score
_pick_global_best_step_layer = _summ_mod._pick_global_best_step_layer
_checkpoint_table = _summ_mod._checkpoint_table
plot_models_step_heatmap = _summ_mod.plot_models_step_heatmap
plot_per_model_training_facets = _summ_mod.plot_per_model_training_facets
plot_scaling_at_steps = _summ_mod.plot_scaling_at_steps
plot_scaling_final_with_fits = _summ_mod.plot_scaling_final_with_fits
plot_global_encoding_vs_step = _summ_mod.plot_global_encoding_vs_step
plot_encoding_heatmap = _summ_mod.plot_encoding_heatmap
plot_lateralization_by_step = _summ_mod.plot_lateralization_by_step
plot_parcel_curves_grid = _summ_mod.plot_parcel_curves_grid
plot_roi_step_heatmap = _summ_mod.plot_roi_step_heatmap
plot_roi_training_curves_by_step = _summ_mod.plot_roi_training_curves_by_step
plot_roi_layer_heatmaps = _summ_mod.plot_roi_layer_heatmaps
write_cross_model_parcel_analysis = _summ_mod.write_cross_model_parcel_analysis
plot_parcel_encoding_heatmap_models = _summ_mod.plot_parcel_encoding_heatmap_models
plot_parcel_mean_encoding_vs_models = _summ_mod.plot_parcel_mean_encoding_vs_models
plot_parcel_curves_grid_models = _summ_mod.plot_parcel_curves_grid_models
plot_parcel_overlay_models = _summ_mod.plot_parcel_overlay_models
plot_parcel_lateralization_heatmap_models = _summ_mod.plot_parcel_lateralization_heatmap_models


# ---------------------------------------------------------------------------
# Metadata for models present on disk but commented out of the registry
# ---------------------------------------------------------------------------

_EXTRA_META: dict[str, dict] = {
    "facebook/opt-30b": {
        "model_id": "facebook/opt-30b",
        "label": "OPT-30B",
        "n_params": 30.0e9,
        "n_layers": 49,
        "registry_family": "opt",
    },
    "EleutherAI/gpt-neo-125m": {
        "model_id": "EleutherAI/gpt-neo-125m",
        "label": "GPT-Neo-125M",
        "n_params": 125e6,
        "n_layers": 13,
        "registry_family": "gptneo",
    },
    "EleutherAI/gpt-neo-1.3B": {
        "model_id": "EleutherAI/gpt-neo-1.3B",
        "label": "GPT-Neo-1.3B",
        "n_params": 1.3e9,
        "n_layers": 25,
        "registry_family": "gptneo",
    },
}


def _n_params_from_label(label: str) -> float:
    """Best-effort n_params extraction from a model label string."""
    m = re.search(r"([\d.]+)\s*B\b", label, re.I)
    if m:
        return float(m.group(1)) * 1e9
    m = re.search(r"([\d.]+)\s*M\b", label, re.I)
    if m:
        return float(m.group(1)) * 1e6
    return np.nan


def _registry_metadata() -> dict[str, dict]:
    """Map model_id / safe_dir → metadata including registry_family."""
    meta: dict[str, dict] = {}
    for fam, specs in LPP_FAMILIES.items():
        for spec in specs:
            cfg = model_spec_to_cfg(spec, fam)
            mid = cfg["model_id"]
            row = {
                "model_id": mid,
                "label": cfg["label"],
                "n_params": float(cfg["n_params"]),
                "n_layers": int(cfg["n_layers"]),
                "registry_family": fam,
            }
            meta[mid] = row
            meta[_safe_model_dir(mid)] = row
    # Supplement with off-registry models that may have results on disk
    for mid, row in _EXTRA_META.items():
        if mid not in meta:
            meta[mid] = row
            meta[_safe_model_dir(mid)] = row
    return meta


# ---------------------------------------------------------------------------
# Discovery helpers (look inside checkpoints/<revision>/ subdirectories)
# ---------------------------------------------------------------------------

def _checkpoint_csvs(model_dir: Path) -> list[Path]:
    """Return all lpp_checkpoint_scores.csv files within checkpoints/ subdirs."""
    return sorted(model_dir.glob("checkpoints/*/lpp_checkpoint_scores.csv"))


def _checkpoint_lw_csvs(model_dir: Path) -> list[Path]:
    return sorted(model_dir.glob("checkpoints/*/lpp_checkpoint_layerwise.csv"))


def _checkpoint_roi_csvs(model_dir: Path) -> list[Path]:
    """Return all lpp_checkpoint_by_roi.csv files within checkpoints/ subdirs."""
    return sorted(model_dir.glob("checkpoints/*/lpp_checkpoint_by_roi.csv"))


def _checkpoint_parcel_csvs(model_dir: Path) -> list[Path]:
    """Return all lpp_checkpoint_by_parcel.csv files within checkpoints/ subdirs."""
    return sorted(model_dir.glob("checkpoints/*/lpp_checkpoint_by_parcel.csv"))


def _checkpoint_fedorenko_csvs(model_dir: Path) -> list[Path]:
    """Return all lpp_checkpoint_by_fedorenko_parcel.csv files within checkpoints/ subdirs."""
    return sorted(model_dir.glob("checkpoints/*/lpp_checkpoint_by_fedorenko_parcel.csv"))



def _has_scores(model_dir: Path) -> bool:
    """True if model_dir has results in either new (checkpoints/*/) or old (flat) format."""
    return bool(_checkpoint_csvs(model_dir)) or (model_dir / "lpp_checkpoint_scores.csv").is_file()


def _discover_families(sweep_dir: Path) -> list[str]:
    found: list[str] = []
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_dir() or p.name in ("summary", "corr_maps", "logs"):
            continue
        if any(_has_scores(sub) for sub in p.iterdir() if sub.is_dir()):
            found.append(p.name)
    return found


def _model_dirs_in_family(sweep_dir: Path, family: str) -> list[Path]:
    fam_dir = sweep_dir / family
    if not fam_dir.is_dir():
        return []
    return [p for p in sorted(fam_dir.iterdir()) if p.is_dir() and _has_scores(p)]


# ---------------------------------------------------------------------------
# Score collection
# ---------------------------------------------------------------------------

def _collect_flat_model_dirs(extra_dir: Path) -> list[tuple[Path, str]]:
    """Discover old-format model dirs. Returns (model_dir, family_name) pairs.

    Handles two layouts:
      <extra_dir>/<model>/lpp_checkpoint_scores.csv   → family = extra_dir.name
      <extra_dir>/<family>/<model>/lpp_checkpoint_scores.csv
    """
    results: list[tuple[Path, str]] = []
    for p in sorted(extra_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "lpp_checkpoint_scores.csv").is_file():
            results.append((p, extra_dir.name))
        else:
            # One level deeper: p is a family dir
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and (sub / "lpp_checkpoint_scores.csv").is_file():
                    results.append((sub, p.name))
    return results


def _load_flat_model_last_step(
    mdir: Path,
    meta: dict[str, dict],
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load old-format flat CSV, keep only the last step row per model."""
    scores = pd.read_csv(mdir / "lpp_checkpoint_scores.csv")
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Keep only the last training step
    last_step = int(scores["step"].max()) if "step" in scores.columns else 0
    scores = scores[scores["step"] == last_step].copy()
    scores = scores.assign(model_dir=mdir.name, model_folder=mdir.name, registry_family=family)

    lbl_0 = str(scores["model_label"].iloc[0]) if "model_label" in scores.columns else mdir.name
    mid = str(scores["model_id"].iloc[0]) if "model_id" in scores.columns else ""
    if mid not in meta and _safe_model_dir(mid) not in meta:
        meta[mid] = {
            "model_id": mid,
            "label": lbl_0,
            "n_params": _n_params_from_label(lbl_0),
            "n_layers": int(scores["n_layers"].iloc[0]) if "n_layers" in scores.columns else np.nan,
            "registry_family": family,
        }
        meta[_safe_model_dir(mid)] = meta[mid]

    lw = pd.DataFrame()
    lw_path = mdir / "lpp_checkpoint_layerwise.csv"
    if lw_path.is_file():
        lw = pd.read_csv(lw_path)
        if "step" in lw.columns:
            lw = lw[lw["step"] == last_step].copy()
    return scores, lw


def _collect_family_scores(
    sweep_dir: Path,
    family: str,
    meta: dict[str, dict],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_frames: list[pd.DataFrame] = []
    lw_by_label: dict[str, pd.DataFrame] = {}
    roi_frames: list[pd.DataFrame] = []
    parcel_frames: list[pd.DataFrame] = []
    fedorenko_frames: list[pd.DataFrame] = []

    for mdir in _model_dirs_in_family(sweep_dir, family):
        ckpt_frames: list[pd.DataFrame] = []
        new_csvs = _checkpoint_csvs(mdir)
        if new_csvs:
            # New format: one CSV per checkpoint subdirectory
            for csv_path in new_csvs:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    ckpt_frames.append(df)
        else:
            # Old flat format: single CSV with all steps — keep only the last step
            flat = mdir / "lpp_checkpoint_scores.csv"
            if flat.is_file():
                df = pd.read_csv(flat)
                if not df.empty:
                    last_step = int(df["step"].max()) if "step" in df.columns else 0
                    ckpt_frames.append(df[df["step"] == last_step].copy())
        if not ckpt_frames:
            continue
        scores = pd.concat(ckpt_frames, ignore_index=True)
        scores = scores.assign(
            model_dir=mdir.name,
            model_folder=mdir.name,
            registry_family=family,
        )

        # Fill in n_params from label when missing (off-registry models)
        if "model_label" in scores.columns:
            lbl_0 = str(scores["model_label"].iloc[0])
        else:
            lbl_0 = mdir.name

        mid = str(scores["model_id"].iloc[0]) if "model_id" in scores.columns else ""
        if mid not in meta and _safe_model_dir(mid) not in meta:
            # Prefer n_params embedded in the CSV (written by run_lpp_checkpoint_sweep);
            # fall back to label-parsing heuristic for older CSVs.
            n_params_csv = None
            if "n_params" in scores.columns:
                v = pd.to_numeric(scores["n_params"].iloc[0], errors="coerce")
                if np.isfinite(float(v)) and float(v) > 0:
                    n_params_csv = float(v)
            inferred = n_params_csv if n_params_csv is not None else _n_params_from_label(lbl_0)
            row = {
                "model_id": mid,
                "label": lbl_0,
                "n_params": inferred,
                "n_layers": int(scores["n_layers"].iloc[0]) if "n_layers" in scores.columns else np.nan,
                "registry_family": family,
            }
            meta[mid] = row
            meta[_safe_model_dir(mid)] = row

        score_frames.append(scores)

        # Layerwise CSVs — same new/old format logic
        lw_frames: list[pd.DataFrame] = []
        new_lw_csvs = _checkpoint_lw_csvs(mdir)
        if new_lw_csvs:
            for lw_path in new_lw_csvs:
                lw = pd.read_csv(lw_path)
                if not lw.empty:
                    lw_frames.append(lw)
        else:
            flat_lw = mdir / "lpp_checkpoint_layerwise.csv"
            if flat_lw.is_file():
                lw = pd.read_csv(flat_lw)
                if not lw.empty:
                    last_step_lw = int(scores["step"].max()) if "step" in scores.columns else 0
                    lw_frames.append(lw[lw["step"] == last_step_lw].copy() if "step" in lw.columns else lw)
        if lw_frames:
            lw_by_label[lbl_0] = pd.concat(lw_frames, ignore_index=True)
        
        last_step = int(scores["step"].max()) if "step" in scores.columns else 0

        # ROI CSVs — new format first, then flat fallback (last step only)
        roi_csv_list: list[pd.DataFrame] = []
        new_roi_csvs = _checkpoint_roi_csvs(mdir)
        if new_roi_csvs:
            for roi_path in new_roi_csvs:
                roi_df = pd.read_csv(roi_path)
                if not roi_df.empty:
                    roi_df = roi_df.assign(model_label=lbl_0, registry_family=family)
                    roi_csv_list.append(roi_df)
        else:
            flat_roi = mdir / "lpp_checkpoint_by_roi.csv"
            if flat_roi.is_file():
                roi_df = pd.read_csv(flat_roi)
                if not roi_df.empty:
                    roi_df = roi_df.assign(model_label=lbl_0, registry_family=family)
                    roi_csv_list.append(roi_df)
        if roi_csv_list:
            roi_frames.extend(roi_csv_list)

        # Parcel CSVs — new format first, then flat fallback (last step only)
        parcel_csv_list: list[pd.DataFrame] = []
        new_parcel_csvs = _checkpoint_parcel_csvs(mdir)
        if new_parcel_csvs:
            for parcel_path in new_parcel_csvs:
                parcel_df = pd.read_csv(parcel_path)
                if not parcel_df.empty:
                    parcel_df = parcel_df.assign(model_label=lbl_0, registry_family=family)
                    parcel_csv_list.append(parcel_df)
        else:
            flat_parcel = mdir / "lpp_checkpoint_by_parcel.csv"
            if flat_parcel.is_file():
                parcel_df = pd.read_csv(flat_parcel)
                if not parcel_df.empty:
                    parcel_df = parcel_df.assign(model_label=lbl_0, registry_family=family)
                    parcel_csv_list.append(parcel_df)
        if parcel_csv_list:
            parcel_frames.extend(parcel_csv_list)

        # Fedorenko parcel CSVs — new format first, then flat fallback
        fedorenko_csv_list: list[pd.DataFrame] = []
        new_fed_csvs = _checkpoint_fedorenko_csvs(mdir)
        if new_fed_csvs:
            for fed_path in new_fed_csvs:
                fed_df = pd.read_csv(fed_path)
                if not fed_df.empty:
                    fed_df = fed_df.assign(model_label=lbl_0, registry_family=family)
                    fedorenko_csv_list.append(fed_df)
        else:
            flat_fed = mdir / "lpp_checkpoint_by_fedorenko_parcel.csv"
            if flat_fed.is_file():
                fed_df = pd.read_csv(flat_fed)
                if not fed_df.empty:
                    fed_df = fed_df.assign(model_label=lbl_0, registry_family=family)
                    fedorenko_csv_list.append(fed_df)
        if fedorenko_csv_list:
            fedorenko_frames.extend(fedorenko_csv_list)

    if not score_frames:
        return pd.DataFrame(), lw_by_label, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    combined = _attach_metadata(pd.concat(score_frames, ignore_index=True), meta)
    roi_all = pd.concat(roi_frames, ignore_index=True) if roi_frames else pd.DataFrame()
    raw_parcel = pd.concat(parcel_frames, ignore_index=True) if parcel_frames else pd.DataFrame()
    parcel_all = _attach_metadata(raw_parcel, meta) if not raw_parcel.empty else pd.DataFrame()
    raw_fed = pd.concat(fedorenko_frames, ignore_index=True) if fedorenko_frames else pd.DataFrame()
    fedorenko_all = _attach_metadata(raw_fed, meta) if not raw_fed.empty else pd.DataFrame()
    return combined, lw_by_label, roi_all, parcel_all, fedorenko_all


# ---------------------------------------------------------------------------
# Per-family summary
# ---------------------------------------------------------------------------

def summarize_one_family(
    sweep_dir: Path,
    family: str,
    meta: dict[str, dict],
    reference_steps: list[int],
    *,
    out_root: Path,
    ppl_map: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[None, None, None]:
    out_dir = out_root / family
    out_dir.mkdir(parents=True, exist_ok=True)

    combined, lw_by_label, roi_all, parcel_all, fedorenko_all = _collect_family_scores(sweep_dir, family, meta)
    if combined.empty:
        print(f"  [{family}] no scores — skipping")
        return None, None, None

    sort_cols = [c for c in ("n_params_b", "model_label", "step") if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Collect by-fold data for layer-profile error bars.
    fold_frames: list[pd.DataFrame] = []
    for mdir in _model_dirs_in_family(sweep_dir, family):
        fold_csv = mdir / "lpp_checkpoint_by_fold.csv"
        if fold_csv.is_file():
            try:
                fdf = pd.read_csv(fold_csv)
                if not fdf.empty:
                    lbl = str(fdf["model_label"].iloc[0]) if "model_label" in fdf.columns else mdir.name
                    fdf = fdf.assign(model_label=lbl)
                    fold_frames.append(fdf)
            except Exception:
                pass
    fold_all = _attach_metadata(pd.concat(fold_frames, ignore_index=True), meta) if fold_frames else pd.DataFrame()

    _write_csv(combined, out_dir / "all_models_lpp_scores.csv")

    # x_col for training-curve plots: tokens_seen when multiple positive values exist,
    # otherwise step (single-checkpoint families have tokens_seen=0 which breaks log scale).
    has_positive_tokens = (
        "tokens_seen" in combined.columns and combined["tokens_seen"].gt(0).any()
    )
    n_unique_steps = combined["step"].nunique() if "step" in combined.columns else 1
    x_col = "tokens_seen" if has_positive_tokens else "step"

    # Training curve plot (meaningful only when multiple checkpoints exist).
    if n_unique_steps > 1:
        plot_lpp_all_models_panel(
            combined,
            x_col=x_col,
            save_path=out_dir / "all_models_training_curves.png",
            title=f"LPP training curves — {family}",
        )

    # Score-by-model bar chart always uses the final checkpoint so all models appear.
    final_step_for_bar = int(combined["step"].max()) if "step" in combined.columns else None
    bar_df = combined[combined["step"] == final_step_for_bar].copy() if final_step_for_bar is not None else combined
    plot_lpp_score_by_model(
        bar_df,
        save_path=out_dir / "all_models_score_by_model.png",
        title=f"LPP brain scores — {family} (step {final_step_for_bar})",
    )

    has_lat = "lateralization_index" in combined.columns and combined["lateralization_index"].notna().any()
    if has_lat:
        # For single-checkpoint families use n_params_b on x so models are spread out.
        lat_x = x_col if n_unique_steps > 1 else "n_params_b"
        plot_lpp_lateralization_panel(
            combined,
            x_col=lat_x,
            save_path=out_dir / "all_models_lateralization.png",
        )

    # Perplexity versions (generated whenever ppl_map has data for this family)
    if ppl_map:
        combined_ppl = _attach_perplexity(combined, ppl_map)
        if combined_ppl["perplexity"].notna().any():
            ppl_x = "inv_perplexity" if "inv_perplexity" in combined_ppl.columns else "perplexity"
            if n_unique_steps <= 1:
                plot_lpp_score_by_model(
                    combined_ppl.sort_values(ppl_x),
                    save_path=out_dir / "all_models_score_vs_perplexity.png",
                    title=f"LPP brain scores vs inverse perplexity — {family}",
                )
            if has_lat:
                plot_lpp_lateralization_panel(
                    combined_ppl,
                    x_col=ppl_x,
                    save_path=out_dir / "all_models_lateralization_vs_perplexity.png",
                    title=f"LPP lateralization vs inverse perplexity — {family}",
                )
    if lw_by_label:
        # Single-checkpoint: layer-profile line chart; multi-checkpoint: heatmap grid.
        if n_unique_steps > 1:
            plot_lpp_layer_heatmap_grid(
                lw_by_label,
                x_col=x_col,
                save_path=out_dir / "all_models_layer_heatmaps_grid.png",
            )
        else:
            plot_lpp_layer_profiles(
                lw_by_label,
                save_path=out_dir / "all_models_layer_profiles.png",
                title=f"Layer profiles — {family}",
            )

    # All-models layer profiles with error bars (always generated when fold data is available).
    if not fold_all.empty and "layer" in fold_all.columns:
        plot_lpp_layer_profiles_all_models(
            fold_all,
            save_path=out_dir / "all_models_layer_profiles_with_errorbars.png",
            title=f"Layer profiles — {family} (final checkpoint ± 1 std across sections)",
        )

    # ROI plots (if available)
    if not roi_all.empty and "roi" in roi_all.columns:
        roi_subdir = out_dir / "roi_plots"
        roi_subdir.mkdir(parents=True, exist_ok=True)
        _write_csv(roi_all, roi_subdir / "all_roi_data.csv")
        
        # Plot per model: ROI training curves
        for model_label in roi_all["model_label"].unique():
            roi_model = roi_all[roi_all["model_label"] == model_label].copy()
            if roi_model.empty:
                continue
            best_roi = _best_layer_per_roi_checkpoint(roi_model)
            if not best_roi.empty and "step" in best_roi.columns and has_positive_tokens:
                ckpt_tbl = _checkpoint_table(best_roi)
                if not ckpt_tbl.empty:
                    plot_roi_training_curves_by_step(
                        best_roi, ckpt_tbl,
                        roi_subdir / f"roi_curves_{model_label}.png",
                        str(model_label),
                    )
                    plot_roi_step_heatmap(
                        best_roi,
                        roi_subdir / f"roi_heatmap_{model_label}.png",
                        str(model_label),
                    )
    
    # Parcel plots (if available)
    if not parcel_all.empty and "parcel" in parcel_all.columns:
        parcel_subdir = out_dir / "parcel_plots"
        parcel_subdir.mkdir(parents=True, exist_ok=True)
        _write_csv(parcel_all, parcel_subdir / "all_parcel_data.csv")
        
        # Plot per model: parcel encoding curves
        for model_label in parcel_all["model_label"].unique():
            parcel_model = parcel_all[parcel_all["model_label"] == model_label].copy()
            if parcel_model.empty:
                continue
            best_parcel = _best_layer_per_parcel_checkpoint(parcel_model)
            if not best_parcel.empty and "step" in best_parcel.columns and has_positive_tokens:
                ckpt_tbl = _checkpoint_table(best_parcel)
                if not ckpt_tbl.empty:
                    # Global parcel encoding vs step
                    plot_global_encoding_vs_step(
                        best_parcel, ckpt_tbl,
                        parcel_subdir / f"parcel_encoding_vs_step_{model_label}.png",
                        str(model_label),
                    )
                    # Parcel encoding heatmap
                    plot_encoding_heatmap(
                        best_parcel,
                        parcel_subdir / f"parcel_encoding_heatmap_{model_label}.png",
                        str(model_label),
                    )
                    # Lateralization heatmap
                    plot_lateralization_by_step(
                        best_parcel,
                        parcel_subdir / f"parcel_lateralization_heatmap_{model_label}.png",
                        str(model_label),
                    )
                    # Parcel curves grid
                    parcels = _all_parcels_ordered(best_parcel)
                    if parcels:
                        plot_parcel_curves_grid(
                            best_parcel, ckpt_tbl,
                            parcel_subdir / f"parcel_curves_grid_{model_label}.png",
                            str(model_label), parcels,
                            title=f"{model_label} — {len(parcels)} parcels vs training step",
                            ncols=8 if len(parcels) > 24 else 6,
                        )

    n_steps = combined["step"].nunique() if "step" in combined.columns else 1
    if n_steps > 1:
        plot_models_step_heatmap(combined, out_dir / "models_x_step_heatmap.png")
        plot_per_model_training_facets(combined, out_dir / "models_training_facets.png")

        ref_steps = [s for s in reference_steps if s in set(combined["step"].tolist())]
        if not ref_steps:
            ref_steps = sorted(combined["step"].unique().astype(int).tolist())
        plot_scaling_at_steps(combined, ref_steps, out_dir / "model_scaling_at_reference_steps.png")

    final_step = int(combined["step"].max())
    final = combined[combined["step"] == final_step].copy()
    _write_csv(final.sort_values("n_params_b"), out_dir / f"model_scores_final_step{final_step}.csv")

    if "log10_n_params" in final.columns and "mean_pearson" in final.columns:
        fit_final = _fit_scaling(
            final["log10_n_params"].to_numpy(dtype=float),
            final["mean_pearson"].to_numpy(dtype=float),
        )
        fit_final["family"] = family
        fit_final["step"] = final_step
        (out_dir / "scaling_fit_results.json").write_text(
            json.dumps(fit_final, indent=2),
            encoding="utf-8",
        )
        plot_scaling_final_with_fits(
            final,
            fit_final,
            out_dir / "model_scaling_final_checkpoint.png",
        )

    # Cross-model parcel summary (all models in one set of plots)
    if not parcel_all.empty and "parcel" in parcel_all.columns:
        _write_parcel_family_summary(parcel_all, out_dir / "parcel_ho_summary")

    if not fedorenko_all.empty and "parcel" in fedorenko_all.columns:
        _write_parcel_family_summary(fedorenko_all, out_dir / "parcel_fedorenko_summary")

    n_models = combined["model_label"].nunique() if "model_label" in combined.columns else "?"
    print(f"  [{family}] {n_models} models, {len(combined)} score rows → {out_dir}")
    return combined, parcel_all, fedorenko_all


# ---------------------------------------------------------------------------
# Parcel cross-model helpers
# ---------------------------------------------------------------------------

def _parcel_snapshot_from_df(parcel_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-parcel best-snapshot DataFrame from already-collected parcel_all.

    For each model, picks the (step, layer) that maximises mean encoding across
    parcels, then returns one row per (model, parcel) at that snapshot.
    """
    if parcel_df.empty or "parcel" not in parcel_df.columns:
        return pd.DataFrame()
    snaps: list[pd.DataFrame] = []
    for model_lbl, grp in parcel_df.groupby("model_label", sort=False):
        step, layer = _pick_global_best_step_layer(grp)
        snap = grp[(grp["step"] == step) & (grp["layer"] == layer)].copy()
        snap["encoding"] = _encoding_score(snap)
        snap["best_step"] = step
        snap["best_layer"] = layer
        snaps.append(snap)
    if not snaps:
        return pd.DataFrame()
    return pd.concat(snaps, ignore_index=True)


def _write_parcel_family_summary(
    parcel_df: pd.DataFrame, out_dir: Path, *, width_scale: float = 1.0
) -> None:
    """Write combined (all models) parcel plots into out_dir."""
    snap = _parcel_snapshot_from_df(parcel_df)
    if snap.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(snap, out_dir / "parcel_at_best_checkpoint_all_models.csv")

    plot_parcel_mean_encoding_vs_models(snap, out_dir / "parcel_mean_encoding_vs_model_size.png",
                                        width_scale=width_scale)
    plot_parcel_encoding_heatmap_models(snap, out_dir / "parcel_encoding_heatmap.png",
                                        width_scale=width_scale)
    plot_parcel_lateralization_heatmap_models(snap, out_dir / "parcel_lateralization_heatmap.png",
                                              width_scale=width_scale)

    all_parcels = _all_parcels_ordered(snap)
    plot_parcel_curves_grid_models(
        snap, out_dir / "parcel_encoding_curves_all_grid.png", all_parcels,
        title=f"All {len(all_parcels)} parcels vs model size",
        ncols=8 if len(all_parcels) > 24 else 4,
        width_scale=width_scale,
    )
    plot_parcel_overlay_models(snap, out_dir / "parcel_encoding_curves_all_overlay.png", all_parcels,
                               width_scale=width_scale)

    top12 = (
        snap.groupby("parcel")["encoding"].mean()
        .sort_values(ascending=False).head(12).index.tolist()
    )
    if top12:
        plot_parcel_curves_grid_models(
            snap, out_dir / "parcel_encoding_curves_top12.png", top12,
            title="Top 12 parcels (mean across models) vs model size", ncols=4,
            width_scale=width_scale,
        )
    print(f"  → {out_dir}")


# ---------------------------------------------------------------------------
# Global summary: all families on one plot, ordered by n_params
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Perplexity helpers
# ---------------------------------------------------------------------------

def _load_perplexity_map(perplexity_dir: Path) -> dict[str, float]:
    """Map model_id → perplexity from individual perplexity.json files."""
    if not perplexity_dir.is_dir():
        return {}
    ppl_map: dict[str, float] = {}
    for p in perplexity_dir.rglob("perplexity.json"):
        try:
            d = json.loads(p.read_text())
            mid = d.get("model_id", "")
            ppl = d.get("perplexity")
            if mid and ppl is not None and np.isfinite(ppl):
                ppl_map[mid] = float(ppl)
        except Exception:
            pass
    return ppl_map


def _attach_perplexity(df: pd.DataFrame, ppl_map: dict[str, float]) -> pd.DataFrame:
    """Add 'perplexity', 'inv_perplexity', and 'log2_perplexity' columns from ppl_map."""
    if not ppl_map or "model_id" not in df.columns:
        return df
    df = df.copy()
    df["perplexity"] = df["model_id"].map(ppl_map)
    mask = df["perplexity"].notna() & df["perplexity"].gt(0)
    df["inv_perplexity"] = np.where(mask, 1.0 / df["perplexity"], np.nan)
    df["log2_perplexity"] = np.where(mask, np.log2(df["perplexity"]), np.nan)
    return df


def _ppl_label(ppl: float) -> str:
    """Format a perplexity value as 2^k."""
    import math
    if not np.isfinite(ppl) or ppl <= 0:
        return ""
    return f"$2^{{{math.log2(ppl):.1f}}}$"


# ---------------------------------------------------------------------------
# Color per registry family
# ---------------------------------------------------------------------------

_FAMILY_COLORS: dict[str, str] = {
    "gpt2": "#0072B2",
    "gptneo": "#56B4E9",
    "opt": "#E69F00",
    "mamba": "#009E73",
    "mamba2": "#66CC99",
    "pythia": "#CC79A7",
    "olmo2": "#D55E00",
    "qwen3": "#AA4499",
    "llama32": "#882255",
    "mistral": "#332288",
}


def _best_layer_row(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model_label, step), keep only the best-layer row = peak mean_pearson."""
    if "model_label" not in df.columns or "mean_pearson" not in df.columns:
        return df
    idx = df.groupby(["model_label", "step"])["mean_pearson"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def _plot_all_models_ordered(
    final_all: pd.DataFrame,
    out_path: Path,
) -> None:
    """Horizontal dot chart: one row per model, sorted by n_params, colored by family."""
    df = final_all.copy()
    df = df.dropna(subset=["mean_pearson"])

    # Sort models by n_params (ascending)
    if "n_params_b" in df.columns:
        order = df.drop_duplicates("model_label").sort_values("n_params_b")["model_label"].tolist()
    else:
        order = sorted(df["model_label"].unique().tolist())

    df["_order"] = df["model_label"].map({lbl: i for i, lbl in enumerate(order)})

    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.45)))

    families_plotted: set[str] = set()
    for _, row in df.sort_values("_order").iterrows():
        fam = str(row.get("registry_family", ""))
        color = _FAMILY_COLORS.get(fam, "#555555")
        label = fam if fam not in families_plotted else "_nolegend_"
        families_plotted.add(fam)
        ax.barh(
            row["_order"],
            row["mean_pearson"],
            color=color,
            alpha=0.85,
            label=label,
            height=0.6,
        )
        ax.text(
            row["mean_pearson"] + 0.001,
            row["_order"],
            f"{row['mean_pearson']:.3f}",
            va="center",
            fontsize=7,
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)

    # Annotate n_params on the left margin
    x_min = ax.get_xlim()[0]
    for i, lbl in enumerate(order):
        sub = df[df["model_label"] == lbl]
        if sub.empty:
            continue
        n_b = sub["n_params_b"].iloc[0]
        if np.isfinite(n_b):
            param_str = f"{n_b:.1f}B" if n_b >= 1 else f"{n_b*1000:.0f}M"
            ax.text(
                x_min - 0.003,
                i,
                param_str,
                va="center",
                ha="right",
                fontsize=7,
                color="#444444",
            )

    ax.set_xlabel("Mean Pearson r (best layer, final checkpoint)")
    ax.set_title("LPP encoding — all models (sorted by #parameters)")
    ax.legend(
        title="Family",
        loc="lower right",
        fontsize=8,
        title_fontsize=8,
        ncol=2,
    )
    ax.invert_yaxis()  # largest n_params at top
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_scaling_all_families(
    final_all: pd.DataFrame,
    out_path: Path,
) -> None:
    """Log-scale scatter: brain score vs n_params, one point per model, colored by family."""
    df = final_all.dropna(subset=["n_params_b", "mean_pearson"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for fam in sorted(df["registry_family"].unique()):
        sub = df[df["registry_family"] == fam].sort_values("n_params_b")
        color = _FAMILY_COLORS.get(fam, "#555555")
        ax.scatter(sub["n_params_b"], sub["mean_pearson"],
                   label=fam, color=color, s=70, alpha=0.9, zorder=3)
        # Connect points within family with a thin line
        if len(sub) > 1:
            ax.plot(sub["n_params_b"], sub["mean_pearson"],
                    color=color, linewidth=0.8, alpha=0.5)
        # Label each point with parameter count (e.g. "1B", "125M")
        for _, r in sub.iterrows():
            n_b = r.get("n_params_b", np.nan)
            if np.isfinite(n_b) and n_b > 0:
                ann = f"{n_b:.1f}B" if n_b >= 1 else f"{n_b * 1000:.0f}M"
            else:
                ann = str(r.get("model_label", ""))
            ax.annotate(
                ann,
                (r["n_params_b"], r["mean_pearson"]),
                textcoords="offset points",
                xytext=(4, 2),
                fontsize=6,
                color="#333333",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions)")
    ax.set_ylabel("Mean Pearson r (best layer, final checkpoint)")
    ax.set_title("LPP encoding — scaling across all families")
    ax.legend(loc="upper left", fontsize=8, ncol=2, title="Family")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_all_models_ordered_vs_perplexity(
    final_all: pd.DataFrame,
    out_path: Path,
) -> None:
    """Same horizontal bar chart as _plot_all_models_ordered but sorted by perplexity."""
    import math
    df = final_all.dropna(subset=["mean_pearson", "perplexity"])
    if df.empty:
        return

    sort_col = "inv_perplexity" if "inv_perplexity" in df.columns else "perplexity"
    ascending = sort_col == "perplexity"
    order = df.drop_duplicates("model_label").sort_values(sort_col, ascending=ascending)["model_label"].tolist()
    df["_order"] = df["model_label"].map({lbl: i for i, lbl in enumerate(order)})

    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.45)))

    families_plotted: set[str] = set()
    for _, row in df.sort_values("_order").iterrows():
        fam = str(row.get("registry_family", ""))
        color = _FAMILY_COLORS.get(fam, "#555555")
        label = fam if fam not in families_plotted else "_nolegend_"
        families_plotted.add(fam)
        ax.barh(row["_order"], row["mean_pearson"], color=color, alpha=0.85, label=label, height=0.6)
        ax.text(row["mean_pearson"] + 0.001, row["_order"],
                f"{row['mean_pearson']:.3f}", va="center", fontsize=7)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)

    x_min = ax.get_xlim()[0]
    for i, lbl in enumerate(order):
        sub = df[df["model_label"] == lbl]
        if sub.empty:
            continue
        ppl = sub["perplexity"].iloc[0]
        if np.isfinite(ppl) and ppl > 0:
            ax.text(x_min - 0.003, i, _ppl_label(ppl),
                    va="center", ha="right", fontsize=7, color="#444444")

    ax.set_xlabel("Mean Pearson r (best layer, final checkpoint)")
    ax.set_title("LPP encoding — all models (sorted by perplexity)")
    ax.legend(title="Family", loc="lower right", fontsize=8, title_fontsize=8, ncol=2)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def _size_label(n_params_b: float) -> str:
    """Format parameter count as a short size string: '1B', '125M', etc."""
    if not np.isfinite(n_params_b) or n_params_b <= 0:
        return ""
    return f"{n_params_b:.1f}B" if n_params_b >= 1 else f"{n_params_b * 1000:.0f}M"


def _plot_scaling_vs_perplexity(
    final_all: pd.DataFrame,
    out_path: Path,
) -> None:
    """Scatter: brain score vs perplexity (log₂ x-axis, 2^k labels), one point per model.

    Dot labels show model size (e.g. '1B', '125M').
    """
    import math
    import matplotlib.ticker as mticker

    x_col = "inv_perplexity" if "inv_perplexity" in final_all.columns else "perplexity"
    df = final_all.dropna(subset=[x_col, "mean_pearson"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for fam in sorted(df["registry_family"].unique()):
        sub = df[df["registry_family"] == fam].sort_values(x_col)
        color = _FAMILY_COLORS.get(fam, "#555555")
        ax.scatter(sub[x_col], sub["mean_pearson"],
                   label=fam, color=color, s=70, alpha=0.9, zorder=3)
        if len(sub) > 1:
            ax.plot(sub[x_col], sub["mean_pearson"],
                    color=color, linewidth=0.8, alpha=0.5)
        for _, r in sub.iterrows():
            n_b = r.get("n_params_b", np.nan)
            ann = _size_label(n_b) if np.isfinite(n_b) and n_b > 0 else str(r.get("model_label", ""))
            ax.annotate(ann, (r[x_col], r["mean_pearson"]),
                        textcoords="offset points", xytext=(4, 2), fontsize=6, color="#333333")

    ax.set_xscale("log", base=2)

    # Axis limits with 2^pad padding; ticks at integer powers of 2.
    x_vals = df[x_col].dropna()
    if not x_vals.empty:
        pad = 0.15
        log2_min = math.log2(x_vals.min())
        log2_max = math.log2(x_vals.max())
        ax.set_xlim(2 ** (log2_min - pad), 2 ** (log2_max + pad))
        ticks = [2 ** k for k in range(
            math.ceil(log2_min - pad),
            math.floor(log2_max + pad) + 1,
        )]
        ax.set_xticks(ticks, minor=False)
        ax.set_xticks([], minor=True)
        if x_col == "inv_perplexity":
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"$2^{{-{round(-math.log2(x))}}}$" if x > 0 else "")
            )
        else:
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"$2^{{{round(math.log2(x))}}}$" if x > 0 else "")
            )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.set_xlabel("Inverse perplexity (log₂ scale, higher = better language model)")
    ax.set_ylabel("Mean Pearson r (best layer, final checkpoint)")
    ax.set_title("LPP encoding — brain score vs inverse perplexity")
    ax.legend(loc="upper right", fontsize=8, ncol=2, title="Family")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def summarize_global(
    family_frames: dict[str, pd.DataFrame],
    *,
    out_root: Path,
    parcel_frames: dict[str, pd.DataFrame] | None = None,
    fedorenko_frames: dict[str, pd.DataFrame] | None = None,
    ppl_map: dict[str, float] | None = None,
    fold_all: pd.DataFrame | None = None,
) -> None:
    if not family_frames:
        return
    out_dir = out_root / "all_families"
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for fam, df in family_frames.items():
        snap = df.copy()
        if "registry_family" not in snap.columns:
            snap["registry_family"] = fam
        parts.append(snap)
    combined = pd.concat(parts, ignore_index=True)
    _write_csv(combined, out_dir / "all_families_lpp_scores.csv")

    # One row per model at its final checkpoint (best layer)
    final_rows: list[pd.DataFrame] = []
    for fam, df in family_frames.items():
        if df.empty:
            continue
        final_step = int(df["step"].max())
        final = df[df["step"] == final_step].copy()
        if "registry_family" not in final.columns:
            final["registry_family"] = fam
        final_rows.append(final)
    if not final_rows:
        return

    final_all = pd.concat(final_rows, ignore_index=True)

    # Keep only the best-layer row per (model, step) in case per-layer rows are present
    final_all = _best_layer_row(final_all)
    _write_csv(
        final_all.sort_values("n_params_b") if "n_params_b" in final_all.columns else final_all,
        out_dir / "all_families_final_checkpoint.csv",
    )

    # 1. Horizontal bar chart: all models sorted by n_params
    _plot_all_models_ordered(final_all, out_dir / "all_models_ordered_by_params.png")

    # 2. Log-scale scatter: scaling across families
    _plot_scaling_all_families(final_all, out_dir / "all_families_scaling_final.png")

    # Perplexity-based versions (only when perplexity data is available)
    if ppl_map:
        final_ppl = _attach_perplexity(final_all, ppl_map)
        if final_ppl["perplexity"].notna().any():
            _plot_all_models_ordered_vs_perplexity(
                final_ppl, out_dir / "all_models_ordered_by_perplexity.png"
            )
            _plot_scaling_vs_perplexity(
                final_ppl, out_dir / "all_families_scaling_vs_perplexity.png"
            )

    # Per-family scaling fits
    fit_rows = []
    for fam, df in family_frames.items():
        if df.empty or "log10_n_params" not in df.columns:
            continue
        final_step = int(df["step"].max())
        final = df[df["step"] == final_step]
        if final["log10_n_params"].notna().sum() < 2:
            continue
        fit = _fit_scaling(
            final["log10_n_params"].to_numpy(dtype=float),
            final["mean_pearson"].to_numpy(dtype=float),
        )
        fit["family"] = fam
        fit_rows.append(fit)
    if fit_rows:
        (out_dir / "scaling_fit_by_family.json").write_text(
            json.dumps(fit_rows, indent=2),
            encoding="utf-8",
        )

    # Layer profiles with error bars — all models across all families, final checkpoint only.
    if fold_all is not None and not fold_all.empty and "layer" in fold_all.columns:
        plot_lpp_layer_profiles_all_models(
            fold_all,
            save_path=out_dir / "all_families_layer_profiles_with_errorbars.png",
            title="Layer profiles — all families (final checkpoint ± 1 std across sections)",
        )

    # Cross-family parcel summary: all models from all families in one set of plots.
    # Double the width so model labels on the x-axis are readable.
    if parcel_frames:
        all_parcel_parts = [df for df in parcel_frames.values() if not df.empty]
        if all_parcel_parts:
            all_parcel_df = pd.concat(all_parcel_parts, ignore_index=True)
            _write_parcel_family_summary(all_parcel_df, out_dir / "parcel_ho_all_families", width_scale=2.0)

    if fedorenko_frames:
        all_fed_parts = [df for df in fedorenko_frames.values() if not df.empty]
        if all_fed_parts:
            all_fed_df = pd.concat(all_fed_parts, ignore_index=True)
            _write_parcel_family_summary(all_fed_df, out_dir / "parcel_fedorenko_all_families", width_scale=2.0)

    print(f"Global summary → {out_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep_dir",
        type=Path,
        default=Path("outputs_jeanzay/lpp/embedding_sweep"),
        help="Directory where checkpoint sweep results are stored (read-only)",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/lpp/embedding_sweep"),
        help="Root directory for all summary outputs (default: outputs/lpp/embedding_sweep)",
    )
    p.add_argument(
        "--extra_dirs",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Additional directories with old-format results "
            "(flat <model>/lpp_checkpoint_scores.csv, multiple steps). "
            "Only the last step per model is included. "
            "Example: --extra_dirs outputs/lpp/embedding_sweep"
        ),
    )
    p.add_argument(
        "--perplexity_dir",
        type=Path,
        default=Path("outputs_jeanzay/lpp/perplexity"),
        help="Directory with perplexity.json files (default: outputs_jeanzay/lpp/perplexity)",
    )
    p.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Subset of families (default: auto-discover under sweep_dir)",
    )
    p.add_argument(
        "--reference_steps",
        type=int,
        nargs="+",
        default=[0, 8000, 32000, 143000],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    out_root = args.out_dir.resolve()
    if not sweep_dir.is_dir():
        sys.exit(f"Missing sweep directory: {sweep_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    meta = _registry_metadata()

    ppl_map = _load_perplexity_map(args.perplexity_dir.resolve())
    if ppl_map:
        print(f"Loaded perplexity for {len(ppl_map)} models from {args.perplexity_dir}")
    else:
        print(f"No perplexity data found under {args.perplexity_dir} — skipping perplexity plots")

    if args.families:
        families = list(args.families)
    else:
        families = _discover_families(sweep_dir)
        if not families:
            families = [f for f in LPP_FAMILY_NAMES if _model_dirs_in_family(sweep_dir, f)]

    if not families:
        sys.exit(f"No family results under {sweep_dir}")

    print(f"Summarizing {len(families)} families under {sweep_dir}: {families}")
    print(f"Outputs → {out_root}")
    family_frames: dict[str, pd.DataFrame] = {}
    parcel_frames: dict[str, pd.DataFrame] = {}
    fedorenko_frames: dict[str, pd.DataFrame] = {}
    fold_frames_global: list[pd.DataFrame] = []
    for fam in families:
        df, parcel_df, fedorenko_df = summarize_one_family(
            sweep_dir, fam, meta, args.reference_steps, out_root=out_root, ppl_map=ppl_map
        )
        if df is not None:
            family_frames[fam] = df
        if parcel_df is not None and not parcel_df.empty:
            parcel_frames[fam] = parcel_df
        if fedorenko_df is not None and not fedorenko_df.empty:
            fedorenko_frames[fam] = fedorenko_df
        # Collect by-fold data for the cross-family layer-profile plot.
        for mdir in _model_dirs_in_family(sweep_dir, fam):
            fold_csv = mdir / "lpp_checkpoint_by_fold.csv"
            if fold_csv.is_file():
                try:
                    fdf = pd.read_csv(fold_csv)
                    if not fdf.empty:
                        lbl = str(fdf["model_label"].iloc[0]) if "model_label" in fdf.columns else mdir.name
                        fdf = fdf.assign(model_label=lbl, registry_family=fam)
                        fold_frames_global.append(fdf)
                except Exception:
                    pass
    fold_all_global = (
        _attach_metadata(pd.concat(fold_frames_global, ignore_index=True), meta)
        if fold_frames_global else pd.DataFrame()
    )

    # Load old-format models from extra_dirs (e.g. outputs/lpp/embedding_sweep for Pythia).
    # Each model directory has a flat lpp_checkpoint_scores.csv with multiple steps;
    # only the last step is kept so they appear as a single point in the global plot.
    extra_dirs = [d.resolve() for d in (args.extra_dirs or []) if d.is_dir()]
    for extra_dir in extra_dirs:
        flat_pairs = _collect_flat_model_dirs(extra_dir)
        if not flat_pairs:
            continue
        extra_rows: list[pd.DataFrame] = []
        for mdir, fam_guess in flat_pairs:
            # Prefer registry family over guessed name
            try:
                probe = pd.read_csv(mdir / "lpp_checkpoint_scores.csv", nrows=1)
                mid = str(probe["model_id"].iloc[0]) if "model_id" in probe.columns else ""
            except Exception:
                mid = ""
            fam = meta.get(mid, meta.get(_safe_model_dir(mid), {})).get("registry_family", fam_guess)
            scores, _ = _load_flat_model_last_step(mdir, meta, fam)
            if not scores.empty:
                extra_rows.append(scores)
                print(f"  [extra] loaded {mdir.name} (family={fam}, "
                      f"step={int(scores['step'].iloc[0])})")
        if extra_rows:
            extra_df = _attach_metadata(pd.concat(extra_rows, ignore_index=True), meta)
            for fam in extra_df["registry_family"].unique():
                chunk = extra_df[extra_df["registry_family"] == fam].copy()
                if fam in family_frames:
                    family_frames[fam] = pd.concat(
                        [family_frames[fam], chunk], ignore_index=True
                    )
                else:
                    family_frames[fam] = chunk

    summarize_global(
        family_frames,
        out_root=out_root / "summary",
        parcel_frames=parcel_frames,
        fedorenko_frames=fedorenko_frames,
        ppl_map=ppl_map,
        fold_all=fold_all_global,
    )
    print("Done.")


if __name__ == "__main__":
    main()
