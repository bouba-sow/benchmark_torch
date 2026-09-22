#!/usr/bin/env python3
"""Natural Stories fMRI encoding checkpoint sweep.

Runs layerwise fMRI encoding (BOLD ~ HRF-convolved LLM embeddings, 26 fROIs)
across LLM families / training checkpoints.  Leave-one-story-out CV (8 stories)
grouped by story.  Himalaya GPU backend with CUDA-stream layer parallelism.

Two-stage pipeline (same as Podcast)
--------------------------------------
Stage 1 — GPU  (--activations_only)
    Runs the LLM forward pass for all 8 stories and all checkpoints.
    Writes .npz files to NS_FMRI_FEATURE_CACHE_DIR.  Cache hit = no model load.
Stage 2 — CPU or GPU  (default)
    Reads cached embeddings; computes HRF design matrices; runs ridge per subject.
    CPU: joblib parallel across layers.
    GPU: himalaya torch_cuda + CUDA streams.

Output structure
----------------
<output_dir>/
  <family>/
    <model_short>/
      ns_fmri_checkpoint_scores.csv
      ns_fmri_checkpoint_layerwise.csv
      ns_fmri_by_roi.csv
      ns_fmri_by_subject.csv
      ns_fmri_training_curve.png
      ns_fmri_layer_heatmap.png
    summary/
      all_models_ns_fmri_scores.csv
      all_models_ns_fmri_training_curves.png

Examples
--------
# Stage 1 — prefetch activations (GPU, one model):
python scripts/run/embedding/neural/ns_fmri.py \\
    --config configs/embedding/neural/ns_fmri.yaml \\
    --from-registry --family gpt2 --array-index 0 \\
    --device cuda --activations_only

# Stage 2 — regression (CPU):
python scripts/run/embedding/neural/ns_fmri.py \\
    --config configs/embedding/neural/ns_fmri.yaml \\
    --from-registry --family gpt2 --array-index 0 \\
    --device cpu --n_jobs 40

# GPU all-in-one:
python scripts/run/embedding/neural/ns_fmri.py \\
    --config configs/embedding/neural/ns_fmri.yaml \\
    --from-registry --family gpt2 --array-index 0 \\
    --device cuda --n_layer_workers 8

# Single model:
python scripts/run/embedding/neural/ns_fmri.py \\
    --config configs/embedding/neural/ns_fmri.yaml \\
    --models gpt2

# Summary after all SLURM jobs:
python scripts/run/embedding/neural/ns_fmri.py \\
    --config configs/embedding/neural/ns_fmri.yaml \\
    --summary_only --out_dir outputs/ns_fmri/embedding_sweep
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from neuroai.benchmarks.base import ModelConfig, RegressionConfig
from neuroai.benchmarks.natural_stories_fmri import (
    NsFmriBenchmarkConfig,
    NsFmriDataConfig,
    NsFmriEncoder,
    NsFmriEncoderResult,
)
from neuroai.datasets.natural_stories_fmri import NS_FMRI_DOCS
from neuroai.features.llm import clear_hf_model_cache
from neuroai.models.registry import (
    build_checkpoint_items,
    registry_model_configs,
    resolve_family_task,
    resolve_load_model_id,
)
from neuroai.sweep import (
    CheckpointItem,
    append_row,
    already_computed,
    load_existing_csv,
    safe_model_dir,
    free_gpu,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw)


def _resolve_alphas(raw: Any) -> list[float]:
    if raw in (None, "null", []):
        alphas = np.logspace(-2, 4, 6).tolist()
    else:
        alphas = [float(a) for a in raw]
    logger.info("Ridge alpha grid: %s", [f"{a:.3g}" for a in alphas])
    return alphas


def _model_output_dir(
    output_dir: Path,
    model_id: str,
    *,
    family: str | None = None,
    model_cfg: dict | None = None,
) -> Path:
    fam = family
    if model_cfg is not None:
        fam = model_cfg.get("registry_family") or model_cfg.get("family") or fam
    base = output_dir / fam if fam else output_dir
    return base / safe_model_dir(model_id)


def _build_benchmark_config(
    cfg: dict,
    *,
    model_id: str,
    revision: str,
    n_params: float,
    batch_size: int,
    device: str,
    output_dir: Path,
    feature_cache_dir: Path | None,
    n_jobs: int,
    n_layer_workers: int = 8,
    load_model_id: str | None = None,
    trust_remote_code: bool = False,
    save_word_predictions: bool | None = None,
) -> NsFmriBenchmarkConfig:
    ds = cfg["dataset"]
    reg = cfg.get("regression", {})
    bench = cfg.get("benchmark", {})

    data_cfg = NsFmriDataConfig(
        data_root=os.path.expandvars(str(ds["data_root"])),
        froi_table=str(ds.get("froi_table", "LANG_MD")),
        docs=ds.get("docs") or None,
        subjects=ds.get("subjects") or None,
        t_r=float(ds.get("t_r", 2.0)),
        word_duration=float(ds.get("word_duration", 0.1)),
        hrf_model=str(ds.get("hrf_model", "glover")),
        n_splits=int(ds.get("n_splits", 8)),
        feature_cache_dir=str(feature_cache_dir) if feature_cache_dir else None,
        max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        context_words=int(ds["context_words"]) if ds.get("context_words") else None,
    )

    hf_model_id = load_model_id or model_id
    model_cfg_obj = ModelConfig(
        model_name=hf_model_id,
        model_revision=revision if revision not in ("", "main") else None,
        device=device,
        batch_size=batch_size,
        n_params=n_params,
        trust_remote_code=trust_remote_code,
    )
    reg_cfg = RegressionConfig(
        n_splits=int(reg.get("n_splits", 8)),
        alpha=_resolve_alphas(reg.get("alpha")),
        scale=bool(reg.get("scale", True)),
        random_state=int(reg.get("random_state", 0)),
    )
    return NsFmriBenchmarkConfig(
        data=data_cfg,
        model=model_cfg_obj,
        regression=reg_cfg,
        output_dir=str(output_dir),
        layerwise=bool(bench.get("layerwise", True)),
        n_jobs=n_jobs,
        n_layer_workers=n_layer_workers,
        tokenizer_no_revision=bool(bench.get("tokenizer_no_revision", True)),
        save_word_predictions=(
            save_word_predictions if save_word_predictions is not None
            else bool(bench.get("save_word_predictions", True))
        ),
    )


# ---------------------------------------------------------------------------
# Stage 1: prefetch activations only
# ---------------------------------------------------------------------------

def run_activations_only(
    checkpoints: list[CheckpointItem],
    *,
    model_id: str,
    n_params: float,
    batch_size: int,
    cfg: dict,
    device: str,
    feature_cache_dir: Path | None,
    model_cfg: dict | None = None,
) -> None:
    """Extract and cache LLM embeddings for all 8 stories — no regression."""
    if feature_cache_dir is None:
        raise ValueError("--activations_only requires --feature_cache_dir or cache.features_dir")

    ds = cfg["dataset"]
    data_root = os.path.expandvars(str(ds["data_root"]))
    froi_table = str(ds.get("froi_table", "LANG_MD"))
    docs: list[str] = ds.get("docs") or NS_FMRI_DOCS
    max_length = int(ds["max_length"]) if ds.get("max_length") else None
    context_words = int(ds["context_words"]) if ds.get("context_words") else None
    bench = cfg.get("benchmark", {})
    tokenizer_no_revision = bool(bench.get("tokenizer_no_revision", True))

    from neuroai.datasets.natural_stories_fmri import load_canonical_words
    from neuroai.features.llm import (
        contextual_word_embeddings_hf_fulltext,
        contextual_word_embeddings_hf,
    )

    for ckpt in checkpoints:
        logger.info(
            "\n[%s] activations ── %s (step=%d) ──", model_id, ckpt.revision, ckpt.step
        )
        load_model_id = (
            resolve_load_model_id(model_cfg, ckpt.revision) if model_cfg else model_id
        )
        model_revision = ckpt.revision if ckpt.revision not in ("", "main") else None

        surp_dtype = "float32" if "gpt-bert" in load_model_id.lower() else None

        for doc_idx, doc in enumerate(docs):
            words_df = load_canonical_words(data_root, doc, froi_table)
            words = words_df["word"].tolist()
            logger.info(
                "  [%s %s] %d words — extracting all layers",
                ckpt.revision, doc, len(words),
            )
            try:
                emb = contextual_word_embeddings_hf_fulltext(
                    words,
                    model_name=load_model_id,
                    revision=model_revision,
                    tokenizer_no_revision=tokenizer_no_revision,
                    layer_index="all",
                    max_length=max_length,
                    cache_dir=str(feature_cache_dir),
                    section_id=doc_idx,
                    device=device,
                    n_params=int(n_params) if n_params else None,
                    trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
                    dtype=surp_dtype,
                )
            except Exception as exc:
                if "fast tokenizer" not in str(exc):
                    raise
                logger.warning("  Falling back to per-word extraction: %s", exc)
                emb = contextual_word_embeddings_hf(
                    words,
                    model_name=load_model_id,
                    revision=model_revision,
                    tokenizer_no_revision=tokenizer_no_revision,
                    layer_index="all",
                    max_length=max_length,
                    cache_dir=str(feature_cache_dir),
                    section_id=doc_idx,
                    device=device,
                    n_params=int(n_params) if n_params else None,
                    trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
                    dtype=surp_dtype,
                )
            logger.info("  [%s %s] done — shape %s", ckpt.revision, doc, emb.shape)

        free_gpu()


# ---------------------------------------------------------------------------
# Per-model sweep
# ---------------------------------------------------------------------------

def _summary_row(
    ckpt: CheckpointItem,
    roi_csv: Path,
    model_revision: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": ckpt.step,
        "revision": model_revision,
        "tokens_seen": ckpt.tokens_seen,
    }
    if not roi_csv.exists():
        return row

    try:
        df = pd.read_csv(roi_csv)
        df = df[df["revision"] == model_revision]
    except Exception:
        return row

    if df.empty:
        return row

    row["mean_best_spearmanr"]  = float(np.nanmean(df["best_spearmanr"]))
    row["mean_best_pearsonr"]   = float(np.nanmean(df["best_pearsonr"]))
    row["n_subjects"]           = int(df["subject"].nunique())
    row["n_rois"]               = int(len(df))

    for net, gdf in df.groupby("network"):
        row[f"{net}_best_spearmanr"] = float(np.nanmean(gdf["best_spearmanr"]))
        row[f"{net}_best_pearsonr"]  = float(np.nanmean(gdf["best_pearsonr"]))

    if "LANG_best_spearmanr" in row and "MD_best_spearmanr" in row:
        row["lang_minus_md_spearmanr"] = (
            row["LANG_best_spearmanr"] - row["MD_best_spearmanr"]
        )

    for froi, gdf in df.groupby("froi"):
        row[f"{froi}_spearmanr"] = float(np.nanmean(gdf["best_spearmanr"]))

    return row


def _layerwise_rows(ckpt: CheckpointItem, lw_csv: Path) -> list[dict]:
    if not lw_csv.exists():
        return []
    try:
        lw = pd.read_csv(lw_csv)
        lw = lw[lw["revision"] == ckpt.revision]
        if lw.empty:
            return []
        agg = (
            lw.groupby(["layer", "network"])
            .agg(mean_spearmanr=("spearmanr", "mean"), mean_pearsonr=("pearsonr", "mean"))
            .reset_index()
        )
        n_layers = agg["layer"].nunique()
        rows = []
        for _, r in agg.iterrows():
            li = int(r["layer"])
            rows.append({
                "step": ckpt.step,
                "revision": ckpt.revision,
                "tokens_seen": ckpt.tokens_seen,
                "layer": li,
                "layer_rel": li / max(1, n_layers - 1),
                "network": r["network"],
                "mean_spearmanr": float(r["mean_spearmanr"]),
                "mean_pearsonr": float(r["mean_pearsonr"]),
            })
        return rows
    except Exception:
        logger.warning("Layerwise row extraction failed", exc_info=True)
        return []


def run_model_sweep(
    checkpoints: list[CheckpointItem],
    *,
    model_id: str,
    n_params: float,
    batch_size: int,
    cfg: dict,
    output_dir: Path,
    device: str,
    feature_cache_dir: Path | None,
    n_jobs: int,
    n_layer_workers: int = 8,
    no_plots: bool = False,
    model_cfg: dict | None = None,
    save_word_predictions: bool | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / "ns_fmri_checkpoint_scores.csv"
    roi_csv    = output_dir / "ns_fmri_by_roi.csv"
    # NsFmriEncoder writes raw per (subject, froi, network, hemisphere, layer)
    # rows directly to "ns_fmri_checkpoint_layerwise.csv", and its own
    # aggregated per-(layer, network) summary to
    # "ns_fmri_checkpoint_layerwise_summary.csv" (see
    # neuroai/benchmarks/natural_stories_fmri.py::_save_checkpoint_summary).
    # This used to point at "ns_fmri_checkpoint_layerwise_detail.csv", a
    # filename nothing ever writes, which silently made the aggregation loop
    # below permanently a no-op.
    layer_csv  = output_dir / "ns_fmri_checkpoint_layerwise.csv"
    lw_csv     = layer_csv

    existing = load_existing_csv(scores_csv)
    summary_rows = existing.to_dict("records") if not existing.empty else []

    for ckpt in checkpoints:
        model_revision = ckpt.revision
        if already_computed(existing, model_revision):
            logger.info("[%s] %s already done — skipping", model_id, model_revision)
            continue

        logger.info(
            "\n[%s] NS-fMRI ── %s (step=%d, tokens≈%s) ──",
            model_id, model_revision, ckpt.step, ckpt.tokens_seen,
        )

        load_model_id = (
            resolve_load_model_id(model_cfg, ckpt.revision) if model_cfg else model_id
        )
        bench_cfg = _build_benchmark_config(
            cfg,
            model_id=model_id,
            load_model_id=load_model_id,
            revision=ckpt.revision,
            n_params=n_params,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            feature_cache_dir=feature_cache_dir,
            n_jobs=n_jobs,
            n_layer_workers=n_layer_workers,
            trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
            save_word_predictions=save_word_predictions,
        )

        try:
            NsFmriEncoder(bench_cfg).run()

            row = _summary_row(ckpt, roi_csv, model_revision)
            append_row(scores_csv, row)
            summary_rows.append(row)
            existing = load_existing_csv(scores_csv)
            # NOTE: no longer re-appending an aggregated "_layerwise_rows"
            # pass here -- it used to write into the *same* raw per-word
            # "ns_fmri_checkpoint_layerwise.csv" that NsFmriEncoder just
            # wrote, which would have corrupted it with mismatched-schema
            # summary rows once lw_csv pointed at the real (existing) file.
            # The aggregated per-(layer, network) view already exists at
            # "ns_fmri_checkpoint_layerwise_summary.csv", written directly by
            # NsFmriEncoder._save_checkpoint_summary.

        except Exception:
            logger.error("[%s] failed at %s", model_id, ckpt.revision, exc_info=True)
            traceback.print_exc()
        finally:
            clear_hf_model_cache()
            free_gpu()

    df = pd.DataFrame(summary_rows)
    if not df.empty and not no_plots:
        _plot_training_curve(df, output_dir, model_id)
        if layer_csv.exists():
            try:
                _plot_layer_heatmap(pd.read_csv(layer_csv), output_dir, model_id)
            except Exception:
                logger.warning("[%s] layer heatmap skipped", model_id, exc_info=True)
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_training_curve(df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if "step" not in df.columns or "mean_best_spearmanr" not in df.columns:
        return
    plot_df = df.sort_values("step")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(plot_df["step"], plot_df["mean_best_spearmanr"], "o-",
            label="Overall (LANG+MD)")
    if "LANG_best_spearmanr" in plot_df.columns:
        ax.plot(plot_df["step"], plot_df["LANG_best_spearmanr"], "s--",
                alpha=0.8, label="LANG")
    if "MD_best_spearmanr" in plot_df.columns:
        ax.plot(plot_df["step"], plot_df["MD_best_spearmanr"], "^--",
                alpha=0.8, label="MD")
    if "lang_minus_md_spearmanr" in plot_df.columns:
        ax.plot(plot_df["step"], plot_df["lang_minus_md_spearmanr"], "D:",
                alpha=0.6, label="LANG − MD")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean best-layer Spearman r")
    ax.set_title(f"NS fMRI — {model_id}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ns_fmri_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_heatmap(layer_df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if layer_df.empty or "step" not in layer_df.columns:
        return
    for net in layer_df["network"].unique():
        ndf = layer_df[layer_df["network"] == net]
        agg = ndf.groupby(["layer", "step"])["mean_spearmanr"].mean().reset_index()
        pivot = agg.pivot(index="layer", columns="step", values="mean_spearmanr")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.6), 5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.01, vmax=0.12)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Layer")
        ax.set_title(f"NS fMRI [{net}] layerwise — {model_id}")
        n_steps = pivot.shape[1]
        ax.set_xticks(range(n_steps))
        ax.set_xticklabels([str(int(s)) for s in pivot.columns], rotation=45, ha="right", fontsize=7)
        fig.colorbar(im, ax=ax, label="Mean Spearman r")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"ns_fmri_layer_heatmap_{net.lower()}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary across models
# ---------------------------------------------------------------------------

def run_summary(output_dir: Path, *, family: str | None = None) -> None:
    search_root = output_dir / family if family else output_dir
    summary_dir = (output_dir / family / "summary") if family else (output_dir / "summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    # ── collect all checkpoint_scores CSVs ──────────────────────────────────
    all_rows: list[pd.DataFrame] = []
    for csv_path in sorted(search_root.glob("**/ns_fmri_checkpoint_scores.csv")):
        model_id_str = csv_path.parent.name.replace("_", "/", 1)
        fam = csv_path.parent.parent.name if csv_path.parent.parent != search_root else None
        df = pd.read_csv(csv_path)
        df.insert(0, "model_id", model_id_str)
        if fam:
            df.insert(0, "family", fam)
        all_rows.append(df)

    if not all_rows:
        logger.warning("No ns_fmri_checkpoint_scores.csv found under %s", search_root)
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(summary_dir / "all_models_ns_fmri_scores.csv", index=False)
    logger.info("Combined CSV: %d rows, %d models", len(combined), combined["model_id"].nunique())

    title_fam = f" ({family})" if family else ""
    models = sorted(combined["model_id"].unique())
    cmap = plt.cm.get_cmap("tab20", len(models))
    color_map = {m: cmap(i) for i, m in enumerate(models)}

    # ── 1. Training curves: Overall / LANG / MD / LANG−MD ───────────────────
    lang_col = next((c for c in combined.columns if c in ("LANG_best_spearmanr", "lang_best_spearmanr")), None)
    md_col   = next((c for c in combined.columns if c in ("MD_best_spearmanr",   "md_best_spearmanr")),   None)
    lm_col   = next((c for c in combined.columns if "lang_minus_md" in c), None)

    has_networks = lang_col is not None and md_col is not None
    n_panels = 4 if (has_networks and lm_col) else (3 if has_networks else 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for mid, grp in combined.groupby("model_id"):
        g = grp.sort_values("step")
        c = color_map[mid]
        axes[0].plot(g["step"], g["mean_best_spearmanr"], "o-", color=c, label=mid)
        if has_networks:
            axes[1].plot(g["step"], g[lang_col], "s-", color=c, label=mid)
            axes[2].plot(g["step"], g[md_col],   "^-", color=c, label=mid)
        if has_networks and lm_col and n_panels == 4:
            axes[3].plot(g["step"], g[lm_col], "D-", color=c, label=mid)

    panel_titles = ["Overall (LANG+MD)", "LANG", "MD", "LANG − MD (specificity)"]
    for i, ax in enumerate(axes):
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean best-layer Spearman r")
        ax.set_title(panel_titles[i])
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=1, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.suptitle(f"NS fMRI — training curves{title_fam}", fontsize=11)
    fig.tight_layout()
    fig.savefig(summary_dir / "all_models_ns_fmri_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Scaling law: final checkpoint, n_params vs Spearman r ────────────
    # Use last step per model (highest token count or step number).
    final = (
        combined.sort_values("step")
        .groupby("model_id")
        .last()
        .reset_index()
    )
    if "n_params" in final.columns and final["n_params"].notna().any():
        fig, axes2 = plt.subplots(1, 2 if has_networks else 1,
                                  figsize=(10 if has_networks else 5, 4))
        if not has_networks:
            axes2 = [axes2]

        for ax, (col, label) in zip(
            axes2,
            [(  "mean_best_spearmanr", "Overall"),
             *([(lang_col, "LANG"), (md_col, "MD")] if has_networks else [])],
        ):
            for _, row in final.iterrows():
                if pd.notna(row.get("n_params")) and pd.notna(row.get(col)):
                    ax.scatter(row["n_params"], row[col],
                               color=color_map[row["model_id"]], s=60, zorder=3)
                    ax.annotate(row["model_id"].split("/")[-1],
                                (row["n_params"], row[col]),
                                fontsize=6, ha="left", va="bottom")
            ax.set_xscale("log")
            ax.set_xlabel("Model parameters")
            ax.set_ylabel("Best-layer Spearman r (final ckpt)")
            ax.set_title(f"Scaling law — {label}")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"NS fMRI — scaling law{title_fam}", fontsize=11)
        fig.tight_layout()
        fig.savefig(summary_dir / "ns_fmri_scaling_law.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 3. Per-fROI bar chart: final checkpoint, mean over models ────────────
    froi_cols = [c for c in final.columns if c.endswith("_spearmanr")
                 and c not in ("mean_best_spearmanr", lang_col, md_col, lm_col,
                               "lang_best_spearmanr", "md_best_spearmanr",
                               "lang_minus_md_spearmanr")]
    if froi_cols:
        froi_means = final[froi_cols].mean().sort_values(ascending=False)
        froi_labels = [c.replace("_spearmanr", "") for c in froi_means.index]
        colors_bar = [
            "#2166ac" if any(l.startswith("Lang") for l in [froi_labels[i]]) else "#d6604d"
            for i in range(len(froi_labels))
        ]
        fig, ax = plt.subplots(figsize=(max(8, len(froi_cols) * 0.55), 4))
        ax.bar(range(len(froi_means)), froi_means.values, color=colors_bar)
        ax.set_xticks(range(len(froi_means)))
        ax.set_xticklabels(froi_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean best-layer Spearman r")
        ax.set_title(f"NS fMRI — per-fROI profile (final ckpt, mean over models){title_fam}")
        ax.axhline(0, color="k", linewidth=0.7)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#2166ac", label="LANG"),
                            Patch(color="#d6604d", label="MD")],
                  fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(summary_dir / "ns_fmri_froi_profile.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 4. LANG vs MD scatter: one dot per model (final ckpt) ───────────────
    if has_networks and final[lang_col].notna().any():
        fig, ax = plt.subplots(figsize=(5, 5))
        for _, row in final.iterrows():
            lv = row.get(lang_col)
            mv = row.get(md_col)
            if pd.notna(lv) and pd.notna(mv):
                ax.scatter(mv, lv, color=color_map[row["model_id"]], s=60, zorder=3)
                ax.annotate(row["model_id"].split("/")[-1], (mv, lv),
                            fontsize=6, ha="left", va="bottom")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="LANG = MD")
        ax.set_xlabel("MD best-layer Spearman r")
        ax.set_ylabel("LANG best-layer Spearman r")
        ax.set_title(f"LANG vs MD specificity{title_fam}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(summary_dir / "ns_fmri_lang_vs_md.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Summary written to %s", summary_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/embedding/neural/ns_fmri.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--activations_only", action="store_true",
                   help="Stage 1: cache LLM embeddings for all stories, skip regression")
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n_jobs", type=int, default=20,
                   help="joblib workers for layer-parallel CPU ridge")
    p.add_argument("--n_layer_workers", type=int, default=8,
                   help="CUDA stream workers for layer-parallel GPU ridge")
    p.add_argument("--feature_cache_dir", default=None)
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--skip_summary", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--save_word_predictions", action=argparse.BooleanOptionalAction, default=None,
                   help="Save each held-out TR's true + out-of-fold predicted BOLD (last layer) "
                        "to ns_fmri_embed_word_predictions/<subject>.parquet (default: "
                        "benchmark.save_word_predictions in config, else True). On by default; "
                        "pass --no-save_word_predictions to disable.")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = Path(
        args.out_dir or cfg.get("output", {}).get("save_dir", "outputs/ns_fmri/embedding_sweep")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    cache_cfg = cfg.get("cache", {})
    feature_cache_dir: Path | None = None
    if args.feature_cache_dir:
        feature_cache_dir = Path(args.feature_cache_dir)
    elif cache_cfg.get("features_dir"):
        _p = os.path.expandvars(str(cache_cfg["features_dir"]))
        if "$" not in _p:
            feature_cache_dir = Path(_p)

    if args.summary_only:
        run_summary(out_root, family=args.family)
        return

    if args.from_registry:
        if args.family is None:
            p.error("--family is required with --from-registry")

        if args.array_index is not None:
            task = resolve_family_task(args.family, args.array_index)
            model_cfgs = [
                m for m in registry_model_configs(task.registry_family)
                if m["model_id"] == task.model_id
            ]
            if not model_cfgs:
                p.error(f"model_id {task.model_id!r} not found in registry family {task.registry_family!r}")
            model_cfg = model_cfgs[0]
            checkpoints = [
                CheckpointItem(rev, step, tokens)
                for rev, step, tokens in build_checkpoint_items(model_cfg)
            ]
            if args.dry_run:
                logger.info("DRY RUN: task=%s, %d checkpoints", task, len(checkpoints))
                return
            if args.activations_only:
                run_activations_only(
                    checkpoints,
                    model_id=task.model_id,
                    n_params=float(model_cfg.get("n_params", 0)),
                    batch_size=int(model_cfg.get("batch_size", 8)),
                    cfg=cfg,
                    device=args.device,
                    feature_cache_dir=feature_cache_dir,
                    model_cfg=model_cfg,
                )
            else:
                run_model_sweep(
                    checkpoints,
                    model_id=task.model_id,
                    n_params=float(model_cfg.get("n_params", 0)),
                    batch_size=int(model_cfg.get("batch_size", 8)),
                    cfg=cfg,
                    output_dir=_model_output_dir(
                        out_root, task.model_id, family=args.family, model_cfg=model_cfg
                    ),
                    device=args.device,
                    feature_cache_dir=feature_cache_dir,
                    n_jobs=args.n_jobs,
                    n_layer_workers=args.n_layer_workers,
                    no_plots=args.no_plots,
                    model_cfg=model_cfg,
                    save_word_predictions=args.save_word_predictions,
                )
            return

        for model_cfg in registry_model_configs(args.family):
            model_id = model_cfg["model_id"]
            checkpoints = [
                CheckpointItem(rev, step, tokens)
                for rev, step, tokens in build_checkpoint_items(model_cfg)
            ]
            output_dir = _model_output_dir(
                out_root, model_id, family=args.family, model_cfg=model_cfg
            )
            logger.info("=== [%s/%s] %d checkpoints ===", args.family, model_id, len(checkpoints))
            if args.dry_run:
                continue
            run_model_sweep(
                checkpoints,
                model_id=model_id,
                n_params=float(model_cfg.get("n_params", 0)),
                batch_size=int(model_cfg.get("batch_size", 8)),
                cfg=cfg,
                output_dir=output_dir,
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                n_jobs=args.n_jobs,
                n_layer_workers=args.n_layer_workers,
                no_plots=args.no_plots,
                model_cfg=model_cfg,
                save_word_predictions=args.save_word_predictions,
            )
            if not args.skip_summary:
                try:
                    run_summary(out_root, family=args.family)
                except Exception:
                    logger.warning("Summary failed", exc_info=True)

    elif args.models:
        for model_id in args.models:
            output_dir = out_root / safe_model_dir(model_id)
            checkpoints = [CheckpointItem(step=0, revision="main", tokens_seen=None)]
            run_model_sweep(
                checkpoints,
                model_id=model_id,
                n_params=0.0,
                batch_size=8,
                cfg=cfg,
                output_dir=output_dir,
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                n_jobs=args.n_jobs,
                n_layer_workers=args.n_layer_workers,
                no_plots=args.no_plots,
                save_word_predictions=args.save_word_predictions,
            )
    else:
        p.error("Specify --from-registry --family <family> or --models <model_id…>")


if __name__ == "__main__":
    main()
