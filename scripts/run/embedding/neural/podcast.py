#!/usr/bin/env python3
"""Podcast ECoG encoding checkpoint sweep.

Runs layerwise high-gamma encoding (activations only, no surprisal) across all
LPP model families / training checkpoints for the Podcast iEEG dataset.

Output structure
----------------
<output_dir>/
  <family>/                    # e.g. gpt2, pythia, mamba
    <model_short>/
      podcast_checkpoint_scores.csv       # per-checkpoint summary (best layer)
      podcast_checkpoint_layerwise.csv    # per-layer Pearson r per checkpoint
      podcast_by_electrode.csv            # per-electrode scores
      podcast_by_subject.csv              # per-subject mean
      podcast_by_region.csv               # per-region mean (electrode arrays)
      podcast_training_curve.png
      podcast_layer_heatmap.png
      podcast_brain_map.png               # electrode scatter coloured by score
    summary/
      all_models_podcast_scores.csv
      all_models_podcast_training_curves.png

Examples
--------
# Single array task (Jean Zay):
python scripts/run/embedding/neural/podcast.py \\
    --config configs/embedding/neural/podcast.yaml \\
    --from-registry --family gpt2 --array-index 0 --device cuda

# Full family locally:
python scripts/run/embedding/neural/podcast.py \\
    --config configs/embedding/neural/podcast.yaml \\
    --from-registry --family pythia --device cuda

# Summary after all SLURM jobs:
python scripts/run/embedding/neural/podcast.py \\
    --config configs/embedding/neural/podcast.yaml \\
    --summary_only --out_dir outputs/podcast/checkpoint_sweep
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
from neuroai.benchmarks.podcast import (
    PodcastBenchmarkConfig,
    PodcastDataConfig,
    PodcastEcogEncoder,
    PodcastEncoderResult,
)
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
    free_gpu,
    load_existing_csv,
    safe_model_dir,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw)


def _resolve_alphas(raw: Any) -> list[float]:
    if raw in (None, "null", []):
        return np.logspace(3.5, 8, 5).tolist()
    if isinstance(raw, str):
        # Allow a numpy expression directly in the YAML, e.g.:
        #   alpha: "np.logspace(2, 6, 8)"
        # Restricted eval: only `np` is in scope, no builtins.
        alphas = eval(raw, {"__builtins__": {}}, {"np": np})
        return [float(a) for a in np.asarray(alphas).ravel()]
    return [float(a) for a in raw]


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
    load_model_id: str | None = None,
    electrode_selection_dir: Path | None = None,
    regression_device: str = "cpu",
    n_layer_workers: int = 4,
    model_cfg: dict | None = None,
    save_word_predictions: bool = False,
    untrained: bool = False,
    untrained_seed: int = 0,
) -> PodcastBenchmarkConfig:
    ds = cfg["dataset"]
    reg = cfg.get("regression", {})
    bench = cfg.get("benchmark", {})

    _cw_raw = ds.get("context_words")
    _ml_raw = ds.get("max_length")
    data_cfg = PodcastDataConfig(
        bids_root=ds["bids_root"],
        subjects=ds.get("subjects") or None,
        tmin=float(ds.get("tmin", -2.0)),
        tmax=float(ds.get("tmax", 2.0)),
        tmin_extract=float(ds.get("tmin_extract", -2.1)),
        downsample_to=float(ds.get("downsample_to", 32.0)),
        drop_bads=bool(ds.get("drop_bads", True)),
        clip_iqr=float(ds["clip_iqr"]) if ds.get("clip_iqr") else None,
        baseline=tuple(ds["baseline"]) if ds.get("baseline") else None,
        baseline_mode=str(ds.get("baseline_mode", "logratio")),
        context_words=int(_cw_raw) if _cw_raw is not None else None,
        max_length=int(_ml_raw) if _ml_raw is not None else None,
        feature_cache_dir=str(feature_cache_dir) if feature_cache_dir else None,
        electrode_selection_dir=str(electrode_selection_dir) if electrode_selection_dir else None,
        wordfreq_language=str(bench.get("wordfreq_language", "en")),
    )
    hf_model_id = load_model_id or model_id
    model_cfg_obj = ModelConfig(
        model_name=hf_model_id,
        model_revision=revision if revision not in ("", "main") else None,
        device=device,
        batch_size=batch_size,
        n_params=n_params,
        untrained=untrained,
        untrained_seed=untrained_seed,
    )
    reg_cfg = RegressionConfig(
        n_splits=int(reg.get("n_splits", 2)),
        alpha=_resolve_alphas(reg.get("alpha")),
        scale=bool(reg.get("scale", True)),
        random_state=int(reg.get("random_state", 0)),
        print_chosen_ridge_alpha=bool(reg.get("print_chosen_ridge_alpha", False)),
    )
    return PodcastBenchmarkConfig(
        data=data_cfg,
        model=model_cfg_obj,
        regression=reg_cfg,
        output_dir=str(output_dir),
        layerwise=bool(bench.get("layerwise", True)),
        feature_mode=str(bench.get("feature_mode", "embeddings")),
        use_lag_matrix=bool(bench.get("use_lag_matrix", True)),
        n_cv_folds=int(bench.get("n_cv_folds", 3)),
        window_size=float(bench.get("window_size", 0.2)),
        step_size=float(bench.get("step_size", 0.05)),
        n_jobs=n_jobs,
        tokenizer_no_revision=bool(bench.get("tokenizer_no_revision", True)),
        trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
        regression_device=regression_device,
        n_layer_workers=n_layer_workers,
        save_word_predictions=save_word_predictions,
        compute_unique_surprisal=bool(bench.get("compute_unique_surprisal", False)),
    )


# ---------------------------------------------------------------------------
# Score row builders
# ---------------------------------------------------------------------------

def _summary_row(
    ckpt: CheckpointItem,
    result: PodcastEncoderResult,
    elec_csv: Path,
    elec_lw_csv: Path,
) -> dict[str, Any]:
    # Read electrode CSV to get all subjects (including ones skipped in this run).
    if elec_csv.exists():
        df = pd.read_csv(elec_csv)
        df = df[df["revision"] == ckpt.revision]
    else:
        df = pd.DataFrame()

    mean_best_r = float(np.nanmean(df["best_pearsonr"])) if not df.empty else np.nan
    mean_flat_r = float(np.nanmean(df["pearsonr"])) if not df.empty else np.nan
    mean_spearman = float(np.nanmean(df["best_spearmanr"])) if not df.empty else np.nan
    n_electrodes = len(df)
    n_subjects = df["subject"].nunique() if not df.empty else 0

    row: dict[str, Any] = {
        "step": ckpt.step,
        "revision": ckpt.revision,
        "tokens_seen": ckpt.tokens_seen,
        "n_words": result.n_words,
        "n_electrodes": n_electrodes,
        "n_subjects": n_subjects,
        "n_subjects_expected": result.n_subjects_expected,
        "mean_best_pearsonr": mean_best_r,
        "mean_flat_pearsonr": mean_flat_r,
        "mean_best_spearmanr": mean_spearman,
    }

    # Best layer from per-electrode layerwise CSV.
    if elec_lw_csv.exists():
        lw_df = pd.read_csv(elec_lw_csv)
        lw_df = lw_df[lw_df["revision"] == ckpt.revision]
        if not lw_df.empty:
            mean_per_layer = lw_df.groupby("layer")["pearsonr"].mean()
            # All-NaN happens when every electrode's every layer scored NaN
            # for this checkpoint (e.g. a run that predates a fix, or a
            # checkpoint where regression genuinely failed everywhere) —
            # .idxmax() raises ValueError: Encountered all NA values instead
            # of just leaving best_layer unset, which crashed the whole
            # sweep on an otherwise-recoverable per-checkpoint summary.
            if mean_per_layer.notna().any():
                best_li = int(mean_per_layer.idxmax())
                row["best_layer"] = best_li
                row["best_layer_mean_pearsonr"] = float(mean_per_layer.iloc[best_li])
    return row


def _layerwise_rows(
    ckpt: CheckpointItem,
    result: PodcastEncoderResult,
    elec_lw_csv: Path,
) -> list[dict]:
    if not elec_lw_csv.exists():
        return []
    lw_df = pd.read_csv(elec_lw_csv)
    lw_df = lw_df[lw_df["revision"] == ckpt.revision]
    if lw_df.empty:
        return []
    agg = lw_df.groupby("layer").agg(
        mean_pearsonr=("pearsonr", "mean"),
        mean_spearmanr=("spearmanr", "mean"),
    ).reset_index()
    n_layers = len(agg)
    rows = []
    for _, r in agg.iterrows():
        li = int(r["layer"])
        rows.append({
            "step": ckpt.step,
            "revision": ckpt.revision,
            "tokens_seen": ckpt.tokens_seen,
            "layer": li,
            "layer_rel": li / max(1, n_layers - 1),
            "mean_pearsonr": float(r["mean_pearsonr"]),
            "mean_spearmanr": float(r["mean_spearmanr"]),
        })
    return rows


# ---------------------------------------------------------------------------
# Stage 1 — activations only (GPU, no regression)
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
    untrained: bool = False,
    untrained_seed: int = 0,
) -> None:
    """Extract and cache LLM activations for every checkpoint without running regression.

    On a cache hit (activations already on disk) the model is never loaded —
    the function returns almost instantly.  On a miss it runs the full forward
    pass, writes the .npz, then frees the GPU before the next checkpoint.

    Designed to run as the GPU stage of a two-stage pipeline:
      Stage 1 (GPU)  — this function: populate $PODCAST_FEATURE_CACHE_DIR
      Stage 2 (CPU)  — run_model_sweep with --device cpu: regression only
    """
    if feature_cache_dir is None:
        raise ValueError(
            "--activations_only requires a feature cache directory. "
            "Set cache.features_dir in the YAML or pass --feature_cache_dir."
        )

    if untrained:
        checkpoints = checkpoints[:1]
    for ckpt in checkpoints:
        logger.info(
            "\n[%s] activations ── %s (step=%d) ──",
            model_id, ckpt.revision, ckpt.step,
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
            output_dir=feature_cache_dir,   # unused for activations, just needs a Path
            feature_cache_dir=feature_cache_dir,
            n_jobs=1,
            model_cfg=model_cfg,
            untrained=untrained,
            untrained_seed=untrained_seed,
        )
        encoder = PodcastEcogEncoder(bench_cfg)
        words_df = encoder._load_transcript()
        words = words_df["word"].astype(str).tolist()
        try:
            emb = encoder._extract_features(words)
            logger.info(
                "[%s] %s — done, shape %s",
                model_id, ckpt.revision,
                "×".join(str(d) for d in emb.shape),
            )
        except Exception:
            logger.error("[%s] %s — failed", model_id, ckpt.revision, exc_info=True)
        finally:
            clear_hf_model_cache()
            free_gpu()


# ---------------------------------------------------------------------------
# Per-model sweep
# ---------------------------------------------------------------------------

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
    no_plots: bool = False,
    model_cfg: dict | None = None,
    electrode_selection_dir: Path | None = None,
    regression_device: str = "cpu",
    n_layer_workers: int = 4,
    save_word_predictions: bool = False,
    untrained: bool = False,
    untrained_seed: int = 0,
) -> pd.DataFrame:
    if untrained:
        # Random init doesn't depend on training-step revision — one row per
        # model is enough (weights never came from any checkpoint).
        checkpoints = checkpoints[:1]
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / "podcast_checkpoint_scores.csv"
    layer_csv = output_dir / "podcast_checkpoint_layerwise.csv"
    elec_csv = output_dir / "podcast_by_electrode.csv"

    existing = load_existing_csv(scores_csv)
    summary_rows = existing.to_dict("records") if not existing.empty else []

    for ckpt in checkpoints:
        if already_computed(existing, ckpt.revision):
            # Verify completeness: if n_subjects < n_subjects_expected, the previous
            # run had subject failures — drop the stale row and recompute.
            stale = False
            if not existing.empty and {"n_subjects", "n_subjects_expected"}.issubset(existing.columns):
                row = existing[existing["revision"] == ckpt.revision]
                if not row.empty:
                    n_got = int(row["n_subjects"].iloc[0])
                    n_exp = int(row["n_subjects_expected"].iloc[0])
                    if n_exp > 0 and n_got < n_exp:
                        stale = True
                        logger.warning(
                            "[%s] %s incomplete (%d/%d subjects) — dropping stale row and recomputing",
                            model_id, ckpt.revision, n_got, n_exp,
                        )
                        existing = existing[existing["revision"] != ckpt.revision]
                        existing.to_csv(scores_csv, index=False)
                        summary_rows = existing.to_dict("records")
            if not stale:
                logger.info("[%s] %s already done — skipping", model_id, ckpt.revision)
                continue

        logger.info(
            "\n[%s] Podcast ── %s (step=%d, tokens≈%s) ──",
            model_id, ckpt.revision, ckpt.step, ckpt.tokens_seen,
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
            electrode_selection_dir=electrode_selection_dir,
            regression_device=regression_device,
            n_layer_workers=n_layer_workers,
            model_cfg=model_cfg,
            save_word_predictions=save_word_predictions,
            untrained=untrained,
            untrained_seed=untrained_seed,
        )

        elec_lw_csv = output_dir / "podcast_by_electrode_layerwise.csv"

        # ---- log full resolved config ----
        alpha_grid = bench_cfg.regression.alpha
        alpha_str = (
            "[" + ", ".join(f"{a:.3g}" for a in alpha_grid) + "]"
            if isinstance(alpha_grid, list) else str(alpha_grid)
        )
        logger.info(
            "  Hyperparameters:\n"
            "    model          : %s  revision=%s  device=%s\n"
            "    epoch window   : tmin=%.2f s  tmax=%.2f s  (extract from %.2f s)\n"
            "    preprocessing  : downsample=%.1f Hz  clip_iqr=%s  baseline=%s  mode=%s\n"
            "    LLM features   : context_words=%s  max_length=%s  layerwise=%s\n"
            "    regression     : use_lag_matrix=%s  device=%s  n_cv_folds=%d\n"
            "    ridge alpha    : grid=%s  n_alphas=%d  scale=%s  random_state=%d\n"
            "    electrode sel  : %s",
            bench_cfg.model.model_name, bench_cfg.model.model_revision,
            bench_cfg.model.device,
            bench_cfg.data.tmin, bench_cfg.data.tmax, bench_cfg.data.tmin_extract,
            bench_cfg.data.downsample_to, bench_cfg.data.clip_iqr,
            bench_cfg.data.baseline, bench_cfg.data.baseline_mode,
            bench_cfg.data.context_words, bench_cfg.data.max_length,
            bench_cfg.layerwise,
            bench_cfg.use_lag_matrix, bench_cfg.regression_device,
            bench_cfg.n_cv_folds,
            alpha_str, len(alpha_grid) if isinstance(alpha_grid, list) else 1,
            bench_cfg.regression.scale, bench_cfg.regression.random_state,
            bench_cfg.data.electrode_selection_dir or "none (all channels)",
        )

        try:
            result = PodcastEcogEncoder(bench_cfg).run()

            # ---- summary CSV (reads electrode + layerwise CSVs for all subjects) ----
            row = _summary_row(ckpt, result, elec_csv, elec_lw_csv)
            append_row(scores_csv, row)
            summary_rows.append(row)
            existing = load_existing_csv(scores_csv)

            # ---- layerwise CSV — one row per layer (reads from per-electrode layerwise CSV) ----
            for row in _layerwise_rows(ckpt, result, elec_lw_csv):
                append_row(layer_csv, row)

            # All per-subject files (electrode, subject, region, timecourse, layerwise)
            # are written incrementally inside PodcastEcogEncoder.run().

        except Exception:
            logger.error("[%s] failed at %s", model_id, ckpt.revision, exc_info=True)
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
        if elec_csv.exists():
            try:
                elec_df = pd.read_csv(elec_csv)
                _plot_brain_map(elec_df, output_dir, model_id)
                reg_csv = output_dir / "podcast_by_region.csv"
                if reg_csv.exists():
                    _plot_region_bar(pd.read_csv(reg_csv), output_dir, model_id)
            except Exception:
                logger.warning("[%s] brain map / region bar skipped", model_id, exc_info=True)
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_training_curve(df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if "step" not in df.columns:
        return
    plot_df = df.sort_values("step")
    fig, ax = plt.subplots(figsize=(9, 4))
    if "mean_best_pearsonr" in plot_df.columns:
        ax.plot(plot_df["step"], plot_df["mean_best_pearsonr"], "o-",
                label="Best layer (mean over electrodes)")
    if "mean_flat_pearsonr" in plot_df.columns:
        ax.plot(plot_df["step"], plot_df["mean_flat_pearsonr"], "s--",
                alpha=0.7, label="Last layer (flat)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title(f"Podcast ECoG — {model_id}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "podcast_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_heatmap(layer_df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if layer_df.empty:
        return
    pivot = layer_df.pivot_table(index="layer", columns="step", values="mean_pearsonr")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.6), 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer")
    ax.set_title(f"Podcast layerwise encoding — {model_id}")
    n_steps = pivot.shape[1]
    step_labels = [str(int(s)) for s in pivot.columns]
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(step_labels, rotation=45, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean Pearson r")
    fig.tight_layout()
    fig.savefig(output_dir / "podcast_layer_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_brain_map(elec_df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    """Scatter plot of electrode MNI positions coloured by best_pearsonr."""
    df = elec_df.dropna(subset=["x", "y", "z", "best_pearsonr"])
    if df.empty:
        return
    # Average over steps/revisions per electrode
    df_agg = (
        df.groupby(["subject", "electrode", "x", "y", "z", "region", "hemisphere"])
        ["best_pearsonr"].mean().reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (xc, yc, view) in zip(
        axes, [("x", "y", "Axial (X–Y)"), ("x", "z", "Coronal (X–Z)")]
    ):
        sc = ax.scatter(
            df_agg[xc], df_agg[yc],
            c=df_agg["best_pearsonr"],
            cmap="RdBu_r", vmin=-0.05, vmax=0.3,
            s=30, alpha=0.85, linewidths=0.3, edgecolors="k",
        )
        ax.set_xlabel(f"MNI {xc.upper()} (mm)")
        ax.set_ylabel(f"MNI {yc.upper()} (mm)")
        ax.set_title(f"{view} — {model_id}")
        ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
        plt.colorbar(sc, ax=ax, label="Mean Pearson r")
        ax.set_aspect("equal")
    fig.suptitle(f"Podcast ECoG brain map — {model_id}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "podcast_brain_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_region_bar(reg_df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    """Bar plot of mean Pearson r per electrode group/region."""
    if reg_df.empty:
        return
    # Flatten multi-level columns if needed
    if isinstance(reg_df.columns, pd.MultiIndex):
        mean_col = [c for c in reg_df.columns if "mean" in str(c).lower() and "pearsonr" in str(c).lower()]
        if not mean_col:
            return
        vals = reg_df[mean_col[0]].sort_values(ascending=False)
    else:
        if "best_pearsonr" not in reg_df.columns:
            return
        vals = reg_df["best_pearsonr"].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, len(vals) * 0.5), 4))
    ax.bar(range(len(vals)), vals.values)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals.index.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Pearson r")
    ax.set_title(f"Podcast by region — {model_id}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "podcast_region_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def run_summary(
    output_dir: Path,
    *,
    family: str | None = None,
    models_filter: list[str] | None = None,
) -> None:
    search_root = output_dir / family if family else output_dir
    summary_dir = (output_dir / family / "summary") if family else (output_dir / "summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    for csv in sorted(search_root.glob("**/podcast_checkpoint_scores.csv")):
        model_id = csv.parent.name.replace("_", "/", 1)
        fam = csv.parent.parent.name if csv.parent.parent != search_root else None
        df = pd.read_csv(csv)
        df.insert(0, "model_id", model_id)
        if fam:
            df.insert(0, "family", fam)
        all_rows.append(df)

    if not all_rows:
        logger.warning("No podcast_checkpoint_scores.csv found under %s", search_root)
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(summary_dir / "all_models_podcast_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    title_family = f" ({family})" if family else ""
    for model_id, grp in combined.groupby("model_id"):
        g = grp.sort_values("step")
        ax.plot(g["step"], g["mean_best_pearsonr"], "o-", label=model_id)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean best-layer Pearson r")
    ax.set_title(f"Podcast ECoG — all models{title_family}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(summary_dir / "all_models_podcast_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Summary written to %s", summary_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/embedding/neural/podcast.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n_jobs", type=int, default=10)
    p.add_argument("--electrode_selection_dir", default=None,
                   help="Dir containing selected_electrodes.csv from data/podcast/electrode_selection.py. "
                        "When set, only language-responsive electrodes are used.")
    p.add_argument("--regression_device", default="cpu",
                   help="Device for ridge regression: 'cuda' uses himalaya GPU backend (default: cpu)")
    p.add_argument("--n_layer_workers", type=int, default=36,
                   help="Layers to fit in parallel on GPU via threads (default: 36). "
                        "Reduce if OOM. Only used when --regression_device cuda.")
    p.add_argument("--save_word_predictions", action="store_true",
                   help="Save each electrode's best-layer out-of-fold predicted lag matrix "
                        "(one .npz per subject) for post-hoc per-word difficulty splits (DLT).")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--activations_only", action="store_true",
                   help="Stage 1: extract and cache LLM activations only (no regression). "
                        "Run on GPU; follow with the full sweep on CPU once cache is warm.")
    p.add_argument("--skip_summary", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--untrained", action="store_true",
        help="Use randomly initialised weights (same architecture) instead of "
             "pretrained checkpoints -- untrained baseline for brain scores.",
    )
    p.add_argument("--untrained_seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(
        args.out_dir
        or cfg.get("output", {}).get("save_dir", "outputs/podcast/embedding_sweep")
    )
    if args.untrained:
        # "_dlt" names this sweep's role as DLT-cost input (word-prediction
        # dumps); drop it for the untrained baseline so the name just reads
        # "..._untrained" instead of "..._dlt_untrained".
        base_name = out_dir.name
        if base_name.endswith("_dlt"):
            base_name = base_name[: -len("_dlt")]
        out_dir = out_dir.parent / f"{base_name}_untrained"

    cache_raw = (cfg.get("cache") or {}).get("features_dir")
    feature_cache_dir = (
        Path(os.path.expandvars(str(cache_raw))).resolve() if cache_raw else None
    )
    if feature_cache_dir and args.untrained:
        feature_cache_dir = feature_cache_dir.parent / f"{feature_cache_dir.name}_untrained"
    if feature_cache_dir:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)

    _esd_raw = args.electrode_selection_dir or (cfg.get("dataset") or {}).get("electrode_selection_dir")
    electrode_selection_dir = (
        Path(os.path.expandvars(str(_esd_raw))).resolve() if _esd_raw else None
    )
    regression_device = args.regression_device

    if args.summary_only:
        run_summary(out_dir, family=args.family, models_filter=args.models)
        return

    if not args.from_registry:
        p.error("Specify --from-registry with --family (and optional --array-index)")

    if not args.family:
        p.error("--from-registry requires --family (e.g. gpt2, pythia, mamba)")

    if args.array_index is not None:
        task = resolve_family_task(args.family, args.array_index, untrained=args.untrained)
        model_cfgs = [
            m for m in registry_model_configs(task.registry_family)
            if m["model_id"] == task.model_id
        ]
        checkpoints = [
            CheckpointItem(rev, step, tokens)
            for rev, step, tokens in build_checkpoint_items(model_cfgs[0])
            if rev == task.revision
        ]
        if args.dry_run:
            print(f"task: {task}")
            return
        if args.activations_only:
            run_activations_only(
                checkpoints,
                model_id=task.model_id,
                n_params=model_cfgs[0]["n_params"],
                batch_size=model_cfgs[0]["batch_size"],
                cfg=cfg,
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                model_cfg=model_cfgs[0],
                untrained=args.untrained,
                untrained_seed=args.untrained_seed,
            )
        else:
            run_model_sweep(
                checkpoints,
                model_id=task.model_id,
                n_params=model_cfgs[0]["n_params"],
                batch_size=model_cfgs[0]["batch_size"],
                cfg=cfg,
                output_dir=_model_output_dir(
                    out_dir, task.model_id, family=args.family, model_cfg=model_cfgs[0]
                ),
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                n_jobs=args.n_jobs,
                no_plots=args.no_plots,
                model_cfg=model_cfgs[0],
                electrode_selection_dir=electrode_selection_dir,
                regression_device=regression_device,
                n_layer_workers=args.n_layer_workers,
                save_word_predictions=args.save_word_predictions,
                untrained=args.untrained,
                untrained_seed=args.untrained_seed,
            )
        return

    # Run whole family
    all_models = registry_model_configs(args.family)
    if args.models:
        all_models = [m for m in all_models if m["model_id"] in args.models]
    if args.dry_run:
        for m in all_models:
            ckpts = [CheckpointItem(r, s, t) for r, s, t in build_checkpoint_items(m)]
            print(f"{m['model_id']}: {len(ckpts)} checkpoints")
        return

    for m in all_models:
        ckpts = [CheckpointItem(r, s, t) for r, s, t in build_checkpoint_items(m)]
        if args.activations_only:
            run_activations_only(
                ckpts,
                model_id=m["model_id"],
                n_params=m["n_params"],
                batch_size=m["batch_size"],
                cfg=cfg,
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                model_cfg=m,
                untrained=args.untrained,
                untrained_seed=args.untrained_seed,
            )
        else:
            run_model_sweep(
                ckpts,
                model_id=m["model_id"],
                n_params=m["n_params"],
                batch_size=m["batch_size"],
                cfg=cfg,
                output_dir=_model_output_dir(out_dir, m["model_id"], model_cfg=m),
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                n_jobs=args.n_jobs,
                no_plots=args.no_plots,
                model_cfg=m,
                electrode_selection_dir=electrode_selection_dir,
                regression_device=regression_device,
                n_layer_workers=args.n_layer_workers,
                save_word_predictions=args.save_word_predictions,
                untrained=args.untrained,
                untrained_seed=args.untrained_seed,
            )

    if not args.skip_summary:
        run_summary(out_dir, family=args.family, models_filter=args.models)


if __name__ == "__main__":
    main()
