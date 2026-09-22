#!/usr/bin/env python3
"""Broderick 2018 EEG encoding checkpoint sweep.

Runs layerwise encoding (1–8 Hz bandpass EEG, 128 channels) across LPP model
families / training checkpoints.  Leave-one-run-out cross-validation over 20
audiobook segments (folds = runs = segments).

Output structure
----------------
<output_dir>/
  <family>/
    <model_short>/
      broderick_checkpoint_scores.csv       — per-checkpoint summary
      broderick_checkpoint_layerwise.csv    — per-layer Pearson r
      broderick_by_channel.csv             — per-channel scores (all subjects)
      broderick_by_subject.csv             — per-subject mean
      broderick_by_region.csv              — per scalp-region mean
      broderick_electrode_timecourse.csv   — time-resolved r per channel
      broderick_by_channel_layerwise.csv   — per-channel × layer Pearson r
      broderick_training_curve.png
      broderick_layer_heatmap.png
    summary/
      all_models_broderick_scores.csv
      all_models_broderick_training_curves.png

Examples
--------
# Single array task (Jean Zay):
python scripts/run/embedding/neural/broderick.py \\
    --config configs/embedding/neural/broderick.yaml \\
    --from-registry --family gpt2 --array-index 0 --device cuda

# Full family:
python scripts/run/embedding/neural/broderick.py \\
    --config configs/embedding/neural/broderick.yaml \\
    --from-registry --family pythia --device cuda

# Summary after all SLURM jobs:
python scripts/run/embedding/neural/broderick.py \\
    --config configs/embedding/neural/broderick.yaml \\
    --summary_only --out_dir outputs/broderick_checkpoint_sweep
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
from neuroai.benchmarks.broderick import (
    BroderickBenchmarkConfig,
    BroderickDataConfig,
    BroderickEegEncoder,
    BroderickEncoderResult,
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
    epoch_cache_dir: Path | None,
    n_jobs: int,
    load_model_id: str | None = None,
    regression_device: str = "cpu",
    n_layer_workers: int = 36,
    n_subjects_batch: int = 1,
    model_cfg: dict | None = None,
    save_word_predictions: bool = True,
    layer_chunk_size: int | None = None,
) -> BroderickBenchmarkConfig:
    ds = cfg["dataset"]
    reg = cfg.get("regression", {})
    bench = cfg.get("benchmark", {})

    _cw_raw = ds.get("context_words")
    _ml_raw = ds.get("max_length")
    data_cfg = BroderickDataConfig(
        bids_root=ds["bids_root"],
        subjects=ds.get("subjects") or None,
        n_runs=int(ds.get("n_runs", 20)),
        tmin=float(ds.get("tmin", -0.2)),
        tmax=float(ds.get("tmax", 1.0)),
        baseline=tuple(ds["baseline"]) if ds.get("baseline") is not None else (-0.2, 0.0),
        l_freq=float(ds.get("l_freq", 1.0)),
        h_freq=float(ds.get("h_freq", 8.0)),
        target_sfreq=float(ds.get("target_sfreq", 32.0)),
        average_ref=bool(ds.get("average_ref", True)),
        drop_bads=bool(ds.get("drop_bads", True)),
        context_words=int(_cw_raw) if _cw_raw is not None else None,
        max_length=int(_ml_raw) if _ml_raw is not None else None,
        feature_cache_dir=str(feature_cache_dir) if feature_cache_dir else None,
        epoch_cache_dir=str(epoch_cache_dir) if epoch_cache_dir else None,
    )
    hf_model_id = load_model_id or model_id
    model_cfg_obj = ModelConfig(
        model_name=hf_model_id,
        model_revision=revision if revision not in ("", "main") else None,
        device=device,
        batch_size=batch_size,
        n_params=n_params,
    )
    reg_cfg = RegressionConfig(
        n_splits=int(reg.get("n_splits", 5)),
        alpha=_resolve_alphas(reg.get("alpha")),
        scale=bool(reg.get("scale", True)),
        random_state=int(reg.get("random_state", 0)),
        print_chosen_ridge_alpha=bool(reg.get("print_chosen_ridge_alpha", False)),
    )
    return BroderickBenchmarkConfig(
        data=data_cfg,
        model=model_cfg_obj,
        regression=reg_cfg,
        output_dir=str(output_dir),
        layerwise=bool(bench.get("layerwise", True)),
        use_lag_matrix=bool(bench.get("use_lag_matrix", True)),
        window_size=float(bench.get("window_size", 0.05)),
        step_size=float(bench.get("step_size", 0.025)),
        n_jobs=n_jobs,
        tokenizer_no_revision=bool(bench.get("tokenizer_no_revision", True)),
        trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
        regression_device=regression_device,
        n_layer_workers=n_layer_workers,
        n_subjects_batch=n_subjects_batch,
        save_word_predictions=save_word_predictions,
        layer_chunk_size=layer_chunk_size,
    )


# ---------------------------------------------------------------------------
# Score row builders
# ---------------------------------------------------------------------------

def _summary_row(
    ckpt: CheckpointItem,
    result: BroderickEncoderResult,
    ch_csv: Path,
    lw_csv: Path,
) -> dict[str, Any]:
    if ch_csv.exists():
        df = pd.read_csv(ch_csv)
        df = df[df["revision"] == ckpt.revision]
    else:
        df = pd.DataFrame()

    mean_best_r = float(np.nanmean(df["best_pearsonr"])) if not df.empty else np.nan
    mean_flat_r = float(np.nanmean(df["pearsonr"])) if not df.empty else np.nan
    mean_spearman = float(np.nanmean(df["best_spearmanr"])) if not df.empty else np.nan
    n_channels = len(df)
    n_subjects = df["subject"].nunique() if not df.empty else 0

    row: dict[str, Any] = {
        "step": ckpt.step,
        "revision": ckpt.revision,
        "tokens_seen": ckpt.tokens_seen,
        "n_words": result.n_words,
        "n_channels": n_channels,
        "n_subjects": n_subjects,
        "n_subjects_expected": result.n_subjects_expected,
        "mean_best_pearsonr": mean_best_r,
        "mean_flat_pearsonr": mean_flat_r,
        "mean_best_spearmanr": mean_spearman,
    }

    if lw_csv.exists():
        lw_df = pd.read_csv(lw_csv)
        lw_df = lw_df[lw_df["revision"] == ckpt.revision]
        if not lw_df.empty:
            mean_per_layer = lw_df.groupby("layer")["pearsonr"].mean()
            best_li = int(mean_per_layer.idxmax())
            row["best_layer"] = best_li
            row["best_layer_mean_pearsonr"] = float(mean_per_layer.iloc[best_li])

    return row


def _layerwise_rows(
    ckpt: CheckpointItem,
    lw_csv: Path,
) -> list[dict]:
    if not lw_csv.exists():
        return []
    lw_df = pd.read_csv(lw_csv)
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
    epoch_cache_dir: Path | None,
    n_jobs: int,
    no_plots: bool = False,
    model_cfg: dict | None = None,
    regression_device: str = "cpu",
    n_layer_workers: int = 36,
    n_subjects_batch: int = 1,
    save_word_predictions: bool = True,
    layer_chunk_size: int | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / "broderick_checkpoint_scores.csv"
    layer_csv = output_dir / "broderick_checkpoint_layerwise.csv"
    ch_csv = output_dir / "broderick_by_channel.csv"
    lw_csv = output_dir / "broderick_by_channel_layerwise.csv"

    existing = load_existing_csv(scores_csv)
    summary_rows = existing.to_dict("records") if not existing.empty else []

    for ckpt in checkpoints:
        if already_computed(existing, ckpt.revision):
            stale = False
            if not existing.empty and {"n_subjects", "n_subjects_expected"}.issubset(existing.columns):
                row = existing[existing["revision"] == ckpt.revision]
                if not row.empty:
                    n_got = int(row["n_subjects"].iloc[0])
                    n_exp = int(row["n_subjects_expected"].iloc[0])
                    if n_exp > 0 and n_got < n_exp:
                        stale = True
                        logger.warning(
                            "[%s] %s incomplete (%d/%d subjects) — recomputing",
                            model_id, ckpt.revision, n_got, n_exp,
                        )
                        existing = existing[existing["revision"] != ckpt.revision]
                        existing.to_csv(scores_csv, index=False)
                        summary_rows = existing.to_dict("records")
            if not stale:
                logger.info("[%s] %s already done — skipping", model_id, ckpt.revision)
                continue

        logger.info(
            "\n[%s] Broderick ── %s (step=%d, tokens≈%s) ──",
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
            epoch_cache_dir=epoch_cache_dir,
            n_jobs=n_jobs,
            regression_device=regression_device,
            n_layer_workers=n_layer_workers,
            n_subjects_batch=n_subjects_batch,
            model_cfg=model_cfg,
            save_word_predictions=save_word_predictions,
            layer_chunk_size=layer_chunk_size,
        )

        try:
            result = BroderickEegEncoder(bench_cfg).run()

            row = _summary_row(ckpt, result, ch_csv, lw_csv)
            append_row(scores_csv, row)
            summary_rows.append(row)
            existing = load_existing_csv(scores_csv)

            for lw_row in _layerwise_rows(ckpt, lw_csv):
                append_row(layer_csv, lw_row)

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
                label="Best layer (mean over channels)")
    if "mean_flat_pearsonr" in plot_df.columns:
        ax.plot(plot_df["step"], plot_df["mean_flat_pearsonr"], "s--",
                alpha=0.7, label="Last layer (flat)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title(f"Broderick EEG — {model_id}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "broderick_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_heatmap(layer_df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if layer_df.empty:
        return
    pivot = layer_df.pivot_table(index="layer", columns="step", values="mean_pearsonr")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.6), 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.05, vmax=0.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer")
    ax.set_title(f"Broderick layerwise encoding — {model_id}")
    n_steps = pivot.shape[1]
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([str(int(s)) for s in pivot.columns],
                       rotation=45, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean Pearson r")
    fig.tight_layout()
    fig.savefig(output_dir / "broderick_layer_heatmap.png", dpi=150, bbox_inches="tight")
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
    for csv in sorted(search_root.glob("**/broderick_checkpoint_scores.csv")):
        model_id_str = csv.parent.name.replace("_", "/", 1)
        fam = csv.parent.parent.name if csv.parent.parent != search_root else None
        df = pd.read_csv(csv)
        df.insert(0, "model_id", model_id_str)
        if fam:
            df.insert(0, "family", fam)
        all_rows.append(df)

    if not all_rows:
        logger.warning("No broderick_checkpoint_scores.csv found under %s", search_root)
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(summary_dir / "all_models_broderick_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    title_fam = f" ({family})" if family else ""
    for mid, grp in combined.groupby("model_id"):
        g = grp.sort_values("step")
        ax.plot(g["step"], g["mean_best_pearsonr"], "o-", label=mid)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean best-layer Pearson r")
    ax.set_title(f"Broderick EEG — all models{title_fam}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        summary_dir / "all_models_broderick_training_curves.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    logger.info("Summary written to %s", summary_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/embedding/neural/broderick.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n_jobs", type=int, default=10)
    p.add_argument("--regression_device", default="cpu")
    p.add_argument("--n_layer_workers", type=int, default=6)
    p.add_argument("--layer_chunk_size", type=int, default=None,
                    help="Cap on layers fit concurrently on GPU per subject. "
                         "Needed for wide-hidden-dim models (opt-30b/66b) that "
                         "OOM the ridge fit even with sharded model loading.")
    p.add_argument("--n_subjects_batch", type=int, default=1,
                   help="Subjects per parallel GPU batch (1=sequential, >1=flat subject×layer parallelism)")
    p.add_argument("--epoch_cache_dir", default=None,
                   help="Dir for cached preprocessed EEG epochs (model-independent)")
    p.add_argument("--save_word_predictions", action=argparse.BooleanOptionalAction, default=True,
                   help="Save each channel's best-layer out-of-fold predicted lag matrix "
                        "(one .npz per subject) for post-hoc per-word difficulty splits (DLT). "
                        "Forces the sequential per-subject path (n_subjects_batch=1 behaviour). "
                        "On by default; pass --no-save_word_predictions to disable.")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--skip_summary", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(
        args.out_dir
        or cfg.get("output", {}).get("save_dir", "outputs/broderick_checkpoint_sweep")
    )

    cache_raw = (cfg.get("cache") or {}).get("features_dir")
    # Expand env vars; treat unexpanded ${VAR} (env var unset) as no-cache.
    if cache_raw:
        expanded = os.path.expandvars(str(cache_raw))
        feature_cache_dir: Path | None = (
            None if expanded.startswith("$") or "${" in expanded
            else Path(expanded).resolve()
        )
    else:
        feature_cache_dir = None
    if feature_cache_dir:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)

    # Epoch cache (EEG, model-independent): CLI arg overrides YAML; same guard.
    epoch_raw = args.epoch_cache_dir or (cfg.get("cache") or {}).get("epochs_dir")
    if epoch_raw:
        epoch_exp = os.path.expandvars(str(epoch_raw))
        epoch_cache_dir: Path | None = (
            None if epoch_exp.startswith("$") or "${" in epoch_exp
            else Path(epoch_exp).resolve()
        )
    else:
        epoch_cache_dir = None
    if epoch_cache_dir:
        epoch_cache_dir.mkdir(parents=True, exist_ok=True)

    regression_device = args.regression_device

    if args.summary_only:
        run_summary(out_dir, family=args.family, models_filter=args.models)
        return

    if not args.from_registry:
        p.error("Specify --from-registry with --family (and optional --array-index)")

    if not args.family:
        p.error("--from-registry requires --family (e.g. gpt2, pythia, mamba)")

    if args.array_index is not None:
        task = resolve_family_task(args.family, args.array_index)
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
            epoch_cache_dir=epoch_cache_dir,
            n_jobs=args.n_jobs,
            no_plots=args.no_plots,
            model_cfg=model_cfgs[0],
            regression_device=regression_device,
            n_layer_workers=args.n_layer_workers,
            n_subjects_batch=args.n_subjects_batch,
            save_word_predictions=args.save_word_predictions,
            layer_chunk_size=args.layer_chunk_size,
        )
        return

    # Full family run
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
        run_model_sweep(
            ckpts,
            model_id=m["model_id"],
            n_params=m["n_params"],
            batch_size=m["batch_size"],
            cfg=cfg,
            output_dir=_model_output_dir(out_dir, m["model_id"], model_cfg=m),
            device=args.device,
            feature_cache_dir=feature_cache_dir,
            epoch_cache_dir=epoch_cache_dir,
            n_jobs=args.n_jobs,
            no_plots=args.no_plots,
            model_cfg=m,
            regression_device=regression_device,
            n_layer_workers=args.n_layer_workers,
            n_subjects_batch=args.n_subjects_batch,
            save_word_predictions=args.save_word_predictions,
            layer_chunk_size=args.layer_chunk_size,
        )

    if not args.skip_summary:
        run_summary(out_dir, family=args.family, models_filter=args.models)


if __name__ == "__main__":
    main()
