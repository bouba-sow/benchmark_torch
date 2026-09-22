#!/usr/bin/env python3
"""Szewczyk & Federmeier N400 EEG encoding checkpoint sweep.

Runs a scalar-output N400 encoding (no per-electrode spatial dimension — the
OSF deposit ships one N400 amplitude per trial, already averaged over 4
centro-parietal ROI clusters) across LLM families / training checkpoints.
GroupKFold CV (n_splits=10) grouped by (subexperiment, item) sentence frame.

Every (subject, layer) ridge fit is an independent task; all of them are
submitted to one flat thread pool of concurrent CUDA-stream workers
(``--n_gpu_workers``, default 32) rather than looping subjects sequentially —
see neuroai/benchmarks/n400_szewczyk.py docstring for why literal one-shot
batching across all subjects is not mathematically valid for this dataset.

Output structure
----------------
<out_dir>/
  <family>/
    <model_short>/
      szewczyk_checkpoint_scores.csv      — per-checkpoint summary
      szewczyk_checkpoint_layerwise.csv   — per-layer mean Spearman r
      szewczyk_by_subject.csv             — per-subject scores
      szewczyk_training_curve.png         — headline scaling law
      szewczyk_cloze_curves.png           — expected/unexpected scaling curves
                                             + cloze-correlation diagnostics
      szewczyk_layer_profile.png          — layer profile at final checkpoint
    summary/
      all_models_szewczyk_scores.csv
      all_models_szewczyk_training_curves.png
      all_models_szewczyk_cloze_curves.png

Examples
--------
# Single SLURM array task:
python scripts/run/embedding/neural/szewczyk.py \\
    --config configs/embedding/neural/szewczyk.yaml \\
    --from-registry --family pythia --array-index 0 --device cuda

# Full family:
python scripts/run/embedding/neural/szewczyk.py \\
    --config configs/embedding/neural/szewczyk.yaml \\
    --from-registry --family pythia --device cuda

# Single model:
python scripts/run/embedding/neural/szewczyk.py \\
    --config configs/embedding/neural/szewczyk.yaml \\
    --models EleutherAI/pythia-1b --device cpu

# Summary after all jobs:
python scripts/run/embedding/neural/szewczyk.py \\
    --config configs/embedding/neural/szewczyk.yaml \\
    --summary_only --out_dir outputs/szewczyk/embedding_sweep
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

from neuroai.benchmarks.n400_szewczyk import (
    SzewczykConfig,
    SzewczykEncoder,
    SzewczykResult,
    already_done,
    save_results,
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
    free_gpu,
    safe_model_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
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


def _build_encoder_config(
    yaml_cfg: dict,
    *,
    model_id: str,
    revision: str,
    n_params: float,
    batch_size: int,
    device: str,
    output_dir: Path,
    feature_cache_dir: Path | None,
    n_gpu_workers: int,
    load_model_id: str | None = None,
    model_cfg: dict | None = None,
    regression_device: str | None = None,
) -> SzewczykConfig:
    ds  = yaml_cfg.get("dataset", {})
    reg = yaml_cfg.get("regression", {})

    alphas_raw = reg.get("alpha")
    alphas = (
        np.logspace(3.5, 8, 5).tolist()
        if alphas_raw in (None, "null", [])
        else [float(a) for a in alphas_raw]
    )

    hf_id = load_model_id or model_id
    rev   = revision if revision not in ("", "main") else None

    yaml_trust = bool(yaml_cfg.get("benchmark", {}).get("trust_remote_code", False))
    trust_remote_code = bool((model_cfg or {}).get("trust_remote_code", yaml_trust))

    return SzewczykConfig(
        data_path=os.path.expandvars(str(ds.get(
            "data_path", "data/szewczyk/osfstorage/bind-5-dset_300-500.rds"
        ))),
        hf_model_name=hf_id,
        hf_revision=rev,
        hf_device=device,
        hf_batch_size=int(batch_size),
        hf_max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        layer_mode="all",
        tokenizer_no_revision=bool(yaml_cfg.get("benchmark", {}).get("tokenizer_no_revision", True)),
        trust_remote_code=trust_remote_code,
        feature_cache_dir=str(feature_cache_dir) if feature_cache_dir else None,
        n_splits=int(reg.get("n_splits", 10)),
        ridge_alphas=alphas,
        scale=bool(reg.get("scale", True)),
        regression_device=regression_device or device,
        n_gpu_workers=n_gpu_workers,
        expected_cloze_threshold=float(reg.get("expected_cloze_threshold", 0.5)),
        out_dir=str(output_dir),
        n_params=n_params,
    )


# ---------------------------------------------------------------------------
# Per-model sweep
# ---------------------------------------------------------------------------

def run_model_sweep(
    checkpoints: list[CheckpointItem],
    *,
    model_id: str,
    n_params: float,
    batch_size: int,
    yaml_cfg: dict,
    output_dir: Path,
    device: str,
    feature_cache_dir: Path | None,
    n_gpu_workers: int,
    no_plots: bool = False,
    model_cfg: dict | None = None,
    regression_device: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for ckpt in checkpoints:
        rev_key = ckpt.revision or "main"

        if already_done(output_dir, model_id, ckpt.revision):
            logger.info("[%s] %s already done — skipping", model_id, rev_key)
            continue

        logger.info(
            "\n[%s] Szewczyk N400 ── %s (step=%d, tokens≈%s) ──",
            model_id, rev_key, ckpt.step, ckpt.tokens_seen,
        )

        load_model_id = (
            resolve_load_model_id(model_cfg, ckpt.revision) if model_cfg else model_id
        )
        enc_cfg = _build_encoder_config(
            yaml_cfg,
            model_id=model_id,
            load_model_id=load_model_id,
            revision=ckpt.revision,
            n_params=n_params,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            feature_cache_dir=feature_cache_dir,
            n_gpu_workers=n_gpu_workers,
            model_cfg=model_cfg,
            regression_device=regression_device,
        )

        try:
            result = SzewczykEncoder(enc_cfg).run()
            save_results(
                result, output_dir,
                step=ckpt.step,
                tokens_seen=ckpt.tokens_seen or 0,
                n_params=n_params,
            )
            peak_layer = int(np.argmax(result.layer_spearmanr)) if result.layer_spearmanr else -1
            logger.info(
                "  ✓ sr=%.4f | peak_layer=%d/%d | expected_sr=%.4f | unexpected_sr=%.4f",
                result.mean_spearmanr,
                peak_layer, result.n_layers,
                result.expected_spearmanr_best, result.unexpected_spearmanr_best,
            )
        except Exception:
            logger.error("[%s] failed at %s", model_id, rev_key, exc_info=True)
            traceback.print_exc()
        finally:
            clear_hf_model_cache()
            free_gpu()

    if not no_plots:
        _plot_all(output_dir, model_id)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_all(output_dir: Path, model_id: str) -> None:
    scores_csv = output_dir / "szewczyk_checkpoint_scores.csv"
    layer_csv  = output_dir / "szewczyk_checkpoint_layerwise.csv"

    if not scores_csv.exists():
        return

    try:
        df = pd.read_csv(scores_csv).sort_values("tokens_seen")
        _plot_training_curve(df, output_dir, model_id)
        _plot_cloze_curves(df, output_dir, model_id)
    except Exception:
        logger.warning("Training/cloze curve plot failed", exc_info=True)

    if layer_csv.exists():
        try:
            ldf = pd.read_csv(layer_csv)
            last_rev = ldf.sort_values("tokens_seen")["revision"].iloc[-1]
            ldf = ldf[ldf["revision"] == last_rev]
            _plot_layer_profile(ldf, output_dir, model_id, rev=last_rev)
        except Exception:
            logger.warning("Layer profile plot failed", exc_info=True)


def _plot_training_curve(df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    if "tokens_seen" in df.columns and "mean_spearmanr" in df.columns:
        ax.errorbar(
            df["tokens_seen"], df["mean_spearmanr"], yerr=df.get("sem_spearmanr"),
            fmt="o-", color="black", lw=2, capsize=3, label="All items (best layer)",
        )
    ax.set_xlabel("Training tokens seen")
    ax.set_ylabel("Spearman r (per-subject, mean ± SEM)")
    ax.set_title(f"Szewczyk & Federmeier N400 — {model_id}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "szewczyk_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cloze_curves(df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    """Two panels: (1) expected vs unexpected scaling curves (median split on
    cloze_p, pooled OOF predictions), (2) continuous cloze-correlation
    diagnostics (does the model's predicted/actual N400 track cloze_p)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    if "expected_spearmanr_best" in df.columns:
        ax.plot(df["tokens_seen"], df["expected_spearmanr_best"], "o-",
                 color="#2196F3", label="Expected (cloze_p ≥ 0.5)", lw=1.8)
    if "unexpected_spearmanr_best" in df.columns:
        ax.plot(df["tokens_seen"], df["unexpected_spearmanr_best"], "o-",
                 color="#F44336", label="Unexpected (cloze_p < 0.5)", lw=1.8)
    ax.set_xlabel("Training tokens seen")
    ax.set_ylabel("Spearman r (predicted vs actual N400, pooled)")
    ax.set_title(f"N400 scaling by expectedness — {model_id}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for col, color, label in (
        ("cloze_corr_pred_best", "#9C27B0", "corr(predicted N400, cloze_p)"),
        ("cloze_corr_actual",    "#4CAF50", "corr(actual N400, cloze_p)"),
    ):
        if col in df.columns:
            ax2.plot(df["tokens_seen"], df[col], "o-", color=color, label=label, lw=1.8)
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("Training tokens seen")
    ax2.set_ylabel("Spearman r")
    ax2.set_title(f"Cloze-probability tracking — {model_id}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "szewczyk_cloze_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_profile(ldf: pd.DataFrame, output_dir: Path, model_id: str, rev: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ldf["layer"], ldf["mean_spearmanr"], "o-k", lw=2)
    best_li = int(ldf.loc[ldf["mean_spearmanr"].idxmax(), "layer"])
    ax.axvline(best_li, color="red", ls="--", lw=0.9, label=f"Best layer={best_li}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman r (mean over subjects)")
    ax.set_title(f"Szewczyk N400 layer profile — {model_id} ({rev})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "szewczyk_layer_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary across models
# ---------------------------------------------------------------------------

def run_summary(output_dir: Path, *, family: str | None = None) -> None:
    search_root = output_dir / family if family else output_dir
    summary_dir = (output_dir / family / "summary") if family else (output_dir / "summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[pd.DataFrame] = []
    for csv in sorted(search_root.glob("**/szewczyk_checkpoint_scores.csv")):
        model_str = csv.parent.name.replace("_", "/", 1)
        fam = csv.parent.parent.name if csv.parent.parent != search_root else None
        df = pd.read_csv(csv)
        df.insert(0, "model_id", model_str)
        if fam:
            df.insert(0, "family", fam)
        all_scores.append(df)

    if not all_scores:
        logger.warning("No szewczyk_checkpoint_scores.csv found under %s", search_root)
        return

    combined = pd.concat(all_scores, ignore_index=True)
    combined.to_csv(summary_dir / "all_models_szewczyk_scores.csv", index=False)

    title_fam = f" ({family})" if family else ""

    fig, ax = plt.subplots(figsize=(11, 5))
    for mid, grp in combined.groupby("model_id"):
        g = grp.sort_values("tokens_seen")
        ax.plot(g["tokens_seen"], g["mean_spearmanr"], "o-", label=mid, lw=1.5)
    ax.set_xlabel("Training tokens seen")
    ax.set_ylabel("Mean Spearman r (best layer, per-subject)")
    ax.set_title(f"Szewczyk N400 — all models{title_fam}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(summary_dir / "all_models_szewczyk_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for mid, grp in combined.groupby("model_id"):
        g = grp.sort_values("tokens_seen")
        axes[0].plot(g["tokens_seen"], g["expected_spearmanr_best"], "o-", lw=1.3, label=mid)
        axes[1].plot(g["tokens_seen"], g["unexpected_spearmanr_best"], "o-", lw=1.3, label=mid)
    axes[0].set_title(f"Expected words (cloze_p ≥ 0.5){title_fam}")
    axes[1].set_title(f"Unexpected words (cloze_p < 0.5){title_fam}")
    for ax in axes:
        ax.set_xlabel("Training tokens seen")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Spearman r (predicted vs actual N400, pooled)")
    axes[1].legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(summary_dir / "all_models_szewczyk_cloze_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Summary written to %s", summary_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/embedding/neural/szewczyk.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="cuda",
                   help="Device for LLM forward pass and regression (cuda or cpu)")
    p.add_argument("--regression_device", default=None,
                   help="Override device for the ridge-fit stage only (default: same as --device). "
                        "Set to 'cpu' to avoid a known cuSOLVER concurrency hang/slowdown when many "
                        "concurrent CUDA-stream torch.linalg.svd calls hit large hidden dims, while "
                        "keeping the LLM forward pass on --device (usually cuda) for speed.")
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None,
                   help="Run specific model IDs (ignores registry)")
    p.add_argument("--n_gpu_workers", type=int, default=32,
                   help="Concurrent CUDA-stream workers for the flattened (subject, layer) ridge sweep")
    p.add_argument("--feature_cache_dir", default=None)
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--skip_summary", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    yaml_cfg = load_config(args.config)
    out_root = Path(
        args.out_dir
        or yaml_cfg.get("output", {}).get("save_dir", "outputs/szewczyk/embedding_sweep")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    cache_cfg = yaml_cfg.get("cache", {})
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

    # ── Registry path ──────────────────────────────────────────────────────
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
                p.error(f"model_id {task.model_id!r} not in registry family {task.registry_family!r}")
            model_cfg = model_cfgs[0]
            checkpoints = [
                CheckpointItem(rev, step, tokens)
                for rev, step, tokens in build_checkpoint_items(model_cfg)
            ]
            if args.dry_run:
                logger.info("DRY RUN: task=%s | %d checkpoints", task, len(checkpoints))
                return
            run_model_sweep(
                checkpoints,
                model_id=task.model_id,
                n_params=float(model_cfg.get("n_params", 0)),
                batch_size=int(model_cfg.get("batch_size", 8)),
                yaml_cfg=yaml_cfg,
                output_dir=_model_output_dir(
                    out_root, task.model_id, family=args.family, model_cfg=model_cfg
                ),
                device=args.device,
                regression_device=args.regression_device,
                feature_cache_dir=feature_cache_dir,
                n_gpu_workers=args.n_gpu_workers,
                no_plots=args.no_plots,
                model_cfg=model_cfg,
            )
            if not args.skip_summary:
                try:
                    run_summary(out_root, family=args.family)
                except Exception:
                    logger.warning("Summary failed", exc_info=True)
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
            logger.info(
                "=== [%s/%s] %d checkpoints ===",
                args.family, model_id, len(checkpoints),
            )
            if args.dry_run:
                logger.info("DRY RUN — skipping")
                continue
            run_model_sweep(
                checkpoints,
                model_id=model_id,
                n_params=float(model_cfg.get("n_params", 0)),
                batch_size=int(model_cfg.get("batch_size", 8)),
                yaml_cfg=yaml_cfg,
                output_dir=output_dir,
                device=args.device,
                regression_device=args.regression_device,
                feature_cache_dir=feature_cache_dir,
                n_gpu_workers=args.n_gpu_workers,
                no_plots=args.no_plots,
                model_cfg=model_cfg,
            )
            if not args.skip_summary:
                try:
                    run_summary(out_root, family=args.family)
                except Exception:
                    logger.warning("Summary failed", exc_info=True)

    # ── Direct model list ──────────────────────────────────────────────────
    elif args.models:
        for model_id in args.models:
            output_dir = out_root / safe_model_dir(model_id)
            checkpoints = [CheckpointItem(revision="main", step=0, tokens_seen=None)]
            if args.dry_run:
                logger.info("DRY RUN — %s", model_id)
                continue
            run_model_sweep(
                checkpoints,
                model_id=model_id,
                n_params=0.0,
                batch_size=8,
                yaml_cfg=yaml_cfg,
                output_dir=output_dir,
                device=args.device,
                regression_device=args.regression_device,
                feature_cache_dir=feature_cache_dir,
                n_gpu_workers=args.n_gpu_workers,
                no_plots=args.no_plots,
            )

    else:
        p.error("Specify --from-registry --family <family> or --models <model_id…>")


if __name__ == "__main__":
    main()
