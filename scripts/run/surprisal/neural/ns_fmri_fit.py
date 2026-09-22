#!/usr/bin/env python3
"""Natural Stories fMRI encoding checkpoint sweep — LME version.

Fits BOLD ~ HRF-convolved surprisal (Lin & Schuler 2025 formula) per fROI,
pooling all 78 subjects and 8 stories into one mixed-effects model per fROI
(pymer4/lme4), across the same LLM checkpoint registry the ridge sweep
(``run/embedding/neural/ns_fmri.py``) uses. Unlike the ridge encoder — which
regresses full hidden-state embeddings — this uses a single scalar predictor
(surprisal), so there is no layerwise dimension and no GPU ridge stage.

Output structure
----------------
<output_dir>/
  <family>/
    <model_short>/
      ns_fmri_lme_checkpoint_scores.csv   # one row per revision (headline)
      ns_fmri_lme_by_roi.csv              # one row per (fROI, revision)
      ns_fmri_lme_by_network.csv          # one row per (network, revision)
      ns_fmri_lme_training_curve.png
    summary/
      all_models_ns_fmri_lme_scores.csv
      all_models_ns_fmri_lme_training_curves.png

Examples
--------
python scripts/run/surprisal/neural/ns_fmri_fit.py \\
    --config configs/surprisal/neural/ns_fmri.yaml \\
    --from-registry --family gpt2 --array-index 0 --device cuda

python scripts/run/surprisal/neural/ns_fmri_fit.py \\
    --config configs/surprisal/neural/ns_fmri.yaml \\
    --summary_only --out_dir outputs/ns_fmri/surprisal_sweep
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from neuroai.benchmarks.base import ModelConfig
from neuroai.benchmarks.natural_stories_fmri_lme import (
    NsFmriLmeBenchmark,
    NsFmriLmeBenchmarkConfig,
    NsFmriLmeConfig,
    NsFmriLmeDataConfig,
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
    load_model_id: str | None = None,
    trust_remote_code: bool = False,
    untrained: bool = False,
    untrained_seed: int = 0,
    n_workers: int | None = None,
    layer_stride: int | None = None,
    last_layer_only: bool = False,
    save_word_predictions: bool | None = None,
) -> NsFmriLmeBenchmarkConfig:
    ds = cfg["dataset"]
    lme = cfg.get("lme", {})

    data_cfg = NsFmriLmeDataConfig(
        data_root=os.path.expandvars(str(ds["data_root"])),
        froi_table=str(ds.get("froi_table", "LANG_MD")),
        docs=ds.get("docs") or None,
        subjects=ds.get("subjects") or None,
        t_r=float(ds.get("t_r", 2.0)),
        word_duration=float(ds.get("word_duration", 0.1)),
        hrf_model=str(ds.get("hrf_model", "glover")),
        max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        tokenizer_no_revision=bool(ds.get("tokenizer_no_revision", True)),
        feature_cache_dir=str(feature_cache_dir) if feature_cache_dir else None,
    )
    hf_model_id = load_model_id or model_id
    model_cfg_obj = ModelConfig(
        model_name=hf_model_id,
        model_revision=revision if revision not in ("", "main") else None,
        device=device,
        batch_size=batch_size,
        n_params=n_params,
        trust_remote_code=trust_remote_code,
        untrained=untrained,
        untrained_seed=untrained_seed,
    )
    lme_cfg = NsFmriLmeConfig(
        chunk_seconds=float(lme.get("chunk_seconds", 30.0)),
        train_frac=float(lme.get("train_frac", 0.75)),
        split_random_state=int(lme.get("split_random_state", 0)),
        min_valid_trs=int(lme.get("min_valid_trs", 50)),
        verbose=bool(lme.get("verbose", False)),
        layer_stride=int(layer_stride if layer_stride is not None else lme.get("layer_stride", 1)),
        last_layer_only=bool(last_layer_only or lme.get("last_layer_only", False)),
        n_workers=int(n_workers if n_workers is not None else lme.get("n_workers", 20)),
        require_cached_surprisal=bool(lme.get("require_cached_surprisal", device == "cpu")),
        include_freq=bool(lme.get("include_freq", True)),
        save_word_predictions=(
            save_word_predictions if save_word_predictions is not None
            else bool(lme.get("save_word_predictions", True))
        ),
    )
    return NsFmriLmeBenchmarkConfig(
        data=data_cfg,
        model=model_cfg_obj,
        lme=lme_cfg,
        output_dir=str(output_dir),
        canonical_model_name=model_id,
    )


# ---------------------------------------------------------------------------
# Score row builders
# ---------------------------------------------------------------------------

def _summary_row(
    ckpt: CheckpointItem,
    roi_csv: Path,
    model_revision: str,
) -> dict[str, Any]:
    """Headline summary for one checkpoint, computed from the LAST layer only.

    ``ns_fmri_lme_by_roi.csv`` now has one row per (fROI, layer, revision) —
    averaging ``mean_delta_loglik_test`` etc. across ALL layers would mix
    early- and late-layer scores into one meaningless number, so the
    checkpoint-scores headline uses the model's own final layer (matching
    the old single-top-layer-surprisal behavior this replaced). Full
    per-layer detail is still in ``ns_fmri_lme_by_roi.csv`` for later
    layer-profile analysis.
    """
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

    if "layer" in df.columns and df["layer"].notna().any():
        row["n_layers"] = int(df["layer"].nunique())
        df = df[df["layer"] == df["layer"].max()]
    if df.empty:
        return row

    row["n_frois"] = int(len(df))
    row["n_converged"] = int(df["converged"].sum())
    row["mean_beta"] = float(np.nanmean(df["beta"]))
    row["mean_delta_loglik_test"] = float(np.nanmean(df["delta_loglik_test"]))
    row["mean_test_spearmanr"] = float(np.nanmean(df["test_spearmanr_full_model"]))
    row["mean_test_pearsonr"] = float(np.nanmean(df["test_pearsonr_full_model"]))

    for net, gdf in df.groupby("network"):
        row[f"{net}_delta_loglik_test"] = float(np.nanmean(gdf["delta_loglik_test"]))
        row[f"{net}_test_spearmanr"] = float(np.nanmean(gdf["test_spearmanr_full_model"]))
        row[f"{net}_beta"] = float(np.nanmean(gdf["beta"]))

    if "LANG_delta_loglik_test" in row and "MD_delta_loglik_test" in row:
        row["lang_minus_md_delta_loglik_test"] = (
            row["LANG_delta_loglik_test"] - row["MD_delta_loglik_test"]
        )

    return row


def _checkpoint_done(existing: pd.DataFrame, revision: str) -> bool:
    """Only skip a checkpoint that previously produced at least one fROI fit."""
    if existing.empty or "revision" not in existing.columns:
        return False
    rows = existing[existing["revision"] == revision]
    if rows.empty:
        return False
    if "n_frois" in rows.columns:
        return bool((rows["n_frois"].fillna(0) > 0).any())
    return True


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
    no_plots: bool = False,
    model_cfg: dict[str, Any] | None = None,
    untrained: bool = False,
    untrained_seed: int = 0,
    n_workers: int | None = None,
    layer_stride: int | None = None,
    last_layer_only: bool = False,
    save_word_predictions: bool | None = None,
) -> pd.DataFrame:
    if untrained:
        # Random init doesn't depend on training-step revision — one row per
        # model is enough (weights never came from any checkpoint).
        checkpoints = checkpoints[:1]
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / "ns_fmri_lme_checkpoint_scores.csv"
    roi_csv = output_dir / "ns_fmri_lme_by_roi.csv"

    existing = load_existing_csv(scores_csv)
    summary_rows = existing.to_dict("records") if not existing.empty else []

    for ckpt in checkpoints:
        if _checkpoint_done(existing, ckpt.revision):
            logger.info("[%s] %s already done — skipping", model_id, ckpt.revision)
            continue

        logger.info(
            "\n[%s] NS-fMRI-LME ── %s (step=%d, tokens≈%s) ──",
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
            trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
            untrained=untrained,
            untrained_seed=untrained_seed,
            n_workers=n_workers,
            layer_stride=layer_stride,
            last_layer_only=last_layer_only,
            save_word_predictions=save_word_predictions,
        )

        try:
            NsFmriLmeBenchmark(bench_cfg).run()
            row = _summary_row(ckpt, roi_csv, ckpt.revision)
            append_row(scores_csv, row)
            summary_rows.append(row)
            existing = load_existing_csv(scores_csv)
        except Exception:
            logger.error("[%s] LME failed at %s", model_id, ckpt.revision, exc_info=True)
        finally:
            clear_hf_model_cache()
            free_gpu()

    df = pd.DataFrame(summary_rows)
    if not df.empty and not no_plots:
        _plot_training_curve(df, output_dir, model_id)
    return df


def _plot_training_curve(df: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    if "step" not in df.columns:
        return
    plot_df = df.sort_values("step")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, label in [
        (axes[0], "mean_delta_loglik_test", "ΔLogLik (held-out)"),
        (axes[1], "mean_test_spearmanr", "Held-out Spearman r"),
    ]:
        if col not in plot_df.columns:
            ax.set_visible(False)
            continue
        ax.plot(plot_df["step"], plot_df[col], "o-", label="Overall")
        lang_col = f"LANG_{col.replace('mean_', '')}"
        md_col = f"MD_{col.replace('mean_', '')}"
        if lang_col in plot_df.columns:
            ax.plot(plot_df["step"], plot_df[lang_col], "s--", alpha=0.8, label="LANG")
        if md_col in plot_df.columns:
            ax.plot(plot_df["step"], plot_df[md_col], "^--", alpha=0.8, label="MD")
        ax.set_xlabel("Training step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Natural Stories fMRI LME — {model_id}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "ns_fmri_lme_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary across models
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
    for csv_path in sorted(search_root.glob("**/ns_fmri_lme_checkpoint_scores.csv")):
        model_id = csv_path.parent.name.replace("_", "/", 1)
        fam = csv_path.parent.parent.name if csv_path.parent.parent != search_root else None
        if models_filter and model_id not in models_filter:
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "model_id", model_id)
        if fam:
            df.insert(0, "family", fam)
        all_rows.append(df)

    if not all_rows:
        logger.warning("No ns_fmri_lme_checkpoint_scores.csv found under %s", search_root)
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(summary_dir / "all_models_ns_fmri_lme_scores.csv", index=False)
    logger.info(
        "Combined CSV: %d rows, %d models", len(combined), combined["model_id"].nunique()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    title_family = f" ({family})" if family else ""
    for model_id, grp in combined.groupby("model_id"):
        g = grp.sort_values("step")
        if "mean_delta_loglik_test" in g.columns:
            ax.plot(g["step"], g["mean_delta_loglik_test"], "o-", label=model_id)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean ΔLogLik (held-out)")
    ax.set_title(f"NS fMRI LME — all models{title_family}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        summary_dir / "all_models_ns_fmri_lme_training_curves.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    logger.info("Summary written to %s", summary_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/surprisal/neural/ns_fmri.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--feature_cache_dir", default=None)
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--skip_summary", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--untrained", action="store_true",
        help="Use randomly initialised weights (same architecture) instead of "
             "pretrained checkpoints -- untrained baseline for brain scores.",
    )
    p.add_argument("--untrained_seed", type=int, default=0)
    p.add_argument("--n_workers", type=int, default=None,
                   help="Concurrent worker processes fitting (fROI, layer) pairs "
                        "(default: lme.n_workers in config, else 20)")
    p.add_argument("--layer_stride", type=int, default=None,
                   help="Fit every Nth layer (default: lme.layer_stride in config, else 1)")
    p.add_argument("--last_layer_only", action="store_true",
                   help="Fit ONLY the final layer, ignoring --layer_stride")
    p.add_argument("--save_word_predictions", action=argparse.BooleanOptionalAction, default=None,
                   help="Save each held-out TR's observed BOLD + baseline/full model predictions "
                        "to ns_fmri_lme_per_word.csv (default: lme.save_word_predictions in config, "
                        "else True). On by default; pass --no-save_word_predictions to disable.")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(
        args.out_dir
        or cfg.get("output", {}).get("save_dir", "outputs/ns_fmri/surprisal_sweep")
    )
    if args.untrained:
        out_dir = out_dir.parent / f"{out_dir.name}_untrained"

    cache_cfg = cfg.get("cache", {})
    feature_cache_dir: Path | None = None
    if args.feature_cache_dir:
        feature_cache_dir = Path(args.feature_cache_dir)
    elif cache_cfg.get("features_dir"):
        _p = os.path.expandvars(str(cache_cfg["features_dir"]))
        if "$" not in _p:
            feature_cache_dir = Path(_p)
    if feature_cache_dir and args.untrained:
        feature_cache_dir = feature_cache_dir.parent / f"{feature_cache_dir.name}_untrained"
    if feature_cache_dir:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        run_summary(out_dir, family=args.family, models_filter=args.models)
        return

    if not args.from_registry:
        p.error("Specify --from-registry with --family (and optional --array-index)")
    if not args.family:
        p.error("--from-registry requires --family")

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
            no_plots=args.no_plots,
            model_cfg=model_cfgs[0],
            untrained=args.untrained,
            untrained_seed=args.untrained_seed,
            n_workers=args.n_workers,
            layer_stride=args.layer_stride,
            last_layer_only=args.last_layer_only,
            save_word_predictions=args.save_word_predictions,
        )
        return

    all_models = registry_model_configs(args.family)
    if args.models:
        all_models = [m for m in all_models if m["model_id"] in args.models]
    if args.dry_run:
        for m in all_models:
            n = len(list(build_checkpoint_items(m)))
            print(f"{m['model_id']}: {n} checkpoints")
        return

    for m in all_models:
        ckpts = [CheckpointItem(rev, step, tok) for rev, step, tok in build_checkpoint_items(m)]
        run_model_sweep(
            ckpts,
            model_id=m["model_id"],
            n_params=m["n_params"],
            batch_size=m["batch_size"],
            cfg=cfg,
            output_dir=_model_output_dir(out_dir, m["model_id"], model_cfg=m),
            device=args.device,
            feature_cache_dir=feature_cache_dir,
            no_plots=args.no_plots,
            model_cfg=m,
            untrained=args.untrained,
            untrained_seed=args.untrained_seed,
            n_workers=args.n_workers,
            layer_stride=args.layer_stride,
            last_layer_only=args.last_layer_only,
            save_word_predictions=args.save_word_predictions,
        )

    if not args.skip_summary:
        run_summary(out_dir, family=args.family, models_filter=args.models)


if __name__ == "__main__":
    main()
