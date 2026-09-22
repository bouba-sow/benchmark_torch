#!/usr/bin/env python3
"""MEG-MASC MEG encoding checkpoint sweep.

Runs layerwise encoding ([0.1, 40] Hz MEG, 208 KIT gradiometers, 50 Hz) across
model families / training checkpoints.  Leave-one-story-out cross-validation
over the four MASC stories (folds = stories; both sessions pooled).

Output structure
----------------
<output_dir>/
  <family>/
    <model_short>/
      megmasc_checkpoint_scores.csv       — per-checkpoint summary
      megmasc_checkpoint_layerwise.csv    — per-layer Pearson r
      megmasc_by_channel.csv              — per-channel scores (all subjects)
      megmasc_by_subject.csv              — per-subject mean
      megmasc_electrode_timecourse.csv    — time-resolved r per channel
      megmasc_by_channel_layerwise.csv    — per-channel × layer Pearson r
      megmasc_layerwise_timecourse.csv    — TRF curves per layer
      megmasc_training_curve.png
      megmasc_layer_heatmap.png
    summary/
      all_models_megmasc_scores.csv
      all_models_megmasc_training_curves.png

Examples
--------
# Single array task (Jean Zay):
python scripts/run/embedding/neural/megmasc.py \\
    --config configs/embedding/neural/megmasc.yaml \\
    --from-registry --family gpt2 --array-index 0 --device cuda

# Full family locally:
python scripts/run/embedding/neural/megmasc.py \\
    --config configs/embedding/neural/megmasc.yaml \\
    --from-registry --family pythia --device cuda

# Summary after all SLURM jobs:
python scripts/run/embedding/neural/megmasc.py \\
    --config configs/embedding/neural/megmasc.yaml \\
    --summary_only --out_dir outputs/megmasc/embedding_sweep
"""

from __future__ import annotations

import argparse
import json
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
from neuroai.benchmarks.megmasc import (
    MegMascBenchmarkConfig,
    MegMascDataConfig,
    MegMascMegEncoder,
    MegMascEncoderResult,
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
    load_existing_csv,
    safe_model_dir,
    free_gpu,
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


# ---------------------------------------------------------------------------
# Subject sampling — sample once, reuse across all model families
# ---------------------------------------------------------------------------

_SUBJECTS_FILE = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "configs" / "configs/subjects/megmasc_subjects.json"


def _resolve_subjects(bids_root: str) -> list[str]:
    """Fixed subject list, created on first call.

    The local MEG-MASC copy has only 11 of the 27 subjects, so there is
    nothing to subsample: the list is simply every ``sub-*`` with at least
    one readable recording, frozen to ``configs/subjects/megmasc_subjects.json`` so
    every model family sees the same set.
    """
    if _SUBJECTS_FILE.exists():
        subjects = json.loads(_SUBJECTS_FILE.read_text())
        logger.info(
            "Loaded fixed subject list (%d subjects) from %s",
            len(subjects), _SUBJECTS_FILE,
        )
        return subjects

    from neuroai.datasets.megmasc import MegMascDataset
    ds = MegMascDataset(bids_root)
    all_subjects = sorted(ds.available_subjects())
    if not all_subjects:
        raise RuntimeError(f"No subjects found under {bids_root}")

    _SUBJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SUBJECTS_FILE.write_text(json.dumps(all_subjects, indent=2))
    logger.info(
        "Discovered %d subjects → saved to %s: %s",
        len(all_subjects), _SUBJECTS_FILE, all_subjects,
    )
    return all_subjects


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
    n_subjects_batch: int = -1,
    workers_per_gpu: int = 15,
    model_cfg: dict | None = None,
    save_word_predictions: bool = True,
) -> MegMascBenchmarkConfig:
    ds = cfg["dataset"]
    reg = cfg.get("regression", {})
    bench = cfg.get("benchmark", {})

    _cw_raw = ds.get("context_words")
    _ml_raw = ds.get("max_length")
    _bl_raw = ds.get("baseline")

    subjects = _resolve_subjects(ds["bids_root"])

    _clamp_raw = ds.get("clamp", 5.0)
    data_cfg = MegMascDataConfig(
        bids_root=ds["bids_root"],
        subjects=subjects,
        stories=ds.get("stories") or None,
        tmin=float(ds.get("tmin", -0.2)),
        tmax=float(ds.get("tmax", 0.8)),
        baseline=tuple(_bl_raw) if _bl_raw is not None else (-0.2, 0.0),
        l_freq=float(ds.get("l_freq", 0.1)) if ds.get("l_freq") is not None else None,
        h_freq=float(ds.get("h_freq", 40.0)) if ds.get("h_freq") is not None else None,
        target_sfreq=float(ds.get("target_sfreq", 50.0)),
        robust_scale=bool(ds.get("robust_scale", True)),
        clamp=float(_clamp_raw) if _clamp_raw is not None else None,
        min_word_duration=float(ds.get("min_word_duration", 0.050)),
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
        n_splits=int(reg.get("n_splits", 4)),
        alpha=_resolve_alphas(reg.get("alpha")),
        scale=bool(reg.get("scale", True)),
        random_state=int(reg.get("random_state", 0)),
        print_chosen_ridge_alpha=bool(reg.get("print_chosen_ridge_alpha", False)),
    )
    return MegMascBenchmarkConfig(
        data=data_cfg,
        model=model_cfg_obj,
        regression=reg_cfg,
        output_dir=str(output_dir),
        layerwise=bool(bench.get("layerwise", True)),
        use_lag_matrix=bool(bench.get("use_lag_matrix", True)),
        lang=str(ds.get("lang", "en")),
        n_jobs=n_jobs,
        regression_device=regression_device,
        n_subjects_batch=n_subjects_batch,
        workers_per_gpu=workers_per_gpu,
        tokenizer_no_revision=bool(bench.get("tokenizer_no_revision", True)),
        trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
        save_word_predictions=save_word_predictions,
    )


# ---------------------------------------------------------------------------
# Score row builders
# ---------------------------------------------------------------------------

def _summary_row(
    ckpt: CheckpointItem,
    result: MegMascEncoderResult,
    ch_csv: Path,
    lw_csv: Path,
) -> dict[str, Any]:
    if ch_csv.exists():
        df = pd.read_csv(ch_csv)
        df = df[df["revision"] == ckpt.revision]
    else:
        df = pd.DataFrame()

    mean_best_r   = float(np.nanmean(df["best_pearsonr"]))   if not df.empty else np.nan
    mean_flat_r   = float(np.nanmean(df["pearsonr"]))        if not df.empty else np.nan
    mean_spearman = float(np.nanmean(df["best_spearmanr"]))  if not df.empty else np.nan
    n_channels    = len(df)
    n_subjects    = df["subject"].nunique() if not df.empty else 0

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
    n_subjects_batch: int = -1,
    workers_per_gpu: int = 15,
    save_word_predictions: bool = True,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv  = output_dir / "megmasc_checkpoint_scores.csv"
    layer_csv   = output_dir / "megmasc_checkpoint_layerwise.csv"
    ch_csv      = output_dir / "megmasc_by_channel.csv"
    lw_csv      = output_dir / "megmasc_by_channel_layerwise.csv"

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
            "\n[%s] MEG-MASC ── %s (step=%d, tokens≈%s) ──",
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
            n_subjects_batch=n_subjects_batch,
            workers_per_gpu=workers_per_gpu,
            model_cfg=model_cfg,
            save_word_predictions=save_word_predictions,
        )

        try:
            result = MegMascMegEncoder(bench_cfg).run()

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
    ax.set_title(f"MEG-MASC MEG — {model_id}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "megmasc_training_curve.png", dpi=150, bbox_inches="tight")
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
    ax.set_title(f"MEG-MASC layerwise encoding — {model_id}")
    n_steps = pivot.shape[1]
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([str(int(s)) for s in pivot.columns],
                       rotation=45, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean Pearson r")
    fig.tight_layout()
    fig.savefig(output_dir / "megmasc_layer_heatmap.png", dpi=150, bbox_inches="tight")
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
    search_root  = output_dir / family if family else output_dir
    summary_dir  = (output_dir / family / "summary") if family else (output_dir / "summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    for csv in sorted(search_root.glob("**/megmasc_checkpoint_scores.csv")):
        model_id_str = csv.parent.name.replace("_", "/", 1)
        fam = csv.parent.parent.name if csv.parent.parent != search_root else None
        df  = pd.read_csv(csv)
        df.insert(0, "model_id", model_id_str)
        if fam:
            df.insert(0, "family", fam)
        all_rows.append(df)

    if not all_rows:
        logger.warning("No megmasc_checkpoint_scores.csv found under %s", search_root)
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(summary_dir / "all_models_megmasc_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    title_fam = f" ({family})" if family else ""
    for mid, grp in combined.groupby("model_id"):
        g = grp.sort_values("step")
        ax.plot(g["step"], g["mean_best_pearsonr"], "o-", label=mid)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean best-layer Pearson r")
    ax.set_title(f"MEG-MASC MEG — all models{title_fam}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        summary_dir / "all_models_megmasc_training_curves.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    logger.info("Summary written to %s", summary_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/embedding/neural/megmasc.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n_jobs", type=int, default=10)
    p.add_argument("--regression_device", default="cpu")
    p.add_argument("--n_subjects_batch", type=int, default=-1,
                   help="-1 = all subjects at once (recommended); >0 = fixed batch size")
    p.add_argument("--workers_per_gpu", type=int, default=15,
                   help="Concurrent MPS tasks per GPU; total pool = n_gpus × workers_per_gpu")
    p.add_argument("--save_word_predictions", action=argparse.BooleanOptionalAction, default=True,
                   help="Save each channel's best-layer out-of-fold predicted lag matrix "
                        "(one .npz per subject) for post-hoc per-word difficulty splits (DLT). "
                        "Forces the sequential per-subject path (no flat subject×layer GPU "
                        "parallelism). On by default; pass --no-save_word_predictions to disable.")
    p.add_argument("--epoch_cache_dir", default=None)
    p.add_argument("--feature_cache_dir", default=None)
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--skip_summary", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.out_dir or cfg.get("output", {}).get("save_dir", "outputs/megmasc/embedding_sweep"))
    out_root.mkdir(parents=True, exist_ok=True)

    cache_cfg = cfg.get("cache", {})
    feature_cache_dir: Path | None = None
    epoch_cache_dir: Path | None   = None
    if args.feature_cache_dir:
        feature_cache_dir = Path(args.feature_cache_dir)
    elif cache_cfg.get("features_dir"):
        _p = cache_cfg["features_dir"]
        if _p and "$" not in str(_p):
            feature_cache_dir = Path(_p)
    if args.epoch_cache_dir:
        epoch_cache_dir = Path(args.epoch_cache_dir)
    elif cache_cfg.get("epochs_dir"):
        _p = cache_cfg["epochs_dir"]
        if _p and "$" not in str(_p):
            epoch_cache_dir = Path(_p)

    if args.summary_only:
        run_summary(out_root, family=args.family)
        return

    if args.from_registry:
        if args.family is None:
            p.error("--family is required with --from-registry")

        # Resolve cache dirs from YAML if not set on CLI
        if feature_cache_dir is None and cache_cfg.get("features_dir"):
            _p = os.path.expandvars(str(cache_cfg["features_dir"]))
            if "$" not in _p:
                feature_cache_dir = Path(_p)
        if epoch_cache_dir is None and cache_cfg.get("epochs_dir"):
            _p = os.path.expandvars(str(cache_cfg["epochs_dir"]))
            if "$" not in _p:
                epoch_cache_dir = Path(_p)

        if args.array_index is not None:
            # FamilyTask is a namedtuple: .model_id, .revision, .registry_family
            task = resolve_family_task(args.family, args.array_index)
            model_cfgs = [
                m for m in registry_model_configs(task.registry_family)
                if m["model_id"] == task.model_id
            ]
            if not model_cfgs:
                p.error(f"model_id {task.model_id!r} not found in registry family {task.registry_family!r}")
            model_cfg = model_cfgs[0]
            checkpoints = [CheckpointItem(rev, step, tokens) for rev, step, tokens in build_checkpoint_items(model_cfg)]
            if args.dry_run:
                logger.info("DRY RUN: task=%s", task)
                return
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
                epoch_cache_dir=epoch_cache_dir,
                n_jobs=args.n_jobs,
                no_plots=args.no_plots,
                model_cfg=model_cfg,
                regression_device=args.regression_device,
                n_subjects_batch=args.n_subjects_batch,
                save_word_predictions=args.save_word_predictions,
                workers_per_gpu=args.workers_per_gpu,
            )
            return

        # Full family run (no --array-index)
        for model_cfg in registry_model_configs(args.family):
            model_id = model_cfg["model_id"]
            checkpoints = [CheckpointItem(rev, step, tokens) for rev, step, tokens in build_checkpoint_items(model_cfg)]
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
                cfg=cfg,
                output_dir=output_dir,
                device=args.device,
                feature_cache_dir=feature_cache_dir,
                epoch_cache_dir=epoch_cache_dir,
                n_jobs=args.n_jobs,
                no_plots=args.no_plots,
                model_cfg=model_cfg,
                regression_device=args.regression_device,
                n_subjects_batch=args.n_subjects_batch,
                save_word_predictions=args.save_word_predictions,
                workers_per_gpu=args.workers_per_gpu,
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
                epoch_cache_dir=epoch_cache_dir,
                n_jobs=args.n_jobs,
                no_plots=args.no_plots,
                regression_device=args.regression_device,
                n_subjects_batch=args.n_subjects_batch,
                save_word_predictions=args.save_word_predictions,
                workers_per_gpu=args.workers_per_gpu,
            )
    else:
        p.error("Specify --from-registry --family <family> or --models <model_id…>")


if __name__ == "__main__":
    main()
