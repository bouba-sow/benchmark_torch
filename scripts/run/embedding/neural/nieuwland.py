#!/usr/bin/env python3
"""Nieuwland EEG N400 encoding checkpoint sweep.

Predicts EEG amplitude (all 6 N400 channels × full epoch time course) from LLM
hidden states across all model families and training checkpoints.

Output structure
----------------
outputs_torch/nieuwland/checkpoint_sweep/
  <family>/                    # e.g. gpt2, pythia, mamba
    <model_short>/
      nieuwland_checkpoint_scores.csv       # per-checkpoint summary (best layer)
      nieuwland_checkpoint_layerwise.csv    # per-layer Spearman r per checkpoint
      nieuwland_by_channel.csv              # per-channel scores
      nieuwland_time_course_<revision>.npz  # full time-resolved data per checkpoint
    summary/
      all_models_nieuwland_scores.csv

Examples
--------
# Single array task (Jean Zay / SLURM):
python scripts/run/embedding/neural/nieuwland.py \\
    --config configs/embedding/neural/nieuwland.yaml \\
    --from-registry --family gpt2 --array-index 0 --device cuda

# Full family locally:
python scripts/run/embedding/neural/nieuwland.py \\
    --config configs/embedding/neural/nieuwland.yaml \\
    --from-registry --family pythia --device cuda

# Summary after all SLURM jobs:
python scripts/run/embedding/neural/nieuwland.py \\
    --config configs/embedding/neural/nieuwland.yaml \\
    --summary_only --out_dir outputs_torch/nieuwland/checkpoint_sweep
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

import numpy as np
import pandas as pd
import yaml

from neuroai.benchmarks.eeg_erp import (
    NieuwlandErpConfig,
    NieuwlandErpEncoder,
    NieuwlandErpResult,
    result_to_checkpoint_row,
    result_to_layerwise_rows,
    result_to_channel_rows,
    save_time_courses,
)
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


def _build_encoder_config(
    cfg: dict,
    *,
    model_id: str,
    revision: str | None,
    n_params: float,
    batch_size: int,
    device: str,
    output_dir: Path,
    feature_cache_dir: Path | None,
    epoch_cache_dir: Path | None,
    regression_device: str,
    n_layer_workers: int,
    load_model_id: str | None = None,
    trust_remote_code: bool = False,
    save_word_predictions: bool = False,
) -> NieuwlandErpConfig:
    ds = cfg.get("dataset", {})
    reg = cfg.get("regression", {})
    bench = cfg.get("benchmark", {})

    hf_model_id = load_model_id or model_id

    return NieuwlandErpConfig(
        bids_root=ds.get("bids_root", "data/nieuwland"),
        included_in_paper_only=bool(ds.get("included_in_paper_only", True)),
        exclude_labs=list(ds.get("exclude_labs", ["glas"])),
        min_trials=int(ds.get("min_trials", 20)),
        # EEG preprocessing
        target_sfreq=float(ds.get("target_sfreq", 32.0)),
        epoch_tmin=float(ds.get("epoch_tmin", -0.2)),
        epoch_tmax=float(ds.get("epoch_tmax", 0.8)),
        baseline=tuple(ds["baseline"]) if ds.get("baseline") else (-0.2, 0.0),
        n400_tmin=float(ds.get("n400_tmin", 0.2)),
        n400_tmax=float(ds.get("n400_tmax", 0.5)),
        target_channels=list(ds.get("target_channels", ["Cz", "C3", "C4", "Pz", "P3", "P4"])),
        artifact_threshold_uv=float(ds.get("artifact_threshold_uv", 100.0)),
        epoch_cache_dir=str(epoch_cache_dir) if epoch_cache_dir else None,
        # LLM features
        hf_model_name=hf_model_id,
        hf_revision=revision if revision not in ("", "main") else None,
        hf_device=device,
        hf_batch_size=int(batch_size),
        hf_max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        layer_mode=bench.get("layer_mode", "all"),
        tokenizer_no_revision=bool(bench.get("tokenizer_no_revision", True)),
        trust_remote_code=trust_remote_code,
        feature_cache_dir=str(feature_cache_dir) if feature_cache_dir else None,
        # Regression
        n_folds=int(reg.get("n_folds", 10)),
        add_lab_covariate=bool(reg.get("add_lab_covariate", True)),
        ridge_alphas=_resolve_alphas(reg.get("alpha")),
        scale=bool(reg.get("scale", True)),
        random_state=int(reg.get("random_state", 0)),
        regression_device=regression_device,
        n_layer_workers=n_layer_workers,
        # Output
        out_dir=str(output_dir),
        n_params=float(n_params),
        save_word_predictions=save_word_predictions,
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
    cfg: dict,
    output_dir: Path,
    device: str,
    feature_cache_dir: Path | None,
    epoch_cache_dir: Path | None,
    regression_device: str,
    n_layer_workers: int,
    family: str | None = None,
    model_cfg: dict | None = None,
    save_word_predictions: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_csv = output_dir / "nieuwland_checkpoint_scores.csv"
    layer_csv = output_dir / "nieuwland_checkpoint_layerwise.csv"
    chan_csv = output_dir / "nieuwland_by_channel.csv"
    # "nieuwland_by_subject.csv" is written directly by NieuwlandErpEncoder.run()
    # (neuroai/benchmarks/eeg_erp.py) with its own schema. There used to be an
    # unused `subj_csv` variable + `result_to_subject_rows` import here from an
    # incomplete previous wiring attempt -- that helper's schema differs from
    # what the encoder already writes (it adds step/tokens_seen/n_params
    # columns), so appending its rows here under the same filename would have
    # reproduced the exact mixed-schema CSV corruption found in earlier sweeps
    # ns_fmri. Removed rather than wired up, since the encoder already owns
    # this file.

    existing = load_existing_csv(scores_csv)
    summary_rows: list[dict] = existing.to_dict("records") if not existing.empty else []

    for ckpt in checkpoints:
        if already_computed(existing, ckpt.revision):
            logger.info("[%s] %s already done — skipping", model_id, ckpt.revision)
            continue

        logger.info(
            "\n[%s] Nieuwland ── %s (step=%d, tokens≈%s) ──",
            model_id, ckpt.revision, ckpt.step, ckpt.tokens_seen,
        )

        load_model_id = (
            resolve_load_model_id(model_cfg, ckpt.revision) if model_cfg else model_id
        )
        enc_cfg = _build_encoder_config(
            cfg,
            model_id=model_id,
            revision=ckpt.revision,
            n_params=n_params,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            feature_cache_dir=feature_cache_dir,
            epoch_cache_dir=epoch_cache_dir,
            regression_device=regression_device,
            n_layer_workers=n_layer_workers,
            load_model_id=load_model_id,
            trust_remote_code=bool((model_cfg or {}).get("trust_remote_code", False)),
            save_word_predictions=save_word_predictions,
        )

        try:
            result: NieuwlandErpResult = NieuwlandErpEncoder(enc_cfg).run()
        except Exception:
            logger.error(
                "[%s] %s failed:\n%s", model_id, ckpt.revision, traceback.format_exc()
            )
            free_gpu()
            continue

        # ── Checkpoint summary row ─────────────────────────────────────────
        ckpt_row = result_to_checkpoint_row(
            result,
            step=ckpt.step,
            tokens_seen=ckpt.tokens_seen,
            n_params=n_params,
        )
        append_row(scores_csv, ckpt_row)
        summary_rows.append(ckpt_row)
        existing = load_existing_csv(scores_csv)

        # ── Layerwise rows ─────────────────────────────────────────────────
        for lrow in result_to_layerwise_rows(
            result, step=ckpt.step, tokens_seen=ckpt.tokens_seen, n_params=n_params
        ):
            append_row(layer_csv, lrow)

        # ── Per-channel rows ───────────────────────────────────────────────
        for crow in result_to_channel_rows(
            result, step=ckpt.step, tokens_seen=ckpt.tokens_seen, n_params=n_params
        ):
            append_row(chan_csv, crow)

        # ── Time course .npz ───────────────────────────────────────────────
        tc_path = output_dir / f"nieuwland_time_course_{ckpt.revision}.npz"
        try:
            save_time_courses(result, tc_path)
        except Exception as exc:
            logger.warning("Time course save failed: %s", exc)

        logger.info(
            "[%s] %s — best_layer=%d  "
            "mean_pc_sr=%.4f  mean_wf_sr=%.4f  "
            "item_mean_spearmanr=%.4f  item_mean_best_layer=%d",
            model_id, ckpt.revision,
            result.best_layer,
            result.mean_subject_perchannel_spearmanr, result.mean_subject_waveform_spearmanr,
            result.item_mean_spearmanr, result.item_mean_best_layer,
        )

        free_gpu()

    return pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def make_summary(output_dir: Path) -> None:
    rows = []
    for csv_path in output_dir.rglob("nieuwland_checkpoint_scores.csv"):
        try:
            df = pd.read_csv(csv_path)
            df["family"] = csv_path.parts[-3] if len(csv_path.parts) >= 3 else "unknown"
            df["model_dir"] = csv_path.parent.name
            rows.append(df)
        except Exception as exc:
            logger.warning("Could not read %s: %s", csv_path, exc)

    if not rows:
        logger.warning("No checkpoint score files found under %s", output_dir)
        return

    summary = pd.concat(rows, ignore_index=True)
    out = output_dir / "summary" / "all_models_nieuwland_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    logger.info("Summary written to %s (%d rows)", out, len(summary))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument(
        "--from-registry", action="store_true",
        help="Use neuroai.models.registry to build checkpoint list",
    )
    p.add_argument("--family", default=None, help="Model family (pythia, gpt2, …)")
    p.add_argument(
        "--array-index", type=int, default=None,
        help="SLURM array index (selects a single model×checkpoint task)",
    )
    p.add_argument("--model", default=None, help="Override: single model ID")
    p.add_argument("--revision", default="main", help="Override: single revision")
    p.add_argument("--device", default="cpu", help="Device for LLM forward pass")
    p.add_argument(
        "--regression_device", default="cpu",
        help="Device for himalaya ridge (cuda recommended for GPU nodes)",
    )
    p.add_argument("--n_layer_workers", type=int, default=36)
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    p.add_argument("--out_dir", default=None, help="Override output directory")
    p.add_argument(
        "--feature_cache_dir", default=None,
        help="Directory for LLM feature .npz caches",
    )
    p.add_argument(
        "--epoch_cache_dir", default=None,
        help="Directory for per-subject epoched EEG caches",
    )
    p.add_argument(
        "--summary_only", action="store_true",
        help="Only aggregate existing CSVs into summary; skip regression",
    )
    p.add_argument(
        "--save_word_predictions", action="store_true",
        help="Save each channel's best-layer out-of-fold item-mean ERP prediction "
             "(one .npz per model, ~80 items) for post-hoc per-item DLT-cost "
             "difficulty splits.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = load_config(args.config)

    output_dir = Path(args.out_dir or cfg.get("output_dir", "outputs_torch/nieuwland/checkpoint_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        make_summary(output_dir)
        return

    feature_cache_dir = (
        Path(args.feature_cache_dir) if args.feature_cache_dir
        else (Path(cfg["cache"]["features_dir"]) if cfg.get("cache", {}).get("features_dir") else None)
    )
    epoch_cache_dir = (
        Path(args.epoch_cache_dir) if args.epoch_cache_dir
        else (Path(cfg["cache"]["epochs_dir"]) if cfg.get("cache", {}).get("epochs_dir") else None)
    )
    regression_device = args.regression_device
    n_layer_workers = args.n_layer_workers

    # ── Single model/revision override ──────────────────────────────────────
    if args.model is not None:
        model_id = args.model
        rev = args.revision
        n_params = 0.0
        checkpoints = [CheckpointItem(revision=rev, step=0, tokens_seen=0)]
        batch_size = args.batch_size or int(cfg.get("model", {}).get("batch_size", 8))
        out = _model_output_dir(output_dir, model_id, family=args.family)
        run_model_sweep(
            checkpoints,
            model_id=model_id,
            n_params=n_params,
            batch_size=batch_size,
            cfg=cfg,
            output_dir=out,
            device=args.device,
            feature_cache_dir=feature_cache_dir,
            epoch_cache_dir=epoch_cache_dir,
            regression_device=regression_device,
            n_layer_workers=n_layer_workers,
            family=args.family,
            save_word_predictions=args.save_word_predictions,
        )
        make_summary(output_dir)
        return

    # ── Registry-based sweep ─────────────────────────────────────────────────
    if not args.from_registry:
        logger.error("Either --from-registry or --model is required")
        sys.exit(1)

    if args.array_index is not None:
        # Single SLURM array task: one (model, checkpoint) pair
        task = resolve_family_task(args.family, args.array_index)
        # FamilyTask is a dataclass — access via attributes, not dict keys
        model_id = task.model_id
        revision = task.revision
        step = task.step
        tokens_seen = task.tokens_seen
        load_model_id = task.load_model_id

        # Get n_params and batch_size from the registry model config dict
        model_cfgs = [
            m for m in registry_model_configs(task.registry_family)
            if m["model_id"] == task.model_id
        ]
        model_cfg_dict = model_cfgs[0] if model_cfgs else {}
        n_params = float(model_cfg_dict.get("n_params", 0.0))
        batch_size = args.batch_size or int(model_cfg_dict.get("batch_size", 8))

        out_dir_m = _model_output_dir(
            output_dir, model_id, family=args.family
        )
        out_dir_m.mkdir(parents=True, exist_ok=True)

        scores_csv = out_dir_m / "nieuwland_checkpoint_scores.csv"
        existing = load_existing_csv(scores_csv)
        if already_computed(existing, revision):
            logger.info("[%s] %s already done — exiting", model_id, revision)
            return

        enc_cfg = _build_encoder_config(
            cfg,
            model_id=model_id,
            revision=revision,
            n_params=n_params,
            batch_size=batch_size,
            device=args.device,
            output_dir=out_dir_m,
            feature_cache_dir=feature_cache_dir,
            epoch_cache_dir=epoch_cache_dir,
            regression_device=regression_device,
            n_layer_workers=n_layer_workers,
            load_model_id=load_model_id,
            trust_remote_code=bool(model_cfg_dict.get("trust_remote_code", False)),
            save_word_predictions=args.save_word_predictions,
        )

        logger.info(
            "\n[%s] Nieuwland ── %s (step=%d) ──", model_id, revision, step
        )
        try:
            result = NieuwlandErpEncoder(enc_cfg).run()
        except Exception:
            logger.error("Failed:\n%s", traceback.format_exc())
            sys.exit(1)

        ckpt = CheckpointItem(revision=revision, step=step, tokens_seen=tokens_seen)
        append_row(scores_csv, result_to_checkpoint_row(
            result, step=ckpt.step, tokens_seen=ckpt.tokens_seen, n_params=n_params
        ))
        layer_csv = out_dir_m / "nieuwland_checkpoint_layerwise.csv"
        for lrow in result_to_layerwise_rows(
            result, step=ckpt.step, tokens_seen=ckpt.tokens_seen, n_params=n_params
        ):
            append_row(layer_csv, lrow)
        chan_csv = out_dir_m / "nieuwland_by_channel.csv"
        for crow in result_to_channel_rows(
            result, step=ckpt.step, tokens_seen=ckpt.tokens_seen, n_params=n_params
        ):
            append_row(chan_csv, crow)
        tc_path = out_dir_m / f"nieuwland_time_course_{ckpt.revision}.npz"
        try:
            save_time_courses(result, tc_path)
        except Exception as exc:
            logger.warning("Time course save failed: %s", exc)

        logger.info(
            "[%s] %s — best_layer=%d  "
            "mean_pc_sr=%.4f  mean_wf_sr=%.4f  "
            "item_mean_spearmanr=%.4f  item_mean_best_layer=%d",
            model_id, ckpt.revision,
            result.best_layer,
            result.mean_subject_perchannel_spearmanr, result.mean_subject_waveform_spearmanr,
            result.item_mean_spearmanr, result.item_mean_best_layer,
        )
        return

    # ── Full family sweep ────────────────────────────────────────────────────
    models = registry_model_configs(args.family)
    for model_cfg in models:
        model_id = model_cfg["model_id"]
        n_params = float(model_cfg.get("n_params", 0.0))
        batch_size = args.batch_size or int(model_cfg.get("batch_size", 8))
        checkpoints = [
            CheckpointItem(rev, step, tok)
            for rev, step, tok in build_checkpoint_items(model_cfg)
        ]
        out = _model_output_dir(output_dir, model_id, family=args.family, model_cfg=model_cfg)
        run_model_sweep(
            checkpoints,
            model_id=model_id,
            n_params=n_params,
            batch_size=batch_size,
            cfg=cfg,
            output_dir=out,
            device=args.device,
            feature_cache_dir=feature_cache_dir,
            epoch_cache_dir=epoch_cache_dir,
            regression_device=regression_device,
            n_layer_workers=n_layer_workers,
            family=args.family,
            model_cfg=model_cfg,
            save_word_predictions=args.save_word_predictions,
        )

    make_summary(output_dir)


if __name__ == "__main__":
    main()
