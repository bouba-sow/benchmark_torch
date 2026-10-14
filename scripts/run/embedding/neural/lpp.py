#!/usr/bin/env python3
"""LPP fMRI Checkpoint Sweep.

Runs the LPP (Le Petit Prince) voxelwise fMRI encoding across multiple model
families and training checkpoints.

Key features
------------
* **YAML-driven** — all settings in ``configs/embedding/neural/lpp.yaml``.
* **Per-layer analysis** — optional ridge encoding for every hidden layer,
  producing layer × checkpoint heatmaps.
* **Lateralization** — optional left/right hemisphere Pearson r and asymmetry
  index (L − R), replicating the methodology of
  Bonnasse-Gahot & Pallier (2025) and the ``llms_brain_lateralization`` repo.
* **Multi-subject** — averages scores across subjects when multiple are listed
  in the config.
* **Resumable** — each checkpoint result is appended to its CSV immediately;
  re-running skips already-computed rows.
* **Incremental plots** — summary figures are regenerated after every model
  completes (unless ``--skip_summary``).

**Where models are defined**

There is **no** hard-coded model registry in this script (unlike
``run_checkpoint_sweep.py``). All models are listed under **``models:``** in
``configs/embedding/neural/lpp.yaml`` (``model_id``, ``label``, ``family``,
checkpoint list, batch size, etc.). **``--models``** filters that YAML list to run
one or more **HuggingFace IDs** in a single invocation.

**Parallel runs (one Slurm job per model)** — share the same ``--out_dir``,
pass e.g. ``--models EleutherAI/pythia-1b`` in one job only, and use
``--skip_summary`` so workers do not write a partial ``summary/``. When every
model has finished, run **once** ``--summary_only`` (same ``--out_dir``) to
aggregate all subfolders into ``summary/*.csv`` and panels.

Output structure
----------------
<output_dir>/
  <model_short>/
    lpp_checkpoint_scores.csv        — one row per checkpoint (overall)
    lpp_checkpoint_layerwise.csv     — one row per (checkpoint, layer) if per_layer
    lpp_checkpoint_by_fold.csv       — one row per LOSO fold × layer (alpha, scores)
    lpp_checkpoint_by_roi.csv        — one row per sphere ROI × layer (when roi_masks_dir set)
    lpp_checkpoint_by_parcel.csv     — one row per HO atlas region × layer (L/R lateralization)
    lpp_checkpoint_by_voxel.csv      — one row per voxel × layer (optional; large)
    lpp_training_curve.png
    lpp_lateralization_curve.png
    lpp_layer_heatmap.png
  summary/
    all_models_lpp_scores.csv
    all_models_lpp_training_curves.png
    all_models_lpp_lateralization.png
    all_models_lpp_layer_heatmaps_grid.png

Examples
--------
# Full sweep with defaults from YAML:
python scripts/run/embedding/neural/lpp.py \\
    --config configs/embedding/neural/lpp.yaml

# Override device and output directory:
python scripts/run/embedding/neural/lpp.py \\
    --config configs/embedding/neural/lpp.yaml \\
    --device cuda --out_dir outputs/lpp_sweep_v2

# Dry-run: only print the checkpoint plan, no computation:
python scripts/run/embedding/neural/lpp.py \\
    --config configs/embedding/neural/lpp.yaml --dry_run

# Single model (e.g. dedicate one Slurm task to this HF id):
python scripts/run/embedding/neural/lpp.py \\
    --config configs/embedding/neural/lpp.yaml \\
    --models EleutherAI/pythia-1b --skip_summary --out_dir outputs/lpp/embedding_sweep

# After all per-model jobs finished — merge summary/ from every model subdir:
python scripts/run/embedding/neural/lpp.py \\
    --config configs/embedding/neural/lpp.yaml \\
    --summary_only --out_dir outputs/lpp/embedding_sweep
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import logging
import os
import sys
from contextlib import contextmanager
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

from neuroai.datasets.le_petit_prince import discover_lpp_derivative_subjects
from neuroai.lpp_encoding import (
    LppEncodingResult,
    LppFmriEncoder,
    LppFmriEncodingConfig,
    LppLayerwiseEncodingResult,
    layerwise_summary_dataframe,
    result_to_dataframe,
)
from neuroai.models.embedding import (
    OLMO2_0425_1B_DEFAULT_REVISIONS,
    OLMO2_1124_7B_DEFAULT_REVISIONS,
    OLMO2_1124_13B_DEFAULT_REVISIONS,
    PYTHIA_CHECKPOINT_STEPS,
    olmo2_stage1_metrics,
    pythia_checkpoint_revision,
)
from neuroai.plotting.lpp import (
    plot_lpp_all_models_panel,
    plot_lpp_lateralization_curve,
    plot_lpp_lateralization_panel,
    plot_lpp_layer_heatmap,
    plot_lpp_layer_heatmap_grid,
    plot_lpp_training_curve,
)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

class _FlushingStreamHandler(logging.StreamHandler):
    """Flush stdout after every log line so Slurm .out files update live."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.flush()
        except OSError:
            pass
        sys.stdout.flush()


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except OSError:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except OSError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[_FlushingStreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint item
# ---------------------------------------------------------------------------

class CheckpointItem(NamedTuple):
    """One HF weight snapshot."""
    revision: str
    step: int
    tokens_seen: int


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw)


def _resolve_feature_cache_dir(cfg: dict) -> Path | None:
    """Absolute lustre path for LLM feature .npz files (shared across step jobs)."""
    raw = os.environ.get("LPP_FEATURE_CACHE_DIR", "").strip()
    if not raw:
        raw = (cfg.get("cache") or {}).get("features_dir")
    if raw in (None, "", "null"):
        return None
    expanded = os.path.expandvars(str(raw))
    if "${" in expanded or "$WORK" in expanded and not os.environ.get("WORK"):
        raise RuntimeError(
            f"Feature cache path not expanded (is WORK set?): {expanded!r}"
        )
    path = Path(expanded).resolve()
    path.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("cache", {})["features_dir"] = str(path)
    return path


def _active_dataset_cfg(cfg: dict) -> dict:
    """Return the active dataset sub-config."""
    key = cfg.get("active_dataset", "lpp")
    datasets = cfg.get("datasets", {})
    if key not in datasets:
        raise KeyError(
            f"active_dataset='{key}' not found in datasets: {list(datasets.keys())}"
        )
    return {"type": key, **datasets[key]}


def _build_checkpoints(model_cfg: dict, cli_steps: list[int] | None) -> list[CheckpointItem]:
    """Build the checkpoint list for one model entry."""
    family = model_cfg.get("family", "pythia")
    ckpt_cfg = model_cfg.get("checkpoints", {})
    ckpt_type = ckpt_cfg.get("type", "steps")

    if family == "olmo2" or ckpt_type == "revisions":
        revisions: list[str] = ckpt_cfg.get("revisions") or []
        out: list[CheckpointItem] = []
        for rev in revisions:
            parsed = olmo2_stage1_metrics(rev)
            if parsed is None:
                logger.warning("OLMo revision %r unrecognised — using step=0, tokens=0", rev)
                out.append(CheckpointItem(revision=rev, step=0, tokens_seen=0))
            else:
                step, tokens = parsed
                out.append(CheckpointItem(revision=rev, step=step, tokens_seen=tokens))
        return out

    if ckpt_type == "main" or family == "main":
        # No training-step checkpoints (e.g. cbr_rnn, gpt2, most non-Pythia/OLMo
        # families) -- a single "main" revision. Without this branch, ckpt_cfg
        # has no "steps" key and silently falls through to the Pythia-style
        # branch below, defaulting to the full PYTHIA_CHECKPOINT_STEPS list
        # (step0, step128, step256, ...) -- revisions that don't exist on repos
        # like cclark/cbr-rnn, causing every "checkpoint" to fail to download.
        return [CheckpointItem(revision="main", step=0, tokens_seen=0)]

    # Pythia-style: integer steps → "stepN" revisions
    steps = cli_steps if cli_steps is not None else ckpt_cfg.get("steps", PYTHIA_CHECKPOINT_STEPS)
    TOKENS_PER_STEP = 2_097_152  # 2^21
    return [
        CheckpointItem(
            revision=pythia_checkpoint_revision(s),
            step=s,
            tokens_seen=s * TOKENS_PER_STEP,
        )
        for s in steps
    ]


def _discover_subjects(dataset_cfg: dict) -> list[str]:
    """Return list of subjects from config or auto-discover from BIDS root."""
    subjects = dataset_cfg.get("subjects")
    if subjects:
        return [str(s) for s in subjects]
    # Auto-discover based on language prefix
    lang = dataset_cfg.get("lang", "en").upper()
    bids_root = Path(dataset_cfg["bids_root"])
    pattern = f"sub-{lang}*"
    found = sorted(p.name for p in bids_root.glob(pattern) if p.is_dir())
    if not found:
        # Fallback: any sub-* directory
        found = sorted(p.name for p in bids_root.glob("sub-*") if p.is_dir())
    logger.info("Auto-discovered %d subjects: %s...", len(found), found[:5])
    return found


def _fmri_unit(dataset_cfg: dict) -> str:
    u = str(dataset_cfg.get("fmri_unit", "single_subject")).lower().strip()
    if u not in ("single_subject", "average_subject"):
        raise ValueError(
            f"datasets.*.fmri_unit must be 'single_subject' or 'average_subject', got {u!r}"
        )
    return u


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "_")


def _safe_revision(revision: str) -> str:
    return revision.replace("/", "_").replace(" ", "_")[:200]


# ---------------------------------------------------------------------------
# Per-checkpoint scoring
# ---------------------------------------------------------------------------

def _save_lpp_word_predictions(
    lw_result: LppLayerwiseEncodingResult,
    model_dir: Path,
    revision: str,
    subject: str,
) -> None:
    """Persist ``LppLayerwiseEncodingResult.fold_predictions`` (last layer only,
    see ``LppFmriEncodingConfig.save_word_predictions``) to one .npz per LOSO
    fold, for post-hoc per-TR difficulty splits (e.g. DLT cost)."""
    pred_dir = model_dir / "lpp_embed_word_predictions" / _safe_model_dir(revision) / subject
    pred_dir.mkdir(parents=True, exist_ok=True)
    for fp in lw_result.fold_predictions or []:
        out_path = pred_dir / f"section{fp['section']}_run{fp['bids_run']}.npz"
        np.savez(
            out_path,
            y_test=fp["y_test"],
            pred=fp["pred"],
            layer=fp["layer"],
            voxel_global_ix=lw_result.voxel_global_ix,
        )
    logger.info(
        "  [%s | %s] saved %d fold prediction file(s) -> %s",
        subject, revision, len(lw_result.fold_predictions or []), pred_dir,
    )


def _run_encoding_for_subject(
    subject: str,
    dataset_cfg: dict,
    encoding_cfg_base: LppFmriEncodingConfig,
    revision: str,
    per_layer: bool,
    compute_lateralization: bool,
    activations_only: bool = False,
) -> tuple[LppEncodingResult | None, LppLayerwiseEncodingResult | None]:
    """Run encoding for one subject at one checkpoint revision."""
    cfg = LppFmriEncodingConfig(
        bids_root=dataset_cfg["bids_root"],
        subject=subject,
        lang=dataset_cfg.get("lang", "en"),
        bold_source=dataset_cfg.get("bold_source", "derivatives"),
        echo=dataset_cfg.get("echo", 2) if dataset_cfg.get("bold_source", "derivatives") == "raw" else None,
        apply_lbg_fixes=dataset_cfg.get("apply_lbg_fixes", False),
        # Feature extraction
        # CBR-RNN is not an AutoModelForCausalLM: use the encoder's dedicated "cbr_rnn" feature mode
        # (neuroai.features.cbr_rnn, the same extractor every other dataset uses) instead of the HF path.
        feature_mode=("cbr_rnn" if "cbr-rnn" in str(encoding_cfg_base.hf_model_name).lower()
                      else encoding_cfg_base.feature_mode),
        hf_model_name=encoding_cfg_base.hf_model_name,
        hf_revision=revision,
        hf_device=encoding_cfg_base.hf_device,
        hf_batch_size=encoding_cfg_base.hf_batch_size,
        hf_context_words=encoding_cfg_base.hf_context_words,
        hf_max_length=encoding_cfg_base.hf_max_length,
        hf_extraction_mode=encoding_cfg_base.hf_extraction_mode,
        hf_lbg_story_text_zip=encoding_cfg_base.hf_lbg_story_text_zip,
        hf_lbg_stride_delta=encoding_cfg_base.hf_lbg_stride_delta,
        hf_lbg_max_length=encoding_cfg_base.hf_lbg_max_length,
        hf_tokenizer_no_revision=encoding_cfg_base.hf_tokenizer_no_revision,
        hf_trust_remote_code=encoding_cfg_base.hf_trust_remote_code,
        hf_n_params=encoding_cfg_base.hf_n_params,
        hf_untrained=encoding_cfg_base.hf_untrained,
        hf_untrained_seed=encoding_cfg_base.hf_untrained_seed,
        layer_mode="all" if per_layer else "last",
        feature_cache_dir=encoding_cfg_base.feature_cache_dir,
        # Preprocessing
        hrf_model=encoding_cfg_base.hrf_model,
        trim_trs_start=encoding_cfg_base.trim_trs_start,
        trim_trs_end=encoding_cfg_base.trim_trs_end,
        ridge_alphas=encoding_cfg_base.ridge_alphas,
        smoothing_fwhm=encoding_cfg_base.smoothing_fwhm,
        high_pass_hz=encoding_cfg_base.high_pass_hz,
        random_state=encoding_cfg_base.random_state,
        picture_nuisance_regressors=encoding_cfg_base.picture_nuisance_regressors,
        max_voxels=encoding_cfg_base.max_voxels,
        reliable_voxel_fraction=encoding_cfg_base.reliable_voxel_fraction,
        reliable_voxel_scores_path=encoding_cfg_base.reliable_voxel_scores_path,
        mask_from_runs=encoding_cfg_base.mask_from_runs,
        strict_lbg_parity=encoding_cfg_base.strict_lbg_parity,
        min_word_duration_sec=encoding_cfg_base.min_word_duration_sec,
        compute_lateralization=compute_lateralization,
        save_corr_maps_dir=encoding_cfg_base.save_corr_maps_dir,
        fmri_unit=encoding_cfg_base.fmri_unit,
        average_subject_subjects=encoding_cfg_base.average_subject_subjects,
        average_subject_precomputed_dir=encoding_cfg_base.average_subject_precomputed_dir,
        ridge_n_jobs=encoding_cfg_base.ridge_n_jobs,
        layer_n_jobs=encoding_cfg_base.layer_n_jobs,
        regression_device=encoding_cfg_base.regression_device,
        n_layer_workers=encoding_cfg_base.n_layer_workers,
        brain_mask_path=encoding_cfg_base.brain_mask_path,
        roi_masks_dir=encoding_cfg_base.roi_masks_dir,
        save_word_predictions=encoding_cfg_base.save_word_predictions,
    )
    encoder = LppFmriEncoder(cfg)
    if activations_only:
        encoder.extract_and_cache_only()
        return None, None
    if per_layer:
        lw_result = encoder.run_layerwise()
        return None, lw_result
    else:
        flat_result = encoder.run_leave_one_run_out()
        return flat_result, None


def _average_flat_results(results: list[LppEncodingResult]) -> dict[str, float]:
    """Average scalar scores across subjects."""
    def _nanmean(vals: list[float]) -> float:
        arr = [v for v in vals if np.isfinite(v)]
        return float(np.mean(arr)) if arr else float("nan")

    return {
        "mean_pearson": _nanmean([r.overall_mean_pearson for r in results]),
        "mean_spearman": _nanmean([getattr(r, "overall_mean_spearman", float("nan")) for r in results]),
        "mean_pearson_left": _nanmean([r.overall_mean_pearson_left for r in results]),
        "mean_pearson_right": _nanmean([r.overall_mean_pearson_right for r in results]),
        "lateralization_index": _nanmean([r.overall_lateralization_index for r in results]),
        "n_subjects": len(results),
    }


def _average_layerwise_results(results: list[LppLayerwiseEncodingResult]) -> dict[str, Any]:
    """Average per-layer scores across subjects."""
    if not results:
        return {}
    n_layers = results[0].n_layers

    def _layer_nanmean(attr: str, li: int) -> float:
        vals = []
        for r in results:
            lst = getattr(r, attr, [])
            if li < len(lst) and np.isfinite(lst[li]):
                vals.append(lst[li])
        return float(np.mean(vals)) if vals else float("nan")

    avg: dict[str, Any] = {"n_layers": n_layers, "n_subjects": len(results)}
    for li in range(n_layers):
        avg[f"layer_{li}_pearson"] = _layer_nanmean("layer_mean_pearson", li)
        avg[f"layer_{li}_spearman"] = _layer_nanmean("layer_mean_spearman", li)
        avg[f"layer_{li}_pearson_left"] = _layer_nanmean("layer_mean_pearson_left", li)
        avg[f"layer_{li}_pearson_right"] = _layer_nanmean("layer_mean_pearson_right", li)
        if np.isfinite(avg[f"layer_{li}_pearson_left"]) and np.isfinite(avg[f"layer_{li}_pearson_right"]):
            avg[f"layer_{li}_lat"] = avg[f"layer_{li}_pearson_left"] - avg[f"layer_{li}_pearson_right"]
        else:
            avg[f"layer_{li}_lat"] = float("nan")

    # Best-layer summary (best layer chosen by Pearson; Spearman reported at the
    # same layer so mean_pearson and mean_spearman refer to one model/layer).
    layer_scores = [avg[f"layer_{li}_pearson"] for li in range(n_layers)]
    best_li = int(np.nanargmax(layer_scores)) if any(np.isfinite(layer_scores)) else 0
    avg["best_layer"] = best_li
    avg["mean_pearson"] = layer_scores[best_li]
    avg["mean_spearman"] = avg.get(f"layer_{best_li}_spearman", float("nan"))
    avg["mean_pearson_left"] = avg.get(f"layer_{best_li}_pearson_left", float("nan"))
    avg["mean_pearson_right"] = avg.get(f"layer_{best_li}_pearson_right", float("nan"))
    avg["lateralization_index"] = avg.get(f"layer_{best_li}_lat", float("nan"))
    for li in range(n_layers):
        alphas = []
        for r in results:
            if li < len(r.layer_mean_alphas) and np.isfinite(r.layer_mean_alphas[li]):
                alphas.append(r.layer_mean_alphas[li])
        avg[f"layer_{li}_alpha"] = float(np.mean(alphas)) if alphas else float("nan")
    return avg


def _checkpoint_sharding_enabled() -> bool:
    """One SLURM task per step → write under checkpoints/<revision>/ (no shared CSV races)."""
    return os.environ.get("LPP_CHECKPOINT_SHARD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _checkpoint_shard_dir(model_dir: Path, revision: str) -> Path:
    out = model_dir / "checkpoints" / _safe_revision(revision)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _merge_checkpoint_shards(model_dir: Path) -> int:
    """Merge per-revision shard CSVs into model_dir/*.csv (for parallel step array jobs)."""
    shard_root = model_dir / "checkpoints"
    if not shard_root.is_dir():
        return 0

    names = [
        "lpp_checkpoint_scores.csv",
        "lpp_checkpoint_layerwise.csv",
        "lpp_checkpoint_by_fold.csv",
        "lpp_checkpoint_by_roi.csv",
        "lpp_checkpoint_by_parcel.csv",
        "lpp_checkpoint_by_voxel.csv",
        "lpp_ridge_alpha_summary.csv",
    ]
    n_merged = 0
    for name in names:
        parts: list[pd.DataFrame] = []
        for sub in sorted(shard_root.iterdir()):
            if not sub.is_dir():
                continue
            p = sub / name
            if p.is_file():
                try:
                    parts.append(pd.read_csv(p))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skip shard %s: %s", p, exc)
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True)
        if "revision" in merged.columns:
            merged = merged.drop_duplicates(subset=["revision"], keep="last")
        if name == "lpp_checkpoint_layerwise.csv" and {"revision", "layer"}.issubset(
            merged.columns
        ):
            merged = merged.drop_duplicates(subset=["revision", "layer"], keep="last")
        out = model_dir / name
        merged.to_csv(out, index=False)
        logger.info("Merged %d shard file(s) → %s (%d rows)", len(parts), out, len(merged))
        n_merged += 1
    return n_merged


def _sync_rows_by_revision(
    rows: list[dict[str, Any]],
    revision: str,
) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("revision")) != revision]


def _read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    try:
        with _csv_file_lock(csv_path):
            return pd.read_csv(csv_path).to_dict("records")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", csv_path, exc)
        return []


def _write_csv_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    with _csv_file_lock(csv_path):
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        else:
            csv_path.unlink(missing_ok=True)


def _snap_alpha_to_grid(alpha: float, grid: np.ndarray, *, rtol: float = 1e-4) -> float:
    """Map a chosen alpha to the nearest value on the configured ridge grid."""
    if not np.isfinite(alpha) or grid.size == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(grid - alpha)))
    if np.isclose(alpha, grid[idx], rtol=rtol):
        return float(grid[idx])
    return float(alpha)


def _collect_chosen_alphas(
    lw_results: list[LppLayerwiseEncodingResult],
    ridge_grid: np.ndarray | None = None,
) -> list[float]:
    """All nested-CV alpha picks (one per LOSO fold × layer)."""
    alphas: list[float] = []
    grid = np.asarray(ridge_grid, dtype=float) if ridge_grid is not None else None
    for res in lw_results:
        for fd in res.fold_details:
            if np.isfinite(fd.best_alpha):
                a = float(fd.best_alpha)
                if grid is not None and grid.size:
                    a = _snap_alpha_to_grid(a, grid)
                alphas.append(a)
    return alphas


def _alpha_counter(alphas: list[float]) -> Counter[float]:
    return Counter(alphas)


def _suggest_reduced_ridge_grid(
    counts: Counter[float],
    *,
    coverage: float = 0.95,
    min_alphas: int = 3,
    max_alphas: int = 8,
) -> list[float]:
    """Smallest sorted alpha subset covering ``coverage`` fraction of picks."""
    if not counts:
        return []
    total = sum(counts.values())
    ranked = counts.most_common()
    kept: list[float] = []
    cum = 0
    for alpha, n in ranked:
        kept.append(alpha)
        cum += n
        if cum / total >= coverage or len(kept) >= max_alphas:
            break
    kept = sorted(set(kept))
    if len(kept) < min_alphas and len(ranked) >= min_alphas:
        kept = sorted(a for a, _ in ranked[:min_alphas])
    return kept


def _format_alpha_table(counts: Counter[float], *, top_n: int = 16) -> str:
    if not counts:
        return "  (no alpha picks recorded)"
    total = sum(counts.values())
    lines = []
    for alpha, n in counts.most_common(top_n):
        pct = 100.0 * n / total
        lines.append(f"  {alpha:12.4g}  n={n:4d}  ({pct:5.1f}%)")
    return "\n".join(lines)


def _log_and_save_alpha_choices(
    model_dir: Path,
    model_label: str,
    ckpt: "CheckpointItem",
    lw_results: list[LppLayerwiseEncodingResult],
    ridge_grid: np.ndarray,
    *,
    shard_exclusive: bool = False,
) -> None:
    """Print chosen ridge alphas and append a per-checkpoint summary row."""
    alphas = _collect_chosen_alphas(lw_results, ridge_grid)
    if not alphas:
        logger.info("[%s] %s — no chosen alphas to summarize", model_label, ckpt.revision)
        return

    counts = _alpha_counter(alphas)
    mode_alpha = counts.most_common(1)[0][0]
    median_alpha = float(np.median(alphas))
    suggested = _suggest_reduced_ridge_grid(counts)

    # Per-layer means (across LOSO folds)
    layer_lines: list[str] = []
    for res in lw_results:
        for li, a in enumerate(res.layer_mean_alphas):
            if np.isfinite(a):
                layer_lines.append(f"L{li}={a:.4g}")
    layer_summary = ", ".join(layer_lines[:8])
    if len(layer_lines) > 8:
        layer_summary += f", … (+{len(layer_lines) - 8} layers)"

    header = (
        f"\n{'=' * 72}\n"
        f"[{model_label}] ridge alpha choices — {ckpt.revision} (step={ckpt.step}, "
        f"n_picks={len(alphas)} = {len(lw_results)} subject(s) × folds × layers)\n"
        f"{'=' * 72}\n"
        f"Mode (most chosen): {mode_alpha:.4g}   Median: {median_alpha:.4g}\n"
        f"Per-layer means (LOSO-averaged): {layer_summary or 'n/a'}\n"
        f"Counts (all fold×layer picks):\n{_format_alpha_table(counts)}\n"
        f"Suggested reduced ridge_alphas (~95% coverage, copy into YAML):\n"
        f"  {suggested}\n"
        f"{'=' * 72}\n"
    )
    print(header, flush=True)
    logger.info(
        "[%s] %s alpha mode=%.4g median=%.4g suggested_grid=%s",
        model_label,
        ckpt.revision,
        mode_alpha,
        median_alpha,
        suggested,
    )

    summary_csv = model_dir / "lpp_ridge_alpha_summary.csv"
    row = {
        "revision": ckpt.revision,
        "step": ckpt.step,
        "tokens_seen": ckpt.tokens_seen,
        "n_picks": len(alphas),
        "mode_alpha": mode_alpha,
        "median_alpha": median_alpha,
        "suggested_ridge_alphas": json.dumps(suggested),
        "alpha_counts_json": json.dumps({str(k): v for k, v in counts.items()}),
    }
    if shard_exclusive:
        _write_csv_rows(summary_csv, [row])
    else:
        existing = _sync_rows_by_revision(_read_csv_rows(summary_csv), ckpt.revision)
        existing.append(row)
        _write_csv_rows(summary_csv, existing)


def _print_model_alpha_aggregate(model_dir: Path, model_label: str) -> None:
    """After all checkpoints: print most-chosen alphas across the whole model run."""
    summary_csv = model_dir / "lpp_ridge_alpha_summary.csv"
    fold_csv = model_dir / "lpp_checkpoint_by_fold.csv"
    counts: Counter[float] = Counter()

    if fold_csv.exists():
        try:
            fold_df = pd.read_csv(fold_csv)
            if "best_alpha" in fold_df.columns:
                for a in fold_df["best_alpha"].dropna():
                    if np.isfinite(a):
                        counts[float(a)] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s for alpha aggregate: %s", fold_csv, exc)

    if not counts and summary_csv.exists():
        return

    if not counts:
        return

    suggested = _suggest_reduced_ridge_grid(counts)
    header = (
        f"\n{'#' * 72}\n"
        f"[{model_label}] AGGREGATE ridge alpha usage (all checkpoints so far)\n"
        f"Total picks: {sum(counts.values())}\n"
        f"{_format_alpha_table(counts, top_n=16)}\n"
        f"Suggested ridge_alphas for faster reruns:\n"
        f"  {suggested}\n"
        f"{'#' * 72}\n"
    )
    print(header, flush=True)
    logger.info("[%s] aggregate suggested ridge_alphas=%s", model_label, suggested)


def _save_granular_checkpoint_outputs(
    model_dir: Path,
    model_id: str,
    model_label: str,
    ckpt: "CheckpointItem",
    lang: str,
    lw_results: list[LppLayerwiseEncodingResult],
    *,
    save_voxel_scores: bool,
    ho_parcelization: "HarvardOxfordParcelization | None" = None,
    shard_exclusive: bool = False,
) -> None:
    """Append granular tables for one checkpoint (replace same revision)."""
    if not lw_results:
        return

    base_meta = {
        "model_id": model_id,
        "model_label": model_label,
        "revision": ckpt.revision,
        "step": ckpt.step,
        "tokens_seen": ckpt.tokens_seen,
        "lang": lang,
    }

    fold_csv = model_dir / "lpp_checkpoint_by_fold.csv"
    roi_csv = model_dir / "lpp_checkpoint_by_roi.csv"
    parcel_csv = model_dir / "lpp_checkpoint_by_parcel.csv"
    voxel_csv = model_dir / "lpp_checkpoint_by_voxel.csv"

    if shard_exclusive:
        fold_rows: list[dict[str, Any]] = []
        roi_rows: list[dict[str, Any]] = []
        parcel_rows: list[dict[str, Any]] = []
        voxel_rows: list[dict[str, Any]] = []
    else:
        fold_rows = _sync_rows_by_revision(_read_csv_rows(fold_csv), ckpt.revision)
        roi_rows = _sync_rows_by_revision(_read_csv_rows(roi_csv), ckpt.revision)
        parcel_rows = _sync_rows_by_revision(_read_csv_rows(parcel_csv), ckpt.revision)
        voxel_rows = _sync_rows_by_revision(_read_csv_rows(voxel_csv), ckpt.revision)

    from neuroai.roi.lpp import parcel_summary_from_correlations, roi_summary_from_correlations

    for res in lw_results:
        subj = res.subject
        for fd in res.fold_details:
            fold_rows.append({**base_meta, "subject": subj, **fd.to_dict()})

        if res.voxel_roi_labels is None:
            roi_labels_sel = None
        else:
            roi_labels_sel = res.voxel_roi_labels

        for li in range(res.n_layers):
            if li >= len(res.layer_voxel_mean_pearson):
                continue
            corrs = res.layer_voxel_mean_pearson[li]
            if corrs.size == 0:
                continue

            if roi_labels_sel is not None and res.voxel_global_ix.size:
                for roi_row in roi_summary_from_correlations(
                    corrs,
                    roi_labels_sel,
                ):
                    roi_rows.append(
                        {
                            **base_meta,
                            "subject": subj,
                            "layer": li,
                            "best_alpha": res.layer_mean_alphas[li]
                            if li < len(res.layer_mean_alphas)
                            else float("nan"),
                            **roi_row,
                        }
                    )

            if ho_parcelization is not None and res.voxel_global_ix.size:
                for parcel_row in parcel_summary_from_correlations(
                    corrs,
                    res.voxel_global_ix,
                    ho_parcelization,
                ):
                    parcel_rows.append(
                        {
                            **base_meta,
                            "subject": subj,
                            "layer": li,
                            "best_alpha": res.layer_mean_alphas[li]
                            if li < len(res.layer_mean_alphas)
                            else float("nan"),
                            **parcel_row,
                        }
                    )

            if save_voxel_scores and res.voxel_global_ix.size:
                roi_sel = (
                    roi_labels_sel
                    if roi_labels_sel is not None
                    else np.full(corrs.shape[0], "", dtype=object)
                )
                for vi, (gix, r_val) in enumerate(
                    zip(res.voxel_global_ix, corrs, strict=False)
                ):
                    voxel_rows.append(
                        {
                            **base_meta,
                            "subject": subj,
                            "layer": li,
                            "voxel_global_ix": int(gix),
                            "voxel_ix": vi,
                            "roi": str(roi_sel[vi]) if vi < len(roi_sel) else "",
                            "pearson_r": float(r_val),
                            "best_alpha": res.layer_mean_alphas[li]
                            if li < len(res.layer_mean_alphas)
                            else float("nan"),
                        }
                    )

    _write_csv_rows(fold_csv, fold_rows)
    logger.info(
        "[%s] lpp_checkpoint_by_fold → %d row(s) (%s)",
        model_label, len(fold_rows), fold_csv,
    )
    if roi_rows:
        _write_csv_rows(roi_csv, roi_rows)
        logger.info(
            "[%s] lpp_checkpoint_by_roi → %d row(s) (%s)",
            model_label, len(roi_rows), roi_csv,
        )
    if parcel_rows:
        _write_csv_rows(parcel_csv, parcel_rows)
        logger.info(
            "[%s] lpp_checkpoint_by_parcel → %d row(s) (%s)",
            model_label, len(parcel_rows), parcel_csv,
        )
    if save_voxel_scores and voxel_rows:
        _write_csv_rows(voxel_csv, voxel_rows)
        logger.info(
            "[%s] lpp_checkpoint_by_voxel → %d row(s) (%s)",
            model_label, len(voxel_rows), voxel_csv,
        )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

@contextmanager
def _csv_file_lock(csv_path: Path):
    """Exclusive lock for concurrent sweep workers (e.g. one SLURM step per task)."""
    lock_path = csv_path.with_suffix(f"{csv_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_existing_csv(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s — starting fresh", csv_path, exc)
    return pd.DataFrame()


def _already_computed(df: pd.DataFrame, revision: str) -> bool:
    if df.empty or "revision" not in df.columns:
        return False
    return bool((df["revision"] == revision).any())


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    with _csv_file_lock(csv_path):
        df_new = pd.DataFrame([row])
        if csv_path.exists():
            df_new.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Per-model sweep
# ---------------------------------------------------------------------------

def _run_model_sweep(
    model_cfg: dict,
    dataset_cfg: dict,
    enc_cfg_base: LppFmriEncodingConfig,
    subjects: list[str],
    checkpoints: list[CheckpointItem],
    analysis_cfg: dict,
    output_cfg: dict,
    output_dir: Path,
    dry_run: bool = False,
    ho_parcelization: Any = None,
    generate_plots: bool = True,
    activations_only: bool = False,
) -> None:
    model_id = model_cfg["model_id"]
    model_label = model_cfg.get("label", model_id.split("/")[-1])
    per_layer = analysis_cfg.get("per_layer", False)
    compute_lat = analysis_cfg.get("lateralization", False)

    model_dir = output_dir / _safe_model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    use_shards = _checkpoint_sharding_enabled()

    flat_csv = model_dir / "lpp_checkpoint_scores.csv"
    lw_csv = model_dir / "lpp_checkpoint_layerwise.csv"
    flat_df = _load_existing_csv(flat_csv)
    lw_df = _load_existing_csv(lw_csv)

    if use_shards:
        logger.info(
            "LPP_CHECKPOINT_SHARD=1: each revision writes to %s/checkpoints/<revision>/",
            model_dir,
        )

    n_layers_hint = model_cfg.get("n_layers")
    if per_layer and n_layers_hint:
        logger.info(
            "Output directory (created now, CSVs fill after each checkpoint): %s",
            model_dir,
        )
        logger.info(
            "per_layer=True: each training checkpoint runs leave-one-run-out over "
            "all story sections (~9 folds) × %d layers. Each layer rebuilds HRF "
            "designs (feature_dim ≈ hidden size) and fits multi-output Ridge on "
            "max_voxels voxels — this can take many minutes per fold; only one "
            "LOSO log line per fold. "
            "lpp_checkpoint_scores.csv / lpp_checkpoint_layerwise.csv are written "
            "only after all %d encoding job(s) finish that checkpoint — not after each fold.",
            n_layers_hint,
            len(subjects),
        )
    else:
        logger.info(
            "Output directory: %s — rows append after each checkpoint (all encoding jobs done).",
            model_dir,
        )

    logger.info(
        "=== Model: %s | %d encoding job(s) | %d checkpoints | per_layer=%s ===",
        model_label, len(subjects), len(checkpoints), per_layer,
    )

    for ckpt in checkpoints:
        ckpt_dir = (
            _checkpoint_shard_dir(model_dir, ckpt.revision) if use_shards else model_dir
        )
        ckpt_flat_csv = ckpt_dir / "lpp_checkpoint_scores.csv"
        ckpt_lw_csv = ckpt_dir / "lpp_checkpoint_layerwise.csv"
        ckpt_flat_df = _load_existing_csv(ckpt_flat_csv)
        if not use_shards:
            ckpt_flat_df = flat_df
        if _already_computed(ckpt_flat_df, ckpt.revision):
            logger.info("[%s] Skipping already-computed revision %s", model_label, ckpt.revision)
            continue

        if dry_run:
            logger.info(
                "[DRY RUN] Would compute revision=%s step=%d tokens=%d",
                ckpt.revision, ckpt.step, ckpt.tokens_seen,
            )
            continue

        logger.info(
            "[%s] revision=%s step=%d tokens=%s",
            model_label, ckpt.revision, ckpt.step,
            f"{ckpt.tokens_seen / 1e9:.2f}B" if ckpt.tokens_seen > 0 else "0",
        )
        if per_layer:
            logger.info(
                "[%s] After cache loads: full LOSO×layers Ridge — CSVs append only when "
                "this checkpoint completes for all encoding jobs.",
                model_label,
            )
        else:
            logger.info(
                "[%s] CSVs append when this checkpoint completes for all encoding jobs.",
                model_label,
            )
        sys.stdout.flush()

        flat_results: list[LppEncodingResult] = []
        lw_results: list[LppLayerwiseEncodingResult] = []

        # Override device / model load options from model config
        device = model_cfg.get("device", enc_cfg_base.hf_device)
        enc_cfg_base.hf_device = device
        enc_cfg_base.hf_model_name = model_cfg.get("load_model_id", model_id)
        enc_cfg_base.hf_trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
        enc_cfg_base.hf_batch_size = model_cfg.get("batch_size", enc_cfg_base.hf_batch_size)
        enc_cfg_base.hf_context_words = model_cfg.get("context_words", enc_cfg_base.hf_context_words)
        enc_cfg_base.hf_max_length = model_cfg.get("max_length", enc_cfg_base.hf_max_length)
        enc_cfg_base.hf_n_params = model_cfg.get("n_params")

        if model_cfg.get("trust_remote_code"):
            logger.info(
                "[%s] trust_remote_code=True for %s",
                model_label,
                enc_cfg_base.hf_model_name,
            )

        # Feature extraction doesn't depend on fMRI subject, only on the
        # stimulus text -- one subject is enough to populate the cache.
        subjects_to_run = subjects[:1] if activations_only else subjects
        for subject in subjects_to_run:
            try:
                flat_r, lw_r = _run_encoding_for_subject(
                    subject=subject,
                    dataset_cfg=dataset_cfg,
                    encoding_cfg_base=enc_cfg_base,
                    revision=ckpt.revision,
                    per_layer=per_layer,
                    compute_lateralization=compute_lat,
                    activations_only=activations_only,
                )
                if flat_r is not None:
                    flat_results.append(flat_r)
                if lw_r is not None:
                    lw_results.append(lw_r)
                    if lw_r.fold_predictions:
                        _save_lpp_word_predictions(
                            lw_r, model_dir, ckpt.revision, subject,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[%s] subject=%s revision=%s FAILED: %s",
                    model_label, subject, ckpt.revision, exc,
                )
                traceback.print_exc()

        # --- Aggregate and save flat results ---
        if not per_layer and flat_results:
            avg = _average_flat_results(flat_results)
            row: dict[str, Any] = {
                "model_id": model_id,
                "model_label": model_label,
                "revision": ckpt.revision,
                "step": ckpt.step,
                "tokens_seen": ckpt.tokens_seen,
                "lang": dataset_cfg.get("lang", "en"),
                **avg,
            }
            if use_shards:
                _write_csv_rows(ckpt_flat_csv, [row])
            else:
                _append_row(flat_csv, row)
                flat_df = _load_existing_csv(flat_csv)
            logger.info(
                "[%s] step=%d mean_pearson=%.4f lat=%.4f → %s",
                model_label, ckpt.step,
                avg.get("mean_pearson", float("nan")),
                avg.get("lateralization_index", float("nan")),
                ckpt_flat_csv,
            )

        # --- Aggregate and save layerwise results ---
        if per_layer and lw_results:
            lw_avg = _average_layerwise_results(lw_results)
            row = {
                "model_id": model_id,
                "model_label": model_label,
                "revision": ckpt.revision,
                "step": ckpt.step,
                "tokens_seen": ckpt.tokens_seen,
                "lang": dataset_cfg.get("lang", "en"),
                **lw_avg,
            }
            n_layers = lw_avg.get("n_layers", 0)
            lw_rows_out: list[dict[str, Any]] = []
            for li in range(n_layers):
                lw_rows_out.append(
                    {
                        "model_id": model_id,
                        "model_label": model_label,
                        "revision": ckpt.revision,
                        "step": ckpt.step,
                        "tokens_seen": ckpt.tokens_seen,
                        "lang": dataset_cfg.get("lang", "en"),
                        "layer": li,
                        "mean_pearson": lw_avg.get(f"layer_{li}_pearson", float("nan")),
                        "mean_pearson_left": lw_avg.get(f"layer_{li}_pearson_left", float("nan")),
                        "mean_pearson_right": lw_avg.get(f"layer_{li}_pearson_right", float("nan")),
                        "lateralization_index": lw_avg.get(f"layer_{li}_lat", float("nan")),
                        "best_alpha": lw_avg.get(f"layer_{li}_alpha", float("nan")),
                    }
                )
            if use_shards:
                _write_csv_rows(ckpt_flat_csv, [row])
                _write_csv_rows(ckpt_lw_csv, lw_rows_out)
            else:
                _append_row(flat_csv, row)
                flat_df = _load_existing_csv(flat_csv)
                for lw_row in lw_rows_out:
                    _append_row(lw_csv, lw_row)
                lw_df = _load_existing_csv(lw_csv)

            if output_cfg.get("save_granular_outputs", True):
                try:
                    _save_granular_checkpoint_outputs(
                        ckpt_dir,
                        model_id,
                        model_label,
                        ckpt,
                        dataset_cfg.get("lang", "en"),
                        lw_results,
                        save_voxel_scores=bool(output_cfg.get("save_voxel_scores", True)),
                        ho_parcelization=ho_parcelization,
                        shard_exclusive=use_shards,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "[%s] granular CSV save failed for %s: %s",
                        model_label,
                        ckpt.revision,
                        exc,
                    )
                    traceback.print_exc()

            _log_and_save_alpha_choices(
                ckpt_dir,
                model_label,
                ckpt,
                lw_results,
                np.asarray(enc_cfg_base.ridge_alphas, dtype=float),
                shard_exclusive=use_shards,
            )
            _print_model_alpha_aggregate(model_dir, model_label)

            logger.info(
                "[%s] step=%d best_layer=%d mean_pearson=%.4f lat=%.4f",
                model_label, ckpt.step,
                lw_avg.get("best_layer", -1),
                lw_avg.get("mean_pearson", float("nan")),
                lw_avg.get("lateralization_index", float("nan")),
            )

        # Free GPU memory between checkpoints
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    if dry_run or not generate_plots:
        return

    # --- Per-model plots ---
    logger.info("[%s] Generating per-model plots ...", model_label)
    flat_df_now = _load_existing_csv(flat_csv)
    if not flat_df_now.empty:
        _plot_model(model_dir, flat_df_now, model_label, analysis_cfg)

    lw_df_now = _load_existing_csv(lw_csv)
    if not lw_df_now.empty and per_layer:
        _plot_model_layerwise(model_dir, lw_df_now, model_label)

    if per_layer:
        _print_model_alpha_aggregate(model_dir, model_label)


def _plot_model(
    model_dir: Path,
    df: pd.DataFrame,
    model_label: str,
    analysis_cfg: dict,
) -> None:
    x_col = "tokens_seen" if "tokens_seen" in df.columns else "step"
    if not df[x_col].dropna().empty:
        plot_lpp_training_curve(
            df, x_col=x_col, model_label=model_label,
            save_path=model_dir / "lpp_training_curve.png",
        )
        if analysis_cfg.get("lateralization", False):
            plot_lpp_lateralization_curve(
                df, x_col=x_col, model_label=model_label,
                save_path=model_dir / "lpp_lateralization_curve.png",
            )


def _plot_model_layerwise(
    model_dir: Path,
    lw_df: pd.DataFrame,
    model_label: str,
) -> None:
    x_col = "tokens_seen" if "tokens_seen" in lw_df.columns else "step"
    if "layer" in lw_df.columns and not lw_df.empty:
        plot_lpp_layer_heatmap(
            lw_df, x_col=x_col, model_label=model_label,
            save_path=model_dir / "lpp_layer_heatmap.png",
        )


# ---------------------------------------------------------------------------
# Summary (all models)
# ---------------------------------------------------------------------------

def _generate_summary(
    output_dir: Path,
    model_cfgs: list[dict],
    analysis_cfg: dict,
) -> None:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_flat_dfs: list[pd.DataFrame] = []
    all_lw_data: dict[str, pd.DataFrame] = {}

    for model_cfg in model_cfgs:
        model_id = model_cfg["model_id"]
        model_dir = output_dir / _safe_model_dir(model_id)
        _merge_checkpoint_shards(model_dir)
        flat_csv = model_dir / "lpp_checkpoint_scores.csv"
        lw_csv = model_dir / "lpp_checkpoint_layerwise.csv"

        if flat_csv.exists():
            df = pd.read_csv(flat_csv)
            if not df.empty:
                all_flat_dfs.append(df)

        if analysis_cfg.get("per_layer", False) and lw_csv.exists():
            df_lw = pd.read_csv(lw_csv)
            if not df_lw.empty:
                label = model_cfg.get("label", model_id.split("/")[-1])
                all_lw_data[label] = df_lw

    if not all_flat_dfs:
        logger.info("No results to summarise yet.")
        return

    combined = pd.concat(all_flat_dfs, ignore_index=True)
    combined_csv = summary_dir / "all_models_lpp_scores.csv"
    combined.to_csv(combined_csv, index=False)
    logger.info("Saved summary CSV → %s (%d rows)", combined_csv, len(combined))

    x_col = "tokens_seen" if "tokens_seen" in combined.columns else "step"

    # All-models training curve
    plot_lpp_all_models_panel(
        combined, x_col=x_col,
        save_path=summary_dir / "all_models_lpp_training_curves.png",
    )

    # Lateralization panel
    if analysis_cfg.get("lateralization", False) and "lateralization_index" in combined.columns:
        plot_lpp_lateralization_panel(
            combined, x_col=x_col,
            save_path=summary_dir / "all_models_lpp_lateralization.png",
        )

    # Layer heatmap grid
    if all_lw_data:
        plot_lpp_layer_heatmap_grid(
            all_lw_data, x_col=x_col,
            save_path=summary_dir / "all_models_lpp_layer_heatmaps_grid.png",
        )

    logger.info("Summary plots saved to %s", summary_dir)


# ---------------------------------------------------------------------------
# Encoding config builder from YAML
# ---------------------------------------------------------------------------

def _build_enc_cfg_base(enc_yaml: dict, dataset_cfg: dict, cache_cfg: dict, out_cfg: dict) -> LppFmriEncodingConfig:
    """Build a template LppFmriEncodingConfig from YAML sub-blocks.

    The model-specific fields (model_name, revision, device, batch_size,
    context_words, max_length) are overridden per-model/per-checkpoint in the
    sweep loop.
    """
    ridge_alphas_raw = enc_yaml.get("ridge_alphas", None)
    if ridge_alphas_raw is not None:
        ridge_alphas = np.asarray(ridge_alphas_raw, dtype=float)
    else:
        ridge_alphas = np.logspace(2.0, 7.0, 16)

    return LppFmriEncodingConfig(
        bids_root=dataset_cfg["bids_root"],
        subject="__placeholder__",       # overridden per subject
        lang=dataset_cfg.get("lang", "en"),
        bold_source=dataset_cfg.get("bold_source", "derivatives"),
        echo=dataset_cfg.get("echo", 2),
        apply_lbg_fixes=dataset_cfg.get("apply_lbg_fixes", False),
        # Feature placeholders — overridden per checkpoint
        feature_mode=enc_yaml.get("feature_mode", "huggingface"),
        hf_model_name="__placeholder__",
        hf_revision=None,
        hf_device="cpu",
        hf_batch_size=4,
        hf_context_words=enc_yaml.get("hf_context_words", 48),
        hf_max_length=enc_yaml.get("hf_max_length", 512),
        hf_extraction_mode=enc_yaml.get("hf_extraction_mode", "context_window"),
        hf_lbg_story_text_zip=enc_yaml.get("hf_lbg_story_text_zip", None),
        hf_lbg_stride_delta=int(enc_yaml.get("hf_lbg_stride_delta", 64)),
        hf_lbg_max_length=enc_yaml.get("hf_lbg_max_length", None),
        hf_tokenizer_no_revision=bool(enc_yaml.get("hf_tokenizer_no_revision", False)),
        hf_trust_remote_code=bool(enc_yaml.get("hf_trust_remote_code", False)),
        layer_mode="last",               # overridden per model
        feature_cache_dir=cache_cfg.get("features_dir", None),
        # Preprocessing
        hrf_model=enc_yaml.get("hrf_model", "glover"),
        trim_trs_start=enc_yaml.get("trim_trs_start", 10),
        trim_trs_end=enc_yaml.get("trim_trs_end", 10),
        ridge_alphas=ridge_alphas,
        smoothing_fwhm=enc_yaml.get("smoothing_fwhm", 4.0),
        high_pass_hz=enc_yaml.get("high_pass_hz", 1.0 / 128.0),
        random_state=enc_yaml.get("random_state", 0),
        picture_nuisance_regressors=enc_yaml.get("picture_nuisance_regressors", False),
        max_voxels=enc_yaml.get("max_voxels", 4000),
        reliable_voxel_fraction=enc_yaml.get("reliable_voxel_fraction", None),
        reliable_voxel_scores_path=enc_yaml.get("reliable_voxel_scores_path", None),
        mask_from_runs=enc_yaml.get("mask_from_runs", 3),
        strict_lbg_parity=bool(enc_yaml.get("strict_lbg_parity", False)),
        min_word_duration_sec=float(enc_yaml.get("min_word_duration_sec", 0.05)),
        compute_lateralization=False,    # overridden per analysis_cfg
        save_corr_maps_dir=None,
        fmri_unit=str(dataset_cfg.get("fmri_unit", "single_subject")).lower().strip(),
        average_subject_subjects=None,
        average_subject_precomputed_dir=dataset_cfg.get(
            "average_subject_precomputed_dir", None
        ),
        ridge_n_jobs=int(enc_yaml.get("ridge_n_jobs", 1)),
        layer_n_jobs=int(enc_yaml.get("layer_n_jobs", 1)),
        regression_device=str(enc_yaml.get("regression_device", "cpu")).lower().strip(),
        n_layer_workers=int(enc_yaml.get("n_layer_workers", 8)),
        brain_mask_path=dataset_cfg.get("lbg_brain_mask_path")
        or enc_yaml.get("brain_mask_path"),
        roi_masks_dir=enc_yaml.get("roi_masks_dir"),
        save_word_predictions=bool(enc_yaml.get("save_word_predictions", True)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LPP fMRI checkpoint sweep (YAML-driven)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/embedding/neural/lpp.yaml"),
        help="Path to YAML config (default: configs/embedding/neural/lpp.yaml)",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Override output directory from YAML.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device for all models (e.g. cuda, cpu).",
    )
    p.add_argument(
        "--ridge_n_jobs",
        type=int,
        default=None,
        help="Override encoding.ridge_n_jobs (parallel alpha grid; joblib).",
    )
    p.add_argument(
        "--layer_n_jobs",
        type=int,
        default=None,
        help="Override encoding.layer_n_jobs (parallel layers per LOSO fold; CPU path).",
    )
    p.add_argument(
        "--regression_device",
        type=str,
        default=None,
        choices=("cpu", "cuda"),
        help="Ridge backend: cpu (sklearn/joblib) or cuda (himalaya multi-output).",
    )
    p.add_argument(
        "--n_layer_workers",
        type=int,
        default=None,
        help="Override encoding.n_layer_workers (CUDA stream workers when regression_device=cuda).",
    )
    p.add_argument(
        "--from-registry",
        action="store_true",
        help="Load models from neuroai.models.registry (use with --family).",
    )
    p.add_argument(
        "--family",
        type=str,
        default=None,
        help="Registry family key (e.g. gpt2, stickbreaking, babylm).",
    )
    p.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="SLURM array task index (one model×checkpoint per task).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Restrict sweep to these HuggingFace model_id values (must match "
            "`model_id` entries under `models:` in the YAML). Space-separated; "
            "use one id per job for parallel cluster runs."
        ),
    )
    p.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Override training steps for Pythia-family models.",
    )
    p.add_argument(
        "--activations_only",
        action="store_true",
        help=(
            "Extract + cache raw per-word features only (one subject, all "
            "layers) -- skips BOLD loading and ridge/LOSO encoding entirely. "
            "For populating feature_cache_dir cheaply (RSA raw-embedding-"
            "cache backfill), not for scoring."
        ),
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print checkpoint plan without running any computation.",
    )
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip all plot generation.",
    )
    p.add_argument(
        "--skip_summary",
        action="store_true",
        help="Skip final summary generation (use on per-model workers).",
    )
    p.add_argument(
        "--summary_only",
        action="store_true",
        help=(
            "Only read per-model CSVs under out_dir and regenerate summary/ "
            "(all_models_lpp_scores.csv + plots). Uses every model in the YAML "
            "`models:` list; missing subdirs are skipped. No encoding run."
        ),
    )
    p.add_argument(
        "--untrained",
        action="store_true",
        help="Use randomly initialised weights (same architecture) instead of "
             "pretrained checkpoints -- untrained baseline for brain scores.",
    )
    p.add_argument("--untrained_seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("Loading config from %s", args.config)
    cfg = load_config(args.config)

    feat_cache = _resolve_feature_cache_dir(cfg)
    if feat_cache is not None and args.untrained:
        feat_cache = feat_cache.parent / f"{feat_cache.name}_untrained"
        feat_cache.mkdir(parents=True, exist_ok=True)
        cfg.setdefault("cache", {})["features_dir"] = str(feat_cache)
    if feat_cache is not None:
        logger.info("LPP feature cache directory: %s", feat_cache)
    else:
        logger.warning(
            "LPP feature cache disabled (set cache.features_dir or LPP_FEATURE_CACHE_DIR)"
        )

    out_cfg_pre = cfg.get("output", {})
    out_dir_pre = Path(args.out_dir or out_cfg_pre.get("save_dir", "outputs/lpp/embedding_sweep"))
    if args.untrained:
        out_dir_pre = out_dir_pre.parent / f"{out_dir_pre.name}_untrained"
    analysis_cfg_pre = cfg.get("analysis", {})

    if args.summary_only:
        if args.dry_run:
            logger.error("--summary_only cannot be combined with --dry_run")
            sys.exit(2)
        full_model_cfgs: list[dict] = cfg.get("models", [])
        if not full_model_cfgs:
            logger.error("Config has no models: list for --summary_only")
            sys.exit(1)
        if args.models:
            full_model_cfgs = [m for m in full_model_cfgs if m["model_id"] in args.models]
        logger.info(
            "summary_only: aggregating under %s (%d model entries in YAML)",
            out_dir_pre,
            len(full_model_cfgs),
        )
        for model_cfg in full_model_cfgs:
            model_dir = out_dir_pre / _safe_model_dir(model_cfg["model_id"])
            n = _merge_checkpoint_shards(model_dir)
            if n:
                logger.info(
                    "Merged checkpoint shards for %s → %s",
                    model_cfg.get("label", model_cfg["model_id"]),
                    model_dir,
                )
        _generate_summary(out_dir_pre, full_model_cfgs, analysis_cfg_pre)
        logger.info("Done.")
        return

    # Resolve active dataset
    dataset_cfg = _active_dataset_cfg(cfg)
    dataset_type = dataset_cfg.get("type", "lpp")
    if dataset_type != "lpp":
        logger.error(
            "This script handles dataset.type='lpp' only; got '%s'.",
            dataset_type,
        )
        sys.exit(1)

    # Resolve subjects / group-mean fMRI (l-bg style)
    fmri_u = _fmri_unit(dataset_cfg)
    if fmri_u == "average_subject":
        avg_explicit = dataset_cfg.get("average_subject_subjects")
        if avg_explicit:
            subjects_full = [str(s) for s in avg_explicit]
        elif dataset_cfg.get("subjects"):
            subjects_full = _discover_subjects(dataset_cfg)
        else:
            subjects_full = discover_lpp_derivative_subjects(
                dataset_cfg["bids_root"], dataset_cfg.get("lang", "en")
            )
            if not subjects_full:
                subjects_full = _discover_subjects(dataset_cfg)
        precomp = dataset_cfg.get("average_subject_precomputed_dir")
        if not subjects_full and precomp:
            precomp_path = Path(os.path.expandvars(str(precomp)))
            if precomp_path.is_dir():
                logger.error(
                    "fmri_unit=average_subject with precomputed_dir=%s but no "
                    "subjects under bids_root=%s (need at least one sub-<LANG>* for "
                    "run↔section timelines and word annotations).",
                    precomp_path,
                    dataset_cfg["bids_root"],
                )
                sys.exit(1)
        if not subjects_full:
            bids = Path(os.path.expandvars(str(dataset_cfg["bids_root"])))
            logger.error(
                "fmri_unit=average_subject but no subjects resolved "
                "(set subjects, average_subject_subjects, or check bids_root). "
                "bids_root=%s exists=%s",
                bids,
                bids.is_dir(),
            )
            sys.exit(1)
        subjects = [subjects_full[0]]
        logger.info(
            "fmri_unit=average_subject: averaging %d participants; timelines from %s",
            len(subjects_full), subjects[0],
        )
    else:
        subjects_full = None
        subjects = _discover_subjects(dataset_cfg)
        if not subjects:
            logger.error("No subjects found. Check bids_root and subjects config.")
            sys.exit(1)

    # Resolve models
    model_cfgs: list[dict] = cfg.get("models", [])
    if args.from_registry:
        if args.family is None:
            logger.error("--from-registry requires --family")
            sys.exit(2)
        from neuroai.models.registry import (
            build_checkpoint_items,
            registry_model_configs,
            resolve_family_task,
        )

        if args.array_index is not None:
            task = resolve_family_task(args.family, args.array_index, untrained=args.untrained)
            model_cfgs = [
                m for m in registry_model_configs(args.family)
                if m["model_id"] == task.model_id
            ]
            if not model_cfgs:
                logger.error(
                    "model_id %r not found in registry family %r",
                    task.model_id,
                    args.family,
                )
                sys.exit(1)
            model_cfgs = [
                {
                    **model_cfgs[0],
                    "load_model_id": task.load_model_id,
                    "_array_checkpoint": (task.revision, task.step, task.tokens_seen),
                }
            ]
            logger.info(
                "Registry task array_index=%d → %s revision=%s step=%d",
                args.array_index,
                task.model_id,
                task.revision,
                task.step,
            )
        else:
            model_cfgs = registry_model_configs(args.family)
            logger.info(
                "Registry family %r → %d model(s)",
                args.family,
                len(model_cfgs),
            )
    elif not model_cfgs:
        logger.error(
            "No models in config. Use models: in YAML or --from-registry --family."
        )
        sys.exit(1)

    if args.models:
        model_cfgs = [m for m in model_cfgs if m["model_id"] in args.models]
    if not model_cfgs:
        logger.error("No models to run (check --models filter or config).")
        sys.exit(1)

    # Output dir
    out_cfg = cfg.get("output", {})
    out_dir = Path(args.out_dir or out_cfg.get("save_dir", "outputs/lpp/embedding_sweep"))
    if args.untrained:
        out_dir = out_dir.parent / f"{out_dir.name}_untrained"

    # Analysis toggles
    analysis_cfg = cfg.get("analysis", {})

    # Encoding config
    enc_yaml = cfg.get("encoding", {})
    cache_cfg = cfg.get("cache", {})
    enc_cfg_base = _build_enc_cfg_base(enc_yaml, dataset_cfg, cache_cfg, out_cfg)
    enc_cfg_base.compute_lateralization = analysis_cfg.get("lateralization", False)
    enc_cfg_base.hf_untrained = args.untrained
    enc_cfg_base.hf_untrained_seed = args.untrained_seed
    if args.ridge_n_jobs is not None:
        enc_cfg_base.ridge_n_jobs = args.ridge_n_jobs
    if args.layer_n_jobs is not None:
        enc_cfg_base.layer_n_jobs = args.layer_n_jobs
    if args.regression_device is not None:
        enc_cfg_base.regression_device = args.regression_device  # type: ignore[assignment]
    if args.n_layer_workers is not None:
        enc_cfg_base.n_layer_workers = args.n_layer_workers
    roi_dir = analysis_cfg.get("roi_masks_dir")
    if roi_dir:
        enc_cfg_base.roi_masks_dir = roi_dir
    mask_path = dataset_cfg.get("lbg_brain_mask_path") or enc_yaml.get("brain_mask_path")
    if mask_path:
        enc_cfg_base.brain_mask_path = mask_path
    ho_parcelization = None
    if analysis_cfg.get("harvard_oxford_parcels", False) and mask_path:
        try:
            from neuroai.roi.lpp import load_harvard_oxford_parcelization

            ho_parcelization = load_harvard_oxford_parcelization(mask_path)
            logger.info(
                "Harvard–Oxford parcels: %d homologous regions (L/R lateralization CSV enabled)",
                ho_parcelization.n_regions,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load Harvard–Oxford parcels: %s", exc)
    enc_cfg_base.fmri_unit = fmri_u
    if fmri_u == "average_subject" and subjects_full is not None:
        enc_cfg_base.average_subject_subjects = tuple(subjects_full)
    else:
        enc_cfg_base.average_subject_subjects = None

    if out_cfg.get("save_corr_maps", False):
        enc_cfg_base.save_corr_maps_dir = out_dir / "corr_maps"

    n_subj_log = len(subjects_full) if subjects_full is not None else len(subjects)
    logger.info(
        "Sweep plan: dataset=%s lang=%s fmri_unit=%s subjects=%d models=%d "
        "per_layer=%s lat=%s ho_parcels=%s regression_device=%s ridge_n_jobs=%d "
        "layer_n_jobs=%s n_layer_workers=%d granular=%s",
        dataset_type,
        dataset_cfg.get("lang", "en"),
        fmri_u,
        n_subj_log,
        len(model_cfgs),
        analysis_cfg.get("per_layer", False),
        analysis_cfg.get("lateralization", False),
        ho_parcelization is not None,
        enc_cfg_base.regression_device,
        enc_cfg_base.ridge_n_jobs,
        enc_cfg_base.layer_n_jobs,
        enc_cfg_base.n_layer_workers,
        out_cfg.get("save_granular_outputs", True),
    )

    if args.dry_run:
        for model_cfg in model_cfgs:
            array_ckpt = model_cfg.get("_array_checkpoint")
            if array_ckpt is not None:
                rev, step, tokens = array_ckpt
                checkpoints = [CheckpointItem(rev, step, tokens)]
            else:
                checkpoints = _build_checkpoints(model_cfg, args.steps)
            if args.untrained:
                checkpoints = checkpoints[:1]
            logger.info(
                "[DRY RUN] model=%s label=%s trust_remote_code=%s checkpoints=%d",
                model_cfg["model_id"],
                model_cfg.get("label"),
                model_cfg.get("trust_remote_code", False),
                len(checkpoints),
            )
            for ckpt in checkpoints:
                logger.info(
                    "  revision=%s step=%d tokens=%s",
                    ckpt.revision, ckpt.step,
                    f"{ckpt.tokens_seen / 1e9:.2f}B" if ckpt.tokens_seen > 0 else "0",
                )
        return

    # --- Main sweep ---
    for model_cfg in model_cfgs:
        model_id = model_cfg["model_id"]
        model_label = model_cfg.get("label", model_id.split("/")[-1])

        array_ckpt = model_cfg.get("_array_checkpoint")
        if array_ckpt is not None:
            rev, step, tokens = array_ckpt
            checkpoints = [CheckpointItem(rev, step, tokens)]
        else:
            checkpoints = _build_checkpoints(model_cfg, args.steps)
        if args.untrained:
            checkpoints = checkpoints[:1]
        if not checkpoints:
            logger.warning("[%s] No checkpoints found — skipping.", model_label)
            continue

        # Device override
        if args.device:
            model_cfg = {**model_cfg, "device": args.device}

        try:
            _run_model_sweep(
                model_cfg=model_cfg,
                dataset_cfg=dataset_cfg,
                enc_cfg_base=enc_cfg_base,
                subjects=subjects,
                checkpoints=checkpoints,
                analysis_cfg=analysis_cfg,
                output_cfg=out_cfg,
                output_dir=out_dir,
                dry_run=False,
                ho_parcelization=ho_parcelization,
                generate_plots=not args.no_plots,
                activations_only=args.activations_only,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] FATAL: %s", model_label, exc)
            traceback.print_exc()

    # --- Summary ---
    if not args.skip_summary:
        logger.info("Generating summary plots and CSV ...")
        _generate_summary(out_dir, model_cfgs, analysis_cfg)

    logger.info("LPP checkpoint sweep complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
