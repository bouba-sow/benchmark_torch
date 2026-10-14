#!/usr/bin/env python3
"""Podcast ECoG pooled LME — one mixed-effects model per lag (Design B).

Fits ``y ~ wlen_z + freq_z + surprisal_z + (1|subject) + (1+surprisal_z|
subject:electrode)`` vs. the same formula without surprisal, once per lag,
pooling every subject and electrode (electrode nested within subject — no
correspondence between electrodes across subjects is assumed). See
``neuroai/regressors/podcast_lme.py`` for the full formula/fallback-ladder
rationale and ``neuroai/benchmarks/podcast_lme.py`` for the pipeline.

CPU-only: surprisal is read from the existing logit-lens cache (already
computed by the ridge/surprisal sweep) — this script never loads an LLM.
Lags are fit concurrently via a spawned process pool within ONE job (not one
SLURM job per lag) — pymer4/rpy2 embeds one R interpreter per process, so
parallelism must be process-based.

Examples
--------
# Single model, all lags, 10 concurrent worker processes:
python scripts/run/surprisal/neural/podcast_fit.py \\
    --config configs/surprisal/neural/podcast.yaml \\
    --model_name openai-community/gpt2 --n_workers 10

# From the registry (single array task, one model):
python scripts/run/surprisal/neural/podcast_fit.py \\
    --config configs/surprisal/neural/podcast.yaml \\
    --from-registry --family gpt2 --array-index 0 --n_workers 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import yaml

from neuroai.benchmarks.base import ModelConfig
from neuroai.benchmarks.podcast_lme import (
    PodcastLmeBenchmark,
    PodcastLmeBenchmarkConfig,
    PodcastLmeConfig,
    PodcastLmeDataConfig,
)
from neuroai.models.registry import registry_model_configs, resolve_family_task, resolve_load_model_id
from neuroai.sweep import safe_model_dir, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw)


def _build_benchmark_config(
    cfg: dict,
    *,
    model_id: str,
    revision: str | None,
    n_params: float | None,
    output_dir: Path,
    feature_cache_dir: Path,
    n_workers: int,
    lag_stride: int,
    trust_remote_code: bool | None = None,
    epoch_cache_dir: Path | None = None,
    save_word_predictions: bool | None = None,
    window_s: tuple[float, float] | None = None,
) -> PodcastLmeBenchmarkConfig:
    ds = cfg["dataset"]
    lme = cfg.get("lme", {})

    data_cfg = PodcastLmeDataConfig(
        bids_root=ds["bids_root"],
        subjects=ds.get("subjects") or None,
        tmin=float(ds.get("tmin", -2.0)),
        tmax=float(ds.get("tmax", 2.0)),
        tmin_extract=float(ds.get("tmin_extract", -2.1)),
        downsample_to=float(ds.get("downsample_to", 32.0)),
        clip_iqr=float(ds["clip_iqr"]) if ds.get("clip_iqr") else None,
        baseline=tuple(ds["baseline"]) if ds.get("baseline") else None,
        baseline_mode=str(ds.get("baseline_mode", "logratio")),
        electrode_selection_dir=ds.get("electrode_selection_dir") or None,
        roi_map_path=ds.get("roi_map_path") or None,
        epoch_cache_dir=str(epoch_cache_dir) if epoch_cache_dir else None,
        feature_cache_dir=str(feature_cache_dir),
        max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        tokenizer_no_revision=bool(ds.get("tokenizer_no_revision", True)),
        trust_remote_code=(
            trust_remote_code if trust_remote_code is not None
            else bool(ds.get("trust_remote_code", False))
        ),
        wordfreq_language=str(ds.get("wordfreq_language", "en")),
    )
    model_cfg = ModelConfig(
        model_name=model_id,
        model_revision=revision if revision not in (None, "", "main") else None,
        device="cpu",
        n_params=n_params,
    )
    lme_cfg = PodcastLmeConfig(
        chunk_words=int(lme.get("chunk_words", 30)),
        train_frac=float(lme.get("train_frac", 0.75)),
        split_random_state=int(lme.get("split_random_state", 0)),
        verbose=bool(lme.get("verbose", False)),
        allow_slope_fallback=bool(lme.get("allow_slope_fallback", True)),
        lag_stride=lag_stride,
        n_workers=n_workers,
        save_word_predictions=(
            save_word_predictions if save_word_predictions is not None
            else bool(lme.get("save_word_predictions", True))
        ),
        window_s=(window_s if window_s is not None else
                  tuple(float(v) for v in lme["window_s"]) if lme.get("window_s") else None),
        roi_wise=bool(lme.get("roi_wise", False)),
    )
    return PodcastLmeBenchmarkConfig(
        data=data_cfg,
        model=model_cfg,
        lme=lme_cfg,
        output_dir=str(output_dir),
        canonical_model_name=model_id,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/podcast.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model_name", default=None, help="Direct HF model id (skip registry)")
    p.add_argument("--revision", default=None)
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--n_workers", type=int, default=10,
                   help="Concurrent worker processes fitting lags (default 10)")
    p.add_argument("--lag_stride", type=int, default=1,
                   help="Fit every Nth lag (default 1 = all lags)")
    p.add_argument("--window_ms", type=float, nargs=2, default=None, metavar=("LO_MS", "HI_MS"),
                   help="Mean-window mode: average the response across lags whose analysis "
                        "time falls in [LO_MS, HI_MS] (milliseconds) into ONE column and fit "
                        "exactly ONE LME on that windowed mean, instead of one LME per lag "
                        "(standard ERP/ECoG 'mean amplitude in a window' convention) -- for a "
                        "checkpoint-scaling sweep, where a full 32-lag grid x 110 checkpoints "
                        "would be far too expensive. Overrides --lag_stride. "
                        "E.g. --window_ms 0 200 for the 0-200ms post-onset window.")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Surprisal cache dir (must already contain the logit-lens cache)")
    p.add_argument("--epoch_cache_dir", default=None,
                   help="Per-subject preprocessed-epoch cache dir (shared across the whole "
                        "family — model-independent, only tmin/tmax/baseline/etc. matter)")
    p.add_argument("--save_word_predictions", action=argparse.BooleanOptionalAction, default=None,
                   help="Save one parquet per lag under out_dir/predictions/ with held-out "
                        "per-word y_true/yhat_base/yhat_full/delta_loglik, for DLT-difficulty "
                        "post-hoc analysis. On by default; pass --no-save_word_predictions to "
                        "skip (large output: millions of rows across a full lag grid).")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = Path(
        args.out_dir or cfg.get("output", {}).get("save_dir", "outputs/podcast/surprisal_sweep")
    )

    cache_cfg = cfg.get("cache", {})
    feature_cache_dir = Path(
        args.feature_cache_dir or os.path.expandvars(str(cache_cfg.get("features_dir", ".cache/podcast_features")))
    )
    epochs_dir_raw = args.epoch_cache_dir or os.path.expandvars(str(cache_cfg.get("epochs_dir", "")))
    epoch_cache_dir = Path(epochs_dir_raw) if epochs_dir_raw else None

    if args.from_registry:
        if not args.family or args.array_index is None:
            p.error("--from-registry requires --family and --array-index")
        task = resolve_family_task(args.family, args.array_index)
        model_cfgs = [m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id]
        load_model_id = resolve_load_model_id(model_cfgs[0], task.revision) if model_cfgs else task.model_id
        model_id = load_model_id
        revision = task.revision
        n_params = model_cfgs[0]["n_params"] if model_cfgs else None
        # Per-model registry entry (e.g. stickbreaking/alibi/forgetting/
        # hard-n-gram, trust_remote_code=True) overrides the YAML default.
        trust_remote_code = (
            bool(model_cfgs[0].get("trust_remote_code", False)) if model_cfgs else None
        )
        output_dir = out_root / args.family / safe_model_dir(task.model_id)
        extra_fields = {"step": task.step, "tokens_seen": task.tokens_seen}
    elif args.model_name:
        model_id = args.model_name
        revision = args.revision
        n_params = None
        trust_remote_code = None
        output_dir = out_root / safe_model_dir(model_id)
        extra_fields = None
    else:
        p.error("Specify --model_name, or --from-registry --family F --array-index I")
        return

    window_s = (args.window_ms[0] / 1000.0, args.window_ms[1] / 1000.0) if args.window_ms else None

    output_dir.mkdir(parents=True, exist_ok=True)
    bench_cfg = _build_benchmark_config(
        cfg,
        model_id=model_id,
        revision=revision,
        n_params=n_params,
        output_dir=output_dir,
        feature_cache_dir=feature_cache_dir,
        n_workers=args.n_workers,
        lag_stride=args.lag_stride,
        trust_remote_code=trust_remote_code,
        epoch_cache_dir=epoch_cache_dir,
        save_word_predictions=args.save_word_predictions,
        window_s=window_s,
    )

    logger.info(
        "Podcast LME sweep: model=%s revision=%s n_workers=%d lag_stride=%d window_s=%s out_dir=%s",
        model_id, revision, args.n_workers, args.lag_stride, window_s, output_dir,
    )
    PodcastLmeBenchmark(bench_cfg).run(extra_fields=extra_fields)
    logger.info("Done.")


if __name__ == "__main__":
    main()
