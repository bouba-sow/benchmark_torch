#!/usr/bin/env python3
"""Broderick 2018 EEG pooled LME — one mixed-effects model per LM layer, region-averaged.

Fits ``eeg ~ wlen_z + freq_z + surprisal_z + prev_surprisal_z +
(1 + wlen_z + freq_z + surprisal_z + prev_surprisal_z | subject) +
(1 + surprisal_z | region)`` vs. the same formula without ``surprisal_z``
and ``prev_surprisal_z``, once per LM layer, pooling every subject and one
of 4 scalp regions (left/right/posterior/occipital — Broderick's 128
BioSemi channels collapsed via ``channel_region``, NOT crossed individually
— see ``neuroai/regressors/broderick_lme.py`` module docstring for why: the
full 128-channel montage would pool ~28M rows, larger than ZuCo's ~5M-row
table that already failed to finish fitting within a 1-4h walltime).

CPU-only stage: on Jean Zay, R (pymer4/lme4) is only available on CPU
partitions, not GPU — so per-layer surprisal must already be cached by
``scripts/run/surprisal/neural/broderick_extract.py`` (GPU stage) before
running this. A cache miss silently falls back to loading the LLM on CPU
(slow, not wrong) rather than hard-erroring — same soft contract as every
other pooled LME in this codebase.

Examples
--------
# Single model, all layers, 10 concurrent worker processes:
python scripts/run/surprisal/neural/broderick_fit.py \\
    --config configs/surprisal/neural/broderick.yaml \\
    --model_name openai-community/gpt2 --n_workers 10

# From the registry (single array task, one model):
python scripts/run/surprisal/neural/broderick_fit.py \\
    --config configs/surprisal/neural/broderick.yaml \\
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
from neuroai.benchmarks.broderick_lme import (
    BroderickLmeBenchmark,
    BroderickLmeBenchmarkConfig,
    BroderickLmeConfig,
    BroderickLmeDataConfig,
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
    n_workers: int,
    layer_stride: int,
    last_layer_only: bool = False,
    device: str | None = None,
    feature_cache_dir: str | None = None,
    trust_remote_code: bool | None = None,
    save_word_predictions: bool = True,
) -> BroderickLmeBenchmarkConfig:
    ds = cfg["dataset"]
    lme = cfg.get("lme", {})
    cache_cfg = cfg.get("cache", {})
    dev = device or "cpu"

    data_cfg = BroderickLmeDataConfig(
        bids_root=os.path.expandvars(str(ds.get("bids_root", "data/broderick"))),
        subjects=ds.get("subjects") or None,
        n_runs=int(ds.get("n_runs", 20)),
        window_tmin=float(ds.get("window_tmin", 0.3)),
        window_tmax=float(ds.get("window_tmax", 0.5)),
        epoch_tmin=float(ds.get("epoch_tmin", -0.2)),
        epoch_tmax=float(ds.get("epoch_tmax", 1.0)),
        baseline=tuple(ds["baseline"]) if ds.get("baseline") else (-0.2, 0.0),
        l_freq=float(ds.get("l_freq", 1.0)),
        h_freq=float(ds.get("h_freq", 8.0)),
        target_sfreq=float(ds.get("target_sfreq", 32.0)),
        average_ref=bool(ds.get("average_ref", True)),
        drop_bads=bool(ds.get("drop_bads", True)),
        max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        tokenizer_no_revision=bool(ds.get("tokenizer_no_revision", True)),
        trust_remote_code=(
            trust_remote_code if trust_remote_code is not None
            else bool(ds.get("trust_remote_code", False))
        ),
        wordfreq_language=str(ds.get("wordfreq_language", "en")),
        device=dev,
        feature_cache_dir=feature_cache_dir or os.path.expandvars(
            str(cache_cfg.get("features_dir", ".cache/broderick_lme_features"))
        ),
    )
    model_cfg = ModelConfig(
        model_name=model_id,
        model_revision=revision if revision not in (None, "", "main") else None,
        device=dev,
        n_params=n_params,
        trust_remote_code=bool(trust_remote_code),
    )
    lme_cfg = BroderickLmeConfig(
        train_frac=float(lme.get("train_frac", 0.75)),
        split_random_state=int(lme.get("split_random_state", 0)),
        verbose=bool(lme.get("verbose", False)),
        allow_slope_fallback=bool(lme.get("allow_slope_fallback", True)),
        layer_stride=layer_stride,
        last_layer_only=last_layer_only,
        n_workers=n_workers,
        # CPU-only stage: R (pymer4/lme4) is CPU-only on Jean Zay. Surprisal
        # is expected to already be cached by
        # run/surprisal/neural/broderick_extract.py (GPU stage) — a cache miss
        # here raises a clear error instead of silently falling back to
        # loading the LLM on CPU.
        require_cached_surprisal=bool(lme.get("require_cached_surprisal", dev == "cpu")),
        save_word_predictions=save_word_predictions,
    )
    return BroderickLmeBenchmarkConfig(
        data=data_cfg,
        model=model_cfg,
        lme=lme_cfg,
        output_dir=str(output_dir),
        canonical_model_name=model_id,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/broderick.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model_name", default=None, help="Direct HF model id (skip registry)")
    p.add_argument("--revision", default=None)
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--n_workers", type=int, default=10,
                   help="Concurrent worker processes fitting layers (default 10)")
    p.add_argument("--layer_stride", type=int, default=1,
                   help="Fit every Nth layer (default 1 = all layers)")
    p.add_argument("--last_layer_only", action="store_true", help="Fit only the final layer")
    p.add_argument("--device", default=None,
                   help="Override dataset device (surprisal extraction fallback only)")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Surprisal cache dir populated by run/surprisal/neural/broderick_extract.py")
    p.add_argument("--save_word_predictions", action=argparse.BooleanOptionalAction, default=True,
                   help="Save each held-out row's observed eeg + baseline/full model predictions "
                        "to a per-word CSV for post-hoc per-word difficulty splits (DLT). "
                        "On by default; pass --no-save_word_predictions to disable.")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = Path(
        args.out_dir or cfg.get("output", {}).get("save_dir", "outputs/broderick/surprisal_sweep")
    )

    if args.from_registry:
        if not args.family or args.array_index is None:
            p.error("--from-registry requires --family and --array-index")
        task = resolve_family_task(args.family, args.array_index)
        model_cfgs = [m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id]
        load_model_id = resolve_load_model_id(model_cfgs[0], task.revision) if model_cfgs else task.model_id
        model_id = load_model_id
        revision = task.revision
        n_params = model_cfgs[0]["n_params"] if model_cfgs else None
        trust_remote_code = (
            bool(model_cfgs[0].get("trust_remote_code", False)) if model_cfgs else None
        )
        output_dir = out_root / args.family / safe_model_dir(task.model_id)
    elif args.model_name:
        model_id = args.model_name
        revision = args.revision
        n_params = None
        trust_remote_code = None
        output_dir = out_root / safe_model_dir(model_id)
    else:
        p.error("Specify --model_name, or --from-registry --family F --array-index I")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    bench_cfg = _build_benchmark_config(
        cfg,
        model_id=model_id,
        revision=revision,
        n_params=n_params,
        output_dir=output_dir,
        n_workers=args.n_workers,
        layer_stride=args.layer_stride,
        last_layer_only=args.last_layer_only,
        device=args.device,
        feature_cache_dir=args.feature_cache_dir,
        trust_remote_code=trust_remote_code,
        save_word_predictions=args.save_word_predictions,
    )

    logger.info(
        "Broderick LME sweep: model=%s revision=%s n_workers=%d layer_stride=%d out_dir=%s",
        model_id, revision, args.n_workers, args.layer_stride, output_dir,
    )
    BroderickLmeBenchmark(bench_cfg).run()
    logger.info("Done.")


if __name__ == "__main__":
    main()
