#!/usr/bin/env python3
"""Broderick 2018 EEG — per-layer logit-lens surprisal extraction (GPU stage).

Populates the surprisal cache consumed by ``run/surprisal/neural/broderick_fit.py``.
Split into its own script/job because on Jean Zay, R (pymer4/lme4, needed
for the LME fit) is only available on CPU partitions, not GPU — so, exactly
like every other pooled LME in this codebase, extraction (needs the LLM,
wants a GPU for larger models) and LME fitting (needs R, CPU-only) must run
as separate jobs sharing a cache directory, not one combined job.

This is a light GPU stage: one forward pass per audiobook segment (20
segments, 11,419 words total) — not sliding-window chunks over a much
longer document.

Examples
--------
python scripts/run/surprisal/neural/broderick_extract.py \\
    --config configs/surprisal/neural/broderick.yaml \\
    --model_name openai-community/gpt2 --device cuda

python scripts/run/surprisal/neural/broderick_extract.py \\
    --config configs/surprisal/neural/broderick.yaml \\
    --from-registry --family gpt2 --array-index 0 --device cuda
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
    BroderickLmeDataConfig,
)
from neuroai.models.registry import registry_model_configs, resolve_family_task, resolve_load_model_id
from neuroai.sweep import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw)


def _data_cfg(ds: dict, *, feature_cache_dir: str) -> BroderickLmeDataConfig:
    return BroderickLmeDataConfig(
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
        wordfreq_language=str(ds.get("wordfreq_language", "en")),
        feature_cache_dir=feature_cache_dir,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/broderick.yaml")
    p.add_argument("--model_name", default=None, help="Direct HF model id (skip registry)")
    p.add_argument("--revision", default=None)
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Override cache.features_dir from the config")
    args = p.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    cache_cfg = cfg.get("cache", {})
    feature_cache_dir = args.feature_cache_dir or os.path.expandvars(
        str(cache_cfg.get("features_dir", ".cache/broderick_lme_features"))
    )
    Path(feature_cache_dir).mkdir(parents=True, exist_ok=True)
    yaml_trust = bool(ds.get("trust_remote_code", False))

    if args.from_registry:
        if not args.family or args.array_index is None:
            p.error("--from-registry requires --family and --array-index")
        task = resolve_family_task(args.family, args.array_index)
        model_cfgs = [m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id]
        model_id = resolve_load_model_id(model_cfgs[0], task.revision) if model_cfgs else task.model_id
        revision = task.revision
        n_params = model_cfgs[0]["n_params"] if model_cfgs else None
        trust_remote_code = bool(model_cfgs[0].get("trust_remote_code", yaml_trust)) if model_cfgs else yaml_trust
    elif args.model_name:
        model_id = args.model_name
        revision = args.revision
        n_params = None
        trust_remote_code = yaml_trust
    else:
        p.error("Specify --model_name, or --from-registry --family F --array-index I")
        return

    data_cfg = _data_cfg(ds, feature_cache_dir=feature_cache_dir)
    data_cfg.trust_remote_code = trust_remote_code
    data_cfg.device = args.device
    model_cfg = ModelConfig(
        model_name=model_id,
        model_revision=revision if revision not in (None, "", "main") else None,
        device=args.device,
        n_params=n_params,
        trust_remote_code=trust_remote_code,
    )
    bench_cfg = BroderickLmeBenchmarkConfig(
        data=data_cfg, model=model_cfg, output_dir=feature_cache_dir, canonical_model_name=model_id,
    )

    logger.info(
        "Broderick surprisal extraction: model=%s revision=%s device=%s cache_dir=%s",
        model_id, revision, args.device, feature_cache_dir,
    )
    surprisal, prev_surprisal = BroderickLmeBenchmark(bench_cfg).extract_and_cache_surprisal()
    logger.info(
        "Done — surprisal %s, prev_surprisal %s cached under %s",
        surprisal.shape, prev_surprisal.shape, feature_cache_dir,
    )


if __name__ == "__main__":
    main()
