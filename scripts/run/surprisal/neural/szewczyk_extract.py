#!/usr/bin/env python3
"""Szewczyk N400 — per-layer logit-lens surprisal extraction (GPU stage).

Populates the surprisal cache consumed by ``run/surprisal/neural/szewczyk_fit.py``.
Split into its own script/job because on Jean Zay, R (pymer4/lme4, needed
for the LME fit) is only available on CPU partitions, not GPU — so, exactly
like Podcast's pooled LME, extraction (needs the LLM, wants a GPU
for larger models) and LME fitting (needs R, CPU-only) must run as separate
jobs sharing a cache directory, not one combined job.

A light GPU stage: ~1,330 unique (context, critical-word) trial items, one
forward pass each (no spillover control needed by the paper formula).

Examples
--------
python scripts/run/surprisal/neural/szewczyk_extract.py \\
    --config configs/surprisal/neural/szewczyk.yaml \\
    --model_name openai-community/gpt2 --device cuda

python scripts/run/surprisal/neural/szewczyk_extract.py \\
    --config configs/surprisal/neural/szewczyk.yaml \\
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
from neuroai.benchmarks.n400_szewczyk_lme import (
    SzewczykLmeBenchmark,
    SzewczykLmeBenchmarkConfig,
    SzewczykLmeDataConfig,
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/szewczyk.yaml")
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
        str(cache_cfg.get("features_dir", ".cache/szewczyk_lme_features"))
    )
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

    data_cfg = SzewczykLmeDataConfig(
        data_path=os.path.expandvars(str(ds.get(
            "data_path", "data/n400_paper_formula/osfstorage/datasets/szewczyk_2022.tsv"
        ))),
        max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        tokenizer_no_revision=bool(ds.get("tokenizer_no_revision", True)),
        trust_remote_code=trust_remote_code,
        device=args.device,
        feature_cache_dir=feature_cache_dir,
    )
    model_cfg = ModelConfig(
        model_name=model_id,
        model_revision=revision if revision not in (None, "", "main") else None,
        device=args.device,
        n_params=n_params,
    )
    bench_cfg = SzewczykLmeBenchmarkConfig(
        data=data_cfg, model=model_cfg, output_dir=feature_cache_dir, canonical_model_name=model_id,
    )

    logger.info(
        "Szewczyk surprisal extraction: model=%s revision=%s device=%s cache_dir=%s",
        model_id, revision, args.device, feature_cache_dir,
    )
    surprisal = SzewczykLmeBenchmark(bench_cfg).extract_and_cache_surprisal()
    logger.info("Done — surprisal %s cached under %s", surprisal.shape, feature_cache_dir)


if __name__ == "__main__":
    main()
