#!/usr/bin/env python3
"""Tuckute fMRI pooled LME — one mixed-effects model per LM layer.

Fits ``y ~ mean_wlen_z + mean_freq_z + surprisal_z + (1 + ... | subject) +
(1 | roi) + (1 + surprisal_z | sentence_id)`` vs. the same formula without
``surprisal_z``, once per LM layer, pooling every subject, ROI, and sentence
together (for one condition — default "B"). See
``neuroai/regressors/tuckute_pooled_lme.py`` for the formula/fallback-ladder
rationale and ``neuroai/benchmarks/tuckute_lme.py`` for the pipeline.

CPU-only stage: on Jean Zay, R (pymer4/lme4) is only available on CPU
partitions, not GPU — so per-layer surprisal must already be cached by
``scripts/run/surprisal/neural/tuckute_extract.py`` (GPU stage) before running
this. A cache miss silently falls back to loading the LLM on CPU (slow, not
wrong) rather than hard-erroring — same soft contract as every other pooled
LME in this codebase.

Examples
--------
# Single model, all layers, 10 concurrent worker processes:
python scripts/run/surprisal/neural/tuckute_fit.py \\
    --config configs/surprisal/neural/tuckute.yaml \\
    --model_name openai-community/gpt2 --n_workers 10

# From the registry (single array task, one model):
python scripts/run/surprisal/neural/tuckute_fit.py \\
    --config configs/surprisal/neural/tuckute.yaml \\
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
from neuroai.benchmarks.tuckute_lme import (
    TuckuteLmeBenchmark,
    TuckuteLmeBenchmarkConfig,
    TuckuteLmeConfig,
    TuckuteLmeDataConfig,
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
    condition: str | None = None,
    device: str | None = None,
    feature_cache_dir: str | None = None,
    trust_remote_code: bool | None = None,
) -> TuckuteLmeBenchmarkConfig:
    ds = cfg["dataset"]
    lme = cfg.get("lme", {})
    cache_cfg = cfg.get("cache", {})

    data_cfg = TuckuteLmeDataConfig(
        data_root=os.path.expandvars(str(ds.get("data_root", "data/tuckute_fmri/osfstorage"))),
        condition=condition or str(ds.get("condition", "B")),
        max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        tokenizer_no_revision=bool(ds.get("tokenizer_no_revision", True)),
        trust_remote_code=(
            trust_remote_code if trust_remote_code is not None
            else bool(ds.get("trust_remote_code", False))
        ),
        wordfreq_language=str(ds.get("wordfreq_language", "en")),
        # CPU-only stage: R (pymer4/lme4) is CPU-only on Jean Zay. Surprisal
        # is expected to already be cached by run/surprisal/neural/tuckute_extract.py
        # (GPU stage) — lme.require_cached_surprisal (set below, defaults to
        # True on a CPU device) makes a cache miss here raise a clear error
        # instead of silently falling back to loading the LLM on CPU, which
        # for the FLA-based trust_remote_code families crashes on import.
        device=device or str(ds.get("device", "cpu")),
        feature_cache_dir=feature_cache_dir or os.path.expandvars(
            str(cache_cfg.get("features_dir", ".cache/tuckute_lme_features"))
        ),
    )
    model_cfg = ModelConfig(
        model_name=model_id,
        model_revision=revision if revision not in (None, "", "main") else None,
        device=data_cfg.device,
        n_params=n_params,
    )
    lme_cfg = TuckuteLmeConfig(
        train_frac=float(lme.get("train_frac", 0.75)),
        split_random_state=int(lme.get("split_random_state", 0)),
        verbose=bool(lme.get("verbose", False)),
        allow_slope_fallback=bool(lme.get("allow_slope_fallback", True)),
        layer_stride=layer_stride,
        last_layer_only=last_layer_only or bool(lme.get("last_layer_only", False)),
        n_workers=n_workers,
        require_cached_surprisal=bool(
            lme.get("require_cached_surprisal", data_cfg.device == "cpu")
        ),
        roi_wise=bool(lme.get("roi_wise", False)),
    )
    return TuckuteLmeBenchmarkConfig(
        data=data_cfg,
        model=model_cfg,
        lme=lme_cfg,
        output_dir=str(output_dir),
        canonical_model_name=model_id,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/tuckute.yaml")
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
    p.add_argument("--last_layer_only", action="store_true",
                   help="Fit ONLY the final layer, ignoring --layer_stride")
    p.add_argument("--condition", default=None, help="Override dataset.condition (B/D/S)")
    p.add_argument("--device", default=None,
                   help="Override dataset.device (surprisal extraction forward pass)")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Surprisal cache dir populated by run/surprisal/neural/tuckute_extract.py")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = Path(
        args.out_dir or cfg.get("output", {}).get("save_dir", "outputs/tuckute/surprisal_sweep")
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
        condition=args.condition,
        device=args.device,
        feature_cache_dir=args.feature_cache_dir,
        trust_remote_code=trust_remote_code,
    )

    logger.info(
        "Tuckute LME sweep: model=%s revision=%s condition=%s n_workers=%d layer_stride=%d out_dir=%s",
        model_id, revision, bench_cfg.data.condition, args.n_workers, args.layer_stride, output_dir,
    )
    TuckuteLmeBenchmark(bench_cfg).run(extra_fields=extra_fields)
    logger.info("Done.")


if __name__ == "__main__":
    main()
