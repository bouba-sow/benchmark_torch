#!/usr/bin/env python3
"""Pereira fMRI voxelwise embedding encoding — himalaya GPU ridge, per
subject, per LM layer.

For each subject and LM layer, predicts every language-atlas voxel from that
layer's mean-pooled-over-words sentence embedding via
``cv_ridge_score_multioutput_gpu`` (5-fold GroupKFold-by-passage, SVD-based
GCV). See ``docs/pereira_embedding_design.md`` for the design and
``neuroai/benchmarks/pereira_embed.py`` for the pipeline.

Unlike ``run/surprisal/neural/pereira_fit.py`` (GPU extraction + CPU-only R fit as two
separate jobs), BOTH stages here need a GPU — this one script does both.

Examples
--------
# Single model, all layers:
python scripts/run/embedding/neural/pereira.py \\
    --config configs/embedding/neural/pereira.yaml \\
    --model_name openai-community/gpt2 --device cuda

# From the registry (single array task, one model):
python scripts/run/embedding/neural/pereira.py \\
    --config configs/embedding/neural/pereira.yaml \\
    --from-registry --family gpt2 --array-index 0 --device cuda

# Alpha-calibration run (wide grid already the config default) — inspect
# outputs/.../pereira_embed_by_layer.csv's ridge_alpha_median column's
# min/median/max across all (subject, layer) rows afterward.
python scripts/run/embedding/neural/pereira.py \\
    --config configs/embedding/neural/pereira.yaml \\
    --model_name openai-community/gpt2 --device cuda \\
    --out_dir outputs/pereira_embed_alpha_calibration
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np
import yaml

from neuroai.benchmarks.base import ModelConfig
from neuroai.benchmarks.pereira_embed import (
    PereiraEmbedBenchmark,
    PereiraEmbedBenchmarkConfig,
    PereiraEmbedConfig,
    PereiraEmbedDataConfig,
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
    layer_stride: int,
    last_layer_only: bool = False,
    atlas: str | None = None,
    device: str | None = None,
    feature_cache_dir: str | None = None,
    trust_remote_code: bool | None = None,
    alpha_logspace: tuple[float, float, int] | None = None,
    n_layer_workers: int | None = None,
) -> PereiraEmbedBenchmarkConfig:
    ds = cfg["dataset"]
    embed = cfg.get("embed", {})
    cache_cfg = cfg.get("cache", {})

    data_cfg = PereiraEmbedDataConfig(
        nc_path=os.path.expandvars(str(ds.get("nc_path", "data/pereira_fmri/Pereira2018_merged.nc"))),
        atlas=atlas or str(ds.get("atlas", "language")),
        max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        tokenizer_no_revision=bool(ds.get("tokenizer_no_revision", True)),
        trust_remote_code=(
            trust_remote_code if trust_remote_code is not None
            else bool(ds.get("trust_remote_code", False))
        ),
        device=device or str(ds.get("device", "cuda")),
        feature_cache_dir=feature_cache_dir or os.path.expandvars(
            str(cache_cfg.get("features_dir", ".cache/pereira_embed_features"))
        ),
    )
    model_cfg = ModelConfig(
        model_name=model_id,
        model_revision=revision if revision not in (None, "", "main") else None,
        device=data_cfg.device,
        n_params=n_params,
    )
    lo, hi, n = alpha_logspace or tuple(embed.get("alpha_logspace", [-3, 9, 12]))
    embed_cfg = PereiraEmbedConfig(
        ridge_alphas=np.logspace(float(lo), float(hi), int(n)),
        n_splits=int(embed.get("n_splits", 5)),
        random_state=int(embed.get("random_state", 0)),
        standardize_target=bool(embed.get("standardize_target", True)),
        layer_stride=layer_stride,
        last_layer_only=last_layer_only,
        n_layer_workers=n_layer_workers if n_layer_workers is not None else int(embed.get("n_layer_workers", 8)),
        require_cached_hidden_states=bool(
            embed.get("require_cached_hidden_states", False)
        ),
    )
    return PereiraEmbedBenchmarkConfig(
        data=data_cfg,
        model=model_cfg,
        embed=embed_cfg,
        output_dir=str(output_dir),
        canonical_model_name=model_id,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/embedding/neural/pereira.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model_name", default=None, help="Direct HF model id (skip registry)")
    p.add_argument("--revision", default=None)
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--layer_stride", type=int, default=1,
                   help="Fit every Nth layer (default 1 = all layers)")
    p.add_argument("--last_layer_only", action="store_true",
                   help="Fit ONLY the final layer, ignoring --layer_stride")
    p.add_argument("--atlas", default=None,
                   help="Override dataset.atlas (language/auditory/MD/DMN/visual)")
    p.add_argument("--device", default=None, help="Override dataset.device")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Hidden-state cache dir")
    p.add_argument("--alpha_min_exp", type=float, default=None,
                   help="Override embed.alpha_logspace[0] (log10 min alpha)")
    p.add_argument("--alpha_max_exp", type=float, default=None,
                   help="Override embed.alpha_logspace[1] (log10 max alpha)")
    p.add_argument("--alpha_n", type=int, default=None,
                   help="Override embed.alpha_logspace[2] (number of alphas)")
    p.add_argument("--n_layer_workers", type=int, default=None,
                   help="Concurrent CUDA-stream layer workers")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = Path(
        args.out_dir or cfg.get("output", {}).get("save_dir", "outputs/pereira/embedding_sweep")
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

    alpha_logspace = None
    if args.alpha_min_exp is not None or args.alpha_max_exp is not None or args.alpha_n is not None:
        base = cfg.get("embed", {}).get("alpha_logspace", [-3, 9, 12])
        alpha_logspace = (
            args.alpha_min_exp if args.alpha_min_exp is not None else base[0],
            args.alpha_max_exp if args.alpha_max_exp is not None else base[1],
            args.alpha_n if args.alpha_n is not None else base[2],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    bench_cfg = _build_benchmark_config(
        cfg,
        model_id=model_id,
        revision=revision,
        n_params=n_params,
        output_dir=output_dir,
        layer_stride=args.layer_stride,
        last_layer_only=args.last_layer_only,
        atlas=args.atlas,
        device=args.device,
        feature_cache_dir=args.feature_cache_dir,
        trust_remote_code=trust_remote_code,
        alpha_logspace=alpha_logspace,
        n_layer_workers=args.n_layer_workers,
    )

    logger.info(
        "Pereira embed sweep: model=%s revision=%s atlas=%s alphas=%s n_layer_workers=%d out_dir=%s",
        model_id, revision, bench_cfg.data.atlas, bench_cfg.embed.ridge_alphas,
        bench_cfg.embed.n_layer_workers, output_dir,
    )
    PereiraEmbedBenchmark(bench_cfg).run(extra_fields=extra_fields)
    logger.info("Done.")


if __name__ == "__main__":
    main()
