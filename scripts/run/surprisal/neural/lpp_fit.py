#!/usr/bin/env python3
"""Le Petit Prince fMRI pooled LME — one mixed-effects model per LM layer.

Fits ``bold ~ wlen_z + freq_z + sentpos_z + surprisal_z +
(1 + surprisal_z | region) + (1 | section)`` vs. the same formula without
``surprisal_z``, once per LM layer, pooling all 10 Fedorenko-parcel x
hemisphere regions and all 9 LPP sections together (average-subject BOLD).
See ``neuroai/regressors/lpp_pooled_lme.py`` for the formula/fallback-ladder
rationale and ``neuroai/benchmarks/lpp_lme.py`` for the pipeline.

CPU-only stage: on Jean Zay, R (pymer4/lme4) is only available on CPU
partitions, not GPU — so per-layer surprisal must already be cached by
``scripts/run/surprisal/neural/lpp_extract.py`` (GPU stage) before running this.
A cache miss silently falls back to loading the LLM on CPU (slow, not wrong)
rather than hard-erroring — same soft contract as every other pooled LME in
this codebase.

Examples
--------
# Single model, all layers, 10 concurrent worker processes:
python scripts/run/surprisal/neural/lpp_fit.py \\
    --config configs/surprisal/neural/lpp.yaml \\
    --model_name openai-community/gpt2 --n_workers 10

# From the registry (single array task, one model):
python scripts/run/surprisal/neural/lpp_fit.py \\
    --config configs/surprisal/neural/lpp.yaml \\
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
from neuroai.benchmarks.lpp_lme import (
    LppLmeBenchmark,
    LppLmeBenchmarkConfig,
    LppLmeConfig,
    LppLmeDataConfig,
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
    use_persubject: bool | None = None,
    device: str | None = None,
    feature_cache_dir: str | None = None,
    trust_remote_code: bool | None = None,
    save_word_predictions: bool | None = None,
) -> LppLmeBenchmarkConfig:
    ds = cfg["dataset"]
    lme = cfg.get("lme", {})
    cache_cfg = cfg.get("cache", {})
    dev = device or "cpu"

    data_cfg = LppLmeDataConfig(
        bids_root=os.path.expandvars(str(ds["bids_root"])),
        lang=str(ds.get("lang", "en")),
        sections=ds.get("sections") or None,
        t_r=float(ds.get("t_r", 2.0)),
        min_word_duration_sec=float(ds.get("min_word_duration_sec", 0.05)),
        hrf_model=str(ds.get("hrf_model", "glover")),
        trim_trs_start=int(ds.get("trim_trs_start", 10)),
        trim_trs_end=int(ds.get("trim_trs_end", 10)),
        max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        tokenizer_no_revision=bool(ds.get("tokenizer_no_revision", True)),
        feature_cache_dir=feature_cache_dir or os.path.expandvars(
            str(cache_cfg.get("features_dir", ".cache/lpp_lme_features"))
        ),
        average_subject_precomputed_dir=os.path.expandvars(
            str(ds["average_subject_precomputed_dir"])
        ),
        reliable_voxel_fraction=(
            float(ds["reliable_voxel_fraction"]) if ds.get("reliable_voxel_fraction") else None
        ),
        reliable_voxel_scores_path=(
            os.path.expandvars(str(ds["reliable_voxel_scores_path"]))
            if ds.get("reliable_voxel_scores_path") else None
        ),
        brain_mask_path=os.path.expandvars(str(ds["brain_mask_path"])),
        fedorenko_atlas_nii_path=os.path.expandvars(str(ds["fedorenko_atlas_nii_path"])),
        fedorenko_labels_txt_path=(
            os.path.expandvars(str(ds["fedorenko_labels_txt_path"]))
            if ds.get("fedorenko_labels_txt_path") else None
        ),
    )
    model_cfg = ModelConfig(
        model_name=model_id,
        model_revision=revision if revision not in (None, "", "main") else None,
        device=dev,
        n_params=n_params,
        trust_remote_code=bool(trust_remote_code),
    )
    lme_cfg = LppLmeConfig(
        train_frac=float(lme.get("train_frac", 0.75)),
        split_random_state=int(lme.get("split_random_state", 0)),
        chunk_seconds=float(lme.get("chunk_seconds", 30.0)),
        verbose=bool(lme.get("verbose", False)),
        allow_slope_fallback=bool(lme.get("allow_slope_fallback", True)),
        layer_stride=layer_stride,
        last_layer_only=last_layer_only or bool(lme.get("last_layer_only", False)),
        use_persubject=(
            use_persubject if use_persubject is not None else bool(lme.get("use_persubject", True))
        ),
        persubject_bold_path=str(lme.get("persubject_bold_path", "data/lpp_fmri_extra/lpp_en_persubject_bold.parquet")),
        n_workers=n_workers,
        # CPU-only stage: R (pymer4/lme4) is CPU-only on Jean Zay. Surprisal
        # is expected to already be cached by
        # run/surprisal/neural/lpp_extract.py (GPU stage) — a cache miss here
        # raises a clear error instead of silently falling back to loading
        # the LLM on CPU. `lme.get(key, default)` only uses `default` when
        # the key is ABSENT -- configs/surprisal/neural/lpp.yaml has this key
        # present with an explicit `null`, which `.get()` returns as-is
        # (None -> False), silently defeating this safety check on every
        # --device cpu run regardless of the intended per-device default.
        # Confirmed in production (2026-09-14): opt-66b's CPU refit tried to
        # load and forward-pass a 66B model on 4 cores instead of raising.
        require_cached_surprisal=bool(
            lme.get("require_cached_surprisal")
            if lme.get("require_cached_surprisal") is not None
            else (dev == "cpu")
        ),
        save_word_predictions=(
            save_word_predictions if save_word_predictions is not None
            else bool(lme.get("save_word_predictions", True))
        ),
        roi_wise=bool(lme.get("roi_wise", False)),
    )
    return LppLmeBenchmarkConfig(
        data=data_cfg,
        model=model_cfg,
        lme=lme_cfg,
        output_dir=str(output_dir),
        canonical_model_name=model_id,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/lpp.yaml")
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
    p.add_argument("--persubject", dest="use_persubject", action="store_true", default=None,
                   help="Fit per-subject BOLD with a (1|subject) term (default: on -- see "
                        "LppLmeConfig.use_persubject's docstring)")
    p.add_argument("--average_subject", dest="use_persubject", action="store_false",
                   help="Fit the original average-subject BOLD (no subject term)")
    p.add_argument("--device", default=None,
                   help="Override dataset device (surprisal extraction fallback only)")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Surprisal cache dir populated by run/surprisal/neural/lpp_extract.py")
    p.add_argument("--save_word_predictions", action=argparse.BooleanOptionalAction, default=None,
                   help="Save each held-out TR's observed BOLD + baseline/full model predictions "
                        "to lpp_lme_per_word.csv (default: lme.save_word_predictions in config, "
                        "else True). On by default; pass --no-save_word_predictions to disable.")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = Path(
        args.out_dir or cfg.get("output", {}).get("save_dir", "outputs/lpp/surprisal_sweep")
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
        trust_remote_code = bool(model_cfgs[0].get("trust_remote_code", False)) if model_cfgs else None
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
        use_persubject=args.use_persubject,
        device=args.device,
        feature_cache_dir=args.feature_cache_dir,
        trust_remote_code=trust_remote_code,
        save_word_predictions=args.save_word_predictions,
    )

    logger.info(
        "LPP LME sweep: model=%s revision=%s n_workers=%d layer_stride=%d out_dir=%s",
        model_id, revision, args.n_workers, args.layer_stride, output_dir,
    )
    LppLmeBenchmark(bench_cfg).run(extra_fields=extra_fields)
    logger.info("Done.")


if __name__ == "__main__":
    main()
