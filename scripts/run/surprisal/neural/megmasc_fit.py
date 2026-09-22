#!/usr/bin/env python3
"""MEG-MASC MEG pooled LME — one mixed-effects model per LM layer, region-averaged.

Fits ``eeg ~ wlen_z + freq_z + surprisal_z + prev_surprisal_z +
(1 + surprisal_z | region)`` vs. the same formula without ``surprisal_z``
and ``prev_surprisal_z``, once per LM layer per SUBJECT (then averaged
across subjects), pooling one of 4 scalp regions (left/right/posterior/
occipital — MEG-MASC's 64-channel BioSemi montage collapsed via
``channel_region``, same coarse quadrant convention as Broderick — see
``neuroai/regressors/megmasc_lme.py`` module docstring for why).

CPU-only stage: on Jean Zay, R (pymer4/lme4) is only available on CPU
partitions, not GPU — so per-layer surprisal must already be cached by
``scripts/run/surprisal/neural/megmasc_extract.py`` (GPU stage) before
running this. A cache miss silently falls back to loading the LLM on CPU
(slow, not wrong) rather than hard-erroring — same soft contract as every
other pooled LME in this codebase.

Examples
--------
# Single model, all layers, 10 concurrent worker processes:
python scripts/run/surprisal/neural/megmasc_fit.py \\
    --config configs/surprisal/neural/megmasc.yaml \\
    --model_name openai-community/gpt2 --n_workers 10

# From the registry (single array task, one model):
python scripts/run/surprisal/neural/megmasc_fit.py \\
    --config configs/surprisal/neural/megmasc.yaml \\
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
from neuroai.benchmarks.megmasc_lme import (
    MegMascLmeBenchmark,
    MegMascLmeBenchmarkConfig,
    MegMascLmeConfig,
    MegMascLmeDataConfig,
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
    device: str | None = None,
    feature_cache_dir: str | None = None,
    epoch_cache_dir: str | None = None,
    trust_remote_code: bool | None = None,
    save_word_predictions: bool = True,
) -> MegMascLmeBenchmarkConfig:
    ds = cfg["dataset"]
    lme = cfg.get("lme", {})
    cache_cfg = cfg.get("cache", {})
    dev = device or "cpu"

    data_cfg = MegMascLmeDataConfig(
        bids_root=os.path.expandvars(str(ds.get("bids_root", "/lustre/fsn1/projects/rech/ldh/una68ug/data/gwilliams"))),
        subjects=ds.get("subjects") or None,
        stories=ds.get("stories") or None,
        window_tmin=float(ds.get("window_tmin", 0.25)),
        window_tmax=float(ds.get("window_tmax", 0.45)),
        epoch_tmin=float(ds.get("epoch_tmin", -0.2)),
        epoch_tmax=float(ds.get("epoch_tmax", 0.8)),
        baseline=tuple(ds["baseline"]) if ds.get("baseline") else (-0.2, 0.0),
        l_freq=float(ds["l_freq"]) if ds.get("l_freq") is not None else 0.1,
        h_freq=float(ds["h_freq"]) if ds.get("h_freq") is not None else 40.0,
        target_sfreq=float(ds.get("target_sfreq", 50.0)),
        robust_scale=bool(ds.get("robust_scale", True)),
        clamp=(float(ds["clamp"]) if ds.get("clamp") is not None else None),
        min_word_duration=float(ds.get("min_word_duration", 0.050)),
        max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        tokenizer_no_revision=bool(ds.get("tokenizer_no_revision", True)),
        trust_remote_code=(
            trust_remote_code if trust_remote_code is not None
            else bool(ds.get("trust_remote_code", False))
        ),
        wordfreq_language=str(ds.get("wordfreq_language", "en")),
        lang=str(ds.get("lang", "en")),
        device=dev,
        feature_cache_dir=feature_cache_dir or os.path.expandvars(
            str(cache_cfg.get("features_dir", ".cache/megmasc_lme_features"))
        ),
        epoch_cache_dir=epoch_cache_dir or (
            os.path.expandvars(str(cache_cfg["epochs_dir"]))
            if cache_cfg.get("epochs_dir") else None
        ),
    )
    model_cfg = ModelConfig(
        model_name=model_id,
        model_revision=revision if revision not in (None, "", "main") else None,
        device=dev,
        n_params=n_params,
        trust_remote_code=bool(trust_remote_code),
    )
    lme_cfg = MegMascLmeConfig(
        train_frac=float(lme.get("train_frac", 0.75)),
        split_random_state=int(lme.get("split_random_state", 0)),
        verbose=bool(lme.get("verbose", False)),
        allow_slope_fallback=bool(lme.get("allow_slope_fallback", True)),
        layer_stride=layer_stride,
        n_workers=n_workers,
        # CPU-only stage: R (pymer4/lme4) is CPU-only on Jean Zay. Surprisal
        # is expected to already be cached by
        # run/surprisal/neural/megmasc_extract.py (GPU stage) — a cache miss
        # here raises a clear error instead of silently falling back to
        # loading the LLM on CPU.
        require_cached_surprisal=bool(lme.get("require_cached_surprisal", dev == "cpu")),
        save_word_predictions=save_word_predictions,
        nuisances=str(lme.get("nuisances", MegMascLmeConfig.nuisances)),
        pooling=str(lme.get("pooling", MegMascLmeConfig.pooling)),
    )
    return MegMascLmeBenchmarkConfig(
        data=data_cfg,
        model=model_cfg,
        lme=lme_cfg,
        output_dir=str(output_dir),
        canonical_model_name=model_id,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/megmasc.yaml")
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
    p.add_argument("--device", default=None,
                   help="Override dataset device (surprisal extraction fallback only)")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Surprisal cache dir populated by run/surprisal/neural/megmasc_extract.py")
    p.add_argument("--epoch_cache_dir", default=None,
                   help="Word-locked epoch cache dir (data/megmasc_epoch_cache.py / the ridge "
                        "encoder's cache) — avoids re-reading raw MEG. Default: cache.epochs_dir "
                        "from the config, or $MEGMASC_EPOCH_CACHE_DIR.")
    p.add_argument("--pooling", choices=["subject", "none"], default=None,
                   help="'subject' (default): ONE pooled LME per layer, all subjects together, "
                        "with (1+surprisal_z|subject) crossed with (1+surprisal_z|region). "
                        "'none': the old two-stage per-subject-then-average design.")
    p.add_argument("--save_word_predictions", action=argparse.BooleanOptionalAction, default=True,
                   help="Save each held-out row's observed eeg + baseline/full model predictions "
                        "to a per-word CSV for post-hoc per-word difficulty splits (DLT). "
                        "On by default; pass --no-save_word_predictions to disable.")
    p.add_argument("--nuisances", default=None,
                   help="LME nuisance terms. Default (config or "
                        "'worddur_z + wlen_z + freq_z + prev_worddur_z'); pass "
                        "'wlen_z + freq_z' for a Broderick-parity run.")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = Path(
        args.out_dir or cfg.get("output", {}).get("save_dir", "outputs/megmasc/surprisal_sweep")
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

    if args.nuisances:
        cfg.setdefault("lme", {})["nuisances"] = args.nuisances
    if args.pooling:
        cfg.setdefault("lme", {})["pooling"] = args.pooling

    output_dir.mkdir(parents=True, exist_ok=True)
    bench_cfg = _build_benchmark_config(
        cfg,
        model_id=model_id,
        revision=revision,
        n_params=n_params,
        output_dir=output_dir,
        n_workers=args.n_workers,
        layer_stride=args.layer_stride,
        device=args.device,
        feature_cache_dir=args.feature_cache_dir,
        epoch_cache_dir=args.epoch_cache_dir,
        trust_remote_code=trust_remote_code,
        save_word_predictions=args.save_word_predictions,
    )

    logger.info(
        "MEG-MASC LME sweep: model=%s revision=%s n_workers=%d layer_stride=%d "
        "pooling=%s out_dir=%s",
        model_id, revision, args.n_workers, args.layer_stride, bench_cfg.lme.pooling, output_dir,
    )
    MegMascLmeBenchmark(bench_cfg).run()
    logger.info("Done.")


if __name__ == "__main__":
    main()
