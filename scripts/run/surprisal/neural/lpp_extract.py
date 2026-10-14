#!/usr/bin/env python3
"""Le Petit Prince fMRI — per-layer logit-lens surprisal extraction (GPU stage).

Populates the surprisal cache consumed by ``run/surprisal/neural/lpp_fit.py``. Split
into its own script/job because on Jean Zay, R (pymer4/lme4, needed for the
LME fit) is only available on CPU partitions, not GPU — so, exactly like
every other pooled LME in this codebase, extraction (needs the LLM, wants a
GPU for larger models) and LME fitting (needs R, CPU-only) must run as
separate jobs sharing a cache directory, not one combined job.

Also computes and caches this (model, revision)'s perplexity on the LPP text
corpus itself (``outputs_jeanzay/lpp/perplexity/...``, same convention as
``scripts/perplexity/pythia_checkpoints_n400.py``) right after
surprisal extraction, reusing the SAME already-loaded model: ``_get_cached_
causal_lm``'s cache key is ``(model_name, revision, device, tokenizer_no_
revision, ...)`` and does NOT include ``load_kw``, so calling it again here
with a matching key is a cache hit, not a second GPU load -- computing
perplexity as a separate later pass would reload every one of the 110
pythia_checkpoints models from scratch for no reason.

Examples
--------
python scripts/run/surprisal/neural/lpp_extract.py \\
    --config configs/surprisal/neural/lpp.yaml \\
    --model_name openai-community/gpt2 --device cuda

python scripts/run/surprisal/neural/lpp_extract.py \\
    --config configs/surprisal/neural/lpp.yaml \\
    --from-registry --family gpt2 --array-index 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
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
    LppLmeDataConfig,
)
from neuroai.models.registry import registry_model_configs, resolve_family_task, resolve_load_model_id
from neuroai.sweep import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _save_lpp_perplexity(
    model_id: str, revision: str, n_params: float | None,
    perplexity: float, n_tokens: int, max_length: int,
) -> None:
    perplexity_dir = Path("outputs_jeanzay") / "lpp/perplexity"
    safe = model_id.replace("/", "_")
    rev_safe = (revision or "main").replace("/", "_")
    out_dir = perplexity_dir / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rev_safe}_lpp_perplexity.json"
    record = {
        "model_id": model_id, "revision": revision or "main", "label": model_id.split("/")[-1],
        "n_params": n_params, "family": "pythia", "dataset": "lpp",
        "perplexity": perplexity, "perplexity_method": "logits_v2_per_word", "n_tokens": n_tokens, "max_length": max_length,
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved LPP perplexity %.4f for %s @ %s", perplexity, model_id, revision or "main")


def _collect_lpp_perplexity_summary() -> None:
    """Rebuild lpp_perplexity_summary.csv from EVERY *_lpp_perplexity.json on
    disk. Single pass, no concurrent-write risk -- run once after all
    extraction jobs finish, not during them (same convention as
    scripts/perplexity/pythia_checkpoints_n400.py's collect_summary)."""
    import pandas as pd

    perplexity_dir = Path("outputs_jeanzay") / "lpp/perplexity"
    rows = []
    for p in sorted(perplexity_dir.glob("*/*_lpp_perplexity.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "error" in d:
            continue
        rows.append(d)
    if not rows:
        logger.warning("No *_lpp_perplexity.json files found under %s", perplexity_dir)
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["model_id", "revision"], keep="last")
    out_csv = perplexity_dir / "lpp_perplexity_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info("wrote %d rows -> %s", len(df), out_csv)


def load_config(config_path: str | Path) -> dict[str, Any]:
    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw)


def _data_cfg(ds: dict, *, feature_cache_dir: str) -> LppLmeDataConfig:
    return LppLmeDataConfig(
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
        feature_cache_dir=feature_cache_dir,
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/lpp.yaml")
    p.add_argument("--model_name", default=None, help="Direct HF model id (skip registry)")
    p.add_argument("--revision", default=None)
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Override cache.features_dir from the config")
    p.add_argument("--skip_perplexity", action="store_true",
                   help="Skip the post-extraction LPP-corpus perplexity computation "
                        "(on by default -- reuses the already-loaded model, see module docstring)")
    p.add_argument("--collect_perplexity_summary", action="store_true",
                   help="Rebuild outputs_jeanzay/lpp/perplexity/lpp_perplexity_summary.csv from "
                        "every *_lpp_perplexity.json on disk and exit (run once after all "
                        "extraction jobs finish, not concurrently with them).")
    args = p.parse_args()

    if args.collect_perplexity_summary:
        _collect_lpp_perplexity_summary()
        return

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    cache_cfg = cfg.get("cache", {})
    feature_cache_dir = args.feature_cache_dir or os.path.expandvars(
        str(cache_cfg.get("features_dir", ".cache/lpp_lme_features"))
    )
    Path(feature_cache_dir).mkdir(parents=True, exist_ok=True)

    if args.from_registry:
        if not args.family or args.array_index is None:
            p.error("--from-registry requires --family and --array-index")
        task = resolve_family_task(args.family, args.array_index)
        model_cfgs = [m for m in registry_model_configs(task.registry_family) if m["model_id"] == task.model_id]
        model_id = resolve_load_model_id(model_cfgs[0], task.revision) if model_cfgs else task.model_id
        revision = task.revision
        n_params = model_cfgs[0]["n_params"] if model_cfgs else None
        trust_remote_code = bool(model_cfgs[0].get("trust_remote_code", False)) if model_cfgs else False
    elif args.model_name:
        model_id = args.model_name
        revision = args.revision
        n_params = None
        trust_remote_code = False
    else:
        p.error("Specify --model_name, or --from-registry --family F --array-index I")
        return

    data_cfg = _data_cfg(ds, feature_cache_dir=feature_cache_dir)
    model_cfg = ModelConfig(
        model_name=model_id,
        model_revision=revision if revision not in (None, "", "main") else None,
        device=args.device,
        n_params=n_params,
        trust_remote_code=trust_remote_code,
    )
    bench_cfg = LppLmeBenchmarkConfig(
        data=data_cfg, model=model_cfg, output_dir=feature_cache_dir, canonical_model_name=model_id,
    )

    logger.info(
        "LPP surprisal extraction: model=%s revision=%s device=%s cache_dir=%s",
        model_id, revision, args.device, feature_cache_dir,
    )
    LppLmeBenchmark(bench_cfg).extract_and_cache_surprisal()
    logger.info("Done — surprisal cached under %s", feature_cache_dir)

    if not args.skip_perplexity:
        _extract_perplexity(model_id, model_cfg, data_cfg, n_params, trust_remote_code)


def _extract_perplexity(
    model_id: str, model_cfg: ModelConfig, data_cfg: LppLmeDataConfig,
    n_params: float | None, trust_remote_code: bool,
) -> None:
    """Reuse the model that ``extract_and_cache_surprisal`` just loaded (see
    module docstring for why this is a cache hit, not a second GPU load)."""
    from neuroai.features.llm import _get_cached_causal_lm
    from scripts.perplexity.lpp import compute_perplexity, load_lpp_text, _max_length_for_model

    revision = model_cfg.model_revision
    out_path = (
        Path("outputs_jeanzay") / "lpp/perplexity" / model_id.replace("/", "_")
        / f"{(revision or 'main').replace('/', '_')}_lpp_perplexity.json"
    )
    if out_path.is_file() and json.loads(out_path.read_text()).get("perplexity_method") == "logits_v2_per_word":
        logger.info("LPP perplexity already cached for %s @ %s — skipping", model_id, revision or "main")
        return
    try:
        tokenizer, model = _get_cached_causal_lm(
            model_id,
            revision=revision,
            device=model_cfg.device,
            load_kw={"trust_remote_code": trust_remote_code},
            tokenizer_no_revision=data_cfg.tokenizer_no_revision,
            n_params=n_params,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_length = _max_length_for_model(model, model_id)
        texts = load_lpp_text(Path(data_cfg.bids_root), data_cfg.lang)
        result = compute_perplexity(model, tokenizer, texts, model_cfg.device, max_length)
        _save_lpp_perplexity(
            model_id, revision or "main", n_params,
            result["perplexity"], result["n_tokens"], max_length,
        )
    except Exception as exc:
        logger.error("LPP perplexity FAILED for %s @ %s: %s", model_id, revision or "main", exc, exc_info=True)


if __name__ == "__main__":
    main()
