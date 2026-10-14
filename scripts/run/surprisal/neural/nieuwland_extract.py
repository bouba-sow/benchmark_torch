#!/usr/bin/env python3
"""Nieuwland N400 — per-layer logit-lens surprisal extraction (GPU stage).

Populates the surprisal cache consumed by ``run/surprisal/neural/nieuwland_fit.py``.
Split into its own script/job because on Jean Zay, R (pymer4/lme4, needed
for the LME fit) is only available on CPU partitions, not GPU — so, exactly
like Podcast's pooled LME, extraction (needs the LLM,
wants a GPU for larger models) and LME fitting (needs R, CPU-only) must run
as separate jobs sharing a cache directory, not one combined job.

EEG-free: uses ``load_all_items`` (pools ``events.tsv`` across subjects),
NOT ``load_all_epochs`` — never touches raw EEG, so this stage is as cheap
as the earlier N400 datasets' (~160 short trial items, two forward passes each).

Examples
--------
python scripts/run/surprisal/neural/nieuwland_extract.py \\
    --config configs/surprisal/neural/nieuwland.yaml \\
    --model_name openai-community/gpt2 --device cuda

python scripts/run/surprisal/neural/nieuwland_extract.py \\
    --config configs/surprisal/neural/nieuwland.yaml \\
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
from neuroai.benchmarks.n400_nieuwland_lme import (
    NieuwlandLmeBenchmark,
    NieuwlandLmeBenchmarkConfig,
    NieuwlandLmeDataConfig,
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


_PERPLEXITY_DIR = Path("outputs_jeanzay") / "nieuwland/perplexity"


def _save_nieuwland_perplexity(
    model_id: str, revision: str, n_params: float | None,
    perplexity: float, n_tokens: int, max_length: int,
) -> None:
    out_dir = _PERPLEXITY_DIR / model_id.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{revision.replace('/', '_')}_nieuwland_perplexity.json"
    out_path.write_text(json.dumps({
        "model_id": model_id, "revision": revision, "n_params": n_params,
        "dataset": "nieuwland", "perplexity": perplexity, "perplexity_method": "logits_v2_per_word",
        "n_tokens": n_tokens, "max_length": max_length,
    }, indent=2))
    logger.info("Saved nieuwland perplexity %.4f for %s @ %s", perplexity, model_id, revision)


def _collect_nieuwland_perplexity_summary() -> None:
    """Rebuild nieuwland_perplexity_summary.csv from every *_nieuwland_perplexity.json
    on disk (run once after all extraction jobs finish, not concurrently)."""
    import pandas as pd

    rows = []
    for p in sorted(_PERPLEXITY_DIR.glob("*/*_nieuwland_perplexity.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        logger.warning("No *_nieuwland_perplexity.json files found under %s", _PERPLEXITY_DIR)
        return
    df = pd.DataFrame(rows)
    out_csv = _PERPLEXITY_DIR / "nieuwland_perplexity_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info("wrote %d rows -> %s", len(df), out_csv)


def _extract_perplexity(
    model_id: str, model_cfg: ModelConfig, data_cfg: "NieuwlandLmeDataConfig",
    n_params: float | None, trust_remote_code: bool,
) -> None:
    """Reuse the model that ``extract_and_cache_surprisal`` just loaded --
    ``_get_cached_causal_lm``'s (size-1) cache key does not include ``load_kw``,
    so this is a cache hit, not a second GPU load (same pattern as
    ``scripts/run/surprisal/neural/lpp_extract.py``)."""
    from neuroai.features.llm import _get_cached_causal_lm
    from scripts.perplexity.lpp import compute_perplexity, _max_length_for_model
    from scripts.perplexity.nieuwland import load_nieuwland_text

    revision = model_cfg.model_revision or "main"
    out_path = (
        _PERPLEXITY_DIR / model_id.replace("/", "_")
        / f"{revision.replace('/', '_')}_nieuwland_perplexity.json"
    )
    if out_path.is_file() and json.loads(out_path.read_text()).get("perplexity_method") == "logits_v2_per_word":
        logger.info("nieuwland perplexity already cached for %s @ %s — skipping", model_id, revision)
        return
    try:
        tokenizer, model = _get_cached_causal_lm(
            model_id,
            revision=model_cfg.model_revision,
            device=model_cfg.device,
            load_kw={"trust_remote_code": trust_remote_code},
            tokenizer_no_revision=data_cfg.tokenizer_no_revision,
            n_params=n_params,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_length = _max_length_for_model(model, model_id)
        texts = load_nieuwland_text(Path(data_cfg.bids_root))
        result = compute_perplexity(model, tokenizer, texts, model_cfg.device, max_length)
        _save_nieuwland_perplexity(
            model_id, revision, n_params,
            result["perplexity"], result["n_tokens"], max_length,
        )
    except Exception as exc:
        logger.error("nieuwland perplexity FAILED for %s @ %s: %s", model_id, revision, exc, exc_info=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/nieuwland.yaml")
    p.add_argument("--model_name", default=None, help="Direct HF model id (skip registry)")
    p.add_argument("--revision", default=None)
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Override cache.features_dir from the config")
    p.add_argument("--skip_perplexity", action="store_true",
                   help="Skip the post-extraction Nieuwland-corpus perplexity computation "
                        "(on by default -- reuses the already-loaded model)")
    p.add_argument("--collect_perplexity_summary", action="store_true",
                   help="Rebuild outputs_jeanzay/nieuwland/perplexity/nieuwland_perplexity_summary.csv "
                        "from every *_nieuwland_perplexity.json on disk and exit.")
    args = p.parse_args()

    if args.collect_perplexity_summary:
        _collect_nieuwland_perplexity_summary()
        return

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    cache_cfg = cfg.get("cache", {})
    feature_cache_dir = args.feature_cache_dir or os.path.expandvars(
        str(cache_cfg.get("features_dir", ".cache/nieuwland_lme_features"))
    )
    # Per-model registry entry (e.g. the FLA-based stickbreaking/alibi/
    # forgetting/hard-n-gram families, trust_remote_code=True) overrides the
    # YAML default — same resolution order as run/embedding/neural/nieuwland.py.
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

    data_cfg = NieuwlandLmeDataConfig(
        bids_root=os.path.expandvars(str(ds.get("bids_root", "data/nieuwland"))),
        subjects=ds.get("subjects"),
        included_in_paper_only=bool(ds.get("included_in_paper_only", True)),
        exclude_labs=list(ds.get("exclude_labs", ["glas"])),
        target_channels=list(ds.get("target_channels", ["Cz", "C3", "C4", "Pz", "P3", "P4"])),
        target_sfreq=float(ds.get("target_sfreq", 32.0)),
        epoch_tmin=float(ds.get("epoch_tmin", -0.2)),
        epoch_tmax=float(ds.get("epoch_tmax", 0.8)),
        baseline=tuple(ds.get("baseline", [-0.2, 0.0])),
        artifact_threshold_uv=float(ds.get("artifact_threshold_uv", 100.0)),
        min_trials=int(ds.get("min_trials", 20)),
        n400_tmin=float(ds.get("n400_tmin", 0.2)),
        n400_tmax=float(ds.get("n400_tmax", 0.5)),
        max_length=int(ds["max_length"]) if ds.get("max_length") else None,
        tokenizer_no_revision=bool(ds.get("tokenizer_no_revision", True)),
        trust_remote_code=trust_remote_code,
        wordfreq_language=str(ds.get("wordfreq_language", "en")),
        device=args.device,
        feature_cache_dir=feature_cache_dir,
    )
    model_cfg = ModelConfig(
        model_name=model_id,
        model_revision=revision if revision not in (None, "", "main") else None,
        device=args.device,
        n_params=n_params,
    )
    bench_cfg = NieuwlandLmeBenchmarkConfig(
        data=data_cfg, model=model_cfg, output_dir=feature_cache_dir, canonical_model_name=model_id,
    )

    logger.info(
        "Nieuwland surprisal extraction: model=%s revision=%s device=%s cache_dir=%s",
        model_id, revision, args.device, feature_cache_dir,
    )
    surprisal, prev_surprisal = NieuwlandLmeBenchmark(bench_cfg).extract_and_cache_surprisal()
    logger.info(
        "Done — surprisal %s, prev_surprisal %s cached under %s",
        surprisal.shape, prev_surprisal.shape, feature_cache_dir,
    )

    if not args.skip_perplexity:
        _extract_perplexity(model_id, model_cfg, data_cfg, n_params, trust_remote_code)


if __name__ == "__main__":
    main()
