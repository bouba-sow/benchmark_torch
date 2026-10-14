#!/usr/bin/env python3
"""Pereira fMRI — per-layer last-word logit-lens surprisal extraction (GPU stage).

Populates the surprisal cache consumed by ``run/surprisal/neural/pereira_fit.py``.
Split into its own script/job because on Jean Zay, R (pymer4/lme4, needed
for the LME fit) is only available on CPU partitions, not GPU — so, exactly
like every other pooled LME in this codebase, extraction (needs the LLM,
wants a GPU for larger models) and LME fitting (needs R, CPU-only) must run
as separate jobs sharing a cache directory, not one combined job.

This is a very light GPU stage: 627 sentences, one forward pass per
sentence — not a full narrative's worth of sliding-window chunks.

Also computes and caches this (model, revision)'s perplexity on the Pereira
sentence corpus itself (``outputs_torch/pereira/perplexity/...``, same
convention as ``scripts/perplexity/pythia_checkpoints_n400.py``)
right after surprisal extraction, reusing the SAME already-loaded model —
see ``run/surprisal/neural/lpp_extract.py``'s module docstring for why this is
a cache hit (``_get_cached_causal_lm``'s cache key excludes ``load_kw``),
not a second GPU load.

Examples
--------
python scripts/run/surprisal/neural/pereira_extract.py \\
    --config configs/surprisal/neural/pereira.yaml \\
    --model_name openai-community/gpt2 --device cuda

python scripts/run/surprisal/neural/pereira_extract.py \\
    --config configs/surprisal/neural/pereira.yaml \\
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
from neuroai.benchmarks.pereira_lme import (
    PereiraLmeBenchmark,
    PereiraLmeBenchmarkConfig,
    PereiraLmeDataConfig,
)
from neuroai.models.registry import registry_model_configs, resolve_family_task, resolve_load_model_id
from neuroai.sweep import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _save_pereira_perplexity(
    model_id: str, revision: str, n_params: float | None,
    perplexity: float, n_tokens: int, max_length: int,
) -> None:
    perplexity_dir = Path("outputs_torch") / "pereira/perplexity"
    safe = model_id.replace("/", "_")
    rev_safe = (revision or "main").replace("/", "_")
    out_dir = perplexity_dir / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rev_safe}_pereira_perplexity.json"
    record = {
        "model_id": model_id, "revision": revision or "main", "label": model_id.split("/")[-1],
        "n_params": n_params, "family": "pythia", "dataset": "pereira",
        "perplexity": perplexity, "n_tokens": n_tokens, "max_length": max_length,
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved Pereira perplexity %.4f for %s @ %s", perplexity, model_id, revision or "main")


def _collect_pereira_perplexity_summary() -> None:
    import pandas as pd

    perplexity_dir = Path("outputs_torch") / "pereira/perplexity"
    rows = []
    for p in sorted(perplexity_dir.glob("*/*_pereira_perplexity.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "error" in d:
            continue
        rows.append(d)
    if not rows:
        logger.warning("No *_pereira_perplexity.json files found under %s", perplexity_dir)
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["model_id", "revision"], keep="last")
    out_csv = perplexity_dir / "pereira_perplexity_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info("wrote %d rows -> %s", len(df), out_csv)


def _extract_perplexity(
    model_id: str, model_cfg: ModelConfig, data_cfg: PereiraLmeDataConfig, n_params: float | None,
) -> None:
    """Reuse the model extract_and_cache_surprisal just loaded."""
    from neuroai.datasets.pereira import PereiraDataset
    from neuroai.features.llm import _get_cached_causal_lm
    from scripts.perplexity.lpp import compute_perplexity, _max_length_for_model

    revision = model_cfg.model_revision
    out_path = (
        Path("outputs_torch") / "pereira/perplexity" / model_id.replace("/", "_")
        / f"{(revision or 'main').replace('/', '_')}_pereira_perplexity.json"
    )
    if out_path.is_file():
        logger.info("Pereira perplexity already cached for %s @ %s — skipping", model_id, revision or "main")
        return
    try:
        sentences, _ = PereiraDataset(data_cfg.nc_path).get_sentences()
        if "cbr-rnn" in model_id.lower() or "cbr_rnn" in model_id.lower():
            # Custom PyTorchModelHubMixin RNN, not AutoModelForCausalLM --
            # see compute_lpp_perplexity.compute_cbr_rnn_perplexity_over_texts's
            # docstring for why this is a separate code path.
            from scripts.perplexity.lpp import compute_cbr_rnn_perplexity_over_texts

            result = compute_cbr_rnn_perplexity_over_texts(model_id, sentences, model_cfg.device)
            _save_pereira_perplexity(
                model_id, revision or "main", n_params,
                result["perplexity"], result["n_tokens"], result["max_length"],
            )
            return
        tokenizer, model = _get_cached_causal_lm(
            model_id,
            revision=revision,
            device=model_cfg.device,
            load_kw={"trust_remote_code": data_cfg.trust_remote_code},
            tokenizer_no_revision=data_cfg.tokenizer_no_revision,
            n_params=n_params,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_length = _max_length_for_model(model, model_id)
        result = compute_perplexity(model, tokenizer, sentences, model_cfg.device, max_length)
        _save_pereira_perplexity(
            model_id, revision or "main", n_params,
            result["perplexity"], result["n_tokens"], max_length,
        )
    except Exception as exc:
        logger.error("Pereira perplexity FAILED for %s @ %s: %s", model_id, revision or "main", exc, exc_info=True)


def load_config(config_path: str | Path) -> dict[str, Any]:
    if "BENCHMARK_ROOT" not in os.environ:
        os.environ["BENCHMARK_ROOT"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/surprisal/neural/pereira.yaml")
    p.add_argument("--model_name", default=None, help="Direct HF model id (skip registry)")
    p.add_argument("--revision", default=None)
    p.add_argument("--from-registry", action="store_true")
    p.add_argument("--family", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--atlas", default=None, help="Override dataset.atlas")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Override cache.features_dir from the config")
    p.add_argument("--skip_perplexity", action="store_true",
                   help="Skip the post-extraction Pereira-corpus perplexity computation")
    p.add_argument("--collect_perplexity_summary", action="store_true",
                   help="Rebuild outputs_torch/pereira/perplexity/pereira_perplexity_summary.csv "
                        "from every *_pereira_perplexity.json on disk and exit.")
    args = p.parse_args()

    if args.collect_perplexity_summary:
        _collect_pereira_perplexity_summary()
        return

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    cache_cfg = cfg.get("cache", {})
    feature_cache_dir = args.feature_cache_dir or os.path.expandvars(
        str(cache_cfg.get("features_dir", ".cache/pereira_lme_features"))
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

    data_cfg = PereiraLmeDataConfig(
        nc_path=os.path.expandvars(str(ds.get("nc_path", "data/pereira_fmri/Pereira2018_merged.nc"))),
        atlas=args.atlas or str(ds.get("atlas", "language")),
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
    bench_cfg = PereiraLmeBenchmarkConfig(
        data=data_cfg, model=model_cfg, output_dir=feature_cache_dir, canonical_model_name=model_id,
    )

    logger.info(
        "Pereira surprisal extraction: model=%s revision=%s device=%s cache_dir=%s",
        model_id, revision, args.device, feature_cache_dir,
    )
    surprisal = PereiraLmeBenchmark(bench_cfg).extract_and_cache_surprisal()
    logger.info("Done — surprisal %s cached under %s", surprisal.shape, feature_cache_dir)

    if not args.skip_perplexity:
        _extract_perplexity(model_id, model_cfg, data_cfg, n_params)


if __name__ == "__main__":
    main()
