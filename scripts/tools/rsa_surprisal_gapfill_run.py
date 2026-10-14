#!/usr/bin/env python3
"""Extraction-only RSA surprisal cache gap fill, sharing one model load across datasets."""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s rsa_surp_gapfill: %(message)s")
log = logging.getLogger("rsa_surp_gapfill")

DATASETS = {
    "naturalstories": ("scripts.run.surprisal.behavioral.naturalstories_extract", "configs/surprisal/behavioral/naturalstories.yaml"),
    "ucl_reading": ("scripts.run.surprisal.behavioral.ucl_reading_extract", "configs/surprisal/behavioral/ucl_reading.yaml"),
    "tuckute": ("scripts.run.surprisal.neural.tuckute_extract", "configs/surprisal/neural/tuckute.yaml"),
    "fedorenko2016": ("scripts.run.surprisal.neural.fedorenko2016_extract", "configs/surprisal/neural/fedorenko2016.yaml"),
    "nieuwland": ("scripts.run.surprisal.neural.nieuwland_extract", "configs/surprisal/neural/nieuwland.yaml"),
    "broderick": ("scripts.run.surprisal.neural.broderick_extract", "configs/surprisal/neural/broderick.yaml"),
    "podcast": ("scripts.run.surprisal.neural.podcast_extract", "configs/surprisal/neural/podcast.yaml"),
    "szewczyk": ("scripts.run.surprisal.neural.szewczyk_extract", "configs/surprisal/neural/szewczyk.yaml"),
}


def _cache_args(ds: str) -> list[str]:
    root = Path(os.environ["BENCHMARK_CACHE_ROOT"])
    dirs = {
        "naturalstories": root / "naturalstories_features",
        "ucl_reading": root / "ucl_reading_features",
        "tuckute": root / "tuckute_lme_features",
        "fedorenko2016": root / "fedorenko2016_lme_features",
        "nieuwland": root / "nieuwland_lme_features",
        "broderick": root / "broderick_lme_features",
        "podcast": root / "podcast_features",
        "szewczyk": root / "szewczyk_lme_features",
    }
    cache = str(dirs[ds])
    if ds in ("naturalstories", "ucl_reading", "podcast"):
        flag = "--cache-dir"
    else:
        flag = "--feature_cache_dir"
    args = [flag, cache]
    if ds == "naturalstories":
        args += ["--dataset-root", str(ROOT / "data/naturalstories")]
    elif ds == "ucl_reading":
        args += ["--dataset-root", str(ROOT / "data/ucl_reading"), "--perplexity-dir", ""]
    elif ds == "podcast":
        args += ["--bids-root", str(ROOT / "data/podcast"), "--skip_perplexity"]
    elif ds not in {"broderick", "szewczyk"}:
        args += ["--skip_perplexity"]
    return args


def _invoke(module, argv: list[str], label: str) -> bool:
    prev = sys.argv
    try:
        sys.argv = [module.__file__, *argv]
        log.info("extract %s: %s", label, " ".join(argv))
        module.main()
        return True
    except SystemExit as exc:
        if exc.code in (0, None):
            return True
        log.exception("failed %s (SystemExit %s)", label, exc.code)
    except Exception:
        log.exception("failed %s", label)
        traceback.print_exc()
        # A kernel failure poisons the CUDA context. Stop instead of recording
        # misleading failures for every remaining dataset in this process.
        exc = sys.exc_info()[1]
        if "CUDA error:" in str(exc):
            raise
    finally:
        sys.argv = prev
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--groups-file", default="scripts/tools/rsa_surprisal_gapfill_groups.json")
    a = p.parse_args()
    groups = json.loads(Path(a.groups_file).read_text())["groups"]
    if not 0 <= a.group < len(groups):
        raise SystemExit(f"group {a.group} is outside 0..{len(groups)-1}")

    # Use the models' native output layer only. RSA needs a single final-layer
    # surprisal value per item; constructing intermediate logit-lens layers is
    # wasted work here. Extraction writes the usual cache files; no LME or fit.
    import neuroai.features.logit_lens as ll
    for name in ("logit_lens_surprisal_trials", "logit_lens_surprisal_all_layers"):
        original = getattr(ll, name)
        def native_only(*args, _orig=original, **kwargs):
            kwargs["native_only"] = True
            return _orig(*args, **kwargs)
        setattr(ll, name, native_only)

    modules = {ds: importlib.import_module(spec[0]) for ds, spec in DATASETS.items()}
    from neuroai.features import llm
    clear_model_cache = llm.clear_hf_model_cache
    for mod in modules.values():
        if hasattr(mod, "clear_hf_model_cache"):
            mod.clear_hf_model_cache = lambda: None

    failures = []
    work = groups[a.group]
    log.info("group %d: %d models, %d dataset-model extractions", a.group, len(work), sum(len(x['datasets']) for x in work))
    for entry in work:
        family, idx, model_id = entry["family"], entry["array_index"], entry["model_id"]
        log.info("MODEL %s family=%s idx=%s datasets=%s", model_id, family, idx, entry["datasets"])
        for ds in entry["datasets"]:
            mod = modules[ds]
            config = DATASETS[ds][1]
            if ds in ("naturalstories", "ucl_reading", "podcast"):
                argv = ["--config", config, "--family", family, "--array-index", str(idx), "--device", "cuda", *_cache_args(ds)]
            else:
                argv = ["--config", config, "--from-registry", "--family", family, "--array-index", str(idx), "--device", "cuda", *_cache_args(ds)]
            if not _invoke(mod, argv, f"{ds}/{model_id}"):
                failures.append((ds, model_id))
        clear_model_cache()
    for mod in modules.values():
        if hasattr(mod, "clear_hf_model_cache"):
            mod.clear_hf_model_cache = clear_model_cache
    if failures:
        log.error("group %d completed with %d failures: %s", a.group, len(failures), failures)
        raise SystemExit(1)
    log.info("group %d complete", a.group)


if __name__ == "__main__":
    main()
