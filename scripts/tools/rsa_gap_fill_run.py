#!/usr/bin/env python3
"""RSA gap-fill worker: for one group of models (see rsa_gap_fill_groups.py),
load each model once and run it through every dataset where it's missing an
embedding and/or surprisal cache. THROWAWAY TOOL -- delete this whole
scripts/tools/rsa_gap_fill_* trio once all six datasets below have full
registry coverage; it exists only to fill gaps left by earlier interrupted
sweeps, not as a permanent part of the benchmark.

Datasets covered (the ones with < full registry coverage as of 2026-09-24):
  - podcast     (embedding)
  - broderick   (embedding + surprisal)
  - nieuwland   (embedding)
  - naturalstories_spr (embedding + remaining surprisal gap)
  - ucl_reading (embedding + surprisal)
  - zuco_reading NR (embedding)

Design
------
Every one of these datasets' real run scripts (scripts/run/embedding/...,
scripts/run/surprisal/...) already has its own idempotent "already done"
check and writes to the exact same cache paths the real SLURM jobs use, so
this script does NOT reimplement any caching/config logic -- it just calls
each real script's ``main()`` in-process with a crafted ``sys.argv``, exactly
mirroring what its SLURM .sh wrapper passes. Re-running (e.g. after a
timeout) is safe: already-finished (model, dataset) cells are fast no-ops.

The one wrinkle: every real script calls ``clear_hf_model_cache()`` after
finishing its own checkpoint loop, which evicts the shared one-slot LLM
cache in ``neuroai.features.llm``. Calling six scripts back-to-back for the
same model would therefore reload the model six times despite being in one
process. This script suppresses that eviction between datasets for the same
model (monkeypatching each imported module's local ``clear_hf_model_cache``
reference to a no-op) and calls the real one exactly once, after all
datasets for that model are done -- so the model is genuinely loaded once
per group-model, then reused, then freed before the next model.

Usage:
    python scripts/tools/rsa_gap_fill_run.py --group 0 \\
        --groups-file scripts/tools/rsa_gap_fill_groups.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rsa_gap_fill")


def _env(name: str) -> str:
    """Required env var -- the wrapping .sh must export these (jz_cache_env.sh
    conventions), same as every real per-dataset SLURM script. Fail loudly
    rather than silently falling back to a possibly-wrong default."""
    val = os.environ.get(name, "").strip()
    if not val:
        raise SystemExit(f"Required env var {name} is not set — source the rsa_gap_fill.sh wrapper.")
    return val


def _run_main(module, argv: list[str], label: str) -> bool:
    """Call module.main() with a crafted argv. Returns True on success."""
    old_argv = sys.argv
    sys.argv = [module.__file__] + argv
    try:
        logger.info("  -> %s: %s", label, " ".join(argv))
        module.main()
        return True
    except SystemExit as exc:
        if exc.code in (0, None):
            return True
        logger.error("  %s FAILED (SystemExit %s)", label, exc.code)
        return False
    except Exception:
        logger.error("  %s FAILED", label, exc_info=True)
        traceback.print_exc()
        return False
    finally:
        sys.argv = old_argv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--groups-file", default="scripts/tools/rsa_gap_fill_groups.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    groups_data = json.loads(Path(args.groups_file).read_text())
    models = groups_data["groups"][args.group]
    logger.info("Group %d: %d models — %s", args.group, len(models), [m["model_id"] for m in models])

    # ── Import every target dataset's run script once ──────────────────────
    import scripts.run.embedding.neural.podcast as podcast_mod
    import scripts.run.embedding.neural.broderick as broderick_mod
    import scripts.run.embedding.neural.nieuwland as nieuwland_mod
    import scripts.run.embedding.behavioral.ucl_reading as ucl_embed_mod
    import scripts.run.embedding.behavioral.zuco_reading as zuco_embed_mod
    import scripts.run.embedding.behavioral.naturalstories as ns_embed_mod
    import scripts.run.surprisal.neural.broderick_extract as broderick_surp_mod
    import scripts.run.surprisal.behavioral.ucl_reading_extract as ucl_surp_mod
    import scripts.run.surprisal.behavioral.naturalstories_extract as ns_surp_mod

    embed_modules = [podcast_mod, broderick_mod, nieuwland_mod, ucl_embed_mod, zuco_embed_mod, ns_embed_mod]
    surp_modules = [broderick_surp_mod, ucl_surp_mod, ns_surp_mod]

    from neuroai.features import llm as llm_mod
    real_clear_hf_model_cache = llm_mod.clear_hf_model_cache

    def _suppress_cache_clear() -> None:
        for m in embed_modules + surp_modules:
            if hasattr(m, "clear_hf_model_cache"):
                m.clear_hf_model_cache = lambda: None

    def _restore_and_clear() -> None:
        for m in embed_modules + surp_modules:
            if hasattr(m, "clear_hf_model_cache"):
                m.clear_hf_model_cache = real_clear_hf_model_cache
        real_clear_hf_model_cache()

    device = args.device

    for entry in models:
        family = entry["family"]
        idx = entry["array_index"]
        model_id = entry["model_id"]
        logger.info("=== Model %s (family=%s idx=%d) ===", model_id, family, idx)

        _suppress_cache_clear()

        # ── podcast (embedding) ──────────────────────────────────────────
        _run_main(podcast_mod, [
            "--config", "configs/embedding/neural/podcast.yaml",
            "--from-registry", "--family", family, "--array-index", str(idx),
            "--device", device,
            "--out_dir", "outputs_jeanzay/podcast/embedding_sweep",
            "--n_jobs", "8",
            "--electrode_selection_dir", "outputs_jeanzay/podcast/electrode_selection",
            "--regression_device", device,
            "--skip_summary", "--no_plots",
        ], f"podcast embed [{model_id}]")

        # ── broderick (embedding) ────────────────────────────────────────
        _run_main(broderick_mod, [
            "--config", "configs/embedding/neural/broderick.yaml",
            "--from-registry", "--family", family, "--array-index", str(idx),
            "--device", device,
            "--out_dir", "outputs_jeanzay/broderick/embedding_sweep",
            "--n_jobs", "8",
            "--regression_device", device,
            "--n_layer_workers", "4",
            "--n_subjects_batch", "1",
            "--skip_summary", "--no_plots",
        ], f"broderick embed [{model_id}]")

        # ── broderick (surprisal) ────────────────────────────────────────
        _run_main(broderick_surp_mod, [
            "--config", "configs/surprisal/neural/broderick.yaml",
            "--from-registry", "--family", family, "--array-index", str(idx),
            "--device", device,
            "--feature_cache_dir", _env("BRODERICK_LME_FEATURE_CACHE_DIR"),
        ], f"broderick surprisal [{model_id}]")

        # ── nieuwland (embedding) ────────────────────────────────────────
        _run_main(nieuwland_mod, [
            "--config", "configs/embedding/neural/nieuwland.yaml",
            "--from-registry", "--family", family, "--array-index", str(idx),
            "--device", device,
            "--out_dir", "outputs_jeanzay/nieuwland/embedding_sweep",
            "--feature_cache_dir", _env("NIEUWLAND_FEATURE_CACHE_DIR"),
            "--epoch_cache_dir", _env("NIEUWLAND_EPOCH_CACHE_DIR"),
            "--regression_device", device,
            "--n_layer_workers", "36",
        ], f"nieuwland embed [{model_id}]")

        # ── ucl_reading (embedding) ──────────────────────────────────────
        _run_main(ucl_embed_mod, [
            "--config", "configs/embedding/behavioral/ucl_reading.yaml",
            "--from-registry", "--family", family, "--output_family", family,
            "--array-index", str(idx),
            "--device", device,
            "--out_dir", "outputs_jeanzay/ucl_reading/embedding_sweep",
            "--skip_summary", "--no_plots",
        ], f"ucl_reading embed [{model_id}]")

        # ── ucl_reading (surprisal) ──────────────────────────────────────
        _run_main(ucl_surp_mod, [
            "--config", "configs/surprisal/behavioral/ucl_reading.yaml",
            "--family", family, "--array-index", str(idx),
            "--device", device,
            "--dataset-root", _env("UCL_READING_DATA_ROOT"),
            "--cache-dir", _env("UCL_READING_FEATURE_CACHE_DIR"),
        ], f"ucl_reading surprisal [{model_id}]")

        # ── zuco_reading NR (embedding) ──────────────────────────────────
        _run_main(zuco_embed_mod, [
            "--config", "configs/embedding/behavioral/zuco_reading_nr.yaml",
            "--from-registry", "--family", family, "--output_family", family,
            "--array-index", str(idx),
            "--device", device,
            "--out_dir", "outputs_jeanzay/zuco_reading/embedding_sweep/nr",
            "--skip_summary", "--no_plots",
        ], f"zuco_reading(NR) embed [{model_id}]")

        # ── naturalstories_spr (embedding, real averaged-subject+himalaya fit) ──
        _run_main(ns_embed_mod, [
            "--config", "configs/embedding/behavioral/naturalstories.yaml",
            "--from-registry", "--family", family, "--array-index", str(idx),
            "--out_dir", "outputs_jeanzay/naturalstories/embedding_sweep",
            "--skip_summary", "--no_plots",
        ], f"naturalstories embed [{model_id}]")

        # ── naturalstories_spr (surprisal, remaining gap) ────────────────
        _run_main(ns_surp_mod, [
            "--config", "configs/surprisal/behavioral/naturalstories.yaml",
            "--family", family, "--array-index", str(idx),
            "--device", device,
            "--dataset-root", _env("BENCHMARK_ROOT") + "/data/naturalstories",
            "--cache-dir", _env("NS_FEATURE_CACHE_DIR"),
        ], f"naturalstories surprisal [{model_id}]")

        _restore_and_clear()
        logger.info("=== Done with %s ===", model_id)

    logger.info("Group %d complete.", args.group)


if __name__ == "__main__":
    main()
