#!/usr/bin/env python3
"""RSA raw-embedding-cache backfill worker: for one group of models (see
rsa_cache_backfill_groups.py), load each model once and run it through every
dataset where its raw per-word/per-sentence embedding cache is missing.
THROWAWAY TOOL -- delete alongside rsa_cache_backfill_groups.py and .sh once
the backfill is done. Separate from and does not touch the unrelated
scripts/tools/rsa_gap_fill_* trio.

Every dataset's real run script already has its own cache-write path and
(for these six) an --activations_only / --extract_only flag that skips the
ridge fit and bypasses the "already done" skip-check -- this script does not
reimplement any caching logic, it just calls each real script's main() in
process with a crafted sys.argv, exactly mirroring its SLURM wrapper.
Re-running after a timeout is safe: an already-cached (model, dataset) pair
is a fast no-op inside the underlying feature-extraction cache lookup itself.

Pereira (only 2 missing models, no activations_only flag needed -- its run
script has no skip-check so a normal run is cheap) is handled by
rsa_cache_backfill_pereira_run.py instead; not part of this script.

Usage:
    python scripts/tools/rsa_cache_backfill_run.py --group 0 \\
        --groups-file scripts/tools/rsa_cache_backfill_groups.json
"""
from __future__ import annotations

import argparse
import json
import logging
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
logger = logging.getLogger("rsa_cache_backfill")


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
    p.add_argument("--groups-file", default="scripts/tools/rsa_cache_backfill_groups.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    groups_data = json.loads(Path(args.groups_file).read_text())
    models = groups_data["groups"][args.group]
    logger.info("Group %d: %d models — %s", args.group, len(models), [m["model_id"] for m in models])

    # ── Import every target dataset's run script once ──────────────────────
    import scripts.run.embedding.neural.fedorenko2016 as fedorenko_mod
    import scripts.run.embedding.behavioral.naturalstories as ns_mod
    import scripts.run.embedding.behavioral.ucl_reading as ucl_mod
    import scripts.run.embedding.behavioral.zuco_reading as zuco_mod
    import scripts.run.embedding.neural.nieuwland as nieuwland_mod
    import scripts.run.embedding.neural.podcast as podcast_mod

    all_modules = [fedorenko_mod, ns_mod, ucl_mod, zuco_mod, nieuwland_mod, podcast_mod]

    from neuroai.features import llm as llm_mod
    real_clear_hf_model_cache = llm_mod.clear_hf_model_cache

    def _suppress_cache_clear() -> None:
        for m in all_modules:
            if hasattr(m, "clear_hf_model_cache"):
                m.clear_hf_model_cache = lambda: None

    def _restore_and_clear() -> None:
        for m in all_modules:
            if hasattr(m, "clear_hf_model_cache"):
                m.clear_hf_model_cache = real_clear_hf_model_cache
        real_clear_hf_model_cache()

    device = args.device

    def _base_argv(config: str, family: str, idx: int) -> list[str]:
        return [
            "--config", config,
            "--from-registry", "--family", family, "--array-index", str(idx),
            "--device", device,
        ]

    for entry in models:
        family = entry["family"]
        idx = entry["array_index"]
        model_id = entry["model_id"]
        missing_datasets = set(entry["missing_datasets"])
        logger.info("=== Model %s (family=%s idx=%d) missing=%s ===", model_id, family, idx, sorted(missing_datasets))

        _suppress_cache_clear()

        if "fedorenko2016" in missing_datasets:
            _run_main(fedorenko_mod, [
                *_base_argv("configs/embedding/neural/fedorenko2016.yaml", family, idx),
                "--activations_only",
            ], f"fedorenko2016 [{model_id}]")

        if "naturalstories" in missing_datasets:
            _run_main(ns_mod, [
                "--config", "configs/embedding/behavioral/naturalstories.yaml",
                "--from-registry", "--family", family, "--array-index", str(idx),
                "--activations_only",
            ], f"naturalstories [{model_id}]")

        if "ucl_reading" in missing_datasets:
            _run_main(ucl_mod, [
                *_base_argv("configs/embedding/behavioral/ucl_reading.yaml", family, idx),
                "--activations_only",
            ], f"ucl_reading [{model_id}]")

        if "zuco_reading_nr" in missing_datasets:
            _run_main(zuco_mod, [
                *_base_argv("configs/embedding/behavioral/zuco_reading_nr.yaml", family, idx),
                "--extract_only",
            ], f"zuco_reading_nr [{model_id}]")

        if "nieuwland" in missing_datasets:
            _run_main(nieuwland_mod, [
                *_base_argv("configs/embedding/neural/nieuwland.yaml", family, idx),
                "--activations_only",
            ], f"nieuwland [{model_id}]")

        if "podcast" in missing_datasets:
            _run_main(podcast_mod, [
                *_base_argv("configs/embedding/neural/podcast.yaml", family, idx),
                "--activations_only",
            ], f"podcast [{model_id}]")

        _restore_and_clear()
        logger.info("=== Done with %s ===", model_id)

    logger.info("Group %d complete.", args.group)


if __name__ == "__main__":
    main()
