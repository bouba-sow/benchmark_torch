#!/usr/bin/env python3
"""RSA gap-fill worker for LPP fMRI + Natural Stories fMRI raw embedding
caches. THROWAWAY TOOL -- delete alongside the other scripts/tools/rsa_*
backfill trios once both datasets have full 118-model cache coverage.

ns_fmri already ships a proper `--activations_only` two-stage flag (extract +
cache only, no regression) -- reused as-is.

lpp_fmri (2026-09-28): originally reused the *real* per-checkpoint ridge-fit
pipeline unmodified via a throwaway --out_dir, on the theory that the cache
write was just a side effect of a normal fit. That was wrong on two counts,
both now fixed properly instead of worked around:
  1. `--from-registry --array-index` mode already resolves to exactly ONE
     checkpoint (see scripts/run/embedding/neural/lpp.py's `_array_checkpoint`
     handling) -- the earlier "full checkpoint-sweep" symptom (40+ min per
     large model, e.g. OLMo2-7B) was the ridge/LOSO-fit machinery itself
     being slow, not a multi-checkpoint expansion bug.
  2. `scripts/run/embedding/neural/lpp.py` now has a real `--activations_only`
     flag (mirrors the other 5 datasets' pattern): `LppFmriEncoder.
     extract_and_cache_only()` extracts + caches every section's features for
     one subject, skipping BOLD/mask loading and ridge/LOSO fitting entirely.
     Validated: ~45s-2.5min per model on one H100 GPU (vs 40+ min before).
  3. Along the way, found and fixed a real pre-existing bug unrelated to this
     backfill: `LppFmriEncoder._extract_features`'s `lbg_fulltext` branch
     never passed `cache_dir` to `contextual_word_embeddings_hf_lbg` at all
     (the sibling non-lbg branch did) -- and a second bug in
     `neuroai.lpp_llm_features._save_cached_features`'s atomic-write tmp
     filename ended in ".tmp" instead of ".npz", so `np.savez_compressed`
     silently re-appended ".npz" and the final `tmp.replace(cache_path)`
     always raised `FileNotFoundError`. Together these meant **no model using
     lbg_fulltext extraction ever cached successfully**, which is most of the
     registry -- likely the dominant root cause of LPP's near-empty cache,
     not just this backfill tool's earlier approach.

configs/embedding/neural/lpp_cache_backfill.yaml (subjects: [sub-EN057]
instead of null) is still used, purely to avoid _discover_subjects()
enumerating all ~50 EN subjects before activations_only trims to one -- not
load-bearing for correctness now.

The throwaway output directory's fit result (if any -- activations_only mode
never calls the fit at all) is discarded; only the cache side effect matters.
Safe to delete outputs_jeanzay/lpp/_cache_backfill_throwaway/ once the
backfill is done.

Usage:
    python scripts/tools/rsa_lpp_nsfmri_run.py --group 0 \\
        --groups-file scripts/tools/rsa_lpp_nsfmri_groups.json
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
logger = logging.getLogger("rsa_lpp_nsfmri_backfill")

LPP_THROWAWAY_OUT_DIR = "outputs_jeanzay/lpp/_cache_backfill_throwaway"


def _run_main(module, argv: list[str], label: str) -> bool:
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
    p.add_argument("--groups-file", default="scripts/tools/rsa_lpp_nsfmri_groups.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    groups_data = json.loads(Path(args.groups_file).read_text())
    models = groups_data["groups"][args.group]
    logger.info("Group %d: %d models — %s", args.group, len(models), [m["model_id"] for m in models])

    import scripts.run.embedding.neural.lpp as lpp_mod
    import scripts.run.embedding.neural.ns_fmri as ns_fmri_mod

    device = args.device

    for entry in models:
        family = entry["family"]
        idx = entry["array_index"]
        model_id = entry["model_id"]
        logger.info("=== Model %s (family=%s idx=%d) ===", model_id, family, idx)

        if entry.get("need_lpp"):
            _run_main(lpp_mod, [
                "--config", "configs/embedding/neural/lpp_cache_backfill.yaml",
                "--from-registry", "--family", family, "--array-index", str(idx),
                "--device", device,
                "--out_dir", LPP_THROWAWAY_OUT_DIR,
                "--skip_summary", "--no_plots",
                "--activations_only",
            ], f"lpp embed (cache backfill, activations_only) [{model_id}]")

        if entry.get("need_ns_fmri"):
            _run_main(ns_fmri_mod, [
                "--config", "configs/embedding/neural/ns_fmri.yaml",
                "--from-registry", "--family", family, "--array-index", str(idx),
                "--device", device,
                "--activations_only",
            ], f"ns_fmri activations_only [{model_id}]")

        logger.info("=== Done with %s ===", model_id)

    logger.info("Group %d complete.", args.group)


if __name__ == "__main__":
    main()
