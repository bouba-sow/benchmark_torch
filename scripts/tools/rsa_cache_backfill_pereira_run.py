#!/usr/bin/env python3
"""RSA raw-embedding-cache backfill for Pereira -- the 2 models missing
(cclark/cbr-rnn, cclark/cbr-rnn-m; see rsa_cache_backfill_pereira.json).

pereira.py's run script has no "already done" skip-check at all, so a plain
run just redoes its (cheap, 1-layer RNN) fit too -- that's fine, no
--activations_only flag exists or is needed here. THROWAWAY TOOL, part of
the same one-off backfill as rsa_cache_backfill_run.py.

Usage:
    python scripts/tools/rsa_cache_backfill_pereira_run.py \\
        --models-file scripts/tools/rsa_cache_backfill_pereira.json
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
logger = logging.getLogger("rsa_cache_backfill_pereira")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-file", default="scripts/tools/rsa_cache_backfill_pereira.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    data = json.loads(Path(args.models_file).read_text())
    models = data["models"]
    logger.info("Pereira backfill: %d models — %s", len(models), [m["model_id"] for m in models])

    import scripts.run.embedding.neural.pereira as pereira_mod

    for entry in models:
        family, idx, model_id = entry["family"], entry["array_index"], entry["model_id"]
        logger.info("=== Pereira: %s (family=%s idx=%d) ===", model_id, family, idx)
        old_argv = sys.argv
        sys.argv = [pereira_mod.__file__, "--config", "configs/embedding/neural/pereira.yaml",
                    "--from-registry", "--family", family, "--array-index", str(idx), "--device", args.device]
        try:
            pereira_mod.main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                logger.error("Pereira [%s] FAILED (SystemExit %s)", model_id, exc.code)
        except Exception:
            logger.error("Pereira [%s] FAILED", model_id, exc_info=True)
            traceback.print_exc()
        finally:
            sys.argv = old_argv

    logger.info("Pereira backfill complete.")


if __name__ == "__main__":
    main()
