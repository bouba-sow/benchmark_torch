#!/usr/bin/env python3
"""One-shot: partition the full model registry into N groups for the RSA
gap-fill (see rsa_gap_fill_run.py). THROWAWAY TOOL -- delete this whole
scripts/tools/rsa_gap_fill_* trio once podcast/broderick/nieuwland/
naturalstories_spr/ucl_reading/zuco_reading(NR) all have embedding+surprisal
caches for every registry model.

Each model appears in exactly one group. The group file is written once and
then read by every array task / resubmission, so a timed-out group always
resumes with the same model list (the per-dataset run_model_sweep /
already_computed / already_done checks inside each real script make
re-running an already-finished (model, dataset) cell a fast no-op).

Usage:
    python scripts/tools/rsa_gap_fill_groups.py --n-groups 5 \\
        --out scripts/tools/rsa_gap_fill_groups.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.models.registry import LPP_FAMILIES, registry_model_configs

# Checkpoint-sweep pseudo-families aren't standalone models -- exclude them,
# same convention as scripts/slurm/stages.yaml's `excluded_families`.
_EXCLUDED_FAMILIES = {"pythia_checkpoints", "babylm_gpt2_checkpoints"}

# opt-66b needs 2 GPUs (bf16 doesn't fit one 80GB A100: 66B * 2 bytes ~=
# 132GB) -- these jobs only request gres:gpu:1. opt-30b fits VRAM-wise but is
# heavy/slow enough on a single A100 to risk starving the rest of its group's
# smaller models of wall-time; both excluded for now (2026-09-24).
_EXCLUDED_MODEL_IDS = {"facebook/opt-66b", "facebook/opt-30b"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-groups", type=int, default=5)
    p.add_argument("--out", default="scripts/tools/rsa_gap_fill_groups.json")
    args = p.parse_args()

    families = [f for f in LPP_FAMILIES if f not in _EXCLUDED_FAMILIES]

    flat: list[dict] = []
    for family in families:
        cfgs = registry_model_configs(family)
        for idx, cfg in enumerate(cfgs):
            if cfg["model_id"] in _EXCLUDED_MODEL_IDS:
                continue
            flat.append({
                "family": family,
                "array_index": idx,
                "model_id": cfg["model_id"],
            })

    groups: list[list[dict]] = [[] for _ in range(args.n_groups)]
    for i, entry in enumerate(flat):
        groups[i % args.n_groups].append(entry)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n_groups": args.n_groups, "groups": groups}, indent=2))

    print(f"{len(flat)} models across {len(families)} families -> {args.n_groups} groups")
    for gi, g in enumerate(groups):
        print(f"  group {gi}: {len(g)} models — {[e['model_id'] for e in g]}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
