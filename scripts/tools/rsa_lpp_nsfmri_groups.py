#!/usr/bin/env python3
"""One-shot: build the group file for the LPP fMRI / Natural Stories fMRI raw
embedding cache backfill. THROWAWAY TOOL -- delete alongside the other
scripts/tools/rsa_*backfill* trio once both datasets have full 118-model
cache coverage.

lpp_fmri cache was found completely empty (0/118) -- every model needs a
backfill run. ns_fmri cache had 18/118 (the small set of architecturally
diverse convergence-test models) -- only the other 100 need it.

Usage:
    python scripts/tools/rsa_lpp_nsfmri_groups.py --n-groups 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-groups", type=int, default=10)
    p.add_argument(
        "--missing-file",
        default="scripts/tools/rsa_ns_fmri_lpp_missing.json",
        help="Output of the missing-model computation (ns_fmri_missing / lpp_missing lists).",
    )
    p.add_argument("--out", default="scripts/tools/rsa_lpp_nsfmri_groups.json")
    args = p.parse_args()

    data = json.loads(Path(args.missing_file).read_text())
    lpp_missing = {e["model_id"]: e for e in data["lpp_missing"]}
    ns_missing_ids = {e["model_id"] for e in data["ns_fmri_missing"]}

    # Union of work: every model needing either dataset gets one entry,
    # flagged with which dataset(s) it still needs. opt-66b needs 2 GPUs
    # (bf16 doesn't fit one 80GB A100) -- excluded here, own dedicated job
    # instead (see rsa_lpp_nsfmri_opt66b_groups.json).
    flat: list[dict] = []
    for model_id, e in lpp_missing.items():
        if model_id == "facebook/opt-66b":
            continue
        flat.append({
            "family": e["family"],
            "array_index": e["array_index"],
            "model_id": model_id,
            "need_lpp": True,
            "need_ns_fmri": model_id in ns_missing_ids,
        })

    groups: list[list[dict]] = [[] for _ in range(args.n_groups)]
    for i, entry in enumerate(flat):
        groups[i % args.n_groups].append(entry)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"n_groups": args.n_groups, "groups": groups}, indent=2))

    print(f"{len(flat)} models need backfill -> {args.n_groups} groups")
    for gi, g in enumerate(groups):
        print(f"  group {gi}: {len(g)} models")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
