#!/usr/bin/env python3
"""Compute per-dataset missing raw-embedding-cache model lists and partition
the union of missing (model, dataset) work into balanced SLURM array groups.

THROWAWAY TOOL for the RSA raw-embedding-cache backfill (2026-09-28) --
unrelated to and does NOT touch the existing scripts/tools/rsa_gap_fill_*
trio (that's a separate, already-mostly-done effort covering a different set
of datasets/tracks).

Datasets covered here (embedding-track RAW CACHE only, no ridge/LME fits --
those are already complete and correct everywhere):
  - fedorenko2016   (FEDORENKO_FEATURE_CACHE_DIR, dir-per-model/dir-per-revision)
  - naturalstories  ($BENCHMARK_ROOT/.cache/naturalstories_features -- NOTE:
                     configs/embedding/behavioral/naturalstories.yaml hardcodes
                     this path, ignoring NS_FEATURE_CACHE_DIR which is actually
                     the surprisal-track's cache dir)
  - ucl_reading     (UCL_READING_EMBED_CACHE_DIR, defaults to
                     $BENCHMARK_CACHE_ROOT/ucl_reading_features -- shared with
                     the surprisal track, NOT ucl_reading_embed_features which
                     is a stale/legacy dir)
  - zuco_reading_nr (ZUCO_READING_EMBED_CACHE_DIR, defaults to
                     $BENCHMARK_CACHE_ROOT/zuco_reading_features/nr)
  - nieuwland       (NIEUWLAND_FEATURE_CACHE_DIR)
  - podcast         (PODCAST_FEATURE_CACHE_DIR)
  - pereira         (PEREIRA_EMBED_FEATURE_CACHE_DIR) -- handled separately,
                     only 2 models missing (cclark/cbr-rnn, cclark/cbr-rnn-m),
                     no activations_only flag needed (pereira.py has no
                     skip-check so a normal run is cheap and safe).

Target model list: the 118-model registry snapshot from
scripts/tools/rsa_gap_fill_groups.json (116 entries) PLUS facebook/opt-30b
(family=opt, array_index=6) and facebook/opt-66b (array_index=7), which that
file excludes but which ARE in scope for this backfill.

File-naming conventions differ per dataset and were verified by reading each
extraction path's cache-key code (neuroai/features/llm.py, neuroai/benchmarks/
eeg_erp.py, neuroai/benchmarks/fedorenko2016.py) and by inspecting actual
files on disk -- see the per-dataset comments below.

Usage:
    python scripts/tools/rsa_cache_backfill_groups.py --n-groups 14 \\
        --out scripts/tools/rsa_cache_backfill_groups.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))

from neuroai.models.registry import resolve_family_task  # noqa: E402
from neuroai.sweep import safe_model_dir  # noqa: E402

ACTIVATIONS_ONLY_DATASETS = [
    "fedorenko2016", "naturalstories", "ucl_reading", "zuco_reading_nr", "nieuwland", "podcast",
]


def _cache_root() -> Path:
    import os
    scratch = os.environ.get("SCRATCH", "").strip()
    root = os.environ.get("BENCHMARK_CACHE_ROOT", "").strip()
    if root:
        return Path(root)
    if scratch:
        return Path(scratch) / ".cache"
    raise SystemExit("Set SCRATCH or BENCHMARK_CACHE_ROOT before running this script.")


def _rev_safe_dash(rev: str | None) -> str:
    """neuroai.features.llm's safe_rev: "/" and "-" both -> "_"."""
    return (rev or "main").replace("/", "_").replace("-", "_")


def build_target_list() -> list[dict]:
    groups = json.loads((REPO_ROOT / "scripts/tools/rsa_gap_fill_groups.json").read_text())["groups"]
    targets = [m for g in groups for m in g]
    targets.append({"family": "opt", "array_index": 6, "model_id": "facebook/opt-30b"})
    targets.append({"family": "opt", "array_index": 7, "model_id": "facebook/opt-66b"})
    assert len(targets) == 118, len(targets)

    resolved = []
    for t in targets:
        task = resolve_family_task(t["family"], t["array_index"])
        resolved.append({
            "family": t["family"],
            "array_index": t["array_index"],
            "model_id": task.model_id,
            "revision": task.revision,
            "safe": safe_model_dir(task.model_id),
        })
    return resolved


def _prefix_set(cache_dir: Path) -> set[str]:
    """Files named <prefix>__en__secNN__...npz -- collect the <prefix> part."""
    prefixes = set()
    if cache_dir.is_dir():
        for f in cache_dir.iterdir():
            if f.name.startswith("logitlens_trials_"):
                continue
            m = re.match(r"^(.*?)__en__sec", f.name)
            if m:
                prefixes.add(m.group(1))
    return prefixes


def compute_missing(resolved: list[dict], cache_root: Path) -> dict[str, list[dict]]:
    missing: dict[str, list[dict]] = {}

    # fedorenko2016: <safe>/<raw_revision>/*.npz (revision NOT sanitized)
    fed_dir = cache_root / "fedorenko2016_features"
    fed_have = set()
    if fed_dir.is_dir():
        for model_dir in fed_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for rev_dir in model_dir.iterdir():
                if rev_dir.is_dir() and any(rev_dir.glob("*.npz")):
                    fed_have.add((model_dir.name, rev_dir.name))
    missing["fedorenko2016"] = [
        r for r in resolved if (r["safe"], r["revision"] or "main") not in fed_have
    ]

    # naturalstories: $BENCHMARK_ROOT/.cache/naturalstories_features (see docstring)
    ns_dir = REPO_ROOT / ".cache" / "naturalstories_features"
    ns_prefixes = _prefix_set(ns_dir)
    missing["naturalstories"] = [
        r for r in resolved if f"{r['safe']}__{_rev_safe_dash(r['revision'])}" not in ns_prefixes
    ]

    # ucl_reading: $BENCHMARK_CACHE_ROOT/ucl_reading_features (shared w/ surprisal)
    ucl_prefixes = _prefix_set(cache_root / "ucl_reading_features")
    missing["ucl_reading"] = [
        r for r in resolved if f"{r['safe']}__{_rev_safe_dash(r['revision'])}" not in ucl_prefixes
    ]

    # zuco_reading_nr: $BENCHMARK_CACHE_ROOT/zuco_reading_features/nr
    zuco_prefixes = _prefix_set(cache_root / "zuco_reading_features" / "nr")
    missing["zuco_reading_nr"] = [
        r for r in resolved if f"{r['safe']}__{_rev_safe_dash(r['revision'])}" not in zuco_prefixes
    ]

    # nieuwland: <safe>_<raw_revision>_<mode>_<hash>.npz (revision NOT sanitized)
    niew_dir = cache_root / "nieuwland_features"
    niew_files = [f.name for f in niew_dir.iterdir()] if niew_dir.is_dir() else []
    def niew_present(r):
        prefix = f"{r['safe']}_{r['revision'] or 'main'}_"
        return any(fn.startswith(prefix) for fn in niew_files)
    missing["nieuwland"] = [r for r in resolved if not niew_present(r)]

    # podcast: $BENCHMARK_CACHE_ROOT/podcast_features
    pod_prefixes = _prefix_set(cache_root / "podcast_features")
    missing["podcast"] = [
        r for r in resolved if f"{r['safe']}__{_rev_safe_dash(r['revision'])}" not in pod_prefixes
    ]

    # pereira: $BENCHMARK_CACHE_ROOT/pereira_embed_features (handled separately)
    per_prefixes = _prefix_set(cache_root / "pereira_embed_features")
    missing["pereira"] = [
        r for r in resolved if f"{r['safe']}__{_rev_safe_dash(r['revision'])}" not in per_prefixes
    ]

    return missing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-groups", type=int, default=14)
    p.add_argument("--out", default="scripts/tools/rsa_cache_backfill_groups.json")
    p.add_argument("--pereira-out", default="scripts/tools/rsa_cache_backfill_pereira.json")
    args = p.parse_args()

    resolved = build_target_list()
    cache_root = _cache_root()
    missing = compute_missing(resolved, cache_root)

    for name, miss in missing.items():
        cached = len(resolved) - len(miss)
        print(f"{name}: {cached}/118 cached, {len(miss)} missing")

    # Union of (model, [datasets missing]) across the 6 activations_only datasets.
    by_model: dict[tuple, dict] = {}
    for ds in ACTIVATIONS_ONLY_DATASETS:
        for r in missing[ds]:
            key = (r["family"], r["array_index"], r["model_id"])
            entry = by_model.setdefault(key, {**r, "missing_datasets": []})
            entry["missing_datasets"].append(ds)

    # opt-66b needs 2 GPUs (bf16 doesn't fit one 80GB A100) -- excluded from the
    # regular single-GPU pools and given its own dedicated job instead (see
    # rsa_cache_backfill_opt66b_groups.json).
    work_items = [e for e in by_model.values() if e["model_id"] != "facebook/opt-66b"]
    work_items.sort(key=lambda e: -len(e["missing_datasets"]))
    print(f"\n{len(work_items)} unique models need >=1 dataset backfilled "
          f"({sum(len(e['missing_datasets']) for e in work_items)} total work items, "
          f"excluding facebook/opt-66b)")

    # Greedy longest-processing-time-first bin packing, balancing total
    # per-model work-item count (a rough proxy for extraction time) across N groups.
    n = max(1, args.n_groups)
    groups: list[list[dict]] = [[] for _ in range(n)]
    loads = [0] * n
    for item in work_items:
        i = loads.index(min(loads))
        groups[i].append(item)
        loads[i] += len(item["missing_datasets"])

    groups = [g for g in groups if g]  # drop empty groups if n > n models
    out = {"n_groups": len(groups), "groups": groups}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(groups)} groups to {args.out} (loads: {sorted(loads, reverse=True)})")

    pereira_out = {"models": missing["pereira"]}
    Path(args.pereira_out).write_text(json.dumps(pereira_out, indent=2))
    print(f"Wrote {len(missing['pereira'])} Pereira-only missing models to {args.pereira_out}")


if __name__ == "__main__":
    main()
