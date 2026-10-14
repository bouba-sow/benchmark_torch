#!/usr/bin/env python3
"""Build eight balanced model groups from the latest extraction-only gap list."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT))
from neuroai.models.registry import flatten_family_tasks, resolve_family_task


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="outputs_jeanzay/logs/surprisal_extract_gapfill_20260928.json")
    p.add_argument("--out", default="scripts/tools/rsa_surprisal_gapfill_groups.json")
    p.add_argument("--n-groups", type=int, default=8)
    p.add_argument("--opt66b-out", default="scripts/tools/rsa_surprisal_gapfill_opt66b.json")
    a = p.parse_args()
    raw = json.loads(Path(a.manifest).read_text())
    by_model = {}
    pythia_main = {t.model_id: t for t in flatten_family_tasks("pythia")}
    for job in raw["jobs"]:
        ds, family = job["dataset"], job["family"]
        for idx in job["indices"]:
            task = resolve_family_task(family, idx)
            selected_family = family
            if "pythia" in family.lower():
                task = pythia_main[task.model_id]
                selected_family = "pythia"
            key = (selected_family, task.array_index, task.model_id, task.revision)
            item = by_model.setdefault(key, {
                "family": selected_family, "array_index": task.array_index, "model_id": task.model_id,
                "revision": task.revision, "datasets": [],
            })
            if ds not in item["datasets"]:
                item["datasets"].append(ds)

    # Greedy balancing by dataset-model extraction count.
    groups = [[] for _ in range(a.n_groups)]
    loads = [0] * a.n_groups
    opt66b = [x for x in by_model.values() if x["model_id"] == "facebook/opt-66b"]
    regular = [x for x in by_model.values() if x["model_id"] != "facebook/opt-66b"]
    for item in sorted(regular, key=lambda x: -len(x["datasets"])):
        g = min(range(a.n_groups), key=loads.__getitem__)
        groups[g].append(item)
        loads[g] += len(item["datasets"])
    out = {"n_groups": a.n_groups, "groups": groups, "dataset_model_pairs": sum(loads), "loads": loads}
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    opt_load = sum(len(x["datasets"]) for x in opt66b)
    Path(a.opt66b_out).write_text(json.dumps({"n_groups": 1, "groups": [opt66b], "dataset_model_pairs": opt_load, "loads": [opt_load]}, indent=2) + "\n")
    print(f"Wrote {a.n_groups} single-GPU groups: {len(regular)} models, {sum(loads)} dataset-model pairs; loads={loads}; OPT-66B separately: {opt_load} pairs")


if __name__ == "__main__":
    main()
