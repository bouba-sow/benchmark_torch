#!/usr/bin/env python3
"""SLURM helpers for LPP per-family checkpoint sweeps.

Examples::

    # Array size for #SBATCH --array=0-N
    python scripts/slurm/family_tasks.py array-size --family pythia

    # Inspect task for array index 3
    python scripts/slurm/family_tasks.py resolve --family gpt2 --index 3

    # List all families and task counts
    python scripts/slurm/family_tasks.py list-families
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.models.registry import (
    LPP_FAMILY_NAMES,
    family_array_size,
    flatten_family_tasks,
    resolve_family_task,
    resolve_load_model_id,
)


def _cmd_list_families(_: argparse.Namespace) -> None:
    for fam in LPP_FAMILY_NAMES:
        n = family_array_size(fam)
        print(f"{fam}\t{n}")


def _cmd_array_size(args: argparse.Namespace) -> None:
    print(family_array_size(args.family, untrained=args.untrained))


def _cmd_resolve(args: argparse.Namespace) -> None:
    task = resolve_family_task(args.family, args.index, untrained=args.untrained)
    print(f"array_index={task.array_index}")
    print(f"family={task.registry_family}")
    print(f"model_id={task.model_id}")
    print(f"load_model_id={task.load_model_id}")
    print(f"label={task.label}")
    print(f"revision={task.revision}")
    print(f"step={task.step}")
    print(f"tokens_seen={task.tokens_seen}")


def _cmd_plan(args: argparse.Namespace) -> None:
    for task in flatten_family_tasks(args.family):
        print(
            f"{task.array_index}\t{task.model_id}\t{task.revision}\t"
            f"step={task.step}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list-families", help="Print family name and array size").set_defaults(
        func=_cmd_list_families
    )

    sz = sub.add_parser("array-size", help="Print N for --array=0-(N-1)")
    sz.add_argument("--family", required=True, choices=LPP_FAMILY_NAMES)
    sz.add_argument(
        "--untrained", action="store_true",
        help="One task per model instead of per model×checkpoint (random init "
             "doesn't depend on training-step revision).",
    )
    sz.set_defaults(func=_cmd_array_size)

    res = sub.add_parser("resolve", help="Describe one array task")
    res.add_argument("--family", required=True, choices=LPP_FAMILY_NAMES)
    res.add_argument("--index", type=int, required=True)
    res.add_argument("--untrained", action="store_true")
    res.set_defaults(func=_cmd_resolve)

    pl = sub.add_parser("plan", help="Print full task table for a family")
    pl.add_argument("--family", required=True, choices=LPP_FAMILY_NAMES)
    pl.set_defaults(func=_cmd_plan)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
