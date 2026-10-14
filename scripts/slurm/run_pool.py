#!/usr/bin/env python3
"""Run several (family, array-index) tasks of one submit.py stage back to back in ONE job.

Uses the same stage table and job-script rendering as ``scripts/submit.py``, so each task
behaves exactly like the corresponding SLURM array task; a failing task does not stop the pool.

    python scripts/slurm/run_pool.py nieuwland.surprisal.extract --env K=V gpt2:0 gpt2:1 opt:3
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
spec = importlib.util.spec_from_file_location("submit", REPO_ROOT / "scripts" / "submit.py")
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", help="stage key, e.g. nieuwland.surprisal.extract")
    p.add_argument("tasks", nargs="+", help="family:index")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VAL")
    args = p.parse_args()

    cfg = submit.load_config()
    stage = cfg["stages"][args.stage]
    env = {k: str(v) for k, v in stage.get("env", {}).items()}
    env.update(dict(item.split("=", 1) for item in args.env))
    failed = []
    for task in args.tasks:
        family, idx = task.rsplit(":", 1)
        script = submit.render_script(cfg, stage, family=family, env=env, summary=False)
        print(f"=== {task} start {time.strftime('%H:%M:%S')}", flush=True)
        rc = subprocess.run(["bash", "-c", script],
                            env={**os.environ, "SLURM_ARRAY_TASK_ID": idx}).returncode
        print(f"=== {task} exit={rc} {time.strftime('%H:%M:%S')}", flush=True)
        if rc:
            failed.append(task)
    print(f"pool done: {len(args.tasks) - len(failed)}/{len(args.tasks)} ok; failed={failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
