#!/usr/bin/env python3
"""Submit a benchmark stage to SLURM -- one command for every dataset x track x stage.

The benchmark is a 2x2 design: {neural, behavioral} datasets x {embedding, surprisal}
tracks. Models, checkpoints and array sizes come from the model registry
(``neuroai/models/registry.py``); everything that differs per dataset lives in
``scripts/slurm/stages.yaml`` (shared runner, entry function, resource profile,
default env). Nothing is hard-coded per model, so a family always covers every
model in the registry.

    python scripts/submit.py list
    python scripts/submit.py geco surprisal extract --family gpt2 pythia        # dry run
    python scripts/submit.py geco surprisal fit --family gpt2 --measure GP --submit
    python scripts/submit.py provo embedding --all-families --submit
    python scripts/submit.py lpp embedding --family opt --indices 0-3,7 --time 04:00:00 --submit
    python scripts/submit.py geco surprisal fit --family gpt2 --summary --submit
    python scripts/submit.py lpp surprisal refit --family gpt2 --submit      # LPP refit with saved word predictions

Dry run is the default: the exact ``sbatch`` command and job script are printed and
nothing is submitted until ``--submit`` is given.

Defaults: 2 h wall time, no QoS. Everything is overridable per call (--time, --account,
--partition, --qos, --gres, --cpus, --constraint, --mem, --max-concurrent, --env KEY=VAL)
or persistently in ``stages.yaml`` (``defaults``, ``profiles``, per-stage keys).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
STAGES_YAML = REPO_ROOT / "scripts" / "slurm" / "stages.yaml"


# --------------------------------------------------------------------------- config
def load_config() -> dict:
    return yaml.safe_load(STAGES_YAML.read_text())


def _stage_parts(key: str) -> tuple[str, str, str]:
    dataset, track, stage = key.split(".")
    return dataset, track, stage


def resolve_stage(cfg: dict, dataset: str, track: str, stage: str | None) -> tuple[str, dict]:
    stages = cfg["stages"]
    if stage is None:
        stage = "fit"
    key = f"{dataset}.{track}.{stage}"
    if key not in stages:
        known = sorted(k for k in stages if k.startswith(f"{dataset}."))
        hint = f" Available for {dataset!r}: {', '.join(known)}" if known else f" Unknown dataset {dataset!r}."
        raise SystemExit(f"No stage {key!r}.{hint}")
    return key, stages[key]


# --------------------------------------------------------------------------- registry
def registry_tasks(family: str, indices: list[int] | None) -> list[tuple[int, float]]:
    """[(array_index, n_params_of_model)] for a family, straight from the registry."""
    from neuroai.models.registry import LPP_FAMILIES, flatten_family_tasks

    if family not in LPP_FAMILIES:
        raise SystemExit(f"Unknown family {family!r}; registry has: {', '.join(sorted(LPP_FAMILIES))}")
    n_params = {spec.model_id: float(getattr(spec, "n_params", 0) or 0) for spec in LPP_FAMILIES[family]}
    tasks = [(t.array_index, n_params.get(t.model_id, 0.0)) for t in flatten_family_tasks(family)]
    if indices is not None:
        keep = set(indices)
        missing = keep - {i for i, _ in tasks}
        if missing:
            raise SystemExit(f"{family}: indices {sorted(missing)} out of range (0..{len(tasks) - 1})")
        tasks = [t for t in tasks if t[0] in keep]
    return tasks


def all_family_names(cfg: dict) -> list[str]:
    from neuroai.models.registry import LPP_FAMILIES

    excluded = set(cfg.get("defaults", {}).get("excluded_families", []))
    return [f for f in LPP_FAMILIES if f not in excluded]


def parse_indices(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def compress(indices: list[int]) -> str:
    """[0,1,2,5] -> '0-2,5' (SLURM --array syntax)."""
    idx = sorted(indices)
    parts, start = [], idx[0]
    for prev, cur in zip(idx, idx[1:] + [None]):
        if cur is None or cur != prev + 1:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = cur
    return ",".join(parts)


# --------------------------------------------------------------------------- resources
def resolve_resources(cfg: dict, stage: dict, args: argparse.Namespace, *, large: bool) -> dict:
    """defaults < profile < large-model class < per-stage keys < command line."""
    res = dict(cfg.get("defaults", {}))
    res.update(cfg["profiles"][stage["profile"]])
    if large and stage["profile"] == "gpu":
        res.update({k: v for k, v in cfg.get("large_model", {}).items() if k != "min_params"})
    for key in ("cpus", "gres", "account", "constraint", "partition", "qos", "time", "mem", "comment"):
        if key in stage:
            res[key] = stage[key]
    for key in ("cpus", "gres", "account", "constraint", "partition", "qos", "time", "mem", "comment"):
        val = getattr(args, key, None)
        if val is not None:
            res[key] = val
    return res


def sbatch_options(res: dict, *, name: str, log: str, array: str, args: argparse.Namespace) -> list[str]:
    opts = ["--parsable", "--export=ALL", f"--job-name={name}", f"--output={log}", f"--array={array}"]
    opts += [f"--account={res['account']}", f"--time={res['time']}", f"--cpus-per-task={res['cpus']}"]
    if res.get("hint"):
        opts.append(f"--hint={res['hint']}")
    for key, flag in (("gres", "--gres"), ("constraint", "--constraint"), ("partition", "--partition"),
                      ("qos", "--qos"), ("mem", "--mem"), ("comment", "--comment")):
        if res.get(key):
            opts.append(f"{flag}={res[key]}")
    if args.dependency:
        opts.append(f"--dependency={args.dependency}")
    return opts


# --------------------------------------------------------------------------- job script
def render_script(cfg: dict, stage: dict, *, family: str, env: dict[str, str], summary: bool) -> str:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f'export BENCHMARK_ROOT="${{BENCHMARK_ROOT:-{REPO_ROOT}}}"',
        'cd "$BENCHMARK_ROOT"',
    ]
    for module in stage.get("modules", []):
        lines.append(f"module load {module}")
    runtime_paths = cfg.get("family_runtime", {}).get(family, {}).get("pythonpath", [])
    if runtime_paths:
        runtime_paths = runtime_paths + stage.get("pythonpath_extra", [])
    for path in runtime_paths:
        lines.append(f'export PYTHONPATH="${{BENCHMARK_ROOT}}/{path}${{PYTHONPATH:+:$PYTHONPATH}}"')
    lines.append(f"export {stage['family_var']}={shlex.quote(family)}")
    for key, val in env.items():
        lines.append(f"export {key}={shlex.quote(str(val))}")
    if summary:
        lines.append("export SUMMARY_ONLY=1")
    lines.append(f'source "${{BENCHMARK_ROOT}}/{stage["runner"]}"')
    fn, summary_fn = stage.get("fn"), stage.get("summary_fn")
    if fn and summary_fn:
        lines += ['if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then', f"  {summary_fn}", "else", f"  {fn}", "fi"]
    elif fn:
        lines.append(fn)
    return "\n".join(lines) + "\n"


def stage_env(stage: dict, args: argparse.Namespace) -> dict[str, str]:
    env = {k: str(v) for k, v in stage.get("env", {}).items()}
    selectors = stage.get("selectors", {})
    for name in ("measure", "task"):
        value = getattr(args, name, None)
        if value is None:
            continue
        if name not in selectors:
            raise SystemExit(f"--{name} is not available for this stage")
        allowed = selectors[name]["values"]
        if value.upper() not in allowed:
            raise SystemExit(f"--{name} must be one of {allowed}")
        env[selectors[name]["env"]] = value.upper()
    for item in args.env or []:
        key, _, val = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--env expects KEY=VAL, got {item!r}")
        env[key] = val
    return env


# --------------------------------------------------------------------------- main
def plan_jobs(cfg: dict, key: str, stage: dict, args: argparse.Namespace) -> list[dict]:
    dataset, track, stg = _stage_parts(key)
    families = args.family or (all_family_names(cfg) if args.all_families else [])
    if not families:
        raise SystemExit("Pick models with --family F [F ...] or --all-families.")
    indices = parse_indices(args.indices) if args.indices else None
    env = stage_env(stage, args)
    min_params = float(cfg.get("large_model", {}).get("min_params", float("inf")))
    jobs = []
    for family in families:
        tasks = registry_tasks(family, [0] if args.summary else indices)
        classes = {False: [i for i, n in tasks if n < min_params], True: [i for i, n in tasks if n >= min_params]}
        for large, idx in classes.items():
            if not idx:
                continue
            res = resolve_resources(cfg, stage, args, large=large)
            array = compress(idx) + (f"%{args.max_concurrent}" if args.max_concurrent else "")
            tag = f"{dataset}_{track}_{stg}_{family}" + ("_large" if large else "")
            log = f"{res['output_dir']}/{tag}_%A_%a.out"
            jobs.append({
                "family": family, "large": large, "n_tasks": len(idx), "array": array, "res": res,
                "cmd": ["sbatch", *sbatch_options(res, name=tag, log=log, array=array, args=args)],
                "script": render_script(cfg, stage, family=family, env=env, summary=args.summary),
            })
    return jobs


def cmd_list(cfg: dict) -> None:
    print(f"{'stage':38s} {'profile':7s} runner")
    for key, st in cfg["stages"].items():
        print(f"{key:38s} {st['profile']:7s} {st['runner']}")
    print(f"\nexcluded from --all-families: {', '.join(cfg.get('defaults', {}).get('excluded_families', [])) or '-'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="dataset slug (e.g. geco, lpp, provo) or 'list'")
    p.add_argument("track", nargs="?", choices=["embedding", "surprisal"])
    p.add_argument("stage", nargs="?", choices=["fit", "extract", "refit"], help="default: fit (refit: LPP only)")
    sel = p.add_argument_group("what to run")
    sel.add_argument("--family", nargs="+", help="registry family names (gpt2, pythia, opt, ...)")
    sel.add_argument("--all-families", action="store_true", help="every registry family except defaults.excluded_families")
    sel.add_argument("--indices", help="only these array indices, e.g. 0-3,7 (default: every model/checkpoint)")
    sel.add_argument("--measure", help="FP | GP | SP (eye-tracking datasets)")
    sel.add_argument("--task", help="NR | SR | TSR (ZuCo Reading)")
    sel.add_argument("--summary", action="store_true", help="run the stage's summary step (array task 0 only)")
    sel.add_argument("--env", action="append", metavar="KEY=VAL", help="extra environment variable (repeatable)")
    res = p.add_argument_group("resources (default: stages.yaml)")
    res.add_argument("--time"); res.add_argument("--account"); res.add_argument("--partition"); res.add_argument("--qos")
    res.add_argument("--gres"); res.add_argument("--constraint"); res.add_argument("--mem")
    res.add_argument("--comment", help='Torch advanced options, e.g. \'gpu_mps=yes\' or \'preemption=yes;requeue=true\'')
    res.add_argument("--cpus", type=int)
    res.add_argument("--max-concurrent", type=int, help="cap simultaneous array tasks (the %%N in --array)")
    res.add_argument("--dependency", help="e.g. afterok:12345")
    p.add_argument("--submit", action="store_true", help="actually call sbatch (default: dry run)")
    p.add_argument("--print-script", action="store_true", help="in a dry run, also print each job script")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config()
    if args.dataset == "list":
        return cmd_list(cfg)
    if args.track is None:
        raise SystemExit("usage: submit.py DATASET {embedding,surprisal} [{fit,extract}] --family ...")
    key, stage = resolve_stage(cfg, args.dataset, args.track, args.stage)
    sys.path.insert(0, str(REPO_ROOT))
    jobs = plan_jobs(cfg, key, stage, args)
    for job in jobs:
        res = job["res"]
        gpu = f"gres={res['gres']}" if res.get("gres") else "cpu"
        print(f"# {key}  family={job['family']}{' (large models)' if job['large'] else ''}  "
              f"{job['n_tasks']} task(s) array={job['array']}  {gpu} cpus={res['cpus']} time={res['time']}"
              f"{' qos=' + res['qos'] if res.get('qos') else ''}")
        print(" ".join(shlex.quote(c) for c in job["cmd"]))
        if args.print_script or not args.submit:
            if args.print_script:
                print(job["script"])
        if args.submit:
            out = subprocess.run(job["cmd"], input=job["script"], text=True, capture_output=True)
            if out.returncode != 0:
                raise SystemExit(f"sbatch failed: {out.stderr.strip()}")
            print(f"  -> submitted job {out.stdout.strip()}")
    if not args.submit:
        print(f"\n(dry run: {len(jobs)} job(s) not submitted -- add --submit)")


if __name__ == "__main__":
    main()
