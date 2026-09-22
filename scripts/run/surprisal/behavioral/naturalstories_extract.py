#!/usr/bin/env python3
"""Precompute and cache Natural Stories per-layer logit-lens surprisal (GPU stage).

Mirrors the podcast prefetch pattern: one GPU job per (model, checkpoint)
pair populates the logit-lens trial-surprisal cache consumed by
NaturalStoriesLmeBenchmark._word_surprisal_all_layers(). Re-running is safe:
``logit_lens_surprisal_trials`` checks its own npz cache before loading the
model, so already-cached (model, revision) pairs are a fast no-op.

The cache format/path is owned by
``neuroai.features.logit_lens._trial_surprisal_cache_path`` (cache_key=
"natural_stories"), not a bespoke CSV — see
``neuroai.benchmarks.natural_stories_lme.NaturalStoriesLmeBenchmark._word_surprisal_all_layers``.

Indexing
--------
--array-size           Print total task count across all families and exit.
--global-index N       Run task N (flat index over all families in registry order).
--family F             Restrict to one family (must pair with --array-index I).
--array-index I        Run task I within family F.

Examples
--------
python scripts/run/surprisal/behavioral/naturalstories_extract.py --array-size
python scripts/run/surprisal/behavioral/naturalstories_extract.py --global-index 0 --device cuda
python scripts/run/surprisal/behavioral/naturalstories_extract.py \\
    --family pythia --array-index 3 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.datasets.natural_stories import NaturalStoriesDataset
from neuroai.features.llm import clear_hf_model_cache
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks, registry_model_configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _all_tasks():
    """Flat list of all FamilyTask objects across every family, in registry order."""
    tasks = []
    for family in LPP_FAMILY_NAMES:
        tasks.extend(flatten_family_tasks(family))
    return tasks


def _total_tasks() -> int:
    return sum(len(flatten_family_tasks(f)) for f in LPP_FAMILY_NAMES)


def _resolve_task(args):
    """Return a FamilyTask from either --global-index or --family/--array-index."""
    if args.global_index is not None:
        tasks = _all_tasks()
        idx = args.global_index
        if idx < 0 or idx >= len(tasks):
            logger.error("--global-index %d out of range [0, %d)", idx, len(tasks))
            sys.exit(1)
        return tasks[idx]

    tasks = flatten_family_tasks(args.family)
    idx = args.array_index
    if idx < 0 or idx >= len(tasks):
        logger.error(
            "--array-index %d out of range [0, %d) for family %r",
            idx,
            len(tasks),
            args.family,
        )
        sys.exit(1)
    return tasks[idx]


def _tokenizer_no_revision(model_id: str, revision: str | None) -> bool:
    """Pythia checkpoints share one tokenizer across all steps — always use 'main'."""
    return bool(revision and "pythia" in model_id.lower())


def _trust_remote_code_for_task(task) -> bool:
    for m in registry_model_configs(task.registry_family):
        if m["model_id"] == task.model_id:
            return bool(m.get("trust_remote_code", False))
    return False


def _n_params_for_task(task) -> float | None:
    """Look up the registry's ``n_params`` for this task's model.

    Without this, ``_get_cached_causal_lm``'s ``_apply_device_map_if_needed``
    defaults the size estimate to 0.0 GiB and wrongly concludes any model
    "fits on one GPU" — pinning it to ``cuda:0`` alone and silently ignoring
    any other GPUs allocated to the job (large models then OOM instead of
    sharding across devices via ``device_map="auto"``).
    """
    for m in registry_model_configs(task.registry_family):
        if m["model_id"] == task.model_id:
            return m.get("n_params")
    return None


def prefetch_one(
    task,
    *,
    dataset_root: str,
    cache_dir: str,
    device: str,
    stories: list[int] | None,
    trust_remote_code: bool = False,
    n_params: float | None = None,
) -> None:
    """Compute and cache per-layer surprisal for one (model, checkpoint) task."""
    model_revision = task.revision if task.revision not in ("main", "") else None
    tok_no_rev = _tokenizer_no_revision(task.load_model_id, model_revision)

    logger.info(
        "Computing per-layer surprisal: %s @ %s  (family=%s step=%d tokens=%dB)",
        task.load_model_id,
        task.revision,
        task.registry_family,
        task.step,
        task.tokens_seen // 1_000_000_000,
    )

    # Load word list with cumulative context column (same as LME pipeline).
    # aggregate=True builds the 'context' column; min_correct=0 keeps all words
    # so the cache is a superset of any LME run's word set.
    ds = NaturalStoriesDataset(dataset_root, stories=stories, min_correct=0, aggregate=True)
    word_df = ds.load_words().reset_index(drop=True)

    contexts = word_df["context"].astype(str).tolist()
    targets = word_df["word"].astype(str).tolist()

    try:
        if task.registry_family == "cbr_rnn":
            from neuroai.features.cbr_rnn import batch_surprisal_cbr_rnn

            batch_surprisal_cbr_rnn(
                task.load_model_id,
                contexts,
                targets,
                device=device,
            )
        else:
            from neuroai.features.logit_lens import logit_lens_surprisal_trials

            surp_dtype = "float32" if "gpt-bert" in task.load_model_id.lower() else None

            arr = logit_lens_surprisal_trials(
                contexts,
                targets,
                task.load_model_id,
                revision=model_revision,
                device=device,
                tokenizer_no_revision=tok_no_rev,
                trust_remote_code=trust_remote_code,
                n_params=n_params,
                cache_dir=cache_dir,
                cache_key="natural_stories",
                dtype=surp_dtype,
            )
            logger.info("Saved per-layer surprisal cache: shape %s", arr.shape)
    except Exception:
        logger.error(
            "Surprisal computation failed: %s @ %s",
            task.load_model_id,
            task.revision,
            exc_info=True,
        )
        raise
    finally:
        if task.registry_family != "cbr_rnn":
            clear_hf_model_cache()


def _load_config(config_path: str) -> dict:
    import yaml

    repo_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    os.environ.setdefault("BENCHMARK_ROOT", str(repo_root))
    scratch = os.environ.get("SCRATCH", "").strip()
    os.environ.setdefault(
        "BENCHMARK_CACHE_ROOT",
        f"{scratch}/.cache" if scratch else f"{os.environ['BENCHMARK_ROOT']}/.cache",
    )
    os.environ.setdefault(
        "NS_FEATURE_CACHE_DIR",
        f"{os.environ['BENCHMARK_CACHE_ROOT']}/naturalstories_features",
    )
    if not Path(config_path).is_file():
        return {}
    with open(config_path, encoding="utf-8") as f:
        raw = os.path.expandvars(f.read())
    return yaml.safe_load(raw) or {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--array-size", action="store_true", help="Print total task count and exit")
    p.add_argument("--global-index", type=int, default=None, metavar="N")
    p.add_argument("--family", default=None, help="Registry family key (e.g. pythia)")
    p.add_argument("--array-index", type=int, default=None, metavar="I")
    p.add_argument("--config", default="configs/surprisal/behavioral/naturalstories.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset-root", default=None, help="Override dataset root from config/env")
    p.add_argument("--cache-dir", default=None, help="Override NS_FEATURE_CACHE_DIR")
    args = p.parse_args()

    if args.array_size:
        n = _total_tasks()
        print(n)
        return

    if args.global_index is None and (args.family is None or args.array_index is None):
        p.error("Specify --global-index N, or --family F --array-index I")

    task = _resolve_task(args)
    print(
        f"Task: family={task.registry_family}  model={task.model_id}"
        f"  load_model_id={task.load_model_id}  revision={task.revision}"
        f"  step={task.step}  global_index={task.array_index}"
    )

    cfg = _load_config(args.config)

    dataset_root = args.dataset_root or os.path.expandvars(
        str(cfg.get("dataset", {}).get("root", "$BENCHMARK_ROOT/data/naturalstories"))
    )
    stories = cfg.get("dataset", {}).get("stories")

    ns_dir = os.environ.get("NS_FEATURE_CACHE_DIR", "").strip()
    cache_dir = args.cache_dir or ns_dir or str(
        cfg.get("cache", {}).get("features_dir", ".cache/naturalstories_features")
    )
    cache_dir = os.path.expandvars(cache_dir)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    logger.info("dataset_root : %s", dataset_root)
    logger.info("cache_dir    : %s", cache_dir)
    logger.info("device       : %s", args.device)

    trust_remote_code = _trust_remote_code_for_task(task)
    n_params = _n_params_for_task(task)
    prefetch_one(
        task,
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        device=args.device,
        stories=stories,
        trust_remote_code=trust_remote_code,
        n_params=n_params,
    )


if __name__ == "__main__":
    main()
