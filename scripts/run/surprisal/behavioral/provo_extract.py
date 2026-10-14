#!/usr/bin/env python3
"""Precompute and cache Provo Corpus per-layer logit-lens surprisal (GPU stage).

Mirrors ``run/surprisal/behavioral/naturalstories_extract.py``: one GPU job per (model, checkpoint)
pair populates the logit-lens trial-surprisal cache consumed by
ProvoLmeBenchmark._word_surprisal_all_layers(). Re-running is safe:
``logit_lens_surprisal_trials`` checks its own npz cache before loading the
model, so already-cached (model, revision) pairs are a fast no-op.

Unlike ZuCo (SR/NR/TSR are three distinct sentence sets, cached separately), Provo's
SP/FP/GP measures share the exact same 55-passage word set — this script therefore
has **no --measure flag and runs once per (model, checkpoint)**, not three times.
The resulting cache (cache_key="provo", keyed only by model/revision) is reused
by all three ``run/surprisal/behavioral/provo_fit.py --measure {SP,FP,GP}`` runs, a 3x
reduction in GPU/model-load cost versus naively mirroring ZuCo's per-task nesting.

The cache format/path is owned by
``neuroai.features.logit_lens._trial_surprisal_cache_path``, not a bespoke CSV.

Corpus perplexity (exp of mean per-word surprisal, nats) is derived from the
DEEPEST logit-lens layer's surprisal (numerically identical to the model's
true final-layer output) — no extra forward pass. Saved as
``<perplexity_dir>/<model_safe>/<revision>_provo_perplexity.json`` plus one row
appended to ``<perplexity_dir>/provo_perplexity_summary.csv``.

Indexing
--------
--array-size           Print total task count across all families and exit.
--global-index N       Run task N (flat index over all families in registry order).
--family F             Restrict to one family (must pair with --array-index I).
--array-index I        Run task I within family F.

Examples
--------
python scripts/run/surprisal/behavioral/provo_extract.py --config configs/surprisal/behavioral/provo_fp.yaml --array-size
python scripts/run/surprisal/behavioral/provo_extract.py --config configs/surprisal/behavioral/provo_fp.yaml --global-index 0 --device cuda
python scripts/run/surprisal/behavioral/provo_extract.py \\
    --config configs/surprisal/behavioral/provo_fp.yaml --family pythia --array-index 3 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np

from neuroai.datasets.provo import provo_word_list
from neuroai.features.llm import clear_hf_model_cache
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks, registry_model_configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _all_tasks():
    tasks = []
    for family in LPP_FAMILY_NAMES:
        tasks.extend(flatten_family_tasks(family))
    return tasks


def _total_tasks() -> int:
    return sum(len(flatten_family_tasks(f)) for f in LPP_FAMILY_NAMES)


def _resolve_task(args):
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
        logger.error("--array-index %d out of range [0, %d) for family %r", idx, len(tasks), args.family)
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


def _save_perplexity(task, surp: np.ndarray, perplexity_dir: str) -> None:
    """Save corpus-level Provo perplexity derived from the deepest layer's surprisal (nats)."""
    import json

    from neuroai.sweep import append_row, load_existing_csv

    valid = np.isfinite(surp)
    n_words = int(valid.sum())
    if n_words == 0:
        logger.warning("No valid surprisal values — skipping perplexity save")
        return
    corpus_ppl = float(np.exp(np.mean(surp[valid])))

    out_dir = Path(perplexity_dir) / task.load_model_id.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    rev_safe = (task.revision or "main").replace("/", "_")
    record = {
        "model_id": task.load_model_id,
        "revision": task.revision,
        "step": task.step,
        "tokens_seen": task.tokens_seen,
        "corpus": "provo",
        "perplexity": round(corpus_ppl, 4), "perplexity_method": "surprisal_v2_sum_per_word",
        "n_words": n_words,
    }

    ppl_json = out_dir / f"{rev_safe}_provo_perplexity.json"
    ppl_json.write_text(json.dumps(record, indent=2))

    summary_csv = Path(perplexity_dir) / "provo_perplexity_summary.csv"
    existing = load_existing_csv(summary_csv)
    already = (
        not existing.empty
        and {"model_id", "revision"}.issubset(existing.columns)
        and ((existing["model_id"] == task.load_model_id) & (existing["revision"] == task.revision)).any()
    )
    if not already:
        append_row(summary_csv, record)

    logger.info("Perplexity saved: %s (ppl=%.3f, n_words=%d)", ppl_json.name, corpus_ppl, n_words)


def prefetch_one(
    task,
    *,
    provo_root: str,
    cache_dir: str,
    device: str,
    exclude_doc_edges: bool,
    trust_remote_code: bool = False,
    n_params: float | None = None,
    perplexity_dir: str | None = None,
) -> None:
    """Compute and cache per-layer surprisal for one (model, checkpoint) task (all 3 measures)."""
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

    # Word list is measure-independent (SP/FP/GP share the same 55 passages) and
    # subject-independent — this single cache serves all three downstream LME runs.
    word_df = provo_word_list(provo_root, exclude_doc_edges=exclude_doc_edges).reset_index(drop=True)

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
            arr = None
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
                cache_key="provo",
                dtype=surp_dtype,
            )
            logger.info("Saved per-layer surprisal cache: shape %s", arr.shape)
    except Exception:
        logger.error("Surprisal computation failed: %s @ %s", task.load_model_id, task.revision, exc_info=True)
        raise
    finally:
        if task.registry_family != "cbr_rnn":
            clear_hf_model_cache()

    if perplexity_dir and arr is not None:
        _save_perplexity(task, np.asarray(arr[:, -1], dtype=np.float64), perplexity_dir)


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
        "PROVO_FEATURE_CACHE_DIR",
        f"{os.environ['BENCHMARK_CACHE_ROOT']}/provo_features",
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
    p.add_argument("--config", default="configs/surprisal/behavioral/provo_fp.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--provo-root", default=None, help="Override Provo dataset root from config/env")
    p.add_argument("--cache-dir", default=None, help="Override PROVO_FEATURE_CACHE_DIR")
    p.add_argument("--perplexity-dir", default=None, help="Override PROVO_PERPLEXITY_DIR (set empty to skip)")
    args = p.parse_args()

    if args.array_size:
        print(_total_tasks())
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
    ds_cfg = cfg.get("dataset", {})

    provo_root = args.provo_root or os.path.expandvars(
        str(ds_cfg.get("root", "$BENCHMARK_ROOT/data/provo/osfstorage"))
    )
    exclude_doc_edges = bool(ds_cfg.get("exclude_doc_edges", True))

    env_dir = os.environ.get("PROVO_FEATURE_CACHE_DIR", "").strip()
    cache_dir = args.cache_dir or env_dir or str(
        cfg.get("cache", {}).get("features_dir", ".cache/provo_features")
    )
    cache_dir = os.path.expandvars(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    env_ppl_dir = os.environ.get("PROVO_PERPLEXITY_DIR", "").strip()
    perplexity_dir = args.perplexity_dir if args.perplexity_dir is not None else env_ppl_dir
    perplexity_dir = os.path.expandvars(perplexity_dir) if perplexity_dir else None
    if perplexity_dir:
        Path(perplexity_dir).mkdir(parents=True, exist_ok=True)

    logger.info("provo_root       : %s", provo_root)
    logger.info("cache_dir        : %s", cache_dir)
    logger.info("perplexity_dir   : %s", perplexity_dir or "(disabled)")
    logger.info("device           : %s", args.device)

    trust_remote_code = _trust_remote_code_for_task(task)
    n_params = _n_params_for_task(task)
    prefetch_one(
        task,
        provo_root=provo_root,
        cache_dir=cache_dir,
        device=args.device,
        exclude_doc_edges=exclude_doc_edges,
        trust_remote_code=trust_remote_code,
        n_params=n_params,
        perplexity_dir=perplexity_dir,
    )


if __name__ == "__main__":
    main()
