#!/usr/bin/env python3
"""Precompute and cache Natural Stories fMRI PER-LAYER surprisal (logit-lens)
for all LME checkpoint tasks.

Mirrors ``run/surprisal/behavioral/naturalstories_extract.py`` (the behavioral SPR LME prefetch
script) but keyed per-doc rather than per multi-story bundle, since the fMRI LME
encoder (``NsFmriLmeBenchmark``) processes one of the 8 canonical stories at a
time. One GPU job per (model, checkpoint) pair populates the logit-lens cache
(``neuroai.features.logit_lens.logit_lens_surprisal_all_layers``, the same
cache used by Fedorenko2016/Tuckute) for all 8 docs under
``NS_FEATURE_CACHE_DIR`` — no bespoke CSV cache needed here anymore (the old
single-top-layer ``batch_surprisal_one_model`` + ``ns_fmri_surprisal_cache_path``
CSV cache has been replaced).

Scope: every registry family, including ``cbr_rnn`` (bypasses
``AutoModelForCausalLM`` entirely and has a single hidden-state vector per
token — no logit-lens layer sweep, but ``batch_surprisal_cbr_rnn`` gives a
genuine ``n_layers=1`` result the rest of the pipeline handles the same way
as any other family). Also includes the FLA-based ``trust_remote_code``
families — stickbreaking, alibi, forgetting, hard_2gram/3gram/5gram — via
``logit_lens.py``'s hook-based hidden-states fallback (see
``docs/history/logit_lens_constrained_models_handoff.md``, verified on Jean Zay H100). Those
families' ``fla`` backend needs a live CUDA driver just to import, so — like
every family here — this only ever runs as a GPU stage. The one narrow
exception within ``babylm`` is the MoE model ``Jtapsa/moep`` specifically
(residual-stream width changes mid-network, so no single final-norm/
unembedding pair is valid for every layer) — that one task fails cleanly with
a clear error while the rest of the family (CLASS-IT, blm-gpt2s) succeeds.

Indexing
--------
--array-size           Print total task count across all families and exit.
--global-index N       Run task N (flat index over all families in registry order).
--family F             Restrict to one family (must pair with --array-index I).
--array-index I        Run task I within family F.

Examples
--------
python scripts/run/surprisal/neural/ns_fmri_extract.py --array-size
python scripts/run/surprisal/neural/ns_fmri_extract.py --global-index 0 --device cuda
python scripts/run/surprisal/neural/ns_fmri_extract.py \\
    --family pythia --array-index 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.datasets.natural_stories_fmri import NS_FMRI_DOCS, load_canonical_words
from neuroai.features.llm import clear_hf_model_cache
from neuroai.features.logit_lens import logit_lens_surprisal_all_layers
from neuroai.models.registry import flatten_family_tasks, registry_model_configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Every registry family. cbr_rnn gets batch_surprisal_cbr_rnn (n_layers=1,
# see prefetch_one) instead of logit-lens. Includes the FLA-based
# trust_remote_code families (stickbreaking, alibi, forgetting, hard-n-gram)
# via logit_lens.py's hook-based hidden-states fallback — see module
# docstring. babylm is included too; only its one MoE model (Jtapsa/moep)
# fails per-task, the rest of the family works.
SUPPORTED_FAMILY_NAMES: tuple[str, ...] = (
    "gpt2", "gptneo", "opt", "mamba", "mamba2",
    "pythia", "olmo2", "qwen3", "llama32", "mistral", "babylm",
    "stickbreaking", "alibi", "forgetting",
    "hard_2gram", "hard_3gram", "hard_5gram",
    "cbr_rnn",
)


def _all_tasks():
    tasks = []
    for family in SUPPORTED_FAMILY_NAMES:
        tasks.extend(flatten_family_tasks(family))
    return tasks


def _total_tasks() -> int:
    return sum(len(flatten_family_tasks(f)) for f in SUPPORTED_FAMILY_NAMES)


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
        logger.error(
            "--array-index %d out of range [0, %d) for family %r",
            idx, len(tasks), args.family,
        )
        sys.exit(1)
    return tasks[idx]


def _trust_remote_code_for_task(task) -> bool:
    for m in registry_model_configs(task.registry_family):
        if m["model_id"] == task.model_id:
            return bool(m.get("trust_remote_code", False))
    return False


def _n_params_for_task(task) -> float | None:
    for m in registry_model_configs(task.registry_family):
        if m["model_id"] == task.model_id:
            return m.get("n_params")
    return None


def _save_ns_fmri_perplexity(
    model_id: str, revision: str, n_params: float | None,
    perplexity: float, n_tokens: int, max_length: int,
) -> None:
    perplexity_dir = Path("outputs_torch") / "ns_fmri/perplexity"
    safe = model_id.replace("/", "_")
    rev_safe = (revision or "main").replace("/", "_")
    out_dir = perplexity_dir / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rev_safe}_ns_fmri_perplexity.json"
    record = {
        "model_id": model_id, "revision": revision or "main", "label": model_id.split("/")[-1],
        "n_params": n_params, "family": "pythia", "dataset": "naturalstories_fmri",
        "perplexity": perplexity, "n_tokens": n_tokens, "max_length": max_length,
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved NS-fMRI perplexity %.4f for %s @ %s", perplexity, model_id, revision or "main")


def _collect_ns_fmri_perplexity_summary() -> None:
    import pandas as pd

    perplexity_dir = Path("outputs_torch") / "ns_fmri/perplexity"
    rows = []
    for p in sorted(perplexity_dir.glob("*/*_ns_fmri_perplexity.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "error" in d:
            continue
        rows.append(d)
    if not rows:
        logger.warning("No *_ns_fmri_perplexity.json files found under %s", perplexity_dir)
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["model_id", "revision"], keep="last")
    out_csv = perplexity_dir / "ns_fmri_perplexity_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info("wrote %d rows -> %s", len(df), out_csv)


def _extract_perplexity(
    task, *, data_root: str, froi_table: str, docs: list[str] | None,
    device: str, trust_remote_code: bool, n_params: float | None,
) -> None:
    """Reuse the model prefetch_one's docs loop just loaded (see module
    docstring / run/surprisal/neural/lpp_extract.py for why re-calling
    _get_cached_causal_lm here, BEFORE clear_hf_model_cache() runs, is a
    cache hit and not a second GPU load). NS-fMRI's own 8-doc corpus is used
    -- NOT naturalstories_spr's: empirically verified NOT interchangeable
    (pythia-14m@step143000: 602 vs 1130 perplexity, a ~47% difference) even
    though the two datasets share most of their underlying stories -- 8/10
    story overlap is not enough to make the aggregate comparable, likely
    because the 2 non-shared stories and the shorter (8 vs 10 doc) corpus
    length both shift the aggregate substantially, not just tokenization."""
    from neuroai.features.llm import _get_cached_causal_lm
    from scripts.perplexity.lpp import compute_perplexity, _max_length_for_model

    model_id = task.load_model_id
    revision = task.revision if task.revision not in ("main", "") else None
    out_path = (
        Path("outputs_torch") / "ns_fmri/perplexity" / model_id.replace("/", "_")
        / f"{(revision or 'main').replace('/', '_')}_ns_fmri_perplexity.json"
    )
    if out_path.is_file():
        logger.info("NS-fMRI perplexity already cached for %s @ %s — skipping", model_id, revision or "main")
        return
    try:
        target_docs = docs or NS_FMRI_DOCS
        texts = [
            " ".join(load_canonical_words(data_root, doc, froi_table)["word"].astype(str).tolist())
            for doc in target_docs
        ]
        if "cbr-rnn" in model_id.lower() or "cbr_rnn" in model_id.lower():
            # Custom PyTorchModelHubMixin RNN, not AutoModelForCausalLM --
            # see compute_lpp_perplexity.compute_cbr_rnn_perplexity_over_texts's
            # docstring for why this is a separate code path.
            from scripts.perplexity.lpp import compute_cbr_rnn_perplexity_over_texts

            result = compute_cbr_rnn_perplexity_over_texts(model_id, texts, device)
            _save_ns_fmri_perplexity(
                model_id, revision or "main", n_params,
                result["perplexity"], result["n_tokens"], result["max_length"],
            )
            return
        tokenizer, model = _get_cached_causal_lm(
            model_id,
            revision=revision,
            device=device,
            load_kw={"trust_remote_code": trust_remote_code},
            tokenizer_no_revision=True,
            n_params=n_params,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_length = _max_length_for_model(model, model_id)
        result = compute_perplexity(model, tokenizer, texts, device, max_length)
        _save_ns_fmri_perplexity(
            model_id, revision or "main", n_params,
            result["perplexity"], result["n_tokens"], max_length,
        )
    except Exception as exc:
        logger.error("NS-fMRI perplexity FAILED for %s @ %s: %s", model_id, revision or "main", exc, exc_info=True)


def prefetch_one(
    task,
    *,
    data_root: str,
    froi_table: str,
    cache_dir: str,
    device: str,
    docs: list[str] | None,
    trust_remote_code: bool = False,
    n_params: float | None = None,
    max_length: int | None = None,
    skip_perplexity: bool = False,
) -> None:
    """Compute and cache per-layer surprisal for one (model, checkpoint) task,
    across all docs. For standard/FLA architectures this is logit-lens
    (``logit_lens_surprisal_all_layers``); for cbr_rnn (single hidden-state
    vector per token, no layer dimension) it's ``batch_surprisal_cbr_rnn``
    instead, giving a genuine ``n_layers=1`` result. Both check their own
    cache BEFORE loading the model, so an all-cache-hit task never touches
    the LLM.
    """
    model_revision = task.revision if task.revision not in ("main", "") else None
    target_docs = docs or NS_FMRI_DOCS
    is_cbr_rnn = task.registry_family == "cbr_rnn"

    try:
        for doc_idx, doc in enumerate(target_docs):
            logger.info(
                "Extracting per-layer surprisal: %s @ %s doc=%s (idx=%d, family=%s step=%d tokens=%dB)",
                task.load_model_id, task.revision, doc, doc_idx,
                task.registry_family, task.step, task.tokens_seen // 1_000_000_000,
            )

            words_df = load_canonical_words(data_root, doc, froi_table)
            words = words_df["word"].astype(str).tolist()

            if is_cbr_rnn:
                from neuroai.features.cbr_rnn import batch_surprisal_cbr_rnn

                contexts = [" ".join(words[: i + 1]) for i in range(len(words))]
                batch_surprisal_cbr_rnn(
                    task.load_model_id,
                    contexts,
                    words,
                    device=device,
                    cache_dir=cache_dir,
                    section_id=doc_idx,
                )
            else:
                logit_lens_surprisal_all_layers(
                    words,
                    task.load_model_id,
                    revision=model_revision,
                    device=device,
                    max_length=max_length,
                    tokenizer_no_revision=True,
                    cache_dir=cache_dir,
                    section_id=doc_idx,
                    n_params=n_params,
                    trust_remote_code=trust_remote_code,
                )

        if not skip_perplexity:
            _extract_perplexity(
                task, data_root=data_root, froi_table=froi_table, docs=docs,
                device=device, trust_remote_code=trust_remote_code, n_params=n_params,
            )
    except Exception:
        logger.error(
            "Surprisal computation failed: %s @ %s", task.load_model_id, task.revision,
            exc_info=True,
        )
        raise
    finally:
        # Evict the cached model ONCE, after all 8 docs are done — NOT per
        # doc. logit_lens_surprisal_all_layers/batch_surprisal_cbr_rnn cache
        # to disk per doc but keep the model itself in an LRU-1 in-memory
        # cache across calls; clearing it inside the loop forced a full
        # reload (~10+ min for a 30B model) on every single doc instead of
        # once per (model, checkpoint) task.
        if not is_cbr_rnn:
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
    p.add_argument("--config", default="configs/surprisal/neural/ns_fmri.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default=None, help="Override data_root from config/env")
    p.add_argument("--cache-dir", default=None, help="Override NS_FEATURE_CACHE_DIR")
    p.add_argument("--skip_perplexity", action="store_true",
                   help="Skip the post-extraction NS-fMRI-corpus perplexity computation")
    p.add_argument("--collect_perplexity_summary", action="store_true",
                   help="Rebuild outputs_torch/ns_fmri/perplexity/ns_fmri_perplexity_summary.csv "
                        "from every *_ns_fmri_perplexity.json on disk and exit.")
    args = p.parse_args()

    if args.collect_perplexity_summary:
        _collect_ns_fmri_perplexity_summary()
        return

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

    data_root = args.data_root or os.path.expandvars(
        str(ds_cfg.get("data_root", "$BENCHMARK_ROOT/data/naturalstories_fmri/osfstorage"))
    )
    froi_table = str(ds_cfg.get("froi_table", "LANG_MD"))
    docs = ds_cfg.get("docs")

    ns_dir = os.environ.get("NS_FEATURE_CACHE_DIR", "").strip()
    cache_dir = args.cache_dir or ns_dir or str(
        cfg.get("cache", {}).get("features_dir", ".cache/naturalstories_features")
    )
    cache_dir = os.path.expandvars(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    logger.info("data_root : %s", data_root)
    logger.info("froi_table: %s", froi_table)
    logger.info("cache_dir : %s", cache_dir)
    logger.info("device    : %s", args.device)

    trust_remote_code = _trust_remote_code_for_task(task)
    n_params = _n_params_for_task(task)
    prefetch_one(
        task,
        data_root=data_root,
        froi_table=froi_table,
        cache_dir=cache_dir,
        device=args.device,
        docs=docs,
        trust_remote_code=trust_remote_code,
        n_params=n_params,
        skip_perplexity=args.skip_perplexity,
    )


if __name__ == "__main__":
    main()
