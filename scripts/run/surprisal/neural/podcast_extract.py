#!/usr/bin/env python3
"""Precompute and cache Podcast per-layer logit-lens surprisal for the LME
pipeline (``neuroai.benchmarks.podcast_lme.PodcastLmeBenchmark``).

Mirrors ``run/surprisal/neural/ns_fmri_extract.py``, but the Podcast transcript is a
single continuous narrative (one "doc"), not 8 canonical stories, so there's
no docs loop — one GPU task per (family, array-index) populates exactly one
cache entry under ``PODCAST_FEATURE_CACHE_DIR``.

``PodcastLmeBenchmark._load_surprisal`` always loads on CPU (``device="cpu"``
hardcoded there) — this script is what lets that stay true in practice: run
this once per model on a GPU node (fast, and the only place a
``trust_remote_code`` family's import — e.g. the FLA/Triton backend needed by
stickbreaking/alibi/forgetting/hard-n-gram, or a custom babylm architecture —
gets a live CUDA driver), so the CPU-only LME SLURM job always hits a warm
cache and never needs to load a model itself.

cbr_rnn is out of scope here: PodcastLmeBenchmark._load_surprisal computes it
directly and cheaply on CPU every time (no cache), so it never runs this
script (and it isn't a HF causal LM, so ``logit_lens_surprisal_all_layers``
wouldn't apply to it anyway).

Indexing
--------
--array-size           Print total task count across SUPPORTED_FAMILY_NAMES and exit.
--global-index N       Run task N (flat index over all families in registry order).
--family F             Restrict to one family (must pair with --array-index I).
--array-index I        Run task I within family F.

Examples
--------
python scripts/run/surprisal/neural/podcast_extract.py --array-size
python scripts/run/surprisal/neural/podcast_extract.py --family babylm --array-index 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.datasets.podcast import load_transcript
from neuroai.features.llm import clear_hf_model_cache
from neuroai.features.logit_lens import logit_lens_surprisal_all_layers
from neuroai.models.registry import flatten_family_tasks, registry_model_configs, resolve_family_task

# Some babylm-community trust_remote_code configs (the gpt-bert-* variants)
# store a raw torch.dtype in their config dict and don't sanitize it before
# transformers' own AutoConfig.from_pretrained -> from_dict logs
# `f"Model config {config}"` (an eager f-string, unconditional on log level).
# That calls the vendor config's __repr__ -> to_json_string() -> json.dumps()
# with no `default=`, so plain json.dumps crashes with
# "TypeError: Object of type dtype is not JSON serializable" before the model
# ever loads. Scoped to this one-model-per-process prefetch script only (not
# neuroai/features/logit_lens.py, which is shared far more broadly) — patch
# json.JSONEncoder.default to stringify anything it doesn't otherwise know,
# matching what passing `default=str` to that json.dumps call would have done.
_orig_json_default = json.JSONEncoder.default


def _json_default_stringify_unknown(self, o):
    try:
        return _orig_json_default(self, o)
    except TypeError:
        return str(o)


json.JSONEncoder.default = _json_default_stringify_unknown

# Same babylm-community gpt-bert-* vendor code also skips calling
# self.post_init() in its custom __init__ (a required call per the
# PreTrainedModel.post_init docstring — it's what sets
# `all_tied_weights_keys`, `_keep_in_fp32_modules`, `_tp_plan`, etc. on every
# other architecture). Without it, transformers 5.13.1's own
# caching_allocator_warmup() -> get_total_byte_count() crashes reading
# `model.all_tied_weights_keys` with an unrelated-looking AttributeError, well
# before any of our code runs. Patch PreTrainedModel.__init__ to guarantee a
# safe empty-dict fallback for any subclass that forgets it — a no-op for
# every well-behaved architecture that already sets it via post_init().
import transformers.modeling_utils as _modeling_utils  # noqa: E402

_orig_pretrained_model_init = _modeling_utils.PreTrainedModel.__init__


def _pretrained_model_init_with_tied_weights_fallback(self, *args, **kwargs):
    _orig_pretrained_model_init(self, *args, **kwargs)
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = {}
    # Same vendor code also declares _keys_to_ignore_on_load_unexpected as a
    # plain list (the old-style convention) — transformers 5.13.1's
    # _adjust_missing_and_unexpected_keys does `self._keys_to_ignore_on_load_unexpected | additional_unexpected_patterns`,
    # which requires a set. Coerce defensively; a no-op for every
    # already-compliant architecture.
    if isinstance(getattr(self, "_keys_to_ignore_on_load_unexpected", None), list):
        self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected)
    if isinstance(getattr(self, "_keys_to_ignore_on_load_missing", None), list):
        self._keys_to_ignore_on_load_missing = set(self._keys_to_ignore_on_load_missing)


_modeling_utils.PreTrainedModel.__init__ = _pretrained_model_init_with_tied_weights_fallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Every registry family the Podcast LME sweep has (or will have) an
# lme_<family>.sh job for. cbr_rnn deliberately excluded — see module docstring.
SUPPORTED_FAMILY_NAMES: tuple[str, ...] = (
    "gpt2", "gptneo", "opt", "mamba", "mamba2",
    "pythia", "olmo2", "qwen3", "llama32", "mistral", "babylm",
    "stickbreaking", "alibi", "forgetting",
    "hard_2gram", "hard_3gram", "hard_5gram",
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

    return resolve_family_task(args.family, args.array_index)


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


def _save_podcast_perplexity(
    model_id: str, revision: str, n_params: float | None,
    perplexity: float, n_tokens: int, max_length: int,
) -> None:
    perplexity_dir = Path("outputs_jeanzay") / "podcast/lme_perplexity"
    safe = model_id.replace("/", "_")
    rev_safe = (revision or "main").replace("/", "_")
    out_dir = perplexity_dir / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rev_safe}_podcast_perplexity.json"
    record = {
        "model_id": model_id, "revision": revision or "main", "label": model_id.split("/")[-1],
        "n_params": n_params, "family": "pythia", "dataset": "podcast",
        "perplexity": perplexity, "perplexity_method": "logits_v2_per_word", "n_tokens": n_tokens, "max_length": max_length,
    }
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("Saved Podcast perplexity %.4f for %s @ %s", perplexity, model_id, revision or "main")


def _collect_podcast_perplexity_summary() -> None:
    import pandas as pd

    perplexity_dir = Path("outputs_jeanzay") / "podcast/lme_perplexity"
    rows = []
    for p in sorted(perplexity_dir.glob("*/*_podcast_perplexity.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "error" in d:
            continue
        rows.append(d)
    if not rows:
        logger.warning("No *_podcast_perplexity.json files found under %s", perplexity_dir)
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["model_id", "revision"], keep="last")
    out_csv = perplexity_dir / "podcast_perplexity_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info("wrote %d rows -> %s", len(df), out_csv)


def _extract_perplexity(
    task, *, words: list[str], device: str, trust_remote_code: bool, n_params: float | None,
) -> None:
    """Reuse the model prefetch_one's logit-lens call just loaded (see
    run/surprisal/neural/lpp_extract.py's module docstring for why re-calling
    _get_cached_causal_lm here, BEFORE clear_hf_model_cache() runs, is a
    cache hit and not a second GPU load). This is a SEPARATE cache namespace
    (outputs_jeanzay/podcast/lme_perplexity/) from the pre-existing standalone
    scripts/perplexity/podcast.py's (outputs_jeanzay/podcast_
    perplexity/) -- that script's own --checkpoints --family pythia_checkpoints
    --all mode already works end-to-end and could be run instead/as well, but
    would reload every one of the 110 checkpoints from scratch a second time;
    this hook gets the same number for free."""
    from neuroai.features.llm import _get_cached_causal_lm
    from scripts.perplexity.lpp import compute_perplexity, _max_length_for_model

    model_id = task.load_model_id
    revision = task.revision if task.revision not in ("main", "") else None
    out_path = (
        Path("outputs_jeanzay") / "podcast/lme_perplexity" / model_id.replace("/", "_")
        / f"{(revision or 'main').replace('/', '_')}_podcast_perplexity.json"
    )
    if out_path.is_file() and json.loads(out_path.read_text()).get("perplexity_method") == "logits_v2_per_word":
        logger.info("Podcast perplexity already cached for %s @ %s — skipping", model_id, revision or "main")
        return
    try:
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
        text = " ".join(w for w in words if w and w == w)  # drop NaN-as-str stragglers
        result = compute_perplexity(model, tokenizer, [text], device, max_length)
        _save_podcast_perplexity(
            model_id, revision or "main", n_params,
            result["perplexity"], result["n_tokens"], max_length,
        )
    except Exception as exc:
        logger.error("Podcast perplexity FAILED for %s @ %s: %s", model_id, revision or "main", exc, exc_info=True)


def prefetch_one(
    task,
    *,
    words: list[str],
    cache_dir: str,
    device: str,
    trust_remote_code: bool = False,
    n_params: float | None = None,
    max_length: int | None = None,
    skip_perplexity: bool = False,
) -> None:
    model_revision = task.revision if task.revision not in ("main", "") else None
    logger.info(
        "Extracting per-layer surprisal: %s @ %s (family=%s)",
        task.load_model_id, task.revision, task.registry_family,
    )
    # BabyLM-community's gpt-bert-* models crash under their own default
    # bf16 weights (custom DWA accumulator does an in-place float32 buffer
    # .set_(), see logit_lens.py's `dtype` docstring) — force float32 for
    # just this family's models, no effect on any other architecture.
    dtype = None
    if "gpt-bert" in task.model_id.lower():
        import torch

        dtype = torch.float32
    is_cbr_rnn = task.registry_family == "cbr_rnn"
    try:
        if is_cbr_rnn:
            # CBR-RNN is a custom PyTorchModelHubMixin model, not a Transformers
            # causal LM. Use the same bounded word contexts as PodcastLmeBenchmark
            # and persist one (word, 1) surprisal array for cache-only consumers.
            from neuroai.datasets.podcast import build_word_context
            from neuroai.features.cbr_rnn import batch_surprisal_cbr_rnn

            batch_surprisal_cbr_rnn(
                task.load_model_id,
                build_word_context(words),
                words,
                device=device,
                n_params=n_params,
                cache_dir=cache_dir,
                section_id=0,
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
                section_id=0,
                lang="en",
                n_params=n_params,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
            )
        if not skip_perplexity and not is_cbr_rnn:
            _extract_perplexity(task, words=words, device=device, trust_remote_code=trust_remote_code, n_params=n_params)
    except Exception:
        logger.error(
            "Surprisal computation failed: %s @ %s", task.load_model_id, task.revision,
            exc_info=True,
        )
        raise
    finally:
        if not is_cbr_rnn:
            clear_hf_model_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--array-size", action="store_true", help="Print total task count and exit")
    p.add_argument("--global-index", type=int, default=None, metavar="N")
    p.add_argument("--family", default=None, help="Registry family key (e.g. babylm)")
    p.add_argument("--array-index", type=int, default=None, metavar="I")
    p.add_argument("--config", default="configs/surprisal/neural/podcast.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bids-root", default=None, help="Override dataset.bids_root from config/env")
    p.add_argument("--cache-dir", default=None, help="Override PODCAST_FEATURE_CACHE_DIR")
    p.add_argument("--skip_perplexity", action="store_true",
                   help="Skip the post-extraction Podcast-corpus perplexity computation")
    p.add_argument("--collect_perplexity_summary", action="store_true",
                   help="Rebuild outputs_jeanzay/podcast/lme_perplexity/podcast_perplexity_summary.csv "
                        "from every *_podcast_perplexity.json on disk and exit.")
    args = p.parse_args()

    if args.collect_perplexity_summary:
        _collect_podcast_perplexity_summary()
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
        f"  array_index={task.array_index}"
    )

    import yaml

    repo_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    os.environ.setdefault("BENCHMARK_ROOT", str(repo_root))
    config_path = args.config
    cfg: dict = {}
    if Path(config_path).is_file():
        with open(config_path, encoding="utf-8") as f:
            raw = os.path.expandvars(f.read())
        cfg = yaml.safe_load(raw) or {}
    ds_cfg = cfg.get("dataset", {})
    cache_cfg = cfg.get("cache", {})

    bids_root = args.bids_root or os.path.expandvars(
        str(ds_cfg.get("bids_root", "$BENCHMARK_ROOT/data/podcast"))
    )
    max_length = int(ds_cfg["max_length"]) if ds_cfg.get("max_length") else None

    cache_dir = args.cache_dir or os.environ.get("PODCAST_FEATURE_CACHE_DIR", "").strip() or os.path.expandvars(
        str(cache_cfg.get("features_dir", ".cache/podcast_features"))
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    logger.info("bids_root : %s", bids_root)
    logger.info("cache_dir : %s", cache_dir)
    logger.info("device    : %s", args.device)

    words_df = load_transcript(bids_root)
    words = words_df["word"].astype(str).tolist()

    trust_remote_code = _trust_remote_code_for_task(task)
    n_params = _n_params_for_task(task)
    prefetch_one(
        task,
        words=words,
        cache_dir=cache_dir,
        device=args.device,
        trust_remote_code=trust_remote_code,
        n_params=n_params,
        max_length=max_length,
        skip_perplexity=args.skip_perplexity,
    )


if __name__ == "__main__":
    main()
