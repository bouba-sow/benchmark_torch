"""Token-level surprisal and entropy helpers (standalone, not a NeuralSet extractor).

These functions work on plain text using HuggingFace causal LMs, and are
called from within the pipeline rather than registered as NeuralSet extractors
(which require a more complex pydantic setup).
"""

from __future__ import annotations

import logging
import os
import time
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _prepare_tokenizer(model_name: str, *, tokenizer_revision: str | None) -> tp.Any:
    from transformers import AutoTokenizer

    tok_kw: dict[str, tp.Any] = {}
    if tokenizer_revision is not None:
        tok_kw["revision"] = tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _surprisal_one_pair(
    model: tp.Any,
    tokenizer: tp.Any,
    device: torch.device,
    context: str,
    target: str,
    *,
    output: tp.Literal["surprisal", "entropy", "both"] = "surprisal",
) -> np.ndarray:
    """Forward pass for one (context, target) pair.

    ``context`` may be either:

    * **Prefix-only** — text strictly before ``target``; we form ``full = context + target``
      and count prefix tokens from ``context`` alone (original behaviour).
    * **Cumulative (NeuralSet ``AddConcatenationContext``)** — ``context`` is the running
      join of words through the current row and **ends with** ``target`` as a substring.
      Then we tokenize ``full = context`` only and set the token boundary like
      :class:`neuralset.extractors.text.HuggingFaceText`: character prefix
      ``context[:-len(target)].rstrip()`` → tokenize to get ``n_ctx`` (prefix token count),
      and surprisal is over ``input_ids[n_ctx:]`` (the target tokens only).
    """
    ctx_s = context.strip()
    tgt_s = target.strip()

    cumulative = bool(tgt_s and ctx_s.endswith(tgt_s))
    if cumulative:
        full = ctx_s
        prefix_text = ctx_s[: -len(tgt_s)].rstrip()
        n_ctx = (
            len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
            if prefix_text
            else 0
        )
    else:
        full = (ctx_s + " " + tgt_s).strip() if ctx_s else tgt_s
        n_ctx = len(tokenizer(ctx_s, add_special_tokens=False)["input_ids"]) if ctx_s else 0

    toks = tokenizer(full, return_tensors="pt")
    total_len = toks["input_ids"].shape[1]

    # Some custom architectures (e.g. small BabyLM-track checkpoints) report a
    # huge/unset tokenizer.model_max_length, so a bare truncation=True is a
    # no-op — a long cumulative context (e.g. a late word in a multi-hundred-
    # word Natural Stories story) can then overflow the model's actual
    # position-embedding table and crash the forward pass. Cap to the
    # model's real window and truncate from the LEFT (drop the oldest
    # context) so the target — always at the end of `full` — is never cut.
    cfg_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    tok_max = getattr(tokenizer, "model_max_length", None)
    candidates = [v for v in (cfg_max, tok_max) if isinstance(v, int) and 0 < v < 1_000_000]
    max_len = min(candidates) if candidates else 2048

    if total_len > max_len:
        drop = total_len - max_len
        toks = {k: v[:, drop:] for k, v in toks.items()}
        n_ctx = max(0, n_ctx - drop)

    with torch.no_grad():
        out = model(**{k: v.to(device) for k, v in toks.items()}, use_cache=False)
    if hasattr(out, "logits") and out.logits is not None:
        logits = out.logits[0].cpu().float()
    elif hasattr(out, "last_hidden_state") and hasattr(model, "lm_head"):
        # Base model without built-in LM head (e.g. Mistral3Model extracted from VLM).
        logits = model.lm_head(out.last_hidden_state)[0].cpu().float()
    else:
        raise AttributeError(
            f"Model output has no .logits and no compatible lm_head fallback: "
            f"output_type={type(out).__name__}, model_type={type(model).__name__}"
        )
    log_probs = torch.log_softmax(logits, dim=-1)
    ids = toks["input_ids"][0]

    target_ids = ids[n_ctx:]
    surp = (
        np.array(
            [
                -log_probs[n_ctx + j - 1, target_ids[j]].item()
                for j in range(len(target_ids))
            ],
            dtype=np.float64,
        )
        if len(target_ids) > 0
        else np.array([0.0], dtype=np.float64)
    )

    if output == "surprisal":
        return np.array([float(surp.mean())], dtype=np.float32)

    ent_arr = -(torch.softmax(logits, dim=-1) * log_probs).sum(-1).numpy()
    ent = float(ent_arr[n_ctx - 1] if n_ctx > 0 else ent_arr[0])

    if output == "entropy":
        return np.array([ent], dtype=np.float32)
    return np.array([float(surp.mean()), ent], dtype=np.float32)


def compute_surprisal(
    model_name: str,
    context: str,
    target: str,
    *,
    output: tp.Literal["surprisal", "entropy", "both"] = "surprisal",
    device: str = "auto",
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
) -> np.ndarray:
    """Compute surprisal / entropy for a target word given its context.

    Parameters
    ----------
    model_name : str
        HuggingFace causal LM identifier (e.g. ``"openai-community/gpt2"``).
    context : str
        Text before ``target``, **or** cumulative text through ``target`` (suffix equals
        ``target``), as produced by NeuralSet ``AddConcatenationContext`` on ``Word`` rows.
    target : str
        Target word whose surprisal to compute.
    output : str
        ``"surprisal"``, ``"entropy"``, or ``"both"`` → 2-vector.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    model_revision : str, optional
        Passed to ``AutoModelForCausalLM.from_pretrained`` (e.g. Pythia ``"step8000"``).
    tokenizer_revision : str, optional
        Passed to ``AutoTokenizer.from_pretrained``. Default: omit (HF default branch).
        For Pythia checkpoints, use ``"main"`` so the tokenizer matches all steps.

    Returns
    -------
    np.ndarray
        Scalar (shape ``(1,)``) or 2-vector (shape ``(2,)``).
    """
    from transformers import AutoModelForCausalLM

    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    tokenizer = _prepare_tokenizer(model_name, tokenizer_revision=tokenizer_revision)
    load_kw: dict[str, tp.Any] = {}
    if model_revision is not None:
        load_kw["revision"] = model_revision
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw).to(device_t)
    model.eval()
    try:
        return _surprisal_one_pair(
            model, tokenizer, device_t, context, target, output=output
        )
    finally:
        del model
        if device_t.type == "cuda":
            torch.cuda.empty_cache()


def _resolve_device_str(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def batch_surprisal_one_model(
    model_name: str,
    contexts: list[str],
    targets: list[str],
    *,
    output: tp.Literal["surprisal", "entropy", "both"] = "surprisal",
    device: str = "auto",
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    n_params: float | None = None,
    trust_remote_code: bool = False,
    untrained: bool = False,
    untrained_seed: int = 0,
) -> np.ndarray:
    """Compute surprisal for many pairs with a **single** model load.

    Use this in checkpoint sweeps so each training revision loads weights once.

    Each ``contexts[i]`` may be a strict prefix before ``targets[i]``, or cumulative
    text ending with ``targets[i]`` (NeuralSet ``AddConcatenationContext``); see
    :func:`_surprisal_one_pair`.

    Parameters
    ----------
    tokenizer_revision : str, optional
        Branch for the tokenizer (Pythia: ``"main"`` is recommended).
    model_revision : str, optional
        Weight revision (Pythia: ``"step512"``, etc.).
    untrained : bool
        If ``True``, use a randomly initialised model (same architecture,
        pretrained weights skipped) instead of the pretrained checkpoint.
    """
    from .llm import _get_cached_causal_lm, _model_input_device, clear_hf_model_cache

    dev_str = _resolve_device_str(device)
    _load_kw: dict[str, object] = {"trust_remote_code": True} if trust_remote_code else {}
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=model_revision,
        device=dev_str,
        load_kw=_load_kw,
        tokenizer_revision=tokenizer_revision,
        tokenizer_no_revision=tokenizer_revision is None,
        n_params=n_params,
        untrained=untrained,
        untrained_seed=untrained_seed,
    )
    device_t = torch.device(_model_input_device(model))
    rows: list[np.ndarray] = []
    try:
        for ctx, tgt in zip(contexts, targets):
            rows.append(
                _surprisal_one_pair(
                    model, tokenizer, device_t, ctx, tgt, output=output
                )
            )
    finally:
        clear_hf_model_cache()

    return np.vstack(rows)


def batch_surprisal_mean_two_words(
    model_name: str,
    contexts_a: list[str],
    targets_a: list[str],
    contexts_b: list[str],
    targets_b: list[str],
    *,
    output: tp.Literal["surprisal", "entropy", "both"] = "surprisal",
    device: str = "auto",
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    n_params: float | None = None,
) -> np.ndarray:
    """Mean of two per-trial surprisal scalars with a **single** model load.

    For each index ``i``, computes
    ``(_surprisal_one_pair(..., A_i) + _surprisal_one_pair(..., B_i)) / 2``,
    where each pair returns the usual mean-token surprisal for that target.

    RT benchmarks use this for e.g. mean(surprisal(penultimate word),
    surprisal(last word)) so the summary mixes agreement on the late verb
    region with the following continuation.
    """
    from .llm import _get_cached_causal_lm, _model_input_device, clear_hf_model_cache

    if output != "surprisal":
        raise ValueError("batch_surprisal_mean_two_words only supports output='surprisal'")
    if not (len(contexts_a) == len(targets_a) == len(contexts_b) == len(targets_b)):
        raise ValueError("context/target lists must have equal length")

    dev_str = _resolve_device_str(device)
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=model_revision,
        device=dev_str,
        load_kw={},
        tokenizer_revision=tokenizer_revision,
        tokenizer_no_revision=tokenizer_revision is None,
        n_params=n_params,
    )
    device_t = torch.device(_model_input_device(model))
    vals: list[float] = []
    try:
        for ca, ta, cb, tb in zip(contexts_a, targets_a, contexts_b, targets_b):
            sa = float(
                _surprisal_one_pair(
                    model, tokenizer, device_t, ca, ta, output="surprisal"
                )[0]
            )
            sb = float(
                _surprisal_one_pair(
                    model, tokenizer, device_t, cb, tb, output="surprisal"
                )[0]
            )
            vals.append((sa + sb) / 2.0)
    finally:
        clear_hf_model_cache()

    return np.asarray(vals, dtype=np.float32).reshape(-1, 1)


batch_surprisal = batch_surprisal_one_model


# ---------------------------------------------------------------------------
# Natural Stories surprisal disk cache
# ---------------------------------------------------------------------------

def ns_surprisal_cache_path(
    cache_dir: str | Path,
    model_name: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    stories: list[int] | None,
) -> Path:
    """Return the CSV cache path for Natural Stories word-level surprisal.

    Cache key encodes model identity (name + revision + tokenizer branch) and
    which story subset was used, so different checkpoint sweeps never collide.
    """
    safe_model = model_name.replace("/", "_")
    safe_rev = (model_revision or "main").replace("/", "_").replace("-", "_")
    tok = (tokenizer_revision or "default").replace("/", "_").replace("-", "_")
    stories_tag = "all" if stories is None else "-".join(str(s) for s in sorted(stories))
    filename = f"{safe_model}__{safe_rev}__tok{tok}__stories{stories_tag}__ns_surprisal.csv"
    return Path(cache_dir) / filename


def load_ns_surprisal_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load a cached surprisal table; return None on miss or corruption."""
    if not cache_path.is_file():
        return None
    try:
        df = pd.read_csv(cache_path)
        if {"story_id", "zone", "surprisal"}.issubset(df.columns):
            logger.info("Surprisal cache hit: %s (%d rows)", cache_path, len(df))
            return df
        logger.warning("Surprisal cache missing required columns: %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to load surprisal cache %s: %s", cache_path, exc)
    return None


def save_ns_surprisal_cache(
    cache_path: Path,
    word_df: pd.DataFrame,
    surprisal: np.ndarray,
) -> None:
    """Atomically write a (story_id, zone, surprisal) CSV cache file.

    Uses a pid-stamped temp file + os.replace, same as the embedding cache,
    so concurrent SLURM array tasks on Lustre don't corrupt each other.
    """
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    table = word_df[["story_id", "zone"]].copy()
    table["surprisal"] = np.asarray(surprisal, dtype=np.float64)

    tmp = cache_path.parent / f"{cache_path.stem}.part{os.getpid()}.csv"
    try:
        table.to_csv(tmp, index=False)
        last_err: BaseException | None = None
        for attempt in range(6):
            try:
                os.replace(tmp, cache_path)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.25 * (attempt + 1))
        if last_err is not None:
            raise last_err
        logger.info("Surprisal cache written: %s (%d rows)", cache_path, len(table))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Natural Stories fMRI (Shain et al. 2020) surprisal disk cache
# ---------------------------------------------------------------------------
# Keyed per-doc (not per multi-story bundle like the SPR cache above) since the
# fMRI encoder already processes one doc at a time; merge key is (sentid, sentpos)
# within that doc, matching load_canonical_words()'s canonical word table.

def ns_fmri_surprisal_cache_path(
    cache_dir: str | Path,
    model_name: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    doc: str,
    untrained: bool = False,
    untrained_seed: int = 0,
) -> Path:
    """Return the CSV cache path for one Natural Stories fMRI doc's word surprisal."""
    safe_model = model_name.replace("/", "_")
    safe_rev = (model_revision or "main").replace("/", "_").replace("-", "_")
    if untrained:
        safe_rev = f"{safe_rev}_untrained_seed{untrained_seed}"
    tok = (tokenizer_revision or "default").replace("/", "_").replace("-", "_")
    safe_doc = str(doc).replace("/", "_")
    filename = f"{safe_model}__{safe_rev}__tok{tok}__doc{safe_doc}__ns_fmri_surprisal.csv"
    return Path(cache_dir) / filename


def load_ns_fmri_surprisal_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load a cached per-doc surprisal table; return None on miss or corruption."""
    if not cache_path.is_file():
        return None
    try:
        df = pd.read_csv(cache_path)
        if {"sentid", "sentpos", "surprisal"}.issubset(df.columns):
            logger.info("NS-fMRI surprisal cache hit: %s (%d rows)", cache_path, len(df))
            return df
        logger.warning("NS-fMRI surprisal cache missing required columns: %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to load NS-fMRI surprisal cache %s: %s", cache_path, exc)
    return None


def save_ns_fmri_surprisal_cache(
    cache_path: Path,
    words_df: pd.DataFrame,
    surprisal: np.ndarray,
) -> None:
    """Atomically write a (sentid, sentpos, surprisal) CSV cache file for one doc."""
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    table = words_df[["sentid", "sentpos"]].copy()
    table["surprisal"] = np.asarray(surprisal, dtype=np.float64)

    tmp = cache_path.parent / f"{cache_path.stem}.part{os.getpid()}.csv"
    try:
        table.to_csv(tmp, index=False)
        last_err: BaseException | None = None
        for attempt in range(6):
            try:
                os.replace(tmp, cache_path)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.25 * (attempt + 1))
        if last_err is not None:
            raise last_err
        logger.info("NS-fMRI surprisal cache written: %s (%d rows)", cache_path, len(table))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# UCL Reading (Frank et al. 2013) surprisal disk cache
# ---------------------------------------------------------------------------
# Distinct filename suffix from the Natural Stories cache above so that the two
# never collide even though both key on (story_id, zone) columns and default
# to a "stories_tag=all" filename component.

def ucl_reading_surprisal_cache_path(
    cache_dir: str | Path,
    model_name: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    sentences: list[int] | None,
) -> Path:
    """Return the CSV cache path for UCL Reading word-level surprisal."""
    safe_model = model_name.replace("/", "_")
    safe_rev = (model_revision or "main").replace("/", "_").replace("-", "_")
    tok = (tokenizer_revision or "default").replace("/", "_").replace("-", "_")
    sent_tag = "all" if sentences is None else "-".join(str(s) for s in sorted(sentences))
    filename = f"{safe_model}__{safe_rev}__tok{tok}__sentences{sent_tag}__ucl_reading_surprisal.csv"
    return Path(cache_dir) / filename


def load_ucl_reading_surprisal_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load a cached UCL Reading surprisal table; return None on miss or corruption."""
    if not cache_path.is_file():
        return None
    try:
        df = pd.read_csv(cache_path)
        if {"story_id", "zone", "surprisal"}.issubset(df.columns):
            logger.info("UCL Reading surprisal cache hit: %s (%d rows)", cache_path, len(df))
            return df
        logger.warning("UCL Reading surprisal cache missing required columns: %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to load UCL Reading surprisal cache %s: %s", cache_path, exc)
    return None


def save_ucl_reading_surprisal_cache(
    cache_path: Path,
    word_df: pd.DataFrame,
    surprisal: np.ndarray,
) -> None:
    """Atomically write a (story_id, zone, surprisal) CSV cache file."""
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    table = word_df[["story_id", "zone"]].copy()
    table["surprisal"] = np.asarray(surprisal, dtype=np.float64)

    tmp = cache_path.parent / f"{cache_path.stem}.part{os.getpid()}.csv"
    try:
        table.to_csv(tmp, index=False)
        last_err: BaseException | None = None
        for attempt in range(6):
            try:
                os.replace(tmp, cache_path)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.25 * (attempt + 1))
        if last_err is not None:
            raise last_err
        logger.info("UCL Reading surprisal cache written: %s (%d rows)", cache_path, len(table))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ZuCo Reading (behavioral, TRT) surprisal disk cache
# ---------------------------------------------------------------------------
# Same (story_id, zone, surprisal) schema as the UCL cache above, but keyed by
# ``task`` (SR/NR/TSR — distinct sentence sets) instead of a ``sentences``
# subset, with a distinct filename suffix so the two never collide.

def zuco_reading_surprisal_cache_path(
    cache_dir: str | Path,
    model_name: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    task: str,
) -> Path:
    """Return the CSV cache path for ZuCo Reading word-level surprisal."""
    safe_model = model_name.replace("/", "_")
    safe_rev = (model_revision or "main").replace("/", "_").replace("-", "_")
    tok = (tokenizer_revision or "default").replace("/", "_").replace("-", "_")
    filename = f"{safe_model}__{safe_rev}__tok{tok}__task{task}__zuco_reading_surprisal.csv"
    return Path(cache_dir) / filename


def load_zuco_reading_surprisal_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load a cached ZuCo Reading surprisal table; return None on miss or corruption."""
    if not cache_path.is_file():
        return None
    try:
        df = pd.read_csv(cache_path)
        if {"story_id", "zone", "surprisal"}.issubset(df.columns):
            logger.info("ZuCo Reading surprisal cache hit: %s (%d rows)", cache_path, len(df))
            return df
        logger.warning("ZuCo Reading surprisal cache missing required columns: %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to load ZuCo Reading surprisal cache %s: %s", cache_path, exc)
    return None


def save_zuco_reading_surprisal_cache(
    cache_path: Path,
    word_df: pd.DataFrame,
    surprisal: np.ndarray,
) -> None:
    """Atomically write a (story_id, zone, surprisal) CSV cache file."""
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    table = word_df[["story_id", "zone"]].copy()
    table["surprisal"] = np.asarray(surprisal, dtype=np.float64)

    tmp = cache_path.parent / f"{cache_path.stem}.part{os.getpid()}.csv"
    try:
        table.to_csv(tmp, index=False)
        last_err: BaseException | None = None
        for attempt in range(6):
            try:
                os.replace(tmp, cache_path)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.25 * (attempt + 1))
        if last_err is not None:
            raise last_err
        logger.info("ZuCo Reading surprisal cache written: %s (%d rows)", cache_path, len(table))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Provo Corpus surprisal disk cache
# ---------------------------------------------------------------------------
# Same (story_id, zone, surprisal) schema as the ZuCo/UCL caches above, but —
# unlike ZuCo's SR/NR/TSR, which are three *different* sentence sets and so get
# nested per-task cache dirs — Provo's SP/FP/GP measures all share the exact same
# 55 passages / word set (only the outcome duration column differs downstream in
# the LME). The cache key is therefore NOT parameterized by measure: one surprisal
# (and KenLM unigram) computation per (model, revision) serves all three measures,
# a 3x reduction in GPU/model-load cost versus naively mirroring ZuCo's nesting.

def provo_surprisal_cache_path(
    cache_dir: str | Path,
    model_name: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
) -> Path:
    """Return the CSV cache path for Provo word-level surprisal (measure-agnostic)."""
    safe_model = model_name.replace("/", "_")
    safe_rev = (model_revision or "main").replace("/", "_").replace("-", "_")
    tok = (tokenizer_revision or "default").replace("/", "_").replace("-", "_")
    filename = f"{safe_model}__{safe_rev}__tok{tok}__provo_surprisal.csv"
    return Path(cache_dir) / filename


def load_provo_surprisal_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load a cached Provo surprisal table; return None on miss or corruption."""
    if not cache_path.is_file():
        return None
    try:
        df = pd.read_csv(cache_path)
        if {"story_id", "zone", "surprisal"}.issubset(df.columns):
            logger.info("Provo surprisal cache hit: %s (%d rows)", cache_path, len(df))
            return df
        logger.warning("Provo surprisal cache missing required columns: %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to load Provo surprisal cache %s: %s", cache_path, exc)
    return None


def save_provo_surprisal_cache(
    cache_path: Path,
    word_df: pd.DataFrame,
    surprisal: np.ndarray,
) -> None:
    """Atomically write a (story_id, zone, surprisal) CSV cache file."""
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    table = word_df[["story_id", "zone"]].copy()
    table["surprisal"] = np.asarray(surprisal, dtype=np.float64)

    tmp = cache_path.parent / f"{cache_path.stem}.part{os.getpid()}.csv"
    try:
        table.to_csv(tmp, index=False)
        last_err: BaseException | None = None
        for attempt in range(6):
            try:
                os.replace(tmp, cache_path)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.25 * (attempt + 1))
        if last_err is not None:
            raise last_err
        logger.info("Provo surprisal cache written: %s (%d rows)", cache_path, len(table))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


# Same (story_id, zone, surprisal) schema as Provo above. GECO's FP/GP measures
# share one word set (56,441-word English novel), same rationale as Provo: one
# surprisal computation per (model, revision) serves both --measure sweeps.

def geco_surprisal_cache_path(
    cache_dir: str | Path,
    model_name: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
) -> Path:
    """Return the CSV cache path for GECO word-level surprisal (measure-agnostic)."""
    safe_model = model_name.replace("/", "_")
    safe_rev = (model_revision or "main").replace("/", "_").replace("-", "_")
    tok = (tokenizer_revision or "default").replace("/", "_").replace("-", "_")
    filename = f"{safe_model}__{safe_rev}__tok{tok}__geco_surprisal.csv"
    return Path(cache_dir) / filename


def load_geco_surprisal_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load a cached GECO surprisal table; return None on miss or corruption."""
    if not cache_path.is_file():
        return None
    try:
        df = pd.read_csv(cache_path)
        if {"story_id", "zone", "surprisal"}.issubset(df.columns):
            logger.info("GECO surprisal cache hit: %s (%d rows)", cache_path, len(df))
            return df
        logger.warning("GECO surprisal cache missing required columns: %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to load GECO surprisal cache %s: %s", cache_path, exc)
    return None


def save_geco_surprisal_cache(
    cache_path: Path,
    word_df: pd.DataFrame,
    surprisal: np.ndarray,
) -> None:
    """Atomically write a (story_id, zone, surprisal) CSV cache file."""
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    table = word_df[["story_id", "zone"]].copy()
    table["surprisal"] = np.asarray(surprisal, dtype=np.float64)

    tmp = cache_path.parent / f"{cache_path.stem}.part{os.getpid()}.csv"
    try:
        table.to_csv(tmp, index=False)
        last_err: BaseException | None = None
        for attempt in range(6):
            try:
                os.replace(tmp, cache_path)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.25 * (attempt + 1))
        if last_err is not None:
            raise last_err
        logger.info("GECO surprisal cache written: %s (%d rows)", cache_path, len(table))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


# Same (story_id, zone, surprisal) schema as Provo/GECO above. MECO's SP/FP/GP
# measures share one word set per language; the cache key is parameterized by
# ``language`` (only "en" is wired up by neuroai.datasets.meco today, but the
# 12-language joint table means a future non-English MecoDataset must not
# collide with the English cache).

def meco_surprisal_cache_path(
    cache_dir: str | Path,
    model_name: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    language: str = "en",
) -> Path:
    """Return the CSV cache path for MECO word-level surprisal (measure-agnostic)."""
    safe_model = model_name.replace("/", "_")
    safe_rev = (model_revision or "main").replace("/", "_").replace("-", "_")
    tok = (tokenizer_revision or "default").replace("/", "_").replace("-", "_")
    filename = f"{safe_model}__{safe_rev}__tok{tok}__meco_{language}_surprisal.csv"
    return Path(cache_dir) / filename


def load_meco_surprisal_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load a cached MECO surprisal table; return None on miss or corruption."""
    if not cache_path.is_file():
        return None
    try:
        df = pd.read_csv(cache_path)
        if {"story_id", "zone", "surprisal"}.issubset(df.columns):
            logger.info("MECO surprisal cache hit: %s (%d rows)", cache_path, len(df))
            return df
        logger.warning("MECO surprisal cache missing required columns: %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to load MECO surprisal cache %s: %s", cache_path, exc)
    return None


def save_meco_surprisal_cache(
    cache_path: Path,
    word_df: pd.DataFrame,
    surprisal: np.ndarray,
) -> None:
    """Atomically write a (story_id, zone, surprisal) CSV cache file."""
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    table = word_df[["story_id", "zone"]].copy()
    table["surprisal"] = np.asarray(surprisal, dtype=np.float64)

    tmp = cache_path.parent / f"{cache_path.stem}.part{os.getpid()}.csv"
    try:
        table.to_csv(tmp, index=False)
        last_err: BaseException | None = None
        for attempt in range(6):
            try:
                os.replace(tmp, cache_path)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.25 * (attempt + 1))
        if last_err is not None:
            raise last_err
        logger.info("MECO surprisal cache written: %s (%d rows)", cache_path, len(table))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


# Same (story_id, zone, surprisal) schema as Provo/GECO/MECO above. OneStop's
# SP/FP/GP measures share the exact same 330-passage word set (see
# neuroai.datasets.onestop.onestop_word_list), so one surprisal computation per
# (model, revision) serves all three --measure sweeps.

def onestop_surprisal_cache_path(
    cache_dir: str | Path,
    model_name: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
) -> Path:
    """Return the CSV cache path for OneStop word-level surprisal (measure-agnostic)."""
    safe_model = model_name.replace("/", "_")
    safe_rev = (model_revision or "main").replace("/", "_").replace("-", "_")
    tok = (tokenizer_revision or "default").replace("/", "_").replace("-", "_")
    filename = f"{safe_model}__{safe_rev}__tok{tok}__onestop_surprisal.csv"
    return Path(cache_dir) / filename


def load_onestop_surprisal_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load a cached OneStop surprisal table; return None on miss or corruption."""
    if not cache_path.is_file():
        return None
    try:
        df = pd.read_csv(cache_path)
        if {"story_id", "zone", "surprisal"}.issubset(df.columns):
            logger.info("OneStop surprisal cache hit: %s (%d rows)", cache_path, len(df))
            return df
        logger.warning("OneStop surprisal cache missing required columns: %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to load OneStop surprisal cache %s: %s", cache_path, exc)
    return None


def save_onestop_surprisal_cache(
    cache_path: Path,
    word_df: pd.DataFrame,
    surprisal: np.ndarray,
) -> None:
    """Atomically write a (story_id, zone, surprisal) CSV cache file."""
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    table = word_df[["story_id", "zone"]].copy()
    table["surprisal"] = np.asarray(surprisal, dtype=np.float64)

    tmp = cache_path.parent / f"{cache_path.stem}.part{os.getpid()}.csv"
    try:
        table.to_csv(tmp, index=False)
        last_err: BaseException | None = None
        for attempt in range(6):
            try:
                os.replace(tmp, cache_path)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.25 * (attempt + 1))
        if last_err is not None:
            raise last_err
        logger.info("OneStop surprisal cache written: %s (%d rows)", cache_path, len(table))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
