"""Token-level surprisal and entropy helpers (standalone, not a NeuralSet extractor).

These functions work on plain text using HuggingFace causal LMs, and are
called from within the pipeline rather than registered as NeuralSet extractors
(which require a more complex pydantic setup).
"""

from __future__ import annotations

import typing as tp

import numpy as np
import torch


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

    toks = tokenizer(full, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model(**{k: v.to(device) for k, v in toks.items()})
    logits = out.logits[0].cpu().float()
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


def batch_surprisal_one_model(
    model_name: str,
    contexts: list[str],
    targets: list[str],
    *,
    output: tp.Literal["surprisal", "entropy", "both"] = "surprisal",
    device: str = "auto",
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
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
    rows: list[np.ndarray] = []
    try:
        for ctx, tgt in zip(contexts, targets):
            rows.append(
                _surprisal_one_pair(
                    model, tokenizer, device_t, ctx, tgt, output=output
                )
            )
    finally:
        del model
        if device_t.type == "cuda":
            torch.cuda.empty_cache()

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
) -> np.ndarray:
    """Mean of two per-trial surprisal scalars with a **single** model load.

    For each index ``i``, computes
    ``(_surprisal_one_pair(..., A_i) + _surprisal_one_pair(..., B_i)) / 2``,
    where each pair returns the usual mean-token surprisal for that target.

    RT benchmarks use this for e.g. mean(surprisal(penultimate word),
    surprisal(last word)) so the summary mixes agreement on the late verb
    region with the following continuation.
    """
    from transformers import AutoModelForCausalLM

    if output != "surprisal":
        raise ValueError("batch_surprisal_mean_two_words only supports output='surprisal'")
    if not (len(contexts_a) == len(targets_a) == len(contexts_b) == len(targets_b)):
        raise ValueError("context/target lists must have equal length")

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
        del model
        if device_t.type == "cuda":
            torch.cuda.empty_cache()

    return np.asarray(vals, dtype=np.float32).reshape(-1, 1)


def batch_surprisal(
    model_name: str,
    contexts: list[str],
    targets: list[str],
    *,
    output: tp.Literal["surprisal", "entropy", "both"] = "surprisal",
    device: str = "auto",
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
) -> np.ndarray:
    """Compute surprisal for a list of (context, target) pairs (one model load).

    Returns
    -------
    np.ndarray of shape ``(n, 1)`` or ``(n, 2)``.
    """
    return batch_surprisal_one_model(
        model_name,
        contexts,
        targets,
        output=output,
        device=device,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    )
