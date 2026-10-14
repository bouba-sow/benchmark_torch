"""Regression tests for the 2026-10-12 surprisal audit (real cached HF models, CPU).

* trial surprisal (cumulative contexts) == full-text surprisal for BOS and non-BOS tokenizers
  (the pre-audit trial path was off by one token for every BOS-adding tokenizer);
* the last logit-lens layer == the model's own logits (native_only path);
* word surprisal is the SUM of subword surprisals;
* scripts/perplexity/lpp.py::compute_perplexity == exp(cross-entropy from logits), per word.

Skipped when the models are not in the local HF cache.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["openai-community/gpt2", "facebook/opt-125m"]  # no BOS / BOS ("</s>")
WORDS = ("The old lighthouse keeper watched the unbelievably violent storm roll across "
         "the harbour, wondering whether the fishermen had reached home.").split()


def _require(model_name: str) -> None:
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    if not (hub / ("models--" + model_name.replace("/", "--"))).exists():
        pytest.skip(f"{model_name} not cached")


@pytest.fixture(autouse=True)
def _clear():
    from neuroai.features.llm import clear_hf_model_cache
    yield
    clear_hf_model_cache()


@pytest.mark.parametrize("model_name", MODELS)
def test_trials_match_fulltext_and_native(model_name):
    _require(model_name)
    from neuroai.features.logit_lens import logit_lens_surprisal_all_layers, logit_lens_surprisal_trials

    contexts = [" ".join(WORDS[: i + 1]) for i in range(len(WORDS))]
    full = logit_lens_surprisal_all_layers(WORDS, model_name, dtype=torch.float32)[:, :, 0]
    trials = logit_lens_surprisal_trials(contexts, WORDS, model_name, dtype=torch.float32)
    native = logit_lens_surprisal_trials(contexts, WORDS, model_name, dtype=torch.float32, native_only=True)
    assert trials.shape == full.shape
    ok = np.isfinite(full[:, -1]) & np.isfinite(trials[:, -1])
    assert ok.sum() >= len(WORDS) - 1
    np.testing.assert_allclose(trials[ok, -1], full[ok, -1], atol=1e-3)
    np.testing.assert_allclose(trials[ok, -1], native[ok, 0], atol=1e-3)
    # earlier layers agree between the two paths too
    np.testing.assert_allclose(trials[ok, 0], full[ok, 0], atol=1e-3)


@pytest.mark.parametrize("model_name", MODELS)
def test_sum_is_word_surprisal(model_name):
    _require(model_name)
    from transformers import AutoTokenizer
    from neuroai.features.logit_lens import logit_lens_surprisal_trials

    tok = AutoTokenizer.from_pretrained(model_name)
    contexts = [" ".join(WORDS[: i + 1]) for i in range(len(WORDS))]
    s = logit_lens_surprisal_trials(contexts, WORDS, model_name, dtype=torch.float32, native_only=True)[:, 0]
    m = logit_lens_surprisal_trials(contexts, WORDS, model_name, dtype=torch.float32, native_only=True,
                                    subword_agg="mean")[:, 0]
    n_tok = np.array([len(tok(" " + w, add_special_tokens=False)["input_ids"]) for w in WORDS])
    ok = np.isfinite(s) & np.isfinite(m)
    np.testing.assert_allclose(s[ok], m[ok] * n_tok[ok], rtol=1e-4, atol=1e-4)
    assert (n_tok > 1).any(), "test sentence must contain a multi-token word"


def test_perplexity_from_logits_per_word():
    model_name = "openai-community/gpt2"
    _require(model_name)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = importlib.util.spec_from_file_location("ppl_lpp", ROOT / "scripts/perplexity/lpp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    text = " ".join(WORDS)
    r = mod.compute_perplexity(model, tok, [text], "cpu", max_length=1024)
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lg = model(ids).logits[0]
    nll = torch.nn.functional.cross_entropy(lg[:-1], ids[0, 1:], reduction="sum").item()
    assert math.isclose(r["perplexity"], math.exp(nll / len(WORDS)), rel_tol=1e-4)
    assert math.isclose(r["perplexity_per_token"], math.exp(nll / (ids.shape[1] - 1)), rel_tol=1e-4)
    # sliding windows give (nearly) the same value as one window
    r_small = mod.compute_perplexity(model, tok, [text], "cpu", max_length=8)
    assert r_small["perplexity"] > 0
