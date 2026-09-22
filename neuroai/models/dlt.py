"""Dependency Locality Theory (Gibson 2000) word-by-word integration cost.

Standalone, CPU-only, spaCy-based — mirrors https://github.com/wtimkey/get_dlt_cost
(Timkey's ``get_dlt_costs.py``) exactly, so results are directly comparable to
that reference tool. Not a NeuralSet extractor; independent of any LLM.

Usage
-----
    costs = dlt_costs_for_text(words, spacy_model="en_core_web_trf")

``words`` is a flat, ordered token/word list for one running text (a story, a
transcript). Sentence boundaries are inferred by spaCy from the joined text,
so pass whole-corpus word lists (not pre-split sentences) when the corpus
already reads as continuous prose.
"""

from __future__ import annotations

import typing as tp

import pandas as pd

_FINITE_VERB_TAGS = {"VBD", "VBP", "VBZ"}


def introduces_discourse_referent(token: tp.Any) -> bool:
    """A token introduces a discourse referent: nouns, proper nouns, finite verbs/aux."""
    if token.pos_ in {"NOUN", "PROPN"}:
        return True
    if token.pos_ in {"VERB", "AUX"} and token.tag_ in _FINITE_VERB_TAGS:
        return True
    return False


def _discourse_referents_between(doc: tp.Any, i: int, j: int) -> int:
    """Count discourse-referent tokens strictly between token indices ``i`` and ``j``."""
    left, right = min(i, j), max(i, j)
    return sum(1 for tok in doc[left + 1:right] if introduces_discourse_referent(tok))


def integration_costs(doc: tp.Any) -> list[int]:
    """Per-token DLT integration cost (Gibson 2000), one int per token in ``doc``."""
    costs = [0] * len(doc)
    for tok in doc:
        if tok.head == tok:  # ROOT
            continue
        dep_i, head_i = tok.i, tok.head.i
        cost_loc = max(dep_i, head_i)
        cost_loc_tok = doc[cost_loc]
        if introduces_discourse_referent(cost_loc_tok):
            costs[cost_loc] = _discourse_referents_between(doc, dep_i, head_i)
            # Implicit gapped relative-clause objects aren't overt in spaCy's
            # parse — approximate by crediting an extra referent when the RC
            # has an overt (non-wh) subject. Matches get_dlt_cost's handling.
            if cost_loc_tok.dep_ == "relcl":
                for tok_j in doc[:cost_loc_tok.i]:
                    if tok_j.head.i == cost_loc_tok.i and tok_j.dep_ == "nsubj":
                        if tok_j.tag_[0] != "W":
                            costs[cost_loc] += 1
        else:
            costs[cost_loc] = 0
    return costs


def dlt_costs_for_doc(doc: tp.Any) -> pd.DataFrame:
    """Per-token storage / integration / total DLT cost for one spaCy ``Doc``."""
    integ = integration_costs(doc)
    store = [int(introduces_discourse_referent(tok)) for tok in doc]
    total = [integ[i] + store[i] for i in range(len(doc))]
    return pd.DataFrame({
        "token_idx": [tok.i for tok in doc],
        "word": [tok.text for tok in doc],
        "head_idx": [tok.head.i for tok in doc],
        "head": [tok.head.text for tok in doc],
        "dep": [tok.dep_ for tok in doc],
        "storage": store,
        "integration": integ,
        "total": total,
    })


def dlt_costs_for_text(
    words: list[str],
    *,
    spacy_model: str = "en_core_web_trf",
    nlp: tp.Any = None,
) -> pd.DataFrame:
    """DLT cost for one running text given as an ordered word list.

    Joins ``words`` with single spaces and re-parses with spaCy (sentence
    splitting happens inside spaCy, same as the reference tool) — so the
    returned frame has one row per **spaCy token**, not necessarily one row
    per input word (spaCy may split/merge on punctuation). Use ``word_index``
    (position in the original ``words`` list, via whitespace-aligned offsets)
    to join back onto corpus word-level tables when an exact 1:1 mapping is
    required upstream.
    """
    if nlp is None:
        import spacy

        nlp = spacy.load(spacy_model)
    text = " ".join(words)
    doc = nlp(text)
    return dlt_costs_for_doc(doc)


def dlt_costs_for_sentences(
    sentences: list[str],
    *,
    spacy_model: str = "en_core_web_trf",
    nlp: tp.Any = None,
) -> pd.DataFrame:
    """DLT cost computed sentence-by-sentence (token indices reset per sentence).

    Prefer this over :func:`dlt_costs_for_text` when the corpus is already
    segmented into sentences/trials (e.g. GECO, Natural Stories) — dependency
    distances should not cross sentence boundaries.
    """
    if nlp is None:
        import spacy

        nlp = spacy.load(spacy_model)
    frames = []
    for si, sent in enumerate(sentences):
        doc = nlp(sent)
        df = dlt_costs_for_doc(doc)
        df.insert(0, "sent_idx", si)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
