"""Shared spaCy-only DLT (Gibson 2000) critical-word cost logic.

Pulled out of ``dlt/costs_nieuwland.py`` so other N400 critical-word
datasets don't need another copy of the same ~80 lines of dependency-parsing logic. Deliberately duplicated from
``neuroai.models.dlt`` / ``dlt/costs.py`` rather than imported —
importing the neuroai package here would pull in
neuroai.models.embedding/surprisal (torch + transformers), which conflicts
with spacy-transformers' pinned transformers version in the isolated
``.venv-dlt`` this module is meant to run in. No neuroai import anywhere in
this file — that's the whole point.
"""

from __future__ import annotations

_FINITE_VERB_TAGS = {"VBD", "VBP", "VBZ"}


def introduces_discourse_referent(token) -> bool:
    if token.pos_ in {"NOUN", "PROPN"}:
        return True
    if token.pos_ in {"VERB", "AUX"} and token.tag_ in _FINITE_VERB_TAGS:
        return True
    return False


def _discourse_referents_between(doc, i: int, j: int) -> int:
    left, right = min(i, j), max(i, j)
    return sum(1 for tok in doc[left + 1:right] if introduces_discourse_referent(tok))


def integration_costs(doc) -> list[int]:
    """DLT integration cost per token (Gibson 2000) — mirrors get_dlt_cost exactly."""
    costs = [0] * len(doc)
    for tok in doc:
        if tok.head == tok:  # ROOT
            continue
        dep_i, head_i = tok.i, tok.head.i
        cost_loc = max(dep_i, head_i)
        cost_loc_tok = doc[cost_loc]
        if introduces_discourse_referent(cost_loc_tok):
            costs[cost_loc] = _discourse_referents_between(doc, dep_i, head_i)
            if cost_loc_tok.dep_ == "relcl":
                for tok_j in doc[:cost_loc_tok.i]:
                    if tok_j.head.i == cost_loc_tok.i and tok_j.dep_ == "nsubj":
                        if tok_j.tag_[0] != "W":
                            costs[cost_loc] += 1
        else:
            costs[cost_loc] = 0
    return costs


def critical_word_cost(context: str, critical_word: str, nlp) -> dict:
    """DLT storage/integration/total for the LAST token of ``context + ' ' + critical_word``.

    The critical word is (almost always) exactly one spaCy token; if it
    tokenizes into more than one (rare — hyphenated compounds etc.), the last
    sub-token is used, matching how the LLM-side feature extraction also takes
    the critical word's last (sub-word) token.
    """
    text = f"{context.strip()} {critical_word.strip()}".strip()
    doc = nlp(text)
    if len(doc) == 0:
        return {"storage": None, "integration": None, "total": None, "n_tokens": 0}
    integ = integration_costs(doc)
    store = [int(introduces_discourse_referent(tok)) for tok in doc]
    last = len(doc) - 1
    return {
        "storage": store[last],
        "integration": integ[last],
        "total": store[last] + integ[last],
        "n_tokens": len(doc),
        "last_token_text": doc[last].text,
    }
