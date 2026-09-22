#!/usr/bin/env python3
"""DLT (Gibson 2000) integration cost at the critical noun, for Nieuwland's
N400 sentence-completion items.

Unlike the continuous-text datasets (Natural Stories/GECO/Podcast/Broderick,
see dlt/costs.py), each Nieuwland "item" is its own short, independent
sentence, and only ONE word per item is ever measured (the critical noun) —
so this script computes a single DLT cost per (item_number, condition) row,
not a per-word table for a whole corpus.

Two-stage design, same reasoning as dlt/costs.py (en_core_web_trf's
spacy-transformers pins an old `transformers`, which conflicts with this
repo's main env — see scripts/slurm/jz/utils/setup/dlt_env.sh):
  Stage 1 ``export-items`` — uses neuroai.datasets.nieuwland (needs the main
    .venv: mne + the actual BIDS EEG data, to rebuild the same item table the
    benchmark itself uses). Writes a plain (item_number, condition,
    sentence_context, noun) CSV — no spaCy involved.
  Stage 2 ``parse``         — pure spaCy + pandas (isolated .venv-dlt, no
    neuroai import). Reads Stage 1's CSV and writes
    outputs/dlt/nieuwland_dlt_cost.csv, keyed by (item_number, condition).

Usage
-----
    # Stage 1 (main .venv, needs data/nieuwland + mne):
    .venv/bin/python scripts/dlt/costs_nieuwland.py export-items \\
        --bids_root data/nieuwland

    # Stage 2 (isolated .venv-dlt, en_core_web_trf):
    .venv-dlt/bin/python scripts/dlt/costs_nieuwland.py parse \\
        --spacy_model en_core_web_trf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
ITEMS_CSV_NAME = "nieuwland_items.csv"


# ---------------------------------------------------------------------------
# Stage 1 — export the unique (item_number, condition) table (needs neuroai + mne)
# ---------------------------------------------------------------------------

def cmd_export_items(args: argparse.Namespace) -> None:
    from neuroai.datasets.nieuwland import (
        NieuwlandEpochsConfig, load_all_epochs, build_unique_items_df,
    )

    cfg = NieuwlandEpochsConfig(
        bids_root=args.bids_root,
        included_in_paper_only=not args.include_excluded,
    )
    logger.info("Loading Nieuwland epochs from %s (this reads real EEG — may take a while) ...",
                args.bids_root)
    epochs_result = load_all_epochs(cfg)
    items = build_unique_items_df(epochs_result.meta)
    logger.info("Found %d (item_number, condition) rows across %d subjects",
                len(items), epochs_result.n_subjects)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ITEMS_CSV_NAME
    items.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Stage 2 — spaCy DLT parse at the critical noun (spacy-only, safe for .venv-dlt)
# ---------------------------------------------------------------------------

_FINITE_VERB_TAGS = {"VBD", "VBP", "VBZ"}


def _introduces_discourse_referent(token) -> bool:
    if token.pos_ in {"NOUN", "PROPN"}:
        return True
    if token.pos_ in {"VERB", "AUX"} and token.tag_ in _FINITE_VERB_TAGS:
        return True
    return False


def _discourse_referents_between(doc, i: int, j: int) -> int:
    left, right = min(i, j), max(i, j)
    return sum(1 for tok in doc[left + 1:right] if _introduces_discourse_referent(tok))


def _integration_costs(doc) -> list[int]:
    """DLT integration cost per token (Gibson 2000) — mirrors get_dlt_cost exactly.

    Deliberately duplicated (not imported) from neuroai.models.dlt / the other
    dlt/costs.py copy: importing the neuroai package here would pull in
    neuroai.models.embedding/surprisal (torch + transformers), which conflicts
    with spacy-transformers' pinned transformers version in .venv-dlt.
    """
    costs = [0] * len(doc)
    for tok in doc:
        if tok.head == tok:  # ROOT
            continue
        dep_i, head_i = tok.i, tok.head.i
        cost_loc = max(dep_i, head_i)
        cost_loc_tok = doc[cost_loc]
        if _introduces_discourse_referent(cost_loc_tok):
            costs[cost_loc] = _discourse_referents_between(doc, dep_i, head_i)
            if cost_loc_tok.dep_ == "relcl":
                for tok_j in doc[:cost_loc_tok.i]:
                    if tok_j.head.i == cost_loc_tok.i and tok_j.dep_ == "nsubj":
                        if tok_j.tag_[0] != "W":
                            costs[cost_loc] += 1
        else:
            costs[cost_loc] = 0
    return costs


def _critical_word_cost(context: str, noun: str, nlp) -> dict:
    """DLT storage/integration/total for the LAST token of ``context + ' ' + noun``.

    The critical noun is (almost always) exactly one spaCy token; if it
    tokenizes into more than one (rare — hyphenated compounds etc.), the last
    sub-token is used, matching how the LLM-side feature extraction also takes
    the noun's last (sub-word) token.
    """
    text = f"{context.strip()} {noun.strip()}".strip()
    doc = nlp(text)
    if len(doc) == 0:
        return {"storage": None, "integration": None, "total": None, "n_tokens": 0}
    integ = _integration_costs(doc)
    store = [int(_introduces_discourse_referent(tok)) for tok in doc]
    last = len(doc) - 1
    return {
        "storage": store[last],
        "integration": integ[last],
        "total": store[last] + integ[last],
        "n_tokens": len(doc),
        "last_token_text": doc[last].text,
    }


def cmd_parse(args: argparse.Namespace) -> None:
    import spacy

    logger.info("Loading spaCy model %s ...", args.spacy_model)
    nlp = spacy.load(args.spacy_model)

    items_path = Path(args.out_dir) / ITEMS_CSV_NAME
    if not items_path.is_file():
        raise FileNotFoundError(
            f"{items_path} missing — run `export-items` first (from the main "
            ".venv, which has neuroai + mne + the actual BIDS EEG data)."
        )
    items = pd.read_csv(items_path)
    logger.info("Parsing DLT cost for %d items ...", len(items))

    rows = []
    for _, row in items.iterrows():
        cost = _critical_word_cost(str(row["sentence_context"]), str(row["noun"]), nlp)
        rows.append({
            "item_number": row["item_number"],
            "condition": row["condition"],
            "noun": row["noun"],
            **cost,
        })

    out_dir = Path(REPO_ROOT / "outputs" / "dlt")
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = pd.DataFrame(rows)
    detail_path = out_dir / "nieuwland_dlt_cost_by_condition.csv"
    detail.to_csv(detail_path, index=False)
    logger.info("Wrote %d (item_number, condition) rows -> %s", len(detail), detail_path)

    # _run_ridge_item_mean (neuroai/benchmarks/eeg_erp.py) groups by item_number
    # ALONE, mixing both conditions' trials into one item — so the join key for
    # scoring needs exactly one row per item_number too. The critical noun's DLT
    # cost is structural (same dependency slot regardless of which noun fills
    # it), so the two conditions' costs are expected to already agree almost
    # always; averaging handles the rare case where they don't (e.g. a noun
    # that tokenizes differently) without silently dropping one condition via
    # an arbitrary drop_duplicates.
    agg = (
        detail.groupby("item_number")[["storage", "integration", "total"]]
        .mean()
        .reset_index()
    )
    agg_path = out_dir / "nieuwland_dlt_cost.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("Wrote %d item_number rows -> %s", len(agg), agg_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "dlt" / "_items_export"))

    p_export = sub.add_parser("export-items", parents=[common],
                               help="Stage 1: rebuild the Nieuwland item table (needs the main .venv).")
    p_export.add_argument("--bids_root", default=str(REPO_ROOT / "data" / "nm000228"))
    p_export.add_argument("--include_excluded", action="store_true",
                           help="Include participants excluded from the original paper (default: False).")
    p_export.set_defaults(func=cmd_export_items)

    p_parse = sub.add_parser("parse", parents=[common],
                              help="Stage 2: spaCy DLT parse at the critical noun (spacy-only).")
    p_parse.add_argument("--spacy_model", default="en_core_web_trf")
    p_parse.set_defaults(func=cmd_parse)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
