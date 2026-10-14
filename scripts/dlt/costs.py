#!/usr/bin/env python3
"""Compute per-word DLT (Gibson 2000) integration cost for the four datasets
selected for the difficulty-split analysis: Natural Stories SPR, GECO
(behavioral) and Podcast, Broderick (neural).

Two-stage design (Jean Zay: en_core_web_trf conflicts with this repo's
`transformers` pin — see scripts/slurm/jz/utils/setup/dlt_env.sh)
--------------------------------------------------------------------
  Stage 1 ``export-words``  — uses the neuroai dataset loaders (needs THIS
    repo's main .venv). Writes a plain word-list CSV per dataset to
    ``<out_dir>/_words_export/`` — no spaCy involved.
  Stage 2 ``parse``         — pure spaCy + pandas (needs ONLY spacy — run
    from the isolated ``.venv-dlt``, no neuroai import). Reads the word-list
    CSVs from Stage 1 and writes the final ``{dataset}_dlt_cost.csv``.

Single-machine convenience: ``all`` runs both stages in one process (fine
locally with en_core_web_sm, where there's no dependency conflict).

Usage
-----
    # Local, one shot (en_core_web_sm):
    .venv/bin/python scripts/dlt/costs.py all --spacy_model en_core_web_sm

    # Jean Zay, split (en_core_web_trf):
    .venv/bin/python scripts/dlt/costs.py export-words
    .venv-dlt/bin/python scripts/dlt/costs.py parse --spacy_model en_core_web_trf
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Only needed for `export-words` (imports neuroai.datasets.*); `parse` has no
# neuroai dependency and works fine without this (e.g. under .venv-dlt).
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

# Each dataset's grouping unit: the column(s) that define one contiguous span
# of running text (dependencies must not be parsed across these boundaries).
GROUP_COLS = {
    "naturalstories": ["story_id"],
    "geco": ["story_id"],
    "podcast": [],          # single continuous transcript
    "broderick": ["run"],
    "provo": ["story_id"],
    "ucl_reading": ["story_id"],
    "meco": ["story_id"],
    "zuco_reading_sr": ["story_id"],
    "zuco_reading_nr": ["story_id"],
    "zuco_reading_tsr": ["story_id"],
    "lpp": ["section"],
    "fedorenko2016": ["sentence_id"],
    "megmasc": ["story"],
    "onestop": ["story_id"],
}


# ---------------------------------------------------------------------------
# Stage 1 — export plain word lists (needs neuroai + the main .venv)
# ---------------------------------------------------------------------------

def _export_naturalstories(root: Path) -> pd.DataFrame:
    from neuroai.datasets.natural_stories import NaturalStoriesDataset

    ds = NaturalStoriesDataset(root)
    word_df = ds.load_words().sort_values(["story_id", "zone"]).reset_index(drop=True)
    word_df["position"] = word_df.groupby("story_id").cumcount()
    return word_df[["story_id", "zone", "position", "word"]]


def _export_geco(root: Path) -> pd.DataFrame:
    from neuroai.datasets.geco import GecoDataset

    ds = GecoDataset(root, measure="FP", aggregate=True)
    word_df = ds.load_words().sort_values(["story_id", "zone"]).reset_index(drop=True)
    word_df["position"] = word_df.groupby("story_id").cumcount()
    return word_df[["story_id", "zone", "position", "word"]]


def _export_podcast(bids_root: Path) -> pd.DataFrame:
    from neuroai.datasets.podcast import load_transcript

    words_df = load_transcript(bids_root)
    words_df["position"] = range(len(words_df))
    return words_df[["word_idx", "position", "word"]]


def _export_broderick(bids_root: Path, n_runs: int = 20) -> pd.DataFrame:
    from neuroai.datasets.broderick import load_all_words

    words_df = load_all_words(bids_root, n_runs=n_runs).sort_values(["run", "word_idx"])
    words_df["position"] = words_df.groupby("run").cumcount()
    return words_df[["run", "word_idx", "position", "word"]].reset_index(drop=True)


def _export_provo(root: Path) -> pd.DataFrame:
    from neuroai.datasets.provo import ProvoDataset

    ds = ProvoDataset(root, measure="FP", aggregate=True)
    word_df = ds.load_words().sort_values(["story_id", "zone"]).reset_index(drop=True)
    word_df["position"] = word_df.groupby("story_id").cumcount()
    return word_df[["story_id", "zone", "position", "word"]]


def _export_ucl_reading(root: Path) -> pd.DataFrame:
    from neuroai.datasets.ucl_reading import UCLReadingDataset

    ds = UCLReadingDataset(root, aggregate=True)
    word_df = ds.load_words().sort_values(["story_id", "zone"]).reset_index(drop=True)
    word_df["position"] = word_df.groupby("story_id").cumcount()
    return word_df[["story_id", "zone", "position", "word"]]


def _export_meco(root: Path) -> pd.DataFrame:
    from neuroai.datasets.meco import MecoDataset

    ds = MecoDataset(root, measure="FP", language="en", aggregate=True)
    word_df = ds.load_words().sort_values(["story_id", "zone"]).reset_index(drop=True)
    word_df["position"] = word_df.groupby("story_id").cumcount()
    return word_df[["story_id", "zone", "position", "word"]]


def _export_zuco_reading(mat_root: Path, task: str) -> pd.DataFrame:
    from neuroai.datasets.zuco_reading import ZuCoReadingDataset

    ds = ZuCoReadingDataset(mat_root, task=task, aggregate=True)
    word_df = ds.load_words().sort_values(["story_id", "zone"]).reset_index(drop=True)
    word_df["position"] = word_df.groupby("story_id").cumcount()
    return word_df[["story_id", "zone", "position", "word"]]


def _export_fedorenko2016(root: Path) -> pd.DataFrame:
    from neuroai.datasets.fedorenko2016 import Fedorenko2016Dataset

    ds = Fedorenko2016Dataset(root)
    stimuli = ds.load_stimuli().sort_values(["sentence_id", "word_num"]).reset_index(drop=True)
    stimuli["position"] = stimuli["word_num"]
    return stimuli[["stimulus_id", "sentence_id", "word_num", "position", "word"]]


def _export_lpp(bids_root: Path, lang: str = "en") -> pd.DataFrame:
    from neuroai.datasets.le_petit_prince import (
        LPP_N_ANNOTATION_SECTIONS, load_word_information_table, words_for_section,
    )

    words_full = load_word_information_table(bids_root, lang, apply_lbg_fixes=False)
    frames = []
    for section in range(1, LPP_N_ANNOTATION_SECTIONS + 1):
        sec_df = words_for_section(words_full, section)
        word = sec_df["word"]
        keep = word.map(lambda w: isinstance(w, str) and w.strip() not in ("", " "))
        sec_df = sec_df.loc[keep].copy()
        sec_df["section"] = section
        sec_df["position"] = range(len(sec_df))
        frames.append(sec_df[["section", "position", "word"]])
    return pd.concat(frames, ignore_index=True)


def _export_onestop(root: Path) -> pd.DataFrame:
    # story_id encodes batch/article/paragraph/difficulty, so each passage (incl.
    # each difficulty variant) is its own running text. Canonical word list over
    # all subjects (same one the surprisal sweep scores).
    from neuroai.datasets.onestop import OnestopDataset

    ds = OnestopDataset(root, measure="FP", aggregate=True)
    word_df = ds.load_words().sort_values(["story_id", "zone"]).reset_index(drop=True)
    word_df["position"] = word_df.groupby("story_id").cumcount()
    return word_df[["story_id", "zone", "position", "word"]]


def _export_megmasc(bids_root: Path) -> pd.DataFrame:
    # Same canonical list as megmasc_lme._load_all_words: global word_idx running
    # index over stories (needed to join with megmasc_lme_per_word.csv).
    from neuroai.benchmarks.megmasc_lme import _load_all_words

    words_df = _load_all_words(str(bids_root), None).sort_values("word_idx")
    words_df["position"] = words_df.groupby("story").cumcount()
    return words_df[["story", "word_idx", "position", "word"]].reset_index(drop=True)


_MECO_ROOT = (
    REPO_ROOT / "data" / "meco" / "osfstorage" / "release 2.0" / "version 2.1"
    / "wave 1" / "primary data" / "eye tracking data"
)
_ZUCO_ROOT = REPO_ROOT / "data" / "q3zws" / "osfstorage"

EXPORTERS = {
    "naturalstories": (_export_naturalstories, REPO_ROOT / "data" / "naturalstories"),
    "geco": (_export_geco, REPO_ROOT / "data" / "geco"),
    "podcast": (_export_podcast, REPO_ROOT / "data" / "Podcast"),
    "broderick": (_export_broderick, REPO_ROOT / "data" / "Broderick"),
    "provo": (_export_provo, REPO_ROOT / "data" / "Provo" / "osfstorage"),
    "ucl_reading": (_export_ucl_reading, REPO_ROOT / "data" / "UCL_reading"),
    "meco": (_export_meco, _MECO_ROOT),
    "zuco_reading_sr": (lambda root: _export_zuco_reading(root, "SR"), _ZUCO_ROOT),
    "zuco_reading_nr": (lambda root: _export_zuco_reading(root, "NR"), _ZUCO_ROOT),
    "zuco_reading_tsr": (lambda root: _export_zuco_reading(root, "TSR"), _ZUCO_ROOT),
    # English only: lpp_lme.py's benchmark (the primary LPP consumer of these
    # costs, see module docstring) uses lang="en" by default, and the spaCy
    # DLT stage assumes English-style dependency parsing (en_core_web_trf) —
    # no French spaCy support is wired up here.
    "lpp": (_export_lpp, REPO_ROOT / "data" / "Lepetitprince_fMRI"),
    "fedorenko2016": (_export_fedorenko2016, REPO_ROOT / "data" / "fedorenko2016"),
    "onestop": (_export_onestop, REPO_ROOT / "data" / "onestop"),
    "megmasc": (_export_megmasc, Path(os.path.expandvars("${MEGMASC_BIDS_ROOT}"))),
}


def cmd_export_words(args: argparse.Namespace) -> None:
    words_dir = Path(args.out_dir) / "_words_export"
    words_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        fn, root = EXPORTERS[name]
        logger.info("[%s] exporting word list from %s", name, root)
        df = fn(root)
        out_path = words_dir / f"{name}_words.csv"
        df.to_csv(out_path, index=False)
        logger.info("[%s] wrote %d words -> %s", name, len(df), out_path)


# ---------------------------------------------------------------------------
# Stage 2 — spaCy DLT parse (needs ONLY spacy — safe for an isolated venv)
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

    Deliberately duplicated (not imported) from neuroai.models.dlt: importing
    the neuroai package here would pull in neuroai.models.embedding/surprisal
    (torch + transformers), which conflicts with spacy-transformers' pinned
    transformers version in the isolated .venv-dlt this stage is meant to run in.
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


def _dlt_costs_for_text(words: list[str], nlp) -> pd.DataFrame:
    """One row per INPUT word (len == len(words)).

    spaCy re-tokenizes the joined text (contractions, trailing punctuation,
    words with internal spaces), so token count != word count and a positional
    ``iloc[:len(words)]`` shifted every cost after the first split token. Tokens
    are mapped back to words by character offset in the single-space-joined text
    and their storage / integration / total are summed (punctuation and clitics
    cost 0, so this is the content token's cost). ``token_idx``/``head*``/``dep``
    describe the word's highest-cost token (first token on ties); ``head_idx`` is
    a spaCy token index, not a word index. ``n_tokens`` is the number of spaCy
    tokens the word was split into.
    """
    from bisect import bisect_right

    doc = nlp(" ".join(words))
    integ = _integration_costs(doc)
    store = [int(_introduces_discourse_referent(tok)) for tok in doc]

    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    by_word: list[list[int]] = [[] for _ in words]
    for tok in doc:
        wi = bisect_right(starts, tok.idx) - 1
        if 0 <= wi < len(words):
            by_word[wi].append(tok.i)

    rows = []
    for w, toks in zip(words, by_word):
        if not toks:
            rows.append((-1, w, -1, "", "", 0, 0, 0, 0))
            continue
        tot = [integ[i] + store[i] for i in toks]
        main = toks[max(range(len(toks)), key=lambda k: (tot[k], -k))]
        rows.append((
            main, w, doc[main].head.i, doc[main].head.text, doc[main].dep_,
            sum(store[i] for i in toks), sum(integ[i] for i in toks), sum(tot), len(toks),
        ))
    return pd.DataFrame(
        rows,
        columns=["token_idx", "word", "head_idx", "head", "dep", "storage", "integration", "total", "n_tokens"],
    )


def _parse_dataset(nlp, words_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    frames = []
    if group_cols:
        groups = [
            (tuple(key if isinstance(key, tuple) else (key,)), g.sort_values("position"))
            for key, g in words_df.groupby(group_cols, sort=True)
        ]
    else:
        groups = [((), words_df.sort_values("position"))]

    for key, g in groups:
        words = g["word"].astype(str).tolist()
        if not words:
            continue
        cost_df = _dlt_costs_for_text(words, nlp)
        g = g.reset_index(drop=True)
        merged = pd.concat([g.drop(columns=["word"]), cost_df], axis=1)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cmd_parse(args: argparse.Namespace) -> None:
    import spacy

    logger.info("Loading spaCy model %s ...", args.spacy_model)
    nlp = spacy.load(args.spacy_model)

    words_dir = Path(args.out_dir) / "_words_export"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        words_path = words_dir / f"{name}_words.csv"
        if not words_path.is_file():
            raise FileNotFoundError(
                f"{words_path} missing — run `dlt/costs.py export-words` first "
                "(from the main .venv, which has the neuroai dataset loaders)."
            )
        logger.info("[%s] parsing DLT cost from %s", name, words_path)
        # keep_default_na=False: a literal word string like "None" (a genuine
        # UCL Reading token, not a missing value) would otherwise round-trip
        # through pandas' default NA-string detection back into an actual
        # float NaN, crashing " ".join(words) below with a TypeError.
        words_df = pd.read_csv(words_path, keep_default_na=False, na_values=[])
        df = _parse_dataset(nlp, words_df, GROUP_COLS[name])
        out_path = out_dir / f"{name}_dlt_cost.csv"
        df.to_csv(out_path, index=False)
        logger.info("[%s] wrote %d rows -> %s", name, len(df), out_path)


def cmd_all(args: argparse.Namespace) -> None:
    """Single-machine convenience: export + parse in one process (no venv split)."""
    cmd_export_words(args)
    cmd_parse(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--datasets", nargs="+", default=list(EXPORTERS.keys()),
                         choices=list(EXPORTERS.keys()))
    common.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "dlt"))

    p_export = sub.add_parser("export-words", parents=[common],
                               help="Stage 1: write plain word lists (needs the main .venv).")
    p_export.set_defaults(func=cmd_export_words)

    p_parse = sub.add_parser("parse", parents=[common],
                              help="Stage 2: spaCy DLT parse (spacy-only — safe for an isolated venv).")
    p_parse.add_argument("--spacy_model", default="en_core_web_trf",
                          help="en_core_web_trf matches get_dlt_cost's own default (best accuracy); "
                               "en_core_web_sm is faster/CPU-cheaper but less accurate.")
    p_parse.set_defaults(func=cmd_parse)

    p_all = sub.add_parser("all", parents=[common],
                            help="Both stages in one process (fine locally with en_core_web_sm).")
    p_all.add_argument("--spacy_model", default="en_core_web_sm")
    p_all.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
