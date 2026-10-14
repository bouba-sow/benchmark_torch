#!/usr/bin/env python3
"""Perplexity of all registry models on one Szewczyk & Federmeier study's stimulus sentences.

Same recipe, CLI and output layout as ``scripts/perplexity/nieuwland.py`` (sliding-window corpus
perplexity, resumable, ``--all`` / ``--family`` / ``--summarize`` / ``--architecture cbr_rnn``); only the
corpus differs. The corpus is the study's unique full sentences (context + critical word) from
``szewczyk_2022.tsv`` (one sentence per item). Pick the study with ``--study {f2007,h2019}``.

    python scripts/perplexity/szewczyk.py --study f2007 --all --device cuda \
        --out_dir outputs_jeanzay/szewczyk_f2007/perplexity
    python scripts/perplexity/szewczyk.py --study f2007 --summarize --out_dir outputs_jeanzay/szewczyk_f2007/perplexity
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO))

TSV = REPO / "data/n400_paper_formula/osfstorage/datasets/szewczyk_2022.tsv"


def _take_study() -> str:
    if "--study" not in sys.argv:
        sys.exit("--study {f2007,h2019} is required")
    i = sys.argv.index("--study")
    study = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return study


def load_szewczyk_text(study: str) -> list[str]:
    from neuroai.datasets.szewczyk_paper import load_study, study_items
    items = study_items(load_study(TSV, study))
    sents = [s.strip() for s in items["FullText"].astype(str)]
    return list(dict.fromkeys(s for s in sents if s))


if __name__ == "__main__":
    study = _take_study()
    spec = importlib.util.spec_from_file_location("_nieuwland_ppl", Path(__file__).with_name("nieuwland.py"))
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    texts = load_szewczyk_text(study)
    base.CORPUS_LABEL = f"szewczyk_{study}"
    base.SUMMARY_CSV = f"szewczyk_{study}_perplexity_summary.csv"
    base.load_nieuwland_text = lambda _root: texts
    if "--bids_root" not in sys.argv:
        sys.argv += ["--bids_root", str(REPO)]          # unused by this corpus; the base CLI only checks it exists
    base.logger.info("Szewczyk %s corpus: %d unique sentences", study, len(texts))
    base.main()
