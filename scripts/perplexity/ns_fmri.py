#!/usr/bin/env python3
"""Perplexity of all registry models on the Natural Stories fMRI corpus (the 8 stories scanned).

Same recipe, CLI and output layout as ``scripts/perplexity/nieuwland.py``; only the corpus differs.
The corpus is each story's canonical word list from the fMRI X table (identical for all subjects), one
text per story. This is deliberately NOT the 10-story SPR corpus (its perplexities differ by ~47%).

    python scripts/perplexity/ns_fmri.py --all --device cuda --out_dir outputs_jeanzay/ns_fmri/perplexity
    python scripts/perplexity/ns_fmri.py --summarize --out_dir outputs_jeanzay/ns_fmri/perplexity
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO))

DATA_ROOT = Path(os.environ.get("NS_FMRI_DATA_ROOT", REPO / "data/naturalstories_fmri/osfstorage"))


def load_ns_fmri_text() -> list[str]:
    from neuroai.datasets.natural_stories_fmri import _load_x_table, load_canonical_words
    docs = list(dict.fromkeys(_load_x_table(str(DATA_ROOT), "LANG_MD")["docid"]))
    texts = []
    for d in docs:
        w = load_canonical_words(str(DATA_ROOT), d, "LANG_MD")
        texts.append(" ".join(w["word"].astype(str)))
    return texts


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("_nieuwland_ppl", Path(__file__).with_name("nieuwland.py"))
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    texts = load_ns_fmri_text()
    base.CORPUS_LABEL = "ns_fmri"
    base.SUMMARY_CSV = "ns_fmri_perplexity_summary.csv"
    base.load_nieuwland_text = lambda _root: texts
    if "--bids_root" not in sys.argv:
        sys.argv += ["--bids_root", str(REPO)]
    base.logger.info("ns_fmri corpus: %d stories, %d words", len(texts), sum(len(t.split()) for t in texts))
    base.main()
