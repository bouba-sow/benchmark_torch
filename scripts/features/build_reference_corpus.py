#!/usr/bin/env python3
"""Fixed English-Wikipedia reference corpus for embedding standardization (Timkey & van Schijndel 2021, arXiv 2109.04404).

They z-score every hidden-state dimension with the mean and std of a Wikipedia corpus (not of the evaluation items).
This builds that corpus once: N_ARTICLES random articles of the 2023-11-01 English Wikipedia dump (local copy on the
shared HuggingFace dataset area), each cut to its first WORDS_PER_ARTICLE words (cleaned of markup residue), seeded.

    .venv/bin/python scripts/features/build_reference_corpus.py [--out $SCRATCH/reference_corpus/wiki_en.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SRC = Path("/lustre/fsmisc/dataset/HuggingFace/wikipedia/wikipedia/20231101.en")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("SCRATCH", ".")) / "reference_corpus/wiki_en.json")
    ap.add_argument("--n-articles", type=int, default=400)
    ap.add_argument("--words-per-article", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    files = sorted(SRC.glob("train-*.parquet"))
    per_file = int(np.ceil(a.n_articles / len(files)))
    segs = []
    for f in files:
        pf = pq.ParquetFile(f)
        n = pf.metadata.num_rows
        pick = set(rng.choice(n, size=min(per_file * 4, n), replace=False).tolist())  # oversample, filter short ones
        got, row0 = 0, 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            for i, text in enumerate(t):
                if row0 + i in pick and got < per_file:
                    words = re.sub(r"\s+", " ", text).split(" ")
                    if len(words) >= a.words_per_article * 2:
                        segs.append(words[: a.words_per_article])
                        got += 1
            row0 += len(t)
    order = rng.permutation(len(segs))[: a.n_articles]
    segs = [segs[i] for i in order]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(source=str(SRC), seed=a.seed, segments=segs)))
    print(f"{len(segs)} segments, {sum(map(len, segs))} words -> {a.out}")


if __name__ == "__main__":
    main()
