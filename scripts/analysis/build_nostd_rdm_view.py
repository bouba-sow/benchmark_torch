#!/usr/bin/env python3
"""Scratch view for the neural RDM-regression summary with RAW embeddings and the (unchanged) surprisal results.

rdm_regression_neural_summary.py reads $SCRATCH/rdm_<name>_units/<track>/...; this builds a root where the embedding track
points to the *_nostd runs (RDM_EMBED_STANDARDIZE=0) and the surprisal track to the original runs, so that
    rdm_regression_neural_summary.py --scratch <view> --datasets <headline datasets>
yields the no-standardization summary. Nothing is copied (symlinks only).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

S = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
FLAT = ["pereira", "tuckute", "broderick", "megmasc", "fedorenko", "podcast_roi"]  # rdm_<name>_units/<track>/...
N400 = ["nieuwland", "szewczyk_f2007", "szewczyk_h2019"]                        # rdm_n400_units/<ds>/<track>/...


def link(dst: Path, src: Path):
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", type=Path, default=S / "nostd_view")
    v = ap.parse_args().view
    for n in FLAT:
        link(v / f"rdm_{n}_units/embedding", S / f"rdm_{n}_units_nostd/embedding")
        link(v / f"rdm_{n}_units/surprisal", S / f"rdm_{n}_units/surprisal")
    for ds in N400:
        link(v / f"rdm_n400_units/{ds}/embedding", S / f"rdm_n400_units_nostd/{ds}/embedding")
        link(v / f"rdm_n400_units/{ds}/surprisal", S / f"rdm_n400_units/{ds}/surprisal")
    print(f"view at {v}")


if __name__ == "__main__":
    main()
