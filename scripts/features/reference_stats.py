#!/usr/bin/env python3
"""Per-model mean/std of last-layer word states over the fixed Wikipedia reference corpus (Timkey & van Schijndel 2021).

Standardization in the paper: z = (x - mu) / sigma with mu, sigma computed over a Wikipedia corpus O (not over the
evaluation items). Here O = scripts/features/build_reference_corpus.py (400 articles x 64 words), word states extracted
with the same function the Provo embedding track uses (``contextual_word_embeddings_hf_fulltext``, last layer), one
forward pass per article segment.

Also stores the paper's rogue-dimension diagnostic: cc_prop[i] = CC_i / sum_j CC_j with
CC_i = mean over 200k random word pairs of u_i v_i / (|u||v|) (their eq. 3-4), plus the mean cosine (anisotropy).

Output: $SCRATCH/reference_stats/<family>/<model_dir>/<revision>.npz  (mu, sigma, cc_prop, anisotropy, n_words).
Array task i of N processes every N-th registry task (SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT).

    python scripts/features/reference_stats.py --family gpt2 --limit 1      # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

import numpy as np

from neuroai.analysis.benchmark_common import EXCLUDED_FAMILY_DIRS
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks

SCRATCH = Path(os.environ.get("SCRATCH", "."))
CORPUS = SCRATCH / "reference_corpus/wiki_en.json"
OUT = SCRATCH / "reference_stats"


def stats(states: np.ndarray, seed: int = 0, n_pairs: int = 200_000) -> dict:
    x = states.astype(np.float64)
    mu, sd = x.mean(0), x.std(0)
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(x), n_pairs), rng.integers(0, len(x), n_pairs)
    keep = i != j
    u, v = x[i[keep]], x[j[keep]]
    cc = (u * v / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))[:, None]).mean(0)
    return dict(mu=mu, sigma=sd, cc_prop=cc / cc.sum(), anisotropy=float(cc.sum()), n_words=len(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", action="append", help="registry family (default: all)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from neuroai.features.llm import contextual_word_embeddings_hf_fulltext
    segs = [[w for w in s if w.strip()] for s in json.loads(CORPUS.read_text())["segments"]]
    tasks = [(f, t) for f in (a.family or LPP_FAMILY_NAMES) if f not in EXCLUDED_FAMILY_DIRS
             for t in flatten_family_tasks(f)]
    k, n = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    tasks = tasks[k::n][: a.limit]
    for family, t in tasks:
        rev = t.revision or "main"
        out = OUT / family / t.model_dir_safe / f"{rev.replace('/', '_')}.npz"
        if out.is_file():
            continue
        t0 = time.time()
        try:
            hf_rev = None if rev == "main" else rev
            parts = [contextual_word_embeddings_hf_fulltext(
                s, t.load_model_id, revision=hf_rev, device=a.device, layer_index="last", cache_dir=None,
                tokenizer_no_revision=bool(hf_rev) and "pythia" in t.load_model_id.lower(),
                trust_remote_code=False,
                dtype="float32" if "gpt-bert" in t.load_model_id.lower() else None) for s in segs]
            states = np.concatenate(parts)
            if not np.isfinite(states).all():
                raise ValueError("non-finite states")
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, **stats(states))
            print(f"OK {family}/{t.model_id}@{rev} {states.shape} {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # keep going; failures are listed at the end of the log
            print(f"ERROR {family}/{t.model_id}@{rev}: {e!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
