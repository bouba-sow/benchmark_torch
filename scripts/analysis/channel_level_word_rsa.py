#!/usr/bin/env python3
"""Embedding RSA for Broderick EEG / MEG-MASC with full per-channel neural patterns.

The default RSA built each word's neural pattern from 4 (Broderick) or 6 (MEG-MASC) regional means
taken from the LME tables. Here the pattern is the per-channel window mean from the cached word-locked
epochs (Broderick: 114 EEG channels, 0.3-0.5 s; MEG-MASC: 208 sensors, 0.25-0.45 s, same windows as the
LME outcome), averaged over repeated presentations of the same word. Embedding states are z-scored per
feature. Everything else (sampling of pairs, scoring, summary) is word_neural_embedding_rsa.py.

    .venv/bin/python scripts/analysis/channel_level_word_rsa.py broderick --out-root <dir>
"""
from __future__ import annotations
import argparse, glob, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.analysis import word_neural_embedding_rsa as W

EPOCHS = {"broderick": ("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/broderick_epochs/sub-*.npz", 0.30, 0.50),
          "megmasc": ("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/megmasc_epochs/sub-*.npz", 0.25, 0.45)}

def patterns(dataset):
    pat, t0, t1 = EPOCHS[dataset]
    out = {}
    for f in sorted(glob.glob(pat)):
        z = np.load(f, allow_pickle=True)
        t = z["times"]; m = (t >= t0 - 1e-9) & (t < t1 - 1e-9) if dataset == "megmasc" else (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
        x = np.asarray(z["epochs"][:, :, m], dtype=np.float32).mean(-1)          # (epoch, channel)
        w = np.asarray(z["word_indices"], dtype=np.int64)
        good_ch = np.isfinite(x).all(0) & (x.std(0) > 1e-12)
        x = x[:, good_ch]
        keep = np.isfinite(x).all(1); x, w = x[keep], w[keep]
        ids, inv = np.unique(w, return_inverse=True)
        mean = np.zeros((len(ids), x.shape[1]), np.float32); np.add.at(mean, inv, x)
        mean /= np.bincount(inv)[:, None]
        sd = mean.std(1) > 1e-12
        out[Path(f).name.split("_")[0]] = (ids[sd], mean[sd])
        print(Path(f).name, "words", int(sd.sum()), "channels", mean.shape[1], flush=True)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=sorted(EPOCHS))
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    W.STANDARDIZE = True
    a.out_root.mkdir(parents=True, exist_ok=True)
    neural = patterns(a.dataset)
    tasks = W._tasks(a.dataset)[: a.limit] if a.limit else W._tasks(a.dataset)
    print(f"{a.dataset}: {len(tasks)} models, {len(neural)} subjects", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(W._score, t, a.dataset, neural, a.out_root) for t in tasks]
        for f in as_completed(futs):
            print(f.result(), flush=True)
    W._summarize(a.dataset, a.out_root)
