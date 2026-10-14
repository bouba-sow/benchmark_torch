#!/usr/bin/env python3
"""Per-word EEG amplitudes of Broderick 2018 for an a-priori N400 ROI, from the raw BIDS recordings (one subject per call).

Same preprocessing as neuroai/benchmarks/broderick_lme.py (average reference over the good channels, 1-8 Hz FIR, 32 Hz, epochs
[-0.2, 1.0] s, baseline [-0.2, 0] s) but, instead of the four BioSemi connector banks, the outcome is the mean over the BioSemi sensors
nearest to the six N400 sites of Nieuwland et al. 2018 (Cz, C3, C4, Pz, P3, P4; neuroai/datasets/nieuwland.py::N400_CHANNELS).
Written columns (one row per word with a valid epoch): chunk (run), word_idx, roi_<tmin>_<tmax> (ROI mean over the window, ms),
s_<sensor> (each ROI sensor, 300-500 ms), bank_A..bank_D (bank means, 300-500 ms; reproduces the existing region tables), n_roi.
    python broderick_roi_table.py sub-002 --out $SCRATCH/broderick_roi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from neuroai.datasets.broderick import BroderickDataset, channel_region, load_all_words, load_eeg_run
from neuroai.datasets.podcast import extract_word_epochs

# BioSemi-128 sensors nearest to the 10-20 sites (mne biosemi128 vs standard_1020 montage; Cz = A1 by the BioSemi convention)
ROI_SENSORS = {"Cz": "A1", "C3": "D19", "C4": "B22", "Pz": "A19", "P3": "A18", "P4": "B4"}
WINDOWS = ((0.3, 0.5), (0.2, 0.4), (0.4, 0.6), (0.2, 0.6))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject")
    ap.add_argument("--bids-root", type=Path, default=ROOT / "data" / "broderick")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ds = BroderickDataset(a.bids_root, subjects=[a.subject], drop_bads=True, n_runs=20)
    words = load_all_words(a.bids_root, n_runs=20)
    good = ds.good_channel_names(a.subject, run=1)
    roi_idx = {n: good.index(s) for n, s in ROI_SENSORS.items() if s in good}
    banks = {}
    for ci, ch in enumerate(good):
        if ch[0] in "ABCD":
            banks.setdefault(ch[0], []).append(ci)
    print(a.subject, len(good), "good channels; ROI sensors present:", {n: ROI_SENSORS[n] for n in roi_idx}, flush=True)
    parts = []
    for run in ds.available_runs(a.subject):
        w = words[words["run"] == run]
        if w.empty:
            continue
        data, sfreq, names = load_eeg_run(a.bids_root, a.subject, run, l_freq=1.0, h_freq=8.0, target_sfreq=32.0,
                                          average_ref=True, good_channels=good)
        if list(names) != good:
            print(f"run {run}: channel order mismatch, skipped", flush=True)
            continue
        ep, valid, t = extract_word_epochs(data, sfreq, recording_tmin=0.0, word_onsets=w["onset"].to_numpy(np.float64),
                                           tmin=-0.2, tmax=1.0)
        if not valid.any():
            continue
        ep = ep - ep[:, :, (t >= -0.2) & (t <= 0.0)].mean(axis=-1, keepdims=True)
        out = {"chunk": run, "word_idx": w["word_idx"].to_numpy(np.int64)[valid]}
        ri = list(roi_idx.values())
        for lo, hi in WINDOWS:
            m = (t >= lo) & (t < hi)
            out[f"roi_{int(lo * 1000)}_{int(hi * 1000)}"] = ep[:, ri][:, :, m].mean(axis=(1, 2))
        m = (t >= 0.3) & (t < 0.5)
        amp = ep[:, :, m].mean(axis=-1)
        for n, ci in roi_idx.items():
            out[f"s_{n}"] = amp[:, ci]
        for b, cis in banks.items():
            out[f"bank_{b}"] = amp[:, cis].mean(axis=1)
        out["n_roi"] = len(ri)
        parts.append(pd.DataFrame(out))
        print(f"run {run}: {int(valid.sum())} words", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(a.out / f"table_{a.subject}.parquet")


if __name__ == "__main__":
    main()
