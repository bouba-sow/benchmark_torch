#!/usr/bin/env python3
"""MEG-MASC signal sanity checks (one subject per call; CPU only, run on SLURM).

Rebuilds word-locked epochs from raw KIT with the benchmark preprocessing
(bandpass 0.1-40 @1000 Hz -> 50 Hz -> RobustScaler+clamp) but WITHOUT baseline
correction, then derives the baseline-corrected variant in memory
(epoch - mean[-0.2, 0]) so both variants come from one raw pass.

Per subject it saves, under --out-dir (NOT outputs/):
  * evoked GFP time course (check 1), both variants
  * per-word outcomes for 4 outcome definitions x 2 windows x 2 variants, split
    into session A / session B / mean (checks 2, 3 and fix 4 feed from this)
  * word covariates (log duration, length) for the positive control

``megmasc_signal_check_reduce.py`` turns those into the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np

from neuroai.benchmarks.megmasc import (
    _bandpass_eeg,
    _extract_word_epochs,
    _robust_scale_clamp,
)
from neuroai.datasets.megmasc import (
    MEGMASC_SENSOR_REGIONS,
    MegMascDataset,
    load_meg_run,
    load_run_words,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("megmasc_signal_check")

SFREQ = 50.0
TMIN, TMAX = -0.2, 0.8
WINDOWS = {"early_80_160": (0.08, 0.16), "late_250_450": (0.25, 0.45)}
REGIONS = sorted(set(MEGMASC_SENSOR_REGIONS))
REGION_IDX = {r: np.array([i for i, x in enumerate(MEGMASC_SENSOR_REGIONS) if x == r]) for r in REGIONS}


def build_epochs(ds: MegMascDataset, subject: str):
    """Return list of dicts (story, session, key, dur, epochs_nobl) per recording."""
    runs = ds.runs()
    runs = runs[runs["subject"] == subject].sort_values(["story", "session"])
    out = []
    for _, run in runs.iterrows():
        rw = load_run_words(run["events_tsv"])
        rw = rw[rw["duration"] >= 0.050].sort_values("word_index")
        meg, sf = load_meg_run(run["con"])
        meg = _bandpass_eeg(meg, sf, 0.1, 40.0)
        from math import gcd
        from scipy.signal import resample_poly
        up, down = int(SFREQ), int(round(sf))
        g = gcd(up, down)
        meg = resample_poly(meg, up // g, down // g, axis=-1).astype(np.float32)
        meg = _robust_scale_clamp(meg, 5.0)
        ep, valid, times = _extract_word_epochs(meg, SFREQ, rw["onset"].to_numpy(float), TMIN, TMAX, None)
        rw = rw[valid]
        out.append(dict(
            story=run["story"], session=str(run["session"]),
            widx=rw["word_index"].to_numpy(int), dur=rw["duration"].to_numpy(float),
            wlen=rw["word"].str.len().to_numpy(int), epochs=ep, times=times,
        ))
        log.info("%s %s %s: %d epochs", subject, run["story"], run["session"], len(rw))
    return out


def outcomes(epochs, times, story_ids, variants):
    """epochs (N, 208, T) for ONE variant -> {(outcome, window): (N, units)}."""
    res = {}
    wins = {k: (times >= a) & (times < b) for k, (a, b) in WINDOWS.items()}
    # spatial filter: leave-one-story-out SVD of evoked (0..0.6 s), sign fixed so
    # the temporal component peaks positive.
    post = (times >= 0.0) & (times < 0.6)
    filt = {}
    for s in np.unique(story_ids):
        tr = story_ids != s
        ev = epochs[tr].mean(0)[:, post]                       # (208, Tpost)
        u, sv, vt = np.linalg.svd(ev, full_matrices=False)
        u1, v1 = u[:, 0], vt[0]
        if v1[np.argmax(np.abs(v1))] < 0:
            u1 = -u1
        filt[s] = u1
    W = np.stack([filt[s] for s in story_ids])                 # (N, 208)
    for wn, m in wins.items():
        amp = epochs[:, :, m].mean(-1)                         # (N, 208)
        reg = np.stack([amp[:, REGION_IDX[r]].mean(1) for r in REGIONS], 1)
        rms = np.stack([np.sqrt((amp[:, REGION_IDX[r]] ** 2).mean(1)) for r in REGIONS], 1)
        res[("region_mean", wn)] = reg
        res[("region_rms", wn)] = rms
        res[("sensor", wn)] = amp
        res[("spatial_filter", wn)] = np.einsum("nc,nct->nt", W, epochs[:, :, m]).mean(-1, keepdims=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--bids-root", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = MegMascDataset(a.bids_root, subjects=[a.subject])
    recs = build_epochs(ds, a.subject)
    times = recs[0]["times"]
    story_names = sorted({r["story"] for r in recs})
    bl_mask = (times >= -0.2) & (times <= 0.0)

    arrays: dict[str, np.ndarray] = {"times": times}
    # rows: one per (recording, word)
    story_ids = np.concatenate([[story_names.index(r["story"])] * len(r["widx"]) for r in recs])
    key = np.concatenate([story_names.index(r["story"]) * 100000 + r["widx"] for r in recs])
    dur = np.concatenate([r["dur"] for r in recs])
    wlen = np.concatenate([r["wlen"] for r in recs])
    E0 = np.concatenate([r["epochs"] for r in recs], 0)
    # sessions: rank within (story) by order of recording
    sess_rank = np.concatenate([[i_s] * len(r["widx"]) for r in recs for i_s in [
        sorted({x["session"] for x in recs if x["story"] == r["story"]}).index(r["session"])]])

    for vname, E in (("nobl", E0), ("bl", E0 - E0[:, :, bl_mask].mean(-1, keepdims=True))):
        gfp = np.sqrt((E.mean(0) ** 2).mean(0))                 # (T,)
        arrays[f"gfp_{vname}"] = gfp
        for s in (0, 1):
            m = sess_rank == s
            if m.any():
                arrays[f"gfp_{vname}_s{s}"] = np.sqrt((E[m].mean(0) ** 2).mean(0))
        for (on, wn), Y in outcomes(E, times, story_ids, vname).items():
            arrays[f"y_{vname}_{on}_{wn}"] = Y.astype(np.float32)

    arrays.update(key=key, sess=sess_rank, story=story_ids, dur=dur, wlen=wlen)
    np.savez_compressed(out_dir / f"{a.subject}.npz", **arrays)
    (out_dir / f"{a.subject}.meta.json").write_text(json.dumps(dict(stories=story_names, n_rows=int(len(key)))))
    log.info("saved %s", out_dir / f"{a.subject}.npz")


if __name__ == "__main__":
    main()
