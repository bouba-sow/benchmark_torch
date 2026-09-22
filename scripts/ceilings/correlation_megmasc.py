#!/usr/bin/env python3
"""MEG-MASC noise ceiling — within-subject session test-retest.

Every MEG-MASC subject (bar the few single-session ones) listened to each of
the four stories **twice**, in two identical sessions. That gives a proper
measurement-reliability ceiling: how well the word-locked MEG response in one
session predicts the same words' response in the other session. It is the
upper bound on what any encoding model can reach, and unlike the subject-split
ceilings used for the eye-tracking / fMRI datasets it needs no Monte-Carlo —
each subject yields a full-data estimate.

Metric is matched to the encoder's ``mean_best_pearsonr``: per channel, the
mean over the 50 lag bins of the across-words Pearson r between session A and
session B. Reported three ways:

  r_tt        raw test-retest r  (reliability of ONE session)
  r_tt_sqrt   sqrt(r_tt)         — bound for predicting a single trial
                                   (closest to the pooled-then-CV encoder)
  r_tt_sb     2 r_tt / (1 + r_tt) — Spearman-Brown, bound for predicting the
                                   2-session average

Outputs (mirrors scripts/ceilings/correlation.py layout):
  <out>/megmasc/ceiling_by_unit.csv    one row per (subject, channel)
  <out>/megmasc/ceiling_headline.csv   mean over channels (+ top-decile)

Usage
-----
    python scripts/ceilings/correlation_megmasc.py --subject sub-01
    python scripts/ceilings/correlation_megmasc.py --array-index 0
    python scripts/ceilings/correlation_megmasc.py --reduce      # headline from by_unit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np
import pandas as pd
import yaml

from neuroai.benchmarks.megmasc import (
    _bandpass_eeg,
    _extract_word_epochs,
    _robust_scale_clamp,
)
from neuroai.datasets.megmasc import MegMascDataset, load_meg_run, load_run_words

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

_SUBJECTS_FILE = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "configs" / "configs/subjects/megmasc_subjects.json"
_OUT_DEFAULT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "outputs_jeanzay" / "noise_ceilings" / "megmasc"


def _cfg(path: str) -> dict:
    os.environ.setdefault("BENCHMARK_ROOT", str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
    return yaml.safe_load(os.path.expandvars(Path(path).read_text()))


def _pearson_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Column-wise Pearson r between two (n, k) arrays -> (k,)."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    num = (A * B).sum(0)
    den = np.sqrt((A * A).sum(0) * (B * B).sum(0))
    out = np.full(A.shape[1], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _spearman_brown(r: np.ndarray) -> np.ndarray:
    d = 1.0 + r
    out = np.full_like(r, np.nan)
    ok = np.abs(d) > 1e-12
    out[ok] = 2.0 * r[ok] / d[ok]
    return out


def _session_epochs(bids_root, run_row, dcfg) -> tuple[np.ndarray, np.ndarray] | None:
    """(word_index, epochs[n_words, n_ch, n_lags]) for one recording, or None."""
    words = load_run_words(run_row["events_tsv"])
    if dcfg["min_word_duration"] > 0:
        words = words[words["duration"] >= dcfg["min_word_duration"]]
    if words.empty:
        return None
    words = words.sort_values("word_index")
    try:
        meg, sf = load_meg_run(run_row["con"])
    except (FileNotFoundError, OSError) as exc:
        logger.warning("  %s: MEG unreadable: %s", run_row["con"], exc)
        return None
    if dcfg["l_freq"] and dcfg["h_freq"]:
        meg = _bandpass_eeg(meg, sf, dcfg["l_freq"], dcfg["h_freq"])
    if dcfg["target_sfreq"] < sf:
        from math import gcd
        from scipy.signal import resample_poly
        up, down = int(round(dcfg["target_sfreq"])), int(round(sf))
        g = gcd(up, down)
        meg = resample_poly(meg, up // g, down // g, axis=-1).astype(np.float32)
    if dcfg["robust_scale"]:
        meg = _robust_scale_clamp(meg, dcfg["clamp"])
    ep, valid, _ = _extract_word_epochs(
        meg, dcfg["target_sfreq"], words["onset"].to_numpy(float),
        dcfg["tmin"], dcfg["tmax"], tuple(dcfg["baseline"]) if dcfg["baseline"] else None,
    )
    if ep.shape[0] == 0:
        return None
    return words["word_index"].to_numpy()[valid], ep


def compute_subject(subject: str, cfg: dict) -> pd.DataFrame:
    ds_cfg = cfg["dataset"]
    bids_root = ds_cfg["bids_root"]
    dcfg = dict(
        l_freq=ds_cfg.get("l_freq"), h_freq=ds_cfg.get("h_freq"),
        target_sfreq=float(ds_cfg.get("target_sfreq", 50.0)),
        robust_scale=bool(ds_cfg.get("robust_scale", True)),
        clamp=ds_cfg.get("clamp", 5.0),
        tmin=float(ds_cfg.get("tmin", -0.2)), tmax=float(ds_cfg.get("tmax", 0.8)),
        baseline=ds_cfg.get("baseline", [-0.2, 0.0]),
        min_word_duration=float(ds_cfg.get("min_word_duration", 0.05)),
    )
    runs = MegMascDataset(bids_root, subjects=[subject]).runs()
    if runs.empty:
        logger.warning("[%s] no runs", subject)
        return pd.DataFrame()
    sessions = sorted(runs["session"].unique())
    if len(sessions) < 2:
        logger.warning("[%s] single session — no test-retest", subject)
        return pd.DataFrame()
    sa, sb = sessions[0], sessions[1]

    yA_all, yB_all = [], []
    for story in sorted(runs["story"].unique()):
        ra = runs[(runs.session == sa) & (runs.story == story)]
        rb = runs[(runs.session == sb) & (runs.story == story)]
        if ra.empty or rb.empty:
            continue
        a = _session_epochs(bids_root, ra.iloc[0], dcfg)
        b = _session_epochs(bids_root, rb.iloc[0], dcfg)
        if a is None or b is None:
            continue
        (wa, ea), (wb, eb) = a, b
        common = np.intersect1d(wa, wb)
        if common.size < 20:
            continue
        ia = pd.Index(wa).get_indexer(common)
        ib = pd.Index(wb).get_indexer(common)
        yA_all.append(ea[ia])   # (nw, nch, nlag)
        yB_all.append(eb[ib])
        logger.info("  [%s] story %-18s %4d common words", subject, story, common.size)
    if not yA_all:
        return pd.DataFrame()
    yA = np.concatenate(yA_all, 0)
    yB = np.concatenate(yB_all, 0)
    n_words, n_ch, n_lag = yA.shape

    # per channel: mean over lags of across-words Pearson r
    r_lag = np.full((n_ch, n_lag), np.nan)
    for l in range(n_lag):
        r_lag[:, l] = _pearson_cols(yA[:, :, l], yB[:, :, l])
    r_tt = np.nanmean(r_lag, axis=1)
    return pd.DataFrame(dict(
        subject=subject,
        channel=[f"ch_{c:02d}" for c in range(n_ch)],
        n_words=n_words,
        r_tt=r_tt,
        r_tt_sqrt=np.sqrt(np.clip(r_tt, 0, None)),
        r_tt_sb=_spearman_brown(r_tt),
    ))


def reduce_headline(out_dir: Path) -> None:
    parts = sorted(out_dir.glob("ceiling_by_unit__*.csv"))
    if not parts:
        logger.error("no per-subject ceiling_by_unit__*.csv under %s", out_dir)
        return
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(out_dir / "ceiling_by_unit.csv", index=False)
    def _stats(s: pd.Series) -> dict:
        v = s.to_numpy(float); v = v[np.isfinite(v)]
        top = np.sort(v)[::-1][: max(1, v.size // 10)]
        return dict(mean=float(v.mean()), sem=float(v.std(ddof=1) / np.sqrt(v.size)),
                    median=float(np.median(v)), top_decile_mean=float(top.mean()),
                    p95=float(np.percentile(v, 95)))
    head = {"n_subjects": int(df.subject.nunique()),
            "n_channels": int(df.channel.nunique()),
            "n_rows": len(df)}
    for col in ("r_tt", "r_tt_sqrt", "r_tt_sb"):
        for k, val in _stats(df[col]).items():
            head[f"{col}_{k}"] = val
    pd.DataFrame([head]).to_csv(out_dir / "ceiling_headline.csv", index=False)
    (out_dir / "ceiling_meta.json").write_text(json.dumps({
        "dataset": "megmasc", "method": "session_test_retest",
        "metric": "mean-over-lags across-words Pearson r between the two "
                  "identical sessions; matched to encoder mean_best_pearsonr",
        "note": "r_tt = single-session reliability; compare encoder r to "
                "r_tt_sqrt (single-trial bound) and r_tt_sb (2-session-avg bound). "
                "Whole-array mean is dragged down by non-language sensors — see "
                "top_decile_mean / p95.",
    }, indent=2))
    logger.info("headline -> %s", out_dir / "ceiling_headline.csv")
    print(pd.DataFrame([head]).T.to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/embedding/neural/megmasc.yaml")
    p.add_argument("--subject", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--out_dir", default=str(_OUT_DEFAULT))
    p.add_argument("--reduce", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.reduce:
        reduce_headline(out_dir)
        return

    cfg = _cfg(args.config)
    subjects = json.loads(_SUBJECTS_FILE.read_text()) if _SUBJECTS_FILE.exists() \
        else sorted(MegMascDataset(cfg["dataset"]["bids_root"]).available_subjects())
    if args.subject:
        subject = args.subject
    elif args.array_index is not None:
        if not (0 <= args.array_index < len(subjects)):
            logger.info("array-index %d out of range", args.array_index)
            return
        subject = subjects[args.array_index]
    else:
        p.error("pass --subject / --array-index / --reduce")

    df = compute_subject(subject, cfg)
    if df.empty:
        logger.warning("[%s] no ceiling rows", subject)
        return
    # one file per subject — concurrent array tasks never touch the same file
    df.to_csv(out_dir / f"ceiling_by_unit__{subject}.csv", index=False)
    logger.info("[%s] %d ch | r_tt mean=%.4f median=%.4f p95=%.4f | sqrt mean=%.4f | sb mean=%.4f",
                subject, len(df), df.r_tt.mean(), df.r_tt.median(), df.r_tt.quantile(.95),
                df.r_tt_sqrt.mean(), df.r_tt_sb.mean())


if __name__ == "__main__":
    main()
