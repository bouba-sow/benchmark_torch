#!/usr/bin/env python
"""Compute split-half noise ceilings for every active benchmark dataset.

Motivation
----------
Model scores in this benchmark are correlations (Pearson / Spearman r) between
LLM-derived predictions and human neural/behavioural responses.  Those scores
are only interpretable relative to the *reliability* of the data: no model can
correlate with a target more strongly than the target correlates with itself.
This script estimates that upper bound (the "noise ceiling") for each dataset.

Method (following van Bree, Styrnal & Hebart, 2025 — "How Much Variance Does
Your Model Explain?"):

1.  Split-half reliability by splitting *subjects* into two random halves
    (these language datasets present each item ~once per subject, so there are
    no within-subject stimulus repetitions to split on — the group-average
    ceiling is the appropriate estimator).  For each split, average each half's
    response per stimulus/item (or per time sample for continuous stimuli) and
    correlate the two halves across items -> r_sh.  Averaged over many random
    splits.
2.  Spearman-Brown correction to extrapolate the half-split reliability to the
    full sample:            r_SB = 2 * r_sh / (1 + r_sh).
3.  **The key correction from the paper**: the reliability r_SB is the ceiling
    for *explained variance* (R^2).  The ceiling for a *correlation* (which is
    what every model score here is) is the SQUARE ROOT:  ceiling_r = sqrt(r_SB).
    Using r_SB directly as a ceiling for a correlation systematically
    *understates* the ceiling and makes models look closer to it than they are.

We therefore report both, but ``ceiling_r`` (= sqrt(r_SB)) is the number to
compare model correlations against.

Everything is computed at two granularities:
  * per native sub-unit (electrode / channel / ROI / fROI / voxel) and
  * a headline value = mean over sub-units (matching how the model score is
    aggregated, e.g. ``mean_spearmanr`` over electrodes).

The split-half correlation uses the SAME correlation family (Pearson vs
Spearman) as the dataset's primary model metric, so the ceiling is on the same
scale as the score it bounds.

Outputs (under ``--out-dir``, default ``outputs/noise_ceilings/``):
  <slug>/ceiling_by_unit.csv        one row per (component, unit)
  <slug>/ceiling_headline.csv       one row per component (mean over units)
  noise_ceilings_summary.csv        all datasets, headline rows concatenated
  noise_ceilings_by_unit.csv        all datasets, per-unit rows concatenated
  noise_ceilings_meta.json          per-dataset status / method / notes

Usage
-----
    python scripts/ceilings/correlation.py --dataset all
    python scripts/ceilings/correlation.py --dataset szewczyk nieuwland
    python scripts/ceilings/correlation.py --dataset all --skip-heavy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Make the repo importable when run as a script.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("noise_ceilings")

DATA_ROOT = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """One self-contained response tensor to compute a ceiling for.

    Provide EITHER ``long`` (tidy, for sparse/item-structured data) OR ``dense``
    (a 3-D array, for dense neural data where a long DataFrame would be huge).

    ``long`` columns:
        subject : hashable   — the unit that is split into halves
        item    : hashable   — the axis correlated ACROSS (stimulus / time sample)
        unit    : hashable   — native sub-unit (electrode/channel/ROI/voxel);
                               use a constant (e.g. "_all") when there is none
        value   : float      — the scalar response

    ``dense`` : (n_items, n_units, n_subjects) float array (NaN = missing);
    ``unit_labels`` : length-n_units labels aligned to axis 1.
    """

    component: str          # e.g. "" / "n4" / "band_t1" / "cond_B" / "LANG"
    corr: str               # "pearson" | "spearman"
    long: pd.DataFrame | None = None
    dense: np.ndarray | None = None
    unit_labels: list | None = None
    # Precomputed per-unit ceilings (skip estimation entirely). Must carry
    # columns: unit, r_sh, r_sb, ceiling_r, ceiling_r2, n_subjects, n_items.
    precomputed: pd.DataFrame | None = None


def spearman_brown(r: float) -> float:
    """Spearman-Brown up-correction of a split-half reliability."""
    if not np.isfinite(r):
        return np.nan
    denom = 1.0 + r
    if abs(denom) < 1e-12:
        return np.nan
    return float(2.0 * r / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties averaged), NaN-free input assumed."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def _paired_corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    """Pearson/Spearman correlation over finite pairs; NaN if degenerate."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if kind == "spearman":
        x = _rankdata(x)
        y = _rankdata(y)
    xs = x - x.mean()
    ys = y - y.mean()
    denom = np.sqrt((xs * xs).sum() * (ys * ys).sum())
    if denom < 1e-12:
        return np.nan
    return float((xs * ys).sum() / denom)


def _ceiling_from_matrix(
    mat: np.ndarray,
    corr: str,
    n_splits: int,
    rng: np.random.Generator,
    min_items: int = 5,
) -> dict:
    """Subject-split-half ceiling for one (item x subject) response matrix.

    Parameters
    ----------
    mat : (n_items, n_subjects) float, may contain NaN (missing item/subject).
    """
    n_items, n_subj = mat.shape
    if n_subj < 2:
        return dict(r_sh=np.nan, r_sh_sem=np.nan, r_sb=np.nan,
                    ceiling_r=np.nan, ceiling_r2=np.nan,
                    n_subjects=n_subj, n_items=n_items, n_splits_used=0)
    half = n_subj // 2
    rs: list[float] = []
    for _ in range(n_splits):
        perm = rng.permutation(n_subj)
        ia, ib = perm[:half], perm[half:2 * half]
        # nanmean across subjects within each half (all-NaN rows -> NaN, dropped)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            a = np.nanmean(mat[:, ia], axis=1)
            b = np.nanmean(mat[:, ib], axis=1)
        keep = np.isfinite(a) & np.isfinite(b)
        if keep.sum() < min_items:
            continue
        r = _paired_corr(a[keep], b[keep], corr)
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return dict(r_sh=np.nan, r_sh_sem=np.nan, r_sb=np.nan,
                    ceiling_r=np.nan, ceiling_r2=np.nan,
                    n_subjects=n_subj, n_items=n_items, n_splits_used=0)
    rs = np.asarray(rs, dtype=np.float64)
    r_sh = float(rs.mean())
    r_sh_sem = float(rs.std(ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0
    r_sb = spearman_brown(r_sh)
    r_sb_c = float(np.clip(r_sb, 0.0, 1.0)) if np.isfinite(r_sb) else np.nan
    ceiling_r = float(np.sqrt(r_sb_c)) if np.isfinite(r_sb_c) else np.nan
    return dict(r_sh=r_sh, r_sh_sem=r_sh_sem, r_sb=r_sb,
                ceiling_r=ceiling_r, ceiling_r2=r_sb_c,
                n_subjects=int(n_subj), n_items=int(n_items),
                n_splits_used=int(len(rs)))


def _rank_columns_nan(mat: np.ndarray) -> np.ndarray:
    """Average-rank each column over its finite entries (NaN preserved)."""
    out = np.full(mat.shape, np.nan, dtype=np.float64)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        f = np.isfinite(col)
        if f.any():
            out[f, j] = _rankdata(col[f])
    return out


def _mean_pairwise_corr(mat: np.ndarray, kind: str, min_overlap: int) -> float:
    """Mean pairwise (between-subject) correlation across items.

    Columns are subjects; correlations use pairwise-complete items. For
    Spearman, columns are rank-transformed once (over each subject's finite
    items) and then correlated linearly.
    """
    if kind == "spearman":
        mat = _rank_columns_nan(mat)
    finite = np.isfinite(mat)
    n = mat.shape[1]
    if finite.all():
        # fast vectorised path (no missing data)
        c = np.corrcoef(mat, rowvar=False)
        iu = np.triu_indices(n, 1)
        vals = c[iu]
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else np.nan
    # pairwise-complete path (subject counts are small)
    vals: list[float] = []
    for i in range(n):
        xi, fi = mat[:, i], finite[:, i]
        for j in range(i + 1, n):
            m = fi & finite[:, j]
            if m.sum() < min_overlap:
                continue
            r = _paired_corr(xi[m], mat[m, j], "pearson")  # already ranked
            if np.isfinite(r):
                vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def _closed_form_ceiling(
    mat: np.ndarray,
    corr: str,
    min_items: int = 5,
    min_overlap: int = 5,
) -> dict:
    """Analytic group-reliability ceiling (Spearman-Brown / standardized
    Cronbach's alpha) — the exact quantity the subject-split-half Monte-Carlo
    estimates, without sampling loops.

    reliability = n*rbar / (1 + (n-1)*rbar),  rbar = mean pairwise between-
    subject correlation across items.  This is the R^2 ceiling (== r_SB);
    ceiling_r = sqrt(reliability).
    """
    n_items, n_subj = mat.shape
    # items usable = rows with at least two finite subjects
    usable_items = int((np.isfinite(mat).sum(axis=1) >= 2).sum())
    if n_subj < 2 or usable_items < min_items:
        return dict(r_sh=np.nan, r_sh_sem=np.nan, r_sb=np.nan,
                    ceiling_r=np.nan, ceiling_r2=np.nan,
                    n_subjects=n_subj, n_items=usable_items, n_splits_used=0)
    rbar = _mean_pairwise_corr(mat, corr, min_overlap)
    if not np.isfinite(rbar):
        return dict(r_sh=np.nan, r_sh_sem=np.nan, r_sb=np.nan,
                    ceiling_r=np.nan, ceiling_r2=np.nan,
                    n_subjects=n_subj, n_items=usable_items, n_splits_used=0)
    r_sb = n_subj * rbar / (1.0 + (n_subj - 1) * rbar)
    r_sb_c = float(np.clip(r_sb, 0.0, 1.0)) if np.isfinite(r_sb) else np.nan
    ceiling_r = float(np.sqrt(r_sb_c)) if np.isfinite(r_sb_c) else np.nan
    return dict(r_sh=float(rbar), r_sh_sem=np.nan, r_sb=float(r_sb),
                ceiling_r=ceiling_r, ceiling_r2=r_sb_c,
                n_subjects=int(n_subj), n_items=usable_items, n_splits_used=0)


def _pivot(long: pd.DataFrame, unit) -> tuple[np.ndarray, np.ndarray]:
    """Return (item x subject) value matrix for one unit and the subject list."""
    sub = long if unit is None else long[long["unit"] == unit]
    m = sub.pivot_table(index="item", columns="subject", values="value",
                        aggfunc="mean")
    return m.to_numpy(dtype=np.float64), m.columns.to_numpy()


def _iter_unit_matrices(block: Block, max_units: int | None, seed: int):
    """Yield (unit_label, item-x-subject matrix) for each native sub-unit.

    Works for both long-form and dense blocks; subsamples units above
    ``max_units`` to keep dense neural datasets tractable.
    """
    if block.dense is not None:
        labels = (block.unit_labels
                  if block.unit_labels is not None
                  else list(range(block.dense.shape[1])))
        idx = np.arange(len(labels))
        subsampled = False
        if max_units is not None and len(idx) > max_units:
            idx = np.random.default_rng(seed).choice(idx, size=max_units,
                                                     replace=False)
            subsampled = True
        for i in idx:
            yield labels[i], block.dense[:, i, :], subsampled
    else:
        long = block.long.dropna(subset=["value"])
        units = list(pd.unique(long["unit"]))
        subsampled = False
        if max_units is not None and len(units) > max_units:
            units = list(np.random.default_rng(seed).choice(
                units, size=max_units, replace=False))
            subsampled = True
        for unit in units:
            mat, _ = _pivot(long, unit)
            yield unit, mat, subsampled


def compute_block_ceiling(
    block: Block,
    n_splits: int,
    seed: int,
    min_items: int,
    max_units: int | None,
    method: str = "split_half",
) -> tuple[pd.DataFrame, dict]:
    """Per-unit + headline ceiling for a block.

    Returns (by_unit_df, headline_dict).
    Headline ceiling = mean over units of the per-unit ceiling_r (this matches
    how the model score is aggregated, e.g. mean r over electrodes).

    ``method``:
      "split_half"  — subject-split Monte-Carlo over ``n_splits`` (default;
                      design-agnostic, correct for unbalanced designs).
      "closed_form" — analytic Spearman-Brown / Cronbach reliability (fast, no
                      sampling loop) but **only valid when every subject has
                      (nearly) every item** (fully-crossed designs, e.g. dense
                      fMRI/EEG timecourses). It OVERESTIMATES on sparse designs
                      where each subject sees a subset of items (podcast,
                      fedorenko); a warning is emitted.
    """
    if block.precomputed is not None:
        by_unit = block.precomputed.copy()
        if "n_splits_used" not in by_unit.columns:
            by_unit["n_splits_used"] = 0
        return _headline_from_by_unit(by_unit, block, subsampled=False,
                                      method="precomputed")

    rows: list[dict] = []
    subsampled = False
    fill_fracs: list[float] = []
    for unit, mat, sub in _iter_unit_matrices(block, max_units, seed):
        subsampled = subsampled or sub
        if method == "closed_form":
            fill_fracs.append(float(np.isfinite(mat).mean()))
            res = _closed_form_ceiling(mat, block.corr, min_items)
        else:
            rng = np.random.default_rng(seed + hash(str(unit)) % 100_000)
            res = _ceiling_from_matrix(mat, block.corr, n_splits, rng, min_items)
        res["unit"] = unit
        rows.append(res)
    if method == "closed_form" and fill_fracs:
        mean_fill = float(np.mean(fill_fracs))
        if mean_fill < 0.9:
            logger.warning(
                "  [%s] closed_form on a non-crossed design (%.0f%% of "
                "subject×item cells populated) — ceiling is OVERESTIMATED; "
                "use --method split_half for this dataset.",
                block.component or "-", 100 * mean_fill)
    by_unit = pd.DataFrame(rows)
    return _headline_from_by_unit(by_unit, block, subsampled, method)


def _headline_from_by_unit(
    by_unit: pd.DataFrame, block: Block, subsampled: bool, method: str,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate per-unit ceilings into a headline row (mean over units)."""
    def _mean_sem(col: str) -> tuple[float, float]:
        v = by_unit[col].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan, np.nan
        sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
        return float(v.mean()), sem

    cr_mean, cr_sem = _mean_sem("ceiling_r")
    cr2_mean, _ = _mean_sem("ceiling_r2")
    rsh_mean, _ = _mean_sem("r_sh")
    headline = dict(
        component=block.component,
        corr=block.corr,
        ceiling_r=cr_mean,             # <-- compare model correlations to this
        ceiling_r_sem=cr_sem,          # spread over units
        ceiling_r2=cr2_mean,           # compare model R^2 to this
        r_sh=rsh_mean,
        n_units=int(by_unit["unit"].nunique()),
        n_subjects=int(by_unit["n_subjects"].max()) if len(by_unit) else 0,
        n_items=int(by_unit["n_items"].max()) if len(by_unit) else 0,
        units_subsampled=subsampled,
        method=method,
    )
    return by_unit, headline


# ---------------------------------------------------------------------------
# Per-dataset handlers.  Each returns (info_dict, list[Block]).
# info_dict keys: name, modality, metric, note.
# ---------------------------------------------------------------------------

def _single_unit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["unit"] = "_all"
    return df


# ---- Szewczyk & Federmeier N400 EEG ---------------------------------------
def handler_szewczyk() -> tuple[dict, list[Block]]:
    from neuroai.datasets.szewczyk import load_data
    path = DATA_ROOT / "Szewczyk_Federmeir/osfstorage/bind-5-dset_300-500.rds"
    df = load_data(path)
    # n400_z already per-subject z-scored (load_data). Pre-averaged over the
    # 300-500ms window and 4 centro-parietal ROI clusters -- no per-electrode
    # topography, so one pseudo-unit ("_all"), same trick as Fedorenko2016.
    # item must be the full (subexperiment, item, critical_word) triple: item
    # numbers repeat across sub-experiments and across cloze completions.
    df["item"] = (df["subexperiment"].astype(str) + "|" + df["item"].astype(str)
                  + "|" + df["critical_word"].astype(str))
    long = _single_unit(df.rename(columns={"n400_z": "value"}))[
        ["subject", "item", "unit", "value"]
    ].dropna(subset=["value"])
    info = dict(name="Szewczyk N400 EEG", modality="EEG", metric="mean_pearsonr",
                note="pre-averaged 300-500ms N400 over 4 centro-parietal ROI clusters "
                     "(no per-electrode topography) -> single pseudo-unit '_all'; "
                     "5 merged sub-experiments; item=(subexperiment,item,critical_word); "
                     "split subjects (already namespaced -> no cross-subexperiment collision)")
    return info, [Block("", "pearson", long)]


# ---- Natural Stories self-paced reading (Pearson, behavioural) ------------
def handler_naturalstories_spr() -> tuple[dict, list[Block]]:
    rt_path = DATA_ROOT / "naturalstories/naturalstories_RTS/processed_RTs.tsv"
    df = pd.read_csv(rt_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    ren = {"WorkerId": "subject", "item": "story_id", "zone": "zone", "RT": "rt"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df = df[(df["rt"] >= 100) & (df["rt"] <= 3000)]
    df["item"] = df["story_id"].astype(str) + "|" + df["zone"].astype(str)
    long = _single_unit(df.rename(columns={"rt": "value"}))[
        ["subject", "item", "unit", "value"]
    ]
    info = dict(name="Natural Stories SPR", modality="Behavioral",
                metric="reading-time reliability",
                note="reading time per word; no sub-units; split subjects")
    return info, [Block("", "pearson", long)]


# ---- Eye-tracking / SPR reading-time datasets (Behavioral) -----------------
def _reading_time_block(component: str, trials: pd.DataFrame) -> Block:
    """Subject-split reliability of one word-level reading-time signal as a
    single Block (component label lets multiple measures -- FP/GP/SP -- of
    the same dataset live as separate rows under one dataset name, same
    pattern as the ZuCo Reading / MECO handlers).

    ``trials`` is the per-(subject, word) trial frame (aggregate=False) with
    columns participant_id, story_id, zone, rt. This bounds a model's *correlation
    with reading time*; note the report's headline behavioral metric is a delta
    log-likelihood (a marginal-fit improvement), which this ceiling does not
    directly bound — so the reading-time reliability is reported as context.
    """
    df = trials.dropna(subset=["rt"]).copy()
    df = df[(df["rt"] >= 50) & (df["rt"] <= 3000)]
    df["item"] = df["story_id"].astype(str) + "|" + df["zone"].astype(str)
    df["subject"] = df["participant_id"].astype(str)
    df["unit"] = "_all"
    long = df.rename(columns={"rt": "value"})[["subject", "item", "unit", "value"]]
    return Block(component, "pearson", long)


def _reading_time_handler(name: str, trials: pd.DataFrame, note: str
                          ) -> tuple[dict, list[Block]]:
    """Single-measure convenience wrapper around _reading_time_block."""
    info = dict(name=name, modality="Behavioral", metric="reading-time reliability",
                note=note)
    return info, [_reading_time_block("", trials)]


def _multi_measure_reading_time_handler(
    name: str, loader, measures: tuple[str, ...], note: str,
) -> tuple[dict, list[Block]]:
    """One Block per measure (FP/GP/SP -- different derived DVs from the SAME
    eye-movement record, not different item pools, so each needs its OWN
    ceiling rather than being pooled like zuco_reading's SR/NR/TSR tasks)."""
    blocks = []
    for m in measures:
        trials = loader(m)
        blocks.append(_reading_time_block(m, trials))
    info = dict(name=name, modality="Behavioral", metric="reading-time reliability",
                note=f"{note}; components={'/'.join(measures)}")
    return info, blocks


def handler_geco() -> tuple[dict, list[Block]]:
    from neuroai.datasets.geco import GecoDataset
    return _multi_measure_reading_time_handler(
        "GECO",
        lambda m: GecoDataset(str(DATA_ROOT / "geco"), m, aggregate=False).load_words(),
        ("FP", "GP"),
        "fixation time per word; split subjects (14 subj)")


def handler_meco() -> tuple[dict, list[Block]]:
    from neuroai.datasets.meco import MecoDataset
    root = DATA_ROOT / "meco/osfstorage/release 2.0/version 2.1/wave 1/primary data/eye tracking data"
    return _multi_measure_reading_time_handler(
        "MECO",
        lambda m: MecoDataset(str(root), m, aggregate=False).load_words(),
        ("FP", "GP", "SP"),
        "fixation time per word; split subjects (English subset)")


def handler_provo() -> tuple[dict, list[Block]]:
    from neuroai.datasets.provo import ProvoDataset
    return _multi_measure_reading_time_handler(
        "Provo",
        lambda m: ProvoDataset(str(DATA_ROOT / "Provo/osfstorage"), m, aggregate=False).load_words(),
        ("FP", "GP", "SP"),
        "fixation time per word; split subjects (84 subj)")


def handler_ucl_reading() -> tuple[dict, list[Block]]:
    from neuroai.datasets.ucl_reading import UCLReadingDataset
    trials = UCLReadingDataset(str(DATA_ROOT / "UCL_reading"), aggregate=False).load_words()
    return _reading_time_handler(
        "UCL Reading", trials, "self-paced reading time per word; split subjects")


def handler_onestop() -> tuple[dict, list[Block]]:
    """New 2026-09-03: OneStop's OnestopDataset.load_words() already renames
    its measure column to "rt" (see neuroai/datasets/onestop.py), so it slots
    directly into _reading_time_block with no adapter needed -- same 3
    measures (FP/GP/SP) as MECO/Provo."""
    from neuroai.datasets.onestop import OnestopDataset
    return _multi_measure_reading_time_handler(
        "OneStop",
        lambda m: OnestopDataset(str(DATA_ROOT / "OneStop"), m, aggregate=False).load_words(),
        ("FP", "GP", "SP"),
        "fixation time per word; split subjects (360 subj)")


# ---- Tuckute fMRI (Spearman, ROI, conditions B/D/S) -----------------------
def handler_tuckute() -> tuple[dict, list[Block]]:
    from neuroai.datasets.tuckute import TuckuteDataset, CONDITIONS
    ds = TuckuteDataset(str(DATA_ROOT / "tuckute_fmri/osfstorage"))
    bdf = ds._brain_df  # participant, item_id, roi, cond, response_target
    blocks = []
    for cond in CONDITIONS:
        sub = bdf[bdf["cond"] == cond]
        long = sub.rename(columns={"target_UID": "subject", "item_id": "item",
                                   "roi": "unit", "response_target": "value"})[
            ["subject", "item", "unit", "value"]
        ].dropna(subset=["value"])
        if len(long):
            blocks.append(Block(f"cond_{cond}", "pearson", long))
    info = dict(name="Tuckute fMRI", modality="fMRI",
                metric="mean_best_pearsonr (per B/D/S)",
                note="unit=LH language fROI; D/S have only 2-3 subjects/item "
                     "(ceiling unstable); dataset also ships its own NC")
    return info, blocks


# ---- Natural Stories fMRI (Spearman, fROI, LANG/MD) -----------------------
def handler_ns_fmri() -> tuple[dict, list[Block]]:
    from neuroai.datasets import natural_stories_fmri as nsf
    data_root = str(DATA_ROOT / "naturalstories_fmri/osfstorage")
    subjects = nsf.available_subjects(data_root, "LANG_MD")
    recs: list[dict] = []
    for s in subjects:
        try:
            per_doc = nsf.load_bold_subject(data_root, s, "LANG_MD")
        except Exception as exc:
            logger.warning("ns_fmri: subject %s failed: %s", s, exc)
            continue
        for doc, (bold, _t, frois, nets) in per_doc.items():
            for j, (froi, net) in enumerate(zip(frois, nets)):
                col = bold[:, j]
                for ti in range(len(col)):
                    recs.append(dict(subject=s, item=f"{doc}|{ti}", unit=froi,
                                     network=net, value=col[ti]))
    df = pd.DataFrame.from_records(recs)
    blocks = []
    for comp, net in (("LANG", "LANG"), ("MD", "MD"), ("all", None)):
        sub = df if net is None else df[df["network"] == net]
        long = sub[["subject", "item", "unit", "value"]].dropna(subset=["value"])
        if len(long):
            blocks.append(Block(comp, "pearson", long))
    info = dict(name="Natural Stories fMRI", modality="fMRI",
                metric="mean_best_pearsonr (LANG / MD)",
                note="fROI BOLD timecourse; item=TR; ISC-style subject split")
    return info, blocks


# ---- Pereira fMRI (Pearson, per-ROI, isolated sentences) -------------------
def handler_pereira() -> tuple[dict, list[Block]]:
    """New 2026-09-03. Same per-region-averaged-BOLD, isolated-sentence shape
    as Tuckute -- one scalar per (subject, roi, sentence), 12 language-network
    ROIs crossed with every subject (not a per-subject voxel/electrode grid)."""
    from neuroai.datasets.pereira import PereiraDataset
    ds = PereiraDataset(str(DATA_ROOT / "pereira_fmri" / "Pereira2018_merged.nc"))
    roi_names = ds.rois("language")
    per_subject = ds.get_condition_per_subject(atlas="language")
    recs: list[dict] = []
    for subj, (bold, item_ids) in per_subject.items():
        for ri, roi in enumerate(roi_names):
            col = bold[:, ri]
            valid = ~np.isnan(col)
            for item_id, v in zip(item_ids[valid], col[valid]):
                recs.append(dict(subject=subj, item=item_id, unit=roi, value=v))
    long = pd.DataFrame.from_records(recs)
    info = dict(name="Pereira fMRI", modality="fMRI", metric="mean_pearsonr (per ROI)",
                note="unit=language-network ROI (12), voxel-averaged; split subjects")
    return info, [Block("", "pearson", long)]


def _find_zuco_root() -> str:
    for cand in (DATA_ROOT / "q3zws/osfstorage", DATA_ROOT / "zuco/osfstorage",
                 DATA_ROOT / "q3zws"):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError("ZuCo osfstorage root not found under data/")


# ---- Nieuwland N400 EEG (Spearman, channels) — needs MNE epoching ---------
def handler_nieuwland(subjects: list[str] | None = None) -> tuple[dict, list[Block]]:
    from neuroai.datasets.nieuwland import (
        NieuwlandEpochsConfig, load_all_epochs, N400_TMIN, N400_TMAX,
    )
    bids_root = DATA_ROOT / "nm000228"
    cache = REPO_ROOT / ".cache" / "nieuwland_epochs"
    cfg = NieuwlandEpochsConfig(bids_root=str(bids_root), subjects=subjects,
                                epoch_cache_dir=str(cache))
    res = load_all_epochs(cfg)
    times = res.times
    mask = (times >= N400_TMIN) & (times <= N400_TMAX)
    # per-channel N400 window mean: (trials, channels)
    win = res.Y[:, :, mask].mean(axis=2)
    meta = res.meta.reset_index(drop=True)
    item = (meta["item_number"].astype(str) + "|" +
            meta["condition"].astype(str)).to_numpy()
    recs: list[dict] = []
    for ci, ch in enumerate(res.channel_names):
        for ti in range(win.shape[0]):
            recs.append(dict(subject=meta["subject"].iat[ti], item=item[ti],
                             unit=ch, value=float(win[ti, ci])))
    long = pd.DataFrame.from_records(recs).dropna(subset=["value"])
    info = dict(name="Nieuwland N400 EEG", modality="EEG",
                metric="mean_subject_perchannel_pearsonr",
                note="unit=centro-parietal channel; item=(item_number,condition); "
                     "split subjects")
    return info, [Block("", "pearson", long)]


# ---- Nieuwland N400 from the paper's own preprocessed per-trial values ----
def handler_nieuwland_paper() -> tuple[dict, list[Block]]:
    """Same 160 items x 334 subjects table the paper-formula LME is fit on
    (``lme.paper_n400_tsv``): the paper's per-trial N400, already averaged over the 6
    centro-parietal channels, so there is a single unit (``_all``)."""
    import re

    tsv = DATA_ROOT / "n400_paper_formula" / "osfstorage" / "datasets" / "nieuwland_2018.tsv"
    df = pd.read_csv(tsv, sep="\t").drop_duplicates()
    df = df.dropna(subset=["n400", "FullText", "TargetWords"])
    key = lambda t: re.sub(r"[^a-z]", "", str(t).lower())  # noqa: E731
    df["item"] = df["FullText"].map(key).str[:20] + "|" + df["TargetWords"].map(key)
    long = pd.DataFrame(dict(subject=df["subject"].astype(str), item=df["item"],
                             unit="_all", value=df["n400"].astype(float)))
    info = dict(name="Nieuwland N400 EEG (paper n400)", modality="EEG",
                metric="paper_n400_item_correlation",
                note="paper's per-trial n400 (6-channel mean, 200-500 ms); "
                     "item=(context prefix, noun) [160]; split subjects [334]")
    return info, [Block("pearson", "pearson", long), Block("spearman", "spearman", long)]


# ---- Szewczyk & Federmeier paper table, ONE study at a time ---------------
def _szewczyk_study_handler(slug: str) -> Callable[[], tuple[dict, list[Block]]]:
    """Split-half subject reliability of the item-mean target the embedding track predicts
    (baseline-regressed N400, neuroai.datasets.szewczyk_paper) for one of the five separate
    experiments in szewczyk_2022.tsv (they are not pooled)."""
    def handler() -> tuple[dict, list[Block]]:
        from neuroai.datasets.szewczyk_paper import STUDIES, TSV, load_study

        df = load_study(REPO_ROOT / TSV, slug)
        long = pd.DataFrame(dict(subject=df["Subject"].astype(str), item=df["item"], unit="_all",
                                 value=df["n400_resid"].astype(float)))
        info = dict(name=f"Szewczyk N400 EEG ({STUDIES[slug]})", modality="EEG",
                    metric="paper_n400_item_correlation",
                    note="baseline-regressed paper N400 (300-500 ms, 4 centro-parietal ROI clusters); "
                         f"item=(context, critical word); split subjects [{df['Subject'].nunique()}]")
        return info, [Block("pearson", "pearson", long), Block("spearman", "spearman", long)]
    return handler


# ---- Michaelov 2024 / Ryskin 2021 N400 from the paper's own tables --------
def _n400_paper_blocks(df: pd.DataFrame, item: pd.Series) -> list[Block]:
    """Per-electrode blocks (units = electrodes, headline = mean over electrodes) plus the
    electrode-averaged variant (one unit ``_all``), for both Pearson and Spearman."""
    long = pd.DataFrame(dict(subject=df["Subject"].astype(str), item=item,
                             unit=df["Electrode"].astype(str), value=df["N400"].astype(float)))
    long = long.dropna(subset=["value"])
    avg = (long.groupby(["subject", "item"], as_index=False)["value"].mean().assign(unit="_all"))
    return [Block("pearson", "pearson", long), Block("spearman", "spearman", long),
            Block("pearson_elecmean", "pearson", avg), Block("spearman_elecmean", "spearman", avg)]


def handler_michaelov_paper() -> tuple[dict, list[Block]]:
    df = pd.read_csv(DATA_ROOT / "n400_paper_formula/osfstorage/datasets/michaelov_2024.tsv", sep="\t")
    df = df.dropna(subset=["N400", "Subject", "Electrode", "ContextCode", "Condition"])
    item = df["ContextCode"].astype(str) + "|" + df["Condition"].astype(str)
    info = dict(name="Michaelov N400 EEG (paper N400)", modality="EEG",
                metric="paper_n400_item_correlation",
                note="paper's per-trial N400 per electrode; item=(ContextCode,Condition) [500]; split subjects [50]")
    return info, _n400_paper_blocks(df, item)


def handler_ryskin_paper() -> tuple[dict, list[Block]]:
    df = pd.read_csv(DATA_ROOT / "n400_paper_formula/osfstorage/datasets/ryskin_2021.tsv", sep="\t")
    df = df.dropna(subset=["N400", "Subject", "Electrode", "Item", "Condition"])
    item = df["Item"].astype(str) + "|" + df["Condition"].astype(str)
    info = dict(name="Ryskin N400 EEG (paper N400)", modality="EEG",
                metric="paper_n400_item_correlation",
                note="paper's per-trial N400 per electrode; item=(Item,Condition) [640]; split subjects [24]")
    return info, _n400_paper_blocks(df, item)


# ---- Fedorenko2016 ECoG (Spearman) — subject-unique electrodes ------------
def handler_fedorenko2016() -> tuple[dict, list[Block]]:
    from neuroai.datasets.fedorenko2016 import Fedorenko2016Dataset
    ds = Fedorenko2016Dataset(str(DATA_ROOT / "fedorenko2016"))
    words = ds.load_word_table()
    elec = ds.load_electrodes()  # electrode, neuroid_id, subject_UID
    elec_cols = [c for c in words.columns if c.startswith("electrode_")]
    col2subj = {f"electrode_{int(r.electrode)}": int(r.subject_UID)
                for r in elec.itertuples()}
    # item axis = word order; group each subject's electrodes into a mean
    # response per word, then split subjects.  (Electrodes are subject-unique
    # and stimuli single-presentation, so a per-electrode split-half over
    # subjects is undefined; the subject-mean-electrode response is the valid
    # signal-reliability estimate at the modality level.)
    words = words.sort_values(["sentence_id", "word_num"]).reset_index(drop=True)
    words["item"] = np.arange(len(words))
    recs: list[dict] = []
    for col in elec_cols:
        subj = col2subj.get(col)
        if subj is None:
            continue
        vals = pd.to_numeric(words[col], errors="coerce").to_numpy()
        # z-score each electrode before pooling within subject
        s = np.nanstd(vals)
        if s > 1e-8:
            vals = (vals - np.nanmean(vals)) / s
        for it, v in zip(words["item"].to_numpy(), vals):
            recs.append(dict(subject=subj, item=int(it), unit="_all", value=v))
    df = pd.DataFrame.from_records(recs)
    # average electrodes within (subject, item)
    long = (df.groupby(["subject", "item", "unit"], as_index=False)["value"]
            .mean())
    info = dict(name="Fedorenko2016 ECoG", modality="iEEG/ECoG",
                metric="mean_best_pearsonr",
                note="subject-unique electrodes + single presentation -> "
                     "per-electrode split-half undefined; ceiling computed on "
                     "subject-averaged electrode response (headline only)")
    return info, [Block("", "pearson", long)]


# ---- Podcast iEEG (Spearman) — subject-unique electrodes, region-level ----
def handler_podcast(bids_root: str | None = None) -> tuple[dict, list[Block]]:
    import mne
    from neuroai.datasets import podcast as pod
    root = Path(bids_root) if bids_root else _find_podcast_root()
    study = pod.PodcastDataset(str(root))
    subjects = study.available_subjects()
    words = study.load_words()
    onset_col = "onset" if "onset" in words.columns else words.columns[0]
    onsets = pd.to_numeric(words[onset_col], errors="coerce").to_numpy(dtype=float)
    n_words = len(onsets)

    # per subject: word x region response, scattered into full word grid
    subj_region_resp: dict[str, pd.DataFrame] = {}
    for s in subjects:
        fif = pod.highgamma_fif_path(root, s)
        if not Path(fif).is_file():
            continue
        try:
            raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
            data = raw.get_data().astype(np.float32)
            sfreq = float(raw.info["sfreq"])
            rec_tmin = float(raw.first_time)
            epochs, valid, _et = pod.extract_word_epochs(
                data, sfreq, rec_tmin, onsets, 0.0, 0.5)
        except Exception as exc:
            logger.warning("podcast: %s failed: %s", s, exc)
            continue
        resp = np.nanmean(epochs, axis=2)  # (n_valid, n_ch)
        regions = _podcast_regions(root, s, raw.ch_names)
        # full (n_words, n_ch) then group columns by region
        full = np.full((n_words, resp.shape[1]), np.nan, dtype=np.float32)
        full[np.asarray(valid, dtype=bool)] = resp
        rdf = pd.DataFrame(full, columns=[f"c{i}" for i in range(full.shape[1])])
        reg_series = pd.Series(regions[:full.shape[1]])
        region_means = {}
        for reg, cols in rdf.T.groupby(reg_series.to_numpy()):
            region_means[reg] = rdf[cols.index].mean(axis=1)
        subj_region_resp[s] = pd.DataFrame(region_means)
    if len(subj_region_resp) < 2:
        raise RuntimeError("Podcast: <2 subjects with highgamma loaded")
    regions_all = sorted(set().union(*[set(d.columns) for d in subj_region_resp.values()]))
    subs = list(subj_region_resp)
    dense = np.full((n_words, len(regions_all), len(subs)), np.nan, dtype=np.float32)
    for si, s in enumerate(subs):
        d = subj_region_resp[s]
        for ri, reg in enumerate(regions_all):
            if reg in d.columns:
                dense[:, ri, si] = d[reg].to_numpy()
    info = dict(name="Podcast iEEG", modality="iEEG/ECoG", metric="mean_pearsonr",
                note="subject-unique electrodes -> ceiling at anatomical-region "
                     "level (regions shared across subjects); item=word (0-500ms "
                     "high-gamma mean)")
    return info, [Block("", "pearson", dense=dense, unit_labels=regions_all)]


def _podcast_regions(root: Path, subject: str, ch_names: list[str]) -> list[str]:
    """Region label per raw channel, matched by channel name to electrodes.tsv."""
    from neuroai.datasets import podcast as pod
    try:
        edf = pod.load_electrodes_tsv(root, subject)
        name_col = "name" if "name" in edf.columns else edf.columns[0]
        for col in ("region", "group", "hemisphere"):
            if col in edf.columns:
                m = dict(zip(edf[name_col].astype(str), edf[col].astype(str)))
                return [m.get(c, "unknown") for c in ch_names]
    except Exception:
        pass
    return ["all"] * len(ch_names)


def _find_podcast_root() -> Path:
    for cand in (DATA_ROOT / "Podcast", DATA_ROOT / "podcast"):
        if cand.exists():
            return cand
    raise FileNotFoundError("Podcast BIDS root not found under data/")


# ---- LPP fMRI (Pearson, voxelwise) — average-subject ceiling from ISC -------
def _loo_isc_to_avg_ceiling(r_loo: np.ndarray, n_avg: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-voxel leave-one-out ISC into the noise ceiling for a model
    that predicts the ``n_avg``-subject AVERAGE.

    Additive equal-variance model: a subject's voxel timecourse is T + e with
    single-subject reliability rho1 = Var(T)/(Var(T)+Var(e)); let u = Var(e)/Var(T).
    Leave-one-out ISC r = corr(subject, mean of the other n_avg-1) satisfies
        1/r^2 = (1+u)(1 + u/(n_avg-1))              (solve the quadratic for u),
    and the reliability of the n_avg-subject average (the model target) is
        rho_N = 1/(1 + u/n_avg),   ceiling_r = sqrt(rho_N).
    A perfect model predicting T reaches this correlation with the average.
    """
    r = np.asarray(r_loo, dtype=np.float64)
    ceiling = np.full(r.shape, np.nan)
    rho = np.full(r.shape, np.nan)
    N = float(n_avg)
    valid = np.isfinite(r) & (r > 0) & (r < 1)
    rv = r[valid]
    disc = N * N - 4.0 * (N - 1.0) * (1.0 - 1.0 / (rv * rv))   # >= 0 here
    u = (-N + np.sqrt(disc)) / 2.0
    u = np.clip(u, 0.0, None)
    rho_N = 1.0 / (1.0 + u / N)
    rho[valid] = rho_N
    ceiling[valid] = np.sqrt(rho_N)
    perfect = np.isfinite(r) & (r >= 1)          # already noise-free
    ceiling[perfect] = 1.0
    rho[perfect] = 1.0
    noise = np.isfinite(r) & (r <= 0)            # no reliable signal
    ceiling[noise] = 0.0
    rho[noise] = 0.0
    return ceiling, rho


def handler_lpp(isc_path: str | None = None, n_avg_subjects: int = 48,
                lang: str = "en") -> tuple[dict, list[Block]]:
    """LPP ceiling for the AVERAGE-SUBJECT target, read directly from the
    precomputed leave-one-out ISC (``data/lpp_fmri_extra/average_subject/isc/isc_{k}trials_<lang>.gz``,
    shape (k, n_vox)); no individual-subject volumes are reloaded. Per voxel we
    average the ISC over trials, then map leave-one-out ISC -> reliability of
    the n_avg-subject average -> ceiling_r = sqrt(reliability)."""
    import joblib
    p = Path(isc_path) if isc_path else (DATA_ROOT / "isc" / f"isc_10trials_{lang}.gz")
    if not p.is_file():
        raise FileNotFoundError(f"LPP ISC file not found: {p}")
    arr = np.asarray(joblib.load(p), dtype=np.float64)
    k = arr.shape[0] if arr.ndim == 2 else 1
    r_loo = np.nanmean(arr, axis=0) if arr.ndim == 2 else arr.ravel()
    ceiling, rho = _loo_isc_to_avg_ceiling(r_loo, n_avg_subjects)
    by_unit = pd.DataFrame(dict(
        unit=[f"vox{i}" for i in range(len(r_loo))],
        r_sh=r_loo,          # per-voxel leave-one-out ISC (single-subject scale)
        r_sb=rho,            # reliability of the n_avg-subject average (R^2 ceiling)
        ceiling_r=ceiling,   # sqrt(reliability) — compare model Pearson r to this
        ceiling_r2=rho,
        n_subjects=n_avg_subjects,
        n_items=k,
    ))
    info = dict(name="LPP fMRI", modality="fMRI", metric="mean_pearson (voxelwise)",
                note=f"PEARSON (only dataset without a Spearman score); "
                     f"average-subject target: ceiling = reliability of the "
                     f"{n_avg_subjects}-subject average derived from precomputed "
                     f"leave-one-out ISC ({p.name}, {len(r_loo)} voxels); "
                     f"headline = mean over all in-mask voxels")
    return info, [Block("", "pearson", precomputed=by_unit)]


# ---- Broderick EEG (Pearson) — continuous, 128ch shared -------------------
def handler_broderick(bids_root: str | None = None,
                      max_subjects: int | None = None) -> tuple[dict, list[Block]]:
    from neuroai.datasets import broderick as bro
    from neuroai.datasets.podcast import extract_word_epochs
    root = Path(bids_root) if bids_root else (DATA_ROOT / "Broderick")
    words = bro.load_all_words(root)  # word_idx, word, onset, run, segment_id
    run_col = "run" if "run" in words.columns else "segment_id"
    runs = sorted(words[run_col].unique())
    subjects = _broderick_subjects(root)
    if max_subjects:
        subjects = subjects[:max_subjects]
    n_words = int(words["word_idx"].max()) + 1

    common_ch: list[str] | None = None
    subj_resp: dict[str, np.ndarray] = {}      # subj -> (n_words, n_ch)
    for s in subjects:
        full = np.full((n_words, 128), np.nan, dtype=np.float32)
        ch_ref: list[str] | None = None
        got = False
        for run in runs:
            w = words[words[run_col] == run]
            onsets = pd.to_numeric(w["onset"], errors="coerce").to_numpy(dtype=float)
            widx = w["word_idx"].to_numpy()
            try:
                eeg, sf, chans = bro.load_eeg_run(root, s, int(run))
            except Exception:
                continue
            eeg = np.asarray(eeg, dtype=np.float32)
            if eeg.ndim != 2:
                continue
            if eeg.shape[0] != len(chans) and eeg.shape[1] == len(chans):
                eeg = eeg.T
            epochs, valid, _et = extract_word_epochs(eeg, float(sf), 0.0,
                                                     onsets, 0.0, 0.4)
            resp = np.nanmean(epochs, axis=2)  # (n_valid, n_ch)
            if ch_ref is None:
                ch_ref = list(chans)
                full = np.full((n_words, len(ch_ref)), np.nan, dtype=np.float32)
            vw = widx[np.asarray(valid, dtype=bool)]
            full[vw, :resp.shape[1]] = resp
            got = True
        if got:
            subj_resp[s] = full
            if common_ch is None and ch_ref is not None:
                common_ch = ch_ref
    if len(subj_resp) < 2:
        raise RuntimeError("Broderick: <2 subjects with EEG loaded")
    subs = list(subj_resp)
    n_ch = min(v.shape[1] for v in subj_resp.values())
    dense = np.stack([subj_resp[s][:, :n_ch] for s in subs], axis=2)
    labels = (common_ch or list(range(n_ch)))[:n_ch]
    info = dict(name="Broderick EEG", modality="EEG", metric="best-layer Pearson r",
                note="128-ch shared montage; word-epoched (0-400ms) response; "
                     "item=word; split subjects")
    return info, [Block("", "pearson", dense=dense, unit_labels=labels)]


def _broderick_subjects(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob("sub-*") if p.is_dir())


# ---- ZuCo Reading (behavioral TRT, pooled SR/NR/TSR) -----------------------
def handler_zuco_reading(tasks: tuple[str, ...] = ("SR", "NR", "TSR")) -> tuple[dict, list[Block]]:
    # Pooled across all three tasks to match the model sweep (zuco_reading_lme /
    # zuco_reading_embed each run SR, NR, TSR as separate configs, but the
    # dataset's own reliability is naturally assessed over the full item pool).
    from neuroai.datasets.zuco_reading import ZuCoReadingDataset
    mat_root = _find_zuco_root()
    frames = []
    for task in tasks:
        try:
            trials = ZuCoReadingDataset(mat_root, task, aggregate=False).load_words()
        except Exception as exc:
            logger.warning("zuco_reading: task %s failed: %s", task, exc)
            continue
        trials = trials.copy()
        trials["story_id"] = task + "|" + trials["story_id"].astype(str)
        frames.append(trials)
    trials = pd.concat(frames, ignore_index=True)
    return _reading_time_handler(
        "ZuCo Reading", trials,
        "total reading time (TRT) per word; split subjects (tasks=SR/NR/TSR pooled)")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[[], tuple[dict, list[Block]]]] = {
    "szewczyk": handler_szewczyk,
    "naturalstories_spr": handler_naturalstories_spr,
    "geco": handler_geco,
    "meco": handler_meco,
    "provo": handler_provo,
    "ucl_reading": handler_ucl_reading,
    "zuco_reading": handler_zuco_reading,
    "tuckute": handler_tuckute,
    "ns_fmri": handler_ns_fmri,
    "nieuwland": handler_nieuwland,
    "nieuwland_paper": handler_nieuwland_paper,
    "szewczyk_f2007": _szewczyk_study_handler("f2007"),
    "szewczyk_h2019": _szewczyk_study_handler("h2019"),
    "michaelov_paper": handler_michaelov_paper,
    "ryskin_paper": handler_ryskin_paper,
    "fedorenko2016": handler_fedorenko2016,
    "podcast": handler_podcast,
    "lpp": handler_lpp,
    "broderick": handler_broderick,
    "onestop": handler_onestop,
    "pereira": handler_pereira,
}

# Light (item-structured / tabular) vs heavy (MNE / nilearn / .fif loading).
LIGHT = {"szewczyk", "naturalstories_spr", "tuckute", "ns_fmri",
         "fedorenko2016", "geco", "provo", "ucl_reading",
         "pereira", "nieuwland_paper", "michaelov_paper", "ryskin_paper",
         "szewczyk_f2007", "szewczyk_h2019"}  # xarray .nc load, no MNE/nilearn
HEAVY = {"nieuwland", "podcast", "lpp", "broderick",
         "meco",  # meco reads a large .rda across many subjects
         "onestop"}  # 360 subjects x 3 measures, each a full reload+modal-word pass


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_dataset(slug: str, args) -> tuple[list[dict], list[dict], dict]:
    """Return (headline_rows, by_unit_rows, meta) for one dataset."""
    handler = HANDLERS[slug]
    logger.info("=== %s ===", slug)
    info, blocks = handler()
    headline_rows: list[dict] = []
    by_unit_rows: list[dict] = []
    for block in blocks:
        by_unit, headline = compute_block_ceiling(
            block, n_splits=args.n_splits, seed=args.seed,
            min_items=args.min_items, max_units=args.max_units,
            method=args.method,
        )
        for r in by_unit.to_dict("records"):
            r.update(dataset=slug, component=block.component)
            by_unit_rows.append(r)
        headline.update(dataset=slug, name=info["name"], modality=info["modality"],
                        metric=info["metric"])
        headline_rows.append(headline)
        logger.info("  [%s|%s] ceiling_r=%.3f (r_SB=%.3f) over %d units, %d subj",
                    slug, block.component or "-",
                    headline["ceiling_r"], headline["ceiling_r2"],
                    headline["n_units"], headline["n_subjects"])
    meta = dict(status="ok", **info)
    return headline_rows, by_unit_rows, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["all"],
                    help="dataset slug(s) or 'all'. Choices: " + ", ".join(HANDLERS))
    ap.add_argument("--out-dir",
                    default=str(REPO_ROOT / "outputs_jeanzay" / "noise_ceilings"),
                    help="output dir (default outputs_jeanzay/noise_ceilings — "
                         "outputs/ is auto-wiped on Jean Zay)")
    ap.add_argument("--method", choices=["split_half", "closed_form"],
                    default="split_half",
                    help="split_half: subject-split Monte-Carlo over --n-splits "
                         "(default; correct for any design). closed_form: analytic "
                         "Cronbach/Spearman-Brown reliability (fast, no loop) but "
                         "valid ONLY for fully-crossed designs (dense fMRI/EEG "
                         "timecourses); overestimates on sparse per-subject-subset "
                         "designs (podcast/fedorenko).")
    ap.add_argument("--n-splits", type=int, default=500,
                    help="random subject splits per unit (split_half method only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-items", type=int, default=5)
    ap.add_argument("--max-units", type=int, default=4000,
                    help="subsample units (voxels/channels) above this count")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="skip MNE/nilearn/.fif datasets (nieuwland, podcast, "
                         "lpp, broderick)")
    ap.add_argument("--rebuild-only", action="store_true",
                    help="do NOT recompute anything; just reassemble the combined "
                         "summary/by_unit CSVs from the per-dataset <slug>/ folders "
                         "already on disk. Run once after all per-dataset jobs finish "
                         "to fix a partial combined summary (e.g. parallel-run race).")
    args = ap.parse_args()

    if args.rebuild_only:
        slugs = []
    elif args.dataset == ["all"]:
        slugs = list(HANDLERS)
    else:
        slugs = args.dataset
    if args.skip_heavy:
        slugs = [s for s in slugs if s not in HEAVY]
    unknown = [s for s in slugs if s not in HANDLERS]
    if unknown:
        ap.error(f"unknown dataset(s): {unknown}. Choices: {list(HANDLERS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_headline: list[dict] = []
    all_by_unit: list[dict] = []
    meta_all: dict[str, dict] = {}

    for slug in slugs:
        try:
            hrows, urows, meta = run_dataset(slug, args)
        except Exception as exc:
            logger.error("%s FAILED: %s", slug, exc)
            logger.debug("%s", traceback.format_exc())
            meta_all[slug] = dict(status="failed", error=str(exc))
            continue
        meta_all[slug] = meta
        all_headline.extend(hrows)
        all_by_unit.extend(urows)
        ds_dir = out_dir / slug
        ds_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(hrows).to_csv(ds_dir / "ceiling_headline.csv", index=False)
        pd.DataFrame(urows).to_csv(ds_dir / "ceiling_by_unit.csv", index=False)

    # Rebuild combined files by globbing ALL per-dataset CSVs on disk, so
    # running datasets in separate invocations accumulates rather than
    # overwriting previously-computed datasets.
    head_parts = [pd.read_csv(p) for p in sorted(out_dir.glob("*/ceiling_headline.csv"))]
    unit_parts = [pd.read_csv(p) for p in sorted(out_dir.glob("*/ceiling_by_unit.csv"))]
    if head_parts:
        cols = ["dataset", "name", "modality", "metric", "component", "corr",
                "ceiling_r", "ceiling_r_sem", "ceiling_r2", "r_sh",
                "n_units", "n_subjects", "n_items", "units_subsampled", "method"]
        df = pd.concat(head_parts, ignore_index=True)
        df = df[[c for c in cols if c in df.columns]].sort_values(["modality", "dataset"])
        df.to_csv(out_dir / "noise_ceilings_summary.csv", index=False)
        if unit_parts:
            pd.concat(unit_parts, ignore_index=True).to_csv(
                out_dir / "noise_ceilings_by_unit.csv", index=False)
        logger.info("\n%s", df.to_string(index=False))

    # Merge meta with any previously-saved meta so status accumulates too.
    meta_path = out_dir / "noise_ceilings_meta.json"
    if meta_path.is_file():
        try:
            prev = json.loads(meta_path.read_text())
            prev.update(meta_all)
            meta_all = prev
        except Exception:
            pass
    with open(meta_path, "w") as fh:
        json.dump(meta_all, fh, indent=2, default=str)
    logger.info("Wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
