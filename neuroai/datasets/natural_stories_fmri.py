"""Natural Stories fMRI dataset adapter.

Wraps the Shain et al. (2020) fMRI corpus:
  78 participants listening to 8 short English narratives (~1000 words each).
  BOLD measured at TR = 2 s, pre-averaged per functionally-defined ROI (fROI).
  Language network (LANG): 6 left-hemisphere fROIs (Fedorenko lab parcels).
  Multiple-demand network (MD): 20 bilateral fROIs.

Data root: data/naturalstories_fmri/osfstorage/
  LANG_MD_ts_X.csv    — word-level predictors (one row per word × subject × fROI)
  LANG_MD_y_train.csv — BOLD at scan TR, training split
  LANG_MD_y_dev.csv   — BOLD at scan TR, dev split
  LANG_MD_y_test.csv  — BOLD at scan TR, test split
  (LANG_* and MD_* contain the same data split by network)

Reference
---------
Shain, C., Blank, I. A., van Schijndel, M., Schuler, W., & Fedorenko, E. (2020).
fMRI reveals language-specific predictive coding during naturalistic sentence
comprehension. Neuropsychologia, 138, 107307.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NS_FMRI_TR: float = 2.0        # repetition time in seconds
NS_FMRI_WORD_DUR: float = 0.1  # fixed event duration for HRF convolution

NS_FMRI_DOCS: list[str] = [
    "Aqua", "Boar", "Elvis", "HighSchool",
    "KingOfBirds", "MatchstickSeller", "MrSticky", "Tulips",
]

NS_LANG_FROIS: list[str] = [
    "LangLAngG", "LangLAntTemp", "LangLIFG",
    "LangLIFGorb", "LangLMFG", "LangLPostTemp",
]

# ---------------------------------------------------------------------------
# Cached full-table loaders (loaded once per process, shared across subjects)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_x_table(data_root: str, froi_table: str) -> pd.DataFrame:
    path = Path(data_root) / f"{froi_table}_ts_X.csv"
    if not path.exists():
        raise FileNotFoundError(f"X table not found: {path}")
    df = pd.read_csv(path, sep=" ")
    logger.info("[NS-fMRI] loaded X table %s (%d rows)", path.name, len(df))
    return df


@lru_cache(maxsize=4)
def _load_y_table(data_root: str, froi_table: str) -> pd.DataFrame:
    """Concatenate train / dev / test BOLD tables into one DataFrame."""
    root = Path(data_root)
    parts: list[pd.DataFrame] = []
    for split in ("train", "dev", "test"):
        p = root / f"{froi_table}_y_{split}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Y table not found: {p}")
        parts.append(pd.read_csv(p, sep=" "))
    df = pd.concat(parts, ignore_index=True)
    logger.info(
        "[NS-fMRI] loaded Y table %s_y_*.csv (%d rows total)", froi_table, len(df)
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def available_subjects(data_root: str, froi_table: str = "LANG_MD") -> list[str]:
    """Return sorted list of subject IDs present in the Y table."""
    y = _load_y_table(data_root, froi_table)
    return sorted(y["subject"].unique().tolist())


def available_frois(data_root: str, froi_table: str = "LANG_MD") -> list[str]:
    """Return sorted list of fROI names present in the Y table."""
    y = _load_y_table(data_root, froi_table)
    return sorted(y["fROI"].unique().tolist())


def load_canonical_words(
    data_root: str,
    doc: str,
    froi_table: str = "LANG_MD",
) -> pd.DataFrame:
    """Word list + onset times for one document (shared across subjects).

    Uses the first subject × fROI in the X table — content is identical for
    all subjects.

    Returns
    -------
    pd.DataFrame with columns:
      word     str   — surface form
      sentid   int   — sentence index
      sentpos  int   — position within sentence (1-based)
      time     float — word onset in seconds from story start
      wlen     int   — character length
    """
    x = _load_x_table(data_root, froi_table)
    doc_x = x[x["docid"] == doc]
    if doc_x.empty:
        raise ValueError(f"Document '{doc}' not found in {froi_table}_ts_X.csv")

    # One representative (subject, fROI) — words are identical across subjects
    s0 = doc_x["subject"].iloc[0]
    f0 = doc_x["fROI"].iloc[0]
    canonical = (
        doc_x[(doc_x["subject"] == s0) & (doc_x["fROI"] == f0)]
        [["word", "sentid", "sentpos", "time", "wlen"]]
        .sort_values("time")
        .reset_index(drop=True)
    )
    logger.debug(
        "[NS-fMRI] canonical words for %s: %d words (%.1f–%.1f s)",
        doc, len(canonical), canonical["time"].min(), canonical["time"].max(),
    )
    return canonical


def load_bold_subject(
    data_root: str,
    subject: str,
    froi_table: str = "LANG_MD",
    docs: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str], list[str]]]:
    """Load BOLD time series for one subject across all (or specified) documents.

    Parameters
    ----------
    data_root   : path to osfstorage/
    subject     : subject ID (e.g. "s061")
    froi_table  : "LANG", "MD", or "LANG_MD"
    docs        : subset of NS_FMRI_DOCS, or None for all

    Returns
    -------
    dict mapping doc_name → (bold, scan_times, froi_names, networks)
      bold       : float64 array (n_TRs, n_fROIs)
      scan_times : float64 array (n_TRs,) — seconds from story start
      froi_names : list[str] of length n_fROIs
      networks   : list[str] — "LANG" or "MD" per fROI
    """
    y = _load_y_table(data_root, froi_table)
    subj_y = y[y["subject"] == subject]
    if subj_y.empty:
        raise ValueError(f"Subject '{subject}' not found in {froi_table}_y_*.csv")

    target_docs = docs or NS_FMRI_DOCS
    result: dict[str, tuple[np.ndarray, np.ndarray, list[str], list[str]]] = {}

    for doc in target_docs:
        doc_y = subj_y[subj_y["docid"] == doc]
        if doc_y.empty:
            logger.warning("[NS-fMRI] %s has no BOLD for doc=%s", subject, doc)
            continue

        # Pivot: rows = TRs, columns = fROIs (sorted alphabetically for reproducibility)
        pivoted = (
            doc_y[["time", "fROI", "BOLD", "network"]]
            .sort_values(["time", "fROI"])
        )
        froi_names = sorted(pivoted["fROI"].unique().tolist())
        scan_times_list: list[float] = sorted(pivoted["time"].unique().tolist())
        n_trs = len(scan_times_list)
        n_frois = len(froi_names)

        bold = np.full((n_trs, n_frois), np.nan, dtype=np.float64)
        froi_idx = {f: j for j, f in enumerate(froi_names)}
        time_idx = {t: i for i, t in enumerate(scan_times_list)}

        for _, row in pivoted.iterrows():
            i = time_idx[row["time"]]
            j = froi_idx[row["fROI"]]
            bold[i, j] = row["BOLD"]

        # Network label per fROI (stable: LANG=1, MD=0 in the data)
        net_map = (
            doc_y[["fROI", "network"]]
            .drop_duplicates()
            .set_index("fROI")["network"]
            .to_dict()
        )
        networks = ["LANG" if net_map.get(f, 0) == 1 else "MD" for f in froi_names]

        result[doc] = (
            bold,
            np.array(scan_times_list, dtype=np.float64),
            froi_names,
            networks,
        )

    return result


# ---------------------------------------------------------------------------
# HRF convolution helper (word-onset events → scan-time design matrix)
# ---------------------------------------------------------------------------

def word_hrf_basis_matrix(
    onsets: np.ndarray,
    durations: np.ndarray,
    frame_times: np.ndarray,
    hrf_model: str = "glover",
) -> np.ndarray:
    """Basis matrix B of shape (n_scans, n_words).

    Column w is the HRF regressor for word w at unit amplitude.
    For linear HRF models, ``design = B @ features`` gives the
    (n_scans, n_features) HRF-convolved design matrix without looping
    over feature dimensions.

    Requires nilearn (``compute_regressor``).
    """
    from nilearn.glm.first_level import compute_regressor

    onsets = np.asarray(onsets, dtype=np.float64).ravel()
    durations = np.asarray(durations, dtype=np.float64).ravel()
    n_words = len(onsets)
    n_scans = len(frame_times)
    B = np.zeros((n_scans, n_words), dtype=np.float64)
    for w in range(n_words):
        exp_condition = np.asarray([[onsets[w]], [durations[w]], [1.0]])
        signal, _ = compute_regressor(exp_condition, hrf_model, frame_times)
        B[:, w] = signal[:, 0]
    return B
