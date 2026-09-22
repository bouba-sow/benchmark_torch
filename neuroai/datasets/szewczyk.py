"""Szewczyk & Federmeier (2021) N400 EEG dataset adapter.

Loads the OSF deposit from ``data/szewczyk/osfstorage/``. The
deposit ships pre-processed R data (``.rds``), not raw EEG: 5 original
experiments have already been merged and the N400 amplitude has already been
averaged over a fixed 300-500 ms window and over 4 centro-parietal ROI
clusters (MiCe, MiPa, LMCe, RMCe) into a single scalar per trial. There is no
per-electrode spatial data in this deposit — unlike the earlier N400 datasetsNieuwland, this benchmark can only predict a scalar N400 amplitude, not a
topographic map.

Experimental design
--------------------
  5 merged sub-experiments ("dataset 1" .. "dataset 5"), ~138 subjects total,
  1,410 unique sentence contexts, ~1,900-2,000 unique (context, critical_word)
  items (many contexts have 2-3 distinct critical-word completions across
  subjects/lists — this is a continuous cloze-probability manipulation, not a
  small set of discrete conditions like Control/Sem/Synt).

  ``Subject`` IDs are already unique across all 5 sub-experiments (verified:
  0/138 subjects appear under more than one ``dataset`` value), though the
  naming convention differs per sub-experiment (``"dataset 1_01"``, plain
  ``"bcr01"``, hashed ``"v1B_IDYSC"``, ...) — so no collision risk when
  pooling all subjects, but don't rely on the string format to infer
  sub-experiment membership; join on ``subexperiment`` instead.
  ``Item`` numbers are only unique *within* a sub-experiment, so the
  GroupKFold group key is ``(subexperiment, item)``, not ``item`` alone.

Reference
---------
Szewczyk, J. M., & Federmeier, K. D. (2021). Context-based facilitation of
semantic access follows both spatiotemporal and cardinal principles. eLife.
OSF deposit: https://osf.io/w96nc
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SZEWCZYK_N400_TMIN: float = 0.300
SZEWCZYK_N400_TMAX: float = 0.500

# Median-split threshold for the secondary "expected vs unexpected" scaling
# curves (not used as a condition boundary in the primary continuous metric).
SZEWCZYK_EXPECTED_CLOZE_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(rds_path: str | Path) -> pd.DataFrame:
    """Load ``bind-5-dset_300-500.rds`` and add derived columns.

    Returns
    -------
    pd.DataFrame with one row per (subexperiment, subject, item, critical_word)
    trial. Added columns: ``context`` (renamed from ``Sentence``),
    ``critical_word`` (renamed from ``word``), ``subject``, ``item``,
    ``subexperiment`` (renamed from ``dataset``), ``n400_z`` (z-scored per
    subject).
    """
    import pyreadr

    result = pyreadr.read_r(str(rds_path))
    df = result[None].copy()

    df = df.rename(columns={
        "Sentence": "context",
        "word":     "critical_word",
        "Subject":  "subject",
        "Item":     "item",
        "dataset":  "subexperiment",
    })
    for col in ("context", "critical_word", "subject", "item", "subexperiment"):
        df[col] = df[col].astype(str)

    # z-score N400 per subject (there is only one collapsed ROI here, so this
    # is the natural analogue of the earlier N400 datasets's per-(subject, electrode)
    # normalisation — it reduces to per-subject normalisation with 1 output).
    grp = df.groupby("subject")["n400"]
    df["n400_z"] = (
        (df["n400"] - grp.transform("mean"))
        / grp.transform("std").clip(lower=1e-8)
    ).astype(np.float32)

    logger.info(
        "Szewczyk: %d rows | %d subjects | %d subexperiments | %d (subexperiment,item) frames",
        len(df),
        df["subject"].nunique(),
        df["subexperiment"].nunique(),
        df[["subexperiment", "item"]].drop_duplicates().shape[0],
    )
    return df


# ---------------------------------------------------------------------------
# Items table
# ---------------------------------------------------------------------------

def build_items_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build the unique-items table — one row per (subexperiment, item, critical_word).

    Each row has: subexperiment, item, critical_word, context, cloze_p,
    item_id (0-based row index into the LLM feature matrix), frame_id
    (integer encoding of (subexperiment, item), used as the GroupKFold group
    key so every critical-word completion of the same sentence frame stays in
    the same CV fold).
    """
    cols = ["subexperiment", "item", "critical_word", "context", "cloze_p"]
    items = (
        df[cols]
        .drop_duplicates(subset=["subexperiment", "item", "critical_word"])
        .sort_values(["subexperiment", "item", "critical_word"])
        .reset_index(drop=True)
    )
    items["item_id"] = items.index

    frames = sorted(set(zip(items["subexperiment"], items["item"])))
    frame_to_int = {f: i for i, f in enumerate(frames)}
    items["frame_id"] = [
        frame_to_int[(se, it)] for se, it in zip(items["subexperiment"], items["item"])
    ]
    items["frame_id"] = items["frame_id"].astype(np.int64)

    logger.info(
        "Szewczyk items: %d unique (subexperiment, item, critical_word) rows | %d frames",
        len(items), len(frames),
    )
    return items


def get_subjects(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique subject IDs (already namespaced per subexperiment)."""
    return sorted(df["subject"].unique().tolist())


# ---------------------------------------------------------------------------
# Per-subject scalar Y
# ---------------------------------------------------------------------------

def build_subject_items(
    df: pd.DataFrame,
    subject: str,
    items: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract one subject's data aligned to the items feature matrix.

    Parameters
    ----------
    df      : full long-form DataFrame from ``load_data``
    subject : subject ID string (e.g. "dataset 1_01")
    items   : unique-items table from ``build_items_df``

    Returns
    -------
    item_indices : (n_sub,) int64 — indices into the items feature matrix
    Y_sub        : (n_sub, 1) float32 — z-scored N400 for this subject
    groups_sub   : (n_sub,) int64 — frame_id for GroupKFold
    cloze_sub    : (n_sub,) float32 — cloze probability for each trial
                   (for the continuous / median-split secondary analyses)
    """
    sub_df = df[df["subject"] == subject]

    sub_pairs = (
        sub_df[["subexperiment", "item", "critical_word", "n400_z"]]
        .merge(
            items[["subexperiment", "item", "critical_word", "item_id", "frame_id", "cloze_p"]],
            on=["subexperiment", "item", "critical_word"],
            how="inner",
        )
        .sort_values("item_id")
        .reset_index(drop=True)
    )

    item_indices = sub_pairs["item_id"].values.astype(np.int64)
    groups_sub   = sub_pairs["frame_id"].values.astype(np.int64)
    Y_sub        = sub_pairs["n400_z"].values.reshape(-1, 1).astype(np.float32)
    cloze_sub    = sub_pairs["cloze_p"].values.astype(np.float32)

    return item_indices, Y_sub, groups_sub, cloze_sub
