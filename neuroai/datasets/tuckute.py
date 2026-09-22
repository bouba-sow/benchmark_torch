"""Tuckute et al. (2023) fMRI dataset loader.

Paper: "Driving and suppressing the human language network using large language
models" — Nature Human Behaviour (2023). https://doi.org/10.1038/s41562-023-01783-7

Dataset: 2,000 isolated 6-word sentences × 10 participants × 6 left-hemisphere
language ROIs.  BOLD responses are sentence-level (no HRF needed).

Three conditions:
  B  — 1,000 naturalistic baseline sentences (all 10 participants)
  D  —   500 sentences that drive   LH language network (2–3 participants each)
  S  —   500 sentences that suppress LH language network (2–3 participants each)

Data layout expected under ``data_root``::

    data_root/
      data/
        brain-lang-data_participant_20230728.csv
        NC-allroi-data.csv
      data.tar          # alternative: CSV files inside the tar

Usage::

    ds = TuckuteDataset(Path("data/tuckute_fmri/osfstorage"))
    sentences, bold, item_ids = ds.get_condition("B")
    # sentences : list[str]  (n_sentences,)
    # bold      : np.ndarray (n_sentences, n_rois)
    # item_ids  : np.ndarray (n_sentences,)  — stable row ordering

    ds.rois          # ['lang_LH_IFGorb', 'lang_LH_IFG', ...]
    ds.noise_ceiling # {'lang_LH_netw': 0.559, ...}
"""

from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path
from functools import cached_property

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical ROI order (matches paper Table 1 + column order in NC file)
LANG_ROIS: list[str] = [
    "lang_LH_netw",
    "lang_LH_IFGorb",
    "lang_LH_IFG",
    "lang_LH_MFG",
    "lang_LH_AntTemp",
    "lang_LH_PostTemp",
]

CONDITIONS: list[str] = ["B", "D", "S"]


# ---------------------------------------------------------------------------
# Low-level CSV readers (handles both extracted dir and in-tar)
# ---------------------------------------------------------------------------

def _read_csv_from_dir_or_tar(data_root: Path, filename: str) -> pd.DataFrame:
    """Read ``filename`` from ``data_root/data/`` or from ``data_root/data.tar``."""
    direct = data_root / "data" / filename
    if direct.is_file():
        return pd.read_csv(direct)

    tar_path = data_root / "data.tar"
    if tar_path.is_file():
        member = f"data/{filename}"
        with tarfile.open(tar_path, "r") as tf:
            try:
                f = tf.extractfile(member)
            except KeyError:
                raise FileNotFoundError(
                    f"{member} not found inside {tar_path}"
                ) from None
            if f is None:
                raise FileNotFoundError(f"Cannot extract {member} from {tar_path}")
            return pd.read_csv(io.BytesIO(f.read()))

    raise FileNotFoundError(
        f"Cannot find {filename}: tried {direct} and inside {data_root / 'data.tar'}"
    )


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class TuckuteDataset:
    """Participant-averaged sentence-level BOLD from Tuckute et al. (2023).

    Parameters
    ----------
    data_root:
        Directory that contains either ``data/`` (extracted) or ``data.tar``.
        Typically ``data/tuckute_fmri/osfstorage``.
    rois:
        Subset of ROIs to use (default: all 6 ``LANG_ROIS``).
    """

    def __init__(
        self,
        data_root: Path | str,
        rois: list[str] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.rois: list[str] = rois if rois is not None else list(LANG_ROIS)
        self._validate_rois()
        logger.info("Loading TuckuteDataset from %s", self.data_root)

    def _validate_rois(self) -> None:
        unknown = [r for r in self.rois if r not in LANG_ROIS]
        if unknown:
            raise ValueError(f"Unknown ROIs: {unknown}. Valid: {LANG_ROIS}")

    # ------------------------------------------------------------------
    # Raw data (loaded once, cached)
    # ------------------------------------------------------------------

    @cached_property
    def _brain_df(self) -> pd.DataFrame:
        df = _read_csv_from_dir_or_tar(
            self.data_root, "brain-lang-data_participant_20230728.csv"
        )
        # Keep only the 6 language ROIs we care about
        df = df[df["roi"].isin(LANG_ROIS)].copy()
        logger.info(
            "Loaded brain data: %d rows, %d participants, %d sentences",
            len(df),
            df["target_UID"].nunique(),
            df["item_id"].nunique(),
        )
        return df

    @cached_property
    def _nc_df(self) -> pd.DataFrame:
        return _read_csv_from_dir_or_tar(self.data_root, "NC-allroi-data.csv")

    # ------------------------------------------------------------------
    # Derived structures (cached)
    # ------------------------------------------------------------------

    @cached_property
    def _sentences_df(self) -> pd.DataFrame:
        """One row per unique sentence: item_id, sentence, cond."""
        return (
            self._brain_df[["item_id", "sentence", "cond"]]
            .drop_duplicates("item_id")
            .sort_values("item_id")
            .reset_index(drop=True)
        )

    @cached_property
    def _bold_avg(self) -> dict[str, pd.DataFrame]:
        """Participant-averaged BOLD per condition.

        Returns dict: cond → DataFrame with item_id as index, ROIs as columns.
        """
        avg: dict[str, pd.DataFrame] = {}
        for cond in CONDITIONS:
            cond_df = self._brain_df[self._brain_df["cond"] == cond]
            # Average over participants for each (item_id, roi)
            pivot = (
                cond_df.groupby(["item_id", "roi"])["response_target"]
                .mean()
                .unstack("roi")   # → (item_id, roi)
            )
            # Reorder columns to canonical ROI order, keep only requested ROIs
            pivot = pivot[[r for r in self.rois if r in pivot.columns]]
            avg[cond] = pivot
        return avg

    @cached_property
    def noise_ceiling(self) -> dict[str, float]:
        """Pearson r noise ceiling per ROI (from NC-allroi-data.csv, B condition)."""
        nc: dict[str, float] = {}
        nc_df = self._nc_df
        for roi in self.rois:
            row = nc_df[nc_df["Unnamed: 0"] == roi]
            if not row.empty:
                nc[roi] = float(row["nc"].iloc[0])
            else:
                nc[roi] = float("nan")
        return nc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_condition(
        self,
        cond: str,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Return sentences, BOLD, and item_ids for one condition.

        Parameters
        ----------
        cond : ``'B'``, ``'D'``, ``'S'``, or ``'all'``

        Returns
        -------
        sentences : list[str]  length n_sentences
        bold      : np.ndarray (n_sentences, n_rois)  participant-averaged
        item_ids  : np.ndarray (n_sentences,) int     stable ordering key
        """
        if cond == "all":
            parts = [self.get_condition(c) for c in CONDITIONS]
            sentences = sum((s for s, _, _ in parts), [])
            bold = np.concatenate([b for _, b, _ in parts], axis=0)
            item_ids = np.concatenate([i for _, _, i in parts], axis=0)
            return sentences, bold, item_ids

        if cond not in CONDITIONS:
            raise ValueError(f"cond must be one of {CONDITIONS + ['all']}, got {cond!r}")

        bold_df = self._bold_avg[cond]
        item_ids = bold_df.index.to_numpy(dtype=int)

        # Map item_ids → sentences in the same order
        sent_map = self._sentences_df.set_index("item_id")["sentence"]
        sentences = [str(sent_map.loc[i]) for i in item_ids]

        bold = bold_df.to_numpy(dtype=np.float64)
        return sentences, bold, item_ids

    @property
    def all_sentences(self) -> list[str]:
        """All 2,000 sentences in item_id order (B then D then S)."""
        sents, _, _ = self.get_condition("all")
        return sents

    def condition_mask(self, cond: str) -> np.ndarray:
        """Boolean mask into ``all_sentences`` for a given condition.

        Useful to index precomputed embeddings (extracted on all_sentences)
        into a single condition.
        """
        _, _, all_ids = self.get_condition("all")
        _, _, cond_ids = self.get_condition(cond)
        cond_set = set(cond_ids.tolist())
        return np.array([i in cond_set for i in all_ids])

    def get_condition_per_subject(
        self,
        cond: str,
    ) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
        """Per-subject (sentences, bold, item_ids) for one condition.

        Parameters
        ----------
        cond : ``'B'``, ``'D'``, ``'S'``, or ``'all'``

        Returns
        -------
        dict mapping subject_id (str) →
            sentences : list[str]           sentences that subject saw
            bold      : np.ndarray (n, n_rois)  that subject's BOLD
            item_ids  : np.ndarray (n,) int
        """
        if cond == "all":
            # Merge each subject's data across all base conditions
            per_cond = {c: self.get_condition_per_subject(c) for c in CONDITIONS}
            all_subjects: set[str] = set()
            for d in per_cond.values():
                all_subjects.update(d.keys())
            result: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
            for subj in sorted(all_subjects):
                sents_all, bold_parts, ids_all = [], [], []
                for c in CONDITIONS:
                    if subj in per_cond[c]:
                        s, b, i = per_cond[c][subj]
                        sents_all.extend(s)
                        bold_parts.append(b)
                        ids_all.append(i)
                if bold_parts:
                    result[subj] = (
                        sents_all,
                        np.concatenate(bold_parts, axis=0),
                        np.concatenate(ids_all, axis=0),
                    )
            return result

        if cond not in CONDITIONS:
            raise ValueError(f"cond must be one of {CONDITIONS + ['all']}, got {cond!r}")

        sent_map = self._sentences_df.set_index("item_id")["sentence"]
        cond_df  = self._brain_df[self._brain_df["cond"] == cond]
        result = {}

        for subj, subj_df in cond_df.groupby("target_UID"):
            pivot = (
                subj_df.groupby(["item_id", "roi"])["response_target"]
                .mean()
                .unstack("roi")
            )
            pivot = pivot[[r for r in self.rois if r in pivot.columns]]
            item_ids = pivot.index.to_numpy(dtype=int)
            sentences = [str(sent_map.loc[i]) for i in item_ids]
            result[str(subj)] = (sentences, pivot.to_numpy(dtype=np.float64), item_ids)

        return result

    @property
    def subjects(self) -> list[str]:
        """All subject IDs in the dataset."""
        return sorted(self._brain_df["target_UID"].unique().astype(str).tolist())

    def n_sentences(self, cond: str = "all") -> int:
        _, _, ids = self.get_condition(cond)
        return len(ids)
