"""ZuCo dataset adapter.

ZuCo: Zurich Cognitive Language Processing Corpus.
  - 12 subjects, 3 reading tasks (silent word-by-word presentation)
  - Task 1 (SR):  400 sentences — movie review sentiment
  - Task 2 (NR):  300 sentences — Wikipedia normal reading
  - Task 3 (TSR): 407 sentence pairs — task-specific relation reading
  - 105-channel EGI EEG at 500 Hz
  - Pre-extracted per-word frequency band features (mean amplitude per electrode)
  - Eye-tracking: FFD, TRT, GD, GPT, SFD per word

EEG feature naming convention (each field is shape (105,)):
  {ET}_{band}{window}
    ET     : TRT (total reading time), FFD (first fixation), GD, GPT, SFD
    band   : t (theta 4–8 Hz), a (alpha 8–13 Hz), b (beta 13–30 Hz), g (gamma 30–49 Hz)
    window : 1 (0–500 ms), 2 (500–1000 ms) relative to word onset

We use TRT-gated features (total reading time) as the default EEG target, as they
capture the full processing window and are most commonly used in the literature.

Data path structure:
  {mat_root}/
    task1- SR/Matlab files/resultsZ{SUBJ}_SR.mat    (12 subjects)
    task2 - NR/Matlab files/resultsZ{SUBJ}_NR.mat
    task3 - TSR/Matlab files/resultsZ{SUBJ}_TSR.mat

Reference
---------
Hollenstein et al. (2018). ZuCo, a simultaneous EEG and eye-tracking resource
for natural sentence reading. Scientific Data 5:180291.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

logger = logging.getLogger(__name__)

ZUCO_N_CHANNELS: int = 105
ZUCO_SFREQ: float = 500.0   # Hz (raw EEG; band features are already averaged)
ZUCO_SUBJECTS: list[str] = [
    "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN",
    "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH",
]
ZUCO_TASKS: list[str] = ["SR", "NR", "TSR"]
ZUCO_BANDS: list[str] = ["t1", "a1", "b1", "g1"]  # 0-500 ms window, all 4 bands
ZUCO_BAND_NAMES: dict[str, str] = {
    "t1": "theta", "t2": "theta2",
    "a1": "alpha", "a2": "alpha2",
    "b1": "beta",  "b2": "beta2",
    "g1": "gamma", "g2": "gamma2",
}

# Directory and file-suffix per task
_TASK_DIR: dict[str, str] = {
    "SR":  "task1- SR",
    "NR":  "task2 - NR",
    "TSR": "task3 - TSR",
}


def mat_path(mat_root: str | Path, task: str, subject: str) -> Path:
    return (
        Path(mat_root) / _TASK_DIR[task] / "Matlab files"
        / f"results{subject}_{task}.mat"
    )


def available_subjects(mat_root: str | Path, task: str) -> list[str]:
    return [s for s in ZUCO_SUBJECTS if mat_path(mat_root, task, s).exists()]


def _load_mat(path: Path) -> np.ndarray:
    """Load sentenceData struct array from a ZuCo mat file."""
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return mat["sentenceData"]


def _word_eeg_band(word_struct, et_metric: str, band_code: str) -> np.ndarray | None:
    """Extract one EEG band array (105,) from a word struct.

    Returns None if the field is missing, empty, or all-NaN.
    """
    field = f"{et_metric}_{band_code}"
    val = getattr(word_struct, field, None)
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float64)
    if arr.size != ZUCO_N_CHANNELS:
        return None
    if np.all(np.isnan(arr)):
        return None
    return arr


def load_task_words(
    mat_root: str | Path,
    task: str,
    subject: str,
    et_metric: str = "TRT",
    bands: list[str] | None = None,
) -> pd.DataFrame:
    """Load word-level EEG band features + eye-tracking for one (task, subject) pair.

    Parameters
    ----------
    mat_root : path to ``osfstorage/`` root (contains ``task1- SR/`` etc.)
    task     : "SR", "NR", or "TSR"
    subject  : one of ZUCO_SUBJECTS
    et_metric: EEG averaging window — "TRT" (default), "FFD", "GD", "GPT", "SFD"
    bands    : list of band codes, e.g. ["t1","a1","b1","g1"]. Default: all 4 × 0–500 ms.

    Returns
    -------
    pd.DataFrame with columns:
      word        str
      sent_id     int   — sentence index within task (0-based), used as CV group key
      word_id     int   — token position within sentence (0-based)
      global_word_idx int — sequential index across all sentences (for embedding lookup)
      t1, a1, b1, g1   ndarray (105,)  — EEG band arrays (one col per band requested)
      nFixations  int
      FFD, TRT, GD, GPT, SFD  float  — raw ET reading times (ms)

    Words with missing EEG (never fixated, omission) are excluded.
    """
    if bands is None:
        bands = list(ZUCO_BANDS)

    path = mat_path(mat_root, task, subject)
    if not path.exists():
        raise FileNotFoundError(f"ZuCo mat file not found: {path}")

    sentence_data = _load_mat(path)

    rows: list[dict] = []
    global_idx = 0

    for sent_id, sent in enumerate(sentence_data):
        words = sent.word
        if not hasattr(words, "__len__"):
            words = np.array([words])

        for word_id, word in enumerate(words):
            # Check EEG validity using the first requested band
            primary = _word_eeg_band(word, et_metric, bands[0])
            if primary is None:
                global_idx += 1
                continue

            row: dict = {
                "word": str(word.content),
                "sent_id": sent_id,
                "word_id": word_id,
                "global_word_idx": global_idx,
                bands[0]: primary,
            }

            # Remaining bands
            for band in bands[1:]:
                arr = _word_eeg_band(word, et_metric, band)
                row[band] = arr if arr is not None else np.full(ZUCO_N_CHANNELS, np.nan)

            # Eye-tracking scalars (raw ms, NaN if missing)
            for et_col in ("nFixations", "FFD", "TRT", "GD", "GPT", "SFD"):
                val = getattr(word, et_col, np.nan)
                row[et_col] = float(np.asarray(val).flat[0]) if np.asarray(val).size else np.nan

            rows.append(row)
            global_idx += 1

    if not rows:
        logger.warning("[ZuCo] %s %s: no valid words found", subject, task)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(
        "[ZuCo] %s %s: %d valid / %d total words across %d sentences",
        subject, task, len(df), global_idx, len(sentence_data),
    )
    return df


def load_canonical_words(
    mat_root: str | Path,
    task: str,
) -> pd.DataFrame:
    """Load the full canonical word list for a task (all sentences, all words).

    Uses the first available subject's mat file — sentence/word content is
    identical across subjects.  Returns ALL words (including omitted ones) so
    that global_word_idx is consistent with ``load_task_words``.

    Returns
    -------
    pd.DataFrame with columns: word, sent_id, word_id, global_word_idx
    """
    subjs = available_subjects(mat_root, task)
    if not subjs:
        raise RuntimeError(f"No mat files found for task {task} under {mat_root}")

    path = mat_path(mat_root, task, subjs[0])
    sentence_data = _load_mat(path)

    rows: list[dict] = []
    global_idx = 0
    for sent_id, sent in enumerate(sentence_data):
        words = sent.word
        if not hasattr(words, "__len__"):
            words = np.array([words])
        for word_id, word in enumerate(words):
            rows.append({
                "word": str(word.content),
                "sent_id": sent_id,
                "word_id": word_id,
                "global_word_idx": global_idx,
            })
            global_idx += 1

    logger.info(
        "[ZuCo] canonical word list for %s: %d words across %d sentences",
        task, global_idx, len(sentence_data),
    )
    return pd.DataFrame(rows)


class ZuCoDataset:
    """Lightweight wrapper for discovering ZuCo mat files."""

    def __init__(
        self,
        mat_root: str | Path,
        tasks: list[str] | None = None,
        subjects: list[str] | None = None,
    ) -> None:
        self.mat_root = Path(mat_root)
        self.tasks = tasks or list(ZUCO_TASKS)
        self._subjects = subjects

    def available_subjects(self, task: str) -> list[str]:
        subjs = available_subjects(self.mat_root, task)
        if self._subjects:
            subjs = [s for s in subjs if s in self._subjects]
        return subjs

    def available_tasks(self) -> list[str]:
        return [t for t in self.tasks if available_subjects(self.mat_root, t)]
