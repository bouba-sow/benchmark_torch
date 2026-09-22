"""Podcast iEEG dataset adapter (BIDS format).

Reference
---------
Hasson lab — Podcast ECoG dataset (OpenNeuro ds005574).
  9 subjects, 1 330 total electrodes, 30-min continuous recording at 512 Hz.
  Stimuli: "This American Life — So a Monkey and a Horse Walk Into a Bar: Act One"
  Preprocessed derivatives: high-gamma (70–200 Hz) Hilbert envelope .fif files.

Dataset root: ``data/podcast/`` (BIDS layout).

Key paths
---------
  stimuli/podcast_transcript.csv                          — word, start (s), end (s)
  sub-{N}/ieeg/sub-{N}_task-podcast_channels.tsv          — channel status
  sub-{N}/ieeg/sub-{N}_space-MNI152NLin2009aSym_electrodes.tsv — MNI coords
  derivatives/ecogprep/sub-{N}/ieeg/sub-{N}_task-podcast_desc-highgamma_ieeg.fif
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PODCAST_SFREQ_RAW: float = 512.0    # original sampling rate
PODCAST_HIGHGAMMA_BAND = (70.0, 200.0)
PODCAST_N_WORDS_APPROX: int = 5136  # approximate number of words in transcript


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def load_transcript(bids_root: str | Path) -> pd.DataFrame:
    """Load word-level transcript with precise onset/offset times.

    Returns
    -------
    pd.DataFrame with columns: word_idx, word, onset (s), offset (s), duration (s).
    ``onset`` is the start time of the word in the podcast (from recording start).
    """
    path = Path(bids_root) / "stimuli" / "podcast_transcript.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Podcast transcript not found: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"start": "onset", "end": "offset"})
    df.insert(0, "word_idx", np.arange(len(df)))
    df["duration"] = df["offset"] - df["onset"]
    return df.reset_index(drop=True)


def build_word_context(words: list[str], context_words: int = 48) -> list[str]:
    """Return a list of context strings: for word i, the last ``context_words`` preceding words."""
    contexts: list[str] = []
    for i in range(len(words)):
        start = max(0, i - context_words)
        ctx = " ".join(words[start:i])
        contexts.append(ctx)
    return contexts


# ---------------------------------------------------------------------------
# Channel and electrode metadata
# ---------------------------------------------------------------------------

def load_channels_tsv(bids_root: str | Path, subject: str) -> pd.DataFrame:
    """Load channels.tsv → DataFrame with columns: name, type, status."""
    path = Path(bids_root) / subject / "ieeg" / f"{subject}_task-podcast_channels.tsv"
    if not path.is_file():
        logger.warning("channels.tsv not found: %s", path)
        return pd.DataFrame(columns=["name", "type", "status"])
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    return df


def load_electrodes_tsv(bids_root: str | Path, subject: str) -> pd.DataFrame:
    """Load MNI-space electrodes.tsv → DataFrame with name, x, y, z, group, hemisphere."""
    path = (
        Path(bids_root) / subject / "ieeg"
        / f"{subject}_space-MNI152NLin2009aSym_electrodes.tsv"
    )
    if not path.is_file():
        logger.warning("MNI electrodes.tsv not found: %s", path)
        return pd.DataFrame(columns=["name", "x", "y", "z", "group"])

    df = pd.read_csv(path, sep="\t", na_values=["n/a"], dtype=str)
    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive hemisphere from MNI x-coordinate sign: positive = right
    if "x" in df.columns:
        df["hemisphere"] = df["x"].apply(
            lambda v: "R" if (v is not None and np.isfinite(float(v)) and float(v) > 0)
            else ("L" if (v is not None and np.isfinite(float(v))) else None)
        )
    else:
        df["hemisphere"] = None

    return df


def _electrode_meta_for_channels(
    elec_df: pd.DataFrame,
    channel_names: list[str],
) -> dict[str, dict]:
    """Build {channel_name: metadata} for every channel in ``channel_names``."""
    elec_index: dict[str, dict] = {}
    if not elec_df.empty and "name" in elec_df.columns:
        elec_index = elec_df.set_index("name").to_dict("index")

    meta: dict[str, dict] = {}
    for name in channel_names:
        e = elec_index.get(name, {})
        x = _to_float(e.get("x"))
        y = _to_float(e.get("y"))
        z = _to_float(e.get("z"))
        group = str(e.get("group", "")) or None
        hemi = e.get("hemisphere") or (
            "R" if (x is not None and x > 0)
            else ("L" if x is not None else None)
        )
        meta[name] = {
            "name": name,
            "x": x,
            "y": y,
            "z": z,
            "region": group,
            "hemisphere": hemi,
        }
    return meta


def _to_float(val) -> float | None:
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Preprocessed fif paths
# ---------------------------------------------------------------------------

def highgamma_fif_path(bids_root: str | Path, subject: str) -> Path:
    """Return the path to the preprocessed high-gamma .fif file."""
    return (
        Path(bids_root) / "derivatives" / "ecogprep" / subject / "ieeg"
        / f"{subject}_task-podcast_desc-highgamma_ieeg.fif"
    )


def raw_fif_path(bids_root: str | Path, subject: str) -> Path:
    """Return the path to the common-average-referenced raw .fif file."""
    return (
        Path(bids_root) / "derivatives" / "ecogprep" / subject / "ieeg"
        / f"{subject}_task-podcast_ieeg.fif"
    )


# ---------------------------------------------------------------------------
# Epoch extraction helpers (MNE-free numpy implementation)
# ---------------------------------------------------------------------------

def extract_word_epochs(
    data: np.ndarray,
    sfreq: float,
    recording_tmin: float,
    word_onsets: np.ndarray,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract word-locked epochs from a continuous neural recording.

    Parameters
    ----------
    data : (n_channels, n_times_total)
        Continuous high-gamma signal (already resampled to sfreq).
    sfreq : float
        Sampling frequency in Hz.
    recording_tmin : float
        Time (s) of the first sample in ``data`` (usually 0).
    word_onsets : (n_words,)
        Word onset times in seconds (absolute, matching transcript).
    tmin, tmax : float
        Epoch window relative to each word onset (e.g. -0.1 to 1.0 s).

    Returns
    -------
    epochs : (n_valid, n_channels, n_epoch_times)
    valid_mask : (n_words,) bool — True for epochs that could be extracted
    epoch_times : (n_epoch_times,) — time axis relative to word onset
    """
    n_channels, n_total = data.shape
    n_epoch_samples = int(round((tmax - tmin) * sfreq))
    epoch_times = np.linspace(tmin, tmax, n_epoch_samples, endpoint=False)

    valid_mask = np.zeros(len(word_onsets), dtype=bool)
    epoch_list: list[np.ndarray] = []

    for wi, onset in enumerate(word_onsets):
        t_start = onset + tmin
        # Convert to sample index relative to start of recording
        idx_start = int(round((t_start - recording_tmin) * sfreq))
        idx_end = idx_start + n_epoch_samples
        if idx_start < 0 or idx_end > n_total:
            continue
        epoch_list.append(data[:, idx_start:idx_end].copy())
        valid_mask[wi] = True

    if not epoch_list:
        return (
            np.zeros((0, n_channels, n_epoch_samples), dtype=np.float32),
            valid_mask,
            epoch_times,
        )

    epochs = np.stack(epoch_list, axis=0).astype(np.float32)
    return epochs, valid_mask, epoch_times


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class PodcastDataset:
    """Adapter for the Podcast BIDS iEEG dataset.

    Parameters
    ----------
    bids_root : str or Path
        Path to the BIDS root (``data/podcast/``).
    subjects : list of str, optional
        Subset of subjects to load.  None = all subjects with a highgamma .fif.
    drop_bads : bool
        Drop channels marked ``status=bad`` in channels.tsv.
    """

    def __init__(
        self,
        bids_root: str | Path,
        *,
        subjects: list[str] | None = None,
        drop_bads: bool = True,
    ) -> None:
        self.bids_root = Path(bids_root)
        self._requested_subjects = subjects
        self.drop_bads = drop_bads

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def available_subjects(self) -> list[str]:
        """Subjects that have a preprocessed high-gamma .fif file."""
        if self._requested_subjects is not None:
            return [
                s for s in self._requested_subjects
                if highgamma_fif_path(self.bids_root, s).is_file()
            ]
        found = sorted(
            d.name for d in self.bids_root.glob("sub-*") if d.is_dir()
            and highgamma_fif_path(self.bids_root, d.name).is_file()
        )
        return found

    def load_words(self) -> pd.DataFrame:
        """Return a word-level DataFrame with timing information.

        Columns: word_idx, word, onset (s), offset (s), duration (s).
        """
        return load_transcript(self.bids_root)

    def good_channel_names(self, subject: str) -> list[str]:
        """Return list of channel names that pass the quality filter."""
        ch_df = load_channels_tsv(self.bids_root, subject)
        if ch_df.empty:
            return []
        mask = ch_df["type"].str.upper().isin({"ECOG", "SEEG", "EEG"})
        if self.drop_bads and "status" in ch_df.columns:
            mask &= ch_df["status"].str.lower().str.strip() == "good"
        return ch_df.loc[mask, "name"].tolist()

    def electrode_meta(self, subject: str, channel_names: list[str]) -> dict[str, dict]:
        """Return {channel_name: metadata} with MNI coordinates and region."""
        elec_df = load_electrodes_tsv(self.bids_root, subject)
        return _electrode_meta_for_channels(elec_df, channel_names)
