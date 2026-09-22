"""Broderick 2018 EEG dataset adapter (BIDS format).

Reference
---------
Broderick et al. (2018) Current Biology — EEG tracks semantic surprise during
continuous audiobook listening.

Dataset: 19 subjects, 20 audiobook segments (~196 s each), 128-channel BioSemi
EEG (A1–D32) at 512 Hz recorded during passive listening to "The Old Man and
the Sea".  Word onsets come from Praat TextGrids (forced alignment), not EEG
triggers.

Key paths
---------
  stimuli/audio{01..20}.TextGrid        — word / phone timings (Praat format)
  stimuli/audio{01..20}.wav
  sub-{001..019}/eeg/
    sub-*_task-listening_run-{01..20}_eeg.vhdr   — raw EEG (BrainVision)
    sub-*_task-listening_run-{01..20}_channels.tsv
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BRODERICK_SFREQ: float = 512.0
BRODERICK_N_SUBJECTS: int = 19
BRODERICK_N_RUNS: int = 20
BRODERICK_N_CHANNELS: int = 128

# BioSemi 128-channel cap: four 32-electrode groups.
# These approximate anatomical regions based on the BioSemi 128-ch standard layout.
_BIOSEMI_REGION: dict[str, str] = {
    "A": "left",
    "B": "right",
    "C": "posterior",
    "D": "occipital",
}


def channel_region(name: str) -> str | None:
    """Map a BioSemi channel name (A1–D32) to a scalp region label."""
    m = re.match(r"^([A-D])\d+$", name.strip())
    if m:
        return _BIOSEMI_REGION.get(m.group(1))
    return None


# ---------------------------------------------------------------------------
# TextGrid parsing
# ---------------------------------------------------------------------------

def parse_textgrid_words(path: str | Path) -> list[tuple[float, float, str]]:
    """Parse a Praat TextGrid file and return word-tier intervals.

    Parameters
    ----------
    path : str or Path
        Path to a ``.TextGrid`` file with tiers named ``phones`` and ``words``.

    Returns
    -------
    list of (xmin, xmax, word_text)
        Intervals from the ``words`` tier, excluding silences (``sil``, ``sp``,
        empty labels).
    """
    path = Path(path)
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            content = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise ValueError(f"Cannot decode TextGrid: {path}")

    # Split into item blocks and find the words tier.
    tier_blocks = re.split(r"\bitem\s*\[\d+\]\s*:", content)
    words_block: str | None = None
    for block in tier_blocks:
        if re.search(r'\bname\s*=\s*"words"', block):
            words_block = block
            break

    if words_block is None:
        raise ValueError(f"No 'words' tier found in TextGrid: {path}")

    interval_re = re.compile(
        r"xmin\s*=\s*([0-9.+\-eE]+)\s+"
        r"xmax\s*=\s*([0-9.+\-eE]+)\s+"
        r'text\s*=\s*"(.*?)"',
        re.DOTALL,
    )
    intervals: list[tuple[float, float, str]] = []
    for m in interval_re.finditer(words_block):
        xmin = float(m.group(1))
        xmax = float(m.group(2))
        word = m.group(3).strip()
        if word and word.lower() not in ("sil", "sp"):
            intervals.append((xmin, xmax, word))

    return intervals


# ---------------------------------------------------------------------------
# Word table helpers
# ---------------------------------------------------------------------------

def load_segment_words(bids_root: str | Path, segment: int) -> pd.DataFrame:
    """Load word table for one audiobook segment.

    Parameters
    ----------
    bids_root : str or Path
    segment : int
        1-based segment index (1–20).

    Returns
    -------
    DataFrame
        Columns: word, onset (s), offset (s), duration (s), run, segment_id.
        ``onset`` and ``offset`` are relative to the start of the segment's
        audio / EEG recording.
    """
    tg_path = Path(bids_root) / "stimuli" / f"audio{segment:02d}.TextGrid"
    if not tg_path.is_file():
        raise FileNotFoundError(f"TextGrid not found: {tg_path}")

    intervals = parse_textgrid_words(tg_path)
    rows = [
        {"word": w, "onset": xmin, "offset": xmax,
         "duration": xmax - xmin, "run": segment, "segment_id": segment}
        for xmin, xmax, w in intervals
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def load_all_words(
    bids_root: str | Path,
    n_runs: int = BRODERICK_N_RUNS,
) -> pd.DataFrame:
    """Load words from all 20 audiobook segments in order.

    Parameters
    ----------
    bids_root : str or Path
    n_runs : int
        Number of segments to load (default 20).

    Returns
    -------
    DataFrame
        Columns: word_idx, word, onset, offset, duration, run, segment_id.
        ``onset``/``offset`` are per-segment (i.e. relative to each segment's
        recording start, not the full audiobook).
    """
    bids_root = Path(bids_root)
    dfs: list[pd.DataFrame] = []
    for seg in range(1, n_runs + 1):
        tg = bids_root / "stimuli" / f"audio{seg:02d}.TextGrid"
        if not tg.is_file():
            logger.warning("TextGrid missing for segment %02d: %s", seg, tg)
            continue
        try:
            dfs.append(load_segment_words(bids_root, seg))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load segment %02d: %s", seg, exc)

    if not dfs:
        raise RuntimeError(f"No TextGrids found under {bids_root}/stimuli/")

    combined = pd.concat(dfs, ignore_index=True)
    combined.insert(0, "word_idx", np.arange(len(combined)))
    return combined


# ---------------------------------------------------------------------------
# EEG path helpers
# ---------------------------------------------------------------------------

def eeg_vhdr_path(bids_root: str | Path, subject: str, run: int) -> Path:
    """Path to the BrainVision .vhdr file for one subject × run."""
    return (
        Path(bids_root) / subject / "eeg"
        / f"{subject}_task-listening_run-{run:02d}_eeg.vhdr"
    )


def channels_tsv_path(bids_root: str | Path, subject: str, run: int) -> Path:
    """Path to channels.tsv for one subject × run."""
    return (
        Path(bids_root) / subject / "eeg"
        / f"{subject}_task-listening_run-{run:02d}_channels.tsv"
    )


# ---------------------------------------------------------------------------
# EEG loading helpers
# ---------------------------------------------------------------------------

def load_channels_tsv(
    bids_root: str | Path,
    subject: str,
    run: int = 1,
) -> pd.DataFrame:
    """Load channels.tsv for one run."""
    path = channels_tsv_path(bids_root, subject, run)
    if not path.is_file():
        logger.warning("channels.tsv not found: %s", path)
        return pd.DataFrame(columns=["name", "type", "status"])
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = df.columns.str.strip()
    return df


def load_eeg_run(
    bids_root: str | Path,
    subject: str,
    run: int,
    *,
    l_freq: float = 1.0,
    h_freq: float = 8.0,
    target_sfreq: float = 64.0,
    average_ref: bool = True,
    good_channels: list[str] | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Load, filter, and downsample one EEG run.

    Parameters
    ----------
    bids_root, subject, run
        BIDS identifiers.
    l_freq, h_freq : float
        Bandpass filter cutoffs (Hz).  Default 1–8 Hz matches the temporal
        scale of word-rate EEG responses (Broderick et al. 2018).
    target_sfreq : float
        Target sampling rate after downsampling (default 64 Hz).
    average_ref : bool
        Apply average reference before filtering.
    good_channels : list of str, optional
        Channel names to keep.  None = keep all EEG channels present.

    Returns
    -------
    data : (n_channels, n_times) float32
        Preprocessed EEG at ``target_sfreq`` Hz.
    sfreq : float
        Actual sampling rate of the returned data.
    ch_names : list of str
        Channel names in the returned data.
    """
    import mne

    vhdr = eeg_vhdr_path(bids_root, subject, run)
    if not vhdr.is_file():
        raise FileNotFoundError(f"EEG file not found: {vhdr}")

    raw = mne.io.read_raw_brainvision(
        str(vhdr), preload=True, verbose=False
    )

    # Pick requested channels (or all EEG channels)
    if good_channels is not None:
        present = [ch for ch in good_channels if ch in raw.ch_names]
        if not present:
            raise RuntimeError(
                f"No good channels found in {vhdr.name} for {subject}"
            )
        raw.pick_channels(present, ordered=True)
    else:
        raw.pick_types(eeg=True, verbose=False)

    # Average reference (set as EEG reference)
    if average_ref:
        raw.set_eeg_reference("average", projection=False, verbose=False)

    # Bandpass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir",
               fir_design="firwin", verbose=False)

    # Downsample
    if raw.info["sfreq"] > target_sfreq + 1:
        raw.resample(target_sfreq, verbose=False)

    data = raw.get_data().astype(np.float32)   # (n_ch, n_t)
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)

    return data, sfreq, ch_names


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class BroderickDataset:
    """Adapter for the Broderick 2018 BIDS EEG dataset.

    Parameters
    ----------
    bids_root : str or Path
        Path to the BIDS root (``data/broderick/``).
    subjects : list of str, optional
        Subset of subjects to load.  None = all subjects with run-01 EEG.
    drop_bads : bool
        Drop channels marked ``status=bad`` in channels.tsv.
    n_runs : int
        Number of audiobook segments to expect (default 20).
    """

    def __init__(
        self,
        bids_root: str | Path,
        *,
        subjects: list[str] | None = None,
        drop_bads: bool = True,
        n_runs: int = BRODERICK_N_RUNS,
    ) -> None:
        self.bids_root = Path(bids_root)
        self._requested_subjects = subjects
        self.drop_bads = drop_bads
        self.n_runs = n_runs

    def available_subjects(self) -> list[str]:
        """Subjects that have at least run-01 EEG data."""
        if self._requested_subjects is not None:
            return [
                s for s in self._requested_subjects
                if eeg_vhdr_path(self.bids_root, s, 1).is_file()
            ]
        return sorted(
            d.name
            for d in self.bids_root.glob("sub-*")
            if d.is_dir() and eeg_vhdr_path(self.bids_root, d.name, 1).is_file()
        )

    def available_runs(self, subject: str) -> list[int]:
        """Run indices (1-based) with an EEG file for this subject."""
        return [
            r for r in range(1, self.n_runs + 1)
            if eeg_vhdr_path(self.bids_root, subject, r).is_file()
        ]

    def good_channel_names(self, subject: str, run: int = 1) -> list[str]:
        """Channel names passing quality filter (uses run-01 channels.tsv)."""
        ch_df = load_channels_tsv(self.bids_root, subject, run)
        if ch_df.empty:
            return []
        mask = ch_df["type"].str.upper() == "EEG"
        if self.drop_bads and "status" in ch_df.columns:
            mask &= ch_df["status"].str.lower().str.strip() == "good"
        return ch_df.loc[mask, "name"].tolist()

    def channel_meta(self, channel_names: list[str]) -> dict[str, dict]:
        """Return ``{channel_name: metadata}`` with region label.

        Broderick EEG has no MNI electrode coordinates, so x/y/z are None.
        """
        return {
            name: {
                "name": name,
                "x": None,
                "y": None,
                "z": None,
                "region": channel_region(name),
                "hemisphere": None,
            }
            for name in channel_names
        }
