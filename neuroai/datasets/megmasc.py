"""MEG-MASC (Gwilliams et al. 2023) dataset adapter.

MEG-MASC: raw magnetoencephalography of English speakers listening to four
fictional stories from the Manually Annotated Sub-Corpus (MASC).

  - 27 participants nominally (this local copy has sub-01 … sub-11); most did
    two identical sessions, a few did one.
  - KIT/Yokogawa whole-head system, 208 axial gradiometers, 1000 Hz.
  - Four stories, presented in a subject-specific order (``task_order`` in
    ``participants.tsv``); ``task-0 … task-3`` map to a story via the
    ``story`` field carried in every ``events.tsv`` row.
  - Word / phoneme onsets from Gentle forced alignment, stored in the
    ``trial_type`` column of each ``*_events.tsv`` as a *stringified Python
    dict* (``kind`` ∈ {sound, phoneme, word}, plus ``word``, ``word_index``,
    ``condition``, ``pronounced`` …).

Key paths (relative to ``bids_root`` = ``$SCRATCH/data/gwilliams``):
  sub-*/ses-*/meg/sub-*_ses-*_task-*_meg.con        — raw KIT recording
  sub-*/ses-*/meg/sub-*_ses-*_task-*_events.tsv     — word/phoneme timings
  stimuli/text/*.txt                                — plain story text

Word onsets are **recording-relative** (each ``.con`` has a variable silent
lead-in), so unlike the Broderick adapter the canonical word
list (:func:`load_story_words`, used once for LLM feature extraction) carries
no usable onsets — the per-recording onsets come from :func:`load_run_words`
and are joined back to the canonical list by ``word_index``.

Reference
---------
Gwilliams, L., Flick, G., Marantz, A., Pylkkänen, L., Poeppel, D., King, J.-R.
(2023). Introducing MEG-MASC: a high-quality magneto-encephalography dataset
for evaluating natural speech processing. Scientific Data 10, 862.
"""

from __future__ import annotations

import ast
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MEGMASC_SFREQ: float = 1000.0     # Hz — native KIT sampling rate
MEGMASC_N_CHANNELS: int = 208     # axial gradiometers

# Coarse sensor grouping for the LME random-effect factor, in the KIT `.con`
# channel order (index 0..207). Derived once from the KIT device-coordinate
# sensor positions of one recording: hemisphere by x-sign (L<0, R>0),
# anterior/central/posterior by y relative to the array median (±25 mm central
# band). This is a *sensor cluster* grouping, NOT an anatomical ROI — a
# "left-posterior" MEG sensor mixes left temporal + parietal + occipital
# cortex. Counts: LA 42, LC 21, LP 42, RA 43, RC 19, RP 41.
MEGMASC_SENSOR_REGIONS: tuple[str, ...] = (
    "RC", "RC", "LP", "RP", "LP", "LP", "RA", "RC", "LA", "RC", "RC", "LC", "LC",
    "RP", "LP", "LP", "LA", "RA", "LC", "LA", "RC", "LC", "RP", "RA", "LA", "LA",
    "RA", "RA", "RA", "RA", "LA", "RA", "LP", "LC", "LC", "LA", "LA", "LC", "LA",
    "LA", "LA", "RA", "LA", "RA", "LA", "RA", "LA", "LA", "LA", "LC", "LA", "LA",
    "LP", "LA", "LA", "LA", "RA", "RP", "RA", "RA", "RC", "RA", "RA", "LP", "LP",
    "LP", "LC", "LC", "LP", "LA", "LC", "LA", "RA", "RA", "RA", "RA", "RA", "RA",
    "RC", "RP", "RA", "LA", "LA", "LA", "LA", "LA", "LA", "LA", "RA", "RA", "RA",
    "RA", "LA", "RA", "LA", "RA", "LA", "LP", "LC", "LP", "LC", "LA", "LA", "LC",
    "RA", "RA", "LA", "RA", "LA", "RA", "LA", "LA", "LC", "RA", "LP", "LC", "LP",
    "LC", "LP", "LC", "RA", "RA", "RA", "RA", "RA", "RP", "RC", "RA", "RP", "RA",
    "RP", "RC", "RC", "LA", "LA", "LP", "RP", "RC", "RP", "RP", "RC", "RP", "RC",
    "RA", "LP", "LC", "RP", "LP", "LP", "LC", "LP", "LP", "RP", "RP", "RP", "RP",
    "RP", "RA", "RC", "RP", "LP", "LP", "LP", "LP", "LP", "LP", "LC", "LA", "RP",
    "RP", "RP", "RP", "RP", "RC", "RP", "RP", "LP", "LP", "LP", "LP", "LP", "LP",
    "LP", "LP", "RP", "RP", "LP", "RP", "RC", "RP", "RC", "RC", "RP", "RP", "RP",
    "RP", "LP", "LP", "LP", "LP", "RP", "RP", "RP", "LP", "RP", "RP", "RP", "RP",
)
assert len(MEGMASC_SENSOR_REGIONS) == MEGMASC_N_CHANNELS


def megmasc_sensor_region(channel: int) -> str:
    """Region label ('LA'|'LC'|'LP'|'RA'|'RC'|'RP') for a 0-based KIT MEG
    channel index — the LME random-effect grouping factor. See
    :data:`MEGMASC_SENSOR_REGIONS`."""
    return MEGMASC_SENSOR_REGIONS[channel]

# task-N → story, per the dataset README ("Task organisation"). Only used as a
# fallback label; the authoritative story name is the ``story`` field inside
# each events.tsv row (read by :func:`_story_from_events`).
TASK_STORIES: tuple[str, ...] = (
    "lw1",
    "cable_spool_fort",
    "easy_money",
    "The_Black_Widow",
)


# ---------------------------------------------------------------------------
# events.tsv parsing
# ---------------------------------------------------------------------------

def _parse_trial_type(val: object) -> dict:
    """Parse one ``trial_type`` cell (a stringified Python dict) → dict.

    Returns an empty dict on anything unparseable.
    """
    if isinstance(val, dict):
        return val
    if not isinstance(val, str):
        return {}
    try:
        out = ast.literal_eval(val)
        return out if isinstance(out, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _read_events(events_tsv: str | Path) -> pd.DataFrame:
    """Read a MEG-MASC ``events.tsv`` and explode the dict column.

    Returns the raw frame with the parsed dict fields merged in as columns
    (``kind``, ``word``, ``word_index``, ``condition``, ``pronounced``,
    ``story`` …), before any filtering.
    """
    df = pd.read_csv(events_tsv, sep="\t")
    meta = df["trial_type"].map(_parse_trial_type)
    fields = ("kind", "word", "word_index", "condition", "pronounced", "story",
              "sequence_id", "speech_rate", "voice")
    for f in fields:
        df[f] = meta.map(lambda d, _f=f: d.get(_f))
    return df


def _story_from_events(events_tsv: str | Path) -> str | None:
    """Story name for one recording, read from the first parseable events row."""
    try:
        df = pd.read_csv(events_tsv, sep="\t", usecols=["trial_type"])
    except (OSError, ValueError):
        return None
    for val in df["trial_type"]:
        d = _parse_trial_type(val)
        story = d.get("story")
        if isinstance(story, str) and story:
            return story
    return None


def _word_rows(events_df: pd.DataFrame) -> pd.DataFrame:
    """Filter an exploded events frame to real, pronounced, in-sentence words.

    Drops the interspersed random word-lists (``condition != 'sentence'``) and
    non-pronounced tokens — the same subset the Gwilliams analyses keep.
    """
    m = (
        (events_df["kind"] == "word")
        & (events_df["condition"] == "sentence")
        & (events_df["pronounced"].fillna(1.0).astype(float) == 1.0)
        & events_df["word"].notna()
        & events_df["word_index"].notna()
    )
    out = events_df.loc[m, ["onset", "duration", "word", "word_index", "story"]].copy()
    out["word_index"] = out["word_index"].astype(int)
    out["word"] = out["word"].astype(str)
    # A story can span multiple sound files, each restarting word_index at 0 —
    # disambiguate by making word_index run monotonically in onset order.
    out = out.sort_values("onset", kind="stable").reset_index(drop=True)
    out["word_index"] = np.arange(len(out), dtype=int)
    return out


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_all_runs(
    bids_root: str | Path,
    subjects: list[str] | None = None,
    stories: list[str] | None = None,
) -> pd.DataFrame:
    """All (subject, session, story) recordings with both a .con and events.tsv.

    Returns
    -------
    DataFrame with columns: subject, session, task, story, con, events_tsv.
    """
    bids_root = Path(bids_root)
    if subjects is None:
        subjects = sorted(d.name for d in bids_root.glob("sub-*") if d.is_dir())

    records: list[dict] = []
    for sub in subjects:
        for ses_dir in sorted((bids_root / sub).glob("ses-*")):
            meg_dir = ses_dir / "meg"
            if not meg_dir.is_dir():
                continue
            for con in sorted(meg_dir.glob(f"{sub}_{ses_dir.name}_task-*_meg.con")):
                events_tsv = con.with_name(
                    con.name.replace("_meg.con", "_events.tsv")
                )
                if not events_tsv.is_file():
                    continue
                task = con.stem.split("task-")[1].split("_")[0]
                story = _story_from_events(events_tsv)
                if story is None:
                    try:
                        story = TASK_STORIES[int(task)]
                    except (ValueError, IndexError):
                        continue
                if stories is not None and story not in stories:
                    continue
                records.append({
                    "subject": sub,
                    "session": ses_dir.name,
                    "task": task,
                    "story": story,
                    "con": str(con),
                    "events_tsv": str(events_tsv),
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Word tables
# ---------------------------------------------------------------------------

def load_run_words(events_tsv: str | Path) -> pd.DataFrame:
    """Per-recording word table with the onsets for *this* ``.con``.

    Columns: word, onset (s, recording-relative), duration (s), word_index
    (0-based, onset order within this recording), story.
    """
    return _word_rows(_read_events(events_tsv))


@lru_cache(maxsize=8)
def _canonical_story_words(bids_root: str, story: str) -> pd.DataFrame:
    """Canonical word list for a story (text is identical across subjects).

    Uses the first recording of that story found under ``bids_root``. Onsets
    are dropped (recording-specific); only ``word`` / ``word_index`` matter —
    those feed one LLM feature-extraction pass shared by every subject.
    """
    runs = discover_all_runs(bids_root, stories=[story])
    if runs.empty:
        raise FileNotFoundError(f"No recording found for story {story!r} under {bids_root}")
    words = load_run_words(runs.iloc[0]["events_tsv"])
    words = words[["word_index", "word"]].copy()
    words["story"] = story
    return words.reset_index(drop=True)


def load_story_words(bids_root: str | Path, story: str) -> pd.DataFrame:
    """Canonical word list for one story — signature matches the Broderick /
    Broderick adapters so the shared encoder can call it uniformly.

    Columns: word_index, word, story, onset (NaN — see module docstring),
    duration (NaN).
    """
    df = _canonical_story_words(str(bids_root), story).copy()
    df["onset"] = np.nan
    df["duration"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Raw MEG loading
# ---------------------------------------------------------------------------

def load_meg_run(con_path: str | Path) -> tuple[np.ndarray, float]:
    """Load one raw KIT recording.

    Returns
    -------
    data : (n_channels, n_samples) float32 — MEG channels only (208 axial grads)
    sfreq : float — native sampling rate (1000 Hz)

    Head-position digitisation (``.mrk`` / ``.pos``) is deliberately not read:
    it is only needed for source localisation, and MNE's KIT reader rejects
    the BIDS ``.pos`` headshape extension anyway.
    """
    import mne

    con_path = str(con_path)
    raw = mne.io.read_raw_kit(con_path, preload=True, verbose="ERROR")
    raw.pick("meg")
    data = raw.get_data().astype(np.float32)
    return data, float(raw.info["sfreq"])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MegMascDataset:
    """Adapter for the MEG-MASC BIDS dataset.

    Parameters
    ----------
    bids_root : str or Path
        BIDS root (``$SCRATCH/data/gwilliams``).
    subjects : list of str, optional
        Subset of subjects. None = all ``sub-*`` directories.
    stories : list of str, optional
        Subset of story IDs. None = all four.
    """

    def __init__(
        self,
        bids_root: str | Path,
        *,
        subjects: list[str] | None = None,
        stories: list[str] | None = None,
    ) -> None:
        self.bids_root = Path(bids_root)
        self._subjects = subjects
        self._stories = stories
        self._runs_cache: pd.DataFrame | None = None

    def runs(self) -> pd.DataFrame:
        if self._runs_cache is None:
            self._runs_cache = discover_all_runs(
                self.bids_root, subjects=self._subjects, stories=self._stories
            )
        return self._runs_cache

    def available_subjects(self) -> list[str]:
        if self._subjects is not None:
            return list(self._subjects)
        r = self.runs()
        return sorted(r["subject"].unique().tolist()) if not r.empty else []

    def available_stories(self) -> list[str]:
        r = self.runs()
        stories = sorted(r["story"].unique().tolist()) if not r.empty else []
        if self._stories is not None:
            stories = [s for s in stories if s in self._stories]
        return stories

    def load_words(self, story: str) -> pd.DataFrame:
        return load_story_words(self.bids_root, story)

    def load_meg(self, con_path: str | Path) -> tuple[np.ndarray, float]:
        return load_meg_run(con_path)
