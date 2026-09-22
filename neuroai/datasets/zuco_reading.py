"""ZuCo behavioral (eye-tracking) reading-time adapter.

Targets **TRT** (total reading time, ms) — the summed duration of all fixations
on a word, the most commonly used ZuCo reading-time measure in the literature.
Reuses the same ``.mat`` files as the shared ZuCo loader (:mod:`neuroai.datasets.zuco`)
but does not require EEG-band validity: a word is a valid behavioral trial iff
it was fixated at least once, independent of whether its EEG features passed
that adapter's band-availability check.

Output schema of :class:`ZuCoReadingDataset.load_words`
---------------------------------------------------------
Mirrors :class:`neuroai.datasets.ucl_reading.UCLReadingDataset` so it can reuse
the same LME fitting code (:func:`neuroai.regressors.ucl_reading_lme.fit_ucl_reading_lme`).

aggregate=True  (one row per word, averaged across subjects):
    word        str     Surface form (as tokenized/presented in the mat file).
    story_id    int     Sentence index within the task (0-based).
    zone        int     1-based word position within the sentence.
    rt_mean     float   Mean TRT across subjects who fixated the word (ms).
    rt_sd       float   Standard deviation across subjects.
    is_edge     bool    True for sentence-initial/-final words.
    wlen        int     Character length of the word.
    context     str     Cumulative sentence text through and including ``word``.

aggregate=False (one row per subject-word trial):
    participant_id  str   ZuCo subject code (e.g. "ZAB").
    story_id, zone, word, is_edge, wlen, context : as above.
    rt              float Raw single-subject TRT (ms).

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

from .zuco import _load_mat, available_subjects, load_canonical_words, mat_path

logger = logging.getLogger(__name__)


def _load_subject_trt(
    mat_root: str | Path,
    task: str,
    subject: str,
    *,
    min_fixations: int,
    min_trt: float,
    max_trt: float,
) -> pd.DataFrame:
    """Raw per-word TRT trials for one subject, independent of EEG validity."""
    path = mat_path(mat_root, task, subject)
    if not path.exists():
        raise FileNotFoundError(f"ZuCo mat file not found: {path}")

    sentence_data = _load_mat(path)

    rows: list[dict] = []
    for sent_id, sent in enumerate(sentence_data):
        words = sent.word
        if not hasattr(words, "__len__"):
            words = np.array([words])
        for word_id, word in enumerate(words):
            n_fix = getattr(word, "nFixations", np.nan)
            n_fix = float(np.asarray(n_fix).flat[0]) if np.asarray(n_fix).size else np.nan
            trt = getattr(word, "TRT", np.nan)
            trt = float(np.asarray(trt).flat[0]) if np.asarray(trt).size else np.nan
            rows.append(
                {
                    "participant_id": subject,
                    "story_id": sent_id,
                    "zone": word_id + 1,
                    "n_fixations": n_fix,
                    "rt": trt,
                }
            )

    df = pd.DataFrame(rows)
    n_before = len(df)
    df = df[
        np.isfinite(df["n_fixations"])
        & (df["n_fixations"] >= min_fixations)
        & np.isfinite(df["rt"])
        & (df["rt"] >= min_trt)
        & (df["rt"] <= max_trt)
    ]
    logger.info(
        "[ZuCo reading] %s %s: kept %d/%d word trials "
        "(fixations>=%d, TRT in [%.0f, %.0f] ms)",
        subject,
        task,
        len(df),
        n_before,
        min_fixations,
        min_trt,
        max_trt,
    )
    return df.drop(columns=["n_fixations"]).reset_index(drop=True)


def load_task_trt_trials(
    mat_root: str | Path,
    task: str,
    *,
    subjects: list[str] | None = None,
    min_fixations: int = 1,
    min_trt: float = 50.0,
    max_trt: float = 5000.0,
) -> pd.DataFrame:
    """Raw per-(subject, word) TRT trials for one ZuCo task, across subjects.

    Returns columns: participant_id, story_id, zone, rt.
    """
    avail = available_subjects(mat_root, task)
    if subjects is not None:
        avail = [s for s in avail if s in subjects]
    if not avail:
        raise RuntimeError(f"No ZuCo subjects available for task {task!r} under {mat_root}")

    frames = [
        _load_subject_trt(
            mat_root,
            task,
            subj,
            min_fixations=min_fixations,
            min_trt=min_trt,
            max_trt=max_trt,
        )
        for subj in avail
    ]
    trials = pd.concat(frames, ignore_index=True)
    logger.info(
        "[ZuCo reading] %s: %d trials, %d subjects, %d sentences",
        task,
        len(trials),
        trials["participant_id"].nunique(),
        trials["story_id"].nunique(),
    )
    return trials


def _word_list(mat_root: str | Path, task: str) -> pd.DataFrame:
    """Per-(story_id, zone) word identity, edge flag, length, and context.

    ``context`` is the cumulative sentence text through and including the
    target word (NeuralSet ``AddConcatenationContext`` convention), matching
    :func:`neuroai.datasets.ucl_reading.UCLReadingDataset._word_list`.
    """
    canonical = load_canonical_words(mat_root, task)
    rows: list[dict] = []
    for sent_id, group in canonical.groupby("sent_id", sort=True):
        tokens = group.sort_values("word_id")["word"].astype(str).tolist()
        n = len(tokens)
        for i, word in enumerate(tokens):
            rows.append(
                {
                    "story_id": int(sent_id),
                    "zone": i + 1,
                    "word": word,
                    "is_edge": (i == 0) or (i == n - 1),
                    "wlen": len(word),
                    "context": " ".join(tokens[: i + 1]),
                }
            )
    return pd.DataFrame(rows)


class ZuCoReadingDataset:
    """Load ZuCo behavioral (TRT) reading times for one task.

    Parameters
    ----------
    mat_root : str or Path
        Dataset root (``osfstorage/``) containing ``task1- SR/`` etc.
    task : str
        One of "SR", "NR", "TSR".
    subjects : list of str, optional
        Subset of ZuCo subject codes to load. Loads all 12 by default.
    min_fixations : int
        Minimum fixation count required to keep a word trial (excludes
        skipped/never-fixated words, whose TRT is not a real reading-time
        observation).
    min_trt, max_trt : float
        Plausible single-trial TRT bounds in ms (implausible TRTs excluded).
    aggregate : bool
        If True (default), one row per word with mean/SD TRT across subjects.
        If False, one row per (subject, word) trial.
    """

    def __init__(
        self,
        mat_root: str | Path,
        task: str,
        *,
        subjects: list[str] | None = None,
        min_fixations: int = 1,
        min_trt: float = 50.0,
        max_trt: float = 5000.0,
        aggregate: bool = True,
    ) -> None:
        self.mat_root = Path(mat_root)
        self.task = task
        self.subjects = subjects
        self.min_fixations = min_fixations
        self.min_trt = min_trt
        self.max_trt = max_trt
        self.aggregate = aggregate

    def load_words(self) -> pd.DataFrame:
        trials = load_task_trt_trials(
            self.mat_root,
            self.task,
            subjects=self.subjects,
            min_fixations=self.min_fixations,
            min_trt=self.min_trt,
            max_trt=self.max_trt,
        )
        word_df = _word_list(self.mat_root, self.task)
        if self.aggregate:
            return self._aggregate(trials, word_df)
        return self._raw(trials, word_df)

    def _aggregate(self, trials: pd.DataFrame, word_df: pd.DataFrame) -> pd.DataFrame:
        agg = (
            trials.groupby(["story_id", "zone"])["rt"]
            .agg(rt_mean="mean", rt_sd="std", rt_n="count")
            .reset_index()
        )
        merged = word_df.merge(agg, on=["story_id", "zone"], how="inner")
        merged["rt_sd"] = merged["rt_sd"].fillna(0.0)
        merged = merged.sort_values(["story_id", "zone"]).reset_index(drop=True)
        logger.info(
            "[ZuCo reading] %s: aggregated %d words across %d sentences (mean TRT)",
            self.task,
            len(merged),
            merged["story_id"].nunique(),
        )
        return merged

    def _raw(self, trials: pd.DataFrame, word_df: pd.DataFrame) -> pd.DataFrame:
        merged = trials.merge(word_df, on=["story_id", "zone"], how="inner")
        logger.info(
            "[ZuCo reading] %s: loaded %d trials (%d subjects, %d sentences)",
            self.task,
            len(merged),
            merged["participant_id"].nunique(),
            merged["story_id"].nunique(),
        )
        return merged
