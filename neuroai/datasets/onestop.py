"""OneStop Corpus eye-tracking reading-time adapter (Berzak, Malmaud & Levy, 2024 /
Berzak et al. 2025 Behavior Research Methods).

Supports three by-word duration measures, selected via ``measure``:

- **SP** (scan path / total dwell time)        -> ``IA_DWELL_TIME``
- **FP** (first-pass / gaze duration, default) -> ``IA_FIRST_RUN_DWELL_TIME``
- **GP** (go-past / regression-path duration)  -> ``IA_REGRESSION_PATH_DURATION``

Same EyeLink DataViewer IA-report format as Provo (missing values are the
literal string ``"."``), so the loading conventions mirror
:mod:`neuroai.datasets.provo` closely.

Dataset layout expected at ``root/``
-------------------------------------
precomputed_reading_measures/ia_Paragraph.csv — one row per (participant_id,
    article_batch, article_id, paragraph_id, difficulty_level, IA_ID): IA-level
    eye-tracking measures for the OneStop paragraph-reading task (360
    participants, 3 counterbalancing batches).

Important data quirks (verified against the released CSV)
-----------------------------------------------------------
1. **``article_id`` is only unique within ``article_batch``** (3 batches x 11
   articles = 33 distinct articles total; naively keying on ``article_id`` alone
   collapses unrelated articles from different batches onto the same id).
   ``story_id`` is therefore encoded as
   ``article_batch * 10_000 + article_id * 100 + paragraph_id * 10 + diff_code``
   (``diff_code`` = 0 for "Adv", 1 for "Ele") — 330 distinct passages, mirroring
   GECO's ``PART * 1000 + TRIAL`` trick for a non-globally-unique trial id.
2. **Practice and repeated-reading trials must be excluded**: ``practice_trial``
   (99,360 rows) and ``repeated_reading_trial`` (422,167 rows) are not part of
   the scoreable first-pass reading-time data.
3. **A small fraction of IA segmentations disagree within the same passage**
   (~0.7% of real trial rows, affecting 15/324 passages and 84/360
   participants): a hyphenated word occasionally wraps across two IA regions
   for some participants/lists (e.g. "top-" + "level") but is a single IA region
   for others ("top-level"), shifting that participant's ``IA_ID`` numbering
   for the rest of the passage. The canonical word at each ``(story_id, zone)``
   is therefore the **modal** ``IA_LABEL`` across all participants who read
   that passage; any trial whose own ``IA_LABEL`` disagrees with the modal word
   for its zone is dropped (same spirit as Provo's ~1%/GECO's ~0.004% drops for
   irreconcilable per-row metadata).
4. **No sentence-number column** (unlike Provo/MECO): sentence boundaries are
   reconstructed by walking each passage's modal word list and marking a break
   after any token ending in ``.``, ``!``, or ``?`` (optionally followed by a
   closing quote/bracket). Verified against all 330 passages this does not
   misfire on: decimal numbers ("96.9%", "1.3" — these never end in bare
   ``.``/``!``/``?``, so they never match), ellipses ("...", "…" — explicitly
   excluded; confirmed to always mark a trailing-off/omission mid-sentence in
   this corpus, never a sentence end), or abbreviations (none of the standard
   forms — "Mr.", "U.S.", "e.g.", "etc." ... — occur in the released passages).
   Resulting segmentation: 3-12 sentences/passage (mean 7.0), sentence length
   1-63 words (mean 17.9); the one 63-word sentence and all seven 1-word
   "sentences" were manually inspected and are genuine (a very long run-on
   sentence and short quoted fragments ending a paragraph, respectively — not
   artifacts of the heuristic).

Output schema of :meth:`OnestopDataset.load_words`
---------------------------------------------------
Mirrors :class:`neuroai.datasets.provo.ProvoDataset` / :class:`neuroai.datasets.geco.GecoDataset`
so it can reuse the same generic LME fitting code
(:func:`neuroai.regressors.ucl_reading_lme.fit_ucl_reading_lme`).

aggregate=True  (one row per word, averaged across subjects who fixated it):
    word        str     Surface form (modal ``IA_LABEL`` across subjects).
    story_id    int     ``article_batch*10_000 + article_id*100 + paragraph_id*10 + diff_code``.
    zone        int     1-based word position within the passage (``IA_ID``).
    rt_mean     float   Mean duration across subjects who fixated the word (ms).
    rt_sd       float   Standard deviation across subjects.
    is_edge     bool    True for sentence- OR passage-initial/-final words.
    wlen        int     Character length of the word.
    context     str     Cumulative passage text through and including ``word``.

aggregate=False (one row per subject-word trial):
    participant_id  str   OneStop ``participant_id``.
    story_id, zone, word, is_edge, wlen, context : as above.
    rt              float Raw single-subject duration for the chosen measure (ms).

Reference
---------
Berzak, Y., Malmaud, J., & Levy, R. (2025). The OneStop Corpus: A large-scale
eye-tracking and comprehension question-answering dataset of naturally occurring
text reading. Behavior Research Methods.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MEASURE_COLUMNS: dict[str, str] = {
    "SP": "IA_DWELL_TIME",
    "FP": "IA_FIRST_RUN_DWELL_TIME",
    "GP": "IA_REGRESSION_PATH_DURATION",
}

_IA_PARAGRAPH_FILENAME = "precomputed_reading_measures/ia_Paragraph.csv"

_STORY_KEY_COLS = ["article_batch", "article_id", "paragraph_id", "difficulty_level"]

# Sentence-final punctuation, optionally followed by a closing quote/bracket.
# Deliberately excludes "..." / "…" — verified (see module docstring) to always
# mark a mid-sentence trailing-off/omission in this corpus, never a true
# sentence boundary. Decimal numbers ("96.9%", "1.3") never end in a bare
# .!? so they never match this pattern (they end in a digit or "%").
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’\)\]]*$")


def _is_sentence_end(word: str) -> bool:
    if word in ("...", "…"):
        return False
    return bool(_SENTENCE_END_RE.search(word))


def _encode_story_id(df: pd.DataFrame) -> pd.Series:
    diff_code = (df["difficulty_level"] == "Ele").astype(int)
    return (
        df["article_batch"].astype(int) * 10_000
        + df["article_id"].astype(int) * 100
        + df["paragraph_id"].astype(int) * 10
        + diff_code
    )


def load_ia_paragraph(root: str | Path) -> pd.DataFrame:
    """Load ``ia_Paragraph.csv``, restricted to real (non-practice,
    non-repeated-reading) trials, with ``story_id``/``zone`` cast and duration
    columns coerced to numeric (the released CSV uses the literal string "."
    for missing values, same convention as Provo's DataViewer export).

    The parse is cached per process (the released CSV is ~5.6 GB): the LME
    benchmark builds two ``OnestopDataset`` instances — subject-aggregated and
    trial-level — that would otherwise each re-read it. The returned frame is
    treated as read-only by all callers (subset/rename produce copies).
    """
    path = Path(root) / _IA_PARAGRAPH_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"OneStop IA-report file not found: {path}")
    return _read_ia_paragraph_cached(str(path.resolve()))


@lru_cache(maxsize=2)
def _read_ia_paragraph_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    usecols = [
        "participant_id",
        *_STORY_KEY_COLS,
        "IA_ID",
        "IA_LABEL",
        "IA_SKIP",
        "practice_trial",
        "repeated_reading_trial",
        *_MEASURE_COLUMNS.values(),
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    n_before = len(df)
    df = df[(~df["practice_trial"]) & (~df["repeated_reading_trial"])]
    logger.info(
        "OneStop: dropped %d/%d practice/repeated-reading rows, kept %d",
        n_before - len(df),
        n_before,
        len(df),
    )

    df["story_id"] = _encode_story_id(df)
    df["zone"] = df["IA_ID"].astype(int)
    for col in _MEASURE_COLUMNS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop(columns=["IA_ID", "practice_trial", "repeated_reading_trial", *_STORY_KEY_COLS])


def _modal_word_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonical (story_id, zone) -> word via modal ``IA_LABEL`` across subjects
    (see module docstring quirk 3: a small fraction of subjects/lists disagree
    on IA segmentation for a hyphenated word wrapping across lines)."""
    counts = raw.groupby(["story_id", "zone", "IA_LABEL"]).size().reset_index(name="n")
    counts = counts.sort_values("n", ascending=False).drop_duplicates(subset=["story_id", "zone"])
    return counts[["story_id", "zone", "IA_LABEL"]].rename(columns={"IA_LABEL": "word"}).sort_values(
        ["story_id", "zone"]
    ).reset_index(drop=True)


class OnestopDataset:
    """Load the OneStop Corpus eye-tracking data for one measure.

    Parameters
    ----------
    root : str or Path
        Dataset root containing ``precomputed_reading_measures/ia_Paragraph.csv``.
    measure : str
        One of "SP" (scan path / total dwell), "FP" (first-pass / gaze
        duration, default), "GP" (go-past / regression-path duration).
    stories : list of int, optional
        Subset of encoded ``story_id`` values to load. Loads all 330 by default.
    min_dur, max_dur : float
        Plausible single-trial duration bounds in ms (implausible durations
        excluded before aggregation) — same defaults as Provo/GECO/MECO.
    exclude_doc_edges : bool
        If True (default), passage-initial/-final words are flagged via
        ``is_edge`` in addition to sentence-initial/-final words.
    aggregate : bool
        If True (default), one row per word with mean/SD duration across the
        subjects who fixated it. If False, one row per (subject, word) trial.
    subjects : list of str, optional
        Restrict the trial-level data to this subset of ``participant_id``
        values (used for the fixed 120-of-360 subset of the LME sweep — see
        ``configs/subjects/onestop_subjects_120.txt``). The canonical word list and the
        modal-word table are still built over **all** subjects, so a subset
        never shifts word identity or its alignment with the measure-
        independent surprisal cache. Loads all 360 subjects by default.
    """

    def __init__(
        self,
        root: str | Path,
        measure: str = "FP",
        *,
        stories: list[int] | None = None,
        min_dur: float = 50.0,
        max_dur: float = 5000.0,
        exclude_doc_edges: bool = True,
        aggregate: bool = True,
        subjects: list[str] | None = None,
    ) -> None:
        measure = measure.upper()
        if measure not in _MEASURE_COLUMNS:
            raise ValueError(f"Unknown OneStop measure {measure!r}; expected one of {sorted(_MEASURE_COLUMNS)}")
        self.root = Path(root)
        self.measure = measure
        self.stories = stories
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.exclude_doc_edges = exclude_doc_edges
        self.aggregate = aggregate
        self.subjects = list(subjects) if subjects is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_words(self) -> pd.DataFrame:
        raw = load_ia_paragraph(self.root)
        if self.stories is not None:
            raw = raw[raw["story_id"].isin(self.stories)]
        # Modal word table / canonical word list are built over ALL subjects so
        # a ``subjects`` subset never shifts word identity or its alignment with
        # the measure-independent surprisal cache.
        modal = _modal_word_table(raw)
        word_df = self._word_list(modal)
        trial_raw = raw
        if self.subjects is not None:
            trial_raw = raw[raw["participant_id"].isin(self.subjects)]
            kept = trial_raw["participant_id"].nunique()
            logger.info(
                "OneStop [%s]: restricted to %d/%d requested subjects present in the data",
                self.measure, kept, len(self.subjects),
            )
        trials = self._load_trials(trial_raw, modal)
        if self.aggregate:
            return self._aggregate(trials, word_df)
        return self._raw(trials, word_df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_trials(self, raw: pd.DataFrame, modal: pd.DataFrame) -> pd.DataFrame:
        measure_col = _MEASURE_COLUMNS[self.measure]
        df = raw[["participant_id", "story_id", "zone", "IA_SKIP", "IA_LABEL", measure_col]].rename(
            columns={measure_col: "rt"}
        )

        n_before = len(df)
        # Drop trials whose own IA_LABEL disagrees with the modal word for that
        # (story_id, zone) — a hyphenated word that wrapped across two IA
        # regions for this participant/list but not for most others (see module
        # docstring quirk 3). Keeping them would silently attach a mismatched
        # word's duration to the canonical word.
        df = df.merge(modal, on=["story_id", "zone"], how="left")
        mismatched = df["IA_LABEL"] != df["word"]
        if mismatched.any():
            logger.info(
                "OneStop [%s]: dropping %d/%d trial rows with IA segmentation "
                "mismatched vs. the modal word for their zone",
                self.measure,
                int(mismatched.sum()),
                n_before,
            )
        df = df[~mismatched].drop(columns=["IA_LABEL", "word"])

        n_before = len(df)
        # IA_SKIP==1 -> the word was never fixated; its duration columns are
        # NaN in the released file for FP/GP, and 0 (not NaN) for SP — drop
        # explicitly first for clarity/consistency across all three measures,
        # mirroring Provo's IA_SKIP handling.
        df = df[df["IA_SKIP"] == 0]
        df = df.dropna(subset=["rt"])
        df = df[(df["rt"] >= self.min_dur) & (df["rt"] <= self.max_dur)]
        logger.info(
            "OneStop [%s]: kept %d/%d word trials (fixated, duration in [%.0f, %.0f] ms)",
            self.measure,
            len(df),
            n_before,
            self.min_dur,
            self.max_dur,
        )
        return df.drop(columns=["IA_SKIP"]).reset_index(drop=True)

    def _word_list(self, modal: pd.DataFrame) -> pd.DataFrame:
        """Per-(story_id, zone) word identity, edge flag, length, and context,
        built from the modal word table (subject-independent)."""
        doc_bounds = modal.groupby("story_id")["zone"].agg(["min", "max"])

        rows: list[dict] = []
        for story_id, group in modal.groupby("story_id", sort=True):
            group = group.sort_values("zone")
            words = group["word"].tolist()
            zones = group["zone"].tolist()
            doc_min = int(doc_bounds.loc[story_id, "min"])
            doc_max = int(doc_bounds.loc[story_id, "max"])

            sent_end_positions: set[int] = set()
            for i, w in enumerate(words):
                if _is_sentence_end(w) or i == len(words) - 1:
                    sent_end_positions.add(i)
            sent_starts = {0} | {i + 1 for i in sent_end_positions if i + 1 < len(words)}

            for i, (zone, word) in enumerate(zip(zones, words)):
                is_sent_edge = (i in sent_starts) or (i in sent_end_positions)
                is_doc_edge = self.exclude_doc_edges and (zone == doc_min or zone == doc_max)
                rows.append(
                    {
                        "story_id": story_id,
                        "zone": zone,
                        "word": word,
                        "is_edge": bool(is_sent_edge or is_doc_edge),
                        "wlen": len(word),
                        "context": " ".join(words[: i + 1]),
                    }
                )
        return pd.DataFrame(rows)

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
            "OneStop [%s]: aggregated %d words across %d passages (mean duration)",
            self.measure,
            len(merged),
            merged["story_id"].nunique(),
        )
        return merged

    def _raw(self, trials: pd.DataFrame, word_df: pd.DataFrame) -> pd.DataFrame:
        merged = trials.merge(word_df, on=["story_id", "zone"], how="inner")
        logger.info(
            "OneStop [%s]: loaded %d trials (%d subjects, %d passages)",
            self.measure,
            len(merged),
            merged["participant_id"].nunique(),
            merged["story_id"].nunique(),
        )
        return merged


def onestop_word_list(root: str | Path, *, exclude_doc_edges: bool = True) -> pd.DataFrame:
    """Measure-independent word list (identity, context, edge flag) for all 330
    OneStop passages — shared across SP/FP/GP since the word set is identical.

    Used by the surprisal prefetch script so surprisal is computed **once** per
    (model, checkpoint), not once per measure.
    """
    raw = load_ia_paragraph(root)
    modal = _modal_word_table(raw)
    return OnestopDataset(root, "FP", exclude_doc_edges=exclude_doc_edges, aggregate=True)._word_list(modal)
