"""Provo Corpus eye-tracking reading-time adapter (Luke and Christianson, 2018).

Supports three by-word duration measures, selected via ``measure``:

- **SP** (scan path / total dwell time)   -> ``IA_DWELL_TIME``
- **FP** (first-pass / gaze duration)      -> ``IA_FIRST_RUN_DWELL_TIME``
- **GP** (go-past / regression-path duration) -> ``IA_REGRESSION_PATH_DURATION``

These correspond to the "Provo SP / Provo FP / Provo GP" rows used by Clark et al.
(2025, COLING) and other Oh/Schuler-style reading-time benchmarks.

Dataset layout expected at ``root/``
-------------------------------------
Provo_Corpus-Eyetracking_Data.csv          — one row per (Participant_ID, Text_ID,
                                              Word_Number): IA-level eye-tracking
                                              measures from EyeLink DataViewer.
Provo_Corpus-Predictability_Norms.csv      — one row per (Text_ID, Word_Number,
                                              cloze response); also carries the full
                                              raw passage text in ``Text`` (latin-1
                                              encoded).

Important data quirk (verified against the released CSVs)
-----------------------------------------------------------
The **passage-initial word of every text is absent** from both CSVs (``Word_Number``
starts at 2) because it cannot be cloze-normed (no preceding context). Concatenating
only the ``Word`` column would therefore silently drop the true first word from every
context string. Instead, the authoritative token sequence for each ``Text_ID`` is
recovered by whitespace-splitting the full passage text in
``Predictability_Norms.csv``'s ``Text`` column (verified to align 1:1 with
``Word_Number`` — token index 1 is the missing passage-initial word, index 2 matches
the first row of the eye-tracking data), mirroring
:func:`neuroai.datasets.ucl_reading.sentence_word_lists`.

Output schema of :meth:`ProvoDataset.load_words`
---------------------------------------------------
Mirrors :class:`neuroai.datasets.ucl_reading.UCLReadingDataset` /
:class:`neuroai.datasets.zuco_reading.ZuCoReadingDataset` so it can reuse the same
generic LME fitting code
(:func:`neuroai.regressors.ucl_reading_lme.fit_ucl_reading_lme`).

aggregate=True  (one row per word, averaged across subjects who fixated it):
    word        str     Surface form (whitespace-split token from the passage text).
    story_id    int     ``Text_ID``.
    zone        int     1-based word position within the passage (``Word_Number``).
    rt_mean     float   Mean duration across subjects who fixated the word (ms).
    rt_sd       float   Standard deviation across subjects.
    is_edge     bool    True for sentence- OR passage-initial/-final words (Provo's
                         own filtering excludes both; see Clark et al. 2025 sec 4.3).
    wlen        int     Character length of the word.
    context     str     Cumulative passage text through and including ``word``.

aggregate=False (one row per subject-word trial):
    participant_id  str   ``Participant_ID`` (e.g. "Sub01").
    story_id, zone, word, is_edge, wlen, context : as above.
    rt              float Raw single-subject duration for the chosen measure (ms).

Reference
---------
Luke, S.G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking
corpus with predictability norms. Behavior Research Methods, 50, 826-833.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# IA-level DataViewer column feeding each measure, and whether skipped words leave
# a NaN there (SP/IA_DWELL_TIME is 0, not NaN, for a skipped word — see docstring
# above and neuroai.datasets.provo module-level notes in the design discussion).
_MEASURE_COLUMNS: dict[str, str] = {
    "SP": "IA_DWELL_TIME",
    "FP": "IA_FIRST_RUN_DWELL_TIME",
    "GP": "IA_REGRESSION_PATH_DURATION",
}

_EYETRACKING_FILENAME = "Provo_Corpus-Eyetracking_Data.csv"
_PREDICTABILITY_FILENAME = "Provo_Corpus-Predictability_Norms.csv"


def load_predictability_norms(root: str | Path) -> pd.DataFrame:
    """Load ``Provo_Corpus-Predictability_Norms.csv`` (latin-1 encoded)."""
    path = Path(root) / _PREDICTABILITY_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Provo predictability-norms file not found: {path}")
    df = pd.read_csv(path, encoding="latin-1")
    df["Text_ID"] = df["Text_ID"].astype(int)
    return df


def passage_word_lists(root: str | Path) -> dict[int, list[str]]:
    """Return ``{text_id: [word1, word2, ...]}`` via whitespace split of the full
    passage text (``Text`` column), 1-based-index-aligned with ``Word_Number``.

    Token index 1 is the passage-initial word, which never appears as a scoreable
    row in either CSV (see module docstring) but is required so that word 2's
    ``context`` includes it.
    """
    norms = load_predictability_norms(root)
    out: dict[int, list[str]] = {}
    for text_id, group in norms.groupby("Text_ID", sort=True):
        text = str(group["Text"].iloc[0])
        out[int(text_id)] = text.strip().split()
    return out


class ProvoDataset:
    """Load the Provo Corpus eye-tracking reading-time data for one measure.

    Parameters
    ----------
    root : str or Path
        Dataset root containing ``Provo_Corpus-Eyetracking_Data.csv`` and
        ``Provo_Corpus-Predictability_Norms.csv``.
    measure : str
        One of "SP" (scan path / total dwell), "FP" (first-pass / gaze duration),
        "GP" (go-past / regression-path duration).
    texts : list of int, optional
        Subset of ``Text_ID`` (1-55) to load. Loads all 55 by default.
    min_dur, max_dur : float
        Plausible single-trial duration bounds in ms (implausible durations
        excluded before aggregation) — same defaults as ZuCo Reading TRT.
    exclude_doc_edges : bool
        If True (default), passage-initial/-final words are flagged via
        ``is_edge`` in addition to sentence-initial/-final words, matching the
        filtering used for Provo in Clark et al. (2025): "data points ... at
        starts and ends of ... documents" are excluded downstream by the shared
        LME prep code's ``filter_sentence_edges`` (which drops any row where
        ``is_edge`` is True).
    aggregate : bool
        If True (default), one row per word with mean/SD duration across the
        subjects who fixated it. If False, one row per (subject, word) trial.
    """

    def __init__(
        self,
        root: str | Path,
        measure: str,
        *,
        texts: list[int] | None = None,
        min_dur: float = 50.0,
        max_dur: float = 5000.0,
        exclude_doc_edges: bool = True,
        aggregate: bool = True,
    ) -> None:
        measure = measure.upper()
        if measure not in _MEASURE_COLUMNS:
            raise ValueError(f"Unknown Provo measure {measure!r}; expected one of {sorted(_MEASURE_COLUMNS)}")
        self.root = Path(root)
        self.measure = measure
        self.texts = texts
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.exclude_doc_edges = exclude_doc_edges
        self.aggregate = aggregate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_words(self) -> pd.DataFrame:
        trials = self._load_trials()
        word_df = self._word_list()
        if self.aggregate:
            return self._aggregate(trials, word_df)
        return self._raw(trials, word_df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_trials(self) -> pd.DataFrame:
        path = self.root / _EYETRACKING_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Provo eye-tracking file not found: {path}")

        usecols = [
            "Participant_ID",
            "Text_ID",
            "Word_Number",
            "IA_SKIP",
            _MEASURE_COLUMNS[self.measure],
        ]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df = df.rename(
            columns={
                "Participant_ID": "participant_id",
                "Text_ID": "story_id",
                "Word_Number": "zone",
                _MEASURE_COLUMNS[self.measure]: "rt",
            }
        )
        # A handful of trailing per-trial IA rows (e.g. an end-of-passage blank
        # region) carry no Word_Number — not real word trials, drop before casting.
        df = df.dropna(subset=["zone"])
        df["story_id"] = df["story_id"].astype(int)
        df["zone"] = df["zone"].astype(int)

        if self.texts is not None:
            df = df[df["story_id"].isin(self.texts)]

        n_before = len(df)
        # A word never fixated (IA_SKIP==1) is not a real reading-time observation
        # for any measure: SP/IA_DWELL_TIME reads 0 (not NaN) for skipped words,
        # while FP/GP are NaN — dropping on IA_SKIP first keeps all three measures
        # consistent instead of relying on per-column NaN behavior.
        df = df[df["IA_SKIP"] == 0]
        df = df.dropna(subset=["rt"])
        df = df[(df["rt"] >= self.min_dur) & (df["rt"] <= self.max_dur)]
        logger.info(
            "Provo [%s]: kept %d/%d word trials (fixated, duration in [%.0f, %.0f] ms)",
            self.measure,
            len(df),
            n_before,
            self.min_dur,
            self.max_dur,
        )
        return df.drop(columns=["IA_SKIP"]).reset_index(drop=True)

    def _sentence_boundaries(self) -> pd.DataFrame:
        """Per-(story_id, zone) sentence id and within-sentence position, from the
        eye-tracking file's own ``Sentence_Number`` / ``Word_In_Sentence_Number``
        columns (subject-independent, so read once and de-duplicated)."""
        path = self.root / _EYETRACKING_FILENAME
        df = pd.read_csv(
            path,
            usecols=["Text_ID", "Word_Number", "Sentence_Number", "Word_In_Sentence_Number"],
            low_memory=False,
        )
        df = df.dropna(subset=["Word_Number"]).drop_duplicates(subset=["Text_ID", "Word_Number"]).rename(
            columns={
                "Text_ID": "story_id",
                "Word_Number": "zone",
                "Sentence_Number": "sent_id",
                "Word_In_Sentence_Number": "sent_pos",
            }
        )
        df["story_id"] = df["story_id"].astype(int)
        df["zone"] = df["zone"].astype(int)
        return df

    def _word_list(self) -> pd.DataFrame:
        """Per-(story_id, zone) word identity, edge flag, length, and context.

        ``context`` is the cumulative passage text through and including the
        target word (NeuralSet ``AddConcatenationContext`` convention), built from
        the full ``Text`` field so the true (unscoreable) passage-initial word is
        still present as context for word 2 onward — see module docstring.
        """
        passages = passage_word_lists(self.root)
        sent_bounds = self._sentence_boundaries()
        sent_lookup = sent_bounds.set_index(["story_id", "zone"])

        # Sentence-final word = last zone with a given sent_id within a story.
        sent_last_zone = (
            sent_bounds.sort_values("zone")
            .groupby(["story_id", "sent_id"])["zone"]
            .max()
        )

        rows: list[dict] = []
        for story_id, tokens in passages.items():
            if self.texts is not None and story_id not in self.texts:
                continue
            n = len(tokens)
            story_sent = sent_bounds[sent_bounds["story_id"] == story_id]
            if story_sent.empty:
                continue
            min_zone = int(story_sent["zone"].min())
            max_zone = int(story_sent["zone"].max())
            last_zone_by_sent = (
                sent_last_zone.xs(story_id, level="story_id")
                if story_id in sent_last_zone.index.get_level_values("story_id")
                else pd.Series(dtype=int)
            )

            n_missing_sent_info = 0
            for zone in sorted(story_sent["zone"].unique()):
                i = zone - 1  # 0-based index into tokens (tokens[0] == word 1, the unscoreable passage-initial word)
                if i < 0 or i >= n:
                    continue
                key = (story_id, zone)
                sent_id_raw = sent_lookup.loc[key, "sent_id"]
                sent_pos_raw = sent_lookup.loc[key, "sent_pos"]
                if pd.isna(sent_id_raw) or pd.isna(sent_pos_raw):
                    # ~1% of Provo words (27/2743) have no Sentence_Number /
                    # Word_In_Sentence_Number in the released CSV — sentence
                    # boundaries can't be determined, so drop rather than guess.
                    n_missing_sent_info += 1
                    continue
                word = tokens[i]
                sent_id = int(sent_id_raw)
                sent_pos = int(sent_pos_raw)
                is_sent_edge = (sent_pos == 1) or (zone == int(last_zone_by_sent.get(sent_id, -1)))
                is_doc_edge = self.exclude_doc_edges and (zone == min_zone or zone == max_zone)
                rows.append(
                    {
                        "story_id": story_id,
                        "zone": zone,
                        "word": word,
                        "is_edge": bool(is_sent_edge or is_doc_edge),
                        "wlen": len(word),
                        "context": " ".join(tokens[: i + 1]),
                    }
                )
            if n_missing_sent_info:
                logger.debug(
                    "Provo: story_id=%d dropped %d word(s) with no Sentence_Number metadata",
                    story_id,
                    n_missing_sent_info,
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
            "Provo [%s]: aggregated %d words across %d passages (mean duration)",
            self.measure,
            len(merged),
            merged["story_id"].nunique(),
        )
        return merged

    def _raw(self, trials: pd.DataFrame, word_df: pd.DataFrame) -> pd.DataFrame:
        merged = trials.merge(word_df, on=["story_id", "zone"], how="inner")
        logger.info(
            "Provo [%s]: loaded %d trials (%d subjects, %d passages)",
            self.measure,
            len(merged),
            merged["participant_id"].nunique(),
            merged["story_id"].nunique(),
        )
        return merged


def provo_word_list(root: str | Path, *, exclude_doc_edges: bool = True) -> pd.DataFrame:
    """Measure-independent word list (identity, context, edge flag) for all 55
    Provo passages — shared across SP/FP/GP since the word set is identical.

    Used by the surprisal prefetch script so surprisal is computed **once** per
    (model, checkpoint), not once per measure (see
    :mod:`neuroai.features.surprisal`'s Provo cache functions, which are keyed
    only by model/revision, not by measure).
    """
    return ProvoDataset(
        root, "SP", exclude_doc_edges=exclude_doc_edges, aggregate=True
    )._word_list()
