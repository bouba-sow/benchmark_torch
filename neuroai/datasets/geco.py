"""GECO Corpus eye-tracking reading-time adapter (Cop, Dirix, Drieghe & Duyck, 2017).

Supports the two by-word duration measures available for the monolingual English
data (GECO provides no raw per-fixation report, so — unlike Provo — a scan-path
/ total-dwell measure cannot be derived; see Clark et al. 2025, COLING, footnote
7: "The SP duration could not be calculated for the GECO and UCL corpora that do
not provide raw eye fixation durations."):

- **FP** (first-pass / gaze duration) -> ``WORD_GAZE_DURATION``
- **GP** (go-past / regression-path duration) -> ``WORD_GO_PAST_TIME``

Dataset layout expected at ``root/``
-------------------------------------
MonolingualReadingData.xlsx  — sheet ``DATA``: one row per (PP_NR, PART, TRIAL,
                                WORD_ID_WITHIN_TRIAL) — per-subject word-level eye-
                                tracking measures for the 14 monolingual English
                                readers of "The Mysterious Affair at Styles".
EnglishMaterial.xlsx          — sheet ``ALL``: one row per WORD_ID ("part-trial-
                                word") with SENTENCE_ID ("part-sentnum") and
                                CHRON_ID (chronological word position within the
                                trial — verified identical to
                                ``WORD_ID_WITHIN_TRIAL`` above, so the two files
                                join cleanly on (story_id, zone)); sheet
                                ``SENTENCE``: full sentence text, unused here
                                (context is built by concatenating ``ALL``'s
                                per-word tokens directly, mirroring Provo).

Important data quirk (verified against the released xlsx files)
-----------------------------------------------------------------
GECO's ``TRIAL`` numbers are **not globally unique** — they restart within each
``PART`` (e.g. both PART 1 and PART 2 contain a TRIAL 5). ``story_id`` is
therefore encoded as ``PART * 1000 + TRIAL`` (588 distinct (PART, TRIAL) pairs
in total, each one screen of text shown to a participant — the natural
"passage" unit, analogous to a Provo text or a Natural Stories item).

Output schema of :meth:`GecoDataset.load_words`
---------------------------------------------------
Mirrors :class:`neuroai.datasets.provo.ProvoDataset` so it can reuse the same
generic LME fitting code (:func:`neuroai.regressors.ucl_reading_lme.fit_ucl_reading_lme`).

aggregate=True  (one row per word, averaged across subjects who fixated it):
    word        str     Surface form.
    story_id    int     ``PART * 1000 + TRIAL``.
    zone        int     1-based chronological word position within the trial.
    rt_mean     float   Mean duration across subjects who fixated the word (ms).
    rt_sd       float   Standard deviation across subjects.
    is_edge     bool    True for sentence- OR trial("document")-initial/-final
                         words, matching Provo's ``exclude_doc_edges`` convention.
    wlen        int     Character length of the word.
    context     str     Cumulative trial text through and including ``word``.

aggregate=False (one row per subject-word trial):
    participant_id  str   ``PP_NR`` (e.g. "pp21").
    story_id, zone, word, is_edge, wlen, context : as above.
    rt              float Raw single-subject duration for the chosen measure (ms).

Reference
---------
Cop, U., Dirix, N., Drieghe, D., & Duyck, W. (2017). Presenting GECO: An eye-
tracking corpus of monolingual and bilingual sentence reading. Behavior
Research Methods, 49(2), 602-615.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MEASURE_COLUMNS: dict[str, str] = {
    "FP": "WORD_GAZE_DURATION",
    "GP": "WORD_GO_PAST_TIME",
}

_READING_DATA_FILENAME = "MonolingualReadingData.xlsx"
_MATERIAL_FILENAME = "EnglishMaterial.xlsx"


def load_english_material(root: str | Path) -> pd.DataFrame:
    """Load ``EnglishMaterial.xlsx``'s ``ALL`` sheet as (story_id, zone, word,
    sentence_id) — subject-independent, so read once and shared across measures.
    """
    path = Path(root) / _MATERIAL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"GECO English material file not found: {path}")

    df = pd.read_excel(path, sheet_name="ALL", usecols=["WORD_ID", "SENTENCE_ID", "WORD"], engine="openpyxl")
    n_before = len(df)
    # A handful of rows (2/54,362 in the released file) have no SENTENCE_ID —
    # sentence membership can't be determined for them, so drop rather than guess
    # (mirrors Provo's handling of its own ~1% missing sentence metadata).
    df = df.dropna(subset=["WORD_ID", "SENTENCE_ID"])
    if len(df) < n_before:
        logger.debug("GECO: dropped %d word(s) with no SENTENCE_ID metadata", n_before - len(df))

    # openpyxl auto-types a WORD cell containing the literal text "FALSE"/"TRUE"
    # as a Python bool (1 occurrence in the released file: the word "False") —
    # recover the original text via str(). A couple of cells are genuinely blank
    # (NaN); those can't be recovered, so drop them.
    word_is_na = df["WORD"].isna()
    if word_is_na.any():
        logger.debug("GECO: dropped %d word(s) with a blank WORD cell", int(word_is_na.sum()))
        df = df[~word_is_na]
    df["WORD"] = df["WORD"].apply(lambda w: str(w) if isinstance(w, bool) else w)

    ids = df["WORD_ID"].str.split("-", expand=True).astype(int)
    sent_ids = df["SENTENCE_ID"].str.split("-", expand=True).astype(int)
    out = pd.DataFrame(
        {
            "story_id": ids[0] * 1000 + ids[1],
            "zone": ids[2],
            "word": df["WORD"].astype(str),
            # sentence_id encoded the same way as story_id (part*1000+sentnum) so it
            # cannot collide across parts, mirroring the story_id convention above.
            "sentence_id": sent_ids[0] * 1000 + sent_ids[1],
        }
    )
    return out


class GecoDataset:
    """Load the GECO Corpus (monolingual English) eye-tracking data for one measure.

    Parameters
    ----------
    root : str or Path
        Dataset root containing ``MonolingualReadingData.xlsx`` and
        ``EnglishMaterial.xlsx``.
    measure : str
        One of "FP" (first-pass / gaze duration), "GP" (go-past / regression-
        path duration). GECO has no raw fixation report, so a scan-path/dwell
        (SP) measure is not available (see module docstring).
    trials : list of int, optional
        Subset of encoded ``story_id`` (``PART * 1000 + TRIAL``) to load. Loads
        all 588 trials by default.
    min_dur, max_dur : float
        Plausible single-trial duration bounds in ms (implausible durations
        excluded before aggregation) — same defaults as Provo/ZuCo Reading.
    exclude_doc_edges : bool
        If True (default), trial-initial/-final words are flagged via
        ``is_edge`` in addition to sentence-initial/-final words, mirroring
        Provo's document-edge exclusion.
    aggregate : bool
        If True (default), one row per word with mean/SD duration across the
        subjects who fixated it. If False, one row per (subject, word) trial.
    """

    def __init__(
        self,
        root: str | Path,
        measure: str,
        *,
        trials: list[int] | None = None,
        min_dur: float = 50.0,
        max_dur: float = 5000.0,
        exclude_doc_edges: bool = True,
        aggregate: bool = True,
    ) -> None:
        measure = measure.upper()
        if measure not in _MEASURE_COLUMNS:
            raise ValueError(f"Unknown GECO measure {measure!r}; expected one of {sorted(_MEASURE_COLUMNS)}")
        self.root = Path(root)
        self.measure = measure
        self.trials = trials
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.exclude_doc_edges = exclude_doc_edges
        self.aggregate = aggregate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_words(self) -> pd.DataFrame:
        trials_df = self._load_trials()
        word_df = self._word_list()
        if self.aggregate:
            return self._aggregate(trials_df, word_df)
        return self._raw(trials_df, word_df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_trials(self) -> pd.DataFrame:
        path = self.root / _READING_DATA_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"GECO reading-time file not found: {path}")

        usecols = ["PP_NR", "PART", "TRIAL", "WORD_ID_WITHIN_TRIAL", "WORD_SKIP", _MEASURE_COLUMNS[self.measure]]
        df = pd.read_excel(path, sheet_name="DATA", usecols=usecols, engine="openpyxl")
        df = df.rename(
            columns={
                "PP_NR": "participant_id",
                "WORD_ID_WITHIN_TRIAL": "zone",
                _MEASURE_COLUMNS[self.measure]: "rt",
            }
        )
        df["story_id"] = df["PART"].astype(int) * 1000 + df["TRIAL"].astype(int)
        df["zone"] = df["zone"].astype(int)
        df = df.drop(columns=["PART", "TRIAL"])

        if self.trials is not None:
            df = df[df["story_id"].isin(self.trials)]

        n_before = len(df)
        # WORD_SKIP==1 means the word was never fixated in first-pass reading, so
        # FP/GP are not real observations for it (both read NaN in the released
        # file) — drop explicitly for clarity, matching Provo's IA_SKIP handling.
        df = df[df["WORD_SKIP"] == 0]
        df = df.dropna(subset=["rt"])
        df = df[(df["rt"] >= self.min_dur) & (df["rt"] <= self.max_dur)]
        logger.info(
            "GECO [%s]: kept %d/%d word trials (fixated, duration in [%.0f, %.0f] ms)",
            self.measure,
            len(df),
            n_before,
            self.min_dur,
            self.max_dur,
        )
        return df.drop(columns=["WORD_SKIP"]).reset_index(drop=True)

    def _word_list(self) -> pd.DataFrame:
        """Per-(story_id, zone) word identity, edge flag, length, and context."""
        material = load_english_material(self.root)
        if self.trials is not None:
            material = material[material["story_id"].isin(self.trials)]
        material = material.sort_values(["story_id", "zone"]).reset_index(drop=True)

        sent_bounds = (
            material.groupby(["story_id", "sentence_id"])["zone"].agg(["min", "max"]).reset_index()
        )
        sent_edges = set()
        for _, row in sent_bounds.iterrows():
            sent_edges.add((row["story_id"], row["min"]))
            sent_edges.add((row["story_id"], row["max"]))

        doc_bounds = material.groupby("story_id")["zone"].agg(["min", "max"])

        rows: list[dict] = []
        for story_id, group in material.groupby("story_id", sort=True):
            words = group["word"].tolist()
            zones = group["zone"].tolist()
            doc_min = int(doc_bounds.loc[story_id, "min"])
            doc_max = int(doc_bounds.loc[story_id, "max"])
            for i, (zone, word) in enumerate(zip(zones, words)):
                is_sent_edge = (story_id, zone) in sent_edges
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
            "GECO [%s]: aggregated %d words across %d trials (mean duration)",
            self.measure,
            len(merged),
            merged["story_id"].nunique(),
        )
        return merged

    def _raw(self, trials: pd.DataFrame, word_df: pd.DataFrame) -> pd.DataFrame:
        merged = trials.merge(word_df, on=["story_id", "zone"], how="inner")
        logger.info(
            "GECO [%s]: loaded %d trials (%d subjects, %d text trials)",
            self.measure,
            len(merged),
            merged["participant_id"].nunique(),
            merged["story_id"].nunique(),
        )
        return merged


def geco_word_list(root: str | Path, *, exclude_doc_edges: bool = True) -> pd.DataFrame:
    """Measure-independent word list (identity, context, edge flag) for all 588
    GECO trials — shared across FP/GP since the word set is identical.

    Used by the surprisal prefetch script so surprisal is computed **once** per
    (model, checkpoint), not once per measure.
    """
    return GecoDataset(
        root, "FP", exclude_doc_edges=exclude_doc_edges, aggregate=True
    )._word_list()
