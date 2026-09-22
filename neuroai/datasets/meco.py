"""MECO (Multilingual Eye-movement Corpus) reading-time adapter — English subset.

MECO (Siegelman et al., 2022) collected eye-tracking during reading of the same
12 encyclopedia-style passages in 13 languages. This adapter only wires up the
**English** subset (``lang == "en"``); the underlying ``.rda`` table covers all
13 languages, so extending to another language later is a one-line change to
``language=``, not new parsing logic — the real added cost of a
fully-multilingual benchmark is per-language surprisal/context, which is out of
scope here.

Supports the three by-word duration measures, selected via ``measure``:

- **SP** (scan path / total dwell time)      -> ``dur``
- **FP** (first-pass / gaze duration)         -> ``firstrun.dur``
- **GP** (go-past / regression-path duration) -> ``firstrun.gopast``

Dataset layout expected at ``root/``
-------------------------------------
joint_l1_data_trimmed_version2.1.rda — one row per (uniform_id, trialid, wordnum):
    word-level eye-tracking measures for all 13 languages/13000ish subjects in
    MECO Release 2.0 / Wave 1. Path:
    ``osfstorage/release 2.0/version 2.1/wave 1/primary data/eye tracking data/``
    relative to the MECO dataset root (pass that directory as ``root`` here, or
    override the filename via ``rda_filename``).

Important data quirk (verified against the released ``.rda``)
-----------------------------------------------------------------
Unlike Provo (whose passage-initial word is entirely absent from both released
CSVs), MECO's word-level table already carries the actual surface form in its
own ``word`` column for every ``wordnum``, including the passage-initial word —
so, unlike Provo/UCL, no separate "recover the missing first word from raw
passage text" step is needed; word identity and ``context`` are both built
directly from this one table.

Output schema of :meth:`MecoDataset.load_words`
---------------------------------------------------
Mirrors :class:`neuroai.datasets.provo.ProvoDataset` / :class:`neuroai.datasets.geco.GecoDataset`
so it can reuse the same generic LME fitting code
(:func:`neuroai.regressors.ucl_reading_lme.fit_ucl_reading_lme`).

aggregate=True  (one row per word, averaged across subjects who fixated it):
    word        str     Surface form.
    story_id    int     ``trialid`` (1-12).
    zone        int     1-based word position within the passage (``wordnum``).
    rt_mean     float   Mean duration across subjects who fixated the word (ms).
    rt_sd       float   Standard deviation across subjects.
    is_edge     bool    True for sentence- OR passage-initial/-final words.
    wlen        int     Character length of the word.
    context     str     Cumulative passage text through and including ``word``.

aggregate=False (one row per subject-word trial):
    participant_id  str   ``uniform_id`` (e.g. "en_3").
    story_id, zone, word, is_edge, wlen, context : as above.
    rt              float Raw single-subject duration for the chosen measure (ms).

Reference
---------
Siegelman, N., Schroeder, S., Acartürk, C., et al. (2022). Expanding horizons of
cross-linguistic research on reading: The Multilingual Eye-movement Corpus
(MECO). Behavior Research Methods, 54, 2843-2863.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MEASURE_COLUMNS: dict[str, str] = {
    "SP": "dur",
    "FP": "firstrun.dur",
    "GP": "firstrun.gopast",
}

_DEFAULT_RDA_FILENAME = "joint_l1_data_trimmed_version2.1.rda"


def load_meco_word_table(root: str | Path, *, language: str = "en", rda_filename: str = _DEFAULT_RDA_FILENAME) -> pd.DataFrame:
    """Load the raw MECO joint word-level table, filtered to one language.

    Cast to plain (story_id, zone, participant_id, word, sentence_id, rt-measure)
    columns; ``.rda`` numeric columns arrive as floats (R has no int type here),
    so ids are explicitly cast to int.
    """
    import pyreadr

    path = Path(root) / rda_filename
    if not path.is_file():
        raise FileNotFoundError(f"MECO word-level .rda file not found: {path}")

    result = pyreadr.read_r(str(path))
    df = next(iter(result.values()))
    df = df[df["lang"] == language].copy()
    if df.empty:
        raise ValueError(f"No MECO rows found for language={language!r} in {path}")

    df["story_id"] = df["trialid"].astype(int)
    df["zone"] = df["wordnum"].astype(int)
    df["sentence_id"] = df["story_id"] * 10_000 + df["sentnum"].astype(int)
    df["participant_id"] = df["uniform_id"].astype(str)
    df["word"] = df["word"].astype(str)
    return df


class MecoDataset:
    """Load the MECO English-subset eye-tracking data for one measure.

    Parameters
    ----------
    root : str or Path
        Directory containing the MECO joint word-level ``.rda`` file (see
        module docstring for the expected path within a MECO OSF download).
    measure : str
        One of "SP" (scan path / total dwell), "FP" (first-pass / gaze
        duration), "GP" (go-past / regression-path duration).
    language : str
        MECO language code (default "en" = English; see module docstring —
        other languages are not otherwise wired up by this benchmark).
    passages : list of int, optional
        Subset of ``trialid`` (1-12) to load. Loads all 12 by default.
    min_dur, max_dur : float
        Plausible single-trial duration bounds in ms (implausible durations
        excluded before aggregation) — same defaults as Provo/GECO.
    exclude_doc_edges : bool
        If True (default), passage-initial/-final words are flagged via
        ``is_edge`` in addition to sentence-initial/-final words.
    aggregate : bool
        If True (default), one row per word with mean/SD duration across the
        subjects who fixated it. If False, one row per (subject, word) trial.
    rda_filename : str
        Override the ``.rda`` filename (default: the Release 2.0 / Version 2.1
        / Wave 1 joint L1 file — see module docstring).
    """

    def __init__(
        self,
        root: str | Path,
        measure: str,
        *,
        language: str = "en",
        passages: list[int] | None = None,
        min_dur: float = 50.0,
        max_dur: float = 5000.0,
        exclude_doc_edges: bool = True,
        aggregate: bool = True,
        rda_filename: str = _DEFAULT_RDA_FILENAME,
    ) -> None:
        measure = measure.upper()
        if measure not in _MEASURE_COLUMNS:
            raise ValueError(f"Unknown MECO measure {measure!r}; expected one of {sorted(_MEASURE_COLUMNS)}")
        self.root = Path(root)
        self.measure = measure
        self.language = language
        self.passages = passages
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.exclude_doc_edges = exclude_doc_edges
        self.aggregate = aggregate
        self.rda_filename = rda_filename

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_words(self) -> pd.DataFrame:
        raw = load_meco_word_table(self.root, language=self.language, rda_filename=self.rda_filename)
        if self.passages is not None:
            raw = raw[raw["story_id"].isin(self.passages)]
        trials = self._load_trials(raw)
        word_df = self._word_list(raw)
        if self.aggregate:
            return self._aggregate(trials, word_df)
        return self._raw(trials, word_df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_trials(self, raw: pd.DataFrame) -> pd.DataFrame:
        measure_col = _MEASURE_COLUMNS[self.measure]
        df = raw[["participant_id", "story_id", "zone", "skip", measure_col]].rename(
            columns={measure_col: "rt"}
        )
        n_before = len(df)
        # skip==1 -> the IA was never fixated in the trial; its duration columns
        # are already NaN in the released table, but filter explicitly for
        # clarity and to stay consistent with Provo/GECO's skip handling.
        df = df[df["skip"] == 0]
        df = df.dropna(subset=["rt"])
        df = df[(df["rt"] >= self.min_dur) & (df["rt"] <= self.max_dur)]
        logger.info(
            "MECO [%s/%s]: kept %d/%d word trials (fixated, duration in [%.0f, %.0f] ms)",
            self.language,
            self.measure,
            len(df),
            n_before,
            self.min_dur,
            self.max_dur,
        )
        return df.drop(columns=["skip"]).reset_index(drop=True)

    def _word_list(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Per-(story_id, zone) word identity, edge flag, length, and context.

        Word identity/sentence membership are subject-independent, so de-dupe
        the raw table down to one row per (story_id, zone) first.
        """
        material = (
            raw[["story_id", "zone", "word", "sentence_id"]]
            .drop_duplicates(subset=["story_id", "zone"])
            .sort_values(["story_id", "zone"])
            .reset_index(drop=True)
        )

        sent_bounds = material.groupby(["story_id", "sentence_id"])["zone"].agg(["min", "max"]).reset_index()
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
            "MECO [%s/%s]: aggregated %d words across %d passages (mean duration)",
            self.language,
            self.measure,
            len(merged),
            merged["story_id"].nunique(),
        )
        return merged

    def _raw(self, trials: pd.DataFrame, word_df: pd.DataFrame) -> pd.DataFrame:
        merged = trials.merge(word_df, on=["story_id", "zone"], how="inner")
        logger.info(
            "MECO [%s/%s]: loaded %d trials (%d subjects, %d passages)",
            self.language,
            self.measure,
            len(merged),
            merged["participant_id"].nunique(),
            merged["story_id"].nunique(),
        )
        return merged


def meco_word_list(
    root: str | Path, *, language: str = "en", exclude_doc_edges: bool = True, rda_filename: str = _DEFAULT_RDA_FILENAME
) -> pd.DataFrame:
    """Measure-independent word list (identity, context, edge flag) for all 12
    MECO passages in one language — shared across SP/FP/GP since the word set
    is identical.

    Used by the surprisal prefetch script so surprisal is computed **once** per
    (model, checkpoint), not once per measure.
    """
    ds = MecoDataset(
        root, "FP", language=language, exclude_doc_edges=exclude_doc_edges, aggregate=True, rda_filename=rda_filename
    )
    raw = load_meco_word_table(root, language=language, rda_filename=rda_filename)
    return ds._word_list(raw)
