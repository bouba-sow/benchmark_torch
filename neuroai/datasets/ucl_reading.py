"""UCL self-paced reading corpus adapter (Frank et al. 2013).

Reference
---------
Frank, S.L., Fernandez Monsalve, I., Thompson, R.L., & Vigliocco, G. (2013).
Reading time data for evaluating broad-coverage models of English sentence
processing. Behavior Research Methods, 45, 1182-1190.

Dataset layout expected at ``root/``
-------------------------------------
stimuli.txt                — sent_nr, sentence, question, answer (latin-1 encoded)
selfpacedreading.RT.txt    — subj_nr, sent_nr, sent_pos, correct, answer_time,
                              word_pos, word, RT  (raw per-subject-per-word trials)
selfpacedreading.subj.txt  — subj_nr, age, age_en, sex, hand, correct
                              (per-subject comprehension-question accuracy)

Output schema of :meth:`UCLReadingDataset.load_words`
------------------------------------------------------
aggregate=True  (one row per word, averaged across subjects):
    word        str     Surface form (whitespace-split token from ``stimuli.txt``).
    story_id    int     Sentence number (``sent_nr``).
    zone        int     1-based word position within the sentence (``word_pos``).
    rt_mean     float   Mean self-paced reading time across subjects (ms).
    rt_sd       float   Standard deviation across subjects.
    is_edge     bool    True for sentence-initial/-final words.
    wlen        int     Character length of the word.
    context     str     Cumulative sentence text through and including ``word``
                         (matches NeuralSet's ``AddConcatenationContext`` — the
                         format :func:`neuroai.features.surprisal.batch_surprisal_one_model`
                         expects).

aggregate=False (one row per subject-word trial):
    participant_id  int   ``subj_nr``.
    story_id, zone, word, is_edge, wlen, context : as above.
    rt              float Raw single-trial RT (ms).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_stimuli(root: str | Path) -> pd.DataFrame:
    """Load ``stimuli.txt`` (latin-1 encoded — contains curly-quote characters)."""
    path = Path(root) / "stimuli.txt"
    if not path.is_file():
        raise FileNotFoundError(f"UCL Reading stimuli file not found: {path}")
    df = pd.read_csv(path, sep="\t", encoding="latin-1")
    df["sent_nr"] = df["sent_nr"].astype(int)
    return df


def sentence_word_lists(root: str | Path) -> dict[int, list[str]]:
    """Return ``{sent_nr: [word1, word2, ...]}`` via whitespace split.

    Confirmed to align exactly with ``selfpacedreading.RT.txt``'s 1-based
    ``word_pos`` (e.g. sentence 2 "Billy wrote on the envelope." splits into
    5 tokens matching ``word_pos`` 1-5).
    """
    stim = load_stimuli(root)
    return {
        int(row.sent_nr): str(row.sentence).strip().split()
        for row in stim.itertuples(index=False)
    }


class UCLReadingDataset:
    """Load the UCL broad-coverage self-paced reading corpus.

    Parameters
    ----------
    root : str or Path
        Dataset root containing ``stimuli.txt`` and ``selfpacedreading.*.txt``.
    sentences : list of int, optional
        Subset of sentence numbers (1-based) to load. Loads all 361 by default.
    min_accuracy : float
        Minimum per-subject comprehension-question accuracy (from
        ``selfpacedreading.subj.txt``) required to keep that subject's trials.
        Default 0.0 keeps all 117 subjects.
    min_rt, max_rt : float
        Plausible single-trial RT bounds in ms (implausible RTs excluded before
        aggregation), matching standard SPR trimming practice.
    aggregate : bool
        If True (default), one row per word with mean/SD RT across subjects.
        If False, one row per (subject, word) trial.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        sentences: list[int] | None = None,
        min_accuracy: float = 0.0,
        min_rt: float = 100.0,
        max_rt: float = 3000.0,
        aggregate: bool = True,
    ) -> None:
        self.root = Path(root)
        self.sentences = sentences
        self.min_accuracy = min_accuracy
        self.min_rt = min_rt
        self.max_rt = max_rt
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
        rt_path = self.root / "selfpacedreading.RT.txt"
        if not rt_path.is_file():
            raise FileNotFoundError(f"UCL Reading RT file not found: {rt_path}")

        df = pd.read_csv(rt_path, sep="\t")
        df = df.rename(
            columns={
                "subj_nr": "participant_id",
                "sent_nr": "story_id",
                "word_pos": "zone",
                "RT": "rt",
            }
        )
        # Word identity comes from the authoritative stimuli.txt tokenization
        # (via ``_word_list``); drop the raw copy to avoid a word_x/word_y merge.
        df = df.drop(columns=["word"])

        if self.min_accuracy > 0:
            subj_path = self.root / "selfpacedreading.subj.txt"
            subj = pd.read_csv(subj_path, sep="\t")
            keep_subj = set(subj.loc[subj["correct"] >= self.min_accuracy, "subj_nr"])
            n_before = df["participant_id"].nunique()
            df = df[df["participant_id"].isin(keep_subj)]
            logger.info(
                "UCLReading: kept %d/%d subjects (accuracy >= %.2f)",
                len(keep_subj),
                n_before,
                self.min_accuracy,
            )

        if self.sentences is not None:
            df = df[df["story_id"].isin(self.sentences)]

        n_before = len(df)
        df = df[(df["rt"] >= self.min_rt) & (df["rt"] <= self.max_rt)]
        logger.info(
            "UCLReading: dropped %d/%d rows with RT outside [%.0f, %.0f] ms",
            n_before - len(df),
            n_before,
            self.min_rt,
            self.max_rt,
        )
        return df.reset_index(drop=True)

    def _word_list(self) -> pd.DataFrame:
        """Per-(story_id, zone) word identity, edge flag, length, and context.

        ``context`` is the cumulative sentence text through and including the
        target word (NeuralSet ``AddConcatenationContext`` convention), so it
        can be passed directly to ``batch_surprisal_one_model``.
        """
        sent_map = sentence_word_lists(self.root)
        rows: list[dict] = []
        for story_id, tokens in sent_map.items():
            if self.sentences is not None and story_id not in self.sentences:
                continue
            n = len(tokens)
            for i, word in enumerate(tokens):
                rows.append(
                    {
                        "story_id": story_id,
                        "zone": i + 1,
                        "word": word,
                        "is_edge": (i == 0) or (i == n - 1),
                        "wlen": len(word),
                        "context": " ".join(tokens[: i + 1]),
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
            "UCLReading: aggregated %d words across %d sentences (mean RT)",
            len(merged),
            merged["story_id"].nunique(),
        )
        return merged

    def _raw(self, trials: pd.DataFrame, word_df: pd.DataFrame) -> pd.DataFrame:
        merged = trials.merge(word_df, on=["story_id", "zone"], how="inner")
        logger.info(
            "UCLReading: loaded %d trials (%d subjects, %d sentences)",
            len(merged),
            merged["participant_id"].nunique(),
            merged["story_id"].nunique(),
        )
        return merged
