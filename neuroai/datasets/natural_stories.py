"""Natural Stories self-paced reading time dataset adapter.

Reference
---------
Futrell et al. (2021). Natural stories corpus revisited.
OpenNeuro: https://openneuro.org/datasets/ds000214

Dataset layout expected at ``root/``
-------------------------------------
words.tsv                         — word list (item, zone, word columns)
naturalstories_RTS/
  processed_RTs.tsv               — raw per-participant reading times
  processed_wordinfo.tsv          — per-word aggregate stats (meanItemRT, ...)

Output schema of :meth:`NaturalStoriesDataset.load_words`
----------------------------------------------------------
word        str     Surface form.
story_id    int     Story index (1-based ``item`` in the original files).
zone        int     Word position within the story (1-based).
rt_mean     float   Mean reading time across participants (ms).
rt_sd       float   Standard deviation of reading times across participants.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_SENT_SPLIT_RE = (
    r"(?<!Mr.|Dr.)(?<!Mrs.)(?<=.[.!?]|[.!?][\'\"]) +"
    r"(?=[^a-z])(?!I asked)(?!I cried)(?!Abby excl)(?!Abby asked)"
)


def load_story_texts(
    root: str | Path,
    *,
    stories: list[int] | None = None,
) -> list[str]:
    """Return one space-joined string per Natural Stories story."""
    ds = NaturalStoriesDataset(root, stories=stories, aggregate=False)
    words = ds._load_word_list()
    if words.empty:
        raise RuntimeError(f"No Natural Stories text found under {root}")

    texts: list[str] = []
    for _story_id, group in words.groupby("story_id", sort=True):
        texts.append(" ".join(group["word"].astype(str).tolist()))
    logger.info("Loaded %d Natural Stories from %s", len(texts), root)
    return texts


class NaturalStoriesDataset:
    """Load the Natural Stories SPR dataset as a word-level DataFrame.

    Parameters
    ----------
    root : str or Path
        Path to the dataset root (directory that contains ``words.tsv`` and
        ``naturalstories_RTS/``).
    stories : list of int, optional
        Subset of story IDs (1-based) to load.  Loads all 10 stories by default.
    min_correct : int, optional
        Minimum number of comprehension questions answered correctly for a
        participant's data to be included (default: 0 = keep all).
    aggregate : bool
        If True (default), return one row per word with mean / std RT.
        If False, return one row per (participant, word) observation.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        stories: list[int] | None = None,
        min_correct: int = 0,
        aggregate: bool = True,
    ) -> None:
        self.root = Path(root)
        self.stories = stories
        self.min_correct = min_correct
        self.aggregate = aggregate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_words(self) -> pd.DataFrame:
        """Return word-level reading-time DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: word, story_id, zone, rt_mean, rt_sd (aggregate mode) or
            word, story_id, zone, rt, participant_id (raw mode).
        """
        rts = self._load_rts()
        words = self._load_word_list()

        sent_map = self._load_sentence_map()

        if self.aggregate:
            return self._aggregate(rts, words, sent_map)
        return self._raw(rts, words, sent_map)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_rts(self) -> pd.DataFrame:
        rt_path = self.root / "naturalstories_RTS" / "processed_RTs.tsv"
        if not rt_path.is_file():
            raise FileNotFoundError(f"Natural Stories RT file not found: {rt_path}")

        df = pd.read_csv(rt_path, sep="\t")
        df.columns = [c.strip() for c in df.columns]

        # Map original column names to standard names
        rename = {
            "WorkerId": "participant_id",
            "correct": "n_correct",
            "item": "story_id",
            "zone": "zone",
            "RT": "rt",
            "word": "word",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        if self.min_correct > 0 and "n_correct" in df.columns:
            before = len(df)
            df = df[df["n_correct"] >= self.min_correct]
            logger.info(
                "NaturalStories: dropped %d rows (n_correct < %d)",
                before - len(df),
                self.min_correct,
            )

        if self.stories is not None:
            df = df[df["story_id"].isin(self.stories)]

        # Exclude implausible RTs (< 100 ms or > 3000 ms) — standard practice
        if "rt" in df.columns:
            before = len(df)
            df = df[(df["rt"] >= 100) & (df["rt"] <= 3000)]
            logger.info(
                "NaturalStories: dropped %d rows with RT outside [100, 3000] ms",
                before - len(df),
            )

        return df

    def _load_word_list(self) -> pd.DataFrame:
        """Load words.tsv — three unnamed columns: whole_key, word_key, word."""
        word_path = self.root / "words.tsv"
        if not word_path.is_file():
            raise FileNotFoundError(f"Natural Stories word list not found: {word_path}")

        rows = []
        with word_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                key = parts[0]  # e.g. "1.1.whole" or "1.2.1"
                word = parts[-1]
                segs = key.split(".")
                if len(segs) < 2:
                    continue
                try:
                    story_id = int(segs[0])
                    zone = int(segs[1])
                except ValueError:
                    continue
                rows.append({"story_id": story_id, "zone": zone, "word": word})

        df = pd.DataFrame(rows).drop_duplicates(subset=["story_id", "zone"])
        df = df.sort_values(["story_id", "zone"]).reset_index(drop=True)

        if self.stories is not None:
            df = df[df["story_id"].isin(self.stories)]

        return df

    def _load_sentence_map(self) -> pd.DataFrame:
        """Map (story_id, zone) -> (sentid, sentpos) using ModelBlocks-like rules."""
        tok_path = self.root / "naturalstories_RTS" / "all_stories.tok"
        if not tok_path.is_file():
            logger.warning("NaturalStories: sentence map missing (%s)", tok_path)
            return pd.DataFrame(columns=["story_id", "zone", "sentid", "sentpos"])

        tok = pd.read_csv(tok_path, sep="\t")
        tok = tok.rename(columns={"item": "story_id"})
        required = {"story_id", "zone", "word"}
        if not required.issubset(tok.columns):
            logger.warning(
                "NaturalStories: sentence map missing required columns; got %s",
                sorted(tok.columns),
            )
            return pd.DataFrame(columns=["story_id", "zone", "sentid", "sentpos"])

        tok["story_id"] = tok["story_id"].astype(int)
        tok["zone"] = tok["zone"].astype(int)
        tok["word"] = tok["word"].astype(str)
        if self.stories is not None:
            tok = tok[tok["story_id"].isin(self.stories)]

        rows: list[dict[str, int]] = []
        global_sentid = 0
        for story_id, group in tok.sort_values(["story_id", "zone"]).groupby(
            "story_id", sort=True
        ):
            zones = group["zone"].to_list()
            words = group["word"].to_list()
            text = " ".join(words)
            sentences = [s.strip() for s in re.split(_SENT_SPLIT_RE, text) if s.strip()]

            i = 0
            for sent in sentences:
                toks = sent.split()
                for sentpos, _ in enumerate(toks, start=1):
                    if i >= len(zones):
                        break
                    rows.append(
                        {
                            "story_id": int(story_id),
                            "zone": int(zones[i]),
                            "sentid": int(global_sentid),
                            "sentpos": int(sentpos),
                        }
                    )
                    i += 1
                global_sentid += 1

            if i < len(zones):
                logger.warning(
                    "NaturalStories: sentence segmentation under-covered story %s (%d/%d tokens); assigning remainder",
                    story_id,
                    i,
                    len(zones),
                )
                for j in range(i, len(zones)):
                    rows.append(
                        {
                            "story_id": int(story_id),
                            "zone": int(zones[j]),
                            "sentid": int(global_sentid),
                            "sentpos": int(j - i + 1),
                        }
                    )
                global_sentid += 1

        return pd.DataFrame(rows)

    def _aggregate(
        self, rts: pd.DataFrame, words: pd.DataFrame, sent_map: pd.DataFrame
    ) -> pd.DataFrame:
        """Return one row per word with mean and SD of RT across participants."""
        agg = (
            rts.groupby(["story_id", "zone"])["rt"]
            .agg(rt_mean="mean", rt_sd="std", rt_n="count")
            .reset_index()
        )
        merged = words.merge(agg, on=["story_id", "zone"], how="left")
        merged["rt_sd"] = merged["rt_sd"].fillna(0.0)
        if not sent_map.empty:
            merged = merged.merge(sent_map, on=["story_id", "zone"], how="left")

        # Build a cumulative context column (within-story, reset per story_id)
        merged = merged.sort_values(["story_id", "zone"]).reset_index(drop=True)
        contexts: list[str] = []
        for _, group in merged.groupby("story_id", sort=False):
            words = group["word"].tolist()
            contexts.extend(" ".join(words[: i + 1]) for i in range(len(words)))
        merged["context"] = contexts

        logger.info(
            "NaturalStories: loaded %d words across %d stories (aggregate RT)",
            len(merged),
            merged["story_id"].nunique(),
        )
        return merged

    def _raw(
        self, rts: pd.DataFrame, words: pd.DataFrame, sent_map: pd.DataFrame
    ) -> pd.DataFrame:
        """Return one row per (participant, word) observation."""
        merged = rts.merge(words, on=["story_id", "zone"], how="left")
        if not sent_map.empty:
            merged = merged.merge(sent_map, on=["story_id", "zone"], how="left")
        logger.info(
            "NaturalStories: loaded %d observations (%d participants, %d stories)",
            len(merged),
            merged["participant_id"].nunique() if "participant_id" in merged.columns else 0,
            merged["story_id"].nunique(),
        )
        return merged
