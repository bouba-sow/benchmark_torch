#!/usr/bin/env python3
"""Save word-frequency and word-length RDMs aligned to behavioral RSA caches.

Distances are absolute differences between scalar lexical features. Each
archive stores the exact stimulus order and (when applicable) sampled pair
indices used by the existing RSA analysis, so it can be added to later nested
RDM regressions without redoing stimulus alignment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from neuroai.features.word_frequency import compute_wlen_wordfreq_covariates
from scripts.analysis.behavioral_rsa import _prepare as prepare_provo_meco
from scripts.analysis.surprisal_rsa import _pair_distances


BEHAVIOR = {
    "provo": ("FP", "GP", "SP"), "meco": ("FP", "GP", "SP"),
    "geco": ("FP", "GP"), "onestop": ("FP", "GP", "SP"),
    "naturalstories": ("SPR",), "ucl_reading": ("SPR",),
    "zuco_reading": ("NR", "SR", "TSR"),
}
KEYS = ["story_id", "zone"]
_PREPARED_PROVO_MECO: dict[str, dict[str, pd.DataFrame]] = {}
_WORD_TABLES: dict[tuple[str, str], pd.DataFrame] = {}


def word_table(dataset: str, measure: str) -> pd.DataFrame:
    """Load canonical surface words used by the existing behavioral track."""
    key = (dataset, measure)
    if key in _WORD_TABLES:
        return _WORD_TABLES[key]
    if dataset == "provo":
        from neuroai.datasets.provo import ProvoDataset
        result = ProvoDataset(ROOT / "data/provo/osfstorage", measure, aggregate=True)._word_list()
    elif dataset == "meco":
        from neuroai.datasets.meco import meco_word_list
        root = ROOT / "data/meco/osfstorage/release 2.0/version 2.1/wave 1/primary data/eye tracking data"
        result = meco_word_list(root, language="en", exclude_doc_edges=True)
    elif dataset == "onestop":
        result = onestop_lexical_table()
    else:
        roots = {
            "geco": ROOT / "data/geco", "naturalstories": ROOT / "data/naturalstories",
            "ucl_reading": ROOT / "data/ucl_reading", "zuco_reading": ROOT / "data/zuco_reading/osfstorage",
        }
        if dataset == "geco":
            from neuroai.datasets.geco import geco_word_list
            result = geco_word_list(roots[dataset], exclude_doc_edges=True)
        elif dataset == "naturalstories":
            from neuroai.datasets.natural_stories import NaturalStoriesDataset
            result = NaturalStoriesDataset(roots[dataset], aggregate=True).load_words()
        elif dataset == "ucl_reading":
            from neuroai.datasets.ucl_reading import UCLReadingDataset
            result = UCLReadingDataset(roots[dataset], aggregate=True).load_words()
        else:
            from neuroai.datasets.zuco_reading import _word_list
            result = _word_list(roots[dataset], measure)
    _WORD_TABLES[key] = result
    return result


@lru_cache(maxsize=1)
def onestop_lexical_table() -> pd.DataFrame:
    """Read the 5.3 GB OneStop source in chunks and retain only modal words."""
    source = ROOT / "data/onestop/precomputed_reading_measures/ia_Paragraph.csv"
    cols = ["article_batch", "article_id", "paragraph_id", "difficulty_level",
            "IA_ID", "IA_LABEL", "practice_trial", "repeated_reading_trial"]
    partial = []
    for chunk in pd.read_csv(source, usecols=cols, chunksize=200_000, low_memory=False):
        chunk = chunk.loc[~chunk.practice_trial & ~chunk.repeated_reading_trial].copy()
        if chunk.empty:
            continue
        diff = (chunk.difficulty_level == "Ele").astype(np.int32)
        chunk["story_id"] = (chunk.article_batch.astype(np.int32) * 10_000
                             + chunk.article_id.astype(np.int32) * 100
                             + chunk.paragraph_id.astype(np.int32) * 10 + diff)
        part = chunk.groupby(["story_id", "IA_ID", "IA_LABEL"], sort=False).size().rename("n").reset_index()
        partial.append(part)
    counts = pd.concat(partial, ignore_index=True)
    counts = counts.groupby(["story_id", "IA_ID", "IA_LABEL"], sort=False, as_index=False).n.sum()
    modal = (counts.sort_values("n", ascending=False)
             .drop_duplicates(["story_id", "IA_ID"])
             .rename(columns={"IA_ID": "zone", "IA_LABEL": "word"}))
    return modal[["story_id", "zone", "word"]].sort_values(KEYS).reset_index(drop=True)


def item_ids(dataset: str, measure: str) -> pd.DataFrame:
    shared = (ROOT / "outputs" / dataset / "pearson_rsa" / "surprisal" /
              "rdm_cache" / measure.lower() / "shared" / "behavior_rdm.npz")
    if shared.is_file():
        with np.load(shared) as archive:
            ids = np.asarray(archive["stimulus_ids"])
        if ids.ndim != 2 or ids.shape[1] != 2:
            raise ValueError(f"Expected (story_id, zone) IDs in shared behavioral RDM: {shared}")
        return pd.DataFrame(ids, columns=KEYS).astype({"story_id": "int64", "zone": "int64"})
    if dataset in {"provo", "meco"}:
        if dataset not in _PREPARED_PROVO_MECO:
            _, _PREPARED_PROVO_MECO[dataset] = prepare_provo_meco(dataset)
        return _PREPARED_PROVO_MECO[dataset][measure][KEYS].copy()
    if dataset in {"geco", "onestop", "naturalstories", "ucl_reading", "zuco_reading"}:
        path = ROOT / "outputs" / dataset / "pearson_rsa" / "embedding" / "response_table" / f"{measure.lower()}.parquet"
        if path.exists():
            return pd.read_parquet(path, columns=KEYS)
        # The response tables are produced by the existing benchmark setup.
        # If absent, derive the same valid word set from its dataset adapter.
        table = word_table(dataset, measure)
        if "is_edge" in table:
            table = table.loc[~table.is_edge.fillna(False).astype(bool)]
        return table[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    raise KeyError(dataset)


def pair_indices(dataset: str, measure: str, ids: np.ndarray) -> np.ndarray | None:
    shared = (ROOT / "outputs" / dataset / "pearson_rsa" / "surprisal" /
              "rdm_cache" / measure.lower() / "shared" / "behavior_rdm.npz")
    if shared.is_file():
        with np.load(shared) as archive:
            if not np.array_equal(ids, archive["stimulus_ids"]):
                raise ValueError(f"Stimulus order changed while reading shared RDM: {shared}")
            return (np.asarray(archive["pair_indices"], dtype=np.int32)
                    if "pair_indices" in archive else None)
    _x, _y, pairs = _pair_distances(np.zeros(len(ids)), np.zeros(len(ids)), dataset, measure)
    return pairs


def save_one(dataset: str, measure: str) -> tuple[int, int]:
    ids = item_ids(dataset, measure)
    words = word_table(dataset, measure)
    if words.duplicated(KEYS).any():
        raise ValueError(f"Duplicate lexical item keys for {dataset}/{measure}")
    aligned = ids.merge(words[KEYS + ["word"]], on=KEYS, how="left", validate="one_to_one")
    if aligned.word.isna().any():
        missing = aligned.loc[aligned.word.isna(), KEYS].head().to_dict("records")
        raise ValueError(f"Cannot align {dataset}/{measure} words; examples: {missing}")
    lexical = compute_wlen_wordfreq_covariates(aligned.word.astype(str).tolist())
    pair = pair_indices(dataset, measure, ids.to_numpy())
    names = {"word_length": lexical[:, 0], "frequency": lexical[:, 1]}
    output = ROOT / "outputs" / dataset / "pearson_rsa" / "confounds" / measure.lower()
    output.mkdir(parents=True, exist_ok=True)
    for name, values in names.items():
        distances = (np.abs(values[pair[:, 0]] - values[pair[:, 1]]) if pair is not None
                     else pdist(values[:, None], metric="euclidean"))
        np.savez_compressed(
            output / f"{name}_rdm.npz", stimulus_ids=ids.to_numpy(dtype=np.int32),
            feature_values=np.asarray(values, dtype=np.float32),
            distances=np.asarray(distances, dtype=np.float32),
            **({"pair_indices": pair} if pair is not None else {}),
            dataset=np.asarray(dataset), measure=np.asarray(measure),
            rdm_type=np.asarray("absolute_difference"),
            feature=np.asarray("wordfreq_zipf" if name == "frequency" else "character_count"),
        )
    return len(ids), len(names["frequency"]) * (len(names["frequency"]) - 1) // 2 if pair is None else len(pair)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(BEHAVIOR), default=sorted(BEHAVIOR))
    args = parser.parse_args()
    for dataset in args.datasets:
        for measure in BEHAVIOR[dataset]:
            n_items, n_pairs = save_one(dataset, measure)
            print(f"{dataset}/{measure}: saved frequency + word length RDMs ({n_items} items, {n_pairs} pairs)", flush=True)


if __name__ == "__main__":
    main()
