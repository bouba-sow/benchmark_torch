#!/usr/bin/env python3
"""Save frequency and word-length RDMs aligned to available neural RSA items.

For the eight neural datasets with cached subject response RDMs, one archive
per subject and feature retains the exact stimulus order and sampled pair
indices. LPP and Natural Stories fMRI do not yet have neural RSA caches; for
those datasets this writes a clearly marked, dataset-level RDM over the
time-chunk items in their saved LME covariate tables.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from neuroai.features.word_frequency import compute_wlen_wordfreq_covariates


DATASETS = ("pereira", "tuckute", "fedorenko2016", "broderick", "megmasc",
            "nieuwland", "szewczyk", "podcast", "lpp", "ns_fmri")
SHARED_ROOT = "pearson_rsa/surprisal/rdm_cache/shared"


def first_table(dataset: str, pattern: str) -> Path:
    paths = sorted((ROOT / "outputs_jeanzay" / dataset / "surprisal_sweep").rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {pattern} found for {dataset}")
    return paths[0]


def _clean_map(frame: pd.DataFrame, id_col: str, length_col: str, freq_col: str):
    sub = frame[[id_col, length_col, freq_col]].dropna().copy()
    # Repeated rows are subject/ROI/electrode replicates. Covariates must agree.
    spread = sub.groupby(id_col)[[length_col, freq_col]].agg(lambda x: float(x.max() - x.min()))
    if (spread.to_numpy() > 1e-5).any():
        raise ValueError(f"Lexical features vary across duplicate stimulus rows: {id_col}")
    sub = sub.groupby(id_col, as_index=False)[[length_col, freq_col]].mean()
    return {str(row[id_col]): (float(row[length_col]), float(row[freq_col]))
            for _, row in sub.iterrows()}


def lexical_maps(dataset: str) -> dict[str, dict[str, tuple[float, float]]]:
    """Return subject-specific or common stimulus ID -> (length, frequency)."""
    if dataset in {"pereira", "tuckute"}:
        p = first_table(dataset, f"{dataset}_lme_pooled_table*.parquet")
        frame = pd.read_parquet(p, columns=["sentence_id", "mean_wlen", "mean_freq"])
        return {"*": _clean_map(frame, "sentence_id", "mean_wlen", "mean_freq")}
    if dataset == "fedorenko2016":
        frame = pd.read_csv(ROOT / "data/fedorenko2016/stimuli.csv")
        lexical = compute_wlen_wordfreq_covariates(frame.word.astype(str).tolist())
        return {"*": {str(item): (float(v[0]), float(v[1]))
                       for item, v in zip(frame.stimulus_id, lexical)}}
    if dataset == "megmasc":
        p = first_table(dataset, "megmasc_lme_pooled_table*.parquet")
        frame = pd.read_parquet(p, columns=["subject", "chunk", "word_idx", "wlen", "freq"])
        frame["stimulus_id"] = frame.chunk.astype(str) + "::" + frame.word_idx.astype(str)
        return {str(s): _clean_map(part, "stimulus_id", "wlen", "freq")
                for s, part in frame.groupby("subject", sort=True)}
    if dataset == "podcast":
        p = first_table(dataset, "podcast_lme_pooled_table*.parquet")
        frame = pd.read_parquet(p, columns=["subject", "chunk", "word_idx", "wlen", "freq"])
        frame["stimulus_id"] = frame.chunk.astype(str) + "::" + frame.word_idx.astype(str)
        return {str(s): _clean_map(part, "stimulus_id", "wlen", "freq")
                for s, part in frame.groupby("subject", sort=True)}
    if dataset == "broderick":
        first = first_table(dataset, "broderick_lme_table_sub-*.parquet")
        result = {}
        for p in sorted(first.parent.glob("broderick_lme_table_sub-*.parquet")):
            frame = pd.read_parquet(p, columns=["chunk", "word_idx", "wlen", "freq"])
            frame["stimulus_id"] = frame.chunk.astype(str) + "::" + frame.word_idx.astype(str)
            subject = "sub-" + p.stem.rsplit("sub-", 1)[1]
            result[subject] = _clean_map(frame, "stimulus_id", "wlen", "freq")
        return result
    if dataset in {"nieuwland", "szewczyk"}:
        source_root = ROOT / "data/n400_paper_formula/osfstorage"
        if dataset == "nieuwland":
            source = pd.read_csv(source_root / "datasets/nieuwland_2018.tsv", sep="\t")
            # One item number has expected/unexpected critical-word variants.
            # The RSA table indexes unique (item_number, condition) pairs with
            # expected before unexpected; higher cloze identifies expected.
            words = (source.groupby(["item", "TargetWords"], as_index=False).cloze.mean()
                     .sort_values(["item", "cloze", "TargetWords"],
                                  ascending=[True, False, True]).reset_index(drop=True))
            words["stimulus_id"] = np.arange(len(words)).astype(str)
        else:
            source = pd.read_csv(source_root / "datasets/szewczyk_2022.tsv", sep="\t",
                                 dtype={"Item": str, "dataset": str, "TargetWords": str,
                                        "Subject": str})
            source["stimulus_id"] = source.dataset.astype(str) + "_" + source.Item.astype(str)
            result = {}
            for subject, part in source.groupby("Subject", sort=True):
                words = part[["stimulus_id", "TargetWords"]].drop_duplicates()
                if words.stimulus_id.duplicated().any():
                    raise ValueError(f"Szewczyk {subject}: one item maps to multiple critical words")
                values = compute_wlen_wordfreq_covariates(words.TargetWords.astype(str).tolist())
                result[str(subject)] = {str(item): (float(v[0]), float(v[1]))
                                        for item, v in zip(words.stimulus_id, values)}
            return result
        values = compute_wlen_wordfreq_covariates(words.TargetWords.astype(str).tolist())
        return {"*": {str(item): (float(v[0]), float(v[1]))
                       for item, v in zip(words.stimulus_id, values)}}
    raise KeyError(dataset)


def _feature_for_id(item, mapping):
    key = str(item)
    if key in mapping:
        return mapping[key]
    # Numeric IDs are sometimes saved as ints by one cache and numeric strings
    # by another; canonicalize without altering nonnumeric sentence/item IDs.
    try:
        numeric = str(int(key))
    except ValueError:
        numeric = key
    if numeric in mapping:
        return mapping[numeric]
    raise KeyError(f"No lexical covariates for stimulus ID {key}")


def save_cached_dataset(dataset: str) -> int:
    shared_root = ROOT / "outputs" / dataset / SHARED_ROOT
    shared_paths = sorted(shared_root.glob("*.npz"))
    if not shared_paths:
        raise FileNotFoundError(f"No shared response RDMs for {dataset}: {shared_root}")
    maps = lexical_maps(dataset)
    n_saved = 0
    for rdm_path in shared_paths:
        with np.load(rdm_path, allow_pickle=True) as archive:
            ids = np.asarray(archive["stimulus_ids"])
            pairs = (np.asarray(archive["pair_indices"], dtype=np.int32)
                     if "pair_indices" in archive else None)
            subject = str(archive["subject"].item()) if "subject" in archive else rdm_path.stem
        # Some upstream N400 caches store string IDs as object arrays. Normalize
        # them to fixed-width Unicode so the new archives need no pickle loading.
        saved_ids = ids.astype(str) if ids.dtype.kind == "O" else ids
        mapping = maps.get(subject, maps.get("*"))
        if mapping is None:
            raise KeyError(f"No covariate map for {dataset}/{subject}")
        lexical = np.asarray([_feature_for_id(item, mapping) for item in ids], dtype=np.float32)
        outdir = ROOT / "outputs" / dataset / "pearson_rsa" / "confounds" / subject
        outdir.mkdir(parents=True, exist_ok=True)
        for col, name, definition in ((0, "word_length", "character_count"),
                                      (1, "frequency", "zipf_frequency")):
            values = lexical[:, col]
            distances = (np.abs(values[pairs[:, 0]] - values[pairs[:, 1]]) if pairs is not None
                         else pdist_scalar(values))
            payload = {"stimulus_ids": saved_ids, "feature_values": values,
                       "distances": np.asarray(distances, dtype=np.float32),
                       "dataset": np.asarray(dataset), "subject": np.asarray(subject),
                       "rdm_type": np.asarray("absolute_difference"),
                       "feature": np.asarray(definition)}
            if pairs is not None:
                payload["pair_indices"] = pairs
            np.savez_compressed(outdir / f"{name}_rdm.npz", **payload)
        n_saved += 1
    return n_saved


def pdist_scalar(values: np.ndarray) -> np.ndarray:
    # Keep this module's runtime imports light; full RDMs only occur for small
    # subject stimulus sets (Pereira, Tuckute, Fedorenko and N400 datasets).
    from scipy.spatial.distance import pdist
    return pdist(np.asarray(values, dtype=np.float64)[:, None], metric="euclidean")


def save_chunk_dataset(dataset: str) -> int:
    """Save confounds over existing LME time-chunk stimulus units for LPP/NS."""
    if dataset == "lpp":
        p = first_table(dataset, "lpp_lme_pooled_table*.parquet")
        table = pd.read_parquet(p, columns=["section", "chunk", "wlen", "freq"])
        table["stimulus_id"] = table.section.astype(str) + "::" + table.chunk.astype(str)
    else:
        p = first_table(dataset, "ns_fmri_lme_nuisance_table*.parquet")
        table = pd.read_parquet(p, columns=["story", "chunk", "wlen", "freq"])
        table["stimulus_id"] = table.story.astype(str) + "::" + table.chunk.astype(str)
    items = table.groupby("stimulus_id", as_index=False)[["wlen", "freq"]].mean().sort_values("stimulus_id")
    ids = items.stimulus_id.astype(str).to_numpy(dtype=str)
    outdir = ROOT / "outputs" / dataset / "pearson_rsa" / "confounds" / "time_chunks"
    outdir.mkdir(parents=True, exist_ok=True)
    for name, col, definition in (("word_length", "wlen", "hrf_convolved_length"),
                                  ("frequency", "freq", "hrf_convolved_frequency")):
        values = items[col].to_numpy(dtype=np.float32)
        np.savez_compressed(outdir / f"{name}_rdm.npz", stimulus_ids=ids,
                            feature_values=values,
                            distances=pdist_scalar(values).astype(np.float32),
                            dataset=np.asarray(dataset), subject=np.asarray("all"),
                            stimulus_level=np.asarray("time_chunk"),
                            rdm_type=np.asarray("absolute_difference"),
                            feature=np.asarray(definition),
                            response_rdm_available=np.asarray(False))
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    for dataset in args.datasets:
        if dataset in {"lpp", "ns_fmri"}:
            n = save_chunk_dataset(dataset)
            print(f"{dataset}: saved two time-chunk RDMs ({n} chunks; neural RSA response cache unavailable)", flush=True)
        else:
            n = save_cached_dataset(dataset)
            print(f"{dataset}: saved two confound RDMs for {n} subjects with exact neural RSA item/pair alignment", flush=True)


if __name__ == "__main__":
    main()
