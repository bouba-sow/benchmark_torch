"""Single-study views of the Szewczyk & Federmeier (2022, JML) paper table.

``data/n400_paper_formula/osfstorage/datasets/szewczyk_2022.tsv`` stacks FIVE separate
experiments (column ``dataset``; different subjects, items and lab sessions). The benchmark
treats each study as its own dataset instead of pooling them, so every function here works on
ONE study. Used by the item-mean embedding track, its correlation noise ceiling, and (through
``sub_experiment`` in ``neuroai/benchmarks/n400_szewczyk_lme.py``) the surprisal/LME track.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TSV = "data/n400_paper_formula/osfstorage/datasets/szewczyk_2022.tsv"

# slug -> value of the tsv ``dataset`` column
STUDIES: dict[str, str] = {
    "f2007": "Federmeier et al. (2007)",          # paper's dataset 1 (primary)
    "h2019": "Hubbard et al. (2019)",             # paper's dataset 4
}


def load_study(tsv_path: str | Path, study: str) -> pd.DataFrame:
    """Trials of one study (``study`` = slug in ``STUDIES`` or the literal ``dataset`` value).

    Adds ``chunk`` (``dataset_Item``, the train/test and random-effect unit), ``item`` (the
    (FullText, TargetWords) key) and ``n400_resid``: the N400 with the pre-stimulus baseline
    regressed out (slope from a within-subject regression on this study), mirroring the paper
    formula's ``scale(bline)`` covariate. Lexical covariates are NOT removed -- they are real
    effects an embedding may legitimately predict.
    """
    name = STUDIES.get(study, study)
    df = pd.read_csv(tsv_path, sep="\t")
    if name not in set(df["dataset"]):
        raise ValueError(f"study {name!r} not in {sorted(df['dataset'].unique())}")
    df = df[df["dataset"] == name].dropna(subset=["n400", "bline"]).reset_index(drop=True)
    df["chunk"] = df["dataset"].astype(str) + "_" + df["Item"].astype(str)
    df["item"] = df["FullText"].astype(str) + "||" + df["TargetWords"].astype(str)
    bd = df["bline"] - df.groupby("Subject")["bline"].transform("mean")
    yd = df["n400"] - df.groupby("Subject")["n400"].transform("mean")
    beta = float((bd * yd).sum() / (bd * bd).sum())
    df["n400_resid"] = df["n400"] - beta * df["bline"]
    df.attrs["bline_beta"] = beta
    return df


def study_items(df: pd.DataFrame) -> pd.DataFrame:
    """Unique items in first-appearance order: ``context`` (text before the critical word),
    ``critical_word``, ``chunk`` (frame/group id), ``item`` key."""
    it = df[["FullText", "TargetWords", "chunk", "item"]].drop_duplicates("item").reset_index(drop=True)
    tw = it["TargetWords"].astype(str).str.strip()
    ft = it["FullText"].astype(str).str.rstrip()
    ok = np.array([f.lower().endswith(t.lower()) for f, t in zip(ft, tw)])
    if not ok.all():
        raise ValueError(f"{(~ok).sum()} items whose FullText does not end with the critical word")
    it["critical_word"] = tw
    it["context"] = [f[: len(f) - len(t)].rstrip() for f, t in zip(ft, tw)]
    return it


def item_means(df: pd.DataFrame, items: pd.DataFrame, col: str = "n400_resid") -> pd.DataFrame:
    """``items`` plus ``y`` (mean of ``col`` over the subjects that saw the item) and ``n_subjects``."""
    g = df.groupby("item")[col].agg(y="mean", n_subjects="size")
    out = items.join(g, on="item")
    if out["y"].isna().any():
        raise ValueError("items without any trial")
    return out
