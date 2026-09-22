"""Shared loader for outputs/dlt_difficulty/*_dlt_difficulty_scores_multi_threshold.csv
(DLT/Gibson-2000 integration-cost easy-vs-hard word split, precomputed by a
separate pipeline). Used by scripts/dlt/hard_bars_v2.py and by
summarize/report.py's hard-case Pareto section.

Common schema per file: model, family, dlt_bin ("easy"/"hard"), score,
n_params, hard_gt_threshold (0,1,2,3,4,5,7). We use hard_gt_threshold==2.0
throughout (the "DLT cost >= 2" hard-word split).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuroai.analysis.benchmark_common import REPO_ROOT, ceiling_for, dll_ceiling_for, lookup_model_meta

DLT_DIR = REPO_ROOT / "outputs" / "dlt_difficulty"

# file stem -> (ceiling slug, track, feature_mode)
DLT_FILES: dict[str, tuple[str, str, str]] = {
    # behavioral, surprisal/LME track
    "geco":                  ("geco", "behavioral", "surprisal"),
    "provo":                 ("provo", "behavioral", "surprisal"),
    "ucl_reading":            ("ucl_reading", "behavioral", "surprisal"),
    "meco":                  ("meco", "behavioral", "surprisal"),
    "zuco_reading_nr":        ("zuco_reading", "behavioral", "surprisal"),
    # behavioral, embedding track (GECO correctly excluded -- no such file)
    "provo_embed":            ("provo", "behavioral", "embedding"),
    "ucl_reading_embed":       ("ucl_reading", "behavioral", "embedding"),
    "meco_embed":             ("meco", "behavioral", "embedding"),
    "zuco_reading_nr_embed":   ("zuco_reading", "behavioral", "embedding"),
    "naturalstories_embed":    ("naturalstories_spr", "behavioral", "embedding"),
    # neural
    "podcast":                ("podcast_ieeg", "neural", "embedding"),
    "podcast_lme":             ("podcast_ieeg", "neural", "surprisal"),
    "fedorenko2016":           ("fedorenko2016_ecog", "neural", "embedding"),
}

# Named dataset-file groupings used by both the bar plots and the hard-case Pareto.
DLT_GROUPS: dict[str, list[str]] = {
    "behavioral_surprisal": ["geco", "provo", "ucl_reading", "meco", "zuco_reading_nr"],
    "behavioral_embedding": ["provo_embed", "ucl_reading_embed", "meco_embed",
                              "zuco_reading_nr_embed", "naturalstories_embed"],
        "neural": ["podcast", "podcast_lme", "fedorenko2016"],
}
# All behavioral files (surprisal + embedding) combined, for the hard-case Pareto's x-axis.
DLT_ALL_BEHAVIORAL: list[str] = DLT_GROUPS["behavioral_surprisal"] + DLT_GROUPS["behavioral_embedding"]


def _load_one(file_stem: str, hard_gt: float = 2.0) -> pd.DataFrame | None:
    path = DLT_DIR / f"{file_stem}_dlt_difficulty_scores_multi_threshold.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df = df[np.isclose(df["hard_gt_threshold"].astype(float), hard_gt)]
    if df.empty:
        return None
    ceiling_slug, _track, _mode = DLT_FILES[file_stem]
    ceiling = ceiling_for(ceiling_slug)
    has_ppl = "perplexity" in df.columns
    rows = []
    for model_id, g in df.groupby("model"):
        family = g["family"].iloc[0]
        n_params_fallback = g["n_params"].iloc[0] if "n_params" in g.columns else None
        meta = lookup_model_meta(model_id, family, n_params_fallback=n_params_fallback)
        easy = g.loc[g["dlt_bin"] == "easy", "score"]
        hard = g.loc[g["dlt_bin"] == "hard", "score"]
        if easy.empty or hard.empty:
            continue
        e, h = float(easy.iloc[0]), float(hard.iloc[0])
        if ceiling is not None and np.isfinite(ceiling) and ceiling > 0:
            e, h = e / ceiling, h / ceiling
        rows.append({
            "model_id": model_id, "family": meta.family, "group": meta.group,
            "label": meta.label, "n_params": meta.n_params,
            "score_easy": e, "score_hard": h,
            "perplexity": float(g["perplexity"].iloc[0]) if has_ppl and pd.notna(g["perplexity"].iloc[0]) else np.nan,
        })
    return pd.DataFrame(rows) if rows else None


def aggregate_group(file_stems: list[str], hard_gt: float = 2.0) -> pd.DataFrame:
    """Ceiling-normalise each dataset file's easy/hard score (done in
    _load_one), then average per model across all listed files it appears in."""
    frames = [d for f in file_stems if (d := _load_one(f, hard_gt)) is not None]
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    agg = (
        long_df.groupby(["model_id", "family", "group", "label", "n_params"], as_index=False)
        .agg(score_easy=("score_easy", "median"), score_hard=("score_hard", "median"),
             perplexity=("perplexity", "median"), n_datasets=("model_id", "size"))
    )
    return agg


# ---------------------------------------------------------------------------
# Surprisal-track behavioral correlation cache (outputs/dlt_corr_cache/) --
# a bounded, ceiling-normalisable Spearman r(yhat_full, fdur) computed only
# over the words in each DLT bin, from scripts/dlt/surprisal_corr.py.
#
# This REPLACES DLT_GROUPS["behavioral_surprisal"] (which reads
# outputs/dlt_difficulty/{name}_dlt_difficulty_scores_multi_threshold.csv --
# confirmed that file's "score" column is a SUM of delta-loglik across the
# bin's words, not a correlation, and is NOT on the same 0..1 ceiling-fraction
# scale as every other cell in the report -- it was wrongly merged into the
# hard-case Pareto's behavioral axis before this fix). Only used by the
# hard-case Pareto (_dlt_pareto_merged_at in summarize/report.py);
# dlt/hard_bars_v2.py's own surprisal panels correctly use the
# deliberately-unnormalised mean-delta-loglik from outputs/dlt_loglik_cache/
# on their own dedicated axis and are untouched by this.
# ---------------------------------------------------------------------------
CORR_CACHE_DIR = REPO_ROOT / "outputs" / "dlt_corr_cache"

# report slug used by DLT_GROUPS["behavioral_surprisal"] -> (corr-cache file
# stem, ceiling slug). Cache files are keyed by the raw dataset name (no
# "_nr" suffix for zuco_reading), unlike DLT_FILES' "zuco_reading_nr" key.
CORR_CACHE_FILES: dict[str, tuple[str, str]] = {
    "geco": ("geco", "geco"),
    "geco_gp": ("geco_gp", "geco_gp"),
    "provo": ("provo", "provo"),
    "provo_gp": ("provo_gp", "provo_gp"),
    "provo_sp": ("provo_sp", "provo_sp"),
    "ucl_reading": ("ucl_reading", "ucl_reading"),
    "meco": ("meco", "meco"),
    "meco_gp": ("meco_gp", "meco_gp"),
    "meco_sp": ("meco_sp", "meco_sp"),
    "zuco_reading_nr": ("zuco_reading", "zuco_reading"),
    "zuco_reading_sr": ("zuco_reading_sr", "zuco_reading_sr"),
    "zuco_reading_tsr": ("zuco_reading_tsr", "zuco_reading_tsr"),
}


def _load_corr_one(report_slug: str, hard_gt: float = 2.0) -> pd.DataFrame | None:
    cache_stem, ceiling_slug = CORR_CACHE_FILES[report_slug]
    path = CORR_CACHE_DIR / f"{cache_stem}.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df = df[np.isclose(df["hard_gt_threshold"].astype(float), hard_gt)]
    if df.empty:
        return None
    ceiling = ceiling_for(ceiling_slug)
    rows = []
    for model_id, g in df.groupby("model"):
        family = g["family"].iloc[0]
        n_params_fallback = g["n_params"].iloc[0] if "n_params" in g.columns else None
        meta = lookup_model_meta(model_id, family, n_params_fallback=n_params_fallback)
        easy = g.loc[g["dlt_bin"] == "easy", "score"]
        hard = g.loc[g["dlt_bin"] == "hard", "score"]
        if easy.empty or hard.empty or pd.isna(easy.iloc[0]) or pd.isna(hard.iloc[0]):
            continue
        e, h = float(easy.iloc[0]), float(hard.iloc[0])
        if ceiling is not None and np.isfinite(ceiling) and ceiling > 0:
            e, h = e / ceiling, h / ceiling
        rows.append({
            "model_id": model_id, "family": meta.family, "group": meta.group,
            "label": meta.label, "n_params": meta.n_params,
            "score_easy": e, "score_hard": h, "perplexity": np.nan,
        })
    return pd.DataFrame(rows) if rows else None


def aggregate_group_pareto_behavioral(hard_gt: float = 2.0) -> pd.DataFrame:
    """Behavioral axis for the hard-case Pareto: embedding-track files via
    _load_one (unchanged, already bounded/ceiling-normalised) + surprisal-
    track files via _load_corr_one (bounded Spearman r, replacing the
    incommensurable summed-delta-loglik data), averaged per model together."""
    frames = [d for f in DLT_GROUPS["behavioral_embedding"] if (d := _load_one(f, hard_gt)) is not None]
    frames += [d for f in CORR_CACHE_FILES if (d := _load_corr_one(f, hard_gt)) is not None]
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    agg = (
        long_df.groupby(["model_id", "family", "group", "label", "n_params"], as_index=False)
        .agg(score_easy=("score_easy", "median"), score_hard=("score_hard", "median"),
             perplexity=("perplexity", "median"), n_datasets=("model_id", "size"))
    )
    return agg


# ---------------------------------------------------------------------------
# Surprisal-track NATIVE metric (outputs/dlt_loglik_cache/) -- mean
# per-observation delta-log-likelihood (full LME - baseline LME) per DLT bin,
# ceiling-normalised via dll_ceiling_for() -- from
# scripts/dlt/delta_loglik.py (behavioral) and
# scripts/dlt/delta_loglik_podcast.py (podcast neural). This is the
# SAME per-observation-DLL-ceiling convention used everywhere else in the
# report now (BehavScore/NeuralScore, figures/pareto_figures.py) --
# CORR_CACHE_FILES/_load_corr_one above predate that convention and score a
# held-out Pearson-correlation-delta instead, a different scale entirely.
# Kept only for summarize/report.py's own hard-case Pareto section
# (unchanged, out of scope here). The paper figures
# (figures/figure2_hard_dlt_pareto.py, build_paper_figure6_dlt_
# threshold_grid.py) use the loaders below so their DLT-restricted axes match
# every other cell in those figures (found 2026-09-08 -- a reader noticed the
# DLT grid's axis looked inconsistent with the rest of the paper).
#
# Reuses the FULL-dataset per-observation DLL ceiling (same one dll_ceiling_
# for() returns for the all-words track) to normalise a DLT-bin-restricted
# mean -- the same reuse-the-whole-dataset-ceiling convention CORR_CACHE_FILES
# already uses above for its bin-restricted correlations.
# ---------------------------------------------------------------------------
LOGLIK_CACHE_DIR = REPO_ROOT / "outputs" / "dlt_loglik_cache"

# report slug -> (loglik-cache file stem, DLL_CEILING_TAGS ceiling slug).
LOGLIK_CACHE_FILES: dict[str, tuple[str, str]] = {
    "geco": ("geco", "geco"),
    "geco_gp": ("geco_gp", "geco_gp"),
    "meco": ("meco", "meco"),
    "meco_gp": ("meco_gp", "meco_gp"),
    "meco_sp": ("meco_sp", "meco_sp"),
    "provo": ("provo", "provo"),
    "provo_gp": ("provo_gp", "provo_gp"),
    "provo_sp": ("provo_sp", "provo_sp"),
    "ucl_reading": ("ucl_reading", "ucl_reading"),
    "zuco_reading_nr": ("zuco_reading", "zuco_reading"),
    "zuco_reading_sr": ("zuco_reading_sr", "zuco_reading_sr"),
    "zuco_reading_tsr": ("zuco_reading_tsr", "zuco_reading_tsr"),
}

# Neural axis: podcast_ieeg is the only neural surprisal dataset with a
# per-observation delta-loglik cache split by DLT bin -- its per-lag
# predictions already carry a per-word delta_loglik column (see
# dlt/delta_loglik_podcast.py's docstring); the other neural
# surprisal datasets in this report have no such per-observation quantity
# available (same infeasibility list as NEURAL_CORR_CACHE_SPEC in
# figures/figure2_hard_dlt_pareto.py).
PODCAST_LOGLIK_CACHE = LOGLIK_CACHE_DIR / "podcast_lme.csv"
PODCAST_DLL_CEILING_SLUG = "podcast_ieeg__surprisal"


def _load_dll_bin(path: Path, ceiling_slug: str, dlt_bin: str, hard_gt: float) -> pd.DataFrame | None:
    """One (dataset, dlt_bin, hard_gt) slice of a dlt_loglik_cache/*.csv file,
    ceiling-normalised -- model_id/family/group/label/n_params/score."""
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df = df[np.isclose(df["hard_gt_threshold"].astype(float), hard_gt) & (df["dlt_bin"] == dlt_bin)]
    if df.empty or "score" not in df.columns:
        return None
    ceiling = dll_ceiling_for(ceiling_slug)
    rows = []
    for model_id, g in df.groupby("model"):
        v = g["score"].iloc[0]
        if pd.isna(v):
            continue
        family = g["family"].iloc[0]
        n_params_fallback = g["n_params"].iloc[0] if "n_params" in g.columns else None
        meta = lookup_model_meta(model_id, family, n_params_fallback=n_params_fallback)
        norm = float(v) / ceiling if (ceiling is not None and ceiling > 0) else float("nan")
        rows.append({"model_id": model_id, "family": meta.family, "group": meta.group,
                     "label": meta.label, "n_params": meta.n_params, "score": norm})
    return pd.DataFrame(rows) if rows else None


def load_behavioral_dll(dlt_bin: str, hard_gt: float = 2.0) -> pd.DataFrame:
    """DLL-ceiling-normalised BehavScore on one DLT bin, averaged per model
    across every LOGLIK_CACHE_FILES dataset/measure entry it appears in."""
    frames = []
    for cache_stem, ceiling_slug in LOGLIK_CACHE_FILES.values():
        d = _load_dll_bin(LOGLIK_CACHE_DIR / f"{cache_stem}.csv", ceiling_slug, dlt_bin, hard_gt)
        if d is not None:
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    return (long_df.groupby(["model_id", "family", "group", "label", "n_params"], as_index=False)
            .agg(behav=("score", "mean")))


def load_podcast_dll(dlt_bin: str, hard_gt: float = 2.0) -> pd.DataFrame:
    """DLL-ceiling-normalised NeuralScore for podcast_ieeg on one DLT bin."""
    d = _load_dll_bin(PODCAST_LOGLIK_CACHE, PODCAST_DLL_CEILING_SLUG, dlt_bin, hard_gt)
    if d is None:
        return pd.DataFrame()
    return d.rename(columns={"score": "neural"})
