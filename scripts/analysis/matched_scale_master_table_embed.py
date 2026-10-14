#!/usr/bin/env python3
"""Assemble the EMBEDDING-path (ridge/BrainScore-style encoding) counterpart
of ``analysis/matched_scale_master_table.py``'s master table. Same row shape and
purpose -- one row per (dataset, group, model), MAIN models only (no Pythia/
BabyLM checkpoint trajectories, matching the current LME table's scope) --
but score = a per-model Pearson r (not a delta-loglik), normalized by the
split-half CORRELATION-path noise ceiling (``outputs/noise_ceilings/<slug>/
ceiling_headline.csv``, computed by ``ceilings/correlation.py``) instead of
the DLL ceiling.

Motivation: every dataset in this benchmark has TWO independent modeling
pipelines -- (a) the surprisal-driven pooled LME / delta-loglik approach
(what ``master_table.csv`` already captures), and (b) a full-hidden-state
ridge encoding of the same behavioral/neural signal (what this script
captures). The user asked for the group-vs-LLM comparison to be decomposed
by BOTH domain (behavioral/neural, already in the regression as `modality`)
AND method (lme/embedding), so a result isn't presented as if it depended on
whichever pipeline happens to include more neural datasets, or on whichever
single method one dataset happens to have looked at.

Per-dataset loaders below are individually written (not one generic reader)
because the embedding-path sweep output schemas differ dataset to dataset
(per-layer-per-revision behavioral CSVs vs. already-"best"-layer neural
summary CSVs vs. per-(layer,subject) Pereira rows, etc.) -- see each loader's
docstring for the exact source file/column it reads and why.

Datasets deliberately EXCLUDED here (reported, not silently dropped):
  - podcast_ieeg: no ridge/embedding sweep exists, only the surprisal/DLT
    trees (podcast/checkpoint_sweep_surprisal, _dlt) -- these ARE the LME
    path, there is no second pipeline for Podcast in this repo.
  - zuco_reading: an embedding sweep dir exists but has essentially no data
    (1 model, opt-66b, as of 2026-09-07) -- not enough coverage for a group
    comparison.
  - nieuwland (N400): no embedding-path sweep either.

Output: outputs_jeanzay/matched_scale_report/master_table_embed.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, REGISTRY_INDEX, REPO_ROOT
from scripts.figures.figure8_scale_matched_comparison import _model_id_from_dirname

OUT_DIR = REPO_ROOT / "outputs_jeanzay" / "matched_scale_report"
OUT_PATH = OUT_DIR / "master_table_embed.csv"
JZ_ROOT = REPO_ROOT / "outputs_jeanzay"
CEILING_DIR = REPO_ROOT / "outputs" / "noise_ceilings"


def _ceiling_r(dataset: str, component: str | None) -> float | None:
    path = CEILING_DIR / dataset / "ceiling_headline.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"component": str})
    df["component"] = df["component"].fillna("")
    row = df[df["component"] == (component or "")]
    if row.empty:
        return None
    return float(row.iloc[0]["ceiling_r"])


def _reg_row(mid: str) -> tuple[str, float] | None:
    if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
        return None
    meta = REGISTRY_INDEX[mid]
    return meta.group, meta.n_params


def _rows_from_layerwise_tree(
    root: Path, csv_name_glob: str, score_col: str, *, revision_col: str = "revision",
) -> pd.DataFrame:
    """Generic loader for the common shape: <root>/<family>/<model>/<csv>
    with one row per (revision, layer[, subject]) and a plain correlation
    column -- take the MAX over layer for revision=="main" (best-layer
    convention, matching how the LME path already picks the best layer/lag
    per model via ``_best_row_per_word``)."""
    rows = []
    if not root.is_dir():
        return pd.DataFrame(rows)
    for fam_dir in sorted(root.iterdir()):
        if not fam_dir.is_dir() or fam_dir.name.endswith("_checkpoints"):
            continue
        for model_dir in sorted(fam_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            csvs = [p for p in model_dir.glob(csv_name_glob) if not p.name.endswith(".lock")]
            if not csvs:
                continue
            mid = _model_id_from_dirname(model_dir.name)
            reg = _reg_row(mid)
            if reg is None:
                continue
            try:
                df = pd.read_csv(csvs[0])
            except Exception:
                continue
            if df.empty or score_col not in df.columns or revision_col not in df.columns:
                continue
            main_df = df[df[revision_col] == "main"]
            if main_df.empty:
                continue
            if "subject" in main_df.columns and "layer" in main_df.columns:
                # Pereira-style: average across subjects within each layer first,
                # then pick the best layer.
                per_layer = main_df.groupby("layer")[score_col].mean()
                score = float(per_layer.max())
            else:
                score = float(main_df[score_col].max())
            group, n_params = reg
            rows.append(dict(group=group, model_id=mid, n_params=n_params, score=score))
    return pd.DataFrame(rows)


def _load_reading_measure(dataset_root: str, measure_subdir: str | None, csv_glob: str) -> pd.DataFrame:
    """FP measures have per-subject ridge fits (plain ``<measure>/`` subdir);
    GP/SP measures for geco/meco/provo/onestop were only ever run as
    subject-AVERAGED ridge fits (plain ``<measure>/`` subdir is empty, data
    lives under ``<measure>_average/`` instead) -- fall back to that when the
    per-subject tree has nothing, and flag it so the caveat is visible."""
    base = JZ_ROOT / dataset_root
    root = base / measure_subdir if measure_subdir else base
    df = _rows_from_layerwise_tree(root, csv_glob, "pearsonr")
    if df.empty and measure_subdir:
        avg_root = base / f"{measure_subdir}_average"
        df = _rows_from_layerwise_tree(avg_root, csv_glob, "pearsonr")
        if not df.empty:
            print(f"    [{dataset_root}/{measure_subdir}] no per-subject data -- "
                  f"using subject-AVERAGED ridge fits from {measure_subdir}_average/ instead")
    elif df.empty:
        # no measure subdir (e.g. ucl_reading): the sweep may exist only as <dataset_root>_average/
        avg_root = base.with_name(base.name + "_average")
        df = _rows_from_layerwise_tree(avg_root, csv_glob, "pearsonr")
        if not df.empty:
            print(f"    [{dataset_root}] no per-subject data -- "
                  f"using subject-AVERAGED ridge fits from {avg_root.name}/ instead")
    return df


def _load_naturalstories_spr() -> pd.DataFrame:
    """Already-"best" per-model summary (flat_pearsonr/best_pearsonr columns,
    no separate layer rows to max over) -- use best_pearsonr directly."""
    root = JZ_ROOT / "naturalstories/embedding_sweep"
    return _rows_from_layerwise_tree(root, "spr_checkpoint_scores.csv", "best_pearsonr")


def _load_fedorenko2016() -> pd.DataFrame:
    """Already dataset-level (one row per revision, mean_best_spearmanr
    column) -- the noise ceiling's metric is mean_best_SPEARMANr (see
    ceiling_headline.csv), so use that column, not pearson."""
    root = JZ_ROOT / "fedorenko2016/embedding_sweep"
    return _rows_from_layerwise_tree(root, "fedorenko2016_checkpoint_scores.csv", "mean_best_spearmanr")


def _load_ns_fmri() -> pd.DataFrame:
    """Already dataset-level: one row per revision with mean_best_pearsonr
    (LANG+MD combined, matching the ceiling's "all" component)."""
    root = JZ_ROOT / "ns_fmri/embedding_sweep"
    return _rows_from_layerwise_tree(root, "ns_fmri_checkpoint_scores.csv", "mean_best_pearsonr")


def _load_tuckute_fmri() -> pd.DataFrame:
    """One row per (model, revision) already; B_mean_best_pearsonr = the
    Baseline-condition mean-pearsonr-across-ROIs column, matching the
    ceiling's "cond_B" component (Tuckute's primary/baseline condition)."""
    root = JZ_ROOT / "tuckute/embedding_sweep"
    return _rows_from_layerwise_tree(root, "tuckute_scores.csv", "B_mean_best_pearsonr")


def _load_lpp_fmri() -> pd.DataFrame:
    """Per-model-dir ``lpp_checkpoint_layerwise.csv`` (layer, mean_pearson,
    revision) -- NOT the ``summary/all_models_lpp_scores.csv`` aggregate,
    which turns out to only cover 5 small Pythia checkpoints, not the full
    main-model set. Directory layout here is flatter than the other embed
    trees (model dirs directly under the root or one non-checkpoint family
    subdir, mixed together) -- ``_rows_from_layerwise_tree`` assumes exactly
    one level of family dirs, so walk explicitly instead."""
    root = JZ_ROOT / "lpp/embedding_sweep"
    rows = []
    if not root.is_dir():
        return pd.DataFrame(rows)
    candidate_dirs = set()
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name in ("summary",) or entry.name.endswith("_checkpoints"):
            continue
        if (entry / "lpp_checkpoint_layerwise.csv").is_file():
            candidate_dirs.add(entry)
        else:
            for sub in entry.iterdir():
                if sub.is_dir() and (sub / "lpp_checkpoint_layerwise.csv").is_file():
                    candidate_dirs.add(sub)
    for model_dir in sorted(candidate_dirs):
        csv_path = model_dir / "lpp_checkpoint_layerwise.csv"
        mid = _model_id_from_dirname(model_dir.name)
        reg = _reg_row(mid)
        if reg is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty or "mean_pearson" not in df.columns or "revision" not in df.columns:
            continue
        main_df = df[df["revision"] == "main"]
        if main_df.empty:
            continue
        group, n_params = reg
        rows.append(dict(group=group, model_id=mid, n_params=n_params, score=float(main_df["mean_pearson"].max())))
    return pd.DataFrame(rows)


def _load_pereira_fmri() -> pd.DataFrame:
    """Per-(layer, subject) rows -- average pearsonr_mean across subjects
    within each layer, then take the best layer (same convention as the
    generic loader's Pereira branch)."""
    root = JZ_ROOT / "pereira/embedding_sweep"
    return _rows_from_layerwise_tree(root, "pereira_embed_by_layer.csv", "pearsonr_mean")


# (dataset slug, ceiling dataset, ceiling component, loader) -- slug matches
# the LME master table's naming (geco_gp/meco_sp/etc.) so the two tables
# stack directly with a "method" column added.
SPECS: list[tuple[str, str, str | None, callable]] = [
    ("geco", "geco", "FP", lambda: _load_reading_measure("geco/embedding_sweep", "fp", "geco_embed_checkpoint_scores.csv")),
    ("geco_gp", "geco", "GP", lambda: _load_reading_measure("geco/embedding_sweep", "gp", "geco_embed_checkpoint_scores.csv")),
    ("meco", "meco", "FP", lambda: _load_reading_measure("meco/embedding_sweep", "fp", "meco_embed_checkpoint_scores.csv")),
    ("meco_gp", "meco", "GP", lambda: _load_reading_measure("meco/embedding_sweep", "gp", "meco_embed_checkpoint_scores.csv")),
    ("meco_sp", "meco", "SP", lambda: _load_reading_measure("meco/embedding_sweep", "sp", "meco_embed_checkpoint_scores.csv")),
    ("provo", "provo", "FP", lambda: _load_reading_measure("provo/embedding_sweep", "fp", "provo_embed_checkpoint_scores.csv")),
    ("provo_gp", "provo", "GP", lambda: _load_reading_measure("provo/embedding_sweep", "gp", "provo_embed_checkpoint_scores.csv")),
    ("provo_sp", "provo", "SP", lambda: _load_reading_measure("provo/embedding_sweep", "sp", "provo_embed_checkpoint_scores.csv")),
    ("onestop_fp", "onestop", "FP", lambda: _load_reading_measure("onestop/embedding_sweep", "fp", "onestop_embed_checkpoint_scores.csv")),
    ("onestop_gp", "onestop", "GP", lambda: _load_reading_measure("onestop/embedding_sweep", "gp", "onestop_embed_checkpoint_scores.csv")),
    ("onestop_sp", "onestop", "SP", lambda: _load_reading_measure("onestop/embedding_sweep", "sp", "onestop_embed_checkpoint_scores.csv")),
    ("ucl_reading", "ucl_reading", None, lambda: _load_reading_measure("ucl_reading/embedding_sweep", None, "ucl_reading_embed_checkpoint_scores.csv")),
    ("naturalstories_spr", "naturalstories_spr", None, _load_naturalstories_spr),
    ("fedorenko2016_ecog", "fedorenko2016", None, _load_fedorenko2016),
    ("ns_fmri", "ns_fmri", "all", _load_ns_fmri),
    ("tuckute_fmri", "tuckute", "cond_B", _load_tuckute_fmri),
    ("lpp_fmri", "lpp", None, _load_lpp_fmri),
    ("pereira_fmri", "pereira", None, _load_pereira_fmri),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for slug, ceiling_ds, component, loader in SPECS:
        print(f"=== {slug} (embedding path) ===")
        df = loader()
        if df.empty:
            print("  no data -- skipping")
            continue
        ceiling = _ceiling_r(ceiling_ds, component)
        df["dataset"] = slug
        df["ceiling"] = ceiling
        df["score_raw"] = df["score"]
        df["score_norm"] = df["score_raw"] / ceiling if ceiling else float("nan")
        n_groups = df["group"].nunique()
        print(f"  {len(df)} main models across {n_groups} group(s), ceiling_r={ceiling}")
        print(df.groupby("group")["model_id"].nunique().to_dict())
        frames.append(df)

    if not frames:
        print("No embedding-path data found at all.")
        return
    master = pd.concat(frames, ignore_index=True)
    master = master[["dataset", "group", "model_id", "n_params", "score_raw", "ceiling", "score_norm"]]
    master.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(master)} rows -> {OUT_PATH}")
    print(master.groupby(["dataset", "group"])["model_id"].nunique().unstack(fill_value=0))


if __name__ == "__main__":
    main()
