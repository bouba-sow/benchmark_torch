#!/usr/bin/env python3
"""Assemble ONE long-format master table for the ICLR matched-scale report:
every model (main checkpoint of every group, plus every Pythia and BabyLM
GPT2 training checkpoint) x every dataset we have last-layer-surprisal
delta-loglik results for.

Columns: dataset, group, model_id, revision, is_checkpoint, checkpoint_family
(pythia_checkpoints/babylm_gpt2_checkpoints/None), n_params, perplexity,
dll_total, n_obs, score_raw (=dll_total/n_obs), score_norm (score_raw/ceiling,
NaN where no delta-loglik ceiling exists -- the 3 N400 datasets).

Data sources (all last-layer-only, out-of-sample delta_loglik_test):
  - Main (single "main"-revision) models of all 4 groups: the DEFAULT sweep
    root each dataset already uses for the standard benchmark (unaffected by
    the checkpoint-collision bug -- that bug only bites when >1 revision of
    the SAME model is fit concurrently, and a "main"-only model has exactly
    one).
  - pythia_checkpoints / babylm_gpt2_checkpoints trajectories: the BUG-FIXED
    trees (revision-namespaced working files) -- see
    figures/figure9_pythia_checkpoint_scaling.py's module docstring for
    which dataset uses which root; reused directly here (OOS_SWEEP_ROOTS /
    N400_SWEEP_ROOTS / FMRI_SWEEP_ROOTS / PERPLEXITY_PATHS imported from that
    module, not re-derived).

Perplexity for main models: looked up in the same per-dataset
<dataset>_perplexity_summary.csv used for checkpoints, keyed on
(model_id, revision="main"). Where a model has >1 "main"-revision row (a
handful of models were perplexity-scored twice, once by an older ad hoc pass
with family=NaN and once by the canonical per-family script), the
family-tagged row is preferred and the ambiguous NaN-family row is dropped --
flagged in the printed summary so it's auditable, not silently resolved.

Output: outputs_jeanzay/matched_scale_report/master_table.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd

from neuroai.analysis.benchmark_common import (
    EXCLUDED_MODEL_IDS, REGISTRY_INDEX, REPO_ROOT, read_lme_by_layer_csv, sum_dll_across_units,
)
from scripts.figures.figure8_scale_matched_comparison import (
    DLL_DATASET_SPECS,
    NOISE_CEILING_DIR,
    _best_row_per_word,
    _model_id_from_dirname,
)
from scripts.figures.figure9_pythia_checkpoint_scaling import (
    FMRI_BY_LAYER_FILENAMES,
    FMRI_SWEEP_ROOTS,
    INSAMPLE_SWEEP_ROOTS,
    N400_SWEEP_ROOTS,
    OOS_SWEEP_ROOTS,
    PERPLEXITY_PATHS,
)

OUT_DIR = REPO_ROOT / "outputs_jeanzay" / "matched_scale_report"
OUT_PATH = OUT_DIR / "master_table.csv"

CHECKPOINT_FAMILIES = ("pythia_checkpoints", "babylm_gpt2_checkpoints")

# build_paper_figure9's N400_SWEEP_ROOTS/FMRI_SWEEP_ROOTS/PERPLEXITY_PATHS use
# SHORT dataset keys for the 5 neural datasets ("lpp", "tuckute", "podcast",
# "fedorenko2016", "naturalstories_fmri") that differ from DLL_DATASET_SPECS'
# LONG keys ("lpp_fmri", "tuckute_fmri", "podcast_ieeg", "fedorenko2016_ecog",
# "ns_fmri") used everywhere else in this script -- bridge the two here
# rather than renaming either module's convention.
SHORT_SLUG: dict[str, str] = {
    "fedorenko2016_ecog": "fedorenko2016",
    "lpp_fmri": "lpp",
    "tuckute_fmri": "tuckute",
    "podcast_ieeg": "podcast",
    "ns_fmri": "naturalstories_fmri",
}

# provo's babylm_gpt2_checkpoints tree lives under the PLAIN "_oos" root, not
# OOS_SWEEP_ROOTS["provo"] (which points at the pythia-only "oldformula_oos"
# ablation tree -- babylm was never part of that ablation). Override just
# this one (dataset, family) pair.
_PROVO_BABYLM_OOS_ROOT = "provo/surprisal_sweep_checkpoints/fp"

# N400 datasets: not in DLL_DATASET_SPECS (no delta_loglik ceiling -- split-
# half Pearson-r ceiling instead, a different unit; scores are reported raw
# here, matching pythia_checkpoint_scaling.pdf's convention). "*_lme_by_layer"
# is the same shape build_paper_figure8 reads for lpp/tuckute/fedorenko2016.
# "megmasc" (Gwilliams MEG-MASC, added 2026-09-04) shares this same
# no-ceiling-yet, raw-score, single-"main"-revision by_layer.csv shape --
# not literally N400, but identical treatment (no ceiling script has been
# written for it yet; main-model sweep data IS present under
# megmasc/surprisal_sweep/<family>/<model>/megmasc_lme_by_layer.csv).
# Neural datasets that are NOT in DLL_DATASET_SPECS but ARE part of the benchmark (2026-10-01: all 18
# datasets enter the matched-scale analysis): slug -> (sweep root under outputs_jeanzay, per-model by-layer file,
# report slug whose DLL ceiling to use via benchmark_common.dll_ceiling_for). Main models only, one row each.
_EXTRA_DATASETS: dict[str, tuple[str, str, str]] = {
    "nieuwland":      ("nieuwland/surprisal_sweep",      "nieuwland_lme_by_layer.csv", "nieuwland_eeg"),
    "megmasc":        ("megmasc/surprisal_sweep",        "megmasc_lme_by_layer.csv",   "megmasc_meg"),
    "broderick":      ("broderick/surprisal_sweep",      "broderick_lme_by_layer.csv", "broderick_eeg"),
    "szewczyk_f2007": ("szewczyk_f2007/surprisal_sweep", "szewczyk_lme_by_layer.csv",  "szewczyk_f2007_eeg"),
    "szewczyk_h2019": ("szewczyk_h2019/surprisal_sweep", "szewczyk_lme_by_layer.csv",  "szewczyk_h2019_eeg"),
}
N400_DATASETS = tuple(_EXTRA_DATASETS)


def _ceiling(slug: str) -> float | None:
    if slug in _EXTRA_DATASETS:
        from neuroai.analysis.benchmark_common import dll_ceiling_for
        return dll_ceiling_for(_EXTRA_DATASETS[slug][2] + "__surprisal")
    spec = DLL_DATASET_SPECS.get(slug)
    if spec is None:
        return None
    path = NOISE_CEILING_DIR / f"delta_loglik_ceiling_{spec['ceiling_tag']}_oos.json"
    if not path.exists():
        return None
    import json
    with open(path) as f:
        return json.load(f)["ceiling_delta_loglik_per_obs"]


def _dataset_root_and_pattern(slug: str) -> tuple[str, str, str, str | None] | None:
    """(root, pattern_kind, pattern, group_col) for a dataset's DEFAULT root
    (main models) and its file pattern. pattern_kind is "glob" (behavioral,
    fnmatch-style pattern) or "name" (N400/neural, exact filename)."""
    spec = DLL_DATASET_SPECS.get(slug)
    if spec is not None:
        return spec["root"], "glob", spec["pattern"], spec.get("group_col")
    if slug in _EXTRA_DATASETS:
        root, fname, _ = _EXTRA_DATASETS[slug]
        return root, "name", fname, None
    return None


def _read_csv_rows(csv_path: Path, dll_col: str, group_col: str | None,
                   keep_max_n_obs: bool = False) -> list[dict]:
    """Every (revision, dll_total, n_obs) triple in one model's raw sweep CSV,
    picking the best layer/lag per revision (and, for group_col e.g. "froi",
    summing dll/n_obs across units first). Non-checkpoint ("main"-only)
    sweeps for several datasets (fedorenko2016/lpp/tuckute/podcast/N400) pre-
    date the checkpoint convention and carry NO "revision" column at all
    (exactly one implicit "main" model per file) -- treated as a single
    revision="main" group here, matching build_paper_figure8's _load_dll
    (which never groups by revision either)."""
    # Robust reader: plain pd.read_csv either raises ParserError (which the
    # caller silently swallows -- this dropped every LPP Mamba/Mamba2 model
    # from the master table) or, worse, silently mis-shifts columns on the
    # sweep files that mix 26- and 29-column rows.
    df = read_lme_by_layer_csv(csv_path)
    if df.empty or dll_col not in df.columns:
        return []
    if keep_max_n_obs and "n_obs" in df.columns:
        # LPP files mix stale grand-average-subject rows (n_obs=19,950) with
        # correct per-subject-pooled rows (957,600 = 48 subjects x more);
        # only the per-subject regime is valid (see summarize/lpp_lme.py).
        n_obs = pd.to_numeric(df["n_obs"], errors="coerce")
        df = df[n_obs == n_obs.max()]
    if "revision" not in df.columns:
        groups = [("main", df)]
    else:
        groups = list(df.groupby("revision"))
    out = []
    for revision, rev_df in groups:
        if group_col:
            per_unit = []
            for _, unit_df in rev_df.groupby(group_col):
                picked = _best_row_per_word(unit_df)
                if picked is None:
                    continue
                per_unit.append(picked)
            total_dll, total_n = sum_dll_across_units(per_unit)
            if total_n == 0:
                continue
            out.append(dict(revision=revision, dll_total=total_dll, n_obs=total_n))
        else:
            picked = _best_row_per_word(rev_df)
            if picked is None:
                continue
            out.append(dict(revision=revision, dll_total=picked[0], n_obs=picked[1]))
    return out


def _load_main_models(slug: str) -> pd.DataFrame:
    """One row per (group, model) main-checkpoint result for one dataset."""
    info = _dataset_root_and_pattern(slug)
    if info is None:
        return pd.DataFrame()
    root, pattern_kind, pattern, group_col = info
    sweep_root = REPO_ROOT / "outputs_jeanzay" / root
    if not sweep_root.is_dir():
        return pd.DataFrame()

    rows = []
    for fam_dir in sorted(sweep_root.iterdir()):
        if not fam_dir.is_dir() or fam_dir.name in CHECKPOINT_FAMILIES:
            continue
        for model_dir in sorted(fam_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if pattern_kind == "glob":
                csvs = [p for p in model_dir.glob(pattern) if not p.name.endswith(".lock")]
                if not csvs:
                    continue
                csv_path = csvs[0]
            else:
                csv_path = model_dir / pattern
                if not csv_path.is_file():
                    continue
            mid = _model_id_from_dirname(model_dir.name)
            if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
                continue
            try:
                recs = _read_csv_rows(csv_path, "delta_loglik_test", group_col,
                                      keep_max_n_obs=(slug == "lpp_fmri"))
            except Exception:
                continue
            main_recs = [r for r in recs if r["revision"] in ("main", "step143000")]
            if not main_recs:
                continue
            r = main_recs[0]
            meta = REGISTRY_INDEX[mid]
            rows.append(dict(
                dataset=slug, group=meta.group, model_id=mid, revision="main",
                is_checkpoint=False, checkpoint_family=None,
                n_params=meta.n_params, dll_total=r["dll_total"], n_obs=r["n_obs"],
            ))
    return pd.DataFrame(rows)


def _load_checkpoints(slug: str, family: str) -> pd.DataFrame:
    """One row per (revision) for the given checkpoint family, for one
    dataset -- reuses figure9's already-established root-per-dataset maps."""
    short = SHORT_SLUG.get(slug, slug)
    if slug in OOS_SWEEP_ROOTS:
        root = OOS_SWEEP_ROOTS[slug]
        if slug == "provo" and family == "babylm_gpt2_checkpoints":
            root = _PROVO_BABYLM_OOS_ROOT
        spec = DLL_DATASET_SPECS[slug]
        pattern_kind, pattern, group_col = "glob", spec["pattern"], spec.get("group_col")
    elif short in N400_SWEEP_ROOTS:
        root = N400_SWEEP_ROOTS[short]
        pattern_kind, pattern, group_col = "name", f"{short}_lme_by_layer.csv", None
    elif short in FMRI_SWEEP_ROOTS:
        root = FMRI_SWEEP_ROOTS[short]
        if short == "naturalstories_fmri":
            pattern_kind, pattern, group_col = "name", "ns_fmri_lme_by_roi.csv", "froi"
        else:
            pattern_kind, pattern, group_col = "name", FMRI_BY_LAYER_FILENAMES[short], None
    elif slug in DLL_DATASET_SPECS:
        # Generic fallback for any dataset whose checkpoint families (when
        # present) live in the SAME tree as its main models -- true for
        # onestop_fp/gp/sp and pereira_fmri (added 2026-09-03), same pattern
        # as the N400/FMRI branches above but without needing a dedicated
        # per-dataset root map, since DLL_DATASET_SPECS already has it.
        spec = DLL_DATASET_SPECS[slug]
        root = spec["root"]
        pattern_kind, pattern, group_col = "glob", spec["pattern"], spec.get("group_col")
    else:
        return pd.DataFrame()

    fam_dir = REPO_ROOT / "outputs_jeanzay" / root / family
    if not fam_dir.is_dir():
        return pd.DataFrame()

    rows = []
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if pattern_kind == "glob":
            csvs = [p for p in model_dir.glob(pattern) if not p.name.endswith(".lock")]
            if not csvs:
                continue
            csv_path = csvs[0]
        else:
            csv_path = model_dir / pattern
            if not csv_path.is_file():
                continue
        mid = _model_id_from_dirname(model_dir.name)
        if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
            continue
        try:
            recs = _read_csv_rows(csv_path, "delta_loglik_test", group_col)
        except Exception:
            continue
        meta = REGISTRY_INDEX[mid]
        for r in recs:
            rows.append(dict(
                dataset=slug, group=meta.group, model_id=mid, revision=r["revision"],
                is_checkpoint=True, checkpoint_family=family,
                n_params=meta.n_params, dll_total=r["dll_total"], n_obs=r["n_obs"],
            ))
    return pd.DataFrame(rows)


# Perplexity summary CSVs for datasets not in figure9's PERPLEXITY_PATHS, or
# where figure9's entry points at the wrong / checkpoints-only file.
#
# Guiding principle (2026-09-10): perplexity is a property of the CORPUS TEXT,
# not of which reading-time measure was recorded from it -- so all measures
# within one behavioral dataset (FP / GP / SP) share a single perplexity
# table, exactly as OneStop's fp/gp/sp already did. That extends to:
#   * geco_gp             -> geco's table  (same English-novel text as geco FP)
#   * meco_gp, meco_sp    -> meco's table  (same passages as meco FP)
#   * provo_gp, provo_sp  -> provo's table (same paragraphs as provo FP)
# ZuCo SR / TSR are DIFFERENT tasks with different sentence sets, so they get
# their own dedicated per-task tables (which already exist on disk).
# naturalstories_spr: figure9's PERPLEXITY_PATHS entry points at
# naturalstories_spr_perplexity_summary.csv, which is Pythia-checkpoints-only
# (11 models, no "main" rows); the real per-model main table is
# naturalstories_perplexity_summary.csv (100 models, n_stories=10 == the SPR
# corpus). That file has no `revision` column -> _perplexity_lookup now
# coerces a revision-less table to revision="main".
_EXTRA_PERPLEXITY_PATHS: dict[str, str] = {
    "onestop_fp": "outputs_jeanzay/onestop/perplexity/onestop_perplexity_summary.csv",
    "onestop_gp": "outputs_jeanzay/onestop/perplexity/onestop_perplexity_summary.csv",
    "onestop_sp": "outputs_jeanzay/onestop/perplexity/onestop_perplexity_summary.csv",
    "pereira_fmri": "outputs_jeanzay/pereira/perplexity/pereira_perplexity_summary.csv",
    "naturalstories_spr": "outputs_jeanzay/naturalstories/perplexity/naturalstories_perplexity_summary.csv",
    # ns_fmri: deliberately NOT wired to any table. Its own text (8 of the 10
    # Natural Stories) gives perplexities ~47% off the SPR 10-story corpus
    # (verified test in figure9's PERPLEXITY_PATHS comment), and no
    # standard-LLM perplexity was ever computed on the fMRI corpus itself.
    # ns_fmri is excluded outright from the matched-scale regression instead
    # (see _EXCLUDE_DATASETS in analysis/matched_scale_regression.py).
    "geco_gp": "outputs_jeanzay/geco/perplexity/geco_perplexity_summary.csv",
    "meco_gp": "outputs_jeanzay/meco/perplexity/meco_perplexity_summary.csv",
    "meco_sp": "outputs_jeanzay/meco/perplexity/meco_perplexity_summary.csv",
    "provo_gp": "outputs_jeanzay/provo/perplexity/provo_perplexity_summary.csv",
    "provo_sp": "outputs_jeanzay/provo/perplexity/provo_perplexity_summary.csv",
    "zuco_reading_sr": "outputs_jeanzay/zuco_reading/perplexity/zuco_reading_sr_perplexity_summary.csv",
    "zuco_reading_tsr": "outputs_jeanzay/zuco_reading/perplexity/zuco_reading_tsr_perplexity_summary.csv",
    # 2026-10-01: ns_fmri now has a perplexity table computed on ITS OWN corpus (the 8 scanned stories;
    # scripts/perplexity/ns_fmri.py), and the two Szewczyk studies on their own stimulus sentences.
    "ns_fmri": "outputs_jeanzay/ns_fmri/perplexity/ns_fmri_perplexity_summary.csv",
    "nieuwland": "outputs_jeanzay/nieuwland/perplexity/nieuwland_perplexity_summary.csv",
    "megmasc": "outputs_jeanzay/megmasc/perplexity/megmasc_perplexity_summary.csv",
    "broderick": "outputs_jeanzay/broderick/perplexity/broderick_perplexity_summary.csv",
    "szewczyk_f2007": "outputs_jeanzay/szewczyk_f2007/perplexity/szewczyk_f2007_perplexity_summary.csv",
    "szewczyk_h2019": "outputs_jeanzay/szewczyk_h2019/perplexity/szewczyk_h2019_perplexity_summary.csv",
    # 2026-10-13 (v2 rerun): per-word logit perplexity for every model now sits in each dataset's own
    # perplexity table, so these take precedence over the older per-model sources below (the embedding-
    # sweep perplexities used the model's own loss, per token). Podcast is derived from the v2 surprisal
    # caches (reports/benchmark_audit_20261012/claude/build_ppl_summaries.py).
    "fedorenko2016_ecog": "outputs_jeanzay/fedorenko2016/perplexity/fedorenko2016_perplexity_summary.csv",
    "tuckute_fmri": "outputs_jeanzay/tuckute/perplexity/tuckute_perplexity_summary.csv",
    "podcast_ieeg": "outputs_jeanzay/podcast/perplexity/podcast_perplexity_summary.csv",
    "lpp_fmri": "outputs_jeanzay/lpp/perplexity/lpp_perplexity_summary.csv",
}

# The three neural datasets whose standalone *_perplexity_summary.csv only
# covers the constrained/BabyLM subset (~39-54 models) -- their standard-LLM
# perplexity lives elsewhere, per-model:
#   * fedorenko2016_ecog: in the embedding checkpoint-sweep score files, one
#     per model dir, `perplexity` column, revision=="main" row (~95/100).
#   * podcast_ieeg:  per-model perplexity.json under
#     outputs_jeanzay/podcast/perplexity/<model_dir>/perplexity.json (100/100).
#   * tuckute_fmri:  per-model perplexity.json under
#     outputs_jeanzay/tuckute/embedding_sweep/<family>/<model_dir>/perplexity.json.
# The one-per-model-dir "perplexity.json" file is the model's FINAL-checkpoint
# perplexity even when its `revision` field carries the real HF branch name
# (e.g. OLMo-2 "stage1-step...-tokens...B", pythia "step143000") rather than
# the literal string "main" -- so _perplexity_from_json_glob() ignores the
# revision field for these and always emits revision="main".
# fedorenko2016 + tuckute additionally fall back to their standalone
# *_perplexity_summary.csv for any model the primary source misses (recovers
# cbr-rnn / cbr-rnn-m). A few OLMo-2 models still have no fedorenko perplexity
# anywhere and are simply left NaN.
# (ns_fmri has NO standard-LLM perplexity on disk -- it is excluded from the
# regression entirely, so it is not handled here at all.)
_FEDORENKO_PPL_SWEEP_ROOT = "outputs_jeanzay/fedorenko2016/embedding_sweep"
_FEDORENKO_PPL_SCORE_FILE = "fedorenko2016_checkpoint_scores.csv"
_PPL_JSON_GLOBS: dict[str, str] = {
    "podcast_ieeg": "outputs_jeanzay/podcast/perplexity/*/perplexity.json",
    "tuckute_fmri": "outputs_jeanzay/tuckute/embedding_sweep/*/*/perplexity.json",
}
# Standalone summary CSVs used only to fill gaps left by the per-model sources.
_PPL_SUMMARY_FALLBACK: dict[str, str] = {
    "fedorenko2016_ecog": "outputs_jeanzay/fedorenko2016/perplexity/fedorenko2016_perplexity_summary.csv",
    "tuckute_fmri": "outputs_jeanzay/tuckute/perplexity/tuckute_perplexity_summary.csv",
}


def _perplexity_from_fedorenko_sweep() -> pd.DataFrame:
    """(model_id, revision='main') -> perplexity, read from the per-model
    fedorenko2016 embedding-sweep score files (`perplexity` column). Covers
    the standard LLMs that the standalone fedorenko2016_perplexity_summary.csv
    (constrained/BabyLM only) misses."""
    root = REPO_ROOT / _FEDORENKO_PPL_SWEEP_ROOT
    if not root.is_dir():
        return pd.DataFrame(columns=["model_id", "revision", "perplexity"])
    rows = []
    for score_path in root.glob(f"*/*/{_FEDORENKO_PPL_SCORE_FILE}"):
        mid = _model_id_from_dirname(score_path.parent.name)
        if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
            continue
        try:
            d = pd.read_csv(score_path)
        except Exception:
            continue
        if "perplexity" not in d.columns:
            continue
        sub = d[d.get("revision", "main").astype(str) == "main"] if "revision" in d.columns else d
        sub = sub.dropna(subset=["perplexity"])
        if sub.empty:
            continue
        rows.append({"model_id": mid, "revision": "main",
                     "perplexity": float(sub["perplexity"].iloc[0])})
    return pd.DataFrame(rows, columns=["model_id", "revision", "perplexity"])


def _perplexity_from_json_glob(pattern: str) -> pd.DataFrame:
    """(model_id, revision='main') -> perplexity, from the one-per-model-dir
    ``perplexity.json`` files matched by `pattern` (relative to REPO_ROOT).
    These files carry a single model's FINAL-checkpoint perplexity; their
    `revision` field may hold the real HF branch name rather than "main", so
    it is ignored. model_id is taken from the JSON's field, else the dir name."""
    rows = []
    for jp in REPO_ROOT.glob(pattern):
        try:
            j = json.loads(jp.read_text())
        except Exception:
            continue
        p = j.get("perplexity")
        if p is None:
            continue
        mid = j.get("model_id") or _model_id_from_dirname(jp.parent.name)
        if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
            continue
        rows.append({"model_id": mid, "revision": "main", "perplexity": float(p)})
    df = pd.DataFrame(rows, columns=["model_id", "revision", "perplexity"])
    return df.drop_duplicates(subset=["model_id", "revision"], keep="first")


def _read_ppl_summary_csv(rel_path: str, slug: str) -> pd.DataFrame:
    """(model_id, revision, perplexity) from a standalone *_perplexity_summary.csv.
    A revision-less table (one row per model, no checkpoint axis) is treated as
    revision='main'."""
    path = REPO_ROOT / rel_path
    if not path.exists():
        return pd.DataFrame(columns=["model_id", "revision", "perplexity"])
    df = pd.read_csv(path)
    if "model_id" not in df.columns or "perplexity" not in df.columns:
        return pd.DataFrame(columns=["model_id", "revision", "perplexity"])
    if "revision" not in df.columns:
        df = df.assign(revision="main")
    df = df.dropna(subset=["model_id", "revision", "perplexity"])
    if "family" in df.columns:
        df = df.sort_values("family", na_position="last")  # NaN family -> dropped by keep="first"
    dupes = df[df.duplicated(subset=["model_id", "revision"], keep=False)]
    if not dupes.empty:
        print(f"  [{slug}] perplexity: {dupes['model_id'].nunique()} model(s) had >1 row for the same "
              f"(model_id, revision) -- kept the family-tagged one, dropped the rest")
    return df.drop_duplicates(subset=["model_id", "revision"], keep="first")[["model_id", "revision", "perplexity"]]


def _perplexity_lookup(slug: str) -> pd.DataFrame:
    """(model_id, revision) -> perplexity for one dataset. Uses the best
    per-model source for that dataset, then fills any remaining gaps from a
    standalone summary CSV where one is configured (fedorenko2016 / tuckute)."""
    if slug in _EXTRA_PERPLEXITY_PATHS:
        primary = _read_ppl_summary_csv(_EXTRA_PERPLEXITY_PATHS[slug], slug)
        return primary
    if slug == "fedorenko2016_ecog":
        primary = _perplexity_from_fedorenko_sweep()
    elif slug in _PPL_JSON_GLOBS:
        primary = _perplexity_from_json_glob(_PPL_JSON_GLOBS[slug])
    else:
        rel_path = _EXTRA_PERPLEXITY_PATHS.get(slug) or PERPLEXITY_PATHS.get(SHORT_SLUG.get(slug, slug))
        primary = _read_ppl_summary_csv(rel_path, slug) if rel_path else \
            pd.DataFrame(columns=["model_id", "revision", "perplexity"])

    fb_path = _PPL_SUMMARY_FALLBACK.get(slug)
    if fb_path:
        fb = _read_ppl_summary_csv(fb_path, slug)
        have = set(zip(primary["model_id"], primary["revision"]))
        extra = fb[~fb.apply(lambda r: (r["model_id"], r["revision"]) in have, axis=1)]
        if len(extra):
            print(f"  [{slug}] perplexity: +{len(extra)} model(s) filled from {Path(fb_path).name}")
            primary = pd.concat([primary, extra], ignore_index=True)
    return primary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_slugs = sorted(set(DLL_DATASET_SPECS) | set(N400_DATASETS))

    frames = []
    for slug in all_slugs:
        print(f"=== {slug} ===")
        main_df = _load_main_models(slug)
        pythia_df = _load_checkpoints(slug, "pythia_checkpoints")
        babylm_df = _load_checkpoints(slug, "babylm_gpt2_checkpoints")
        ppl = _perplexity_lookup(slug)

        parts = [d for d in (main_df, pythia_df, babylm_df) if not d.empty]
        if not parts:
            print("  no data at all -- skipping")
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df.merge(ppl, on=["model_id", "revision"], how="left")

        ceiling = _ceiling(slug)
        df["ceiling"] = ceiling
        df["score_raw"] = df["dll_total"] / df["n_obs"]
        df["score_norm"] = df["score_raw"] / ceiling if ceiling else float("nan")

        n_ppl_missing = df["perplexity"].isna().sum()
        print(f"  {len(df)} rows ({main_df['model_id'].nunique() if not main_df.empty else 0} main models, "
              f"{pythia_df['revision'].nunique() if not pythia_df.empty else 0} pythia ckpts, "
              f"{babylm_df['revision'].nunique() if not babylm_df.empty else 0} babylm ckpts), "
              f"ceiling={'yes' if ceiling else 'NO (raw score only)'}, "
              f"{n_ppl_missing} row(s) missing perplexity")
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)
    master = master[[
        "dataset", "group", "model_id", "revision", "is_checkpoint", "checkpoint_family",
        "n_params", "perplexity", "dll_total", "n_obs", "score_raw", "ceiling", "score_norm",
    ]]
    master.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(master)} rows -> {OUT_PATH}")
    print(master.groupby(["dataset", "group"])["model_id"].nunique().unstack(fill_value=0))


if __name__ == "__main__":
    main()
