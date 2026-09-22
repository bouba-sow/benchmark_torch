#!/usr/bin/env python3
"""Paper figure: Pythia training-checkpoint trajectories across the 5
non-provo behavioral datasets (naturalstories_spr, ucl_reading, geco, meco,
zuco_reading -- provo is handled separately in
figures/figure10_provo_ohschuler.py / figures/figure11_provo_formula_ablation.py
since it also carries the Oh & Schuler formula ablation).

For each of the 10 Pythia sizes, traces its 11 training checkpoints (steps
0, 128, 256, 512, 2000, 8000, 16000, 32000, 64000, 128000, 143000 --
LPP_PYTHIA_STEPS) through delta log-likelihood, reading the "pythia_checkpoints"
registry family's raw sweep output (neuroai/models/registry.py). Reads the
BUG-FIXED sweep trees only (revision-namespaced working files -- see
neuroai/benchmarks/ucl_reading_lme.py / natural_stories_lme.py); the older
default sweep roots without the "_oos"/"_insample" suffix are known-corrupted
for this family (checkpoint-collision bug, fixed 2026-08-26) and are never
read here.

Two pages, each a single figure with a 4-row x 6-column grid (one column per
behavioral dataset; row B/D pack N400 EEG into columns 0-2 and fMRI --LPP,
Tuckute, Natural Stories-- into columns 3-5, since N400 alone only used half
the row's width):
  Row A. Final checkpoint (step 143000) only, score vs n_params (log-x) --
    the classic scaling-law curve, one point per size -- behavioral datasets.
  Row B. Same as row A, for N400 EEG (cols 0-2) and fMRI (cols 3-5).
  Row C. EVERY completed checkpoint, pooled score-vs-perplexity scatter
    (log-x) across all 10 sizes, star = fully trained (step 143000), dashed =
    pooled quadratic fit in log10(perplexity) -- matches Oh & Schuler (2023,
    arXiv:2304.11389) Fig. 1/2 middle panels' pooled-curve convention --
    behavioral datasets.
  Row D. Same as row C, for N400 EEG (cols 0-2) and fMRI (cols 3-5).

Page 1 -- out-of-sample (delta_loglik_test, from the standard *_oos sweep
  trees, frequency + spillover covariates retained -- this pipeline's default
  formula).
Page 2 -- in-sample (delta_loglik_train, from the standard *_insample sweep
  trees, fit AND evaluated on the full exploratory set, no train/test split --
  same formula as page 1, matching Oh & Schuler's in-sample convention but
  NOT their formula, which is provo's dedicated ablation instead).

Per-checkpoint perplexity (outputs_jeanzay/<dataset>_perplexity/
<dataset>_perplexity_summary.csv, with a "revision" column) now covers all 5
datasets (added 2026-08-26 via scripts/perplexity/pythia_checkpoints_3datasets.py
for naturalstories_spr/ucl_reading/zuco_reading; geco/meco already had it from
the earlier embedding-sweep era).

Output: reports/paper/pythia_checkpoint_scaling.pdf (2 pages: OOS, in-sample)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, REGISTRY_INDEX, REPO_ROOT
from neuroai.analysis.paper_style import despine, use_paper_style
from scripts.analysis.perplexity_matched_scaling import _best_row_insample
from scripts.figures.figure8_scale_matched_comparison import DLL_DATASET_SPECS, _best_row_per_word, _model_id_from_dirname

OUT_PATH = REPO_ROOT / "reports" / "paper" / "pythia_checkpoint_scaling.pdf"

DATASET_ORDER: list[str] = [
    "naturalstories_spr", "ucl_reading",  # SPR
    "geco", "meco", "provo", "zuco_reading",  # eye-tracking
]

FINAL_STEP_REVISION = "step143000"
_FINAL_REVISIONS = {"step143000", "main"}

# N400 EEG datasets -- added as a THIRD row below the 6-column behavioral
# grid (final-checkpoint-vs-size only, no perplexity/pooled-checkpoint panel
# B -- no per-checkpoint perplexity table exists for these three yet). Each
# dataset's own paper-exact LME formula (see neuroai/regressors/
# n400_pooled_lme.py::fit_n400_lme_one_layer_paper) was used for the
# pythia_checkpoints sweep, same as every other model in these datasets'
# post-formula-fix reruns. LAST LAYER ONLY was fit per checkpoint (see
# scripts/submit.py <dataset> surprisal fit --family pythia_checkpoints), so
# _best_row_per_word's "pick the max-delta_loglik_test row" is a no-op here
# (exactly one row per (model, revision)) -- reused anyway for consistency
# and defensiveness (e.g. against a stray duplicate row).
N400_DATASET_ORDER: list[str] = ["nieuwland"]
N400_SWEEP_ROOTS: dict[str, str] = {
    "nieuwland": "nieuwland/surprisal_sweep",
}

# fMRI datasets -- added as a further TWO rows (C: vs size, F: vs perplexity)
# below the N400 rows, same convention: each dataset's own OOS/in-sample LME
# sweep (delta_loglik_test / delta_loglik_train), pythia_checkpoints family,
# LAST LAYER ONLY per checkpoint. LPP and Tuckute fit ONE pooled LME per
# layer (same by_layer.csv shape as the N400 tables above -- reused directly
# by _load_fmri_dll_per_checkpoint); Natural Stories fMRI fits one LME PER
# fROI (not pooled), so its rows are summed dll/n_obs across fROIs first --
# same electrode-level aggregation convention as
# build_paper_figure8_scale_matched_comparison uses for the iEEG datasets.
FMRI_DATASET_ORDER: list[str] = ["lpp", "tuckute", "naturalstories_fmri"]
FMRI_SWEEP_ROOTS: dict[str, str] = {
    "lpp": "lpp/surprisal_sweep",
    "tuckute": "tuckute/surprisal_sweep",
    "naturalstories_fmri": "ns_fmri/surprisal_sweep",
}
FMRI_BY_LAYER_FILENAMES: dict[str, str] = {
    "lpp": "lpp_lme_by_layer.csv",
    "tuckute": "tuckute_lme_by_layer.csv",
}

# Fedorenko2016 / Podcast iEEG-ECoG -- added as a further row pair (E: vs
# size, F: vs perplexity) below the fMRI row, since N400 (3 cols) + fMRI
# (3 cols) already fill all 6 columns of the shared row. Same pooled-
# single-DV shape as LPP/Tuckute's by_layer.csv (one row per layer/lag,
# last_layer_only / mean-window-mode so effectively one row per (model,
# revision)) -- reuses _load_fmri_dll_per_checkpoint's non-
# naturalstories_fmri branch directly (its "layer"-vs-"lag" column-name
# difference doesn't matter -- the picker only reads delta_loglik_test/
# n_obs_test, not the index column's name).
EXTRA_IEEG_DATASET_ORDER: list[str] = ["fedorenko2016", "podcast"]
FMRI_SWEEP_ROOTS["fedorenko2016"] = "fedorenko2016/surprisal_sweep"
FMRI_BY_LAYER_FILENAMES["fedorenko2016"] = "fedorenko2016_lme_by_layer.csv"
FMRI_SWEEP_ROOTS["podcast"] = "podcast/surprisal_sweep"
FMRI_BY_LAYER_FILENAMES["podcast"] = "podcast_lme_by_lag.csv"

# Bug-fixed out-of-sample sweep trees (revision-namespaced working files).
# provo uses its "oldformula" ablation variant (see
# figures/figure11_provo_formula_ablation.py) -- that IS this pipeline's
# standard freq+spillover formula, just named for the ablation it was run
# alongside; provo has no plain "_oos"-suffixed tree of its own. The other 5
# datasets' plain "_oos" trees ARE the standard-formula rerun. The default
# (no-suffix) roots in DLL_DATASET_SPECS are NOT safe for this family for any
# of the 6 -- kept separate on purpose, see module docstring.
OOS_SWEEP_ROOTS: dict[str, str] = {
    "naturalstories_spr": "naturalstories/surprisal_sweep_checkpoints",
    "geco": "geco/surprisal_sweep_checkpoints/fp",
    "meco": "meco/surprisal_sweep_checkpoints/fp",
    "provo": "provo/surprisal_sweep_checkpoints/fp",
    "ucl_reading": "ucl_reading/surprisal_sweep_checkpoints",
    "zuco_reading": "zuco_reading/surprisal_sweep/nr",
}

# Bug-fixed in-sample sweep trees -- same caveat as OOS_SWEEP_ROOTS above.
# NOT analyze_perplexity_matched_scaling.INSAMPLE_SWEEP_ROOTS: that module's
# "provo" entry ("provo_checkpoint_sweep_lme_dlt_insample/fp") points at a
# directory that was never created once the ablation naming took over --
# defined locally here instead so provo resolves to the real, bug-fixed tree.
INSAMPLE_SWEEP_ROOTS: dict[str, str] = {
    "naturalstories_spr": "naturalstories/surprisal_sweep_insample",
    "geco": "geco/surprisal_sweep_insample/fp",
    "meco": "meco/surprisal_sweep_insample/fp",
    "provo": "provo/surprisal_sweep_checkpoints_insample/fp",
    "ucl_reading": "ucl_reading/surprisal_sweep_insample",
    "zuco_reading": "zuco_reading/surprisal_sweep_insample/nr",
}

# Per-checkpoint perplexity summary CSVs -- now available for all 6 datasets
# here (naturalstories_spr/ucl_reading/zuco_reading added 2026-08-26).
PERPLEXITY_PATHS: dict[str, str] = {
    "naturalstories_spr": "outputs_jeanzay/naturalstories/spr_perplexity/naturalstories_spr_perplexity_summary.csv",
    "geco": "outputs_jeanzay/geco/perplexity/geco_perplexity_summary.csv",
    "meco": "outputs_jeanzay/meco/perplexity/meco_perplexity_summary.csv",
    "provo": "outputs_jeanzay/provo/perplexity/provo_perplexity_summary.csv",
    "ucl_reading": "outputs_jeanzay/ucl_reading/perplexity/ucl_reading_perplexity_summary.csv",
    "zuco_reading": "outputs_jeanzay/zuco_reading/perplexity/zuco_reading_perplexity_summary.csv",
    # N400 EEG datasets -- computed via
    # scripts/perplexity/pythia_checkpoints_n400.py (same summary-csv
    # convention, so _load_perplexity below needs no N400-specific branch).
    "nieuwland": "outputs_jeanzay/nieuwland/perplexity/nieuwland_perplexity_summary.csv",
    # fMRI datasets -- computed "for free" alongside surprisal extraction
    # (see run/surprisal/neural/lpp_extract.py / run/surprisal/neural/tuckute_extract.py
    # / run/surprisal/neural/ns_fmri_extract.py module docstrings/_extract_perplexity
    # docstrings). naturalstories_fmri has its OWN dedicated per-checkpoint
    # perplexity table (added 2026-08-27) -- earlier this session it briefly
    # reused naturalstories_spr's instead, on the assumption that perplexity
    # (an aggregate scalar) would be roughly interchangeable across the two
    # datasets' 8/10-shared-story corpora; EMPIRICALLY WRONG (verified via a
    # direct test: pythia-14m@step143000 gives 602 on NS-fMRI's own 8-doc
    # corpus vs 1130 on SPR's 10-doc corpus, a 47% difference) -- the 2
    # non-shared stories and shorter corpus length shift the aggregate far
    # more than tokenization differences alone would. Do not reuse SPR's here.
    "lpp": "outputs_jeanzay/lpp/perplexity/lpp_perplexity_summary.csv",
    "tuckute": "outputs_jeanzay/tuckute/perplexity/tuckute_perplexity_summary.csv",
    "naturalstories_fmri": "outputs_jeanzay/ns_fmri/perplexity/ns_fmri_perplexity_summary.csv",
    "fedorenko2016": "outputs_jeanzay/fedorenko2016/perplexity/fedorenko2016_perplexity_summary.csv",
    "podcast": "outputs_jeanzay/podcast/lme_perplexity/podcast_perplexity_summary.csv",
}


def _load_perplexity(slug: str) -> dict[tuple[str, str], float]:
    rel_path = PERPLEXITY_PATHS.get(slug)
    if rel_path is None:
        return {}
    path = REPO_ROOT / rel_path
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "revision" not in df.columns:
        return {}
    df = df.dropna(subset=["model_id", "revision", "perplexity"])
    df = df.drop_duplicates(subset=["model_id", "revision"], keep="first")
    return {(r.model_id, r.revision): float(r.perplexity) for r in df.itertuples()}


def _load_dll_per_checkpoint_oos(slug: str) -> pd.DataFrame:
    """Out-of-sample analogue of analyze_perplexity_matched_scaling's
    _load_dll_per_checkpoint_insample, reading OOS_SWEEP_ROOTS (bug-fixed)
    instead of DLL_DATASET_SPECS' default (unsafe-for-this-family) root."""
    root = OOS_SWEEP_ROOTS.get(slug)
    spec = DLL_DATASET_SPECS.get(slug)
    if root is None or spec is None:
        return pd.DataFrame()

    sweep_root = REPO_ROOT / "outputs_jeanzay" / root
    fam_dir = sweep_root / "pythia_checkpoints"
    if not fam_dir.is_dir():
        return pd.DataFrame()
    rows = []
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_paths = [p for p in model_dir.glob(spec["pattern"]) if not p.name.endswith(".lock")]
        if not csv_paths:
            continue
        mid = _model_id_from_dirname(model_dir.name)
        if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
            continue
        df = pd.read_csv(csv_paths[0])
        if df.empty or "delta_loglik_test" not in df.columns or "revision" not in df.columns:
            continue
        meta = REGISTRY_INDEX[mid]
        for revision, rev_df in df.groupby("revision"):
            picked = _best_row_per_word(rev_df)
            if picked is None:
                continue
            dll_total, n_obs = picked
            rows.append(dict(
                model_id=mid, revision=revision, n_params=meta.n_params,
                dll_total=dll_total, n_obs=n_obs, score_raw=dll_total / n_obs,
            ))
    return pd.DataFrame(rows)


# naturalstories_spr's own dedicated (100%-full-data, no train/test split)
# in-sample sweep (naturalstories/surprisal_sweep_insample) was never
# finished -- only 4/10 sizes, a handful of checkpoints each (the
# *_parallel/_parallel_tail/_tail2 jobs stalled/were never completed). Rather
# than block the in-sample page on relaunching that sweep, fall back to the
# delta_loglik_train column already sitting in the (complete, 110/110)
# bug-fixed OOS tree -- the in-sample fit-and-score on that sweep's 75%
# train partition, same "OOS fit also reports the in-sample 75%-training-set
# delta-loglik, so no separate in-sample re-fit is needed" convention already
# used for the babylm_gpt2_checkpoints family on every other dataset (see e.g.
# scripts/submit.py (see docs/history/launcher_migration.csv)).
_INSAMPLE_FALLBACK_TO_OOS_TRAIN: set[str] = {"naturalstories_spr"}


def _load_dll_per_checkpoint_insample(slug: str) -> pd.DataFrame:
    """In-sample analogue of _load_dll_per_checkpoint_oos above, reading
    INSAMPLE_SWEEP_ROOTS (bug-fixed, and correct for provo unlike
    analyze_perplexity_matched_scaling's copy -- see that dict's docstring).

    For slugs in _INSAMPLE_FALLBACK_TO_OOS_TRAIN, reads OOS_SWEEP_ROOTS
    instead and picks delta_loglik_train/n_obs (75%-train-partition
    in-sample fit) -- see that set's comment above."""
    if slug in _INSAMPLE_FALLBACK_TO_OOS_TRAIN:
        root = OOS_SWEEP_ROOTS.get(slug)
    else:
        root = INSAMPLE_SWEEP_ROOTS.get(slug)
    spec = DLL_DATASET_SPECS.get(slug)
    if root is None or spec is None:
        return pd.DataFrame()

    sweep_root = REPO_ROOT / "outputs_jeanzay" / root
    fam_dir = sweep_root / "pythia_checkpoints"
    if not fam_dir.is_dir():
        return pd.DataFrame()
    rows = []
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_paths = [p for p in model_dir.glob(spec["pattern"]) if not p.name.endswith(".lock")]
        if not csv_paths:
            continue
        mid = _model_id_from_dirname(model_dir.name)
        if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
            continue
        df = pd.read_csv(csv_paths[0])
        if df.empty or "delta_loglik_train" not in df.columns or "revision" not in df.columns:
            continue
        meta = REGISTRY_INDEX[mid]
        for revision, rev_df in df.groupby("revision"):
            picked = _best_row_insample(rev_df)
            if picked is None:
                continue
            dll_total, n_obs = picked
            rows.append(dict(
                model_id=mid, revision=revision, n_params=meta.n_params,
                dll_total=dll_total, n_obs=n_obs, score_raw=dll_total / n_obs,
            ))
    return pd.DataFrame(rows)


def _load_n400_dll_per_checkpoint(slug: str, *, out_of_sample: bool) -> pd.DataFrame:
    """N400 analogue of _load_dll_per_checkpoint_oos/_insample above --
    reads outputs_jeanzay/<slug>_lme_sweep/pythia_checkpoints/<model>/
    <slug>_lme_by_layer.csv directly (no "fp"-style measure subfolder, no
    ceiling-normalisation column to merge in)."""
    root = N400_SWEEP_ROOTS.get(slug)
    if root is None:
        return pd.DataFrame()

    fam_dir = REPO_ROOT / "outputs_jeanzay" / root / "pythia_checkpoints"
    if not fam_dir.is_dir():
        return pd.DataFrame()
    dll_col = "delta_loglik_test" if out_of_sample else "delta_loglik_train"
    picker = _best_row_per_word if out_of_sample else _best_row_insample

    rows = []
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_path = model_dir / f"{slug}_lme_by_layer.csv"
        if not csv_path.is_file():
            continue
        mid = _model_id_from_dirname(model_dir.name)
        if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
            continue
        df = pd.read_csv(csv_path)
        if df.empty or dll_col not in df.columns or "revision" not in df.columns:
            continue
        meta = REGISTRY_INDEX[mid]
        for revision, rev_df in df.groupby("revision"):
            picked = picker(rev_df)
            if picked is None:
                continue
            dll_total, n_obs = picked
            rows.append(dict(
                model_id=mid, revision=revision, n_params=meta.n_params,
                dll_total=dll_total, n_obs=n_obs, score_raw=dll_total / n_obs,
            ))
    return pd.DataFrame(rows)


def _load_fmri_dll_per_checkpoint(slug: str, *, out_of_sample: bool, lang_only: bool = True) -> pd.DataFrame:
    """fMRI analogue of _load_n400_dll_per_checkpoint above. lpp/tuckute read
    a single pooled by_layer.csv per model (identical shape to the N400
    tables -- one row per layer, last_layer_only so effectively one row per
    (model, revision)); naturalstories_fmri instead reads by_roi.csv (one LME
    PER fROI, not pooled) and sums dll_total/n_obs across fROIs per
    (model, revision) -- same electrode-level aggregation convention as
    build_paper_figure8_scale_matched_comparison uses for the iEEG datasets.

    LPP (Fedorenko-parcel-averaged BOLD) and Tuckute (fixed ``LANG_ROIS``
    crossed random effect) are ALREADY language-network-only by construction
    -- ``lang_only`` only affects naturalstories_fmri, whose ``froi_table``
    default ("LANG_MD") pools 26 fROIs: 6 language-selective ("LANG" network)
    + 20 domain-general multiple-demand ("MD" network, Fedorenko et al.'s own
    localizer contrast, not language-selective). Summing dll/n_obs over all
    26 was diluting the language-selective signal with 20 ROIs that have no
    strong reason to show a surprisal effect at all -- ``lang_only=True``
    (default) restricts the sum to ``network == "LANG"`` fROIs only, reusing
    each (fROI, layer, revision)'s ALREADY-FIT LME row (the "network" column
    is already in ns_fmri_lme_by_roi.csv) rather than refitting anything."""
    root = FMRI_SWEEP_ROOTS.get(slug)
    if root is None:
        return pd.DataFrame()
    fam_dir = REPO_ROOT / "outputs_jeanzay" / root / "pythia_checkpoints"
    if not fam_dir.is_dir():
        return pd.DataFrame()
    dll_col = "delta_loglik_test" if out_of_sample else "delta_loglik_train"
    picker = _best_row_per_word if out_of_sample else _best_row_insample

    rows = []
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        mid = _model_id_from_dirname(model_dir.name)
        if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
            continue
        meta = REGISTRY_INDEX[mid]

        if slug == "naturalstories_fmri":
            csv_path = model_dir / "ns_fmri_lme_by_roi.csv"
            if not csv_path.is_file():
                continue
            df = pd.read_csv(csv_path)
            if df.empty or dll_col not in df.columns or "revision" not in df.columns:
                continue
            if lang_only and "network" in df.columns:
                df = df[df["network"] == "LANG"]
                if df.empty:
                    continue
            for revision, rev_df in df.groupby("revision"):
                total_dll, total_n = 0.0, 0.0
                for _, froi_df in rev_df.groupby("froi"):
                    picked = picker(froi_df)
                    if picked is None:
                        continue
                    total_dll += picked[0]
                    total_n += picked[1]
                if total_n == 0:
                    continue
                rows.append(dict(
                    model_id=mid, revision=revision, n_params=meta.n_params,
                    dll_total=total_dll, n_obs=total_n, score_raw=total_dll / total_n,
                ))
        else:
            csv_path = model_dir / FMRI_BY_LAYER_FILENAMES[slug]
            if not csv_path.is_file():
                continue
            df = pd.read_csv(csv_path)
            if df.empty or dll_col not in df.columns or "revision" not in df.columns:
                continue
            for revision, rev_df in df.groupby("revision"):
                picked = picker(rev_df)
                if picked is None:
                    continue
                dll_total, n_obs = picked
                rows.append(dict(
                    model_id=mid, revision=revision, n_params=meta.n_params,
                    dll_total=dll_total, n_obs=n_obs, score_raw=dll_total / n_obs,
                ))
    return pd.DataFrame(rows)


def _panel_a(ax, df: pd.DataFrame, sample_label: str = "in-sample") -> None:
    """Full single-panel version (title, axis labels) -- used standalone by
    figures/figure10_provo_ohschuler.py / figures/figure11_provo_formula_ablation.py.
    See _panel_a_grid for the compact multi-dataset-grid version used below."""
    final = df[df["revision"] == FINAL_STEP_REVISION].sort_values("n_params")
    if final.empty:
        ax.text(0.5, 0.5, "no final-checkpoint data", ha="center", va="center",
                 transform=ax.transAxes, fontsize=8, color="#999999")
        return
    ax.plot(final["n_params"], final["dll_total"], marker="o", ms=4, lw=1.3, color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xlabel("model size (parameters, log scale)", fontsize=8)
    ax.set_ylabel(f"delta log-lik ({sample_label}, total)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.98, "A. final checkpoint vs size", transform=ax.transAxes, fontsize=7.5,
             fontweight="bold", va="top", ha="left", color="#333333")


def _panel_b(ax, df: pd.DataFrame, ppl_map: dict[tuple[str, str], float], sample_label: str = "in-sample") -> None:
    """Full single-panel version -- see _panel_a's note."""
    rows = []
    for r in df.itertuples():
        ppl = ppl_map.get((r.model_id, r.revision))
        if ppl is None or ppl <= 0:
            continue
        rows.append(dict(model_id=r.model_id, n_params=r.n_params, perplexity=ppl,
                          dll_total=r.dll_total, is_final=r.revision in _FINAL_REVISIONS))
    merged = pd.DataFrame(rows)
    if merged.empty:
        ax.text(0.5, 0.5, "no per-checkpoint perplexity yet", ha="center", va="center",
                 transform=ax.transAxes, fontsize=8, color="#999999")
        ax.set_xlabel("perplexity (log scale)", fontsize=8)
        ax.set_ylabel(f"delta log-lik ({sample_label}, total, not per-word/ceiling-normalised)", fontsize=8)
        return

    sizes_sorted = sorted(merged["n_params"].unique())
    cmap = plt.get_cmap("viridis")
    colors = merged["n_params"].map(dict(zip(sizes_sorted, cmap(np.linspace(0.1, 0.9, len(sizes_sorted))))))

    non_final = merged[~merged["is_final"]]
    final = merged[merged["is_final"]]
    ax.scatter(non_final["perplexity"], non_final["dll_total"], s=12, c=colors[non_final.index],
               alpha=0.7, edgecolor="none", label="checkpoint")
    ax.scatter(final["perplexity"], final["dll_total"], s=55, c=colors[final.index], marker="*",
               edgecolor="#333333", linewidth=0.4, zorder=5, label="fully trained (step 143000)")

    if len(merged) >= 4:
        logp = np.log10(merged["perplexity"].to_numpy())
        y = merged["dll_total"].to_numpy()
        coefs = np.polyfit(logp, y, deg=2)
        grid = np.linspace(logp.min(), logp.max(), 100)
        ax.plot(10 ** grid, np.polyval(coefs, grid), color="#333333", lw=1.3, linestyle="--",
                zorder=4, label="pooled quadratic fit")

    ax.set_xscale("log")
    ax.set_xlabel("perplexity (log scale)", fontsize=8)
    ax.set_ylabel(f"delta log-lik ({sample_label}, total, not per-word/ceiling-normalised)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="best")
    ax.text(0.02, 0.98, "B. every model x checkpoint, pooled (star = fully trained)",
             transform=ax.transAxes, fontsize=7, fontweight="bold", va="top", ha="left", color="#333333")


# ---------------------------------------------------------------------------
# Paper-style grid (matches figures/figure_babylm_checkpoint_scaling.py's
# _new_fig/_decorate exactly -- lettered left-aligned titles, despined axes,
# tight constrained_layout, no row-legend text/fig.suptitle). Pythia's ONE
# addition babylm didn't need: the perplexity pages color each point by
# Pythia SIZE (9-10 of them, vs. babylm's single model), so those two pages
# get exactly one shared bottom legend instead of per-subplot ones.
# ---------------------------------------------------------------------------

_NCOL = 5
_LETTERS = "abcdefghijklmnopqrst"
_FIT_C = "#333333"

# slug -> (short display name, group). Group determines line/marker color on
# the "vs size" pages (perplexity pages color by Pythia size instead).
_DATASET_META: dict[str, tuple[str, str]] = {
    "naturalstories_spr": ("Natural Stories · SPR", "SPR"),
    "ucl_reading": ("UCL Reading · SPR", "SPR"),
    "geco": ("GECO · first-pass", "eye"),
    "meco": ("MECO · first-pass", "eye"),
    "provo": ("Provo · first-pass", "eye"),
    "zuco_reading": ("ZuCo · TRT", "eye"),
    "nieuwland": ("Nieuwland · N400", "N400"),
    "lpp": ("Le Petit Prince · fMRI", "fMRI"),
    "tuckute": ("Tuckute · fMRI", "fMRI"),
    "naturalstories_fmri": ("Natural Stories · fMRI", "fMRI"),
    "fedorenko2016": ("Fedorenko 2016 · ECoG", "iEEG"),
    "podcast": ("Podcast · ECoG 0–200 ms", "iEEG"),
}
GROUP_COLOR = {"SPR": "#4C72B0", "eye": "#55A868", "N400": "#C44E52",
               "fMRI": "#8172B2", "iEEG": "#CCB974"}


def _format_n_params(n: float) -> str:
    if n >= 1e9:
        v = n / 1e9
        return f"{v:.0f}B" if v == int(v) else f"{v:.1f}B"
    v = n / 1e6
    return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"


def _all_dataset_specs(loader):
    """Flat, row-major list of every dataset panel: (slug, per-dataset
    loader). Order: behavioral, N400, fMRI, iEEG -- 14 datasets, tiled into
    a 5-col x 3-row grid (1 blank cell)."""
    specs = [(slug, loader) for slug in DATASET_ORDER]
    specs += [(slug, _load_n400_dll_per_checkpoint) for slug in N400_DATASET_ORDER]
    specs += [(slug, _load_fmri_dll_per_checkpoint) for slug in FMRI_DATASET_ORDER]
    specs += [(slug, _load_fmri_dll_per_checkpoint) for slug in EXTRA_IEEG_DATASET_ORDER]
    return specs


def _new_fig(n_panels: int):
    """3-row x 5-col paper grid, tight spacing, no figure title -- identical
    layout to figures/figure_babylm_checkpoint_scaling.py's _new_fig."""
    nrow = -(-n_panels // _NCOL)  # ceil
    fig, axes = plt.subplots(nrow, _NCOL, figsize=(13.4, 2.55 * nrow + 0.35),
                              constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.010, h_pad=0.020, wspace=0.012, hspace=0.030)
    axes = axes.ravel()
    for ax in axes[n_panels:]:
        ax.remove()
    return fig, axes


def _decorate(ax, i: int, n_panels: int, slug: str, xlabel: str, ylabel: str) -> None:
    col = i % _NCOL
    is_bottom = (i + _NCOL) >= n_panels
    short, _group = _DATASET_META[slug]
    ax.set_title(f"({_LETTERS[i]}) {short}", loc="left", fontsize=7.6, color="#222222", pad=3.5)
    if col == 0:
        ax.set_ylabel(ylabel, fontsize=8)
    if is_bottom:
        ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=6.4, length=2.5, pad=1.5)
    ax.margins(x=0.04)
    despine(ax)


def _make_size_page(pdf: PdfPages, loader, sample_label: str, *, out_of_sample: bool) -> None:
    """Score (final checkpoint only) vs model size -- one line per dataset,
    colored by dataset group (matches babylm's trajectory-page styling; no
    legend needed here, only one series per panel)."""
    specs = _all_dataset_specs(loader)
    fig, axes = _new_fig(len(specs))
    for i, (slug, row_loader) in enumerate(specs):
        ax = axes[i]
        df = row_loader(slug) if row_loader is loader else row_loader(slug, out_of_sample=out_of_sample)
        _group = _DATASET_META[slug][1]
        c = GROUP_COLOR[_group]
        final = df[df["revision"] == FINAL_STEP_REVISION].sort_values("n_params") if not df.empty else df
        if not final.empty:
            ax.plot(final["n_params"], final["dll_total"], "-o", ms=3.2,
                    lw=1.1, color=c, mfc=c, mec="none")
            ax.set_xscale("log")
        _decorate(ax, i, len(specs), slug, "model size (params, log)", f"ΔLog-lik ({sample_label}, final ckpt)")
        n_models = df["model_id"].nunique() if not df.empty else 0
        print(f"[{sample_label}/size/{slug}] {n_models} models, {len(df)} points")
    pdf.savefig(fig)
    plt.close(fig)


def _make_perplexity_page(pdf: PdfPages, loader, sample_label: str, *, out_of_sample: bool) -> None:
    """Score (every checkpoint) vs corpus perplexity -- colored by Pythia
    SIZE (9-10 of them), one shared legend for the whole page instead of a
    legend per subplot."""
    specs = _all_dataset_specs(loader)

    all_n_params: set[float] = set()
    dfs: dict[str, pd.DataFrame] = {}
    for slug, row_loader in specs:
        df = row_loader(slug) if row_loader is loader else row_loader(slug, out_of_sample=out_of_sample)
        dfs[slug] = df
        if not df.empty:
            all_n_params.update(df["n_params"].unique().tolist())
    sizes_sorted = sorted(all_n_params)
    cmap = plt.get_cmap("viridis")
    color_map = dict(zip(sizes_sorted, cmap(np.linspace(0.1, 0.9, len(sizes_sorted)))))

    fig, axes = _new_fig(len(specs))
    for i, (slug, _row_loader) in enumerate(specs):
        ax = axes[i]
        df = dfs[slug]
        ppl_map = _load_perplexity(slug)
        rows = []
        for r in df.itertuples():
            ppl = ppl_map.get((r.model_id, r.revision))
            if ppl is None or ppl <= 0:
                continue
            rows.append(dict(n_params=r.n_params, perplexity=ppl, dll_total=r.dll_total,
                              is_final=r.revision in _FINAL_REVISIONS))
        merged = pd.DataFrame(rows)
        if not merged.empty:
            colors = merged["n_params"].map(color_map)
            non_final = merged[~merged["is_final"]]
            final = merged[merged["is_final"]]
            ax.scatter(non_final["perplexity"], non_final["dll_total"], s=9,
                       c=colors[non_final.index], edgecolors="white", linewidths=0.3, zorder=3)
            ax.scatter(final["perplexity"], final["dll_total"], s=55, c=colors[final.index],
                       marker="*", edgecolors="white", linewidths=0.4, zorder=5)
            if len(merged) >= 6:
                logp = np.log10(merged["perplexity"].to_numpy())
                y = merged["dll_total"].to_numpy()
                coefs = np.polyfit(logp, y, deg=2)
                grid = np.linspace(logp.min(), logp.max(), 100)
                ax.plot(10 ** grid, np.polyval(coefs, grid), "--", color=_FIT_C, lw=0.9, zorder=4)
            ax.set_xscale("log")
        _decorate(ax, i, len(specs), slug, "corpus perplexity (log)", f"ΔLog-lik ({sample_label}, every ckpt)")
        n_models = df["model_id"].nunique() if not df.empty else 0
        print(f"[{sample_label}/perplexity/{slug}] {n_models} models, {len(df)} points")

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color_map[n],
                markersize=6, label=_format_n_params(n))
        for n in sizes_sorted
    ]
    handles.append(Line2D([0], [0], marker="*", linestyle="none", color="#666666",
                            markersize=9, markeredgecolor="#333333", label="fully trained"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               fontsize=7.5, frameon=False)

    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    use_paper_style()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUT_PATH) as pdf:
        _make_size_page(pdf, _load_dll_per_checkpoint_oos, "out-of-sample", out_of_sample=True)
        _make_perplexity_page(pdf, _load_dll_per_checkpoint_oos, "out-of-sample", out_of_sample=True)
        _make_size_page(pdf, _load_dll_per_checkpoint_insample, "in-sample", out_of_sample=False)
        _make_perplexity_page(pdf, _load_dll_per_checkpoint_insample, "in-sample", out_of_sample=False)

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
