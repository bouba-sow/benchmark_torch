#!/usr/bin/env python3
"""Paper figure: scale-matched model-group comparison.

Controls for the parameter-count confound between model groups by comparing
mean normalised score (score / noise ceiling) only within narrow parameter
windows where two or more groups actually overlap:

  Window B (~10-50M params):  Standard LLM vs Memory-Constrained vs
                               Constrained-BabyLM vs BabyLM.
  Window A (~85-165M params): Standard LLM vs BabyLM only (no
                               Memory-Constrained / Constrained-BabyLM model
                               is that large).

Produced separately per LLM-signal feature mode, since the two are scored
with different metrics and are NOT directly comparable:
  - "surprisal": scalar token-level surprisal -> LME delta-R (behavioral
    datasets' own primary metric; neural datasets' "__surprisal" variant).
    NOTE: pending the delta-loglik noise ceiling replacing this delta-R
    metric everywhere (see "dll" below), this mode's normalisation is
    known-noisy -- treat as provisional.
  - "embedding": hidden-state ridge encoding (neural datasets' own primary
    metric; behavioral datasets' "__embed" variant).
  - "dll": per-word (or per-observation) delta log-likelihood (full LME minus
    baseline-only LME, out-of-sample), normalised by each dataset's own
    split-half delta-loglik noise ceiling (Timkey et al. 2026 method, with
    ECoG/fMRI-adapted variants; see scripts/ceilings/dll_behavioral.py
    and outputs/noise_ceilings/delta_loglik_ceiling_<tag>_oos.json). This is
    the more principled replacement for "surprisal"/"embedding" above,
    computed from each model's RAW per-layer (or per-lag, or per-fROI)
    sweep output under outputs_jeanzay/ (NOT the outputs/ pre-aggregated
    summary trees, which are stale/incomplete locally for several datasets;
    see DLL_DATASET_SPECS for the exact raw source + filename pattern per
    dataset). The best layer/lag per model is taken by delta_loglik_test,
    matching the "peak-layer" convention used for every other metric in the
    report; ns_fmri additionally aggregates per-fROI (each fROI keeps its own
    best layer, matching how its "mean_delta_loglik_test" summary column is
    built), summing delta_loglik_test and n_obs_test across fROIs before
    dividing. "*_corrupted_yhat_full_backup" measure-subdir siblings
    (present for provo, meco, zuco_reading) are known-bad backups and are
    skipped. Covers all 6 behavioral datasets + 5 neural datasets
    (fedorenko2016_ecog, lpp_fmri, tuckute_fmri, podcast_ieeg, ns_fmri).
    Datasets without a delta-loglik ceiling (e.g. broderick_eeg) are skipped.

Layout: one page per (feature mode, window) combination. Each page is a
single figure with a 5-column subplot grid (average-across-datasets panel
first, then one subplot per dataset). Window A pages show only the 2 groups
that window covers (LLM, BabyLM); Window B pages show all 4. No descriptive
titles inside panels -- just the dataset's short slug as an axis label,
matching the "A./B."-only, caption-in-the-paper convention used elsewhere in
neuroai/analysis/paper_style.py.

Final two pages ("perplexity/A", "perplexity/B") show each window's groups'
mean corpus perplexity (log scale) -- context for whether a group's
score/ceiling differences track how good the models are as language models
per se. Per model, perplexity is the MEDIAN across every available
per-dataset perplexity.json under outputs_jeanzay/ (lpp/naturalstories/geco/
meco/provo/ucl_reading/zuco_reading/podcast _perplexity dirs, plus
fedorenko2016/embedding_sweep's and tuckute's own per-model perplexity.json)
-- median rather than mean because a couple of these sources are corpora
shared with the Pythia/OLMo2 *checkpoint* sweep and include early
(near-random, astronomically-high-perplexity) training steps alongside the
final one in the same directory tree; those step subdirectories are excluded
by name, but median is a second layer of robustness against any other
one-off broken corpus fit. No perplexity source at all exists for
fedorenko2016_ecog/tuckute's own LME sweep corpora beyond what's already
folded in above, and no separate one exists for ns_fmri, but this is a
single global per-model number pooled across whatever corpora are available
-- not tied 1:1 to the 11 dll datasets -- so that gap doesn't reduce coverage
here.

Output: reports/paper/scale_matched_comparison.pdf
  (surprisal/A, surprisal/B, embedding/A, embedding/B, dll/A [11 datasets],
   dll/B [11 datasets], perplexity/A, perplexity/B)
"""
from __future__ import annotations

import json
import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from neuroai.analysis.benchmark_common import (
    DATASET_FAMILY,
    DATASET_FAMILY_ORDER,
    DATASETS_BY_SLUG,
    EXCLUDED_MODEL_IDS,
    filter_degenerate_dll_rows,
    sum_dll_across_units,
    GROUP_COLORS,
    GROUP_DISPLAY,
    GROUP_ORDER,
    REGISTRY_INDEX,
    REPO_ROOT,
    read_lme_by_layer_csv,
)
from neuroai.analysis.paper_style import use_paper_style

SUMMARY_DIR = REPO_ROOT / "outputs" / "benchmark_summary"
NOISE_CEILING_DIR = REPO_ROOT / "outputs" / "noise_ceilings"
OUT_DIR = REPO_ROOT / "reports" / "paper"
OUT_PATH = OUT_DIR / "scale_matched_comparison.pdf"

# Delta-loglik LME sweep specs: RAW per-model trees under outputs_jeanzay/
# (not the outputs/ pre-aggregated summaries -- see module docstring).
#   root: sweep directory relative to outputs_jeanzay/ (family dirs live
#     directly under this; geco/meco/provo/zuco_reading/podcast have an extra
#     measure-name level -- a "*_corrupted_yhat_full_backup" sibling of that
#     level is a known-bad backup and must NOT be read).
#   pattern: filename glob (relative to each model dir) carrying
#     delta_loglik_test + n_obs_test. Behavioral files have one row per layer
#     (all main-checkpoint / step 0); neural "by_layer"/"by_lag" files are the
#     same shape (single pooled LME across electrodes/voxels, just layer- or
#     lag-indexed instead).
#   ceiling_tag: suffix of delta_loglik_ceiling_<tag>_oos.json (differs from
#     the dataset slug for lpp_fmri and podcast_ieeg).
#   group_col: only for ns_fmri -- per-fROI aggregation (see docstring).
DLL_DATASET_SPECS: dict[str, dict] = {
    "naturalstories_spr": dict(root="naturalstories/surprisal_sweep",
                                pattern="*_lme_checkpoint_scores.csv", ceiling_tag="naturalstories_spr"),
    "geco": dict(root="geco/surprisal_sweep/fp",
                 pattern="*_lme_checkpoint_scores.csv", ceiling_tag="geco"),
    "meco": dict(root="meco/surprisal_sweep/fp",
                 pattern="*_lme_checkpoint_scores.csv", ceiling_tag="meco"),
    "provo": dict(root="provo/surprisal_sweep/fp",
                  pattern="*_lme_checkpoint_scores.csv", ceiling_tag="provo"),
    "ucl_reading": dict(root="ucl_reading/surprisal_sweep",
                         pattern="*_lme_checkpoint_scores.csv", ceiling_tag="ucl_reading"),
    # Root was "zuco_reading/checkpoint_sweep_lme_dlt/nr" (the pre-refit,
    # 50/50-split tree) until 2026-09-09 -- fixed to match its SR/TSR
    # siblings below and the corrected 75/25 OOS resweep (word predictions
    # backfilled 2026-09-08) that every other current DLL-track figure in
    # this report already uses (dlt/delta_loglik.py's DATASET_SPEC,
    # neuroai.analysis.dlt_difficulty.py's caches, etc.) -- this was the one
    # remaining consumer still pointed at the stale root (found while
    # auditing family_scaling_lastlayer.pdf for a "good normalization"
    # confirmation request).
    "zuco_reading": dict(root="zuco_reading/surprisal_sweep/nr",
                          pattern="*_lme_checkpoint_scores.csv", ceiling_tag="zuco_reading"),
    # SR/TSR -- added 2026-09-08, data landed after NR was first wired in.
    # Root uses the "_dlt_oos" canonical tree (see
    # summarize/benchmark.py's load_zuco_reading_sr/tsr docstrings).
    "zuco_reading_sr": dict(root="zuco_reading/surprisal_sweep/sr",
                             pattern="*_lme_checkpoint_scores.csv", ceiling_tag="zuco_reading_sr"),
    "zuco_reading_tsr": dict(root="zuco_reading/surprisal_sweep/tsr",
                              pattern="*_lme_checkpoint_scores.csv", ceiling_tag="zuco_reading_tsr"),
    "fedorenko2016_ecog": dict(root="fedorenko2016/surprisal_sweep",
                                pattern="*_lme_by_layer.csv", ceiling_tag="fedorenko2016_ecog"),
    "lpp_fmri": dict(root="lpp/surprisal_sweep",
                      pattern="*_lme_by_layer.csv", ceiling_tag="lpp_fmri_persubject"),
    "tuckute_fmri": dict(root="tuckute/surprisal_sweep",
                          pattern="*_lme_by_layer.csv", ceiling_tag="tuckute_fmri"),
    "podcast_ieeg": dict(root="podcast/surprisal_sweep",
                          pattern="*_lme_by_lag.csv", ceiling_tag="podcast_ecog"),
    "ns_fmri": dict(root="ns_fmri/surprisal_sweep",
                     pattern="*_lme_by_roi.csv", ceiling_tag="ns_fmri_fmri", group_col="froi"),
    # Added 2026-09-03: OneStop (3 eye-tracking measures) and Pereira fMRI --
    # not yet in DATASET_FAMILY/DATASET_ORDER below (figure8's own dataset
    # grid), so adding these entries doesn't change that figure; they ARE
    # picked up by analysis/matched_scale_master_table.py's generic "any slug
    # in DLL_DATASET_SPECS" fallback. Neither has a pythia_checkpoints
    # trajectory yet (only babylm_gpt2_checkpoints for pereira) -- see that
    # script's docstring -- so both currently only feed the regression's
    # size-matched terms, not its perplexity terms.
    "onestop_fp": dict(root="onestop/surprisal_sweep/fp",
                        pattern="*_lme_checkpoint_scores.csv", ceiling_tag="onestop"),
    "onestop_gp": dict(root="onestop/surprisal_sweep/gp",
                        pattern="*_lme_checkpoint_scores.csv", ceiling_tag="onestop_gp"),
    "onestop_sp": dict(root="onestop/surprisal_sweep/sp",
                        pattern="*_lme_checkpoint_scores.csv", ceiling_tag="onestop_sp"),
    "pereira_fmri": dict(root="pereira/surprisal_sweep",
                          pattern="*_lme_by_layer.csv", ceiling_tag="pereira_fmri"),
    # Added 2026-09-04: GECO/MECO/Provo's GP (go-past) and MECO/Provo's SP
    # (scan-path/total dwell) measures -- main-model data already existed
    # under a "gp"/"sp" sibling subfolder of each dataset's default root
    # (same convention as onestop_gp/sp above), just never had a
    # DLL_DATASET_SPECS entry pointing at it. The plain "geco"/"meco"/"provo"
    # entries above are implicitly FP (their root's "fp" subfolder) and are
    # left as-is rather than renamed, to avoid touching every existing
    # figure/script that already keys off those three slugs.
    "geco_gp": dict(root="geco/surprisal_sweep/gp",
                     pattern="*_lme_checkpoint_scores.csv", ceiling_tag="geco_gp"),
    "meco_gp": dict(root="meco/surprisal_sweep/gp",
                     pattern="*_lme_checkpoint_scores.csv", ceiling_tag="meco_gp"),
    "meco_sp": dict(root="meco/surprisal_sweep/sp",
                     pattern="*_lme_checkpoint_scores.csv", ceiling_tag="meco_sp"),
    "provo_gp": dict(root="provo/surprisal_sweep/gp",
                      pattern="*_lme_checkpoint_scores.csv", ceiling_tag="provo_gp"),
    "provo_sp": dict(root="provo/surprisal_sweep/sp",
                      pattern="*_lme_checkpoint_scores.csv", ceiling_tag="provo_sp"),
}

WINDOW_B = (10e6, 50e6)    # all 4 groups
WINDOW_A = (85e6, 165e6)   # LLM + BabyLM only

# Window B, LLM group only: restrict to Pythia-31M, dropping Pythia-14M.
# Rationale: BabyLM's only Window-B models (GPTBERT-10M-*) are all 33.0M --
# more than 2x Pythia-14M, so keeping both Pythia points biased the LLM mean
# down against a group that sits uniformly near the window's upper end, and
# n=2 made that mean's SEM unstable besides. Pythia-31M (31M) is the closest
# size match available in the registry to BabyLM's 33M, so it alone is kept.
WINDOW_B_GROUP_OVERRIDE: dict[str, tuple[float, float]] = {
    "LLM": (25e6, 50e6),
}

# Canonical dataset order (fixed scientific ordering, matches the rest of the
# report). broderick_eeg excluded -- see neuroai/analysis/benchmark_common.py notes (incomplete sweep).
DATASET_ORDER: list[str] = [
    slug for fam in DATASET_FAMILY_ORDER for slug in DATASET_FAMILY
    if DATASET_FAMILY[slug] == fam and slug in DATASETS_BY_SLUG  # DATASET_FAMILY may list datasets that are not registered
]


def _csv_slug(base_slug: str, mode: str) -> str:
    """Map a canonical dataset slug to the all_models_scores.csv slug that
    carries the requested feature mode, given each dataset's own default mode."""
    default_mode = DATASETS_BY_SLUG[base_slug].feature_mode
    if default_mode == mode:
        return base_slug
    suffix = "__surprisal" if mode == "surprisal" else "__embed"
    return base_slug + suffix


def _load(slug: str) -> pd.DataFrame:
    path = SUMMARY_DIR / slug / "all_models_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["n_params"] = pd.to_numeric(df["n_params"], errors="coerce")
    df["score_norm"] = pd.to_numeric(df["score_norm"], errors="coerce")
    return df.dropna(subset=["n_params", "score_norm", "group"])


def _model_id_from_dirname(dirname: str) -> str:
    """``org_repo`` sweep directory name -> ``org/repo`` HF model id (first
    underscore only, matching neuroai.analysis.benchmark_common.model_dir_to_model_id)."""
    return dirname.replace("_", "/", 1)


def _best_row_per_word(per_layer: pd.DataFrame) -> tuple[float, float] | None:
    """(delta_loglik_test, n_obs_test) at the row with max delta_loglik_test.
    None if the frame is empty or every row errored (delta_loglik_test all NaN
    -- e.g. a checkpoint whose LME fit failed to converge)."""
    if per_layer.empty or "delta_loglik_test" not in per_layer.columns:
        return None
    valid = per_layer.dropna(subset=["delta_loglik_test", "n_obs_test"])
    if valid.empty:
        return None
    valid = filter_degenerate_dll_rows(valid)
    if valid.empty:
        return None
    best = valid.loc[valid["delta_loglik_test"].idxmax()]
    return float(best["delta_loglik_test"]), float(best["n_obs_test"])


def _load_dll(slug: str) -> pd.DataFrame:
    """Per-model delta-loglik-per-word (or per-observation) / noise-ceiling
    for one dataset, built from its RAW sweep output under outputs_jeanzay/
    -- the complete, current source (see module docstring). Picks the best
    layer/lag per model by delta_loglik_test, matching the "peak-layer"
    convention used for every other metric in the report; for ns_fmri
    (group_col="froi"), each fROI keeps its own best layer and
    delta_loglik_test/n_obs_test are summed across fROIs before dividing."""
    spec = DLL_DATASET_SPECS.get(slug)
    if spec is None:
        return pd.DataFrame()
    ceiling_path = NOISE_CEILING_DIR / f"delta_loglik_ceiling_{spec['ceiling_tag']}_oos.json"
    if not ceiling_path.exists():
        return pd.DataFrame()
    with open(ceiling_path) as f:
        ceiling = json.load(f)["ceiling_delta_loglik_per_obs"]

    sweep_root = REPO_ROOT / "outputs_jeanzay" / spec["root"]
    group_col = spec.get("group_col")
    rows = []
    for fam_dir in sorted(sweep_root.iterdir()) if sweep_root.is_dir() else []:
        if not fam_dir.is_dir():
            continue
        for model_dir in sorted(fam_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            csv_paths = [p for p in model_dir.glob(spec["pattern"]) if not p.name.endswith(".lock")]
            if not csv_paths:
                continue
            mid = _model_id_from_dirname(model_dir.name)
            if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
                continue
            df = read_lme_by_layer_csv(csv_paths[0])
            if df.empty or "delta_loglik_test" not in df.columns:
                continue

            if group_col:
                per_unit = []
                for _, unit_df in df.groupby(group_col):
                    picked = _best_row_per_word(unit_df)
                    if picked is None:
                        continue
                    per_unit.append(picked)
                total_dll, total_n = sum_dll_across_units(per_unit)
                if total_n == 0:
                    continue
                per_word = total_dll / total_n
            else:
                picked = _best_row_per_word(df)
                if picked is None:
                    continue
                per_word = picked[0] / picked[1]

            meta = REGISTRY_INDEX[mid]
            rows.append(dict(group=meta.group, n_params=meta.n_params, score_norm=per_word / ceiling))
    return pd.DataFrame(rows)


# Per-dataset perplexity.json roots (see module docstring for what's pooled
# and why median-across-sources is used). Two directory shapes:
#   root/<model_dir>/perplexity.json                    (most _perplexity dirs)
#   root/<family>/<model_dir>/perplexity.json            (checkpoint-sweep dirs)
# rglob("perplexity.json") handles both; "stepN" subdirs (Pythia/OLMo2
# checkpoint-sweep intermediate steps living inside the same tree) are
# excluded by parent-dirname so only each model's own single/main-checkpoint
# perplexity is kept.
PERPLEXITY_ROOTS: tuple[str, ...] = (
    "lpp/perplexity", "naturalstories/perplexity", "geco/perplexity", "meco/perplexity",
    "provo/perplexity", "ucl_reading/perplexity", "zuco_reading/perplexity",
    "podcast/perplexity",
    "fedorenko2016/embedding_sweep", "tuckute",
)


def _load_perplexity() -> pd.DataFrame:
    """One row per registry model: group, n_params, and its MEDIAN perplexity
    across every available corpus (see PERPLEXITY_ROOTS)."""
    per_model: dict[str, list[float]] = {}
    for root in PERPLEXITY_ROOTS:
        base = REPO_ROOT / "outputs_jeanzay" / root
        if not base.is_dir():
            continue
        for p in base.rglob("perplexity.json"):
            if p.parent.name.lower().startswith("step"):
                continue
            try:
                d = json.load(open(p))
            except (json.JSONDecodeError, OSError):
                continue
            mid, ppl = d.get("model_id"), d.get("perplexity")
            if mid is None or ppl is None:
                continue
            per_model.setdefault(mid, []).append(float(ppl))

    rows = []
    for mid, vals in per_model.items():
        if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
            continue
        meta = REGISTRY_INDEX[mid]
        rows.append(dict(group=meta.group, n_params=meta.n_params, perplexity=float(np.median(vals))))
    return pd.DataFrame(rows)


def _window_stats(
    df: pd.DataFrame, lo: float, hi: float,
    group_override: dict[str, tuple[float, float]] | None = None,
    value_col: str = "score_norm",
) -> dict[str, tuple[float, float, int]]:
    """group -> (mean, sem, n) within [lo, hi] params (or the group's own
    override range, if given one in ``group_override``)."""
    out = {}
    for g in GROUP_ORDER:
        g_lo, g_hi = (group_override or {}).get(g, (lo, hi))
        sub = df[(df["n_params"] >= g_lo) & (df["n_params"] <= g_hi) & (df["group"] == g)]
        vals = sub[value_col].to_numpy()
        if len(vals) == 0:
            continue
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        out[g] = (float(vals.mean()), float(sem), int(len(vals)))
    return out


def _draw(ax, stats: dict, label: str, ylabel: str = "score/ceiling", log_scale: bool = False) -> None:
    groups = [g for g in GROUP_ORDER if g in stats]
    x = np.arange(len(groups))
    for i, g in enumerate(groups):
        m, sem, _ = stats[g]
        ax.bar(i, m, width=0.6, yerr=sem, color=GROUP_COLORS[g], capsize=3,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_DISPLAY[g] for g in groups], fontsize=6.5, rotation=20, ha="right")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel(ylabel, fontsize=7)
    if log_scale:
        ax.set_yscale("log")
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=7,
             fontweight="bold", va="top", ha="left", color="#333333")
    if not groups:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                 fontsize=7, color="#999999")


def _page(
    pdf: PdfPages, mode: str, window_label: str, lo: float, hi: float,
    group_override: dict[str, tuple[float, float]] | None = None,
) -> None:
    per_dataset: dict[str, pd.DataFrame] = {}
    if mode == "dll":
        dll_order = [s for s in DATASET_ORDER if s in DLL_DATASET_SPECS] + \
                    [s for s in DLL_DATASET_SPECS if s not in DATASET_ORDER]
        for slug in dll_order:
            df = _load_dll(slug)
            if not df.empty:
                per_dataset[slug] = df
    else:
        for slug in DATASET_ORDER:
            df = _load(_csv_slug(slug, mode))
            if not df.empty:
                per_dataset[slug] = df

    panels = ["__AVERAGE__"] + list(per_dataset.keys())
    ncols = 5
    nrows = ceil(len(panels) / ncols)
    fig = plt.figure(figsize=(2.6 * ncols, 2.5 * nrows + 0.7))
    axes = fig.subplots(nrows, ncols)
    axes = np.atleast_2d(axes)

    # Average-across-datasets panel: mean-of-dataset-means, SEM across datasets.
    avg: dict[str, list[float]] = {g: [] for g in GROUP_ORDER}
    for slug, df in per_dataset.items():
        for g, (m, _, _) in _window_stats(df, lo, hi, group_override).items():
            avg[g].append(m)

    def _agg(d: dict[str, list[float]]) -> dict[str, tuple[float, float, int]]:
        out = {}
        for g, vals in d.items():
            if not vals:
                continue
            arr = np.asarray(vals)
            sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            out[g] = (float(arr.mean()), float(sem), len(arr))
        return out

    for i, panel in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        if panel == "__AVERAGE__":
            _draw(ax, _agg(avg), "Average")
        else:
            _draw(ax, _window_stats(per_dataset[panel], lo, hi, group_override), panel)

    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    legend_handles = [Patch(facecolor=GROUP_COLORS[g], label=GROUP_DISPLAY[g]) for g in GROUP_ORDER]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    caption = f"{mode} -- window {window_label} (~{lo/1e6:.0f}-{hi/1e6:.0f}M params)"
    if group_override:
        overrides = ", ".join(
            f"{GROUP_DISPLAY[g]} restricted to ~{go_lo/1e6:.0f}-{go_hi/1e6:.0f}M"
            for g, (go_lo, go_hi) in group_override.items()
        )
        caption += f" [{overrides}]"
    fig.text(0.01, 0.995, caption, fontsize=9, color="#666666", ha="left", va="top")
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    pdf.savefig(fig)
    plt.close(fig)


def _perplexity_page(
    pdf: PdfPages, ppl_df: pd.DataFrame, window_label: str, lo: float, hi: float,
    group_override: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Single-panel page: mean (log-scale) corpus perplexity per group,
    within the same param window used for the score panels."""
    stats = _window_stats(ppl_df, lo, hi, group_override, value_col="perplexity")
    fig, ax = plt.subplots(figsize=(4.0, 4.2))
    _draw(ax, stats, "Perplexity", ylabel="median corpus perplexity (log scale)", log_scale=True)

    legend_handles = [Patch(facecolor=GROUP_COLORS[g], label=GROUP_DISPLAY[g]) for g in GROUP_ORDER]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    caption = f"perplexity -- window {window_label} (~{lo/1e6:.0f}-{hi/1e6:.0f}M params)"
    if group_override:
        overrides = ", ".join(
            f"{GROUP_DISPLAY[g]} restricted to ~{go_lo/1e6:.0f}-{go_hi/1e6:.0f}M"
            for g, (go_lo, go_hi) in group_override.items()
        )
        caption += f" [{overrides}]"
    fig.text(0.02, 0.98, caption, fontsize=9, color="#666666", ha="left", va="top")
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    use_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        for mode in ("surprisal", "embedding", "dll"):
            _page(pdf, mode, "A", *WINDOW_A)
            _page(pdf, mode, "B", *WINDOW_B, group_override=WINDOW_B_GROUP_OVERRIDE)
        ppl_df = _load_perplexity()
        _perplexity_page(pdf, ppl_df, "A", *WINDOW_A)
        _perplexity_page(pdf, ppl_df, "B", *WINDOW_B, group_override=WINDOW_B_GROUP_OVERRIDE)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
