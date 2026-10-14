"""Assemble reports/benchmark_report.md from outputs/benchmark_summary/<slug>/.

Reads the per-dataset outputs written by scripts/summarize/benchmark.py, builds
the executive-summary / per-dataset figures, and writes a portable markdown
report with all embedded PNGs copied into reports/figures/ (so reports/ is
standalone and does not link back into outputs/).

Usage:
    .venv/bin/python scripts/summarize/report.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.analysis.benchmark_common import (  # noqa: E402
    BABYLM_FAMILIES,
    BEHAVIORAL_FAMILIES,
    BEHAVIORAL_MODALITIES,
    BENCHMARK_SUMMARY_ROOT,
    DATASET_FAMILY,
    DATASET_FAMILY_ORDER,
    DATASET_METHODS,
    DATASETS,
    DATASETS_BY_SLUG,
    LLM_FAMILIES,
    MEMORY_CONSTRAINED_FAMILIES,
    MODALITY_BUCKET,
    NEURAL_MODALITIES,
    NORM_YLABEL,
    REPO_ROOT,
    REPORT_TO_CEILING,
    GROUP_COLORS,
    GROUP_DISPLAY,
    GROUP_ORDER,
    add_ceiling_line,
    ceiling_for,
    draw_group_scatter,
    draw_scaling_by_perplexity,
    draw_scaling_law,
    load_noise_ceilings,
    mean_sem_by_group,
    plot_scaling_law_by_group,
    plot_two_category_bars,
    sem,
)
from neuroai.analysis.paper_style import (  # noqa: E402
    GROUP_COLORS_PAPER,
    log_size_map,
    use_paper_style,
)
from neuroai.analysis.dlt_difficulty import (  # noqa: E402
    DLT_ALL_BEHAVIORAL, DLT_DIR, DLT_GROUPS, aggregate_group, aggregate_group_pareto_behavioral,
)

REPORTS_ROOT = REPO_ROOT / "reports"
FIGURES_ROOT = REPORTS_ROOT / "figures"

# The 16 base datasets (one feature-mode each). The secondary feature-mode
# variants live only in the "Feature modes" section, not in the per-dataset
# appendix loops (which would otherwise double-count the behavioral cells).
PRIMARY_DATASETS = [d for d in DATASETS if d.base_slug is None]

# The "four versions" axis used throughout Main Results: feature mode x layer
# selection. Module-level (not local to build_main_results) since several
# other section builders iterate the same four combinations.
MODES = ("embedding", "surprisal")
LAYERS = ("best", "last")

PDF_CSS = """
@page { size: A4 portrait; margin: 1.8cm; }
body { font-family: "DejaVu Sans", Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.4; color: #1a1a1a; }
h1 { font-size: 22pt; border-bottom: 3px solid #1f77b4; padding-bottom: 6px; }
h2 { font-size: 16pt; color: #1f4e79; margin-top: 28px; border-bottom: 1px solid #ccc; padding-bottom: 3px; page-break-before: always; }
h1 + p, h2:first-of-type { page-break-before: avoid; }
h3 { font-size: 13pt; color: #1f4e79; margin-top: 18px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: left; }
th { background-color: #eef3f8; }
img { max-width: 100%; margin: 8px 0; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
strong { color: #111; }
hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
"""


def convert_report_to_pdf(md_path: Path, pdf_path: Path) -> None:
    """Render the markdown report to a standalone PDF (images/tables included)
    via markdown -> HTML -> WeasyPrint, so the report can be shared/printed
    without a markdown viewer."""
    import markdown as md_lib
    from weasyprint import HTML

    html_body = md_lib.markdown(
        md_path.read_text(), extensions=["tables", "fenced_code", "sane_lists"]
    )
    html_doc = f"<html><head><meta charset='utf-8'><style>{PDF_CSS}</style></head><body>{html_body}</body></html>"
    HTML(string=html_doc, base_url=str(md_path.parent)).write_pdf(str(pdf_path))

MODALITIES: list[str] = ["fMRI", "EEG", "iEEG / ECoG", "MEG", "Behavioral"]


def _slugs_for_modality(modality: str) -> list[str]:
    return [slug for slug, m in MODALITY_BUCKET.items() if m == modality]


def _mode_slug(base_slug: str, mode: str) -> str | None:
    """Summary slug for (base dataset, feature_mode), or None if that cell does
    not exist (e.g. naturalstories has no embedding cell)."""
    for d in DATASETS:
        if d.base == base_slug and d.feature_mode == mode:
            return d.slug
    return None


def _load_model_df(slug: str) -> pd.DataFrame | None:
    f = BENCHMARK_SUMMARY_ROOT / slug / "all_models_scores.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if df.empty:
        return None
    # The baked-in ``group`` can be stale (e.g. after a regrouping, when only some
    # datasets were re-summarized); a model split across two groups is duplicated
    # by the group-by in _bucket_model_df. Re-derive it from the single source.
    from neuroai.analysis.benchmark_common import family_group
    df["group"] = [family_group(f, m) for f, m in zip(df["family"], df["model_id"])]
    # Guarantee a ceiling-normalised column (score / noise ceiling). Older CSVs
    # without it fall back to the raw score.
    if "score_norm" not in df.columns:
        df["score_norm"] = df["score"]
    # Guarantee best-layer / last-layer (raw + ceiling-normalised) columns.
    for c in ("score_best", "score_last"):
        if c not in df.columns:
            df[c] = df["score"]
    for c in ("score_best_norm", "score_last_norm"):
        if c not in df.columns:
            df[c] = df.get(c.replace("_norm", ""), df["score_norm"])
    # NS fMRI: composites (Pareto etc.) use the LANGUAGE network only (6 fROIs), not the pooled language+MD score.
    # Set env NS_FMRI_NETWORK=all to reproduce the previous all-fROI behaviour. Per-dataset summary CSVs keep both.
    import os as _os
    if slug in ("ns_fmri", "ns_fmri__surprisal") and _os.environ.get("NS_FMRI_NETWORK", "lang") == "lang" \
            and "score_lang_norm" in df.columns:
        df = df.dropna(subset=["score_lang_norm"]).copy()
        for c in ("score_norm", "score_best_norm", "score_last_norm"):
            df[c] = df["score_lang_" + c.replace("score_", "")] if ("score_lang_" + c.replace("score_", "")) in df.columns else df["score_lang_norm"]
        df["score"] = df["score_lang"]
    return df


# Ceiling-normalised score column for a given layer selection.
def _layer_col(layer_sel: str) -> str:
    return "score_last_norm" if layer_sel == "last" else "score_best_norm"


def _modality_model_df(modality: str, mode: str = "embedding", layer_sel: str = "best",
                        exclude_bases: frozenset[str] = frozenset(), return_long: bool = False) -> pd.DataFrame:
    """Average each model's ceiling-normalised score (score / noise ceiling)
    across all datasets in this modality that it appears in. Normalising per
    dataset first puts every dataset on the same 0..1 "fraction of the data's
    noise ceiling" scale before averaging, so datasets with very different
    ceilings are combined fairly (the returned ``score`` column is normalised).

    ``mode`` selects the feature-mode variant (embedding vs surprisal) of each
    dataset; ``layer_sel`` picks the best-layer or last-layer score. Every cell
    (embedding AND surprisal, neural AND behavioral) now has a valid noise
    ceiling and is normalised against it; the fraction-of-best fallback below
    only fires for the rare dataset with no ceiling data available at all.

    ``exclude_bases``: base dataset slugs to drop entirely (e.g. Pereira's
    surprisal-track ceiling is ~0.003 -- near-zero raw scores get amplified
    ~100-300x by that tiny denominator, so a model's noise on Pereira alone
    can dominate the whole modality average; see 2026-09-14 investigation)."""
    src = _layer_col(layer_sel)
    frames = []
    for base in _slugs_for_modality(modality):
        if base in exclude_bases:
            continue
        slug = _mode_slug(base, mode)
        if slug is None:
            continue
        df = _load_model_df(slug)
        if df is None:
            continue
        cols = ["model_id", "family", "group", "label", "n_params", src]
        sub = df[cols].copy().rename(columns={src: "score_norm"})
        sub["perplexity"] = df["perplexity"] if "perplexity" in df.columns else pd.NA
        # Every dataset (neural and behavioral, embedding and surprisal) now has
        # a valid noise ceiling and is on a common 0..1 fraction-of-ceiling
        # scale. This fallback only fires if a ceiling is genuinely unavailable.
        if ceiling_for(slug) is None:
            mx = sub["score_norm"].max()
            if pd.notna(mx) and mx > 0:
                sub["score_norm"] = sub["score_norm"] / mx
        frames.append(sub.assign(dataset=slug))
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    long_df["perplexity"] = pd.to_numeric(long_df["perplexity"], errors="coerce")
    if return_long:
        return long_df
    agg = (
        long_df.groupby(["model_id", "family", "group", "label", "n_params"], as_index=False)
        .agg(score=("score_norm", "median"), perplexity=("perplexity", "median"),
             n_datasets=("dataset", "nunique"))
    )
    return agg


def _modality_long_df(modality: str, mode: str = "embedding", layer_sel: str = "best") -> pd.DataFrame:
    """Same source data as ``_modality_model_df``, but WITHOUT the final
    per-model collapse -- keeps one row per (model, dataset) with the
    ``dataset`` column intact. Feeds the mixed-effects group comparison
    (``_fit_group_mixedlm``), which needs per-dataset rows for its
    ``(1 | dataset)`` random intercept -- averaging each model's score across
    datasets FIRST (as ``_modality_model_df`` does, for the descriptive
    scatter-bar plot) silently reweights datasets by how many models they
    share and throws away the between-dataset variance a mixed model is
    supposed to absorb."""
    src = _layer_col(layer_sel)
    frames = []
    for base in _slugs_for_modality(modality):
        slug = _mode_slug(base, mode)
        if slug is None:
            continue
        df = _load_model_df(slug)
        if df is None:
            continue
        cols = ["model_id", "family", "group", "label", "n_params", src]
        sub = df[cols].copy().rename(columns={src: "score_norm"})
        sub["perplexity"] = df["perplexity"] if "perplexity" in df.columns else pd.NA
        if ceiling_for(slug) is None:
            mx = sub["score_norm"].max()
            if pd.notna(mx) and mx > 0:
                sub["score_norm"] = sub["score_norm"] / mx
        frames.append(sub.assign(dataset=slug))
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    long_df["perplexity"] = pd.to_numeric(long_df["perplexity"], errors="coerce")
    return long_df


def _robust_z(s: pd.Series) -> pd.Series:
    """Median/MAD z-score (1.4826 = the constant that makes MAD a consistent
    estimator of SD under normality) -- robust to the extreme outlier
    datasets/models this benchmark actually has (e.g. a badly-converged fit,
    or a dataset whose DLL ceiling estimate is itself degenerate), unlike a
    raw mean/SD z-score where one such outlier distorts every other point's
    standardization on that axis."""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or pd.isna(mad):
        sd = s.std()
        return (s - med) / sd if sd else pd.Series(0.0, index=s.index)
    return (s - med) / (1.4826 * mad)


def _raw_layer_col(layer_sel: str) -> str:
    return "score_last" if layer_sel == "last" else "score_best"


def _modality_model_df_zscore(modality: str, mode: str = "embedding", layer_sel: str = "best",
                               raw: bool = False, return_long: bool = False) -> pd.DataFrame:
    """Per-model composite score for one modality bucket, built to be fair
    across datasets with very different score_norm SCALES (not just
    different noise ceilings) -- e.g. UCL Reading's behavioral baseline sits
    ~70x higher than Provo's.
    A plain average of score_norm across datasets lets whichever dataset has
    the largest RAW magnitude dominate a model's composite position, which
    is exactly what feeds the Pareto-front scatter's axes.

    Fix: within each dataset, robust-z-standardize score_norm across every
    model scored on it FIRST (so a dataset contributes according to how much
    it actually discriminates between models, not its absolute scale), THEN
    average a model's z-scores across the datasets it appears in -- same
    per-dataset-then-per-model order as ``_modality_model_df``, just
    standardized before combining. See ``_modality_model_df``'s docstring for
    the shared loading logic this mirrors.

    ``raw=True`` (2026-09-14): skip noise-ceiling division entirely and
    robust-z-standardize the RAW (unnormalized) score directly -- "z-scoring
    alone", as a comparison point against the default "ceiling-normalize,
    then z-score" ordering. Motivation: ceiling-normalization is itself only
    as trustworthy as the ceiling estimate (see Pereira fMRI's ~0.003 DLL
    ceiling investigation, 2026-09-14) -- z-scoring the raw score sidesteps
    that estimate altogether, at the cost of losing the "fraction of
    explainable signal" interpretation."""
    src = _raw_layer_col(layer_sel) if raw else _layer_col(layer_sel)
    frames = []
    for base in _slugs_for_modality(modality):
        slug = _mode_slug(base, mode)
        if slug is None:
            continue
        df = _load_model_df(slug)
        if df is None:
            continue
        cols = ["model_id", "family", "group", "label", "n_params", src]
        sub = df[cols].copy().rename(columns={src: "score_norm"})
        sub["perplexity"] = df["perplexity"] if "perplexity" in df.columns else pd.NA
        sub = sub.dropna(subset=["score_norm"])
        if len(sub) < 2:
            continue
        sub["score_z"] = _robust_z(sub["score_norm"])
        frames.append(sub.assign(dataset=slug))
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    long_df["perplexity"] = pd.to_numeric(long_df["perplexity"], errors="coerce")
    if return_long:
        return long_df.drop(columns=["score_norm"]).rename(columns={"score_z": "score_norm"})
    agg = (
        long_df.groupby(["model_id", "family", "group", "label", "n_params"], as_index=False)
        .agg(score=("score_z", "median"), perplexity=("perplexity", "median"),
             n_datasets=("dataset", "nunique"))
    )
    return agg


def _bucket_model_df_zscore(modalities: tuple[str, ...], mode: str = "embedding", layer_sel: str = "best",
                             raw: bool = False) -> pd.DataFrame:
    """Z-scored counterpart of ``_bucket_model_df`` -- see
    ``_modality_model_df_zscore`` for why. ``raw`` forwards to it."""
    frames = [_modality_model_df_zscore(m, mode, layer_sel, raw=raw, return_long=True) for m in modalities]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)  # one row per (model, dataset): median pools ALL datasets in the bucket
    agg_kwargs = {"score": ("score_norm", "median")}
    if "perplexity" in long_df.columns:
        agg_kwargs["perplexity"] = ("perplexity", "median")
    return long_df.groupby(["model_id", "family", "group", "label", "n_params"], as_index=False).agg(**agg_kwargs)


def _neural_behavioral_merged_zscore(mode: str = "embedding", layer_sel: str = "best", raw: bool = False) -> pd.DataFrame:
    """Z-scored counterpart of ``_neural_behavioral_merged`` -- feeds the
    corrected Pareto front (see build_pareto_section_zscore). ``raw=True``
    z-scores unnormalized scores directly, skipping noise-ceiling division."""
    nd = _bucket_model_df_zscore(NEURAL_MODALITIES, mode, layer_sel, raw=raw).dropna(subset=["score"])
    nd = nd[["model_id", "label", "group", "n_params", "perplexity", "score"]].rename(columns={"score": "neural"})
    bd = _bucket_model_df_zscore(BEHAVIORAL_MODALITIES, mode, layer_sel, raw=raw).dropna(subset=["score"])
    bd = bd[["model_id", "score"]].rename(columns={"score": "behav"})
    return nd.merge(bd, on="model_id", how="inner").reset_index(drop=True)


def _bucket_long_df(modalities: tuple[str, ...], mode: str = "embedding", layer_sel: str = "best") -> pd.DataFrame:
    """``_modality_long_df`` merged across several modalities (Neural =
    fMRI+EEG+iEEG/ECoG vs Behavioral) -- per-(model, dataset) rows, for the
    mixed-effects group comparison."""
    frames = [_modality_long_df(m, mode, layer_sel) for m in modalities]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fit_group_mixedlm(long_df: pd.DataFrame) -> pd.DataFrame | None:
    """Mixed-effects group comparison: score_norm ~ C(group, Treatment('LLM'))
    + (1 | dataset). Every non-LLM group's coefficient is its score shift vs.
    LLM, controlling for between-dataset baseline differences via the random
    intercept -- same model (Model C) used in
    analysis/matched_scale_regression.py,
    applied here to this report's own dataset pool. Returns None if there's
    too little data (<2 datasets, or a group missing) to fit."""
    import statsmodels.formula.api as smf

    d = long_df.dropna(subset=["score_norm", "group", "dataset"]).copy()
    if d["dataset"].nunique() < 2 or d["group"].nunique() < 2 or "LLM" not in d["group"].unique():
        return None
    d["group"] = pd.Categorical(d["group"], categories=GROUP_ORDER)
    try:
        model = smf.mixedlm("score_norm ~ C(group, Treatment('LLM'))", data=d, groups=d["dataset"])
        result = model.fit(reml=True, method="lbfgs")
    except Exception:
        return None
    summ = result.summary().tables[1].reset_index().rename(
        columns={"index": "term", "Coef.": "coef", "Std.Err.": "std_err", "P>|z|": "p_value"})
    rows = []
    for g in GROUP_ORDER:
        if g == "LLM":
            continue
        term = f"C(group, Treatment('LLM'))[T.{g}]"
        row = summ[summ["term"] == term]
        if row.empty:
            continue
        rows.append(dict(group=g, coef=float(row.iloc[0]["coef"]), std_err=float(row.iloc[0]["std_err"]),
                          p_value=float(row.iloc[0]["p_value"]),
                          n=int((d["group"] == g).sum()), n_datasets=int(d.loc[d["group"] == g, "dataset"].nunique())))
    return pd.DataFrame(rows) if rows else None


def _plot_group_effects_mixedlm(neural_long: pd.DataFrame, behavioral_long: pd.DataFrame,
                                 mode: str = "embedding", layer_sel: str = "best"):
    """Bar chart: each non-LLM group's mixed-effects coefficient vs. LLM
    (0 = no difference from LLM), Neural and Behavioral side by side, with
    95% CI and a significance star. Replaces the old "average each model's
    score, then average per group" scatter-bar (see git history /
    _plot_neural_behavioral_bars) -- that naive aggregation let datasets
    with more per-model coverage dominate the group mean and threw away the
    (1 | dataset) random-intercept correction, which produced a materially
    different (and, on manual audit, less trustworthy) picture from the
    project's own matched-scale mixed-effects regression."""
    import matplotlib.pyplot as plt

    panels = [("Neural", _fit_group_mixedlm(neural_long)), ("Behavioral", _fit_group_mixedlm(behavioral_long))]
    if all(fit is None for _, fit in panels):
        return None

    groups = [g for g in GROUP_ORDER if g != "LLM"]
    domain_colors = {"Neural": "#DD8452", "Behavioral": "#4C72B0"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x0 = np.arange(len(groups))
    bar_w = 0.35
    for di, (title, fit) in enumerate(panels):
        offs = x0 + (di - 0.5) * bar_w
        heights, errs, stars = [], [], []
        for g in groups:
            if fit is None or g not in fit["group"].values:
                heights.append(0.0); errs.append(0.0); stars.append("")
                continue
            r = fit[fit["group"] == g].iloc[0]
            heights.append(r["coef"]); errs.append(1.96 * r["std_err"])
            stars.append("*" if r["p_value"] < 0.05 else "")
        ax.bar(offs, heights, bar_w, yerr=errs, capsize=3, label=title,
               color=domain_colors[title], alpha=0.85)
        for xi, h, e, s in zip(offs, heights, errs, stars):
            if s:
                ax.text(xi, h + np.sign(h) * (e + 0.005), s, ha="center",
                        va="bottom" if h >= 0 else "top", fontsize=11, fontweight="bold")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x0)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("coefficient vs. LLM (score_norm units, 95% CI)", fontsize=9.5)
    ax.set_title(f"Mixed-effects group comparison — {mode} features, {layer_sel} layer\n"
                 "score_norm ~ group + (1 | dataset); * = p<0.05 vs. LLM", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, title="domain", title_fontsize=8.5)
    fig.tight_layout()
    out = FIGURES_ROOT / f"group_effects_mixedlm_{mode}_{layer_sel}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _family_model_df(ds_family: str, mode: str = "embedding", layer_sel: str = "best") -> pd.DataFrame:
    """One row per model for a DATASET FAMILY (fMRI / ECoG / N400 / EEG / SPR /
    Eye-tracking), with its mean normalised score and mean perplexity across that
    family's datasets, for the given feature ``mode``.

    Every family (embedding or surprisal, neural or behavioral) is now
    ceiling-normalised (fraction of the noise ceiling) against its own dataset's
    split-half reliability.
    """
    src = _layer_col(layer_sel)
    base_slugs = [s for s, f in DATASET_FAMILY.items() if f == ds_family]
    frames = []
    for base in base_slugs:
        slug = _mode_slug(base, mode)
        if slug is None:
            continue
        df = _load_model_df(slug)
        if df is None:
            continue
        sub = df[["model_id", "family", "group", "label", "n_params", src]].copy().rename(columns={src: "score_norm"})
        sub["perplexity"] = df["perplexity"] if "perplexity" in df.columns else pd.NA
        if ceiling_for(slug) is None:          # fallback if ceiling genuinely unavailable
            mx = sub["score_norm"].max()
            if pd.notna(mx) and mx > 0:
                sub["score_norm"] = sub["score_norm"] / mx
        frames.append(sub.assign(dataset=slug))
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    long_df["perplexity"] = pd.to_numeric(long_df["perplexity"], errors="coerce")
    return (
        long_df.groupby(["model_id", "family", "group", "label", "n_params"], as_index=False)
        .agg(score=("score_norm", "median"), perplexity=("perplexity", "median"),
             n_datasets=("dataset", "nunique"))
    )


_SCALING_X_SPECS = {
    "n_params": ("n_params", "perplexity", "Parameters"),
    "perplexity": ("perplexity", "n_params", "Perplexity"),
}


def build_family_scaling_figure(mode: str = "embedding", layer_sel: str = "best",
                                 x_col: str = "n_params") -> Path | None:
    """THE main result figure, for one feature ``mode`` and ``layer_sel``.

    A 1 x 6 grid: columns = dataset family (fMRI, ECoG, N400/ERP, EEG, SPR,
    Eye-tracking); x-axis = ``x_col`` (parameters or perplexity, log scale).

    Encodings (this is how we fit 4 variables onto 2 axes):
      x      = x_col, log scale
      y      = normalised score (fraction of noise ceiling, neural and
               behavioral families alike)
      colour = model group  (the ONLY thing colour ever encodes)
      size   = the *other* quantitative variable -> the confound is visible in-panel
      facet  = dataset family
    Per-group linear fits (on log x) are drawn so the sign of each trend is legible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_paper_style()
    fams = [f for f in DATASET_FAMILY_ORDER if not _family_model_df(f, mode, layer_sel).empty]
    if not fams:
        return None
    dfs = {f: _family_model_df(f, mode, layer_sel) for f in fams}

    x_col, size_col, xlabel = _SCALING_X_SPECS[x_col]

    # Global log range so a dot size means the same thing across all panels.
    def _rng(col):
        v = pd.concat([d[col] for d in dfs.values()])
        v = pd.to_numeric(v, errors="coerce").dropna()
        v = v[v > 0]
        return (float(np.log10(v.min())), float(np.log10(v.max()))) if len(v) else (0.0, 1.0)
    size_rng = _rng(size_col)

    fig, axes = plt.subplots(1, len(fams), figsize=(3.6 * len(fams), 4.2), squeeze=False)
    axes = axes[0]
    # Each family panel gets its OWN y-axis. Families genuinely differ in
    # achievable fraction-of-ceiling range (surprisal especially), and a
    # single legitimate outlier (a dataset whose ceiling is mismatched to its
    # score, see the report caveat -- can read >100%) would
    # otherwise stretch every other panel's axis and crush their resolution.
    for ci, fam in enumerate(fams):
        ax = axes[ci]
        d = dfs[fam].dropna(subset=["score", x_col])
        d = d[d[x_col] > 0]
        if d.empty:
            ax.set_axis_off()
            continue
        sizes = log_size_map(d[size_col], *size_rng, s_min=12, s_max=110)
        for g in GROUP_ORDER:
            sub = d[d["group"] == g]
            if sub.empty:
                continue
            s = sizes[(d["group"] == g).to_numpy()]
            ax.scatter(sub[x_col], sub["score"], s=s, color=GROUP_COLORS_PAPER[g],
                       alpha=0.6, edgecolors="white", linewidths=0.4, zorder=2)
            if len(sub) >= 3:   # per-group fit on log-x
                lx = np.log10(sub[x_col].to_numpy(dtype=float))
                ly = sub["score"].to_numpy(dtype=float)
                b, a = np.polyfit(lx, ly, 1)
                xs = np.linspace(lx.min(), lx.max(), 50)
                ax.plot(10 ** xs, b * xs + a, color=GROUP_COLORS_PAPER[g],
                        lw=1.6, ls="--", alpha=0.9, zorder=3)
        ax.set_xscale("log")
        ax.set_title(fam, fontweight="semibold")
        ax.set_xlabel(f"{xlabel} (log)")
        if ci == 0:
            ax.set_ylabel("Fraction of noise ceiling")
        # No cross-panel y harmonisation (see note above) -- each family
        # keeps its own auto-scaled range.

    # one legend for the whole figure: colour = group, size = the other variable
    handles = [Line2D([], [], marker="o", ls="None", color=GROUP_COLORS_PAPER[g],
                      markersize=7, label=GROUP_DISPLAY.get(g, g)) for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"Scaling by dataset family — {mode} features, {layer_sel} layer — vs "
        f"{xlabel.lower()} (dot size = {_SCALING_X_SPECS['perplexity' if x_col == 'n_params' else 'n_params'][2].lower()})",
        y=1.02)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = FIGURES_ROOT / f"scaling_by_dataset_family_{mode}_{layer_sel}_{x_col}.png"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _pareto_front(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """Non-dominated set maximising both xcol and ycol (the Pareto front)."""
    keep = []
    for i, r in df.iterrows():
        dominated = ((df[xcol] >= r[xcol]) & (df[ycol] >= r[ycol]) &
                     ((df[xcol] > r[xcol]) | (df[ycol] > r[ycol]))).any()
        if not dominated:
            keep.append(i)
    return df.loc[keep].sort_values(xcol)


def _neural_behavioral_merged(mode: str = "embedding", layer_sel: str = "best",
                              exclude_neural_bases: frozenset[str] = frozenset()) -> pd.DataFrame:
    nd = _bucket_model_df(NEURAL_MODALITIES, mode, layer_sel, exclude_neural_bases).dropna(subset=["score"])
    nd = nd[["model_id", "label", "group", "n_params", "perplexity", "score"]].rename(
        columns={"score": "neural"})
    bd = _bucket_model_df(BEHAVIORAL_MODALITIES, mode, layer_sel).dropna(subset=["score"])
    bd = bd[["model_id", "score"]].rename(columns={"score": "behav"})
    return nd.merge(bd, on="model_id", how="inner").reset_index(drop=True)


def _two_dataset_merged(x_slug: str | None, y_slug: str | None, layer_sel: str = "best") -> pd.DataFrame:
    """Same shape as _neural_behavioral_merged (model_id/label/group/n_params/
    perplexity/neural/behav) but built from two SPECIFIC datasets' own
    already-ceiling-normalised score, not a modality-wide average."""
    if x_slug is None or y_slug is None:
        return pd.DataFrame()
    xdf, ydf = _load_model_df(x_slug), _load_model_df(y_slug)
    if xdf is None or ydf is None:
        return pd.DataFrame()
    src = _layer_col(layer_sel)
    xd = xdf.dropna(subset=[src])[["model_id", "label", "group", "n_params", "perplexity", src]].rename(
        columns={src: "neural"})
    yd = ydf.dropna(subset=[src])[["model_id", src]].rename(columns={src: "behav"})
    return xd.merge(yd, on="model_id", how="inner").reset_index(drop=True)


_PARETO_SIZE_SPECS = {
    "n_params": ("n_params", "Model size (params)", lambda n: f"{n/1e9:.0f}B" if n >= 1e9 else f"{n/1e6:.0f}M",
                 (1e7, 1e8, 1e9, 1e10, 7e10)),
    "perplexity": ("perplexity", "Perplexity", lambda n: f"{n/1000:.0f}k" if n >= 1000 else f"{n:.0f}",
                   (10, 100, 1e3, 1e4)),
}


def _draw_pareto_figure(m: pd.DataFrame, x_label: str, y_label: str, suptitle: str,
                         out_name: str, size_cols: tuple[str, ...] = ("n_params",)) -> tuple[Path | None, pd.DataFrame]:
    """Shared plotting core: x/y trade-off scatter + Pareto front, one dot per
    model, colour = group, dot area = one or more size encodings (parameter
    count and/or perplexity -- one panel per entry in ``size_cols``). Models
    on the Pareto front (no other model beats them on BOTH axes) are ringed
    and connected. ``m`` must have columns model_id, label, group, n_params,
    perplexity, neural (x), behav (y)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_paper_style()
    if m.empty:
        return None, m
    front = _pareto_front(m, "neural", "behav")

    rows = [_PARETO_SIZE_SPECS[c] for c in size_cols]

    fig, axes = plt.subplots(1, len(rows), figsize=(7.5 * len(rows), 7), squeeze=False)
    axes = axes[0]
    for ax, (size_col, size_title, fmt, refs) in zip(axes, rows):
        d = m.dropna(subset=[size_col]) if size_col == "perplexity" else m
        v = pd.to_numeric(d[size_col], errors="coerce")
        v = v[v > 0]
        lo, hi = (float(np.log10(v.min())), float(np.log10(v.max()))) if len(v) else (0.0, 1.0)
        sizes = log_size_map(d[size_col], lo, hi, s_min=25, s_max=400)
        for g in GROUP_ORDER:
            sub = d[d["group"] == g]
            if sub.empty:
                continue
            ss = sizes[(d["group"] == g).to_numpy()]
            ax.scatter(sub["neural"], sub["behav"], s=ss, color=GROUP_COLORS_PAPER[g],
                       alpha=0.55, edgecolors="white", linewidths=0.5, zorder=2, label=GROUP_DISPLAY.get(g, g))
        # Pareto front: connect + ring + label
        ax.plot(front["neural"], front["behav"], color="#333333", lw=1.3, ls="-",
                alpha=0.7, zorder=3)
        ax.scatter(front["neural"], front["behav"], s=170, facecolors="none",
                   edgecolors="#333333", linewidths=1.6, zorder=4)
        for _, r in front.iterrows():
            ax.annotate(r["label"], (r["neural"], r["behav"]), fontsize=6.5,
                        xytext=(4, 4), textcoords="offset points", zorder=5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"dot size = {size_title.lower()}")
        # size legend
        ref = [r for r in refs if lo <= np.log10(r) <= hi]
        handles = [Line2D([], [], marker="o", ls="None", markerfacecolor="grey",
                          markeredgecolor="white", alpha=0.6,
                          markersize=float(np.sqrt(log_size_map(pd.Series([r]), lo, hi, 25, 400)[0])),
                          label=fmt(r)) for r in ref]
        leg1 = ax.legend(handles=handles, title=size_title, loc="upper right",
                         fontsize=7, title_fontsize=7, labelspacing=1.2, borderpad=0.8)
        ax.add_artist(leg1)
    # group-colour legend (once)
    gh = [Line2D([], [], marker="o", ls="None", color=GROUP_COLORS_PAPER[g], markersize=8, label=GROUP_DISPLAY.get(g, g))
          for g in GROUP_ORDER]
    gh.append(Line2D([], [], marker="o", ls="None", markerfacecolor="none",
                     markeredgecolor="#333", markersize=10, label="Pareto-optimal"))
    axes[0].legend(handles=gh, loc="lower left", fontsize=8)
    fig.suptitle(suptitle, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES_ROOT / out_name
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out, front


def _draw_pareto_grid(panels: list[tuple[pd.DataFrame, str]], x_label: str, y_label: str,
                       suptitle: str, out_name: str, size_col: str = "n_params") -> Path | None:
    """All (mode, layer) combos as one 2x2 grid (one page) instead of one
    full-page figure per combo -- same per-panel scatter + Pareto front as
    _draw_pareto_figure, single size_col only (no side-by-side size variants)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_paper_style()
    if all(m.empty for m, _ in panels):
        return None
    size_col, size_title, _, _ = _PARETO_SIZE_SPECS[size_col]

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, (m, title) in zip(axes.flat, panels):
        if m.empty:
            ax.set_title(f"{title} (no data)", fontsize=10)
            ax.set_axis_off()
            continue
        front = _pareto_front(m, "neural", "behav")
        v = pd.to_numeric(m[size_col], errors="coerce")
        v = v[v > 0]
        lo, hi = (float(np.log10(v.min())), float(np.log10(v.max()))) if len(v) else (0.0, 1.0)
        sizes = log_size_map(m[size_col], lo, hi, s_min=18, s_max=220)
        for g in GROUP_ORDER:
            sub = m[m["group"] == g]
            if sub.empty:
                continue
            ss = sizes[(m["group"] == g).to_numpy()]
            ax.scatter(sub["neural"], sub["behav"], s=ss, color=GROUP_COLORS_PAPER[g],
                       alpha=0.55, edgecolors="white", linewidths=0.5, zorder=2, label=GROUP_DISPLAY.get(g, g))
        ax.plot(front["neural"], front["behav"], color="#333333", lw=1.1, ls="-", alpha=0.7, zorder=3)
        ax.scatter(front["neural"], front["behav"], s=120, facecolors="none",
                   edgecolors="#333333", linewidths=1.4, zorder=4)
        for _, r in front.iterrows():
            ax.annotate(r["label"], (r["neural"], r["behav"]), fontsize=5.5,
                        xytext=(3, 3), textcoords="offset points", zorder=5)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.tick_params(labelsize=8)

    gh = [Line2D([], [], marker="o", ls="None", color=GROUP_COLORS_PAPER[g], markersize=8, label=GROUP_DISPLAY.get(g, g))
          for g in GROUP_ORDER]
    gh.append(Line2D([], [], marker="o", ls="None", markerfacecolor="none",
                     markeredgecolor="#333", markersize=10, label="Pareto-optimal"))
    fig.legend(handles=gh, loc="lower center", ncol=len(gh), frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(f"{suptitle} — dot size = {size_title.lower()}", y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = FIGURES_ROOT / out_name
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_pareto_row(panels: list[tuple[pd.DataFrame, str]], x_label: str, y_label: str,
                      suptitle: str, out_name: str, size_col: str = "n_params") -> Path | None:
    """1xN row of Pareto scatter panels (e.g. one per DLT-hardness threshold),
    same per-panel scatter + front convention as _draw_pareto_grid, laid out
    in a single row instead of a 2x2 grid."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_paper_style()
    if all(m.empty for m, _ in panels):
        return None
    size_col, size_title, _, _ = _PARETO_SIZE_SPECS[size_col]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 8.2))
    axes = np.atleast_1d(axes)
    for ax, (m, title) in zip(axes, panels):
        if m.empty:
            ax.set_title(f"{title}\n(no data)", fontsize=9)
            ax.set_axis_off()
            continue
        front = _pareto_front(m, "neural", "behav")
        v = pd.to_numeric(m[size_col], errors="coerce")
        v = v[v > 0]
        lo, hi = (float(np.log10(v.min())), float(np.log10(v.max()))) if len(v) else (0.0, 1.0)
        sizes = log_size_map(m[size_col], lo, hi, s_min=14, s_max=160)
        for g in GROUP_ORDER:
            sub = m[m["group"] == g]
            if sub.empty:
                continue
            ss = sizes[(m["group"] == g).to_numpy()]
            ax.scatter(sub["neural"], sub["behav"], s=ss, color=GROUP_COLORS_PAPER[g],
                       alpha=0.55, edgecolors="white", linewidths=0.5, zorder=2, label=GROUP_DISPLAY.get(g, g))
        ax.plot(front["neural"], front["behav"], color="#333333", lw=1.0, ls="-", alpha=0.7, zorder=3)
        ax.scatter(front["neural"], front["behav"], s=90, facecolors="none",
                   edgecolors="#333333", linewidths=1.2, zorder=4)
        for _, r in front.iterrows():
            ax.annotate(r["label"], (r["neural"], r["behav"]), fontsize=6.5,
                        xytext=(4, 4), textcoords="offset points", zorder=5)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="semibold")
        ax.tick_params(labelsize=8)

    gh = [Line2D([], [], marker="o", ls="None", color=GROUP_COLORS_PAPER[g], markersize=8, label=GROUP_DISPLAY.get(g, g))
          for g in GROUP_ORDER]
    gh.append(Line2D([], [], marker="o", ls="None", markerfacecolor="none",
                     markeredgecolor="#333", markersize=10, label="Pareto-optimal"))
    fig.legend(handles=gh, loc="lower center", ncol=len(gh), frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle(f"{suptitle} — dot size = {size_title.lower()}", y=1.03)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = FIGURES_ROOT / out_name
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def build_pareto_figure(mode: str = "embedding", layer_sel: str = "best",
                         size_cols: tuple[str, ...] = ("n_params",)) -> tuple[Path | None, pd.DataFrame]:
    """Neural vs behavioral trade-off with the Pareto front, for one feature
    mode — x/y are each the mean across every dataset in that modality."""
    m = _neural_behavioral_merged(mode, layer_sel)
    suffix = "_".join(size_cols)
    return _draw_pareto_figure(
        m, "BrainScore  (mean fraction of noise ceiling) →",
        "BehavScore  (mean fraction of noise ceiling) →",
        f"Neural–behavioral trade-off and Pareto front — {mode} features, "
        f"{layer_sel} layer  (top-right = good at both)",
        f"pareto_neural_behavioral_{mode}_{layer_sel}_{suffix}.png",
        size_cols=size_cols,
    )


def build_pareto_figure_lpp_ns(mode: str = "embedding", layer_sel: str = "best",
                                size_cols: tuple[str, ...] = ("n_params",)) -> tuple[Path | None, pd.DataFrame]:
    """Same trade-off, but x/y are ONE specific dataset each (LPP fMRI vs
    Natural Stories SPR) instead of a modality-wide average — a cleaner,
    single-dataset-pair read of the same neural/behavioral tension. Natural
    Stories SPR has no embedding track, so this is only populated for
    mode="surprisal"."""
    x_slug = _mode_slug("lpp_fmri", mode)
    y_slug = _mode_slug("naturalstories_spr", mode)
    m = _two_dataset_merged(x_slug, y_slug, layer_sel)
    suffix = "_".join(size_cols)
    return _draw_pareto_figure(
        m, "LPP fMRI score  (fraction of noise ceiling) →",
        "Natural Stories SPR score  (fraction of noise ceiling) →",
        f"LPP vs Natural Stories SPR trade-off and Pareto front — {mode} features, "
        f"{layer_sel} layer  (top-right = good at both)",
        f"pareto_lpp_naturalstories_{mode}_{layer_sel}_{suffix}.png",
        size_cols=size_cols,
    )


def _adaptive_ylim(ax, value_arrays: list[np.ndarray], pad_frac: float = 0.35) -> None:
    """Zoom the y-axis to the actual spread of the plotted per-model values
    (ignoring the ax.bar(...) zero baseline), with padding on each side.
    Falls back to leaving the axis alone if the data has ~zero spread."""
    all_v = np.concatenate([v[np.isfinite(v)] for v in value_arrays if len(v)])
    if len(all_v) < 2:
        return
    lo, hi = float(all_v.min()), float(all_v.max())
    span = hi - lo
    if span <= 0:
        return
    pad = span * pad_frac
    ax.set_ylim(lo - pad, hi + pad)


def _plot_per_dataset_group_bars(mode: str, layer_sel: str) -> Path | None:
    """One page: a 4x4 grid, one panel per PRIMARY dataset, each panel a
    group-mean +/- SEM bar chart (all 4 groups) of that ONE dataset's own
    ceiling-normalised score at the given (mode, layer_sel) -- i.e. the
    per-dataset breakdown behind the modality-aggregated bars shown earlier."""
    import matplotlib.pyplot as plt

    src = _layer_col(layer_sel)
    n = len(PRIMARY_DATASETS)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    rng = np.random.default_rng(0)
    any_data = False
    for ax, info in zip(axes.flat, PRIMARY_DATASETS):
        slug = _mode_slug(info.slug, mode)
        df = _load_model_df(slug) if slug else None
        if df is None or df.empty or src not in df.columns:
            ax.set_title(f"{info.name} (no data)", fontsize=8)
            ax.set_axis_off()
            continue
        d = df.dropna(subset=[src])
        if d.empty:
            ax.set_title(f"{info.name} (no data)", fontsize=8)
            ax.set_axis_off()
            continue
        any_data = True
        groups = [g for g in GROUP_ORDER if (d["group"] == g).any()]
        all_vals: list[np.ndarray] = []
        for gi, g in enumerate(groups):
            sub = d[d["group"] == g]
            vals = sub[src].to_numpy(dtype=float)
            all_vals.append(vals)
            m, s = float(vals.mean()), sem(vals)
            ax.bar(gi, m, width=0.65, color=GROUP_COLORS[g], alpha=0.25,
                   edgecolor=GROUP_COLORS[g], linewidth=1.2, zorder=1)
            xj = gi + rng.uniform(-0.2, 0.2, len(sub))
            ax.scatter(xj, vals, s=14, color=GROUP_COLORS[g], alpha=0.6,
                       edgecolors="white", linewidths=0.3, zorder=2)
            ax.errorbar(gi, m, yerr=s, fmt="D", color="black", markersize=4.5,
                        markerfacecolor="white", markeredgewidth=1.0,
                        capsize=3.5, elinewidth=1.3, zorder=4)
        # Behavioral panels: r_full = r_base + delta_r rides on a large, model-
        # independent baseline (see conversation -- switching the headline
        # metric from delta-r to full-model r means between-group differences
        # are now a tiny fraction of the y-range if the axis is forced to
        # include the ax.bar(...) zero baseline). Zoom to the data's own
        # range instead so those differences are actually visible; neural
        # panels keep the zero-anchored default since their spread is large
        # relative to the baseline.
        if info.response == "behavioral" and all_vals:
            _adaptive_ylim(ax, all_vals)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\n(n={int((d['group'] == g).sum())})" for g in groups], fontsize=5.8)
        ax.set_title(info.name, fontsize=8.5, fontweight="semibold")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[n:]:
        ax.set_axis_off()
    if not any_data:
        plt.close(fig)
        return None
    fig.suptitle(f"Per-dataset group means — {mode} features, {layer_sel} layer "
                 "(fraction of noise ceiling)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES_ROOT / f"per_dataset_group_bars_{mode}_{layer_sel}.png"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def build_per_dataset_group_bars_section() -> str:
    """Four pages (one per feature mode x layer selection), each a 4x4 grid of
    every primary dataset's own group-mean bar chart -- the per-dataset detail
    behind the modality-aggregated 'Mean performance by model group' figure."""
    lines = ["## Per-Dataset Group Means\n"]
    for mode in MODES:
        for layer_sel in LAYERS:
            lines.append(f"\n### {mode.capitalize()} — {layer_sel.capitalize()} layer\n")
            out = _plot_per_dataset_group_bars(mode, layer_sel)
            if out is not None:
                lines.append(f"\n![Per-dataset group means, {mode}, {layer_sel}](figures/{out.name})\n")
            else:
                lines.append("\n*(no data for this feature mode / layer combination)*\n")
    return "\n".join(lines)


def build_dlt_hard_word_section() -> str:
    """DLT (Gibson 2000) integration-cost difficulty split: regular vs
    hard-words-only (DLT cost >= 2) group bars, for behavioral-surprisal,
    behavioral-embedding, and neural tracks. Figures are pre-generated by
    scripts/dlt/hard_bars_v2.py (fast, reads the precomputed
    outputs/dlt_difficulty/*_multi_threshold.csv files -- NOT regenerated by
    this script) and just embedded here if present."""
    lines = ["## DLT-Hard-Word Difficulty Split (DLT cost ≥ 2)\n"]
    specs = [
        ("behavioral_surprisal", "Behavioral — Surprisal features"),
        ("behavioral_embedding", "Behavioral — Embedding features"),
        ("neural", "Neural"),
    ]
    any_fig = False
    for key, title in specs:
        fname = f"dlt_hard_bars_{key}_per_dataset.png"
        src = REPO_ROOT / "outputs" / "dlt_difficulty" / fname
        rel = _copy_fig(src, fname)
        if rel:
            lines.append(f"\n### {title}\n")
            lines.append(f"\n![DLT hard-word split, {title}]({rel})\n")
            any_fig = True
    if not any_fig:
        lines.append("\n*(DLT hard-word figures not yet generated — run scripts/dlt/hard_bars_v2.py)*\n")
    return "\n".join(lines)


def build_average_behavioral_scaling_section() -> str:
    """Scaling law for behavioral-embedding models fit on the SUBJECT-AVERAGED
    RT signal (one ridge fit on the mean RT across subjects per model),
    instead of the usual per-subject-then-averaged convention. Averaged over
    GECO / MECO / Provo / UCL Reading / ZuCo Reading. Pre-generated by
    scripts/figures/average_embed_scaling.py."""
    lines = ["## Scaling Law — Behavioral Embedding (Subject-Averaged Fit)\n"]
    src = REPO_ROOT / "outputs" / "behavioral_embed_average_scaling.png"
    rel = _copy_fig(src, "behavioral_embed_average_scaling.png")
    if rel:
        lines.append(f"\n![Behavioral embedding scaling law, subject-averaged fit]({rel})\n")
    else:
        lines.append("\n*(figure not yet generated — run scripts/figures/average_embed_scaling.py)*\n")
    return "\n".join(lines)


def _dlt_pareto_merged_at(threshold: float, use_col: str) -> pd.DataFrame:
    """Pareto input at a given DLT hardness threshold: x = mean ceiling-
    normalised score across ALL available behavioral datasets (surprisal +
    embedding), y = mean ceiling-normalised score across the NEURAL datasets
    that have hard-word DLT data ready (podcast/fedorenko2016 --
    fedorenko2016 drops out above threshold=1 since it has 0 words at
    DLT cost >= 2, a genuine data limitation of that stimulus set, not a
    bug). ``use_col`` selects score_easy / score_hard / score_all (the last
    is the unweighted mean of easy+hard, an approximation of the whole-
    corpus score reconstructed from the exhaustive easy/hard split at
    threshold=1, where cost is integer-valued so easy(cost<=0) and
    hard(cost>=1) partition all words with no gap).

    Behavioral axis: embedding-track files (already bounded/ceiling-
    normalised) + surprisal-track files via the bounded Spearman-r-vs-fdur
    correlation cache (outputs/dlt_corr_cache/, see
    aggregate_group_pareto_behavioral) -- NOT the raw outputs/dlt_difficulty
    surprisal files, whose "score" column is a sum of delta-loglik across the
    bin's words and is incommensurable with the ceiling-normalised
    correlation scale used here."""
    behav = aggregate_group_pareto_behavioral(hard_gt=threshold)
    neural = aggregate_group(DLT_GROUPS["neural"], hard_gt=threshold)
    if behav.empty or neural.empty:
        return pd.DataFrame()
    for d in (behav, neural):
        d["score_all"] = d[["score_easy", "score_hard"]].mean(axis=1)
    bd = behav[["model_id", use_col]].rename(columns={use_col: "behav"})
    nd = neural[["model_id", "label", "group", "n_params", "perplexity", use_col]].rename(
        columns={use_col: "neural"})
    return nd.merge(bd, on="model_id", how="inner").reset_index(drop=True)


def _dlt_hard_word_count(threshold: float) -> int:
    """Sum n_hard (or n_easy+n_hard at threshold=1 for the 'all words' panel)
    across every dataset file contributing to the behavioral+neural hard-case
    Pareto, from the precomputed *_dlt_threshold_bin_counts.json files."""
    import json
    total = 0
    for file_stem in DLT_ALL_BEHAVIORAL + DLT_GROUPS["neural"]:
        p = DLT_DIR / f"{file_stem}_dlt_threshold_bin_counts.json"
        if not p.is_file():
            continue
        try:
            counts = json.loads(p.read_text()).get(f"{threshold:.1f}", {})
        except Exception:
            continue
        total += int(counts.get("n_hard", 0))
    return total


def build_hard_case_pareto_section() -> str:
    """Figures only: one row of 5 Pareto panels -- all words, then DLT cost
    >= 1/2/3/4 -- so the reader can see how the neural-behavioral trade-off
    shifts as words get syntactically harder, with the hard-word count in
    each panel's title so the (shrinking) sample size is visible. Same
    dataset selection throughout: all available behavioral datasets on x,
    the neural datasets with hard-word DLT data ready on y."""
    lines = ["## Neural–Behavioral Trade-off — Across DLT-Hardness Thresholds\n"]
    panels = []
    m_all = _dlt_pareto_merged_at(1.0, "score_all")
    n_all = _dlt_hard_word_count(1.0)
    panels.append((m_all, f"All words\n(Σn≈{n_all:,})"))
    for thr in (1.0, 2.0, 3.0, 4.0):
        m = _dlt_pareto_merged_at(thr, "score_hard")
        n = _dlt_hard_word_count(thr)
        panels.append((m, f"cost ≥ {thr:g}\n(Σn_hard≈{n:,} words, summed across datasets)"))
    out = _draw_pareto_row(
        panels, "BrainScore  (fraction of ceiling) →", "BehavScore  (fraction of ceiling) →",
        "Neural–behavioral trade-off across DLT-hardness thresholds  (top-right = good at both)",
        "pareto_hard_case_thresholds_n_params.png",
    )
    if out is not None:
        lines.append(f"\n![Hard-case neural-behavioral Pareto fronts across thresholds](figures/{out.name})\n")
    else:
        lines.append("\n*(no models with both scores — hard-case DLT data not yet complete)*\n")
    return "\n".join(lines)


def build_pythia_checkpoint_section(variant: str = "perplexity") -> str:
    """Pythia training-checkpoint sweep summaries -- pre-generated, just
    copied in from outputs/{dataset}_pythia_checkpoints_summary/.
    ``variant``: "perplexity" (score vs perplexity across every completed
    checkpoint -- main body) or "size" (score vs model size at the final
    checkpoint -- appendix)."""
    title = "Pythia Training-Checkpoint Curves (Embedding)" + (" — vs Perplexity" if variant == "perplexity" else " — vs Model Size")
    lines = [f"## {title}\n"]
    specs = [("provo", "Provo"), ("meco", "MECO")]
    any_fig = False
    for slug, name in specs:
        src_dir = REPO_ROOT / "outputs" / f"{slug}_pythia_checkpoints_summary"
        src = src_dir / f"{slug}_pythia_by_{variant}.png"
        rel = _copy_fig(src, f"pythia_ckpt_{slug}_by_{variant}.png")
        if rel:
            lines.append(f"\n### {name}\n")
            lines.append(f"\n![{name} Pythia checkpoints by {variant}]({rel})\n")
            any_fig = True
    if not any_fig:
        lines.append("\n*(no Pythia checkpoint summary figures found.)*\n")
    return "\n".join(lines)


def build_roi_section() -> str:
    """ROI/region score profiles -- pre-generated by summarize/benchmark.py's
    process_dataset() (plot_roi_profile) for every dataset with has_roi=True,
    embedded here for the 4 with a genuine anatomical/functional ROI or
    electrode-region breakdown (fMRI parcels for lpp/ns_fmri/tuckute, scalp
    region for podcast iEEG) rather than raw per-channel labels."""
    lines = ["## ROI / Region Score Profiles\n"]
    specs = [
        ("lpp_fmri", "Le Petit Prince fMRI"),
        ("ns_fmri", "Natural Stories fMRI"),
        ("tuckute_fmri", "Tuckute fMRI"),
        ("podcast_ieeg", "Podcast iEEG"),
    ]
    any_fig = False
    for slug, name in specs:
        src = BENCHMARK_SUMMARY_ROOT / slug / "roi_profile.png"
        rel = _copy_fig(src, f"roi_profile_{slug}.png")
        if rel:
            lines.append(f"\n### {name}\n")
            lines.append(f"\n![{name} ROI profile]({rel})\n")
            any_fig = True
    if not any_fig:
        lines.append("\n*(no ROI profile figures found -- run scripts/summarize/benchmark.py first.)*\n")
    return "\n".join(lines)


def build_feature_mode_section() -> str:
    """The feature-mode 'doubling': {neural, behavioral} x {embedding, surprisal}.

    Regenerates the coverage table + comparison figure (via
    plot_feature_mode_comparison) and embeds them, stating the native metric and
    layer selection (best vs last) for each cell.
    """
    import subprocess

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "figures/feature_mode_comparison.py")],
        check=False, capture_output=True,
    )
    fm_dir = BENCHMARK_SUMMARY_ROOT / "_feature_mode"
    cov_path = fm_dir / "coverage.csv"

    lines = ["## Feature modes: embedding vs surprisal\n"]
    if cov_path.is_file():
        cov = pd.read_csv(cov_path).fillna("")
        lines.append("\n| Dataset | Response | Embedding (n, layers) | Surprisal (n, layers) |")
        lines.append("|---|---|---|---|")
        for _, r in cov.iterrows():
            n_emb, n_sup = int(r["embedding"]), int(r["surprisal"])
            sup_layers = "last only" if r["dataset"] == "podcast_ieeg" else "best+last"
            emb = f"{n_emb} · best+last" if n_emb else "— not computed"
            sup = f"{n_sup} · {sup_layers}" if n_sup else "— not computed"
            lines.append(f"| {r['dataset']} | {r['response']} | {emb} | {sup} |")
        lines.append("")
    for layer_sel in ("best", "last"):
        chunks = sorted(fm_dir.glob(f"comparison_{layer_sel}_*.png"),
                        key=lambda p: int(p.stem.split("_")[-1]))
        if not chunks:
            continue
        lines.append(f"\n### Embedding vs Surprisal — {layer_sel.capitalize()} layer\n")
        for p in chunks:
            rel = _copy_fig(p, f"feature_mode_{p.name}")
            if rel:
                lines.append(f"\n![Feature-mode comparison {layer_sel} {p.stem.split('_')[-1]}]({rel})\n")
    return "\n".join(lines)


def build_methods_table() -> str:
    """The standardization contract: exactly what is computed for each dataset.

    These are NOT uniform (windows/baselines differ) — stating them explicitly is
    the honest thing to do until the sweeps are re-run on a common convention.
    """
    lines = ["## Datasets and Metrics\n"]
    lines.append("| Dataset | Family | N | Window | Baseline | Metric | Aggregation |")
    lines.append("|---|---|---|---|---|---|---|")
    for fam in DATASET_FAMILY_ORDER:
        for info in PRIMARY_DATASETS:
            if DATASET_FAMILY.get(info.slug) != fam:
                continue
            m = DATASET_METHODS.get(info.slug)
            if m is None:
                continue
            lines.append(
                f"| {info.name} | {fam} | {info.n_subjects} | {m.window} | "
                f"{m.baseline} | {m.metric} | {m.aggregation} |"
            )
    lines.append("")
    return "\n".join(lines)


def _bucket_model_df(modalities: tuple[str, ...], mode: str = "embedding", layer_sel: str = "best",
                      exclude_bases: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Same as _modality_model_df but merged across several modalities (used
    for the Neural = fMRI+EEG+iEEG/ECoG vs Behavioral bar chart)."""
    frames = [_modality_model_df(m, mode, layer_sel, exclude_bases, return_long=True) for m in modalities]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)  # one row per (model, dataset): median pools ALL datasets in the bucket
    agg_kwargs = {"score": ("score_norm", "median")}
    if "perplexity" in long_df.columns:
        agg_kwargs["perplexity"] = ("perplexity", "median")
    agg = (
        long_df.groupby(["model_id", "family", "group", "label", "n_params"], as_index=False)
        .agg(**agg_kwargs)
    )
    return agg


def _copy_fig(src: Path, dest_name: str) -> str | None:
    if not src.exists():
        return None
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    dest = FIGURES_ROOT / dest_name
    shutil.copy2(src, dest)
    return f"figures/{dest_name}"


def build_executive_summary() -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Grid sized to fit len(MODALITIES) panels, not a hardcoded 2x2 -- MEG was
    # added as a 5th modality, which zip() would otherwise silently drop under
    # a fixed 2x2 grid (only 4 axes).
    _ncols = 2
    _nrows = -(-len(MODALITIES) // _ncols)  # ceil division
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(17, 6.5 * _nrows))
    for _ax in list(axes.flat)[len(MODALITIES):]:
        _ax.set_axis_off()  # unused trailing slot(s)
    lines = ["## Executive Summary\n"]
    lines.append(
        f"All {len(MODALITIES)} panels ({', '.join(MODALITIES[:-1])}, **and {MODALITIES[-1]}**) are on the "
        "SAME scale: each score is **normalised by the dataset's noise ceiling** "
        "(`score / ceiling_r`, where the ceiling is the square root of the "
        "split-half reliability of the data — see `outputs_jeanzay/noise_ceilings/`). "
        "So a value of 0.4 means the model captures 40% of the attainable "
        "correlation; a model's plotted value is the average of its per-dataset "
        "normalised scores across that modality's datasets (normalising per "
        "dataset first puts datasets with very different ceilings on the same "
        "footing). Every score in this report is a **Pearson r** (or a held-out "
        "delta Pearson r for the behavioral/surprisal cells, ceiling-normalised "
        "against the dataset's own reading-time split-half reliability — see "
        "Datasets and Metrics above). Marker **shape** encodes model family, "
        "**color** encodes model group; diamonds show group mean +/- SEM; grey "
        "dotted vertical lines mark reference parameter counts.\n"
    )
    neural_axes = []
    for ax, modality in zip(axes.flat, MODALITIES):
        df = _modality_model_df(modality)
        if df.empty:
            ax.set_title(f"{modality} (no data)")
            continue
        df = df.dropna(subset=["score"])
        draw_group_scatter(ax, df, "score", ylabel=NORM_YLABEL, title=modality)
        neural_axes.append(ax)
    # Shared y-axis across all four (now all ceiling-normalised) panels.
    if neural_axes:
        ymax = max(ax.get_ylim()[1] for ax in neural_axes)
        for ax in neural_axes:
            ax.set_ylim(0.0, ymax)
    fig.suptitle("Cross-modality overview: every panel as a fraction of the noise ceiling "
                 "— colored by group, shaped by family",
                 fontsize=14)
    fig.tight_layout()
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIGURES_ROOT / "executive_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    lines.append(f"![Executive summary](figures/{out.name})\n")

    neural_df = _bucket_model_df(NEURAL_MODALITIES)      # score = mean fraction of ceiling
    behavioral_df = _bucket_model_df(BEHAVIORAL_MODALITIES)  # score = mean fraction of ceiling

    # ---- Cross-dataset general summary: global scaling laws --------------------
    # Ceiling-normalisation now makes every dataset comparable, neural AND
    # behavioral alike, so each modality bucket collapses into a single
    # benchmark-wide scaling law on the same fraction-of-ceiling scale. Kept as
    # two separate panels (rather than merged into one) since neural and
    # behavioral still draw on different underlying dataset pools.
    import matplotlib.pyplot as _plt
    lines.append(
        "\n### Cross-dataset scaling laws\n\n"
        "Because every score (neural AND behavioral) is now on the same "
        "fraction-of-noise-ceiling scale, we can collapse each modality bucket "
        "into a single benchmark-wide scaling law. Each model's value is its "
        "mean normalised score across the datasets it was run on, versus its "
        "**perplexity** (mean across those datasets; log scale) — a more "
        "meaningful axis than raw size, since it asks directly whether better "
        "language modelling yields better brain/behaviour prediction and places "
        "all model families (including the n-gram models) on a common footing. "
        "Per-group linear fits shown; **neural** and **behavioral** are kept as "
        "separate panels (different dataset pools), though both share one scale.\n"
    )
    NEURAL_YL = "Mean fraction of noise ceiling (neural)"
    BEHAV_YL = "Mean fraction of noise ceiling (behavioral)"
    if not neural_df.empty or not behavioral_df.empty:
        figS, (axN, axB) = _plt.subplots(1, 2, figsize=(16, 6))
        for ax, df, title, yl in ((axN, neural_df, "Neural", NEURAL_YL),
                                  (axB, behavioral_df, "Behavioral", BEHAV_YL)):
            if df.empty or "perplexity" not in df.columns or df["perplexity"].notna().sum() == 0:
                ax.set_title(f"{title} (no perplexity data)")
                ax.set_axis_off()
                continue
            draw_scaling_by_perplexity(ax, df, "score", "perplexity", ylabel=yl,
                                       title=title, label_col=None)
        figS.suptitle("Global scaling laws vs perplexity: neural vs behavioral", fontsize=14)
        figS.tight_layout()
        spath = FIGURES_ROOT / "global_scaling_laws.png"
        figS.savefig(spath, dpi=150, bbox_inches="tight")
        _plt.close(figS)
        lines.append(f"![Global scaling laws](figures/{spath.name})\n")
    # ...and each broken out per model category (2x2).
    if not neural_df.empty:
        plot_scaling_law_by_group(
            neural_df, "score", FIGURES_ROOT / "global_neural_scaling_by_group.png",
            ylabel=NEURAL_YL, ppl_col="perplexity",
            title="Global neural scaling law (vs perplexity) by model category", label_points=False,
        )
        lines.append("\nNeural, broken out per model category:\n")
        lines.append("![Global neural scaling law by category](figures/global_neural_scaling_by_group.png)\n")
    if not behavioral_df.empty:
        plot_scaling_law_by_group(
            behavioral_df, "score", FIGURES_ROOT / "global_behavioral_scaling_by_group.png",
            ylabel=BEHAV_YL, ppl_col="perplexity",
            title="Global behavioral scaling law (vs perplexity) by model category", label_points=False,
        )
        lines.append("\nBehavioral, broken out per model category:\n")
        lines.append("![Global behavioral scaling law by category](figures/global_behavioral_scaling_by_group.png)\n")

    # ---- Neural vs Behavioral, mean per group (1x2, same fraction-of-ceiling scale) ---
    lines.append(
        "\n### Mean performance by model group\n\n"
        "Columns: **Neural** (each model's mean fraction of the noise ceiling across the "
        "neural datasets) and **Behavioral** (each model's mean fraction of the noise "
        "ceiling across the behavioral datasets, using the held-out delta-Pearson-r "
        "headline metric). Both columns are now on the same fraction-of-ceiling scale.\n\n"
        "Bars show the group mean and the white diamond the mean +/- SEM, but **every "
        "individual model is also plotted as a dot**, so the within-group spread is "
        "visible rather than hidden behind an error bar. The **same models appear in both "
        "rows; only the dot size changes**:\n\n"
        "* **Top row — dot area = parameter count** (log scale). In *Neural* the large "
        "models sit at the top of the LLM cloud; in *Behavioral* they sink to the bottom "
        "— the inverse-scaling effect, visible directly.\n"
        "* **Bottom row — dot area = perplexity** (log scale; bigger dot = *higher* "
        "perplexity, i.e. a weaker language model). This is the perplexity-confound view: "
        "within Memory-Constrained, the highest-perplexity models are exactly the ones "
        "scoring highest on *Behavioral* and lowest on *Neural*.\n"
    )
    bar_path = _plot_neural_behavioral_bars(neural_df, behavioral_df)
    if bar_path is not None:
        lines.append(f"![Neural vs Behavioral by group](figures/{bar_path.name})\n")

    return "\n".join(lines)


def _fmt_params(n: float) -> str:
    return f"{n / 1e9:.0f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def _fmt_ppl(n: float) -> str:
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return f"{n:.0f}"


S_MIN, S_MAX = 18.0, 340.0  # dot AREA range (pt^2)


def _log_range(dfs: list[pd.DataFrame], col: str) -> tuple[float, float]:
    """Global log10 range of ``col`` across dfs, so a dot of a given size means
    the same thing in every panel of that row."""
    vals = pd.concat([d[col] for d in dfs if not d.empty and col in d.columns])
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    vals = vals[vals > 0]
    if not len(vals):
        return 0.0, 1.0
    return float(np.log10(vals.min())), float(np.log10(vals.max()))


def _sizes_from(series, lo: float, hi: float) -> np.ndarray:
    """Map a value to dot AREA on a log10 scale (params span 14M..66B and
    perplexity spans ~8..70k, so a linear map would hide almost everything)."""
    v = pd.to_numeric(series, errors="coerce")
    v = v.fillna(10 ** lo).clip(lower=1.0)          # missing -> smallest dot
    lp = np.log10(v)
    norm = (lp - lo) / (hi - lo) if hi > lo else np.full(len(lp), 0.5)
    return S_MIN + np.clip(np.asarray(norm, dtype=float), 0.0, 1.0) * (S_MAX - S_MIN)


_BAR_SIZE_SPECS = {
    "n_params": ("n_params", "Model size (params)", _fmt_params, (1e7, 1e8, 1e9, 1e10, 7e10)),
    "perplexity": ("perplexity", "Perplexity", _fmt_ppl, (10, 100, 1e3, 1e4, 1e5)),
}


def _plot_neural_behavioral_bars(neural_df: pd.DataFrame, behavioral_df: pd.DataFrame, mode: str = "embedding",
                                  layer_sel: str = "best", size_col: str = "n_params"):
    """1x2: columns = Neural | Behavioral, dot size encodes ``size_col``
    (parameters or perplexity).

    Bar = group mean, white diamond = mean +/- SEM, and every individual model is
    a jittered dot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [
        (neural_df, "Neural", "Mean fraction of noise ceiling"),
        (behavioral_df, "Behavioral", "Mean fraction of noise ceiling"),
    ]
    if all(df.empty for df, _, _ in panels):
        return None
    dfs = [d for d, _, _ in panels]

    size_col, legend_title, fmt, ref_candidates = _BAR_SIZE_SPECS[size_col]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    lo, hi = _log_range(dfs, size_col)
    rng = np.random.default_rng(0)
    for ci, (df, title, ylabel) in enumerate(panels):
        ax = axes[ci]
        if df.empty:
            ax.set_title(f"{title} (no data)")
            ax.set_axis_off()
            continue
        d = df.dropna(subset=["score"])
        groups = [g for g in GROUP_ORDER if (d["group"] == g).any()]
        all_vals: list[np.ndarray] = []
        for gi, g in enumerate(groups):
            sub = d[d["group"] == g]
            vals = sub["score"].to_numpy(dtype=float)
            all_vals.append(vals)
            m, s = float(vals.mean()), sem(vals)
            ax.bar(gi, m, width=0.62, color=GROUP_COLORS[g], alpha=0.20,
                   edgecolor=GROUP_COLORS[g], linewidth=1.4, zorder=1)
            sizes = _sizes_from(sub[size_col], lo, hi)
            order = np.argsort(-sizes)   # big dots behind, small stay visible
            xj = gi + rng.uniform(-0.22, 0.22, len(sub))
            ax.scatter(xj[order], vals[order], s=sizes[order],
                       color=GROUP_COLORS[g], alpha=0.55, edgecolors="white",
                       linewidths=0.6, zorder=2)
            ax.errorbar(gi, m, yerr=s, fmt="D", color="black", markersize=7,
                        markerfacecolor="white", markeredgewidth=1.6,
                        capsize=6, elinewidth=2, zorder=4)
        # Zoom behavioral panels to their own data range -- see _adaptive_ylim
        # docstring / _plot_per_dataset_group_bars for why (r_full compression).
        if title == "Behavioral" and all_vals:
            _adaptive_ylim(ax, all_vals)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\n(n={int((d['group'] == g).sum())})" for g in groups],
                           fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — dot size = {legend_title.lower()}")
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    # size legend (on the right-hand panel)
    refs = [r for r in ref_candidates if lo <= np.log10(r) <= hi]
    handles = [
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="grey",
               markeredgecolor="white", alpha=0.65,
               markersize=float(np.sqrt(_sizes_from(pd.Series([r]), lo, hi)[0])),
               label=fmt(r))
        for r in refs
    ]
    if handles and not panels[1][0].empty:
        axes[1].legend(handles=handles, title=legend_title, fontsize=8,
                       title_fontsize=8, loc="best", frameon=True,
                       labelspacing=1.3, borderpad=0.9, handletextpad=1.2)

    fig.suptitle(
        f"Mean performance by model group — {mode} features, {layer_sel} layer — each dot is one "
        f"model; bar = group mean, white diamond = mean +/- SEM; dot area = {legend_title.lower()}.",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGURES_ROOT / f"neural_vs_behavioral_by_group_{mode}_{layer_sel}_{size_col}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


N_QUANTILE_BINS = 8


def _quantile_bins(series: pd.Series, n_bins: int, fmt) -> tuple[pd.Series, list[str]]:
    """Equal-COUNT (quantile) bins of a positive-valued series, so every bar in
    the binned plot rests on a comparable, non-trivial sample size — fixed-width
    or fixed-log-spaced bins would leave some bins with 1-2 models and others
    with 50+ given how skewed perplexity/params are. Returns (bin index per row,
    human-readable range label per bin, using ``fmt`` for the endpoints)."""
    s = pd.to_numeric(series, errors="coerce")
    valid = s[s > 0]
    n_bins_eff = min(n_bins, valid.nunique())
    if n_bins_eff < 2:
        return pd.Series(np.nan, index=series.index), []
    codes, edges = pd.qcut(valid, n_bins_eff, retbins=True, duplicates="drop")
    bin_idx = pd.Series(np.nan, index=series.index)
    bin_idx.loc[valid.index] = codes.cat.codes.astype(float)
    labels = [f"{fmt(edges[i])}–{fmt(edges[i + 1])}" for i in range(len(edges) - 1)]
    return bin_idx, labels


def _plot_binned_bars(neural_df: pd.DataFrame, behavioral_df: pd.DataFrame, mode: str = "embedding",
                       layer_sel: str = "best", n_bins: int = N_QUANTILE_BINS):
    """Same story as _plot_neural_behavioral_bars but replaces dot-SIZE encoding
    with quantile BINS on the x-axis — bin edges are shared across groups (same
    perplexity/param range for everyone), and each bin holds one narrow bar PER
    architecture group (mean +/- SEM of that group's models in that range), so
    ranges are directly comparable AND group composition within a range is
    directly comparable, without eyeballing dot sizes or colors.
    """
    import matplotlib.pyplot as plt

    panels = [
        (neural_df, "Neural", "Mean fraction of noise ceiling"),
        (behavioral_df, "Behavioral", "Mean fraction of noise ceiling"),
    ]
    if all(df.empty for df, _, _ in panels):
        return None
    rows = [("perplexity", "Perplexity", _fmt_ppl), ("n_params", "Model size (params)", _fmt_params)]

    n_groups = len(GROUP_ORDER)
    sub_width = 0.8 / n_groups  # total cluster width 0.8, small gap to next bin
    legend_seen: set[str] = set()  # label each group once, in whichever bin/panel it first appears

    fig, axes = plt.subplots(2, 2, figsize=(19, 13))
    for ri, (bin_col, row_title, fmt) in enumerate(rows):
        for ci, (df, title, ylabel) in enumerate(panels):
            ax = axes[ri, ci]
            if df.empty:
                ax.set_title(f"{title} (no data)")
                ax.set_axis_off()
                continue
            d = df.dropna(subset=["score", bin_col]).copy()
            d = d[d[bin_col] > 0]
            if len(d) < 6:
                ax.set_title(f"{title} (too few models with {bin_col})")
                ax.set_axis_off()
                continue
            bin_idx, labels = _quantile_bins(d[bin_col], n_bins, fmt)
            d["_bin"] = bin_idx
            d = d.dropna(subset=["_bin"])
            n_bins_eff = len(labels)
            for b in range(n_bins_eff):
                sub = d[d["_bin"] == b]
                for gi, g in enumerate(GROUP_ORDER):
                    gsub = sub[sub["group"] == g]
                    x = b + (gi - (n_groups - 1) / 2) * sub_width
                    if gsub.empty:
                        continue
                    vals = gsub["score"].to_numpy(dtype=float)
                    m = float(vals.mean())
                    ax.bar(x, m, width=sub_width * 0.9, color=GROUP_COLORS[g], alpha=0.75,
                           edgecolor=GROUP_COLORS[g], linewidth=1.0, zorder=1,
                           label=GROUP_DISPLAY.get(g, g) if g not in legend_seen else None)
                    legend_seen.add(g)
                    s = sem(vals) if len(vals) >= 2 else 0.0
                    if len(vals) >= 2:
                        ax.errorbar(x, m, yerr=s, fmt="none", color="black",
                                    capsize=2.5, elinewidth=1.1, zorder=2)
                    y_text = m + s if m >= 0 else m - s
                    va = "bottom" if m >= 0 else "top"
                    ax.text(x, y_text, str(len(vals)), ha="center", va=va,
                            fontsize=5.5, color=GROUP_COLORS[g], rotation=90, alpha=0.85)
            ax.set_xticks(range(n_bins_eff))
            ax.set_xticklabels(labels, fontsize=7.5, rotation=0)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — binned by {row_title.lower()}")
            ax.set_xlabel(row_title)
            ax.grid(True, axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.margins(y=0.15)  # headroom for the per-bar n= labels above/below each bar
    handles, hlabels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles += h
        hlabels += l
    if handles:
        fig.legend(handles, hlabels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Mean performance by quantile range ({n_bins} bins, shared edges across groups) — {mode} features, "
        f"{layer_sel} layer.\nEach bin has one narrow bar per architecture group (mean +/- SEM of that "
        "group's models in that range); small numbers above/below each bar are that group's model count.",
        fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = FIGURES_ROOT / f"neural_vs_behavioral_by_range_{mode}_{layer_sel}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def build_bar_plots_section() -> str:
    """Figures only: mean performance by model group, vs model size (params)."""
    lines = ["## Mean Performance by Model Group\n"]
    for mode in MODES:
        for layer_sel in LAYERS:
            neural_df = _bucket_model_df(NEURAL_MODALITIES, mode, layer_sel)
            behavioral_df = _bucket_model_df(BEHAVIORAL_MODALITIES, mode, layer_sel)
            bar_path = _plot_neural_behavioral_bars(neural_df, behavioral_df, mode, layer_sel, size_col="n_params")
            if bar_path is not None:
                lines.append(f"\n### {mode.capitalize()} — {layer_sel.capitalize()} layer\n")
                lines.append(f"![Mean performance by group, {mode}, {layer_sel}](figures/{bar_path.name})\n")
    return "\n".join(lines)


def build_group_effects_mixedlm_section() -> str:
    """Mixed-effects counterpart of build_bar_plots_section() -- 2026-09-07,
    added because the scatter-bar panel above averages each model's score
    across datasets FIRST, then averages per group: datasets with more
    per-model coverage silently dominate that group mean, and the panel
    throws away the between-dataset baseline variance a random intercept is
    meant to absorb. This section fits
    ``score_norm ~ C(group, Treatment('LLM')) + (1 | dataset)`` on the SAME
    underlying per-model-per-dataset data instead (Model C from
    analysis/matched_scale_regression.py), and shows each group's coefficient
    vs. LLM (0 = no difference) rather than an absolute mean -- read this
    panel, not the scatter-bar one above, as this report's answer to "does
    group X differ from LLM."""
    lines = ["## Mean Performance by Model Group — Mixed-Effects Estimate\n",
             "score_norm ~ group + (1 | dataset), fit separately per (feature mode, layer, domain) — "
             "bars are each group's coefficient vs. LLM (0 = no difference), not an absolute mean. "
             "See the scatter-bar panels above for the raw per-model distribution.\n"]
    for mode in MODES:
        for layer_sel in LAYERS:
            neural_long = _bucket_long_df(NEURAL_MODALITIES, mode, layer_sel)
            behavioral_long = _bucket_long_df(BEHAVIORAL_MODALITIES, mode, layer_sel)
            bar_path = _plot_group_effects_mixedlm(neural_long, behavioral_long, mode, layer_sel)
            if bar_path is not None:
                lines.append(f"\n### {mode.capitalize()} — {layer_sel.capitalize()} layer\n")
                lines.append(f"![Mixed-effects group comparison, {mode}, {layer_sel}](figures/{bar_path.name})\n")
    return "\n".join(lines)


def build_pareto_section() -> str:
    """Figures only: neural-behavioral Pareto fronts, vs model size (params).
    All 4 (mode x layer) combos in one 2x2-grid figure per Pareto pair, so
    each pair is one page instead of four."""
    lines = ["## Neural–Behavioral Trade-off (Pareto Front)\n"]
    panels = [(_neural_behavioral_merged(mode, layer_sel), f"{mode.capitalize()}, {layer_sel} layer")
              for mode in MODES for layer_sel in LAYERS]
    out = _draw_pareto_grid(
        panels, "BrainScore (mean fraction of noise ceiling) →",
        "BehavScore (mean fraction of noise ceiling) →",
        "Neural–behavioral trade-off and Pareto front (top-right = good at both)",
        "pareto_neural_behavioral_grid_n_params.png",
    )
    if out is not None:
        lines.append(f"\n![Neural-behavioral Pareto front grid](figures/{out.name})\n")

    lines.append("\n## Neural–Behavioral Trade-off: LPP fMRI vs Natural Stories SPR\n")
    panels_ns = [
        (_two_dataset_merged(_mode_slug("lpp_fmri", mode), _mode_slug("naturalstories_spr", mode), layer_sel),
         f"{mode.capitalize()}, {layer_sel} layer")
        for mode in MODES for layer_sel in LAYERS
    ]
    out_ns = _draw_pareto_grid(
        panels_ns, "LPP fMRI score (fraction of noise ceiling) →",
        "Natural Stories SPR score (fraction of noise ceiling) →",
        "LPP vs Natural Stories SPR trade-off and Pareto front (top-right = good at both)",
        "pareto_lpp_naturalstories_grid_n_params.png",
    )
    if out_ns is not None:
        lines.append(f"\n![LPP vs Natural Stories SPR Pareto front grid](figures/{out_ns.name})\n")
    return "\n".join(lines)


def build_pareto_section_zscore() -> str:
    """Robust-z-scored counterpart of build_pareto_section() -- 2026-09-07,
    added because the raw-average Pareto front lets whichever dataset has the
    largest absolute score_norm SCALE (not just a different noise ceiling)
    dominate a model's axis position: UCL Reading's behavioral baseline
    ~70x Provo's. Each dataset is
    robust-z-standardized across models BEFORE averaging into the composite
    axis here, so every dataset contributes by how much it discriminates
    between models, not by its raw magnitude. Read this section, not the raw
    one above, as this report's actual Pareto front."""
    lines = ["## Neural–Behavioral Trade-off (Pareto Front) — Z-Scored\n",
             "Same trade-off as the raw-average Pareto front above, but each dataset is robust-z-standardized "
             "(median/MAD) across models BEFORE being averaged into the Neural/Behavioral composite score, so no "
             "single dataset's absolute score_norm scale (e.g. UCL Reading) dominates a model's axis "
             "position. Axes are in SD units relative to the model population, not fraction of ceiling.\n"]
    panels = [(_neural_behavioral_merged_zscore(mode, layer_sel), f"{mode.capitalize()}, {layer_sel} layer")
              for mode in MODES for layer_sel in LAYERS]
    out = _draw_pareto_grid(
        panels, "BrainScore (mean robust z-score across datasets) →",
        "BehavScore (mean robust z-score across datasets) →",
        "Neural–behavioral trade-off and Pareto front, z-scored (top-right = good at both)",
        "pareto_neural_behavioral_grid_zscore.png",
    )
    if out is not None:
        lines.append(f"\n![Neural-behavioral Pareto front grid, z-scored](figures/{out.name})\n")
    return "\n".join(lines)


def build_scaling_section() -> str:
    """Figures only: scaling by dataset family, vs model size (params)."""
    lines = ["## Scaling by Dataset Family\n"]
    for mode in MODES:
        for layer_sel in LAYERS:
            fam_fig = build_family_scaling_figure(mode, layer_sel, x_col="n_params")
            if fam_fig is not None:
                lines.append(f"\n### {mode.capitalize()} — {layer_sel.capitalize()} layer\n")
                lines.append(f"![Scaling by dataset family, {mode}, {layer_sel}](figures/{fam_fig.name})\n")
    return "\n".join(lines)


def build_supplementary_perplexity_section() -> str:
    """Appendix counterpart of the three main-body figure sections, sized/
    plotted against perplexity instead of model size, plus the quantile-range
    bar plots (perplexity AND params rows, unsplit) and the Pareto-optimal
    model-list tables."""
    lines = ["## Supplementary: vs Perplexity\n"]

    lines.append("\n### Mean Performance by Model Group\n")
    for mode in MODES:
        for layer_sel in LAYERS:
            neural_df = _bucket_model_df(NEURAL_MODALITIES, mode, layer_sel)
            behavioral_df = _bucket_model_df(BEHAVIORAL_MODALITIES, mode, layer_sel)
            bar_path = _plot_neural_behavioral_bars(neural_df, behavioral_df, mode, layer_sel, size_col="perplexity")
            if bar_path is not None:
                lines.append(f"\n#### {mode.capitalize()} — {layer_sel.capitalize()} layer\n")
                lines.append(f"![Mean performance by group vs perplexity, {mode}, {layer_sel}](figures/{bar_path.name})\n")

    lines.append("\n### Neural–Behavioral Trade-off\n")
    panels_ppl = [(_neural_behavioral_merged(mode, layer_sel), f"{mode.capitalize()}, {layer_sel} layer")
                  for mode in MODES for layer_sel in LAYERS]
    out_ppl = _draw_pareto_grid(
        panels_ppl, "BrainScore (mean fraction of noise ceiling) →",
        "BehavScore (mean fraction of noise ceiling) →",
        "Neural–behavioral trade-off and Pareto front (top-right = good at both)",
        "pareto_neural_behavioral_grid_perplexity.png", size_col="perplexity",
    )
    if out_ppl is not None:
        lines.append(f"\n![Neural-behavioral Pareto front grid vs perplexity](figures/{out_ppl.name})\n")

    lines.append("\n### Scaling by Dataset Family\n")
    for mode in MODES:
        for layer_sel in LAYERS:
            fam_fig = build_family_scaling_figure(mode, layer_sel, x_col="perplexity")
            if fam_fig is not None:
                lines.append(f"\n#### {mode.capitalize()} — {layer_sel.capitalize()} layer\n")
                lines.append(f"![Scaling by dataset family vs perplexity, {mode}, {layer_sel}](figures/{fam_fig.name})\n")

    lines.append(f"\n### Mean Performance by Quantile Range ({N_QUANTILE_BINS} bins, perplexity & params)\n")
    for mode in MODES:
        for layer_sel in LAYERS:
            neural_df = _bucket_model_df(NEURAL_MODALITIES, mode, layer_sel)
            behavioral_df = _bucket_model_df(BEHAVIORAL_MODALITIES, mode, layer_sel)
            range_path = _plot_binned_bars(neural_df, behavioral_df, mode, layer_sel)
            if range_path is not None:
                lines.append(f"\n#### {mode.capitalize()} — {layer_sel.capitalize()} layer\n")
                lines.append(f"![Mean performance by range, {mode}, {layer_sel}](figures/{range_path.name})\n")

    lines.append("\n### Pareto-Optimal Model Lists\n")
    pareto_specs = [
        ("Neural vs Behavioral (modality mean)", build_pareto_figure, "Neural", "Behavioral"),
        ("LPP fMRI vs Natural Stories SPR", build_pareto_figure_lpp_ns, "LPP fMRI", "Natural Stories SPR"),
    ]
    for label, fn, xname, yname in pareto_specs:
        lines.append(f"\n#### {label}\n")
        for mode in MODES:
            for layer_sel in LAYERS:
                _, front = fn(mode, layer_sel)
                lines.append(f"\n**{mode.capitalize()}, {layer_sel} layer**\n")
                if front.empty:
                    lines.append("*(no models with both scores in this combination)*\n")
                    continue
                lines.append(f"| Model | Group | {xname} | {yname} | Params |")
                lines.append("|---|---|---|---|---|")
                for _, r in front.sort_values("neural", ascending=False).iterrows():
                    p = f"{r['n_params']/1e9:.0f}B" if r['n_params'] >= 1e9 else f"{r['n_params']/1e6:.0f}M"
                    lines.append(f"| {r['label']} | {r['group']} | {r['neural']:.3f} | {r['behav']:.3f} | {p} |")
    return "\n".join(lines)


def _dataset_blurb(slug: str) -> str:
    info = DATASETS_BY_SLUG[slug]
    df = _load_model_df(slug)
    lines = [f"### {info.name}\n"]
    lines.append(
        f"**Modality:** {info.modality}  \n"
        f"**Stimuli:** {info.stimuli}  \n"
        f"**N:** {info.n_subjects}  \n"
        f"**Metric:** {info.metric}\n"
    )
    if df is None or df.empty:
        lines.append("_No data available for this dataset._\n")
        return "\n".join(lines)
    df = df.dropna(subset=["score"]).copy()
    ceiling = ceiling_for(slug)
    if "score_norm" not in df.columns:
        df["score_norm"] = df["score"] / ceiling if ceiling else df["score"]
    best = df.loc[df["score"].idxmax()]
    if ceiling is None:
        # Delta metric (behavioral): no noise ceiling — report native units.
        group_rank = df.groupby("group")["score"].mean().sort_values(ascending=False)
        rank_str = "; ".join(f"{g} (mean={v:.4f})" for g, v in group_rank.items())
        lines.append(
            f"Best-scoring model: **{best['label']}** ({best['family']}, {best['group']} group), "
            f"score = {best['score']:.4f} ({info.metric}). This is a *delta* metric "
            f"(improvement in fit), so it is not normalised by a noise ceiling. "
            f"Group ranking by mean score: {rank_str}. N models evaluated: {len(df)}.\n"
        )
        return "\n".join(lines)
    # Rank groups by their mean fraction of the noise ceiling.
    group_rank = df.groupby("group")["score_norm"].mean().sort_values(ascending=False)
    rank_str = "; ".join(f"{g} ({v * 100:.0f}% of ceiling)" for g, v in group_rank.items())
    lines.append(
        f"Best-scoring model: **{best['label']}** ({best['family']}, {best['group']} group), "
        f"raw score = {best['score']:.4f} = **{best['score_norm'] * 100:.0f}% of the "
        f"noise ceiling** (`ceiling_r = {ceiling:.3f}`). "
        f"Group ranking by fraction of ceiling: {rank_str}. "
        f"N models evaluated: {len(df)}.\n"
    )
    return "\n".join(lines)


def _dataset_figures(slug: str) -> str:
    info = DATASETS_BY_SLUG[slug]
    src_dir = BENCHMARK_SUMMARY_ROOT / slug
    lines = []
    fig_specs = [
        ("overview.png", "Overview (scaling law / perplexity / layer profile / timecourse)"),
        ("scaling_by_group.png", "Scaling law by model category (LLM / Memory-Constrained / Constrained-BabyLM / BabyLM separately)"),
        ("roi_profile.png", "ROI / spatial profile"),
    ]
    for fname, caption in fig_specs:
        src = src_dir / fname
        if not src.exists():
            continue
        dest_name = f"{slug}_{fname}"
        rel = _copy_fig(src, dest_name)
        if rel:
            lines.append(f"**{caption}**\n\n![{info.name} {caption}]({rel})\n")
    return "\n".join(lines)


def build_dataset_deep_dive() -> str:
    lines = ["## Per-Dataset Deep Dive\n"]
    for info in PRIMARY_DATASETS:
        lines.append(_dataset_blurb(info.slug))
        lines.append(_dataset_figures(info.slug))
    return "\n".join(lines)


def build_dataset_summary_table() -> str:
    lines = ["## Datasets at a Glance\n"]
    lines.append("| # | Dataset | Modality | Stimuli | N | Metric |")
    lines.append("|---|---|---|---|---|---|")
    for i, info in enumerate(PRIMARY_DATASETS, start=1):
        lines.append(f"| {i} | {info.name} | {info.modality} | {info.stimuli} | {info.n_subjects} | {info.metric} |")
    lines.append(f"\n**{len(PRIMARY_DATASETS)} active datasets** spanning fMRI, EEG, iEEG, ECoG, and behavioral (self-paced reading) modalities.\n")
    return "\n".join(lines)


def build_distance_to_ceiling() -> str:
    """Table + figure: for each dataset, the best model's fraction of the noise
    ceiling and each group's mean fraction — where models saturate vs. where
    there is headroom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_ORDER

    lines = ["## Distance to Noise Ceiling\n"]
    rows = []
    for info in PRIMARY_DATASETS:
        df = _load_model_df(info.slug)
        ceiling = ceiling_for(info.slug)
        if df is None or df.empty or ceiling is None:
            continue
        df = df.dropna(subset=["score"]).copy()
        if "score_norm" not in df.columns:
            df["score_norm"] = df["score"] / ceiling
        best = df.loc[df["score"].idxmax()]
        gmeans = df.groupby("group")["score_norm"].mean()
        rows.append({
            "slug": info.slug, "name": info.name, "modality": info.modality,
            "ceiling": ceiling, "best_raw": best["score"],
            "best_frac": best["score_norm"], "best_model": best["label"],
            **{g: gmeans.get(g, float("nan")) for g in GROUP_ORDER},
        })
    if not rows:
        lines.append("_No ceiling data available._\n")
        return "\n".join(lines)
    rows.sort(key=lambda r: r["best_frac"], reverse=True)

    lines.append("| Dataset | Modality | ceiling_r | Best model | Best raw | Best % of ceiling |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['modality']} | {r['ceiling']:.3f} | {r['best_model']} "
            f"| {r['best_raw']:.4f} | **{r['best_frac']*100:.0f}%** |"
        )
    lines.append("")

    # Horizontal bar figure: best fraction per dataset, colored by best model's... group-agnostic (single bar/dataset).
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(rows))))
    names = [r["name"] for r in rows]
    fracs = [r["best_frac"] for r in rows]
    y = range(len(rows))
    ax.barh(list(y), fracs, color="#4C72B0", alpha=0.85)
    ax.axvline(1.0, ls="--", lw=1.4, color="#444", alpha=0.9)
    ax.text(1.0, len(rows) - 0.5, " noise ceiling", va="top", ha="left",
            fontsize=8, color="#444", style="italic")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Best model's score / noise ceiling")
    ax.set_xlim(0, max(1.08, max(fracs) * 1.05))
    ax.set_title("How close is the best model to each dataset's noise ceiling?")
    for i, f in enumerate(fracs):
        ax.text(f + 0.01, i, f"{f*100:.0f}%", va="center", fontsize=7.5)
    fig.tight_layout()
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIGURES_ROOT / "distance_to_ceiling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    lines.append(f"![Distance to noise ceiling](figures/{out.name})\n")

    lines.append(build_behavioral_reliability_table())
    return "\n".join(lines)


def build_behavioral_reliability_table() -> str:
    """Reading-time split-half reliability for the behavioral datasets.

    This IS the ceiling used to normalise every behavioral score in this report
    (score / ceiling_r, same as every neural dataset) — the headline behavioral
    metric is now a held-out delta-Pearson-r (a bounded correlation-delta), not
    the legacy delta-log-likelihood, so this raw-RT split-half reliability is a
    valid bound for it. See the Datasets and Metrics section above.
    """
    nc = load_noise_ceilings()
    lines = ["\n### Behavioral: reading-time split-half reliability\n"]
    behavioral_slugs = [s for s in DATASETS_BY_SLUG if MODALITY_BUCKET.get(s) == "Behavioral"]
    rows = []
    for slug in behavioral_slugs:
        info = DATASETS_BY_SLUG[slug]
        # Sub-measure slugs (geco_gp, meco_sp, ...) don't appear verbatim in
        # the "dataset" column -- the noise-ceiling CSV keys them as the base
        # dataset (geco/meco/...) with a "component" column (FP/GP/SP), the
        # same mapping REPORT_TO_CEILING/ceiling_for() already use for score
        # normalisation. A raw `nc["dataset"] == slug` match (the old code
        # here) silently missed every _gp/_sp slug and printed "failed to
        # compute" even though their ceilings were fine and already in use
        # elsewhere in the report (found 2026-09-08 -- a reader flagged the
        # GP/SP rows as looking like a missing-ceiling data gap).
        # NB: deliberately NOT ceiling_for(slug) here -- for surprisal/LME-
        # track datasets that also has a delta-log-likelihood ceiling and
        # checks it FIRST (a different scale entirely), which would silently
        # swap this table's "raw-RT split-half reliability" (0.5-0.9 range,
        # what the docstring above promises) for the much smaller DLL
        # ceiling. Read the correlation-scale ceiling_r straight out of the
        # noise-ceiling CSV via the same REPORT_TO_CEILING component mapping
        # ceiling_for() itself uses internally, just stopping short of its
        # DLL branch.
        sub = pd.DataFrame()
        if nc is not None:
            nc_slug, comps = REPORT_TO_CEILING.get(slug, (slug, None))
            sub = nc[nc["dataset"] == nc_slug]
            if comps is not None:
                sub = sub[sub["component"].astype(str).isin(comps)]
        ceiling_vals = pd.to_numeric(sub["ceiling_r"], errors="coerce").dropna() if len(sub) else pd.Series(dtype=float)
        ceiling_r = float(ceiling_vals.mean()) if len(ceiling_vals) else None
        rows.append({
            "name": info.name,
            "ceiling_r": ceiling_r,
            "n_subjects": int(sub["n_subjects"].iloc[0]) if len(sub) else None,
            "n_items": int(pd.to_numeric(sub["n_items"], errors="coerce").sum()) if len(sub) else None,
        })
    lines.append("| Dataset | Reading-time reliability (ceiling_r) | N subjects | N words/items |")
    lines.append("|---|---|---|---|")
    for r in rows:
        if r["ceiling_r"] is None:
            lines.append(f"| {r['name']} | _failed to compute_ | — | — |")
        else:
            lines.append(
                f"| {r['name']} | {r['ceiling_r']:.3f} | {r['n_subjects']} | {r['n_items']} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_model_registry_appendix() -> str:
    from neuroai.analysis.benchmark_common import family_group

    lines = ["## Model Registry Appendix\n"]

    lines.append("### Standard LLMs\n")
    lines.append("| Family | # Models | Sizes | Layers | Checkpoint sweep |")
    lines.append("|---|---|---|---|---|")
    for family in LLM_FAMILIES:
        specs = REGISTRY_INDEX_BY_FAMILY.get(family, [])
        if not specs:
            continue
        sizes = ", ".join(
            f"{s.n_params/1e6:.0f}M" if s.n_params < 1e9 else f"{s.n_params/1e9:.1f}B" for s in specs
        )
        layers = ", ".join(str(s.n_layers) for s in specs)
        sweep = "Training checkpoints" if specs[0].checkpoint_kind in ("pythia", "olmo2") else "Size sweep only"
        lines.append(f"| {family} | {len(specs)} | {sizes} | {layers} | {sweep} |")

    # MEMORY_CONSTRAINED_FAMILIES splits into two GROUPS depending on the
    # individual model, not the family: a "babylm" in the model_id makes it
    # Constrained-BabyLM, everything else (Pile-trained, etc.) stays
    # Memory-Constrained -- see family_group(). Split each family's specs
    # accordingly rather than listing the whole family under one heading.
    mem_specs_by_family: dict[str, list] = {}
    cbabylm_specs_by_family: dict[str, list] = {}
    for family in MEMORY_CONSTRAINED_FAMILIES:
        for spec in REGISTRY_INDEX_BY_FAMILY.get(family, []):
            bucket = cbabylm_specs_by_family if family_group(family, spec.model_id) == "Constrained-BabyLM" else mem_specs_by_family
            bucket.setdefault(family, []).append(spec)

    lines.append(
        "\n### Memory-Constrained Models\n\n"
        "Architectures with an explicitly bounded working-memory / context mechanism, "
        "trained on non-BabyLM data (Pile, etc.). These come from different lines of "
        "work, not a single paper: `alibi`, `stickbreaking`, and `forgetting` are "
        "attention-bias variants; `cbr_rnn` is a separate, unrelated recurrent "
        "architecture with a fixed-size memory state.\n"
    )
    lines.append("| Family | # Models | Sizes | Layers | Notes |")
    lines.append("|---|---|---|---|---|")
    for family, specs in mem_specs_by_family.items():
        sizes = ", ".join(
            f"{s.n_params/1e6:.0f}M" if s.n_params < 1e9 else f"{s.n_params/1e9:.1f}B" for s in specs
        )
        layers = ", ".join(str(s.n_layers) for s in specs)
        notes = "Pile-trained, 2L & 4L variants" if family != "cbr_rnn" else "Recurrent architecture, bounded memory state"
        lines.append(f"| {family} | {len(specs)} | {sizes} | {layers} | {notes} |")

    lines.append(
        "\n### Constrained-BabyLM Models\n\n"
        "The same bounded-memory architectures above (`alibi`, `stickbreaking`, "
        "`forgetting`, plus any `cbr_rnn`/other variant trained this way), but trained "
        "on BabyLM-10M / BabyLM-100M data instead of Pile -- kept as a separate group "
        "from both plain Memory-Constrained (different training data) and plain BabyLM "
        "(different, non-memory-bounded architectures).\n"
    )
    lines.append("| Family | # Models | Sizes | Layers | Notes |")
    lines.append("|---|---|---|---|---|")
    for family, specs in cbabylm_specs_by_family.items():
        sizes = ", ".join(
            f"{s.n_params/1e6:.0f}M" if s.n_params < 1e9 else f"{s.n_params/1e9:.1f}B" for s in specs
        )
        layers = ", ".join(str(s.n_layers) for s in specs)
        lines.append(f"| {family} | {len(specs)} | {sizes} | {layers} | BabyLM-10M / BabyLM-100M, 2L & 4L variants |")

    lines.append("\n### BabyLM Models\n")
    lines.append("| Model | Params | Layers |")
    lines.append("|---|---|---|")
    for family in BABYLM_FAMILIES:
        for spec in REGISTRY_INDEX_BY_FAMILY.get(family, []):
            size = f"{spec.n_params/1e6:.0f}M" if spec.n_params < 1e9 else f"{spec.n_params/1e9:.1f}B"
            lines.append(f"| {spec.label} | {size} | {spec.n_layers} |")

    n_llm = sum(len(REGISTRY_INDEX_BY_FAMILY.get(f, [])) for f in LLM_FAMILIES)
    n_mem = sum(len(v) for v in mem_specs_by_family.values())
    n_cbabylm = sum(len(v) for v in cbabylm_specs_by_family.values())
    n_baby = sum(len(REGISTRY_INDEX_BY_FAMILY.get(f, [])) for f in BABYLM_FAMILIES)
    lines.append(
        f"\n**Totals:** {n_llm} Standard LLM model variants, {n_mem} Memory-Constrained "
        f"model variants, {n_cbabylm} Constrained-BabyLM model variants, {n_baby} BabyLM "
        f"model variants ({n_llm + n_mem + n_cbabylm + n_baby} total).\n"
    )
    return "\n".join(lines)


def build_caveats() -> str:
    from neuroai.models.registry import LPP_FAMILIES
    from neuroai.analysis.benchmark_common import family_group

    gtot = {g: 0 for g in GROUP_ORDER}
    for fam, specs in LPP_FAMILIES.items():
        # Per-MODEL, not per-family: Memory-Constrained families split into two
        # groups (Constrained-BabyLM vs Memory-Constrained) depending on the
        # individual model_id, so a family-level count can't distinguish them.
        for spec in specs:
            gtot[family_group(fam, spec.model_id)] += 1

    lines = ["## Caveats\n"]
    lines.append(
        f"**Group sizes:** {gtot['LLM']} LLM, {gtot['Memory-Constrained']} Memory-Constrained, "
        f"{gtot['Constrained-BabyLM']} Constrained-BabyLM, {gtot['BabyLM']} BabyLM.\n"
    )

    # Base dataset name (without the "(embedding)"/"(surprisal)" suffix used by
    # secondary-mode DatasetInfo rows), keyed by base slug. Primary rows
    # (base_slug is None) give the clean name; secondary-only bases fall back
    # to stripping the trailing "(...)" from their own name.
    base_names: dict[str, str] = {}
    for d in DATASETS:
        if d.base_slug is None:
            base_names[d.base] = d.name
    for d in DATASETS:
        base_names.setdefault(d.base, d.name.split(" (")[0])
    n_subjects_by_base: dict[str, str] = {}
    for d in PRIMARY_DATASETS:
        n_subjects_by_base[d.base] = d.n_subjects
    order_bases = list(dict.fromkeys(d.base for d in PRIMARY_DATASETS))

    def _coverage_table(mode: str) -> tuple[list[str], list[tuple[str, dict[str, int], str]]]:
        rows = []
        for base in order_bases:
            name = base_names[base]
            n_subj = n_subjects_by_base.get(base, "")
            slug = _mode_slug(base, mode)
            if slug is None:
                rows.append((name, {g: 0 for g in GROUP_ORDER}, "n/a"))
                continue
            df = _load_model_df(slug)
            if df is None:
                rows.append((name, {g: 0 for g in GROUP_ORDER}, n_subj))
                continue
            counts = df["group"].value_counts()
            rows.append((name, {g: int(counts.get(g, 0)) for g in GROUP_ORDER}, n_subj))
        table_lines = [f"### {mode.capitalize()} track\n"]
        table_lines.append("| Dataset | " + " | ".join(f"N ({g})" for g in GROUP_ORDER) + " | N subjects/electrodes |")
        table_lines.append("|---|" + "---|" * len(GROUP_ORDER) + "---|")
        for name, counts, n_subj in rows:
            table_lines.append(f"| {name} | " + " | ".join(str(counts[g]) for g in GROUP_ORDER) + f" | {n_subj} |")
        table_lines.append("")
        return table_lines, rows

    all_rows_for_missing: list[tuple[str, dict[str, int], str]] = []
    for mode in MODES:
        table_lines, rows = _coverage_table(mode)
        lines.extend(table_lines)
        # "n/a" (no track for that dataset in this mode) is excluded from the
        # missing-coverage summary below -- that's not a gap, it's expected
        # (e.g. Natural Stories SPR has no embedding track).
        all_rows_for_missing.extend((name, counts, n_subj) for name, counts, n_subj in rows if n_subj != "n/a")

    missing_by_group = {g: [name for name, counts, _ in all_rows_for_missing if counts[g] == 0] for g in GROUP_ORDER}
    cov_bits = [f"**no {g} models** yet: {', '.join(names)}"
                for g, names in missing_by_group.items() if names]
    if cov_bits:
        lines.append("\n**Missing group coverage:** " + "; ".join(cov_bits) + ".\n")
    else:
        lines.append(f"\n**Group coverage:** every dataset/track has all {len(GROUP_ORDER)} model groups.\n")
    lines.append(
        "\n**Notes:** Natural Stories fMRI perplexity reuses Natural Stories SPR's perplexity "
        "(same 8 stories). ZuCo's within-subject score vs. between-subject ceiling can exceed "
        "100% (approximate ceiling, lower bound); channel 104 excluded (degenerate). All scores "
        "are shown as fraction of `ceiling_r` (sqrt split-half Spearman-Brown reliability); "
        "iEEG/ECoG ceilings are region-level; Tuckute D/S conditions are unstable (2-3 "
        "subjects/item); LPP ceiling is high by construction (average-subject target).\n"
    )
    return "\n".join(lines)


def main() -> None:
    global REGISTRY_INDEX_BY_FAMILY
    from neuroai.models.registry import LPP_FAMILIES

    REGISTRY_INDEX_BY_FAMILY = LPP_FAMILIES

    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    sections = []
    sections.append("# Working Memory Benchmark\n")

    # --- main body: figures only, top-down, vs model size -----------------
    sections.append(build_bar_plots_section())
    sections.append(build_group_effects_mixedlm_section())
    sections.append(build_per_dataset_group_bars_section())
    sections.append(build_roi_section())
    sections.append(build_pareto_section())
    sections.append(build_pareto_section_zscore())
    sections.append(build_hard_case_pareto_section())
    sections.append(build_scaling_section())
    sections.append(build_average_behavioral_scaling_section())
    sections.append(build_dlt_hard_word_section())
    sections.append(build_pythia_checkpoint_section(variant="perplexity"))

    # --- appendix: tables + supplementary (vs perplexity/size) figures ----
    sections.append("\n\n---\n\n# Appendix\n")
    sections.append(build_methods_table())
    sections.append(build_model_registry_appendix())
    sections.append(build_dataset_summary_table())
    sections.append(build_distance_to_ceiling())
    sections.append(build_caveats())
    sections.append(build_feature_mode_section())
    sections.append(build_pythia_checkpoint_section(variant="size"))
    sections.append(build_supplementary_perplexity_section())

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_ROOT / "benchmark_report.md"
    report_path.write_text("\n".join(sections))
    print(f"Wrote {report_path}")

    pdf_path = REPORTS_ROOT / "benchmark_report.pdf"
    convert_report_to_pdf(report_path, pdf_path)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
