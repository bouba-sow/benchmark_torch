#!/usr/bin/env python3
"""RDM-regression track: Pareto fronts, scaling plots and a small report (last layer only).

Reads the joined table written by scripts/analysis/rdm_report.py and writes, into reports/rsa_summary/:
  rdm_pareto_{embedding,surprisal}_{confounds,noconf}.{pdf,png}   neural vs behavioral composite
  rdm_scaling_{embedding,surprisal}_{confounds,noconf}.{pdf,png}  delta LL vs model size, one panel per dataset/measure
  rdm_scaling_summary.csv, rdm_pareto_scores.csv, rdm_regression_report.md

Score ("confounds" = delta LL beyond word length + frequency, "noconf" = plain delta LL), last layer.
* Pareto front (same layout as scripts/figures/pareto_figures.py: neural composite vs behavioral composite,
  bubble size = log parameters, colour = model group, black front): each unit's score is normalised by its
  noise ceiling and a model's composite is the MEDIAN over units.
    - every unit: FULL-sample last-layer delta LL / split-half ceiling (Timkey et al. 2026 recipe in RDM space; the
      reference is built from half the participants, so ratios can exceed 1). Datasets without a usable ceiling are
      left out; the rule never looks at model scores (--ceilings rdm_ceiling_values.csv).
  Units: every reading measure and every neural dataset counts once (as in the benchmark's dataset list).
* Scaling plots: raw last-layer delta LL per pair against log10(parameters), OLS line over non-Pythia models
  (descriptive, like the existing RSA scaling figure); no ceiling normalisation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullLocator
from scipy import stats

from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, REPO_ROOT, family_group, lookup_model_meta
from neuroai.analysis.paper_style import log_size_map, use_paper_style

GROUP_COLORS = {**GROUP_COLORS, "Constrained-BabyLM": "#ff7f0e"}  # same palette as build_rsa_report.py / figure3, figure6

REPORT = REPO_ROOT / "reports" / "rsa_summary"
OUT = REPORT / "rdm_regression"  # tables, report text and the raw scaling panels
FIG, FIGPDF = REPORT / "figures", REPORT / "figures_pdf"  # the six headline figures
# delta LL is always defined against the confound-only model (word length + frequency RDMs)
VARIANTS = {"len_freq": ""}
VARIANT_TEXT = {"len_freq": "beyond word length + frequency"}
NEURAL_UNITS = ["pereira", "tuckute", "podcast", "broderick", "megmasc",
                "nieuwland_avg", "szewczyk_f2007_avg", "szewczyk_h2019_avg"]
TITLES = {"pereira": "Pereira · fMRI", "tuckute": "Tuckute B · fMRI", "fedorenko2016": "Fedorenko · ECoG",
          "podcast": "Podcast · ECoG", "broderick": "Broderick · EEG", "megmasc": "MEG-MASC · MEG",
          "nieuwland_avg": "Nieuwland · N400 (subject mean)",
          "szewczyk_f2007_avg": "Federmeier et al. (2007) · N400 (subject mean)",
          "szewczyk_h2019_avg": "Hubbard et al. (2019) · N400 (subject mean)"}
BEHAV_TITLES = {"provo": "Provo", "meco": "MECO", "geco": "GECO", "onestop": "OneStop",
                "naturalstories": "Natural Stories", "ucl_reading": "UCL Reading", "zuco_reading": "ZuCo"}
# panel layout (4 columns) grouped like the existing RSA overview
LAYOUT = [
    [("pereira", "all"), ("tuckute", "all"), ("podcast", "all"), ("broderick", "all")],
    [("megmasc", "all"), ("nieuwland_avg", "all"), ("szewczyk_f2007_avg", "all"), ("szewczyk_h2019_avg", "all")],
    [("provo", "FP"), ("meco", "FP"), ("geco", "FP"), ("onestop", "FP")],
    [("provo", "GP"), ("meco", "GP"), ("geco", "GP"), ("onestop", "GP")],
    [("provo", "SP"), ("meco", "SP"), ("onestop", "SP"), None],
    [("naturalstories", "SPR"), ("ucl_reading", "SPR"), None, None],
    [("zuco_reading", "NR"), ("zuco_reading", "SR"), ("zuco_reading", "TSR"), None],
]
EXCLUDED_DATASETS = ["fedorenko2016"]  # no reliable cross-subject signal at the RDM level (ceiling < 0); excluded 2026-10-07
RSA_NOTE = ""  # set by scripts/figures/rsa_pareto.py
MIN_COVERAGE = 0.75
EXCLUDED: list = []  # datasets left out of the Pareto composites (set in main)


def _group(family, model_id, stored):
    """Current registry group (neuroai.analysis.benchmark_common.family_group); falls back to the stored one for non-registry rows."""
    try:
        return family_group(family, model_id)
    except KeyError:
        return stored


def panel_title(dataset: str, unit: str) -> str:
    return TITLES[dataset] if dataset in TITLES else f"{BEHAV_TITLES[dataset]} · {unit}"


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[~d.dataset.isin(["nieuwland", "szewczyk", "szewczyk_avg", *EXCLUDED_DATASETS])].copy()
    d["n_params"] = pd.to_numeric(d.n_params, errors="coerce")
    d["group"] = [_group(f, m, g) for f, m, g in zip(d.family, d.model_id, d.group)]  # current registry grouping, not the one stored in the table
    d["norm"] = np.nan
    sh = d.ceiling_type == "split_half_half_target"
    d.loc[sh, "norm"] = d.loc[sh, "ratio_to_ceiling"]
    lo = d.ceiling_type == "loso_same_target"
    d.loc[lo, "norm"] = d.loc[lo, "ratio_to_upper"]
    d["is_neural"] = d.dataset.isin(NEURAL_UNITS)
    d["unit_key"] = d.dataset + "/" + d.unit
    return d


UNIT_DATASET = {"nieuwland": "nieuwland_avg", "szewczyk_f2007": "szewczyk_f2007_avg", "szewczyk_h2019": "szewczyk_h2019_avg"}


def with_neural_units(d: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Replace the neural rows by the per-unit design (docs/rdm_regression_neural_units.md). No noise ceilings exist for it yet."""
    n = pd.read_csv(path, dtype={"revision": str})
    n = n[~n.dataset.isin(EXCLUDED_DATASETS)]
    meta = [lookup_model_meta(m, f) for m, f in zip(n.model_id, n.family)]
    new = pd.DataFrame({"dataset": n.dataset.map(lambda x: UNIT_DATASET.get(x, x)), "track": n.track, "family": n.family,
                        "model_id": n.model_id, "revision": n.revision, "unit": "all", "confounds": "len_freq",
                        "last_layer_delta_ll_per_obs": n.delta_ll_per_obs, "group": [m.group for m in meta],
                        "n_params": [m.n_params for m in meta]})
    new["norm"] = np.nan
    new["is_neural"] = True
    new["unit_key"] = new.dataset + "/" + new.unit
    keep = d[~d.is_neural]
    return pd.concat([keep, new], ignore_index=True)


def apply_ceilings(d: pd.DataFrame, path: Path) -> pd.DataFrame:
    """norm = last-layer dLL / split-half noise ceiling (rdm_ceiling_table.py), for the Pareto composites.
    Only datasets without a usable (finite, positive) ceiling are left out (norm = NaN); the rule never looks at model
    scores. Ratios above 1 are kept: the numerator is the full-sample model score and the ceiling a half-sample
    split-half reference, so ratios can exceed 1."""
    cv = pd.read_csv(path)
    cv["dataset"] = cv.dataset_key.map(lambda v: UNIT_DATASET.get(v, v))
    cv["unit"] = cv.unit.astype(str)
    d = d.drop(columns=["norm"], errors="ignore").merge(
        cv[["dataset", "unit", "ceiling_dll", "excluded_from_pareto"]], on=["dataset", "unit"], how="left")
    ok = (d.confounds == "len_freq") & (~d.excluded_from_pareto.fillna(True).astype(bool)) & (d.ceiling_dll > 0)
    d["norm"] = np.where(ok, d.last_layer_delta_ll_per_obs / d.ceiling_dll, np.nan)
    d.attrs["excluded"] = sorted(cv.loc[cv.excluded_from_pareto, "dataset"].unique())
    return d


def attach_benchmark(d: pd.DataFrame) -> pd.DataFrame:
    """Benchmark score per model: embedding = ridge Pearson r, surprisal = LME test dLL / obs (rsa/model_scores_*.csv)."""
    inv = {v: k for k, v in UNIT_DATASET.items()}
    out = []
    for track, name in (("embedding", "model_scores_correlation.csv"), ("surprisal", "model_scores_dll.csv")):
        sc = pd.read_csv(REPORT / "rsa" / name, dtype={"revision": str})
        sc = sc[sc.track == track][["dataset", "measure", "family", "model_id", "revision", "benchmark_score"]].copy()
        sc["measure"] = sc.measure.fillna("all").astype(str).str.upper().replace({"ALL": "all"})
        x = d[d.track == track].copy()
        x["dataset_b"] = x.dataset.map(lambda v: inv.get(v, v))
        x["unit_b"] = x.unit.astype(str).str.upper().replace({"ALL": "all"})
        x = x.merge(sc, left_on=["dataset_b", "unit_b", "family", "model_id", "revision"],
                    right_on=["dataset", "measure", "family", "model_id", "revision"], how="left", suffixes=("", "_sc"))
        out.append(x)
    return pd.concat(out, ignore_index=True)


def draw_benchmark(d: pd.DataFrame, track: str, variant: str) -> None:
    """RDM-regression delta LL (y) versus the benchmark score (x), one panel per dataset/measure, across models."""
    ncols, nrows = 4, len(LAYOUT)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.4, nrows * 3.4), squeeze=False)
    xlabel = "Ridge encoding Pearson r (benchmark)" if track == "embedding" else "LME ΔLL / test observation (benchmark)"
    for ri, row in enumerate(LAYOUT):
        for ci, panel in enumerate(row):
            ax = axes[ri][ci]
            if panel is None:
                ax.set_visible(False)
                continue
            ds, unit = panel
            sub = d[(d.dataset == ds) & (d.unit == unit) & (d.track == track) & (d.confounds == variant)]
            sub = sub[sub.last_layer_delta_ll_per_obs.notna() & sub.benchmark_score.notna()]
            ax.set_title(panel_title(ds, unit), fontsize=11, fontweight="bold", loc="left", pad=8)
            if sub.empty:
                ax.text(.5, .5, "no results", ha="center", va="center", transform=ax.transAxes, color="#777777")
                continue
            for g in GROUP_ORDER:
                sg = sub[sub.group == g]
                if len(sg):
                    ax.scatter(sg.benchmark_score, sg.last_layer_delta_ll_per_obs, s=18, alpha=.7, color=GROUP_COLORS[g],
                               marker="o" if track == "embedding" else "^", linewidths=0, rasterized=True)
            x, y = sub.benchmark_score.to_numpy(float), sub.last_layer_delta_ll_per_obs.to_numpy(float)
            ax.text(.02, .985, f"n={len(sub)}", transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
            k = 0
            for g in GROUP_ORDER:
                sg = sub[sub.group == g]
                xg, yg = sg.benchmark_score.to_numpy(float), sg.last_layer_delta_ll_per_obs.to_numpy(float)
                if len(sg) >= 5 and len(np.unique(xg)) >= 3:
                    fg = stats.linregress(xg, yg)
                    xs = np.linspace(xg.min(), xg.max(), 50)
                    ax.plot(xs, fg.intercept + fg.slope * xs, color=GROUP_COLORS[g], lw=1.6, zorder=3)
                    k += 1
                    ax.text(.02, .985 - .062 * k, f"{GROUP_DISPLAY.get(g, g)}: r={fg.rvalue:+.2f} (n={len(sg)})",
                            transform=ax.transAxes, va="top", fontsize=7, color=GROUP_COLORS[g],
                            bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=.5))
            lo_, hi_ = np.percentile(y, [1, 99])
            pad = 0.3 * (hi_ - lo_) if hi_ > lo_ else 1e-6
            ax.set_ylim(lo_ - pad, hi_ + pad)
            n_hidden = int(((y < lo_ - pad) | (y > hi_ + pad)).sum())
            if n_hidden:
                ax.text(.02, .02, f"{n_hidden} point(s) outside view", transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=7.5, color="#b00020")
            ax.axhline(0, color="#cccccc", linewidth=.6, zorder=0)
            ax.grid(alpha=.13)
            ax.set_xlabel(xlabel, fontsize=8.5)
            ax.set_ylabel("RDM-regression ΔLL per pair", fontsize=8.5)
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)
    handles = [Line2D([], [], marker="o" if track == "embedding" else "^", linestyle="None", color=GROUP_COLORS[g], markersize=6,
                      label=GROUP_DISPLAY.get(g, g)) for g in GROUP_ORDER]
    fig.suptitle(f"{track.capitalize()} · RDM-regression ΔLL versus benchmark · last layer", fontsize=18, y=.995)
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(.5, .975), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .02, 1, .955), h_pad=2.0, w_pad=2.2)
    fig.text(.5, .008, "Each point is a model. ΔLL: held-out gain over the confound-only RDM regression. One line (and r) per model category; descriptive; "
             "panel axes vary.", ha="center", fontsize=9, color="#555555")
    for suf in ("png", "pdf"):
        fig.savefig(OUT / f"rdm_dll_vs_benchmark_{track}.{suf}", dpi=180, facecolor="white")
    plt.close(fig)


def pareto_front(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    keep = []
    for i, r in df.iterrows():
        dominated = ((df[xcol] >= r[xcol]) & (df[ycol] >= r[ycol]) & ((df[xcol] > r[xcol]) | (df[ycol] > r[ycol]))).any()
        if not dominated:
            keep.append(i)
    return df.loc[keep].sort_values(xcol)


def composites(d: pd.DataFrame, track: str, variant: str, col: str = "norm") -> pd.DataFrame:
    s = d[(d.track == track) & (d.confounds == variant) & d[col].notna()]
    n_neural = s[s.is_neural].unit_key.nunique()
    n_behav = s[~s.is_neural].unit_key.nunique()
    key = ["family", "model_id", "revision"]
    neural = s[s.is_neural].groupby(key).agg(neural=(col, "median"), n_neural=("unit_key", "nunique"))
    behav = s[~s.is_neural].groupby(key).agg(behav=(col, "median"), n_behav=("unit_key", "nunique"))
    meta = s.groupby(key).agg(group=("group", "first"), n_params=("n_params", "first"))
    m = meta.join(neural).join(behav).reset_index()
    m = m[(m.n_neural >= MIN_COVERAGE * n_neural) & (m.n_behav >= MIN_COVERAGE * n_behav)].dropna(subset=["neural", "behav"])
    m["label"] = [lookup_model_meta(i, f).label for i, f in zip(m.model_id, m.family)]
    m.attrs.update(n_neural_units=n_neural, n_behav_units=n_behav)
    return m


def draw_pareto(m: pd.DataFrame, track: str, variant: str, mode: str = "ceiling") -> Path:
    front = pareto_front(m, "neural", "behav")
    v = m.n_params[m.n_params > 0].astype(float)
    lo, hi = float(np.log10(v.min())), float(np.log10(v.max()))
    sizes = log_size_map(m.n_params, lo, hi, s_min=25, s_max=400)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    for g in GROUP_ORDER:
        mask = (m.group == g).to_numpy()
        if mask.any():
            ax.scatter(m.neural[mask], m.behav[mask], s=sizes[mask], color=GROUP_COLORS[g], alpha=0.6,
                       edgecolors="white", linewidths=0.5, zorder=2, marker="o")
    if len(front) > 1:
        ax.plot(front.neural, front.behav, color="black", lw=1.8, zorder=3, alpha=0.9)
    # front points are numbered on the plot; the key (left to right along the front) sits beside the legends
    names = []
    for k, (_, r) in enumerate(front.sort_values("neural").iterrows()):
        ax.scatter([r.neural], [r.behav], s=90, facecolors="none", edgecolors="black", linewidths=1.8, zorder=4)
        ax.annotate(str(k + 1), (r.neural, r.behav), fontsize=7.5, fontweight="bold", xytext=(5, 6 + 7 * (k % 2)),
                    textcoords="offset points", zorder=6, bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=.6))
        names.append(f"{k + 1:>2}  {r.label}")
    ax.text(1.02, 1.0, "Pareto front\n(left to right)\n\n" + "\n".join(names), transform=ax.transAxes, va="top", ha="left",
            fontsize=7, family="monospace", linespacing=1.35)
    what = {"ceiling": "ΔLL / noise ceiling", "rsa": "Pearson RSA / noise ceiling"}.get(mode, "robust z-scored ΔLL")
    ax.set_xlabel(f"BrainScore (median over units of {what})", fontsize=12)
    ax.set_ylabel(f"BehavScore (median over units of {what})", fontsize=12)
    ax.set_title(f"{track.capitalize()} · {VARIANT_TEXT[variant]} · last layer", fontsize=11, loc="left")
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", linestyle="None", color=GROUP_COLORS[g], markeredgecolor="white",
                      markersize=8, label=GROUP_DISPLAY.get(g, g)) for g in GROUP_ORDER if (m.group == g).any()]
    refs = [r for r in (1e7, 1e8, 1e9, 1e10, 7e10) if lo <= np.log10(r) <= hi]
    size_handles = [Line2D([], [], marker="o", linestyle="None", markerfacecolor="grey", markeredgecolor="white", alpha=0.65,
                           markersize=float(np.sqrt(log_size_map(pd.Series([r]), lo, hi)[0])),
                           label=f"{r / 1e9:.0f}B" if r >= 1e9 else f"{r / 1e6:.0f}M") for r in refs]
    # legends inside the axes, as in the paper LME / ridge Pareto figures: surprisal both at the bottom right (stacked),
    # embedding both in the top right (side by side, in added headroom)
    top = track == "embedding"
    corner = "upper right" if top else "lower right"
    if top:
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.28 * (y1 - y0))
    (x_size, y_group, y_size) = (0.66, 1.0, 1.0) if top else (1.0, 0.30, 0.0)
    leg = ax.legend(handles=handles, fontsize=8, title_fontsize=8, title="Model group", loc=corner, bbox_to_anchor=(1.0, y_group), frameon=True)
    ax.add_artist(leg)
    if size_handles:
        ax.legend(handles=size_handles, title="model size", fontsize=8, title_fontsize=8, loc=corner, bbox_to_anchor=(x_size, y_size),
                  frameon=True, labelspacing=1.2, borderpad=0.9)
        ax.add_artist(leg)
    note = (RSA_NOTE if mode == "rsa" else
            "median over units of last-layer ΔLL / split-half noise ceiling (Timkey et al.). Fedorenko excluded (no reliable RDM-level ceiling)."
            if mode == "ceiling" else
            "median over units of the last-layer ΔLL robust z-scored, (x - median) / (1.4826 MAD), across the models of this track within each "
            "dataset/measure; no noise ceiling, all datasets included.")
    fig.text(0.01, 0.005, f"{len(m)} models; {m.attrs['n_neural_units']} neural and {m.attrs['n_behav_units']} behavioral units, " + note,
             fontsize=6.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 0.84, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    FIGPDF.mkdir(parents=True, exist_ok=True)
    suffix = {"ceiling": "", "rsa": "_rsa"}.get(mode, "_zscore")
    fig.savefig(FIG / f"{track}_pareto{suffix}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(FIGPDF / f"{track}_pareto{suffix}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return FIG / f"{track}_pareto{suffix}.png"


def draw_scaling(d: pd.DataFrame, track: str, variant: str, records: list) -> None:
    ncols, nrows = 4, len(LAYOUT)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.4, nrows * 3.4), squeeze=False)
    for ri, row in enumerate(LAYOUT):
        for ci, panel in enumerate(row):
            ax = axes[ri][ci]
            if panel is None:
                ax.set_visible(False)
                continue
            ds, unit = panel
            sub = d[(d.dataset == ds) & (d.unit == unit) & (d.track == track) & (d.confounds == variant)]
            sub = sub[np.isfinite(sub.n_params) & (sub.n_params > 0) & sub.last_layer_delta_ll_per_obs.notna()]
            # Pythia checkpoints are a training trajectory, not independent model-size
            # observations; exclude them from both the scaling plot and its fit.
            sub = sub[sub.family != "pythia"]
            ax.set_title(panel_title(ds, unit), fontsize=11, fontweight="bold", loc="left", pad=8)
            if sub.empty:
                ax.text(.5, .5, "no results", ha="center", va="center", transform=ax.transAxes, color="#777777")
                continue
            for g in GROUP_ORDER:
                s = sub[sub.group == g]
                if len(s):
                    ax.scatter(s.n_params, s.last_layer_delta_ll_per_obs, s=18, alpha=.7, color=GROUP_COLORS[g],
                               linewidths=0, rasterized=True)
            x, y = np.log10(sub.n_params.to_numpy(float)), sub.last_layer_delta_ll_per_obs.to_numpy(float)
            slope = r = p = np.nan
            if len(np.unique(x)) >= 3:
                fit = stats.linregress(x, y)
                slope, r, p = fit.slope, fit.rvalue, fit.pvalue
            group_fits = []
            for g in GROUP_ORDER:
                sg = sub[sub.group == g]
                xg, yg = np.log10(sg.n_params.to_numpy(float)), sg.last_layer_delta_ll_per_obs.to_numpy(float)
                if len(sg) >= 5 and len(np.unique(xg)) >= 3:
                    fg = stats.linregress(xg, yg)
                    xs = np.linspace(xg.min(), xg.max(), 50)
                    ax.plot(10 ** xs, fg.intercept + fg.slope * xs, color=GROUP_COLORS[g], lw=1.6, zorder=3)
                    group_fits.append((g, fg.slope, fg.rvalue, fg.pvalue, len(sg)))
            # robust y view: ignore a few extreme outliers (they would flatten the panel) and say how many are hidden
            lo_, hi_ = np.percentile(y, [1, 99])
            pad = 0.3 * (hi_ - lo_) if hi_ > lo_ else 1e-6
            ax.set_ylim(lo_ - pad, hi_ + pad)
            n_hidden = int(((y < lo_ - pad) | (y > hi_ + pad)).sum())
            ceiling = sub.ceiling_delta_ll_per_obs.dropna()
            ceil_up = sub.ceiling_upper_delta_ll_per_obs.dropna() if "ceiling_upper_delta_ll_per_obs" in sub else pd.Series(dtype=float)
            ax.text(.02, .985, f"n={len(sub)}", transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
            for k, (g, gs, gr, gp, gn) in enumerate(group_fits):
                ax.text(.02, .985 - .062 * (k + 1), f"{GROUP_DISPLAY.get(g, g)}: r={gr:+.2f} (n={gn})", transform=ax.transAxes,
                        va="top", fontsize=7, color=GROUP_COLORS[g], bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=.5))
            if n_hidden:
                ax.text(.02, .02, f"{n_hidden} point(s) outside view", transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=7.5, color="#b00020")
            if len(ceiling):
                txt = f"ceiling {ceiling.iloc[0]:.1e}" + (f" (upper {ceil_up.iloc[0]:.1e})" if len(ceil_up) else "")
                ax.text(.98, .02, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555555")
            for g, gs, gr, gp, gn in group_fits:
                records.append({"dataset": ds, "unit": unit, "track": track, "confounds": variant, "group": g, "n_models": gn,
                                "slope_per_10x_params": gs, "pearson_r_log10_params": gr, "p_value": gp})
            records.append({"dataset": ds, "unit": unit, "track": track, "confounds": variant, "group": "all", "n_models": len(sub),
                            "slope_per_10x_params": slope, "pearson_r_log10_params": r, "p_value": p,
                            "n_points_outside_view": n_hidden,
                            "median_delta_ll": float(np.median(y)), "max_delta_ll": float(np.max(y)),
                            "ceiling": float(ceiling.iloc[0]) if len(ceiling) else np.nan,
                            "ceiling_upper": float(ceil_up.iloc[0]) if len(ceil_up) else np.nan})
            ax.set_xscale("log")
            ax.axhline(0, color="#cccccc", linewidth=.6, zorder=0)
            ax.grid(alpha=.13)
            ax.set_xlabel("Model parameters (log scale)")
            ax.set_ylabel("ΔLL per pair")
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    handles = [Line2D([], [], marker="o", linestyle="None", color=GROUP_COLORS[g], markersize=6,
                      label=GROUP_DISPLAY.get(g, g)) for g in GROUP_ORDER]
    handles += [Line2D([], [], color="#444444", lw=1.6, label="linear fit per model category")]
    fig.suptitle(f"{track.capitalize()} ΔLL by model size · {VARIANT_TEXT[variant]} · last layer", fontsize=18, y=.995)
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(.5, .975), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .02, 1, .955), h_pad=2.0, w_pad=2.2)
    fig.text(.5, .008, "Held-out RDM ΔLL per pair, last layer. One OLS line per model category on log10(parameters) (r in each panel; Pythia "
             "excluded); descriptive, not a scaling-law claim. "
             "Panels ignore extreme outliers (count shown in red). Ceilings shown for reference.",
             ha="center", fontsize=9, color="#555555")
    base = OUT / (f"rdm_scaling_{track}" + (f"_{VARIANTS[variant]}" if VARIANTS[variant] else ""))
    for suf in ("pdf", "png"):
        fig.savefig(f"{base}.{suf}", dpi=180, facecolor="white")
    plt.close(fig)


def write_report(d: pd.DataFrame, pareto: dict, scaling: pd.DataFrame) -> None:
    has_ceil = "ceiling_dll" in d.columns
    lines = ["# RDM-regression track: Pareto fronts and scaling (last layer)", "",
             "Held-out ΔLL of a model RDM (embedding: last layer; surprisal: |Δ surprisal|) predicting the neural or "
             "behavioral RDM, scored with item-fold cross-validation (Schütt et al. 2023; Kaniuth & Hebart), "
             "always against the confound-only baseline: ΔLL = LL(nuisance RDMs + model RDM) "
             "minus LL(nuisance RDMs), held out. Complementary to the LME "
             "headline. Method and validation: `docs/rdm_regression.md`, `docs/rdm_regression_neural_units.md`.", "",
             "## Figures", ""]
    for track in ("embedding", "surprisal"):
        lines += [f"* {track.capitalize()}: [Pareto](rdm_pareto_{track}.pdf) · [scaling](rdm_scaling_{track}.pdf)"]
    lines += ["", "![Embedding Pareto front](rdm_pareto_embedding.png)",
              "![Surprisal Pareto front](rdm_pareto_surprisal.png)", "",
              "![Embedding scaling](rdm_scaling_embedding.png)",
              "![Surprisal scaling](rdm_scaling_surprisal.png)", "",
              "## How the scores are built", "",
              "* Pareto axes: each unit's last-layer ΔLL divided by that unit's split-half noise ceiling, median over units "
              "(every reading measure and neural dataset is one unit). The numerator is the FULL-sample model score, the "
              "ceiling a HALF-sample reference (Timkey et al. 2026 recipe in RDM space: the item-mean RDM of one random "
              "half of the participants predicts the other half's, both directions summed, 15 splits, same confound RDMs "
              "and same pair likelihood). The reference is built from half the participants, so it is conservative and "
              "ratios can exceed 1; they are not fractions of explainable variance. Datasets are left out only when they "
              "have no usable ceiling (Fedorenko: subject-specific electrodes), never because of model scores. The "
              "per-subject Nieuwland and legacy Szewczyk runs are not used; the subject-averaged Nieuwland, Federmeier 2007 "
              "and Hubbard 2019 datasets are.",
              "* Scaling plots use raw ΔLL (no ceiling normalisation): OLS line on log10(parameters) over all models, "
              "descriptive only.",
              "* Always read the ratio next to the raw ΔLL and the ceiling: a ratio to a ceiling near 0 is unstable.", "",
              "## Noise ceilings (split-half ΔLL per pair, same nuisance baseline as the models)", "",
              "| Dataset / unit | Split-half ceiling | Models above it (embedding / surprisal) |", "| --- | ---: | --- |"]
    if has_ceil:
        sub = d[d.confounds == "len_freq"]
        for (ds, unit), g in sub.groupby(["dataset", "unit"], sort=False):
            c = g.ceiling_dll.iloc[0]
            if not np.isfinite(c):
                lines.append(f"| {panel_title(ds, unit)} | none | n/a |")
                continue
            n_above = {t: int((g[g.track == t].last_layer_delta_ll_per_obs > c).sum()) for t in ("embedding", "surprisal")}
            n_tot = {t: int((g.track == t).sum()) for t in ("embedding", "surprisal")}
            lines.append(f"| {panel_title(ds, unit)} | {c:.2e} | {n_above['embedding']}/{n_tot['embedding']} / "
                         f"{n_above['surprisal']}/{n_tot['surprisal']} |")
    else:
        lines.append("| (no ceiling table passed: run with `--ceilings`) | | |")
    lines += ["", "Model scores above the ceiling occur because the full-sample score is compared with a half-sample "
              "reference (and the reference has finite reliability); they are reported, not removed.", "",
              "## Reading the figures (caveats)", "",
              "* Neural ΔLL values are tiny and many neural units have no model whose bootstrap interval is above zero; "
              "differences along the neural axis are largely noise, so the behavioral axis carries most of the ordering "
              "of the fronts.",
              "* The scaling lines are straight fits on log10(parameters). Several behavioral panels rise to about 1 to 3B "
              "parameters and then flatten or fall for the largest models (visible for surprisal on Provo, MECO), "
              "which a linear fit does not describe; slopes and r are descriptive.",
              "* ΔLL values are tiny per pair (1e-5 to 1e-2); ceilings are given above for scale. "
              "Negative ΔLL means the extra predictor hurts held-out fit.", "",
              "## Pareto fronts", ""]
    for (track, var), (m, front) in pareto.items():
        lines += [f"### {track.capitalize()} · {VARIANT_TEXT[var]}", "",
                  f"{len(m)} models with >= {int(MIN_COVERAGE * 100)}% unit coverage. Non-dominated models:", "",
                  "| Model | Group | Params | Neural | Behavioral |", "| --- | --- | ---: | ---: | ---: |"]
        for r in front.sort_values("behav", ascending=False).itertuples():
            lines.append(f"| {r.label} | {r.group} | {r.n_params / 1e9:.2f}B | {r.neural:.4f} | {r.behav:.4f} |")
        lines.append("")
    lines += ["## Scaling slopes (ΔLL per tenfold parameters, with word length + frequency baseline)", "",
              "Slope per tenfold parameters over non-Pythia models. Pythia checkpoints are excluded because they are a training trajectory, not independent model-size observations.", "",
              "| Unit | Embedding slope | r | Surprisal slope | r |", "| --- | ---: | ---: | ---: | ---: |"]
    sc = scaling[scaling.confounds == "len_freq"]
    for ds, unit in [p for row in LAYOUT for p in row if p]:
        e = sc[(sc.dataset == ds) & (sc.unit == unit) & (sc.track == "embedding")]
        s = sc[(sc.dataset == ds) & (sc.unit == unit) & (sc.track == "surprisal")]
        f = lambda t, c, fmt: (format(t[c].iloc[0], fmt) if len(t) and np.isfinite(t[c].iloc[0]) else "")  # noqa: E731
        lines.append(f"| {panel_title(ds, unit)} | {f(e, 'slope_per_10x_params', '+.2e')} | "
                     f"{f(e, 'pearson_r_log10_params', '+.2f')} | {f(s, 'slope_per_10x_params', '+.2e')} | "
                     f"{f(s, 'pearson_r_log10_params', '+.2f')} |")
    lines += ["", "Raw numbers for every model (ΔLL, bootstrap 95% interval, ceiling, ratio): "
              "`$SCRATCH/rdm_report/rdm_last_layer_results.csv`; panel statistics: `rdm_scaling_summary.csv`."]
    (OUT / "rdm_regression_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, default=Path(os.environ["SCRATCH"]) / "rdm_report" / "rdm_last_layer_results.csv")
    ap.add_argument("--out", type=Path, default=None, help="output folder of tables/raw panels (default reports/rsa_summary/rdm_regression)")
    ap.add_argument("--report-root", type=Path, default=None,
                    help="report root replacing reports/rsa_summary (figures/, figures_pdf/, rdm_regression/, rsa/model_scores_*.csv), "
                         "e.g. reports/rsa_summary_nostd, so the standardized figures are not overwritten")
    ap.add_argument("--ceilings", type=Path, default=None,
                    help="rdm_ceiling_values.csv from rdm_ceiling_table.py: Pareto axes = dLL / ceiling, datasets with a model above it left out")
    ap.add_argument("--neural-summary", type=Path, default=None,
                    help="neural_rdm_summary.csv from rdm_regression_neural_summary.py; replaces the neural rows of --table")
    a = ap.parse_args()
    global OUT, REPORT, FIG, FIGPDF
    if a.report_root is not None:
        REPORT = a.report_root
        OUT, FIG, FIGPDF = REPORT / "rdm_regression", REPORT / "figures", REPORT / "figures_pdf"
    if a.out is not None:
        OUT = a.out
    use_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load(a.table)
    if a.neural_summary is not None:
        d = with_neural_units(d, a.neural_summary)
    if a.ceilings is not None:
        d = apply_ceilings(d, a.ceilings)
    EXCLUDED[:] = d.attrs.get("excluded", [])
    def robust_z(x: pd.Series) -> pd.Series:
        """(x - median) / (1.4826 * MAD): z-score that a few extreme models cannot distort."""
        mad = 1.4826 * (x - x.median()).abs().median()
        return (x - x.median()) / mad if mad > 0 else x * np.nan

    d["z"] = d.groupby(["dataset", "unit", "track", "confounds"]).last_layer_delta_ll_per_obs.transform(robust_z)
    db = attach_benchmark(d)
    pareto, pareto_rows, records = {}, [], []
    for track in ("embedding", "surprisal"):
        for var in VARIANTS:
            m = composites(d, track, var)
            front = pareto_front(m, "neural", "behav") if len(m) else m
            pareto[(track, var)] = (m, front)
            if len(m):
                draw_pareto(m, track, var)
            draw_scaling(d, track, var, records)
            draw_benchmark(db, track, var)
            import shutil
            FIG.mkdir(parents=True, exist_ok=True)
            FIGPDF.mkdir(parents=True, exist_ok=True)
            shutil.copy(OUT / f"rdm_scaling_{track}.png", FIG / f"{track}_dll.png")
            shutil.copy(OUT / f"rdm_scaling_{track}.pdf", FIGPDF / f"{track}_dll.pdf")
            if len(m):
                pareto_rows.append(m.assign(track=track, confounds=var, on_front=m.index.isin(front.index), score="ceiling_normalised"))
            print(f"{track}/{var}: {len(m)} models, front {len(front)}", flush=True)
            mz = composites(d, track, var, "z")
            fz = pareto_front(mz, "neural", "behav") if len(mz) else mz
            if len(mz):
                draw_pareto(mz, track, var, "zscore")
                pareto_rows.append(mz.assign(track=track, confounds=var, on_front=mz.index.isin(fz.index), score="robust_zscore"))
            print(f"{track}/{var} z-score: {len(mz)} models, front {len(fz)} ({mz.attrs.get('n_neural_units')} neural, {mz.attrs.get('n_behav_units')} behavioral units)", flush=True)
    scaling = pd.DataFrame(records)
    scaling.to_csv(OUT / "rdm_scaling_summary.csv", index=False)
    if pareto_rows:
        pd.concat(pareto_rows).to_csv(OUT / "rdm_pareto_scores.csv", index=False)
    write_report(d, pareto, scaling)
    print(f"wrote figures, tables and report to {OUT}")


if __name__ == "__main__":
    main()
