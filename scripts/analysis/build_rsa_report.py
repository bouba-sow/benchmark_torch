#!/usr/bin/env python3
"""Build one RSA report from saved scores; no feature extraction or RDM fitting."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = "/tmp/rsa-report-matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullLocator
import numpy as np
import pandas as pd

from neuroai.analysis.benchmark_common import (
    EXCLUDED_FAMILY_DIRS, GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, family_group,
)
from neuroai.models.registry import LPP_FAMILIES, LPP_FAMILY_NAMES, flatten_family_tasks

OUT = Path(os.environ["RSA_REPORT_OUT"]) if os.environ.get("RSA_REPORT_OUT") else ROOT / "reports/rsa_summary"
# RSA_OUTPUTS_ROOT: alternative root holding <dataset>/pearson_rsa (e.g. the raw/no-standardization mirror)
RSA_OUTPUTS_ROOT = Path(os.environ["RSA_OUTPUTS_ROOT"]) if os.environ.get("RSA_OUTPUTS_ROOT") else ROOT / "outputs"
# Layout: figures/ (the final PNGs), figures_pdf/ (vector copies), rsa/ (this script's tables and all panel figures),
# rdm_regression/ (RDM-regression track), audits/, old/. README.md stays at the top.
RSA = OUT / "rsa"
FIG = OUT / "figures"
FIGPDF = OUT / "figures_pdf"
RSA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
FIGPDF.mkdir(parents=True, exist_ok=True)
KEYS = ["family", "model_id", "revision"]
EXPECTED_KEYS = {(family, task.model_id, task.revision)
                 for family in LPP_FAMILY_NAMES if family not in EXCLUDED_FAMILY_DIRS
                 for task in flatten_family_tasks(family)}
MODEL_LAYERS = {(family, spec.model_id): spec.n_layers for family in LPP_FAMILY_NAMES
                if family not in EXCLUDED_FAMILY_DIRS for spec in LPP_FAMILIES[family]}
NEURAL = {
    "pereira": "Pereira · fMRI", "tuckute": "Tuckute B · fMRI",
    "broderick": "Broderick · EEG",  # Fedorenko 2016 ECoG excluded from the RSA / RDM-regression reports (2026-10-07)
    "megmasc": "MEG-MASC · MEG", "nieuwland": "Nieuwland · N400",
    "szewczyk_f2007": "Federmeier et al. (2007) · N400",
    "szewczyk_h2019": "Hubbard et al. (2019) · N400",
    "podcast": "Podcast · ECoG (−0.2–0.8 s)",
    "lpp": "LPP · fMRI", "ns_fmri": "Natural Stories · fMRI",
}
BEHAVIOR = {
    "provo": ("Provo", ("FP", "GP", "SP")),
    "meco": ("MECO English", ("FP", "GP", "SP")),
    "geco": ("GECO", ("FP", "GP")),
    "onestop": ("OneStop", ("FP", "GP", "SP")),
    "naturalstories": ("Natural Stories", ("SPR",)),
    "ucl_reading": ("UCL Reading", ("SPR",)),
    "zuco_reading": ("ZuCo", ("NR", "SR", "TSR")),
}
# Same model-category palette as the rest of the benchmark figures (figure3/figure6
# use orange for Constrained-BabyLM).
GROUP_COLOR = {**GROUP_COLORS, "Constrained-BabyLM": "#ff7f0e"}
COLORS = {"embedding": "#2166ac", "surprisal": "#c62828"}
MARKERS = {"embedding": "o", "surprisal": "^"}


def relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_summary(dataset, track, measure=""):
    base = RSA_OUTPUTS_ROOT / dataset / "pearson_rsa"
    if dataset == "podcast":
        path = base / track / "lagwise" / "model_summary.csv"
    elif track == "embedding" and dataset in {"pereira", "tuckute", "fedorenko2016"}:
        path = base / "model_summary.csv"
    elif track == "embedding" and dataset in {"provo", "meco"}:
        path = base / measure.lower() / "model_summary.csv"
    else:
        path = base / track / measure.lower() / "model_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=KEYS + ["rsa_pearson"])
    df = pd.read_csv(path, dtype={k: str for k in KEYS})
    selected = [tuple(row) in EXPECTED_KEYS for row in df[KEYS].itertuples(index=False, name=None)]
    if not all(selected):
        print(f"Excluded {len(df) - sum(selected)} non-benchmark rows from {path}", flush=True)
    df = df.loc[selected].copy()
    if df.duplicated(KEYS).any():
        raise ValueError(f"Duplicate model/revision keys: {path}")
    df["rsa_source"] = relative(path)
    return df


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def neural_reference(dataset, frames):
    """Use the saved RSA encoding comparator; fill missing exact revisions from scores."""
    refs = {}
    if dataset == "nieuwland":
        # The Nieuwland benchmark was updated to the paper's 160-item mean N400
        # fit. Its report comparator must come from the last-layer Pearson r in
        # those new per-model layerwise files; the RSA summary still contains
        # the older epoch-based encoding values and must not override them.
        root = ROOT / "outputs_jeanzay/nieuwland/embedding_sweep"
        files = list(root.glob("*/*/nieuwland_paper_item_mean_layerwise.csv"))
        frames_by_model = []
        for path in files:
            layerwise = pd.read_csv(path, dtype={"revision": str})
            if not {"family", "model_id", "revision", "layer", "pearsonr"} <= set(layerwise):
                raise ValueError(f"Missing final-layer Pearson columns: {path}")
            layerwise["source"] = relative(path)
            frames_by_model.append(layerwise)
        if not frames_by_model:
            raise FileNotFoundError(f"No updated Nieuwland item-mean fit files under {root}")
        scores = pd.concat(frames_by_model, ignore_index=True)
        scores["layer"] = pd.to_numeric(scores.layer, errors="coerce")
        keys = ["family", "model_id", "revision"]
        last_layer = scores.loc[
            scores.layer.eq(scores.groupby(keys).layer.transform("max"))
        ].copy()
        if last_layer.duplicated(keys).any():
            raise ValueError("Duplicate Nieuwland final-layer fit scores")
        available = set(last_layer[keys].itertuples(index=False, name=None))
        missing = EXPECTED_KEYS - available
        if missing:
            raise RuntimeError(f"Nieuwland updated fit scores missing {len(missing)} registry models")
        for row in last_layer.itertuples(index=False):
            if not np.isfinite(row.pearsonr):
                raise RuntimeError(f"Nonfinite Nieuwland final-layer Pearson r: {row.model_id}")
            refs[(row.family, row.model_id, row.revision)] = (float(row.pearsonr), row.source)
        return refs
    for frame in frames:
        col = "encoding_pearson_50_250ms" if dataset == "podcast" else "encoding_pearson"
        if col not in frame:
            continue
        for _, row in frame.iterrows():
            if np.isfinite(row[col]):
                refs.setdefault(tuple(row[k] for k in KEYS), (float(row[col]), row.rsa_source))
    tasks = pd.concat(frames, ignore_index=True).drop_duplicates(KEYS)
    for _, row in tasks.iterrows():
        key = tuple(row[k] for k in KEYS)
        if key in refs:
            continue
        base = ROOT / "outputs_jeanzay" / dataset / "embedding_sweep" / row.family / row.model_id.replace("/", "_")
        filename = {"nieuwland": "nieuwland_checkpoint_layerwise.csv",
                    "fedorenko2016": "fedorenko2016_by_electrode.csv",
                    "megmasc": "megmasc_by_subject.csv",
                    "szewczyk": "szewczyk_by_subject.csv"}.get(dataset)
        if filename is None or not (base / filename).exists():
            continue
        path = base / filename
        scores = pd.read_csv(path, dtype={"revision": str})
        scores = scores.loc[scores.revision == row.revision]
        if scores.empty:
            continue
        if dataset == "fedorenko2016":
            value = scores.pearsonr.mean()
        elif dataset == "nieuwland":
            value = scores.loc[scores.layer == scores.layer.max(), "mean_pearsonr"].mean()
        else:
            col = "last_layer_mean_pearsonr" if "last_layer_mean_pearsonr" in scores else "pearsonr_mean"
            value = scores.groupby("subject")[col].mean().mean()
        if np.isfinite(value):
            refs[key] = (float(value), relative(path))
    return refs


def likelihood_reference(dataset, measure, frames):
    refs = {}
    tasks = pd.concat(frames, ignore_index=True).drop_duplicates(KEYS)
    name = {"naturalstories": "spr", "ucl_reading": "ucl_reading_spr",
            "zuco_reading": "zuco_reading_trt"}.get(dataset, dataset)
    subdir = "" if dataset in {"naturalstories", "ucl_reading"} else measure.lower()
    for _, row in tasks.iterrows():
        path = (ROOT / "outputs_jeanzay" / dataset / "surprisal_sweep" / subdir /
                row.family / row.model_id.replace("/", "_") / f"{name}_lme_checkpoint_scores.csv")
        if not path.exists():
            continue
        scores = pd.read_csv(path, dtype={"revision": str})
        scores = scores.loc[scores.revision == row.revision]
        if "measure" in scores:
            scores = scores.loc[scores.measure.str.upper() == measure]
        if scores.empty or not {"layer", "delta_loglik_test", "n_obs_test"} <= set(scores):
            continue
        # Include every saved final-layer score, without outlier filtering.
        # A missing final-layer score must never fall back to an earlier layer.
        last_layer = pd.to_numeric(scores.layer).max()
        last = scores.loc[(scores.layer == last_layer) & (scores.n_obs_test > 0)]
        value = (last.delta_loglik_test / last.n_obs_test).replace([np.inf, -np.inf], np.nan).mean()
        if np.isfinite(value):
            refs[tuple(row[k] for k in KEYS)] = (float(value), relative(path))
    return refs


def behavioral_pearson_reference(dataset, measure, frames):
    """Final-layer Pearson r from the subject-averaged embedding ridge CSVs."""
    if dataset in {"geco", "meco", "provo", "onestop"}:
        subdir = f"{measure.lower()}_average"
        name = f"{dataset}_embed_checkpoint_scores.csv"
        root = ROOT / "outputs_jeanzay" / dataset / "embedding_sweep" / subdir
    elif dataset == "zuco_reading":
        name = "zuco_reading_embed_checkpoint_scores.csv"
        root = ROOT / "outputs_jeanzay/zuco_reading/embedding_sweep" / f"{measure.lower()}_average"
    elif dataset == "ucl_reading":
        name = "ucl_reading_embed_checkpoint_scores.csv"
        root = ROOT / "outputs_jeanzay/ucl_reading/embedding_sweep_average"
    else:
        name = "spr_checkpoint_layerwise.csv"
        root = ROOT / "outputs_jeanzay/naturalstories/embedding_sweep"
    refs = {}
    tasks = pd.concat(frames, ignore_index=True).drop_duplicates(KEYS)
    for row in tasks.itertuples():
        path = root / row.family / row.model_id.replace("/", "_") / name
        if not path.exists():
            continue
        scores = pd.read_csv(path, dtype={"revision": str})
        if not {"revision", "layer", "pearsonr"} <= set(scores):
            continue
        scores = scores.loc[scores.revision == row.revision]
        if "measure" in scores:
            scores = scores.loc[scores.measure.astype(str).str.upper() == measure]
        last_layer = MODEL_LAYERS[(row.family, row.model_id)] - 1
        last = scores.loc[pd.to_numeric(scores.layer, errors="coerce") == last_layer]
        if "n_items" in last:
            last = last.loc[pd.to_numeric(last.n_items, errors="coerce") > 2]
        value = pd.to_numeric(last.pearsonr, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        if np.isfinite(value):
            refs[(row.family, row.model_id, row.revision)] = (float(value), relative(path))
    return refs


def benchmark_summary_reference(dataset, measure, track):
    """Read the saved benchmark score for the same feature track as RSA."""
    if dataset in {"lpp", "ns_fmri"}:
        return {}
    if track == "embedding":
        name = {"naturalstories": "naturalstories_spr",
                "ucl_reading": "ucl_reading",
                "zuco_reading": "zuco_reading"}.get(dataset, dataset)
        if dataset not in {"naturalstories", "ucl_reading"}:
            name += {"FP": "", "GP": "_gp", "SP": "_sp",
                     "NR": "", "SR": "_sr", "TSR": "_tsr"}[measure]
        name += "__embed"
    else:
        name = {"pereira": "pereira_fmri", "tuckute": "tuckute_fmri",
                "fedorenko2016": "fedorenko2016_ecog", "broderick": "broderick_eeg",
                "megmasc": "megmasc_meg", "nieuwland": "nieuwland_eeg",
                "szewczyk_f2007": "szewczyk_f2007_eeg",
                "szewczyk_h2019": "szewczyk_h2019_eeg", "podcast": "podcast_ieeg"}[dataset]
        if dataset != "pereira":
            name += "__surprisal"
    path = ROOT / "outputs/benchmark_summary" / name / "all_models_scores.csv"
    if not path.exists():
        return {}
    scores = pd.read_csv(path)
    if not {"family", "model_id", "score_last"} <= set(scores):
        raise ValueError(f"Missing benchmark score columns: {path}")
    scores = scores.loc[[(f, m) in {(key[0], key[1]) for key in EXPECTED_KEYS}
                          for f, m in scores[["family", "model_id"]].itertuples(index=False, name=None)]]
    if scores.duplicated(["family", "model_id"]).any():
        raise ValueError(f"Duplicate benchmark models: {path}")
    return {(row.family, row.model_id, key[2]): (float(row.score_last), relative(path))
            for row in scores.itertuples() if np.isfinite(row.score_last)
            for key in EXPECTED_KEYS if key[:2] == (row.family, row.model_id)}


def collect(comparison):
    points, summary = [], []
    panels = {"neural": [], "behavioral": []}
    specs = [("neural", ds, title, "") for ds, title in NEURAL.items()]
    specs += [("behavioral", ds, f"{title} · {m}", m)
              for ds, (title, measures) in BEHAVIOR.items() for m in measures]
    selected_track = "embedding" if comparison == "correlation" else "surprisal"
    for modality, dataset, title, measure in specs:
        frames = [read_summary(dataset, selected_track, measure)]
        if dataset == "podcast":
            frame = frames[0]
            references = {comparison: {
                tuple(row[k] for k in KEYS): (float(row.benchmark_score), row.rsa_source)
                for _, row in frame.iterrows() if np.isfinite(row.benchmark_score)
            }}
        elif modality == "neural":
            references = {"correlation": neural_reference(dataset, frames),
                          "dll": benchmark_summary_reference(dataset, measure, "surprisal")}
        else:
            references = {"correlation": behavioral_pearson_reference(dataset, measure, frames),
                          "dll": likelihood_reference(dataset, measure, frames)}
        panels[modality].append((dataset, measure, title))
        for track, df in [(selected_track, frames[0])]:
            df = df.copy()
            refs = references[comparison]
            df["benchmark_score"] = [refs.get(tuple(r[k] for k in KEYS), (np.nan, ""))[0]
                                     for _, r in df.iterrows()]
            df["benchmark_source"] = [refs.get(tuple(r[k] for k in KEYS), (np.nan, ""))[1]
                                      for _, r in df.iterrows()]
            for k, v in dict(dataset=dataset, measure=measure, modality=modality, track=track, title=title).items():
                df[k] = v
            df["benchmark_metric"] = (
                ("neural_embedding_last_layer_encoding_pearson" if dataset == "podcast" else
                 "neural_embedding_encoding_pearson") if modality == "neural" and comparison == "correlation" else
                "neural_surprisal_benchmark_score" if modality == "neural" else
                "behavioral_embedding_last_layer_pearson" if comparison == "correlation" else
                "behavioral_surprisal_final_layer_delta_loglik_test_per_obs")
            finite = np.isfinite(df.rsa_pearson.astype(float))
            pair = finite & np.isfinite(df.benchmark_score.astype(float))
            size = pd.to_numeric(df.get("n_params", pd.Series(index=df.index, dtype=float)), errors="coerce")
            log_size = np.log10(size.where(size > 0))
            summary.append(dict(modality=modality, dataset=dataset, measure=measure, title=title,
                                track=track, n_expected=len(EXPECTED_KEYS), n_rsa=int(finite.sum()), n_paired=int(pair.sum()),
                                mean_rsa=df.loc[finite, "rsa_pearson"].mean(),
                                score_rsa_r=correlation(df.benchmark_score, df.rsa_pearson),
                                size_rsa_r=correlation(log_size, df.rsa_pearson)))
            if not df.empty:
                points.append(df)
        print(f"Collected {title}", flush=True)
    return pd.concat(points, ignore_index=True), pd.DataFrame(summary), panels


def plot(points, summary, panels, modality, comparison):
    count = len(panels)
    ncols = 2 if modality == "neural" else 4
    nrows = (count + ncols - 1) // ncols
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.4, nrows * 3.6), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        if panel is None:
            ax.set_visible(False)
            continue
        dataset, measure, title = panel
        sub = points.loc[(points.dataset == dataset) & (points.measure == measure)]
        stats = summary.loc[(summary.dataset == dataset) & (summary.measure == measure)]
        panel_modality = stats.modality.iloc[0]
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=9)
        if sub.empty:
            ax.text(.5, .5, "No saved RSA scores", ha="center", va="center", transform=ax.transAxes, color="#777777")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        track = "embedding" if comparison == "correlation" else "surprisal"
        for i, track in enumerate((track,)):
            values = sub.loc[sub.track == track].dropna(subset=["benchmark_score", "rsa_pearson"])
            values = values.loc[np.isfinite(values.benchmark_score) & np.isfinite(values.rsa_pearson)]
            color = COLORS[track]
            if "group" not in values or values.group.isna().any():
                values = values.assign(group=[family_group(f, m) for f, m in zip(values.family, values.model_id)])
            for g in GROUP_ORDER:
                vg = values.loc[values.group == g]
                if len(vg):
                    ax.scatter(vg.benchmark_score, vg.rsa_pearson, s=18, alpha=.7, color=GROUP_COLOR[g],
                               marker=MARKERS[track], linewidths=0, rasterized=True)
            stat = stats.loc[stats.track == track].iloc[0]
            ax.text(.02, .98, f"RSA {stat.n_rsa}/{len(EXPECTED_KEYS)}, paired {stat.n_paired}", color="#333333",
                    transform=ax.transAxes, va="top", fontsize=8, bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
            k = 0
            for g in GROUP_ORDER:
                vg = values.loc[values.group == g]
                if len(vg) >= 5 and vg.benchmark_score.nunique() >= 3:
                    rg = correlation(vg.benchmark_score, vg.rsa_pearson)
                    slope, intercept = np.polyfit(vg.benchmark_score, vg.rsa_pearson, 1)
                    xs = np.array([vg.benchmark_score.min(), vg.benchmark_score.max()])
                    ax.plot(xs, intercept + slope * xs, color=GROUP_COLOR[g], linewidth=1.5, zorder=3)
                    k += 1
                    ax.text(.02, .98 - .066 * k, f"{GROUP_DISPLAY.get(g, g)}: r={rg:+.2f} (n={len(vg)})", color=GROUP_COLOR[g],
                            transform=ax.transAxes, va="top", fontsize=7,
                            bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=.5))
        ax.margins(y=.26, x=.07)
        ax.axhline(0, color="#cccccc", linewidth=.6, zorder=0)
        ax.grid(alpha=.13)
        ax.set_ylabel("RSA Pearson r")
        ax.set_xlabel("Pearson r" if comparison == "correlation" else
                      "ΔLL / test observation")
        if comparison == "dll":
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useMathText=True)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    track = "embedding" if comparison == "correlation" else "surprisal"
    handles = [Line2D([], [], marker=MARKERS[track], color=GROUP_COLOR[g], label=GROUP_DISPLAY.get(g, g),
                      linestyle="-", markersize=5) for g in GROUP_ORDER]
    title = {"neural": "Neural RSA", "behavioral": "Behavioral RSA",
             "overall": "RSA comparisons · all neural and behavioral datasets"}[modality]
    title += " versus " + ("embedding Pearson r" if comparison == "correlation"
                            else "surprisal ΔLL / test observation")
    fig.suptitle(title, fontsize=18, y=.993)
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(.5, .976), frameon=False)
    fig.tight_layout(rect=(0, .02, 1, .951), h_pad=2.2, w_pad=2.3)
    fig.text(.5, .008, "Each point requires both RSA and a benchmark score; one line (and r) per model category. RSA coverage and paired counts are shown separately.\nLines and correlations are descriptive; panel axes vary.", ha="center", fontsize=9, color="#555555")
    for suffix in ("png", "pdf"):
        fig.savefig(RSA / f"{modality}_{comparison}.{suffix}", dpi=180, facecolor="white")
        if (modality, comparison) in {("overall", "correlation"), ("neural", "correlation"),
                                      ("behavioral", "dll")}:
            fig.savefig(RSA / f"{modality}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def overview_order(panels):
    """Group the combined overview by neural modality, then behavioral measure."""
    by_key = {(dataset, measure): (dataset, measure, title)
              for dataset, measure, title in panels}
    blocks = [
        [("pereira", ""), ("tuckute", ""), ("lpp", ""), ("ns_fmri", "")],
        [("podcast", ""), ("broderick", ""), ("megmasc", ""), None],
        [("nieuwland", ""), ("szewczyk_f2007", ""),
         ("szewczyk_h2019", ""), None],
        [("provo", "FP"), ("meco", "FP"), ("geco", "FP"), ("onestop", "FP")],
        [("provo", "GP"), ("meco", "GP"), ("geco", "GP"), ("onestop", "GP")],
        [("provo", "SP"), ("meco", "SP"), ("onestop", "SP"), None],
        [("naturalstories", "SPR"), ("ucl_reading", "SPR"), None, None],
        [("zuco_reading", "NR"), ("zuco_reading", "SR"),
         ("zuco_reading", "TSR"), None],
    ]
    return [None if key is None else by_key[key] for block in blocks for key in block]


def plot_scaling(points, panels):
    """One scaling figure per track (rsa_scaling_embedding / rsa_scaling_surprisal)."""
    records = []
    for track in COLORS:
        records += _plot_scaling_track(points, panels, track)
    pd.DataFrame(records).to_csv(RSA / "scaling_summary.csv", index=False)


def _plot_scaling_track(points, panels, track):
    """Scatter RSA against parameter count, with descriptive log-size fits."""
    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.4, nrows * 3.6), squeeze=False)
    records = []
    for ax, panel in zip(axes.flat, panels):
        if panel is None:
            ax.set_visible(False)
            continue
        dataset, measure, title = panel
        sub = points.loc[(points.dataset == dataset) & (points.measure == measure)]
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=9)
        available = sub.loc[np.isfinite(pd.to_numeric(sub.n_params, errors="coerce")) &
                            (pd.to_numeric(sub.n_params, errors="coerce") > 0) &
                            np.isfinite(pd.to_numeric(sub.rsa_pearson, errors="coerce"))]
        if available.empty:
            ax.text(.5, .5, "No RSA / model-size results", ha="center", va="center",
                    transform=ax.transAxes, color="#777777")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        for idx, track in enumerate([track]):
            values = sub.loc[sub.track == track].copy()
            values["n_params"] = pd.to_numeric(values.n_params, errors="coerce")
            values["rsa_pearson"] = pd.to_numeric(values.rsa_pearson, errors="coerce")
            values = values.loc[np.isfinite(values.n_params) & (values.n_params > 0) &
                                np.isfinite(values.rsa_pearson)]
            if values.empty:
                continue
            color = "#333333"
            if "group" not in values or values.group.isna().any():
                values["group"] = [family_group(f, m) for f, m in zip(values.family, values.model_id)]
            x = np.log10(values.n_params.to_numpy(float))
            y = values.rsa_pearson.to_numpy(float)
            ax.scatter(values.n_params, y, s=18, alpha=.75, marker=MARKERS[track],
                       c=values.group.map(GROUP_COLOR).fillna("grey"), linewidths=0, rasterized=True)
            r = correlation(x, y)
            slope = float(np.polyfit(x, y, 1)[0]) if len(np.unique(x)) >= 2 else np.nan
            ax.text(.02, .98, f"n={len(values)}", color="#333333", transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
            k = 0
            for g in GROUP_ORDER:
                vg = values.loc[values.group == g]
                xg, yg = np.log10(vg.n_params.to_numpy(float)), vg.rsa_pearson.to_numpy(float)
                if len(vg) >= 5 and len(np.unique(xg)) >= 3:
                    xs = np.linspace(xg.min(), xg.max(), 50)
                    ax.plot(10.0 ** xs, np.polyval(np.polyfit(xg, yg, 1), xs), color=GROUP_COLOR[g], linewidth=1.5, zorder=3)
                    k += 1
                    ax.text(.02, .98 - .066 * k, f"{GROUP_DISPLAY.get(g, g)}: r={correlation(xg, yg):+.2f} (n={len(vg)})",
                            color=GROUP_COLOR[g], transform=ax.transAxes, va="top", fontsize=7,
                            bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=.5))
            records.append(dict(modality="neural" if dataset in NEURAL else "behavioral",
                                dataset=dataset, measure=measure, title=title, track=track,
                                n_models=len(values), n_distinct_sizes=len(np.unique(x)),
                                pearson_r_log10_size=r, rsa_change_per_10x_size=slope))
        ax.set_xscale("log")
        ax.axhline(0, color="#cccccc", linewidth=.6, zorder=0)
        ax.grid(alpha=.13)
        ax.set_xlabel("Model parameters (log scale)")
        ax.set_ylabel("RSA Pearson r")
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"{track.capitalize()} RSA by model size across datasets", fontsize=18, y=.993)
    handles = [Line2D([], [], marker=MARKERS[track], linestyle="-", color=GROUP_COLOR[g],
                      label=GROUP_DISPLAY.get(g, g), markersize=6) for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(.5, .976), frameon=False)
    fig.tight_layout(rect=(0, .02, 1, .951), h_pad=2.2, w_pad=2.3)
    fig.text(.5, .008, "One line per model category fits RSA against log10(parameter count). Descriptive trends only.",
             ha="center", fontsize=9, color="#555555")
    for suffix in ("png", "pdf"):
        fig.savefig(RSA / f"rsa_scaling_{track}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    return records


def table(summary, panels):
    lines = ["| Dataset / measure | RSA N / paired | Mean RSA | r(score, RSA) |",
             "| --- | ---: | ---: | ---: |"]
    def fmt(x):
        return f"{x:.3f}" if np.isfinite(x) else "—"
    for dataset, measure, title in panels:
        s = summary.loc[(summary.dataset == dataset) & (summary.measure == measure)].iloc[0]
        row = [title, f"{s.n_rsa} / {s.n_paired}", fmt(s.mean_rsa), fmt(s.score_rsa_r)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-saved", action="store_true",
                        help="Redraw the report from its own saved CSVs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit = []
    results = {}
    for comparison in ("correlation", "dll"):
        if args.from_saved:
            points = pd.read_csv(RSA / f"model_scores_{comparison}.csv", dtype={k: str for k in KEYS})
            if not all(tuple(row) in EXPECTED_KEYS for row in points[KEYS].itertuples(index=False, name=None)):
                raise ValueError("Saved report contains non-benchmark models; rebuild without --from-saved")
            summary = pd.read_csv(RSA / f"summary_{comparison}.csv")
            for frame in (points, summary):
                frame["measure"] = frame.measure.fillna("")
            panels = {modality: list(summary.loc[summary.modality == modality,
                      ["dataset", "measure", "title"]].drop_duplicates().itertuples(index=False, name=None))
                      for modality in ("neural", "behavioral")}
        else:
            points, summary, panels = collect(comparison)
        results[comparison] = (points, summary, panels)
        for row in summary.itertuples():
            part = points.loc[(points.dataset == row.dataset) & (points.measure == row.measure) &
                              (points.track == row.track)]
            present = set(part.loc[np.isfinite(part.rsa_pearson), KEYS].itertuples(index=False, name=None))
            for key in sorted(EXPECTED_KEYS - present):
                audit.append(dict(comparison=comparison, dataset=row.dataset, measure=row.measure,
                                  track=row.track, **dict(zip(KEYS, key)), reason="missing finite RSA"))
            for item in part.loc[np.isfinite(part.rsa_pearson) & ~np.isfinite(part.benchmark_score)].itertuples():
                audit.append(dict(comparison=comparison, dataset=row.dataset, measure=row.measure,
                                  track=row.track, **{k: getattr(item, k) for k in KEYS},
                                  reason="missing valid benchmark comparator"))
        points.to_csv(RSA / f"model_scores_{comparison}.csv", index=False)
        summary.to_csv(RSA / f"summary_{comparison}.csv", index=False)
        for modality in panels:
            plot(points, summary, panels[modality], modality, comparison)
        ordered_panels = overview_order(panels["neural"] + panels["behavioral"])
        plot(points, summary, ordered_panels, "overall", comparison)
        if comparison == "correlation":
            # Preserve the original CSV names as aliases for the correlation comparison.
            points.to_csv(RSA / "model_scores.csv", index=False)
            summary.to_csv(RSA / "summary.csv", index=False)
    scaling_points = pd.concat([results["correlation"][0], results["dll"][0]], ignore_index=True)
    plot_scaling(scaling_points, overview_order(results["correlation"][2]["neural"] +
                                               results["correlation"][2]["behavioral"]))
    pd.DataFrame(audit, columns=["comparison", "dataset", "measure", "track", *KEYS, "reason"]).to_csv(
        RSA / "coverage_gaps.csv", index=False)
    _, correlation_summary, panels = results["correlation"]
    _, dll_summary, _ = results["dll"]
    report = r"""# RSA summary

Exact registry model/family/revision targets only (historical checkpoint families excluded). The **correlation** figures show embedding RSA in blue against embedding benchmark Pearson r. The **ΔLL** figures show surprisal RSA in red against surprisal benchmark ΔLL per test observation. **N / paired** gives the number of RSA scores / scores matched to the benchmark by model, family, and revision. Mean RSA uses all available scores; correlations use finite matched scores.

For stimuli $i,j$, embedding vectors $h$, surprisal $s_i=-\log p(w_i\mid w_{<i})$, neural patterns $y$, and mean reading times $t$:

$$D^E_{ij}=1-\mathrm{corr}(h_i,h_j),\qquad D^S_{ij}=|s_i-s_j|.$$
$$D^N_{ij}=1-\mathrm{corr}(y_i,y_j),\qquad D^B_{ij}=|t_i-t_j|.$$

For scalar N400 responses, $D^N_{ij}=|y_i-y_j|$. With $v(D)$ denoting the matching upper-triangle entries (all or the saved sampled pairs):

$$R_m=\mathrm{corr}\big(v(D^m),v(D^{response})\big),\qquad r_{comparison}=\mathrm{corr}_{models}(R_m,B_m).$$

Embedding RSA uses per-feature z-scored last-layer states (each dimension standardized across the scored stimuli, following Timkey & van Schijndel 2021, arXiv:2109.04404) before the 1 − Pearson model RDM; surprisal RSA is unchanged. N400 embedding and surprisal RSA use the subject-averaged item-mean N400 (the ridge target). The old-versus-new scores are in [standardization_evolution.md](standardization_evolution.md). Neural RSA averages subject scores (Pereira first averages experiments within subjects). The correlation comparison uses the embedding encoding Pearson score. Nieuwland's comparator uses the updated paper item-mean N400 fit's final-layer Pearson r; its existing RSA values are unchanged. Podcast embedding RSA and its ridge comparator both use the final hidden layer and only the 32 scored lags from −0.2 to 0.8 s. Neural and model RDMs are computed separately at each lag; the plot shows each model's mean RSA and mean same-lag benchmark score. The embedding and surprisal tracks retain their own native 32 Hz time grids. The ΔLL comparison uses the saved neural surprisal benchmark score. Sentence datasets use their saved sentence embeddings and pooled sentence surprisal.

## Neural datasets

### Correlation benchmark

""" + table(correlation_summary, panels["neural"]) + r"""

![Neural RSA versus embedding correlation](neural_correlation.png)

[Vector PDF](neural_correlation.pdf)

### ΔLL benchmark

""" + table(dll_summary, panels["neural"]) + r"""

![Neural RSA versus surprisal ΔLL](neural_dll.png)

[Vector PDF](neural_dll.pdf)

## Behavioral datasets

The correlation comparison uses the subject-averaged embedding ridge fit's last-layer **Pearson r** from its raw CSV. The ΔLL comparison uses the existing **surprisal** model's held-out behavioral improvement at its highest saved layer:

$$B_m=\frac{\Delta LL_{test,m}}{n_{test,m}}=\frac{LL_{full,m}-LL_{baseline,m}}{n_{test,m}}.$$

Each comparison shows the matching RSA track only. ΔLL comes from the existing behavioral benchmark, not a regression on RDM entries. All finite saved final-layer scores are included without outlier filtering. Missing or nonfinite final-layer scores cannot be plotted; an earlier layer is never substituted.

### Correlation benchmark

""" + table(correlation_summary, panels["behavioral"]) + r"""

![Behavioral RSA versus embedding correlation](behavioral_correlation.png)

[Vector PDF](behavioral_correlation.pdf)

### ΔLL benchmark

""" + table(dll_summary, panels["behavioral"]) + r"""

![Behavioral RSA versus surprisal ΔLL](behavioral_dll.png)

[Vector PDF](behavioral_dll.pdf)

## Lexical-confound RDMs

Separate word-frequency and word-length RDMs are saved under `outputs/<dataset>/pearson_rsa/confounds/`. For behavioral datasets, the scalar features are English Zipf frequency $z$ and character count $\ell$, with $D^{freq}_{ij}=|z_i-z_j|$ and $D^{length}_{ij}=|\ell_i-\ell_j|$. Each archive stores stimulus IDs, feature values, distances, and the saved response RDM's pair indices when sampled. See [Provo FP frequency](../../outputs/provo/pearson_rsa/confounds/fp/frequency_rdm.npz) and [word length](../../outputs/provo/pearson_rsa/confounds/fp/word_length_rdm.npz).

For the seven neural datasets with cached response RDMs, confound RDMs are saved per subject and retain the exact stimulus order and pairs. Federmeier 2007 and Hubbard 2019 now have embedding and surprisal RSA summaries from their item-level caches, but no separate saved response-RDM or confound-RDM archives. LPP and Natural Stories fMRI also have no saved neural RSA response RDM; their confounds are over the 181 and 97 time chunks in the LME covariate tables and are not pair-aligned to a neural response RDM. See [Pereira subject 018 frequency](../../outputs/pereira/pearson_rsa/confounds/018/frequency_rdm.npz) and [the neural confound builder](../../scripts/analysis/save_neural_lexical_confounds_rdms.py); the [behavioral builder](../../scripts/analysis/save_lexical_confounds_rdms.py) regenerates the behavioral RDMs.

FP: first-pass; GP: go-past; SP: total dwell; SPR: self-paced reading. ZuCo NR/SR/TSR identify its reading tasks. LPP and NS-fMRI RSA are deferred. UCL embedding RSA excludes incomplete caches. Coverage and stimulus sets can differ between tracks; these are descriptive model correlations, with no independent-pair significance tests or scaling-law claims.

[Correlation model scores and source paths](model_scores_correlation.csv) · [ΔLL model scores and source paths](model_scores_dll.csv) · [Table statistics for correlation](summary_correlation.csv) · [Table statistics for ΔLL](summary_dll.csv)

## All datasets in one figure

Each figure has one subplot per dataset/measure. The correlation figure plots embedding RSA against embedding benchmark Pearson r; the ΔLL figure plots surprisal RSA against surprisal ΔLL per test observation. The x-axis label is identical across all panels in each figure.

![All datasets versus embedding correlation](overall_correlation.png)

[Vector PDF](overall_correlation.pdf)

![All datasets versus surprisal ΔLL](overall_dll.png)

[Vector PDF](overall_dll.pdf)

## RSA by model size

Each subplot shows RSA against model parameters on a log scale, with one figure per track (embeddings, surprisal) and a trend line per panel. The fitted line uses $\log_{10}(\mathrm{parameters})$; its slope is the RSA change per tenfold increase. These trends are descriptive and do not alone establish a scaling law.

![Embedding RSA versus parameter count](rsa_scaling_embedding.png)

![Surprisal RSA versus parameter count](rsa_scaling_surprisal.png)

[Embedding PDF](rsa_scaling_embedding.pdf) · [Surprisal PDF](rsa_scaling_surprisal.pdf) · [Per-panel scaling statistics](scaling_summary.csv)

## RDM-regression scaling and Pareto plots

The item-averaged RDM-regression figures include separate Federmeier 2007 and Hubbard 2019 N400 panels. They use held-out ΔLL and split-half ceilings, which differ from the Pearson RSA results above. See the [RDM-regression report](rdm_regression_report.md), [embedding scaling](rdm_scaling_embedding.png), [surprisal scaling](rdm_scaling_surprisal.png), [embedding Pareto front](rdm_pareto_embedding.png), and [surprisal Pareto front](rdm_pareto_surprisal.png).

Rebuild: `.venv/bin/python scripts/analysis/build_rsa_report.py`.
"""
    report += "\n## Coverage audit\n\n[Exact missing RSA and comparator entries](coverage_gaps.csv). " \
              "Cache extraction readiness does not establish a finite RSA result or a valid benchmark comparator. " \
              "Historical extraction logs are not used to certify current coverage.\n"
    import re as _re
    report = _re.sub(r"\]\(((?:neural|behavioral|overall|rsa_scaling|model_scores|summary|scaling_summary|coverage_gaps|standardization_evolution)[^)/]*)\)", r"](rsa/\1)", report)
    report = _re.sub(r"\]\((rdm_[^)/]*)\)", r"](rdm_regression/\1)", report)
    report = report.replace("](rsa/coverage_gaps.csv)", "](rsa/coverage_gaps.csv)")
    figs = ("\n## Final figures\n\n`figures/` holds the six headline PNGs (vector copies in `figures_pdf/`): "
            "`embedding_correlation_with_benchmark`, `surprisal_correlation_with_benchmark` (RSA score versus benchmark), "
            "`embedding_dll`, `surprisal_dll` (RDM-regression ΔLL versus model size) and the two RSA scaling plots "
            "`scaling_embedding`, `scaling_surprisal` (RSA versus model size). Tables and all panel figures are in `rsa/`, the RDM-regression track in "
            "`rdm_regression/`, audits in `audits/`, superseded files in `old/`.\n")
    report = report.replace("\n## ", figs + "\n## ", 1) if "\n## " in report else report + figs
    (OUT / "README.md").write_text(report)
    # headline RSA figures (the other six come from scripts/figures/rdm_regression_figures.py)
    import shutil as _sh
    for src, name in (("overall_correlation", "embedding_correlation_with_benchmark"), ("overall_dll", "surprisal_correlation_with_benchmark")):
        _sh.copy(RSA / f"{src}.png", FIG / f"{name}.png")
        _sh.copy(RSA / f"{src}.pdf", FIGPDF / f"{name}.pdf")
    for track in ("embedding", "surprisal"):
        _sh.copy(RSA / f"rsa_scaling_{track}.png", FIG / f"scaling_{track}.png")
        _sh.copy(RSA / f"rsa_scaling_{track}.pdf", FIGPDF / f"scaling_{track}.pdf")
    print(f"Report saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
