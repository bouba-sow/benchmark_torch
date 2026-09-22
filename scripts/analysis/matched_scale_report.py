#!/usr/bin/env python3
"""ICLR matched-scale report: pooled, binned comparison of the 4 model
groups (LLM / Memory-Constrained / Constrained-BabyLM / BabyLM) across the
whole perplexity (and, separately, model-size) range, averaged over every
dataset at once.

Reads outputs_jeanzay/matched_scale_report/master_table.csv (built by
analysis/matched_scale_master_table.py -- run that first).

SCORE: score_norm = (dll_total / n_obs) / noise_ceiling for the 11 datasets
with a delta-loglik ceiling (all 11 checked positive -- see
_check_ceiling_signs -- so normalising never flips a score's sign). Because
every dataset's score is on this SAME ceiling-fraction scale, points can be
POOLED across all 11 datasets directly -- one (model, checkpoint, dataset)
triple is one point in the pool, not one point per dataset. The 3 N400
datasets have no delta-loglik ceiling (a different-unit split-half r ceiling
instead) so they're pooled separately using RAW per-word dll -- never mixed
into the ceiling-normalised pool.

Design: bin log10(x) (x = perplexity or n_params) into BIN_EDGES quantiles
of the pooled LLM group's own distribution (LLM has the densest, most
continuous coverage -- 110 Pythia checkpoints x however many datasets -- so
its quantiles give sensible, evenly-populated range boundaries). Every
group's points falling in a bin are averaged, regardless of which dataset
they came from. One bar chart: x-axis = perplexity/size range, 4 bars per
range = the 4 groups' pooled mean score (+/- SE) in that range.

Outputs:
  outputs_jeanzay/matched_scale_report/binned_group_means.csv
  reports/paper/matched_scale_report.pdf
  reports/paper/matched_scale_report_summary.md
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

from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, REPO_ROOT
from neuroai.analysis.paper_style import despine, use_paper_style
from scripts.figures.figure8_scale_matched_comparison import DLL_DATASET_SPECS, NOISE_CEILING_DIR

MASTER_PATH = REPO_ROOT / "outputs_jeanzay" / "matched_scale_report" / "master_table.csv"
BINNED_OUT = REPO_ROOT / "outputs_jeanzay" / "matched_scale_report" / "binned_group_means.csv"
PDF_OUT = REPO_ROOT / "reports" / "paper" / "matched_scale_report.pdf"
MD_OUT = REPO_ROOT / "reports" / "paper" / "matched_scale_report_summary.md"

AXES = [("perplexity", "corpus perplexity"), ("n_params", "model size (params)")]
N400_DATASETS = ("nieuwland", "megmasc")  # no ceiling -> raw score, own pool
N_BINS = 6
# n_params gets more bins than perplexity: model sizes cluster at a handful
# of standard sizes (14M/31M/70M/...), so 6 quantile bins lump 14M-70M (three
# whole Pythia sizes) into one bar. 10 bins splits that into 14M-31M and
# 31M-70M -- both still have every group represented (checked: Constrained-
# BabyLM/Memory-Constrained/LLM in the first, BabyLM/Constrained-BabyLM/LLM/
# Memory-Constrained in the second) -- without fragmenting the upper range
# into single-group, uninformative bins any worse than 6 bins already did
# (nothing above ~400M has more than LLM in it either way).
N_BINS_BY_AXIS: dict[str, int] = {"perplexity": 6, "n_params": 10}

# Behavioral (SPR + eye-tracking, all human RT/fixation-based) vs neural
# (fMRI + ECoG) split of the 11 ceiling-normalised datasets -- N400 (also
# neural/EEG but with no delta-loglik ceiling) stays in its own separate raw-
# score pool above, not folded into either split.
BEHAVIORAL_DATASETS = ("naturalstories_spr", "ucl_reading", "geco", "meco", "provo", "zuco_reading",
                        "onestop_fp", "onestop_gp", "onestop_sp",
                        "geco_gp", "meco_gp", "meco_sp", "provo_gp", "provo_sp",
                        # zuco_reading's SR/TSR tasks -- data landed 2026-09-08
                        # (ceiling already existed, NR-only pending note above
                        # no longer applies).
                        "zuco_reading_sr", "zuco_reading_tsr")
NEURAL_DATASETS = ("lpp_fmri", "tuckute_fmri", "ns_fmri", "fedorenko2016_ecog", "podcast_ieeg", "pereira_fmri")


def _check_ceiling_signs() -> str:
    negative, checked = [], []
    for slug, spec in DLL_DATASET_SPECS.items():
        path = NOISE_CEILING_DIR / f"delta_loglik_ceiling_{spec['ceiling_tag']}_oos.json"
        if not path.exists():
            continue
        import json
        with open(path) as f:
            c = json.load(f)["ceiling_delta_loglik_per_obs"]
        checked.append(slug)
        if c <= 0:
            negative.append((slug, c))
    if negative:
        raise SystemExit(f"Non-positive noise ceiling(s) -- normalising would flip sign: {negative}")
    return f"all positive (checked: {', '.join(sorted(checked))})"


def _pool(master: pd.DataFrame, datasets: list[str], score_col: str, axis: str) -> pd.DataFrame:
    df = master[master["dataset"].isin(datasets)]
    sub = df[["dataset", "group", axis, score_col]].dropna()
    return sub.rename(columns={score_col: "score"})


def _bin_edges(pool: pd.DataFrame, axis: str, n_bins: int = N_BINS) -> np.ndarray:
    """Quantile edges of the LLM group's own pooled log10(x) distribution --
    LLM has the densest coverage (Pythia's 110 checkpoints x every dataset),
    so its quantiles give evenly-populated bins other groups' sparser points
    then fall into."""
    llm_x = np.log10(pool.loc[pool["group"] == "LLM", axis].to_numpy())
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(llm_x, qs))
    return edges


def _format_range(lo: float, hi: float, axis: str) -> str:
    def fmt(v):
        if axis == "n_params":
            if v >= 1e9:
                return f"{v / 1e9:.1f}B"
            return f"{v / 1e6:.0f}M"
        if v >= 1000:
            return f"{v / 1000:.1f}k"
        return f"{v:.0f}" if v >= 10 else f"{v:.1f}"
    return f"{fmt(lo)}–{fmt(hi)}"


def compute_binned_means(master: pd.DataFrame) -> pd.DataFrame:
    ceiling_datasets = [d for d in DLL_DATASET_SPECS if d not in N400_DATASETS]
    rows = []
    for axis, _ in AXES:
        pools = [
            ("ceiling_normalised", ceiling_datasets, "score_norm"),
            ("n400_raw", list(N400_DATASETS), "score_raw"),
        ]
        pools += [
            ("behavioral_only", list(BEHAVIORAL_DATASETS), "score_norm"),
            ("neural_only", list(NEURAL_DATASETS), "score_norm"),
        ]
        n_bins = N_BINS_BY_AXIS[axis]
        for pool_name, datasets, score_col in pools:
            pool = _pool(master, datasets, score_col, axis)
            if pool.empty or (pool["group"] == "LLM").sum() < n_bins:
                continue
            edges = _bin_edges(pool, axis, n_bins=n_bins)
            logx = np.log10(pool[axis].to_numpy())
            bin_idx = np.digitize(logx, edges[1:-1])  # 0..len(edges)-2
            pool = pool.assign(bin=bin_idx)
            for b in range(len(edges) - 1):
                lo, hi = 10 ** edges[b], 10 ** edges[b + 1]
                bdf = pool[pool["bin"] == b]
                for g in GROUP_ORDER:
                    gdf = bdf[bdf["group"] == g]
                    if gdf.empty:
                        continue
                    n = len(gdf)
                    se = gdf["score"].std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
                    rows.append(dict(pool=pool_name, axis=axis, bin=b, range_lo=lo, range_hi=hi,
                                      range_label=_format_range(lo, hi, axis), group=g,
                                      mean=gdf["score"].mean(), se=se, n=n,
                                      n_datasets=gdf["dataset"].nunique()))
    return pd.DataFrame(rows)


def _binned_bar_page(pdf: PdfPages, binned: pd.DataFrame, axis: str, pool_name: str,
                      title: str, score_label: str) -> None:
    sub = binned[(binned["axis"] == axis) & (binned["pool"] == pool_name)].sort_values("bin")
    if sub.empty:
        return
    bins = sorted(sub["bin"].unique())
    groups = [g for g in GROUP_ORDER if g in sub["group"].unique()]
    range_labels = [sub.loc[sub["bin"] == b, "range_label"].iloc[0] for b in bins]

    fig, ax = plt.subplots(figsize=(max(9.0, 1.7 * len(bins)), 5.2))
    n_groups = len(groups)
    bar_w = 0.82 / n_groups
    x0 = np.arange(len(bins))

    for gi, g in enumerate(groups):
        offs = x0 + (gi - (n_groups - 1) / 2) * bar_w
        heights, errs, ns = [], [], []
        for b in bins:
            row = sub[(sub["bin"] == b) & (sub["group"] == g)]
            if row.empty:
                heights.append(np.nan); errs.append(0.0); ns.append(0)
                continue
            heights.append(float(row["mean"].iloc[0]))
            errs.append(float(row["se"].iloc[0]) if pd.notna(row["se"].iloc[0]) else 0.0)
            ns.append(int(row["n"].iloc[0]))
        ax.bar(offs, heights, width=bar_w * 0.94, color=GROUP_COLORS[g], label=GROUP_DISPLAY[g],
               yerr=errs, capsize=1.5, error_kw=dict(lw=0.7))
        for xo, h, n in zip(offs, heights, ns):
            if n and not np.isnan(h):
                ax.text(xo, 0.0, f"n={n}", ha="center",
                        va="bottom" if h >= 0 else "top", fontsize=5.8, color="#555555", rotation=90)

    ax.axhline(0, color="#999999", lw=0.8, zorder=0)
    ax.set_xticks(x0)
    ax.set_xticklabels(range_labels, fontsize=8)
    ax.set_ylabel(score_label, fontsize=9)
    ax.set_xlabel(f"{dict(AXES)[axis]} range (bin edges = LLM group's own quantiles)", fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=10)
    ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.tick_params(labelsize=8)
    despine(ax)
    fig.subplots_adjust(top=0.90, bottom=0.22, left=0.08, right=0.98)
    pdf.savefig(fig)
    plt.close(fig)


def _method_notes_page(binned: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.axis("off")
    ceiling_signs = _check_ceiling_signs()
    lines = [
        "MATCHED-SCALE / MATCHED-PERPLEXITY REPORT — method notes",
        "",
        "Score: (delta log-lik / n_words) / noise_ceiling, for the 11 datasets with a",
        "delta-loglik ceiling. Noise ceilings are " + ceiling_signs + ",",
        "so normalising by them never flips a score's sign.",
        "",
        "Because every one of those 11 datasets' score is on this SAME ceiling-",
        "fraction scale, every (model, checkpoint, dataset) point is POOLED into",
        "one distribution per group -- not kept dataset-by-dataset. The 3 N400",
        "datasets (no delta-loglik ceiling) are pooled separately using RAW",
        "per-word delta log-lik, own panel, never mixed with the 11 above.",
        "",
        "Binning: x-axis ranges are quantile edges of the LLM group's OWN pooled",
        "log10(x) distribution (LLM has the densest coverage -- Pythia's 110",
        "training checkpoints x every dataset -- so its quantiles give evenly-",
        "populated bins for the sparser groups to fall into). Every group's",
        "points landing in a range are averaged together, whatever dataset or",
        "checkpoint step they came from. Bars: pooled mean +/- 1 SE; n= is the",
        "point count backing that bar (labelled on each bar).",
        "",
        "A missing bar means that group has zero models in that range -- not",
        "zero score.",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
            family="monospace", va="top", ha="left")
    return fig


def main() -> None:
    use_paper_style()
    if not MASTER_PATH.exists():
        raise SystemExit(f"{MASTER_PATH} not found -- run analysis/matched_scale_master_table.py first")
    master = pd.read_csv(MASTER_PATH)

    # 2026-09-07: match analysis/matched_scale_regression.py's MAIN_MODELS_ONLY
    # scope -- without this, pooled/binned means silently mix in every
    # Pythia/BabyLM training-checkpoint row alongside main models. Early
    # Pythia checkpoints (barely-trained, tiny) drag the pooled LLM mean
    # down while BabyLM's own checkpoint trajectory does not get the same
    # treatment applied to it consistently, which previously made BabyLM
    # look like it "beats" LLM on behavioral datasets in this report even
    # though the main-models-only regression shows the opposite. Keep in
    # sync with analysis/matched_scale_regression.py's flag of the same name.
    n_before = len(master)
    master = master[~master["is_checkpoint"]].copy()
    print(f"Main models only: kept {len(master)}/{n_before} rows "
          f"(dropped {n_before - len(master)} pythia/babylm checkpoint rows)")

    print("Pooling and binning across all datasets...")
    binned = compute_binned_means(master)
    BINNED_OUT.parent.mkdir(parents=True, exist_ok=True)
    binned.to_csv(BINNED_OUT, index=False)
    print(f"Wrote {len(binned)} rows -> {BINNED_OUT}")

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF_OUT) as pdf:
        fig = _method_notes_page(binned)
        pdf.savefig(fig)
        plt.close(fig)
        for axis, axis_label in AXES:
            _binned_bar_page(pdf, binned, axis, "ceiling_normalised",
                              f"All 4 groups, pooled across 11 datasets — binned by {axis_label}",
                              "mean score / noise ceiling (pooled)")
            _binned_bar_page(pdf, binned, axis, "n400_raw",
                              f"N400 only (raw score, NOT ceiling-normalised) — binned by {axis_label}",
                              "mean raw delta log-lik / word (pooled)")
        for axis, axis_label in AXES:
            _binned_bar_page(pdf, binned, axis, "behavioral_only",
                              f"Behavioral only (SPR + eye-tracking, 6 datasets) — binned by {axis_label}",
                              "mean score / noise ceiling (pooled)")
            _binned_bar_page(pdf, binned, axis, "neural_only",
                              f"Neural only (fMRI + ECoG, 5 datasets) — binned by {axis_label}",
                              "mean score / noise ceiling (pooled)")
    print(f"Wrote {PDF_OUT}")

    md_lines = ["# Matched-scale / matched-perplexity report (pooled, binned)\n",
                "Score = (delta log-lik / n_words) / noise_ceiling, pooled across all 11 "
                "ceiling-having datasets (every model x checkpoint x dataset point averaged "
                "together within each perplexity/size range); N400 shown separately as raw "
                "score (no ceiling exists for it).\n"]
    for axis, label in AXES:
        sub = binned[(binned["axis"] == axis) & (binned["pool"] == "ceiling_normalised")]
        md_lines.append(f"## {label.capitalize()}\n")
        if sub.empty:
            md_lines.append("(no data)\n")
            continue
        cols = ["range_label", "group", "mean", "se", "n", "n_datasets"]
        table = sub.sort_values(["bin", "group"])[cols].round(4)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False))
        md_lines.extend([header, sep, body, ""])
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(md_lines))
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    main()
