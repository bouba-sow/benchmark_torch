#!/usr/bin/env python3
"""Simple per-model regression on the matched-scale data (redesign 2026-09-10).

Supervisor asked to strip the pooled mixed-effects model down to something
that fits on one slide:

  * ONE row per model (not per model x dataset). Each model's outcome is the
    mean of its ceiling-normalised scores (score_norm = delta-loglik/n_words
    over the delta-loglik noise ceiling) across every dataset it was run on.
    N = number of models (~100), so this is an ordinary least-squares fit --
    no random effects, no reference-flip trick.

  * MINIMUM features:
        median_score ~ C(category, Treatment('LLM')) + log10(n_params) + log10(perplexity)
    category in {LLM, Memory-Constrained, Constrained-BabyLM, BabyLM};
    log10(perplexity) is the mean over datasets of that model's per-corpus
    log10 perplexity (perplexity differs by stimulus set).

  * Same dataset universe as the old Model C: the 21 datasets with a real
    delta-loglik noise ceiling (N400 set + MEG-MASC have none; ns_fmri dropped
    -- no per-model perplexity on the fMRI corpus). Main (non-checkpoint)
    models only.

The primary fit pools behavioral + neural datasets into one mean per model.
Because that mixes reading-time and brain scores -- and because models were
not all run on the same datasets -- three secondary fits are reported next to
it (same formula):
    - behavioral-only mean
    - neural-only mean
    - dataset-centered: subtract each dataset's mean score_norm before
      averaging, so a model tested only on "easy" datasets is not flattered.

Outputs:
  outputs_jeanzay/matched_scale_report/simple_regression_per_model.csv
  outputs_jeanzay/matched_scale_report/simple_regression_coefficients.csv
  outputs_jeanzay/matched_scale_report/simple_regression_summary.txt
  reports/paper/matched_scale_simple_regression.pdf
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
import statsmodels.formula.api as smf
from matplotlib.backends.backend_pdf import PdfPages

from neuroai.analysis.paper_style import use_paper_style

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
MASTER_PATH = REPO_ROOT / "outputs_jeanzay" / "matched_scale_report" / "master_table.csv"
OUT_DIR = REPO_ROOT / "outputs_jeanzay" / "matched_scale_report"
PDF_OUT = REPO_ROOT / "reports" / "paper" / "matched_scale_simple_regression.pdf"

# Hard-coded copies of build_matched_scale_report.{BEHAVIORAL,NEURAL,N400}_DATASETS
# (importing that module pulls in the whole figure8 stack -- slow). Keep in sync.
BEHAVIORAL_DATASETS = [
    "geco", "geco_gp", "meco", "meco_gp", "meco_sp", "naturalstories_spr",
    "onestop_fp", "onestop_gp", "onestop_sp", "provo", "provo_gp", "provo_sp",
    "ucl_reading", "zuco_reading", "zuco_reading_sr", "zuco_reading_tsr",
]
NEURAL_DATASETS = [
    "fedorenko2016_ecog", "lpp_fmri", "ns_fmri", "pereira_fmri",
    "podcast_ieeg", "tuckute_fmri",
]
N400_DATASETS = ["megmasc", "nieuwland"]
_EXCLUDE_DATASETS = ["ns_fmri"]  # no per-model perplexity on the fMRI corpus

GROUP_ORDER = ["LLM", "Memory-Constrained", "Constrained-BabyLM", "BabyLM"]
GROUP_COLORS = {
    "LLM": "#1f77b4", "Memory-Constrained": "#d62728",
    "Constrained-BabyLM": "#9467bd", "BabyLM": "#2ca02c",
}
FORMULA = "median_score ~ C(category, Treatment('LLM')) + log_params + log_ppl"

# Pretty term names for output
_TERM_LABEL = {
    "Intercept": "Intercept (LLM)",
    "C(category, Treatment('LLM'))[T.Memory-Constrained]": "Memory-Constrained vs LLM",
    "C(category, Treatment('LLM'))[T.Constrained-BabyLM]": "Constrained-BabyLM vs LLM",
    "C(category, Treatment('LLM'))[T.BabyLM]": "BabyLM vs LLM",
    "log_params": "log10(n_params)",
    "log_ppl": "log10(perplexity)",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _rows() -> pd.DataFrame:
    """Filtered per-(model, dataset) rows: 21 ceiling-having datasets, main
    models only, with a modality tag and a per-row log10(perplexity)."""
    df = pd.read_csv(MASTER_PATH)
    df = df[~df["dataset"].isin(N400_DATASETS + _EXCLUDE_DATASETS)]
    df = df[df["score_norm"].notna()]
    df = df[~df["is_checkpoint"]].copy()
    df["modality"] = np.where(df["dataset"].isin(BEHAVIORAL_DATASETS), "behavioral", "neural")
    with np.errstate(divide="ignore"):
        df["log_ppl_row"] = np.where(df["perplexity"] > 0, np.log10(df["perplexity"]), np.nan)
    # dataset-centered score: score_norm minus that dataset's mean over models
    df["score_norm_c"] = df["score_norm"] - df.groupby("dataset")["score_norm"].transform("mean")
    return df


def _per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per model."""
    g = df.groupby("model_id")
    pm = pd.DataFrame({
        "category": g["group"].first(),
        "n_params": g["n_params"].first(),
        "n_datasets": g["dataset"].nunique(),
        "n_behavioral": g.apply(lambda x: (x["modality"] == "behavioral").sum(), include_groups=False),
        "n_neural": g.apply(lambda x: (x["modality"] == "neural").sum(), include_groups=False),
        "median_score_all": g["score_norm"].median(),
        "median_score_centered": g["score_norm_c"].median(),
        "log_ppl": g["log_ppl_row"].median(),
    })
    beh = (df[df["modality"] == "behavioral"].groupby("model_id")["score_norm"].median()
           .rename("median_score_behavioral"))
    neu = (df[df["modality"] == "neural"].groupby("model_id")["score_norm"].median()
           .rename("median_score_neural"))
    pm = pm.join(beh).join(neu)
    pm["log_params"] = np.log10(pm["n_params"])
    pm["category"] = pd.Categorical(pm["category"], categories=GROUP_ORDER)
    return pm.reset_index()


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _fit(pm: pd.DataFrame, outcome: str, label: str) -> tuple[pd.DataFrame, dict]:
    d = pm.rename(columns={outcome: "median_score"})
    d = d[d["median_score"].notna() & d["log_params"].notna() & d["log_ppl"].notna()].copy()
    res = smf.ols(FORMULA, data=d).fit()
    ci = res.conf_int()
    ci.columns = ["ci_lo", "ci_hi"]
    tab = pd.concat([res.params.rename("coef"), res.bse.rename("std_err"),
                     res.tvalues.rename("t"), res.pvalues.rename("p_value"), ci], axis=1)
    tab = tab.reset_index().rename(columns={"index": "term"})
    tab.insert(0, "fit", label)
    tab["term_label"] = tab["term"].map(_TERM_LABEL).fillna(tab["term"])
    meta = {"fit": label, "outcome": outcome, "n": int(res.nobs),
            "r2": float(res.rsquared), "r2_adj": float(res.rsquared_adj)}
    return tab, meta


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3f}"


def _summary_text(coefs: pd.DataFrame, metas: list[dict]) -> str:
    lines = []
    lines.append("SIMPLE PER-MODEL REGRESSION -- matched-scale data")
    lines.append("=" * 64)
    lines.append(f"formula: {FORMULA}")
    lines.append("one row per model; OLS; 21 ceiling-having datasets; main models only")
    lines.append("")
    for m in metas:
        lines.append(f"[{m['fit']}]  outcome={m['outcome']}  N={m['n']}  "
                     f"R^2={m['r2']:.3f}  adj R^2={m['r2_adj']:.3f}")
        sub = coefs[coefs["fit"] == m["fit"]]
        for _, r in sub.iterrows():
            if r["term"] == "Intercept":
                continue
            star = "***" if r["p_value"] < 1e-3 else "**" if r["p_value"] < 1e-2 else "*" if r["p_value"] < 0.05 else ""
            lines.append(f"    {r['term_label']:<28} {r['coef']:+.4f}  "
                         f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  p={_fmt_p(r['p_value'])} {star}")
        lines.append("")
    return "\n".join(lines)


def _forest(ax, coefs: pd.DataFrame, meta: dict) -> None:
    sub = coefs[(coefs["fit"] == meta["fit"]) & (coefs["term"] != "Intercept")].iloc[::-1]
    y = np.arange(len(sub))
    for yi, (_, r) in zip(y, sub.iterrows()):
        color = "#333333"
        for grp, gc in GROUP_COLORS.items():
            if r["term_label"].startswith(f"{grp} vs "):
                color = gc
        ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=color, lw=2)
        ax.plot(r["coef"], yi, "o", color=color, ms=6)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["term_label"])
    ax.set_xlabel("coefficient (median_score units, 95% CI)")
    ax.set_title(f"{meta['fit']}  (N={meta['n']}, adj $R^2$={meta['r2_adj']:.2f})", fontsize=10)
    ax.grid(True, axis="x", alpha=0.2)


def _scatter(ax, pm: pd.DataFrame) -> None:
    for grp in GROUP_ORDER:
        s = pm[pm["category"] == grp]
        ax.scatter(s["log_params"], s["median_score_all"], s=26, alpha=0.85,
                   color=GROUP_COLORS[grp], edgecolor="white", linewidth=0.4, label=grp)
    ax.set_xlabel("log10(n_params)")
    ax.set_ylabel("median score_norm across datasets")
    ax.set_title("Per-model outcome vs size", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, frameon=False)


def main() -> None:
    use_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)

    rows = _rows()
    pm = _per_model(rows)
    print(f"{len(pm)} models  |  {rows['dataset'].nunique()} datasets  |  "
          f"{rows.shape[0]} model x dataset rows")
    print(pm["category"].value_counts().to_string())

    fits = [
        ("median_score_all", "primary: all datasets pooled"),
        ("median_score_behavioral", "behavioral-only"),
        ("median_score_neural", "neural-only"),
        ("median_score_centered", "dataset-centered"),
    ]
    coef_frames, metas = [], []
    for outcome, label in fits:
        tab, meta = _fit(pm, outcome, label)
        coef_frames.append(tab)
        metas.append(meta)
    coefs = pd.concat(coef_frames, ignore_index=True)

    pm.to_csv(OUT_DIR / "simple_regression_per_model.csv", index=False)
    coefs.to_csv(OUT_DIR / "simple_regression_coefficients.csv", index=False)
    txt = _summary_text(coefs, metas)
    (OUT_DIR / "simple_regression_summary.txt").write_text(txt + "\n")
    print("\n" + txt)

    with PdfPages(PDF_OUT) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        _forest(axes[0], coefs, metas[0])
        _scatter(axes[1], pm)
        fig.suptitle("Simple per-model regression: median_score ~ category + log10(size) + log10(perplexity)",
                     fontsize=11, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        for ax, meta in zip(axes.ravel(), metas):
            _forest(ax, coefs, meta)
        fig.suptitle("Same formula, four outcome definitions", fontsize=11, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)
    print(f"\nwrote {PDF_OUT}")
    print(f"wrote {OUT_DIR/'simple_regression_per_model.csv'}")
    print(f"wrote {OUT_DIR/'simple_regression_coefficients.csv'}")


if __name__ == "__main__":
    main()
