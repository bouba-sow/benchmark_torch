#!/usr/bin/env python3
"""Matched-scale regression with ROBUST scaling (paper analysis, 2026-10-12).

Replaces the noise-ceiling normalisation of ``matched_scale_regression.py`` for the paper's regression: each dataset unit's
raw model scores are scaled as (x - median) / IQR over the models of that unit, so coefficients are in interquartile
ranges of model scores within a dataset and no noise-ceiling estimate enters the regression (ceilings are used only to
describe headroom).

Inputs: outputs/benchmark_summary/<unit>/all_models_scores.csv (raw score_last / score_best, family), perplexity from
outputs_jeanzay/matched_scale_report/master_table.csv, groups from ``family_group`` (the summary files' own group column is
stale). 27 dataset units, 118 models, both tracks, two layer rules (last = primary, best = sensitivity).

Model (reference = Standard LLM, dataset-specific intercepts, OLS), pooled over behavioral and neural dataset units:
  s ~ memory * data + log10 size + log10 ppl
(2026-10-12: the paper presents only this model; the earlier Model 1 and the behavioral / neural split are no longer fitted.)
Two standard errors are stored: the plain model-based one, and a cluster-robust one clustered by model family (22 clusters),
because models of one family -- and the same model across datasets -- are not independent. The paper tables show the
cluster-robust SE. (A two-way family x corpus clustering was tried and rejected: with a handful of clusters per subset the
covariance estimate is not positive definite.)

Outputs:
  outputs_jeanzay/matched_scale_report/robust_regression_coefficients.csv
  paperwriting/sections/results_main_table.tex        last layer, both tracks (Table: matched-scale regression)
  paperwriting/sections/results_layer_table.tex       best layer, same layout (appendix)
  paperwriting/sections/results_medians_table.tex     unadjusted group medians (last layer)
  paperwriting/sections/results_numbers.tex           \\newcommand macros for every coefficient cited in the text
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from neuroai.analysis.benchmark_common import family_group

_spec = importlib.util.spec_from_file_location("msr", REPO_ROOT / "scripts/analysis/matched_scale_regression.py")
msr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msr)

OUT_CSV = REPO_ROOT / "outputs_jeanzay" / "matched_scale_report" / "robust_regression_coefficients.csv"
PAPER_SECTIONS = REPO_ROOT / "paperwriting" / "sections"
NEURAL_UNITS = {"lpp_fmri", "ns_fmri", "pereira_fmri", "tuckute_fmri", "fedorenko2016_ecog", "broderick", "megmasc",
                "nieuwland", "szewczyk_f2007", "szewczyk_h2019", "podcast_ieeg"}
TRACKS = ("embedding", "surprisal")
LAYERS = {"last": "score_last", "best": "score_best"}
TERMS = ["memory", "data", "memory:data", "both", "log_params", "log_ppl"]
TERM_LABEL = {"memory": r"Memory constraint $\beta_M$", "data": r"Data constraint $\beta_D$",
              "memory:data": r"Memory $\times$ Data $\beta_{MD}$", "both": r"Both constraints $\beta_M{+}\beta_D{+}\beta_{MD}$",
              "log_params": r"$\log_{10}$ size $\beta_S$", "log_ppl": r"$\log_{10}$ perplexity $\beta_P$"}
MACRO_TERM = {"memory": "Mem", "data": "Data", "memory:data": "MD", "both": "Both", "log_params": "Size", "log_ppl": "Ppl"}


def build_units(track: str, col: str) -> pd.DataFrame:
    ppl = pd.read_csv(msr.SURPRISAL_MASTER)
    ppl = ppl[~ppl["is_checkpoint"]][["dataset", "model_id", "perplexity"]].drop_duplicates(["dataset", "model_id"])
    parts = []
    for src, unit in msr._SUMMARY_UNITS[track].items():
        d = pd.read_csv(msr.SUMMARY_ROOT / src / "all_models_scores.csv")
        x = d[col].where(np.isfinite(d[col]))
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        parts.append(pd.DataFrame({
            "dataset": unit, "model_id": d["model_id"], "family": d["family"],
            "group": [family_group(f, m) for f, m in zip(d["family"], d["model_id"])],
            "n_params": d["n_params"], "raw": x, "score": (x - med) / (q3 - q1)}))
    t = pd.concat(parts).merge(ppl, on=["dataset", "model_id"], how="left")
    t = t[t["score"].notna() & (t["perplexity"] > 0)].copy()
    t["log_params"] = np.log10(t["n_params"])
    t["log_ppl"] = np.log10(t["perplexity"])
    t["memory"] = t["group"].isin(msr._MEMORY_GROUPS).astype(int)
    t["data"] = t["group"].isin(msr._DATA_GROUPS).astype(int)
    t["modality"] = np.where(t["dataset"].isin(NEURAL_UNITS), "neural", "behavioral")
    t["corpus"] = t["dataset"].map(msr._base_dataset)
    return t


def _fit(d: pd.DataFrame, formula: str):
    """OLS with dataset intercepts; returns (model-based fit, cluster-robust fit clustered by model family)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = smf.ols(formula, d).fit()
        clus = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["family"])[0]})
    return base, clus


def _row(res_b, res_c, key, label, n, units):
    return dict(term=label, coef=float(res_b.params[key]), se_model=float(res_b.bse[key]), p_model=float(res_b.pvalues[key]),
                se_cluster=float(res_c.bse[key]), p_cluster=float(res_c.pvalues[key]), n=n, units=units)


def _both(res_b, res_c, keys, label, n, units):
    out = {}
    for tag, res in (("model", res_b), ("cluster", res_c)):
        names = list(res.params.index)
        row = np.zeros((1, len(names)))
        for k in keys:
            row[0, names.index(k)] = 1.0
        t = res.t_test(row)
        out[tag] = (float(np.ravel(t.effect)[0]), float(np.ravel(t.sd)[0]), float(np.ravel(t.pvalue)[0]))
    return dict(term=label, coef=out["model"][0], se_model=out["model"][1], p_model=out["model"][2],
                se_cluster=out["cluster"][1], p_cluster=out["cluster"][2], n=n, units=units)


def fit_scope(d: pd.DataFrame) -> list[dict]:
    """Model 2 on the given rows (a scope = pooled, behavioral only or neural only); same formula in every scope."""
    n, units = len(d), d["dataset"].nunique()
    rb, rc = _fit(d, "score ~ memory * data + log_params + log_ppl + C(dataset)")
    rows = [_row(rb, rc, k, t, n, units) for k, t in [("memory", "memory"), ("data", "data"), ("memory:data", "memory:data"),
                                                      ("log_params", "log_params"), ("log_ppl", "log_ppl")]]
    rows.append(_both(rb, rc, ["memory", "data", "memory:data"], "both", n, units))
    return rows


def run() -> pd.DataFrame:
    out = []
    for track in TRACKS:
        for layer, col in LAYERS.items():
            t = build_units(track, col)
            for r in fit_scope(t):
                out.append(dict(track=track, layer=layer, scope="pooled", **r))
            if layer == "last":  # unadjusted medians over all units (descriptive)
                for g, v in t.groupby("group").score.median().items():
                    out.append(dict(track=track, layer=layer, scope="pooled", term=f"median:{g}", coef=float(v),
                                    se_model=np.nan, p_model=np.nan, se_cluster=np.nan, p_cluster=np.nan,
                                    n=len(t), units=t["dataset"].nunique()))
    df = pd.DataFrame(out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


# ---------------------------------------------------------------------------- LaTeX
def _star(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def _num(x: float) -> str:
    s = f"{abs(x):.2f}"
    return ("$-$" if x < 0 and s != "0.00" else "") + s


def _cell(r) -> str:
    return rf"{_num(r.coef)} ({r.se_cluster:.2f}){_star(r.p_cluster)}"


def _coef_table(df: pd.DataFrame, layer: str, caption: str, label: str) -> str:
    d = df[(df["layer"] == layer) & (~df["term"].str.startswith("median:"))]
    rows = []
    for term in TERMS:
        cells = []
        for track in TRACKS:
            r = d[(d["track"] == track) & (d["term"] == term)].iloc[0]
            cells.append(_cell(r))
        rows.append(f"{TERM_LABEL[term]} & " + " & ".join(cells) + r" \\")
    ns = []
    for track in TRACKS:
        r = d[(d["track"] == track) & (d["term"] == "memory")].iloc[0]
        ns.append(f"{int(r.n)}/{int(r.units)}")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Term}} & \textbf{{Embedding}} & \textbf{{Surprisal}} \\
\midrule
{body}
\midrule
Rows / units & {" & ".join(ns)} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _medians_table(df: pd.DataFrame) -> str:
    d = df[df["term"].str.startswith("median:")].copy()
    d["group"] = d["term"].str.replace("median:", "", regex=False)
    order = ["LLM", "Memory-Constrained", "BabyLM", "Constrained-BabyLM"]
    display = {"LLM": "Standard LLM"}
    rows = []
    for g in order:
        cells = []
        for track in TRACKS:
            r = d[(d["track"] == track) & (d["group"] == g)]
            cells.append(_num(float(r.coef.iloc[0])) if len(r) else "--")
        rows.append(f"{display.get(g, g)} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Unadjusted group medians of the robust-scaled score (last layer, all dataset units): units are interquartile ranges of model scores within a dataset, 0 is the median model. These differences mix the effect of a constraint with differences in model size and perplexity (Table~\ref{{tab:models}}).}}
\label{{tab:medians}}
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Group}} & \textbf{{Embedding}} & \textbf{{Surprisal}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _macros(df: pd.DataFrame) -> str:
    names = {"embedding": "Emb", "surprisal": "Sur"}
    scopes = {"pooled": "Pool", "behavioral": "Beh", "neural": "Neu"}
    lay = {"last": "Last", "best": "Best"}
    lines = ["% Generated by scripts/analysis/matched_scale_robust.py -- do not edit by hand.",
             r"\providecommand{\RBsign}[1]{#1}"]
    for r in df[~df["term"].str.startswith("median:")].itertuples():
        base = f"R{lay[r.layer]}{names[r.track]}{scopes[r.scope]}{MACRO_TERM[r.term]}"
        val = f"{abs(r.coef):.2f}"
        signed = (r"$-$" if r.coef < 0 and val != "0.00" else "+") + val
        lines.append(rf"\newcommand{{\{base}}}{{{signed}}}")
        lines.append(rf"\newcommand{{\{base}SE}}{{{r.se_cluster:.2f}}}")
        lines.append(rf"\newcommand{{\{base}P}}{{{r.p_cluster:.3f}}}")
    for r in df[df["term"].str.startswith("median:")].itertuples():
        g = r.term.split(":")[1].replace("-", "").replace("Constrained", "C").replace("BabyLM", "Baby").replace("Memory", "Mem")
        lines.append(rf"\newcommand{{\RMed{names[r.track]}{scopes[r.scope]}{g}}}{{{_num(r.coef)}}}")
    return "\n".join(lines) + "\n"


def write_latex(df: pd.DataFrame) -> None:
    cap_main = (r"Matched-scale regression, last layer. Coefficients are shifts in interquartile ranges of model scores within a dataset, "
                r"relative to a Standard LLM of the same size and perplexity; cluster-robust standard errors "
                r"(clustered by model family) in parentheses. \textsuperscript{*}$p<0.05$, \textsuperscript{**}$p<0.01$, "
                r"\textsuperscript{***}$p<0.001$. Last row: model--unit rows / dataset units.")
    cap_best = (r"As Table~\ref{tab:main-regression} but with each model's best layer (embedding: highest held-out score; "
                r"surprisal: highest training $\Delta LL$).")
    (PAPER_SECTIONS / "results_main_table.tex").write_text(_coef_table(df, "last", cap_main, "tab:main-regression"))
    (PAPER_SECTIONS / "results_layer_table.tex").write_text(_coef_table(df, "best", cap_best, "tab:best-layer-regression"))
    (PAPER_SECTIONS / "results_medians_table.tex").write_text(_medians_table(df))
    (PAPER_SECTIONS / "results_numbers.tex").write_text(_macros(df))
    print("wrote LaTeX tables and macros to", PAPER_SECTIONS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-latex", action="store_true", help="only write the coefficient CSV")
    args = ap.parse_args()
    df = run()
    print(f"wrote {OUT_CSV} ({len(df)} rows)")
    if not args.no_latex:
        write_latex(df)


if __name__ == "__main__":
    main()
