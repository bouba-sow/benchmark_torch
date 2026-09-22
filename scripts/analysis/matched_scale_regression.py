#!/usr/bin/env python3
"""Matched-scale regression -- compact 2-page formula + table report.

Redesigned 2026-09-10 (supervisor: "simplify"). Two analyses of the same
matched-scale master table, differing only in the unit of observation:

  PAGE 1 -- mixed-effects, per (model x dataset) row (2035 rows):
      score_norm ~ C(group, ref=LLM) [+ log10(n_params)] [+ log10(perplexity)] + (1 | dataset)

  PAGE 2 -- ordinary least squares, one row per model (100 rows), outcome is
  that model's median score_norm across datasets:
      median_score ~ C(group, ref=LLM) [+ log10(n_params)] [+ log10(median_perplexity)]

  PAGE 3 -- page 1 plus a dataset-type control: a modality (behavioral /
  neural) main effect and a group x modality interaction, so each class's gap
  vs LLM is reported separately for behavioral and neural data:
      score_norm ~ C(group, ref=LLM) * C(modality) [+ controls] + (1 | dataset)

  PAGE 4 / 5 -- one simple 4-feature model (supervisor request): the four
  classes as two binary dimensions, memory (memory-limited architecture) and
  data (developmental BabyLM data budget), instead of three group dummies:
      HS ~ memory + data + log10(size) + log10(perplexity)  [+ (1 | dataset)]
  Page 4 is plain OLS on one row per model (100 rows; score and perplexity
  median over the model's datasets). Page 5 keeps every (model, dataset)
  row (2035) and adds the dataset random intercept. Each table adds an
  "implied gap vs LLM" block: Memory-Constrained = beta_M, BabyLM = beta_D,
  Constrained-BabyLM = beta_M + beta_D (a linear contrast).

Pages 1-3 show four columns: M0 no controls, M1 +size, M2 +perplexity,
M3 +size+perplexity -- plus the R (lme4 / base lm) formula and the maths.

  * score_norm = (delta log-lik / n_words) / delta-loglik noise ceiling
    (BehavScore / NeuralScore), from
    outputs_jeanzay/matched_scale_report/master_table.csv.
  * group: LLM / Memory-Constrained / Constrained-BabyLM / BabyLM. Reference =
    LLM, so each group coefficient is that class's score shift vs a standard
    LLM with that column's controls held fixed.
  * 21 datasets with a real delta-loglik ceiling (N400 set + MEG-MASC have
    none; ns_fmri dropped -- no per-model perplexity on the fMRI corpus).
    Main (non-checkpoint) models only. Both pages use the same rows -- those
    with a perplexity value -- so the columns (and the two pages) are
    directly comparable.

Outputs:
  reports/matched_scale_regression_report.tex / .pdf
  outputs_jeanzay/matched_scale_report/regression_coefficients.csv
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
MASTER_PATH = REPO_ROOT / "outputs_jeanzay" / "matched_scale_report" / "master_table.csv"
OUT_DIR = REPO_ROOT / "outputs_jeanzay" / "matched_scale_report"
TEX_OUT = REPO_ROOT / "reports" / "matched_scale_regression_report.tex"

MAIN_MODELS_ONLY = True
_EXCLUDE_DATASETS = ["ns_fmri"]
_N400_DATASETS = ["megmasc", "nieuwland"]
GROUP_ORDER = ["LLM", "Memory-Constrained", "Constrained-BabyLM", "BabyLM"]

# Dataset type -> modality (copied from build_matched_scale_report; ns_fmri
# already dropped above).
_BEHAVIORAL_DATASETS = [
    "geco", "geco_gp", "meco", "meco_gp", "meco_sp", "naturalstories_spr",
    "onestop_fp", "onestop_gp", "onestop_sp", "provo", "provo_gp", "provo_sp",
    "ucl_reading", "zuco_reading", "zuco_reading_sr", "zuco_reading_tsr",
]
_NEURAL_DATASETS = [
    "fedorenko2016_ecog", "lpp_fmri", "pereira_fmri", "podcast_ieeg", "tuckute_fmri",
]

# (id, short column header, RHS after "<outcome> ~ ")
_VARIANTS = [
    ("M0", "no controls", "C(group, Treatment('LLM'))"),
    ("M1", "+size", "C(group, Treatment('LLM')) + log_params"),
    ("M2", "+ppl", "C(group, Treatment('LLM')) + log_ppl"),
    ("M3", "+size+ppl", "C(group, Treatment('LLM')) + log_params + log_ppl"),
]

# Page 3: group x modality. "{mod}" is filled with the C(modality, ...) term so
# the same template can be fit twice -- once with each modality as reference --
# to read a correctly-SE'd group gap for behavioral and for neural.
_VARIANTS_MOD = [
    ("M0", "no controls", "C(group, Treatment('LLM')) * {mod}"),
    ("M1", "+size", "C(group, Treatment('LLM')) * {mod} + log_params"),
    ("M2", "+ppl", "C(group, Treatment('LLM')) * {mod} + log_ppl"),
    ("M3", "+size+ppl", "C(group, Treatment('LLM')) * {mod} + log_params + log_ppl"),
]
_MOD_B = "C(modality, Treatment('behavioral'))"
_MOD_N = "C(modality, Treatment('neural'))"

# statsmodels term name -> display row label
_TERMS = [
    ("Intercept", "Intercept (LLM)"),
    ("C(group, Treatment('LLM'))[T.Memory-Constrained]", "Memory-Constrained vs LLM"),
    ("C(group, Treatment('LLM'))[T.Constrained-BabyLM]", "Constrained-BabyLM vs LLM"),
    ("C(group, Treatment('LLM'))[T.BabyLM]", "BabyLM vs LLM"),
    ("log_params", r"$\log_{10}$(n\_params)"),
    ("log_ppl", r"$\log_{10}$(perplexity)"),
]

# ---- Simple 4-feature model (supervisor request) -------------------------
# The four classes as two binary dimensions instead of three group dummies:
#   memory = 1 for the memory-limited architectures  (Memory-Constrained, Constrained-BabyLM)
#   data   = 1 for the developmental (BabyLM) data budget (BabyLM, Constrained-BabyLM)
#   Standard LLM (0,0)  Memory-Constrained (1,0)  BabyLM (0,1)  Constrained-BabyLM (1,1)
# One model, four predictors -- no interaction, no nested variants:
_MEMORY_GROUPS = {"Memory-Constrained", "Constrained-BabyLM"}
_DATA_GROUPS = {"BabyLM", "Constrained-BabyLM"}
_RHS_FAC = "memory + data + log_params + log_ppl"

_TERMS_FAC = [
    ("Intercept", "Intercept (Standard LLM)"),
    ("memory", r"Memory constraint\ \ $\beta_M$"),
    ("data", r"Data constraint\ \ $\beta_D$"),
    ("log_params", r"$\log_{10}$(n\_params)"),
    ("log_ppl", r"$\log_{10}$(perplexity)"),
]

# Page 6: add the memory x data interaction (equivalent in df to the original
# 4-group coding).
_RHS_FAC_INT = "memory * data + log_params + log_ppl"
_TERMS_FAC_INT = [
    ("Intercept", "Intercept (Standard LLM)"),
    ("memory", r"Memory constraint\ \ $\beta_M$"),
    ("data", r"Data constraint\ \ $\beta_D$"),
    ("memory:data", r"Memory $\times$ Data\ \ $\beta_{MD}$"),
    ("log_params", r"$\log_{10}$(n\_params)"),
    ("log_ppl", r"$\log_{10}$(perplexity)"),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _prepare() -> pd.DataFrame:
    """Per-(model, dataset) rows: 21 ceiling-having datasets, main models,
    restricted to rows that have a perplexity value."""
    df = pd.read_csv(MASTER_PATH)
    df = df[~df["dataset"].isin(_N400_DATASETS + _EXCLUDE_DATASETS)]
    df = df[df["score_norm"].notna()]
    if MAIN_MODELS_ONLY:
        df = df[~df["is_checkpoint"]]
    df = df.copy()
    df["log_params"] = np.log10(df["n_params"])
    with np.errstate(divide="ignore"):
        df["log_ppl"] = np.where(df["perplexity"] > 0, np.log10(df["perplexity"]), np.nan)
    df = df.dropna(subset=["log_params", "log_ppl"]).copy()
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER)
    df["modality"] = np.select(
        [df["dataset"].isin(_BEHAVIORAL_DATASETS), df["dataset"].isin(_NEURAL_DATASETS)],
        ["behavioral", "neural"], default="?")
    if (df["modality"] == "?").any():
        bad = sorted(df.loc[df["modality"] == "?", "dataset"].unique())
        raise ValueError(f"dataset(s) not in behavioral/neural lists: {bad}")
    df["modality"] = pd.Categorical(df["modality"], categories=["behavioral", "neural"])
    df["memory"] = df["group"].isin(_MEMORY_GROUPS).astype(int)
    df["data"] = df["group"].isin(_DATA_GROUPS).astype(int)
    return df


def _collapse_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: median score_norm over its datasets; log_ppl = median of
    the per-dataset log10(perplexity)."""
    g = df.groupby("model_id", observed=True)
    pm = pd.DataFrame({
        "group": g["group"].first(),
        "memory": g["memory"].first(),
        "data": g["data"].first(),
        "log_params": g["log_params"].first(),
        "log_ppl": g["log_ppl"].median(),
        "n_datasets": g["dataset"].nunique(),
        "score_norm": g["score_norm"].median(),   # named score_norm so formulas match
    }).reset_index()
    pm["group"] = pd.Categorical(pm["group"], categories=GROUP_ORDER)
    return pm


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

def _fit_lmm(df: pd.DataFrame, rhs: str):
    return smf.mixedlm(f"score_norm ~ {rhs}", data=df, groups=df["dataset"]).fit(
        reml=True, method="lbfgs")


def _fit_ols(pm: pd.DataFrame, rhs: str):
    return smf.ols(f"score_norm ~ {rhs}", data=pm).fit()


def _stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def _cell(res, term: str) -> str:
    if term not in res.params.index:
        return r"\textemdash"
    return f"{res.params[term]:+.3f} ({res.bse[term]:.3f}){_stars(res.pvalues[term])}"


def _icc(res) -> float:
    tau2 = float(np.asarray(res.cov_re)[0, 0])
    return tau2 / (tau2 + float(res.scale))


def _fe_names(res) -> list[str]:
    """Fixed-effect parameter names (MixedLMResults.t_test wants an r_matrix
    with len(fe_params) columns, not len(params); OLS has no fe_params)."""
    return list(getattr(res, "fe_params", res.params).index)


def _contrast(res, terms: list[str]):
    """(estimate, se, p) for the sum of the named coefficients, e.g.
    ['memory', 'data'] -> Constrained-BabyLM vs Standard LLM."""
    names = _fe_names(res)
    row = np.zeros((1, len(names)))
    for t in terms:
        row[0, names.index(t)] = 1.0
    tt = res.t_test(row)
    return (float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0]),
            float(np.ravel(tt.pvalue)[0]))


def _contrast_cell(res, terms: list[str]) -> str:
    est, se, p = _contrast(res, terms)
    return f"{est:+.3f} ({se:.3f}){_stars(p)}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_R_LMM = r"""# lme4 / lmerTest ; d = one row per (model, dataset) ; group: factor, ref = "LLM"
M0 <- lmer(score_norm ~ group + (1|dataset), data=d)
M1 <- lmer(score_norm ~ group + log10(n_params) + (1|dataset), data=d)
M2 <- lmer(score_norm ~ group + log10(perplexity) + (1|dataset), data=d)
M3 <- lmer(score_norm ~ group + log10(n_params) + log10(perplexity) + (1|dataset), data=d)"""

_R_OLS = r"""# base R ; pm = one row per model
#   median_score      = median of score_norm      over that model's datasets
#   median_log10_ppl  = median of log10(perplexity) over that model's datasets
M0 <- lm(median_score ~ group, data=pm)
M1 <- lm(median_score ~ group + log10(n_params), data=pm)
M2 <- lm(median_score ~ group + median_log10_ppl, data=pm)
M3 <- lm(median_score ~ group + log10(n_params) + median_log10_ppl, data=pm)"""

_MATH_LMM = r"""\[
\begin{aligned}
y_{ij} = {}& \beta_0
 + \beta_1\,\mathrm{MC}_i + \beta_2\,\mathrm{CB}_i + \beta_3\,\mathrm{BL}_i \\
 & + \beta_4\,\log_{10}\mathrm{params}_i
   + \beta_5\,\log_{10}\mathrm{ppl}_{ij}
   + u_j + \varepsilon_{ij}
\end{aligned}
\]
\vspace{-4pt}
\[
u_j \sim \mathcal{N}(0,\tau^2),\qquad
\varepsilon_{ij} \sim \mathcal{N}(0,\sigma^2),\qquad
\mathrm{ICC} = \frac{\tau^2}{\tau^2+\sigma^2}
\]
Observation $i$ in dataset $j$. $\mathrm{MC}_i,\mathrm{CB}_i,\mathrm{BL}_i$ are
0/1 indicators for the model's class (LLM = reference $\Rightarrow$ all three
$=0$); $u_j$ is the dataset random intercept. M0--M2 drop $\beta_4$ and/or
$\beta_5$."""

_MATH_OLS = r"""\[
\tilde y_i = \operatorname*{median}_{j=1,\dots,K_i}\, y_{ij}
\qquad\text{(median over the $K_i$ datasets model $i$ was run on)}
\]
\vspace{-4pt}
\[
\tilde y_i = \beta_0
 + \beta_1\,\mathrm{MC}_i + \beta_2\,\mathrm{CB}_i + \beta_3\,\mathrm{BL}_i
 + \beta_4\,\log_{10}\mathrm{params}_i
 + \beta_5\,\widetilde{\log_{10}\mathrm{ppl}}_i
 + \varepsilon_i,\qquad
\varepsilon_i \sim \mathcal{N}(0,\sigma^2)
\]
One row per model ($N$ = number of models). No dataset term -- each model
contributes a single averaged score. M0--M2 drop $\beta_4$ and/or $\beta_5$."""

_R_LMM_MOD = r"""# lme4 ; d = one row per (model, dataset) ; modality: factor {behavioral, neural}
M0 <- lmer(score_norm ~ group * modality + (1|dataset), data=d)
M1 <- lmer(score_norm ~ group * modality + log10(n_params) + (1|dataset), data=d)
M2 <- lmer(score_norm ~ group * modality + log10(perplexity) + (1|dataset), data=d)
M3 <- lmer(score_norm ~ group * modality + log10(n_params) + log10(perplexity) + (1|dataset), data=d)"""

_MATH_LMM_MOD = r"""\[
\begin{aligned}
y_{ij} = {}& \beta_0
 + \underbrace{\gamma_1\mathrm{MC}_i + \gamma_2\mathrm{CB}_i + \gamma_3\mathrm{BL}_i}_{\text{class gap, behavioral}}
 + \delta_0\,\mathrm{Neural}_j \\
 & + \underbrace{\delta_1(\mathrm{MC}_i\!\cdot\!\mathrm{Neural}_j)
   + \delta_2(\mathrm{CB}_i\!\cdot\!\mathrm{Neural}_j)
   + \delta_3(\mathrm{BL}_i\!\cdot\!\mathrm{Neural}_j)}_{\text{extra gap on neural data}} \\
 & + \beta_4\log_{10}\mathrm{params}_i + \beta_5\log_{10}\mathrm{ppl}_{ij}
   + u_j + \varepsilon_{ij}
\end{aligned}
\]
\vspace{-4pt}
\[
u_j \sim \mathcal{N}(0,\tau^2),\qquad \varepsilon_{ij}\sim\mathcal{N}(0,\sigma^2)
\]
$\mathrm{Neural}_j = 1$ if dataset $j$ is neural. A class's behavioral gap vs
LLM is $\gamma_k$; its neural gap is $\gamma_k + \delta_k$. The table below
reports each directly (the neural block is the same model re-fit with
\texttt{{neural}} as the modality reference, so every SE / $p$ is exact).
M0--M2 drop $\beta_4$ and/or $\beta_5$."""


def _table_mod(behav_fits: list, neural_fits: list) -> str:
    """group x modality table: behavioral gaps from the behavioral-reference
    fits, neural gaps from the neural-reference fits; shared terms (size, ppl,
    modality main effect, variance components) from the behavioral fits."""
    col_spec = "l" + "r" * len(_VARIANTS_MOD)
    head = " & ".join([r"\textbf{Term}"] + [rf"\textbf{{{v[0]}}}" for v in _VARIANTS_MOD]) + r" \\"
    sub = " & ".join([""] + [rf"{{\scriptsize {v[1]}}}" for v in _VARIANTS_MOD]) + r" \\"
    grp_terms = _TERMS[1:4]  # the three "X vs LLM" rows

    def block(title, fits):
        out = [rf"\multicolumn{{{1+len(_VARIANTS_MOD)}}}{{@{{}}l}}{{\itshape {title}}} \\"]
        for term, label in grp_terms:
            out.append(f"\\quad {label} & " + " & ".join(_cell(r, term) for r in fits) + r" \\")
        return "\n".join(out)

    mod_term = f"{_MOD_B}[T.neural]"
    tail = "\n".join([
        f"{lab} & " + " & ".join(_cell(r, term) for r in behav_fits) + r" \\"
        for term, lab in [("log_params", _TERMS[4][1]), ("log_ppl", _TERMS[5][1]),
                          (mod_term, "Neural vs behavioral (LLM)")]
    ])
    var_rows = "\n".join([
        "Random-intercept SD & " + " & ".join(
            f"{np.sqrt(np.asarray(r.cov_re)[0,0]):.4f}" for r in behav_fits) + r" \\",
        "Residual SD & " + " & ".join(f"{np.sqrt(r.scale):.4f}" for r in behav_fits) + r" \\",
        "ICC (dataset) & " + " & ".join(f"{_icc(r):.3f}" for r in behav_fits) + r" \\",
        "$N$ rows & " + " & ".join(str(int(r.nobs)) for r in behav_fits) + r" \\",
    ])
    return rf"""\begin{{center}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{{col_spec}}}
\toprule
{head}
{sub}
\midrule
{block("Behavioral (class gap vs LLM)", behav_fits)}
\addlinespace
{block("Neural (class gap vs LLM)", neural_fits)}
\midrule
{tail}
\midrule
{var_rows}
\bottomrule
\end{{tabular}}
\end{{center}}"""


def _table(fits: list, extra_rows: list[tuple[str, list[str]]], n_label: str, n_val) -> str:
    col_spec = "l" + "r" * len(_VARIANTS)
    head = " & ".join([r"\textbf{Term}"] + [rf"\textbf{{{v[0]}}}" for v in _VARIANTS]) + r" \\"
    sub = " & ".join([""] + [rf"{{\scriptsize {v[1]}}}" for v in _VARIANTS]) + r" \\"
    body = "\n".join(
        f"{label} & " + " & ".join(_cell(res, term) for res in fits) + r" \\"
        for term, label in _TERMS
    )
    extra = "\n".join(f"{lab} & " + " & ".join(vals) + r" \\" for lab, vals in extra_rows)
    n_row = f"{n_label} & " + " & ".join([str(n_val)] * len(_VARIANTS)) + r" \\"
    return rf"""\begin{{center}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{{col_spec}}}
\toprule
{head}
{sub}
\midrule
{body}
\midrule
{extra}
{n_row}
\bottomrule
\end{{tabular}}
\end{{center}}"""


_MATH_FAC_OLS = r"""\[
\widetilde{\mathrm{HS}}_i = \beta_0
 + \beta_M\,\mathrm{Memory}_i + \beta_D\,\mathrm{Data}_i
 + \beta_S\,\log_{10}\mathrm{params}_i
 + \beta_P\,\widetilde{\log_{10}\mathrm{ppl}}_i
 + \varepsilon_i
\]"""

_MATH_FAC_LMM = r"""\[
\mathrm{HS}_{ij} = \beta_0
 + \beta_M\,\mathrm{Memory}_i + \beta_D\,\mathrm{Data}_i
 + \beta_S\,\log_{10}\mathrm{params}_i
 + \beta_P\,\log_{10}\mathrm{ppl}_{ij}
 + u_j + \varepsilon_{ij}
\]"""

_MATH_FAC_LMM_INT = r"""\[
\mathrm{HS}_{ij} = \beta_0
 + \beta_M\,\mathrm{Memory}_i + \beta_D\,\mathrm{Data}_i
 + \beta_{MD}\,(\mathrm{Memory}_i\!\cdot\!\mathrm{Data}_i)
 + \beta_S\,\log_{10}\mathrm{params}_i
 + \beta_P\,\log_{10}\mathrm{ppl}_{ij}
 + u_j + \varepsilon_{ij}
\]"""

_FAC_DEF = (r"\texttt{memory}\,$=1$ for the memory-limited architectures "
            r"(Memory-Constrained, Constrained-BabyLM); \texttt{data}\,$=1$ for "
            r"the BabyLM data budget (BabyLM, Constrained-BabyLM). "
            r"$\beta_M+\beta_D$ is the Constrained-BabyLM effect.")

_FAC_DEF_INT = (_FAC_DEF + r" With the interaction, Constrained-BabyLM "
                r"$=\beta_M+\beta_D+\beta_{MD}$ (equivalent to the original "
                r"4-group coding).")


def _table_fac(res, extra_rows, n_label: str, n_val, terms=_TERMS_FAC, combo=("memory", "data")) -> str:
    """Single-fit factorial table: the coefficients (+ the Constrained-BabyLM
    combo contrast) then fit statistics."""
    rows = []
    for term, label in terms:
        rows.append(f"{label} & {_cell(res, term)}" + r" \\")
        if term == (combo[-1] if len(combo) > 2 else "data"):
            combo_label = r"$\beta_M+\beta_D" + (r"+\beta_{MD}$" if len(combo) > 2 else "$")
            rows.append(rf"Memory and data constraint\ \ {combo_label} & "
                        + _contrast_cell(res, list(combo)) + r" \\")
    body = "\n".join(rows)
    extra = "\n".join(f"{lab} & {val}" + r" \\" for lab, val in extra_rows)
    return rf"""\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Term}} & \textbf{{Coefficient (SE)}} \\
\midrule
{body}
\midrule
{extra}
{n_label} & {n_val} \\
\bottomrule
\end{{tabular}}
\end{{center}}"""


def _build_report(df: pd.DataFrame, pm: pd.DataFrame, lmm_fits: list, ols_fits: list,
                  p3_behav: list, p3_neural: list, fac_ols, fac_lmm, fac_lmm_int) -> None:
    n_obs = int(lmm_fits[0].nobs)
    n_ds = df["dataset"].nunique()
    n_models = df["model_id"].nunique()
    gc = df.groupby("group", observed=True)["model_id"].nunique().reindex(GROUP_ORDER)
    grp_line = (f"LLM {int(gc['LLM'])}, Memory-Constrained {int(gc['Memory-Constrained'])}, "
                f"Constrained-BabyLM {int(gc['Constrained-BabyLM'])}, BabyLM {int(gc['BabyLM'])}")

    lmm_extra = [
        ("Random-intercept SD",
         [f"{np.sqrt(np.asarray(r.cov_re)[0, 0]):.4f}" for r in lmm_fits]),
        ("Residual SD", [f"{np.sqrt(r.scale):.4f}" for r in lmm_fits]),
        ("ICC (dataset)", [f"{_icc(r):.3f}" for r in lmm_fits]),
    ]
    ols_extra = [
        ("Residual SD", [f"{np.sqrt(r.scale):.4f}" for r in ols_fits]),
        ("$R^2$", [f"{r.rsquared:.3f}" for r in ols_fits]),
        ("adj.\\ $R^2$", [f"{r.rsquared_adj:.3f}" for r in ols_fits]),
    ]
    fac_ols_extra = [
        ("Residual SD", f"{np.sqrt(fac_ols.scale):.4f}"),
        ("$R^2$", f"{fac_ols.rsquared:.3f}"),
        ("adj.\\ $R^2$", f"{fac_ols.rsquared_adj:.3f}"),
    ]
    fac_lmm_extra = [
        ("Random-intercept SD", f"{np.sqrt(np.asarray(fac_lmm.cov_re)[0, 0]):.4f}"),
        ("Residual SD", f"{np.sqrt(fac_lmm.scale):.4f}"),
        ("ICC (dataset)", f"{_icc(fac_lmm):.3f}"),
    ]
    fac_lmm_int_extra = [
        ("Random-intercept SD", f"{np.sqrt(np.asarray(fac_lmm_int.cov_re)[0, 0]):.4f}"),
        ("Residual SD", f"{np.sqrt(fac_lmm_int.scale):.4f}"),
        ("ICC (dataset)", f"{_icc(fac_lmm_int):.3f}"),
    ]

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{fancyvrb}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{5pt}}
\begin{{document}}

%% ---------------- PAGE 1 : mixed-effects, per (model x dataset) row ----------
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (1) mixed-effects, per observation}}
\end{{center}}

Linear mixed model on the {n_obs} per-(model, dataset) rows, fit by REML.
Outcome \texttt{{score\_norm}} = ceiling-normalised score
$(\Delta\text{{log-lik}}/n_\text{{words}})$ divided by each dataset's
$\Delta$log-lik split-half noise ceiling (BehavScore\,/\,NeuralScore).

\textbf{{R (lme4):}}
\begin{{Verbatim}}[fontsize=\scriptsize]
{_R_LMM}
\end{{Verbatim}}

\textbf{{Model:}}
{_MATH_LMM}

\textbf{{Sample:}} {n_obs} (model $\times$ dataset) rows over {n_ds} datasets
with a real $\Delta$log-lik noise ceiling (N400 datasets and MEG-MASC have
none; \texttt{{ns\_fmri}} dropped). Main (non-checkpoint) models only:
{n_models} models --- {grp_line}. All four columns use the same rows.

{_table(lmm_fits, lmm_extra, "$N$ rows", n_obs)}

Each cell: coefficient (standard error).
\textsuperscript{{*}}$p<0.05$,\;
\textsuperscript{{**}}$p<0.01$,\;
\textsuperscript{{***}}$p<0.001$.

\newpage
%% ---------------- PAGE 2 : OLS, one row per model --------------------------
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (2) OLS, one row per model}}
\end{{center}}

Same data, but each model is collapsed to a single outcome: the median of its
\texttt{{score\_norm}} over the datasets it was run on. $N = {len(pm)}$ models,
so this is an ordinary least-squares fit with no dataset term. This is the
``{len(pm)}-row'' version --- coefficients are not expected to match page~1,
which weights every (model, dataset) observation and absorbs between-dataset
differences into a random intercept.

\textbf{{R (base):}}
\begin{{Verbatim}}[fontsize=\scriptsize]
{_R_OLS}
\end{{Verbatim}}

\textbf{{Model:}}
{_MATH_OLS}

\textbf{{Sample:}} {len(pm)} models --- {grp_line}. Outcome and
$\widetilde{{\log_{{10}}\mathrm{{ppl}}}}_i$ are medians over each model's datasets
(median {int(pm['n_datasets'].median())} of {n_ds}).

{_table(ols_fits, ols_extra, "$N$ models", len(pm))}

Each cell: coefficient (standard error).
\textsuperscript{{*}}$p<0.05$,\;
\textsuperscript{{**}}$p<0.01$,\;
\textsuperscript{{***}}$p<0.001$.

\newpage
%% ---------------- PAGE 3 : mixed-effects, group x modality ----------------
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (3) mixed-effects, behavioral vs neural split}}
\end{{center}}

Page 1 re-fit with a \texttt{{modality}} (behavioral / neural) main effect
\emph{{and}} a \texttt{{group $\times$ modality}} interaction, so each class's
gap vs LLM is estimated separately for behavioral and for neural datasets.
Same {n_obs} rows, same \texttt{{(1 $\mid$ dataset)}} random intercept, REML.

\textbf{{R (lme4):}}
\begin{{Verbatim}}[fontsize=\scriptsize]
{_R_LMM_MOD}
\end{{Verbatim}}

\textbf{{Model:}}
{_MATH_LMM_MOD}

\textbf{{Sample:}} {n_obs} rows --- behavioral
{int((df['modality'] == 'behavioral').sum())}
(over {int(df.loc[df['modality'] == 'behavioral', 'dataset'].nunique())} datasets),
neural {int((df['modality'] == 'neural').sum())}
(over {int(df.loc[df['modality'] == 'neural', 'dataset'].nunique())} datasets).
Models: {grp_line}.

{_table_mod(p3_behav, p3_neural)}

Each cell: coefficient (standard error).
\textsuperscript{{*}}$p<0.05$,\;
\textsuperscript{{**}}$p<0.01$,\;
\textsuperscript{{***}}$p<0.001$.

\newpage
%% ---------------- PAGE 4 : simple 4-feature model, no random intercept ----
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (4) 4-feature model, OLS, one row per model}}
\end{{center}}

{_FAC_DEF}

\textbf{{Model:}}
{_MATH_FAC_OLS}

\textbf{{Samples:}} $N = {len(pm)}$ models (LLM {int(gc['LLM'])},
Memory-Constrained {int(gc['Memory-Constrained'])},
Constrained-BabyLM {int(gc['Constrained-BabyLM'])},
BabyLM {int(gc['BabyLM'])}); \texttt{{score\_norm}} and $\log_{{10}}$(perplexity)
are medians over each model's datasets.

{_table_fac(fac_ols, fac_ols_extra, "$N$ models", len(pm))}

{{\footnotesize Coefficient (SE).\;
\textsuperscript{{*}}$p<0.05$\;
\textsuperscript{{**}}$p<0.01$\;
\textsuperscript{{***}}$p<0.001$}}

\newpage
%% ---------------- PAGE 5 : simple 4-feature model + random intercept ------
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (5) 4-feature model, mixed, + dataset random intercept}}
\end{{center}}

{_FAC_DEF}

\textbf{{Model:}}
{_MATH_FAC_LMM}

\textbf{{Samples:}} $N = {n_obs}$ (model $\times$ dataset) rows over {n_ds}
datasets, {n_models} models (LLM {int(gc['LLM'])},
Memory-Constrained {int(gc['Memory-Constrained'])},
Constrained-BabyLM {int(gc['Constrained-BabyLM'])},
BabyLM {int(gc['BabyLM'])}); REML.

{_table_fac(fac_lmm, fac_lmm_extra, "$N$ rows", n_obs)}

{{\footnotesize Coefficient (SE).\;
\textsuperscript{{*}}$p<0.05$\;
\textsuperscript{{**}}$p<0.01$\;
\textsuperscript{{***}}$p<0.001$.\;
ICC $=\hat\tau^2/(\hat\tau^2+\hat\sigma^2)$}}

\newpage
%% ---------------- PAGE 6 : 4-feature model + interaction + random intercept
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (6) 4-feature model + interaction, mixed, + dataset random intercept}}
\end{{center}}

{_FAC_DEF_INT}

\textbf{{Model:}}
{_MATH_FAC_LMM_INT}

\textbf{{Samples:}} $N = {n_obs}$ (model $\times$ dataset) rows over {n_ds}
datasets, {n_models} models (LLM {int(gc['LLM'])},
Memory-Constrained {int(gc['Memory-Constrained'])},
Constrained-BabyLM {int(gc['Constrained-BabyLM'])},
BabyLM {int(gc['BabyLM'])}); REML.

{_table_fac(fac_lmm_int, fac_lmm_int_extra, "$N$ rows", n_obs, terms=_TERMS_FAC_INT, combo=("memory", "data", "memory:data"))}

{{\footnotesize Coefficient (SE).\;
\textsuperscript{{*}}$p<0.05$\;
\textsuperscript{{**}}$p<0.01$\;
\textsuperscript{{***}}$p<0.001$.\;
ICC $=\hat\tau^2/(\hat\tau^2+\hat\sigma^2)$}}

\end{{document}}
"""
    TEX_OUT.write_text(tex)
    print(f"wrote {TEX_OUT}")

    cmd = (f"source /usr/share/Modules/init/bash && module load texlive/20240312 && "
           f"cd {TEX_OUT.parent} && "
           f"pdflatex -interaction=nonstopmode -halt-on-error {TEX_OUT.name} >/dev/null 2>&1 && "
           f"pdflatex -interaction=nonstopmode -halt-on-error {TEX_OUT.name} >/dev/null 2>&1")
    rc = subprocess.run(["bash", "-c", cmd]).returncode
    pdf = TEX_OUT.with_suffix(".pdf")
    if rc == 0 and pdf.exists():
        for ext in (".aux", ".log", ".out"):
            TEX_OUT.with_suffix(ext).unlink(missing_ok=True)
        print(f"wrote {pdf}")
    else:
        print(f"WARNING: pdflatex failed (rc={rc}); .tex written, compile manually")


def _log_fits(tag: str, fits: list, variants=_VARIANTS) -> list[dict]:
    rows = []
    for (mid, desc, _), res in zip(variants, fits):
        print(f"\n=== [{tag}] {mid} ({desc})   N={int(res.nobs)}")
        for term in res.params.index:
            if term == "Group Var":
                continue
            if term in dict(_TERMS) or term in dict(_TERMS_FAC) or ":" in term or "modality" in term:
                print(f"    {term:<58} {res.params[term]:+.4f} "
                      f"({res.bse[term]:.4f}) p={res.pvalues[term]:.3g}")
            rows.append({"analysis": tag, "model": mid, "controls": desc, "term": term,
                         "coef": float(res.params[term]), "std_err": float(res.bse[term]),
                         "p_value": float(res.pvalues[term])})
    return rows


def main() -> None:
    if not MASTER_PATH.exists():
        raise SystemExit(f"{MASTER_PATH} not found -- run analysis/matched_scale_master_table.py first")
    df = _prepare()
    pm = _collapse_per_model(df)

    lmm_fits = [_fit_lmm(df, rhs) for _, _, rhs in _VARIANTS]
    ols_fits = [_fit_ols(pm, rhs) for _, _, rhs in _VARIANTS]
    p3_behav = [_fit_lmm(df, rhs.format(mod=_MOD_B)) for _, _, rhs in _VARIANTS_MOD]
    p3_neural = [_fit_lmm(df, rhs.format(mod=_MOD_N)) for _, _, rhs in _VARIANTS_MOD]
    fac_ols = smf.ols(f"score_norm ~ {_RHS_FAC}", data=pm).fit()    # per-model (100 rows), no RE
    fac_lmm = _fit_lmm(df, _RHS_FAC)                                # per-observation, + (1|dataset)
    fac_lmm_int = _fit_lmm(df, _RHS_FAC_INT)                        # + memory:data interaction

    _fac_variant = [("fac", "memory+data+size+ppl", _RHS_FAC)]
    _fac_int_variant = [("fac_int", "memory*data+size+ppl", _RHS_FAC_INT)]
    coef_rows = (_log_fits("lmm_per_observation", lmm_fits)
                 + _log_fits("ols_per_model", ols_fits)
                 + _log_fits("lmm_group_x_modality_behav_ref", p3_behav, _VARIANTS_MOD)
                 + _log_fits("lmm_group_x_modality_neural_ref", p3_neural, _VARIANTS_MOD)
                 + _log_fits("factorial_ols_per_model_no_random_intercept", [fac_ols], _fac_variant)
                 + _log_fits("factorial_lmm_per_observation_random_intercept", [fac_lmm], _fac_variant)
                 + _log_fits("factorial_lmm_random_intercept_interaction", [fac_lmm_int], _fac_int_variant))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coef_rows).to_csv(OUT_DIR / "regression_coefficients.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'regression_coefficients.csv'}")

    _build_report(df, pm, lmm_fits, ols_fits, p3_behav, p3_neural, fac_ols, fac_lmm, fac_lmm_int)


if __name__ == "__main__":
    main()
