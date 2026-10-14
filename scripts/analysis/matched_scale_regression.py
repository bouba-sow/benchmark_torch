#!/usr/bin/env python3
"""Matched-scale regression -- compact 3-page formula + table report.

Redesigned 2026-09-10 (supervisor: "simplify"); trimmed 2026-10-07 to the last two
models of the previous 6-page report plus one new model. The four classes are coded
as two binary dimensions, memory (memory-limited architecture) and data (developmental
BabyLM data budget), instead of three group dummies. All three are mixed models on
every (model x dataset) row with a dataset random intercept:

  PAGE 1 -- score_norm ~ memory + data + log10(size) + log10(perplexity) + (1 | dataset)
  PAGE 2 -- score_norm ~ memory * data + log10(size) + log10(perplexity) + (1 | dataset)
            (adds the memory x data interaction; equivalent in df to the 4-group coding)
  PAGE 3 -- score_norm ~ memory * data * C(modality) + log10(size) * C(modality)
                         + log10(perplexity) * C(modality) + (1 | dataset)
            (page 2 plus a behavioral / neural modality main effect and its interactions
            with memory, data, size and perplexity, so every effect is reported separately
            for behavioral and for neural datasets -- the same idea as the group x modality
            page of the earlier report)

  PAGE 4 -- (always, added 2026-10-13) page 3 on a per-dataset robust z of the score, (x - median) / (IQR/1.349),
            over ALL dataset units: no noise ceiling is used, so no dataset is excluded.

  PAGE 5 -- page 3 with dataset-level random SLOPES for size and perplexity:
            ... + (1 + log10(size) + log10(perplexity) | dataset)
            (preferred modality model: datasets within a modality need not share the size / perplexity slopes;
            size and perplexity are mean-centered here so the random-effect covariance is well conditioned --
            this moves only the intercept and the modality main effect, not the class effects)

Each table adds an "implied gap vs LLM" row: Constrained-BabyLM = beta_M + beta_D
(+ beta_MD with the interaction), a linear contrast.

  * score_norm = (delta log-lik / n_words) / delta-loglik noise ceiling
    (BehavScore / NeuralScore), from
    outputs_jeanzay/matched_scale_report/master_table.csv.
  * Reference = standard LLM (memory = data = 0), so each coefficient is a score shift
    vs a standard LLM with size and perplexity held fixed.
  * ALL 18 benchmark datasets (27 dataset x measure units). Main (non-checkpoint)
    models only; only rows with a perplexity value, so the three pages use the same rows.

Two tracks (--track): "surprisal" (default; master_table.csv, delta-loglik / DLL ceiling) and "embedding"
(master_table_embed.csv, ridge-encoding Pearson r / correlation ceiling; its perplexity is joined from the surprisal
master table on (dataset, model_id) since perplexity is a property of the corpus text and the model).

  --layer last (added 2026-10-12): scores are each model's LAST layer (no layer selection, same rule for every model and
  both tracks), read from outputs/benchmark_summary/<dataset>/all_models_scores.csv (column score_last_norm) instead of the
  best-layer master tables, and a fifth page is added:

  PAGE 6 -- page 3 with dataset random intercepts AND slopes (size, perplexity; independent variance components) plus crossed
            random intercepts for model FAMILY and for CORPUS (geco/meco/onestop/provo/zuco measures share items and
            participants, so FP/GP/SP are separate dataset units but one corpus):
            ... + (1 | dataset) + (0 + size | dataset) + (0 + ppl | dataset) + (1 | corpus) + (1 | family)
            (preferred: models of one family are not independent and the constraint groups contain few families, so
            without the family term the group contrasts are over-confident)
  PAGE 7 -- (surprisal only) page 6 refit without Broderick, Federmeier 2007, Hubbard 2019 and Pereira, whose delta-loglik
            noise ceilings are unusable or unstable (models above the ceiling / mostly negative scores).

Surprisal track (2026-10-12): Broderick, Federmeier 2007, Hubbard 2019 and Pereira are EXCLUDED by default (their delta-loglik
ceilings are exceeded by many models or most normalized scores are negative, which makes the pooled coefficients meaningless);
--keep-unstable-ceilings restores all 27 units. The embedding track uses correlation ceilings and is not affected.

Default report = pages 1-4 (dataset random intercept); --extended adds page 5 (random slopes) and, with --layer last, pages 6-7.

Outputs (<track> = surprisal | embedding):
  reports/matched_scale_regression_report_<track>.tex / .pdf
  outputs_jeanzay/matched_scale_report/regression_coefficients_<track>.csv
  outputs_jeanzay/matched_scale_report/master_table_<track>_last.csv   (only with --layer last)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from neuroai.analysis.benchmark_common import family_group

REPO_ROOT =next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT_DIR = REPO_ROOT / "outputs_jeanzay" / "matched_scale_report"
SURPRISAL_MASTER = OUT_DIR / "master_table.csv"
TRACK_MASTER = {"surprisal": SURPRISAL_MASTER, "embedding": OUT_DIR / "master_table_embed.csv"}
# Set by main() from --track.
TRACK = "surprisal"
MASTER_PATH = SURPRISAL_MASTER
TEX_OUT = REPO_ROOT / "reports" / "matched_scale_regression_report_surprisal.tex"
# Set by main() from --layer: "best" = the master tables as built (best layer), "last" = last-layer summary scores.
LAYER = "best"
SUMMARY_ROOT = REPO_ROOT / "outputs" / "benchmark_summary"

# benchmark_summary directory -> dataset unit name used by the master tables.
_SUMMARY_UNITS = {
    "embedding": {
        "geco__embed": "geco", "geco_gp__embed": "geco_gp", "meco__embed": "meco", "meco_gp__embed": "meco_gp",
        "meco_sp__embed": "meco_sp", "naturalstories_spr__embed": "naturalstories_spr", "onestop__embed": "onestop_fp",
        "onestop_gp__embed": "onestop_gp", "onestop_sp__embed": "onestop_sp", "provo__embed": "provo",
        "provo_gp__embed": "provo_gp", "provo_sp__embed": "provo_sp", "ucl_reading__embed": "ucl_reading",
        "zuco_reading__embed": "zuco_reading", "zuco_reading_sr__embed": "zuco_reading_sr",
        "zuco_reading_tsr__embed": "zuco_reading_tsr",
        "lpp_fmri": "lpp_fmri", "ns_fmri": "ns_fmri", "pereira_fmri__embed": "pereira_fmri", "tuckute_fmri": "tuckute_fmri",
        "fedorenko2016_ecog": "fedorenko2016_ecog", "broderick_eeg": "broderick", "megmasc_meg": "megmasc",
        "nieuwland_eeg": "nieuwland", "szewczyk_f2007_eeg": "szewczyk_f2007", "szewczyk_h2019_eeg": "szewczyk_h2019",
        "podcast_ieeg": "podcast_ieeg",
    },
    "surprisal": {
        "geco": "geco", "geco_gp": "geco_gp", "meco": "meco", "meco_gp": "meco_gp", "meco_sp": "meco_sp",
        "naturalstories_spr": "naturalstories_spr", "onestop": "onestop_fp", "onestop_gp": "onestop_gp",
        "onestop_sp": "onestop_sp", "provo": "provo", "provo_gp": "provo_gp", "provo_sp": "provo_sp",
        "ucl_reading": "ucl_reading", "zuco_reading": "zuco_reading", "zuco_reading_sr": "zuco_reading_sr",
        "zuco_reading_tsr": "zuco_reading_tsr",
        "lpp_fmri__surprisal": "lpp_fmri", "ns_fmri__surprisal": "ns_fmri", "pereira_fmri": "pereira_fmri",
        "tuckute_fmri__surprisal": "tuckute_fmri", "fedorenko2016_ecog__surprisal": "fedorenko2016_ecog",
        "podcast_ieeg__surprisal": "podcast_ieeg", "broderick_eeg__surprisal": "broderick",
        "megmasc_meg__surprisal": "megmasc", "nieuwland_eeg__surprisal": "nieuwland",
        "szewczyk_f2007_eeg__surprisal": "szewczyk_f2007", "szewczyk_h2019_eeg__surprisal": "szewczyk_h2019",
    },
}
# Datasets whose delta-loglik noise ceiling is unusable (models above it, or mostly negative scores); dropped on page 6.
_UNSTABLE_DLL_CEILING = ["broderick", "szewczyk_f2007", "szewczyk_h2019", "pereira_fmri"]


def _build_last_layer_table(track: str, col: str = "score_last_norm") -> pd.DataFrame:
    """Per-(dataset unit, model) rows from the benchmark summary (column ``col``: score_last_norm or score_best_norm), with the
    perplexity of the surprisal master table (a property of corpus text and model) and the model family."""
    ppl = pd.read_csv(SURPRISAL_MASTER)
    ppl = ppl[~ppl["is_checkpoint"]][["dataset", "model_id", "perplexity"]].drop_duplicates(["dataset", "model_id"])
    # The master table only has perplexity for models its own score source found; fill every other model from the
    # dataset's perplexity table directly (same lookup the master table uses).
    from scripts.analysis.matched_scale_master_table import _perplexity_lookup
    fills = []
    for unit in set(_SUMMARY_UNITS[track].values()):
        lk = _perplexity_lookup(unit)
        lk = lk[lk["revision"].astype(str) == "main"]
        fills.append(pd.DataFrame({"dataset": unit, "model_id": lk["model_id"], "perplexity": lk["perplexity"]}))
    ppl = pd.concat([ppl.dropna(subset=["perplexity"]), *fills], ignore_index=True).drop_duplicates(
        ["dataset", "model_id"], keep="first")
    parts = []
    for src, unit in _SUMMARY_UNITS[track].items():
        d = pd.read_csv(SUMMARY_ROOT / src / "all_models_scores.csv")
        # The group column baked into the summary CSVs can be stale (it was still the pre-2026-10-07 grouping with
        # mamba/mamba2/rwkv4 as Memory-Constrained in 26 of 27 files), so assign it from the registry-based rule.
        grp = [family_group(f, m) for f, m in zip(d["family"], d["model_id"])]
        parts.append(pd.DataFrame({"dataset": unit, "group": grp, "model_id": d["model_id"], "family": d["family"],
                                   "n_params": d["n_params"], "score_norm": d[col]}))
    tab = pd.concat(parts, ignore_index=True).drop_duplicates(["dataset", "model_id"])  # summary CSVs can repeat a model
    tab = tab.merge(ppl, on=["dataset", "model_id"], how="left")
    tab["is_checkpoint"] = False
    return tab


_OUTCOME_TEXT = {
    "surprisal": (r"Outcome \texttt{score\_norm} = $\Delta$log-lik per word divided by each "
                  r"dataset's $\Delta$log-lik split-half noise ceiling."),
    "embedding": (r"Outcome \texttt{score\_norm} = ridge-encoding Pearson $r$ (hidden states $\to$ response) divided "
                  r"by each dataset's split-half correlation noise ceiling; datasets without an embedding sweep "
                  r"are absent."),
}

MAIN_MODELS_ONLY = True
# --robust-z (2026-10-13): replace score_norm by a per-dataset robust z of the same score,
# (x - median) / (IQR / 1.349), computed over the regression rows of that dataset unit. Removes the dependence on
# each dataset's noise ceiling (scale and sign), so datasets with unusable ceilings can stay in. IQR / 1.349 equals
# the SD for normal data; a dataset with IQR 0 falls back to 1.4826 * MAD.
ROBUST_Z = False
_ROBUST_Z_TEXT = (r"Outcome = per-dataset robust z of the score, $(x - \mathrm{median}) / (\mathrm{IQR}/1.349)$ over the "
                  r"models of that dataset unit, so no noise ceiling is used; coefficients are in model-spread units.")
# 2026-10-01: NO dataset is excluded any more -- all 18 benchmark datasets enter. A "dataset" unit below is one
# dataset x reading-time measure (e.g. geco FP and geco GP are two units), 27 units from the 18 datasets.
_EXCLUDE_DATASETS: list[str] = []
_N400_DATASETS: list[str] = []


def _base_dataset(slug: str) -> str:
    for b in ("zuco_reading", "onestop", "geco", "meco", "provo"):
        if slug.startswith(b):
            return b
    return slug


GROUP_ORDER = ["LLM", "Memory-Constrained", "Constrained-BabyLM", "BabyLM"]

# Dataset type -> modality (copied from build_matched_scale_report).
_BEHAVIORAL_DATASETS = [
    "geco", "geco_gp", "meco", "meco_gp", "meco_sp", "naturalstories_spr",
    "onestop_fp", "onestop_gp", "onestop_sp", "provo", "provo_gp", "provo_sp",
    "ucl_reading", "zuco_reading", "zuco_reading_sr", "zuco_reading_tsr",
]
_NEURAL_DATASETS = [
    "fedorenko2016_ecog", "lpp_fmri", "pereira_fmri", "podcast_ieeg", "tuckute_fmri",
    "ns_fmri", "nieuwland", "megmasc", "broderick", "szewczyk_f2007", "szewczyk_h2019",
]

# The four classes as two binary dimensions:
#   memory = 1 for the memory-limited architectures  (Memory-Constrained, Constrained-BabyLM)
#   data   = 1 for the developmental (BabyLM) data budget (BabyLM, Constrained-BabyLM)
#   Standard LLM (0,0)  Memory-Constrained (1,0)  BabyLM (0,1)  Constrained-BabyLM (1,1)
_MEMORY_GROUPS = {"Memory-Constrained", "Constrained-BabyLM"}
_DATA_GROUPS = {"BabyLM", "Constrained-BabyLM"}

# Page 1: main effects only.
_RHS_FAC = "memory + data + log_params + log_ppl"
_TERMS_FAC = [
    ("Intercept", "Intercept (Standard LLM)"),
    ("memory", r"Memory constraint\ \ $\beta_M$"),
    ("data", r"Data constraint\ \ $\beta_D$"),
    ("log_params", r"$\log_{10}$(n\_params)"),
    ("log_ppl", r"$\log_{10}$(perplexity)"),
]

# Page 2: add the memory x data interaction.
_RHS_FAC_INT = "memory * data + log_params + log_ppl"
_TERMS_FAC_INT = [
    ("Intercept", "Intercept (Standard LLM)"),
    ("memory", r"Memory constraint\ \ $\beta_M$"),
    ("data", r"Data constraint\ \ $\beta_D$"),
    ("memory:data", r"Memory $\times$ Data\ \ $\beta_{MD}$"),
    ("log_params", r"$\log_{10}$(n\_params)"),
    ("log_ppl", r"$\log_{10}$(perplexity)"),
]

# Page 3: page 2 x modality. "{mod}" is the C(modality, ...) term, so the same template is fit twice -- once with
# each modality as reference -- to read a correctly-SE'd effect for behavioral and for neural.
_RHS_FAC_INT_MOD = "memory * data * {mod} + log_params * {mod} + log_ppl * {mod}"
_MOD_B = "C(modality, Treatment('behavioral'))"
_MOD_N = "C(modality, Treatment('neural'))"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _prepare() -> pd.DataFrame:
    """Per-(model, dataset) rows: all benchmark datasets, main models,
    restricted to rows that have a perplexity value."""
    df = pd.read_csv(MASTER_PATH)
    if TRACK == "embedding" and "perplexity" not in df.columns:
        # main models only already; perplexity comes from the surprisal master table (main rows)
        ppl = pd.read_csv(SURPRISAL_MASTER)
        ppl = ppl[~ppl["is_checkpoint"]][["dataset", "model_id", "perplexity"]].drop_duplicates(["dataset", "model_id"])
        df = df.merge(ppl, on=["dataset", "model_id"], how="left")
    df = df[~df["dataset"].isin(_N400_DATASETS + _EXCLUDE_DATASETS)]
    df = df[df["score_norm"].notna()]
    if MAIN_MODELS_ONLY and "is_checkpoint" in df.columns:
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
    df["corpus"] = df["dataset"].map(_base_dataset)
    return _robust_z(df) if ROBUST_Z else df


def _robust_z(df: pd.DataFrame) -> pd.DataFrame:
    """score_norm -> per-dataset robust z, (x - median) / (IQR / 1.349); IQR 0 falls back to 1.4826 * MAD."""
    def _rz(x: pd.Series) -> pd.Series:
        q1, q3 = x.quantile([0.25, 0.75])
        scale = (q3 - q1) / 1.349
        if not scale > 0:
            scale = 1.4826 * (x - x.median()).abs().median()
        return (x - x.median()) / scale if scale > 0 else x * np.nan
    df = df.copy()
    df["score_norm"] = df.groupby("dataset", observed=True)["score_norm"].transform(_rz)
    return df[df["score_norm"].notna()].copy()


def _prepare_all_robust_z() -> pd.DataFrame:
    """Page-4 sample: every dataset unit (no unstable-ceiling exclusion), outcome = per-dataset robust z."""
    global _EXCLUDE_DATASETS
    saved = _EXCLUDE_DATASETS
    _EXCLUDE_DATASETS = []
    try:
        d = _prepare()
    finally:
        _EXCLUDE_DATASETS = saved
    return d if ROBUST_Z else _robust_z(d)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

def _fit_lmm(df: pd.DataFrame, rhs: str):
    """REML mixed model with a dataset random intercept. lbfgs alone can collapse the random-intercept variance to 0
    (singular covariance, non-finite log-likelihood, NaN intercept SE) on the full 2026-10 table, so several
    optimizers are tried and the valid fit (finite log-likelihood, tau^2 > 0) with the highest REML log-likelihood wins."""
    best = None
    for method in ("bfgs", "lbfgs", "powell"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = smf.mixedlm(f"score_norm ~ {rhs}", data=df, groups=df["dataset"]).fit(reml=True, method=method)
            except Exception:  # noqa: BLE001 -- a failed optimizer just means try the next one
                continue
        if not (np.isfinite(res.llf) and float(np.asarray(res.cov_re)[0, 0]) > 0 and np.isfinite(res.bse).all()):
            continue
        if best is None or res.llf > best.llf:
            best = res
    if best is None:
        raise RuntimeError(f"no optimizer gave a valid mixed-model fit for: {rhs}")
    return best


def _fit_lmm_rs(df: pd.DataFrame, rhs: str):
    """REML mixed model with a dataset random intercept AND random slopes for size and perplexity. Size and
    perplexity are mean-centered (class effects are unaffected; only the intercept / modality main effect move) so
    the 3x3 random-effect covariance is well conditioned. Same optimizer policy as _fit_lmm."""
    d = df.copy()
    d["log_params"] = d["log_params"] - d["log_params"].mean()
    d["log_ppl"] = d["log_ppl"] - d["log_ppl"].mean()
    best = None
    for method in ("bfgs", "lbfgs", "powell"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = smf.mixedlm(f"score_norm ~ {rhs}", data=d, groups=d["dataset"],
                                  re_formula="~log_params + log_ppl").fit(reml=True, method=method)
            except Exception:  # noqa: BLE001
                continue
        if not (np.isfinite(res.llf) and np.isfinite(res.bse).all()):
            continue
        if best is None or res.llf > best.llf:
            best = res
    if best is None:
        raise RuntimeError(f"no optimizer gave a valid random-slope fit for: {rhs}")
    return best


_VC_FORMULA = {"dataset": "0 + C(dataset)", "dataset x size": "0 + C(dataset):log_params",
               "dataset x ppl": "0 + C(dataset):log_ppl", "corpus": "0 + C(corpus)", "family": "0 + C(family)"}


def _fit_lmm_vc(df: pd.DataFrame, rhs: str):
    """REML mixed model with independent variance components: dataset intercept, dataset-specific size and perplexity
    slopes, corpus intercept and model-family intercept (one all-rows group, as statsmodels needs for crossed effects).
    Size and perplexity are mean-centered as in _fit_lmm_rs."""
    d = df.copy()
    d["_all"] = 1
    d["log_params"] = d["log_params"] - d["log_params"].mean()
    d["log_ppl"] = d["log_ppl"] - d["log_ppl"].mean()
    mod = smf.mixedlm(f"score_norm ~ {rhs}", data=d, groups="_all", re_formula="0", vc_formula=_VC_FORMULA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(reml=True, method=["lbfgs"], maxiter=300)
    if not (np.isfinite(res.llf) and np.isfinite(res.bse).all()):
        raise RuntimeError(f"variance-component fit did not give a valid result for: {rhs}")
    res.vc_names = list(mod.exog_vc.names)
    return res


def _vc_extra(res) -> list[tuple[str, str]]:
    sd = dict(zip(res.vc_names, np.sqrt(np.asarray(res.vcomp))))
    return [
        ("Dataset random-intercept SD", f"{sd['dataset']:.4f}"),
        (r"Dataset random-slope SD, $\log_{10}$(n\_params)", f"{sd['dataset x size']:.4f}"),
        (r"Dataset random-slope SD, $\log_{10}$(perplexity)", f"{sd['dataset x ppl']:.4f}"),
        ("Corpus random-intercept SD", f"{sd['corpus']:.4f}"),
        ("Model-family random-intercept SD", f"{sd['family']:.4f}"),
        ("Residual SD", f"{np.sqrt(res.scale):.4f}"),
    ]


def _rs_extra(res) -> list[tuple[str, str]]:
    sd = np.sqrt(np.diag(np.asarray(res.cov_re)))
    return [
        ("Random-intercept SD", f"{sd[0]:.4f}"),
        (r"Random-slope SD, $\log_{10}$(n\_params)", f"{sd[1]:.4f}"),
        (r"Random-slope SD, $\log_{10}$(perplexity)", f"{sd[2]:.4f}"),
        ("Residual SD", f"{np.sqrt(res.scale):.4f}"),
    ]


def _rs_boundary(res) -> bool:
    """True if the random-effect covariance is (near-)singular, i.e. some slope variance is ~0 (boundary fit)."""
    ev = np.linalg.eigvalsh(np.asarray(res.cov_re))
    return bool(ev.min() / ev.max() < 1e-3)


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
    with len(fe_params) columns, not len(params))."""
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


def _lmm_extra(res) -> list[tuple[str, str]]:
    return [
        ("Random-intercept SD", f"{np.sqrt(np.asarray(res.cov_re)[0, 0]):.4f}"),
        ("Residual SD", f"{np.sqrt(res.scale):.4f}"),
        ("ICC (dataset)", f"{_icc(res):.3f}"),
    ]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

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

_MATH_FAC_LMM_MOD = r"""\[
\begin{aligned}
\mathrm{HS}_{ij} = {}& \beta_0
 + \underbrace{\beta_M\mathrm{M}_i + \beta_D\mathrm{D}_i + \beta_{MD}\,\mathrm{M}_i\mathrm{D}_i}_{\text{constraint effects, behavioral}}
 + \delta_0\,\mathrm{Neural}_j \\
 & + \beta_S\log_{10}\mathrm{params}_i + \beta_P\log_{10}\mathrm{ppl}_{ij} \\
 & + \underbrace{\delta_M\,\mathrm{M}_i\mathrm{Neural}_j + \delta_D\,\mathrm{D}_i\mathrm{Neural}_j
   + \delta_{MD}\,\mathrm{M}_i\mathrm{D}_i\mathrm{Neural}_j
   + \delta_S\log_{10}\mathrm{params}_i\mathrm{Neural}_j + \delta_P\log_{10}\mathrm{ppl}_{ij}\mathrm{Neural}_j}_{\text{extra effect on neural data}} \\
 & + u_j + \varepsilon_{ij}
\end{aligned}
\]"""

_FAC_DEF = (r"\texttt{memory}\,$=1$ for the memory-limited architectures "
            r"(Memory-Constrained, Constrained-BabyLM); \texttt{data}\,$=1$ for "
            r"the BabyLM data budget (BabyLM, Constrained-BabyLM). "
            r"$\beta_M+\beta_D$ is the Constrained-BabyLM effect.")

_FAC_DEF_INT = (_FAC_DEF + r" With the interaction, Constrained-BabyLM "
                r"$=\beta_M+\beta_D+\beta_{MD}$ (equivalent to the original "
                r"4-group coding).")

_MATH_FAC_LMM_MOD_RS = r"""\[
u_j = (u_{0j},\,u_{Sj},\,u_{Pj}) \sim \mathcal{N}(0,\Sigma),\qquad
\text{random intercept + random slopes for }\log_{10}\mathrm{params},\ \log_{10}\mathrm{ppl}
\]
\[
\mathrm{HS}_{ij} = (\text{page-3 fixed effects})
 + u_{0j} + u_{Sj}\log_{10}\mathrm{params}_i + u_{Pj}\log_{10}\mathrm{ppl}_{ij} + \varepsilon_{ij}
\]"""

_FAC_DEF_MOD_RS = (r"The page-3 model with dataset-level random \emph{slopes} for $\log_{10}$(size) and "
                   r"$\log_{10}$(perplexity) in addition to the random intercept, so datasets (including the "
                   r"very differently scaled neural ones) may differ in how strongly size and perplexity matter. "
                   r"This is the preferred specification: it keeps the modality-specific fixed slopes of page~3 "
                   r"but does not let a few datasets' slopes drive the standard errors. Size and perplexity are "
                   r"mean-centered, so the intercept is the standard-LLM, behavioral prediction at the average "
                   r"size and perplexity (class effects are unaffected by the centering).")

_FAC_DEF_MOD = (r"Page~2 with a \texttt{modality} (behavioral / neural) main effect and its interactions with "
                r"\texttt{memory}, \texttt{data}, \texttt{memory}$\times$\texttt{data}, $\log_{10}$(size) and "
                r"$\log_{10}$(perplexity), so every effect is estimated separately for behavioral and neural "
                r"datasets. A behavioral effect is "
                r"$\beta_\cdot$; its neural counterpart is $\beta_\cdot+\delta_\cdot$. The table reports each "
                r"directly (the neural block is the same model re-fit with \texttt{neural} as the modality "
                r"reference, so every SE / $p$ is exact). Constrained-BabyLM "
                r"$=\beta_M+\beta_D+\beta_{MD}$ within each modality.")

_SEPARATE_MODALITY = {"surprisal", "embedding"}  # tracks whose page 3 is two separate fits (2026-10-13)

_MATH_FAC_LMM_SEP = r"""\[
\mathrm{HS}_{ij} = \beta_0^{(m)}
 + \beta_M^{(m)}\mathrm{M}_i + \beta_D^{(m)}\mathrm{D}_i + \beta_{MD}^{(m)}\,\mathrm{M}_i\mathrm{D}_i
 + \beta_S^{(m)}\log_{10}\mathrm{params}_i + \beta_P^{(m)}\log_{10}\mathrm{ppl}_{ij}
 + u_j^{(m)} + \varepsilon_{ij}^{(m)},\qquad m\in\{\text{behavioral},\text{neural}\}
\]"""

_FAC_DEF_SEP = (r"Page~2 fitted \emph{separately} on the behavioral and on the neural dataset units, so every "
                r"coefficient, the dataset random-intercept variance and the residual variance are specific to each "
                r"modality. (A single model with modality interactions gives the same point estimates, but its one "
                r"residual variance is pooled over modalities whose residual spread differs several-fold, which makes "
                r"the behavioral standard errors too large and the neural ones too small.) Constrained-BabyLM "
                r"$=\beta_M+\beta_D+\beta_{MD}$ within each modality.")


def _table_fac_sep(fit_b, fit_n) -> str:
    """Page-3 table for two separate fits: each block, intercept and variance components from its own fit."""
    def block(title, res):
        out = [rf"\multicolumn{{2}}{{@{{}}l}}{{\itshape {title}}} \\"]
        for term, label in [("memory", r"Memory constraint\ \ $\beta_M$"),
                            ("data", r"Data constraint\ \ $\beta_D$"),
                            ("memory:data", r"Memory $\times$ Data\ \ $\beta_{MD}$")]:
            out.append(rf"\quad {label} & {_cell(res, term)} \\")
        out.append(r"\quad Memory and data constraint\ \ $\beta_M+\beta_D+\beta_{MD}$ & "
                   + _contrast_cell(res, ["memory", "data", "memory:data"]) + r" \\")
        out.append(rf"\quad $\log_{{10}}$(n\_params)\ \ $\beta_S$ & {_cell(res, 'log_params')} \\")
        out.append(rf"\quad $\log_{{10}}$(perplexity)\ \ $\beta_P$ & {_cell(res, 'log_ppl')} \\")
        out.append(rf"\quad Intercept (Standard LLM) & {_cell(res, 'Intercept')} \\")
        out += [rf"\quad {lab} & {val} \\" for lab, val in _lmm_extra(res)]
        out.append(rf"\quad $N$ rows ({res.model.group_labels.__len__()} dataset units) & {int(res.nobs)} \\")
        return "\n".join(out)
    return rf"""\begin{{center}}
\footnotesize
\renewcommand{{\arraystretch}}{{0.95}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Term}} & \textbf{{Coefficient (SE)}} \\
\midrule
{block("Behavioral datasets (separate fit; effect vs standard LLM)", fit_b)}
\addlinespace
{block("Neural datasets (separate fit; effect vs standard LLM)", fit_n)}
\bottomrule
\end{{tabular}}
\end{{center}}"""


_TAIL_NOTE_RS = (r"{\footnotesize Coefficient (SE).\;"
                 r"\textsuperscript{*}$p<0.05$\;"
                 r"\textsuperscript{**}$p<0.01$\;"
                 r"\textsuperscript{***}$p<0.001$.}")

_RS_BOUNDARY_NOTE = (r"{\footnotesize Note: the estimated random-effect covariance is close to singular (some slope "
                     r"variances are near zero), i.e. this fit lies on the boundary of the parameter space. Standard "
                     r"errors are therefore approximate; point estimates agree with page~3.}")

_TAIL_NOTE = (r"{\footnotesize Coefficient (SE).\;"
              r"\textsuperscript{*}$p<0.05$\;"
              r"\textsuperscript{**}$p<0.01$\;"
              r"\textsuperscript{***}$p<0.001$.\;"
              r"ICC $=\hat\tau^2/(\hat\tau^2+\hat\sigma^2)$}")


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


def _table_fac_mod(fit_b, fit_n, n_val, extra_rows=None, intercept_label="Intercept (Standard LLM, behavioral)") -> str:
    """Page-3 table: behavioral effects from the behavioral-reference fit, neural effects from the neural-reference
    fit; shared terms (intercept, modality main effect, variance components) from the behavioral fit."""
    combo = ["memory", "data", "memory:data"]

    def block(title, res):
        out = [rf"\multicolumn{{2}}{{@{{}}l}}{{\itshape {title}}} \\"]
        for term, label in [("memory", r"Memory constraint\ \ $\beta_M$"),
                            ("data", r"Data constraint\ \ $\beta_D$"),
                            ("memory:data", r"Memory $\times$ Data\ \ $\beta_{MD}$")]:
            out.append(rf"\quad {label} & {_cell(res, term)} \\")
        out.append(r"\quad Memory and data constraint\ \ $\beta_M+\beta_D+\beta_{MD}$ & "
                   + _contrast_cell(res, combo) + r" \\")
        out.append(rf"\quad $\log_{{10}}$(n\_params)\ \ $\beta_S$ & {_cell(res, 'log_params')} \\")
        out.append(rf"\quad $\log_{{10}}$(perplexity)\ \ $\beta_P$ & {_cell(res, 'log_ppl')} \\")
        return "\n".join(out)

    mod_term = f"{_MOD_B}[T.neural]"
    tail = "\n".join(f"{lab} & {_cell(fit_b, term)} \\\\" for term, lab in [
        ("Intercept", intercept_label),
        (mod_term, "Neural vs behavioral (Standard LLM)"),
    ])
    extra = "\n".join(f"{lab} & {val}" + r" \\" for lab, val in (extra_rows or _lmm_extra(fit_b)))
    return rf"""\begin{{center}}
\footnotesize
\renewcommand{{\arraystretch}}{{0.95}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Term}} & \textbf{{Coefficient (SE)}} \\
\midrule
{block("Behavioral datasets (effect vs standard LLM)", fit_b)}
\addlinespace
{block("Neural datasets (effect vs standard LLM)", fit_n)}
\midrule
{tail}
\midrule
{extra}
$N$ rows & {n_val} \\
\bottomrule
\end{{tabular}}
\end{{center}}"""


_MATH_VC = r"""\[
\mathrm{HS}_{ij} = (\text{page-3 fixed effects})
 + u_{0j} + u_{Sj}\log_{10}\mathrm{params}_i + u_{Pj}\log_{10}\mathrm{ppl}_{ij} + c_{k(j)} + f_{m(i)} + \varepsilon_{ij}
\]
with independent $u_{0j},u_{Sj},u_{Pj}$ (dataset $j$), $c_{k(j)}$ (corpus of dataset $j$) and $f_{m(i)}$ (family of model $i$)."""

_FAC_DEF_VC = (r"The page-3 model with dataset random intercepts and slopes for $\log_{10}$(size) and $\log_{10}$(perplexity) "
               r"(independent variance components), plus crossed random intercepts for the \emph{corpus} (the FP / GP / SP "
               r"measures of one corpus, and the three ZuCo tasks, share items and participants) and for the model "
               r"\emph{family} (models of one family are not independent, and each constraint group contains only a few "
               r"families). This is the preferred specification: without the family and corpus terms the group contrasts "
               r"are over-confident. Size and perplexity are mean-centered.")


def _vc_page(title: str, vc_b, vc_n, d: pd.DataFrame, note: str = "") -> str:
    n_obs = int(vc_b.nobs)
    d_beh = int(d.loc[d["modality"] == "behavioral", "dataset"].nunique())
    d_neu = int(d.loc[d["modality"] == "neural", "dataset"].nunique())
    n_beh = int((d["modality"] == "behavioral").sum())
    n_neu = int((d["modality"] == "neural").sum())
    detail = (rf"\textbf{{Samples:}} $N = {n_obs}$ rows; behavioral {n_beh} rows ({d_beh} dataset units), neural {n_neu} rows "
              rf"({d_neu} dataset units); {d['corpus'].nunique()} corpora, {d['family'].nunique()} model families, "
              rf"{d['model_id'].nunique()} models; REML. {note}")
    return rf"""\newpage
\begin{{center}}
{{\large\bfseries {title}}}
\end{{center}}

{_FAC_DEF_VC}

\textbf{{Model:}}
{_MATH_VC}

{detail}

{_table_fac_mod(vc_b, vc_n, n_obs, extra_rows=_vc_extra(vc_b), intercept_label=r"Intercept (Standard LLM, behavioral, mean size and ppl)")}

{_TAIL_NOTE_RS}
"""


ROBUST_INFERENCE_DIR = REPO_ROOT / "reports" / "robust_inference"
_RI_TERMS = [("M", r"Memory only $\beta_M$"), ("D", r"Data only $\beta_D$"), ("M:D", r"Memory $\times$ Data $\beta_{MD}$"),
             ("both", r"Both constraints $\beta_M{+}\beta_D{+}\beta_{MD}$"), ("size", r"$\log_{10}$ parameters"),
             ("ppl", r"$\log_{10}$ perplexity")]


def _robust_inference_page() -> str:
    """Page 5 (added 2026-10-14): robust inference from scripts/analysis/robust_inference.R (lineage / model / corpus
    random effects, unit slopes, joint behavioral x neural model, robustness variants, lineage cluster bootstrap).
    Empty when reports/robust_inference/coefficients.csv does not exist (run scripts/slurm/jz/utils/robust_inference.sbatch)."""
    path = ROBUST_INFERENCE_DIR / "coefficients.csv"
    if not path.exists():
        return ""
    co = pd.read_csv(path)
    co = co[co["track"] == TRACK]

    def cell(r) -> str:
        if r is None:
            return "--"
        return rf"${r['est']:+.3f}$ $[{r['lo']:+.3f}, {r['hi']:+.3f}]${_stars(r['p'])}"

    def get(analysis, variant, domain, term):
        x = co[(co["analysis"] == analysis) & (co["variant"] == variant) & (co["domain"] == domain) & (co["term"] == term)]
        return None if x.empty else x.iloc[0]

    def rng(domain, term) -> str:
        x = co[(co["analysis"].isin(["robustness", "leave_one_unit_out"])) & (co["domain"] == domain) & (co["term"] == term)
               & (co["variant"] != "robust_z")]
        if x.empty:
            return "--"
        return rf"${x['est'].min():+.3f}$ to ${x['est'].max():+.3f}$ ({int((x['p'] < .05).sum())}/{len(x)})"

    boot_var = co.loc[co["analysis"] == "lineage_bootstrap", "variant"].iloc[0]
    rows = []
    for dom in ("behavioral", "neural"):
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{dom.capitalize()} datasets}}}} \\")
        for t, lab in _RI_TERMS:
            rows.append(" & ".join([lab, cell(get("ladder", "S0", dom, t)), cell(get("ladder", "S2", dom, t)),
                                    cell(get("lineage_bootstrap", boot_var, dom, t)), rng(dom, t),
                                    cell(get("joint", "ceiling-normalized", "neural - behavioral", t)) if dom == "neural" else ""])
                        + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    has_dagger = TRACK == "surprisal"
    fig = ROBUST_INFERENCE_DIR / f"forest_{TRACK}.pdf"
    return rf"""\newpage
%% ---------------- PAGE 5 : robust inference ----------------------------------
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (5) robust inference for the page-3 model}}
\end{{center}}

Page 3 treats the {{\it model $\times$ dataset}} rows as independent apart from a dataset random intercept, although models
of one family share architecture and data, the same model appears in every dataset unit, and FP/GP/SP units of one corpus
share items and participants. This page refits the page-3 model with that dependence modeled
(\texttt{{scripts/analysis/robust\_inference.R}}; lme4/lmerTest, REML, Satterthwaite df):
\begin{{itemize}}\setlength{{\itemsep}}{{0pt}}
\item \textbf{{S0}} = page 3: $Y \sim M{{*}}D + \text{{size}} + \text{{ppl}} + (1\mid\text{{unit}})$, separate behavioral / neural fits.
\item \textbf{{S2}} (preferred) = S0 $+ (1\mid\text{{lineage}}) + (1\mid\text{{model}}) + (1\mid\text{{corpus}})
      + (0+\text{{size}}\mid\text{{unit}}) + (0+\text{{ppl}}\mid\text{{unit}})$. \emph{{Lineage}} = model family, with the single
      registry family \texttt{{babylm}} split into its five independent submissions (GPT-2 baselines, GPT-BERT, MoEP,
      CLASS-IT, BLM-GPT2s), 25 lineages in total; size and ppl centered.
\item \textbf{{Bootstrap}}: S0 refit on {boot_var.split(",")[1].strip()} with lineages resampled with replacement within
      each constraint group; percentile 95\% CI, two-sided bootstrap $p$. Few lineages per constrained group (5--6), so
      this CI is approximate.
\item \textbf{{S2 range}}: min--max of the S2 estimate over the robustness variants (winsorized 5--95\% within unit,
      without GPT-BERT models, $+\log_{{10}}$ depth{", without the four units with flagged references (UCL, Broderick, Federmeier 2007, Hubbard 2019)" if has_dagger else ""})
      and over leave-one-unit-out refits; in parentheses the number of refits with $p<.05$.
\item \textbf{{Neural $-$ behavioral}}: joint model on all rows,
      $Y \sim (M{{*}}D + \text{{size}} + \text{{ppl}})\times\text{{domain}}$ with the S2 random effects plus lineage $\times$ domain
      and model $\times$ domain intercepts; the domain interaction directly tests whether a coefficient differs between
      behavioral and neural prediction.
\end{{itemize}}
Cells: estimate [95\% CI]; \textsuperscript{{*}}$p<.05$, \textsuperscript{{**}}$p<.01$, \textsuperscript{{***}}$p<.001$.

\noindent\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\toprule
Term & S0 (page 3) & S2 (preferred) & Lineage bootstrap & S2 range (sig./refits) & Neural $-$ behavioral \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}

Several S1/S2 fits are singular (a lineage or model variance at zero once the other term absorbs it); fixed effects are
unaffected. Full results: \texttt{{reports/robust\_inference/coefficients.csv}}.

{rf"\begin{{center}}\includegraphics[width=\textwidth,height=0.42\textheight,keepaspectratio]{{robust_inference/forest_{TRACK}.pdf}}\end{{center}}" if fig.exists() else ""}
"""


def _robust_z_page(rz_b, rz_n, d: pd.DataFrame) -> str:
    n_obs = int(rz_b.nobs)
    d_beh = int(d.loc[d["modality"] == "behavioral", "dataset"].nunique())
    d_neu = int(d.loc[d["modality"] == "neural", "dataset"].nunique())
    n_beh = int((d["modality"] == "behavioral").sum())
    n_neu = int((d["modality"] == "neural").sum())
    return rf"""\newpage
%% ---------------- PAGE 4 : page 3 on robust-z scores, all datasets -----------
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (4) page-3 model, robust-z outcome, all datasets}}
\end{{center}}

Same model as page 3, but the outcome is not divided by a noise ceiling: each dataset unit's scores are replaced by
their robust z across the models of that unit, $(x - \mathrm{{median}}) / (\mathrm{{IQR}}/1.349)$. This needs no ceiling,
so every dataset unit is included (also those whose $\Delta$LL ceiling is unusable), and an outlying model cannot set
the scale. Coefficients are in units of the between-model spread of a dataset (1 $\approx$ one between-model SD); the
dataset random intercept is kept. The cost is that scores no longer say how close a model is to the ceiling.

\textbf{{Samples:}} $N = {n_obs}$ rows; behavioral {n_beh} rows ({d_beh} dataset units), neural {n_neu} rows
({d_neu} dataset units); {d['model_id'].nunique()} models; REML. Scores are each model's
{"last layer" if LAYER == "last" else "best layer"}.

{_table_fac_mod(rz_b, rz_n, n_obs)}

{_TAIL_NOTE}
"""


def _build_report(df: pd.DataFrame, fac_lmm, fac_lmm_int, mod_b, mod_n, rs_b=None, rs_n=None, vc_pages: str = "",
                  rz_page: str = "") -> None:
    n_obs = int(fac_lmm.nobs)
    n_ds = df["dataset"].nunique()
    n_models = df["model_id"].nunique()
    gc = df.groupby("group", observed=True)["model_id"].nunique().reindex(GROUP_ORDER)
    grp_line = (f"LLM {int(gc['LLM'])}, Memory-Constrained {int(gc['Memory-Constrained'])}, "
                f"Constrained-BabyLM {int(gc['Constrained-BabyLM'])}, BabyLM {int(gc['BabyLM'])}")
    samples = (rf"\textbf{{Samples:}} $N = {n_obs}$ (model $\times$ dataset) rows over {n_ds} dataset units "
               rf"(reading-time measures FP/GP/SP are separate units), {n_models} models "
               rf"({grp_line}); REML. {_ROBUST_Z_TEXT if ROBUST_Z else _OUTCOME_TEXT[TRACK]} "
               + (r"Scores are each model's last layer (no layer selection)." if LAYER == "last"
                  else r"Scores are each model's best layer (maximum over layers of the held-out score).")
               + (r" Excluded (unusable or unstable $\Delta$LL noise ceilings): Broderick, Federmeier et al. (2007), "
                  r"Hubbard et al. (2019), Pereira." if _EXCLUDE_DATASETS and TRACK == "surprisal" else ""))
    n_beh = int((df["modality"] == "behavioral").sum())
    n_neu = int((df["modality"] == "neural").sum())
    d_beh = int(df.loc[df["modality"] == "behavioral", "dataset"].nunique())
    d_neu = int(df.loc[df["modality"] == "neural", "dataset"].nunique())

    # Page 4 (dataset random slopes) is only part of the --extended report.
    rs_page = "" if rs_b is None else rf"""\newpage
%% ---------------- PAGE 4 : page 3 + random slopes ---------------------------
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (5) page-3 model + dataset random slopes{"" if vc_pages else " (preferred)"}}}
\end{{center}}

{_FAC_DEF_MOD_RS if not vc_pages else _FAC_DEF_MOD_RS.replace("This is the preferred specification: it keeps", "It keeps")}

\textbf{{Model:}}
{_MATH_FAC_LMM_MOD_RS}

{samples} Behavioral {n_beh} rows (over {d_beh} datasets), neural {n_neu} rows (over {d_neu} datasets).

{_table_fac_mod(rs_b, rs_n, n_obs, extra_rows=_rs_extra(rs_b), intercept_label=r"Intercept (Standard LLM, behavioral, mean size and ppl)")}

{_TAIL_NOTE_RS}
{_RS_BOUNDARY_NOTE if _rs_boundary(rs_b) else ""}
"""

    if TRACK in _SEPARATE_MODALITY:
        page3 = rf"""{{\large\bfseries Matched-scale regression --- (3) 4-feature model + interaction, separate behavioral and neural fits}}
\end{{center}}

{_FAC_DEF_SEP}

\textbf{{Model:}} ($\mathrm{{M}}$ = Memory, $\mathrm{{D}}$ = Data; fitted once per modality $m$)
{_MATH_FAC_LMM_SEP}

{samples} Behavioral {n_beh} rows (over {d_beh} datasets), neural {n_neu} rows (over {d_neu} datasets).

{_table_fac_sep(mod_b, mod_n)}"""
    else:
        page3 = rf"""{{\large\bfseries Matched-scale regression --- (3) 4-feature model + interaction $\times$ modality, mixed}}
\end{{center}}

{_FAC_DEF_MOD}

\textbf{{Model:}} ($\mathrm{{M}}$ = Memory, $\mathrm{{D}}$ = Data, $\mathrm{{Neural}}_j=1$ if dataset $j$ is neural)
{_MATH_FAC_LMM_MOD}

{samples} Behavioral {n_beh} rows (over {d_beh} datasets), neural {n_neu} rows (over {d_neu} datasets).

{_table_fac_mod(mod_b, mod_n, n_obs)}"""

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.8in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{3pt}}
\begin{{document}}
\small

%% ---------------- PAGE 1 : 4-feature model + dataset random intercept ------
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (1) 4-feature model, mixed, + dataset random intercept}}
\end{{center}}

{_FAC_DEF}

\textbf{{Model:}}
{_MATH_FAC_LMM}

{samples}

{_table_fac(fac_lmm, _lmm_extra(fac_lmm), "$N$ rows", n_obs)}

{_TAIL_NOTE}

\newpage
%% ---------------- PAGE 2 : 4-feature model + interaction + random intercept
\begin{{center}}
{{\large\bfseries Matched-scale regression --- (2) 4-feature model + interaction, mixed, + dataset random intercept}}
\end{{center}}

{_FAC_DEF_INT}

\textbf{{Model:}}
{_MATH_FAC_LMM_INT}

{samples}

{_table_fac(fac_lmm_int, _lmm_extra(fac_lmm_int), "$N$ rows", n_obs, terms=_TERMS_FAC_INT, combo=("memory", "data", "memory:data"))}

{_TAIL_NOTE}

\newpage
%% ---------------- PAGE 3 : page 2 x modality --------------------------------
\begin{{center}}
{page3}

{_TAIL_NOTE}

{rz_page}

{_robust_inference_page()}

{rs_page}

{vc_pages}
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


def _log_fits(tag: str, fits: list, variants: list) -> list[dict]:
    rows = []
    for (mid, desc, _), res in zip(variants, fits):
        print(f"\n=== [{tag}] {mid} ({desc})   N={int(res.nobs)}")
        for term in res.params.index:
            if term == "Group Var":
                continue
            print(f"    {term:<70} {res.params[term]:+.4f} "
                  f"({res.bse[term]:.4f}) p={res.pvalues[term]:.3g}")
            rows.append({"analysis": tag, "model": mid, "controls": desc, "term": term,
                         "coef": float(res.params[term]), "std_err": float(res.bse[term]),
                         "p_value": float(res.pvalues[term])})
    return rows


def main() -> None:
    global TRACK, MASTER_PATH, TEX_OUT, LAYER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=sorted(TRACK_MASTER), default="surprisal")
    ap.add_argument("--layer", choices=["best", "last", "bestsummary"], default="best",
                    help="best: master tables as built (default, unchanged); last: last-layer scores from "
                         "outputs/benchmark_summary (27 units, registry groups); bestsummary: best-layer scores from the "
                         "same summary tables (output files get the suffix _bestlayer)")
    ap.add_argument("--keep-unstable-ceilings", action="store_true",
                    help="surprisal track only: keep Broderick, Federmeier 2007, Hubbard 2019 and Pereira, whose "
                         "delta-loglik noise ceilings are unusable or unstable (default: they are excluded)")
    ap.add_argument("--robust-z", action="store_true",
                    help="outcome = per-dataset robust z of the score instead of the ceiling-normalized score "
                         "(output files get the suffix _robustz)")
    ap.add_argument("--extended", action="store_true",
                    help="also fit/report page 4 (dataset random slopes) and, with --layer last, pages 5-6 "
                         "(family + corpus random effects). Default: pages 1-3 only (dataset random intercept).")
    args = ap.parse_args()
    TRACK, LAYER = args.track, args.layer
    MASTER_PATH = TRACK_MASTER[TRACK]
    TEX_OUT = REPO_ROOT / "reports" / f"matched_scale_regression_report_{TRACK}.tex"
    if not SURPRISAL_MASTER.exists():
        raise SystemExit(f"{SURPRISAL_MASTER} not found -- run analysis/matched_scale_master_table.py first")
    if LAYER in ("last", "bestsummary"):
        tag = "last" if LAYER == "last" else "bestlayer"
        MASTER_PATH = OUT_DIR / f"master_table_{TRACK}_{tag}.csv"
        if LAYER == "bestsummary":
            TEX_OUT = REPO_ROOT / "reports" / f"matched_scale_regression_report_{TRACK}_bestlayer.tex"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _build_last_layer_table(TRACK, "score_last_norm" if LAYER == "last" else "score_best_norm").to_csv(MASTER_PATH, index=False)
        print(f"wrote {MASTER_PATH}")
    elif not MASTER_PATH.exists():
        raise SystemExit(f"{MASTER_PATH} not found -- run analysis/matched_scale_master_table_embed.py first")
    global _EXCLUDE_DATASETS, ROBUST_Z
    ROBUST_Z = args.robust_z
    if ROBUST_Z:
        TEX_OUT = TEX_OUT.with_name(TEX_OUT.stem + "_robustz.tex")
    if TRACK == "surprisal" and not args.keep_unstable_ceilings:
        _EXCLUDE_DATASETS = list(_UNSTABLE_DLL_CEILING)
    df = _prepare()

    fac_lmm = _fit_lmm(df, _RHS_FAC)                                       # page 1
    fac_lmm_int = _fit_lmm(df, _RHS_FAC_INT)                               # page 2
    if TRACK in _SEPARATE_MODALITY:                                        # page 3: separate fits per modality
        mod_b = _fit_lmm(df[df["modality"] == "behavioral"], _RHS_FAC_INT)
        mod_n = _fit_lmm(df[df["modality"] == "neural"], _RHS_FAC_INT)
    else:                                                                  # page 3: one model x modality
        mod_b = _fit_lmm(df, _RHS_FAC_INT_MOD.format(mod=_MOD_B))          # behavioral = reference
        mod_n = _fit_lmm(df, _RHS_FAC_INT_MOD.format(mod=_MOD_N))          # neural = reference
    rs_b = rs_n = None
    if args.extended:
        rs_b = _fit_lmm_rs(df, _RHS_FAC_INT_MOD.format(mod=_MOD_B))        # page 4 (random slopes), behavioral ref
        rs_n = _fit_lmm_rs(df, _RHS_FAC_INT_MOD.format(mod=_MOD_N))        # page 4, neural ref

    coef_rows = (
        _log_fits("factorial_lmm_random_intercept", [fac_lmm],
                  [("fac", "memory+data+size+ppl", _RHS_FAC)])
        + _log_fits("factorial_lmm_random_intercept_interaction", [fac_lmm_int],
                    [("fac_int", "memory*data+size+ppl", _RHS_FAC_INT)])
        + (_log_fits("factorial_lmm_separate_behavioral", [mod_b],
                     [("fac_int_sep", "memory*data+size+ppl, behavioral datasets only", _RHS_FAC_INT)])
           + _log_fits("factorial_lmm_separate_neural", [mod_n],
                       [("fac_int_sep", "memory*data+size+ppl, neural datasets only", _RHS_FAC_INT)])
           if TRACK in _SEPARATE_MODALITY else
           _log_fits("factorial_lmm_interaction_x_modality_behav_ref", [mod_b],
                     [("fac_int_mod", "memory*data*modality+size+ppl", _RHS_FAC_INT_MOD)])
           + _log_fits("factorial_lmm_interaction_x_modality_neural_ref", [mod_n],
                       [("fac_int_mod", "memory*data*modality+size+ppl", _RHS_FAC_INT_MOD)])))
    if args.extended:
        coef_rows += (_log_fits("factorial_lmm_random_slopes_behav_ref", [rs_b],
                                [("fac_int_mod_rs", "page3 + (1+size+ppl|dataset), centered size/ppl", _RHS_FAC_INT_MOD)])
                      + _log_fits("factorial_lmm_random_slopes_neural_ref", [rs_n],
                                  [("fac_int_mod_rs", "page3 + (1+size+ppl|dataset), centered size/ppl", _RHS_FAC_INT_MOD)]))
    vc_pages = ""
    if args.extended and LAYER == "last":
        rhs_b = _RHS_FAC_INT_MOD.format(mod=_MOD_B)
        rhs_n = _RHS_FAC_INT_MOD.format(mod=_MOD_N)
        vc_b, vc_n = _fit_lmm_vc(df, rhs_b), _fit_lmm_vc(df, rhs_n)
        coef_rows += (_log_fits("factorial_lmm_family_corpus_behav_ref", [vc_b],
                                [("fac_int_mod_vc", "page3 + dataset slopes + corpus + family", _RHS_FAC_INT_MOD)])
                      + _log_fits("factorial_lmm_family_corpus_neural_ref", [vc_n],
                                  [("fac_int_mod_vc", "page3 + dataset slopes + corpus + family", _RHS_FAC_INT_MOD)]))
        vc_pages = _vc_page("Matched-scale regression --- (6) page-3 model + dataset slopes + corpus and family random "
                            "intercepts (preferred)", vc_b, vc_n, df)
        if TRACK == "surprisal" and args.keep_unstable_ceilings:
            d6 = df[~df["dataset"].isin(_UNSTABLE_DLL_CEILING)]
            v6b, v6n = _fit_lmm_vc(d6, rhs_b), _fit_lmm_vc(d6, rhs_n)
            coef_rows += (_log_fits("factorial_lmm_family_corpus_excl4_behav_ref", [v6b],
                                    [("fac_int_mod_vc_excl4", "page5 without " + ",".join(_UNSTABLE_DLL_CEILING), _RHS_FAC_INT_MOD)])
                          + _log_fits("factorial_lmm_family_corpus_excl4_neural_ref", [v6n],
                                      [("fac_int_mod_vc_excl4", "page5 without " + ",".join(_UNSTABLE_DLL_CEILING), _RHS_FAC_INT_MOD)]))
            vc_pages += _vc_page("Matched-scale regression --- (7) page 6 without the four datasets with unusable "
                                 "$\\Delta$LL ceilings", v6b, v6n, d6,
                                 note=r"Dropped: Broderick, Federmeier et al. (2007), Hubbard et al. (2019) and Pereira "
                                      r"(models above the ceiling or mostly negative normalized scores).")
    # Page 4: page-3 model on robust-z scores over ALL dataset units (no ceiling, no exclusion).
    d_rz = _prepare_all_robust_z()
    rz_b = _fit_lmm(d_rz, _RHS_FAC_INT_MOD.format(mod=_MOD_B))
    rz_n = _fit_lmm(d_rz, _RHS_FAC_INT_MOD.format(mod=_MOD_N))
    coef_rows += (_log_fits("factorial_lmm_robustz_all_behav_ref", [rz_b],
                            [("fac_int_mod_rz", "page3 on per-dataset robust z, all dataset units", _RHS_FAC_INT_MOD)])
                  + _log_fits("factorial_lmm_robustz_all_neural_ref", [rz_n],
                              [("fac_int_mod_rz", "page3 on per-dataset robust z, all dataset units", _RHS_FAC_INT_MOD)]))
    rz_page = _robust_z_page(rz_b, rz_n, d_rz)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coef_path = OUT_DIR / (f"regression_coefficients_{TRACK}{'_bestlayer' if LAYER == 'bestsummary' else ''}"
                           f"{'_robustz' if ROBUST_Z else ''}.csv")
    pd.DataFrame(coef_rows).to_csv(coef_path, index=False)
    print(f"\nwrote {coef_path}")

    _build_report(df, fac_lmm, fac_lmm_int, mod_b, mod_n, rs_b, rs_n, vc_pages=vc_pages, rz_page=rz_page)


if __name__ == "__main__":
    main()
