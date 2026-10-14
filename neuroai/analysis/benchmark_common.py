"""Shared infrastructure for scripts/summarize/benchmark.py.

Group assignment, registry-based model metadata lookup, and generic
"scientific-ready" plotting helpers reused across all 12 datasets so that
scripts/summarize/benchmark.py can stay dataset-agnostic for
aggregation/plotting and only vary in per-dataset raw-CSV parsing.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuroai.models.registry import LPP_FAMILIES

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUTPUTS_JEANZAY = REPO_ROOT / "outputs_jeanzay"
BENCHMARK_SUMMARY_ROOT = REPO_ROOT / "outputs" / "benchmark_summary"

# Column names appended to the *_lme_by_layer.csv writer after some existing
# sweep files' headers were already written -- a resumed sweep run under the
# updated code then appends rows with 3 extra trailing fields below a header
# that still only names the original columns (seen 2026-09-17 while
# investigating BLOOM's missing LPP/Tuckute/Fedorenko2016/Nieuwland surprisal
# coverage: several pre-BLOOM models' by_layer.csv files mix a 26-column
# header with a tail of 29-column data rows). Plain ``pd.read_csv`` raises
# ParserError on this; ``on_bad_lines="skip"`` avoids the crash but silently
# DROPS every mismatched row (in one observed file, 6 of 9 layers -- exactly
# the layers a "pick the best layer" analysis most needs), so neither is
# safe to use directly on these files.
_LME_BY_LAYER_LEGACY_EXTRA_COLS = ("revision", "step", "tokens_seen")


def read_lme_by_layer_csv(path: Path) -> pd.DataFrame:
    """Robust reader for ``*_lme_by_layer.csv`` sweep files.

    NEVER delegates the ragged-row decision to ``pd.read_csv``'s own
    exception handling: its C engine does not reliably raise ParserError on
    a mismatched row count -- on at least one observed file it silently
    "succeeded" while shifting tokens across column boundaries (e.g. two
    numeric values landing concatenated as one string in the ``layer``
    column, `error`/`fit_wall_time_s` values silently zeroed), which is far
    worse than a loud crash. So this always parses the file itself with
    ``csv.reader`` (quote-safe) and checks every row's length against the
    header before ever handing anything to pandas -- padding a stale header
    with ``_LME_BY_LAYER_LEGACY_EXTRA_COLS`` and short/older rows with blanks
    when needed -- so old- and new-format rows within the same file both
    survive, correctly aligned, instead of being silently dropped or
    mis-shifted."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        return pd.DataFrame()
    header, data_rows = rows[0], rows[1:]
    max_len = max((len(r) for r in data_rows), default=len(header))
    max_len = max(max_len, len(header))
    if max_len > len(header):
        extra = list(_LME_BY_LAYER_LEGACY_EXTRA_COLS)
        while len(header) < max_len and extra:
            header.append(extra.pop(0))
        if len(header) < max_len:
            header += [f"extra_{i}" for i in range(max_len - len(header))]
    norm_rows = [r + [""] * (max_len - len(r)) for r in data_rows]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(norm_rows)
    buf.seek(0)
    return pd.read_csv(buf)


# ---------------------------------------------------------------------------
# Degenerate-layer filter for the "best layer = max delta_loglik_test" DLL
# convention (figures/figure8_scale_matched_comparison.py's
# _best_row_per_word, summarize/benchmark.py's _best_last_dll_per_obs).
#
# Found 2026-09-08: OPT-125M's ns_fmri fit, froi=MDlanglocLmPFC, layer 11 has
# beta=-0.072 (t=-1.49, unremarkable) but delta_loglik_test=21638 -- a
# numerical blow-up in the held-out log-likelihood computation itself (every
# other layer/froi in that same model's table sits in [-21, 2]), not a real
# effect. Since "best layer" picks the MAX delta_loglik_test, this one
# degenerate cell silently became that model's reported ns_fmri score.
# GPT-Neo-2.7B had the same failure mode on a different froi.
#
# Fix: drop rows whose delta_loglik_test is an extreme robust-outlier
# relative to the OTHER rows in that SAME model's per-layer/per-unit table,
# before taking the max -- adapts to each dataset's own scale instead of a
# hardcoded magic-number cutoff (this dataset's legitimate deltas range over
# several orders of magnitude across datasets, so no single absolute
# threshold works everywhere). z_thresh=8 is deliberately generous -- a
# genuinely much-better best layer might show z of 2-4 relative to its
# table-mates; 21638 vs. a spread of [-21, 2] is nowhere close to that.
# ---------------------------------------------------------------------------

def sum_dll_across_units(records: list[tuple[float, float]], z_thresh: float = 8.0) -> tuple[float, float]:
    """Sum (delta_loglik_test, n_obs_test) pairs already picked per unit
    (e.g. one per fROI, each at that unit's own best layer), filtering
    degenerate-outlier UNITS first via the same robust-z logic as
    ``filter_degenerate_dll_rows`` -- needed in addition to that per-layer
    filter for models with too few layers to detect an outlier layer within
    one unit's own table (e.g. cbr-rnn-m has exactly 1 layer, so its
    ns_fmri MDlanglocLMFG blow-up -- delta_loglik_test=17640 with an
    unremarkable beta/t_stat, vs. [-8, 8] for its other fROIs -- shows up
    ACROSS units instead of across layers; found 2026-09-08 as a second
    instance of the same numerical bug found in OPT-125M/GPT-Neo-2.7B's
    per-layer blow-ups). Returns (0.0, 0.0) if nothing survives."""
    if not records:
        return 0.0, 0.0
    df = pd.DataFrame(records, columns=["delta_loglik_test", "n_obs_test"])
    df = filter_degenerate_dll_rows(df)
    if df.empty:
        return 0.0, 0.0
    return float(df["delta_loglik_test"].sum()), float(df["n_obs_test"].sum())


def flag_degenerate_dll_rows(df: pd.DataFrame, col: str = "delta_loglik_test", z_thresh: float = 8.0) -> pd.Series:
    """AUDIT ONLY. Boolean mask (True = suspicious) of rows of ``df`` whose ``col`` value is a
    robust outlier (median/MAD z-score > z_thresh) relative to the OTHER rows of the same table.
    Nothing is dropped by this: flagged rows are candidates to investigate or rerun."""
    flags = pd.Series(False, index=df.index)
    if col not in df.columns or len(df) < 4:
        return flags
    vals = pd.to_numeric(df[col], errors="coerce")
    valid = vals.dropna()
    if len(valid) < 4:
        return flags
    med = valid.median()
    mad = (valid - med).abs().median()
    if mad == 0 or pd.isna(mad):
        return flags
    z = (vals - med).abs() / (1.4826 * mad)
    return (z > z_thresh).fillna(False)


def filter_degenerate_dll_rows(df: pd.DataFrame, col: str = "delta_loglik_test", z_thresh: float = 8.0) -> pd.DataFrame:
    """No-op, kept so existing call sites keep working. Results are NOT filtered (decision
    2026-10-01): a suspicious fit is investigated or rerun, never silently dropped. Use
    :func:`flag_degenerate_dll_rows` to list candidates."""
    return df


# ---------------------------------------------------------------------------
# Layer-selection rule for every "best layer" surprisal number (decision 2026-10-01).
#
# OLD rule: the layer (or lag / fROI-layer row) with the max HELD-OUT delta_loglik_test, scored on that
# same held-out set -- the winner's-curse rule: on small test sets (Tuckute, Nieuwland, Pereira, the
# single Szewczyk studies) it inflated the headline by 2-15x (mean over models) and reshuffled the model
# ranking (scripts/analysis/train_vs_test_layer_selection.py).
# NEW rule: pick the row with the max TRAINING delta_loglik_train, then report delta_loglik_test /
# n_obs_test at that row. Needs no refit: both columns are stored in every per-layer results file.
# Set LAYER_SELECTION = "test" to reproduce the old numbers.
# ---------------------------------------------------------------------------
LAYER_SELECTION = "train"
_WARNED_NO_TRAIN = {"done": False}


def select_best_row(valid: pd.DataFrame) -> pd.Series:
    """The row reported as a model's best layer / lag / fROI-layer. ``valid`` already holds only rows with a
    finite ``delta_loglik_test`` and ``n_obs_test > 0``. Under ``LAYER_SELECTION == "train"`` the row with the
    max ``delta_loglik_train`` wins (rows without a finite train value cannot be selected); if no row has a
    train value the old test-based rule is used instead (warned once)."""
    if LAYER_SELECTION == "train" and "delta_loglik_train" in valid.columns:
        tr = pd.to_numeric(valid["delta_loglik_train"], errors="coerce")
        if tr.notna().any():
            return valid.loc[tr.idxmax()]
    if LAYER_SELECTION == "train" and not _WARNED_NO_TRAIN["done"]:
        _WARNED_NO_TRAIN["done"] = True
        import warnings
        warnings.warn("select_best_row: no delta_loglik_train available -> falling back to test-based selection",
                      stacklevel=2)
    return valid.loc[valid["delta_loglik_test"].idxmax()]

# ---------------------------------------------------------------------------
# Group assignment (fixed, ad hoc — do not derive from elsewhere in the repo)
# ---------------------------------------------------------------------------

LLM_FAMILIES: tuple[str, ...] = (
    "gpt2", "gptneo", "opt", "pythia", "pythia_checkpoints", "olmo2",
    "qwen3", "llama32", "mistral",
    # mamba/mamba2 are Standard LLMs again (2026-10-07; they were briefly grouped as Memory-Constrained, 2026-09-20 to 2026-10-07)
    "mamba", "mamba2",
    # rwkv4 joins them (2026-10-07): like Mamba, a fixed-size recurrent state at web/Pile scale, no human-motivated memory constraint
    "rwkv4",
)
# Memory-Constrained: architectures with an explicitly limited working-memory /
# context mechanism. Not a single paper/author — alibi, stickbreaking, and
# forgetting are from Fox et al.; cbr_rnn (a recurrent architecture with a
# bounded memory state) is a separate line of work.
# Memory-Constrained also includes the hard n-gram surprisal baselines
# (hard_2gram/3gram/5gram): an n-gram model has a hard bounded context, i.e. the
# extreme case of a limited working-memory model, so it is grouped here.
# bloom/rwkv4/fox/molm added 2026-09-12: same constrained mechanisms as
# alibi/cbr_rnn/forgetting/stickbreaking above (ALiBi, bounded recurrent
# state, forgetting gate, stick-breaking MoE respectively) but web-scale
# trained rather than Pile/BabyLM-10-100M -- extends this group's scale
# ceiling from ~48M to the billions. The architecture-vs-training-data
# confound this introduces is intentionally NOT split into a separate group;
# it's handled downstream by regressing out perplexity (see
# analysis/perplexity_matched_scaling.py / analysis/matched_scale_regression.py).
MEMORY_CONSTRAINED_FAMILIES: tuple[str, ...] = (
    "alibi", "stickbreaking", "forgetting", "cbr_rnn",
    "hard_2gram", "hard_3gram", "hard_5gram",
    "bloom", "fox", "molm",
)
# The "fox" registry family (Lin et al. 2025 release, zhixuan-lin/*) holds two FoX models and their two MATCHED BASELINES. The
# baselines carry NO memory constraint -- verified 2026-10-12 from the cached config.json: transformer-llama-760m is a plain
# TransformerForCausalLM (RoPE), transformer-pro-760m is a ForgettingTransformerForCausalLM with fgate_type="none" (RoPE, output
# gate, k/v shift, no forget gate), whereas fox-llama/fox-pro have fgate_type="full". They are therefore Standard LLMs (a
# 760M, 48B-token matched pair for the FoX models); family_group() checks this list before the family rule.
UNCONSTRAINED_BASELINE_MODEL_IDS: tuple[str, ...] = (
    "zhixuan-lin/transformer-llama-760m-longcrawl64-48b",
    "zhixuan-lin/transformer-pro-760m-longcrawl64-48b",
)
BABYLM_FAMILIES: tuple[str, ...] = ("babylm", "babylm_gpt2_checkpoints")

# Within the Memory-Constrained families (alibi/stickbreaking/forgetting/
# hard_ngram), every model_id is suffixed "_pile_" or "_babylm_" (e.g.
# "Lanni-ni/alibi_babylm_100m_2layer" vs "Lanni-ni/alibi_pile_2layer") --
# the babylm-data-trained variants get their own group (below), separate
# from both the Pile-trained Memory-Constrained models and the
# standard-architecture BabyLM models. cbr_rnn has no babylm-trained variant
# in the registry ("cclark/cbr-rnn", "cclark/cbr-rnn-m" only) so it always
# stays "Memory-Constrained".
GROUP_ORDER: tuple[str, ...] = ("LLM", "Memory-Constrained", "Constrained-BabyLM", "BabyLM")

# Display-only label shown in figures/tables; the internal "LLM" identifier
# above (used for data lookups, coloring, filtering) is left untouched.
GROUP_DISPLAY: dict[str, str] = {
    "LLM": "Standard LLM",
    "Memory-Constrained": "Memory-Constrained",
    "Constrained-BabyLM": "Constrained-BabyLM",
    "BabyLM": "BabyLM",
}

GROUP_COLORS: dict[str, str] = {
    "LLM": "#1f77b4",
    "Memory-Constrained": "#d62728",
    "Constrained-BabyLM": "#9467bd",
    "BabyLM": "#2ca02c",
}
GROUP_X_OFFSET: dict[str, float] = {
    "LLM": -0.225,
    "Memory-Constrained": -0.075,
    "Constrained-BabyLM": 0.075,
    "BabyLM": 0.225,
}

# ---------------------------------------------------------------------------
# Noise ceilings (from scripts/ceilings/correlation.py)
# ---------------------------------------------------------------------------
# The report normalises every score by its dataset's noise ceiling (ceiling_r =
# sqrt of the Spearman-Brown reliability; see outputs*/noise_ceilings/README.md),
# so 1.0 == "as good as the data reliability allows". Each report dataset maps to
# a noise-ceiling dataset slug + the component(s) matching the score column the
# report plots (averaged when the report averages several components). None =>
# average all components present for that dataset.
_NC_CANDIDATES = (
    # outputs/noise_ceilings/ is the actively-maintained tree (multi-measure
    # FP/GP/SP components, ceilings/correlation.py's canonical --out-dir)
    # -- MUST be checked first. outputs_jeanzay/noise_ceilings/ is a stale,
    # abandoned duplicate (last touched 2026-08-11, single-measure only,
    # component always NaN) that was silently shadowing every ceiling fix
    # made to the canonical tree until this reorder (found 2026-09-08 while
    # debugging why geco__embed etc. reported "no noise ceiling found"
    # despite outputs/noise_ceilings/geco/ceiling_headline.csv being correct).
    REPO_ROOT / "outputs" / "noise_ceilings" / "noise_ceilings_summary.csv",
    OUTPUTS_JEANZAY / "noise_ceilings" / "noise_ceilings_summary.csv",
)

# Behavioral datasets' headline score is now a held-out delta-Pearson-r
# (test_delta_pearsonr: full-model minus baseline-only correlation with RT) --
# a bounded correlation-delta, so it IS ceiling-normalisable against the
# dataset's own split-half reading-time reliability, unlike the old
# delta-log-likelihood metric (kept as score_legacy). See
# scripts/ceilings/correlation.py's *_reading_time_handler / handler_naturalstories_spr.
REPORT_TO_CEILING: dict[str, tuple[str, tuple[str, ...] | None]] = {
    "lpp_fmri":            ("lpp", None),
    "podcast_ieeg":        ("podcast", None),
    "nieuwland_eeg":       ("nieuwland", None),
    "szewczyk_f2007_eeg":  ("szewczyk_f2007", None),
    "szewczyk_h2019_eeg":  ("szewczyk_h2019", None),
    # noise_ceilings_summary.csv's "broderick" row (nc_slug != report slug,
    # like pereira_fmri above) -- added 2026-09-18 alongside re-enabling
    # broderick_eeg in DATASETS.
    "broderick_eeg":       ("broderick", None),
    "ns_fmri":             ("ns_fmri", ("all",)),
    "fedorenko2016_ecog":  ("fedorenko2016", None),
    "tuckute_fmri":        ("tuckute", ("cond_B", "cond_D", "cond_S")),
    "naturalstories_spr":  ("naturalstories_spr", None),
    # noise_ceilings_summary.csv has ONE dataset row per corpus, split into
    # FP/GP/SP by "component" -- comps=None would average ALL of them
    # together, which is wrong for the plain (FP) slug. Filter to FP
    # explicitly (fixed 2026-09-08 -- previously silently blended FP+GP).
    "geco":                ("geco", ("FP",)),
    "meco":                ("meco", ("FP",)),
    "provo":               ("provo", ("FP",)),
    "ucl_reading":         ("ucl_reading", None),
    "zuco_reading":        ("zuco_reading", None),
    # SR/TSR embedding tracks reuse the same pooled ceiling as NR -- the
    # correlation-path noise ceiling for zuco_reading is computed once,
    # pooled across all three tasks (SR/NR/TSR), matching the model sweep's
    # own convention (see ceilings/correlation.py's handler_zuco_reading).
    "zuco_reading_sr":     ("zuco_reading", None),
    "zuco_reading_tsr":    ("zuco_reading", None),
    "onestop":             ("onestop", ("FP",)),
    "pereira_fmri":        ("pereira", None),  # nc_slug is "pereira", not "pereira_fmri" (fixed 2026-09-08)
    # noise_ceilings_summary.csv has ONE dataset row per corpus (geco/meco/
    # provo/onestop), split into FP/GP/SP by "component" -- not a separate
    # "dataset=geco_gp" row -- so these must filter geco/meco/provo/onestop's
    # own rows by component, same pattern as tuckute_fmri/ns_fmri below.
    "geco_gp":             ("geco", ("GP",)),
    "meco_gp":             ("meco", ("GP",)),
    "meco_sp":             ("meco", ("SP",)),
    "provo_gp":            ("provo", ("GP",)),
    "provo_sp":            ("provo", ("SP",)),
    "onestop_gp":          ("onestop", ("GP",)),
    "onestop_sp":          ("onestop", ("SP",)),
}

# ---------------------------------------------------------------------------
# Delta-log-likelihood noise ceilings (Timkey et al. 2026 split-half method;
# see scripts/ceilings/dll_behavioral.py and its ECoG/fMRI/N400
# variants). Report slug -> the exact suffix of
# outputs/noise_ceilings/delta_loglik_ceiling_<tag>.json (everything between
# "delta_loglik_ceiling_" and ".json" -- N400 datasets' filenames embed "_oos"
# in the middle, not just at the end, so this is the literal suffix, not a
# "<tag>_oos" template). Takes priority over REPORT_TO_CEILING in
# ceiling_for() for every surprisal/LME-track dataset now scored as
# per-observation delta log-likelihood (delta_loglik_test / n_obs_test) --
# see scripts/summarize/benchmark.py's _load_lme_reading /
# _load_neural_surprisal / _load_podcast_surprisal_full /
# _load_ns_fmri_surprisal_full. Datasets NOT listed here (all __embed slugs,
# and every other neural dataset's primary embedding-ridge score) keep using
# REPORT_TO_CEILING's correlation-scale ceiling unchanged.
DLL_CEILING_TAGS: dict[str, str] = {
    "naturalstories_spr":            "naturalstories_spr_oos",
    "geco":                          "geco_oos",
    "meco":                          "meco_oos",
    "provo":                         "provo_oos",
    "ucl_reading":                   "ucl_reading_oos",
    "zuco_reading":                  "zuco_reading_oos",
    "zuco_reading_sr":               "zuco_reading_sr_oos",
    "zuco_reading_tsr":              "zuco_reading_tsr_oos",
    "onestop":                       "onestop_oos",
    "pereira_fmri":                  "pereira_fmri_oos",
    # Added 2026-09-07: GECO/MECO/Provo's GP (go-past) and MECO/Provo/OneStop's
    # SP (scan-path/total dwell) measures -- ceilings already existed from an
    # earlier session (outputs/noise_ceilings/delta_loglik_ceiling_<tag>_oos.json),
    # just never had a report-slug entry pointing at them.
    "geco_gp":                       "geco_gp_oos",
    "meco_gp":                       "meco_gp_oos",
    "meco_sp":                       "meco_sp_oos",
    "provo_gp":                      "provo_gp_oos",
    "provo_sp":                      "provo_sp_oos",
    "onestop_gp":                    "onestop_gp_oos",
    "onestop_sp":                    "onestop_sp_oos",
    "lpp_fmri__surprisal":           "lpp_fmri_persubject_oos",
    "ns_fmri__surprisal":            "ns_fmri_fmri_oos",
    "podcast_ieeg__surprisal":       "podcast_region_lags_oos",   # per-region LME, mean over 0-600 ms lags (2026-10-08); was podcast_ecog_oos
    "fedorenko2016_ecog__surprisal": "fedorenko2016_ecog_oos",
    "tuckute_fmri__surprisal":       "tuckute_fmri_oos",
    "nieuwland_eeg__surprisal":      "nieuwland_paper_n400_oos_nieuwland_paper",
    "szewczyk_f2007_eeg__surprisal":  "szewczyk_f2007_oos",
    "szewczyk_h2019_eeg__surprisal":  "szewczyk_h2019_oos",
    "broderick_eeg__surprisal":      "broderick_oos",
    "megmasc_meg__surprisal":        "megmasc",
}

_DLL_CEILING_CACHE: dict[str, float | None] = {}


def dll_ceiling_for(report_slug: str) -> float | None:
    """``ceiling_delta_loglik_per_obs`` for a report slug in DLL_CEILING_TAGS,
    or None if the slug isn't one of those (or the JSON is missing/unreadable) --
    the fallback-to-REPORT_TO_CEILING case, handled in ceiling_for()."""
    tag = DLL_CEILING_TAGS.get(report_slug)
    if tag is None:
        return None
    if tag in _DLL_CEILING_CACHE:
        return _DLL_CEILING_CACHE[tag]
    path = REPO_ROOT / "outputs" / "noise_ceilings" / f"delta_loglik_ceiling_{tag}.json"
    val: float | None = None
    if path.is_file():
        try:
            with open(path) as f:
                val = float(json.load(f)["ceiling_delta_loglik_per_obs"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            val = None
    _DLL_CEILING_CACHE[tag] = val
    return val


# megmasc's noise ceiling lives under outputs_jeanzay/noise_ceilings/megmasc/
# (its own ceilings/correlation_megmasc.py output), not merged into the
# master outputs*/noise_ceilings/noise_ceilings_summary.csv REPORT_TO_CEILING
# reads via load_noise_ceilings() -- special-cased here instead of adding a
# new handler to scripts/ceilings/correlation.py (out of this task's scope).
# Uses r_tt_sqrt_mean (the split-half test-retest ceiling's single-trial
# bound -- this project's documented convention for "the one to compare the
# encoder to"), NOT r_tt_mean (a smaller, differently-scaled quantity).
_MEGMASC_CEILING_PATH = OUTPUTS_JEANZAY / "noise_ceilings" / "megmasc" / "ceiling_headline.csv"
_MEGMASC_CEILING_CACHE: dict[str, float | None] = {}


def _megmasc_ceiling() -> float | None:
    if "val" in _MEGMASC_CEILING_CACHE:
        return _MEGMASC_CEILING_CACHE["val"]
    val: float | None = None
    if _MEGMASC_CEILING_PATH.is_file():
        try:
            row = pd.read_csv(_MEGMASC_CEILING_PATH).iloc[0]
            val = float(row["r_tt_sqrt_mean"])
        except (OSError, KeyError, IndexError, ValueError):
            val = None
    _MEGMASC_CEILING_CACHE["val"] = val
    return val


_NC_CACHE: dict[str, pd.DataFrame] = {}


def load_noise_ceilings() -> pd.DataFrame | None:
    """Load the combined noise-ceiling summary (cached); None if absent."""
    if "df" not in _NC_CACHE:
        df = None
        for p in _NC_CANDIDATES:
            if p.is_file():
                df = pd.read_csv(p)
                break
        _NC_CACHE["df"] = df
    return _NC_CACHE["df"]


def ceiling_for(report_slug: str) -> float | None:
    """Headline ceiling_r for a report dataset (mean over its mapped
    components), or None if unavailable OR non-positive.

    The non-positive check matters: summarize/benchmark.py's write-time
    normalisation guards ``ceiling > 0`` before dividing (a negative or zero
    ceiling is never usable), leaving score_best_norm/score_last_norm as RAW
    scores when it isn't. Every read-time consumer (summarize/report.py's
    _modality_model_df / Pareto fronts / "Distance to Noise Ceiling" table)
    only checks ``ceiling_for(slug) is None`` before deciding whether to
    apply ITS OWN no-ceiling fallback -- without this guard, a negative
    ceiling (e.g. a dataset with a split-half-unstable ceiling, ceiling_r
    ~= -0.0069) would silently pass that check, so the composite would trust
    and blend in that dataset's RAW un-normalised scores alongside every
    other dataset's properly ceiling/max-normalised ones (found 2026-09-08
    auditing exactly this mismatch). Centralising the >0 check here makes
    every consumer see the same "no valid ceiling" answer consistently.
    """
    return _ceiling_for_raw(report_slug)


def _ceiling_for_raw(report_slug: str) -> float | None:
    val = __ceiling_for_uncached(report_slug)
    if val is None or not math.isfinite(val) or val <= 0:
        return None
    return val


def __ceiling_for_uncached(report_slug: str) -> float | None:
    """Headline ceiling_r for a report dataset (mean over its mapped
    components), or None if unavailable.

    Surprisal/LME-track datasets in DLL_CEILING_TAGS are scored as
    per-observation delta log-likelihood and are normalised against their own
    delta-loglik split-half ceiling (dll_ceiling_for()) -- checked FIRST,
    ahead of the correlation-scale REPORT_TO_CEILING lookup below (a
    completely different scale; must never be mixed with it). Embedding-track
    cells (ridge/correlation scale) fall through to REPORT_TO_CEILING as
    before. Secondary variants (``__embed`` / ``__surprisal``) reuse their
    base dataset's ceiling (same underlying signal, same metric family) when
    not separately listed in REPORT_TO_CEILING.
    """
    dll = dll_ceiling_for(report_slug)
    if dll is not None:
        return dll
    if report_slug == "megmasc_meg":
        return _megmasc_ceiling()
    df = load_noise_ceilings()
    if report_slug not in REPORT_TO_CEILING and "__" in report_slug:
        report_slug = report_slug.split("__", 1)[0]
    if df is None or report_slug not in REPORT_TO_CEILING:
        return None
    nc_slug, comps = REPORT_TO_CEILING[report_slug]
    sub = df[df["dataset"] == nc_slug]
    if comps is not None:
        sub = sub[sub["component"].astype(str).isin(comps)]
    vals = pd.to_numeric(sub["ceiling_r"], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else None


def add_ceiling_line(ax, y: float = 1.0, label: str = "noise ceiling") -> None:
    """Draw the noise-ceiling reference line on a ceiling-normalised axis."""
    ax.axhline(y, ls="--", lw=1.4, color="#444", alpha=0.9, zorder=1.5)
    ax.text(0.995, y, f" {label}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="#444", style="italic")
    ymin, ymax = ax.get_ylim()
    if ymax < y * 1.05:
        ax.set_ylim(ymin, y * 1.08)


# Y-axis label used everywhere the score is ceiling-normalised.
NORM_YLABEL = "score / noise ceiling"


def family_group(family: str, model_id: str = "") -> str:
    if model_id in UNCONSTRAINED_BASELINE_MODEL_IDS:
        return "LLM"
    if family in LLM_FAMILIES:
        return "LLM"
    if family in MEMORY_CONSTRAINED_FAMILIES:
        return "Constrained-BabyLM" if "babylm" in model_id.lower() else "Memory-Constrained"
    if family in BABYLM_FAMILIES:
        return "BabyLM"
    raise KeyError(f"Unknown family {family!r} — not assigned to any group")


# ---------------------------------------------------------------------------
# Per-family markers — one distinct shape per family, color still encodes group
# ---------------------------------------------------------------------------

_MARKER_CYCLE: tuple[str, ...] = ("o", "s", "^", "v", "D", "P", "X", "*", "h", "<", ">", "p", "8", "d", "+")

FAMILY_ORDER: tuple[str, ...] = tuple(LPP_FAMILIES.keys())
FAMILY_MARKERS: dict[str, str] = {
    family: _MARKER_CYCLE[i % len(_MARKER_CYCLE)] for i, family in enumerate(FAMILY_ORDER)
}


def family_marker(family: str) -> str:
    return FAMILY_MARKERS.get(family, "o")


# ---------------------------------------------------------------------------
# Registry-based model metadata (n_params, n_layers, label) with CSV fallback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMeta:
    model_id: str
    family: str
    group: str
    label: str
    n_params: float
    n_layers: int | None


def _build_registry_index() -> dict[str, ModelMeta]:
    index: dict[str, ModelMeta] = {}
    for family, specs in LPP_FAMILIES.items():
        for spec in specs:
            group = family_group(family, spec.model_id)
            index[spec.model_id] = ModelMeta(
                model_id=spec.model_id,
                family=family,
                group=group,
                label=spec.label,
                n_params=spec.n_params,
                n_layers=spec.n_layers,
            )
    return index


REGISTRY_INDEX: dict[str, ModelMeta] = _build_registry_index()

# Models excluded from every report table/figure regardless of dataset: broken
# upstream checkpoints, not a data-quality issue in any particular sweep.
EXCLUDED_MODEL_IDS: frozenset[str] = frozenset({
    # Bad perplexities (broken upstream checkpoint) -- see BabyLM-community's repo.
    "BabyLM-community/BabyLM-2026-Baseline-Strict-Interaction",
    "BabyLM-community/BabyLM-2026-Baseline-Strict-Small-Interaction",
})

# Top-level family directories under outputs_jeanzay/<sweep_root>/ that must be
# skipped when walking a sweep root for its model list: these hold a separate,
# unrelated experiment (a Pythia training-checkpoint sweep) that happens to
# live as a sibling of the regular per-family final-checkpoint directories in
# some behavioral embed sweep roots (meco/provo/ucl_reading/zuco_reading
# checkpoint_sweep_embed). Same models as the regular "pythia" family, so
# including it double-counts every Pythia model in the LLM group.
EXCLUDED_FAMILY_DIRS: frozenset[str] = frozenset({
    "pythia_checkpoints",
    "babylm_gpt2_checkpoints",
})


def filter_registry_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep exact benchmark targets, excluding historical checkpoint families."""
    from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
    expected = {(family, task.model_id, task.revision)
                for family in LPP_FAMILY_NAMES if family not in EXCLUDED_FAMILY_DIRS
                for task in flatten_family_tasks(family)}
    if frame.empty:
        return frame.copy()
    keys = frame[["family", "model_id", "revision"]].astype(str)
    return frame.loc[[key in expected for key in keys.itertuples(index=False, name=None)]].copy()


def model_dir_to_model_id(model_dir: str) -> str:
    """``org_repo`` directory name -> ``org/repo`` HF model id (first underscore only)."""
    if "/" in model_dir:
        return model_dir
    return model_dir.replace("_", "/", 1)


def lookup_model_meta(
    model_id: str,
    family: str,
    n_params_fallback: float | None = None,
    n_layers_fallback: int | None = None,
    label_fallback: str | None = None,
) -> ModelMeta:
    """Resolve a model's metadata from the registry, falling back to CSV-embedded values."""
    meta = REGISTRY_INDEX.get(model_id)
    if meta is not None:
        return meta
    group = family_group(family, model_id)
    return ModelMeta(
        model_id=model_id,
        family=family,
        group=group,
        label=label_fallback or model_id,
        n_params=float(n_params_fallback) if n_params_fallback is not None else float("nan"),
        n_layers=int(n_layers_fallback) if n_layers_fallback is not None else None,
    )


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------


def sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def mean_sem_by_group(df: pd.DataFrame, value_col: str, group_col: str = "group") -> pd.DataFrame:
    """Per-group mean +/- SEM across models (one row per model expected in df)."""
    rows = []
    for group in GROUP_ORDER:
        sub = df.loc[df[group_col] == group, value_col].dropna()
        if sub.empty:
            continue
        rows.append(
            {
                "group": group,
                "mean": float(sub.mean()),
                "sem": sem(sub.to_numpy()),
                "n": int(sub.size),
            }
        )
    return pd.DataFrame(rows)


def _human_params(n_params: float) -> str:
    if n_params >= 1e9:
        return f"{n_params / 1e9:.1f}B"
    return f"{n_params / 1e6:.0f}M"


# ---------------------------------------------------------------------------
# Plotting helpers — "scientific ready": labeled axes, legends, fit lines
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def _family_legend_handles(df: pd.DataFrame, family_col: str = "family", group_col: str = "group"):
    """One legend handle per (family) present in df, marker=family shape,
    color=that family's group color."""
    handles = []
    for family in sorted(df[family_col].dropna().unique(), key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 999):
        sub = df.loc[df[family_col] == family]
        if sub.empty:
            continue
        group = sub[group_col].iloc[0]
        handles.append(
            plt.Line2D(
                [], [], marker=family_marker(family), linestyle="None",
                color=GROUP_COLORS.get(group, "grey"), markersize=7, label=family,
            )
        )
    return handles


def _add_param_scale_reference(ax, df: pd.DataFrame, group_col: str = "group", n_params_col: str = "n_params", ref_group: str = "LLM") -> None:
    """Light vertical dashed reference lines (not one per model) showing where
    specific parameter counts fall along the sorted-by-size x-index, so the
    reader can gauge the size range even though x is an arbitrary index."""
    sub = df.loc[df[group_col] == ref_group].dropna(subset=[n_params_col])
    if sub.empty:
        # fall back to whichever group has the widest parameter range
        best_group, best_span = None, -1
        for group in GROUP_ORDER:
            g = df.loc[df[group_col] == group].dropna(subset=[n_params_col])
            if len(g) < 2:
                continue
            span = np.log10(g[n_params_col].max()) - np.log10(g[n_params_col].min())
            if span > best_span:
                best_span, best_group = span, group
        if best_group is None:
            return
        sub = df.loc[df[group_col] == best_group].dropna(subset=[n_params_col])
    sub = sub.sort_values(n_params_col).reset_index(drop=True)
    if len(sub) < 2:
        return
    n_refs = min(5, len(sub))
    idxs = np.unique(np.linspace(0, len(sub) - 1, n_refs).astype(int))
    ymax = ax.get_ylim()[1]
    for i in idxs:
        val = sub.loc[i, n_params_col]
        ax.axvline(i, color="grey", alpha=0.15, linestyle=":", zorder=0)
        ax.text(i, ymax, _human_params(val), rotation=90, fontsize=6.5, color="grey",
                 ha="right", va="top", alpha=0.7)


def draw_group_scatter(
    ax,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    group_col: str = "group",
    family_col: str = "family",
    show_param_scale: bool = True,
) -> None:
    """Every model's score at x = arbitrary index within its group (sorted by
    size, small per-group x-offset). Marker shape encodes family, color encodes
    group. Diamond + error bar shows group mean +/- SEM."""
    for group in GROUP_ORDER:
        sub = df.loc[df[group_col] == group].dropna(subset=[value_col])
        if sub.empty:
            continue
        sub = sub.sort_values("n_params")
        x = np.arange(len(sub)) + GROUP_X_OFFSET[group]
        for marker in sub[family_col].unique():
            fam_mask = sub[family_col] == marker
            ax.scatter(
                x[fam_mask.to_numpy()], sub.loc[fam_mask, value_col],
                color=GROUP_COLORS[group], alpha=0.65, s=64,
                marker=family_marker(marker), zorder=2, edgecolors="none",
            )
        m, s = sub[value_col].mean(), sem(sub[value_col].to_numpy())
        ax.errorbar(
            [len(sub) / 2 - 0.5 + GROUP_X_OFFSET[group] * 3], [m], yerr=[s],
            fmt="D", color=GROUP_COLORS[group], markersize=9, markeredgecolor="black",
            markeredgewidth=0.8, capsize=4, elinewidth=1.5,
            label=f"{group} mean +/- SEM (n={len(sub)})", zorder=3,
        )
    if show_param_scale:
        _add_param_scale_reference(ax, df, group_col=group_col)
    ax.set_xlabel("Model index within group (sorted by parameter count; grey dotted\n"
                   "lines mark reference parameter counts to show the size range)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    group_handles, group_labels = ax.get_legend_handles_labels()
    family_handles = _family_legend_handles(df, family_col=family_col, group_col=group_col)
    ax.legend(
        handles=group_handles + family_handles, labels=group_labels + [h.get_label() for h in family_handles],
        fontsize=6.5, loc="best", frameon=True, ncol=2,
    )


def plot_group_summary(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    ylabel: str,
    title: str,
    group_col: str = "group",
) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 0.15 * len(df) + 3), 6))
    draw_group_scatter(ax, df, value_col, ylabel, title, group_col=group_col)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _add_group_fit_lines(ax, df: pd.DataFrame, x_col: str, value_col: str, group_col: str = "group") -> None:
    """Per-group linear fit of value_col ~ log10(x_col), drawn across that
    group's own x-range."""
    for group in GROUP_ORDER:
        sub = df.loc[df[group_col] == group].dropna(subset=[value_col, x_col])
        sub = sub.loc[sub[x_col] > 0]
        if len(sub) < 2:
            continue
        x = np.log10(sub[x_col].to_numpy())
        y = sub[value_col].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(10 ** xs, slope * xs + intercept, color=GROUP_COLORS[group],
                 linestyle="--", linewidth=1.6, alpha=0.85, zorder=1)


def draw_scaling_law(
    ax,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    group_col: str = "group",
    family_col: str = "family",
    label_col: str | None = "label",
    fit_line: bool = True,
) -> None:
    """log10(n_params) vs score, all 3 groups overlaid on shared axes, with a
    per-group linear fit (dashed) and, optionally, a text label per point."""
    plotted = False
    for group in GROUP_ORDER:
        sub = df.loc[df[group_col] == group].dropna(subset=[value_col, "n_params"])
        if sub.empty:
            continue
        for family in sub[family_col].unique():
            fam_sub = sub.loc[sub[family_col] == family]
            ax.scatter(
                fam_sub["n_params"], fam_sub[value_col], color=GROUP_COLORS[group],
                alpha=0.75, s=64, marker=family_marker(family), zorder=2,
            )
        if label_col and label_col in sub.columns:
            for _, r in sub.iterrows():
                ax.annotate(
                    str(r[label_col]), (r["n_params"], r[value_col]),
                    fontsize=5, alpha=0.75, xytext=(3, 2), textcoords="offset points",
                )
        plotted = True
    if fit_line:
        _add_group_fit_lines(ax, df, "n_params", value_col, group_col=group_col)
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if plotted:
        handles = [plt.Line2D([], [], marker="o", linestyle="None", color=GROUP_COLORS[g], label=GROUP_DISPLAY.get(g, g)) for g in GROUP_ORDER if (df[group_col] == g).any()]
        if fit_line:
            handles.append(plt.Line2D([], [], color="grey", linestyle="--", label="per-group linear fit"))
        ax.legend(handles=handles, fontsize=7, loc="best", frameon=True)


def plot_scaling_law(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path | None,
    ylabel: str,
    title: str,
    ax=None,
    label_points: bool = False,
    group_col: str = "group",
) -> None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    draw_scaling_law(ax, df, value_col, ylabel, title, group_col=group_col, label_col="label" if label_points else None)
    if standalone:
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


def plot_scaling_law_by_group(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    ylabel: str,
    title: str,
    group_col: str = "group",
    ceiling_line: bool = False,
    label_points: bool = True,
    ppl_col: str | None = None,
) -> None:
    """2x2 scaling-law grid: All groups + one panel per model category (LLM /
    Memory-Constrained / BabyLM), so each category's trend is readable on its own
    instead of being crushed together on a single shared axis.

    x-axis is parameter count by default; pass ``ppl_col`` to plot vs perplexity.
    """
    x_col = ppl_col or "n_params"
    panels: list[tuple[str, pd.DataFrame]] = [("All groups", df)]
    panels += [(g, df.loc[df[group_col] == g]) for g in GROUP_ORDER]
    # Grid sized to fit "All groups" + one panel per category, not a hardcoded
    # 2x2 -- with 4 categories that's 5 panels, which zip() would silently
    # truncate to 4 under a fixed 2x2 grid.
    ncols = 2
    nrows = -(-len(panels) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.5 * nrows))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_axis_off()  # unused trailing slot(s)
    for ax, (panel_title, sub) in zip(axes.flat, panels):
        sub = sub.dropna(subset=[value_col])
        if sub.empty or sub[x_col].notna().sum() == 0:
            ax.set_title(f"{panel_title} (no data)")
            ax.set_axis_off()
            continue
        n = sub["model_id"].nunique() if "model_id" in sub.columns else len(sub)
        label_col = "label" if label_points else None
        if ppl_col:
            draw_scaling_by_perplexity(ax, sub, value_col, ppl_col, ylabel,
                                       f"{panel_title} (n={n})", group_col=group_col,
                                       label_col=label_col)
        else:
            draw_scaling_law(ax, sub, value_col, ylabel, f"{panel_title} (n={n})",
                            group_col=group_col, label_col=label_col)
        if ceiling_line:
            add_ceiling_line(ax)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def draw_layer_profiles(
    ax,
    layer_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    layer_frac_col: str = "layer_frac",
    group_col: str = "group",
    model_col: str = "model_id",
    n_bins: int = 21,
) -> None:
    """x = layer/n_layers, one line per group (mean across that group's models),
    shaded band = +/- SEM (spread) within group across models at each relative depth."""
    bins = np.linspace(0.0, 1.0, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plotted = False
    for group in GROUP_ORDER:
        sub = layer_df.loc[layer_df[group_col] == group].dropna(subset=[value_col, layer_frac_col])
        if sub.empty:
            continue
        sub = sub.copy()
        sub["_bin"] = pd.cut(sub[layer_frac_col], bins=bins, labels=False, include_lowest=True)
        agg = sub.groupby("_bin")[value_col].agg(mean="mean", sem=lambda s: sem(s.to_numpy()))
        x = bin_centers[agg.index.astype(int)]
        ax.plot(x, agg["mean"], color=GROUP_COLORS[group], label=f"{group} (n_models={sub[model_col].nunique()})")
        ax.fill_between(x, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=GROUP_COLORS[group], alpha=0.2)
        plotted = True
    ax.set_xlabel("Relative layer depth (layer / n_layers)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if plotted:
        ax.legend(fontsize=7, loc="best", frameon=True)


def draw_layer_profiles_interp(
    ax,
    layer_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    layer_frac_col: str = "layer_frac",
    group_col: str = "group",
    model_col: str = "model_id",
) -> None:
    """Same purpose as draw_layer_profiles (x = relative layer depth, one
    line per group +/- SEM band across that group's models), but resamples
    each model onto a SHARED grid via linear interpolation instead of
    binning into a fixed number of bins.

    draw_layer_profiles's fixed n_bins=21 histogram-style binning means
    models with few layers (e.g. a 2-4 layer constrained model) only land in
    a handful of the 21 bins -- most bins for a small/heterogeneous group
    (e.g. BabyLM, ~11 models spanning 11-13 raw layers each) then average
    over just 1-2 models, making the group mean noisy/jagged bin-to-bin
    (worse than the underlying per-model curves actually are).

    Here the grid resolution is DATA-DRIVEN: N = the max number of distinct
    layer_frac points any single model in ``layer_df`` actually has (not an
    arbitrary round number), so the grid is exactly as fine as the deepest
    model warrants. Every other model is resampled onto that same N-point
    grid via ``np.interp`` (linear interpolation between its own measured
    points; a 1-layer model has nothing to interpolate between, so it's
    treated as a constant -- flat line at its one value across the whole
    grid). Every model then contributes a value at every grid point, so the
    group mean at each point is always computed over the full group instead
    of whatever handful of models happened to land in that bin.

    The tradeoff versus draw_layer_profiles: points between a shallow
    model's actual measured layers are interpolated (invented), not
    measured -- this trades "some fabricated in-between values" for "far
    less sampling noise in the group mean," which is preferable when the
    group is small and unevenly deep (exactly the BabyLM case this was
    built for), but means the resulting curve should be read as a smoothed
    trend, not literally N measured layers per model.
    """
    max_pts = layer_df.groupby(model_col)[layer_frac_col].nunique().max() if not layer_df.empty else 0
    n_grid = max(2, int(max_pts) if pd.notna(max_pts) else 2)
    grid = np.linspace(0.0, 1.0, n_grid)
    plotted = False
    for group in GROUP_ORDER:
        sub = layer_df.loc[layer_df[group_col] == group].dropna(subset=[value_col, layer_frac_col])
        if sub.empty:
            continue
        curves = []
        for _mid, g in sub.groupby(model_col):
            g = g.sort_values(layer_frac_col)
            xs = g[layer_frac_col].to_numpy(dtype=float)
            ys = g[value_col].to_numpy(dtype=float)
            if len(xs) == 1:
                curves.append(np.full(n_grid, ys[0]))
            else:
                curves.append(np.interp(grid, xs, ys))
        if not curves:
            continue
        arr = np.vstack(curves)  # (n_models, n_grid)
        mean = arr.mean(axis=0)
        sems = np.array([sem(arr[:, j]) for j in range(n_grid)])
        ax.plot(grid, mean, color=GROUP_COLORS[group], label=f"{group} (n_models={sub[model_col].nunique()})")
        ax.fill_between(grid, mean - sems, mean + sems, color=GROUP_COLORS[group], alpha=0.2)
        plotted = True
    ax.set_xlabel("Relative layer depth (layer / n_layers)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if plotted:
        ax.legend(fontsize=7, loc="best", frameon=True)


def plot_layer_profiles(
    layer_df: pd.DataFrame,
    value_col: str,
    out_path: Path | None,
    ylabel: str,
    title: str,
    ax=None,
    group_col: str = "group",
) -> None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    draw_layer_profiles(ax, layer_df, value_col, ylabel, title, group_col=group_col)
    if standalone:
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


def draw_scaling_by_perplexity(
    ax,
    df: pd.DataFrame,
    value_col: str,
    ppl_col: str,
    ylabel: str,
    title: str,
    group_col: str = "group",
    family_col: str = "family",
    label_col: str | None = "label",
    fit_line: bool = True,
) -> None:
    plotted = False
    for group in GROUP_ORDER:
        sub = df.loc[df[group_col] == group].dropna(subset=[value_col, ppl_col])
        if sub.empty:
            continue
        for family in sub[family_col].unique():
            fam_sub = sub.loc[sub[family_col] == family]
            ax.scatter(
                fam_sub[ppl_col], fam_sub[value_col], color=GROUP_COLORS[group],
                alpha=0.75, s=64, marker=family_marker(family), zorder=2,
            )
        if label_col and label_col in sub.columns:
            for _, r in sub.iterrows():
                ax.annotate(
                    str(r[label_col]), (r[ppl_col], r[value_col]),
                    fontsize=5, alpha=0.75, xytext=(3, 2), textcoords="offset points",
                )
        plotted = True
    if fit_line:
        _add_group_fit_lines(ax, df, ppl_col, value_col, group_col=group_col)
    ax.set_xscale("log")
    ax.set_xlabel("Perplexity (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if plotted:
        handles = [plt.Line2D([], [], marker="o", linestyle="None", color=GROUP_COLORS[g], label=GROUP_DISPLAY.get(g, g)) for g in GROUP_ORDER if (df[group_col] == g).any()]
        if fit_line:
            handles.append(plt.Line2D([], [], color="grey", linestyle="--", label="per-group linear fit"))
        ax.legend(handles=handles, fontsize=7, loc="best", frameon=True)


def draw_binned_trend_by_perplexity(
    ax,
    df: pd.DataFrame,
    value_col: str,
    ppl_col: str,
    ylabel: str,
    title: str,
    group_col: str = "group",
    n_bins: int = 6,
    show_scatter: bool = True,
) -> None:
    """Quantile-bin each group's models by log10(perplexity) and plot the
    binned mean +/- SEM as a connected line (with the raw points faint in the
    background). This reveals a non-monotonic (e.g. inverted-U) shape directly
    from local averages, without assuming a linear or quadratic functional
    form up front — the shape is whatever the connected line does."""
    plotted = False
    for group in GROUP_ORDER:
        sub = df.loc[df[group_col] == group].dropna(subset=[value_col, ppl_col])
        sub = sub.loc[sub[ppl_col] > 0]
        if len(sub) < n_bins * 2:
            continue
        sub = sub.copy()
        sub["_log_ppl"] = np.log10(sub[ppl_col])
        if show_scatter:
            ax.scatter(sub[ppl_col], sub[value_col], color=GROUP_COLORS[group], alpha=0.18, s=28, zorder=1)
        try:
            sub["_bin"] = pd.qcut(sub["_log_ppl"], q=min(n_bins, sub["_log_ppl"].nunique()), duplicates="drop")
        except ValueError:
            continue
        agg = sub.groupby("_bin", observed=True).agg(
            x=("_log_ppl", "mean"), mean=(value_col, "mean"), sem=(value_col, lambda s: sem(s.to_numpy())), n=(value_col, "size"),
        ).sort_values("x")
        if len(agg) < 3:
            continue
        x_plot = 10 ** agg["x"]
        ax.plot(x_plot, agg["mean"], color=GROUP_COLORS[group], linewidth=2.2, marker="o", markersize=6,
                 label=f"{group} (n={len(sub)}, {len(agg)} bins)", zorder=3)
        ax.fill_between(x_plot, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=GROUP_COLORS[group], alpha=0.22, zorder=2)
        plotted = True
    ax.set_xscale("log")
    ax.set_xlabel("Perplexity (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if plotted:
        ax.legend(fontsize=7, loc="best", frameon=True)


def plot_scaling_by_perplexity(
    df: pd.DataFrame,
    value_col: str,
    ppl_col: str,
    out_path: Path | None,
    ylabel: str,
    title: str,
    ax=None,
    label_points: bool = False,
    group_col: str = "group",
) -> None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    draw_scaling_by_perplexity(ax, df, value_col, ppl_col, ylabel, title, group_col=group_col, label_col="label" if label_points else None)
    if standalone:
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


def plot_roi_profile(
    roi_df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    ylabel: str,
    title: str,
    roi_col: str = "roi",
    group_col: str = "group",
    ceiling_line: bool = False,
) -> None:
    """Bar plot of mean +/- SEM score per ROI, grouped by model group."""
    rois = list(pd.unique(roi_df[roi_col]))
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(rois) + 3), 6))
    width = 0.8 / max(1, len(GROUP_ORDER))
    x_base = np.arange(len(rois))
    for gi, group in enumerate(GROUP_ORDER):
        sub = roi_df.loc[roi_df[group_col] == group]
        if sub.empty:
            continue
        means, sems = [], []
        for roi in rois:
            vals = sub.loc[sub[roi_col] == roi, value_col].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(sem(vals.to_numpy()) if len(vals) else 0.0)
        ax.bar(
            x_base + (gi - 1) * width, means, width=width, yerr=sems, capsize=3,
            color=GROUP_COLORS[group], alpha=0.8, label=GROUP_DISPLAY.get(group, group),
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels(rois, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best", frameon=True)
    if ceiling_line:
        add_ceiling_line(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def draw_timecourse(
    ax,
    tc_df: pd.DataFrame,
    time_col: str,
    value_col: str,
    ylabel: str,
    title: str,
    xlabel: str = "Time (s)",
    group_col: str = "group",
) -> None:
    """One line per group (mean across electrodes/models), shaded +/- SEM band."""
    plotted = False
    for group in GROUP_ORDER:
        sub = tc_df.loc[tc_df[group_col] == group].dropna(subset=[value_col, time_col])
        if sub.empty:
            continue
        agg = sub.groupby(time_col)[value_col].agg(mean="mean", sem=lambda s: sem(s.to_numpy())).sort_index()
        ax.plot(agg.index, agg["mean"], color=GROUP_COLORS[group], label=GROUP_DISPLAY.get(group, group))
        ax.fill_between(agg.index, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=GROUP_COLORS[group], alpha=0.2)
        plotted = True
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if plotted:
        ax.legend(fontsize=7, loc="best", frameon=True)


def plot_timecourse(
    tc_df: pd.DataFrame,
    time_col: str,
    value_col: str,
    out_path: Path | None,
    ylabel: str,
    title: str,
    ax=None,
    xlabel: str = "Time (s)",
    group_col: str = "group",
) -> None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    draw_timecourse(ax, tc_df, time_col, value_col, ylabel, title, xlabel=xlabel, group_col=group_col)
    if standalone:
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


def build_grid(
    panels: list[tuple[str, Callable]],
    out_path: Path,
    suptitle: str | None = None,
    ncols: int = 2,
    panel_size: tuple[float, float] = (8.0, 6.5),
    legend_df: pd.DataFrame | None = None,
) -> None:
    """Combine several draw_*(ax, ...) panels into one grid figure (2x2 / 3x2 /
    etc. depending on panel count), each panel "big enough" to read on its own.
    If legend_df is given, a single shared family-marker legend is drawn below
    the grid instead of repeating it in every panel."""
    n = len(panels)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, draw_fn) in zip(axes, panels):
        draw_fn(ax)
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.0)
    if legend_df is not None and not legend_df.empty:
        handles = _family_legend_handles(legend_df)
        fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="lower center",
                    ncol=min(8, len(handles)), fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98) if legend_df is not None else (0, 0, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_two_category_bars(
    data: pd.DataFrame,
    out_path: Path,
    ylabel: str,
    title: str,
    category_col: str = "category",
    group_col: str = "group",
    mean_col: str = "mean",
    sem_col: str = "sem",
    n_col: str = "n",
) -> None:
    """Grouped bar chart: one cluster of 3 group-bars per category (e.g.
    Behavioral vs Neural), mean +/- SEM."""
    categories = list(pd.unique(data[category_col]))
    fig, ax = plt.subplots(figsize=(3.5 * len(categories) + 2, 6))
    width = 0.8 / len(GROUP_ORDER)
    x_base = np.arange(len(categories))
    for gi, group in enumerate(GROUP_ORDER):
        sub = data.loc[data[group_col] == group].set_index(category_col)
        means = [sub.loc[c, mean_col] if c in sub.index else np.nan for c in categories]
        sems = [sub.loc[c, sem_col] if c in sub.index else 0.0 for c in categories]
        ns = [int(sub.loc[c, n_col]) if c in sub.index else 0 for c in categories]
        bars = ax.bar(
            x_base + (gi - 1) * width, means, width=width, yerr=sems, capsize=4,
            color=GROUP_COLORS[group], alpha=0.85, label=GROUP_DISPLAY.get(group, group), edgecolor="black", linewidth=0.5,
        )
        for bar, n in zip(bars, ns):
            if n == 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(1e-6, abs(bar.get_height())),
                     f"n={n}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Perplexity loading
# ---------------------------------------------------------------------------


def load_perplexity_dir(ppl_root: Path, nested: bool = False) -> pd.DataFrame:
    """Read outputs_jeanzay/<dataset>_perplexity/<model_dir>/*.json into a DataFrame.

    Two naming schemes exist in the sweeps and BOTH must be handled, or a whole
    dataset family silently loses its perplexity:
      * ``perplexity.json``                                  (naturalstories, ucl, neural)
      * ``<revision>_<corpus>_perplexity.json``              (geco, meco, provo)
    When a model has several checkpoint files, the final one (max tokens_seen) wins.

    ``nested=True`` is for datasets with no dedicated ``<dataset>_perplexity/``
    top-level dir (Tuckute writes ``perplexity.json`` inside its own checkpoint-
    sweep tree instead, ``tuckute/<family>/<model_dir>/perplexity.json``) —
    walks one extra level (``ppl_root/<family>/<model_dir>/``) rather than
    ``ppl_root/<model_dir>/`` directly.
    """
    rows = []
    if not ppl_root.exists():
        return pd.DataFrame(columns=["model_id", "perplexity"])
    model_dirs = (
        [d for family_dir in sorted(ppl_root.iterdir()) if family_dir.is_dir()
         for d in sorted(family_dir.iterdir())]
        if nested else sorted(ppl_root.iterdir())
    )
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        files = [f for f in dict.fromkeys(
            [model_dir / "perplexity.json", *sorted(model_dir.glob("*perplexity*.json"))]
        ) if f.is_file()]
        best: dict | None = None
        for f in files:
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("perplexity") is None:
                continue
            if best is None or (data.get("tokens_seen") or 0) >= (best.get("tokens_seen") or 0):
                best = data
        if best is None:
            continue
        model_id = best.get("model_id") or model_dir_to_model_id(model_dir.name)
        rows.append({"model_id": model_id, "perplexity": float(best["perplexity"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset registry (static metadata for text blurbs / report generator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetInfo:
    slug: str
    name: str
    modality: str
    stimuli: str
    n_subjects: str
    metric: str
    has_perplexity: bool
    has_roi: bool
    has_timecourse: bool
    # --- feature-mode axis --------------------------------------------------
    # ``response`` is the thing being predicted (neural vs behavioral); it is
    # derived from modality.  ``feature_mode`` is the LLM signal used to predict
    # it: "embedding" (hidden states) or "surprisal" (scalar, per-layer).
    # ``base_slug`` ties the two feature-mode variants of one dataset together
    # so the report can facet them side by side.  Secondary-mode datasets use a
    # composite slug ``<base>__<mode>`` and set base_slug to the base dataset.
    feature_mode: str = "embedding"
    base_slug: str | None = None
    # Which layer the reported score comes from: "best" (swept, best layer
    # chosen) or "last" (final/output layer only, e.g. scalar output surprisal).
    layer_sel: str = "best"

    @property
    def response(self) -> str:
        return "behavioral" if self.modality == "Behavioral" else "neural"

    @property
    def layer_tag(self) -> str:
        return "best layer" if self.layer_sel == "best" else "last layer"

    @property
    def base(self) -> str:
        return self.base_slug or self.slug


DATASETS: list[DatasetInfo] = [
    DatasetInfo("lpp_fmri", "Le Petit Prince fMRI", "fMRI", "Audio book (~200 min, 9 sections)", "~50 EN", "Voxelwise encoding (Pearson r)", True, True, False),
    DatasetInfo("naturalstories_spr", "Natural Stories SPR", "Behavioral", "10 stories (~10,000 words)", "181", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("geco", "GECO", "Behavioral", "Eye-tracking, English novel (first-pass RT)", "14", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("meco", "MECO", "Behavioral", "Eye-tracking, multilingual reading (first-pass RT)", "multi", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("provo", "Provo", "Behavioral", "Eye-tracking, 55 short passages (first-pass RT)", "84", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("ucl_reading", "UCL Reading", "Behavioral", "Self-paced reading + eye-tracking sentences", "—", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("zuco_reading", "ZuCo Reading", "Behavioral", "Natural reading, total reading time (eye-tracking)", "12", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    # SR (sentiment reading) / TSR (task-specific reading) -- added 2026-09-08,
    # data landed after NR was first wired in (see load_zuco_reading_sr/tsr).
    DatasetInfo("zuco_reading_sr", "ZuCo Reading (SR)", "Behavioral", "Sentiment-reading passages (eye-tracking)", "12", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("zuco_reading_tsr", "ZuCo Reading (TSR)", "Behavioral", "Task-specific reading passages (eye-tracking)", "12", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("onestop", "OneStop", "Behavioral", "Eye-tracking, 33 news articles x 2 difficulty levels (first-pass RT)", "360", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    # GP (go-past) / SP (scan-path, total dwell) measures -- added 2026-09-07
    # so this report's dataset coverage matches analysis/matched_scale_master_table.py.
    DatasetInfo("geco_gp", "GECO (GP)", "Behavioral", "Eye-tracking, English novel (go-past RT)", "14", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("meco_gp", "MECO (GP)", "Behavioral", "Eye-tracking, multilingual reading (go-past RT)", "multi", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("meco_sp", "MECO (SP)", "Behavioral", "Eye-tracking, multilingual reading (scan-path RT)", "multi", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("provo_gp", "Provo (GP)", "Behavioral", "Eye-tracking, 55 short passages (go-past RT)", "84", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("provo_sp", "Provo (SP)", "Behavioral", "Eye-tracking, 55 short passages (scan-path RT)", "84", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("onestop_gp", "OneStop (GP)", "Behavioral", "Eye-tracking, 33 news articles x 2 difficulty levels (go-past RT)", "360", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("onestop_sp", "OneStop (SP)", "Behavioral", "Eye-tracking, 33 news articles x 2 difficulty levels (scan-path RT)", "360", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    DatasetInfo("podcast_ieeg", "Podcast iEEG", "iEEG", "Podcast audio (French)", "9 (285 elec.)", "Electrode-level ridge r (Spearman)", True, True, True),
    DatasetInfo("nieuwland_eeg", "Nieuwland N400 EEG", "EEG (N400)", "Sentence completion", "291", "Amplitude regression (Spearman r)", True, True, False),
    DatasetInfo("szewczyk_f2007_eeg", "Federmeier et al. (2007) N400 EEG", "EEG (N400)", "Sentence reading (cloze manipulation)", "32", "Amplitude regression (Spearman r)", True, False, False),
    DatasetInfo("szewczyk_h2019_eeg", "Hubbard et al. (2019) N400 EEG", "EEG (N400)", "Sentence reading (cloze manipulation)", "32", "Amplitude regression (Spearman r)", True, False, False),
    # broderick_eeg ENABLED 2026-09-18 (embedding/checkpoint track only) --
    # per explicit request, despite its surprisal/LME track staying excluded
    # (non-convergent by-layer fits, no `converged` flag yet -- see
    # note in summarize/benchmark.py above NEURAL_SURPRISAL_SPEC). The
    # checkpoint sweep itself is complete across families (see
    # maintenance/audit_sweep_gaps.py's 2026-09-18 unskip of broderick_ckpt).
    DatasetInfo("broderick_eeg", "Broderick EEG", "EEG", "20 audiobook segments (~196s each, 128-ch BioSemi)", "19", "Best-layer ridge encoding (Spearman r)", True, True, True),
    DatasetInfo("pereira_fmri", "Pereira fMRI", "fMRI", "627 isolated sentences (243- and 384-sentence experiments)", "≤10", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal"),
    # megmasc_meg (Gwilliams MEG-MASC) RE-ENABLED 2026-09-18 per explicit
    # request, despite its noise ceiling still not being fully settled: a
    # second, differently-computed ceiling
    # (delta_loglik_ceiling_megmasc_loso.json, leave-one-subject-out) sits
    # alongside the original r_tt_sqrt_mean test-retest ceiling
    # _megmasc_ceiling() actually reads, both dated within the same week --
    # still being iterated on. Using r_tt_sqrt_mean for now (this project's
    # documented convention); revisit if/when the LOSO ceiling is reconciled.
    DatasetInfo("megmasc_meg", "MEG-MASC (Gwilliams)", "MEG", "4 naturalistic English stories (KIT/Yokogawa, 208 channels)", "11", "Best-layer ridge encoding (Spearman r)", True, False, True),
    DatasetInfo("ns_fmri", "Natural Stories fMRI", "fMRI", "8 stories (Shain et al.)", "43", "Voxelwise/fROI encoding (Spearman r)", True, True, False),
    DatasetInfo("fedorenko2016_ecog", "Fedorenko2016 ECoG", "ECoG", "8 sentence sets (word-by-word)", "5", "Electrode-level ridge (Spearman r)", True, True, False),
    DatasetInfo("tuckute_fmri", "Tuckute fMRI", "fMRI", "2,000 isolated 6-word sentences", "10", "ROI encoding (Spearman r)", True, True, False),

    # ------------------------------------------------------------------
    # Secondary feature-mode variants (the "doubling").  Composite slug
    # <base>__<mode>; base_slug pairs them with the primary above so the
    # report can facet embedding vs surprisal for the same dataset.
    # ------------------------------------------------------------------
    # Behavioral + EMBEDDING (hidden states -> RT ridge, per-subject then averaged):
    DatasetInfo("naturalstories_spr__embed", "Natural Stories SPR (embedding)", "Behavioral", "10 stories (~10,000 words)", "181", "Embedding->RT ridge (Spearman r, per-subject then averaged), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="naturalstories_spr"),
    DatasetInfo("geco__embed", "GECO (embedding)", "Behavioral", "Eye-tracking, English novel (first-pass RT)", "14", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="geco"),
    DatasetInfo("meco__embed", "MECO (embedding)", "Behavioral", "Eye-tracking, multilingual reading (first-pass RT)", "multi", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="meco"),
    DatasetInfo("provo__embed", "Provo (embedding)", "Behavioral", "Eye-tracking, 55 short passages (first-pass RT)", "84", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="provo"),
    DatasetInfo("ucl_reading__embed", "UCL Reading (embedding)", "Behavioral", "Self-paced reading + eye-tracking sentences", "—", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="ucl_reading"),
    DatasetInfo("zuco_reading__embed", "ZuCo Reading (embedding)", "Behavioral", "Natural reading passages (eye-tracking RT)", "12", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="zuco_reading"),
    DatasetInfo("zuco_reading_sr__embed", "ZuCo Reading SR (embedding)", "Behavioral", "Sentiment-reading passages (eye-tracking)", "12", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="zuco_reading_sr"),
    DatasetInfo("zuco_reading_tsr__embed", "ZuCo Reading TSR (embedding)", "Behavioral", "Task-specific reading passages (eye-tracking)", "12", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="zuco_reading_tsr"),
    DatasetInfo("onestop__embed", "OneStop (embedding)", "Behavioral", "Eye-tracking, 33 news articles x 2 difficulty levels (first-pass RT)", "360", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="onestop"),
    # GP/SP embedding-track variants + Pereira fMRI embedding -- added
    # 2026-09-08 (previously missing entirely; data existed on disk under
    # <dataset>_checkpoint_sweep_embed/<measure>_average/ but had no
    # DatasetInfo/SECONDARY_SPEC entry pointing at it).
    DatasetInfo("geco_gp__embed", "GECO GP (embedding)", "Behavioral", "Eye-tracking, English novel (go-past RT)", "14", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="geco_gp"),
    DatasetInfo("meco_gp__embed", "MECO GP (embedding)", "Behavioral", "Eye-tracking, multilingual reading (go-past RT)", "multi", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="meco_gp"),
    DatasetInfo("meco_sp__embed", "MECO SP (embedding)", "Behavioral", "Eye-tracking, multilingual reading (scan-path RT)", "multi", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="meco_sp"),
    DatasetInfo("provo_gp__embed", "Provo GP (embedding)", "Behavioral", "Eye-tracking, 55 short passages (go-past RT)", "84", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="provo_gp"),
    DatasetInfo("provo_sp__embed", "Provo SP (embedding)", "Behavioral", "Eye-tracking, 55 short passages (scan-path RT)", "84", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="provo_sp"),
    DatasetInfo("onestop_gp__embed", "OneStop GP (embedding)", "Behavioral", "Eye-tracking, 33 news articles x 2 difficulty levels (go-past RT)", "360", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="onestop_gp"),
    DatasetInfo("onestop_sp__embed", "OneStop SP (embedding)", "Behavioral", "Eye-tracking, 33 news articles x 2 difficulty levels (scan-path RT)", "360", "Embedding->RT ridge (Spearman r, grand-average-subject), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="onestop_sp"),
    DatasetInfo("pereira_fmri__embed", "Pereira fMRI (embedding)", "fMRI", "627 isolated sentences (243- and 384-sentence experiments)", "≤10", "Embedding->BOLD ridge (Pearson r, per-ROI then averaged), ceiling-normalised", True, False, False, feature_mode="embedding", base_slug="pereira_fmri"),
    # Neural + SURPRISAL (pooled LME of the neural signal on scalar surprisal —
    # the SAME method as behavioral-surprisal, so the surprisal track is fair;
    # native metric = held-out delta Pearson r, ceiling-normalised like every
    # other cell in the report; legacy delta log-likelihood kept as score_legacy).
    DatasetInfo("podcast_ieeg__surprisal", "Podcast iEEG (surprisal)", "iEEG", "Podcast audio (French)", "9 (285 elec.)", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling; per-region LME, mean over 0-600 ms lags, last layer)", True, False, False, feature_mode="surprisal", base_slug="podcast_ieeg", layer_sel="last"),
    DatasetInfo("ns_fmri__surprisal", "Natural Stories fMRI (surprisal)", "fMRI", "8 stories (Shain et al.)", "43", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal", base_slug="ns_fmri"),
    DatasetInfo("lpp_fmri__surprisal", "Le Petit Prince fMRI (surprisal)", "fMRI", "Audio book (~200 min, 9 sections)", "~50 EN", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal", base_slug="lpp_fmri"),
    DatasetInfo("nieuwland_eeg__surprisal", "Nieuwland N400 EEG (surprisal)", "EEG (N400)", "Sentence completion", "291", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal", base_slug="nieuwland_eeg"),
    DatasetInfo("szewczyk_f2007_eeg__surprisal", "Federmeier et al. (2007) N400 EEG (surprisal)", "EEG (N400)", "Sentence reading (cloze manipulation)", "32", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal", base_slug="szewczyk_f2007_eeg"),
    DatasetInfo("szewczyk_h2019_eeg__surprisal", "Hubbard et al. (2019) N400 EEG (surprisal)", "EEG (N400)", "Sentence reading (cloze manipulation)", "32", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal", base_slug="szewczyk_h2019_eeg"),
    DatasetInfo("fedorenko2016_ecog__surprisal", "Fedorenko2016 ECoG (surprisal)", "ECoG", "8 sentence sets (word-by-word)", "5", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal", base_slug="fedorenko2016_ecog"),
    DatasetInfo("tuckute_fmri__surprisal", "Tuckute fMRI (surprisal)", "fMRI", "2,000 isolated 6-word sentences", "10", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal", base_slug="tuckute_fmri"),
    # broderick_eeg__surprisal ENABLED 2026-09-28: the pooled-LME rewrite
    # (crossed (1|subject)+(1+surprisal_z|region), see
    # neuroai/regressors/broderick_lme.py) converges 100% across the full
    # registry (1979/1979 layer-fits, zero fallback) -- the earlier
    # per-subject-then-averaged approach's missing convergence flag is now
    # a real per-layer column.
    DatasetInfo("broderick_eeg__surprisal", "Broderick EEG (surprisal)", "EEG", "20 audiobook segments (~196s each, 128-ch BioSemi)", "19", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal", base_slug="broderick_eeg"),
    # megmasc_meg__surprisal ENABLED 2026-09-28: sweep was already complete
    # (118 models) but the composite was never registered/wired to a loader.
    DatasetInfo("megmasc_meg__surprisal", "MEG-MASC (surprisal)", "MEG", "4 naturalistic English stories (KIT/Yokogawa, 208 channels)", "11", "Normalized ΔLL (held-out Δ log-likelihood / noise ceiling, best layer)", True, False, False, feature_mode="surprisal", base_slug="megmasc_meg"),
]

DATASETS_BY_SLUG: dict[str, DatasetInfo] = {d.slug: d for d in DATASETS}

# Slugs of the "primary" datasets (one feature-mode entry each — the original
# benchmark).  The secondary variants above are the doubling.
PRIMARY_SLUGS: tuple[str, ...] = tuple(d.slug for d in DATASETS if d.base_slug is None)

# ---------------------------------------------------------------------------
# Dataset families — the scientific grouping used for the faceted figures.
# Finer than MODALITY_BUCKET: ERP/N400 is separated from continuous EEG, and
# self-paced reading is separated from eye-tracking. Podcast is ECoG (not iEEG).
# ---------------------------------------------------------------------------

DATASET_FAMILY: dict[str, str] = {
    "lpp_fmri": "fMRI",
    "ns_fmri": "fMRI",
    "tuckute_fmri": "fMRI",
    "podcast_ieeg": "ECoG",
    "fedorenko2016_ecog": "ECoG",
    "nieuwland_eeg": "N400/ERP",
    "szewczyk_f2007_eeg": "N400/ERP",
    "szewczyk_h2019_eeg": "N400/ERP",
    "naturalstories_spr": "SPR",
    "ucl_reading": "SPR",
    "geco": "Eye-tracking",
    "meco": "Eye-tracking",
    "provo": "Eye-tracking",
    "zuco_reading": "Eye-tracking",
    "zuco_reading_sr": "Eye-tracking",
    "zuco_reading_tsr": "Eye-tracking",
    "onestop": "Eye-tracking",
    "pereira_fmri": "fMRI",
    "geco_gp": "Eye-tracking",
    "meco_gp": "Eye-tracking",
    "meco_sp": "Eye-tracking",
    "provo_gp": "Eye-tracking",
    "provo_sp": "Eye-tracking",
    "onestop_gp": "Eye-tracking",
    "onestop_sp": "Eye-tracking",
}

DATASET_FAMILY_ORDER: tuple[str, ...] = (
    "fMRI", "ECoG", "N400/ERP", "EEG", "SPR", "Eye-tracking",
)

# Which families are ceiling-normalised (neural) vs delta-metric (behavioral).
BEHAVIORAL_FAMILIES: tuple[str, ...] = ("SPR", "Eye-tracking")


# ---------------------------------------------------------------------------
# Methods metadata — the exact analysis window / baseline / metric / aggregation
# per dataset. Audited from neuroai/benchmarks/*.py. These are NOT uniform (see
# the Methods table in the report): windows and baselines differ across datasets,
# though the aggregation convention is consistent (per-unit Pearson r at the best
# layer, then mean over units / subjects).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetMethods:
    window: str        # analysis window relative to word/stimulus onset
    baseline: str      # baseline correction interval
    sfreq: str         # sampling rate used for scoring
    metric: str        # correlation family actually scored
    aggregation: str   # what the reported scalar averages over


DATASET_METHODS: dict[str, DatasetMethods] = {
    "lpp_fmri": DatasetMethods(
        "HRF-convolved (no epoch)", "—", "TR (2 s)", "Pearson r",
        "per-voxel r → mean over voxels"),
    "ns_fmri": DatasetMethods(
        "HRF-convolved (no epoch)", "—", "TR", "Pearson r",
        "per-fROI best-layer ρ → mean over fROIs (LANG/MD)"),
    "tuckute_fmri": DatasetMethods(
        "sentence-level BOLD", "—", "per sentence", "Pearson r",
        "per-ROI best-layer ρ → mean over ROIs, per condition (B/D/S)"),
    "podcast_ieeg": DatasetMethods(
        "−0.2 → 0.8 s (scored; raw epochs extracted over −1 → 1 s)", "none (z-scored per electrode, train-fold stats)", "32 Hz", "Pearson r",
        "high-γ, per-electrode lag-matrix ridge; best-layer ρ → mean over electrodes"),
    "fedorenko2016_ecog": DatasetMethods(
        "word-level scalar (no time axis)", "—", "—", "Pearson r",
        "per-electrode best-layer ρ → mean over subjects"),
    "nieuwland_eeg": DatasetMethods(
        "−0.2 → 0.8 s", "(−0.2, 0)", "32 Hz", "Pearson r",
        "N400 = mean 200–500 ms; per-channel ρ → mean over channels & subjects"),
    "szewczyk_f2007_eeg": DatasetMethods(
        "N400 window only (300–500 ms)", "—", "—", "Pearson r",
        "item-mean N400 (mean over subjects per item), one study of the Szewczyk & Federmeier collection; out-of-fold ρ vs item means"),
    "szewczyk_h2019_eeg": DatasetMethods(
        "N400 window only (300–500 ms)", "—", "—", "Pearson r",
        "item-mean N400 (mean over subjects per item), one study of the Szewczyk & Federmeier collection; out-of-fold ρ vs item means"),
    "naturalstories_spr": DatasetMethods(
        "word reading time", "—", "—", "Δ Pearson r (held-out)",
        "held-out Δr of LME(RT ~ surprisal) over baseline LME, ceiling-normalised"),
    "ucl_reading": DatasetMethods(
        "word reading time", "—", "—", "Δ Pearson r (held-out)",
        "held-out Δr of LME(RT ~ surprisal) over baseline LME, ceiling-normalised"),
    "geco": DatasetMethods(
        "first-pass fixation time", "—", "—", "Δ Pearson r (held-out)",
        "held-out Δr of LME(RT ~ surprisal) over baseline LME, ceiling-normalised"),
    "meco": DatasetMethods(
        "first-pass fixation time", "—", "—", "Δ Pearson r (held-out)",
        "held-out Δr of LME(RT ~ surprisal) over baseline LME, ceiling-normalised"),
    "provo": DatasetMethods(
        "first-pass fixation time", "—", "—", "Δ Pearson r (held-out)",
        "held-out Δr of LME(RT ~ surprisal) over baseline LME, ceiling-normalised"),
}

# Modality bucket for the executive-summary panel (fMRI / EEG / iEEG-ECoG / Behavioral)
MODALITY_BUCKET: dict[str, str] = {
    "lpp_fmri": "fMRI",
    "ns_fmri": "fMRI",
    "tuckute_fmri": "fMRI",
    "naturalstories_spr": "Behavioral",
    "geco": "Behavioral",
    "meco": "Behavioral",
    "provo": "Behavioral",
    "ucl_reading": "Behavioral",
    "zuco_reading": "Behavioral",
    "zuco_reading_sr": "Behavioral",
    "zuco_reading_tsr": "Behavioral",
    "onestop": "Behavioral",
    "geco_gp": "Behavioral",
    "meco_gp": "Behavioral",
    "meco_sp": "Behavioral",
    "provo_gp": "Behavioral",
    "provo_sp": "Behavioral",
    "onestop_gp": "Behavioral",
    "onestop_sp": "Behavioral",
    "podcast_ieeg": "iEEG / ECoG",
    "fedorenko2016_ecog": "iEEG / ECoG",
    "nieuwland_eeg": "EEG",
    "szewczyk_f2007_eeg": "EEG",
    "szewczyk_h2019_eeg": "EEG",
    "broderick_eeg": "EEG",
    "pereira_fmri": "fMRI",
    "megmasc_meg": "MEG",
}

# MEG is physically distinct from EEG (magnetometers, not scalp electrodes;
# its own noise-ceiling scale) -- a genuinely separate bucket, not folded
# into "EEG".
NEURAL_MODALITIES: tuple[str, ...] = ("fMRI", "EEG", "iEEG / ECoG", "MEG")
BEHAVIORAL_MODALITIES: tuple[str, ...] = ("Behavioral",)

# Derived, not hand-maintained: every dataset slug whose modality bucket is
# neural (fMRI/EEG/iEEG-ECoG), i.e. every non-Behavioral dataset. Use this
# instead of excluding slugs by name (that broke silently when geco/meco/provo/
# ucl_reading — all Behavioral — were added and a name-based exclude list only
# knew about "naturalstories_spr").
NEURAL_SLUGS: tuple[str, ...] = tuple(s for s, m in MODALITY_BUCKET.items() if m in NEURAL_MODALITIES)
