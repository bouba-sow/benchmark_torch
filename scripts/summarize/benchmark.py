"""Unified benchmark summary pipeline.

Reads raw sweep results directly from ``outputs_jeanzay/`` for all 12 active
datasets and writes a consistent output tree to
``outputs/benchmark_summary/<dataset_slug>/`` (models grouped into LLM /
Memory-Constrained / BabyLM per neuroai/analysis/benchmark_common.py).

Does not read or write anything under outputs/<dataset>_summary/ and does not
modify outputs_jeanzay/ (read-only raw source of truth). See scripts/summarize_*.py
for per-dataset reference implementations (read-only format references).

Usage:
    .venv/bin/python scripts/summarize/benchmark.py [--datasets lpp_fmri ns_fmri ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.analysis.benchmark_common import (  # noqa: E402
    BENCHMARK_SUMMARY_ROOT,
    DATASETS_BY_SLUG,
    EXCLUDED_FAMILY_DIRS,
    EXCLUDED_MODEL_IDS,
    filter_degenerate_dll_rows,
    sum_dll_across_units,
    MEMORY_CONSTRAINED_FAMILIES,
    NORM_YLABEL,
    OUTPUTS_JEANZAY,
    REGISTRY_INDEX,
    add_ceiling_line,
    ceiling_for,
    lookup_model_meta,
    load_perplexity_dir,
    model_dir_to_model_id,
    build_grid,
    plot_layer_profiles,
    plot_roi_profile,
    plot_scaling_by_perplexity,
    plot_scaling_law,
    plot_scaling_law_by_group,
    plot_timecourse,
)


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None
    if df.empty:
        return None
    return df


def _dlt_sibling(sweep_root: Path) -> Path | None:
    """The ``<topdir>_dlt`` counterpart of a sweep root, if it exists. The `_dlt`
    suffix goes on the top-level dir under outputs_jeanzay (before any subdir like
    ``/fp``), e.g. geco/surprisal_sweep/fp -> geco/surprisal_sweep/fp."""
    try:
        parts = sweep_root.relative_to(OUTPUTS_JEANZAY).parts
    except ValueError:
        return None
    if not parts:
        return None
    cand = OUTPUTS_JEANZAY.joinpath(parts[0] + "_dlt", *parts[1:])
    return cand if cand.is_dir() else None


def _iter_model_score_files(sweep_root: Path, scores_filename: str, _prefer_dlt: bool = False):
    """Yield (family, model_dir_path, scores_csv_path) for every model, handling
    both flat layout (<family>/<model_dir>/<scores>.csv) and the LPP-style nested
    layout (<family>/<model_dir>/checkpoints/<revision>/<scores>.csv).

    ``_prefer_dlt`` is opt-in and used **only for behavioral LME** loaders, where
    the ``<topdir>_dlt`` sibling is the canonical multilayer re-run: each model is
    taken from the `_dlt` dir if present there, else from the non-dlt dir (per-model,
    so coverage never drops). It is OFF by default because for NEURAL datasets the
    ``_dlt`` dirs are a *different* analysis (DLT difficulty), not an embedding
    re-run, so those loaders must read their own canonical dir as written by the
    run scripts."""
    if _prefer_dlt:
        dlt = _dlt_sibling(sweep_root)
        if dlt is not None:
            seen: set[str] = set()
            for fam, md, files in _iter_model_score_files(dlt, scores_filename, _prefer_dlt=False):
                seen.add(md.name)
                yield fam, md, files
            for fam, md, files in _iter_model_score_files(sweep_root, scores_filename, _prefer_dlt=False):
                if md.name not in seen:
                    yield fam, md, files
            return
    if not sweep_root.exists():
        return
    for family_dir in sorted(sweep_root.iterdir()):
        if not family_dir.is_dir() or family_dir.name == "summary":
            continue
        if family_dir.name in EXCLUDED_FAMILY_DIRS:
            continue
        family = family_dir.name
        for model_dir in sorted(family_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            direct = model_dir / scores_filename
            if direct.exists():
                yield family, model_dir, [direct]
                continue
            ckpt_root = model_dir / "checkpoints"
            if ckpt_root.is_dir():
                files = sorted(ckpt_root.glob(f"*/{scores_filename}"))
                if files:
                    yield family, model_dir, files


def _pick_final_row(df: pd.DataFrame) -> pd.Series:
    """Given a model's per-checkpoint rows, pick the row with max tokens_seen/step."""
    sort_col = "tokens_seen" if "tokens_seen" in df.columns else ("step" if "step" in df.columns else None)
    if sort_col is not None:
        df = df.sort_values(sort_col)
    return df.iloc[-1]


def _model_id_from_dir(model_dir: Path) -> str:
    return model_dir_to_model_id(model_dir.name)


# Real delta-Pearson-r / delta-Spearman-r values (test_delta_pearsonr,
# test_delta_spearmanr, surprisal_delta_*) are always tiny in practice --
# adding one scalar surprisal predictor to an already-fit baseline LME rarely
# moves the correlation with held-out RT by more than a few hundredths.
# Verified empirically: pooling every naturalstories_spr/geco/meco/provo/
# ucl_reading/zuco_reading (model, layer) row, the real signal clusters below
# ~0.05 with a clean bimodal gap to a second cluster around ~0.5+ that comes
# from a data-quality bug in specific (model, layer) LME fits -- most visible
# in naturalstories_spr (~24% of rows affected), where the *baseline*-only
# model's held-out correlation occasionally comes out as a sentinel-looking
# value (exactly -1.0, or implausibly near 0 while the full model's held-out
# correlation is a normal ~0.5), producing a nonsense "delta" once subtracted.
# These rows report converged=True, so the existing convergence filter alone
# doesn't catch them. Filtering by the delta's own magnitude (not tied to any
# one dataset) is deliberately generic and a no-op on already-clean data.
DELTA_CORR_SANITY_MAX = 0.1


def _sanitize_delta_col(df: pd.DataFrame, col: str, max_abs: float = DELTA_CORR_SANITY_MAX) -> pd.DataFrame:
    """NaN out delta-correlation rows whose magnitude is implausible (see
    DELTA_CORR_SANITY_MAX docstring above) so they can't be picked up by a
    max-over-layers ``best`` selection or land as a garbage ``last`` layer."""
    if col not in df.columns:
        return df
    vals = pd.to_numeric(df[col], errors="coerce")
    df = df.copy()
    df[col] = vals.where(vals.abs() <= max_abs)
    return df


def _best_last_dll_per_obs(df: pd.DataFrame, layer_col: str = "layer") -> tuple[float, float]:
    """Per-observation delta-log-likelihood (delta_loglik_test / n_obs_test) at
    the best layer (max delta_loglik_test, NOT max of the ratio itself -- the
    "best layer" selection is defined on delta_loglik_test's own scale, matching
    the convention already established in
    scripts/figures/figure8_scale_matched_comparison.py's _best_row_per_word)
    and at the last (highest-index) layer. NaN/NaN if delta_loglik_test or
    n_obs_test is missing/unusable everywhere."""
    need = {"delta_loglik_test", "n_obs_test"}
    if not need <= set(df.columns):
        return float("nan"), float("nan")
    d = df.copy()
    d["delta_loglik_test"] = pd.to_numeric(d["delta_loglik_test"], errors="coerce")
    d["n_obs_test"] = pd.to_numeric(d["n_obs_test"], errors="coerce")
    valid = d.dropna(subset=["delta_loglik_test", "n_obs_test"])
    valid = valid[valid["n_obs_test"] > 0]
    valid = filter_degenerate_dll_rows(valid)
    if valid.empty:
        return float("nan"), float("nan")
    best_row = valid.loc[valid["delta_loglik_test"].idxmax()]
    best = float(best_row["delta_loglik_test"]) / float(best_row["n_obs_test"])
    if layer_col in valid.columns:
        lay = pd.to_numeric(valid[layer_col], errors="coerce")
        if lay.notna().any():
            last_layer = lay.max()
            last_rows = valid[lay == last_layer]
            last = float((last_rows["delta_loglik_test"] / last_rows["n_obs_test"]).mean())
            return best, last
    return best, best


def _best_last_over_layers(df: pd.DataFrame, score_col: str, layer_col: str = "layer") -> tuple[float, float]:
    """(best-layer, last-layer) score from a per-layer table. best = max over
    layers; last = score at the highest layer index. When there is no usable layer
    axis, both fall back to the last non-null value (best == last)."""
    s = pd.to_numeric(df.get(score_col), errors="coerce")
    if s is None or not s.notna().any():
        return float("nan"), float("nan")
    if layer_col not in df.columns:
        v = s.dropna()
        return float(v.max()), float(v.iloc[-1])
    lay = pd.to_numeric(df[layer_col], errors="coerce")
    best = float(s.max())
    valid = lay.notna() & s.notna()
    if not valid.any():
        return best, best
    last_layer = lay[valid].max()
    last = float(s[valid & (lay == last_layer)].mean())
    return best, last


# ---------------------------------------------------------------------------
# Standardized scoring convention
# ---------------------------------------------------------------------------
# One rule for every dataset, recomputed post-hoc from the per-unit/per-lag/
# per-layer detail CSVs the sweeps already wrote (no jobs are re-run):
#
#   metric       : Pearson rho            (LPP is the exception - it only ever
#                                           computed Pearson; to be re-run later)
#   window       : -0.2 -> 0.8 s           (lag-based datasets only: podcast. Others have no lags. This is a
#                                           POST-HOC sub-window applied here regardless
#                                           of each dataset's own raw epoch extraction
#                                           range -- podcast's raw epochs are now
#                                           extracted over only [-1, 1] s with no
#                                           baseline correction (z-scored per electrode
#                                           on train-fold stats instead, see
#                                           configs/embedding/neural/podcast.yaml),
#                                           down from the older [-2, 2] s + baseline
#                                           convention; the [-0.2, 0.8] scoring
#                                           sub-window is still a valid subset and is
#                                           kept for comparability across lag-based datasets.)
#   layer        : GLOBAL best layer       (one layer per model, chosen on the
#                                           standardized score - a form of
#                                           normalisation across models)
#   aggregation  : per-unit rho -> mean WITHIN subject -> mean ACROSS subjects
#                  (each SUBJECT is one observation, so a patient with 200
#                   electrodes does not outweigh one with 20)
#
# The legacy headline score is kept alongside as ``score_legacy`` so the shift
# introduced by standardising is visible rather than silent.
STD_WINDOW: tuple[float, float] = (-0.2, 0.8)

# ZuCo channel 104 (last of 105) is a degenerate/non-brain channel: Pearson is
# exactly 1.0 in every row. Excluded from both the score and the noise ceiling.
ZUCO_BAD_ELECTRODE: int = 104


def _in_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["time_s"] >= STD_WINDOW[0]) & (df["time_s"] <= STD_WINDOW[1])]


def _std_from_layerwise_timecourse(base: Path, fname: str, metric: str = "spearmanr_mean") -> float | None:
    """Full convention from one file (subject x layer x time_s x <metric>).

    ``pearsonr_mean``/``spearmanr_mean`` is already the mean over that
    subject's channels, so: mean over lags in window -> mean over subjects ->
    take the best layer. Used by broderick.
    """
    d = _read_csv_safe(base / fname)
    need = {"subject", "layer", "time_s", metric}
    if d is None or not need <= set(d.columns):
        return None
    w = _in_window(d)
    if w.empty:
        return None
    per_subject_layer = w.groupby(["layer", "subject"])[metric].mean()
    per_layer = per_subject_layer.groupby("layer").mean()   # subjects weighted equally
    return float(per_layer.max()) if len(per_layer) else None


def _std_from_unit_timecourse(base: Path, fname: str, unit_col: str, metric: str = "spearmanr") -> float | None:
    """Window + subject-first from subject x unit x time_s x <metric>.

    Used by podcast (which has no per-layer timecourse, so the layer selection
    stays as the pipeline's; the window is the important fix there: podcast was
    scored over -2 -> +2 s while every other lag dataset used -0.2 -> 1.0 s).
    """
    d = _read_csv_safe(base / fname)
    need = {"subject", unit_col, "time_s", metric}
    if d is None or not need <= set(d.columns):
        return None
    w = _in_window(d)
    if w.empty:
        return None
    per_unit = w.groupby(["subject", unit_col])[metric].mean()   # over lags
    per_subject = per_unit.groupby("subject").mean()                  # over units
    return float(per_subject.mean())                                  # over subjects


def _std_from_by_subject(base: Path, fname: str, cols: list[str]) -> float | None:
    """Subject-first aggregation for datasets with no time axis: mean over
    subjects of the per-subject score (mean of ``cols`` if several). ``cols`` is passed in fully-resolved (metric already baked
    into the column names by the caller)."""
    d = _read_csv_safe(base / fname)
    if d is None or "subject" not in d.columns:
        return None
    have = [c for c in cols if c in d.columns]
    if not have:
        return None
    per_subject = d.groupby("subject")[have].mean().mean(axis=1)
    per_subject = per_subject.dropna()
    return float(per_subject.mean()) if len(per_subject) else None


def _std_zuco(base: Path, metric: str = "spearmanr") -> float | None:
    """ZuCo full convention from zuco_checkpoint_layerwise.csv
    (subject x task x band x electrode x layer x <metric>).

    Global best layer + subject-first, over ALL tasks/bands. The legacy score used
    ``best_pearsonr_mean`` = the best layer chosen *per electrode*, which maxes over
    layers on noise and inflated ZuCo above its own noise ceiling (105%).
    """
    d = _read_csv_safe(base / "zuco_checkpoint_layerwise.csv")
    need = {"subject", "layer", metric}
    if d is None or not need <= set(d.columns):
        return None
    # Electrode 104 (the last of the 105 channels) is a degenerate / non-brain
    # channel: its Pearson is exactly 1.000 in EVERY subject/band/layer row, so
    # it is excluded here (and in the noise-ceiling handler) as junk data.
    if "electrode" in d.columns:
        d = d[d["electrode"] != ZUCO_BAD_ELECTRODE]
    per_subject_layer = d.groupby(["layer", "subject"])[metric].mean()
    per_layer = per_subject_layer.groupby("layer").mean()   # subjects weighted equally
    return float(per_layer.max()) if len(per_layer) else None


def _std_or(std: float | None, legacy):
    """Prefer the standardized score; fall back to the legacy headline."""
    return std if std is not None else legacy


def _final_base(score_files: list[Path]) -> Path:
    """Directory of the final checkpoint's score file (where the detail CSVs live)."""
    return score_files[-1].parent


def _finalize_model_df(rows: list[dict], ppl_df: pd.DataFrame | None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[~df["model_id"].isin(EXCLUDED_MODEL_IDS)]
    if df.empty:
        return df
    if ppl_df is not None and not ppl_df.empty:
        df = df.merge(ppl_df, on="model_id", how="left")
    else:
        df["perplexity"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Per-dataset loaders. Each returns a dict of DataFrames:
#   model_df: one row per model (final checkpoint) with columns
#             [model_id, family, group, label, n_params, n_layers, score, perplexity, score_sem?]
#   layer_df: long format [model_id, family, group, layer, layer_frac, score]
#   roi_df:   optional long format [model_id, family, group, roi, score]
#   timecourse_df: optional long format [model_id, family, group, time_s, score]
# ---------------------------------------------------------------------------


def _lpp_flattened_memory_models(sweep_root: Path, scores_filename: str):
    """LPP sweep quirk: the alibi/stickbreaking/forgetting family dirs exist but are
    empty — those 18 models were actually run as flat ``Lanni-ni_<family>_...``
    dirs directly under the sweep root (no family subdir), with a checkpoints/main/
    subdir. See _lpp_flattened_registry_models below for BabyLM/cbr_rnn, which follow the same flat-dir pattern but cannot be matched by a shared name prefix."""
    for model_dir in sorted(sweep_root.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith("Lanni-ni_"):
            continue
        family = next(
            (f for f in MEMORY_CONSTRAINED_FAMILIES if f"_{f}_" in model_dir.name),
            None,
        )
        if family is None:
            continue
        files = sorted(model_dir.glob(f"checkpoints/*/{scores_filename}"))
        if not files:
            direct = model_dir / scores_filename
            files = [direct] if direct.exists() else []
        if files:
            yield family, model_dir, files


def _lpp_flattened_registry_models(sweep_root: Path, scores_filename: str, already_seen: set[str]):
    """LPP sweep quirk, part 2: BabyLM (CLASS-IT/BLM-GPT2s/MoEP/GPT2-Strict/
    GPT-BERT variants) and cbr_rnn/cbr_rnn-m are ALSO run as flat
    ``<org>_<repo>/checkpoints/main/`` dirs directly under the sweep root, same
    as the Lanni-ni Memory-Constrained models -- there is no ``babylm/`` or
    ``cbr_rnn/`` family subdir with contents (those exist but are empty, same
    quirk as the Memory-Constrained ones). Unlike the Lanni-ni case, these
    dir names don't share one common prefix/pattern, so family is resolved by
    looking the reconstructed model_id up in the model registry directly
    (robust to naming, not name-pattern-matching). ``already_seen`` is the set
    of directory names already yielded by the nested-family walk or the
    Lanni-ni flat walk, so this never double-counts a model.
    """
    for model_dir in sorted(sweep_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name in already_seen or model_dir.name.startswith("Lanni-ni_"):
            continue
        model_id = model_dir_to_model_id(model_dir.name)
        meta = REGISTRY_INDEX.get(model_id)
        if meta is None:
            continue
        files = sorted(model_dir.glob(f"checkpoints/*/{scores_filename}"))
        if not files:
            direct = model_dir / scores_filename
            files = [direct] if direct.exists() else []
        if files:
            yield meta.family, model_dir, files


def load_lpp_fmri() -> dict[str, pd.DataFrame]:
    sweep_root = OUTPUTS_JEANZAY / "lpp/embedding_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "lpp/perplexity")

    model_rows, layer_rows, roi_rows = [], [], []
    nested = list(_iter_model_score_files(sweep_root, "lpp_checkpoint_scores.csv"))
    lanni = list(_lpp_flattened_memory_models(sweep_root, "lpp_checkpoint_scores.csv"))
    seen_dirs = {d.name for _, d, _ in nested} | {d.name for _, d, _ in lanni}
    extra = list(_lpp_flattened_registry_models(sweep_root, "lpp_checkpoint_scores.csv", seen_dirs))
    lpp_iter = nested + lanni + extra
    for family, model_dir, score_files in lpp_iter:
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        final = _pick_final_row(scores)
        meta = lookup_model_meta(
            model_id, family,
            n_params_fallback=final.get("n_params"),
            n_layers_fallback=final.get("n_layers"),
            label_fallback=final.get("model_label"),
        )
        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": final.get("mean_pearson"),
        })

        for f in score_files:
            base = f.parent
            lw = _read_csv_safe(base / "lpp_checkpoint_layerwise.csv")
            if lw is not None and "layer" in lw.columns and "mean_pearson" in lw.columns:
                n_layers = meta.n_layers or lw["layer"].max() + 1
                for _, r in lw.iterrows():
                    layer_rows.append({
                        "model_id": model_id, "family": family, "group": meta.group,
                        "layer": r["layer"], "layer_frac": r["layer"] / max(1, n_layers - 1),
                        "score": r.get("mean_pearson"),
                    })
            roi = _read_csv_safe(base / "lpp_checkpoint_by_roi.csv")
            if roi is not None and "roi" in roi.columns and "mean_pearson" in roi.columns:
                for _, r in roi.iterrows():
                    roi_rows.append({
                        "model_id": model_id, "family": family, "group": meta.group,
                        "roi": r["roi"], "score": r.get("mean_pearson"),
                    })

    return {
        "model_df": _finalize_model_df(model_rows, ppl_df),
        "layer_df": pd.DataFrame(layer_rows),
        "roi_df": pd.DataFrame(roi_rows),
    }


def load_naturalstories_spr() -> dict[str, pd.DataFrame]:
    sweep_root = OUTPUTS_JEANZAY / "naturalstories/surprisal_sweep"
    return _load_lme_reading(
        sweep_root, "spr_lme_checkpoint_scores.csv",
        OUTPUTS_JEANZAY / "naturalstories/perplexity",
    )


def _load_lme_reading(sweep_root: Path, scores_filename: str, ppl_dir: Path) -> dict:
    """Shared loader for LME reading-time (behavioral) datasets — Natural
    Stories SPR, GECO, MECO, Provo, UCL-reading, ZuCo Reading, OneStop. The
    headline score is **per-observation delta log-likelihood**
    (delta_loglik_test / n_obs_test, held out) at the model's best layer (max
    delta_loglik_test) -- ceiling-normalised downstream via ceiling_for() /
    dll_ceiling_for() against each dataset's own delta-loglik noise ceiling
    (Timkey et al. 2026 split-half method; see
    outputs/noise_ceilings/delta_loglik_ceiling_<tag>_oos.json). The previous
    held-out delta-Pearson-r headline is kept as ``score_legacy``."""
    ppl_df = load_perplexity_dir(ppl_dir)
    model_rows = []
    # Behavioral LME: the <topdir>_dlt sibling is the canonical multilayer re-run.
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, scores_filename, _prefer_dlt=True):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        if "converged" in scores.columns:
            converged = scores[scores["converged"] == True]  # noqa: E712
            if not converged.empty:
                scores = converged
        final = _pick_final_row(scores)
        # Restrict to the final checkpoint, then take best/last over LAYERS (the
        # multilayer LME writes one row per layer).
        key = "tokens_seen" if "tokens_seen" in scores.columns else ("step" if "step" in scores.columns else None)
        if key is not None and pd.notna(final.get(key)):
            scores = scores[scores[key] == final[key]]
        meta = lookup_model_meta(model_id, family, n_params_fallback=final.get("n_params"))
        # Primary score = held-out per-observation delta log-likelihood at the
        # best layer (delta_loglik_test / n_obs_test) -- see docstring. A model
        # with no usable delta_loglik_test/n_obs_test is excluded rather than
        # scored on a different metric.
        best, last = _best_last_dll_per_obs(scores, "layer")
        if best != best:  # NaN
            continue
        # legacy: held-out delta Pearson r (this pipeline's earlier convention).
        legacy_col = next((c for c in ("test_delta_pearsonr", "surprisal_delta_pearsonr")
                           if c in scores.columns and pd.to_numeric(scores[c], errors="coerce").notna().any()), None)
        legacy_best = float("nan")
        if legacy_col:
            leg_scores = _sanitize_delta_col(scores, legacy_col)
            legacy_best = _best_last_over_layers(leg_scores, legacy_col, "layer")[0]
        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": best, "score_best": best, "score_last": last,
            "score_legacy": legacy_best,
        })
    return {
        "model_df": _finalize_model_df(model_rows, ppl_df),
        "layer_df": pd.DataFrame(),
        "roi_df": pd.DataFrame(),
    }


def load_geco() -> dict[str, pd.DataFrame]:
    # geco/surprisal_sweep/fp is the canonical multilayer re-run
    # (100 models, has n_obs_test) -- same "<topdir>_dlt/fp" convention as
    # meco/provo below, just pointed at directly since geco's own non-dlt
    # sibling doesn't exist any more (nothing for _prefer_dlt to compare
    # against). geco/lme_sweep_dlt (previously used here) is a stale,
    # differently-named directory with NO n_obs_test column -- do not use it.
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "geco/surprisal_sweep" / "fp",
        "geco_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "geco/perplexity",
    )


def load_meco() -> dict[str, pd.DataFrame]:
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "meco/surprisal_sweep" / "fp",
        "meco_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "meco/perplexity",
    )


def load_provo() -> dict[str, pd.DataFrame]:
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "provo/surprisal_sweep" / "fp",
        "provo_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "provo/perplexity",
    )


def load_ucl_reading() -> dict[str, pd.DataFrame]:
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "ucl_reading/surprisal_sweep",
        "ucl_reading_spr_lme_checkpoint_scores.csv",
        OUTPUTS_JEANZAY / "ucl_reading/perplexity",
    )


def load_zuco_reading() -> dict[str, pd.DataFrame]:
    # ZuCo natural-reading total-reading-time LME (measure dir "nr"). Pointed
    # directly at the "_dlt_oos" (out-of-sample) variant -- the canonical one
    # -- rather than the auto-resolved "_dlt" sibling _prefer_dlt would find
    # (zuco_reading/checkpoint_sweep_lme_dlt/nr also exists and also has
    # n_obs_test, but is superseded).
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "zuco_reading/surprisal_sweep" / "nr",
        "zuco_reading_trt_lme_checkpoint_scores.csv",
        OUTPUTS_JEANZAY / "zuco_reading/perplexity",
    )


def load_zuco_reading_sr() -> dict[str, pd.DataFrame]:
    # SR (sentiment reading) -- data landed 2026-09-08, same convention as NR
    # above (same "_dlt_oos" canonical root, same measure-suffixed filename).
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "zuco_reading/surprisal_sweep" / "sr",
        "zuco_reading_trt_lme_checkpoint_scores.csv",
        OUTPUTS_JEANZAY / "zuco_reading/perplexity",
    )


def load_zuco_reading_tsr() -> dict[str, pd.DataFrame]:
    # TSR (task-specific reading) -- data landed 2026-09-08, same convention.
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "zuco_reading/surprisal_sweep" / "tsr",
        "zuco_reading_trt_lme_checkpoint_scores.csv",
        OUTPUTS_JEANZAY / "zuco_reading/perplexity",
    )


def load_onestop() -> dict[str, pd.DataFrame]:
    # OneStop eye-tracking (first-pass RT, measure dir "fp") -- new dataset,
    # same convention as geco/meco/provo.
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "onestop/surprisal_sweep" / "fp",
        "onestop_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "onestop/perplexity",
    )


# ---------------------------------------------------------------------------
# GECO/MECO/Provo/OneStop's GP (go-past) and MECO/Provo/OneStop's SP
# (scan-path/total dwell) measures -- 2026-09-07, added so this report's
# "Behavioral" bucket has the same measure coverage as the matched-scale
# analysis (analysis/matched_scale_master_table.py). Same "<topdir>_dlt/<measure>"
# root, same scores_filename, as each dataset's FP loader above -- only the
# measure subdir changes. Perplexity is a property of the corpus text, not
# of which eye-tracking measure was recorded, so each reuses its FP sibling's
# perplexity dir.
# ---------------------------------------------------------------------------

def load_geco_gp() -> dict[str, pd.DataFrame]:
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "geco/surprisal_sweep" / "gp",
        "geco_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "geco/perplexity",
    )


def load_meco_gp() -> dict[str, pd.DataFrame]:
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "meco/surprisal_sweep" / "gp",
        "meco_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "meco/perplexity",
    )


def load_meco_sp() -> dict[str, pd.DataFrame]:
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "meco/surprisal_sweep" / "sp",
        "meco_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "meco/perplexity",
    )


def load_provo_gp() -> dict[str, pd.DataFrame]:
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "provo/surprisal_sweep" / "gp",
        "provo_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "provo/perplexity",
    )


def load_provo_sp() -> dict[str, pd.DataFrame]:
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "provo/surprisal_sweep" / "sp",
        "provo_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "provo/perplexity",
    )


def load_onestop_gp() -> dict[str, pd.DataFrame]:
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "onestop/surprisal_sweep" / "gp",
        "onestop_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "onestop/perplexity",
    )


def load_onestop_sp() -> dict[str, pd.DataFrame]:
    return _load_lme_reading(
        OUTPUTS_JEANZAY / "onestop/surprisal_sweep" / "sp",
        "onestop_lme_checkpoint_scores.csv", OUTPUTS_JEANZAY / "onestop/perplexity",
    )


def load_pereira_embed() -> dict[str, pd.DataFrame]:
    """Pereira fMRI, embedding/ridge track -- outputs_jeanzay/pereira/embedding_sweep/
    (NOT empty, despite load_pereira_fmri's docstring above claiming so --
    that was true at some earlier point and never updated; confirmed present
    with ~97 main models as of 2026-09-07, see
    analysis/matched_scale_master_table_embed.py's _load_pereira_fmri, which
    this mirrors). One row per (layer, subject, revision) --
    pearsonr_mean is already the per-layer, per-subject mean-over-voxels
    Pearson r; average across subjects within each layer FIRST, then take
    the best layer, matching the ROI-averaging convention every other neural
    embedding dataset already uses."""
    sweep_root = OUTPUTS_JEANZAY / "pereira/embedding_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "pereira/perplexity")
    rows = []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "pereira_embed_by_layer.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        if "layer" not in df.columns or "pearsonr_mean" not in df.columns:
            continue
        if "revision" in df.columns:
            final = _pick_final_row(df)
            key = "tokens_seen" if "tokens_seen" in df.columns else ("step" if "step" in df.columns else None)
            if key is not None and pd.notna(final.get(key)):
                df = df[df[key] == final[key]]
        per_layer = df.groupby("layer")["pearsonr_mean"].mean()
        if per_layer.empty:
            continue
        best = float(per_layer.max())
        last = float(per_layer.loc[per_layer.index.max()])
        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        rows.append({
            "model_id": model_id, "family": meta.family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": best, "score_best": best, "score_last": last,
            "score_legacy": float("nan"),
        })
    return {"model_df": _finalize_model_df(rows, ppl_df), "layer_df": pd.DataFrame(), "roi_df": pd.DataFrame()}


def load_pereira_fmri() -> dict[str, pd.DataFrame]:
    """Pereira fMRI, surprisal/LME track. Its embedding-track sibling is
    ``pereira_fmri__embed`` (load_pereira_embed, above).

    Normalized DLL: reads the raw per-model per-layer files directly
    (outputs_jeanzay/pereira/surprisal_sweep/<family>/<model>/pereira_lme_by_layer.csv
    -- no pre-built outputs/pereira_lme_summary/ exists, unlike
    lpp_fmri__surprisal/tuckute_fmri__surprisal/etc.), picking the best layer
    by delta_loglik_test and scoring delta_loglik_test/n_obs_test there --
    same convention as every other surprisal-track dataset (see
    _best_last_dll_per_obs), mirroring _load_podcast_surprisal_full's
    directly-from-raw-files shape."""
    sweep_root = OUTPUTS_JEANZAY / "pereira/surprisal_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "pereira/perplexity")
    rows = []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "pereira_lme_by_layer.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        best, last = _best_last_dll_per_obs(scores, "layer")
        if best != best:  # NaN
            continue
        legacy = float("nan")
        if "test_delta_pearsonr" in scores.columns:
            leg = _sanitize_delta_col(scores, "test_delta_pearsonr")
            s = pd.to_numeric(leg["test_delta_pearsonr"], errors="coerce")
            if s.notna().sum():
                legacy = float(s.max())
        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        rows.append({
            "model_id": model_id, "family": meta.family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": best, "score_best": best, "score_last": last,
            "score_legacy": legacy,
        })
    return {"model_df": _finalize_model_df(rows, ppl_df), "layer_df": pd.DataFrame(), "roi_df": pd.DataFrame()}


def load_podcast_ieeg() -> dict[str, pd.DataFrame]:
    # podcast/checkpoint_sweep (no suffix) is the old pre-rerun location and no
    # longer exists; podcast/embedding_sweep is the current canonical dir
    # (window now [-1, 1] s, no baseline correction -- see configs/embedding/neural/podcast.yaml).
    sweep_root = OUTPUTS_JEANZAY / "podcast/embedding_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "podcast/perplexity")

    model_rows, layer_rows, roi_rows, tc_rows = [], [], [], []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "podcast_checkpoint_scores.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        final = _pick_final_row(scores)
        meta = lookup_model_meta(model_id, family, n_params_fallback=final.get("n_params"))
        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": _std_or(_std_from_unit_timecourse(
                _final_base(score_files), "podcast_electrode_timecourse.csv", "electrode"),
                final.get("mean_best_spearmanr")),
            "score_legacy": final.get("mean_best_pearsonr"),
        })

        base = model_dir if (model_dir / "podcast_checkpoint_scores.csv").exists() else score_files[-1].parent
        lw = _read_csv_safe(base / "podcast_checkpoint_layerwise.csv")
        if lw is not None and "layer" in lw.columns:
            n_layers = meta.n_layers or (lw["layer"].max() + 1)
            for _, r in lw.iterrows():
                layer_frac = r["layer_rel"] if "layer_rel" in lw.columns and pd.notna(r.get("layer_rel")) else r["layer"] / max(1, n_layers - 1)
                layer_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "layer": r["layer"], "layer_frac": layer_frac, "score": r.get("mean_pearsonr"),
                })
        region = _read_csv_safe(base / "podcast_by_region.csv")
        if region is not None and "region" in region.columns:
            col = "best_pearsonr_mean" if "best_pearsonr_mean" in region.columns else None
            if col:
                for _, r in region.iterrows():
                    roi_rows.append({
                        "model_id": model_id, "family": family, "group": meta.group,
                        "roi": r["region"], "score": r.get(col),
                    })
        tc = _read_csv_safe(base / "podcast_electrode_timecourse.csv")
        if tc is not None and "time_s" in tc.columns and "pearsonr" in tc.columns:
            agg = tc.groupby("time_s", as_index=False)["pearsonr"].mean()
            for _, r in agg.iterrows():
                tc_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "time_s": r["time_s"], "score": r["pearsonr"],
                })

    return {
        "model_df": _finalize_model_df(model_rows, ppl_df),
        "layer_df": pd.DataFrame(layer_rows),
        "roi_df": pd.DataFrame(roi_rows),
        "timecourse_df": pd.DataFrame(tc_rows),
    }


def load_nieuwland_eeg() -> dict[str, pd.DataFrame]:
    # nieuwland/checkpoint_sweep (no suffix) is the old pre-rerun location and
    # no longer exists; _dlt is the current canonical dir.
    sweep_root = OUTPUTS_JEANZAY / "nieuwland/embedding_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "nieuwland/perplexity")

    model_rows, layer_rows, roi_rows = [], [], []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "nieuwland_checkpoint_scores.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        final = _pick_final_row(scores)
        meta = lookup_model_meta(model_id, family, n_params_fallback=final.get("n_params"), n_layers_fallback=final.get("n_layers"))
        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": _std_or(_std_from_by_subject(
                _final_base(score_files), "nieuwland_by_subject.csv", ["perchannel_spearmanr"]),
                final.get("mean_subject_perchannel_spearmanr")),
            "score_legacy": final.get("mean_subject_perchannel_pearsonr"),
        })

        base = model_dir if (model_dir / "nieuwland_checkpoint_scores.csv").exists() else score_files[-1].parent
        lw = _read_csv_safe(base / "nieuwland_checkpoint_layerwise.csv")
        if lw is not None and "layer" in lw.columns:
            n_layers = meta.n_layers or (lw["layer"].max() + 1)
            score_col = "pc_sr" if "pc_sr" in lw.columns else "mean_pearsonr"
            for _, r in lw.iterrows():
                layer_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "layer": r["layer"], "layer_frac": r["layer"] / max(1, n_layers - 1), "score": r.get(score_col),
                })
        chan = _read_csv_safe(base / "nieuwland_by_channel.csv")
        if chan is not None and "channel" in chan.columns:
            col = "pearsonr" if "pearsonr" in chan.columns else None
            if col:
                for _, r in chan.iterrows():
                    roi_rows.append({
                        "model_id": model_id, "family": family, "group": meta.group,
                        "roi": r["channel"], "score": r.get(col),
                    })

    return {
        "model_df": _finalize_model_df(model_rows, ppl_df),
        "layer_df": pd.DataFrame(layer_rows),
        "roi_df": pd.DataFrame(roi_rows),
    }


def load_szewczyk_eeg() -> dict[str, pd.DataFrame]:
    # One ROI-averaged N400 scalar per (subject, trial): no electrode/ROI breakdown.
    sweep_root = OUTPUTS_JEANZAY / "szewczyk/embedding_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "szewczyk_perplexity")

    model_rows, layer_rows = [], []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "szewczyk_checkpoint_scores.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        final = _pick_final_row(scores)
        meta = lookup_model_meta(model_id, family, n_params_fallback=final.get("n_params"), n_layers_fallback=final.get("n_layers"))
        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": _std_or(_std_from_by_subject(
                _final_base(score_files), "szewczyk_by_subject.csv", ["mean_spearmanr"]),
                final.get("mean_spearmanr")),
            "score_legacy": final.get("mean_pearsonr"),
            "score_sem": final.get("sem_spearmanr"),
        })

        base = model_dir if (model_dir / "szewczyk_checkpoint_scores.csv").exists() else score_files[-1].parent
        lw = _read_csv_safe(base / "szewczyk_checkpoint_layerwise.csv")
        if lw is not None and "layer" in lw.columns:
            n_layers = meta.n_layers or (lw["layer"].max() + 1)
            for _, r in lw.iterrows():
                layer_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "layer": r["layer"], "layer_frac": r["layer"] / max(1, n_layers - 1), "score": r.get("mean_pearsonr"),
                })

    return {
        "model_df": _finalize_model_df(model_rows, ppl_df),
        "layer_df": pd.DataFrame(layer_rows),
        "roi_df": pd.DataFrame(),
    }


def load_megmasc_meg() -> dict[str, pd.DataFrame]:
    """Gwilliams MEG-MASC, embedding/encoding track only -- no delta-loglik
    noise ceiling exists yet for its surprisal/LME track, so that track is
    not registered. New "MEG" modality (distinct sensor physics from EEG).
    megmasc's per-model output files use the standard lag-based
    column/filename shapes."""
    sweep_root = OUTPUTS_JEANZAY / "megmasc/embedding_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "megmasc/perplexity")

    model_rows, layer_rows, tc_rows = [], [], []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "megmasc_checkpoint_scores.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        final = _pick_final_row(scores)
        meta = lookup_model_meta(model_id, family, n_params_fallback=final.get("n_params"))
        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": _std_or(_std_from_layerwise_timecourse(
                _final_base(score_files), "megmasc_layerwise_timecourse.csv"),
                final.get("mean_best_spearmanr")),
            "score_legacy": final.get("mean_best_pearsonr"),
        })

        base = model_dir if (model_dir / "megmasc_checkpoint_scores.csv").exists() else score_files[-1].parent
        lw = _read_csv_safe(base / "megmasc_checkpoint_layerwise.csv")
        if lw is not None and "layer" in lw.columns:
            n_layers = meta.n_layers or (lw["layer"].max() + 1)
            for _, r in lw.iterrows():
                layer_frac = r["layer_rel"] if "layer_rel" in lw.columns and pd.notna(r.get("layer_rel")) else r["layer"] / max(1, n_layers - 1)
                layer_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "layer": r["layer"], "layer_frac": layer_frac, "score": r.get("mean_pearsonr"),
                })
        tc = _read_csv_safe(base / "megmasc_mean_timecourse.csv")
        if tc is not None and "time_s" in tc.columns and "pearsonr" in tc.columns:
            for _, r in tc.iterrows():
                tc_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "time_s": r["time_s"], "score": r["pearsonr"],
                })

    return {
        "model_df": _finalize_model_df(model_rows, ppl_df),
        "layer_df": pd.DataFrame(layer_rows),
        "roi_df": pd.DataFrame(),
        "timecourse_df": pd.DataFrame(tc_rows),
    }


def load_broderick_eeg() -> dict[str, pd.DataFrame]:
    sweep_root = OUTPUTS_JEANZAY / "broderick_checkpoint_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "broderick/perplexity")

    model_rows, layer_rows, roi_rows, tc_rows = [], [], [], []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "broderick_checkpoint_scores.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        final = _pick_final_row(scores)
        meta = lookup_model_meta(model_id, family, n_params_fallback=final.get("n_params"))
        n_subjects = final.get("n_subjects")
        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": _std_or(_std_from_layerwise_timecourse(
                _final_base(score_files), "broderick_layerwise_timecourse.csv"),
                final.get("mean_best_spearmanr")),
            "score_legacy": final.get("mean_best_pearsonr"),
            "n_subjects": n_subjects,
            "n_subjects_anomalous": bool(pd.notna(n_subjects) and n_subjects < 19),
        })

        base = model_dir if (model_dir / "broderick_checkpoint_scores.csv").exists() else score_files[-1].parent
        lw = _read_csv_safe(base / "broderick_checkpoint_layerwise.csv")
        if lw is not None and "layer" in lw.columns:
            n_layers = meta.n_layers or (lw["layer"].max() + 1)
            for _, r in lw.iterrows():
                layer_frac = r["layer_rel"] if "layer_rel" in lw.columns and pd.notna(r.get("layer_rel")) else r["layer"] / max(1, n_layers - 1)
                layer_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "layer": r["layer"], "layer_frac": layer_frac, "score": r.get("mean_pearsonr"),
                })
        region = _read_csv_safe(base / "broderick_by_region.csv")
        if region is not None and "region" in region.columns and "best_pearsonr_mean" in region.columns:
            for _, r in region.iterrows():
                roi_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "roi": r["region"], "score": r.get("best_pearsonr_mean"),
                })
        tc = _read_csv_safe(base / "broderick_mean_timecourse.csv")
        if tc is not None and "time_s" in tc.columns and "pearsonr" in tc.columns:
            for _, r in tc.iterrows():
                tc_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "time_s": r["time_s"], "score": r["pearsonr"],
                })

    return {
        "model_df": _finalize_model_df(model_rows, ppl_df),
        "layer_df": pd.DataFrame(layer_rows),
        "roi_df": pd.DataFrame(roi_rows),
        "timecourse_df": pd.DataFrame(tc_rows),
    }


def load_ns_fmri() -> dict[str, pd.DataFrame]:
    sweep_root = OUTPUTS_JEANZAY / "ns_fmri/embedding_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "naturalstories/perplexity")

    model_rows, layer_rows, roi_rows = [], [], []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "ns_fmri_checkpoint_scores.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        meta = lookup_model_meta(model_id, family)
        if frames:
            scores = pd.concat(frames, ignore_index=True)
            final = _pick_final_row(scores)
            score = final.get("mean_best_spearmanr")
            score_legacy = final.get("mean_best_pearsonr")
        else:
            # fall back to by_roi.csv when checkpoint_scores.csv is missing/malformed
            base_dir = model_dir
            roi_csv = _read_csv_safe(base_dir / "ns_fmri_checkpoint_by_roi.csv") or _read_csv_safe(base_dir / "ns_fmri_by_roi.csv")
            if roi_csv is None or "best_spearmanr" not in roi_csv.columns:
                continue
            score = roi_csv["best_spearmanr"].mean()
            score_legacy = roi_csv["best_pearsonr"].mean() if "best_pearsonr" in roi_csv.columns else float("nan")

        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": _std_or(_std_from_by_subject(
                _final_base(score_files), "ns_fmri_by_subject.csv", ["best_spearmanr_mean"]),
                score),
            "score_legacy": score_legacy,
        })

        base = score_files[-1].parent if frames else model_dir
        lw = _read_csv_safe(base / "ns_fmri_checkpoint_layerwise.csv")
        if lw is not None and "layer" in lw.columns:
            lw = lw.copy()
            lw["layer"] = pd.to_numeric(lw["layer"], errors="coerce")
            lw = lw.dropna(subset=["layer"])
            n_layers = meta.n_layers or (lw["layer"].max() + 1)
            value_col = "pearsonr" if "pearsonr" in lw.columns else None
            if value_col:
                agg = lw.groupby("layer", as_index=False)[value_col].mean()
                for _, r in agg.iterrows():
                    layer_rows.append({
                        "model_id": model_id, "family": family, "group": meta.group,
                        "layer": r["layer"], "layer_frac": r["layer"] / max(1, n_layers - 1), "score": r[value_col],
                    })
        roi = _read_csv_safe(base / "ns_fmri_checkpoint_by_roi.csv") or _read_csv_safe(base / "ns_fmri_by_roi.csv")
        if roi is not None and "froi" in roi.columns and "best_pearsonr" in roi.columns:
            agg = roi.groupby("froi", as_index=False)["best_pearsonr"].mean()
            for _, r in agg.iterrows():
                roi_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "roi": r["froi"], "score": r["best_pearsonr"],
                })

    return {
        "model_df": _finalize_model_df(model_rows, ppl_df),
        "layer_df": pd.DataFrame(layer_rows),
        "roi_df": pd.DataFrame(roi_rows),
    }


def load_fedorenko2016_ecog() -> dict[str, pd.DataFrame]:
    sweep_root = OUTPUTS_JEANZAY / "fedorenko2016/embedding_sweep"

    model_rows, layer_rows = [], []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "fedorenko2016_checkpoint_scores.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        final = _pick_final_row(scores)
        meta = lookup_model_meta(model_id, family, n_params_fallback=final.get("n_params"))
        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": _std_or(_std_from_by_subject(
                _final_base(score_files), "fedorenko2016_by_subject.csv", ["best_spearmanr_mean"]),
                final.get("mean_best_spearmanr")),
            "score_legacy": final.get("mean_best_pearsonr"),
            "perplexity": final.get("perplexity"),
        })

        base = model_dir if (model_dir / "fedorenko2016_checkpoint_scores.csv").exists() else score_files[-1].parent
        lw = _read_csv_safe(base / "fedorenko2016_checkpoint_layerwise.csv")
        if lw is not None and "layer" in lw.columns:
            n_layers = meta.n_layers or (lw["layer"].max() + 1)
            for _, r in lw.iterrows():
                layer_frac = r["layer_rel"] if "layer_rel" in lw.columns and pd.notna(r.get("layer_rel")) else r["layer"] / max(1, n_layers - 1)
                layer_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "layer": r["layer"], "layer_frac": layer_frac, "score": r.get("mean_pearsonr"),
                })

    # NOTE: does not go through _finalize_model_df (perplexity is already
    # embedded per-row above, and that helper would overwrite it with NaN when
    # passed ppl_df=None) -- but the EXCLUDED_MODEL_IDS filter it normally
    # applies still needs to happen here explicitly.
    model_df = pd.DataFrame(model_rows)
    if not model_df.empty:
        model_df = model_df[~model_df["model_id"].isin(EXCLUDED_MODEL_IDS)]
    return {
        "model_df": model_df,
        "layer_df": pd.DataFrame(layer_rows),
        "roi_df": pd.DataFrame(),
    }


def load_tuckute_fmri() -> dict[str, pd.DataFrame]:
    sweep_root = OUTPUTS_JEANZAY / "tuckute/embedding_sweep"
    conditions = ("B", "D", "S")

    model_rows, layer_rows, roi_rows = [], [], []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "tuckute_scores.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        final = _pick_final_row(scores)
        meta = lookup_model_meta(model_id, family, n_params_fallback=final.get("n_params"))
        cond_scores = [final.get(f"{c}_mean_best_spearmanr") for c in conditions]
        cond_scores = [c for c in cond_scores if pd.notna(c)]
        primary = float(np.mean(cond_scores)) if cond_scores else np.nan
        cond_scores_legacy = [final.get(f"{c}_mean_best_pearsonr") for c in conditions]
        cond_scores_legacy = [c for c in cond_scores_legacy if pd.notna(c)]
        primary_legacy = float(np.mean(cond_scores_legacy)) if cond_scores_legacy else np.nan

        ppl = None
        base = model_dir if (model_dir / "tuckute_scores.csv").exists() else score_files[-1].parent
        ppl_json = base / "perplexity.json"
        if ppl_json.exists():
            import json
            try:
                data = json.loads(ppl_json.read_text())
                if data.get("corpus") in ("tuckute", "", None):
                    ppl = data.get("perplexity")
            except (json.JSONDecodeError, OSError):
                pass

        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": primary, "score_legacy": primary_legacy, "perplexity": ppl,
            "score_B": final.get("B_mean_best_spearmanr"),
            "score_D": final.get("D_mean_best_spearmanr"),
            "score_S": final.get("S_mean_best_spearmanr"),
        })

        lw = _read_csv_safe(base / "tuckute_layerwise.csv")
        if lw is not None and "layer" in lw.columns:
            n_layers = meta.n_layers or (lw["layer"].max() + 1)
            value_col = "pearsonr_mean" if "pearsonr_mean" in lw.columns else "pearsonr"
            if value_col in lw.columns:
                agg = lw.groupby("layer", as_index=False)[value_col].mean()
                for _, r in agg.iterrows():
                    layer_rows.append({
                        "model_id": model_id, "family": family, "group": meta.group,
                        "layer": r["layer"], "layer_frac": r["layer"] / max(1, n_layers - 1), "score": r[value_col],
                    })
        roi = _read_csv_safe(base / "tuckute_by_roi.csv")
        if roi is not None and "roi" in roi.columns and "pearsonr" in roi.columns:
            best = roi.sort_values("pearsonr").groupby(["roi"], as_index=False).tail(1) if "layer" in roi.columns else roi
            agg = best.groupby("roi", as_index=False)["pearsonr"].mean()
            for _, r in agg.iterrows():
                roi_rows.append({
                    "model_id": model_id, "family": family, "group": meta.group,
                    "roi": r["roi"], "score": r["pearsonr"],
                })

    model_df = pd.DataFrame(model_rows)
    if not model_df.empty:
        model_df = model_df[~model_df["model_id"].isin(EXCLUDED_MODEL_IDS)]
    return {
        "model_df": model_df,
        "layer_df": pd.DataFrame(layer_rows),
        "roi_df": pd.DataFrame(roi_rows),
    }


# ---------------------------------------------------------------------------
# Secondary feature-mode loaders (the "doubling"): behavioral+embedding and
# neural+surprisal.  Each keeps its NATIVE metric (per the design decision):
#   behavioral+embedding -> best-layer Pearson r (per-subject then averaged)
#   neural+surprisal     -> Pearson r (best layer);  LPP has Pearson only.
# Spec: slug -> (sweep_root_rel, scores_filename, score_col, pick, ppl_dir_rel)
#   pick="layer_max": file has one row per layer at one checkpoint -> take the
#                     final checkpoint then the max score over layers.
#   pick="final_row": one row per checkpoint -> take the final checkpoint's cell.
# ---------------------------------------------------------------------------
# de Varda, Berzak, Fedorenko & Levy (2026) grand-average-subject methodology
# (average RT across subjects per word -> one ridge CV per layer, no
# per-subject fitting) -- "_average"-suffixed sibling dirs of the originals.
SECONDARY_SPEC: dict[str, tuple[str, str, str, str, str]] = {
    # behavioral + EMBEDDING
    "geco__embed":        ("geco/embedding_sweep/fp_average",  "geco_embed_checkpoint_scores.csv",        "spearmanr", "layer_max", "geco/perplexity"),
    "meco__embed":        ("meco/embedding_sweep/fp_average",  "meco_embed_checkpoint_scores.csv",        "spearmanr", "layer_max", "meco/perplexity"),
    "provo__embed":       ("provo/embedding_sweep/fp_average", "provo_embed_checkpoint_scores.csv",       "spearmanr", "layer_max", "provo/perplexity"),
    # Irregular naming: "_average" is appended to the whole sweep dir here,
    # not to a "fp"-style measure subdir (ucl_reading has no gp/sp measures).
    "ucl_reading__embed": ("ucl_reading/embedding_sweep_average", "ucl_reading_embed_checkpoint_scores.csv", "spearmanr", "layer_max", "ucl_reading/perplexity"),
    "zuco_reading__embed":("zuco_reading/embedding_sweep/nr_average", "zuco_reading_embed_checkpoint_scores.csv", "spearmanr", "layer_max", "zuco_reading/perplexity"),
    "zuco_reading_sr__embed":("zuco_reading/embedding_sweep/sr_average", "zuco_reading_embed_checkpoint_scores.csv", "spearmanr", "layer_max", "zuco_reading/perplexity"),
    "zuco_reading_tsr__embed":("zuco_reading/embedding_sweep/tsr_average", "zuco_reading_embed_checkpoint_scores.csv", "spearmanr", "layer_max", "zuco_reading/perplexity"),
    "onestop__embed":     ("onestop/embedding_sweep/fp_average", "onestop_embed_checkpoint_scores.csv", "spearmanr", "layer_max", "onestop/perplexity"),
    # GP (go-past) / SP (scan-path, total dwell) measures -- added 2026-09-08.
    # Per-subject ridge fits were never run for these measures (only the
    # subject-averaged "_average" tree exists), same caveat as the surprisal-
    # track GP/SP additions.
    "geco_gp__embed":     ("geco/embedding_sweep/gp_average",   "geco_embed_checkpoint_scores.csv",        "spearmanr", "layer_max", "geco/perplexity"),
    "meco_gp__embed":     ("meco/embedding_sweep/gp_average",   "meco_embed_checkpoint_scores.csv",        "spearmanr", "layer_max", "meco/perplexity"),
    "meco_sp__embed":     ("meco/embedding_sweep/sp_average",   "meco_embed_checkpoint_scores.csv",        "spearmanr", "layer_max", "meco/perplexity"),
    "provo_gp__embed":    ("provo/embedding_sweep/gp_average",  "provo_embed_checkpoint_scores.csv",       "spearmanr", "layer_max", "provo/perplexity"),
    "provo_sp__embed":    ("provo/embedding_sweep/sp_average",  "provo_embed_checkpoint_scores.csv",       "spearmanr", "layer_max", "provo/perplexity"),
    "onestop_gp__embed":  ("onestop/embedding_sweep/gp_average","onestop_embed_checkpoint_scores.csv",     "spearmanr", "layer_max", "onestop/perplexity"),
    "onestop_sp__embed":  ("onestop/embedding_sweep/sp_average","onestop_embed_checkpoint_scores.csv",     "spearmanr", "layer_max", "onestop/perplexity"),
}

# Neural + SURPRISAL: pooled-LME-per-layer, read from the already-summarized
# outputs/<ds>_lme_summary CSVs (same delta-log-likelihood metric as behavioral
# surprisal).  slug -> (summary_csv (rel to outputs/), score_col, pick, ppl_rel)
#   pick="best_layer": file has one row per (model, layer) -> per model take the
#                      layer with max score_col.
#   pick="row":        file already has one row per model (best layer / peak lag).
_OUTPUTS = BENCHMARK_SUMMARY_ROOT.parent
# Primary score is now per-observation delta log-likelihood (delta_loglik_test
# / n_obs_test) at the best layer -- ceiling-normalised downstream via
# ceiling_for()/dll_ceiling_for(). "score_col" below is now only the LEGACY
# column (the previous held-out delta-Pearson-r headline), kept as
# score_legacy for reference; it is no longer used to pick the primary score.
NEURAL_SURPRISAL_SPEC: dict[str, tuple[str, str, str, str]] = {
    "nieuwland_eeg__surprisal":     ("nieuwland/lme_summary/all_models_lme_by_layer.csv",     "test_delta_pearsonr", "by_layer", "nieuwland/perplexity"),
    "szewczyk_eeg__surprisal":      ("szewczyk/lme_summary/all_models_lme_by_layer.csv",      "test_delta_pearsonr", "by_layer", "szewczyk_perplexity"),
    "fedorenko2016_ecog__surprisal":("fedorenko2016/lme_summary/all_models_lme_by_layer.csv", "test_delta_pearsonr", "by_layer", "fedorenko2016/perplexity"),
    # "tuckute/perplexity" doesn't exist as a top-level dir; the ppl_rel here
    # is handled specially in _load_neural_surprisal (nested walk of "tuckute").
    "tuckute_fmri__surprisal":      ("tuckute/lme_summary/all_models_lme_by_layer.csv",       "test_delta_pearsonr", "by_layer", "tuckute"),
    "lpp_fmri__surprisal":          ("lpp/lme_summary/all_models_lme_by_layer.csv",           "test_delta_pearsonr", "by_layer", "lpp/perplexity"),
    # ns_fmri: outputs/ns_fmri/lme_summary has no raw all_models_lme_by_layer.csv
    # (only pre-aggregated best/last-layer summaries with no n_obs_test), so
    # this reads straight from the raw per-model per-fROI files (see
    # _load_ns_fmri_surprisal_full), same as podcast below.
    "ns_fmri__surprisal":           (None,                                                     "mean_test_delta_pearsonr",  "ns_fmri_raw",  "ns_fmri/perplexity"),
    "podcast_ieeg__surprisal":      (None,                                                     "test_delta_pearsonr",       "podcast_raw", "podcast/perplexity"),
}


def _load_podcast_surprisal_full() -> dict[str, pd.DataFrame]:
    """podcast_ieeg__surprisal, normalized-DLL variant: reads the raw
    per-model per-lag files directly (small, ~30 rows each) and picks the lag
    with the max delta_loglik_test as the "peak" (matching the best-layer
    convention used everywhere else in this pipeline), then scores
    delta_loglik_test / n_obs_test at that lag."""
    sweep_root = OUTPUTS_JEANZAY / "podcast/surprisal_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "podcast/perplexity")
    rows = []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "podcast_lme_by_lag.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        lags = pd.concat(frames, ignore_index=True)
        peak, _ = _best_last_dll_per_obs(lags, "lag")
        if peak != peak:  # NaN
            continue
        legacy = float("nan")
        if "test_delta_pearsonr" in lags.columns:
            leg = _sanitize_delta_col(lags, "test_delta_pearsonr")
            s = pd.to_numeric(leg["test_delta_pearsonr"], errors="coerce")
            if s.notna().sum():
                legacy = float(s.max())
        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        rows.append({
            "model_id": model_id, "family": meta.family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": peak, "score_best": peak, "score_last": peak,
            "score_legacy": legacy,
        })
    return {"model_df": _finalize_model_df(rows, ppl_df), "layer_df": pd.DataFrame(), "roi_df": pd.DataFrame()}


def _load_ns_fmri_surprisal_full() -> dict[str, pd.DataFrame]:
    """ns_fmri__surprisal, normalized-DLL variant: reads the raw per-model
    per-fROI per-layer files directly (outputs_jeanzay/ns_fmri/surprisal_sweep/
    <family>/<model>/ns_fmri_lme_by_roi.csv -- the pre-aggregated
    outputs/ns_fmri/lme_summary CSVs carry no n_obs_test), summing
    delta_loglik_test and n_obs_test across fROIs (each fROI keeps its own
    best layer) before taking the per-observation ratio -- same convention as
    figures/figure8_scale_matched_comparison.py's _load_dll(group_col="froi")."""
    sweep_root = OUTPUTS_JEANZAY / "ns_fmri/surprisal_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "ns_fmri/perplexity")
    rows = []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "ns_fmri_lme_by_roi.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        if "froi" not in df.columns:
            continue
        per_unit: list[tuple[float, float]] = []
        for _, unit_df in df.groupby("froi"):
            best_dll_n = _best_last_dll_per_obs(unit_df, "layer")
            if best_dll_n[0] != best_dll_n[0]:  # NaN -> this fROI unusable
                continue
            # Recover the raw (delta_loglik_test, n_obs_test) at that fROI's
            # best layer to sum across fROIs before dividing (summing ratios
            # directly would be wrong).
            d = unit_df.copy()
            d["delta_loglik_test"] = pd.to_numeric(d["delta_loglik_test"], errors="coerce")
            d["n_obs_test"] = pd.to_numeric(d["n_obs_test"], errors="coerce")
            valid = d.dropna(subset=["delta_loglik_test", "n_obs_test"])
            valid = valid[valid["n_obs_test"] > 0]
            valid = filter_degenerate_dll_rows(valid)
            if valid.empty:
                continue
            best_row = valid.loc[valid["delta_loglik_test"].idxmax()]
            per_unit.append((float(best_row["delta_loglik_test"]), float(best_row["n_obs_test"])))
        total_dll, total_n = sum_dll_across_units(per_unit)
        if total_n == 0:
            continue
        score = total_dll / total_n
        legacy = float("nan")
        if "test_delta_pearsonr" in df.columns:
            leg = _sanitize_delta_col(df, "test_delta_pearsonr")
            s = pd.to_numeric(leg["test_delta_pearsonr"], errors="coerce")
            if s.notna().sum():
                legacy = float(s.max())
        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        rows.append({
            "model_id": model_id, "family": meta.family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": score, "score_best": score, "score_last": score,
            "score_legacy": legacy,
        })
    return {"model_df": _finalize_model_df(rows, ppl_df), "layer_df": pd.DataFrame(), "roi_df": pd.DataFrame()}


def _load_neural_surprisal(slug: str) -> dict[str, pd.DataFrame]:
    csv_rel, score_col, kind, ppl_rel = NEURAL_SURPRISAL_SPEC[slug]
    if kind == "podcast_raw":
        return _load_podcast_surprisal_full()
    if kind == "ns_fmri_raw":
        return _load_ns_fmri_surprisal_full()
    df = _read_csv_safe(_OUTPUTS / csv_rel)
    if df is None or df.empty or score_col not in df.columns or "model_id" not in df.columns:
        return {"model_df": pd.DataFrame(), "layer_df": pd.DataFrame(), "roi_df": pd.DataFrame()}
    if "converged" in df.columns:
        conv = df[df["converged"] == True]  # noqa: E712
        if not conv.empty:
            df = conv
    df = df.copy()
    if "delta" in score_col:
        df = _sanitize_delta_col(df, score_col)

    # Legacy headline (this pipeline's previous convention): held-out delta
    # Pearson r, i.e. NEURAL_SURPRISAL_SPEC's own "score_col" ("test_delta_pearsonr")
    # -- the primary score below is now per-observation delta log-likelihood,
    # not this column.
    legacy_col = score_col if score_col in df.columns else None
    if legacy_col:
        df = _sanitize_delta_col(df, legacy_col)

    # Per-model {model_id: (best, last)} scores.
    bl: dict[str, tuple[float, float]] = {}
    legacy_best: dict[str, float] = {}
    meta_row: dict[str, pd.Series] = {}
    if kind == "by_layer":
        for mid, g in df.groupby("model_id"):
            bl[mid] = _best_last_dll_per_obs(g, "layer")
            if legacy_col:
                legacy_best[mid] = _best_last_over_layers(g, legacy_col, "layer")[0]
            meta_row[mid] = g.iloc[0]
    elif kind == "ns_pair":
        last_df = _read_csv_safe(_OUTPUTS / csv_rel.replace("best_layer", "last_layer"))
        last_map = {}
        if last_df is not None and score_col in last_df.columns:
            last_map = dict(zip(last_df["model_id"], pd.to_numeric(last_df[score_col], errors="coerce")))
        for _, r in df.drop_duplicates("model_id", keep="last").iterrows():
            mid = r["model_id"]
            best = float(r[score_col]) if pd.notna(r[score_col]) else float("nan")
            bl[mid] = (best, float(last_map.get(mid, best)))
            if legacy_col:
                legacy_best[mid] = float(r[legacy_col]) if pd.notna(r[legacy_col]) else float("nan")
            meta_row[mid] = r
    else:  # single: no layer axis (podcast peak) -> best == last
        for _, r in df.drop_duplicates("model_id", keep="last").iterrows():
            mid = r["model_id"]
            v = float(r[score_col]) if pd.notna(r[score_col]) else float("nan")
            bl[mid] = (v, v)
            if legacy_col:
                legacy_best[mid] = float(r[legacy_col]) if pd.notna(r[legacy_col]) else float("nan")
            meta_row[mid] = r

    # Tuckute has no dedicated <dataset>_perplexity/ dir -- perplexity.json
    # lives nested inside its own checkpoint-sweep tree instead.
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / ppl_rel, nested=(ppl_rel == "tuckute"))
    rows = []
    for mid, (best, last) in bl.items():
        r = meta_row[mid]
        family = r.get("family")
        if not isinstance(family, str) or not family:
            family = model_dir_to_model_id(str(mid)).split("/")[0]
        meta = lookup_model_meta(mid, family, n_params_fallback=r.get("n_params"))
        rows.append({
            "model_id": mid, "family": meta.family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": best, "score_best": best, "score_last": last,
            "score_legacy": legacy_best.get(mid, float("nan")),
        })
    model_df = _finalize_model_df(rows, ppl_df)
    # ns_fmri summary carries its own perplexity column.
    if "perplexity" in df.columns and not model_df.empty:
        pmap = dict(zip(df["model_id"], pd.to_numeric(df["perplexity"], errors="coerce")))
        model_df["perplexity"] = model_df["perplexity"].fillna(model_df["model_id"].map(pmap))
    return {"model_df": model_df, "layer_df": pd.DataFrame(), "roi_df": pd.DataFrame()}


# NOTE: broderick_eeg__surprisal is intentionally not registered: its by-layer LME
# CSVs carry no `converged` flag and contain non-significant blown-up fits that
# a max-over-layers picks up. Re-enable (via NEURAL_SURPRISAL_SPEC) once the
# Broderick LME sweep emits a convergence flag.


def _load_secondary(slug: str) -> dict[str, pd.DataFrame]:
    root_rel, fname, score_col, pick, ppl_rel = SECONDARY_SPEC[slug]
    sweep_root = OUTPUTS_JEANZAY / root_rel
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / ppl_rel)
    rows: list[dict] = []
    iters = list(_iter_model_score_files(sweep_root, fname))
    # LPP-surprisal reuses the flat Lanni-ni memory-model layout.
    if slug == "lpp_fmri__surprisal":
        iters += list(_lpp_flattened_memory_models(sweep_root, fname))
    for family, model_dir, score_files in iters:
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        if score_col not in scores.columns:
            continue
        legacy_col = "pearsonr" if "pearsonr" in scores.columns else None
        if pick == "layer_max":
            final = _pick_final_row(scores)  # restrict to the final checkpoint
            key = "tokens_seen" if "tokens_seen" in scores.columns else ("step" if "step" in scores.columns else None)
            if key is not None and pd.notna(final.get(key)):
                scores = scores[scores[key] == final[key]]
            best, last = _best_last_over_layers(scores, score_col, "layer")
            legacy_best = _best_last_over_layers(scores, legacy_col, "layer")[0] if legacy_col else float("nan")
        else:  # final_row (single row per model)
            final = _pick_final_row(scores)
            best = last = pd.to_numeric(pd.Series([final.get(score_col)]), errors="coerce").iloc[0]
            legacy_best = pd.to_numeric(pd.Series([final.get(legacy_col)]), errors="coerce").iloc[0] if legacy_col else float("nan")
        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": best, "score_best": best, "score_last": last,
            "score_legacy": legacy_best,
        })
    return {"model_df": _finalize_model_df(rows, ppl_df), "layer_df": pd.DataFrame(), "roi_df": pd.DataFrame()}


def load_naturalstories_embed() -> dict[str, pd.DataFrame]:
    """Behavioral + EMBEDDING for Natural Stories SPR (best-layer Pearson r,
    per-subject then averaged) -- same native metric as the other behavioral
    embedding tracks (SECONDARY_SPEC), but naturalstories' embed sweep writes
    a separate per-layer file (spr_checkpoint_layerwise.csv) rather than the
    single long layer x checkpoint CSV SECONDARY_SPEC's "layer_max" pick
    expects, so it needs its own loader."""
    sweep_root = OUTPUTS_JEANZAY / "naturalstories/embedding_sweep"
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / "naturalstories/perplexity")
    model_rows = []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, "spr_checkpoint_layerwise.csv"):
        model_id = _model_id_from_dir(model_dir)
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        if "spearmanr" not in scores.columns:
            continue
        final = _pick_final_row(scores)
        key = "tokens_seen" if "tokens_seen" in scores.columns else ("step" if "step" in scores.columns else None)
        if key is not None and pd.notna(final.get(key)):
            scores = scores[scores[key] == final[key]]
        best, last = _best_last_over_layers(scores, "spearmanr", "layer")
        legacy_best = _best_last_over_layers(scores, "pearsonr", "layer")[0] if "pearsonr" in scores.columns else float("nan")
        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        model_rows.append({
            "model_id": model_id, "family": family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "n_layers": meta.n_layers,
            "score": best, "score_best": best, "score_last": last,
            "score_legacy": legacy_best,
        })
    return {
        "model_df": _finalize_model_df(model_rows, ppl_df),
        "layer_df": pd.DataFrame(),
        "roi_df": pd.DataFrame(),
    }


LOADERS = {
    "lpp_fmri": load_lpp_fmri,
    "naturalstories_spr": load_naturalstories_spr,
    "naturalstories_spr__embed": load_naturalstories_embed,
    "geco": load_geco,
    "meco": load_meco,
    "provo": load_provo,
    "ucl_reading": load_ucl_reading,
    "podcast_ieeg": load_podcast_ieeg,
    "nieuwland_eeg": load_nieuwland_eeg,
    "szewczyk_eeg": load_szewczyk_eeg,
    "broderick_eeg": load_broderick_eeg,
    "ns_fmri": load_ns_fmri,
    "fedorenko2016_ecog": load_fedorenko2016_ecog,
    "tuckute_fmri": load_tuckute_fmri,
    "zuco_reading": load_zuco_reading,
    "zuco_reading_sr": load_zuco_reading_sr,
    "zuco_reading_tsr": load_zuco_reading_tsr,
    "onestop": load_onestop,
    "geco_gp": load_geco_gp,
    "meco_gp": load_meco_gp,
    "meco_sp": load_meco_sp,
    "provo_gp": load_provo_gp,
    "provo_sp": load_provo_sp,
    "onestop_gp": load_onestop_gp,
    "onestop_sp": load_onestop_sp,
    "pereira_fmri": load_pereira_fmri,
    "pereira_fmri__embed": load_pereira_embed,
    "megmasc_meg": load_megmasc_meg,
    # secondary feature-mode variants (the doubling)
    **{slug: (lambda s=slug: _load_secondary(s)) for slug in SECONDARY_SPEC},
    **{slug: (lambda s=slug: _load_neural_surprisal(s)) for slug in NEURAL_SURPRISAL_SPEC},
}


def process_dataset(slug: str) -> None:
    info = DATASETS_BY_SLUG[slug]
    print(f"[{slug}] loading raw data...")
    data = LOADERS[slug]()
    model_df: pd.DataFrame = data["model_df"]
    if model_df.empty:
        print(f"[{slug}] WARNING: no models found, skipping")
        return
    model_df = model_df.dropna(subset=["score"]).copy()

    layer_df: pd.DataFrame = data.get("layer_df", pd.DataFrame())
    tc_df: pd.DataFrame = data.get("timecourse_df", pd.DataFrame())
    roi_df: pd.DataFrame = data.get("roi_df", pd.DataFrame())
    for d in (layer_df, tc_df, roi_df):
        if not d.empty and "model_id" in d.columns:
            d.drop(d.index[d["model_id"].isin(EXCLUDED_MODEL_IDS)], inplace=True)

    # Ensure best-layer AND last-layer scores exist for every dataset. Secondary
    # loaders provide them directly; for the neural-embedding loaders derive the
    # last-layer score from the per-model layer_df (score at the highest layer).
    if "score_best" not in model_df.columns:
        model_df["score_best"] = model_df["score"]
    if "score_last" not in model_df.columns:
        last = None
        if not layer_df.empty and {"model_id", "layer", "score"} <= set(layer_df.columns):
            lay = layer_df.copy()
            lay["layer"] = pd.to_numeric(lay["layer"], errors="coerce")
            maxl = lay.groupby("model_id")["layer"].transform("max")
            last = lay[lay["layer"] == maxl].groupby("model_id")["score"].mean()
        model_df["score_last"] = (
            model_df["model_id"].map(last).fillna(model_df["score"]) if last is not None
            else model_df["score"]
        )

    # Ceiling-normalise every plotted score so 1.0 == the data's noise ceiling.
    ceiling = ceiling_for(slug)
    val_col, ylabel, ceiling_line = "score", info.metric, False
    if ceiling is not None and np.isfinite(ceiling) and ceiling > 0:
        for d in (model_df, layer_df, tc_df, roi_df):
            if not d.empty and "score" in d.columns:
                d["score_norm"] = d["score"] / ceiling
        for c in ("score_best", "score_last"):
            model_df[f"{c}_norm"] = model_df[c] / ceiling
        model_df["ceiling_r"] = ceiling
        # Normalise scores by the ceiling but do NOT draw the 1.0 reference line
        # (it forces the y-axis up to 1.0 and crushes the readable 0..0.4 range).
        val_col, ylabel, ceiling_line = "score_norm", NORM_YLABEL, False
        print(f"[{slug}] ceiling_r={ceiling:.3f} -> scores normalised (1.0 = ceiling)")
    else:
        # No ceiling (delta metric): keep raw; the report rescales to fraction-of-best.
        for c in ("score_best", "score_last"):
            model_df[f"{c}_norm"] = model_df[c]
        print(f"[{slug}] WARNING: no noise ceiling found; plotting raw scores")

    out_dir = BENCHMARK_SUMMARY_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(out_dir / "all_models_scores.csv", index=False)
    print(f"[{slug}] {len(model_df)} models -> {out_dir/'all_models_scores.csv'}")

    has_ppl = info.has_perplexity and model_df["perplexity"].notna().any()

    # One combined "overview" grid per dataset: scaling law, score-vs-perplexity,
    # layer profile, timecourse — whichever apply — each panel big enough to read
    # on its own, per-model markers by family, model-name labels, per-group fit lines.
    def _ceil(ax):
        if ceiling_line:
            add_ceiling_line(ax)

    panels: list[tuple[str, callable]] = [
        ("Scaling law", lambda ax: (plot_scaling_law(
            model_df, val_col, None, ylabel=ylabel, title="Scaling law", ax=ax, label_points=True,
        ), _ceil(ax))),
    ]
    if has_ppl:
        panels.append(("Score vs. perplexity", lambda ax: (plot_scaling_by_perplexity(
            model_df, val_col, "perplexity", None, ylabel=ylabel, title="Score vs. perplexity",
            ax=ax, label_points=True,
        ), _ceil(ax))))
    else:
        print(f"[{slug}] no perplexity data available, skipping score-vs-perplexity panel")
    if not layer_df.empty:
        panels.append(("Layer profile", lambda ax: (plot_layer_profiles(
            layer_df, val_col, None, ylabel=ylabel, title="Layer profile", ax=ax,
        ), _ceil(ax))))
    if info.has_timecourse and not tc_df.empty:
        panels.append(("Timecourse", lambda ax: (plot_timecourse(
            tc_df, "time_s", val_col, None, ylabel=ylabel, title="Timecourse", ax=ax,
        ), _ceil(ax))))

    build_grid(
        panels, out_dir / "overview.png", suptitle=f"{info.name} — overview",
        ncols=2, panel_size=(9, 7), legend_df=model_df,
    )

    # 2x2 scaling-law grid, one panel per model category (see the results for
    # each group separately rather than all overlaid on one axis).
    plot_scaling_law_by_group(
        model_df, val_col, out_dir / "scaling_by_group.png", ylabel=ylabel,
        title=f"{info.name} — scaling law by model category", ceiling_line=ceiling_line,
    )

    if info.has_roi and not roi_df.empty:
        plot_roi_profile(
            roi_df, val_col, out_dir / "roi_profile.png",
            ylabel=ylabel, title=f"{info.name} — ROI / spatial profile",
            ceiling_line=ceiling_line,
        )

    print(f"[{slug}] done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS_BY_SLUG.keys()),
                         help="Subset of dataset slugs to process (default: all 12)")
    args = parser.parse_args()

    for slug in args.datasets:
        if slug not in DATASETS_BY_SLUG:
            print(f"Unknown dataset slug {slug!r}, skipping", file=sys.stderr)
            continue
        try:
            process_dataset(slug)
        except Exception as e:  # noqa: BLE001
            print(f"[{slug}] FAILED: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
