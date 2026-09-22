#!/usr/bin/env python3
"""Behavioral-surprisal group bars, split by DLT-cost difficulty: regular
(all words) vs hard-only (DLT integration cost >= a threshold, default 2).

For each of the 6 behavioral surprisal datasets (Natural Stories SPR, GECO,
MECO, Provo, UCL Reading, ZuCo Reading NR), the model-choice pipeline already
writes, per model, per layer:
  - a small aggregate ``*_lme_checkpoint_scores.csv`` with the headline
    ``test_delta_pearsonr`` per layer (this IS the "regular" score already
    shown elsewhere in the report -- no need to touch the big per-word file
    to get it).
  - a large per-(subject, word, layer) ``*_lme_per_word.csv`` with the
    held-out actual value (``fdur``) and the full/baseline model's held-out
    predictions (``yhat_full``/``yhat_baseline``) -- this is what has to be
    filtered to DLT-hard words and re-scored.

So per model we only need to touch the per-word file for the ONE best layer
and the ONE last layer (picked from the small scores CSV, matching the
report's own best/last-layer selection) -- not all ~20-30 swept layers. Read
via polars (fast columnar CSV parsing) projecting only the columns needed.

Caches every (dataset, model) result to
``outputs/dlt_hard_bar_cache/<dataset>_hard_scores_thr<T>.csv`` so a rerun
(or the plotting step alone) is instant; per-model results are appended
incrementally so a killed run resumes rather than restarting.

Usage
-----
    .venv/bin/python scripts/dlt/hard_word_bars.py compute --n_jobs 8
    .venv/bin/python scripts/dlt/hard_word_bars.py plot
    .venv/bin/python scripts/dlt/hard_word_bars.py all --n_jobs 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from neuroai.analysis.benchmark_common import (  # noqa: E402
    GROUP_COLORS,
    GROUP_ORDER,
    REPO_ROOT,
    ceiling_for,
    lookup_model_meta,
    sem,
)
from scripts.summarize.benchmark import (  # noqa: E402
    DELTA_CORR_SANITY_MAX,
    OUTPUTS_JEANZAY,
    _iter_model_score_files,
    _model_id_from_dir,
    _pick_final_row,
    _read_csv_safe,
    _sanitize_delta_col,
)

DLT_DIR = REPO_ROOT / "outputs" / "dlt"
CACHE_DIR = REPO_ROOT / "outputs" / "dlt_hard_bar_cache"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

HARD_THRESHOLD = 2.0  # DLT integration ("total") cost >= this = "hard"

# dataset_key -> (sweep_root, scores_filename, per_word_filename, dlt_cost_csv)
DATASET_SPEC: dict[str, tuple[Path, str, str, Path]] = {
    "naturalstories_spr": (
        OUTPUTS_JEANZAY / "naturalstories/surprisal_sweep",
        "spr_lme_checkpoint_scores.csv", "spr_lme_per_word.csv",
        DLT_DIR / "naturalstories_dlt_cost.csv",
    ),
    "geco": (
        OUTPUTS_JEANZAY / "geco/lme_sweep_dlt",
        "geco_lme_checkpoint_scores.csv", "geco_FP_lme_per_word.csv",
        DLT_DIR / "geco_dlt_cost.csv",
    ),
    "meco": (
        OUTPUTS_JEANZAY / "meco/surprisal_sweep" / "fp",
        "meco_lme_checkpoint_scores.csv", "meco_FP_lme_per_word.csv",
        DLT_DIR / "meco_dlt_cost.csv",
    ),
    "provo": (
        OUTPUTS_JEANZAY / "provo/surprisal_sweep" / "fp",
        "provo_lme_checkpoint_scores.csv", "provo_FP_lme_per_word.csv",
        DLT_DIR / "provo_dlt_cost.csv",
    ),
    "ucl_reading": (
        OUTPUTS_JEANZAY / "ucl_reading/surprisal_sweep",
        "ucl_reading_spr_lme_checkpoint_scores.csv", "ucl_reading_spr_lme_per_word.csv",
        DLT_DIR / "ucl_reading_dlt_cost.csv",
    ),
    "zuco_reading": (
        OUTPUTS_JEANZAY / "zuco_reading/surprisal_sweep" / "nr",
        "zuco_reading_trt_lme_checkpoint_scores.csv", "zuco_reading_NR_lme_per_word.csv",
        DLT_DIR / "zuco_reading_nr_dlt_cost.csv",
    ),
}

# GECO temporarily excluded: its per-word files are inconsistently schema'd
# across families (some got the newer multilayer/full-metrics "_dlt" rerun,
# others are still the legacy delta_loglik-only schema with no held-out
# yhat_full/yhat_baseline/fdur) -- the fallback logic in _process_one_model
# handles it, but GECO is left out of the default dataset set / aggregate
# plot for now until that coverage question is revisited.
EXCLUDED_DATASETS: frozenset[str] = frozenset({"geco"})

CACHE_COLUMNS = [
    "model_id", "family", "group", "label", "n_params", "revision",
    "best_layer", "last_layer",
    "score_best_all", "score_last_all", "score_best_hard", "score_last_hard",
    "n_words_best_all", "n_words_last_all", "n_words_best_hard", "n_words_last_hard",
]


# ---------------------------------------------------------------------------
# Stage 1: per-model best/last layer + hard-word rescoring
# ---------------------------------------------------------------------------

def _pick_best_last_layer(scores: pd.DataFrame) -> tuple[int | None, int | None]:
    """Mirror ``_best_last_over_layers``'s layer selection (best = layer with
    max sanitized test_delta_pearsonr; last = highest layer index among rows
    that survive sanitization), but return the LAYER INDEX, not the score."""
    df = _sanitize_delta_col(scores, "test_delta_pearsonr", max_abs=DELTA_CORR_SANITY_MAX)
    s = pd.to_numeric(df["test_delta_pearsonr"], errors="coerce")
    lay = pd.to_numeric(df["layer"], errors="coerce")
    valid = lay.notna() & s.notna()
    if not valid.any():
        return None, None
    best_idx = s[valid].idxmax()
    best_layer = int(lay.loc[best_idx])
    last_layer = int(lay[valid].max())
    return best_layer, last_layer


def _final_revision(scores: pd.DataFrame) -> tuple[str, float, float]:
    """Final-checkpoint revision string + that revision's best/last-layer
    test_delta_pearsonr (the ready-made "regular"/all-word score)."""
    key = "tokens_seen" if "tokens_seen" in scores.columns else ("step" if "step" in scores.columns else None)
    if key is not None and scores[key].notna().any():
        final_val = scores[key].max()
        scores = scores[scores[key] == final_val]
    revision = str(_pick_final_row(scores).get("revision", "main"))
    scores = scores[scores["revision"].astype(str) == revision] if "revision" in scores.columns else scores
    return revision, scores


def _pooled_delta_pearsonr(df: pd.DataFrame) -> tuple[float, int]:
    """Pooled corr(fdur, yhat_full) - corr(fdur, yhat_baseline), matching
    ``neuroai.regressors.lme._corrs`` exactly (single pooled Pearson r across
    all rows, not per-subject-then-averaged)."""
    if df.empty:
        return float("nan"), 0
    yt = df["fdur"].to_numpy(dtype=np.float64)
    yb = df["yhat_baseline"].to_numpy(dtype=np.float64)
    yf = df["yhat_full"].to_numpy(dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yb) & np.isfinite(yf)
    n = int(valid.sum())
    if n < 3:
        return float("nan"), n
    r_b = float(np.corrcoef(yt[valid], yb[valid])[0, 1])
    r_f = float(np.corrcoef(yt[valid], yf[valid])[0, 1])
    if not (np.isfinite(r_b) and np.isfinite(r_f)):
        return float("nan"), n
    delta = r_f - r_b
    if abs(delta) > DELTA_CORR_SANITY_MAX:
        return float("nan"), n
    return delta, n


def _per_word_columns(per_word_path: Path) -> set[str]:
    import polars as pl

    return set(pl.scan_csv(per_word_path).collect_schema().names())


def _max_layer_in_per_word(per_word_path: Path, revision: str) -> int | None:
    """Fallback for datasets whose small aggregate scores CSV has no ``layer``
    column at all (GECO: confirmed true for every family, not just the
    memory-constrained ones -- ``geco_lme_checkpoint_scores.csv`` is a plain
    per-checkpoint file with no layer axis, even though the co-located
    per-word file DOES sweep layers). Reads only the ``layer``/``revision``
    columns (cheap) to recover the highest swept layer directly from data."""
    import polars as pl

    lf = (
        pl.scan_csv(per_word_path, schema_overrides={"sentid": pl.Int64, "sentpos": pl.Int64})
        .filter(pl.col("revision").cast(pl.Utf8) == revision)
        .select("layer")
    )
    vals = lf.collect(engine="streaming").to_series()
    if vals.is_empty():
        return None
    return int(vals.max())


def _scores_for_layer(
    per_word_path: Path, layer: int, revision: str, dlt_key: pd.DataFrame, hard_threshold: float,
) -> tuple[float, int, float, int]:
    """(score_all, n_all, score_hard, n_hard) for one (layer, revision), from
    a single read of the per-word file."""
    import polars as pl

    lf = (
        pl.scan_csv(per_word_path, schema_overrides={"sentid": pl.Int64, "sentpos": pl.Int64})
        .filter((pl.col("layer") == layer) & (pl.col("revision").cast(pl.Utf8) == revision))
        .select(["sentid", "sentpos", "fdur", "yhat_baseline", "yhat_full"])
    )
    pw = lf.collect(engine="streaming").to_pandas()
    if pw.empty:
        return float("nan"), 0, float("nan"), 0
    pw["story_id"] = pw["sentid"].astype(int)
    pw["zone"] = pw["sentpos"].astype(int)
    pw = pw.merge(dlt_key, on=["story_id", "zone"], how="left")
    score_all, n_all = _pooled_delta_pearsonr(pw)
    hard = pw[pw["total"] >= hard_threshold]
    score_hard, n_hard = _pooled_delta_pearsonr(hard)
    return score_all, n_all, score_hard, n_hard


def _hard_score_for_layer(
    per_word_path: Path, layer: int, revision: str, dlt_key: pd.DataFrame, hard_threshold: float,
) -> tuple[float, int]:
    _, _, score_hard, n_hard = _scores_for_layer(per_word_path, layer, revision, dlt_key, hard_threshold)
    return score_hard, n_hard


def _process_one_model(
    family: str, model_dir: Path, per_word_path: Path, scores_path: Path,
    dlt_key: pd.DataFrame, hard_threshold: float,
) -> dict | None:
    """Never raises -- any per-model failure (malformed/partial scores CSV,
    missing columns, unreadable per-word file, ...) is logged and skipped so
    one bad model can't take down the whole (parallel) batch."""
    model_id = _model_id_from_dir(model_dir)
    try:
        scores = _read_csv_safe(scores_path)
        if scores is None or scores.empty or "test_delta_pearsonr" not in scores.columns:
            return None
        if "converged" in scores.columns:
            conv = scores[scores["converged"] == True]  # noqa: E712
            if not conv.empty:
                scores = conv

        pw_cols = _per_word_columns(per_word_path)
        has_hard_cols = {"fdur", "yhat_baseline", "yhat_full", "layer", "sentid", "sentpos"} <= pw_cols

        if "layer" in scores.columns:
            revision, scores_final = _final_revision(scores)
            if scores_final.empty or "layer" not in scores_final.columns:
                return None
            best_layer, last_layer = _pick_best_last_layer(scores_final)
            if best_layer is None:
                return None
            s = pd.to_numeric(
                _sanitize_delta_col(scores_final, "test_delta_pearsonr", max_abs=DELTA_CORR_SANITY_MAX)["test_delta_pearsonr"],
                errors="coerce",
            )
            lay = pd.to_numeric(scores_final["layer"], errors="coerce")
            score_best_all = float(s[lay == best_layer].iloc[0]) if (lay == best_layer).any() else float("nan")
            score_last_all = float(s[lay == last_layer].iloc[0]) if (lay == last_layer).any() else float("nan")
            n_obs_col = "n_obs_test" if "n_obs_test" in scores_final.columns else None
            n_all_best = int(scores_final.loc[lay == best_layer, n_obs_col].iloc[0]) if n_obs_col and (lay == best_layer).any() else -1
            n_all_last = int(scores_final.loc[lay == last_layer, n_obs_col].iloc[0]) if n_obs_col and (lay == last_layer).any() else -1

            if not has_hard_cols:
                score_best_hard = score_last_hard = float("nan")
                n_hard_best = n_hard_last = -1
            else:
                score_best_hard, n_hard_best = _hard_score_for_layer(per_word_path, best_layer, revision, dlt_key, hard_threshold)
                if last_layer == best_layer:
                    score_last_hard, n_hard_last = score_best_hard, n_hard_best
                else:
                    score_last_hard, n_hard_last = _hard_score_for_layer(per_word_path, last_layer, revision, dlt_key, hard_threshold)
        elif has_hard_cols:
            # Small aggregate scores CSV has no layer axis at all (confirmed
            # true for GECO across every family, including standard LLMs --
            # not just the memory-constrained ones) even though the co-located
            # per-word file DOES sweep layers -- some models' per-word files
            # already got the newer multilayer/full-metrics rerun. Recover
            # "last layer" directly from the per-word file's own data and
            # recompute both the all-word and hard-word scores from it in one
            # pass; "best layer" search would need scoring every swept layer
            # (expensive), so we fall back to using the same (last) layer for
            # both -- flagged via best_layer == last_layer in the cache.
            revision, _ = _final_revision(scores)
            last_layer = _max_layer_in_per_word(per_word_path, revision)
            if last_layer is None:
                return None
            best_layer = last_layer
            score_last_all, n_all_last, score_last_hard, n_hard_last = _scores_for_layer(
                per_word_path, last_layer, revision, dlt_key, hard_threshold,
            )
            score_best_all, n_all_best = score_last_all, n_all_last
            score_best_hard, n_hard_best = score_last_hard, n_hard_last
        else:
            # Neither the small scores CSV nor the per-word file has a layer
            # axis or held-out yhat_full/yhat_baseline/fdur columns (legacy
            # single-implicit-layer schema, only delta_loglik saved) -- these
            # models predate the multilayer "_dlt" rerun and genuinely cannot
            # be rescored for a DLT-hard subset (no held-out predictions were
            # ever saved). Still surface the "all words" score (straight from
            # the small scores CSV's final checkpoint) so the regular/left
            # panel keeps full coverage; the hard/right panel just won't have
            # this model. best_layer/last_layer = -1 flags "no layer info".
            revision, scores_final = _final_revision(scores)
            if scores_final.empty:
                return None
            row = _pick_final_row(scores_final)
            val = pd.to_numeric(pd.Series([row.get("test_delta_pearsonr")]), errors="coerce").iloc[0]
            if not np.isfinite(val) or abs(val) > DELTA_CORR_SANITY_MAX:
                return None
            best_layer = last_layer = -1
            score_best_all = score_last_all = float(val)
            n_all_best = n_all_last = int(row.get("n_obs_test", -1)) if pd.notna(row.get("n_obs_test")) else -1
            score_best_hard = score_last_hard = float("nan")
            n_hard_best = n_hard_last = -1

        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        return {
            "model_id": model_id, "family": meta.family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "revision": revision,
            "best_layer": best_layer, "last_layer": last_layer,
            "score_best_all": score_best_all, "score_last_all": score_last_all,
            "score_best_hard": score_best_hard, "score_last_hard": score_last_hard,
            "n_words_best_all": n_all_best, "n_words_last_all": n_all_last,
            "n_words_best_hard": n_hard_best, "n_words_last_hard": n_hard_last,
        }
    except Exception:
        logger.exception("[%s/%s] failed -- skipping this model", family, model_dir.name)
        return None


def compute_dataset(dataset_key: str, hard_threshold: float = HARD_THRESHOLD, n_jobs: int = 4, force: bool = False) -> Path:
    sweep_root, scores_filename, per_word_filename, dlt_cost_csv = DATASET_SPEC[dataset_key]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{dataset_key}_hard_scores_thr{hard_threshold:g}.csv"

    done_ids: set[str] = set()
    existing_rows: list[pd.DataFrame] = []
    if cache_path.is_file() and not force:
        prev = pd.read_csv(cache_path)
        existing_rows.append(prev)
        done_ids = set(prev["model_id"])
        logger.info("[%s] resuming: %d models already cached", dataset_key, len(done_ids))

    dlt_key = pd.read_csv(dlt_cost_csv)[["story_id", "zone", "total"]].drop_duplicates(["story_id", "zone"])

    todo = [
        (family, model_dir, files[0])
        for family, model_dir, files in _iter_model_score_files(sweep_root, per_word_filename)
        if _model_id_from_dir(model_dir) not in done_ids
    ]
    logger.info("[%s] %d models to process (n_jobs=%d)", dataset_key, len(todo), n_jobs)

    from joblib import Parallel, delayed

    BATCH = max(n_jobs, 8)
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        try:
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_process_one_model)(
                    family, model_dir, per_word_path, model_dir / scores_filename, dlt_key, hard_threshold,
                )
                for family, model_dir, per_word_path in batch
            )
        except Exception:
            # A worker crash (OOM-killed, segfault, ...) takes the whole batch
            # with it, but earlier batches are already flushed to cache_path --
            # log and move on to the next batch rather than aborting the dataset.
            logger.exception(
                "[%s] batch %d-%d crashed (worker likely OOM-killed) -- skipping this batch, "
                "rerun the same command later to retry it (already-cached models are untouched)",
                dataset_key, i, min(i + BATCH, len(todo)),
            )
            continue
        rows = [r for r in results if r is not None]
        if rows:
            existing_rows.append(pd.DataFrame(rows))
            out = pd.concat(existing_rows, ignore_index=True).drop_duplicates("model_id", keep="last")
            out.to_csv(cache_path, index=False)
            existing_rows = [out]
        logger.info("[%s] %d/%d models done", dataset_key, min(i + BATCH, len(todo)), len(todo))

    logger.info("[%s] wrote %s", dataset_key, cache_path)
    return cache_path


# ---------------------------------------------------------------------------
# Stage 2: plotting
# ---------------------------------------------------------------------------

def _aggregate_across_datasets(hard_threshold: float, score_col: str) -> pd.DataFrame:
    """Ceiling-normalise each dataset's ``score_col``, then average per model
    across whichever of the included behavioral datasets it appears in --
    same normalise-then-average convention as ``_modality_model_df`` in
    summarize/report.py. Skips EXCLUDED_DATASETS (see its docstring)."""
    frames = []
    for dataset_key in DATASET_SPEC:
        if dataset_key in EXCLUDED_DATASETS:
            continue
        cache_path = CACHE_DIR / f"{dataset_key}_hard_scores_thr{hard_threshold:g}.csv"
        if not cache_path.is_file():
            logger.warning("[%s] no cache at %s -- skipping in aggregate", dataset_key, cache_path)
            continue
        df = pd.read_csv(cache_path)
        if df.empty or score_col not in df.columns:
            continue
        ceiling = ceiling_for(dataset_key)
        d = df[["model_id", "family", "group", "label", "n_params", score_col]].copy()
        d = d.rename(columns={score_col: "score"})
        if ceiling is not None and np.isfinite(ceiling) and ceiling > 0:
            d["score_norm"] = d["score"] / ceiling
        else:
            mx = d["score"].max()
            d["score_norm"] = d["score"] / mx if pd.notna(mx) and mx > 0 else d["score"]
        d["dataset"] = dataset_key
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    agg = (
        long_df.groupby(["model_id", "family", "group", "label", "n_params"], as_index=False)
        .agg(score=("score_norm", "mean"), n_datasets=("dataset", "nunique"))
    )
    return agg


def _draw_group_bar_axis(ax, d: pd.DataFrame, title: str) -> list[tuple[float, float]]:
    """Mean +/- SEM only (diamond + whiskers) per group -- no bar rectangles
    or per-model scatter, so a handful of outlier models can't force the
    y-axis wide enough to flatten the mean markers into invisibility.
    Returns the plotted (mean, sem) pairs so the caller can size the y-axis
    off the error bars themselves rather than the raw (outlier-prone) data."""
    groups = [g for g in GROUP_ORDER if (d["group"] == g).any()]
    mean_sems: list[tuple[float, float]] = []
    for gi, g in enumerate(groups):
        sub = d[d["group"] == g]
        vals = sub["score"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        m, s = float(vals.mean()), sem(vals)
        mean_sems.append((m, s))
        ax.errorbar(gi, m, yerr=s, fmt="D", color=GROUP_COLORS[g], markersize=7,
                    markerfacecolor=GROUP_COLORS[g], markeredgecolor="black", markeredgewidth=0.8,
                    ecolor=GROUP_COLORS[g], capsize=5, elinewidth=1.8, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n={int((d['group'] == g).sum())})" for g in groups], fontsize=8.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.axhline(0, color="grey", linewidth=0.8, zorder=0)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    return mean_sems


def plot_hard_word_bars(hard_threshold: float = HARD_THRESHOLD) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for layer_sel, all_col, hard_col in (
        ("best", "score_best_all", "score_best_hard"),
        ("last", "score_last_all", "score_last_hard"),
    ):
        d_all = _aggregate_across_datasets(hard_threshold, all_col)
        d_hard = _aggregate_across_datasets(hard_threshold, hard_col)
        if d_all.empty and d_hard.empty:
            logger.warning("[%s] no data for either panel -- skipping figure", layer_sel)
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        ms_all = _draw_group_bar_axis(axes[0], d_all, "All words")
        ms_hard = _draw_group_bar_axis(axes[1], d_hard, f"Hard words only (DLT cost ≥ {hard_threshold:g})")

        # Size the shared y-axis off the error bars (mean +/- SEM) actually
        # drawn, not the raw per-model scores -- a few outlier models would
        # otherwise force the axis wide enough to flatten every mean marker.
        lo = min((m - s for m, s in ms_all + ms_hard), default=0.0)
        hi = max((m + s for m, s in ms_all + ms_hard), default=1.0)
        pad = (hi - lo) * 0.25 or 0.01
        for ax in axes:
            ax.set_ylim(lo - pad, hi + pad)
        axes[0].set_ylabel("score / noise ceiling\n(mean across behavioral datasets)")
        handles = [plt.Line2D([0], [0], marker="D", color=GROUP_COLORS[g], markersize=7,
                              markerfacecolor=GROUP_COLORS[g], markeredgecolor="black",
                              markeredgewidth=0.8, linestyle="none")
                   for g in GROUP_ORDER]
        fig.legend(handles, GROUP_ORDER, loc="upper center", ncol=len(GROUP_ORDER),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))
        fig.suptitle(
            f"Behavioral surprisal, {layer_sel} layer: regular vs DLT-hard words only\n"
            "(mean ± SEM across models, ceiling-normalised, averaged over "
            "Natural Stories SPR / MECO / Provo / UCL Reading / ZuCo Reading -- "
            "GECO excluded for now, inconsistent per-word schema across families)",
            fontsize=11, y=1.1,
        )
        fig.tight_layout()
        out = FIGURES_DIR / f"dlt_hard_bars_{layer_sel}_layer_thr{hard_threshold:g}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", out)
        out_paths.append(out)
    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["compute", "plot", "all"])
    ap.add_argument("--datasets", nargs="*", default=[k for k in DATASET_SPEC if k not in EXCLUDED_DATASETS])
    ap.add_argument("--hard_threshold", type=float, default=HARD_THRESHOLD)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.cmd in ("compute", "all"):
        for ds in args.datasets:
            compute_dataset(ds, hard_threshold=args.hard_threshold, n_jobs=args.n_jobs, force=args.force)
    if args.cmd in ("plot", "all"):
        plot_hard_word_bars(hard_threshold=args.hard_threshold)


if __name__ == "__main__":
    main()
