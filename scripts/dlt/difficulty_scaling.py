#!/usr/bin/env python3
"""DLT-cost difficulty-split scaling laws.

For each of the four datasets (Natural Stories SPR, GECO — behavioral;
Podcast, Broderick — neural), split words into easy/hard bins by DLT
integration cost — fixed cutoff at 0 (easy = no integration cost, hard = any
nonzero cost; see the module-level comment above ``dlt_cutoff``) — and plot
the existing scaling-law curve (score vs n_params/perplexity) **separately
per bin** — i.e. does the LLM's advantage concentrate on long-dependency
(hard) words, or is it flat across difficulty?

Requires, per dataset:
  - DLT cost table:      outputs/dlt/{dataset}_dlt_cost.csv        (already computed)
  - Aggregate scores:    <sweep_root>/**/{spr,geco,podcast,broderick}_checkpoint_scores.csv
                         (for n_params / perplexity per model — already exists)
  - Per-word data:
      naturalstories: <sweep_root>/**/spr_lme_per_word.csv
      geco:           <sweep_root>/**/geco_{FP,GP}_lme_per_word.csv
      podcast:        <sweep_root>/**/podcast_word_predictions/*.npz   (needs --save_word_predictions rerun)
      broderick:      <sweep_root>/**/broderick_word_predictions/*.npz (needs --save_word_predictions rerun)

Usage
-----
    .venv/bin/python scripts/dlt/difficulty_scaling.py \\
        --dataset naturalstories --sweep_root outputs_jeanzay/naturalstories_lme_sweep \\
        --dlt_cost outputs/dlt/naturalstories_dlt_cost.csv \\
        --out_dir outputs/dlt_difficulty
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())


# ---------------------------------------------------------------------------
# DLT bin cutoff — fixed at 0, not the median.
#
# A median split coincided with this for every dataset checked (median==0 for
# Natural Stories / GECO / Podcast / Broderick — DLT cost is a sparse, mostly-
# zero signal, so the middle value of a sorted list that skewed lands on 0).
# But tying the rule to "whatever the median happens to be" made the criterion
# implicit and dataset-dependent — it would silently shift to a different,
# non-zero cutoff for some future corpus with <50% zero-cost words, changing
# what "easy"/"hard" means without anyone noticing. Fixing it at 0 makes the
# rule explicit and dataset-independent: easy = no DLT integration cost at
# all, hard = any nonzero cost.
# ---------------------------------------------------------------------------

def dlt_cutoff(dlt_cost: pd.DataFrame, *, cost_col: str = "total") -> float:
    return 0.0


def log_dlt_distribution(dlt_cost: pd.DataFrame, *, dataset: str, cost_col: str = "total") -> dict:
    """Print + return summary stats of the DLT-cost column actually used for the
    easy/hard split, so a flat-looking easy-vs-hard plot can be told apart from
    "the split itself carries little signal for this corpus" — e.g. DLT total
    cost is right-skewed with a lot of zeros, and the "hard" bin (any nonzero
    cost) is itself dominated by barely-nonzero words (cost 1-3), with truly
    long-dependency words (cost >10) making up under 1% of most corpora.
    """
    t = dlt_cost[cost_col]
    cutoff = dlt_cutoff(dlt_cost, cost_col=cost_col)
    pct_levels = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    percentiles = {f"p{p}": float(np.percentile(t, p)) for p in pct_levels}
    stats = {
        "dataset": dataset,
        "n_words": int(len(t)),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "median": float(t.median()),
        "cutoff": cutoff,
        "std": float(t.std()),
        "frac_at_or_below_cutoff": float((t <= cutoff).mean()),
        "frac_exactly_zero": float((t == 0).mean()),
        **percentiles,
    }
    pct_str = "  ".join(f"p{p}={percentiles[f'p{p}']:g}" for p in pct_levels)
    logger.info(
        "[%s] DLT '%s' distribution — n=%d  min=%.1f  max=%.1f  mean=%.3f  median=%.3f  "
        "std=%.3f  cutoff=%.1f (fixed, not the median)  frac<=cutoff=%.3f  frac==0=%.3f",
        dataset, cost_col, stats["n_words"], stats["min"], stats["max"], stats["mean"],
        stats["median"], stats["std"], cutoff, stats["frac_at_or_below_cutoff"], stats["frac_exactly_zero"],
    )
    logger.info("[%s] DLT '%s' percentiles — %s", dataset, cost_col, pct_str)
    return stats


def log_threshold_bin_counts(
    dlt_cost: pd.DataFrame,
    thresholds: list[float],
    *,
    dataset: str,
    cost_col: str = "total",
    shared_easy_cutoff: float | None = None,
) -> dict[float, dict[str, float | int]]:
    """Log (and return) how many stimulus words fall in easy/hard for each
    hard>t threshold — so a flat easy-vs-hard gap can be read against bin size
    (e.g. Provo hard>3 is already a tiny tail).
    """
    t = dlt_cost[cost_col].to_numpy(dtype=float)
    n = int(len(t))
    out: dict[float, dict[str, float | int]] = {}
    logger.info("[%s] DLT bin sizes for hard_gt_thresholds (n_words=%d):", dataset, n)
    for thr in thresholds:
        if shared_easy_cutoff is not None:
            n_easy = int(np.sum(np.isfinite(t) & (t <= shared_easy_cutoff)))
            n_hard = int(np.sum(np.isfinite(t) & (t > thr)))
            n_excl = n - n_easy - n_hard  # mid band: shared_easy < cost ≤ thr
            logger.info(
                "  hard>%g  easy(≤%g)=%d (%.1f%%)  hard=%d (%.1f%%)  excluded_mid=%d (%.1f%%)",
                thr, shared_easy_cutoff, n_easy, 100 * n_easy / n if n else 0.0,
                n_hard, 100 * n_hard / n if n else 0.0,
                n_excl, 100 * n_excl / n if n else 0.0,
            )
        else:
            n_easy = int(np.sum(np.isfinite(t) & (t <= thr)))
            n_hard = int(np.sum(np.isfinite(t) & (t > thr)))
            logger.info(
                "  hard>%g  easy(≤%g)=%d (%.1f%%)  hard=%d (%.1f%%)",
                thr, thr, n_easy, 100 * n_easy / n if n else 0.0,
                n_hard, 100 * n_hard / n if n else 0.0,
            )
        out[float(thr)] = {"n_easy": n_easy, "n_hard": n_hard, "n_words": n}
    return out


def assign_bin(cost: np.ndarray, cutoff: float) -> np.ndarray:
    """'easy' (<= cutoff) / 'hard' (> cutoff). NaN cost -> NaN bin (dropped later)."""
    out = np.full(cost.shape, None, dtype=object)
    valid = np.isfinite(cost)
    out[valid & (cost <= cutoff)] = "easy"
    out[valid & (cost > cutoff)] = "hard"
    return out


# ---------------------------------------------------------------------------
# Behavioral (Natural Stories / GECO): per-word delta log-likelihood
# ---------------------------------------------------------------------------

def _find_files(sweep_root: Path, filename: str) -> list[Path]:
    return sorted(sweep_root.rglob(filename))


def _registry_n_params_lookup() -> dict[str, float]:
    """{safe_model_dir(model_id): n_params} for every registered model.

    Needed because the LME sweeps' own ``*_checkpoint_scores.csv`` (written by
    the runner scripts, e.g. geco_lme_checkpoint_scores.csv) carries no
    model/n_params column at all — only measure/step/revision/tokens_seen/stats
    (model identity is implicit in the directory, not a column).
    """
    from neuroai.models.registry import registry_model_configs
    from neuroai.sweep import safe_model_dir

    return {safe_model_dir(m["model_id"]): m["n_params"] for m in registry_model_configs()}


def _registry_model_id_lookup() -> dict[str, str]:
    """{safe_model_dir(model_id): model_id} — reverse of ``_registry_n_params_lookup``'s
    key. Needed only by the podcast_lme path: its pooled table/predictions parquet
    carry no model/revision columns at all (unlike the ridge encoder's .npz files,
    which save "model"/"revision" explicitly) — model identity lives solely in the
    directory name — so joining perplexity (keyed by raw HF model_id) requires
    mapping back from that directory name first.
    """
    from neuroai.models.registry import registry_model_configs
    from neuroai.sweep import safe_model_dir

    return {safe_model_dir(m["model_id"]): m["model_id"] for m in registry_model_configs()}


def _load_perplexity_lookup(perplexity_dir: Path | None) -> pd.DataFrame | None:
    """Columns: model, [revision], perplexity — read directly from each model's
    own ``*_perplexity.json`` (or ``perplexity.json``) under ``perplexity_dir``.

    NOT from the aggregated ``*_perplexity_summary.csv`` — that file was found
    to be wrong for at least the hard_Ngram families (every 2/4-layer,
    babylm/pile variant reporting the exact same value, 50277.0042, while each
    model's own per-model JSON has distinct, plausible values). Whatever wrote
    that summary CSV clobbered/misjoined rows; the per-model JSONs are the
    ground truth (same forward pass, just not re-aggregated).
    """
    import json

    if perplexity_dir is None or not perplexity_dir.is_dir():
        return None
    rows: list[dict] = []
    for json_path in perplexity_dir.rglob("*.json"):
        try:
            d = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "model_id" not in d or "perplexity" not in d:
            continue
        rows.append({
            "model": d["model_id"],
            "revision": d.get("revision", "main"),
            "perplexity": d["perplexity"],
        })
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates()


def build_behavioral_scaling_table(
    *,
    sweep_root: Path,
    dlt_cost: pd.DataFrame,
    per_word_filename: str,
    scores_filename: str,
    perplexity_dir: Path | None = None,
    cutoff: float | None = None,
) -> pd.DataFrame:
    cutoff = dlt_cutoff(dlt_cost) if cutoff is None else float(cutoff)
    tables = build_behavioral_scaling_table_multi_cutoff(
        sweep_root=sweep_root,
        dlt_cost=dlt_cost,
        per_word_filename=per_word_filename,
        scores_filename=scores_filename,
        perplexity_dir=perplexity_dir,
        cutoffs={"_single": cutoff},
    )
    return tables["_single"]


def build_behavioral_scaling_table_multi_cutoff(
    *,
    sweep_root: Path,
    dlt_cost: pd.DataFrame,
    per_word_filename: str,
    scores_filename: str,
    perplexity_dir: Path | None = None,
    cutoffs: dict[object, float],
) -> dict[object, pd.DataFrame]:
    """Same as ``build_behavioral_scaling_table`` but computes ALL ``cutoffs``
    (keyed by whatever label the caller wants back, e.g. a threshold float or
    "median"/"p95") from a single read of each per-word CSV — reading and
    re-parsing the same (possibly large) files once per cutoff instead of once
    total was the dominant cost of a multi-threshold scan.
    """
    dlt_key = dlt_cost[["story_id", "zone", "total"]].drop_duplicates(["story_id", "zone"])
    n_params_lookup = _registry_n_params_lookup()
    perplexity_lookup = _load_perplexity_lookup(perplexity_dir)

    per_word_files = _find_files(sweep_root, per_word_filename)
    if not per_word_files:
        raise FileNotFoundError(
            f"No {per_word_filename} found under {sweep_root} — rerun the sweep "
            "(per-word saving is unconditional in the current code, but only for new runs)."
        )

    rows_by_cutoff: dict[object, list[pd.DataFrame]] = {name: [] for name in cutoffs}
    for pw_path in per_word_files:
        per_word = pd.read_csv(pw_path)
        # sentid/sentpos are exactly str(story_id)/int(zone) for both LME styles used
        # here (see neuroai/regressors/lme.py::prepare_spr_lme_frame and
        # neuroai/regressors/ucl_reading_lme.py::prepare_ucl_reading_lme_frame).
        per_word["story_id"] = per_word["sentid"].astype(int)
        per_word["zone"] = per_word["sentpos"].astype(int)
        per_word = per_word.merge(dlt_key, on=["story_id", "zone"], how="left")
        total = per_word["total"].to_numpy(dtype=float)

        group_cols = [c for c in ("model", "revision") if c in per_word.columns]

        # Model identity lives in the directory, not in the checkpoint_scores.csv —
        # join tokens_seen by revision within THIS file's own directory only, so a
        # "main"/"step1000" revision from one model never matches another model's row.
        model_short = pw_path.parent.name
        family = pw_path.parent.parent.name
        scores_path = pw_path.parent / scores_filename
        scores_df = pd.read_csv(scores_path) if scores_path.is_file() else None

        for cutoff_name, cutoff_val in cutoffs.items():
            pw = per_word.copy()
            pw["dlt_bin"] = assign_bin(total, cutoff_val)
            pw = pw.dropna(subset=["dlt_bin"])

            # Mean ± SEM across subjects of each subject's total Δloglik in the
            # bin (enables shaded error bands). Without subject, fall back to a
            # single grand sum and no SE.
            if "subject" in pw.columns:
                per_subj = (
                    pw.groupby(group_cols + ["subject", "dlt_bin"], dropna=False)["delta_loglik"]
                    .sum()
                    .reset_index()
                )
                agg = (
                    per_subj.groupby(group_cols + ["dlt_bin"], dropna=False)["delta_loglik"]
                    .agg(score="mean", score_se="sem", n_subjects="count")
                    .reset_index()
                )
            else:
                agg = (
                    pw.groupby(group_cols + ["dlt_bin"])["delta_loglik"]
                    .sum()
                    .reset_index()
                    .rename(columns={"delta_loglik": "score"})
                )
                agg["score_se"] = np.nan
                agg["n_subjects"] = np.nan
            agg["model_short"] = model_short
            agg["family"] = family
            agg["n_params"] = n_params_lookup.get(model_short, np.nan)
            if scores_df is not None and "revision" in agg.columns \
                    and "tokens_seen" in scores_df.columns and "revision" in scores_df.columns:
                agg = agg.merge(
                    scores_df[["revision", "tokens_seen"]].drop_duplicates("revision"),
                    on="revision", how="left",
                )
            rows_by_cutoff[cutoff_name].append(agg)

    out: dict[object, pd.DataFrame] = {}
    for cutoff_name, rows in rows_by_cutoff.items():
        d = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not d.empty and perplexity_lookup is not None:
            join_cols = ["model", "revision"] if "revision" in perplexity_lookup.columns else ["model"]
            d = d.merge(perplexity_lookup, on=join_cols, how="left")
        out[cutoff_name] = d
    return out


# ---------------------------------------------------------------------------
# Neural (Podcast / Broderick): per-word out-of-fold predicted lag matrices
# ---------------------------------------------------------------------------

def _colwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of a with the matching column of b,
    ignoring rows that are NaN in either. a, b: (n_words, n_times). Returns
    (n_times,). Columns with <3 valid rows or ~0 std (<1e-8) -> NaN.

    Two-pass (centre first) in float64 — the naive one-pass covariance formula
    loses precision on the float32 predictions (catastrophic cancellation), which
    is why the vectorised result must NOT use it.
    """
    a = a.astype(np.float64); b = b.astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    n = m.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.where(m, a, 0.0).sum(0) / n
        mb = np.where(m, b, 0.0).sum(0) / n
        ac = np.where(m, a - ma, 0.0)
        bc = np.where(m, b - mb, 0.0)
        cov = (ac * bc).sum(0)
        va = (ac * ac).sum(0)
        vb = (bc * bc).sum(0)
        r = cov / np.sqrt(va * vb)
    # Match the original per-lag guard: drop cols with <3 valid rows or std<1e-8
    # (variance = va/n < 1e-16).
    bad = (n < 3) | (va / n < 1e-16) | (vb / n < 1e-16)
    r[bad] = np.nan
    return r


def _rank_cols(x: np.ndarray) -> np.ndarray:
    """Column-wise average ranks (ties averaged) with NaNs left in place.

    Fast path (no NaNs — Podcast/Broderick prediction arrays): pure-numpy
    argsort ranks. Exact float ties are vanishingly rare in continuous neural
    predictions (~0.04% of adjacent sorted pairs), so skipping scipy's
    average-tie pass is a negligible approximation and ~20-30% faster than
    ``scipy.stats.rankdata`` on the wide (n_words × n_elec*n_times) matrices
    this path builds. Slow path keeps the original per-column scipy behaviour.
    """
    if not np.isnan(x).any():
        n = x.shape[0]
        # mergesort is stable; with near-zero ties the rank order matches
        # scipy's average-rank up to those rare exact collisions.
        order = np.argsort(x, axis=0, kind="mergesort")
        ranks = np.empty(x.shape, dtype=np.float64)
        np.put_along_axis(
            ranks, order, np.arange(1, n + 1, dtype=np.float64)[:, None], axis=0,
        )
        return ranks

    from scipy.stats import rankdata

    out = np.full(x.shape, np.nan, dtype=np.float64)
    for j in range(x.shape[1]):
        col = x[:, j]
        f = np.isfinite(col)
        if f.sum() >= 3:
            out[f, j] = rankdata(col[f], method="average")
    return out


def _score_electrodes_batch(oof_all: np.ndarray, y_all: np.ndarray, mask: np.ndarray,
                             metric: str) -> np.ndarray:
    """Mean per-lag correlation for EVERY electrode in one shot.

    ``oof_all``/``y_all``: (n_elec, n_words, n_times). ``mask``: (n_words,) bool.
    Returns (n_elec,) float scores (NaN if an electrode has no valid lags).

    Reshapes to (n_words_in_bin, n_elec * n_times) so ranking + column-wise
    correlation run once across the whole (electrode, lag) grid instead of
    once per electrode.
    """
    # Keep float32 through the reshape; only promote for the correlation pass.
    # Ranking on float32 is fine (order statistics); _colwise_corr needs float64
    # to avoid catastrophic cancellation on centred products.
    o = np.asarray(oof_all[:, mask, :])  # (E, W, T)
    y = np.asarray(y_all[:, mask, :])
    n_elec, n_words, n_times = o.shape
    o2 = o.transpose(1, 0, 2).reshape(n_words, n_elec * n_times)
    y2 = y.transpose(1, 0, 2).reshape(n_words, n_elec * n_times)

    if metric == "spearman":
        with np.errstate(invalid="ignore"):
            # std in float64 for the near-constant guard (matches old path)
            raw_ok = (
                (np.nanstd(o2.astype(np.float64, copy=False), axis=0) >= 1e-8)
                & (np.nanstd(y2.astype(np.float64, copy=False), axis=0) >= 1e-8)
            )
        a, b = _rank_cols(o2), _rank_cols(y2)
        r = _colwise_corr(a, b)
        r[~raw_ok] = np.nan
    else:
        r = _colwise_corr(o2.astype(np.float64, copy=False),
                          y2.astype(np.float64, copy=False))

    r = r.reshape(n_elec, n_times)
    with np.errstate(invalid="ignore"):
        return np.nanmean(r, axis=1)


def _score_electrode(oof: np.ndarray, y_true: np.ndarray, metric: str) -> float:
    """Per-lag correlation over words, averaged over lags.

    ``oof``/``y_true``: (n_words_in_bin, n_times) for one electrode.
    Kept for single-electrode call sites; the hot path uses
    ``_score_electrodes_batch`` instead.
    """
    scores = _score_electrodes_batch(
        oof[np.newaxis, :, :], y_true[np.newaxis, :, :],
        np.ones(oof.shape[0], dtype=bool), metric,
    )
    return float(scores[0]) if np.isfinite(scores[0]) else float("nan")


def _process_npz_multi_cutoff(npz_path: Path, dlt_key: pd.Series, cutoffs: dict[object, float],
                              npz_id_key: str, metric: str,
                              *, shared_easy_cutoff: float | None = None) -> list[dict]:
    """All (cutoff, bin) rows for one .npz (= one subject/model).

    Loads ``oof_predictions``/``y_true`` once. When ``shared_easy_cutoff`` is set
    (multi-threshold hard-tail scans), the expensive "easy" Spearman score is
    computed **once** at that cutoff and reused for every threshold row; only
    the (small) hard bin is re-scored per cutoff. That matters because on
    Podcast the easy bin is ~90%+ of words and was dominating runtime when
    re-ranked once per threshold.

    Plot semantics with ``shared_easy_cutoff=t0``: each panel shows
    easy ≤ t0 (shared) vs hard > t_i. Words with t0 < cost ≤ t_i are excluded
    from both series — a deliberate "clear easy vs clear hard" contrast.
    """
    data = np.load(npz_path, allow_pickle=True)
    if "model" not in data or "revision" not in data:
        raise KeyError(
            f"{npz_path} has no model/revision fields — rerun the sweep with "
            "--save_word_predictions to regenerate.")
    if npz_id_key not in data:
        raise KeyError(
            f"{npz_path} has no {npz_id_key!r} field — it predates the id-key fix; "
            "rerun the sweep with --save_word_predictions to regenerate."
        )
    words_idx = data[npz_id_key]
    cost = dlt_key.reindex(words_idx).to_numpy(dtype=float)
    oof_all, y_all = data["oof_predictions"], data["y_true"]  # (n_elec, n_words, n_times)
    model = str(data["model"])
    revision = str(data["revision"])
    n_params = float(data["n_params"]) if "n_params" in data else np.nan
    model_short = npz_path.parent.parent.name
    family = npz_path.parent.parent.parent.name

    def _row(cutoff_name, bin_name, per_elec) -> dict:
        return {
            "model": model,
            "revision": revision,
            "n_params": n_params,
            "model_short": model_short,
            "family": family,
            "npz_file": npz_path.name,
            "cutoff_name": cutoff_name,
            "dlt_bin": bin_name,
            "score": float(np.nanmean(per_elec)),
            "n_electrodes": int(np.sum(np.isfinite(per_elec))),
        }

    rows: list[dict] = []

    # Optional shared easy score (computed once).
    shared_easy_score: np.ndarray | None = None
    if shared_easy_cutoff is not None:
        easy_mask = assign_bin(cost, shared_easy_cutoff) == "easy"
        if int(easy_mask.sum()) >= 3:
            shared_easy_score = _score_electrodes_batch(oof_all, y_all, easy_mask, metric)

    for cutoff_name, cutoff in cutoffs.items():
        if shared_easy_score is not None:
            rows.append(_row(cutoff_name, "easy", shared_easy_score))
            hard_mask = assign_bin(cost, cutoff) == "hard"
            if int(hard_mask.sum()) >= 3:
                hard_score = _score_electrodes_batch(oof_all, y_all, hard_mask, metric)
                rows.append(_row(cutoff_name, "hard", hard_score))
        else:
            bin_labels = assign_bin(cost, cutoff)
            for bin_name in ("easy", "hard"):
                mask = bin_labels == bin_name
                if int(mask.sum()) < 3:
                    continue
                per_elec = _score_electrodes_batch(oof_all, y_all, mask, metric)
                rows.append(_row(cutoff_name, bin_name, per_elec))
    return rows


def build_neural_scaling_table_multi_cutoff(
    *,
    sweep_root: Path,
    dlt_cost: pd.DataFrame,
    npz_dirname: str,
    word_key: str = "word_idx",
    npz_id_key: str | None = None,
    metric: str = "spearman",
    perplexity_dir: Path | None = None,
    n_jobs: int = 1,
    cutoffs: dict[object, float],
    shared_easy_cutoff: float | None = None,
    exclude_subjects: tuple[str, ...] = (),
) -> dict[object, pd.DataFrame]:
    """Like ``build_neural_scaling_table`` but computes ALL ``cutoffs`` from a
    single pass over the .npz files.

    ``shared_easy_cutoff``: if set, score the easy bin once at this cutoff and
    reuse it for every threshold (only hard is re-scored per cutoff). Use for
    multi-threshold hard-tail scans — see ``_process_npz_multi_cutoff``.

    ``exclude_subjects``: npz files whose filename contains ``_{subject}.``
    for any subject in this tuple are skipped entirely — e.g. Broderick's
    sub-001 (corrupted EEG epochs; excluded everywhere else in this repo, see
    configs/embedding/neural/broderick.yaml). Word-prediction npz files carry
    no subject-exclusion of their own (only the sweep config does), so a sweep
    root written before/outside that config can still contain sub-001 files.
    """
    npz_id_key = npz_id_key or word_key
    dlt_key = dlt_cost[[word_key, "total"]].drop_duplicates(word_key).set_index(word_key)["total"]
    perplexity_lookup = _load_perplexity_lookup(perplexity_dir)

    npz_files = sorted(sweep_root.rglob(f"{npz_dirname}/*.npz"))
    if exclude_subjects:
        n_before = len(npz_files)
        npz_files = [
            p for p in npz_files
            if not any(f"_{subj}." in p.name for subj in exclude_subjects)
        ]
        logger.info("excluded %d/%d .npz files matching subjects=%s",
                    n_before - len(npz_files), n_before, exclude_subjects)
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found under {sweep_root}/**/{npz_dirname}/ — rerun the sweep "
            "with --save_word_predictions."
        )
    logger.info(
        "Found %d .npz files under %s/**/%s/ (n_jobs=%d, cutoffs=%s, shared_easy=%s)",
        len(npz_files), sweep_root, npz_dirname, n_jobs, list(cutoffs), shared_easy_cutoff,
    )

    if n_jobs == 1:
        results = [
            _process_npz_multi_cutoff(
                p, dlt_key, cutoffs, npz_id_key, metric,
                shared_easy_cutoff=shared_easy_cutoff,
            )
            for p in npz_files
        ]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_process_npz_multi_cutoff)(
                p, dlt_key, cutoffs, npz_id_key, metric,
                shared_easy_cutoff=shared_easy_cutoff,
            )
            for p in npz_files
        )
    rows = [r for sub in results for r in sub]

    per_subject = pd.DataFrame(rows)
    out: dict[object, pd.DataFrame] = {}
    for cutoff_name in cutoffs:
        if per_subject.empty:
            out[cutoff_name] = per_subject
            continue
        sub = per_subject[per_subject["cutoff_name"] == cutoff_name]
        if sub.empty:
            out[cutoff_name] = sub
            continue
        agg = (
            sub.groupby(
                ["model", "revision", "n_params", "model_short", "family", "dlt_bin"],
                dropna=False,
            )["score"]
            .agg(score="mean", score_se="sem", n_subjects="count")
            .reset_index()
        )
        if perplexity_lookup is not None:
            join_cols = ["model", "revision"] if "revision" in perplexity_lookup.columns else ["model"]
            agg = agg.merge(perplexity_lookup, on=join_cols, how="left")
        out[cutoff_name] = agg
    return out


def build_neural_scaling_table(
    *,
    sweep_root: Path,
    dlt_cost: pd.DataFrame,
    npz_dirname: str,
    word_key: str = "word_idx",
    npz_id_key: str | None = None,
    metric: str = "spearman",
    perplexity_dir: Path | None = None,
    n_jobs: int = 1,
    cutoff: float | None = None,
    exclude_subjects: tuple[str, ...] = (),
) -> pd.DataFrame:
    """``word_key`` names the join-id column in the DLT-cost CSV; ``npz_id_key``
    names the corresponding array *inside each .npz* — defaults to ``word_key``
    since every dataset so far happens to use the same name for both (Podcast/
    Broderick: "word_idx"; Nieuwland: "item_number"), but they're conceptually
    distinct in case a future dataset's npz array is named differently.
    """
    cutoff = dlt_cutoff(dlt_cost) if cutoff is None else float(cutoff)
    tables = build_neural_scaling_table_multi_cutoff(
        sweep_root=sweep_root,
        dlt_cost=dlt_cost,
        npz_dirname=npz_dirname,
        word_key=word_key,
        npz_id_key=npz_id_key,
        metric=metric,
        perplexity_dir=perplexity_dir,
        n_jobs=n_jobs,
        cutoffs={"_single": cutoff},
        shared_easy_cutoff=None,
        exclude_subjects=exclude_subjects,
    )
    return tables["_single"]


# ---------------------------------------------------------------------------
# Podcast pooled LME (Design B): one mixed-effects model per lag. Rather than
# picking a single "peak lag" per model (which risks a mild selection bias —
# the peak is chosen by the very quantity, delta_loglik_test, that the DLT
# split then further partitions) this AVERAGES delta_loglik across every
# available lag, per (subject, electrode, word_idx), before joining DLT cost —
# a lag-robust per-word effect size, not contingent on one noisy time point.
# Podcast is one continuous transcript (no story_id/zone), so the join key is
# a flat word_idx — same convention as the ridge encoder's
# podcast_word_predictions/*.npz. Treated like the behavioral path for
# multi-threshold scans (cheap delta_loglik sums, no need for the neural .npz
# path's "shared easy score" optimisation).
# ---------------------------------------------------------------------------

def _lag_averaged_per_word(model_dir: Path, predictions_dirname: str) -> pd.DataFrame | None:
    """Concatenate every available ``predictions/lag*.parquet`` for one model
    and collapse down to ONE value per (subject, word_idx):

    1. Average ``delta_loglik`` across lags, per (subject, electrode,
       word_idx) — held-out membership can differ slightly lag-to-lag (a lag
       can be missing e.g. after a crash), so grouping + mean handles that
       naturally (average over however many lags actually have that row).
    2. Average across electrodes, per (subject, word_idx) — otherwise a
       subject with more recorded electrodes would contribute
       disproportionately more to their own bin total than a subject with
       fewer electrodes, for reasons having nothing to do with word difficulty.
    """
    pred_dir = model_dir / predictions_dirname
    lag_files = sorted(pred_dir.glob("lag*.parquet")) if pred_dir.is_dir() else []
    if not lag_files:
        return None
    parts = [pd.read_parquet(p, columns=["subject", "electrode", "word_idx", "delta_loglik"])
             for p in lag_files]
    all_lags = pd.concat(parts, ignore_index=True)
    lag_avg = (
        all_lags.groupby(["subject", "electrode", "word_idx"], dropna=False)["delta_loglik"]
        .mean()
        .reset_index()
    )
    return (
        lag_avg.groupby(["subject", "word_idx"], dropna=False)["delta_loglik"]
        .mean()
        .reset_index()
    )


def build_podcast_lme_scaling_table_multi_cutoff(
    *,
    sweep_root: Path,
    dlt_cost: pd.DataFrame,
    perplexity_dir: Path | None = None,
    cutoffs: dict[object, float],
    by_lag_filename: str = "podcast_lme_by_lag.csv",
    predictions_dirname: str = "predictions",
    word_key: str = "word_idx",
    shared_easy_cutoff: float | None = None,
) -> dict[object, pd.DataFrame]:
    """Like ``build_behavioral_scaling_table_multi_cutoff`` but for the pooled
    podcast_lme sweep: one lag-averaged parquet build per model, then all
    ``cutoffs`` computed from that single read.

    ``shared_easy_cutoff``: if set, easy is always cost ≤ this fixed value for
    every threshold row (matching the neural .npz path's convention) — only
    the hard tail (cost > cutoffs[name]) moves. Words with
    shared_easy_cutoff < cost ≤ cutoffs[name] are excluded from both bins.
    """
    dlt_key = dlt_cost[[word_key, "total"]].drop_duplicates(word_key)
    n_params_lookup = _registry_n_params_lookup()
    model_id_lookup = _registry_model_id_lookup()
    perplexity_lookup = _load_perplexity_lookup(perplexity_dir)

    by_lag_files = _find_files(sweep_root, by_lag_filename)
    if not by_lag_files:
        raise FileNotFoundError(
            f"No {by_lag_filename} found under {sweep_root} — has the podcast_lme "
            "sweep been run yet?"
        )

    rows_by_cutoff: dict[object, list[pd.DataFrame]] = {name: [] for name in cutoffs}
    for by_lag_path in by_lag_files:
        model_dir = by_lag_path.parent
        per_word = _lag_averaged_per_word(model_dir, predictions_dirname)
        if per_word is None:
            logger.warning("%s: no %s/lag*.parquet found — skipping", model_dir, predictions_dirname)
            continue
        n_lags_seen = len(sorted((model_dir / predictions_dirname).glob("lag*.parquet")))
        per_word = per_word.merge(dlt_key, on=word_key, how="left")
        total = per_word["total"].to_numpy(dtype=float)

        model_short = model_dir.name
        family = model_dir.parent.name
        model_id = model_id_lookup.get(model_short)

        for cutoff_name, cutoff_val in cutoffs.items():
            pw = per_word.copy()
            if shared_easy_cutoff is not None:
                easy_mask = total <= shared_easy_cutoff
                hard_mask = total > cutoff_val
                pw["dlt_bin"] = np.where(easy_mask, "easy", np.where(hard_mask, "hard", None))
            else:
                pw["dlt_bin"] = assign_bin(total, cutoff_val)
            pw = pw.dropna(subset=["dlt_bin"])

            # Mean ± SEM across subjects of each subject's PER-WORD AVERAGE
            # (lag- and electrode-averaged) Δloglik in the bin — not a sum —
            # so bin size (easy is ~24x more words than hard) doesn't by
            # itself determine which bin scores higher; this isolates the
            # per-word effect size difference, which is the actual DLT
            # question (does the model explain hard words MORE PER WORD, not
            # just explain more words because there are more of them).
            per_subj = (
                pw.groupby(["subject", "dlt_bin"], dropna=False)["delta_loglik"]
                .mean()
                .reset_index()
            )
            agg = (
                per_subj.groupby(["dlt_bin"], dropna=False)["delta_loglik"]
                .agg(score="mean", score_se="sem", n_subjects="count")
                .reset_index()
            )
            agg["model_short"] = model_short
            agg["family"] = family
            agg["model"] = model_id
            agg["n_lags_averaged"] = n_lags_seen
            agg["n_params"] = n_params_lookup.get(model_short, np.nan)
            rows_by_cutoff[cutoff_name].append(agg)

    out: dict[object, pd.DataFrame] = {}
    for cutoff_name, rows in rows_by_cutoff.items():
        d = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not d.empty and perplexity_lookup is not None and "model" in d.columns:
            # podcast_lme has no revision/checkpoint dimension — join on model
            # only, taking the "main" row when a model has more than one.
            pp = perplexity_lookup
            if "revision" in pp.columns:
                pp = pp[pp["revision"] == "main"] if (pp["revision"] == "main").any() else pp
            d = d.merge(pp[["model", "perplexity"]].drop_duplicates("model"), on="model", how="left")
        out[cutoff_name] = d
    return out


def build_podcast_lme_scaling_table(
    *,
    sweep_root: Path,
    dlt_cost: pd.DataFrame,
    perplexity_dir: Path | None = None,
    cutoff: float | None = None,
    by_lag_filename: str = "podcast_lme_by_lag.csv",
    predictions_dirname: str = "predictions",
    word_key: str = "word_idx",
) -> pd.DataFrame:
    cutoff = dlt_cutoff(dlt_cost) if cutoff is None else float(cutoff)
    tables = build_podcast_lme_scaling_table_multi_cutoff(
        sweep_root=sweep_root,
        dlt_cost=dlt_cost,
        perplexity_dir=perplexity_dir,
        cutoffs={"_single": cutoff},
        by_lag_filename=by_lag_filename,
        predictions_dirname=predictions_dirname,
        word_key=word_key,
    )
    return tables["_single"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_BIN_COLOR = {"easy": "#2a78d6", "hard": "#e34948"}  # fixed order, matches the dataviz palette
_BAND_ALPHA = 0.25  # fill_between alpha for ±SEM bands (same hue as the line)

# Figure export defaults. Subplot panels are intentionally large (≈3× the
# original 4.2" width) so the multi-threshold grid is wide, not a thin vertical
# band. Always also write a PDF sibling for lossless zoom.
_FIG_DPI = 400
_PANEL_W, _PANEL_H = 12.6, 7.0  # inches per family subplot (3× original width)
_FONT_TITLE = 14
_FONT_TICK = 11
_FONT_LEGEND = 14
_FONT_SUP = 18
_MARKER_SIZE = 6.0
_LINE_WIDTH = 2.0


def _score_ylim(dfs: list[pd.DataFrame]) -> tuple[float, float, float]:
    """Shared y-limits including ±SEM bands when present."""
    lo, hi = np.inf, -np.inf
    for df in dfs:
        if df is None or df.empty or "score" not in df.columns:
            continue
        s = df["score"].to_numpy(dtype=float)
        se = (
            df["score_se"].to_numpy(dtype=float)
            if "score_se" in df.columns
            else np.zeros_like(s)
        )
        se = np.where(np.isfinite(se), se, 0.0)
        valid = np.isfinite(s)
        if not valid.any():
            continue
        lo = min(lo, float(np.nanmin(s[valid] - se[valid])))
        hi = max(hi, float(np.nanmax(s[valid] + se[valid])))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0, 0.08
    pad = (hi - lo) * 0.08 or 1.0
    return lo, hi, pad


def _plot_bin_curve(ax, g: pd.DataFrame, x: str, *, color: str, label: str | None = None) -> None:
    """Line + same-hue ±SEM band (when ``score_se`` is present)."""
    g = g.sort_values(x)
    ax.plot(
        g[x], g["score"],
        marker="o", ms=_MARKER_SIZE, lw=_LINE_WIDTH, color=color, label=label, zorder=2,
    )
    if "score_se" not in g.columns or g["score_se"].isna().all():
        return
    se = g["score_se"].to_numpy(dtype=float)
    se = np.where(np.isfinite(se), se, 0.0)
    y = g["score"].to_numpy(dtype=float)
    ax.fill_between(
        g[x].to_numpy(dtype=float), y - se, y + se,
        color=color, alpha=_BAND_ALPHA, linewidth=0, zorder=1,
    )


def _save_figure(fig, out_path: Path) -> None:
    """Save PNG (preview) + PDF (vector, zoom-safe) next to each other.

    PNG dpi is capped so multi-threshold grids (tens of inches tall/wide) don't
    try to allocate multi-gigapixel rasters. The PDF has no such limit — zoom
    freely there.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w_in, h_in = fig.get_size_inches()
    # Keep the longer PNG edge under ~12k px (Discord / viewers / memory).
    max_px = 12_000
    png_dpi = _FIG_DPI
    if max(w_in, h_in) * png_dpi > max_px:
        png_dpi = max(72, int(max_px / max(w_in, h_in)))

    fig.savefig(out_path, dpi=png_dpi, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info(
        "wrote %s (dpi=%d, %.1f MB) and %s (%.1f MB)",
        out_path, png_dpi, out_path.stat().st_size / 1e6,
        pdf_path, pdf_path.stat().st_size / 1e6,
    )

_SUPERSCRIPT = str.maketrans("0123456789-.", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻·")
_NICE_LOG2_STEPS = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]


def plot_dlt_distribution(
    dlt_cost: pd.DataFrame, *, dataset: str, cutoff: float, cost_col: str = "total", out_path: Path
) -> None:
    """Histogram (log-count, since DLT cost is heavily right-skewed/sparse) +
    CDF of the DLT-cost column, with the fixed easy/hard cutoff (0) marked on
    both — lets you see at a glance how much of the corpus falls in the sparse
    nonzero tail vs at exactly zero cost.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = dlt_cost[cost_col].to_numpy(dtype=float)
    max_val = int(np.nanmax(t))
    bin_edges = np.arange(0, max_val + 2) - 0.5  # integer-centered bins
    counts, edges = np.histogram(t, bins=bin_edges)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [_BIN_COLOR["easy"] if c <= cutoff else _BIN_COLOR["hard"] for c in centers]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.bar(centers, counts, width=0.9, color=colors)
    ax.set_yscale("log")
    ax.set_xlabel(f"DLT {cost_col} cost (integer)")
    ax.set_ylabel("word count (log scale)")
    ax.set_title(f"Histogram (n={len(t)})")
    ax.axvline(cutoff, color="black", ls="--", lw=1)
    ax.text(cutoff, ax.get_ylim()[1], f" cutoff={cutoff:g}", va="top", fontsize=9)

    ax2 = axes[1]
    sorted_vals = np.sort(t)
    cum_frac = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax2.plot(sorted_vals, cum_frac, color="#0b0b0b", lw=1.6)
    ax2.axvline(cutoff, color="black", ls="--", lw=1)
    ax2.axhline(0.5, color="gray", ls=":", lw=1)
    ax2.set_xlabel(f"DLT {cost_col} cost")
    ax2.set_ylabel("cumulative fraction of words")
    ax2.set_title("CDF")

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(True, alpha=0.25)

    fig.suptitle(f"{dataset}: DLT cost distribution (easy = blue, cost=0; hard = red, cost>0)",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out_path)
    plt.close(fig)
    logger.info("[%s] wrote distribution figure -> %s", dataset, out_path)


def _make_pow2_locator(nbins: int = 5):
    """Tick positions evenly spaced in log2 space, snapped to a 'nice' step.

    Plain ``LogLocator(base=2)`` only places ticks at *integer* powers of 2 —
    for a family whose models all sit within one octave (e.g. the hard_Ngram
    baselines, which are deliberately near-identical), that can mean zero
    ticks fall inside the view. This guarantees ~4-6 ticks regardless of how
    narrow the range is, falling back to fractional exponents when needed.
    """
    import math

    import matplotlib.ticker

    class Pow2Locator(matplotlib.ticker.Locator):
        def tick_values(self, vmin: float, vmax: float) -> np.ndarray:
            vmin = max(vmin, 1e-12)
            lo, hi = math.log2(vmin), math.log2(vmax)
            raw_step = (hi - lo) / nbins or 1.0
            step = next((s for s in _NICE_LOG2_STEPS if s >= raw_step), _NICE_LOG2_STEPS[-1])
            start = math.floor(lo / step) * step
            stop = math.ceil(hi / step) * step + step / 2
            exponents = np.arange(start, stop, step)
            return 2.0 ** exponents

        def __call__(self) -> np.ndarray:
            vmin, vmax = self.axis.get_view_interval()
            return self.tick_values(vmin, vmax)

    return Pow2Locator()


def _pow2_label(value: float, _pos: int) -> str:
    """Tick label as '2^n' (unicode superscript) — fractional if the tick step is."""
    import math

    if value <= 0:
        return ""
    exponent = math.log2(value)
    rounded = round(exponent)
    text = str(rounded) if abs(exponent - rounded) < 1e-6 else f"{exponent:.2f}"
    return f"2{text.translate(_SUPERSCRIPT)}"


def plot_difficulty_scaling_by_family(
    df: pd.DataFrame, *, dataset: str, x: str, out_path: Path
) -> None:
    """One subplot per model family, all sharing the same y-axis range, so the
    easy/hard gap is visually comparable across families — not just within one.
    """
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    if x not in df.columns or df[x].isna().all():
        logger.warning("[%s] no %s column available — skipping %s plot", dataset, x, x)
        return
    d = df.dropna(subset=[x, "score", "family"]).copy()
    if d.empty:
        logger.warning("[%s] no rows with both %s and score — skipping plot", dataset, x)
        return

    families = sorted(d["family"].unique())
    n = len(families)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    y_lo, y_hi, y_pad = _score_ylim([d])

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(_PANEL_W * ncols, _PANEL_H * nrows), sharey=True, squeeze=False,
    )
    for i, family in enumerate(families):
        ax = axes[i // ncols][i % ncols]
        sub = d[d["family"] == family]
        for dlt_bin, color in _BIN_COLOR.items():
            g = sub[sub["dlt_bin"] == dlt_bin]
            if g.empty:
                continue
            _plot_bin_curve(ax, g, x, color=color, label=dlt_bin)
        if x == "perplexity":
            # Base-2 log — denser tick spacing (powers of 2) than base-10, so
            # even a narrow perplexity range still gets several readable ticks.
            # Labels show the exponent itself ("2^n"), not the raw value.
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_pow2_label))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.xaxis.set_major_locator(_make_pow2_locator())
            # Zoom to THIS family's own perplexity range — models within a family
            # (e.g. the hard_Ngram baselines) often sit close together, and the
            # default log autoscale otherwise snaps out to the next whole decade,
            # squeezing all the points into a sliver in the middle of the panel.
            x_vals = sub[x].dropna()
            if len(x_vals) and x_vals.min() > 0:
                x_lo, x_hi = x_vals.min(), x_vals.max()
                margin = (x_hi / x_lo) ** 0.15 if x_hi > x_lo else 1.05
                ax.set_xlim(x_lo / margin, x_hi * margin)
        else:
            ax.set_xscale("log")
        ax.set_title(family, fontsize=_FONT_TITLE)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=_FONT_TICK)

    # Blank any unused grid cells
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles = [
        plt.Line2D([0], [0], color=c, marker="o", ms=_MARKER_SIZE, lw=_LINE_WIDTH, label=b)
        for b, c in _BIN_COLOR.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=_FONT_LEGEND,
               bbox_to_anchor=(0.5, 1.02))
    xlabel = "Model size (parameters)" if x == "n_params" else "Perplexity"
    fig.supxlabel(xlabel, fontsize=_FONT_TITLE)
    fig.supylabel("score (mean ± SEM across subjects)", fontsize=_FONT_TITLE)
    fig.suptitle(
        f"{dataset}: scaling by DLT-cost difficulty (easy vs hard), by family — x = {xlabel.lower()}",
        fontsize=_FONT_SUP, y=1.05,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out_path)
    plt.close(fig)
    logger.info("[%s] wrote %s (%d families, shared y=[%.1f, %.1f])",
                dataset, out_path, n, y_lo - y_pad, y_hi + y_pad)


def plot_threshold_grid_by_family(
    *,
    df_by_threshold: list[tuple[float, pd.DataFrame]],
    dataset: str,
    x: str,
    out_path: Path,
    shared_easy_cutoff: float | None = None,
    bin_counts: dict[float, dict[str, float | int]] | None = None,
) -> None:
    """Stack multiple hard-threshold splits in one figure.

    ``shared_easy_cutoff``: if set, titles/legend note that the easy curve is
    shared (cost ≤ that cutoff) while only hard > t varies per row — matches
    the neural multi-threshold scoring optimisation.

    ``bin_counts``: optional ``{threshold: {n_easy, n_hard, ...}}`` from
    ``log_threshold_bin_counts`` — annotated in each row's panel titles.
    """
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    valid = [(thr, df) for thr, df in df_by_threshold if df is not None and not df.empty]
    if not valid:
        logger.warning("[%s] no non-empty threshold tables — skipping %s", dataset, out_path)
        return

    if not any(x in df.columns and not df[x].isna().all() for _, df in valid):
        logger.warning("[%s] no %s column available in any threshold table — skipping %s plot",
                        dataset, x, x)
        return
    # Drop rows missing x/score/family per-threshold (not the whole table) — a
    # dataset without a perplexity_dir simply has no "perplexity" column at all,
    # which dropna(subset=...) would otherwise raise a KeyError on.
    valid = [
        (thr, df.dropna(subset=[c for c in (x, "score", "family") if c in df.columns]))
        for thr, df in valid
    ]
    valid = [(thr, df) for thr, df in valid if x in df.columns and not df.empty]
    if not valid:
        logger.warning("[%s] no rows with both %s and score — skipping %s plot", dataset, x, x)
        return

    families = sorted(set(
        fam
        for _, df in valid
        for fam in df["family"].dropna().unique().tolist()
    ))
    if not families:
        logger.warning("[%s] no families across thresholds — skipping %s", dataset, out_path)
        return

    y_lo, y_hi, y_pad = _score_ylim([df for _, df in valid])

    n = len(families)
    ncols = min(4, n)
    fam_rows = math.ceil(n / ncols)
    total_rows = fam_rows * len(valid)

    fig, axes = plt.subplots(
        total_rows, ncols,
        figsize=(_PANEL_W * ncols, _PANEL_H * total_rows),
        sharey=True, squeeze=False,
    )

    for t_idx, (thr, df_thr) in enumerate(valid):
        d = df_thr
        row_offset = t_idx * fam_rows
        counts = (bin_counts or {}).get(float(thr), {})
        n_hard = counts.get("n_hard")
        n_easy = counts.get("n_easy")
        if shared_easy_cutoff is not None:
            thr_label = f"easy≤{shared_easy_cutoff:g}, hard>{thr:g}"
        else:
            thr_label = f"hard > {thr:g}"
        if n_hard is not None and n_easy is not None:
            thr_label = f"{thr_label}\nn_easy={n_easy}, n_hard={n_hard}"
        elif n_hard is not None:
            thr_label = f"{thr_label}\nn_hard={n_hard}"

        for i, family in enumerate(families):
            ax = axes[row_offset + i // ncols][i % ncols]
            sub = d[d["family"] == family]

            for dlt_bin, color in _BIN_COLOR.items():
                g = sub[sub["dlt_bin"] == dlt_bin]
                if g.empty:
                    continue
                _plot_bin_curve(ax, g, x, color=color)

            if x == "perplexity":
                ax.set_xscale("log", base=2)
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_pow2_label))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.xaxis.set_major_locator(_make_pow2_locator())
                x_vals = sub[x].dropna()
                if len(x_vals) and x_vals.min() > 0:
                    x_lo, x_hi = x_vals.min(), x_vals.max()
                    margin = (x_hi / x_lo) ** 0.15 if x_hi > x_lo else 1.05
                    ax.set_xlim(x_lo / margin, x_hi * margin)
            else:
                ax.set_xscale("log")

            title = f"{family}\n({thr_label})" if i < ncols else family
            ax.set_title(title, fontsize=_FONT_TITLE)
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=_FONT_TICK)

        for j in range(n, fam_rows * ncols):
            axes[row_offset + j // ncols][j % ncols].axis("off")

    if shared_easy_cutoff is not None:
        handles = [
            plt.Line2D([0], [0], color=_BIN_COLOR["easy"], marker="o", ms=_MARKER_SIZE, lw=_LINE_WIDTH,
                       label=f"easy (cost≤{shared_easy_cutoff:g}, shared)"),
            plt.Line2D([0], [0], color=_BIN_COLOR["hard"], marker="o", ms=_MARKER_SIZE, lw=_LINE_WIDTH,
                       label="hard (cost>t, per row)"),
        ]
    else:
        handles = [
            plt.Line2D([0], [0], color=c, marker="o", ms=_MARKER_SIZE, lw=_LINE_WIDTH, label=b)
            for b, c in _BIN_COLOR.items()
        ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=_FONT_LEGEND,
               bbox_to_anchor=(0.5, 1.01))
    xlabel = "Model size (parameters)" if x == "n_params" else "Perplexity"
    fig.supxlabel(xlabel, fontsize=_FONT_TITLE)
    fig.supylabel("score (mean ± SEM across subjects)", fontsize=_FONT_TITLE)
    if shared_easy_cutoff is not None:
        fig.suptitle(
            f"{dataset}: hard-tail DLT thresholds (shared easy≤{shared_easy_cutoff:g}), "
            f"by family — x = {xlabel.lower()}",
            fontsize=_FONT_SUP, y=1.02,
        )
    else:
        fig.suptitle(
            f"{dataset}: scaling by DLT difficulty across hard thresholds, by family — x = {xlabel.lower()}",
            fontsize=_FONT_SUP, y=1.02,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out_path)
    plt.close(fig)
    logger.info("[%s] wrote %s (threshold rows=%d, families=%d, dpi=%d)",
                dataset, out_path, len(valid), n, _FIG_DPI)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BEHAVIORAL = {
    "naturalstories": dict(per_word_filename="spr_lme_per_word.csv",
                           scores_filename="spr_lme_checkpoint_scores.csv"),
    "geco": dict(per_word_filename="geco_FP_lme_per_word.csv",
                 scores_filename="geco_lme_checkpoint_scores.csv"),
    "provo": dict(per_word_filename="provo_FP_lme_per_word.csv",
                  scores_filename="provo_lme_checkpoint_scores.csv"),
    "ucl_reading": dict(per_word_filename="ucl_reading_spr_lme_per_word.csv",
                        scores_filename="ucl_reading_spr_lme_checkpoint_scores.csv"),
    "meco": dict(per_word_filename="meco_FP_lme_per_word.csv",
                 scores_filename="meco_lme_checkpoint_scores.csv"),
    "zuco_reading_sr": dict(per_word_filename="zuco_reading_SR_lme_per_word.csv",
                            scores_filename="zuco_reading_trt_lme_checkpoint_scores.csv"),
    "zuco_reading_nr": dict(per_word_filename="zuco_reading_NR_lme_per_word.csv",
                            scores_filename="zuco_reading_trt_lme_checkpoint_scores.csv"),
    "zuco_reading_tsr": dict(per_word_filename="zuco_reading_TSR_lme_per_word.csv",
                             scores_filename="zuco_reading_trt_lme_checkpoint_scores.csv"),
}
NEURAL = {
    "podcast": dict(npz_dirname="podcast_word_predictions", word_key="word_idx"),
    "broderick": dict(npz_dirname="broderick_word_predictions", word_key="word_idx",
                      exclude_subjects=("sub-001",)),
    "nieuwland": dict(npz_dirname="nieuwland_word_predictions", word_key="item_number"),
}
PODCAST_LME = {
    "podcast_lme": dict(),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(BEHAVIORAL) + list(NEURAL) + list(PODCAST_LME))
    ap.add_argument("--sweep_root", default=None, type=Path,
                     help="Root of checkpoint-sweep outputs (per-word CSVs or .npz dirs). "
                          "Not required when --scores_csv is set (replot-only).")
    ap.add_argument("--dlt_cost", required=True, type=Path)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "dlt_difficulty"))
    ap.add_argument("--perplexity_dir", default=None, type=Path,
                     help="e.g. outputs_jeanzay/geco/perplexity/ — the per-model perplexity.json "
                          "tree (NOT the *_perplexity_summary.csv, which was found to have wrong, "
                          "collapsed-to-identical values for at least the hard_Ngram families). "
                          "Enables the by-perplexity figure alongside the by-n_params one.")
    ap.add_argument("--n_jobs", type=int, default=1,
                     help="parallel workers over .npz files (neural datasets). "
                          "Use e.g. 30 on a many-core node; -1 = all cores.")
    ap.add_argument(
        "--hard_gt_thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Optional hard-threshold scan (hard > t), one subplot row per threshold, all in a "
             "single figure. Works for BOTH behavioral (per-word CSV) and neural (.npz) datasets "
             "— e.g. --hard_gt_thresholds 1 2 3 4 5. Without this, falls back to the single fixed "
             "cutoff at 0 (easy = no DLT cost, hard = any cost).",
    )
    ap.add_argument(
        "--scores_csv",
        type=Path,
        default=None,
        help="Replot from an existing difficulty-scores CSV (skip recomputing from sweep_root). "
             "For hard_gt grids the CSV must have a hard_gt_threshold column. SEM bands are drawn "
             "when score_se is present.",
    )
    args = ap.parse_args()

    if args.scores_csv is None and args.sweep_root is None:
        ap.error("one of --sweep_root or --scores_csv is required")

    dlt_cost = pd.read_csv(args.dlt_cost)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dlt_stats = log_dlt_distribution(dlt_cost, dataset=args.dataset)
    import json
    (out_dir / f"{args.dataset}_dlt_stats.json").write_text(json.dumps(dlt_stats, indent=2))
    plot_dlt_distribution(
        dlt_cost, dataset=args.dataset, cutoff=dlt_stats["cutoff"],
        out_path=out_dir / f"{args.dataset}_dlt_distribution.png",
    )

    is_behavioral = args.dataset in BEHAVIORAL
    is_podcast_lme = args.dataset in PODCAST_LME
    df_by_threshold: list[tuple[float, pd.DataFrame]] | None = None
    # Neural (.npz) and podcast_lme multi-threshold scans share one easy bin
    # (cost ≤ 0) and only vary the hard tail — for the .npz path this also
    # avoids re-ranking the ~65-90% "easy" words once per threshold (that was
    # dominating runtime); for podcast_lme it's just the more sensible
    # semantics (easy always means "zero DLT cost", not "cost ≤ whatever the
    # hard threshold happens to be this row"). Behavioral LME sums are cheap
    # AND don't have this asymmetric-bin-size framing in the same way, so they
    # keep the classic easy≤t / hard>t split per threshold.
    shared_easy_cutoff: float | None = None
    bin_counts: dict[float, dict[str, float | int]] | None = None

    if args.scores_csv is not None:
        # Replot-only path: reuse a previously written difficulty-scores CSV
        # (optionally with score_se for ±SEM bands).
        df = pd.read_csv(args.scores_csv)
        logger.info("[%s] loaded %d rows from --scores_csv %s", args.dataset, len(df), args.scores_csv)
        if "hard_gt_threshold" in df.columns:
            if not is_behavioral:
                shared_easy_cutoff = 0.0
            thr_vals = (
                [float(t) for t in args.hard_gt_thresholds]
                if args.hard_gt_thresholds
                else sorted(df["hard_gt_threshold"].dropna().unique().tolist())
            )
            bin_counts = log_threshold_bin_counts(
                dlt_cost, thr_vals, dataset=args.dataset,
                shared_easy_cutoff=shared_easy_cutoff,
            )
            (out_dir / f"{args.dataset}_dlt_threshold_bin_counts.json").write_text(
                json.dumps({str(k): v for k, v in bin_counts.items()}, indent=2)
            )
            df_by_threshold = []
            for thr in thr_vals:
                df_thr = df[df["hard_gt_threshold"] == thr].copy()
                df_by_threshold.append((thr, df_thr))
            args.hard_gt_thresholds = thr_vals  # ensure plot branch below
            if "score_se" not in df.columns:
                logger.warning(
                    "[%s] scores CSV has no score_se — plots will have no ±SEM bands "
                    "(recompute without --scores_csv to add them)",
                    args.dataset,
                )
        elif args.hard_gt_thresholds:
            ap.error("--scores_csv has no hard_gt_threshold column but --hard_gt_thresholds was set")
    elif args.hard_gt_thresholds:
        thr_vals = [float(t) for t in args.hard_gt_thresholds]
        shared_easy_cutoff = None if is_behavioral else 0.0
        bin_counts = log_threshold_bin_counts(
            dlt_cost, thr_vals, dataset=args.dataset,
            shared_easy_cutoff=shared_easy_cutoff,
        )
        (out_dir / f"{args.dataset}_dlt_threshold_bin_counts.json").write_text(
            json.dumps({str(k): v for k, v in bin_counts.items()}, indent=2)
        )
        if is_behavioral:
            tables = build_behavioral_scaling_table_multi_cutoff(
                sweep_root=args.sweep_root,
                dlt_cost=dlt_cost,
                perplexity_dir=args.perplexity_dir,
                cutoffs={thr: thr for thr in thr_vals},
                **BEHAVIORAL[args.dataset],
            )
        elif is_podcast_lme:
            tables = build_podcast_lme_scaling_table_multi_cutoff(
                sweep_root=args.sweep_root,
                dlt_cost=dlt_cost,
                perplexity_dir=args.perplexity_dir,
                cutoffs={thr: thr for thr in thr_vals},
                shared_easy_cutoff=shared_easy_cutoff,
                **PODCAST_LME[args.dataset],
            )
        else:
            # Neural (.npz) path — joblib Parallel over files (--n_jobs);
            # each .npz decompressed once; easy scored once at cost≤0.
            tables = build_neural_scaling_table_multi_cutoff(
                sweep_root=args.sweep_root,
                dlt_cost=dlt_cost,
                perplexity_dir=args.perplexity_dir,
                n_jobs=args.n_jobs,
                cutoffs={thr: thr for thr in thr_vals},
                shared_easy_cutoff=shared_easy_cutoff,
                **NEURAL[args.dataset],
            )
        df_by_threshold = []
        for thr in thr_vals:
            df_thr = tables[thr]
            if not df_thr.empty:
                df_thr["hard_gt_threshold"] = thr
            df_by_threshold.append((thr, df_thr))
        df = pd.concat([d for _, d in df_by_threshold], ignore_index=True)
    elif is_behavioral:
        df = build_behavioral_scaling_table(sweep_root=args.sweep_root, dlt_cost=dlt_cost,
                                             perplexity_dir=args.perplexity_dir,
                                             **BEHAVIORAL[args.dataset])
    elif is_podcast_lme:
        df = build_podcast_lme_scaling_table(sweep_root=args.sweep_root, dlt_cost=dlt_cost,
                                              perplexity_dir=args.perplexity_dir,
                                              **PODCAST_LME[args.dataset])
    else:
        df = build_neural_scaling_table(sweep_root=args.sweep_root, dlt_cost=dlt_cost,
                                         perplexity_dir=args.perplexity_dir,
                                         n_jobs=args.n_jobs,
                                         **NEURAL[args.dataset])

    if args.hard_gt_thresholds:
        csv_path = out_dir / f"{args.dataset}_dlt_difficulty_scores_multi_threshold.csv"
        if args.scores_csv is None:
            df.to_csv(csv_path, index=False)
            logger.info("[%s] wrote %d rows -> %s", args.dataset, len(df), csv_path)
        else:
            logger.info("[%s] replotting from %s (not rewriting scores CSV)", args.dataset, args.scores_csv)

        plot_threshold_grid_by_family(
            df_by_threshold=df_by_threshold or [],
            dataset=args.dataset,
            x="n_params",
            shared_easy_cutoff=shared_easy_cutoff,
            bin_counts=bin_counts,
            out_path=out_dir / f"{args.dataset}_dlt_difficulty_by_params_hard_gt_grid.png",
        )
        plot_threshold_grid_by_family(
            df_by_threshold=df_by_threshold or [],
            dataset=args.dataset,
            x="perplexity",
            shared_easy_cutoff=shared_easy_cutoff,
            bin_counts=bin_counts,
            out_path=out_dir / f"{args.dataset}_dlt_difficulty_by_perplexity_hard_gt_grid.png",
        )
    else:
        csv_path = out_dir / f"{args.dataset}_dlt_difficulty_scores.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.scores_csv is None:
            df.to_csv(csv_path, index=False)
            logger.info("[%s] wrote %d rows -> %s", args.dataset, len(df), csv_path)
        else:
            logger.info("[%s] replotting from %s (not rewriting scores CSV)", args.dataset, args.scores_csv)

        plot_difficulty_scaling_by_family(
            df, dataset=args.dataset, x="n_params",
            out_path=out_dir / f"{args.dataset}_dlt_difficulty_by_params.png",
        )
        plot_difficulty_scaling_by_family(
            df, dataset=args.dataset, x="perplexity",
            out_path=out_dir / f"{args.dataset}_dlt_difficulty_by_perplexity.png",
        )


if __name__ == "__main__":
    main()
