#!/usr/bin/env python3
"""DLT easy/hard difficulty-scaling CSVs for the three npz schemas NOT covered
by ``dlt/difficulty_scaling.py``'s ``build_neural_scaling_table_multi_cutoff``
(that one assumes the standard ``(n_elec, n_words, n_times)`` OOF-prediction
shape used by Podcast/Broderick/Nieuwland):

  rt_embed      — RT-prediction embedding tracks (Provo, MeCo, UCL Reading,
                   ZuCo Reading SR/NR/TSR, Natural Stories). One .npz per
                   subject: oof_predictions (n_layers, n_trials), rt_true
                   (n_trials,), story_id/zone join keys, NO n_params/y_true.
                   A "best layer" is picked per subject from the FULL (all
                   words) Spearman correlation, then easy/hard bins are scored
                   at that one layer (mirrors the LME per-word tables, which
                   also commit to a single upstream-chosen layer).
  fedorenko2016 — one .npz per model: oof_predictions (n_layers, n_words,
                   n_electrodes), y_true (n_words, n_electrodes), join key
                   stimulus_id. Electrodes pool across subjects (subject_uid).
                   Best layer picked from the full-word per-electrode score.

Output convention matches the rest of the DLT pipeline: writes
``{dataset}_dlt_difficulty_scores_multi_threshold.csv`` to --out_dir with
columns model, revision, n_params, model_short, family, dlt_bin, score,
score_se, hard_gt_threshold[, perplexity] — easy is always shared at cost<=0,
only the hard tail varies across --hard_gt_thresholds. Also writes the
threshold bin-count JSON and the by-params/by-perplexity grid PNGs, reusing
plot_dlt_difficulty_scaling.plot_threshold_grid_by_family.

Usage
-----
    .venv/bin/python scripts/dlt/difficulty_scores.py \\
        --dataset provo_embed --sweep_root outputs_jeanzay/provo/embedding_sweep \\
        --npz_dirname provo_embed_word_predictions \\
        --dlt_cost outputs/dlt/provo_dlt_cost.csv \\
        --hard_gt_thresholds 1 2 3 5 7
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from scripts.dlt.difficulty_scaling import (  # noqa: E402
    REPO_ROOT,
    _load_perplexity_lookup,
    _registry_n_params_lookup,
    _score_electrodes_batch,
    assign_bin,
    dlt_cutoff,
    log_dlt_distribution,
    log_threshold_bin_counts,
    plot_dlt_distribution,
    plot_threshold_grid_by_family,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RT-embedding schema (Provo, MeCo, UCL Reading, ZuCo Reading, Natural Stories)
# ---------------------------------------------------------------------------

def _process_rt_embed_npz(
    npz_path: Path, dlt_key: pd.Series, cutoffs: dict[object, float],
    n_params_lookup: dict[str, float], *, shared_easy_cutoff: float | None,
) -> list[dict]:
    data = np.load(npz_path, allow_pickle=True)
    rt = np.asarray(data["rt_true"], dtype=float)
    story_id = np.asarray(data["story_id"])
    zone = np.asarray(data["zone"])
    model = str(data["model"])
    revision = str(data["revision"])
    model_short = npz_path.parent.parent.name
    family = npz_path.parent.parent.parent.name

    idx = pd.MultiIndex.from_arrays([story_id, zone])
    cost = dlt_key.reindex(idx).to_numpy(dtype=float)
    valid_rt = np.isfinite(rt)

    if "oof_predictions" in data:
        # Current schema: (n_layers, n_trials) — pick the best layer from the
        # full (all-words) Spearman correlation.
        oof = np.asarray(data["oof_predictions"])
        n_layers = oof.shape[0]
        full_scores = np.full(n_layers, np.nan)
        for layer in range(n_layers):
            o = oof[layer]
            m = valid_rt & np.isfinite(o)
            if int(m.sum()) >= 3:
                r = spearmanr(o[m], rt[m]).correlation
                full_scores[layer] = r if np.isfinite(r) else np.nan
        if np.all(np.isnan(full_scores)):
            return []
        best_layer = int(np.nanargmax(full_scores))
        o_best = oof[best_layer]
    elif "rt_pred_oof" in data:
        # Older schema (some large/slow models never got rerun with the
        # multi-layer save): a single already-chosen-best-layer OOF
        # prediction vector, with the chosen layer recorded separately.
        o_best = np.asarray(data["rt_pred_oof"], dtype=float)
        best_layer = int(data["layer"]) if "layer" in data else -1
    else:
        return []

    def _bin_score(mask: np.ndarray) -> float | None:
        m = mask & valid_rt & np.isfinite(o_best)
        if int(m.sum()) < 3:
            return None
        r = spearmanr(o_best[m], rt[m]).correlation
        return float(r) if np.isfinite(r) else None

    base = dict(
        model=model, revision=revision, model_short=model_short, family=family,
        n_params=n_params_lookup.get(model_short, np.nan),
        subject=npz_path.stem, best_layer=best_layer,
    )
    rows: list[dict] = []
    shared_easy_score = None
    if shared_easy_cutoff is not None:
        easy_mask = assign_bin(cost, shared_easy_cutoff) == "easy"
        shared_easy_score = _bin_score(easy_mask)

    for cutoff_name, cutoff in cutoffs.items():
        if shared_easy_score is not None:
            rows.append({**base, "cutoff_name": cutoff_name, "dlt_bin": "easy", "score": shared_easy_score})
            hard_mask = assign_bin(cost, cutoff) == "hard"
            s = _bin_score(hard_mask)
            if s is not None:
                rows.append({**base, "cutoff_name": cutoff_name, "dlt_bin": "hard", "score": s})
        else:
            bin_labels = assign_bin(cost, cutoff)
            for bin_name in ("easy", "hard"):
                s = _bin_score(bin_labels == bin_name)
                if s is not None:
                    rows.append({**base, "cutoff_name": cutoff_name, "dlt_bin": bin_name, "score": s})
    return rows


def build_rt_embed_scaling_table_multi_cutoff(
    *, sweep_root: Path, dlt_cost: pd.DataFrame, npz_dirname: str,
    perplexity_dir: Path | None = None, cutoffs: dict[object, float],
    shared_easy_cutoff: float | None = None,
) -> dict[object, pd.DataFrame]:
    dlt_key = (
        dlt_cost[["story_id", "zone", "total"]]
        .drop_duplicates(["story_id", "zone"])
        .set_index(["story_id", "zone"])["total"]
    )
    n_params_lookup = _registry_n_params_lookup()
    perplexity_lookup = _load_perplexity_lookup(perplexity_dir)

    npz_files = sorted(sweep_root.rglob(f"{npz_dirname}/*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz found under {sweep_root}/**/{npz_dirname}/")
    logger.info("found %d .npz files under %s/**/%s/", len(npz_files), sweep_root, npz_dirname)

    all_rows: list[dict] = []
    for p in npz_files:
        all_rows.extend(_process_rt_embed_npz(
            p, dlt_key, cutoffs, n_params_lookup, shared_easy_cutoff=shared_easy_cutoff,
        ))
    if not all_rows:
        return {name: pd.DataFrame() for name in cutoffs}

    raw = pd.DataFrame(all_rows)
    out: dict[object, pd.DataFrame] = {}
    for cutoff_name in cutoffs:
        sub = raw[raw["cutoff_name"] == cutoff_name]
        if sub.empty:
            out[cutoff_name] = pd.DataFrame()
            continue
        agg = (
            sub.groupby(["model", "revision", "model_short", "family", "dlt_bin"], dropna=False)["score"]
            .agg(score="mean", score_se="sem", n_subjects="count")
            .reset_index()
        )
        agg["n_params"] = agg["model_short"].map(n_params_lookup)
        if perplexity_lookup is not None:
            join_cols = ["model", "revision"] if "revision" in perplexity_lookup.columns else ["model"]
            agg = agg.merge(perplexity_lookup, on=join_cols, how="left")
        out[cutoff_name] = agg
    return out


# ---------------------------------------------------------------------------
# Fedorenko2016 schema: one .npz per model, layer axis, electrodes pool subjects
# ---------------------------------------------------------------------------

def _process_fedorenko2016_npz(
    npz_path: Path, dlt_key: pd.Series, cutoffs: dict[object, float],
    n_params_lookup: dict[str, float], *, shared_easy_cutoff: float | None, metric: str = "spearman",
) -> list[dict]:
    data = np.load(npz_path, allow_pickle=True)
    oof_all = np.asarray(data["oof_predictions"])  # (n_layers, n_words, n_electrodes)
    y_true = np.asarray(data["y_true"])  # (n_words, n_electrodes)
    stimulus_id = np.asarray(data["stimulus_id"])
    subject_uid = np.asarray(data["subject_uid"])
    model = str(data["model"])
    revision = str(data["revision"])
    model_short = npz_path.parent.parent.name
    family = npz_path.parent.parent.parent.name
    n_subjects = int(len(np.unique(subject_uid)))

    cost = dlt_key.reindex(stimulus_id).to_numpy(dtype=float)
    n_layers = oof_all.shape[0]
    all_valid = np.ones(oof_all.shape[1], dtype=bool)

    # oof_all[l]: (n_words, n_electrodes) -> transpose to (n_elec, n_words, 1)
    y_t = y_true.T[:, :, None]

    def _per_elec(layer: int, mask: np.ndarray) -> np.ndarray:
        o_t = oof_all[layer].T[:, :, None]
        return _score_electrodes_batch(o_t, y_t, mask, metric)

    full_scores = np.full(n_layers, np.nan)
    for layer in range(n_layers):
        pe = _per_elec(layer, all_valid)
        if np.any(np.isfinite(pe)):
            full_scores[layer] = float(np.nanmean(pe))
    if np.all(np.isnan(full_scores)):
        return []
    best_layer = int(np.nanargmax(full_scores))

    base = dict(
        model=model, revision=revision, model_short=model_short, family=family,
        n_params=n_params_lookup.get(model_short, np.nan),
        n_subjects=n_subjects, best_layer=best_layer,
    )

    def _row(cutoff_name, bin_name, per_elec) -> dict:
        finite = per_elec[np.isfinite(per_elec)]
        n = len(finite)
        return {
            **base, "cutoff_name": cutoff_name, "dlt_bin": bin_name,
            "score": float(np.mean(finite)) if n else np.nan,
            "score_se": float(np.std(finite, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "n_electrodes": n,
        }

    rows: list[dict] = []
    shared_easy_per_elec = None
    if shared_easy_cutoff is not None:
        easy_mask = assign_bin(cost, shared_easy_cutoff) == "easy"
        if int(easy_mask.sum()) >= 3:
            shared_easy_per_elec = _per_elec(best_layer, easy_mask)

    for cutoff_name, cutoff in cutoffs.items():
        if shared_easy_per_elec is not None:
            rows.append(_row(cutoff_name, "easy", shared_easy_per_elec))
            hard_mask = assign_bin(cost, cutoff) == "hard"
            if int(hard_mask.sum()) >= 3:
                rows.append(_row(cutoff_name, "hard", _per_elec(best_layer, hard_mask)))
        else:
            bin_labels = assign_bin(cost, cutoff)
            for bin_name in ("easy", "hard"):
                mask = bin_labels == bin_name
                if int(mask.sum()) >= 3:
                    rows.append(_row(cutoff_name, bin_name, _per_elec(best_layer, mask)))
    return rows


def build_fedorenko2016_scaling_table_multi_cutoff(
    *, sweep_root: Path, dlt_cost: pd.DataFrame, npz_dirname: str = "fedorenko2016_word_predictions",
    perplexity_dir: Path | None = None, cutoffs: dict[object, float],
    shared_easy_cutoff: float | None = None, metric: str = "spearman",
) -> dict[object, pd.DataFrame]:
    dlt_key = dlt_cost[["stimulus_id", "total"]].drop_duplicates("stimulus_id").set_index("stimulus_id")["total"]
    n_params_lookup = _registry_n_params_lookup()
    perplexity_lookup = _load_perplexity_lookup(perplexity_dir)

    npz_files = sorted(sweep_root.rglob(f"{npz_dirname}/*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz found under {sweep_root}/**/{npz_dirname}/")
    logger.info("found %d .npz files under %s/**/%s/", len(npz_files), sweep_root, npz_dirname)

    rows_by_cutoff: dict[object, list[dict]] = {name: [] for name in cutoffs}
    for p in npz_files:
        for row in _process_fedorenko2016_npz(
            p, dlt_key, cutoffs, n_params_lookup, shared_easy_cutoff=shared_easy_cutoff, metric=metric,
        ):
            rows_by_cutoff[row["cutoff_name"]].append(row)

    out: dict[object, pd.DataFrame] = {}
    for cutoff_name, rows in rows_by_cutoff.items():
        d = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not d.empty and perplexity_lookup is not None:
            join_cols = ["model", "revision"] if "revision" in perplexity_lookup.columns else ["model"]
            d = d.merge(perplexity_lookup, on=join_cols, how="left")
        out[cutoff_name] = d
    return out



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

RT_EMBED = {
    "provo_embed": dict(npz_dirname="provo_embed_word_predictions"),
    "meco_embed": dict(npz_dirname="meco_embed_word_predictions"),
    "ucl_reading_embed": dict(npz_dirname="ucl_reading_embed_word_predictions"),
    "zuco_reading_sr_embed": dict(npz_dirname="zuco_reading_embed_word_predictions"),
    "zuco_reading_nr_embed": dict(npz_dirname="zuco_reading_embed_word_predictions"),
    "zuco_reading_tsr_embed": dict(npz_dirname="zuco_reading_embed_word_predictions"),
    "naturalstories_embed": dict(npz_dirname="naturalstories_embed_word_predictions"),
}
FEDORENKO2016 = {"fedorenko2016": dict()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                     choices=list(RT_EMBED) + list(FEDORENKO2016))
    ap.add_argument("--sweep_root", required=True, type=Path)
    ap.add_argument("--dlt_cost", required=True, type=Path)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "dlt_difficulty"))
    ap.add_argument("--perplexity_dir", default=None, type=Path)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--hard_gt_thresholds", type=float, nargs="+", required=True)
    args = ap.parse_args()

    dlt_cost = pd.read_csv(args.dlt_cost)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dlt_stats = log_dlt_distribution(dlt_cost, dataset=args.dataset)
    (out_dir / f"{args.dataset}_dlt_stats.json").write_text(json.dumps(dlt_stats, indent=2))
    plot_dlt_distribution(
        dlt_cost, dataset=args.dataset, cutoff=dlt_stats["cutoff"],
        out_path=out_dir / f"{args.dataset}_dlt_distribution.png",
    )

    thr_vals = [float(t) for t in args.hard_gt_thresholds]
    shared_easy_cutoff = 0.0
    bin_counts = log_threshold_bin_counts(
        dlt_cost, thr_vals, dataset=args.dataset, shared_easy_cutoff=shared_easy_cutoff,
    )
    (out_dir / f"{args.dataset}_dlt_threshold_bin_counts.json").write_text(
        json.dumps({str(k): v for k, v in bin_counts.items()}, indent=2)
    )
    cutoffs = {thr: thr for thr in thr_vals}

    if args.dataset in RT_EMBED:
        tables = build_rt_embed_scaling_table_multi_cutoff(
            sweep_root=args.sweep_root, dlt_cost=dlt_cost,
            perplexity_dir=args.perplexity_dir, cutoffs=cutoffs,
            shared_easy_cutoff=shared_easy_cutoff, **RT_EMBED[args.dataset],
        )
    else:
        tables = build_fedorenko2016_scaling_table_multi_cutoff(
            sweep_root=args.sweep_root, dlt_cost=dlt_cost,
            perplexity_dir=args.perplexity_dir, cutoffs=cutoffs,
            shared_easy_cutoff=shared_easy_cutoff, **FEDORENKO2016[args.dataset],
        )

    df_by_threshold = []
    for thr in thr_vals:
        df_thr = tables[thr]
        if not df_thr.empty:
            df_thr = df_thr.copy()
            df_thr["hard_gt_threshold"] = thr
        df_by_threshold.append((thr, df_thr))
    df = pd.concat([d for _, d in df_by_threshold], ignore_index=True)

    csv_path = out_dir / f"{args.dataset}_dlt_difficulty_scores_multi_threshold.csv"
    df.to_csv(csv_path, index=False)
    logger.info("[%s] wrote %d rows -> %s", args.dataset, len(df), csv_path)

    plot_threshold_grid_by_family(
        df_by_threshold=df_by_threshold, dataset=args.dataset, x="n_params",
        shared_easy_cutoff=shared_easy_cutoff, bin_counts=bin_counts,
        out_path=out_dir / f"{args.dataset}_dlt_difficulty_by_params_hard_gt_grid.png",
    )
    plot_threshold_grid_by_family(
        df_by_threshold=df_by_threshold, dataset=args.dataset, x="perplexity",
        shared_easy_cutoff=shared_easy_cutoff, bin_counts=bin_counts,
        out_path=out_dir / f"{args.dataset}_dlt_difficulty_by_perplexity_hard_gt_grid.png",
    )


if __name__ == "__main__":
    main()
