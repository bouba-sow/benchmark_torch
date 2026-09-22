#!/usr/bin/env python3
"""Scaling law for behavioral-embedding models fit on the SUBJECT-AVERAGED
reading-time signal (one ridge fit per model on the mean RT across subjects),
instead of the usual per-subject-then-averaged convention.

Reads the "_average" sibling sweep dirs (same file layout as the normal
per-subject embedding sweeps, just missing *_by_subject.csv):
  outputs_jeanzay/geco/embedding_sweep/fp_average/
  outputs_jeanzay/meco/embedding_sweep/fp_average/
  outputs_jeanzay/provo/embedding_sweep/fp_average/
  outputs_jeanzay/ucl_reading/embedding_sweep_average/
  outputs_jeanzay/zuco_reading/embedding_sweep/nr_average/

Ceiling-normalises each dataset (same noise ceiling as its per-subject
sibling -- same underlying signal), averages per model across all 5, and
plots score vs n_params (log-x), colour/facet = 4-group taxonomy.

Usage: .venv/bin/python3 scripts/figures/average_embed_scaling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroai.analysis.benchmark_common import OUTPUTS_JEANZAY, REPO_ROOT, ceiling_for, draw_scaling_law
from scripts.summarize.benchmark import (
    _best_last_over_layers,
    _iter_model_score_files,
    _pick_final_row,
    _read_csv_safe,
    _finalize_model_df,
    load_perplexity_dir,
    lookup_model_meta,
)

OUT_DIR = REPO_ROOT / "outputs"

# slug -> (sweep_root, scores_filename, score_col, ceiling_slug, perplexity_rel)
AVERAGE_SPEC: dict[str, tuple[Path, str, str, str, str]] = {
    "geco":         (OUTPUTS_JEANZAY / "geco/embedding_sweep" / "fp_average",
                      "geco_embed_checkpoint_scores.csv", "pearsonr", "geco", "geco/perplexity"),
    "meco":         (OUTPUTS_JEANZAY / "meco/embedding_sweep" / "fp_average",
                      "meco_embed_checkpoint_scores.csv", "pearsonr", "meco", "meco/perplexity"),
    "provo":        (OUTPUTS_JEANZAY / "provo/embedding_sweep" / "fp_average",
                      "provo_embed_checkpoint_scores.csv", "pearsonr", "provo", "provo/perplexity"),
    "ucl_reading":  (OUTPUTS_JEANZAY / "ucl_reading/embedding_sweep_average",
                      "ucl_reading_embed_checkpoint_scores.csv", "pearsonr", "ucl_reading", "ucl_reading/perplexity"),
    "zuco_reading": (OUTPUTS_JEANZAY / "zuco_reading/embedding_sweep" / "nr_average",
                      "zuco_reading_embed_checkpoint_scores.csv", "pearsonr", "zuco_reading", "zuco_reading/perplexity"),
}


def _load_one(slug: str) -> pd.DataFrame:
    sweep_root, fname, score_col, ceiling_slug, ppl_rel = AVERAGE_SPEC[slug]
    ppl_df = load_perplexity_dir(OUTPUTS_JEANZAY / ppl_rel)
    ceiling = ceiling_for(ceiling_slug)
    rows: list[dict] = []
    for family, model_dir, score_files in _iter_model_score_files(sweep_root, fname):
        model_id = model_dir.name.replace("_", "/", 1) if "/" not in model_dir.name else model_dir.name
        frames = [d for f in score_files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        scores = pd.concat(frames, ignore_index=True)
        if score_col not in scores.columns:
            continue
        final = _pick_final_row(scores)
        key = "tokens_seen" if "tokens_seen" in scores.columns else ("step" if "step" in scores.columns else None)
        if key is not None and pd.notna(final.get(key)):
            scores = scores[scores[key] == final[key]]
        best, _last = _best_last_over_layers(scores, score_col, "layer")
        meta = lookup_model_meta(model_id, family, n_params_fallback=None)
        score_norm = best / ceiling if (ceiling is not None and best == best and ceiling > 0) else float("nan")
        rows.append({
            "model_id": model_id, "family": meta.family, "group": meta.group, "label": meta.label,
            "n_params": meta.n_params, "score": score_norm,
        })
    df = _finalize_model_df(rows, ppl_df)
    return df


def build() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for slug in AVERAGE_SPEC:
        d = _load_one(slug)
        if d.empty:
            print(f"[{slug}] no models found in _average sweep dir -- skipping")
            continue
        out[slug] = d
        print(f"[{slug}] {d['model_id'].nunique()} models")
    return out


def main() -> None:
    per_dataset = build()
    if not per_dataset:
        print("No data -- nothing to plot.")
        return
    slugs = list(per_dataset)
    ncols = 3
    nrows = -(-len(slugs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.2 * nrows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax in axes_flat[len(slugs):]:
        ax.set_axis_off()
    for ax, slug in zip(axes_flat, slugs):
        df = per_dataset[slug]
        draw_scaling_law(
            ax, df, "score",
            ylabel="score / noise ceiling",
            title=f"{slug} (n={df['model_id'].nunique()})",
        )
    fig.suptitle("Behavioral embedding scaling law -- subject-averaged RT fit (per dataset)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "behavioral_embed_average_scaling.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  (datasets={slugs})")


if __name__ == "__main__":
    main()
