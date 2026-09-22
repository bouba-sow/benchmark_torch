#!/usr/bin/env python3
"""Paper figure: 2x2 ablation of Provo's Pythia checkpoint trajectories --
{Oh & Schuler (2023) formula, this pipeline's original formula} x
{out-of-sample, in-sample} -- all four run AFTER the revision-namespaced-file
bug fix (neuroai/benchmarks/ucl_reading_lme.py), to isolate how much of the
clean inverse-U shape is the bug fix itself vs. the Oh & Schuler formula
change (no frequency covariate, no spillover, crossed (1 | sentid) random
intercept) vs. the in-sample-vs-OOS convention.

Four pages, same two-panel design as figures/figure10_provo_ohschuler.py:
  A. Final checkpoint (step 143000) only, score vs n_params (log-x).
  B. Every checkpoint, pooled score-vs-perplexity scatter, star = fully
     trained, dashed = pooled quadratic fit in log10(perplexity).

Sweep roots (all under outputs_jeanzay/, all bug-fixed):
  ohschuler_oos:        provo/surprisal_sweep_ohschuler_oos/fp
  ohschuler_insample:   provo/surprisal_sweep_ohschuler_insample/fp
  oldformula_oos:       provo/surprisal_sweep_checkpoints/fp
  oldformula_insample:  provo/surprisal_sweep_checkpoints_insample/fp

Output: reports/paper/provo_formula_ablation.pdf (4 pages)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, REGISTRY_INDEX, REPO_ROOT
from neuroai.analysis.paper_style import use_paper_style
from scripts.analysis.perplexity_matched_scaling import _best_row_insample, _load_perplexity_per_checkpoint
from scripts.figures.figure8_scale_matched_comparison import _best_row_per_word, _model_id_from_dirname
from scripts.figures.figure9_pythia_checkpoint_scaling import _panel_a, _panel_b

OUT_PATH = REPO_ROOT / "reports" / "paper" / "provo_formula_ablation.pdf"

VARIANTS = [
    ("ohschuler_oos", "Oh & Schuler formula, out-of-sample", "out-of-sample", False),
    ("ohschuler_insample", "Oh & Schuler formula, in-sample", "in-sample", True),
    ("oldformula_oos", "Original formula (freq + spillover), out-of-sample", "out-of-sample", False),
    ("oldformula_insample", "Original formula (freq + spillover), in-sample", "in-sample", True),
]


def _load(variant: str, insample: bool) -> pd.DataFrame:
    sweep_root = REPO_ROOT / "outputs_jeanzay" / f"provo_checkpoint_sweep_lme_dlt_{variant}" / "fp" / "pythia_checkpoints"
    if not sweep_root.is_dir():
        return pd.DataFrame()
    rows = []
    for model_dir in sorted(sweep_root.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_paths = [p for p in model_dir.glob("*_lme_checkpoint_scores.csv") if not p.name.endswith(".lock")]
        if not csv_paths:
            continue
        mid = _model_id_from_dirname(model_dir.name)
        if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
            continue
        df = pd.read_csv(csv_paths[0])
        needed_col = "delta_loglik_train" if insample else "delta_loglik_test"
        if df.empty or needed_col not in df.columns or "revision" not in df.columns:
            continue
        meta = REGISTRY_INDEX[mid]
        for revision, rev_df in df.groupby("revision"):
            picked = _best_row_insample(rev_df) if insample else _best_row_per_word(rev_df)
            if picked is None:
                continue
            dll_total, n_obs = picked
            rows.append(dict(
                model_id=mid, revision=revision, n_params=meta.n_params,
                dll_total=dll_total, score_raw=dll_total / n_obs,
            ))
    return pd.DataFrame(rows)


def main() -> None:
    use_paper_style()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ppl_map = _load_perplexity_per_checkpoint("provo")

    with PdfPages(OUT_PATH) as pdf:
        for variant, title, sample_label, insample in VARIANTS:
            df = _load(variant, insample)
            print(f"[{variant}] {df['model_id'].nunique() if not df.empty else 0} models, {len(df)} points")

            fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.5, 4.2))
            if df.empty:
                for ax in (axA, axB):
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                             transform=ax.transAxes, fontsize=8, color="#999999")
            else:
                _panel_a(axA, df, sample_label=sample_label)
                _panel_b(axB, df, ppl_map, sample_label=sample_label)
            fig.text(0.01, 0.98, f"provo — {title} (bug-fixed)", fontsize=9,
                      color="#666666", ha="left", va="top")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
