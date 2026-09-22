#!/usr/bin/env python3
"""Paper figure: Provo Pythia checkpoint trajectories under Oh & Schuler
(2023, arXiv:2304.11389)'s own LME formula -- no frequency covariate, no
spillover, crossed (1 | sentid) random intercept -- and their own
out-of-sample convention (held-out delta_loglik_test), reading the
dedicated sweep at outputs_jeanzay/provo/surprisal_sweep_ohschuler_oos/.

Post-dates the revision-namespaced-file fix (neuroai/benchmarks/
ucl_reading_lme.py / natural_stories_lme.py) for the checkpoint-collision
bug that corrupted every earlier pythia_checkpoints sweep (concurrent
checkpoints of the same model shared one working-file path and raced,
collapsing 11 distinct training steps down to 1-2 duplicate fits) --
this is the first genuinely-correct per-checkpoint sweep.

Same two-panel design as figures/figure9_pythia_checkpoint_scaling.py:
  A. Final checkpoint (step 143000) only, score vs n_params (log-x).
  B. Every checkpoint, pooled score-vs-perplexity scatter (all 10 sizes on
     one curve, not one line per size -- matches Oh & Schuler's own Fig. 1/2
     middle-panel design), star = fully trained, dashed = pooled quadratic
     fit in log10(perplexity).

Output: reports/paper/provo_ohschuler_checkpoint_scaling.pdf (1 page)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from neuroai.analysis.benchmark_common import EXCLUDED_MODEL_IDS, REGISTRY_INDEX, REPO_ROOT
from neuroai.analysis.paper_style import use_paper_style
from scripts.analysis.perplexity_matched_scaling import _load_perplexity_per_checkpoint
from scripts.figures.figure8_scale_matched_comparison import _best_row_per_word, _model_id_from_dirname
from scripts.figures.figure9_pythia_checkpoint_scaling import _panel_a, _panel_b

OUT_PATH = REPO_ROOT / "reports" / "paper" / "provo_ohschuler_checkpoint_scaling.pdf"
SWEEP_ROOT = REPO_ROOT / "outputs_jeanzay" / "provo/surprisal_sweep_ohschuler_oos" / "fp"
FINAL_STEP_REVISION = "step143000"


def _load_provo_ohschuler_oos() -> pd.DataFrame:
    rows = []
    fam_dir = SWEEP_ROOT / "pythia_checkpoints"
    if not fam_dir.is_dir():
        return pd.DataFrame()
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_paths = [p for p in model_dir.glob("*_lme_checkpoint_scores.csv") if not p.name.endswith(".lock")]
        if not csv_paths:
            continue
        mid = _model_id_from_dirname(model_dir.name)
        if mid in EXCLUDED_MODEL_IDS or mid not in REGISTRY_INDEX:
            continue
        df = pd.read_csv(csv_paths[0])
        if df.empty or "delta_loglik_test" not in df.columns or "revision" not in df.columns:
            continue
        meta = REGISTRY_INDEX[mid]
        for revision, rev_df in df.groupby("revision"):
            picked = _best_row_per_word(rev_df)
            if picked is None:
                continue
            dll_total, n_obs_test = picked
            rows.append(dict(
                model_id=mid, revision=revision, n_params=meta.n_params,
                dll_total=dll_total, score_raw=dll_total / n_obs_test,
            ))
    return pd.DataFrame(rows)


def main() -> None:
    use_paper_style()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = _load_provo_ohschuler_oos()
    ppl_map = _load_perplexity_per_checkpoint("provo")
    print(f"provo (Oh & Schuler, OOS): {df['model_id'].nunique() if not df.empty else 0} models, "
          f"{len(df)} (model, checkpoint) points")

    with PdfPages(OUT_PATH) as pdf:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.5, 4.2))
        if df.empty:
            for ax in (axA, axB):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                         transform=ax.transAxes, fontsize=8, color="#999999")
        else:
            _panel_a(axA, df, sample_label="out-of-sample")
            _panel_b(axB, df, ppl_map, sample_label="out-of-sample")
        fig.text(0.01, 0.98, "provo — Oh & Schuler (2023) formula, out-of-sample", fontsize=9,
                  color="#666666", ha="left", va="top")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
