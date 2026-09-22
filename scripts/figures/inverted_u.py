"""Binned-trend ("curve") visualization of the score-vs-perplexity relationship,
to see the inverted-U for LLMs (N400 datasets) and the reversed monotonic trend
for Memory-Constrained models (finding #4 and #5 in reports/neural_trends_findings.md)
on the same axes, per dataset.

Usage:
    .venv/bin/python scripts/figures/inverted_u.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.analysis.benchmark_common import DATASETS_BY_SLUG, build_grid, draw_binned_trend_by_perplexity  # noqa: E402
from scripts.summarize.benchmark import LOADERS  # noqa: E402

ANALYSIS_DIR = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "outputs" / "benchmark_summary" / "_analysis"

# The N400 datasets plus 2 non-N400 datasets with
# the strongest Memory-Constrained perplexity reversal (LPP fMRI, Podcast iEEG),
# so the figure shows both "where it's strong" and "where it's absent/weak".
SLUGS = ["nieuwland_eeg", "szewczyk_eeg", "lpp_fmri", "podcast_ieeg"]


def main() -> None:
    panels = []
    for slug in SLUGS:
        info = DATASETS_BY_SLUG[slug]
        model_df = LOADERS[slug]()["model_df"].dropna(subset=["score"])
        panels.append((
            info.name,
            lambda ax, df=model_df, info=info: draw_binned_trend_by_perplexity(
                ax, df, "score", "perplexity", ylabel=info.metric, title=info.name,
            ),
        ))
    out_path = ANALYSIS_DIR / "perplexity_binned_trend.png"
    build_grid(panels, out_path, suptitle="Score vs. perplexity: binned mean +/- SEM curve, LLM vs. Memory-Constrained\n(N400 datasets: Nieuwland, Szewczyk | contrast: LPP fMRI, Podcast iEEG)", ncols=3, panel_size=(7, 6))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
