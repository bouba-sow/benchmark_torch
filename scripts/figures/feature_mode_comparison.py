#!/usr/bin/env python
"""Feature-mode 'doubling' overview: the 2x2 of {neural, behavioral} x
{embedding, surprisal}.

Reads the per-cell summaries written by summarize/benchmark.py
(outputs/benchmark_summary/<slug>/all_models_scores.csv) and produces:

  outputs/benchmark_summary/_feature_mode/coverage.csv      model counts, 2x2
  outputs/benchmark_summary/_feature_mode/comparison.png    embedding vs surprisal
        scaling law faceted per dataset (only datasets that have BOTH modes).

Each cell keeps its NATIVE metric (per design), so panels are labelled with
their own y-units and are compared within a dataset, not across the grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from neuroai.analysis.benchmark_common import (  # noqa: E402
    BENCHMARK_SUMMARY_ROOT, DATASETS, DATASETS_BY_SLUG, NORM_YLABEL,
    ceiling_for, plot_scaling_law,
)

MODES = ("embedding", "surprisal")
OUT = BENCHMARK_SUMMARY_ROOT / "_feature_mode"


def _cell_slug(base: str, mode: str) -> str | None:
    """Summary slug for (base dataset, feature_mode), or None if not defined."""
    for d in DATASETS:
        if d.base == base and d.feature_mode == mode:
            return d.slug
    return None


def _load(slug: str) -> pd.DataFrame:
    f = BENCHMARK_SUMMARY_ROOT / slug / "all_models_scores.csv"
    if not f.is_file():
        return pd.DataFrame()
    return pd.read_csv(f)


def build_coverage() -> pd.DataFrame:
    bases = dict.fromkeys(d.base for d in DATASETS)  # ordered-unique
    rows = []
    for base in bases:
        info = DATASETS_BY_SLUG.get(base) or next(d for d in DATASETS if d.base == base)
        row = {"dataset": base, "response": info.response}
        for mode in MODES:
            slug = _cell_slug(base, mode)
            df = _load(slug) if slug else pd.DataFrame()
            n = int(df["score"].notna().sum()) if "score" in df else 0
            row[mode] = n
            row[f"{mode}_layer"] = DATASETS_BY_SLUG[slug].layer_tag if (slug and n) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _panel(ax, base: str, mode: str, layer_sel: str = "best") -> None:
    slug = _cell_slug(base, mode)
    info = DATASETS_BY_SLUG[slug] if slug else None
    df = _load(slug) if slug else pd.DataFrame()
    if info is None or df.empty or "n_params" not in df:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"{base}\n{mode}: not computed", ha="center", va="center")
        return
    metric = info.metric.split("(")[0].strip()
    # process_dataset already wrote ceiling-normalised score_{best,last}_norm.
    col = f"score_{layer_sel}_norm"
    if col not in df.columns:
        col = "score"
    ceil = ceiling_for(slug)
    ylabel = NORM_YLABEL if (ceil is not None and ceil > 0) else metric
    n = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
    plot_scaling_law(
        df, col, None, ylabel=ylabel,
        title=f"{base} — {mode} / {layer_sel} layer ({metric})  n={n}",
        ax=ax, label_points=False,
    )


def build_comparison(cov: pd.DataFrame, layer_sel: str = "best", per_fig: int = 3) -> list[Path]:
    """Emit several page-sized figures (``per_fig`` datasets each, embedding left /
    surprisal right) for one layer selection, rather than one tall overflowing figure."""
    both = cov[(cov["embedding"] > 0) & (cov["surprisal"] > 0)].copy()
    both = both.sort_values(["response", "dataset"])
    bases = list(both["dataset"])
    if not bases:
        print("no datasets with both feature modes; skipping comparison figures")
        return []
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob(f"comparison_{layer_sel}_*.png"):   # clear stale chunks
        old.unlink()
    written: list[Path] = []
    chunks = [bases[i:i + per_fig] for i in range(0, len(bases), per_fig)]
    for ci, chunk in enumerate(chunks, start=1):
        fig, axes = plt.subplots(len(chunk), 2, figsize=(15, 4.6 * len(chunk)), squeeze=False)
        for i, base in enumerate(chunk):
            for j, mode in enumerate(MODES):
                _panel(axes[i][j], base, mode, layer_sel)
        fig.suptitle(
            f"Feature-mode doubling — {layer_sel} layer ({ci}/{len(chunks)}): "
            "embedding (left) vs surprisal (right)", fontsize=14, y=1.004)
        fig.tight_layout()
        out = OUT / f"comparison_{layer_sel}_{ci}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    print(f"wrote {len(written)} {layer_sel}-layer comparison figures ({len(bases)} datasets, {per_fig}/page)")
    return written


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cov = build_coverage()
    cov.to_csv(OUT / "coverage.csv", index=False)
    # pretty print the 2x2
    print("Feature-mode coverage (models scored per cell):\n")
    hdr = f"{'dataset':22s} {'response':11s} {'embedding':>10s} {'surprisal':>10s}"
    print(hdr); print("-" * len(hdr))
    for _, r in cov.iterrows():
        print(f"{r['dataset']:22s} {r['response']:11s} {r['embedding']:10d} {r['surprisal']:10d}")
    print("-" * len(hdr))
    for resp in ("neural", "behavioral"):
        sub = cov[cov["response"] == resp]
        print(f"{resp:22s} {'':11s} {sub['embedding'].sum():10d} {sub['surprisal'].sum():10d}")
    print(f"\nwrote {OUT/'coverage.csv'}")
    for old in OUT.glob("comparison_*.png"):   # clear any legacy single-set chunks
        if old.stem.split("_")[1] not in ("best", "last"):
            old.unlink()
    for layer_sel in ("best", "last"):
        build_comparison(cov, layer_sel)


if __name__ == "__main__":
    main()
