#!/usr/bin/env python3
"""Summarize the MECo pythia_checkpoints sweep: best-layer score vs model
size and vs perplexity.

Reads whatever's completed so far under
outputs_jeanzay/meco/embedding_sweep/fp/pythia_checkpoints/ and
outputs_jeanzay/meco/perplexity/meco_perplexity_summary.csv — safe to
re-run at any point while the sweep is still filling in.

Outputs (under --out_dir, default same summary/ folder as the regular sweep):
  meco/pythia_checkpoints_summary.csv   merged per-(model,checkpoint) table
  meco_pythia_by_size.png               final-checkpoint score vs n_params
  meco_pythia_by_perplexity.png         every completed checkpoint, score vs perplexity
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuroai.analysis.paper_style import despine, use_paper_style
from neuroai.models.registry import LPP_FAMILIES


def _n_params_map() -> dict[str, float]:
    return {s.model_id: s.n_params for s in LPP_FAMILIES["pythia"]}


def load_scores(sweep_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(sweep_dir / "*" / "meco_embed_checkpoint_scores.csv")))
    if not paths:
        raise FileNotFoundError(f"No meco_embed_checkpoint_scores.csv found under {sweep_dir}")
    rows = []
    for p in paths:
        model_id = Path(p).parent.name.replace("_", "/", 1)
        df = pd.read_csv(p)
        df["model_id"] = model_id
        rows.append(df)
    combined = pd.concat(rows, ignore_index=True)
    # Best layer per (model, checkpoint) — same convention as LayerwiseRidgeResults.best_layer_score().
    idx = combined.groupby(["model_id", "revision"])["spearmanr"].idxmax()
    best = combined.loc[idx].reset_index(drop=True)
    return best


def load_perplexity(perplexity_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(perplexity_csv)
    df = df[df["model_id"].str.contains("pythia", case=False, na=False)]
    return df[["model_id", "revision", "perplexity"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep_dir", default="outputs_jeanzay/meco/embedding_sweep/fp/pythia_checkpoints")
    ap.add_argument("--perplexity_csv", default="outputs_jeanzay/meco/perplexity/meco_perplexity_summary.csv")
    ap.add_argument("--out_dir", default=None, help="Default: <sweep_dir>/summary")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sweep_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = load_scores(sweep_dir)
    ppl = load_perplexity(Path(args.perplexity_csv))

    merged = best.merge(ppl, on=["model_id", "revision"], how="left")
    n_params = _n_params_map()
    merged["n_params"] = merged["model_id"].map(n_params)
    merged = merged.sort_values(["n_params", "step"]).reset_index(drop=True)

    csv_path = out_dir / "meco/pythia_checkpoints_summary.csv"
    merged.to_csv(csv_path, index=False)
    n_missing_ppl = merged["perplexity"].isna().sum()
    print(f"Loaded {len(merged)} (model,checkpoint) rows "
          f"({merged['model_id'].nunique()} models); {n_missing_ppl} missing perplexity.")
    print(f"Wrote {csv_path}")

    use_paper_style()

    # ---- Plot 1: TRUE final-checkpoint (step=143000) score vs model size ----
    # NOT groupby(...).idxmax("step") — for a model whose sweep hasn't reached
    # 143000 yet, the max *available* step is some early/untrained checkpoint,
    # which would silently masquerade as "final" and fake a size-related decline
    # that's actually just "this model isn't trained yet". Require the real
    # final step explicitly and report what's excluded.
    FINAL_STEP = 143000
    final = merged[merged["step"] == FINAL_STEP].sort_values("n_params")
    all_models = sorted(merged["model_id"].unique(), key=lambda m: n_params.get(m, 0))
    incomplete = [m for m in all_models if m not in set(final["model_id"])]
    if incomplete:
        print(f"NOTE: {len(incomplete)} model(s) haven't reached step={FINAL_STEP} yet "
              f"— excluded from the by-size plot (sweep still running): {incomplete}")

    fig, ax = plt.subplots(figsize=(7, 5))
    if not final.empty:
        ax.plot(final["n_params"], final["spearmanr"], marker="o", color="#3B6FA0", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (parameters)")
    ax.set_ylabel("Best-layer Spearman r")
    subtitle = "final checkpoint (step 143000) per size"
    if incomplete:
        subtitle += f" — {len(incomplete)} larger model(s) still training, not shown"
    ax.set_title(f"MECo — Pythia checkpoint sweep (embedding): score vs model size\n({subtitle})")
    despine(ax)
    fig.tight_layout()
    size_path = out_dir / "meco_pythia_by_size.png"
    fig.savefig(size_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {size_path}")

    # ---- Plot 2: every completed checkpoint, score vs perplexity ----
    have_ppl = merged.dropna(subset=["perplexity"]).copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if not have_ppl.empty:
        sizes_sorted = sorted(have_ppl["n_params"].unique())
        cmap = plt.get_cmap("viridis")
        norm_vals = np.linspace(0.15, 0.9, len(sizes_sorted))
        color_by_size = dict(zip(sizes_sorted, cmap(norm_vals)))
        for n, group in have_ppl.groupby("n_params"):
            group = group.sort_values("perplexity")
            label = group["model_id"].iloc[0].split("/")[-1]
            ax.plot(group["perplexity"], group["spearmanr"], marker="o", ms=4, lw=1,
                    color=color_by_size[n], label=label, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("Perplexity (log scale)")
        ax.set_ylabel("Best-layer Spearman r")
        ax.set_title("MECo — Pythia checkpoint sweep (embedding): score vs perplexity\n(every completed checkpoint, partial — sweep still running)")
        ax.legend(fontsize=7, ncol=2, frameon=False, loc="best")
        despine(ax)
    fig.tight_layout()
    ppl_path = out_dir / "meco_pythia_by_perplexity.png"
    fig.savefig(ppl_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {ppl_path}")


if __name__ == "__main__":
    main()
