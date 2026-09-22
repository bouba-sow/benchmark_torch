#!/usr/bin/env python3
"""Summarize Podcast ECoG surprisal (logit-lens) results: last-layer vs best-layer.

Main metric: **Pearson r**.

Every logit-lens run scores each electrode at every layer
(``podcast_by_electrode_layerwise.csv``: one row per subject × electrode ×
layer, with both ``pearsonr`` and ``spearmanr``). This script:

  * takes the **last-layer** score straight from ``podcast_by_electrode.csv``
    (column ``pearsonr`` — the standard, single-number surprisal everyone
    reports), and
  * recomputes the **best layer per electrode by argmax over Pearson r**
    across layers from the layerwise CSV (independent of whatever metric the
    pipeline's stored ``best_layer``/``best_pearsonr`` columns used — those
    are picked by Spearman argmax by default, see ``neural_pipeline.py``'s
    ``best_layer_index(metric="spearmanr")``).

That gives two scaling laws (brain score vs. model size) side by side, plus a
few diagnostic plots to see whether middle-layer surprisal is more human-like
than final-layer surprisal, and if so, at what relative depth the advantage
sits.

Reads from the final-checkpoint (top-level ``<family>/<model>/``) results of a
podcast surprisal sweep — same layout as ``summarize/podcast.py``'s default mode.
Checkpoint-sweep training dynamics are out of scope here.

Outputs (default ``outputs/podcast/summary_surprisal/``):
  all_models_surprisal_scores.csv
  scaling_law_last_layer.png       — regular (last-layer) surprisal
  scaling_law_best_layer.png       — best-layer surprisal
  scaling_law_side_by_side.png     — both panels together for quick comparison
  delta_best_minus_last.png        — best-layer advantage vs model size
  best_layer_depth.png             — mean relative depth of the best layer vs model size
  layer_profiles.png               — per-model score-vs-relative-depth curves

  If the sweep ran with ``benchmark.compute_unique_surprisal: true``
  (podcast_unique_surprisal_by_electrode.csv present):
  all_models_unique_surprisal_scores.csv
  scaling_law_unique_contribution.png — surprisal's delta r over the [word
    length, KenLM frequency] baseline vs model size — the direct test of
    whether a raw-correlation scaling trend survives controlling for the
    frequency/length confound Oh et al. (2024) propose for inverse scaling,
    or is explained by it.

Usage
-----
  python scripts/summarize/podcast_surprisal.py
  python scripts/summarize/podcast_surprisal.py \\
      --sweep_dir outputs_jeanzay/podcast/checkpoint_sweep_surprisal \\
      --out_dir   outputs/podcast/summary_surprisal
  python scripts/summarize/podcast_surprisal.py --families gpt2 pythia olmo2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import scripts.summarize.podcast as sp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORR = "pearsonr"  # main metric for this script
LAST_LAYER_METRIC = "mean_pearsonr_last"
BEST_LAYER_METRIC = "mean_pearsonr_best"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _best_layer_by_pearson(lw_csv: Path, revision: str) -> tuple[float, float] | None:
    """Per electrode, argmax pearsonr across layers; return (mean best r, mean best layer)."""
    try:
        df = pd.read_csv(lw_csv)
    except Exception as exc:
        print(f"    [warn] failed to read {lw_csv}: {exc}", file=sys.stderr)
        return None
    if df.empty or CORR not in df.columns:
        return None
    if "revision" in df.columns:
        df = df[df["revision"] == revision]
    if df.empty:
        return None
    idx = df.groupby(["subject", "electrode"])[CORR].idxmax()
    best = df.loc[idx]
    return float(best[CORR].mean()), float(best["layer"].mean())


def load_surprisal_models(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """One row per model: mean last-layer and best-layer Pearson r across electrodes."""
    rows: list[dict] = []

    for elec_csv in sorted(sweep_dir.glob("*/*/podcast_by_electrode.csv")):
        if sp._is_checkpoint_staging_path(elec_csv):
            continue
        family = elec_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = elec_csv.parent.name
        model_id = model_dir.replace("_", "/", 1)

        try:
            df = pd.read_csv(elec_csv)
        except Exception as exc:
            print(f"    [warn] failed to read {elec_csv}: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue

        if "revision" in df.columns and df["revision"].nunique() > 1:
            revision = sp._pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == revision]
        elif "revision" in df.columns:
            revision = str(df["revision"].iloc[0])
        else:
            revision = "main"

        if CORR not in df.columns:
            print(f"    [warn] missing {CORR} column: {elec_csv}", file=sys.stderr)
            continue

        lw_csv = elec_csv.parent / "podcast_by_electrode_layerwise.csv"
        best = _best_layer_by_pearson(lw_csv, revision) if lw_csv.is_file() else None
        if best is None:
            print(f"    [warn] no per-layer data for best-layer-by-Pearson: {elec_csv}", file=sys.stderr)
            continue

        m = meta.get(model_id, meta.get(model_dir, {}))
        n_layers = m.get("n_layers", np.nan)
        mean_best_r, mean_best_layer = best

        row = {
            "family": family,
            "model_id": model_id,
            "label": m.get("label", model_id.split("/")[-1]),
            "n_params": m.get("n_params", np.nan),
            "n_layers": n_layers,
            "revision": revision,
            "n_electrodes": len(df),
            "n_subjects": int(df["subject"].nunique()) if "subject" in df.columns else 0,
            LAST_LAYER_METRIC: float(df[CORR].mean()),
            BEST_LAYER_METRIC: mean_best_r,
            "mean_best_layer": mean_best_layer,
        }
        if pd.notna(n_layers) and n_layers > 1:
            row["mean_best_layer_rel"] = mean_best_layer / (n_layers - 1)

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["family", "n_params"]).reset_index(drop=True)


UNIQUE_LAST_LAYER_METRIC = "mean_delta_pearsonr_last"
UNIQUE_BEST_LAYER_METRIC = "mean_delta_pearsonr_best"


def load_unique_surprisal_models(
    sweep_dir: Path,
    families: list[str] | None,
    meta: dict[str, dict],
) -> pd.DataFrame:
    """One row per model: mean delta Pearson r (surprisal's unique contribution
    over the [wlen, kenlm_freq] baseline) at the last layer and at the
    per-electrode best layer (by argmax delta), from
    ``podcast_unique_surprisal_by_electrode.csv`` — only present when the
    sweep ran with ``benchmark.compute_unique_surprisal: true``.
    """
    rows: list[dict] = []

    for u_csv in sorted(sweep_dir.glob("*/*/podcast_unique_surprisal_by_electrode.csv")):
        if sp._is_checkpoint_staging_path(u_csv):
            continue
        family = u_csv.parent.parent.name
        if families and family not in families:
            continue
        model_dir = u_csv.parent.name
        model_id = model_dir.replace("_", "/", 1)

        try:
            df = pd.read_csv(u_csv)
        except Exception as exc:
            print(f"    [warn] failed to read {u_csv}: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue

        if "revision" in df.columns and df["revision"].nunique() > 1:
            revision = sp._pick_final_revision(df["revision"].dropna().unique().tolist())
            df = df[df["revision"] == revision]
        elif "revision" in df.columns:
            revision = str(df["revision"].iloc[0])
        else:
            revision = "main"
        if df.empty:
            continue

        n_layers_seen = int(df["layer"].max()) + 1
        last_layer_df = df[df["layer"] == df["layer"].max()]
        best_idx = df.groupby(["subject", "electrode"])["delta_pearsonr"].idxmax()
        best_df = df.loc[best_idx]

        m = meta.get(model_id, meta.get(model_dir, {}))
        row = {
            "family": family,
            "model_id": model_id,
            "label": m.get("label", model_id.split("/")[-1]),
            "n_params": m.get("n_params", np.nan),
            "n_layers": m.get("n_layers", n_layers_seen),
            "revision": revision,
            "n_electrodes": last_layer_df["electrode"].nunique(),
            "n_subjects": int(df["subject"].nunique()),
            UNIQUE_LAST_LAYER_METRIC: float(last_layer_df["delta_pearsonr"].mean()),
            UNIQUE_BEST_LAYER_METRIC: float(best_df["delta_pearsonr"].mean()),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["family", "n_params"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _ylabel_mean_pearson(*, window: bool = False) -> str:
    suffix = f" {sp._window_label()}" if window else ""
    return f"Mean Pearson r{suffix}"


def _patch_summarize_podcast_to_pearson() -> None:
    """Repoint summarize_podcast's shared plot helpers (ylabel text, layer-profile
    metric/column) from Spearman to Pearson, since this script's main metric is
    Pearson r. Only affects functions called from this script within this process."""
    sp._ylabel_mean_spearman = _ylabel_mean_pearson
    sp.LAYER_CORR = CORR
    sp.LAYER_PROFILE_METRIC = "mean_pearsonr"


def plot_side_by_side(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), sharey=True)
    sp._scaling_law_ax(
        axes[0], df, LAST_LAYER_METRIC,
        title="Last layer (regular surprisal)",
        ylabel=_ylabel_mean_pearson(),
    )
    sp._scaling_law_ax(
        axes[1], df, BEST_LAYER_METRIC,
        title="Best layer surprisal",
        ylabel=_ylabel_mean_pearson(),
    )
    fig.suptitle("Podcast ECoG — Surprisal scaling law: last vs. best layer", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = out_dir / "scaling_law_side_by_side.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_delta(df: pd.DataFrame, out_dir: Path) -> None:
    """Best-layer minus last-layer score, per model, vs model size."""
    plot_df = df.dropna(subset=["n_params", LAST_LAYER_METRIC, BEST_LAYER_METRIC]).copy()
    if plot_df.empty:
        print("  [skip] no data for delta plot")
        return
    plot_df["delta"] = plot_df[BEST_LAYER_METRIC] - plot_df[LAST_LAYER_METRIC]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(0, color="gray", lw=1.0, ls="--", zorder=1)

    for family, grp in plot_df.groupby("family"):
        color = sp.FAMILY_COLORS.get(family, "gray")
        marker = sp.FAMILY_MARKERS.get(family, "o")
        grp_s = grp.sort_values("n_params")
        ax.scatter(grp_s["n_params"], grp_s["delta"], label=family,
                   color=color, marker=marker, s=80, zorder=3, alpha=0.9)
        if len(grp_s) > 1:
            ax.plot(grp_s["n_params"], grp_s["delta"], color=color, lw=0.8, alpha=0.45)
        for _, row in grp_s.iterrows():
            ax.annotate(row["label"], (row["n_params"], row["delta"]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters", fontsize=10)
    ax.set_ylabel("Best-layer r − Last-layer r", fontsize=10)
    ax.set_title("Podcast ECoG — Best-layer surprisal advantage over last-layer", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fname = out_dir / "delta_best_minus_last.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_best_layer_depth(df: pd.DataFrame, out_dir: Path) -> None:
    """Mean relative depth of the winning layer vs model size."""
    if "mean_best_layer_rel" not in df.columns:
        print("  [skip] no best_layer column found")
        return
    plot_df = df.dropna(subset=["n_params", "mean_best_layer_rel"]).copy()
    if plot_df.empty:
        print("  [skip] no data for best-layer-depth plot")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.7, zorder=1, label="last layer")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5, zorder=1, label="middle layer")

    for family, grp in plot_df.groupby("family"):
        color = sp.FAMILY_COLORS.get(family, "gray")
        marker = sp.FAMILY_MARKERS.get(family, "o")
        grp_s = grp.sort_values("n_params")
        ax.scatter(grp_s["n_params"], grp_s["mean_best_layer_rel"], label=family,
                   color=color, marker=marker, s=80, zorder=3, alpha=0.9)
        if len(grp_s) > 1:
            ax.plot(grp_s["n_params"], grp_s["mean_best_layer_rel"], color=color, lw=0.8, alpha=0.45)
        for _, row in grp_s.iterrows():
            ax.annotate(row["label"], (row["n_params"], row["mean_best_layer_rel"]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Number of parameters", fontsize=10)
    ax.set_ylabel("Mean relative depth of best layer (0=first, 1=last)", fontsize=10)
    ax.set_title("Podcast ECoG — Where does the best surprisal layer sit?", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fname = out_dir / "best_layer_depth.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_unique_contribution_scaling_law(df: pd.DataFrame, out_dir: Path) -> None:
    """Scaling law for surprisal's UNIQUE contribution (delta r) over the
    [wlen, kenlm_freq] baseline — the direct test of whether a raw-correlation
    scaling trend survives controlling for word length/frequency, or is
    explained by it (Oh et al. 2024's confound story for inverse scaling).
    """
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), sharey=True)
    sp._scaling_law_ax(
        axes[0], df, UNIQUE_LAST_LAYER_METRIC,
        title="Last layer — unique contribution",
        ylabel="Mean delta Pearson r (full − baseline)",
    )
    sp._scaling_law_ax(
        axes[1], df, UNIQUE_BEST_LAYER_METRIC,
        title="Best layer — unique contribution",
        ylabel="Mean delta Pearson r (full − baseline)",
    )
    for ax in axes:
        ax.axhline(0, color="gray", lw=1.0, ls="--", zorder=1)
    fig.suptitle(
        "Podcast ECoG — Surprisal's unique contribution over [word length, "
        "KenLM frequency] baseline",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fname = out_dir / "scaling_law_unique_contribution.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sweep_dir", type=Path,
        default=Path("outputs_jeanzay/podcast/checkpoint_sweep_surprisal"),
        help="Read-only surprisal sweep results (default: outputs_jeanzay/podcast/checkpoint_sweep_surprisal)",
    )
    p.add_argument(
        "--out_dir", type=Path,
        default=Path("outputs/podcast/summary_surprisal"),
        help="Output directory (default: outputs/podcast/summary_surprisal)",
    )
    p.add_argument(
        "--families", nargs="+", default=None,
        help="Restrict to these families (default: all)",
    )
    args = p.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not sweep_dir.is_dir():
        sys.exit(f"sweep_dir not found: {sweep_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from : {sweep_dir}")
    print(f"Writing to   : {out_dir}\n")

    _patch_summarize_podcast_to_pearson()
    meta = sp._registry_meta()

    df = load_surprisal_models(sweep_dir, args.families, meta)
    if df.empty:
        sys.exit("No podcast_by_electrode.csv files with surprisal columns found — nothing to summarize.")

    print(f"Found {len(df)} models across {df['family'].nunique()} families:\n")
    hdr = f"  {'family':<10} {'model':<28} {'n_subj':>6} {'n_elec':>7} {'last r':>8} {'best r':>8} {'Δ':>7}"
    print(hdr)
    print("  " + "-" * len(hdr))
    for _, row in df.iterrows():
        delta = row[BEST_LAYER_METRIC] - row[LAST_LAYER_METRIC]
        print(
            f"  {row['family']:<10} {row['label']:<28} "
            f"{row['n_subjects']:>6} {row['n_electrodes']:>7} "
            f"{row[LAST_LAYER_METRIC]:>8.4f} {row[BEST_LAYER_METRIC]:>8.4f} {delta:>+7.4f}"
        )
    print()

    out_csv = out_dir / "all_models_surprisal_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # Two scaling laws: last layer vs best layer
    sp.plot_scaling_law(df, out_dir, metric=LAST_LAYER_METRIC,
                        suffix="_last_layer", title_extra=" — last layer (regular surprisal)")
    sp.plot_scaling_law(df, out_dir, metric=BEST_LAYER_METRIC,
                        suffix="_best_layer", title_extra=" — best layer surprisal")
    plot_side_by_side(df, out_dir)
    plot_delta(df, out_dir)
    plot_best_layer_depth(df, out_dir)

    print("\nLoading layerwise data...")
    lw_df = sp.load_layerwise(sweep_dir, args.families, meta)
    sp.plot_layer_profiles(lw_df, out_dir)

    print("\nLoading unique-surprisal-contribution data "
          "(podcast_unique_surprisal_by_electrode.csv, if present)...")
    unique_df = load_unique_surprisal_models(sweep_dir, args.families, meta)
    if unique_df.empty:
        print("  [skip] no podcast_unique_surprisal_by_electrode.csv found — "
              "re-run the sweep with benchmark.compute_unique_surprisal: true to get this.")
    else:
        u_csv = out_dir / "all_models_unique_surprisal_scores.csv"
        unique_df.to_csv(u_csv, index=False)
        print(f"  Saved: {u_csv}")
        hdr2 = f"  {'family':<10} {'model':<28} {'last Δr':>9} {'best Δr':>9}"
        print(hdr2)
        print("  " + "-" * len(hdr2))
        for _, row in unique_df.iterrows():
            print(
                f"  {row['family']:<10} {row['label']:<28} "
                f"{row[UNIQUE_LAST_LAYER_METRIC]:>9.4f} {row[UNIQUE_BEST_LAYER_METRIC]:>9.4f}"
            )
        plot_unique_contribution_scaling_law(unique_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
