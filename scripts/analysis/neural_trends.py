"""Exploratory analysis of the neural (non-behavioral) benchmark datasets,
looking for cross-dataset patterns worth writing up: rank consistency across
datasets, group effects (LLM vs Memory-Constrained vs BabyLM), scaling-law
saturation, perplexity relationships, and layer-depth-of-peak-alignment.

Reuses scripts/summarize/benchmark.py's LOADERS (already validated against the
raw sweep data) so this analysis is consistent with the report, not a
reimplementation. Read-only: does not touch outputs/ or outputs_jeanzay/.

Usage:
    .venv/bin/python scripts/analysis/neural_trends.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.analysis.benchmark_common import GROUP_ORDER, NEURAL_SLUGS, sem  # noqa: E402
from scripts.summarize.benchmark import LOADERS  # noqa: E402


pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def load_all():
    data = {}
    for slug in NEURAL_SLUGS:
        print(f"loading {slug}...", file=sys.stderr)
        d = LOADERS[slug]()
        d["model_df"] = d["model_df"].dropna(subset=["score"]).copy()
        data[slug] = d
    return data


def cross_dataset_consistency(data: dict) -> pd.DataFrame:
    """Spearman rank correlation of model scores between every pair of
    datasets, restricted to models common to both."""
    wide = None
    for slug, d in data.items():
        s = d["model_df"].set_index("model_id")["score"].rename(slug)
        wide = s.to_frame() if wide is None else wide.join(s, how="outer")
    corr = wide.corr(method="spearman", min_periods=8)
    n_common = wide.notna().sum()
    return corr, wide, n_common


def group_effects(data: dict) -> pd.DataFrame:
    rows = []
    for slug, d in data.items():
        df = d["model_df"]
        stats_by_group = {}
        for g in GROUP_ORDER:
            vals = df.loc[df["group"] == g, "score"].dropna().to_numpy()
            if len(vals) >= 2:
                stats_by_group[g] = (vals.mean(), vals.std(ddof=1), len(vals))
        row = {"dataset": slug}
        if "LLM" in stats_by_group and "Memory-Constrained" in stats_by_group:
            m1, s1, n1 = stats_by_group["LLM"]
            m2, s2, n2 = stats_by_group["Memory-Constrained"]
            pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            d_cohen = (m1 - m2) / pooled_sd if pooled_sd > 0 else np.nan
            row["llm_mean"] = m1
            row["mc_mean"] = m2
            row["llm_vs_mc_cohens_d"] = d_cohen
            row["llm_vs_mc_pct_gap"] = 100 * (m1 - m2) / abs(m1) if m1 != 0 else np.nan
        if "BabyLM" in stats_by_group and "LLM" in stats_by_group:
            m1, _, _ = stats_by_group["LLM"]
            m3, _, n3 = stats_by_group["BabyLM"]
            row["babylm_mean"] = m3
            row["babylm_n"] = n3
            row["llm_vs_babylm_pct_gap"] = 100 * (m1 - m3) / abs(m1) if m1 != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("dataset")


def scaling_saturation(data: dict) -> pd.DataFrame:
    """For the LLM group (widest size range), fit score ~ log10(n_params)
    separately in the lower half and upper half of the size range; a much
    smaller upper-half slope indicates diminishing returns / saturation."""
    rows = []
    for slug, d in data.items():
        df = d["model_df"]
        sub = df.loc[df["group"] == "LLM"].dropna(subset=["score", "n_params"])
        sub = sub.loc[sub["n_params"] > 0].sort_values("n_params")
        if len(sub) < 8:
            continue
        x = np.log10(sub["n_params"].to_numpy())
        y = sub["score"].to_numpy()
        median = np.median(x)
        lo_mask, hi_mask = x <= median, x > median
        if lo_mask.sum() < 3 or hi_mask.sum() < 3:
            continue
        slope_lo = np.polyfit(x[lo_mask], y[lo_mask], 1)[0]
        slope_hi = np.polyfit(x[hi_mask], y[hi_mask], 1)[0]
        slope_full, _ = np.polyfit(x, y, 1)
        r_full = stats.pearsonr(x, y)[0]
        rows.append({
            "dataset": slug, "n_models": len(sub),
            "slope_lower_half": slope_lo, "slope_upper_half": slope_hi,
            "saturation_ratio": slope_hi / slope_lo if slope_lo != 0 else np.nan,
            "slope_full": slope_full, "r_full": r_full,
        })
    return pd.DataFrame(rows).set_index("dataset")


def perplexity_relationship(data: dict) -> pd.DataFrame:
    """Quadratic fit of score ~ log10(perplexity) for the LLM group: a
    significantly positive quadratic coefficient means score eventually
    *drops* for the very-lowest-perplexity (best) models (inverted-U), not
    just monotonically improves."""
    rows = []
    for slug, d in data.items():
        df = d["model_df"]
        sub = df.loc[df["group"] == "LLM"].dropna(subset=["score", "perplexity"])
        sub = sub.loc[sub["perplexity"] > 0]
        if len(sub) < 8:
            continue
        x = np.log10(sub["perplexity"].to_numpy())
        y = sub["score"].to_numpy()
        # Fit linear and quadratic, compare
        lin = np.polyfit(x, y, 1)
        quad = np.polyfit(x, y, 2)
        lin_pred = np.polyval(lin, x)
        quad_pred = np.polyval(quad, x)
        ss_res_lin = np.sum((y - lin_pred) ** 2)
        ss_res_quad = np.sum((y - quad_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else np.nan
        r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else np.nan
        # best (lowest) 25% perplexity models: is the local slope of score vs log(perplexity) still negative (improving) or flipped?
        thresh = np.percentile(x, 25)
        best_mask = x <= thresh
        if best_mask.sum() >= 3:
            local_slope = np.polyfit(x[best_mask], y[best_mask], 1)[0]
        else:
            local_slope = np.nan
        rows.append({
            "dataset": slug, "n_models": len(sub),
            "quad_coef": quad[0], "r2_linear": r2_lin, "r2_quadratic": r2_quad,
            "delta_r2": r2_quad - r2_lin,
            "slope_best_quartile_perplexity": local_slope,
        })
    return pd.DataFrame(rows).set_index("dataset")


def peak_layer_depth(data: dict) -> pd.DataFrame:
    """Per-model relative depth (layer/n_layers) of peak score, from each
    dataset's own layer_df (already computed by the loaders), then averaged
    per group per dataset."""
    rows = []
    for slug, d in data.items():
        layer_df = d.get("layer_df", pd.DataFrame())
        if layer_df.empty:
            continue
        peaks = (
            layer_df.dropna(subset=["score", "layer_frac"])
            .sort_values("score")
            .groupby(["model_id", "group"], as_index=False)
            .tail(1)
        )
        for g in GROUP_ORDER:
            vals = peaks.loc[peaks["group"] == g, "layer_frac"].dropna().to_numpy()
            if len(vals) >= 2:
                rows.append({
                    "dataset": slug, "group": g, "mean_peak_depth": vals.mean(),
                    "sem_peak_depth": sem(vals), "n_models": len(vals),
                })
    return pd.DataFrame(rows)


def small_model_spotlight(data: dict) -> None:
    """cbr_rnn (17M, Memory-Constrained) and BabyLM (80-140M) vs param-matched peers."""
    print("\n" + "=" * 70)
    print("SPOTLIGHT: cbr_rnn (17M recurrent, Memory-Constrained) percentile rank")
    print("=" * 70)
    for slug, d in data.items():
        df = d["model_df"]
        row = df.loc[df["model_id"] == "cclark/cbr-rnn"]
        if row.empty:
            continue
        score = row["score"].iloc[0]
        # compare against all models with n_params < 100M (small-model regime)
        small = df.loc[df["n_params"] < 100e6].dropna(subset=["score"])
        if len(small) < 3:
            continue
        pct = 100 * (small["score"] < score).mean()
        print(f"  {slug:22s}  cbr_rnn score={score:.4f}  percentile among n_params<100M (n={len(small)}): {pct:5.1f}%")

    print("\n" + "=" * 70)
    print("SPOTLIGHT: BabyLM models (80-140M) vs similarly-sized peers")
    print("=" * 70)
    for slug, d in data.items():
        df = d["model_df"]
        baby = df.loc[df["group"] == "BabyLM"].dropna(subset=["score"])
        if baby.empty:
            continue
        peers = df.loc[(df["n_params"] >= 60e6) & (df["n_params"] <= 200e6)].dropna(subset=["score"])
        if len(peers) < 3:
            continue
        for _, r in baby.iterrows():
            pct = 100 * (peers["score"] < r["score"]).mean()
            print(f"  {slug:22s}  {r['label']:20s} score={r['score']:.4f}  percentile among 60-200M peers (n={len(peers)}): {pct:5.1f}%")


def main() -> None:
    data = load_all()

    print("\n" + "=" * 70)
    print("1. CROSS-DATASET RANK CONSISTENCY (Spearman rho between datasets' model scores)")
    print("=" * 70)
    corr, wide, n_common = cross_dataset_consistency(data)
    print(corr.round(2))
    print("\nN common models per dataset pair (diagonal = N models with a score):")
    print(n_common)

    print("\n" + "=" * 70)
    print("2. GROUP EFFECTS: LLM vs Memory-Constrained vs BabyLM per dataset")
    print("=" * 70)
    ge = group_effects(data)
    print(ge.round(3))

    print("\n" + "=" * 70)
    print("3. SCALING-LAW SATURATION (LLM group): slope in lower vs upper half of size range")
    print("=" * 70)
    sat = scaling_saturation(data)
    print(sat.round(4))

    print("\n" + "=" * 70)
    print("4. PERPLEXITY RELATIONSHIP (LLM group): linear vs quadratic fit")
    print("=" * 70)
    ppl = perplexity_relationship(data)
    print(ppl.round(4))

    print("\n" + "=" * 70)
    print("5. PEAK-LAYER DEPTH (relative layer of best score) by group")
    print("=" * 70)
    peak = peak_layer_depth(data)
    piv = peak.pivot_table(index="dataset", columns="group", values="mean_peak_depth")
    print(piv.round(3))

    small_model_spotlight(data)

    out_dir = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "outputs" / "benchmark_summary" / "_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_dir / "cross_dataset_rank_correlation.csv")
    ge.to_csv(out_dir / "group_effects.csv")
    sat.to_csv(out_dir / "scaling_saturation.csv")
    ppl.to_csv(out_dir / "perplexity_relationship.csv")
    peak.to_csv(out_dir / "peak_layer_depth.csv")
    print(f"\nSaved CSVs to {out_dir}")


if __name__ == "__main__":
    main()
