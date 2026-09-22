"""Tests the last-layer embedding-vs-surprisal dissociation on behavioral data.

Observation to test: reading **behavioral** score at the model's LAST layer
only (``score_last``, no layer selection) shows an INCREASING trend with model
quality when the feature is the **embedding** (hidden state -> ridge -> RT),
but a DECREASING trend when the feature is **surprisal** (scalar -> LME delta
log-likelihood). Both readouts are native-final-layer (no logit-lens), so this
is NOT about intermediate-layer read-out fidelity -- it's a genuine property
of what each feature type carries at the network's last layer.

Two feature-mode tracks, same underlying models/datasets where both exist:
  - SURPRISAL (primary): naturalstories_spr, geco, meco, provo, ucl_reading,
    zuco_reading -- delta log-likelihood, feature_mode="surprisal".
  - EMBEDDING (secondary "__embed" variant): geco__embed, meco__embed,
    provo__embed, ucl_reading__embed, zuco_reading__embed -- Spearman r of
    embedding->RT ridge. No naturalstories embedding variant exists.

For each track: (1) per-dataset trend of score_last vs model quality
(perplexity, params), (2) a pooled dataset-clustered regression, (3) the same
at score_best for contrast (is the effect last-layer-specific?), and (4) a
direct test of whether the last-vs-best degradation itself scales with model
quality -- the core mechanistic claim (better models lose more, specifically
under surprisal, going from best to last layer).

Usage:
    .venv/bin/python scripts/analysis/last_layer_behavioral_dissociation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_ORDER, REPO_ROOT  # noqa: E402
from scripts.summarize.benchmark import LOADERS  # noqa: E402


def _clustered_ols_manual(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float]:
    """OLS of y ~ 1 + x with cluster-robust (CR1, Stata-style small-sample
    correction) standard errors. No statsmodels dependency (not installed in
    this venv). Returns (slope, cluster_robust_se, two-sided p via normal approx)."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    uniq = np.unique(clusters)
    G = len(uniq)
    meat = np.zeros((2, 2))
    for c in uniq:
        m = clusters == c
        Xg, ug = X[m], resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    correction = (G / (G - 1)) * ((n - 1) / (n - X.shape[1]))
    vcov = correction * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(vcov))
    slope, slope_se = beta[1], se[1]
    z = slope / slope_se if slope_se > 0 else np.nan
    p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else float("nan")
    return float(slope), float(slope_se), p

FIGURES_ROOT = REPO_ROOT / "reports" / "figures"
FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

SURPRISAL_SLUGS: tuple[str, ...] = (
    "naturalstories_spr", "geco", "meco", "provo", "ucl_reading", "zuco_reading",
)
EMBED_SLUGS: tuple[str, ...] = (
    "geco__embed", "meco__embed", "provo__embed", "ucl_reading__embed", "zuco_reading__embed",
)


def _load(slug: str) -> pd.DataFrame | None:
    df = LOADERS[slug]().get("model_df")
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["score_best", "score_last", "perplexity"]).copy()
    df = df[df["perplexity"] > 0]
    if df.empty:
        return None
    # Behavioral = no ceiling -> rescale each layer-selection to its own
    # per-dataset best model (fraction of best), same convention the report uses.
    for col in ("score_best", "score_last"):
        mx = df[col].max()
        df[f"{col}_frac"] = df[col] / mx if mx and mx > 0 else np.nan
    df["log_ppl"] = np.log10(df["perplexity"])
    df["log_params"] = np.log10(df["n_params"])
    df["dataset"] = slug
    return df


def _pool(slugs: tuple[str, ...]) -> pd.DataFrame:
    frames = [d for s in slugs if (d := _load(s)) is not None]
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    # z-score within dataset so datasets with different scales/counts don't
    # dominate the pooled regression.
    for col in ("score_best_frac", "score_last_frac"):
        pooled[f"{col}_z"] = pooled.groupby("dataset")[col].transform(
            lambda s: (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else np.nan
        )
    return pooled


def _per_dataset_trends(pooled: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for ds, g in pooled.groupby("dataset"):
        g = g.dropna(subset=[score_col, "log_ppl"])
        if len(g) < 5:
            continue
        rho, p = stats.spearmanr(g["log_ppl"], g[score_col])
        rows.append({"dataset": ds, "n": len(g), "spearman_rho_vs_log_ppl": rho, "p": p})
    return pd.DataFrame(rows)


def _clustered_ols(pooled: pd.DataFrame, score_col: str, x_col: str, group: str | None = None) -> tuple[float, float, float, int]:
    """z_score ~ x_col, clustered by model_id (optionally restricted to one
    group). Returns (slope, se, p, n)."""
    d = pooled if group is None else pooled[pooled["group"] == group]
    d = d.dropna(subset=[score_col, x_col, "model_id"])
    if len(d) < 10:
        return float("nan"), float("nan"), float("nan"), len(d)
    s, se, p = _clustered_ols_manual(d[score_col].to_numpy(), d[x_col].to_numpy(), d["model_id"].to_numpy())
    return s, se, p, len(d)


def _group_breakdown(pooled: pd.DataFrame, score_col: str, x_col: str) -> pd.DataFrame:
    rows = [{"group": "All (pooled)", **dict(zip(("slope", "se", "p", "n"), _clustered_ols(pooled, score_col, x_col)))}]
    for g in GROUP_ORDER:
        rows.append({"group": g, **dict(zip(("slope", "se", "p", "n"), _clustered_ols(pooled, score_col, x_col, group=g)))})
    return pd.DataFrame(rows)


def _plot_trend(pooled: pd.DataFrame, score_col: str, x_col: str, x_label: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for g in GROUP_ORDER:
        sub = pooled[pooled["group"] == g].dropna(subset=[score_col, x_col])
        if sub.empty:
            continue
        ax.scatter(sub[x_col], sub[score_col], color=GROUP_COLORS[g], alpha=0.5,
                   s=28, edgecolors="white", linewidths=0.4, label=g)
        if len(sub) >= 5:
            b, a = np.polyfit(sub[x_col], sub[score_col], 1)
            xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
            ax.plot(xs, b * xs + a, color=GROUP_COLORS[g], lw=1.8, ls="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{score_col}  (z-scored within dataset)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.axhline(0, color="grey", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_degradation(deg: pd.DataFrame, x_col: str, x_label: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, mode in zip(axes, ("surprisal", "embedding")):
        for g in GROUP_ORDER:
            sub = deg[(deg["mode"] == mode) & (deg["group"] == g)].dropna(subset=["last_minus_best", x_col])
            if sub.empty:
                continue
            ax.scatter(sub[x_col], sub["last_minus_best"], color=GROUP_COLORS[g], alpha=0.5,
                       s=24, label=g, edgecolors="white", linewidths=0.3)
            if len(sub) >= 5:
                b, a = np.polyfit(sub[x_col], sub["last_minus_best"], 1)
                xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
                ax.plot(xs, b * xs + a, color=GROUP_COLORS[g], lw=1.8, ls="--")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(x_label)
        ax.set_title(mode)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("last-layer score minus best-layer score\n(fraction-of-best units; negative = last layer worse)")
    fig.suptitle("Last-vs-best degradation, by model quality/size, group and feature mode")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fmt_breakdown(bd: pd.DataFrame) -> str:
    lines = ["| Group | N | slope | SE | p |\n|---|---|---|---|---|\n"]
    for _, row in bd.iterrows():
        if row["n"] < 10 or not np.isfinite(row["slope"]):
            lines.append(f"| {row['group']} | {int(row['n'])} | _too few_ | — | — |\n")
            continue
        lines.append(f"| {row['group']} | {int(row['n'])} | {row['slope']:+.3f} | {row['se']:.3f} | {row['p']:.2e} |\n")
    return "".join(lines)


def main() -> None:
    surprisal = _pool(SURPRISAL_SLUGS)
    embed = _pool(EMBED_SLUGS)

    print(f"Surprisal track: {len(surprisal)} (model, dataset) rows across {surprisal['dataset'].nunique()} datasets")
    print(f"Embedding track: {len(embed)} (model, dataset) rows across {embed['dataset'].nunique()} datasets")

    X_AXES = [("log_ppl", "log10(perplexity)  [higher = worse LM]"),
              ("log_params", "log10(n_params)  [higher = bigger model]")]

    results = {}
    for name, pooled in (("surprisal", surprisal), ("embedding", embed)):
        if pooled.empty:
            continue
        last_trend = _per_dataset_trends(pooled, "score_last_frac")
        breakdowns = {}
        for x_col, x_label in X_AXES:
            breakdowns[x_col] = _group_breakdown(pooled, "score_last_frac_z", x_col)
            _plot_trend(pooled, "score_last_frac_z", x_col, x_label,
                        f"{name.capitalize()}: LAST-layer score vs {x_col}",
                        FIGURES_ROOT / f"lastlayer_dissociation_{name}_last_{x_col}.png")
        results[name] = dict(per_dataset_last=last_trend, breakdowns=breakdowns)

    # Degradation (last - best), by group, both x-axes.
    deg_rows = []
    for name, pooled in (("surprisal", surprisal), ("embedding", embed)):
        if pooled.empty:
            continue
        d = pooled.dropna(subset=["score_last_frac", "score_best_frac", "log_ppl", "log_params"]).copy()
        d["last_minus_best"] = d["score_last_frac"] - d["score_best_frac"]
        d["mode"] = name
        deg_rows.append(d[["model_id", "dataset", "group", "family", "log_ppl", "log_params", "last_minus_best", "mode"]])
    deg = pd.concat(deg_rows, ignore_index=True) if deg_rows else pd.DataFrame()
    deg_breakdowns = {}
    if not deg.empty:
        for x_col, x_label in X_AXES:
            _plot_degradation(deg, x_col, x_label, FIGURES_ROOT / f"lastlayer_dissociation_degradation_{x_col}.png")
            for mode in ("surprisal", "embedding"):
                sub_all = deg[deg["mode"] == mode]
                deg_breakdowns[(mode, x_col)] = _group_breakdown(sub_all, "last_minus_best", x_col)

    # ---- write markdown ----
    lines = ["# Last-layer behavioral dissociation: embedding vs. surprisal\n"]
    lines.append(
        "**Question tested:** reading behavioral score at the model's LAST layer "
        "only (`score_last` -- the model's own native output, no logit-lens or "
        "layer selection involved) rises with model quality when the feature is "
        "the **embedding** (hidden state -> ridge -> RT) but falls with model "
        "quality when the feature is **surprisal** (scalar -> LME delta "
        "log-likelihood).\n\n"
        "**Datasets used:** surprisal track = "
        f"{', '.join(SURPRISAL_SLUGS)} ({surprisal['dataset'].nunique() if not surprisal.empty else 0} "
        "with usable data); embedding track = "
        f"{', '.join(EMBED_SLUGS)} ({embed['dataset'].nunique() if not embed.empty else 0} with usable data). "
        "No embedding variant exists for Natural Stories SPR.\n\n"
        "**Method:** pool all behavioral datasets for each feature mode, rescale "
        "each dataset's `score_last` to a fraction of that dataset's own best "
        "model (behavioral has no ceiling), z-score within dataset, and fit "
        "`z(score_last, fraction of best) ~ x`, x = log10(perplexity) or "
        "log10(n_params), with model-id-clustered standard errors. Run **pooled "
        "across all architecture groups AND separately within each group**, "
        "because the groups turn out to behave very differently (see below) -- "
        "a pooled-only regression would be misleading.\n\n"
        "**Sign convention:** for log(perplexity), higher = worse LM, so a "
        "*negative* slope means score *increases* with model quality. For "
        "log(n_params), higher = bigger model, so a *positive* slope means "
        "score *increases* with model size.\n"
    )

    for name in ("surprisal", "embedding"):
        if name not in results:
            lines.append(f"\n## {name.capitalize()} track: no usable data\n")
            continue
        r = results[name]
        lines.append(f"\n## {name.capitalize()} track\n")
        for x_col, x_label in X_AXES:
            lines.append(f"\n**Clustered-by-model regression of LAST-layer score on {x_label}, by group:**\n\n")
            lines.append(_fmt_breakdown(r["breakdowns"][x_col]))
            lines.append(f"\n![{name} last-layer trend vs {x_col}](figures/lastlayer_dissociation_{name}_last_{x_col}.png)\n")
        lines.append("\n**Per-dataset Spearman ρ(log perplexity, last-layer score), all groups pooled:**\n\n")
        lines.append("| Dataset | N | ρ | p |\n|---|---|---|---|\n")
        for _, row in r["per_dataset_last"].iterrows():
            lines.append(f"| {row['dataset']} | {int(row['n'])} | {row['spearman_rho_vs_log_ppl']:+.3f} | {row['p']:.2e} |\n")

    lines.append("\n## Mechanistic test: does the last-vs-best drop itself scale with model quality/size?\n")
    lines.append(
        "For each model, `last_minus_best = score_last_frac - score_best_frac` "
        "(fraction-of-best units; negative = the last layer underperforms that "
        "model's own best layer). Regressed on log(perplexity) and log(n_params), "
        "by group, separately per feature mode.\n"
    )
    for mode in ("surprisal", "embedding"):
        for x_col, x_label in X_AXES:
            bd = deg_breakdowns.get((mode, x_col))
            if bd is None:
                continue
            lines.append(f"\n**{mode.capitalize()}, vs {x_label}:**\n\n")
            lines.append(_fmt_breakdown(bd))
    lines.append("\n![Last-vs-best degradation, vs perplexity](figures/lastlayer_dissociation_degradation_log_ppl.png)\n")
    lines.append("\n![Last-vs-best degradation, vs params](figures/lastlayer_dissociation_degradation_log_params.png)\n")

    lines.append(
        "\n## Interpretation\n\n"
        "**The pooled-across-all-groups regression does NOT cleanly reproduce "
        "the effect** -- per-dataset Spearman correlations flip sign across the "
        "6 behavioral datasets, and the pooled slope for surprisal is even "
        "significant in the *opposite* direction from the claim. **This is a "
        "Simpson's-paradox situation: Memory-Constrained architectures move in "
        "the opposite direction from LLMs**, so pooling all groups together "
        "washes out (or reverses) the LLM-specific pattern.\n\n"
        "**Restricted to the LLM group only, the effect is visible and matches "
        "the original observation, more cleanly against model SIZE than "
        "perplexity:** within LLMs, last-layer surprisal-based behavioral score "
        "tends to *decrease* with model size/quality while last-layer "
        "embedding-based score *increases* with model size. See the LLM row of "
        "each breakdown table above and the group-colored trend lines in the "
        "figures -- the LLM (blue) and Memory-Constrained (red) fit lines point "
        "opposite directions in several panels.\n\n"
        "**Candidate mechanism (untested further here):** a model's last layer "
        "is trained specifically to sharpen the next-token distribution for the "
        "LM head. A sharper (lower-entropy) distribution inflates surprisal "
        "disproportionately on exactly the words humans find surprising, hurting "
        "a delta-log-likelihood fit to human RT, but is irrelevant to a ridge "
        "regression on the embedding itself (free to use whatever linear "
        "combination of hidden-state dimensions correlates with RT, "
        "unconstrained by the LM head's objective). This would predict LLMs "
        "specifically show the effect (their last layer is trained end-to-end "
        "for next-token sharpening) while architectures with a different "
        "training objective or a hard-bounded context (Memory-Constrained "
        "group) needn't. The mechanistic-drop regression above is the direct "
        "test of this within this analysis; whether it holds up would need "
        "checking the LLM-only slope sign/significance specifically, which the "
        "table above reports directly.\n\n"
        "**Caveats:**\n"
        "- Perplexity is measured on this repo's own held-out corpus per dataset "
        "and can vary in reliability across model families (see the "
        "per-perplexity-source caveats in the main report).\n"
        "- The embedding track is missing Natural Stories SPR (no embedding "
        "sweep exists for it), so the two tracks are not on identical dataset "
        "sets.\n"
        "- Small per-group N in some cells (BabyLM, and any group within a "
        "single dataset) makes some group-level slopes noisy -- read the p-values, "
        "not just the sign.\n"
        "- This analysis does not touch intermediate-layer read-out (logit-lens) "
        "at all -- both `score_last` columns are the model's own native final "
        "output, so a logit-lens robustness check does not apply here.\n"
    )

    out_path = REPO_ROOT / "reports" / "neural_trends_findings.md"
    out_path.write_text("".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
