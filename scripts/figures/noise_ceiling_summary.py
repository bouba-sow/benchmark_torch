#!/usr/bin/env python3
"""Audit + summary PDF for every delta-loglik noise ceiling used by the
matched-scale report's score_norm.

Method (Timkey et al. 2026, PNAS supplement, Section 3.B "Filler model
goodness-of-fit"): for n_splits random equal splits of subjects/items into
halves A/B, compute the per-item mean of the response measure within one
half, use it AS THE PREDICTOR (standing in for a "perfect" surprisal) in a
model fit to the OTHER half's raw trials, and take that model's delta log-
likelihood (full vs. baseline) over the number of observations. Repeat in
both directions, sum, average over splits -> per-observation ceiling.

This script does NOT recompute anything -- it reads every existing
outputs/noise_ceilings/delta_loglik_ceiling_*_oos.json, checks:
  1. Sign (must be positive -- a negative ceiling would flip score_norm's
     sign relative to score_raw when used as a divisor elsewhere).
  2. Split convergence rate (n_splits_converged / n_splits_requested).
  3. Relative noise (ceiling_sd / ceiling -- how much the estimate itself
     wobbles across the 15/10 random splits).
  4. n_obs backing each ceiling (too few observations = an unstable split-
     half mean).
and flags anything outside a reasonable range, then renders one table + one
bar-with-error-bars page.

Output: reports/paper/noise_ceiling_summary.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from neuroai.analysis.paper_style import despine, use_paper_style

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
CEILING_DIR = REPO_ROOT / "outputs" / "noise_ceilings"
OUT_PATH = REPO_ROOT / "reports" / "paper" / "noise_ceiling_summary.pdf"

DATASET_LABELS: dict[str, str] = {
    "naturalstories_spr": "Natural Stories (SPR)",
    "ucl_reading": "UCL Reading (SPR)",
    "geco": "GECO (eye, FP)",
    "meco": "MECO (eye, FP)",
    "provo": "Provo (eye, FP)",
    "zuco_reading": "ZuCo (eye, TRT)",
    "lpp_fmri_persubject": "Le Petit Prince (fMRI)",
    "tuckute": "Tuckute (fMRI)",
    "ns_fmri": "Natural Stories (fMRI)",
    "fedorenko2016": "Fedorenko 2016 (ECoG)",
    "podcast": "Podcast (ECoG)",
}
# Datasets flagged in the ceiling scripts' own docstrings as small-N and
# therefore expected to have a noisier split-half estimate than the rest.
SMALL_N_DATASETS = {"geco": "14 subjects", "zuco_reading": "12 subjects",
                     "fedorenko2016": "5 subjects", "podcast": "9 subjects"}

REL_SD_WARN = 0.30       # ceiling_sd / ceiling above this -> flag
CONVERGENCE_WARN = 0.80  # converged/requested below this -> flag
MIN_N_OBS_WARN = 1000    # smallest per-iteration n_obs below this -> flag


def load_records() -> list[dict]:
    records = []
    for path in sorted(CEILING_DIR.glob("delta_loglik_ceiling_*_oos.json")):
        d = json.loads(path.read_text())
        ceiling = d.get("ceiling_delta_loglik_per_obs")
        sd = d.get("ceiling_delta_loglik_per_obs_sd")
        n_obs_list = d.get("per_iteration_n_obs") or []
        conv = d.get("n_splits_converged") or 0
        req = d.get("n_splits_requested") or 0
        rel_sd = (sd / ceiling) if (ceiling and sd is not None and ceiling != 0) else float("nan")
        conv_rate = conv / req if req else float("nan")
        flags = []
        if ceiling is None or not np.isfinite(ceiling) or ceiling <= 0:
            flags.append("NON-POSITIVE/MISSING CEILING")
        if np.isfinite(rel_sd) and rel_sd > REL_SD_WARN:
            flags.append(f"high relative noise ({rel_sd:.0%})")
        if np.isfinite(conv_rate) and conv_rate < CONVERGENCE_WARN:
            flags.append(f"low convergence ({conv}/{req})")
        if n_obs_list and min(n_obs_list) < MIN_N_OBS_WARN:
            flags.append(f"small n_obs (min={min(n_obs_list)})")
        records.append(dict(
            file=path.name, dataset=d.get("dataset"), method=d.get("method"),
            ceiling=ceiling, sd=sd, rel_sd=rel_sd,
            n_converged=conv, n_requested=req, conv_rate=conv_rate,
            n_obs_min=min(n_obs_list) if n_obs_list else None,
            n_obs_max=max(n_obs_list) if n_obs_list else None,
            flags=flags,
        ))
    return records


def _table_page(pdf: PdfPages, records: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.axis("off")
    lines = [
        "DELTA-LOGLIK NOISE CEILING AUDIT",
        "Method: split-half item-mean-as-predictor ΔLL, out-of-sample "
        "(matches every real model's delta_loglik_test convention)",
        "Reference: Timkey et al. 2026, PNAS supplement §3.B",
        "",
        f"{'dataset':<24}{'ceiling':>10}{'±SD':>10}{'rel.SD':>8}{'splits':>10}"
        f"{'n_obs range':>22}  flags",
        "-" * 110,
    ]
    n_flagged = 0
    for r in records:
        label = DATASET_LABELS.get(r["dataset"], r["dataset"])
        splits = f"{r['n_converged']}/{r['n_requested']}"
        n_obs_rng = f"{r['n_obs_min']:,}–{r['n_obs_max']:,}" if r["n_obs_min"] else "n/a"
        flag_str = "; ".join(r["flags"]) if r["flags"] else "ok"
        if r["flags"]:
            n_flagged += 1
        lines.append(
            f"{label:<24}{r['ceiling']:>10.5f}{r['sd']:>10.5f}{r['rel_sd']:>7.0%}"
            f"{splits:>10}{n_obs_rng:>22}  {flag_str}"
        )
    lines.append("-" * 110)
    lines.append(f"{n_flagged}/{len(records)} datasets flagged" if n_flagged
                 else f"All {len(records)} datasets: positive ceiling, "
                      f"≥{CONVERGENCE_WARN:.0%} split convergence, "
                      f"<{REL_SD_WARN:.0%} relative noise, ≥{MIN_N_OBS_WARN:,} obs/split — no red flags")
    lines.append("")
    lines.append("Small-N datasets (flagged in the ceiling scripts' own docstrings as expected")
    lines.append("to be noisier than the rest, independent of the checks above):")
    for slug, n in SMALL_N_DATASETS.items():
        label = DATASET_LABELS.get(slug, slug)
        r = next((x for x in records if x["dataset"] == slug), None)
        rel_sd_str = f"{r['rel_sd']:.0%} relative SD" if r else "not found"
        lines.append(f"  {label} ({n}): {rel_sd_str}")
    ax.text(0.01, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=7.8,
            family="monospace", va="top", ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def _bar_page(pdf: PdfPages, records: list[dict]) -> None:
    recs = sorted(records, key=lambda r: r["ceiling"] or 0)
    labels = [DATASET_LABELS.get(r["dataset"], r["dataset"]) for r in recs]
    vals = [r["ceiling"] for r in recs]
    errs = [r["sd"] for r in recs]
    colors = ["#d62728" if r["flags"] else "#1f77b4" for r in recs]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(recs))
    ax.barh(y, vals, xerr=errs, color=colors, height=0.65, capsize=3, error_kw=dict(lw=0.9))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("ceiling Δ log-lik / observation (± SD across splits)", fontsize=9)
    ax.set_title("Noise ceiling by dataset", loc="left", fontsize=11, fontweight="bold")
    for slug in SMALL_N_DATASETS:
        if DATASET_LABELS.get(slug) in labels:
            i = labels.index(DATASET_LABELS[slug])
            ax.text(vals[i] + errs[i] + 0.0005, i, "small N", fontsize=6.5, color="#888888", va="center")
    despine(ax)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    use_paper_style()
    records = load_records()
    if not records:
        raise SystemExit(f"No ceiling JSONs found under {CEILING_DIR}")

    print(f"Loaded {len(records)} ceiling files")
    for r in records:
        flag_str = f" -- FLAGGED: {'; '.join(r['flags'])}" if r["flags"] else ""
        print(f"  {r['dataset']:<22} ceiling={r['ceiling']:.5f} sd={r['sd']:.5f} "
              f"rel_sd={r['rel_sd']:.0%} splits={r['n_converged']}/{r['n_requested']}{flag_str}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        _table_page(pdf, records)
        _bar_page(pdf, records)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
