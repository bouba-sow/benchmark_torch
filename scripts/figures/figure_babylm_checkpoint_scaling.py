#!/usr/bin/env python3
"""Paper figure: BabyLM-2025 GPT-2 (100M) training-checkpoint trajectories
across all 12 behavioural + neural datasets.

The BabyLM baseline ``BabyLM-community/babylm-baseline-100m-gpt2`` publishes 29
training checkpoints named by tokens seen (``chck_1M`` … ``chck_1000M``) plus a
separate ``main`` branch (an early-stopped / best-eval checkpoint, NOT
identical to ``chck_1000M``). This is the single-model analogue of
``scripts/figures/figure9_pythia_checkpoint_scaling.py`` — there is no
parameter-scaling axis, so the trajectory is traced against **training
tokens** instead.

Reads the ``babylm_gpt2_checkpoints`` registry-family sweep output written by
scripts/slurm/jz/surprisal/*/…babylm_gpt2_checkpoints*.sh (surprisal is
last-layer-only, so each per-checkpoint LME fit is on the model's true final
layer). Per-checkpoint corpus perplexity comes from
``outputs_jeanzay/<dataset>_perplexity/<dataset>_perplexity_summary.csv``
(or the per-revision JSONs if the summary is absent).

Two pages, each a 3×5 panel grid (one panel per dataset, grouped SPR →
eye-tracking → N400 EEG → fMRI):

  Page 1 — training trajectory: total ΔLog-Lik (out-of-sample,
    delta_loglik_test) vs training tokens (log-x) over the 29 ``chck_*``
    checkpoints; ``main`` drawn as a star at its nominal 1e9-token position.
  Page 2 — score vs perplexity: the same ΔLog-Lik pooled against corpus
    perplexity (log-x), scatter + dashed quadratic fit in log10(perplexity)
    — matches figure 9's rows C/D (Oh & Schuler 2023 pooled-curve
    convention).

Output: reports/paper/babylm_checkpoint_scaling.pdf
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from neuroai.analysis.benchmark_common import REPO_ROOT
from neuroai.analysis.paper_style import despine, use_paper_style

OUT_PATH = REPO_ROOT / "reports" / "paper" / "babylm_checkpoint_scaling.pdf"
OUTPUTS = REPO_ROOT / "outputs_jeanzay"
MODEL_ID = "BabyLM-community/babylm-baseline-100m-gpt2"
MODEL_DIR = MODEL_ID.replace("/", "_")
MAIN_NOMINAL_TOKENS = 1.0e9  # where to draw the `main` star on the token axis

# slug, display, group, lme-csv path (under outputs_jeanzay), aggregation,
# perplexity-summary path (under outputs_jeanzay)
DATASETS: list[tuple[str, str, str, str, str, str]] = [
    ("naturalstories_spr", "Natural Stories (SPR)", "SPR",
     f"naturalstories/surprisal_sweep_checkpoints/babylm_gpt2_checkpoints/{MODEL_DIR}/spr_lme_checkpoint_scores.csv",
     "row", "naturalstories/spr_perplexity/naturalstories_spr_perplexity_summary.csv"),
    ("ucl_reading", "UCL Reading (SPR)", "SPR",
     f"ucl_reading/surprisal_sweep_checkpoints/babylm_gpt2_checkpoints/{MODEL_DIR}/ucl_reading_spr_lme_checkpoint_scores.csv",
     "row", "ucl_reading/perplexity/ucl_reading_perplexity_summary.csv"),
    ("geco", "GECO (first-pass)", "eye",
     f"geco/surprisal_sweep_checkpoints/fp/babylm_gpt2_checkpoints/{MODEL_DIR}/geco_lme_checkpoint_scores.csv",
     "row", "geco/perplexity/geco_perplexity_summary.csv"),
    ("meco", "MECO (first-pass)", "eye",
     f"meco/surprisal_sweep_checkpoints/fp/babylm_gpt2_checkpoints/{MODEL_DIR}/meco_lme_checkpoint_scores.csv",
     "row", "meco/perplexity/meco_perplexity_summary.csv"),
    ("provo", "Provo (first-pass)", "eye",
     f"provo/surprisal_sweep_checkpoints/fp/babylm_gpt2_checkpoints/{MODEL_DIR}/provo_lme_checkpoint_scores.csv",
     "row", "provo/perplexity/provo_perplexity_summary.csv"),
    ("zuco_reading", "ZuCo Reading (TRT)", "eye",
     f"zuco_reading/surprisal_sweep/nr/babylm_gpt2_checkpoints/{MODEL_DIR}/zuco_reading_trt_lme_checkpoint_scores.csv",
     "row", "zuco_reading/perplexity/zuco_reading_perplexity_summary.csv"),
    ("nieuwland", "Nieuwland (N400)", "N400",
     f"nieuwland/surprisal_sweep/babylm_gpt2_checkpoints/{MODEL_DIR}/nieuwland_lme_by_layer.csv",
     "row", "nieuwland/perplexity/nieuwland_perplexity_summary.csv"),
    ("lpp", "Le Petit Prince (fMRI)", "fMRI",
     f"lpp/surprisal_sweep/babylm_gpt2_checkpoints/{MODEL_DIR}/lpp_lme_by_layer.csv",
     "row", "lpp/perplexity/lpp_perplexity_summary.csv"),
    ("tuckute", "Tuckute (fMRI)", "fMRI",
     f"tuckute/surprisal_sweep/babylm_gpt2_checkpoints/{MODEL_DIR}/tuckute_lme_by_layer.csv",
     "row", "tuckute/perplexity/tuckute_perplexity_summary.csv"),
    ("ns_fmri", "Natural Stories (fMRI)", "fMRI",
     f"ns_fmri/surprisal_sweep/babylm_gpt2_checkpoints/{MODEL_DIR}/ns_fmri_lme_by_roi.csv",
     "froi", "ns_fmri/perplexity/ns_fmri_perplexity_summary.csv"),
    ("fedorenko2016", "Fedorenko 2016 (iEEG)", "iEEG",
     f"fedorenko2016/surprisal_sweep/babylm_gpt2_checkpoints/{MODEL_DIR}/fedorenko2016_lme_by_layer.csv",
     "row", "fedorenko2016/perplexity/fedorenko2016_perplexity_summary.csv"),
    ("podcast", "Podcast ECoG (0-200 ms)", "iEEG",
     f"podcast/surprisal_sweep/babylm_gpt2_checkpoints/{MODEL_DIR}/podcast_lme_by_lag.csv",
     "row", "podcast/lme_perplexity/podcast_perplexity_summary.csv"),
]

GROUP_COLOR = {"SPR": "#4C72B0", "eye": "#55A868", "N400": "#C44E52",
               "fMRI": "#8172B2", "iEEG": "#CCB974"}


def _tokens(revision: str) -> float:
    """`chck_<N>M` -> N*1e6 training tokens; `main` -> NaN (drawn separately)."""
    m = re.fullmatch(r"chck_(\d+)M", revision)
    return float(m.group(1)) * 1e6 if m else np.nan


def _load_scores(rel_csv: str, agg: str) -> pd.DataFrame:
    """revision -> (dll_total, n_obs, dll_per_word) for this dataset.

    ``agg='row'``   : one already-best-layer row per revision.
    ``agg='froi'``  : per (froi, layer) rows — take each froi's max-
                      delta_loglik_test layer, then sum dll_test / n_obs_test
                      across frois (same as figure 9's ns_fmri handling).
    """
    p = OUTPUTS / rel_csv
    if not p.is_file():
        return pd.DataFrame(columns=["revision", "dll_total", "n_obs", "dll_per_word"])
    df = pd.read_csv(p)
    if "delta_loglik_test" not in df.columns or "revision" not in df.columns:
        return pd.DataFrame(columns=["revision", "dll_total", "n_obs", "dll_per_word"])

    rows: list[dict] = []
    for rev, g in df.groupby("revision"):
        g = g.dropna(subset=["delta_loglik_test"])
        if g.empty:
            continue
        if agg == "froi" and "froi" in g.columns:
            best = g.loc[g.groupby("froi")["delta_loglik_test"].idxmax()]
        else:
            best = g.loc[[g["delta_loglik_test"].idxmax()]]
        dll = float(best["delta_loglik_test"].sum())
        n = float(best["n_obs_test"].sum()) if "n_obs_test" in best.columns else np.nan
        rows.append({"revision": str(rev), "dll_total": dll, "n_obs": n,
                     "dll_per_word": dll / n if n and np.isfinite(n) and n > 0 else np.nan})
    return pd.DataFrame(rows)


def _load_perplexity(rel_csv: str, ds_slug: str) -> dict[str, float]:
    """revision -> corpus perplexity for MODEL_ID's babylm_gpt2 checkpoints.

    The per-revision ``*_perplexity.json`` files are the source of truth (one
    written per checkpoint by its prefetch/extract entrypoint). The
    ``*_perplexity_summary.csv`` roll-ups mix model families and have been
    seen stale (e.g. ucl_reading's every-row-identical), so they are only a
    fallback when no JSONs are on disk.
    """
    out: dict[str, float] = {}
    json_dir = OUTPUTS / Path(rel_csv).parent / MODEL_DIR  # e.g. podcast/lme_perplexity/<model>/
    for jp in sorted(json_dir.glob("*_perplexity.json")):
        try:
            d = json.loads(jp.read_text())
            out[str(d["revision"])] = float(d["perplexity"])
        except Exception:  # noqa: BLE001
            pass
    if out:
        return out
    p = OUTPUTS / rel_csv
    if p.is_file():
        df = pd.read_csv(p)
        if "model_id" in df.columns:
            df = df[df["model_id"] == MODEL_ID]
        if {"revision", "perplexity"}.issubset(df.columns):
            for _, r in df.iterrows():
                try:
                    out[str(r["revision"])] = float(r["perplexity"])
                except (TypeError, ValueError):
                    pass
    return out


def _series(rel_csv: str, agg: str, rel_ppl: str, ds_slug: str) -> pd.DataFrame:
    s = _load_scores(rel_csv, agg)
    if s.empty:
        return s
    ppl = _load_perplexity(rel_ppl, ds_slug)
    s["perplexity"] = s["revision"].map(ppl)
    s["tokens"] = s["revision"].map(_tokens)
    s["is_main"] = s["revision"] == "main"
    return s


def _quad_fit(x_log: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ok = np.isfinite(x_log) & np.isfinite(y)
    if ok.sum() < 6:
        return None
    c = np.polyfit(x_log[ok], y[ok], 2)
    xx = np.linspace(x_log[ok].min(), x_log[ok].max(), 100)
    return 10.0**xx, np.polyval(c, xx)


_NCOL = 5
_LETTERS = "abcdefghijklmnopqrst"
_MAIN_C = "#D1495B"
_FIT_C = "#333333"
_SHORT = {
    "naturalstories_spr": "Natural Stories · SPR",
    "ucl_reading": "UCL Reading · SPR",
    "geco": "GECO · first-pass",
    "meco": "MECO · first-pass",
    "provo": "Provo · first-pass",
    "zuco_reading": "ZuCo · TRT",
    "nieuwland": "Nieuwland · N400",
    "lpp": "Le Petit Prince · fMRI",
    "tuckute": "Tuckute · fMRI",
    "ns_fmri": "Natural Stories · fMRI",
    "fedorenko2016": "Fedorenko 2016 · ECoG",
    "podcast": "Podcast · ECoG 0–200 ms",
}


def _new_fig():
    """3-row x 5-col paper grid, tight spacing, no figure title."""
    nrow = -(-len(DATASETS) // _NCOL)  # ceil
    fig, axes = plt.subplots(nrow, _NCOL, figsize=(13.4, 2.55 * nrow + 0.35),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.010, h_pad=0.020,
                                    wspace=0.012, hspace=0.030)
    axes = axes.ravel()
    for ax in axes[len(DATASETS):]:
        ax.remove()
    return fig, axes


def _decorate(ax, i: int, xlabel: str) -> None:
    col = i % _NCOL
    is_bottom = (i + _NCOL) >= len(DATASETS)  # bottom-most populated panel in its column
    ax.set_title(f"({_LETTERS[i]}) {_SHORT[DATASETS[i][0]]}", loc="left", fontsize=7.6,
                 color="#222222", pad=3.5)
    if col == 0:
        ax.set_ylabel("ΔLog-lik (OOS)", fontsize=8)
    if is_bottom:
        ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=6.4, length=2.5, pad=1.5)
    ax.margins(x=0.04)
    despine(ax)


def _page_trajectory(pdf: PdfPages, data: dict[str, pd.DataFrame]) -> None:
    fig, axes = _new_fig()
    for i, (slug, _disp, group, *_rest) in enumerate(DATASETS):
        ax = axes[i]
        s = data[slug]
        c = GROUP_COLOR[group]
        if not s.empty:
            traj = s[~s["is_main"]].sort_values("tokens")
            ax.plot(traj["tokens"], traj["dll_total"], "-o", ms=2.4, lw=0.9,
                    color=c, mfc=c, mec="none")
            mn = s[s["is_main"]]
            if not mn.empty:
                ax.axvline(MAIN_NOMINAL_TOKENS, color=_MAIN_C, lw=0.5, ls=":", alpha=0.5)
                ax.plot([MAIN_NOMINAL_TOKENS], [float(mn["dll_total"].iloc[0])], "*",
                        ms=7.5, color=_MAIN_C, mec="white", mew=0.4, zorder=5)
            ax.set_xscale("log")
        _decorate(ax, i, "training tokens")
    pdf.savefig(fig)
    plt.close(fig)


def _page_perplexity(pdf: PdfPages, data: dict[str, pd.DataFrame]) -> None:
    fig, axes = _new_fig()
    for i, (slug, _disp, group, *_rest) in enumerate(DATASETS):
        ax = axes[i]
        s = data[slug]
        c = GROUP_COLOR[group]
        if not s.empty and s["perplexity"].notna().any():
            reg = s[~s["is_main"]]
            ax.scatter(reg["perplexity"], reg["dll_total"], s=9, color=c,
                       edgecolors="white", linewidths=0.3, zorder=3)
            mn = s[s["is_main"]]
            if not mn.empty and np.isfinite(mn["perplexity"].iloc[0]):
                ax.scatter([mn["perplexity"].iloc[0]], [mn["dll_total"].iloc[0]],
                           marker="*", s=70, color=_MAIN_C, edgecolors="white",
                           linewidths=0.4, zorder=5)
            fit = _quad_fit(np.log10(s["perplexity"].to_numpy(dtype=float)),
                            s["dll_total"].to_numpy(dtype=float))
            if fit is not None:
                ax.plot(fit[0], fit[1], "--", color=_FIT_C, lw=0.9, zorder=4)
            ax.set_xscale("log")
        _decorate(ax, i, "corpus perplexity")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    use_paper_style()
    data = {slug: _series(csv, agg, ppl, slug)
            for slug, _disp, _grp, csv, agg, ppl in DATASETS}
    for k, v in data.items():
        if v.empty:
            print(f"  {k:20s}  NO DATA")
            continue
        n = v["revision"].nunique()
        uniq = v["dll_total"].round(3).nunique()
        flag = "  <-- suspiciously few distinct values (sweep bug?)" if n >= 10 and uniq <= 3 else ""
        print(f"  {k:20s}  {n:2d}/29 checkpoints, {uniq:2d} distinct ΔLL values{flag}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        _page_trajectory(pdf, data)
        _page_perplexity(pdf, data)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
