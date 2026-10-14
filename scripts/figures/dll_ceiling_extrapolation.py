#!/usr/bin/env python3
"""Plot the DLL noise ceiling vs number of source subjects + Schrimpf-style exponential
extrapolation (output of scripts/ceilings/dll_n400.py --mode extrap), with model scores as
reference lines.

    .venv/bin/python scripts/figures/dll_ceiling_extrapolation.py \
        --json $SCRATCH/ceilings_recheck/delta_loglik_ceiling_szewczyk_paper_n400_oos_szewczyk_paper_Federmeieretal2007_extrap.json \
        --models-glob 'outputs_jeanzay/szewczyk/surprisal_sweep_f2007/gpt2/*/szewczyk_lme_by_layer.csv' \
        --out figure.png --title 'Federmeier et al. (2007)'
"""
import argparse, csv, glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8b8a84", "#e6e5e0"
BLUE = "#2a78d6"      # series 1 (validated reference palette); single series, so one hue

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--models-glob", default=None)
ap.add_argument("--split-half", type=float, default=None, help="finite-N split-half ceiling to mark")
ap.add_argument("--in-sample", action="store_true", help="model lines = delta_loglik_train / n_obs (in-sample)")
ap.add_argument("--title", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()

d = json.load(open(a.json))
pts = d["points"]
x = np.array([p["size"] for p in pts], float)
v = np.array([p["per_obs"] for p in pts], float)
sizes = sorted(set(x))
mean = np.array([v[x == s].mean() for s in sizes])
sem = np.array([v[x == s].std(ddof=1) / np.sqrt((x == s).sum()) for s in sizes])
v0, (lo, hi), tau = d["ceiling_delta_loglik_per_obs"], d["ci95"], d["tau0"]

# the exponential is only meaningful if it actually bends within (or near) the observed range
identifiable = (tau < 5 * max(sizes)) and (hi < 20 * max(lo, 1e-6)) and hi < 1.0
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.scatter(x, v, s=10, color=BLUE, alpha=0.18, linewidths=0, zorder=2)            # individual fits
xs = np.linspace(0, max(sizes) * (2.2 if identifiable else 1.0), 300)
ax.plot(xs, v0 * (1 - np.exp(-xs / tau)), color=BLUE, lw=2, zorder=3)              # fitted curve
ax.errorbar(sizes, mean, yerr=1.96 * sem, fmt="o", color=BLUE, ms=6, mfc="#fcfcfb", mec=BLUE, mew=2,
            elinewidth=1.5, capsize=0, zorder=4)                                   # mean per source size
if identifiable:
    ax.axhspan(lo, hi, color=BLUE, alpha=0.10, lw=0, zorder=1)
    ax.axhline(v0, color=BLUE, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(xs[-1], v0, f"  extrapolated ceiling {v0:.4f}\n  (95% CI {lo:.4f}–{hi:.4f})", va="center", ha="left",
            color=INK, fontsize=8.5, clip_on=False)
else:
    ax.text(xs[-1], mean[-1], f"  no plateau reached:\n  exponential fit not\n  identifiable\n  (v0 95% CI {lo:.3f} to >{min(hi, 10):.0f})",
            va="center", ha="left", color=INK, fontsize=8.5, clip_on=False)
if a.split_half is not None:
    ax.axhline(a.split_half, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(0.3, a.split_half, f"split-half (finite N) {a.split_half:.4f}", va="bottom", ha="left", color=INK2, fontsize=8)
if a.models_glob:
    best = {}
    for f in glob.glob(a.models_glob):
        rows = [r for r in csv.DictReader(open(f)) if not r["error"] and r["delta_loglik_test"]]
        if rows:
            b = max(rows, key=lambda r: float(r["delta_loglik_test"]))
            if a.in_sample:
                rows = [r for r in rows if r.get("delta_loglik_train")]
                b = max(rows, key=lambda r: float(r["delta_loglik_train"]))
                best[Path(f).parent.name.replace("openai-community_", "")] = float(b["delta_loglik_train"]) / float(b["n_obs"])
            else:
                best[Path(f).parent.name.replace("openai-community_", "")] = float(b["delta_loglik_test"]) / float(b["n_obs_test"])
    for name in (min(best, key=best.get), max(best, key=best.get)) if best else ():
        ax.axhline(best[name], color=INK2, lw=1, zorder=1)
        ax.text(0.3, best[name], f"{name} (best layer) {best[name]:.4f}", va="bottom", ha="left", color=INK2, fontsize=8)
ax.set_xlim(0, xs[-1]); top = max(hi if identifiable else 0, max(best.values()) if a.models_glob and best else 0, mean.max() + 1.96 * sem.max())
ax.set_ylim(min(0, v.min()) * 1.1, top * 1.15)
ax.set_xlabel("Number of source subjects behind the oracle", color=INK2)
ax.set_ylabel("ΔLL per trial (in-sample)" if a.in_sample else "ΔLL per held-out trial", color=INK2)
ax.set_title((a.title + " — " if a.title else "") + "DLL noise ceiling vs. number of subjects", loc="left", color=INK, fontsize=11)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8.5)
fig.text(0.01, 0.005, "dots: individual fits · circles: mean ± 95% CI per size · line: v0(1−exp(−x/τ0)), Schrimpf et al. 2021",
         color=MUTED, fontsize=7)
fig.subplots_adjust(left=0.13, right=0.74, top=0.90, bottom=0.14)
fig.savefig(a.out, facecolor=fig.get_facecolor()); print("wrote", a.out)
