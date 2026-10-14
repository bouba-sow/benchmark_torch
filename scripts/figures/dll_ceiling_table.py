#!/usr/bin/env python3
"""Regenerates paperwriting/sections/dll_ceiling_table.tex (split-half delta-log-likelihood noise ceilings).

One row per surprisal-track dataset unit (27), from outputs/noise_ceilings/delta_loglik_ceiling_<tag>.json (tags from
neuroai/analysis/benchmark_common.DLL_CEILING_TAGS), plus the fraction of the 118 models whose last-layer score exceeds the
ceiling (outputs/benchmark_summary/<unit>/all_models_scores.csv, column score_last_norm). Rows with a relative SD above 50% or
more than 10% of models above the ceiling are flagged as unreliable: for those the ceiling is not a usable upper bound.

2026-10-12: rebuilt as a script (the previous table was typed by hand and lacked Federmeier 2007, Hubbard 2019, Broderick and
MEG-MASC). The ceilings are used only to describe headroom; the paper's regression uses robust scaling.

Usage: .venv/bin/python scripts/figures/dll_ceiling_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np
import pandas as pd

from neuroai.analysis.benchmark_common import DLL_CEILING_TAGS, REPO_ROOT

OUT_PATH = REPO_ROOT / "paperwriting" / "sections" / "dll_ceiling_table.tex"
SUMMARY = REPO_ROOT / "outputs" / "benchmark_summary"
NC = REPO_ROOT / "outputs" / "noise_ceilings"

LABEL = {
    "geco": "GECO (FP)", "geco_gp": "GECO (GP)", "meco": "MECO (FP)", "meco_gp": "MECO (GP)", "meco_sp": "MECO (SP)",
    "naturalstories_spr": "Natural Stories (SPR)", "ucl_reading": "UCL Reading (SPR)",
    "onestop": "OneStop (FP)", "onestop_gp": "OneStop (GP)", "onestop_sp": "OneStop (SP)",
    "provo": "Provo (FP)", "provo_gp": "Provo (GP)", "provo_sp": "Provo (SP)",
    "zuco_reading": "ZuCo Reading (NR)", "zuco_reading_sr": "ZuCo Reading (SR)", "zuco_reading_tsr": "ZuCo Reading (TSR)",
    "pereira_fmri": "Pereira (fMRI)", "lpp_fmri__surprisal": "Le Petit Prince (fMRI)",
    "ns_fmri__surprisal": "Natural Stories (fMRI)", "tuckute_fmri__surprisal": "Tuckute (fMRI)",
    "podcast_ieeg__surprisal": "Podcast (ECoG)", "fedorenko2016_ecog__surprisal": "Fedorenko2016 (ECoG)",
    "nieuwland_eeg__surprisal": "Nieuwland (N400)", "szewczyk_f2007_eeg__surprisal": "Federmeier et al.\\ (2007) (N400)",
    "szewczyk_h2019_eeg__surprisal": "Hubbard et al.\\ (2019) (N400)", "broderick_eeg__surprisal": "Broderick (EEG)",
    "megmasc_meg__surprisal": "MEG-MASC (MEG)",
}
NOTE = {
    "geco": "small $N$ (14 subj.)", "geco_gp": "small $N$ (14 subj.)", "fedorenko2016_ecog__surprisal": "small $N$ (5 subj.)",
    "pereira_fmri": "small $N$ ($\\leq$10 subj.)", "podcast_ieeg__surprisal": "9 subj.; weighted over 3 regions, 0--600\\,ms lags",
    "zuco_reading": "small $N$ (12 subj.)", "zuco_reading_sr": "small $N$ (12 subj.)", "zuco_reading_tsr": "small $N$ (12 subj.)",
    "szewczyk_f2007_eeg__surprisal": "32 subj.; item-mean oracle", "szewczyk_h2019_eeg__surprisal": "32 subj.; item-mean oracle",
    "broderick_eeg__surprisal": "19 subj.", "megmasc_meg__surprisal": "27 subj.",
}
UNRELIABLE = r"$\ddagger$"


def main() -> None:
    rows = []
    for slug, label in LABEL.items():
        tag = DLL_CEILING_TAGS[slug]
        j = json.load(open(NC / f"delta_loglik_ceiling_{tag}.json"))
        c, sd = float(j["ceiling_delta_loglik_per_obs"]), float(j.get("ceiling_delta_loglik_per_obs_sd", np.nan))
        if "n_splits_converged" in j:
            conv = f"{j['n_splits_converged']}/{j['n_splits_requested']}"
        elif "n_iterations_used" in j:
            conv = f"{j['n_iterations_used']}/{j['n_iterations_requested']}"
        else:
            conv = "--"
        s = pd.read_csv(SUMMARY / slug / "all_models_scores.csv")["score_last_norm"].dropna()
        above = float((s > 1).mean())
        rel = sd / c if c else np.nan
        flag = (rel > 0.5) or (above > 0.10)
        rows.append(dict(label=label + (UNRELIABLE if flag else ""), c=c, sd=sd, rel=rel, conv=conv, above=above,
                         note=NOTE.get(slug, "")))
    lines = [
        r"{\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}p{3.5cm}cccccp{3.0cm}@{}}",
        r"\caption{Delta log-likelihood noise ceilings for the surprisal track, computed by the split-half item-mean-as-predictor "
        r"procedure of Appendix~\ref{app:dll-ceiling}, following \citet{timkey2026dissociation}. Ceiling is per-observation $\Delta LL$; "
        r"``splits conv.'' is the number of converged splits out of those requested. ``Above'' is the share of the 118 models whose "
        r"last-layer $\Delta LL$ exceeds the ceiling. Rows marked $\ddagger$ have a relative SD above 50\% or more than 10\% of models "
        r"above the ceiling; their ceiling is not a usable upper bound. Ceilings are used only to describe headroom; the regression "
        r"uses robust scaling (Appendix~\ref{app:normalization}).}",
        r"\label{tab:dll-ceilings} \\",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Ceiling} & \textbf{$\pm$SD} & \shortstack{\textbf{Rel.}\\\textbf{SD}} & \shortstack{\textbf{Splits}\\\textbf{conv.}} & \textbf{Above} & \textbf{Note} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{c}{\tablename\ \thetable{} -- continued} \\",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Ceiling} & \textbf{$\pm$SD} & \shortstack{\textbf{Rel.}\\\textbf{SD}} & \shortstack{\textbf{Splits}\\\textbf{conv.}} & \textbf{Above} & \textbf{Note} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for r in rows:
        sd = "--" if not np.isfinite(r["sd"]) else f"{r['sd']:.5f}"
        rel = "--" if not np.isfinite(r["rel"]) else f"{r['rel']*100:.0f}\\%"
        lines.append(f"{r['label']} & {r['c']:.5f} & {sd} & {rel} & {r['conv']} & {r['above']*100:.0f}\\% & {r['note']} \\\\")
    lines += [r"\end{longtable}", "}"]
    text = "\n".join(lines)
    OUT_PATH.write_text(text + "\n")
    print(f"wrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
