#!/usr/bin/env python3
"""Regenerates paperwriting/sections/corr_ceiling_table.tex as a GENUINE
embedding/ridge-track-only noise-ceiling table.

Fixes a real bug found 2026-09-09: the previous corr_ceiling_table.tex was a
verbatim copy of summarize/report.py's "Distance to Noise Ceiling"
table (build_distance_to_ceiling(), which iterates PRIMARY_DATASETS and
reports each dataset's own PRIMARY/canonical track -- ΔLL for behavioral-
primary datasets including Pereira fMRI, correlation for neural-primary
datasets) captioned as if every row were a "√r_SB" correlation-scale
embedding-track ceiling. For every behavioral-family row (and Pereira fMRI,
whose canonical DatasetInfo entry is feature_mode="surprisal"), the table was
silently showing the ΔLL/noise-ceiling value instead (e.g. Pereira fMRI:
0.003, matching outputs/noise_ceilings/delta_loglik_ceiling_pereira_fmri_oos.json
exactly -- NOT the genuine embedding-track split-half ceiling, 0.507, from
outputs/noise_ceilings/noise_ceilings_summary.csv). That ΔLL-scale table is
correctly reproduced separately in dll_ceiling_table.tex; this script instead
builds the actual embedding-only counterpart so the two appendix tables stop
overlapping/conflicting under mismatched captions.

Iterates every DATASETS entry with feature_mode == "embedding" (23 as of
2026-09-09: 6 neural-primary "bare" slugs + 17 "<base>__embed" secondary
variants), plus MEG-MASC (megmasc_meg) pulled in directly the same way
figures/figure3_family_scaling.py's _megmasc_family_df does (excluded
from DATASETS/MODALITY_BUCKET pending its DLL-track ceiling being settled,
but its embedding-track correlation ceiling is fine to include here).

Usage: .venv/bin/python3 scripts/figures/corr_ceiling_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

from neuroai.analysis.benchmark_common import REPO_ROOT
from scripts.summarize.report import DATASETS, _load_model_df, ceiling_for

OUT_PATH = REPO_ROOT / "paperwriting" / "sections" / "corr_ceiling_table.tex"

# Display name override for the embedding-track rows -- DatasetInfo.name for
# "<base>__embed" secondary entries carries an explicit "(embedding)" suffix
# (e.g. "GECO (embedding)") to disambiguate from its surprisal sibling in the
# main report; that's redundant here since EVERY row in this table is the
# embedding track, so it's stripped for a cleaner paper table (matches the
# original table's plain "GECO"/"UCL Reading"/etc. naming).
def _display_name(name: str) -> str:
    return name.replace(" (embedding)", "")


def main() -> None:
    rows = []
    for info in DATASETS:
        if info.feature_mode != "embedding":
            continue
        df = _load_model_df(info.slug)
        ceiling = ceiling_for(info.slug)
        if df is None or df.empty or ceiling is None or ceiling <= 0:
            continue
        df = df.dropna(subset=["score"])
        if df.empty:
            continue
        best = df.loc[df["score"].idxmax()]
        frac = float(best["score"]) / ceiling
        frac_last = float(df["score_last_norm"].max()) if "score_last_norm" in df.columns else float("nan")
        rows.append({"name": _display_name(info.name), "modality": info.modality,
                     "ceiling": ceiling, "frac": frac, "frac_last": frac_last})

    # MEG-MASC: excluded from DATASETS (its DLL/surprisal-track ceiling isn't
    # settled yet, per neuroai/analysis/benchmark_common.py's DATASETS list note) but its
    # embedding-track correlation ceiling is fine -- same special-case
    # figures/figure3_family_scaling.py's _megmasc_family_df() uses.
    meg_df = _load_model_df("megmasc_meg")
    if any(r["name"].startswith("MEG-MASC") for r in rows):
        meg_df = None  # already a DATASETS entry (2026-10-12): do not add a duplicate row
    if meg_df is not None and not meg_df.empty and "score_last_norm" in meg_df.columns:
        meg_ceiling = ceiling_for("megmasc_meg")
        if meg_ceiling is not None and meg_ceiling > 0:
            meg_df = meg_df.dropna(subset=["score"])
            if not meg_df.empty:
                best = meg_df.loc[meg_df["score"].idxmax()]
                rows.append({"name": "MEG-MASC (Gwilliams)", "modality": "MEG",
                             "ceiling": meg_ceiling, "frac": float(best["score"]) / meg_ceiling,
                             "frac_last": float(meg_df["score_last_norm"].max())})

    rows.sort(key=lambda r: r["frac_last"], reverse=True)

    lines = [
        r"{\small\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}p{4.2cm}lccc@{}}",
        r"\caption{Split-half correlation noise ceilings (embedding track, "
        r"$\sqrt{r_{\mathrm{SB}}}$, Appendix~\ref{app:corr-ceiling}) for every dataset "
        r"in the embedding/ridge track, and the fraction of that ceiling reached by "
        r"the best model, with the last layer (our primary rule) and with the best layer. "
        r"These ceilings are used only to describe headroom; the regression uses robust scaling (Appendix~\ref{app:normalization}). Rows marked $\dagger$ have a ceiling below 0.01 -- an "
        r"absolute reliability so low that a high percent-of-ceiling figure reflects "
        r"very little recoverable signal in the human data itself, not a strong model "
        r"fit; we report these for completeness but do not treat them as headline "
        r"results.}",
        r"\label{tab:corr-ceilings} \\",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Modality} & \shortstack{\textbf{Ceiling}\\\textbf{($\sqrt{r_{\mathrm{SB}}}$)}} & \shortstack{\textbf{Best model,}\\\textbf{last layer (\%)}} & \shortstack{\textbf{Best model,}\\\textbf{best layer (\%)}} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{c}{\tablename\ \thetable{} -- continued} \\",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Modality} & \shortstack{\textbf{Ceiling}\\\textbf{($\sqrt{r_{\mathrm{SB}}}$)}} & \shortstack{\textbf{Best model,}\\\textbf{last layer (\%)}} & \shortstack{\textbf{Best model,}\\\textbf{best layer (\%)}} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{5}{r}{\textit{continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for r in rows:
        dagger = r"$\dagger$" if r["ceiling"] < 0.01 else ""
        lines.append(f"{r['name']}{dagger} & {r['modality']} & {r['ceiling']:.3f} & {r['frac_last']*100:.0f}\\% & {r['frac']*100:.0f}\\% \\\\")
    lines.append(r"\end{longtable}")
    lines.append("}")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
