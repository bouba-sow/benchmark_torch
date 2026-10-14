#!/usr/bin/env python3
"""Export the last-layer matched-scale regression rows for the robust-inference analysis (robust_inference.R).

Uses the same data preparation as the paper's Table 2 (scripts/analysis/matched_scale_regression.py, --layer last,
all 27 units kept in both tracks), so the base specification of robust_inference.R must reproduce that table.
Adds the columns the robustness fits need:

  lineage   clustering unit for dependence: the model family, except that the single registry family "babylm" is split
            into its five independent submissions (GPT-2 baselines, GPT-BERT, MoEP, CLASS-IT, BLM-GPT2s), otherwise
            the data contrast would rest on one cluster.
  n_layers  depth, from the benchmark summary.
  gptbert / masked   GPT-BERT models / masked-focus GPT-BERT models (hybrid or masked objectives).
  dagger    unit whose reference is flagged in the paper's ceiling tables (split SD > .5 x reference or >10% of
            models above it): UCL SPR, Broderick, Federmeier 2007, Hubbard 2019 (surprisal track only).

Writes <out>/rows_<track>.csv. Reads outputs only; never refits a benchmark result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

import scripts.analysis.matched_scale_regression as msr

DAGGER_SURPRISAL = ["ucl_reading", "broderick", "szewczyk_f2007", "szewczyk_h2019"]


def _lineage(family: str, model_id: str) -> str:
    if family != "babylm":
        return family
    m = model_id.lower()
    for key, name in (("gpt-bert", "babylm_gptbert"), ("baseline-gpt2", "babylm_gpt2"), ("moep", "babylm_moep"),
                      ("class_it", "babylm_classit"), ("blm-gpt2s", "babylm_blmgpt2s")):
        if key in m:
            return name
    raise ValueError(f"unknown BabyLM submission: {model_id}")


def export(track: str, out: Path) -> Path:
    msr.TRACK, msr.LAYER = track, "last"
    msr.MASTER_PATH = out / f"master_table_{track}_last.csv"
    msr._EXCLUDE_DATASETS = []
    msr.ROBUST_Z = False
    msr._build_last_layer_table(track, "score_last_norm").to_csv(msr.MASTER_PATH, index=False)
    df = msr._prepare()

    depth = {}
    for src in msr._SUMMARY_UNITS[track]:
        s = pd.read_csv(msr.SUMMARY_ROOT / src / "all_models_scores.csv", usecols=["model_id", "n_layers"])
        for m, n in zip(s["model_id"], s["n_layers"]):
            depth.setdefault(m, n)
    df["n_layers"] = df["model_id"].map(depth)
    df["lineage"] = [_lineage(f, m) for f, m in zip(df["family"], df["model_id"])]
    mid = df["model_id"].str.lower()
    df["gptbert"] = mid.str.contains("gpt-bert").astype(int)
    df["masked"] = mid.str.contains("masked-focus").astype(int)
    df["dagger"] = (df["dataset"].isin(DAGGER_SURPRISAL) & (track == "surprisal")).astype(int)
    cols = ["dataset", "modality", "corpus", "model_id", "family", "lineage", "group", "memory", "data", "n_params",
            "n_layers", "perplexity", "log_params", "log_ppl", "gptbert", "masked", "dagger", "score_norm"]
    path = out / f"rows_{track}.csv"
    df[cols].to_csv(path, index=False)
    n = df.groupby("modality", observed=True).agg(rows=("score_norm", "size"), units=("dataset", "nunique"),
                                                  models=("model_id", "nunique"))
    print(f"{track}: wrote {path}\n{n.to_string()}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "robust_inference")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for track in ("embedding", "surprisal"):
        export(track, a.out)


if __name__ == "__main__":
    main()
