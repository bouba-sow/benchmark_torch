#!/usr/bin/env python3
"""Summary table of the RDM-regression noise ceilings next to the model scores, for both tracks.

Ceiling = Timkey et al. (2026) split-half delta LL in RDM space (neuroai/analysis/rdm_ceiling.py): 15 random splits of the
participants, one half's item-mean RDM predicts the other half's beyond the confound RDMs, both directions summed and divided
by the total number of test pairs. Neural datasets use the same grouping as the headline regression (scripts/analysis/
rdm_ceiling_units.py, run on halves of the subjects); behavioral datasets use the existing split-half ceilings
($SCRATCH/rdm_ceiling). Fedorenko has no units shared across subjects: its ceiling uses the item-mean response of one half of the subjects as the shared predictor (rdm_ceiling_fedorenko.py); it is NEGATIVE, so Fedorenko is left out of the ceiling-normalised Pareto.

Model columns (per track): median and best last-layer delta LL over the 118 models with the headline aggregation
(scripts/analysis/rdm_regression_neural_summary.py; behavioral: $SCRATCH/rdm_report_std). model/ceiling uses the FULL-sample
model score against a HALF-sample ceiling, so it is conservative in the models' favour (ratios can exceed 1); the like-for-like
half-target scoring is not part of this table.

Writes reports/rsa_summary/rdm_regression/rdm_ceiling_table_{dll,rsa}.{csv,md}. The RSA table covers the neural datasets only
(RSA = the plain Pearson r between the model RDM and the unit response RDM stored by the regression, same units and target as the ceiling).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
SCR = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
OUT = ROOT / "reports/rsa_summary/rdm_regression"
NEURAL = [("pereira", "pereira", "Pereira · fMRI"), ("tuckute", "tuckute", "Tuckute B · fMRI"),
          ("broderick", "broderick", "Broderick · EEG"), ("megmasc", "megmasc", "MEG-MASC · MEG"),
          ("podcast", "podcast", "Podcast · ECoG (ROIs)"), ("nieuwland", "nieuwland", "Nieuwland · N400"),
          ("f2007", "szewczyk_f2007", "Federmeier 2007 · N400"), ("h2019", "szewczyk_h2019", "Hubbard 2019 · N400")]
BEHAV = {"provo": "Provo", "meco": "MECO", "geco": "GECO", "onestop": "OneStop", "naturalstories": "Natural Stories",
         "ucl_reading": "UCL Reading", "zuco_reading": "ZuCo"}


def neural_ceiling(key: str) -> tuple[float, float]:
    d = pd.read_csv(SCR / f"rdm_ceiling_units/ceilings_{key}.csv")
    if key == "pereira":
        d = d.groupby("passage_set")[["dll_ceiling", "rsa_ceiling"]].mean()
    elif key == "podcast":
        d = d[d.roi != "netw"]
    return float(d.dll_ceiling.mean()), float(d.rsa_ceiling.mean())


def main():
    ms = pd.read_csv(SCR / "rdm_neural_summary/neural_rdm_summary.csv")
    beh = pd.read_csv(SCR / "rdm_report_std/rdm_last_layer_results.csv", dtype={"revision": str})
    beh = beh[beh.dataset.isin(BEHAV) & (beh.confounds == "len_freq")]
    dll_rows, rsa_rows, values = [], [], []

    def stats(prefix, v, ceil):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        ok = len(v) and ceil is not None and np.isfinite(ceil) and ceil > 0
        return {f"{prefix}_n_models": len(v), f"{prefix}_median": float(np.median(v)) if len(v) else np.nan,
                f"{prefix}_best": float(v.max()) if len(v) else np.nan,
                f"{prefix}_best_over_ceiling": float(v.max() / ceil) if ok else np.nan,
                f"{prefix}_n_above_ceiling": int((v > ceil).sum()) if ok else np.nan,
                f"{prefix}_frac_pos": float((v > 0).mean()) if len(v) else np.nan}

    for key, ds, label in NEURAL:  # Fedorenko excluded (its ceiling is negative, rdm_ceiling_fedorenko.py)
        c, rc = neural_ceiling(key)  # Fedorenko: item-mean predictor ceiling (scripts/analysis/rdm_ceiling_fedorenko.py)
        for rows, ceil, col in ((dll_rows, c, "delta_ll_per_obs"), (rsa_rows, rc, "rsa_pearson")):
            r = dict(dataset=label, ceiling=ceil)
            for t in ("embedding", "surprisal"):
                r.update(stats(t, ms[(ms.dataset == ds) & (ms.track == t)][col], ceil))
            rows.append(r)
            if col == "delta_ll_per_obs":
                values.append(dict(dataset_key=ds, unit="all", ceiling_dll=ceil, ceiling_rsa=rc,
                                   n_above_embedding=r["embedding_n_above_ceiling"], n_above_surprisal=r["surprisal_n_above_ceiling"]))
    for ds, label in BEHAV.items():
        cs = pd.read_csv(SCR / f"rdm_ceiling/{ds}/ceiling_summary.csv")
        cs = cs[cs.confounds == "len_freq"].drop_duplicates(["unit"])[["unit", "ceiling_delta_ll_per_obs"]]
        for _, u in cs.iterrows():
            r = dict(dataset=f"{label} · {u.unit}", ceiling=float(u.ceiling_delta_ll_per_obs))
            for t in ("embedding", "surprisal"):
                r.update(stats(t, beh[(beh.dataset == ds) & (beh.unit == u.unit) & (beh.track == t)].last_layer_delta_ll_per_obs, r["ceiling"]))
            dll_rows.append(r)
            values.append(dict(dataset_key=ds, unit=u.unit, ceiling_dll=r["ceiling"], ceiling_rsa=np.nan,
                               n_above_embedding=r["embedding_n_above_ceiling"], n_above_surprisal=r["surprisal_n_above_ceiling"]))
    v = pd.DataFrame(values)
    # a dataset is left out of the Pareto fronts only when it has no usable ceiling (decided independently of model scores;
    # models above the ceiling are kept and reported, since the split-half predictor is a noisy reference, not a hard bound)
    bad = v.groupby("dataset_key").apply(lambda g: bool((~(g.ceiling_dll > 0)).any()), include_groups=False)
    v["excluded_from_pareto"] = v.dataset_key.map(bad)
    OUT.mkdir(parents=True, exist_ok=True)
    v.to_csv(OUT / "rdm_ceiling_values.csv", index=False)
    f = lambda x, p=2: "n/a" if not np.isfinite(x) else (f"{x:.2e}" if p == 0 else f"{x:.{p}f}")  # noqa: E731
    texts = []
    for name, rows in (("dll", dll_rows), ("rsa", rsa_rows)):
        t = pd.DataFrame(rows)
        t.to_csv(OUT / f"rdm_ceiling_table_{name}.csv", index=False)
        unit = "ΔLL per pair" if name == "dll" else "Pearson r"
        head = [f"| Dataset | ceiling ({unit}) | emb. best | emb. best/ceiling | emb. models > ceiling | emb. median | surp. best | surp. best/ceiling | surp. models > ceiling | surp. median |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for r in t.itertuples():
            n_e, n_s = r.embedding_n_models, r.surprisal_n_models
            head.append(f"| {r.dataset} | {f(r.ceiling, 0 if name == 'dll' else 3)} | {f(r.embedding_best, 0 if name == 'dll' else 3)} | "
                        f"{f(r.embedding_best_over_ceiling)} | {'n/a' if not np.isfinite(r.embedding_n_above_ceiling) else f'{int(r.embedding_n_above_ceiling)}/{n_e}'} | "
                        f"{f(r.embedding_median, 0 if name == 'dll' else 3)} | {f(r.surprisal_best, 0 if name == 'dll' else 3)} | "
                        f"{f(r.surprisal_best_over_ceiling)} | {'n/a' if not np.isfinite(r.surprisal_n_above_ceiling) else f'{int(r.surprisal_n_above_ceiling)}/{n_s}'} | "
                        f"{f(r.surprisal_median, 0 if name == 'dll' else 3)} |")
        (OUT / f"rdm_ceiling_table_{name}.md").write_text("\n".join(head) + "\n")
        texts.append(f"### {'ΔLL' if name == 'dll' else 'RSA'} ceiling\n\n" + "\n".join(head))
    print("\n\n".join(texts))


if __name__ == "__main__":
    main()
