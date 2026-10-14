#!/usr/bin/env python3
"""Sensitivity of the surprisal headline to HOW the best layer is chosen.

"headline" = the OLD rule (LAYER_SELECTION="test"); "train_sel" = the NEW default (LAYER_SELECTION="train", since 2026-10-01).
Old rule (scripts/summarize/benchmark.py::_best_last_dll_per_obs): per model, take the layer
with the max HELD-OUT delta_loglik_test and report delta_loglik_test / n_obs_test there -- i.e. the layer
is selected on the very data it is scored on (an optimistic, "winner's curse" rule on small test sets).
Alternative: select the layer on the TRAINING delta-log-lik (delta_loglik_train) and report the held-out
value at that layer.  Needs no re-fitting: both columns are already in every per-layer results file.

Every surprisal-track loader goes through _best_last_dll_per_obs, so we run all of them twice, patching
that one function.  Nothing in outputs/ is written; results go to --out.
"""
import argparse, glob, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("bm", REPO / "scripts/summarize/benchmark.py")
bm = importlib.util.module_from_spec(spec); sys.modules["bm"] = bm; spec.loader.exec_module(bm)
import neuroai.analysis.benchmark_common as bc
from neuroai.analysis.benchmark_common import DATASETS, read_lme_by_layer_csv

MISSING = {"n": 0}   # kept for the report line below (train columns are checked by select_best_row)


def run_loaders(slugs):
    out = {}
    for slug in slugs:
        res = {}
        for tag, mode in (("headline", "test"), ("train_sel", "train")):
            bc.LAYER_SELECTION = mode   # select_best_row reads this module global at call time
            try:
                r = bm.LOADERS[slug]()
                md = r["model_df"] if isinstance(r, dict) else r
                res[tag] = md.set_index("model_id")["score_best"].astype(float) if len(md) else pd.Series(dtype=float)
            except Exception as e:  # noqa: BLE001
                res[tag] = pd.Series(dtype=float); print(f"[{slug}] {tag} failed: {type(e).__name__}: {e}", file=sys.stderr)
        out[slug] = res
    bc.LAYER_SELECTION = "train"
    return out


def new_szewczyk_studies():
    out = {}
    for s in ("f2007", "h2019"):
        h, t = {}, {}
        for p in glob.glob(str(REPO / f"outputs_jeanzay/szewczyk_{s}/surprisal_sweep/*/*/szewczyk_lme_by_layer.csv")):
            d = read_lme_by_layer_csv(p); mid = p.split("/")[-2]
            bc.LAYER_SELECTION = "test"; h[mid] = bm._best_last_dll_per_obs(d)[0]
            bc.LAYER_SELECTION = "train"; t[mid] = bm._best_last_dll_per_obs(d)[0]
        out[f"szewczyk_{s}"] = {"headline": pd.Series(h), "train_sel": pd.Series(t)}
    return out


def summarize(slug, r):
    a, b = r["headline"], r["train_sel"]
    j = pd.concat([a.rename("h"), b.rename("t")], axis=1).dropna()
    if len(j) < 5:
        return dict(dataset=slug, n_models=len(j), note="insufficient / no train column")
    top = j["h"].nlargest(10).index
    sp = j["h"].rank().corr(j["t"].rank())
    return dict(dataset=slug, n_models=len(j), headline_mean=j.h.mean(), trainsel_mean=j.t.mean(),
                mean_inflation=j.h.mean() / j.t.mean() if j.t.mean() > 0 else np.nan,
                headline_top10=j.h[top].mean(), trainsel_top10=j.t[top].mean(),
                top10_inflation=j.h[top].mean() / j.t[top].mean() if j.t[top].mean() > 0 else np.nan,
                headline_max=j.h.max(), trainsel_max=j.t.max(), rank_corr=sp, frac_models_lower=(j.t < j.h).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    slugs = [d.slug for d in DATASETS if getattr(d, "feature_mode", "") == "surprisal" and d.slug in bm.LOADERS]
    slugs += [k for k in bm.LOADERS if k.endswith("__surprisal") and k not in slugs]
    print("datasets:", slugs, file=sys.stderr)
    res = run_loaders(slugs); res.update(new_szewczyk_studies())
    tab = pd.DataFrame([summarize(k, v) for k, v in res.items()])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); tab.to_csv(a.out, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(tab.round(4).to_string(index=False)); print("loader calls missing train columns:", MISSING["n"])
