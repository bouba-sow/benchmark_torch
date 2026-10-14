#!/usr/bin/env python3
"""Per-unit RDM regression (held-out delta LL) for Fedorenko 2016 ECoG: one unit per (subject, electrode).

Same design as rdm_regression_pereira_units.py. Electrodes are subject specific, so there are no group units.

  items        : the 331 words covered by ALL 118 models (eligible_ids.json of the previous RDM-regression run)
  response RDM : |z(y_i) - z(y_j)|, y = the electrode's word response (LME table column y), all 54,615 word pairs
  confounds    : |d word length|, |d word frequency| (LME nuisances wlen, freq)
  model RDM    : surprisal -> |d word surprisal| (last layer); embedding -> 1 - Pearson of last-layer word states,
                 each feature z-scored across the 331 words
  regression   : plain OLS (base y ~ 1 + confounds, full + model RDM), item-fold CV, delta LL per held-out pair.

Output: <out-root>/<track>/units.csv (resumable per model). Never writes under outputs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_regression import RdmCvPlan, all_pairs, standardize_features
from neuroai.sweep import safe_model_dir
from scripts.analysis import other_neural_rsa as O
from scripts.analysis import surprisal_rsa as S
from scripts.analysis.rdm_regression_pereira_units import corr_rdm

SCR = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if ROOT / "outputs" in args.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    elig = sorted(json.load(open(SCR / "rdm_regression_r30/fedorenko2016/eligible_ids.json"))["ids"]["*"], key=int)
    sur_tasks = S._tasks("fedorenko2016", ("",))
    fam, mid, rev = sur_tasks[0]
    table = pd.read_parquet(S._neural_cache_paths("fedorenko2016", fam, mid, rev)[1],
                            columns=["subject", "electrode", "stimulus_id", "wlen", "freq", "y"])
    table["stimulus_id"] = table.stimulus_id.astype(str)
    cov = table.drop_duplicates("stimulus_id").set_index("stimulus_id")[["wlen", "freq"]].loc[elig]
    n = len(elig)
    pairs = all_pairs(n)
    conf = np.column_stack([np.abs(cov[c].to_numpy(float)[pairs[:, 0]] - cov[c].to_numpy(float)[pairs[:, 1]]) for c in ("wlen", "freq")])
    plans = []
    for (sub, el), part in table.groupby(["subject", "electrode"]):
        y = part.set_index("stimulus_id").y.reindex(elig).to_numpy(float)
        if not np.isfinite(y).all() or y.std() == 0:
            continue
        y = (y - y.mean()) / y.std()
        seed = int.from_bytes(hashlib.sha256(f"fedorenko2016:{sub}".encode()).digest()[:4], "little")
        plans.append((str(sub), str(el), RdmCvPlan(np.abs(y[pairs[:, 0]] - y[pairs[:, 1]]), n, None, conf, args.n_repeats, seed=seed)))
    print(f"{n} words, {len(plans)} electrode units", flush=True)

    ds_words = O.Fedorenko2016Dataset(ROOT / "data/fedorenko2016").load_word_table()
    stim_row = {str(s): i for i, s in enumerate(ds_words.stimulus_id)}
    rows_idx = np.array([stim_row[s] for s in elig])
    sentences, _, n_words = O._prepare_fedorenko(ROOT / "data/fedorenko2016")
    enc_root = ROOT / "outputs_jeanzay/fedorenko2016/embedding_sweep"
    cache_root = Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/fedorenko2016_features")
    emb_tasks = O._tasks("fedorenko2016", enc_root)

    for track in (["embedding", "surprisal"] if args.track == "both" else [args.track]):
        out = args.out_root / track / "units.csv"
        done = pd.read_csv(out, dtype={"revision": str, "subject": str, "unit": str}) if out.is_file() else pd.DataFrame()
        seen = set(zip(done.model_id, done.revision)) if len(done) else set()
        tasks = emb_tasks if track == "embedding" else sur_tasks
        if args.limit:
            tasks = tasks[:args.limit]
        rows, count = done.to_dict("records") if len(done) else [], 0
        for family, model_id, revision in tasks:
            if (model_id, revision) in seen:
                continue
            try:
                if track == "surprisal":
                    v = S._load_neural_surprisal("fedorenko2016", family, model_id, revision)
                    if v is None:
                        continue
                    s = v.assign(stimulus_id=v.stimulus_id.astype(str)).drop_duplicates("stimulus_id").set_index("stimulus_id").surprisal.reindex(elig).to_numpy(np.float64)
                    if not np.isfinite(s).all():
                        print(f"skip {model_id}: missing surprisal", file=sys.stderr, flush=True)
                        continue
                    rdm = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
                else:
                    paths = list((cache_root / safe_model_dir(model_id) / revision).glob("fedorenko2016_*.npz"))
                    if len(paths) != 1:
                        continue
                    st = O._load_fedorenko(paths[0], sentences, n_words)[rows_idx]
                    rdm = corr_rdm(standardize_features(st), pairs)
                if not np.isfinite(rdm).all() or rdm.std() == 0:
                    raise ValueError("bad model RDM")
                for sub, el, plan in plans:
                    r = plan.delta_ll(rdm)
                    rows.append(dict(track=track, family=family, model_id=model_id, revision=revision, subject=sub,
                                     unit=el, n_items=n, n_pairs=plan.n_pairs, delta_ll_per_obs=r.delta_ll_per_obs,
                                     delta_ll_per_obs_sd=r.delta_ll_per_obs_sd, rsa_pearson=plan.pearson(rdm),
                                     frac_pairs_tested=r.frac_pairs_tested))
                count += 1
            except Exception:
                print(f"ERROR {model_id}@{revision}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            if count and count % 10 == 0:
                out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False)
        if rows:
            out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{track}: {count} new models", flush=True)


if __name__ == "__main__":
    main()
