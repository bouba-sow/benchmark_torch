#!/usr/bin/env python3
"""Per-unit RDM regression (held-out delta LL) for Tuckute 2023, condition B, language + MD + DMN ROIs.

Same design as rdm_regression_pereira_units.py. Unit = (subject, ROI) with ROI from the supplementary
'allroi' file (lang_*, md_*, dmn_*; the 360 Glasser parcels are left out, 4320 units would not fit in memory)
plus a GROUP unit per ROI (mean of the z-scored responses over the 10 subjects).

  response RDM : |z(y_i) - z(y_j)|, y = ROI response to B sentence i, z-scored across the ~988 sentences
  confounds    : |d mean word length|, |d mean frequency| (the LME nuisances, pooled-table columns)
  model RDM    : surprisal -> |d sentence surprisal| (last layer); embedding -> 1 - Pearson of last-layer
                 states, each feature z-scored across the sentences
  regression   : plain OLS, base y ~ 1 + confounds vs full + model RDM, item-fold CV, delta LL per held-out pair.

Parallelism is over subjects (--shard I/N over [10 subjects + GROUP]); each shard scores every model.
Output: <out-root>/<track>/units_<subject>.csv (resumable per model). Never writes under outputs/.
"""
from __future__ import annotations

import argparse
import hashlib
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
from neuroai.datasets.tuckute import TuckuteDataset
from neuroai.sweep import safe_model_dir
from scripts.analysis import other_neural_rsa as O
from scripts.analysis import surprisal_rsa as S
from scripts.analysis.rdm_regression_pereira_units import corr_rdm

SI = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug")) / "tuckute_si/data_SI/brain-allroi-data_participant_20231001.csv"
PREFIXES = ("lang_", "md_", "dmn_")


def seed_for(subject: str) -> int:
    return int.from_bytes(hashlib.sha256(f"tuckute:{subject}".encode()).digest()[:4], "little")


def load_responses():
    df = pd.read_csv(SI, usecols=["roi", "response_target", "item_id", "target_UID", "cond"])
    df = df[(df.cond == "B") & df.roi.str.startswith(PREFIXES)]
    out = {}
    for sub, part in df.groupby("target_UID"):
        w = part.pivot_table(index="item_id", columns="roi", values="response_target", aggfunc="mean")
        out[str(sub)] = w
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--limit", type=int, help="score only the first N models per track")
    ap.add_argument("--shard", metavar="I/N")
    args = ap.parse_args()
    if ROOT / "outputs" in args.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    resp = load_responses()
    subjects = sorted(resp) + ["GROUP"]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        subjects = subjects[i::n]
    ds = TuckuteDataset(ROOT / "data/tuckute_fmri/osfstorage")
    sentences, _, all_ids = ds.get_condition("all")
    pos = {int(i): k for k, i in enumerate(all_ids)}
    enc_root = ROOT / "outputs_jeanzay/tuckute/embedding_sweep"
    cache_root = enc_root / ".embed_cache"
    tasks_emb = O._tasks("tuckute", enc_root)
    tasks_sur = S._tasks("tuckute", ("",))
    fam, mid, rev = tasks_sur[0]
    sent_tab = (pd.read_parquet(S._neural_cache_paths("tuckute", fam, mid, rev)[1], columns=["sentence_id", "mean_wlen", "mean_freq"])
                .assign(sentence_id=lambda d: d.sentence_id.astype(int)).drop_duplicates("sentence_id").set_index("sentence_id"))
    tracks = ["embedding", "surprisal"] if args.track == "both" else [args.track]

    for subject in subjects:
        if subject == "GROUP":
            common = sorted(set.intersection(*[set(w.dropna(how="all").index) for w in resp.values()]))
            cols = sorted(set.intersection(*[set(w.columns) for w in resp.values()]))
            z = []
            for w in resp.values():
                x = w.loc[common, cols].to_numpy(float)
                z.append((x - x.mean(0)) / x.std(0))
            Y, ids, rois = np.mean(z, axis=0), np.array(common, int), cols
        else:
            w = resp[subject]
            ok = w.notna().all(axis=1)
            w = w[ok]
            Y, ids, rois = w.to_numpy(float), w.index.to_numpy(int), list(w.columns)
        keep = np.isfinite(Y).all(0) & (Y.std(0) > 0)
        Y, rois = Y[:, keep], [r for r, k in zip(rois, keep) if k]
        order = np.argsort(ids); ids, Y = ids[order], Y[order]
        n = len(ids); pairs = all_pairs(n)
        conf = np.column_stack([np.abs(sent_tab.loc[ids, c].to_numpy(float)[pairs[:, 0]] - sent_tab.loc[ids, c].to_numpy(float)[pairs[:, 1]])
                                for c in ("mean_wlen", "mean_freq")])
        plans = []
        for j, roi in enumerate(rois):
            y = (Y[:, j] - Y[:, j].mean()) / Y[:, j].std()
            plans.append((roi, RdmCvPlan(np.abs(y[pairs[:, 0]] - y[pairs[:, 1]]), n, None, conf, args.n_repeats, seed=seed_for(subject))))
        print(f"{subject}: {n} sentences, {len(plans)} ROIs", flush=True)
        rows_idx = np.array([pos[int(i)] for i in ids])
        for track in tracks:
            out = args.out_root / track / f"units_{subject}.csv"
            done = pd.read_csv(out, dtype={"revision": str, "subject": str}) if out.is_file() else pd.DataFrame()
            seen = set(zip(done.model_id, done.revision)) if len(done) else set()
            tasks = tasks_emb if track == "embedding" else tasks_sur
            if args.limit:
                tasks = tasks[:args.limit]
            rows, count = done.to_dict("records") if len(done) else [], 0
            for family, model_id, revision in tasks:
                if (model_id, revision) in seen:
                    continue
                try:
                    if track == "surprisal":
                        v = S._load_neural_surprisal("tuckute", family, model_id, revision)
                        if v is None:
                            continue
                        sv = v.assign(sentence_id=v.sentence_id.astype(int)).drop_duplicates("sentence_id").set_index("sentence_id").surprisal
                        s = sv.reindex(ids).to_numpy(np.float64)
                        if not np.isfinite(s).all():
                            continue
                        rdm = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
                    else:
                        mdir = cache_root / safe_model_dir(model_id) / revision.replace("/", "_")
                        path = mdir / "embeddings__embv2.npz"  # fixed features of the 34 models changed in the 2026-10-12 audit
                        if not path.exists():
                            path = mdir / "embeddings.npz"
                        if not path.exists():
                            continue
                        st = O._load_tuckute(path, sentences)[rows_idx]
                        rdm = corr_rdm(standardize_features(st), pairs)
                    if not np.isfinite(rdm).all() or rdm.std() == 0:
                        raise ValueError("bad model RDM")
                    for roi, plan in plans:
                        r = plan.delta_ll(rdm)
                        rows.append(dict(track=track, family=family, model_id=model_id, revision=revision, subject=subject,
                                         roi=roi, atlas=roi.split("_")[0], n_items=n, n_pairs=plan.n_pairs,
                                         delta_ll_per_obs=r.delta_ll_per_obs, delta_ll_per_obs_sd=r.delta_ll_per_obs_sd,
                                         rsa_pearson=plan.pearson(rdm), frac_pairs_tested=r.frac_pairs_tested))
                    count += 1
                except Exception:
                    print(f"ERROR {model_id}@{revision}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                if count and count % 10 == 0:
                    out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False)
            if rows:
                out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{subject} {track}: {count} new models", flush=True)


if __name__ == "__main__":
    main()
