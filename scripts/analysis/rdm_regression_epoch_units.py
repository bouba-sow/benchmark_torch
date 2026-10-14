#!/usr/bin/env python3
"""Per-unit RDM regression (held-out delta LL) for Broderick 2018 EEG and MEG-MASC: every channel + the LME regions.

Same design as rdm_regression_pereira_units.py. Unit = (subject, channel) for the ~114 retained scalp channels of
that subject, and (subject, region) for the LME regions (left, right, posterior, occipital; response from the LME
table). The response is the 300-500 ms mean after word onset (the LME outcome window). GROUP units average the
z-scored responses over the subjects that have the channel (>= 10 subjects, only channels/words with enough data).

  items        : words covered by ALL 118 models (eligible_ids.json of the previous RDM-regression run)
  pairs        : 250k pairs sampled deterministically per subject (surprisal_rsa._pair_distances), same for all models
  response RDM : |z(y_i) - z(y_j)|
  confounds    : |d word length|, |d word frequency| (LME nuisances wlen, freq)
  model RDM    : surprisal -> |d word surprisal| (last layer); embedding -> 1 - Pearson of last-layer states,
                 each feature z-scored across the items
  regression   : plain OLS (base y ~ 1 + confounds, full + model RDM), item-fold CV, delta LL per held-out pair.

Parallelism over subjects (--shard I/N over [18 subjects + GROUP]); each shard scores every model.
Output: <out-root>/<track>/units_<subject>.csv, resumable per model. Never writes under outputs/.
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

import glob

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_regression import RdmCvPlan, standardize_features
from scripts.analysis import surprisal_rsa as S
from scripts.analysis import word_neural_embedding_rsa as W

CACHE = "/lustre/fsn1/projects/rech/ldh/una68ug/.cache"
SCR = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
# window convention follows the RSA channel scripts: broderick inclusive of t1, megmasc t < t1
CONFIG = {
    "broderick": dict(epochs=f"{CACHE}/broderick_epochs/sub-*.npz", t0=0.30, t1=0.50, inclusive=True, value="eeg"),
    "megmasc": dict(epochs=f"{CACHE}/megmasc_epochs/sub-*.npz", t0=0.25, t1=0.45, inclusive=False, value="y"),
}
MIN_GROUP_SUBJECTS = 10
DS = "broderick"  # set from --dataset in main()


def blocked_corr_rdm(states: np.ndarray, pairs: np.ndarray, block: int = 16384) -> np.ndarray:
    x = states.astype(np.float32).copy()
    x -= x.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    if (nrm <= np.finfo(np.float32).tiny).any():
        raise ValueError("Constant hidden-state vector")
    x /= nrm
    out = np.empty(len(pairs), dtype=np.float64)
    for s in range(0, len(pairs), block):
        b = pairs[s:s + block]
        out[s:s + len(b)] = 1.0 - np.clip(np.einsum("ij,ij->i", x[b[:, 0]], x[b[:, 1]]), -1.0, 1.0)
    return out


def load_subject(path: str):
    z = np.load(path, allow_pickle=True)
    t = z["times"]
    cfg = CONFIG[DS]
    m = (t >= cfg["t0"] - 1e-9) & ((t <= cfg["t1"] + 1e-9) if cfg["inclusive"] else (t < cfg["t1"] - 1e-9))
    x = np.asarray(z["epochs"][:, :, m], dtype=np.float32).mean(-1)
    w = np.asarray(z["word_indices"], dtype=np.int64)
    ids, inv = np.unique(w, return_inverse=True)
    mean = np.zeros((len(ids), x.shape[1]), np.float32)
    np.add.at(mean, inv, np.nan_to_num(x))
    cnt = np.zeros((len(ids), x.shape[1]), np.float32)
    np.add.at(cnt, inv, np.isfinite(x).astype(np.float32))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, mean / cnt, np.nan)
    names = [str(c) for c in z["ch_names"]] if "ch_names" in z.files else [f"ch{j:03d}" for j in range(mean.shape[1])]
    return ids, mean, names


def zcol(v):
    return (v - np.nanmean(v, axis=0)) / np.nanstd(v, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(CONFIG))
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", metavar="I/N")
    ap.add_argument("--group-only", action="store_true", help="score only the subject-averaged GROUP units (pair-count sensitivity runs)")
    ap.add_argument("--models", type=Path, help="text file with one model_id per line: score only these models")
    args = ap.parse_args()
    if ROOT / "outputs" in args.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    global DS
    DS = args.dataset
    elig = json.load(open(SCR / f"rdm_regression_r30/{DS}/eligible_ids.json"))["ids"]
    files = {Path(f).name.split("_")[0]: f for f in sorted(glob.glob(CONFIG[DS]["epochs"]))}
    subjects = sorted(files) + ["GROUP"]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        subjects = subjects[i::n]
    if args.group_only:
        subjects = ["GROUP"]
    if os.environ.get("RDM_MAX_PAIRS"):  # pair-count sensitivity runs only; default = surprisal_rsa.MAX_PAIRS
        S.MAX_PAIRS = int(os.environ["RDM_MAX_PAIRS"])

    sur_tasks = S._tasks(DS, ("",))
    emb_tasks = [(f, m, r) for f, m, r, _, _ in W._tasks(DS)]
    emb_cache = {(f, m, r): c for f, m, r, c, _ in W._tasks(DS)}
    if args.models:
        keep = {l.strip() for l in open(args.models) if l.strip()}
        sur_tasks = [t for t in sur_tasks if t[1] in keep or t[1].split("/")[-1] in keep]
        emb_tasks = [t for t in emb_tasks if t[1] in keep or t[1].split("/")[-1] in keep]
    fam, mid, rev = sur_tasks[0]
    table = pd.read_parquet(S._neural_cache_paths(DS, fam, mid, rev)[1],
                            columns=["subject", "region", "word_idx", CONFIG[DS]["value"], "wlen", "freq"])
    word_cov = table.drop_duplicates("word_idx").set_index("word_idx")[["wlen", "freq"]]
    reg = {s: p.pivot_table(index="word_idx", columns="region", values=CONFIG[DS]["value"], aggfunc="mean") for s, p in table.groupby("subject")}
    del table

    # Per-subject responses on the eligible words: channels + regions
    def subject_matrix(s):
        ids, mean, chs = load_subject(files[s])
        words = np.array(sorted(int(k.split("::")[1]) for k in elig[s]))
        pos = {w: i for i, w in enumerate(ids)}
        keep = np.array([w in pos for w in words])
        words = words[keep]
        Y = mean[[pos[w] for w in words]]
        R = reg[s].reindex(words).to_numpy(float)
        cols = [("channel", c) for c in chs] + [("region", r) for r in reg[s].columns]
        return words, np.column_stack([Y, R]), cols

    for subject in subjects:
        if subject == "GROUP":
            allw = np.array(sorted({int(k.split("::")[1]) for k in elig[next(iter(elig))]}))
            acc: dict[tuple[str, str], np.ndarray] = {}
            cnt: dict[tuple[str, str], np.ndarray] = {}
            for s in files:
                words, Y, cols = subject_matrix(s)
                idx = np.searchsorted(allw, words)
                for j, key in enumerate(cols):
                    col = Y[:, j]
                    ok = np.isfinite(col)
                    if ok.sum() < 100 or np.nanstd(col) == 0:
                        continue
                    z = np.full(len(allw), np.nan); z[idx[ok]] = (col[ok] - col[ok].mean()) / col[ok].std()
                    a = acc.setdefault(key, np.zeros(len(allw))); c = cnt.setdefault(key, np.zeros(len(allw)))
                    ok_idx = idx[ok]
                    a[ok_idx] += z[ok_idx]
                    c[ok_idx] += 1
            cols = [k for k in sorted(acc) if (cnt[k] >= MIN_GROUP_SUBJECTS).sum() > 1000]
            good = np.all([cnt[k] >= MIN_GROUP_SUBJECTS for k in cols], axis=0)
            words = allw[good]
            Y = np.column_stack([acc[k][good] / cnt[k][good] for k in cols])
        else:
            words, Y, cols = subject_matrix(subject)
        ok_cols = [j for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).all() and Y[:, j].std() > 0]
        # keep only words finite in all kept columns; fall back to per-column NaN handling is not supported
        finite_rows = np.isfinite(Y[:, ok_cols]).all(axis=1)
        words, Y = words[finite_rows], Y[finite_rows]
        cols = [cols[j] for j in ok_cols]; Y = Y[:, ok_cols]
        n = len(words)
        _, _, pairs = S._pair_distances(np.arange(n, dtype=float), np.arange(n, dtype=float), DS, subject)
        conf = np.column_stack([np.abs(word_cov.loc[words, c].to_numpy(float)[pairs[:, 0]] -
                                       word_cov.loc[words, c].to_numpy(float)[pairs[:, 1]]) for c in ("wlen", "freq")])
        seed = int.from_bytes(hashlib.sha256(f"{DS}:{subject}".encode()).digest()[:4], "little")
        plans = []
        for j, (kind, name) in enumerate(cols):
            y = (Y[:, j] - Y[:, j].mean()) / Y[:, j].std()
            plans.append((kind, name, RdmCvPlan(np.abs(y[pairs[:, 0]] - y[pairs[:, 1]]), n, pairs, conf, args.n_repeats, seed=seed)))
        print(f"{subject}: {n} words, {len(plans)} units", flush=True)
        for track in (["embedding", "surprisal"] if args.track == "both" else [args.track]):
            out = args.out_root / track / f"units_{subject}.csv"
            done = pd.read_csv(out, dtype={"revision": str, "subject": str}) if out.is_file() else pd.DataFrame()
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
                        v = S._load_neural_surprisal(DS, family, model_id, revision)
                        if v is None:
                            continue
                        v = v.assign(w=v.stimulus_id.str.rsplit("::", n=1).str[1].astype(np.int64))
                        s = v.groupby("w").surprisal.mean().reindex(words).to_numpy(np.float64)
                        if not np.isfinite(s).all():
                            print(f"skip {model_id}: missing surprisal for some words", file=sys.stderr, flush=True)
                            continue
                        rdm = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
                    else:
                        st = W._last_layer_states(emb_cache[(family, model_id, revision)])
                        st = standardize_features(np.asarray(st[words], dtype=np.float32))
                        rdm = blocked_corr_rdm(st, pairs)
                    if not np.isfinite(rdm).all() or rdm.std() == 0:
                        raise ValueError("bad model RDM")
                    for kind, name, plan in plans:
                        r = plan.delta_ll(rdm)
                        rows.append(dict(track=track, family=family, model_id=model_id, revision=revision, subject=subject,
                                         kind=kind, unit=name, n_items=n, n_pairs=plan.n_pairs,
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
