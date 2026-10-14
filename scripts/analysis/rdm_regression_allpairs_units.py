#!/usr/bin/env python3
"""ALL-PAIRS per-unit RDM regression (held-out delta LL) for Podcast (ROI GROUP units, 32 lags) and MEG-MASC (channel GROUP units).

Same design as rdm_regression_podcast_roi_units.py / rdm_regression_epoch_units.py (docs/rdm_regression_neural_units.md): response RDM
|z(y_i) - z(y_j)|, confounds |d word length| and |d word frequency|, plain OLS, item-fold CV (same item partitions as the production
runs: same seed), delta LL per held-out pair + the RSA correlation, last layer only. The difference: every pair of words is used
(Podcast 13.2M, MEG-MASC 29.5M) instead of a random subset of 100k / 250k pairs, and the memory stays bounded by processing ONE unit at
a time (a CV plan holds ~40 bytes per pair), so a task needs one core.

Three steps, all resumable, nothing is written under outputs/:
  prepare    (once)            builds the units: words, covariates, one z-scored response column per unit  -> <out>/<ds>/prep.npz + units.csv
  emb-cache  (array over models) all-pairs embedding RDM of each model from a Gram matrix (1 - Pearson of the per-feature standardized
                               last-layer states), float32, one .npy per model                          -> <out>/<ds>/emb_rdm/
  score      (array over units)  for each of its units and each model of each track: delta LL + RSA     -> <out>/<ds>/<track>/units_<shard>.csv
``--cap N`` reproduces the production random pair sample (surprisal_rsa._pair_distances, same seed) instead of all pairs; it is used to
check that this script returns the production numbers.

    python rdm_regression_allpairs_units.py prepare   podcast --out-root $SCRATCH/rdm_allpairs
    python rdm_regression_allpairs_units.py emb-cache podcast --out-root ... --shard I/N
    python rdm_regression_allpairs_units.py score     podcast --out-root ... --shard I/N [--track surprisal|embedding|both]
"""
from __future__ import annotations

import argparse
import glob
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
from neuroai.analysis.rdm_regression_dense import DenseDesign
from scripts.analysis import podcast_embedding_rsa as PE
from scripts.analysis import rdm_regression_epoch_units as E
from scripts.analysis import rdm_regression_podcast_roi_units as P
from scripts.analysis import surprisal_rsa as S
from scripts.analysis import word_neural_embedding_rsa as W

SCR = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
DATASETS = ("podcast", "megmasc", "broderick")
UNIT_COLS = {"podcast": ["subject", "roi", "lag", "lag_time_s"], "megmasc": ["subject", "kind", "unit"]}


# ----------------------------------------------------------------------------------------------- data preparation (copied from production)
def prepare_epoch(ds: str, kind: str = "channel", min_subjects: int = E.MIN_GROUP_SUBJECTS):
    """GROUP units of rdm_regression_epoch_units.py (same items, same columns) for megmasc / broderick; the columns of the requested ``kind`` (channel | region) are returned as units; the item set is the same for both."""
    E.DS = ds
    elig = json.load(open(SCR / f"rdm_regression_r30/{ds}/eligible_ids.json"))["ids"]
    files = {Path(f).name.split("_")[0]: f for f in sorted(glob.glob(E.CONFIG[ds]["epochs"]))}
    fam, mid, rev = S._tasks(ds, ("",))[0]
    table = pd.read_parquet(S._neural_cache_paths(ds, fam, mid, rev)[1],
                            columns=["subject", "region", "word_idx", E.CONFIG[ds]["value"], "wlen", "freq"])
    word_cov = table.drop_duplicates("word_idx").set_index("word_idx")[["wlen", "freq"]]
    reg = {s: p.pivot_table(index="word_idx", columns="region", values=E.CONFIG[ds]["value"], aggfunc="mean") for s, p in table.groupby("subject")}
    del table

    def subject_matrix(s):
        ids, mean, chs = E.load_subject(files[s])
        words = np.array(sorted(int(k.split("::")[1]) for k in elig[s]))
        pos = {w: i for i, w in enumerate(ids)}
        keep = np.array([w in pos for w in words])
        words = words[keep]
        Y = mean[[pos[w] for w in words]]
        R = reg[s].reindex(words).to_numpy(float)
        cols = [("channel", c) for c in chs] + [("region", r) for r in reg[s].columns]
        return words, np.column_stack([Y, R]), cols

    allw = np.array(sorted({int(k.split("::")[1]) for k in elig[next(iter(elig))]}))
    acc, cnt = {}, {}
    for s in files:
        words, Y, cols = subject_matrix(s)
        idx = np.searchsorted(allw, words)
        for j, key in enumerate(cols):
            col = Y[:, j]
            ok = np.isfinite(col)
            if ok.sum() < 100 or np.nanstd(col) == 0:
                continue
            z = np.full(len(allw), np.nan)
            z[idx[ok]] = (col[ok] - col[ok].mean()) / col[ok].std()
            a = acc.setdefault(key, np.zeros(len(allw)))
            c = cnt.setdefault(key, np.zeros(len(allw)))
            ok_idx = idx[ok]
            a[ok_idx] += z[ok_idx]
            c[ok_idx] += 1
    cols = [k for k in sorted(acc) if (cnt[k] >= min_subjects).sum() > 1000]
    good = np.all([cnt[k] >= min_subjects for k in cols], axis=0)
    words = allw[good]
    Y = np.column_stack([acc[k][good] / cnt[k][good] for k in cols])
    ok_cols = [j for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).all() and Y[:, j].std() > 0]
    finite_rows = np.isfinite(Y[:, ok_cols]).all(axis=1)
    words, Y = words[finite_rows], Y[finite_rows]
    cols = [cols[j] for j in ok_cols]
    Y = Y[:, ok_cols]
    keep = [j for j, (k, _) in enumerate(cols) if k == kind]
    units = pd.DataFrame([dict(subject="GROUP", kind=kind, unit=cols[j][1]) for j in keep])
    resp = np.column_stack([(Y[:, j] - Y[:, j].mean()) / Y[:, j].std() for j in keep]).astype(np.float32)
    cov = word_cov.loc[words]
    return words, cov.wlen.to_numpy(float), cov.freq.to_numpy(float), resp, units


def prepare_podcast(min_subjects: int = 3):
    """GROUP ROI units of rdm_regression_podcast_roi_units.py: (GROUP, roi) x 32 lags."""
    sur_tasks = S._tasks("podcast", ("",))
    fam, mid, rev = sur_tasks[0]
    npy, tpath = S._neural_cache_paths("podcast", fam, mid, rev)
    table = pd.read_parquet(tpath, columns=["subject", "electrode", "chunk", "word_idx", "wlen", "freq"])
    y_all = np.load(npy, mmap_mode="r")
    sel = pd.read_csv(P.MAP)
    sel = sel[sel.selected & sel.assigned & (sel.hemisphere == "L")][["subject", "channel", "region"]]
    words = np.sort(table.word_idx.unique()).astype(np.int64)
    n = len(words)
    wpos = {w: k for k, w in enumerate(words)}
    cov = table.drop_duplicates("word_idx").set_index("word_idx").loc[words, ["wlen", "freq"]]
    resp = {}
    for subject, part in table.groupby("subject"):
        rows_by_el = part.groupby("electrode").indices
        per_roi = {}
        for _, r in sel[sel.subject == subject].iterrows():
            if r.channel not in rows_by_el:
                continue
            idx = part.index.to_numpy()[rows_by_el[r.channel]]
            e = np.full((n, P.N_LAGS), np.nan, np.float32)
            e[part.word_idx.map(wpos).to_numpy()[rows_by_el[r.channel]]] = np.asarray(y_all[idx], dtype=np.float32)
            e = P.zcol(e)
            per_roi.setdefault(r.region, []).append(e)
            per_roi.setdefault("netw", []).append(e)
        for roi, lst in per_roi.items():
            resp[(subject, roi)] = P.zcol(np.nanmean(np.stack(lst), axis=0))
    group = {}
    for roi in sorted({k[1] for k in resp}):
        subs = [v for (s, r), v in resp.items() if r == roi]
        if len(subs) >= min_subjects:
            group[("GROUP", roi)] = P.zcol(np.mean(np.stack(subs), axis=0))
    cols, units = [], []
    for (subject, roi), y in group.items():
        for lag in range(P.N_LAGS):
            col = y[:, lag]
            if not np.isfinite(col).all() or col.std() == 0:
                continue
            cols.append(col.astype(np.float32))
            units.append(dict(subject=subject, roi=roi, lag=lag, lag_time_s=-0.2 + lag / 32))
    return words, cov.wlen.to_numpy(float), cov.freq.to_numpy(float), np.column_stack(cols), pd.DataFrame(units)


def load_prep(root: Path, ds: str):
    z = np.load(root / ds / "prep.npz")
    return z["words"], z["wlen"], z["freq"], z["resp"], pd.read_csv(root / ds / "units.csv")


# ----------------------------------------------------------------------------------------------- models
def surprisal_tasks(ds: str):
    return S._tasks(ds, ("",))


def embedding_states(ds: str, family: str, model_id: str, revision: str, words: np.ndarray):
    """Last-layer states of the words, per-feature standardized across the words (as in production)."""
    if ds != "podcast":
        cache = {(f, m, r): c for f, m, r, c, _ in W._tasks(ds)}.get((family, model_id, revision))
        if cache is None:
            return None
        st = W._last_layer_states(cache)
    else:
        cache = PE._cache_for(model_id, revision)
        if cache is None:
            return None
        with np.load(cache) as z:
            f = z["features"]
        st = np.asarray((f[:, None, :] if f.ndim == 2 else f)[:, -1, :], dtype=np.float32)
    return standardize_features(np.asarray(st[words], dtype=np.float32))


def gram_rdm(x: np.ndarray) -> np.ndarray:
    """1 - Pearson between all pairs of rows of x, in all_pairs() order (upper triangle, row-major), float32, via one Gram matrix."""
    x = np.asarray(x, dtype=np.float32).copy()
    x -= x.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    if (nrm <= np.finfo(np.float32).tiny).any():
        raise ValueError("Constant hidden-state vector")
    x /= nrm
    g = x @ x.T
    n = len(x)
    out = np.empty(n * (n - 1) // 2, dtype=np.float32)
    o = 0
    for i in range(n - 1):
        m = n - i - 1
        out[o:o + m] = 1.0 - np.clip(g[i, i + 1:], -1.0, 1.0)
        o += m
    return out


def pair_abs_diff(v: np.ndarray) -> np.ndarray:
    """|v_i - v_j| for all pairs i < j, in all_pairs() order, without index arrays."""
    return np.concatenate([np.abs(v[i] - v[i + 1:]) for i in range(len(v) - 1)])


def safe(s: str) -> str:
    return s.replace("/", "_")


def emb_path(root: Path, ds: str, family: str, model_id: str, revision: str, cap: int | None = None) -> Path:
    return root / ds / (f"emb_rdm_cap{cap}" if cap else "emb_rdm") / f"{family}__{safe(model_id)}__{safe(revision)}.npy"


def capped_pairs(ds: str, n: int, cap: int) -> np.ndarray:
    """The production random pair sample (surprisal_rsa._pair_distances, same seed) with ``cap`` pairs."""
    S.MAX_PAIRS = cap
    lab = ("podcast", "roi") if ds == "podcast" else (ds, "GROUP")
    return S._pair_distances(np.arange(n, dtype=float), np.arange(n, dtype=float), *lab)[2]


def surprisal_vector(ds: str, family: str, model_id: str, revision: str, words: np.ndarray):
    v = S._load_neural_surprisal(ds, family, model_id, revision)
    if v is None:
        return None
    v = v.assign(w=v.stimulus_id.str.rsplit("::", n=1).str[1].astype(np.int64))
    s = v.groupby("w").surprisal.mean().reindex(words).to_numpy(np.float64)
    return s if np.isfinite(s).all() else None


# ----------------------------------------------------------------------------------------------- steps
def step_prepare(a):
    out = a.out_root / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    words, wlen, freq, resp, units = (prepare_podcast() if a.dataset == "podcast" else prepare_epoch(a.dataset, a.kind))
    np.savez(out / "prep.npz", words=words, wlen=wlen, freq=freq, resp=resp)
    units.to_csv(out / "units.csv", index=False)
    n = len(words)
    print(f"{a.dataset}: {n} words ({n * (n - 1) // 2} pairs), {len(units)} units", flush=True)


def shard_of(items, spec):
    if not spec:
        return items
    i, n = (int(x) for x in spec.split("/"))
    return items[i::n]


def step_emb_cache(a):
    words, *_ = load_prep(a.prep_root or a.out_root, a.dataset)
    tasks = shard_of(surprisal_tasks(a.dataset), a.shard)
    if a.models:
        keep = {l.strip() for l in open(a.models) if l.strip()}
        tasks = [t for t in tasks if t[1] in keep or t[1].split("/")[-1] in keep]
    (emb_path(a.out_root, a.dataset, "x", "x", "x", a.cap).parent).mkdir(parents=True, exist_ok=True)
    pairs = capped_pairs(a.dataset, len(words), a.cap) if a.cap else None
    for family, model_id, revision in tasks:
        path = emb_path(a.out_root, a.dataset, family, model_id, revision, a.cap)
        if path.is_file():
            continue
        try:
            st = embedding_states(a.dataset, family, model_id, revision, words)
            if st is None:
                print(f"no embedding cache for {model_id}", flush=True)
                continue
            rdm = E.blocked_corr_rdm(st, pairs).astype(np.float32) if a.cap else gram_rdm(st)
            if not np.isfinite(rdm).all() or rdm.std() == 0:
                raise ValueError("bad model RDM")
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, rdm)
            tmp.rename(path)
            print(f"cached {model_id}", flush=True)
        except Exception:
            print(f"ERROR {model_id}@{revision}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def step_score(a):
    ds = a.dataset
    words, wlen, freq, resp, units = load_prep(a.prep_root or a.out_root, ds)
    dense = (not a.cap) and a.engine == "dense"
    # RDM_DROP_WORDS: json list of word ids to leave out for every model and both tracks (v2 surprisal has no value for
    # them in some models, e.g. Broderick segment-initial words). Embedding RDMs cached over all prep words are subset
    # to the remaining pairs (emb_pair_keep), so the emb-cache step need not be rerun.
    emb_pair_keep = None
    if os.environ.get("RDM_DROP_WORDS"):
        drop = {int(w) for w in json.load(open(os.environ["RDM_DROP_WORDS"]))}
        keepw = np.array([int(w) not in drop for w in words])
        if not keepw.all():
            if not dense:
                raise ValueError("RDM_DROP_WORDS is only supported with the dense all-pairs engine")
            i, j = np.triu_indices(len(words), 1)
            emb_pair_keep = keepw[i] & keepw[j]
            words, wlen, freq, resp = words[keepw], wlen[keepw], freq[keepw], resp[keepw]
            print(f"{ds}: dropped {int((~keepw).sum())} words (RDM_DROP_WORDS)", flush=True)
    n = len(words)
    seed = int.from_bytes(hashlib.sha256(b"podcast:roi" if ds == "podcast" else f"{ds}:GROUP".encode()).digest()[:4], "little")
    if dense:  # all pairs, matrix-product engine (neuroai/analysis/rdm_regression_dense.py): no pair index arrays
        pairs = None
        design = DenseDesign(n, [pair_abs_diff(wlen), pair_abs_diff(freq)], a.n_repeats, seed=seed)
        n_pairs = n * (n - 1) // 2
    else:
        if a.cap:  # the production random pair sample, with a.cap pairs
            pairs = capped_pairs(ds, n, a.cap)
        else:
            pairs = all_pairs(n)
        conf = np.column_stack([np.abs(c[pairs[:, 0]] - c[pairs[:, 1]]) for c in (wlen, freq)])
        n_pairs = len(pairs)
    print(f"{ds}: {n} words, {n_pairs} pairs, {len(units)} units, engine={'dense' if dense else 'sparse'}", flush=True)

    tracks = ["embedding", "surprisal"] if a.track == "both" else [a.track]
    tasks = surprisal_tasks(ds)
    if a.models:
        keep = {l.strip() for l in open(a.models) if l.strip()}
        tasks = [t for t in tasks if t[1] in keep or t[1].split("/")[-1] in keep]
    sur = {}
    if "surprisal" in tracks:
        for family, model_id, revision in tasks:
            s = surprisal_vector(ds, family, model_id, revision, words)
            if s is None:
                print(f"skip {model_id}: no surprisal", file=sys.stderr, flush=True)
            else:
                sur[(family, model_id, revision)] = s
    my_units = shard_of(list(range(len(units))), a.shard)
    if a.limit_units:
        my_units = my_units[:a.limit_units]
    tag = (a.shard or "0/1").replace("/", "of")
    outs = {t: a.out_root / ds / t / f"units_s{tag}.csv" for t in tracks}
    rows = {}
    done = {}
    for t, f in outs.items():
        d = pd.read_csv(f, dtype={"revision": str, "subject": str}) if f.is_file() else pd.DataFrame()
        rows[t] = d.to_dict("records") if len(d) else []
        done[t] = {int(u) for u in d.unit_idx} if len(d) else set()
    for ui in my_units:
        todo = [t for t in tracks if ui not in done[t]]
        if not todo:
            continue
        lab = units.iloc[ui].to_dict()
        y = resp[:, ui].astype(np.float64)
        if dense:
            plan = design.plan(pair_abs_diff(y))
        else:
            plan = RdmCvPlan(np.abs(y[pairs[:, 0]] - y[pairs[:, 1]]), n, pairs, conf, a.n_repeats, seed=seed)
        for track in todo:
            models = [(f, m, r) for f, m, r in tasks if (track == "embedding" and emb_path(a.out_root, ds, f, m, r, a.cap).is_file())
                      or (track == "surprisal" and (f, m, r) in sur)]
            for family, model_id, revision in models:
                try:
                    if track == "surprisal":
                        s = sur[(family, model_id, revision)]
                        rdm = pair_abs_diff(s) if dense else np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
                    else:
                        full = np.load(emb_path(a.out_root, ds, family, model_id, revision, a.cap), mmap_mode="r")
                        rdm = np.asarray(full if emb_pair_keep is None else full[emb_pair_keep], dtype=np.float64)
                    r = plan.delta_ll(rdm, (family, model_id, revision, track)) if dense else plan.delta_ll(rdm)
                    rows[track].append(dict(track=track, family=family, model_id=model_id, revision=revision, unit_idx=ui, **lab,
                                            n_items=n, n_pairs=plan.n_pairs, delta_ll_per_obs=r.delta_ll_per_obs,
                                            delta_ll_per_obs_sd=r.delta_ll_per_obs_sd, rsa_pearson=plan.pearson(rdm),
                                            frac_pairs_tested=r.frac_pairs_tested))
                except Exception:
                    print(f"ERROR {model_id}@{revision} unit {ui}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            outs[track].parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows[track]).to_csv(outs[track], index=False)
            done[track].add(ui)
            print(f"unit {ui} {lab} {track}: {len(models)} models", flush=True)
        del plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "emb-cache", "score"])
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--prep-root", type=Path, help="where prep.npz / units.csv are read (default: --out-root)")
    ap.add_argument("--shard", metavar="I/N", help="emb-cache: every N-th model; score: every N-th unit")
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--kind", choices=["channel", "region"], default="channel", help="prepare (megmasc / broderick): which GROUP units to build")
    ap.add_argument("--engine", choices=["dense", "sparse"], default="dense", help="all pairs: matrix-product engine (default) or the bincount RdmCvPlan (validation)")
    ap.add_argument("--cap", type=int, help="use the production random pair sample of this size instead of all pairs (validation)")
    ap.add_argument("--models", type=Path, help="text file with one model_id per line (default: all models)")
    ap.add_argument("--limit-units", type=int)
    a = ap.parse_args()
    if ROOT / "outputs" in a.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    {"prepare": step_prepare, "emb-cache": step_emb_cache, "score": step_score}[a.step](a)


if __name__ == "__main__":
    main()
