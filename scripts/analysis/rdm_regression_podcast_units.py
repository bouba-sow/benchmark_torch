#!/usr/bin/env python3
"""Per-unit RDM regression (held-out delta LL) for Podcast ECoG: (subject, electrode, lag).

Same design as rdm_regression_pereira_units.py. Electrodes are subject specific (6-63 per subject, 285 in total), so no
group units. The LME outcome has 32 lags (-0.2 .. 0.78 s, 32 Hz); to keep the number of regressions manageable the lags
are scored one by one by default (--lag-width 1: 285 electrodes x 32 lags); --lag-width 8 averages them into four 250 ms
windows (-0.20..0.05, 0.05..0.30, 0.30..0.55, 0.55..0.80 s). The response of each unit is z-scored.

  items        : all 5134 words (complete for every subject); 100k pairs sampled per subject, same for all models
  response RDM : |z(y_i) - z(y_j)|
  confounds    : |d word length|, |d word frequency| (LME nuisances wlen, freq)
  model RDM    : surprisal -> |d word surprisal| (last layer, as in the LME table); embedding -> 1 - Pearson of
                 last-layer word states, each feature z-scored across the words
  regression   : plain OLS (base y ~ 1 + confounds, full + model RDM), item-fold CV, delta LL per held-out pair.

Parallelism over (subject, lag block): --lag-chunks 4 blocks of 8 lags, --shard I/N over the 36 (subject, block) jobs. Output:
<out-root>/<track>/units_<subject>_c<block>.csv, resumable
per model. Never writes under outputs/.
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

from neuroai.analysis.rdm_regression import RdmCvPlan, standardize_features
from scripts.analysis import podcast_embedding_rsa as PE
from scripts.analysis import surprisal_rsa as S
from scripts.analysis.rdm_regression_epoch_units import blocked_corr_rdm

MAX_PAIRS = 100_000
N_LAGS = 32  # 32 Hz lags, time = -0.2 + k/32 s


def lag_windows(width: int) -> dict:
    """Windows of ``width`` consecutive lags; width=1 is every lag separately."""
    out = {}
    for a in range(0, N_LAGS, width):
        b = min(a + width, N_LAGS)
        out[f"lags{a:02d}-{b - 1:02d}_{-0.2 + a / 32:+.3f}_{-0.2 + (b - 1) / 32:+.3f}"] = slice(a, b)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", metavar="I/N")
    ap.add_argument("--lag-chunks", type=int, default=4, help="split the 32 lags into this many blocks that are scored by separate shards")
    ap.add_argument("--lag-width", type=int, default=1, help="lags averaged per unit (1 = all 32 lags separately; 8 = four 250 ms windows)")
    args = ap.parse_args()
    WINDOWS = lag_windows(args.lag_width)
    if ROOT / "outputs" in args.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    sur_tasks = S._tasks("podcast", ("",))
    fam, mid, rev = sur_tasks[0]
    npy, tpath = S._neural_cache_paths("podcast", fam, mid, rev)
    table = pd.read_parquet(tpath, columns=["subject", "electrode", "chunk", "word_idx", "wlen", "freq"])
    y_all = np.load(npy, mmap_mode="r")
    jobs = [(sub, c) for sub in sorted(table.subject.unique()) for c in range(args.lag_chunks)]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        jobs = jobs[i::n]
    block = N_LAGS // args.lag_chunks
    emb_tasks = []
    for family, model_id, revision in sur_tasks:
        cache = PE._cache_for(model_id, revision)
        if cache is not None:
            emb_tasks.append((family, model_id, revision, cache))
    S.MAX_PAIRS = MAX_PAIRS

    for subject, chunk in jobs:
        part = table[table.subject == subject]
        idx = part.index.to_numpy()
        words = np.sort(part.word_idx.unique()).astype(np.int64)
        n = len(words)
        wpos = {w: k for k, w in enumerate(words)}
        cov = part.drop_duplicates("word_idx").set_index("word_idx").loc[words, ["wlen", "freq"]]
        _, _, pairs = S._pair_distances(np.arange(n, dtype=float), np.arange(n, dtype=float), "podcast", subject)
        conf = np.column_stack([np.abs(cov[c].to_numpy(float)[pairs[:, 0]] - cov[c].to_numpy(float)[pairs[:, 1]]) for c in ("wlen", "freq")])
        seed = int.from_bytes(hashlib.sha256(f"podcast:{subject}".encode()).digest()[:4], "little")
        yv = np.asarray(y_all[idx], dtype=np.float32)
        wrow = part.word_idx.map(wpos).to_numpy()
        plans = []
        for el, g in part.groupby("electrode"):
            rows = part.index.get_indexer(g.index)
            for wname, sl in WINDOWS.items():
                if sl.start // block != chunk:
                    continue
                col = np.full(n, np.nan)
                col[wrow[rows]] = yv[rows][:, sl].mean(axis=1)
                if not np.isfinite(col).all() or col.std() == 0:
                    continue
                z = (col - col.mean()) / col.std()
                plans.append((str(el), wname, RdmCvPlan(np.abs(z[pairs[:, 0]] - z[pairs[:, 1]]), n, pairs, conf, args.n_repeats, seed=seed)))
        print(f"{subject} lag block {chunk}: {n} words, {len(plans)} units", flush=True)
        for track in (["embedding", "surprisal"] if args.track == "both" else [args.track]):
            out = args.out_root / track / f"units_{subject}_c{chunk}.csv"
            done = pd.read_csv(out, dtype={"revision": str, "subject": str, "electrode": str}) if out.is_file() else pd.DataFrame()
            seen = set(zip(done.model_id, done.revision)) if len(done) else set()
            tasks = emb_tasks if track == "embedding" else [(f, m, r, None) for f, m, r in sur_tasks]
            if args.limit:
                tasks = tasks[:args.limit]
            rows_out, count = done.to_dict("records") if len(done) else [], 0
            for family, model_id, revision, cache in tasks:
                if (model_id, revision) in seen:
                    continue
                try:
                    if track == "surprisal":
                        v = S._load_neural_surprisal("podcast", family, model_id, revision)
                        if v is None:
                            continue
                        v = v[v.subject.astype(str) == str(subject)].copy()
                        v["w"] = v.stimulus_id.str.rsplit("::", n=1).str[1].astype(np.int64)
                        s = v.groupby("w").surprisal.mean().reindex(words).to_numpy(np.float64)
                        if not np.isfinite(s).all():
                            print(f"skip {model_id}: missing surprisal", file=sys.stderr, flush=True)
                            continue
                        rdm = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
                    else:
                        with np.load(cache) as z:
                            f = z["features"]
                        st = np.asarray((f[:, None, :] if f.ndim == 2 else f)[:, -1, :], dtype=np.float32)
                        rdm = blocked_corr_rdm(standardize_features(st[words]), pairs)
                    if not np.isfinite(rdm).all() or rdm.std() == 0:
                        raise ValueError("bad model RDM")
                    for el, wname, plan in plans:
                        r = plan.delta_ll(rdm)
                        rows_out.append(dict(track=track, family=family, model_id=model_id, revision=revision, subject=subject,
                                             electrode=el, window=wname, n_items=n, n_pairs=plan.n_pairs,
                                             delta_ll_per_obs=r.delta_ll_per_obs, delta_ll_per_obs_sd=r.delta_ll_per_obs_sd,
                                             rsa_pearson=plan.pearson(rdm), frac_pairs_tested=r.frac_pairs_tested))
                    count += 1
                except Exception:
                    print(f"ERROR {model_id}@{revision}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                if count and count % 10 == 0:
                    out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows_out).to_csv(out, index=False)
            if rows_out:
                out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows_out).to_csv(out, index=False)
            print(f"{subject} {track}: {count} new models", flush=True)


if __name__ == "__main__":
    main()
