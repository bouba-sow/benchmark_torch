#!/usr/bin/env python3
"""ROI-level RDM regression (held-out delta LL) for Podcast ECoG, all 32 lags.

Electrodes are grouped with the previous ROI analysis: outputs/podcast/electrode_roi_map_language_responsive.csv
(scripts/data/podcast_electrode_roi_map.py + podcast_electrode_selection.py): the 285 language-responsive electrodes of the
benchmark, assigned to the Fedorenko language parcels IFGorb, IFG, MFG, AntTemp, PostTemp. Left hemisphere only, as in the
previous ROI figures; an 'netw' unit averages all selected left electrodes (like lang_LH_netw).

  unit (subject, ROI)  : each electrode's lag response is z-scored across words, averaged over the subject's electrodes
                         in the ROI, and z-scored again
  unit (GROUP, ROI)    : mean of the per-subject ROI responses over subjects (>= --min-subjects subjects)
  response RDM         : |z(y_i) - z(y_j)| for every lag separately
  items / pairs        : all 5134 words, 100k pairs sampled once (same for all units and models)
  confounds            : |d word length|, |d word frequency| (LME nuisances)
  model RDM            : surprisal -> |d word surprisal| (last layer); embedding -> 1 - Pearson of last-layer states,
                         each feature z-scored across the words
  regression           : plain OLS (base y ~ 1 + confounds, full + model RDM), item-fold CV, delta LL per held-out pair.

The per-ROI LME ΔLL (outputs_jeanzay/podcast/surprisal_roi_lastlayer/*/podcast_lme_by_roi.csv) can be compared ROI by ROI.
Parallelism over lag blocks (--lag-chunks, --shard I/N). Output: <out-root>/<track>/units_roi_c<block>.csv (resumable per
model). Never writes under outputs/.
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

MAP = ROOT / "outputs/podcast/electrode_roi_map_language_responsive.csv"
MAX_PAIRS = int(os.environ.get("RDM_MAX_PAIRS", 100_000))  # env override only for the pair-count sensitivity runs
N_LAGS = 32


def zcol(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0)
    sd[sd == 0] = np.nan
    return (x - x.mean(axis=0)) / sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--min-subjects", type=int, default=3)
    ap.add_argument("--lag-chunks", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", metavar="I/N")
    ap.add_argument("--group-only", action="store_true", help="score only the subject-averaged GROUP units (pair-count sensitivity runs)")
    ap.add_argument("--models", type=Path, help="text file with one model_id per line: score only these models")
    args = ap.parse_args()
    if ROOT / "outputs" in args.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    chunks = list(range(args.lag_chunks))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        chunks = chunks[i::n]
    block = N_LAGS // args.lag_chunks

    sur_tasks = S._tasks("podcast", ("",))
    if args.models:
        keep = {l.strip() for l in open(args.models) if l.strip()}
        sur_tasks = [t for t in sur_tasks if t[1] in keep or t[1].split("/")[-1] in keep]
    fam, mid, rev = sur_tasks[0]
    npy, tpath = S._neural_cache_paths("podcast", fam, mid, rev)
    table = pd.read_parquet(tpath, columns=["subject", "electrode", "chunk", "word_idx", "wlen", "freq"])
    y_all = np.load(npy, mmap_mode="r")
    sel = pd.read_csv(MAP)
    sel = sel[sel.selected & sel.assigned & (sel.hemisphere == "L")][["subject", "channel", "region"]]
    words = np.sort(table.word_idx.unique()).astype(np.int64)
    n = len(words)
    wpos = {w: k for k, w in enumerate(words)}
    cov = table.drop_duplicates("word_idx").set_index("word_idx").loc[words, ["wlen", "freq"]]

    # per-subject ROI responses: (n_words, 32) per (subject, roi)
    resp: dict[tuple[str, str], np.ndarray] = {}
    for subject, part in table.groupby("subject"):
        rows_by_el = part.groupby("electrode").indices
        per_roi: dict[str, list] = {}
        for _, r in sel[sel.subject == subject].iterrows():
            if r.channel not in rows_by_el:
                continue
            idx = part.index.to_numpy()[rows_by_el[r.channel]]
            e = np.full((n, N_LAGS), np.nan, np.float32)
            e[part.word_idx.map(wpos).to_numpy()[rows_by_el[r.channel]]] = np.asarray(y_all[idx], dtype=np.float32)
            e = zcol(e)
            per_roi.setdefault(r.region, []).append(e)
            per_roi.setdefault("netw", []).append(e)
        for roi, lst in per_roi.items():
            resp[(subject, roi)] = zcol(np.nanmean(np.stack(lst), axis=0))
    group = {}
    for roi in sorted({k[1] for k in resp}):
        subs = [v for (s, r), v in resp.items() if r == roi]
        if len(subs) >= args.min_subjects:
            group[("GROUP", roi)] = zcol(np.mean(np.stack(subs), axis=0))
    units = group if args.group_only else {**resp, **group}
    print("units:", {r: sum(1 for (s, rr) in resp if rr == r) for r in sorted({k[1] for k in resp})}, "| GROUP ROIs:",
          sorted(k[1] for k in group), flush=True)

    S.MAX_PAIRS = MAX_PAIRS
    _, _, pairs = S._pair_distances(np.arange(n, dtype=float), np.arange(n, dtype=float), "podcast", "roi")
    conf = np.column_stack([np.abs(cov[c].to_numpy(float)[pairs[:, 0]] - cov[c].to_numpy(float)[pairs[:, 1]]) for c in ("wlen", "freq")])
    seed = int.from_bytes(hashlib.sha256(b"podcast:roi").digest()[:4], "little")
    emb_tasks = []
    for family, model_id, revision in sur_tasks:
        cache = PE._cache_for(model_id, revision)
        if cache is not None:
            emb_tasks.append((family, model_id, revision, cache))

    for chunk in chunks:
        lags = range(chunk * block, (chunk + 1) * block)
        plans = []
        for (subject, roi), y in units.items():
            for lag in lags:
                col = y[:, lag]
                if not np.isfinite(col).all() or col.std() == 0:
                    continue
                plans.append((subject, roi, lag, RdmCvPlan(np.abs(col[pairs[:, 0]] - col[pairs[:, 1]]), n, pairs, conf, args.n_repeats, seed=seed)))
        print(f"lag block {chunk}: {len(plans)} units", flush=True)
        for track in (["embedding", "surprisal"] if args.track == "both" else [args.track]):
            out = args.out_root / track / f"units_roi_c{chunk}.csv"
            done = pd.read_csv(out, dtype={"revision": str, "subject": str}) if out.is_file() else pd.DataFrame()
            seen = set(zip(done.model_id, done.revision)) if len(done) else set()
            tasks = emb_tasks if track == "embedding" else [(f, m, r, None) for f, m, r in sur_tasks]
            if args.limit:
                tasks = tasks[:args.limit]
            rows, count = done.to_dict("records") if len(done) else [], 0
            for family, model_id, revision, cache in tasks:
                if (model_id, revision) in seen:
                    continue
                try:
                    if track == "surprisal":
                        v = S._load_neural_surprisal("podcast", family, model_id, revision)
                        if v is None:
                            continue
                        v = v.assign(w=v.stimulus_id.str.rsplit("::", n=1).str[1].astype(np.int64))
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
                    for subject, roi, lag, plan in plans:
                        r = plan.delta_ll(rdm)
                        rows.append(dict(track=track, family=family, model_id=model_id, revision=revision, subject=subject,
                                         roi=roi, lag=lag, lag_time_s=-0.2 + lag / 32, n_items=n, n_pairs=plan.n_pairs,
                                         delta_ll_per_obs=r.delta_ll_per_obs, delta_ll_per_obs_sd=r.delta_ll_per_obs_sd,
                                         rsa_pearson=plan.pearson(rdm), frac_pairs_tested=r.frac_pairs_tested))
                    count += 1
                except Exception:
                    print(f"ERROR {model_id}@{revision}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                if count and count % 10 == 0:
                    out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False)
            if rows:
                out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False)
            print(f"block {chunk} {track}: {count} new models", flush=True)


if __name__ == "__main__":
    main()
