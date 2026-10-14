#!/usr/bin/env python3
"""Split-half noise ceilings for the per-unit RDM regression (docs/rdm_regression_neural_units.md).

For every response unit that is shared across subjects (the GROUP units of the headline), 15 random splits of the
subjects into halves A/B are drawn. The half-mean response of each half gives a response RDM |m_i - m_j|, and

  dll_ceiling : delta LL of predicting half B's RDM from half A's RDM beyond the confound RDMs, both directions summed
                (Timkey et al. 2026 recipe in RDM space, neuroai/analysis/rdm_ceiling.py; same item folds, same
                confounds, same Gaussian pair likelihood as the models)
  r_half      : Pearson correlation between the two half RDMs (split-half reliability of the RDM)
  rsa_ceiling : sqrt(2 r_half / (1 + r_half)), i.e. the Spearman-Brown corrected reliability of the full-sample RDM,
                square-rooted: the highest RSA score any model could reach with the full-sample target

The dll_ceiling is conservative (a half-sample predictor); to compare models with it like-for-like they must be scored
on the same half targets (not done here, see docs).

Datasets: pereira (ROI x passage set), tuckute (ROI), broderick / megmasc (channels with >= 10 subjects),
podcast (ROI x lag), nieuwland / f2007 / h2019 (one item-mean unit). Fedorenko has no shared units: see rdm_ceiling_fedorenko.py (item-mean predictor).

    .venv/bin/python scripts/analysis/rdm_ceiling_units.py <dataset> --out-root $SCRATCH/rdm_ceiling_units
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_ceiling import RdmTargetReplicates, combine_halves, split_half_targets
from neuroai.analysis.rdm_ceiling_dense import DenseCeilingDesign, dense_unit_ceiling
from neuroai.analysis.rdm_regression import all_pairs
from scripts.analysis import surprisal_rsa as S

SCR = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
N_SPLITS = 15
ENGINE = "sparse"  # --engine dense: all pairs with matrix products (neuroai/analysis/rdm_ceiling_dense.py), podcast and the LME regions only
REGIONS = False  # --regions: the LME regions of megmasc / broderick instead of the channels
SHARD = None  # (i, n): process every n-th unit starting at i (set from --shard; array jobs of the 1M-pair run)


def pair_abs_diff(v: np.ndarray) -> np.ndarray:
    return np.concatenate([np.abs(v[i] - v[i + 1:]) for i in range(len(v) - 1)])


def ceiling_fn(ds: str, n: int, cov, cols=("wlen", "freq")):
    """unit ceiling function of (values, subject idx, item idx, seed) for the active engine; sparse = the production pair sample."""
    if ENGINE == "dense":
        design = DenseCeilingDesign(n, [pair_abs_diff(cov[c].to_numpy(float)) for c in cols])
        return lambda v, s_, i_, seed: dense_unit_ceiling(v, s_, i_, design, N_SPLITS, seed)
    if os.environ.get("RDM_MAX_PAIRS"):
        S.MAX_PAIRS = int(os.environ["RDM_MAX_PAIRS"])
    lab = ("podcast", "roi") if ds == "podcast" else (ds, "GROUP")
    _, _, pairs = S._pair_distances(np.arange(n, dtype=float), np.arange(n, dtype=float), *lab)
    conf = np.column_stack([np.abs(cov[c].to_numpy(float)[pairs[:, 0]] - cov[c].to_numpy(float)[pairs[:, 1]]) for c in cols])
    return lambda v, s_, i_, seed: unit_ceiling(v, s_, i_, n, pairs, conf, seed)


def in_shard(k: int) -> bool:
    return SHARD is None or k % SHARD[1] == SHARD[0]


def unit_ceiling(values, subj, item, n_items, pairs, conf, seed: int) -> dict:
    """Ceilings of one unit from its (subject, item, value) observations."""
    means, valid, half_pairs = split_half_targets(values, subj, item, n_items, N_SPLITS, seed)
    i, j = pairs[:, 0], pairs[:, 1]
    resp = np.abs(means[:, i] - means[:, j])
    wvalid = valid[:, i] * valid[:, j] > 0
    rs = []
    for ta, tb in half_pairs:
        m = wvalid[ta]
        if m.sum() > 10 and resp[ta][m].std() > 0 and resp[tb][m].std() > 0:
            rs.append(np.corrcoef(resp[ta][m], resp[tb][m])[0, 1])
    r_half = float(np.mean(rs)) if rs else np.nan
    rt = RdmTargetReplicates(resp, n_items, pairs, conf, item_valid=valid, n_partitions=5, seed=seed)
    partner = np.arange(len(resp)) ^ 1
    delta, n_test = rt.score_partners(partner)
    comb = combine_halves(delta, n_test, half_pairs)            # (1, variants, splits, partitions)
    v = rt.variants.index("len_freq")
    per_split = np.nanmean(comb[0, v], axis=-1)
    r_sb = 2 * r_half / (1 + r_half) if np.isfinite(r_half) and r_half > -1 else np.nan
    return dict(r_half=r_half, rsa_ceiling=float(np.sqrt(r_sb)) if np.isfinite(r_sb) and r_sb > 0 else np.nan,
                dll_ceiling=float(np.mean(per_split)), dll_ceiling_sd=float(np.std(per_split, ddof=1)),
                n_subjects=int(np.max(subj)) + 1, n_items=n_items, n_pairs=len(pairs))


def obs_from_matrix(Y: np.ndarray, zscore: bool = True):
    """(subjects x items) with NaN -> flat (values, subject index, item index); z-score per subject."""
    S_, N = Y.shape
    if zscore:
        mu, sd = np.nanmean(Y, axis=1, keepdims=True), np.nanstd(Y, axis=1, keepdims=True)
        sd[sd == 0] = np.nan
        Y = (Y - mu) / sd
    sub, itm = np.nonzero(np.isfinite(Y))
    return Y[sub, itm], sub, itm


# ----------------------------------------------------------------------------- loaders
def seed_of(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little")


def run_pereira(out):
    from neuroai.datasets.pereira import PereiraDataset
    ds = PereiraDataset(ROOT / "data/pereira_fmri/Pereira2018_merged.nc")
    fam, mid, rev = S._tasks("pereira", ("",))[0]
    sent = (pd.read_parquet(S._neural_cache_paths("pereira", fam, mid, rev)[1], columns=["sentence_id", "sentlen", "mean_wlen", "mean_freq"])
            .drop_duplicates("sentence_id").set_index("sentence_id"))
    _, all_ids = ds.get_sentences()
    all_ids = np.asarray(all_ids).astype(str)
    acc: dict = {}
    for atlas in ("language", "MD", "DMN", "visual", "auditory"):
        per = ds.get_condition_per_subject(atlas)
        rois = ds.rois(atlas)
        for subject, (bold, ids) in per.items():
            ids = np.asarray(ids).astype(str)
            for pset in ("243sentences", "384sentences"):
                items = np.sort(all_ids[np.char.startswith(all_ids, pset)])
                pos = {s: k for k, s in enumerate(items)}
                sel = np.flatnonzero(np.char.startswith(ids, pset))
                if len(sel) < 6:
                    continue
                for j, roi in enumerate(rois):
                    y = bold[sel, j]
                    if not np.isfinite(y).all() or y.std() == 0:
                        continue
                    acc.setdefault((pset, atlas, roi), {})[subject] = (np.array([pos[s] for s in ids[sel]]), y)
    rows = []
    for (pset, atlas, roi), bysub in sorted(acc.items()):
        if len(bysub) < 4:
            continue
        items = np.sort(all_ids[np.char.startswith(all_ids, pset)])
        n = len(items)
        Y = np.full((len(bysub), n), np.nan)
        for k, (idx, y) in enumerate(bysub.values()):
            Y[k, idx] = y
        pairs = all_pairs(n)
        c = sent.loc[items, ["sentlen", "mean_wlen", "mean_freq"]].to_numpy(float)
        conf = np.column_stack([np.abs(c[pairs[:, 0], q] - c[pairs[:, 1], q]) for q in range(3)])
        v, s_, i_ = obs_from_matrix(Y)
        rows.append(dict(passage_set=pset, atlas=atlas, roi=roi, **unit_ceiling(v, s_, i_, n, pairs, conf, seed_of("pereira", pset))))
        print(rows[-1]["atlas"], rows[-1]["roi"], pset, round(rows[-1]["dll_ceiling"], 6), round(rows[-1]["rsa_ceiling"], 3), flush=True)
    return pd.DataFrame(rows)


def run_tuckute(out):
    from scripts.analysis.rdm_regression_tuckute_units import load_responses
    resp = load_responses()
    fam, mid, rev = S._tasks("tuckute", ("",))[0]
    st = (pd.read_parquet(S._neural_cache_paths("tuckute", fam, mid, rev)[1], columns=["sentence_id", "mean_wlen", "mean_freq"])
          .assign(sentence_id=lambda d: d.sentence_id.astype(int)).drop_duplicates("sentence_id").set_index("sentence_id"))
    common = sorted(set.intersection(*[set(w.dropna(how="all").index) for w in resp.values()]))
    cols = sorted(set.intersection(*[set(w.columns) for w in resp.values()]))
    n = len(common)
    pairs = all_pairs(n)
    c = st.loc[common, ["mean_wlen", "mean_freq"]].to_numpy(float)
    conf = np.column_stack([np.abs(c[pairs[:, 0], q] - c[pairs[:, 1], q]) for q in range(2)])
    rows = []
    for roi in cols:
        Y = np.stack([w.loc[common, roi].to_numpy(float) for w in resp.values()])
        if not np.isfinite(Y).any() or np.nanstd(Y) == 0:
            continue
        v, s_, i_ = obs_from_matrix(Y)
        rows.append(dict(roi=roi, atlas=roi.split("_")[0], **unit_ceiling(v, s_, i_, n, pairs, conf, seed_of("tuckute"))))
        print(roi, round(rows[-1]["dll_ceiling"], 7), round(rows[-1]["rsa_ceiling"], 3), flush=True)
    return pd.DataFrame(rows)


def _epoch_inputs(ds):
    from scripts.analysis import rdm_regression_epoch_units as E
    E.DS = ds
    cfg = E.CONFIG[ds]
    elig = json.load(open(SCR / f"rdm_regression_r30/{ds}/eligible_ids.json"))["ids"]
    files = {Path(f).name.split("_")[0]: f for f in sorted(glob.glob(cfg["epochs"]))}
    words = np.array(sorted(int(k.split("::")[1]) for k in elig[next(iter(elig))]))
    fam, mid, rev = S._tasks(ds, ("",))[0]
    table = pd.read_parquet(S._neural_cache_paths(ds, fam, mid, rev)[1], columns=["word_idx", "wlen", "freq"])
    cov = table.drop_duplicates("word_idx").set_index("word_idx").loc[words, ["wlen", "freq"]]
    return E, elig, files, words, cov


def run_epoch_regions(ds):
    """Ceilings of the LME-region GROUP units (L/R x A/C/P for MEG-MASC, the 4 regions of Broderick): per-subject region responses from the
    LME table, split over subjects exactly as for the channels (same words, pairs, covariates and seed)."""
    E, elig, files, words, cov = _epoch_inputs(ds)
    N = len(words)
    fam, mid, rev = S._tasks(ds, ("",))[0]
    val = E.CONFIG[ds]["value"]
    table = pd.read_parquet(S._neural_cache_paths(ds, fam, mid, rev)[1], columns=["subject", "region", "word_idx", val])
    subs = list(files)
    regs: dict[str, np.ndarray] = {}
    for k, s_ in enumerate(subs):
        p = table[table.subject == s_].pivot_table(index="word_idx", columns="region", values=val, aggfunc="mean").reindex(words)
        for r in p.columns:
            regs.setdefault(r, np.full((len(subs), N), np.nan))[k] = p[r].to_numpy(float)
    ceil = ceiling_fn(ds, N, cov)
    rows = []
    for name, Y in sorted(regs.items()):
        if np.isfinite(Y).any(axis=1).sum() < E.MIN_GROUP_SUBJECTS:
            continue
        v, s2, i_ = obs_from_matrix(Y)
        rows.append(dict(channel=name, **ceil(v, s2, i_, seed_of(ds))))
        print(name, round(rows[-1]["dll_ceiling"], 8), round(rows[-1]["rsa_ceiling"], 3), flush=True)
    return pd.DataFrame(rows)


def run_epoch(ds, out):
    if REGIONS:
        return run_epoch_regions(ds)
    E, elig, files, words, cov = _epoch_inputs(ds)
    N = len(words)
    pos = {w: i for i, w in enumerate(words)}
    chans: dict[str, np.ndarray] = {}
    for k, (sub, f) in enumerate(files.items()):
        ids, mean, names = E.load_subject(f)
        ok = np.array([w in pos for w in ids])
        idx = np.array([pos[w] for w in ids[ok]])
        for c, name in enumerate(names):
            col = mean[ok, c]
            if np.isfinite(col).sum() < 100 or np.nanstd(col) == 0:
                continue
            arr = chans.setdefault(name, np.full((len(files), N), np.nan))
            arr[k, idx] = col
        print("loaded", sub, flush=True)
    if os.environ.get("RDM_MAX_PAIRS"):  # same override as the regression runs (pair-count runs); default 250,000
        S.MAX_PAIRS = int(os.environ["RDM_MAX_PAIRS"])
    _, _, pairs = S._pair_distances(np.arange(N, dtype=float), np.arange(N, dtype=float), ds, "GROUP")
    conf = np.column_stack([np.abs(cov[c].to_numpy(float)[pairs[:, 0]] - cov[c].to_numpy(float)[pairs[:, 1]]) for c in ("wlen", "freq")])
    rows = []
    k_unit = -1
    for name, Y in sorted(chans.items()):
        if np.isfinite(Y).any(axis=1).sum() < E.MIN_GROUP_SUBJECTS:
            continue
        k_unit += 1
        if not in_shard(k_unit):
            continue
        v, s_, i_ = obs_from_matrix(Y)
        rows.append(dict(channel=name, **unit_ceiling(v, s_, i_, N, pairs, conf, seed_of(ds))))
        print(name, round(rows[-1]["dll_ceiling"], 8), round(rows[-1]["rsa_ceiling"], 3), flush=True)
    return pd.DataFrame(rows)


def run_podcast(out):
    from scripts.analysis import rdm_regression_podcast_roi_units as P
    fam, mid, rev = S._tasks("podcast", ("",))[0]
    npy, tpath = S._neural_cache_paths("podcast", fam, mid, rev)
    table = pd.read_parquet(tpath, columns=["subject", "electrode", "chunk", "word_idx", "wlen", "freq"])
    y_all = np.load(npy, mmap_mode="r")
    sel = pd.read_csv(P.MAP)
    sel = sel[sel.selected & sel.assigned & (sel.hemisphere == "L")][["subject", "channel", "region"]]
    words = np.sort(table.word_idx.unique()).astype(np.int64)
    n = len(words)
    wpos = {w: k for k, w in enumerate(words)}
    cov = table.drop_duplicates("word_idx").set_index("word_idx").loc[words, ["wlen", "freq"]]
    resp: dict = {}
    for subject, part in table.groupby("subject"):
        by_el = part.groupby("electrode").indices
        per_roi: dict = {}
        for _, r in sel[sel.subject == subject].iterrows():
            if r.channel not in by_el:
                continue
            idx = part.index.to_numpy()[by_el[r.channel]]
            e = np.full((n, P.N_LAGS), np.nan, np.float32)
            e[part.word_idx.map(wpos).to_numpy()[by_el[r.channel]]] = np.asarray(y_all[idx], dtype=np.float32)
            e = P.zcol(e)
            per_roi.setdefault(r.region, []).append(e)
            per_roi.setdefault("netw", []).append(e)
        for roi, lst in per_roi.items():
            resp[(subject, roi)] = P.zcol(np.nanmean(np.stack(lst), axis=0))
    if ENGINE != "dense":
        S.MAX_PAIRS = P.MAX_PAIRS
    ceil = ceiling_fn("podcast", n, cov)
    rows = []
    k_unit = -1
    for roi in sorted({k[1] for k in resp}):
        subs = [v for (s, r), v in resp.items() if r == roi]
        if len(subs) < 3:
            continue
        for lag in range(P.N_LAGS):
            Y = np.stack([s[:, lag] for s in subs])
            if not np.isfinite(Y).any():
                continue
            k_unit += 1
            if not in_shard(k_unit):
                continue
            v, s_, i_ = obs_from_matrix(Y, zscore=False)       # already z-scored per subject-ROI
            rows.append(dict(roi=roi, lag=lag, lag_time_s=-0.2 + lag / 32,
                             **ceil(v, s_, i_, seed_of("podcast", roi, lag))))
        print(roi, flush=True)
    return pd.DataFrame(rows)


def n400_inputs(study):
    """Trial-level N400 data of one study: (values, subject idx, item idx, n_items, pairs, confounds, seed)."""
    from neuroai.features.word_frequency import compute_wlen_wordfreq_covariates
    if study == "nieuwland":
        from neuroai.datasets.nieuwland import load_all_items
        from scripts.run.embedding.neural.nieuwland_paper_item_mean import PAPER_TSV, _key, item_means
        items = item_means(load_all_items(ROOT / "data/nieuwland", included_in_paper_only=True, exclude_labs=["glas"]))
        paper = pd.read_csv(PAPER_TSV, sep="\t").drop_duplicates().dropna(subset=["n400", "FullText", "TargetWords"])
        paper["k"] = paper["FullText"].map(_key).str[:20] + "|" + paper["TargetWords"].map(_key)
        kpos = {k: i for i, k in enumerate(items["k"])}
        item_idx = paper["k"].map(kpos).to_numpy()
        subj = pd.factorize(paper["subject"])[0]
        val = paper["n400"].to_numpy(float)
        words = items.noun.astype(str).tolist()
        n = len(items)
    else:
        from neuroai.datasets.szewczyk_paper import STUDIES, TSV, load_study, study_items
        df = load_study(ROOT / TSV, study)
        it = study_items(df)
        kpos = {k: i for i, k in enumerate(it["item"])}
        item_idx = df["item"].map(kpos).to_numpy()
        subj = pd.factorize(df["Subject"])[0]
        val = df["n400_resid"].to_numpy(float)
        words = it.critical_word.astype(str).tolist()
        n = len(it)
        # same extra nuisance covariates as rdm_regression_n400_units.py (paper formula, calculate_aics_szewczyk.R)
        extra = df.groupby("item")[["pos_start", "old20", "concr"]].first().loc[it["item"]].to_numpy(np.float64)
    ok = np.isfinite(val) & np.isfinite(item_idx.astype(float))
    lex = compute_wlen_wordfreq_covariates(words).astype(np.float64)
    if study != "nieuwland":
        lex = np.column_stack([lex, extra])
    pairs = all_pairs(n)
    conf = np.column_stack([np.abs(lex[pairs[:, 0], q] - lex[pairs[:, 1], q]) for q in range(lex.shape[1])])
    return val[ok], subj[ok], item_idx[ok].astype(int), n, pairs, conf, seed_of("n400", study)


def run_n400(study, out):
    val, subj, item, n, pairs, conf, seed = n400_inputs(study)
    r = unit_ceiling(val, subj, item, n, pairs, conf, seed)
    return pd.DataFrame([dict(unit="item_mean", **r)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["pereira", "tuckute", "broderick", "megmasc", "podcast", "nieuwland", "f2007", "h2019"])
    ap.add_argument("--out-root", type=Path, default=SCR / "rdm_ceiling_units")
    ap.add_argument("--engine", choices=["sparse", "dense"], default="sparse", help="dense: all pairs by matrix products (podcast and --regions)")
    ap.add_argument("--regions", action="store_true", help="megmasc / broderick: ceilings of the LME-region GROUP units (writes ceilings_<ds>_regions.csv)")
    ap.add_argument("--shard", metavar="I/N", help="megmasc / broderick / podcast: process every N-th unit starting at I; writes ceilings_<ds>_s<I>of<N>.csv")
    a = ap.parse_args()
    if ROOT / "outputs" in a.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    a.out_root.mkdir(parents=True, exist_ok=True)
    d = a.dataset
    global SHARD, REGIONS, ENGINE
    REGIONS = a.regions
    ENGINE = a.engine
    if a.shard:
        SHARD = tuple(int(x) for x in a.shard.split("/"))
    df = (run_pereira(a.out_root) if d == "pereira" else run_tuckute(a.out_root) if d == "tuckute" else
          run_epoch(d, a.out_root) if d in ("broderick", "megmasc") else run_podcast(a.out_root) if d == "podcast" else
          run_n400(d, a.out_root))
    df.insert(0, "dataset", d)
    df.to_csv(a.out_root / (f"ceilings_{d}_regions.csv" if REGIONS else f"ceilings_{d}_s{SHARD[0]}of{SHARD[1]}.csv" if SHARD else f"ceilings_{d}.csv"), index=False)
    print(f"{d}: {len(df)} units | mean dll_ceiling {df.dll_ceiling.mean():.3e} | mean rsa_ceiling {df.rsa_ceiling.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
