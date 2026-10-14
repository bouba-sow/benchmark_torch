#!/usr/bin/env python3
"""Split-half RDM noise ceiling + like-for-like half-target model scores (last layer).

Behavioral datasets (per reading measure) and the grand-average N400 datasets (nieuwland_avg,
szewczyk_avg). Timkey et al. (2026) recipe in RDM space, see neuroai/analysis/rdm_ceiling.py:
15 random participant splits into halves A/B; the item-mean RDM of one half is the predictor of the
other half's item-mean RDM (both directions summed); every model's last-layer RDM is scored on the
SAME half targets with the same item folds, so model/ceiling is like-for-like.

Reuses the validated scoring pipelines of rdm_regression_behavioral.py / _other.py / _neural.py by
replacing their scoring step (same approach as rdm_bootstrap_run.py). Output:
<out-root>/<dataset>/ceiling_<unit>.npz (ceiling: variants x splits x partitions),
<out-root>/<dataset>/<track>/<family>/<model>/ceil_<rev>.npz (half-target model scores, same shape) and
`--summarize-only` -> <out-root>/<dataset>/ceiling_summary.csv (model, ceiling, ratio).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["RDM_LAST_LAYER_ONLY"] = "1"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_ceiling import RdmTargetReplicates, combine_halves, split_half_targets
from neuroai.analysis.rdm_regression import all_pairs
from neuroai.sweep import safe_model_dir
from scripts.analysis import rdm_regression_behavioral as base
from scripts.analysis import rdm_regression_neural as neural

BEHAVIORAL = tuple(base.DATASET_MEASURES)
AVG = tuple(neural.AVG)
N_SPLITS, N_PARTITIONS = 15, 5
SUBJECT_COLUMNS = ("participant_id", "subject", "subject_id", "participant")


class PlanStub:
    def __init__(self, response, n_items, pairs=None, confounds=None, n_repeats=30, n_folds=None, seed=0):
        self.response, self.n_items, self.pairs, self.confounds = response, n_items, pairs, confounds
        self.state = None
        self.lock = threading.Lock()


base.RdmCvPlan = PlanStub
neural.RdmCvPlan = PlanStub
CFG: dict = {}
STORE: dict = {}
STORE_LOCK = threading.Lock()


# ----------------------------------------------------------------------------- subject-level trials
def behavioral_trials(dataset: str, measure: str) -> pd.DataFrame:
    """Trial rows (participant, story_id, zone, rt) defined exactly like the production item means."""
    if dataset in ("provo", "meco"):
        from neuroai.datasets.meco import MecoDataset
        from neuroai.datasets.provo import ProvoDataset
        from scripts.analysis.behavioral_rsa import _dataset_root
        root = _dataset_root(dataset)
        ds = (ProvoDataset(root, measure, aggregate=False) if dataset == "provo"
              else MecoDataset(root, measure, language="en", aggregate=False))
        trials = ds.load_words().dropna(subset=["rt"])
        trials = trials.loc[~trials.is_edge.astype(bool)]
    else:
        from scripts.analysis.surprisal_rsa import _behavior_cache_paths, _tasks
        family, model_id, revision = _tasks(dataset, base.DATASET_MEASURES[dataset])[0]
        _, word_path = _behavior_cache_paths(dataset, family, model_id, revision, measure)
        trials = pd.read_parquet(word_path.with_name(word_path.name.replace("word_base", "trials_base")))
        trials = trials.loc[np.isfinite(pd.to_numeric(trials.rt, errors="coerce"))].copy()
        if "is_edge" in trials:
            trials = trials.loc[~trials.is_edge.fillna(False).astype(bool)]
    col = next((c for c in SUBJECT_COLUMNS if c in trials.columns), None)
    if col is None:
        raise KeyError(f"No participant column in trials of {dataset}/{measure}: {list(trials.columns)}")
    out = trials[[col, "story_id", "zone", "rt"]].rename(columns={col: "participant"})
    out["rt"] = pd.to_numeric(out.rt)
    return out


def avg_n400_trials(dataset: str) -> pd.DataFrame:
    """(participant, item, n400) rows with the item keys of rdm_regression_neural.load_average_unit."""
    from scripts.analysis.surprisal_rsa import _neural_cache_paths
    base_ds = neural.AVG[dataset]
    family, model_id, revision, _ = next(t for t in neural.surprisal_tasks(base_ds)
                                         if Path(_neural_cache_paths(base_ds, *t[:3])[1]).is_file())
    table = _neural_cache_paths(base_ds, family, model_id, revision)[1]
    chunk_to_item = neural._szewczyk_chunk_to_item() if base_ds == "szewczyk" else None
    raw_col = "chunk" if base_ds == "szewczyk" else "item_id"
    df = pd.read_parquet(table, columns=["subject", raw_col, "n400"])
    df["participant"] = df.subject.astype(str)
    df["item"] = [str(chunk_to_item[(s, str(r))]) if chunk_to_item else neural.canon_ids([r])[0]
                  for s, r in zip(df.participant, df[raw_col])]
    df = df.loc[np.isfinite(df.n400.to_numpy(dtype=float))]
    return df[["participant", "item", "n400"]]


def _arrays_for_unit(dataset, unit, trials, item_keys):
    """Trial arrays (value, subject index, item index) restricted to the unit's items."""
    if dataset in AVG:
        key_col, val_col = "item", "n400"
        keys = pd.Series(np.arange(len(item_keys)), index=pd.Index(item_keys, name="item"))
        item_idx = trials["item"].map(keys)
    else:
        key_col, val_col = None, "rt"
        ids = pd.MultiIndex.from_arrays([item_keys[:, 0], item_keys[:, 1]])
        lookup = pd.Series(np.arange(len(ids)), index=ids)
        idx = pd.MultiIndex.from_arrays([trials.story_id.to_numpy(), trials.zone.to_numpy()])
        item_idx = pd.Series(lookup.reindex(idx).to_numpy(), index=trials.index)
    ok = item_idx.notna()
    subj = pd.factorize(trials.loc[ok, "participant"])[0]
    return trials.loc[ok, val_col].to_numpy(dtype=np.float64), subj, item_idx[ok].to_numpy(dtype=np.int64)


def _state(stub: PlanStub, dataset: str, unit: str, item_keys, trials_fn):
    """Lazily build (targets object, halves, ceiling) once per unit; verified against the shared RDM."""
    with stub.lock:
        if stub.state is not None:
            return stub.state
        n = stub.n_items
        pairs = all_pairs(n) if stub.pairs is None else np.asarray(stub.pairs)
        values, subj, item = _arrays_for_unit(dataset, unit, trials_fn(), item_keys)
        # alignment check: item means over ALL participants must reproduce the shared response RDM
        full = np.bincount(item, weights=values, minlength=n) / np.maximum(np.bincount(item, minlength=n), 1)
        probe = slice(0, min(3000, len(pairs)))
        got = np.abs(full[pairs[probe, 0]] - full[pairs[probe, 1]])
        want = np.asarray(stub.response)[probe]
        err = float(np.abs(got - want).max())
        if err > 1e-3 * max(1.0, float(np.abs(want).max())):
            raise ValueError(f"{dataset}/{unit}: subject-level trials do not reproduce the shared RDM (max err {err:.3g})")
        seed = (zlib.crc32(f"{dataset}|{unit}".encode()) + CFG["seed"]) % (2**31)
        means, valid, halves = split_half_targets(values, subj, item, n, n_splits=N_SPLITS, seed=seed)
        filled = np.nan_to_num(means)
        responses = np.abs(filled[:, pairs[:, 0]] - filled[:, pairs[:, 1]])
        rep = RdmTargetReplicates(responses, n, pairs, stub.confounds, valid, n_partitions=N_PARTITIONS, seed=seed)
        partner = np.empty(2 * N_SPLITS, dtype=int)
        for ta, tb in halves:
            partner[ta], partner[tb] = tb, ta
        d, nt = rep.score_partners(partner)
        ceiling = combine_halves(d, nt, halves)[0]                         # (V, S, P)
        out = CFG["out_root"] / dataset / f"ceiling_{unit}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, ceiling=ceiling, variants=np.array(rep.variants),
                            valid_item_fraction=valid[::2].mean(axis=1), n_test=nt, align_err=err,
                            n_participants=int(subj.max()) + 1)
        os.replace(tmp, out)
        stub.state = (rep, halves, ceiling)
        print(f"ceiling {dataset}/{unit}: variants {rep.variants}, mean {np.nanmean(ceiling, axis=(1, 2)).round(5)}, "
              f"{int(subj.max()) + 1} participants, valid items/split {valid[::2].mean():.2f}", flush=True)
        return stub.state


def _score(stub, unit, item_keys, trials_fn, dataset, track, task, layer, rdm, extra):
    rep, halves, _ = _state(stub, dataset, unit, item_keys, trials_fn)
    d, nt = rep.score_models(np.asarray(rdm, dtype=np.float64)[None])
    scores = combine_halves(d, nt, halves)[0]                              # (V, S, P)
    with STORE_LOCK:
        STORE.setdefault((track, task[0], task[1], task[2]), {})[unit] = scores.astype(np.float32)
    mean = np.nanmean(scores, axis=(1, 2))
    return [{"dataset": dataset, "track": track, "family": task[0], "model_id": task[1], "revision": task[2],
             "unit": unit, "layer": layer, "confounds": v, "half_target_delta_ll_per_obs": mean[i], **extra}
            for i, v in enumerate(rep.variants)]


def rows_behavioral(meas, measure, dataset, track, task, layer, rdm):
    return _score(meas["plans"]["len_freq"], measure, meas["ids"],
                  lambda: behavioral_trials(dataset, measure), dataset, track, task, layer, rdm, {"measure": measure})


def unit_avg(unit, dataset, track, task, layer, rdm):
    return _score(unit["plans"]["len_freq"], unit["subject"], unit["ids"],
                  lambda: avg_n400_trials(dataset), dataset, track, task, layer, rdm, {})


base._rows = rows_behavioral
neural._score_unit = unit_avg


# ----------------------------------------------------------------------------- tasks / summary
def _npz_path(root, dataset, track, task):
    return root / dataset / track / task[0] / safe_model_dir(task[1]) / f"ceil_{task[2].replace('/', '_')}.npz"


def _csv_paths(root, dataset, track, task, measures):
    if dataset in AVG:
        return [neural._out_path(root, dataset, track, task[:3])]
    return [base._out_path(root, dataset, track, m, task[:3]) for m in measures]


def run_task(fn, call_args, task, dataset, track, root, measures):
    npz = _npz_path(root, dataset, track, task)
    if npz.is_file():
        return f"existing {task[1]}@{task[2]}"
    for p in _csv_paths(root, dataset, track, task, measures):
        p.unlink(missing_ok=True)
    message = fn(task, dataset, *call_args)
    arrays = STORE.pop((track, task[0], task[1], task[2]), None)
    if arrays:
        arrays["__meta__"] = np.array([task[0], task[1], task[2]])
        npz.parent.mkdir(parents=True, exist_ok=True)
        tmp = npz.with_name(npz.name + f".{os.getpid()}.{threading.get_ident()}.tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, npz)
    return message


def summarize(dataset: str, root: Path):
    ceilings = {}
    for p in (root / dataset).glob("ceiling_*.npz"):
        with np.load(p) as z:
            ceilings[p.stem[len("ceiling_"):]] = (z["ceiling"], [str(v) for v in z["variants"]])
    rows = []
    for track in ("embedding", "surprisal"):
        for path in (root / dataset / track).glob("*/*/ceil_*.npz"):
            with np.load(path) as z:
                units = {k: z[k] for k in z.files if not k.startswith("__")}
                family, model_id, revision = (str(x) for x in z["__meta__"])
            for unit, arr in units.items():
                if unit not in ceilings:
                    continue
                ceil, variants = ceilings[unit]
                for i, v in enumerate(variants):
                    m, c = float(np.nanmean(arr[i])), float(np.nanmean(ceil[i]))
                    per_split = np.nanmean(arr[i], axis=-1) / np.nanmean(ceil[i], axis=-1)
                    rows.append({"dataset": dataset, "track": track, "family": family, "model_id": model_id,
                                 "revision": revision, "unit": unit, "confounds": v,
                                 "half_target_delta_ll_per_obs": m, "ceiling_delta_ll_per_obs": c,
                                 "ratio_to_ceiling": m / c if c > 0 else np.nan,
                                 "ratio_split_sd": float(np.nanstd(per_split, ddof=1))})
    if not rows:
        print(f"{dataset}: nothing to summarize", flush=True)
        return
    out = pd.DataFrame(rows)
    path = root / dataset / "ceiling_summary.csv"
    out.sort_values(["track", "unit", "confounds", "family", "model_id", "revision"]).to_csv(path, index=False)
    print(f"{dataset}: {len(out)} rows -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=BEHAVIORAL + AVG)
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--elig-root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", metavar="I/N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    out_root = args.out_root or (scratch / "rdm_ceiling" if scratch else None)
    if out_root is None:
        ap.error("--out-root is required when $SCRATCH is unset")
    if ROOT / "outputs" in out_root.resolve().parents or out_root.resolve() == ROOT / "outputs":
        ap.error("refusing to write under outputs/")
    elig_root = args.elig_root or (scratch / "rdm_regression_r30" if scratch else None)
    if args.summarize_only:
        summarize(args.dataset, out_root)
        return
    shard = None
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            ap.error("--shard I/N needs 0 <= I < N")
        shard = (i, n)
    CFG.update(seed=args.seed, out_root=out_root)
    dataset, tracks = args.dataset, (["embedding", "surprisal"] if args.track == "both" else [args.track])
    if dataset in AVG:
        prepared = neural.prepare(dataset, 1, 0, elig_root)
        measures = ()
        task_fns = {"embedding": (neural.embedding_tasks, neural.score_embedding),
                    "surprisal": (neural.surprisal_tasks, neural.score_surprisal)}
        call_args = (prepared, out_root)
        list_tasks = lambda track: task_fns[track][0](dataset)  # noqa: E731
    else:
        measures = base.DATASET_MEASURES[dataset]
        prepared = {m: base.load_measure(dataset, m, 1, 0) for m in measures}
        embed_fn, embed_tasks = base.score_embedding, (lambda: base.embedding_tasks(dataset))
        if dataset not in ("provo", "meco"):
            from scripts.analysis import rdm_regression_behavioral_other as other
            prepared["_ctx"] = other.prepare_context(dataset, measures)
            embed_fn, embed_tasks = other.score_embedding, (lambda: other.embedding_tasks(dataset, measures))
        task_fns = {"embedding": (lambda ds: embed_tasks(), embed_fn),
                    "surprisal": (lambda ds: base.surprisal_tasks(ds, measures), base.score_surprisal)}
        call_args = (measures, prepared, out_root)
        list_tasks = lambda track: task_fns[track][0](dataset)  # noqa: E731
    failures = 0
    for track in tracks:
        tasks = list_tasks(track)
        if args.limit:
            tasks = tasks[:args.limit]
        if shard:
            tasks = tasks[shard[0]::shard[1]]
        fn = task_fns[track][1]
        print(f"{track}: {len(tasks)} models", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_task, fn, call_args, t, dataset, track, out_root, measures): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    print(fut.result(), flush=True)
                except Exception:
                    failures += 1
                    t = futures[fut]
                    print(f"ERROR {t[1]}@{t[2]}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    if not shard:
        summarize(dataset, out_root)
    if failures:
        sys.exit(f"{failures} model(s) failed; see ERROR lines above")


if __name__ == "__main__":
    main()
