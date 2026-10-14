#!/usr/bin/env python3
"""Production stimulus bootstrap of the last-layer RDM delta LL (neuroai/analysis/rdm_bootstrap.py).

For every model (embedding: last layer; surprisal) and every response unit (behavioral measure, or
neural subject / grand-average N400 RDM) this stores the delta LL of each bootstrap resample and fold
partition, plus point estimate, bootstrap SE and 95% percentile interval per confound variant.

Reuses the scoring pipelines of rdm_regression_behavioral.py / _other.py / _neural.py unchanged
(so model RDMs are exactly the validated ones) by replacing their scoring step; RDM_LAST_LAYER_ONLY=1
makes them skip the other layers.

Resamples are drawn ONCE per dataset over the union of its stimuli and every unit uses the counts of
its own stimuli: subjects/measures that share stimuli share the resample, so averaging subjects does
not shrink the stimulus-sampling variance. The fold partitions are fixed per unit, so all models
(both tracks) are paired on identical resamples and partitions. Output: <out-root>/<dataset>/<track>/
<family>/<model>/boot_<rev>.npz (array per unit: variants x (1+B) x partitions, replicate 0 = original
data) plus a per-model CSV; `--summarize-only` writes <out-root>/<dataset>/bootstrap_summary.csv.
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

from neuroai.analysis.rdm_bootstrap import BootResult, RdmBootstrap
from neuroai.sweep import safe_model_dir
from scripts.analysis import rdm_regression_behavioral as base
from scripts.analysis import rdm_regression_neural as neural

BEHAVIORAL = tuple(base.DATASET_MEASURES)
NEURAL = ("pereira", "tuckute", "fedorenko2016", "broderick", "megmasc", "podcast", "nieuwland_avg", "szewczyk_avg")
PER_SUBJECT_NEURAL = ("pereira", "tuckute", "fedorenko2016", "broderick", "megmasc", "podcast")


class PlanStub:
    """Stands in for RdmCvPlan while the existing loaders build their units: keeps the raw arrays."""

    def __init__(self, response, n_items, pairs=None, confounds=None, n_repeats=30, n_folds=None, seed=0):
        self.response, self.n_items, self.pairs, self.confounds = response, n_items, pairs, confounds
        self.boot = None
        self.lock = threading.Lock()


base.RdmCvPlan = PlanStub
neural.RdmCvPlan = PlanStub

CFG: dict = {}          # dataset, n_boot, n_partitions, seed, counts (B, U), index {stimulus key -> column}
STORE: dict = {}        # (track, family, model, revision) -> {unit name: array (V, 1+B, P)}
STORE_LOCK = threading.Lock()


def init_resamples(dataset: str, unit_keys: list[list[str]], n_boot: int, n_partitions: int, seed: int):
    universe = sorted({k for keys in unit_keys for k in keys})
    rng = np.random.default_rng([seed, zlib.crc32(dataset.encode()), 7])
    u = len(universe)
    CFG.update(dataset=dataset, n_boot=n_boot, n_partitions=n_partitions, seed=seed,
               index={k: i for i, k in enumerate(universe)},
               counts=rng.multinomial(u, np.full(u, 1.0 / u), size=n_boot))
    print(f"{dataset}: {n_boot} resamples over {u} distinct stimuli, {n_partitions} partitions", flush=True)


def _boot_for(stub: PlanStub, unit: str, stimulus_keys: list[str]) -> RdmBootstrap:
    with stub.lock:
        if stub.boot is None:
            cols = np.fromiter((CFG["index"][k] for k in stimulus_keys), dtype=np.int64, count=len(stimulus_keys))
            seed = (CFG["seed"] * 1_000_003 + zlib.crc32(f"{CFG['dataset']}|{unit}".encode())) % (2**31)
            stub.boot = RdmBootstrap(stub.response, stub.n_items, stub.pairs, stub.confounds,
                                     n_partitions=CFG["n_partitions"], seed=seed,
                                     counts=CFG["counts"][:, cols], include_original=True)
        return stub.boot


def _score(stub, unit, stimulus_keys, dataset, track, task, layer, rdm, extra):
    boot = _boot_for(stub, unit, stimulus_keys)
    res = boot.delta_ll_batch(np.asarray(rdm, dtype=np.float64)[None])
    with STORE_LOCK:
        STORE.setdefault((track, task[0], task[1], task[2]), {})[unit] = res.delta[0].astype(np.float32)
    point, se, se_naive, ci = res.point[0], res.se()[0], res.se_naive()[0], res.ci()[:, 0, :]
    return [{"dataset": dataset, "track": track, "family": task[0], "model_id": task[1], "revision": task[2],
             "unit": unit, "layer": layer, "confounds": v, "n_items": boot.n_items, "n_pairs": boot.n_pairs,
             "point_delta_ll_per_obs": point[i], "boot_se": se[i], "boot_sd_uncorrected": se_naive[i], "ci_low": ci[0, i], "ci_high": ci[1, i],
             "n_boot": CFG["n_boot"], "n_partitions": boot.P, **extra} for i, v in enumerate(res.variants)]


def boot_rows_behavioral(meas, measure, dataset, track, task, layer, rdm):
    keys = [f"{s}_{z}" for s, z in meas["ids"].tolist()]
    return _score(meas["plans"]["len_freq"], measure, keys, dataset, track, task, layer, rdm, {"measure": measure})


def boot_unit_neural(unit, dataset, track, task, layer, rdm):
    return _score(unit["plans"]["len_freq"], unit["subject"], unit["ids"].tolist(), dataset, track, task, layer,
                  rdm, {"subject": unit["subject"]})


base._rows = boot_rows_behavioral
neural._score_unit = boot_unit_neural


def _npz_path(root: Path, dataset: str, track: str, task: tuple) -> Path:
    return (root / dataset / track / task[0] / safe_model_dir(task[1]) /
            f"boot_{task[2].replace('/', '_')}.npz")


def _csv_paths(root, dataset, track, task, measures):
    if dataset in NEURAL:
        return [neural._out_path(root, dataset, track, task[:3])]
    return [base._out_path(root, dataset, track, m, task[:3]) for m in measures]


def run_task(fn, args_tuple, task, dataset, track, root, measures):
    """Run one model; it is 'done' only when its npz exists (partial CSVs from a crash are discarded)."""
    npz = _npz_path(root, dataset, track, task)
    if npz.is_file():
        return f"existing {task[1]}@{task[2]}"
    for p in _csv_paths(root, dataset, track, task, measures):
        p.unlink(missing_ok=True)
    message = fn(task, dataset, *args_tuple)
    arrays = STORE.pop((track, task[0], task[1], task[2]), None)
    if arrays:
        arrays["__meta__"] = np.array([task[0], task[1], task[2]])
        npz.parent.mkdir(parents=True, exist_ok=True)
        tmp = npz.with_name(npz.name + f".{os.getpid()}.{threading.get_ident()}.tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, npz)
    return message


def summarize(dataset: str, root: Path):
    rows = []
    for track in ("embedding", "surprisal"):
        for path in (root / dataset / track).glob("*/*/boot_*.npz"):
            with np.load(path) as z:
                units = {k: z[k] for k in z.files if not k.startswith("__")}
                family, model_id, revision = (str(x) for x in z["__meta__"])
            variants = ("none", "len_freq")
            groups = ({u: [a] for u, a in units.items()} if dataset in BEHAVIORAL
                      else {"mean_over_units": list(units.values())})
            for name, arrs in groups.items():
                delta = np.mean(np.stack(arrs), axis=0)                    # (V, 1+B, P)
                r = BootResult(delta[None].astype(np.float64), variants, include_original=True)
                se, se_naive, ci, point = r.se()[0], r.se_naive()[0], r.ci()[:, 0, :], r.point[0]
                for i, v in enumerate(variants):
                    rows.append({"dataset": dataset, "track": track, "family": family, "model_id": model_id,
                                 "revision": revision, "unit": name, "n_units": len(arrs), "confounds": v,
                                 "point_delta_ll_per_obs": point[i], "boot_se": se[i],
                                 "boot_sd_uncorrected": se_naive[i], "ci_low": ci[0, i], "ci_high": ci[1, i]})
    if not rows:
        print(f"{dataset}: nothing to summarize", flush=True)
        return
    out = pd.DataFrame(rows)
    path = root / dataset / "bootstrap_summary.csv"
    out.sort_values(["track", "unit", "confounds", "family", "model_id", "revision"]).to_csv(path, index=False)
    print(f"{dataset}: {len(out)} rows -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=BEHAVIORAL + NEURAL)
    ap.add_argument("--track", choices=["embedding", "surprisal", "both"], default="both")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--elig-root", type=Path, default=None,
                    help="where rdm_regression_neural.py wrote <dataset>/eligible_ids.json "
                         "(default $SCRATCH/rdm_regression_r30)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", metavar="I/N")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-partitions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    out_root = args.out_root or (scratch / "rdm_bootstrap_r200" if scratch else None)
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
    tracks = ["embedding", "surprisal"] if args.track == "both" else [args.track]
    dataset = args.dataset
    if dataset in NEURAL:
        prepared = neural.prepare(dataset, 1, 0, elig_root)
        measures = ()
        init_resamples(dataset, [u["ids"].tolist() for u in prepared["units"]], args.n_boot, args.n_partitions, args.seed)
        task_fns = {"embedding": (neural.embedding_tasks, neural.score_embedding),
                    "surprisal": (neural.surprisal_tasks, neural.score_surprisal)}
        call_args = (prepared, out_root)
        list_tasks = lambda track: task_fns[track][0](dataset)  # noqa: E731
    else:
        measures = base.DATASET_MEASURES[dataset]
        prepared = {m: base.load_measure(dataset, m, 1, 0) for m in measures}
        init_resamples(dataset, [[f"{s}_{z}" for s, z in prepared[m]["ids"].tolist()] for m in measures],
                       args.n_boot, args.n_partitions, args.seed)
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
