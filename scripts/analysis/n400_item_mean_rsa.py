#!/usr/bin/env python3
"""N400 RSA on the subject-averaged item-mean target (same target as the ridge benchmark).

The ridge benchmark (``nieuwland_paper_item_mean.py``, ``szewczyk_item_mean.py``) predicts the
per-item mean N400 over subjects. ``n400_paper_rsa.py`` instead scores each subject's own item
scalars and averages the scores, which is much noisier. Here:

  neural distance : |y_i - y_j| on the subject-averaged item mean ``y`` (identical to the ridge target)
  embedding RDM   : 1 - Pearson between last-layer item vectors (as before)
  surprisal RDM   : |s_i - s_j| on the last-layer surprisal averaged over subjects per item
  score           : Pearson over all item pairs -> one value per model (no subject averaging)

Output goes to ``--out-root`` (default ``$SCRATCH/n400_item_mean_rsa``), never to ``outputs/``.

    .venv/bin/python scripts/analysis/n400_item_mean_rsa.py nieuwland f2007 h2019
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/run/embedding/neural"))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from neuroai.analysis.benchmark_common import lookup_model_meta
from neuroai.benchmarks.eeg_erp import _feature_cache_path, _items_hash
from neuroai.datasets.nieuwland import load_all_items
from neuroai.datasets.szewczyk_paper import item_means as szewczyk_item_means
from neuroai.datasets.szewczyk_paper import load_study, study_items, TSV
from neuroai.sweep import safe_model_dir
from nieuwland_paper_item_mean import item_means as nieuwland_item_means
from scripts.analysis import n400_paper_rsa as P

NIEUWLAND_CACHE = Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/nieuwland_features")


STANDARDIZE = False


def _z(states):
    """Per-feature z-score across items (Timkey & van Schijndel 2021), if enabled."""
    return (states - states.mean(0)) / (states.std(0) + 1e-8) if STANDARDIZE else states


def _score(model_dist, y):
    return float(pearsonr(model_dist, pdist(y[:, None], metric="euclidean")).statistic)


def _pairs_ok(values):
    return np.isfinite(values).all()


class Szewczyk:
    def __init__(self, study):
        self.dataset = f"szewczyk_{study}"
        self.study = study
        df = load_study(ROOT / TSV, study)
        self.items = study_items(df)
        self.y = szewczyk_item_means(df, self.items).y.to_numpy(float)
        self.item_key = (df[["Subject", "chunk", "item"]].drop_duplicates()
                         .set_index(["Subject", "chunk"])["item"].to_dict())
        self.positions = {str(i): k for k, i in enumerate(self.items.item.astype(str))}

    def embedding(self, family, task):
        states = P.feature_array(family, task, self.items, self.study)
        return None if states is None else pdist(_z(states), metric="correlation")

    def surprisal(self, family, task):
        root = (ROOT / "outputs_jeanzay" / self.dataset / "surprisal_sweep" / family
                / safe_model_dir(task.model_id))
        rev = (task.revision or "main").replace("/", "_")
        tp, npy = root / f"szewczyk_lme_pooled_table_{rev}.parquet", root / f"szewczyk_lme_pooled_surprisal_{rev}.npy"
        if not (tp.is_file() and npy.is_file()):
            return None
        table = pd.read_parquet(tp, columns=["subject", "chunk"])
        vals = np.load(npy, mmap_mode="r")
        if vals.shape[0] != len(table):
            raise ValueError(f"row mismatch {npy}")
        table["surprisal"] = np.asarray(vals[:, -1], dtype=float)
        table["item"] = [self.item_key[(str(s), str(c))] for s, c in zip(table.subject, table.chunk)]
        per_item = table.groupby("item").surprisal.mean()
        s = np.array([per_item.get(i, np.nan) for i in self.items.item])
        return None if not _pairs_ok(s) else pdist(s[:, None], metric="euclidean")


class Nieuwland:
    dataset = "nieuwland"

    def __init__(self):
        items = load_all_items(ROOT / "data/nieuwland", included_in_paper_only=True, exclude_labs=["glas"])
        self.items = nieuwland_item_means(items)
        self.y = self.items.y.to_numpy(float)

    def embedding(self, family, task):
        rev = task.revision or "main"
        ih = _items_hash(self.items["sentence_context"].tolist(), self.items["noun"].tolist())
        cache = _feature_cache_path(NIEUWLAND_CACHE, task.load_model_id, None if rev == "main" else rev, "all", ih)
        if not cache.is_file():
            return None
        f = np.load(cache)["features"]
        states = f if f.ndim == 2 else f[:, -1, :]
        if not np.isfinite(states).all():
            raise ValueError(f"non-finite features {cache}")
        return pdist(_z(states), metric="correlation")

    def surprisal(self, family, task):
        root = (ROOT / "outputs_jeanzay" / "nieuwland" / "surprisal_sweep" / family / safe_model_dir(task.model_id))
        rev = (task.revision or "main").replace("/", "_")
        pairs = [(root / "nieuwland_lme_pooled_surprisal.npy", root / "nieuwland_lme_pooled_table.parquet"),
                 (root / f"nieuwland_lme_pooled_surprisal_{rev}.npy", root / f"nieuwland_lme_pooled_table_{rev}.parquet")]
        npy, tp = next(((a, b) for a, b in pairs if a.is_file() and b.is_file()), (None, None))
        if npy is None:
            return None
        table = pd.read_parquet(tp, columns=["item_id"])
        vals = np.load(npy, mmap_mode="r")
        if vals.shape[0] != len(table):
            raise ValueError(f"row mismatch {npy}")
        table["surprisal"] = np.asarray(vals[:, -1], dtype=float)
        per_item = table.groupby("item_id").surprisal.mean()
        s = np.array([per_item.get(int(i), np.nan) for i in self.items.item_condition_id])
        return None if not _pairs_ok(s) else pdist(s[:, None], metric="euclidean")


def run(name, out_root):
    src = Nieuwland() if name == "nieuwland" else Szewczyk(name)
    dataset = src.dataset
    # Nieuwland item-mean results live in embedding_sweep_paper_n400 (nieuwland_paper.sh), the Szewczyk ones in embedding_sweep.
    files = [p for sub in ("embedding_sweep", "embedding_sweep_paper_n400")
             for p in (ROOT / "outputs_jeanzay" / dataset / sub).glob("*/*/*item_mean_scores.csv")]
    enc = pd.concat([pd.read_csv(p, dtype={"revision": str}) for p in files])
    enc = enc[["family", "model_id", "revision", "last_layer_pearsonr"]].rename(
        columns={"last_layer_pearsonr": "encoding_pearson"})
    keys = {(r.family, r.model_id, r.revision) for r in enc.itertuples()}
    tasks = [(f, t) for f, t in P.registry_tasks() if (f, t.model_id, t.revision or "main") in keys]
    rows = {"embedding": [], "surprisal": []}
    for n, (family, task) in enumerate(tasks, 1):
        rev = task.revision or "main"
        print(f"{dataset}: {n}/{len(tasks)} {task.model_id}@{rev}", flush=True)
        for track in rows:
            dist = getattr(src, track)(family, task)
            if dist is None:
                continue
            rows[track].append(dict(family=family, model_id=task.model_id, revision=rev,
                                    n_items=len(src.y), rsa_pearson=_score(dist, src.y)))
    for track, data in rows.items():
        summary = pd.DataFrame(data)
        meta = [lookup_model_meta(r.model_id, r.family) for r in summary.itertuples()]
        summary["group"] = [m.group for m in meta]
        summary["n_params"] = [m.n_params for m in meta]
        if track == "embedding":
            summary = summary.merge(enc, on=["family", "model_id", "revision"], how="left", validate="one_to_one")
        out = out_root / dataset / track
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "model_summary.csv", index=False)
        print(f"{dataset}/{track}: {len(summary)} models", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("studies", nargs="+", choices=["nieuwland", "f2007", "h2019"])
    parser.add_argument("--out-root", type=Path,
                        default=Path(os.environ.get("SCRATCH", "/tmp")) / "n400_item_mean_rsa")
    parser.add_argument("--standardize-features", action="store_true")
    args = parser.parse_args()
    STANDARDIZE = args.standardize_features
    for s in args.studies:
        run(s, args.out_root)
