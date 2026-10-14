#!/usr/bin/env python3
"""RDM regression (held-out delta LL) for the N400 sets: Nieuwland 2018, Federmeier 2007, Hubbard 2019.

One GROUP unit per dataset: the response is the subject-averaged item-mean N400 (the ridge target), a single scalar per
item, so the response RDM is |y_i - y_j| (z-scored). The per-subject item scalars carry no usable signal on their own
(RSA ~ 0), which is why no per-subject units are fitted.

  confounds    : |d critical-word length|, |d Zipf frequency| (compute_wlen_wordfreq_covariates, as in the
                 neural confound RDMs). Federmeier 2007 / Hubbard 2019 also get the other nuisance covariates of the
                 paper formula (calculate_aics_szewczyk.R): |d word position|, |d OLD20|, |d concreteness|. The baseline
                 amplitude (bline) is already regressed out of the target (n400_resid); the random slopes have no
                 counterpart on a subject-averaged item target. The Nieuwland formula has no covariates beyond lab.
  model RDM    : reuse of n400_item_mean_rsa.py (surprisal: |d last-layer surprisal averaged over subjects; embedding:
                 1 - Pearson of last-layer states, each feature z-scored across the items)
  regression   : plain OLS (base y ~ 1 + confounds, full + model RDM), item-fold CV, delta LL per held-out pair.

Output: <out-root>/<dataset>/<track>/units.csv. Never writes under outputs/.
    .venv/bin/python scripts/analysis/rdm_regression_n400_units.py nieuwland f2007 h2019 --out-root $SCRATCH/rdm_n400_units
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

import numpy as np
import pandas as pd

from neuroai.analysis.rdm_regression import RdmCvPlan
from neuroai.features.word_frequency import compute_wlen_wordfreq_covariates
from neuroai.datasets.szewczyk_paper import TSV, load_study
from scripts.analysis import n400_item_mean_rsa as M
from scripts.analysis import n400_paper_rsa as P


def run(name: str, out_root: Path, n_repeats: int, limit: int | None):
    src = M.Nieuwland() if name == "nieuwland" else M.Szewczyk(name)
    words = (src.items.noun if name == "nieuwland" else src.items.critical_word).astype(str).tolist()
    lex = compute_wlen_wordfreq_covariates(words).astype(np.float64)
    n = len(src.y)
    iu = np.triu_indices(n, 1)
    if name != "nieuwland":  # paper-formula item covariates (constant within an item)
        trials = load_study(ROOT / TSV, name).groupby("item")[["pos_start", "old20", "concr"]].first()
        extra = trials.loc[src.items.item].to_numpy(np.float64)
        lex = np.column_stack([lex, extra])
    conf = np.column_stack([np.abs(lex[iu[0], j] - lex[iu[1], j]) for j in range(lex.shape[1])])
    y = (src.y - src.y.mean()) / src.y.std()
    seed = int.from_bytes(hashlib.sha256(f"n400:{name}".encode()).digest()[:4], "little")
    plan = RdmCvPlan(np.abs(y[iu[0]] - y[iu[1]]), n, None, conf, n_repeats, seed=seed)
    # Nieuwland item-mean results live in embedding_sweep_paper_n400 (nieuwland_paper.sh), the Szewczyk ones in embedding_sweep.
    files = [p for sub in ("embedding_sweep", "embedding_sweep_paper_n400")
             for p in (ROOT / "outputs_jeanzay" / src.dataset / sub).glob("*/*/*item_mean_scores.csv")]
    keys = set()
    for p in files:
        t = pd.read_csv(p, dtype={"revision": str})
        keys.update(t[["family", "model_id", "revision"]].itertuples(index=False, name=None))
    tasks = [(f, t) for f, t in P.registry_tasks() if (f, t.model_id, t.revision or "main") in keys]
    if limit:
        tasks = tasks[:limit]
    print(f"{src.dataset}: {n} items, {len(tasks)} models", flush=True)
    for track in ("embedding", "surprisal"):
        rows = []
        for family, task in tasks:
            rev = task.revision or "main"
            try:
                dist = getattr(src, track)(family, task)
                if dist is None:
                    continue
                dist = np.asarray(dist, dtype=np.float64)
                if not np.isfinite(dist).all() or dist.std() == 0:
                    continue
                r = plan.delta_ll(dist)
                rows.append(dict(dataset=src.dataset, track=track, family=family, model_id=task.model_id, revision=rev,
                                 subject="GROUP", unit="item_mean", n_items=n, n_confounds=conf.shape[1], n_pairs=plan.n_pairs,
                                 delta_ll_per_obs=r.delta_ll_per_obs, delta_ll_per_obs_sd=r.delta_ll_per_obs_sd,
                                 rsa_pearson=plan.pearson(dist), frac_pairs_tested=r.frac_pairs_tested))
            except Exception:
                print(f"ERROR {task.model_id}@{rev}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        out = out_root / src.dataset / track / "units.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{src.dataset}/{track}: {len(rows)} models", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("studies", nargs="+", choices=["nieuwland", "f2007", "h2019"])
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if ROOT / "outputs" in a.out_root.resolve().parents:
        ap.error("refusing to write under outputs/")
    M.STANDARDIZE = os.environ.get("RDM_EMBED_STANDARDIZE") != "0"  # =0: raw states (no-standardization runs)
    for s in a.studies:
        run(s, a.out_root, a.n_repeats, a.limit)
