"""Recompute two cached Provo FP models for the RSA investigation (read-only)."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/rsa-audit-matplotlib"
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from neuroai.analysis.rdm_regression import RdmCvPlan, standardize_features
from scripts.analysis import rdm_regression_behavioral as B
from scripts.analysis.behavioral_rsa import _cache_paths
from scripts.analysis.word_neural_embedding_rsa import _last_layer_states

S = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
OUT = ROOT / "reports/rsa_summary/audits/rsa_investigation_20261005"

def main():
    meas = B.load_measure("provo", "FP", 30, 0)
    ids, plan = meas["ids"], meas["plans"]["len_freq"]
    shared = ROOT / "outputs/provo/pearson_rsa/surprisal/rdm_cache/fp/shared/behavior_rdm.npz"
    with np.load(shared) as a:
        y = np.asarray(a["distances"], float)
    c = np.column_stack([np.load(ROOT / f"outputs/provo/pearson_rsa/confounds/fp/{name}_rdm.npz")["distances"]
                         for name in ("word_length", "frequency")])
    results = []
    tasks = [("gpt2", "openai-community/gpt2", "main"),
             ("hard_2gram", "Lanni-ni/hard_2gram_babylm_100m_4layer", "main")]
    for family, model, revision in tasks:
        paths = _cache_paths("provo", model, revision, {int(s):None for s in np.unique(ids[:,0])})
        states = {s:_last_layer_states(p) for s,p in paths.items()}
        x = np.stack([states[int(s)][int(z)-1] for s,z in ids])
        before = x.copy()
        z = standardize_features(x)
        sd = x.astype(float).std(0)
        print(model, "shape", x.shape, "std min/max", sd.min(), sd.max(),
              "near constant", int((sd <= 1e-12).sum()), "finite", np.isfinite(z).all(),
              "unchanged", np.array_equal(before, x), flush=True)
        saved = pd.read_csv(S / f"rdm_regression_std/provo/embedding/fp/{family}/{model.replace('/', '_')}/rdm_regression_main.csv")
        saved = saved[saved.confounds == "len_freq"].iloc[-1]
        for variant, xx in (("raw", x), ("standardized", z)):
            m = pdist(xx, metric="correlation")
            r = plan.delta_ll(m)
            # Explicit dense fitting checks optimized sufficient statistics.
            from neuroai.analysis.rdm_regression import all_pairs, item_fold_assignment
            pairs = all_pairs(len(x))
            fold = item_fold_assignment(len(x), 5, np.random.default_rng(0))
            design = np.column_stack([np.ones(len(y)),c,m])
            optimized = RdmCvPlan(y,len(x),confounds=c,n_repeats=1,seed=0)
            # Full five-fold value is checked separately by a dense reference.
            tot = 0.; nte = 0
            for f in range(5):
                tr = (fold[pairs[:,0]] != f) & (fold[pairs[:,1]] != f)
                te = (fold[pairs[:,0]] == f) & (fold[pairs[:,1]] == f)
                ll = []
                for q in (3,4):
                    beta = np.linalg.lstsq(design[tr,:q],y[tr],rcond=None)[0]
                    s2 = np.mean((y[tr]-design[tr,:q]@beta)**2)
                    ll.append(-.5*np.sum(np.log(2*np.pi*s2)+(y[te]-design[te,:q]@beta)**2/s2))
                tot += ll[1]-ll[0]; nte += te.sum()
            error = abs(tot/nte-optimized.delta_ll(m).delta_ll_per_obs)
            row = dict(model_id=model, variant=variant, n_items=len(x), rsa=plan.pearson(m),
                       dll=r.delta_ll_per_obs, saved_std_dll=saved.delta_ll_per_obs,
                       saved_error=abs(r.delta_ll_per_obs-saved.delta_ll_per_obs),
                       dense_solver_error=error, rdm_min=m.min(),rdm_max=m.max(),rdm_sd=m.std())
            results.append(row)
            print(row, flush=True)
    pd.DataFrame(results).to_csv(OUT / "cached_provo_recomputation.csv", index=False)

def nieuwland():
    import hashlib
    from scripts.analysis import n400_item_mean_rsa as M
    from scripts.analysis import n400_paper_rsa as P
    from neuroai.features.word_frequency import compute_wlen_wordfreq_covariates
    M.STANDARDIZE = True
    src = M.Nieuwland()
    task = next(t for f,t in P.registry_tasks() if f == "gpt2" and t.model_id == "openai-community/gpt2")
    lex = compute_wlen_wordfreq_covariates(src.items.noun.astype(str).tolist()).astype(float)
    i,j = np.triu_indices(len(src.y),1)
    conf = np.column_stack([abs(lex[i,q]-lex[j,q]) for q in range(2)])
    seed = int.from_bytes(hashlib.sha256(b"n400:nieuwland").digest()[:4], "little")
    plan = RdmCvPlan(abs(src.y[i]-src.y[j]),len(src.y),confounds=conf,n_repeats=10,seed=seed)
    rows = []
    for track in ("embedding", "surprisal"):
        dist = getattr(src,track)("gpt2",task)
        r = plan.delta_ll(dist)
        saved = pd.read_csv(S / f"rdm_n400_units/nieuwland/{track}/units.csv")
        saved = saved[saved.model_id == task.model_id].iloc[0]
        row = dict(dataset="nieuwland",track=track,model_id=task.model_id,n_items=len(src.y),
                   rsa=plan.pearson(dist),dll=r.delta_ll_per_obs,
                   saved_error=abs(r.delta_ll_per_obs-saved.delta_ll_per_obs),
                   saved_rsa_error=abs(plan.pearson(dist)-saved.rsa_pearson),
                   split_sd=r.delta_ll_per_obs_sd)
        rows.append(row)
        print(row,flush=True)
    pd.DataFrame(rows).to_csv(OUT / "cached_nieuwland_recomputation.csv",index=False)

if __name__ == "__main__":
    nieuwland() if "--nieuwland-only" in sys.argv else main()
