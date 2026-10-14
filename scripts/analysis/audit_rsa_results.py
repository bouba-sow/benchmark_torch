"""Read-only RSA audit; writes diagnostics under reports/rsa_summary/audits.

Run with the project interpreter. Does not change scores or regenerate figures.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/rsa-audit-matplotlib"
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from neuroai.analysis.rdm_regression import RdmCvPlan, standardize_features
from scripts.figures import rdm_regression_figures as F
from scripts.analysis.build_rsa_report import EXPECTED_KEYS

S = Path(os.environ.get("SCRATCH", "/lustre/fsn1/projects/rech/ldh/una68ug"))
OUT = ROOT / "reports/rsa_summary/audits/rsa_investigation_20261005"
OUT.mkdir(parents=True, exist_ok=True)
K = ["family", "model_id", "revision"]

def registry(d):
    return d.loc[[tuple(r) in EXPECTED_KEYS for r in d[K].itertuples(index=False, name=None)]].copy()

def front(d):
    return F.pareto_front(d, "neural", "behav")

def main():
    d = F.with_neural_units(F.load(S / "rdm_report_std/rdm_last_layer_results.csv"),
                            S / "rdm_neural_summary/neural_rdm_summary.csv")
    dup = d[d.duplicated(["dataset", "unit", "track", "confounds"] + K, keep=False)]
    dup.to_csv(OUT / "duplicate_input_rows.csv", index=False)
    d = registry(d)
    source = pd.read_csv(S / "rdm_report_std/rdm_last_layer_results.csv")
    behavioral = source[source.dataset.isin(F.BEHAV_TITLES) &
                        (source.confounds == "len_freq")].copy()
    behavioral["full_ratio"] = behavioral.last_layer_delta_ll_per_obs / behavioral.ceiling_delta_ll_per_obs
    behavioral["full_to_half_score"] = behavioral.last_layer_delta_ll_per_obs / behavioral.half_target_delta_ll_per_obs
    behavioral.to_csv(OUT / "behavioral_full_vs_half.csv", index=False)
    cv = pd.read_csv(F.OUT / "rdm_ceiling_values.csv")
    cv["dataset"] = cv.dataset_key.map(lambda x: F.UNIT_DATASET.get(x, x))
    norm = F.apply_ceilings(d, F.OUT / "rdm_ceiling_values.csv")
    saved = pd.read_csv(F.OUT / "rdm_pareto_scores.csv")
    summaries, comps = [], []
    for track in ("embedding", "surprisal"):
        current = F.composites(norm, track, "len_freq")
        old = saved[(saved.track == track) & (saved.score == "ceiling_normalised")]
        check = current.merge(old, on=K, suffixes=("", "_saved"), validate="one_to_one")
        print(track, "reconstructed", len(check), "max saved error",
              max((check.neural-check.neural_saved).abs().max(),
                  (check.behav-check.behav_saved).abs().max()), flush=True)
        # Sensitivity: retain every unit with a positive reference; no score-dependent exclusions.
        allnorm = d.drop(columns="norm").merge(cv[["dataset", "unit", "ceiling_dll"]],
                                               on=["dataset", "unit"], validate="many_to_one")
        allnorm["norm"] = np.where(allnorm.ceiling_dll > 0,
                                  allnorm.last_layer_delta_ll_per_obs / allnorm.ceiling_dll, np.nan)
        for variant, inp in (("current", norm), ("all_positive_references", allnorm)):
            c = F.composites(inp, track, "len_freq")
            c["on_front"] = c.index.isin(front(c).index)
            c["track"], c["variant"] = track, variant
            comps.append(c)
        part = norm[(norm.track == track) & (norm.confounds == "len_freq")]
        for (ds, unit), g in part.groupby(["dataset", "unit"]):
            summaries.append(dict(dataset=ds, unit=unit, track=track, n=len(g),
                                  median_dll=g.last_layer_delta_ll_per_obs.median(),
                                  fraction_positive=(g.last_layer_delta_ll_per_obs > 0).mean(),
                                  ceiling=g.ceiling_dll.iloc[0], included=g.norm.notna().any(),
                                  median_ratio=g.norm.median() if g.norm.notna().any() else np.nan))
    pd.DataFrame(summaries).to_csv(OUT / "unit_diagnostics.csv", index=False)
    comp = pd.concat(comps, ignore_index=True)
    comp.to_csv(OUT / "pareto_sensitivity.csv", index=False)
    print(comp.groupby(["track", "variant"]).agg(n=("neural", "size"),
          neural=("neural", "median"), behav=("behav", "median"),
          n_neural=("n_neural", "min"), fronts=("on_front", "sum")).to_string(), flush=True)
    print("Alternative fronts:", comp[(comp.variant != "current") & comp.on_front][
          ["track", "label", "neural", "behav"]].to_string(index=False), flush=True)
    # Gaussian example: distance transformation attenuates scalar prediction signals.
    rng = np.random.default_rng(7)
    x, e = rng.normal(size=(2, 1500))
    rows = []
    for rho in (0., .1, .3, .5, .8):
        y = rho*x + np.sqrt(1-rho*rho)*e
        xd, yd = pdist(x[:, None]), pdist(y[:, None])
        score = RdmCvPlan(yd, len(y), n_repeats=5, seed=0).delta_ll(xd)
        rows.append(dict(requested_rho=rho, item_r=np.corrcoef(x,y)[0,1],
                         distance_r=np.corrcoef(xd,yd)[0,1], dll=score.delta_ll_per_obs))
    pd.DataFrame(rows).to_csv(OUT / "distance_attenuation.csv", index=False)
    # Numerical safety / equivalence to the explicit paper formula.
    a = rng.normal(size=(200, 64)).astype(np.float32)
    a[:, 0] = 10
    z = standardize_features(a)
    print("standardization: input unchanged", bool(np.all(a[:,0] == 10)),
          "constant output", float(abs(z[:,0]).max()),
          "max mean", float(abs(z[:,1:].mean(0)).max()),
          "max std error", float(abs(z[:,1:].std(0)-1).max()), flush=True)
    # Scoring invariance to affine scaling of already constructed RDM vectors.
    p = RdmCvPlan(yd, len(y), n_repeats=2)
    a_score = p.delta_ll(xd).delta_ll_per_obs
    b_score = RdmCvPlan(yd*3+4, len(y), n_repeats=2).delta_ll(xd*2-7).delta_ll_per_obs
    print("RDM affine scaling dLL error", abs(a_score-b_score), flush=True)
    print("Audit written to", OUT, flush=True)

if __name__ == "__main__":
    main()
