#!/usr/bin/env python3
"""Aggregate megmasc_signal_check.py outputs -> text report + CSV.

Metrics per (variant, outcome, window), mean over units then over subjects:
  retest_r   session-A vs session-B across-word Pearson r (check 2)
  xsubj_r    subject vs mean-of-other-subjects across-word r, i.e. what the
             split-half DLL-ceiling oracle relies on (check 2/fix 4 decider)
  ctrl_dur   |r| with log word duration (positive control, check 3)
  ctrl_len   |r| with word length
"""
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd


def corr_cols(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    d = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return (a * b).sum(0) / np.where(d == 0, np.nan, d)


ap = argparse.ArgumentParser()
ap.add_argument("--in-dir", required=True)
ap.add_argument("--out-csv", required=True)
a = ap.parse_args()

subs = {}
for f in sorted(glob.glob(f"{a.in_dir}/sub-*.npz")):
    subs[Path(f).stem] = dict(np.load(f))
print("subjects:", list(subs))

# ---- check 1: evoked GFP
times = next(iter(subs.values()))["times"]
print("\n== CHECK 1: evoked global field power (peak time, peak/baseline ratio) ==")
for v in ("nobl", "bl"):
    rows = []
    for s, d in subs.items():
        g = d[f"gfp_{v}"]; post = times > 0.0
        pk = times[post][np.argmax(g[post])]
        pre = g[(times >= -0.2) & (times < 0)].mean()
        rows.append((s, pk, g[post].max() / pre))
    print(v, "peak times (s):", " ".join(f"{r[1]:.2f}" for r in rows),
          "| peak/pre ratio mean %.2f" % np.mean([r[2] for r in rows]))
g = np.mean([d["gfp_bl"] for d in subs.values()], 0)
print("grand GFP (bl) every 100ms:", " ".join(f"{t:+.1f}:{x:.3f}" for t, x in zip(times[::5], g[::5])))

# ---- per-word tables
def word_table(d, name):
    Y = d[name]; k = d["key"]; s = d["sess"]
    return k, s, Y

res = []
variants = ("nobl", "bl")
names = sorted({n[len("y_nobl_"):] for d in subs.values() for n in d if n.startswith("y_nobl_")})
for v in variants:
    for nm in names:
        full = f"y_{v}_{nm}"
        ret, xs, cd, cl = [], [], [], []
        # mean-over-sessions per unique word per subject
        per = {}
        for s, d in subs.items():
            k, se, Y = d["key"], d["sess"], d[full]
            df = pd.DataFrame(Y); df["k"] = k; df["se"] = se
            A = df[df.se == 0].groupby("k").mean().drop(columns="se")
            B = df[df.se == 1].groupby("k").mean().drop(columns="se") if (se == 1).any() else None
            if B is not None:
                c = A.index.intersection(B.index)
                ret.append(np.nanmean(corr_cols(A.loc[c].to_numpy(), B.loc[c].to_numpy())))
            M = df.drop(columns="se").groupby("k").mean()
            per[s] = M
            cov = pd.DataFrame(dict(k=k, dur=np.log(d["dur"]), wl=d["wlen"])).groupby("k").mean()
            c = M.index.intersection(cov.index)
            cd.append(np.nanmean(np.abs(corr_cols(M.loc[c].to_numpy(), np.repeat(cov.loc[c, "dur"].to_numpy()[:, None], M.shape[1], 1)))))
            cl.append(np.nanmean(np.abs(corr_cols(M.loc[c].to_numpy(), np.repeat(cov.loc[c, "wl"].to_numpy()[:, None], M.shape[1], 1)))))
        common = set.intersection(*[set(m.index) for m in per.values()])
        common = sorted(common)
        for s, m in per.items():
            others = np.mean([per[t].loc[common].to_numpy() for t in per if t != s], 0)
            xs.append(np.nanmean(corr_cols(m.loc[common].to_numpy(), others)))
        res.append(dict(variant=v, outcome_window=nm, retest_r=np.mean(ret), xsubj_r=np.mean(xs),
                        ctrl_dur=np.mean(cd), ctrl_len=np.mean(cl), n_common_words=len(common)))
R = pd.DataFrame(res)
R.to_csv(a.out_csv, index=False)
pd.set_option("display.width", 200)
print("\n== CHECKS 2/3 + FIX 4 (r across words; higher = more signal) ==")
print(R.round(4).to_string(index=False))
