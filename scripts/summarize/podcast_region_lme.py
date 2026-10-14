#!/usr/bin/env python3
"""Summarize the Podcast per-REGION surprisal sweep (region-averaged electrodes, lags 0-600 ms).

Reads
  <sweep_dir>/<family>/<model>/podcast_region_lme_by_lag.csv     one row per (region, lag)
  <ceiling_dir>/delta_loglik_ceiling_podcast_region_oos_lagNN.json one ceiling per lag, per region

Score (mean over lags, as for the other lag-based headline conventions; no peak picking):
  dll[r, l]   = delta_loglik_test / n_obs_test of region r at lag l
  region r    : S_r = mean_l dll[r, l]            ceiling C_r = mean_l c[r, l]      score S_r / C_r
  headline    : score_norm = sum_r n_r * (S_r / C_r) / sum_r n_r      (EACH REGION NORMALISED BY ITS OWN
                ceiling, then n_r-weighted; switched 2026-10-09 from the ratio of means S / C)
  reference   : C = sum_r n_r * C_r / sum_r n_r (the ceiling stored in ceiling_summary.json), and
                score_dll = score_norm * C is the headline in delta-LL units, so that score_dll / C == score_norm
                wherever the benchmark divides a delta-LL score by the dataset ceiling.
                score_dll_pooled = sum_r n_r S_r / sum_r n_r is the old (ratio-of-means numerator).
n_r is the model's held-out row count in region r (identical across lags, and across models).
A model enters only if every (region, lag) fit is present, error-free and finite; otherwise it is
reported in the diagnostics but gets no headline score (no silent partial averages).

Outputs (default outputs_jeanzay/podcast/summary_region/):
  all_models_region_by_lag.csv   raw rows of all models (+ family, model_id)
  ceiling_by_region_lag.csv      the per-lag, per-region ceilings used
  model_summary.csv              one row per model: S, S/C, per-region S_r and S_r/C_r, diagnostics
  ceiling_summary.json           combined + per-region ceilings (key ceiling_delta_loglik_per_obs = C)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np
import pandas as pd

DEFAULT_SWEEP = Path("outputs_jeanzay/podcast/surprisal_region_sweep")
DEFAULT_CEIL = Path("outputs_jeanzay/noise_ceilings/podcast_lags_region")
DEFAULT_OUT = Path("outputs_jeanzay/podcast/summary_region")


def load_ceilings(ceil_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(ceil_dir.glob("delta_loglik_ceiling_podcast_region_oos_lag*.json")):
        d = json.loads(f.read_text())
        lag = int(re.search(r"lag(\d+)", f.name).group(1))
        for region, v in d["per_region"].items():
            rows.append({"lag": lag, "region": region, "ceiling": v["ceiling_delta_loglik_per_obs"],
                         "ceiling_sd": v["ceiling_delta_loglik_per_obs_sd"],
                         "n_iterations_used": v["n_iterations_used"]})
    return pd.DataFrame(rows)


def load_sweep(sweep_dir: Path, families: list[str] | None) -> pd.DataFrame:
    frames = []
    for f in sorted(sweep_dir.glob("*/*/podcast_region_lme_by_lag.csv")):
        family, model = f.parent.parent.name, f.parent.name
        if families and family not in families:
            continue
        d = pd.read_csv(f)
        d.insert(0, "model", model)
        d.insert(0, "family", family)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(sweep: pd.DataFrame, ceil: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    regions = sorted(ceil["region"].unique())
    lags = sorted(ceil["lag"].unique())
    cmat = ceil.pivot(index="region", columns="lag", values="ceiling").reindex(index=regions, columns=lags)
    if cmat.isna().any().any():
        raise SystemExit(f"ceilings missing/NaN for some (region, lag):\n{cmat[cmat.isna().any(axis=1)]}")
    c_r = cmat.mean(axis=1)                                   # C_r = mean over lags

    rows, n_r_ref = [], None
    for (family, model), g in sweep.groupby(["family", "model"]):
        g = g.copy()
        for c in ("delta_loglik_test", "n_obs_test"):
            g[c] = pd.to_numeric(g[c], errors="coerce")
        ok = g[g["error"].isna() & np.isfinite(g["delta_loglik_test"]) & (g["n_obs_test"] > 0)]
        row = {"family": family, "model": model, "n_fits": len(g), "n_ok": len(ok),
               "n_not_converged": int((~g["converged"].astype(bool)).sum()) if "converged" in g else np.nan}
        have = {(r, int(l)) for r, l in zip(ok["region"], ok["lag"])}
        complete = all((r, l) in have for r in regions for l in lags)
        row["complete"] = complete
        if complete:
            ok = ok[ok["region"].isin(regions) & ok["lag"].isin(lags)].drop_duplicates(["region", "lag"])
            ok = ok.assign(dll=ok["delta_loglik_test"] / ok["n_obs_test"])
            s_r = ok.groupby("region")["dll"].mean().reindex(regions)
            n_r = ok.groupby("region")["n_obs_test"].first().reindex(regions)
            S = float((n_r * s_r).sum() / n_r.sum())
            C = float((n_r * c_r).sum() / n_r.sum())
            norm = float((n_r * (s_r / c_r)).sum() / n_r.sum())
            row.update(score_dll=norm * C, score_dll_pooled=S, ceiling=C, score_norm=norm)
            for r in regions:
                row[f"dll_{r}"] = float(s_r[r])
                row[f"norm_{r}"] = float(s_r[r] / c_r[r])
            n_r_ref = n_r if n_r_ref is None else n_r_ref
        rows.append(row)
    summ = pd.DataFrame(rows)
    n_r = n_r_ref if n_r_ref is not None else pd.Series(np.nan, index=regions)
    C = float((n_r * c_r).sum() / n_r.sum()) if n_r.notna().all() else float("nan")
    info = {
        "method": "split_half_region_averaged_per_region_lme_mean_over_lags_per_region_normalised",
        "lags": [int(l) for l in lags], "regions": regions,
        "ceiling_delta_loglik_per_obs": C,
        "per_region_ceiling": {r: float(c_r[r]) for r in regions},
        "per_region_n_obs_test": {r: (None if pd.isna(n_r[r]) else int(n_r[r])) for r in regions},
        "weights": "n_obs_test of the sweep (held-out rows per region)",
    }
    return summ, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", default=str(DEFAULT_SWEEP))
    ap.add_argument("--ceiling_dir", default=str(DEFAULT_CEIL))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--families", nargs="*", default=None)
    args = ap.parse_args()

    ceil = load_ceilings(Path(args.ceiling_dir))
    sweep = load_sweep(Path(args.sweep_dir), args.families)
    if ceil.empty or sweep.empty:
        raise SystemExit(f"nothing to summarize (ceilings: {len(ceil)} rows, sweep: {len(sweep)} rows)")
    summ, info = summarize(sweep, ceil)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "all_models_region_by_lag.csv", index=False)
    ceil.to_csv(out / "ceiling_by_region_lag.csv", index=False)
    summ.to_csv(out / "model_summary.csv", index=False)
    (out / "ceiling_summary.json").write_text(json.dumps(info, indent=2))
    n_c = int(summ["complete"].sum())
    print(f"{len(summ)} models, {n_c} complete; combined ceiling = {info['ceiling_delta_loglik_per_obs']:.6f}")
    print("per-region ceiling:", {k: round(v, 5) for k, v in info["per_region_ceiling"].items()})
    if n_c:
        print(summ.loc[summ["complete"], "score_norm"].describe().round(4).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
