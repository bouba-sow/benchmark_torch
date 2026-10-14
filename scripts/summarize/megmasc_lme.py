#!/usr/bin/env python3
"""Summarize MEG-MASC (Gwilliams) pooled-LME-per-layer surprisal sweeps.

Reads ``<sweep_dir>/<family>/<model>/megmasc_lme_by_layer.csv`` — one row
per LM layer, pooled across subjects/channels. Primary metric:
``delta_loglik_test`` (held-out log-likelihood gain from adding per-layer
logit-lens surprisal), scored per-observation and ceiling-normalized
downstream (see ``megmasc_meg__surprisal`` in
``scripts/summarize/benchmark.py``'s ``NEURAL_SURPRISAL_SPEC``).

A crashed layer (OOM-killed worker, unconverged fit) leaves a row with
``error`` set and all score columns NaN — dropped from every summary stat,
same convention as every other pooled-LME dataset in this codebase.

This is deliberately minimal (aggregation only, no plots) — mirrors
``scripts/summarize/broderick_lme.py``'s aggregation logic exactly, since
megmasc's raw per-model sweep schema is identical.

Outputs (default ``outputs/megmasc/lme_summary/``):
  all_models_lme_by_layer.csv  — concatenated raw per-layer rows, all models
  peak_summary.csv             — one row per model: peak/mean metrics,
                                  convergence + crash diagnostics

Usage
-----
  python scripts/summarize/megmasc_lme.py --families gpt2 pythia ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import numpy as np
import pandas as pd

from neuroai.models.registry import LPP_FAMILIES, model_spec_to_cfg

DEFAULT_SWEEP_DIR = Path("outputs_jeanzay/megmasc/surprisal_sweep")
DEFAULT_OUT_DIR = Path("outputs/megmasc/lme_summary")

LME_METRICS: list[tuple[str, str]] = [
    ("delta_loglik_test", "ΔLogLik (test, held-out)"),
    ("test_delta_pearsonr", "ΔPearson r (test)"),
    ("test_delta_spearmanr", "ΔSpearman r (test)"),
]


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "_")


def _registry_meta() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for fam, specs in LPP_FAMILIES.items():
        for spec in specs:
            cfg = model_spec_to_cfg(spec, fam)
            row = {
                "model_id": spec.model_id,
                "model_label": spec.label,
                "n_params": float(spec.n_params),
                "family": fam,
            }
            meta[spec.model_id] = row
            meta[_safe_model_dir(spec.model_id)] = row
    return meta


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _discover_families(sweep_dir: Path) -> list[str]:
    families: list[str] = []
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_dir() or p.name == "summary":
            continue
        if list(p.glob("*/megmasc_lme_by_layer.csv")):
            families.append(p.name)
    return families


def _load_family(sweep_dir: Path, family: str, meta: dict[str, dict]) -> pd.DataFrame:
    fam_dir = sweep_dir / family
    parts: list[pd.DataFrame] = []
    for model_dir in sorted(fam_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_path = model_dir / "megmasc_lme_by_layer.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        model_id = model_dir.name.replace("_", "/", 1)
        m = meta.get(model_id, meta.get(model_dir.name, {}))
        df.insert(0, "model_id", m.get("model_id", model_id))
        df.insert(1, "family", family)
        df["model_label"] = m.get("model_label", model_id)
        df["n_params"] = m.get("n_params", np.nan)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _peak_summary_row(model_id: str, model_label: str, family: str, n_params: float, df: pd.DataFrame) -> dict:
    n_total = len(df)
    crashed = df["error"].notna()
    n_crashed = int(crashed.sum())
    ok = df.loc[~crashed].copy()
    n_converged = int(ok["converged"].fillna(False).astype(bool).sum()) if "converged" in ok else 0
    n_slope_dropped = int(ok["slope_dropped"].fillna(False).astype(bool).sum()) if "slope_dropped" in ok else 0

    row: dict = {
        "model_id": model_id,
        "model_label": model_label,
        "family": family,
        "n_params": n_params,
        "n_params_b": n_params / 1e9 if np.isfinite(n_params) else np.nan,
        "n_layers_total": n_total,
        "n_layers_crashed": n_crashed,
        "n_layers_converged": n_converged,
        "n_layers_slope_dropped": n_slope_dropped,
        "crash_rate": n_crashed / n_total if n_total else np.nan,
    }
    last_layer_idx = int(ok["layer"].max()) if "layer" in ok and not ok.empty else None
    for metric, _ in LME_METRICS:
        if metric not in ok.columns or not np.isfinite(ok[metric]).any():
            row[f"peak_{metric}"] = np.nan
            row[f"peak_{metric}_layer"] = np.nan
            row[f"mean_{metric}"] = np.nan
            row[f"last_{metric}"] = np.nan
            continue
        finite = ok[np.isfinite(ok[metric])]
        peak_idx = finite[metric].idxmax()
        row[f"peak_{metric}"] = float(finite.loc[peak_idx, metric])
        row[f"peak_{metric}_layer"] = float(finite.loc[peak_idx, "layer"])
        row[f"mean_{metric}"] = float(finite[metric].mean())
        last_row = ok.loc[ok["layer"] == last_layer_idx] if last_layer_idx is not None else pd.DataFrame()
        last_val = last_row[metric].iloc[0] if not last_row.empty else np.nan
        row[f"last_{metric}"] = float(last_val) if np.isfinite(last_val) else np.nan
    return row


def summarize_family(sweep_dir: Path, family: str, out_root: Path, meta: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = out_root / family
    df = _load_family(sweep_dir, family, meta)
    if df.empty:
        print(f"[{family}] no LME layer results found")
        return df, pd.DataFrame()

    _write_csv(df, out_dir / "all_layers.csv")

    rows = []
    for model_id, grp in df.groupby("model_id"):
        rows.append(_peak_summary_row(
            model_id, grp["model_label"].iloc[0], family, grp["n_params"].iloc[0], grp,
        ))
    summary = pd.DataFrame(rows).sort_values("n_params_b", kind="mergesort").reset_index(drop=True)
    _write_csv(summary, out_dir / "peak_summary.csv")

    n_crashed_total = int(summary["n_layers_crashed"].sum())
    if n_crashed_total:
        print(f"[{family}] {n_crashed_total} crashed/unconverged layer fit(s) across {len(summary)} model(s)")
    print(f"[{family}] wrote LME layer summaries to {out_dir}")
    return df, summary


def run_lme_summary(sweep_dir: Path | str, out_dir: Path | str | None = None, *, families: list[str] | None = None) -> None:
    sweep_path = Path(sweep_dir).resolve()
    out_path = Path(out_dir).resolve() if out_dir else DEFAULT_OUT_DIR.resolve()

    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Missing sweep dir: {sweep_path}")

    meta = _registry_meta()
    fams = families or _discover_families(sweep_path)
    if not fams:
        print(f"No megmasc_lme_by_layer.csv found under {sweep_path}")
        return

    print(f"Summarizing MEG-MASC LME families: {fams}")
    print(f"Reading from: {sweep_path}")
    print(f"Writing to:   {out_path}")

    all_layers: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    for fam in fams:
        df, summary = summarize_family(sweep_path, fam, out_path, meta)
        if not df.empty:
            all_layers.append(df)
        if not summary.empty:
            all_summaries.append(summary)

    if all_layers:
        combined_layers = pd.concat(all_layers, ignore_index=True)
        _write_csv(combined_layers, out_path / "all_models_lme_by_layer.csv")
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        _write_csv(combined_summary, out_path / "peak_summary.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep_dir", type=Path, default=DEFAULT_SWEEP_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--families", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_lme_summary(args.sweep_dir, args.out_dir, families=args.families)
    print("Done.")


if __name__ == "__main__":
    main()
