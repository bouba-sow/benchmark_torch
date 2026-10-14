#!/usr/bin/env python3
"""Compare last-layer Pereira embedding RSA with existing ridge encoding scores.

Each RSA score is a Pearson correlation between two Pearson-distance RDMs.
No model is fitted. Run from the repository root:

    .venv/bin/python scripts/analysis/pereira_rsa.py --all-scored
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = "/tmp/pereira-rsa-matplotlib"

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

from neuroai.analysis.benchmark_common import filter_registry_models
from neuroai.analysis.pereira_rsa import load_last_layer, prepare_neural_rdms, score_last_layer
from neuroai.datasets.pereira import PereiraDataset
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
from neuroai.sweep import safe_model_dir


def _tasks(encoding_root: Path) -> list[tuple[str, str, str, str]]:
    """Registry tasks that actually have a Pereira encoding score."""
    tasks = []
    revisions_by_file: dict[Path, set[str]] = {}
    for family in LPP_FAMILY_NAMES:
        for task in flatten_family_tasks(family):
            path = encoding_root / family / task.model_dir_safe / "pereira_embed_by_layer.csv"
            if not path.exists():
                continue
            if path not in revisions_by_file:
                revisions_by_file[path] = set(pd.read_csv(path, usecols=["revision"])["revision"].astype(str))
            if task.revision in revisions_by_file[path]:
                tasks.append((family, task.model_id, task.load_model_id, task.revision))
    return list(dict.fromkeys(tasks))


def _summarize(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in output_root.glob("*/*/comparison_*.csv"):
        by_subject = pd.read_csv(path, dtype={"subject": str})
        if by_subject.empty:
            continue
        rows.append({
            "family": path.parent.parent.name,
            "model_id": by_subject.model_id.iloc[0],
            "revision": by_subject.revision.iloc[0],
            "n_subjects": by_subject.subject.nunique(),
            "encoding_pearson": by_subject.encoding_pearson.mean(),
            "rsa_pearson": by_subject.rsa_pearson.mean(),
        })
    summary = filter_registry_models(pd.DataFrame(rows))
    if not summary.empty:
        from neuroai.analysis.benchmark_common import lookup_model_meta
        metadata = [lookup_model_meta(row.model_id, row.family) for row in summary.itertuples()]
        summary["group"] = [item.group for item in metadata]
        summary["n_params"] = [item.n_params for item in metadata]
        summary.sort_values(["family", "model_id", "revision"], inplace=True)
        summary.to_csv(output_root / "model_summary.csv", index=False)
        if len(summary) >= 3:
            correlation = pearsonr(summary.encoding_pearson, summary.rsa_pearson)
            print(f"Across {len(summary)} models: encoding vs RSA Pearson r = {correlation.statistic:.4f} "
                  "(descriptive; related checkpoints are not independent)", flush=True)
            _plot_comparison(summary, output_root / "encoding_vs_rsa.png", correlation.statistic)
            _scaling_summary(summary, output_root)
    return summary


def _plot_comparison(summary: pd.DataFrame, path: Path, across_model_r: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    for group in GROUP_ORDER:
        part = summary.loc[summary.group == group]
        if not part.empty:
            axis.scatter(part.encoding_pearson, part.rsa_pearson, s=38, alpha=0.75,
                         color=GROUP_COLORS[group], label=GROUP_DISPLAY[group])
    axis.set_xlabel("Last-layer ridge encoding (Pearson r)")
    axis.set_ylabel("Last-layer fit-free RSA (Pearson r)")
    axis.set_title(f"Pereira 2018 · {len(summary)} models · across-model r = {across_model_r:.2f}")
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _scaling_summary(summary: pd.DataFrame, output_root: Path) -> None:
    """Describe size trends; families are kept separate in the saved table."""
    valid = summary.loc[np.isfinite(summary.n_params) & (summary.n_params > 0)].copy()
    valid["log10_params"] = np.log10(valid.n_params)
    rows = []
    scopes = [("all", "all models", valid)]
    scopes += [("group", name, part) for name, part in valid.groupby("group")]
    scopes += [("family", name, part) for name, part in valid.groupby("family")]
    for scope, name, part in scopes:
        if len(part) < 3 or part.log10_params.nunique() < 3:
            continue
        for metric in ("rsa_pearson", "encoding_pearson"):
            fit = linregress(part.log10_params, part[metric])
            rows.append({"scope": scope, "name": name, "metric": metric,
                         "n_models": len(part), "slope_per_log10_param": fit.slope,
                         "pearson_r": fit.rvalue, "r_squared": fit.rvalue ** 2})
    variable_families = valid.groupby("family").filter(lambda part: part.log10_params.nunique() > 1)
    centered_size = variable_families.log10_params - variable_families.groupby("family").log10_params.transform("mean")
    for metric in ("rsa_pearson", "encoding_pearson"):
        centered_score = variable_families[metric] - variable_families.groupby("family")[metric].transform("mean")
        slope = float((centered_size * centered_score).sum() / (centered_size ** 2).sum())
        r = float(pearsonr(centered_size, centered_score).statistic)
        rows.append({"scope": "within_family", "name": "family-centered", "metric": metric,
                     "n_models": len(variable_families), "slope_per_log10_param": slope,
                     "pearson_r": r, "r_squared": r ** 2})
    scaling = pd.DataFrame(rows)
    scaling.to_csv(output_root / "scaling_summary.csv", index=False)
    _plot_scaling(valid, output_root / "rsa_scaling.png")
    print(scaling.loc[scaling.scope.isin(["all", "group", "within_family"])].to_string(index=False,
          float_format=lambda value: f"{value:.4f}"), flush=True)


def _plot_scaling(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    for axis, metric, ylabel in zip(
        axes, ("rsa_pearson", "encoding_pearson"),
        ("Pearson RSA score", "Ridge encoding Pearson r"),
    ):
        for group in GROUP_ORDER:
            part = summary.loc[summary.group == group]
            if part.empty:
                continue
            axis.scatter(part.n_params, part[metric], s=30, alpha=0.7,
                         color=GROUP_COLORS[group], label=GROUP_DISPLAY[group])
        axis.set_xscale("log")
        axis.set_xlabel("Model parameters")
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-scored", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild summary and plot without reading neural data")
    parser.add_argument("--model", nargs=2, action="append", metavar=("FAMILY", "MODEL_ID"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/pereira_embed_features"))
    parser.add_argument("--data", type=Path, default=ROOT / "data/pereira_fmri/Pereira2018_merged.nc")
    parser.add_argument("--encoding-root", type=Path, default=ROOT / "outputs_jeanzay/pereira/embedding_sweep")
    parser.add_argument("--standardize-features", action="store_true",
                        help="z-score each embedding feature across the sentences before the model RDM")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/pereira/pearson_rsa")
    args = parser.parse_args()
    if args.summarize_only:
        _summarize(args.output_root)
        return
    tasks = _tasks(args.encoding_root) if args.all_scored else []
    tasks.extend((family, model_id, model_id, args.revision) for family, model_id in (args.model or []))
    tasks = list(dict.fromkeys(tasks))
    if not tasks:
        parser.error("Specify --all-scored or --model FAMILY MODEL_ID")
    args.output_root.mkdir(parents=True, exist_ok=True)

    # The large NetCDF read happens once. All subsequent model scores use the
    # same neural RDMs and differ only in their cached last-layer states.
    dataset = PereiraDataset(args.data)
    _, item_ids = dataset.get_sentences()
    neural_rdms = prepare_neural_rdms(item_ids, dataset.get_voxelwise_per_subject("language"))
    print(f"Prepared {len(neural_rdms)} subject/experiment neural RDMs", flush=True)

    for number, (family, model_id, cache_model_id, revision) in enumerate(tasks, start=1):
        out_dir = args.output_root / family / safe_model_dir(model_id)
        comparison_path = out_dir / f"comparison_{revision.replace('/', '_')}.csv"
        if comparison_path.exists():
            continue
        try:
            states = load_last_layer(args.cache_dir, cache_model_id, revision, len(item_ids))
        except FileNotFoundError as error:
            print(f"[{number}/{len(tasks)}] skipped {model_id}: {error}", flush=True)
            continue
        if args.standardize_features:
            states = (states - states.mean(0)) / (states.std(0) + 1e-8)
        rsa = score_last_layer(states, neural_rdms, model_id=model_id, revision=revision)
        out_dir.mkdir(parents=True, exist_ok=True)
        rsa.to_csv(out_dir / f"rsa_by_experiment_{revision.replace('/', '_')}.csv", index=False)
        by_subject = rsa.groupby(["model_id", "revision", "subject"], as_index=False)["rsa_pearson"].mean()

        score_path = args.encoding_root / family / safe_model_dir(model_id) / "pereira_embed_by_layer.csv"
        scores = pd.read_csv(score_path, dtype={"subject": str})
        scores = scores.loc[(scores.revision == revision) & scores.error.isna()]
        last_layer = scores.layer.max()
        scores = scores.loc[scores.layer == last_layer, ["subject", "pearsonr_mean"]]
        by_subject = by_subject.merge(scores, on="subject", validate="one_to_one")
        if by_subject.subject.nunique() != rsa.subject.nunique():
            raise ValueError(f"Subject mismatch between RSA and encoding for {model_id}@{revision}")
        by_subject.rename(columns={"pearsonr_mean": "encoding_pearson"}, inplace=True)
        by_subject.to_csv(comparison_path, index=False)
        print(f"[{number}/{len(tasks)}] {model_id}: RSA {by_subject.rsa_pearson.mean():.4f}, "
              f"encoding {by_subject.encoding_pearson.mean():.4f}", flush=True)

    _summarize(args.output_root)


if __name__ == "__main__":
    main()
