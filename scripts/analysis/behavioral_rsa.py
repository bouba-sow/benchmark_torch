#!/usr/bin/env python3
"""Cache-only last-layer Pearson RSA for Provo and English MECO reading times.

For each FP/GP/SP measure, correlate model Pearson distances with absolute
differences in grand-average word reading time. No regression is fitted.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = "/tmp/behavioral-rsa-matplotlib"

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import linregress, pearsonr

from neuroai.analysis.benchmark_common import GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, lookup_model_meta
from neuroai.benchmarks.subject_average import average_items_across_subjects
from neuroai.datasets.meco import MecoDataset, meco_word_list
from neuroai.datasets.provo import ProvoDataset, passage_word_lists
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
from neuroai.sweep import safe_model_dir

MEASURES = ("FP", "GP", "SP")
CACHE_ROOT = Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache")


def _dataset_root(dataset: str) -> Path:
    if dataset == "provo":
        return ROOT / "data/provo/osfstorage"
    return (ROOT / "data/meco/osfstorage/release 2.0/version 2.1/wave 1/"
            "primary data/eye tracking data")


def _prepare(dataset: str):
    root = _dataset_root(dataset)
    if dataset == "provo":
        sections = {int(story): list(words) for story, words in passage_word_lists(root).items()}
    else:
        word_table = meco_word_list(root, language="en", exclude_doc_edges=True)
        sections = {int(story): part.sort_values("zone").word.astype(str).tolist()
                    for story, part in word_table.groupby("story_id")}
    responses = {}
    for measure in MEASURES:
        ds = (ProvoDataset(root, measure, aggregate=False) if dataset == "provo"
              else MecoDataset(root, measure, language="en", aggregate=False))
        trials = ds.load_words().dropna(subset=["rt"])
        trials = trials.loc[~trials.is_edge.astype(bool)]
        items = average_items_across_subjects(trials, item_key_cols=("story_id", "zone"))
        items = items.sort_values(["story_id", "zone"]).reset_index(drop=True)
        for row in items.itertuples():
            if row.story_id not in sections or not 1 <= row.zone <= len(sections[row.story_id]):
                raise ValueError(f"Invalid {dataset} item {(row.story_id, row.zone)}")
        responses[measure] = items
    return sections, responses


def _tasks(dataset: str) -> list[tuple[str, str, str]]:
    root = ROOT / "outputs_jeanzay" / dataset / "embedding_sweep"
    tasks = []
    filename = f"{dataset}_embed_checkpoint_scores.csv"
    for family in LPP_FAMILY_NAMES:
        for task in flatten_family_tasks(family):
            model_dir = root / "fp_average" / family / task.model_dir_safe
            path = model_dir / filename
            if path.exists() and task.revision in set(pd.read_csv(path, usecols=["revision"]).revision.astype(str)):
                tasks.append((family, task.model_id, task.revision))
    return list(dict.fromkeys(tasks))


def _cache_paths(dataset: str, model_id: str, revision: str, sections: dict[int, list[str]]):
    root = CACHE_ROOT / f"{dataset}_embed_features"
    safe = model_id.replace("/", "_")
    rev = revision.replace("/", "_").replace("-", "_")
    candidates = list(root.glob(f"{safe}__{rev}__en__sec*__all.npz"))
    paths = {}
    for path in candidates:
        match = re.search(r"__sec(\d+)__", path.name)
        if match:
            section = int(match.group(1))
            # Prefer the full-text cache used by the benchmark when both modes exist.
            if section not in paths or "__fulltext_" in path.name:
                paths[section] = path
    missing = set(sections) - paths.keys()
    return None if missing else paths


STANDARDIZE = False


def _load_states(paths: dict[int, Path], sections: dict[int, list[str]]):
    states = {}
    for section, words in sorted(sections.items()):
        with np.load(paths[section]) as archive:
            features = archive["features"]
            if features.ndim != 3 or len(features) != len(words):
                raise ValueError(f"Cache word count/shape mismatch: {paths[section]}")
            states[section] = np.asarray(features[:, -1, :], dtype=np.float32)
    if STANDARDIZE:  # per-feature z-score over every cached word (Timkey & van Schijndel 2021)
        allw = np.concatenate(list(states.values()))
        mu, sd = allw.mean(0), allw.std(0) + 1e-8
        states = {k: (v - mu) / sd for k, v in states.items()}
    widths = {values.shape[1] for values in states.values()}
    if len(widths) != 1 or not all(np.isfinite(values).all() for values in states.values()):
        raise ValueError("Inconsistent or nonfinite last-layer cached states")
    return states


def _score_model(task, dataset: str, sections, responses, response_distances, output_root: Path):
    family, model_id, revision = task
    paths = _cache_paths(dataset, model_id, revision, sections)
    if paths is None:
        return f"missing cache {model_id}@{revision}"
    if all((output_root / measure.lower() / family / safe_model_dir(model_id) /
            f"rsa_{revision.replace('/', '_')}.csv").exists() for measure in MEASURES):
        return f"existing {model_id}@{revision}"
    states = _load_states(paths, sections)
    model_rdm_cache = {}
    for measure, items in responses.items():
        output = (output_root / measure.lower() / family / safe_model_dir(model_id) /
                  f"rsa_{revision.replace('/', '_')}.csv")
        if output.exists():
            continue
        item_key = tuple(zip(items.story_id.to_numpy(), items.zone.to_numpy()))
        if item_key not in model_rdm_cache:
            matrix = np.stack([states[int(story)][int(zone) - 1] for story, zone in item_key])
            model_rdm_cache[item_key] = pdist(matrix, metric="correlation")
        model_distances = model_rdm_cache[item_key]
        rt_distances = response_distances[measure]
        if not np.isfinite(model_distances).all() or not np.isfinite(rt_distances).all():
            raise ValueError(f"Undefined RDM for {dataset}/{measure}/{model_id}")
        score = float(pearsonr(model_distances, rt_distances).statistic)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(f"{output.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        pd.DataFrame([{"family": family, "model_id": model_id, "revision": revision,
                       "measure": measure, "n_items": len(items), "n_pairs": len(rt_distances),
                       "rsa_pearson": score}]).to_csv(temp, index=False)
        os.replace(temp, output)
    return f"scored {model_id}@{revision}"


def _summarize(dataset: str, output_root: Path):
    from matplotlib import pyplot as plt

    encoding_root = ROOT / "outputs_jeanzay" / dataset / "embedding_sweep"
    for measure in MEASURES:
        measure_root = output_root / measure.lower()
        rows = []
        for path in measure_root.glob("*/*/rsa_*.csv"):
            rsa = pd.read_csv(path).iloc[0]
            score_path = (encoding_root / f"{measure.lower()}_average" / rsa.family /
                          safe_model_dir(rsa.model_id) / f"{dataset}_embed_checkpoint_scores.csv")
            if not score_path.exists():
                continue
            encoding = pd.read_csv(score_path)
            encoding = encoding.loc[encoding.revision == rsa.revision]
            if encoding.empty:
                continue
            last = encoding.loc[encoding.layer == encoding.layer.max()].iloc[0]
            meta = lookup_model_meta(rsa.model_id, rsa.family)
            rows.append({"family": rsa.family, "model_id": rsa.model_id,
                         "revision": rsa.revision, "n_items": int(rsa.n_items),
                         "rsa_pearson": rsa.rsa_pearson, "encoding_pearson": last.pearsonr,
                         "group": meta.group, "n_params": meta.n_params})
        if not rows:
            print(f"{dataset}/{measure}: no paired scores", flush=True)
            continue
        summary = pd.DataFrame(rows).sort_values(["family", "model_id", "revision"])
        measure_root.mkdir(parents=True, exist_ok=True)
        summary.to_csv(measure_root / "model_summary.csv", index=False)
        across = (float(pearsonr(summary.encoding_pearson, summary.rsa_pearson).statistic)
                  if len(summary) >= 2 else float('nan'))
        print(f"{dataset}/{measure}: {len(summary)} models, encoding vs RSA r={across:.4f}", flush=True)
        valid = summary.loc[np.isfinite(summary.n_params) & (summary.n_params > 0)].copy()
        valid["log10_params"] = np.log10(valid.n_params)
        scaling = []
        scopes = [('all', 'all models', valid)]
        scopes += [('group', name, part) for name, part in valid.groupby('group')]
        scopes += [('family', name, part) for name, part in valid.groupby('family')]
        for scope, name, part in scopes:
            if len(part) < 3 or part.log10_params.nunique() < 3:
                continue
            for metric in ('rsa_pearson', 'encoding_pearson'):
                fit = linregress(part.log10_params, part[metric])
                scaling.append({'scope': scope, 'name': name, 'metric': metric,
                                'n_models': len(part), 'slope_per_log10_param': fit.slope,
                                'pearson_r': fit.rvalue})
        repeated = valid.groupby('family').filter(lambda part: part.log10_params.nunique() > 1)
        if len(repeated) >= 3:
            x = repeated.log10_params - repeated.groupby('family').log10_params.transform('mean')
            for metric in ('rsa_pearson', 'encoding_pearson'):
                y = repeated[metric] - repeated.groupby('family')[metric].transform('mean')
                scaling.append({'scope': 'within_family', 'name': 'family-centered', 'metric': metric,
                                'n_models': len(repeated), 'slope_per_log10_param': float((x*y).sum()/(x*x).sum()),
                                'pearson_r': float(pearsonr(x, y).statistic)})
        pd.DataFrame(scaling).to_csv(measure_root / "scaling_summary.csv", index=False)

        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for group in GROUP_ORDER:
            part = summary.loc[summary.group == group]
            if part.empty:
                continue
            color = GROUP_COLORS[group]
            axes[0].scatter(part.encoding_pearson, part.rsa_pearson, color=color,
                            label=GROUP_DISPLAY[group], s=30, alpha=.7)
            axes[1].scatter(part.n_params, part.rsa_pearson, color=color, s=30, alpha=.7)
        axes[0].set(xlabel='Last-layer ridge encoding (Pearson r)',
                    ylabel='Behavioral RSA (Pearson r)', title=f'Across-model r = {across:.2f}')
        axes[1].set_xscale('log')
        axes[1].set(xlabel='Model parameters', ylabel='Behavioral RSA (Pearson r)')
        axes[0].legend(frameon=False, fontsize=7)
        figure.suptitle(f'{dataset.upper()} {measure} · {len(summary)} models')
        figure.tight_layout()
        figure.savefig(measure_root / 'rsa_summary.png', dpi=180)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["provo", "meco"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--standardize-features", action="store_true")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()
    global STANDARDIZE
    STANDARDIZE = args.standardize_features
    output_root = args.out_root or ROOT / "outputs" / args.dataset / "pearson_rsa"
    if args.summarize_only:
        _summarize(args.dataset, output_root)
        return
    sections, responses = _prepare(args.dataset)
    response_distances = {measure: pdist(items.rt.to_numpy(dtype=float)[:, None], metric='euclidean')
                          for measure, items in responses.items()}
    tasks = _tasks(args.dataset)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{args.dataset}: {len(tasks)} scored models; " +
          ", ".join(f"{m}={len(t)} items" for m, t in responses.items()), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_score_model, task, args.dataset, sections, responses,
                               response_distances, output_root)
                   for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    _summarize(args.dataset, output_root)


if __name__ == "__main__":
    main()
