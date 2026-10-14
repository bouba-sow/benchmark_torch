#!/usr/bin/env python3
"""Last-layer Pearson RSA for Tuckute B sentences and Fedorenko 2016 words.

Reads existing embedding caches only. Example:
  python scripts/analysis/other_neural_rsa.py tuckute --workers 8
  python scripts/analysis/other_neural_rsa.py fedorenko2016 --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = "/tmp/neural-rsa-matplotlib"

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

from neuroai.analysis.benchmark_common import filter_registry_models, EXCLUDED_FAMILY_DIRS
from neuroai.analysis.pereira_rsa import pearson_rdm
from neuroai.analysis.benchmark_common import (
    GROUP_COLORS, GROUP_DISPLAY, GROUP_ORDER, lookup_model_meta,
)
from neuroai.datasets.fedorenko2016 import Fedorenko2016Dataset
from neuroai.datasets.tuckute import TuckuteDataset
from neuroai.models.registry import LPP_FAMILY_NAMES, flatten_family_tasks
from neuroai.sweep import safe_model_dir


def _tasks(dataset: str, encoding_root: Path) -> list[tuple[str, str, str]]:
    score_file = "tuckute_by_roi.csv" if dataset == "tuckute" else "fedorenko2016_by_electrode.csv"
    tasks = []
    for family in LPP_FAMILY_NAMES:
        if family in EXCLUDED_FAMILY_DIRS:
            continue
        for task in flatten_family_tasks(family):
            if dataset == "tuckute" and (encoding_root / ".embed_cache" / task.model_dir_safe /
                                          task.revision / "embeddings.npz").exists():
                tasks.append((family, task.model_id, task.revision))
                continue
            path = encoding_root / family / task.model_dir_safe / score_file
            if not path.exists():
                continue
            revisions = pd.read_csv(path, usecols=["revision"]).revision.astype(str)
            if task.revision in set(revisions):
                tasks.append((family, task.model_id, task.revision))
    return list(dict.fromkeys(tasks))


def _prepare_tuckute(data_root: Path):
    ds = TuckuteDataset(data_root)
    all_sentences, _, all_ids = ds.get_condition("all")
    selected = np.flatnonzero(ds.condition_mask("B"))
    id_to_index = {int(item): i for i, item in enumerate(all_ids)}
    neural = []
    for subject, (sentences, bold, ids) in ds.get_condition_per_subject("B").items():
        if len(ids) != len(set(ids)) or len(ids) != len(sentences):
            raise ValueError(f"Invalid Tuckute stimulus IDs for subject {subject}")
        valid = np.isfinite(bold).all(axis=1)
        bold, ids = bold[valid], ids[valid]
        if len(ids) < 3:
            raise ValueError(f"Too few valid Tuckute responses for subject {subject}")
        indices = np.array([id_to_index[int(item)] for item in ids])
        if not np.isin(indices, selected).all():
            raise ValueError(f"Subject {subject} has non-B sentences")
        neural.append((str(subject), indices, pearson_rdm(bold)))
    return all_sentences, selected, neural


def _prepare_fedorenko(data_root: Path):
    ds = Fedorenko2016Dataset(data_root)
    words = ds.load_word_table()
    electrodes = ds.load_electrodes()
    sentence_texts = [" ".join(group.word.astype(str)) for _, group in
                      words.sort_values(["sentence_id", "word_num"]).groupby("sentence_id")]
    ordered = words.sort_values(["sentence_id", "word_num"])
    if not np.array_equal(ordered.stimulus_id.to_numpy(), words.stimulus_id.to_numpy()):
        raise ValueError("Fedorenko cache word order differs from neural table")
    neural = []
    for subject, group in electrodes.groupby("subject_UID"):
        columns = [f"electrode_{int(e)}" for e in group.electrode if f"electrode_{int(e)}" in words]
        values = words[columns].to_numpy(dtype=float)
        keep = np.isfinite(values).all(axis=0) & (np.std(values, axis=0) > 1e-8)
        values = values[:, keep]
        if values.shape[1] < 2:
            continue
        values = (values - values.mean(axis=0)) / values.std(axis=0)
        neural.append((str(subject), np.arange(len(words)), pearson_rdm(values)))
    return sentence_texts, neural, len(words)


def _load_tuckute(path: Path, sentences: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if data["sentences"].tolist() != sentences:
            raise ValueError(f"Cache sentence mismatch: {path}")
        states = np.asarray(data["embeddings"][:, -1, :], dtype=np.float32)
    return states


def _load_fedorenko(path: Path, sentences: list[str], n_words: int) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if data["sentences"].tolist() != sentences:
            raise ValueError(f"Cache sentence mismatch: {path}")
        states = np.asarray(data["word_embeddings"][:, -1, :], dtype=np.float32)
    if len(states) != n_words:
        raise ValueError(f"Cache word count mismatch: {path}")
    return states


def _encoding_score(dataset: str, path: Path, revision: str) -> float:
    if dataset == "tuckute":
        scores = pd.read_csv(path / "tuckute_by_roi.csv")
        scores = scores.loc[(scores.revision == revision) & (scores.cond == "B")]
        scores = scores.loc[scores.layer == scores.layer.max()]
        if scores.roi.nunique() != 6:
            raise ValueError(f"Expected six Tuckute ROIs: {path}")
        return float(scores.pearsonr.mean())
    scores = pd.read_csv(path / "fedorenko2016_by_electrode.csv")
    scores = scores.loc[scores.revision == revision]
    return float(scores.pearsonr.mean())


def _encoding_subject_score(path: Path, revision: str) -> float:
    """Fedorenko sensitivity check giving each of the five subjects equal weight."""
    scores = pd.read_csv(path / "fedorenko2016_by_subject.csv")
    scores = scores.loc[scores.revision == revision]
    if scores.subject_uid.nunique() != 5:
        raise ValueError(f"Expected five Fedorenko subjects: {path}")
    return float(scores.pearsonr_mean.mean())


def _standardize(states: np.ndarray) -> np.ndarray:
    """z-score every feature across stimuli (unsupervised; uses stimuli only, never the neural data).

    Last-layer states are anisotropic: a few high-variance dimensions dominate the 1 - Pearson RDM.
    Standardising removes that.
    """
    return (states - states.mean(0)) / (states.std(0) + 1e-8)


def _score_model(task, dataset, encoding_root, cache_root, output_root, context, standardize=False):
    family, model_id, revision = task
    model_dir = safe_model_dir(model_id)
    output = output_root / family / model_dir / f"rsa_{revision.replace('/', '_')}.csv"
    if output.exists():
        return f"existing {model_id}@{revision}"
    if dataset == "tuckute":
        sentences, selected, neural = context
        path = cache_root / model_dir / revision.replace("/", "_") / "embeddings.npz"
        if not path.exists():
            return f"missing cache {model_id}@{revision}"
        states = _load_tuckute(path, sentences)
        if not np.isfinite(states).all():
            raise ValueError(f"Nonfinite embeddings: {path}")
        # Compute only the common baseline subset; other conditions are sparse.
        chosen = states[selected]
        rdm = pearson_rdm(_standardize(chosen) if standardize else chosen)
        index = {int(i): k for k, i in enumerate(selected)}
        neural = [(subject, np.array([index[int(i)] for i in indices]), distances)
                  for subject, indices, distances in neural]
    else:
        sentences, neural, n_words = context
        path_dir = cache_root / model_dir / revision
        paths = list(path_dir.glob("fedorenko2016_*.npz"))
        if len(paths) != 1:
            return f"missing/ambiguous cache {model_id}@{revision}"
        states = _load_fedorenko(paths[0], sentences, n_words)
        if not np.isfinite(states).all():
            raise ValueError(f"Nonfinite embeddings: {paths[0]}")
        rdm = pearson_rdm(_standardize(states) if standardize else states)
    rows = []
    for subject, indices, neural_rdm in neural:
        triangle = np.triu_indices(len(indices), k=1)
        model_pairs = rdm[np.ix_(indices, indices)][triangle]
        neural_pairs = neural_rdm[triangle]
        rows.append({"family": family, "model_id": model_id, "revision": revision,
                     "subject": subject, "n_stimuli": len(indices), "n_pairs": len(model_pairs),
                     "rsa_pearson": float(pearsonr(model_pairs, neural_pairs).statistic)})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, output)
    return f"scored {model_id}@{revision}: {np.mean([r['rsa_pearson'] for r in rows]):.4f}"


def _summarize(dataset: str, encoding_root: Path, output_root: Path):
    rows = []
    for path in output_root.glob("*/*/rsa_*.csv"):
        rsa = pd.read_csv(path, dtype={"subject": str})
        if rsa.empty:
            continue
        first = rsa.iloc[0]
        try:
            encoding = _encoding_score(dataset, encoding_root / first.family /
                                       safe_model_dir(first.model_id), first.revision)
        except (FileNotFoundError, ValueError):
            encoding = float("nan")
        meta = lookup_model_meta(first.model_id, first.family)
        rows.append({"family": first.family, "model_id": first.model_id,
                     "revision": first.revision, "n_subjects": rsa.subject.nunique(),
                     "rsa_pearson": rsa.rsa_pearson.mean(), "encoding_pearson": encoding,
                     "group": meta.group, "n_params": meta.n_params})
        if dataset == 'fedorenko2016':
            rows[-1]['encoding_subject_mean'] = _encoding_subject_score(
                encoding_root / first.family / safe_model_dir(first.model_id), first.revision)
    if not rows:
        print("No paired RSA/encoding scores yet", flush=True)
        return
    summary = filter_registry_models(pd.DataFrame(rows)).sort_values(["family", "model_id", "revision"])
    summary.to_csv(output_root / "model_summary.csv", index=False)
    paired = summary.loc[np.isfinite(summary.encoding_pearson) & np.isfinite(summary.rsa_pearson)]
    if len(paired) >= 3:
        across = pearsonr(paired.encoding_pearson, paired.rsa_pearson).statistic
        print(f"{dataset}: {len(summary)} models; encoding vs RSA r={across:.4f} (descriptive)", flush=True)
        if dataset == 'fedorenko2016':
            subject_across = pearsonr(summary.encoding_subject_mean, summary.rsa_pearson).statistic
            print(f"{dataset}: equal-subject encoding vs RSA r={subject_across:.4f} (sensitivity)", flush=True)
        _plot_summary(dataset, summary, across, output_root)
    size = summary.loc[np.isfinite(summary.n_params) & (summary.n_params > 0)].copy()
    size["log10_params"] = np.log10(size.n_params)
    scaling = []
    for scope, name, subset in ([('all', 'all models', size)] +
                                [('group', name, part) for name, part in size.groupby('group')] +
                                [('family', name, part) for name, part in size.groupby('family')]):
        if len(subset) < 3 or subset.log10_params.nunique() < 3:
            continue
        for metric in ('rsa_pearson', 'encoding_pearson'):
            finite = subset.loc[np.isfinite(subset[metric])]
            if len(finite) < 3 or finite.log10_params.nunique() < 3:
                continue
            fit = linregress(finite.log10_params, finite[metric])
            scaling.append({'scope': scope, 'name': name, 'metric': metric,
                            'n_models': len(finite), 'slope_per_log10_param': fit.slope,
                            'pearson_r': fit.rvalue})
    repeated = size.groupby('family').filter(lambda part: part.log10_params.nunique() > 1)
    if len(repeated) >= 3:
        centered_size = repeated.log10_params - repeated.groupby('family').log10_params.transform('mean')
        for metric in ('rsa_pearson', 'encoding_pearson'):
            finite = repeated.loc[np.isfinite(repeated[metric])]
            centered_size = finite.log10_params - finite.groupby('family').log10_params.transform('mean')
            if len(finite) < 3 or (centered_size ** 2).sum() == 0:
                continue
            centered_score = finite[metric] - finite.groupby('family')[metric].transform('mean')
            slope = float((centered_size * centered_score).sum() / (centered_size ** 2).sum())
            scaling.append({'scope': 'within_family', 'name': 'family-centered', 'metric': metric,
                            'n_models': len(finite), 'slope_per_log10_param': slope,
                            'pearson_r': float(pearsonr(centered_size, centered_score).statistic)})
    pd.DataFrame(scaling).to_csv(output_root / "scaling_summary.csv", index=False)


def _plot_summary(dataset: str, summary: pd.DataFrame, across: float, output_root: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    for group in GROUP_ORDER:
        part = summary.loc[summary.group == group]
        if not part.empty:
            axis.scatter(part.encoding_pearson, part.rsa_pearson, s=38, alpha=.75,
                         color=GROUP_COLORS[group], label=GROUP_DISPLAY[group])
    axis.set(xlabel='Last-layer ridge encoding (Pearson r)',
             ylabel='Last-layer fit-free RSA (Pearson r)',
             title=f'{dataset} · {len(summary)} RSA / {np.isfinite(summary.encoding_pearson).sum()} paired · r = {across:.2f}')
    axis.legend(frameon=False, fontsize=8)
    axis.spines[['top', 'right']].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_root / 'encoding_vs_rsa.png', dpi=180)
    plt.close(figure)

    valid = summary.loc[np.isfinite(summary.n_params) & (summary.n_params > 0)]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    for axis, metric, ylabel in zip(axes, ('rsa_pearson', 'encoding_pearson'),
                                    ('Pearson RSA score', 'Ridge encoding Pearson r')):
        for group in GROUP_ORDER:
            part = valid.loc[valid.group == group]
            if not part.empty:
                axis.scatter(part.n_params, part[metric], s=30, alpha=.7,
                             color=GROUP_COLORS[group], label=GROUP_DISPLAY[group])
        axis.set_xscale('log')
        axis.set(xlabel='Model parameters', ylabel=ylabel)
        axis.spines[['top', 'right']].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_root / 'rsa_scaling.png', dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["tuckute", "fedorenko2016"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--standardize-features", action="store_true",
                        help="z-score each embedding feature across stimuli before the model RDM")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="default outputs/<dataset>/pearson_rsa; use a scratch dir to avoid mixing with old scores")
    args = parser.parse_args()
    encoding_root = ROOT / "outputs_jeanzay" / args.dataset / "embedding_sweep"
    output_root = args.out_root or ROOT / "outputs" / args.dataset / "pearson_rsa"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        _summarize(args.dataset, encoding_root, output_root)
        return
    tasks = _tasks(args.dataset, encoding_root)
    if args.limit:
        tasks = tasks[:args.limit]
    if args.dataset == "tuckute":
        context = _prepare_tuckute(ROOT / "data/tuckute_fmri/osfstorage")
        cache_root = encoding_root / ".embed_cache"
    else:
        context = _prepare_fedorenko(ROOT / "data/fedorenko2016")
        cache_root = Path("/lustre/fsn1/projects/rech/ldh/una68ug/.cache/fedorenko2016_features")
    print(f"{args.dataset}: {len(tasks)} scored registry tasks; {len(context[-1] if args.dataset == 'tuckute' else context[1])} neural RDMs", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_score_model, task, args.dataset, encoding_root,
                               cache_root, output_root, context,
                               args.standardize_features) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    _summarize(args.dataset, encoding_root, output_root)


if __name__ == "__main__":
    main()
