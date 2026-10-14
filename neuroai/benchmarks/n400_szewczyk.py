"""Szewczyk & Federmeier N400 EEG encoding benchmark: LLM activations → N400.

Unlike Nieuwland, this deposit ships a single scalar N400
amplitude per (subject, item) trial (already averaged over 4 centro-parietal
ROI clusters in the original preprocessing) — there is no per-electrode
spatial dimension to regress against. The benchmark is therefore a per-subject
*single-output* ridge regression rather than a multi-electrode one.

GPU strategy
------------
A true single-shot batched fit across all ~138 subjects is not valid here:
subjects only respond to a Latin-square subset of the ~2,300 items, so they
cannot be stacked as columns of one shared-X ridge problem without a masked/
weighted formulation. Instead, every (subject, layer) pair is treated as one
independent task (~138 subjects x ~n_layers, e.g. ~4,000 tasks per checkpoint)
and all of them are submitted to a single flat ``ThreadPoolExecutor``, each
running ``cv_ridge_score_multioutput_gpu`` (SVD/GCV-based — alpha is chosen
analytically per fold, no nested CV grid search) on its own CUDA stream. Each
fit is tiny (<=300 rows), so a single H100 can run dozens of these
concurrently — this gets near-"all subjects at once" GPU utilisation without
requiring invalid batched math.

Pipeline
--------
  1. Load bind-5-dset_300-500.rds -> ~2,300 unique (subexperiment, item,
     critical_word) items (continuous cloze-probability manipulation, no
     discrete conditions).
  2. Extract LLM hidden states at the critical word for all items (reuses
     ``extract_sentence_features`` from the Nieuwland/eeg_erp benchmark).
  3. Flatten (subject, layer) into one task queue; fit single-output ridge
     (GroupKFold by (subexperiment, item) frame) for every task concurrently.
  4. Per subject: best layer = argmax Spearman r across layers. Two cheap
     GroupKFold+RidgeCV OOF-prediction passes (best layer, last layer; CPU,
     sklearn) feed the secondary cloze-based diagnostics.
  5. Pool OOF predictions across subjects, then report:
       - primary: per-subject Spearman r (predicted vs actual N400) -> mean+-SEM
       - secondary (continuous): pooled Spearman(predicted N400, cloze_p) and
         Spearman(actual N400, cloze_p), as a sanity check on effect validity
       - secondary (median-split, for scaling-law plots): pooled Spearman
         (predicted vs actual) restricted to cloze_p >= 0.5 ("expected") and
         cloze_p < 0.5 ("unexpected") separately.

Outputs (written to out_dir)
----------------------------
  szewczyk_checkpoint_scores.csv     — one row per checkpoint
  szewczyk_checkpoint_layerwise.csv  — one row per (checkpoint, layer)
  szewczyk_by_subject.csv            — one row per (checkpoint, subject)

Reference
---------
Szewczyk, J. M., & Federmeier, K. D. (2021). eLife. OSF: https://osf.io/w96nc
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import threading
import typing as tp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr as _scipy_pearsonr
from scipy.stats import spearmanr as _scipy_spearmanr
from sklearn.linear_model import RidgeCV as _SkRidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from ..benchmarks.eeg_erp import extract_sentence_features
from ..datasets.szewczyk import (
    build_items_df,
    build_subject_items,
    get_subjects,
    load_data,
)
from ..regressors.ridge import CVScore, cv_ridge_score_multioutput_gpu

logger = logging.getLogger(__name__)
_csv_lock = threading.Lock()
_fit_cache_lock = threading.Lock()


def _fit_cache_path(fit_cache_dir: str | Path, model_name: str, revision: str | None) -> Path:
    safe_model = model_name.replace("/", "_")
    safe_rev = (revision or "main").replace("/", "_")
    from ..features.llm import embedding_cache_tag

    return Path(fit_cache_dir) / f"szewczyk_fit_cache_{safe_model}_{safe_rev}{embedding_cache_tag(model_name)}.jsonl"


def _load_fit_cache(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    """Reload a JSONL fit cache: {(subject, layer): (spearmanr, pearsonr)}.

    Tolerant of a truncated last line (SLURM SIGKILL at the wall-time limit
    can cut the process off mid-write).
    """
    out: dict[tuple[str, int], tuple[float, float]] = {}
    if not path.is_file():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(rec["subject"], int(rec["layer"]))] = (rec["spearmanr"], rec["pearsonr"])
    return out


def _append_fit_cache(path: Path, subject: str, layer: int, spearmanr: float, pearsonr: float) -> None:
    rec = json.dumps({"subject": subject, "layer": layer,
                       "spearmanr": spearmanr, "pearsonr": pearsonr})
    with _fit_cache_lock:
        with open(path, "a") as f:
            f.write(rec + "\n")
            f.flush()
            os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(_scipy_spearmanr(a, b).statistic)


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(_scipy_pearsonr(a, b).statistic)


def _csv_append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import csv
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _szewczyk_cpu_single_output_score(
    X: np.ndarray,       # (n_sub, D)
    y: np.ndarray,       # (n_sub,)
    groups: np.ndarray,
    n_splits: int,
    *,
    ridge_alphas,
    scale: bool,
    standardize_target: bool,
) -> CVScore:
    """CPU counterpart to ``cv_ridge_score_multioutput_gpu`` for n_targets=1.

    Module-level (not a method) so it can be called from
    :func:`_szewczyk_fit_subject_all_layers`, which itself must be
    module-level/picklable to run under ``ProcessPoolExecutor``.

    Deliberately does NOT call ``cv_ridge_score_multioutput`` (ridge.py):
    that helper's ``_fit_ridge_for_fold`` picks ``RidgeCV(cv=<int>)`` for
    large-enough folds, which routes through sklearn's ``GridSearchCV`` —
    and that path silently returns 1D predictions for a single-column y
    while the caller still expects 2D, corrupting the per-target
    correlation helpers (a shape (n,) vs (1,) broadcast bug that never
    surfaced for the earlier N400 datasets since they always have >1 electrode
    targets). Using ``RidgeCV``'s default ``cv=None`` (GCV) here — same as
    ``SzewczykEncoder._oof_predictions`` — sidesteps that entirely.
    """
    gkf = GroupKFold(n_splits=n_splits)
    alphas = np.array(ridge_alphas)
    pr_folds: list[float] = []
    sr_folds: list[float] = []
    alpha_folds: list[float] = []

    for tr_idx, te_idx in gkf.split(X, groups=groups):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if scale:
            sc = RobustScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        if standardize_target:
            y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
            y_std = y_std if y_std > 1e-8 else 1.0
            y_tr_fit = (y_tr - y_mean) / y_std
        else:
            y_tr_fit, y_mean, y_std = y_tr, 0.0, 1.0
        model = _SkRidgeCV(alphas=alphas)
        model.fit(X_tr, y_tr_fit)
        alpha_folds.append(float(model.alpha_))
        y_pred = model.predict(X_te) * y_std + y_mean
        pr_folds.append(_safe_pearsonr(y_te, y_pred))
        sr_folds.append(_safe_spearmanr(y_te, y_pred))

    return CVScore(
        pearsonr=float(np.nanmean(pr_folds)),
        spearmanr=float(np.nanmean(sr_folds)),
        r2=float("nan"),
        mse=float("nan"),
        ridge_alpha=float(np.nanmedian(alpha_folds)) if alpha_folds else float("nan"),
    )


# Shared feature array for CPU worker processes, set once in the parent
# process (see SzewczykEncoder.run) right before spawning a ``fork``-context
# ProcessPoolExecutor -- fork's copy-on-write semantics make this a
# zero-copy handoff to every worker (no per-task or per-worker pickling of
# the full (n_items, n_layers, D) array), unlike passing it as a submit()
# argument every call.
_SZEWCZYK_WORKER_FEATURES: np.ndarray | None = None


def _szewczyk_fit_subject_all_layers(
    subj: str,
    idx: np.ndarray,
    y_col: np.ndarray,
    groups: np.ndarray,
    *,
    n_layers: int,
    n_splits: int,
    ridge_alphas,
    scale: bool,
    standardize_target: bool,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Fit the CPU ridge for every layer of one subject, in one process.

    Module-level (picklable) so it can run as a ``ProcessPoolExecutor`` task.
    Real multiprocessing (not ``ThreadPoolExecutor``) is required here: each
    individual fit is tiny (<=300 rows), and profiling showed the GIL isn't
    released enough during sklearn's small-matrix RobustScaler/RidgeCV calls
    for threads to get real parallelism -- 32 "concurrent" threads ended up
    serializing against the GIL with heavy context-switch overhead, an effect
    that gets worse with hidden dim (RobustScaler's per-column stats scale
    with D). Parallelizing at the SUBJECT level (one process per subject, all
    of that subject's layers fit sequentially within the process) instead of
    per (subject, layer) keeps the number of processes bounded to
    ``len(subjects)`` (~138) rather than needing one per (subject, layer)
    task (~3,450+) -- mirrors ``geco_embed.py::_fit_subject_layers``.

    Reads the shared item-embedding array from the module-level
    ``_SZEWCZYK_WORKER_FEATURES`` (set via fork inheritance, not passed as an
    argument) to avoid pickling a large per-subject-per-layer slice on every
    task dispatch.
    """
    features = _SZEWCZYK_WORKER_FEATURES
    assert features is not None, (
        "_SZEWCZYK_WORKER_FEATURES not set -- worker process wasn't forked "
        "from a parent that set it, or the pool's mp_context isn't 'fork'"
    )
    n_splits_eff = min(n_splits, len(np.unique(groups)))
    sr = np.full(n_layers, np.nan, dtype=np.float32)
    pr = np.full(n_layers, np.nan, dtype=np.float32)
    for li in range(n_layers):
        X_li = features[idx, li, :].astype(np.float32)
        score = _szewczyk_cpu_single_output_score(
            X_li, y_col, groups, n_splits_eff,
            ridge_alphas=ridge_alphas, scale=scale,
            standardize_target=standardize_target,
        )
        sr[li] = score.spearmanr
        pr[li] = score.pearsonr
    return subj, sr, pr


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SzewczykConfig:
    """Full configuration for one Szewczyk checkpoint run."""

    data_path: str | Path = "data/szewczyk/osfstorage/bind-5-dset_300-500.rds"

    # LLM ─────────────────────────────────────────────────────────────────────
    hf_model_name: str = "EleutherAI/pythia-1b"
    hf_revision: str | None = None
    hf_device: str = "cpu"
    hf_batch_size: int = 8
    hf_max_length: int | None = None
    layer_mode: tp.Union[tp.Literal["last", "all"], int] = "all"
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    feature_cache_dir: str | Path | None = None
    # Per-(subject, layer) ridge-fit cache (GPU path only). The flattened
    # sweep is ~4,000-7,000 independent fits and can run past a SLURM wall
    # limit on the biggest checkpoints (see 2026-09-24 szewczyk timeout on
    # MoLM-700M-8B, 49 layers -- lost the *entire* sweep since only the
    # LLM-feature cache survived). Each fit result is appended to a JSONL
    # file as soon as it completes; a restart reloads it and only re-submits
    # (subject, layer) pairs missing from it. None disables the cache
    # (matches old behaviour).
    fit_cache_dir: str | Path | None = None

    # Regression ──────────────────────────────────────────────────────────────
    n_splits: int = 10                  # GroupKFold folds (grouped by frame)
    ridge_alphas: list[float] = field(
        default_factory=lambda: np.logspace(3.5, 8, 5).tolist()
    )
    scale: bool = True
    standardize_target: bool = True
    regression_device: str = "cpu"      # "cuda" -> GPU SVD/GCV path
    n_gpu_workers: int = 32             # flattened (subject, layer) concurrent workers
    expected_cloze_threshold: float = 0.5

    # Output ──────────────────────────────────────────────────────────────────
    out_dir: str | Path = "outputs/szewczyk/embedding_sweep"
    n_params: float = 0.0               # informational only (filled by sweep)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SzewczykSubjectResult:
    subject: str
    n_items: int
    best_layer: int
    last_layer: int
    mean_spearmanr: float
    mean_pearsonr: float
    last_layer_mean_spearmanr: float
    last_layer_mean_pearsonr: float
    layer_spearmanr: list[float]
    layer_pearsonr: list[float]


@dataclass
class SzewczykResult:
    model_name: str
    revision: str | None
    n_layers: int
    n_subjects: int
    subject_results: list[SzewczykSubjectResult]

    # Primary: per-subject Spearman/Pearson at best layer, mean +- SEM
    mean_spearmanr: float
    sem_spearmanr: float
    mean_pearsonr: float
    last_layer_mean_spearmanr: float
    last_layer_sem_spearmanr: float
    last_layer_mean_pearsonr: float
    layer_spearmanr: list[float]
    layer_pearsonr: list[float]

    # Secondary (continuous, pooled OOF across subjects)
    cloze_corr_pred_best: float
    cloze_corr_pred_last: float
    cloze_corr_actual: float

    # Secondary (median-split, pooled OOF across subjects) — feeds the two
    # "expected vs unexpected" scaling-law plots.
    expected_spearmanr_best: float
    unexpected_spearmanr_best: float
    expected_spearmanr_last: float
    unexpected_spearmanr_last: float
    n_expected_items: int
    n_unexpected_items: int


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class SzewczykEncoder:
    """Encode Szewczyk & Federmeier N400 items with LLM hidden states via ridge.

    Usage
    -----
    cfg    = SzewczykConfig(data_path="...", hf_model_name="...", ...)
    result = SzewczykEncoder(cfg).run()
    """

    def __init__(self, cfg: SzewczykConfig) -> None:
        self.cfg = cfg

    def run(self) -> SzewczykResult:
        cfg = self.cfg
        logger.info(
            "SzewczykEncoder: %s (rev=%s)", cfg.hf_model_name, cfg.hf_revision or "main"
        )

        # ── Load data ─────────────────────────────────────────────────────────
        df    = load_data(cfg.data_path)
        items = build_items_df(df)
        subjects = get_subjects(df)

        # ── Extract LLM features for all items (done once) ───────────────────
        features = self._extract_features(items)
        layerwise = features.ndim == 3
        n_layers  = features.shape[1] if layerwise else 1
        if not layerwise:
            features = features[:, np.newaxis, :]

        last_layer = n_layers - 1

        # ── Per-subject data, pre-fetched once ────────────────────────────────
        min_items = cfg.n_splits * 2
        subj_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for subj in subjects:
            idx, Y, groups, cloze = build_subject_items(df, subj, items)
            n_groups = len(np.unique(groups))
            if len(idx) < min_items or n_groups < 2:
                logger.warning("Szewczyk: subject %s has only %d items (%d frames) — skipping",
                               subj, len(idx), n_groups)
                continue
            subj_data[subj] = (idx, Y, groups, cloze)

        # ── GPU backend ────────────────────────────────────────────────────────
        # cv_ridge_score_multioutput_gpu is a pure-torch SVD/GCV implementation
        # (no himalaya dependency) — only torch.cuda availability matters here.
        use_gpu = cfg.regression_device != "cpu"
        if use_gpu:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA unavailable — using CPU")
                use_gpu = False
            else:
                # torch.linalg.svd (used inside cv_ridge_score_multioutput_gpu)
                # does a one-time, non-thread-safe CUDA/MAGMA backend
                # registration on its first call in the process. If two
                # (subject, layer) worker threads race to trigger that first
                # call concurrently, PyTorch raises "RuntimeError: lazy
                # wrapper should be called at most once" and that fit is lost
                # (same fix as podcast.py / broderick.py's eigh-based
                # GPU-batched regression). Force it here, single threaded,
                # before spawning the pool below.
                _svd_warmup = torch.eye(2, device=f"cuda:{torch.cuda.current_device()}")
                torch.linalg.svd(_svd_warmup, full_matrices=False)
                del _svd_warmup

        # ── Flatten (subject, layer) into one task queue ──────────────────────
        subj_layer_sr = {s: np.full(n_layers, np.nan, dtype=np.float32) for s in subj_data}
        subj_layer_pr = {s: np.full(n_layers, np.nan, dtype=np.float32) for s in subj_data}

        if use_gpu:
            def _fit_one(subj: str, li: int) -> tuple[str, int, CVScore]:
                idx, Y, groups, _cloze = subj_data[subj]
                X_li = features[idx, li, :].astype(np.float32)
                # Cap folds to this subject's actual number of unique frames —
                # GroupKFold raises if n_splits > n_groups, and not every
                # subject has >= cfg.n_splits distinct (subexperiment, item)
                # frames.
                n_splits_eff = min(cfg.n_splits, len(np.unique(groups)))
                import torch
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    scores = cv_ridge_score_multioutput_gpu(
                        X_li, Y, groups=groups, n_splits=n_splits_eff,
                        alpha=cfg.ridge_alphas, scale=cfg.scale,
                        standardize_target=cfg.standardize_target,
                    )
                stream.synchronize()
                return subj, li, scores[0]

            all_tasks = [(s, li) for s in subj_data for li in range(n_layers)]

            fit_cache_path: Path | None = None
            if cfg.fit_cache_dir:
                fit_cache_path = _fit_cache_path(cfg.fit_cache_dir, cfg.hf_model_name, cfg.hf_revision)
                cached = _load_fit_cache(fit_cache_path)
                for (s, li), (sr_v, pr_v) in cached.items():
                    if s in subj_layer_sr and li < n_layers:
                        subj_layer_sr[s][li] = sr_v
                        subj_layer_pr[s][li] = pr_v
                if cached:
                    logger.info(
                        "Szewczyk: resuming from fit cache — %d/%d (subject, layer) fits already done",
                        len(cached), len(all_tasks),
                    )

            tasks = [(s, li) for s, li in all_tasks if not np.isfinite(subj_layer_sr[s][li])]
            n_workers = max(1, min(cfg.n_gpu_workers, len(tasks) or 1))
            logger.info(
                "Szewczyk: %d subjects x %d layers = %d ridge fits, %d remaining, %d concurrent workers (gpu=True)",
                len(subj_data), n_layers, len(all_tasks), len(tasks), n_workers,
            )
            if tasks:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_fit_one, s, li): (s, li) for s, li in tasks}
                    for fut in as_completed(futures):
                        subj, li, score = fut.result()
                        subj_layer_sr[subj][li] = score.spearmanr
                        subj_layer_pr[subj][li] = score.pearsonr
                        if fit_cache_path is not None:
                            _append_fit_cache(fit_cache_path, subj, li, score.spearmanr, score.pearsonr)
        else:
            # Real multiprocessing, parallelized per SUBJECT (not per
            # (subject, layer)) -- see _szewczyk_fit_subject_all_layers's
            # docstring for why ThreadPoolExecutor doesn't scale here (GIL
            # contention on many small sklearn calls). fork context lets
            # every worker see `features` via _SZEWCZYK_WORKER_FEATURES with
            # zero-copy inheritance instead of pickling it per task/worker.
            global _SZEWCZYK_WORKER_FEATURES
            _SZEWCZYK_WORKER_FEATURES = features
            n_workers = max(1, min(cfg.n_gpu_workers, os.cpu_count() or 8, len(subj_data)))
            logger.info(
                "Szewczyk: %d subjects x %d layers = %d ridge fits, %d concurrent worker processes (gpu=False)",
                len(subj_data), n_layers, len(subj_data) * n_layers, n_workers,
            )
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {
                    pool.submit(
                        _szewczyk_fit_subject_all_layers,
                        subj, idx, Y[:, 0], groups,
                        n_layers=n_layers, n_splits=cfg.n_splits,
                        ridge_alphas=cfg.ridge_alphas, scale=cfg.scale,
                        standardize_target=cfg.standardize_target,
                    ): subj
                    for subj, (idx, Y, groups, _cloze) in subj_data.items()
                }
                for fut in as_completed(futures):
                    subj, sr, pr = fut.result()
                    subj_layer_sr[subj] = sr
                    subj_layer_pr[subj] = pr
            _SZEWCZYK_WORKER_FEATURES = None

        # ── Per-subject best layer, OOF predictions, and pooled accumulators ──
        subject_results: list[SzewczykSubjectResult] = []
        pool_pred_best: list[np.ndarray] = []
        pool_pred_last: list[np.ndarray] = []
        pool_actual:    list[np.ndarray] = []
        pool_cloze:     list[np.ndarray] = []

        for subj in subj_data:
            idx, Y, groups, cloze = subj_data[subj]
            sr = subj_layer_sr[subj]
            pr = subj_layer_pr[subj]

            if not np.any(np.isfinite(sr)):
                continue
            best_li = int(np.nanargmax(sr))

            X_best = features[idx, best_li, :].astype(np.float32)
            y_pred_best = self._oof_predictions(X_best, Y[:, 0], groups)

            if last_layer != best_li:
                X_last = features[idx, last_layer, :].astype(np.float32)
                y_pred_last = self._oof_predictions(X_last, Y[:, 0], groups)
            else:
                y_pred_last = y_pred_best

            subject_results.append(SzewczykSubjectResult(
                subject=subj,
                n_items=len(idx),
                best_layer=best_li,
                last_layer=last_layer,
                mean_spearmanr=float(sr[best_li]),
                mean_pearsonr=float(pr[best_li]),
                last_layer_mean_spearmanr=float(sr[last_layer]),
                last_layer_mean_pearsonr=float(pr[last_layer]),
                layer_spearmanr=sr.tolist(),
                layer_pearsonr=pr.tolist(),
            ))

            pool_pred_best.append(y_pred_best)
            pool_pred_last.append(y_pred_last)
            pool_actual.append(Y[:, 0])
            pool_cloze.append(cloze)

            logger.info(
                "  subject=%s | n=%d | best_layer=%d | sr=%.4f",
                subj, len(idx), best_li, sr[best_li],
            )

        # ── Pooled secondary diagnostics ──────────────────────────────────────
        all_pred_best = np.concatenate(pool_pred_best)
        all_pred_last = np.concatenate(pool_pred_last)
        all_actual    = np.concatenate(pool_actual)
        all_cloze     = np.concatenate(pool_cloze)

        cloze_corr_pred_best = _safe_spearmanr(all_pred_best, all_cloze)
        cloze_corr_pred_last = _safe_spearmanr(all_pred_last, all_cloze)
        cloze_corr_actual    = _safe_spearmanr(all_actual, all_cloze)

        expected_mask = all_cloze >= cfg.expected_cloze_threshold
        unexpected_mask = ~expected_mask

        expected_spearmanr_best   = _safe_spearmanr(all_pred_best[expected_mask], all_actual[expected_mask])
        unexpected_spearmanr_best = _safe_spearmanr(all_pred_best[unexpected_mask], all_actual[unexpected_mask])
        expected_spearmanr_last   = _safe_spearmanr(all_pred_last[expected_mask], all_actual[expected_mask])
        unexpected_spearmanr_last = _safe_spearmanr(all_pred_last[unexpected_mask], all_actual[unexpected_mask])

        # ── Aggregate across subjects ──────────────────────────────────────────
        result = self._aggregate(
            subject_results=subject_results,
            n_layers=n_layers,
            cloze_corr_pred_best=cloze_corr_pred_best,
            cloze_corr_pred_last=cloze_corr_pred_last,
            cloze_corr_actual=cloze_corr_actual,
            expected_spearmanr_best=expected_spearmanr_best,
            unexpected_spearmanr_best=unexpected_spearmanr_best,
            expected_spearmanr_last=expected_spearmanr_last,
            unexpected_spearmanr_last=unexpected_spearmanr_last,
            n_expected_items=int(expected_mask.sum()),
            n_unexpected_items=int(unexpected_mask.sum()),
        )
        return result

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, items: pd.DataFrame) -> np.ndarray:
        cfg = self.cfg
        fcache = Path(cfg.feature_cache_dir) if cfg.feature_cache_dir else None
        return extract_sentence_features(
            contexts=items["context"].tolist(),
            nouns=items["critical_word"].tolist(),
            model_name=cfg.hf_model_name,
            revision=cfg.hf_revision,
            layer_mode=cfg.layer_mode,
            device=cfg.hf_device,
            batch_size=cfg.hf_batch_size,
            max_length=cfg.hf_max_length,
            tokenizer_no_revision=cfg.tokenizer_no_revision,
            cache_dir=fcache,
            n_params=cfg.n_params if cfg.n_params > 0 else None,
            trust_remote_code=cfg.trust_remote_code,
        )

    # ------------------------------------------------------------------
    # OOF predictions (CPU, GroupKFold, one layer at a time)
    # ------------------------------------------------------------------

    def _oof_predictions(
        self,
        X: np.ndarray,      # (n_sub, D)
        y: np.ndarray,      # (n_sub,)
        groups: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        y_pred = np.full_like(y, np.nan, dtype=np.float32)
        n_groups = len(np.unique(groups))
        n_splits = min(cfg.n_splits, n_groups)
        gkf      = GroupKFold(n_splits=n_splits)
        alphas   = np.array(cfg.ridge_alphas)

        for tr_idx, te_idx in gkf.split(X, groups=groups):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr        = y[tr_idx]
            if cfg.scale:
                sc   = RobustScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)
            model = _SkRidgeCV(alphas=alphas)
            model.fit(X_tr, y_tr)
            y_pred[te_idx] = model.predict(X_te).astype(np.float32)

        return y_pred

    # ------------------------------------------------------------------
    # CPU fallback: single-output GroupKFold ridge scoring
    # ------------------------------------------------------------------

    def _cpu_single_output_score(
        self,
        X: np.ndarray,       # (n_sub, D)
        y: np.ndarray,       # (n_sub,)
        groups: np.ndarray,
        n_splits: int,
    ) -> CVScore:
        """Instance-method wrapper around :func:`_szewczyk_cpu_single_output_score`
        (kept for any external caller expecting the old method signature) —
        see that module-level function's docstring for the actual logic and
        why it exists as a standalone (picklable) function."""
        cfg = self.cfg
        return _szewczyk_cpu_single_output_score(
            X, y, groups, n_splits,
            ridge_alphas=cfg.ridge_alphas, scale=cfg.scale,
            standardize_target=cfg.standardize_target,
        )

    # ------------------------------------------------------------------
    # Aggregate across subjects
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        subject_results: list[SzewczykSubjectResult],
        n_layers: int,
        **pooled_kw: tp.Any,
    ) -> SzewczykResult:
        cfg = self.cfg

        n_fin = lambda x: max(1, np.sum(np.isfinite(x)))
        sub_sr  = np.array([s.mean_spearmanr for s in subject_results], dtype=np.float64)
        sub_pr  = np.array([s.mean_pearsonr  for s in subject_results], dtype=np.float64)
        sub_lsr = np.array([s.last_layer_mean_spearmanr for s in subject_results], dtype=np.float64)
        sub_lpr = np.array([s.last_layer_mean_pearsonr  for s in subject_results], dtype=np.float64)

        layer_sr_mat = np.array([s.layer_spearmanr for s in subject_results])
        layer_pr_mat = np.array([s.layer_pearsonr  for s in subject_results])
        layer_spearmanr = [float(np.nanmean(layer_sr_mat[:, li])) for li in range(n_layers)]
        layer_pearsonr  = [float(np.nanmean(layer_pr_mat[:, li])) for li in range(n_layers)]

        return SzewczykResult(
            model_name=cfg.hf_model_name,
            revision=cfg.hf_revision,
            n_layers=n_layers,
            n_subjects=len(subject_results),
            subject_results=subject_results,
            mean_spearmanr=float(np.nanmean(sub_sr)),
            sem_spearmanr=float(np.nanstd(sub_sr, ddof=1) / np.sqrt(n_fin(sub_sr))),
            mean_pearsonr=float(np.nanmean(sub_pr)),
            last_layer_mean_spearmanr=float(np.nanmean(sub_lsr)),
            last_layer_sem_spearmanr=float(np.nanstd(sub_lsr, ddof=1) / np.sqrt(n_fin(sub_lsr))),
            last_layer_mean_pearsonr=float(np.nanmean(sub_lpr)),
            layer_spearmanr=layer_spearmanr,
            layer_pearsonr=layer_pearsonr,
            **pooled_kw,
        )


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

def result_to_checkpoint_row(
    result: SzewczykResult,
    *,
    step: int = 0,
    tokens_seen: int = 0,
    n_params: float = 0.0,
) -> dict[str, tp.Any]:
    return {
        "model":                       result.model_name,
        "revision":                    result.revision or "main",
        "step":                        step,
        "tokens_seen":                 tokens_seen,
        "n_params":                    n_params,
        "n_subjects":                  result.n_subjects,
        "n_layers":                    result.n_layers,
        "mean_spearmanr":              result.mean_spearmanr,
        "sem_spearmanr":               result.sem_spearmanr,
        "mean_pearsonr":               result.mean_pearsonr,
        "last_layer_mean_spearmanr":   result.last_layer_mean_spearmanr,
        "last_layer_sem_spearmanr":    result.last_layer_sem_spearmanr,
        "last_layer_mean_pearsonr":    result.last_layer_mean_pearsonr,
        "cloze_corr_pred_best":        result.cloze_corr_pred_best,
        "cloze_corr_pred_last":        result.cloze_corr_pred_last,
        "cloze_corr_actual":           result.cloze_corr_actual,
        "expected_spearmanr_best":     result.expected_spearmanr_best,
        "unexpected_spearmanr_best":   result.unexpected_spearmanr_best,
        "expected_spearmanr_last":     result.expected_spearmanr_last,
        "unexpected_spearmanr_last":   result.unexpected_spearmanr_last,
        "n_expected_items":            result.n_expected_items,
        "n_unexpected_items":          result.n_unexpected_items,
    }


def result_to_layerwise_rows(
    result: SzewczykResult,
    *,
    step: int = 0,
    tokens_seen: int = 0,
    n_params: float = 0.0,
) -> list[dict[str, tp.Any]]:
    return [
        {
            "model":          result.model_name,
            "revision":       result.revision or "main",
            "step":           step,
            "tokens_seen":    tokens_seen,
            "n_params":       n_params,
            "layer":          li,
            "mean_spearmanr": result.layer_spearmanr[li],
            "mean_pearsonr":  result.layer_pearsonr[li],
        }
        for li in range(result.n_layers)
    ]


def result_to_subject_rows(
    result: SzewczykResult,
    *,
    step: int = 0,
    tokens_seen: int = 0,
    n_params: float = 0.0,
) -> list[dict[str, tp.Any]]:
    return [
        {
            "model":                     result.model_name,
            "revision":                  result.revision or "main",
            "step":                      step,
            "tokens_seen":               tokens_seen,
            "n_params":                  n_params,
            "subject":                   sr.subject,
            "n_items":                   sr.n_items,
            "best_layer":                sr.best_layer,
            "last_layer":                sr.last_layer,
            "mean_spearmanr":            sr.mean_spearmanr,
            "mean_pearsonr":             sr.mean_pearsonr,
            "last_layer_mean_spearmanr": sr.last_layer_mean_spearmanr,
            "last_layer_mean_pearsonr":  sr.last_layer_mean_pearsonr,
        }
        for sr in result.subject_results
    ]


def save_results(
    result: SzewczykResult,
    out_dir: str | Path,
    *,
    step: int = 0,
    tokens_seen: int = 0,
    n_params: float = 0.0,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kw = dict(step=step, tokens_seen=tokens_seen, n_params=n_params)
    ck_row  = result_to_checkpoint_row(result, **kw)
    lw_rows = result_to_layerwise_rows(result, **kw)
    su_rows = result_to_subject_rows(result, **kw)

    with _csv_lock:
        _csv_append(out / "szewczyk_checkpoint_scores.csv",    [ck_row])
        _csv_append(out / "szewczyk_checkpoint_layerwise.csv", lw_rows)
        _csv_append(out / "szewczyk_by_subject.csv",           su_rows)

    logger.info(
        "Szewczyk saved: step=%d | sr=%.4f+-%.4f | %d subjects -> %s",
        step, result.mean_spearmanr, result.sem_spearmanr, result.n_subjects, out,
    )


# ---------------------------------------------------------------------------
# Restart guard
# ---------------------------------------------------------------------------

def already_done(
    out_dir: str | Path,
    model_name: str,
    revision: str | None,
) -> bool:
    csv_path = Path(out_dir) / "szewczyk_checkpoint_scores.csv"
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path, usecols=["model", "revision"])
        rev_key = revision or "main"
        return bool(((df["model"] == model_name) & (df["revision"] == rev_key)).any())
    except Exception:
        return False
