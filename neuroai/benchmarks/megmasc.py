"""MEG-MASC (Gwilliams et al. 2023) MEG encoding benchmark.

Predicts word-locked MEG (208 KIT axial gradiometers, [0.1, 40] Hz, 50 Hz)
from LLM hidden states using per-channel lag-matrix ridge regression with
leave-one-story-out (LOSO) cross-validation over the four MASC stories.

Pipeline (per subject)
----------------------
1. Build one canonical word list across the four stories (text is identical
   across subjects) and extract LLM features once (English, full-text
   sliding-window pass).
2. For each (session, story) recording the subject has:
   a. Read the raw KIT .con (MNE), MEG channels only.
   b. Bandpass [0.1, 40] Hz at 1000 Hz, polyphase-resample to 50 Hz,
      per-channel RobustScaler + clamp to [-5, 5] (arXiv:2412.17829 recipe,
      tmax stretched to 0.8 s). No notch — 50 Hz mains is outside the band.
   c. Extract word-locked epochs [tmin, tmax] using this recording's own
      (recording-relative) onsets; baseline-correct.
3. Pool epochs across stories AND sessions; story index = CV group
   (LOSO, 4 folds; both sessions of a story share a fold).
4. Per-channel lag-matrix ridge regression → TimeResolvedScore per channel.
   Brain score = mean over lags of median-over-folds Pearson r.
5. Layer sweep: repeat steps 1(features)–4 per transformer layer; track best.

GPU parallelism
---------------
The flat (subject × layer) multiprocessing path places embeddings and per-subject
EEG arrays in OS shared memory and submits all (subj, layer) tasks to a
ProcessPoolExecutor with a spawn context (required: CUDA cannot be forked).
When CUDA MPS is enabled at the SLURM level (-C mps + --exclusive), the spawned
processes share a single GPU context via the MPS daemon — eliminating the CUDA
context-switching crashes observed on Jean Zay without MPS.

Do NOT attempt to start the MPS daemon manually. On Jean Zay, MPS is managed
entirely by SLURM. The Python code is MPS-agnostic; only the SLURM script needs:
  #SBATCH -C mps
  #SBATCH --exclusive

Outputs
-------
Per-checkpoint (written to output_dir):
  megmasc_checkpoint_scores.csv      — best-layer summary per checkpoint
  megmasc_by_channel.csv             — per-channel scores (all subjects)
  megmasc_by_subject.csv             — per-subject mean
  megmasc_electrode_timecourse.csv   — time-resolved r per channel
  megmasc_by_channel_layerwise.csv   — per-channel × layer Pearson r
  megmasc_layerwise_timecourse.csv   — TRF curves (mean±std) per layer
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..benchmarks.base import ModelConfig, RegressionConfig, word_predictions_dir
from ..datasets.megmasc import (
    MEGMASC_SFREQ,
    MegMascDataset,
    load_meg_run,
    load_run_words,
    load_story_words,
)
from ..features.llm import (
    contextual_word_embeddings_hf,
    contextual_word_embeddings_hf_fulltext,
)
from ..regressors.ridge import (
    CVScore,
    TimeResolvedScore,
    cv_ridge_score_lag_matrix,
    cv_ridge_score_lag_matrix_gpu_batched,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MegMascDataConfig:
    """Dataset location and epoch-extraction parameters."""

    bids_root: str
    subjects: list[str] | None = None
    stories: list[str] | None = None
    # Epoch window (s relative to word onset). Shared MEG recipe (LittlePrince
    # M/EEG paper, arXiv:2412.17829) with tmax stretched 0.6 -> 0.8 s.
    tmin: float = -0.2
    tmax: float = 0.8
    # Baseline correction: subtract mean over [bl_tmin, 0] from each epoch
    baseline: tuple[float, float] | None = (-0.2, 0.0)
    # Raw KIT is 1000 Hz. Bandpass at the native rate then resample to 50 Hz.
    # [0.1, 40] Hz keeps broadband/evoked MEG; 50 Hz mains sits outside it so
    # no notch is applied (verified: MEG-MASC line peak is at 50 Hz).
    l_freq: float | None = 0.1
    h_freq: float | None = 40.0
    target_sfreq: float = 50.0
    # Per-channel RobustScaler on the continuous signal, then clamp to
    # [-clamp, clamp] (same outlier handling as the LittlePrince MEG paper).
    robust_scale: bool = True
    clamp: float | None = 5.0
    # Drop words shorter than this (forced-alignment artifacts)
    min_word_duration: float = 0.050
    # Disk caches
    feature_cache_dir: str | None = None
    epoch_cache_dir: str | None = None
    # LLM context
    context_words: int | None = None
    max_length: int | None = None


@dataclass
class MegMascBenchmarkConfig:
    """Top-level config for the MEG-MASC MEG encoding benchmark."""

    data: MegMascDataConfig
    model: ModelConfig
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    output_dir: str = "outputs"
    layerwise: bool = True
    use_lag_matrix: bool = True
    lang: str = "en"                 # English — passed to feature extractor
    n_jobs: int = 20                 # joblib workers for CPU-only fallback path
    regression_device: str = "cpu"  # "cpu" or "cuda"
    n_subjects_batch: int = 5        # subjects per parallel batch; keep small to limit shared-memory peak RAM
    workers_per_gpu: int = 10         # concurrent MPS tasks per GPU (4 GPUs × 10 = 40 total)
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    # Save each channel's best-layer out-of-fold predicted lag matrix AND the
    # actual observed EEG (needed to score any post-hoc word subset, e.g. an
    # easy/hard DLT-cost split) — same convention as Broderick's
    # SAVE_WORD_PREDICTIONS. On by default. Forces the sequential per-subject
    # CPU path (OOF capture isn't wired through the multiprocess flat-batch
    # GPU path), same restriction as Broderick.
    save_word_predictions: bool = True


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChannelScore:
    """Encoding score for one EEG channel × one subject."""

    subject: str
    name: str
    region: str | None
    score: CVScore
    time_course: TimeResolvedScore | None = None
    layer_scores: list[CVScore] | None = None
    layer_time_courses: list[TimeResolvedScore] | None = None
    # Last-layer refit (with OOF predictions) when save_word_predictions=True.
    # Populated alongside time_course (the winning-layer refit) — together
    # these are the only two layers word predictions are ever saved for; see
    # _save_word_predictions.
    last_layer_time_course: TimeResolvedScore | None = None

    def best_layer_index(self, metric: str = "pearsonr") -> int | None:
        if self.layer_scores:
            vals = np.asarray([getattr(s, metric) for s in self.layer_scores], dtype=float)
            if vals.size > 0 and np.any(np.isfinite(vals)):
                return int(np.nanargmax(vals))
        return None

    def best_layer_pearsonr(self) -> float:
        if self.layer_scores:
            vals = np.asarray([s.pearsonr for s in self.layer_scores], dtype=float)
            return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else float("nan")
        return self.score.pearsonr

    def best_layer_spearmanr(self) -> float:
        if self.layer_scores:
            vals = np.asarray([s.spearmanr for s in self.layer_scores], dtype=float)
            return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else float("nan")
        return self.score.spearmanr

    def timecourse_rows(self) -> list[dict]:
        if self.time_course is None:
            return []
        tc = self.time_course
        base = {"subject": self.subject, "channel": self.name, "region": self.region}
        return [
            {**base, "time_s": float(tc.window_centers[t]),
             "pearsonr": float(tc.pearsonr[t]),
             "spearmanr": float(tc.spearmanr[t])}
            for t in range(len(tc.window_centers))
        ]

    def to_dict(self, include_layers: bool = False) -> dict:
        d: dict[str, tp.Any] = {
            "subject": self.subject,
            "channel": self.name,
            "region": self.region,
            **self.score.to_dict(),
            "best_pearsonr": self.best_layer_pearsonr(),
            "best_spearmanr": self.best_layer_spearmanr(),
            "best_layer": self.best_layer_index("pearsonr"),
        }
        if self.time_course is not None:
            d["peak_pearsonr"] = self.time_course.peak_score("pearsonr")
            d["peak_spearmanr"] = self.time_course.peak_score("spearmanr")
            d["peak_time_s"] = self.time_course.peak_time("pearsonr")
        if include_layers and self.layer_scores:
            for li, ls in enumerate(self.layer_scores):
                d[f"layer_{li}_pearsonr"] = ls.pearsonr
                d[f"layer_{li}_spearmanr"] = ls.spearmanr
        return d


@dataclass
class MegMascEncoderResult:
    """Aggregated encoding results for one model checkpoint."""

    model_name: str
    model_revision: str | None
    n_words: int
    channel_scores: list[ChannelScore]
    n_subjects_expected: int = 0

    def mean_pearsonr(self) -> float:
        vals = [c.best_layer_pearsonr() for c in self.channel_scores]
        return float(np.nanmean(vals)) if vals else float("nan")

    def mean_spearmanr(self) -> float:
        vals = [c.best_layer_spearmanr() for c in self.channel_scores]
        return float(np.nanmean(vals)) if vals else float("nan")

    def to_dataframe(self, include_layers: bool = False) -> pd.DataFrame:
        return pd.DataFrame(
            [c.to_dict(include_layers=include_layers) for c in self.channel_scores]
        )

    def channel_timecourse_df(self) -> pd.DataFrame:
        rows: list[dict] = []
        for c in self.channel_scores:
            rows.extend(c.timecourse_rows())
        return pd.DataFrame(rows)

    def by_subject(self) -> pd.DataFrame:
        df = self.to_dataframe()
        cols = ["best_pearsonr", "best_spearmanr", "pearsonr", "spearmanr"]
        agg = df.groupby("subject")[cols].agg(["mean", "std", "count"])
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        return agg.reset_index()


# ---------------------------------------------------------------------------
# EEG processing helpers
# ---------------------------------------------------------------------------

def _bandpass_eeg(
    data: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter. data: (n_channels, n_samples)."""
    from scipy.signal import butter, sosfiltfilt
    nyq = sfreq / 2.0
    sos = butter(order, [l_freq / nyq, h_freq / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def _robust_scale_clamp(
    data: np.ndarray,          # (n_channels, n_samples)
    clamp: float | None,
) -> np.ndarray:
    """Per-channel RobustScaler (median / IQR) on the continuous signal, then
    clamp to ``[-clamp, clamp]``. Matches the LittlePrince M/EEG paper
    (arXiv:2412.17829) outlier handling. Channels with a degenerate IQR are
    left centred-only."""
    med = np.median(data, axis=1, keepdims=True)
    q75, q25 = np.percentile(data, [75, 25], axis=1, keepdims=True)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    out = (data - med) / iqr
    if clamp is not None:
        out = np.clip(out, -clamp, clamp)
    return out.astype(np.float32)


def _extract_word_epochs(
    eeg: np.ndarray,          # (n_channels, n_samples)
    sfreq: float,
    word_onsets: np.ndarray,  # seconds from recording start
    tmin: float,
    tmax: float,
    baseline: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract word-locked epochs from continuous EEG.

    Returns
    -------
    epochs : (n_valid, n_channels, n_times) float32
    valid_mask : (n_words,) bool
    times : (n_times,) float  — seconds relative to word onset
    """
    n_channels, n_samples = eeg.shape
    tmin_samp = int(round(tmin * sfreq))
    tmax_samp = int(round(tmax * sfreq))
    n_times = tmax_samp - tmin_samp
    times = np.arange(tmin_samp, tmax_samp) / sfreq

    epochs_list: list[np.ndarray] = []
    valid_mask = np.zeros(len(word_onsets), dtype=bool)

    for i, onset in enumerate(word_onsets):
        onset_samp = int(round(onset * sfreq))
        start = onset_samp + tmin_samp
        end = onset_samp + tmax_samp
        if start < 0 or end > n_samples:
            continue
        epoch = eeg[:, start:end].copy()  # (n_channels, n_times)

        if baseline is not None:
            bl_tmin, bl_tmax = baseline
            bl_mask = (times >= bl_tmin) & (times <= bl_tmax)
            if bl_mask.any():
                epoch = epoch - epoch[:, bl_mask].mean(axis=-1, keepdims=True)

        epochs_list.append(epoch.astype(np.float32))
        valid_mask[i] = True

    if not epochs_list:
        return (
            np.zeros((0, n_channels, n_times), dtype=np.float32),
            valid_mask,
            times.astype(np.float32),
        )

    return (
        np.stack(epochs_list, axis=0),   # (n_valid, n_channels, n_times)
        valid_mask,
        times.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Epoch cache helpers
# ---------------------------------------------------------------------------

def _epoch_cache_key(dcfg: MegMascDataConfig) -> str:
    """MD5 of preprocessing hyperparameters — versions the epoch cache."""
    params: dict = {
        "l_freq": dcfg.l_freq,
        "h_freq": dcfg.h_freq,
        "target_sfreq": dcfg.target_sfreq,
        "tmin": dcfg.tmin,
        "tmax": dcfg.tmax,
        "baseline": list(dcfg.baseline) if dcfg.baseline is not None else None,
        "min_word_duration": dcfg.min_word_duration,
        "robust_scale": dcfg.robust_scale,
        "clamp": dcfg.clamp,
    }
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def _load_epoch_cache(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (epochs, groups, times, word_indices)."""
    d = np.load(path, allow_pickle=False)
    return d["epochs"], d["groups"], d["times"], d["word_indices"]


def _save_epoch_cache(
    path: Path,
    epochs: np.ndarray,
    groups: np.ndarray,
    times: np.ndarray,
    word_indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: concurrent SLURM array workers (different model checkpoints)
    # all build the same model-independent epoch cache on the first run — a
    # partial file from a crashed/racing writer must never be picked up.
    import os
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            epochs=epochs.astype(np.float32),
            groups=groups.astype(np.int32),
            times=times.astype(np.float32),
            word_indices=word_indices.astype(np.int32),
        )
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Multiprocessing worker (module-level for spawn-safe pickling)
#
# Called by ProcessPoolExecutor spawned processes. Each worker:
#   1. Attaches to shared memory for embeddings (zero-copy) and EEG epochs.
#   2. Extracts the layer slice from the embedding array.
#   3. Runs GPU batched lag-matrix ridge for all channels simultaneously.
#   4. Releases the GPU cache to prevent accumulation across tasks.
#
# CUDA MPS note: this worker calls into CUDA from a spawned process. If CUDA
# MPS is enabled by SLURM (-C mps + --exclusive), all spawned workers share one
# GPU context via the MPS daemon — no manual nvidia-cuda-mps-control needed.
# Without MPS, multiple spawned CUDA contexts on the same GPU cause crashes.
# ---------------------------------------------------------------------------

def _gpu_pool_initializer(gpu_id: int) -> None:
    """Called once per worker process: pin this process to a single GPU forever.

    Using a per-GPU pool with an initializer (rather than round-robin set_device
    inside each task) ensures that every worker process creates exactly ONE CUDA
    context on ONE GPU for its entire lifetime.  Without this, workers reassigned
    to different GPUs accumulate stale CUDA contexts (~200-500 MiB each) that
    torch.cuda.empty_cache() cannot release because it only clears the current
    device.  With 32 workers and 1850 tasks cycling across 4 GPUs, this caused
    ~24 contexts on a single GPU and OOM.
    """
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.set_device(gpu_id)


def _mp_worker_fit(task: dict) -> tuple[str, int, list]:
    """Spawned worker: fit one (subject, layer) ridge regression on GPU.

    Workers are pre-pinned to their GPU via _gpu_pool_initializer.
    set_device here is a no-op safety net in case the task is run outside
    the normal per-GPU pool (e.g. during testing).
    """
    from multiprocessing.shared_memory import SharedMemory

    import numpy as _np
    from neuroai.regressors.ridge import cv_ridge_score_lag_matrix_gpu_batched as _fit

    subj: str = task["subj"]
    li: int = task["li"]

    # Redundant when workers are pre-pinned via initializer, but harmless.
    try:
        import torch as _torch
        _torch.cuda.set_device(task.get("gpu_id", 0))
    except Exception:
        pass

    # Attach to embeddings shared memory; extract layer slice; detach immediately.
    # Embeddings are stored as (n_layers, n_words, D) — transposed so that
    # emb_full[li] is a *contiguous* (n_words, D) block. Without the transpose,
    # emb_full[:, li, :] strides across the entire array, causing TLB stalls when
    # many workers run simultaneously on a large embedding (e.g. 9+ GB for 12B models).
    shm_emb = SharedMemory(name=task["emb_shm_name"])
    emb_full = _np.ndarray(task["emb_shape"], dtype=task["emb_dtype"], buffer=shm_emb.buf)
    X_li = _np.array(emb_full[li] if task["emb_layerwise"] else emb_full)
    shm_emb.close()
    del emb_full  # drop view into shared memory immediately

    # Attach to per-subject EEG epochs; copy; detach.
    shm_eeg = SharedMemory(name=task["eeg_shm_name"])
    eeg = _np.array(_np.ndarray(task["eeg_shape"], dtype=task["eeg_dtype"], buffer=shm_eeg.buf))
    shm_eeg.close()

    tcs = _fit(X_li, eeg, task["times"], **task["batched_kw"])

    # Free CPU arrays and flush the GPU caching allocator.
    # With 8 concurrent workers per GPU, unreleased allocations across tasks
    # accumulate and cause OOM on subsequent tasks in the same worker process.
    del X_li, eeg
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return subj, li, tcs


# Thread lock for safe concurrent CSV appends across the sequential path.
_csv_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Per-channel CPU regression worker (module-level for joblib pickling)
# ---------------------------------------------------------------------------

def _regress_channel_cpu(
    ci: int,
    emb: np.ndarray,        # (n_words, [n_layers,] D)
    y_ci: np.ndarray,       # (n_words, n_times)
    times: np.ndarray,
    groups: np.ndarray,
    ch_name: str,
    subject: str,
    n_splits: int,
    reg_kwargs: dict,
    save_word_predictions: bool = False,
) -> ChannelScore:
    layerwise = emb.ndim == 3

    def _best_li(scores: list[CVScore]) -> int:
        vals = np.asarray([s.pearsonr for s in scores], dtype=float)
        return int(np.nanargmax(vals)) if np.any(np.isfinite(vals)) else 0

    if layerwise:
        n_layers = emb.shape[1]
        layer_tcs: list[TimeResolvedScore] = []
        for li in range(n_layers):
            tc = cv_ridge_score_lag_matrix(
                emb[:, li, :], y_ci, times, groups=groups,
                n_splits=n_splits, **reg_kwargs,
            )
            layer_tcs.append(tc)
        layer_scores = [tc.to_mean_cvScore() for tc in layer_tcs]
        best_li = _best_li(layer_scores)
        last_li = n_layers - 1
        time_course = layer_tcs[best_li]
        last_layer_time_course = layer_tcs[last_li]
        if save_word_predictions:
            # Refit only the winning + last layer with OOF predictions kept —
            # avoids holding a (n_words × n_times) array per layer in
            # memory/CSV. These are the only two layers word predictions are
            # ever saved for (see ChannelScore.last_layer_time_course).
            time_course = cv_ridge_score_lag_matrix(
                emb[:, best_li, :], y_ci, times,
                groups=groups, n_splits=n_splits, return_oof_predictions=True, **reg_kwargs,
            )
            last_layer_time_course = (
                time_course if last_li == best_li
                else cv_ridge_score_lag_matrix(
                    emb[:, last_li, :], y_ci, times,
                    groups=groups, n_splits=n_splits, return_oof_predictions=True, **reg_kwargs,
                )
            )
        return ChannelScore(
            subject=subject, name=ch_name, region=None,
            score=layer_scores[-1],
            time_course=time_course,
            layer_scores=layer_scores,
            layer_time_courses=layer_tcs,
            last_layer_time_course=last_layer_time_course,
        )
    else:
        tc = cv_ridge_score_lag_matrix(
            emb, y_ci, times, groups=groups,
            n_splits=n_splits, return_oof_predictions=save_word_predictions, **reg_kwargs,
        )
        return ChannelScore(
            subject=subject, name=ch_name, region=None,
            score=tc.to_mean_cvScore(),
            time_course=tc,
        )


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class MegMascMegEncoder:
    """Predict MEG-MASC MEG ([0.1, 40] Hz, 50 Hz) from LLM hidden states.

    Architecture mirrors BroderickEegEncoder with three differences:
    - Raw KIT .con read via MNE; preprocessing done here (bandpass+resample).
    - Cross-validation: leave-one-story-out (4 stories, sessions pooled).
    - Language: English (lang="en").
    """

    def __init__(self, cfg: MegMascBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(self) -> MegMascEncoderResult:
        words_df = self._load_all_words()
        if words_df.empty:
            raise RuntimeError(f"No words found under {self.cfg.data.bids_root}")

        words = words_df["word"].astype(str).tolist()
        emb = self._extract_features(words)
        logger.info(
            "[%s] features shape: %s",
            self.cfg.model.model_name,
            "×".join(str(d) for d in emb.shape),
        )

        # Free LLM from GPU immediately after caching features to disk.
        from ..features.llm import clear_hf_model_cache
        clear_hf_model_cache()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

        ds = MegMascDataset(
            self.cfg.data.bids_root,
            subjects=self.cfg.data.subjects,
            stories=self.cfg.data.stories,
        )
        subjects = ds.available_subjects()
        if not subjects:
            raise RuntimeError(
                f"No subjects with EEG data found under {self.cfg.data.bids_root}"
            )
        logger.info(
            "[%s] running on %d subjects", self.cfg.model.model_name, len(subjects)
        )

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ch_csv    = out_dir / "megmasc_by_channel.csv"
        subj_csv  = out_dir / "megmasc_by_subject.csv"
        tc_csv    = out_dir / "megmasc_electrode_timecourse.csv"
        lw_csv    = out_dir / "megmasc_by_channel_layerwise.csv"
        lw_tc_csv = out_dir / "megmasc_layerwise_timecourse.csv"

        model_revision = self.cfg.model.model_revision or "main"

        # Skip subjects already computed for this revision (restart-safe).
        done_subjects: set[str] = set()
        if ch_csv.exists():
            try:
                _ex = pd.read_csv(ch_csv, usecols=["subject", "revision"])
                done_subjects = set(
                    _ex.loc[_ex["revision"] == model_revision, "subject"].unique()
                )
                if done_subjects:
                    logger.info(
                        "Skipping %d already-done subject(s): %s",
                        len(done_subjects), sorted(done_subjects),
                    )
            except Exception:
                logger.warning("Could not read existing channel CSV; re-running all.")

        subjects_todo = [s for s in subjects if s not in done_subjects]
        all_scores: list[ChannelScore] = []

        use_parallel = (
            self.cfg.n_subjects_batch != 0
            and self.cfg.regression_device != "cpu"
        )
        if self.cfg.save_word_predictions and use_parallel:
            logger.info(
                "save_word_predictions=True — flat (subject x layer) MP batch path "
                "now carries return_oof_predictions through to each worker and "
                "saves word predictions per subject as soon as its layers finish."
            )

        if use_parallel:
            batch_size = (
                len(subjects_todo) if self.cfg.n_subjects_batch < 0
                else self.cfg.n_subjects_batch
            )
            n_batches = max(1, (len(subjects_todo) + batch_size - 1) // batch_size)

            # Create GPU pools ONCE for all batches — avoids paying the ~30-second
            # spawn-and-CUDA-init cost on each batch.  Shared memory is still
            # created/destroyed per batch; only the worker processes are reused.
            import multiprocessing as _mpp_run
            from concurrent.futures import ProcessPoolExecutor as _PPE_run
            try:
                import torch as _torch_run
                _n_gpus_run = max(_torch_run.cuda.device_count(), 1)
            except Exception:
                _n_gpus_run = 1
            _wpg_run = self.cfg.workers_per_gpu
            _ctx_run = _mpp_run.get_context("spawn")
            _shared_pools = [
                _PPE_run(
                    max_workers=_wpg_run,
                    mp_context=_ctx_run,
                    initializer=_gpu_pool_initializer,
                    initargs=(gpu_id,),
                )
                for gpu_id in range(_n_gpus_run)
            ]
            logger.info(
                "[Persistent pools] %d GPU(s) × %d workers = %d processes (shared across %d batches)",
                _n_gpus_run, _wpg_run, _n_gpus_run * _wpg_run, n_batches,
            )
            try:
                for i in range(0, len(subjects_todo), batch_size):
                    batch = subjects_todo[i : i + batch_size]
                    batch_num = i // batch_size + 1
                    logger.info(
                        "=== Parallel batch %d/%d: %d subjects ===",
                        batch_num, n_batches, len(batch),
                    )
                    try:
                        scores = self._run_subjects_batch_parallel(
                            batch, ds, words_df, emb, out_dir, model_revision,
                            ch_csv, subj_csv, tc_csv, lw_csv, lw_tc_csv,
                            gpu_pools=_shared_pools, n_gpus=_n_gpus_run,
                        )
                        all_scores.extend(scores)
                    except Exception:
                        logger.error("Parallel batch %s failed", batch, exc_info=True)
            finally:
                for _p in _shared_pools:
                    _p.shutdown(wait=False, cancel_futures=True)
        else:
            for subj in subjects_todo:
                logger.info("--- Subject: %s ---", subj)
                try:
                    scores = self._run_subject(subj, ds, words_df, emb)
                    all_scores.extend(scores)
                    self._save_subject_scores(
                        subj, scores, out_dir, model_revision,
                        ch_csv, subj_csv, tc_csv, lw_csv, lw_tc_csv,
                    )
                except Exception:
                    logger.error("Subject %s failed", subj, exc_info=True)

        # Mean time course across all subjects
        if tc_csv.exists():
            try:
                tc_all = pd.read_csv(tc_csv)
                tc_all = tc_all[tc_all["revision"] == model_revision]
                if not tc_all.empty:
                    mean_tc = (
                        tc_all.groupby("time_s")
                        .agg(
                            pearsonr=("pearsonr", "mean"),
                            sem_pearsonr=("pearsonr", "sem"),
                            spearmanr=("spearmanr", "mean"),
                            sem_spearmanr=("spearmanr", "sem"),
                            n_channels=("pearsonr", "count"),
                        )
                        .reset_index()
                    )
                    mean_tc["revision"] = model_revision
                    mean_tc_csv = out_dir / "megmasc_mean_timecourse.csv"
                    mean_tc.to_csv(
                        mean_tc_csv, mode="a",
                        header=not mean_tc_csv.exists(), index=False,
                    )
            except Exception:
                logger.warning("Mean timecourse aggregation failed", exc_info=True)

        return MegMascEncoderResult(
            model_name=self.cfg.model.model_name,
            model_revision=self.cfg.model.model_revision,
            n_words=len(words),
            channel_scores=all_scores,
            n_subjects_expected=len(subjects),
        )

    # ------------------------------------------------------------------
    # Word loading
    # ------------------------------------------------------------------

    def _load_all_words(self) -> pd.DataFrame:
        """Canonical word list across all four stories; assign global word_idx.

        Story text is identical across subjects, so this list drives a single
        LLM feature-extraction pass. Onsets are recording-specific and filled
        in per-run by :meth:`_load_subject_epoch_data`; the short-word filter
        is likewise applied there (durations are not known here).
        """
        ds = MegMascDataset(
            self.cfg.data.bids_root, stories=self.cfg.data.stories
        )
        stories = ds.available_stories()
        if not stories:
            return pd.DataFrame()

        dfs = []
        global_idx = 0
        for story in stories:
            try:
                df = load_story_words(self.cfg.data.bids_root, story)
            except FileNotFoundError:
                logger.warning("No recording found for story %s — skipping", story)
                continue

            df = df.sort_values("word_index").reset_index(drop=True)
            df["global_word_idx"] = np.arange(
                global_idx, global_idx + len(df), dtype=np.int32
            )
            global_idx += len(df)
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, words: list[str]) -> np.ndarray:
        """Extract LLM embeddings for the full (English) word list."""
        dcfg = self.cfg.data
        mcfg = self.cfg.model
        layer_index: str | int = "all" if self.cfg.layerwise else "last"
        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

        try:
            emb = contextual_word_embeddings_hf_fulltext(
                words,
                mcfg.model_name,
                layer_index=layer_index,
                revision=mcfg.model_revision,
                device=mcfg.device,
                max_length=dcfg.max_length,
                cache_dir=dcfg.feature_cache_dir,
                section_id=0,
                lang=self.cfg.lang,
                tokenizer_no_revision=self.cfg.tokenizer_no_revision,
                n_params=mcfg.n_params,
                trust_remote_code=self.cfg.trust_remote_code,
                dtype=surp_dtype,
            )
        except ValueError as exc:
            if "fast tokenizer" not in str(exc):
                raise
            logger.warning("Falling back to per-word extraction (slow tokenizer): %s", exc)
            emb = contextual_word_embeddings_hf(
                words,
                mcfg.model_name,
                layer_index=layer_index,
                revision=mcfg.model_revision,
                device=mcfg.device,
                context_words=dcfg.context_words,
                max_length=dcfg.max_length,
                cache_dir=dcfg.feature_cache_dir,
                section_id=0,
                lang=self.cfg.lang,
                tokenizer_no_revision=self.cfg.tokenizer_no_revision,
                n_params=mcfg.n_params,
                trust_remote_code=self.cfg.trust_remote_code,
                dtype=surp_dtype,
            )
        return emb  # (n_words, [n_layers,] D)

    # ------------------------------------------------------------------
    # Per-subject epoch loading
    # ------------------------------------------------------------------

    def _load_subject_epoch_data(
        self,
        subject: str,
        ds: MegMascDataset,
        words_df: pd.DataFrame,
        emb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Load (or restore from cache) word epochs for one subject.

        Returns (emb_indexed, epochs, story_groups, times, word_indices) or None.
          emb_indexed : (n_valid_words, [n_layers,] D)
          epochs      : (n_valid_words, n_channels, n_times)
          story_groups: (n_valid_words,) int — story index for LOSO CV
          times       : (n_times,) float — seconds relative to word onset
          word_indices: (n_valid_words,) int — indexes into ``words_df``/``emb``
            (row order preserved from epoch extraction) — used to join OOF
            predictions back to word text/onset for the DLT-cost analysis.
        """
        dcfg = self.cfg.data

        # All (session, story) runs available for this subject
        runs_df = ds.runs()
        subj_runs = runs_df[runs_df["subject"] == subject]
        if subj_runs.empty:
            logger.warning("  [%s] No EEG runs found — skipping", subject)
            return None

        cache_key = _epoch_cache_key(dcfg)
        epoch_cache_path: Path | None = (
            Path(dcfg.epoch_cache_dir) / f"{subject}_{cache_key}.npz"
            if dcfg.epoch_cache_dir else None
        )

        if epoch_cache_path is not None and epoch_cache_path.exists():
            logger.info("  Loading cached epochs: %s", epoch_cache_path.name)
            epochs_all, groups_all, times, word_indices_all = _load_epoch_cache(
                epoch_cache_path
            )
        else:
            # Build a sorted list of unique stories for stable CV group indices.
            stories_seen = sorted(subj_runs["story"].unique())
            story_to_group = {s: i for i, s in enumerate(stories_seen)}

            # Global story list (same order as words_df) for group lookup
            global_stories = words_df["story"].unique().tolist()
            global_story_to_group = {s: story_to_group.get(s, -1) for s in global_stories}

            all_epochs: list[np.ndarray] = []
            all_word_indices: list[np.ndarray] = []
            all_groups: list[int] = []
            times: np.ndarray | None = None

            # Canonical (word_index -> global_word_idx) map per story.
            canon_by_story = {
                s: dict(zip(g["word_index"].astype(int), g["global_word_idx"].astype(int)))
                for s, g in words_df.groupby("story")
            }

            for _, run in subj_runs.iterrows():
                story = run["story"]
                canon_map = canon_by_story.get(story)
                if not canon_map:
                    logger.warning("  Story %s: absent from canonical word list — skipping", story)
                    continue

                # This recording's own word onsets (recording-relative seconds).
                run_words = load_run_words(run["events_tsv"])
                if dcfg.min_word_duration > 0:
                    run_words = run_words[
                        run_words["duration"] >= dcfg.min_word_duration
                    ].copy()
                run_words = run_words[run_words["word_index"].isin(canon_map)]
                if run_words.empty:
                    logger.warning("  Story %s: no aligned words in %s — skipping",
                                   story, Path(run["events_tsv"]).name)
                    continue

                run_words = run_words.sort_values("word_index")
                global_indices = run_words["word_index"].astype(int).map(canon_map).to_numpy()
                if global_indices.max() >= len(emb):
                    logger.warning(
                        "  Story %s: word index out of bounds (max=%d, emb size=%d)",
                        story, global_indices.max(), len(emb),
                    )
                    continue
                word_onsets = run_words["onset"].to_numpy(dtype=np.float64)

                try:
                    meg, meg_sfreq = load_meg_run(run["con"])   # (n_channels, n_samples), Hz
                except (FileNotFoundError, OSError) as exc:
                    logger.warning("  Story %s: MEG not readable: %s", story, exc)
                    continue

                # Bandpass at the native (1000 Hz) rate BEFORE downsampling so
                # the filter sees the correct Nyquist frequency. No notch: the
                # [0.1, 40] Hz band already excludes the 50 Hz line.
                if dcfg.l_freq is not None and dcfg.h_freq is not None:
                    meg = _bandpass_eeg(meg, meg_sfreq, dcfg.l_freq, dcfg.h_freq)

                # Polyphase downsample to target_sfreq (same method as MNE's
                # raw.resample()).
                if dcfg.target_sfreq < meg_sfreq:
                    from math import gcd as _gcd
                    from scipy.signal import resample_poly as _resample_poly
                    up = int(round(dcfg.target_sfreq))
                    down = int(round(meg_sfreq))
                    g = _gcd(up, down)
                    meg = _resample_poly(meg, up // g, down // g, axis=-1).astype(np.float32)

                if dcfg.robust_scale:
                    meg = _robust_scale_clamp(meg, dcfg.clamp)

                epochs, valid_mask, epoch_times = _extract_word_epochs(
                    meg,
                    dcfg.target_sfreq,
                    word_onsets,
                    dcfg.tmin,
                    dcfg.tmax,
                    dcfg.baseline,
                )
                n_valid = int(valid_mask.sum())
                logger.info(
                    "  %s/%s story %s: %d / %d words extracted",
                    run["subject"], run["session"], story, n_valid, len(word_onsets),
                )
                if n_valid == 0:
                    continue

                if times is None:
                    times = epoch_times

                group_idx = story_to_group.get(story, -1)
                all_epochs.append(epochs)
                all_word_indices.append(global_indices[valid_mask])
                all_groups.extend([group_idx] * n_valid)

            if not all_epochs:
                logger.warning("  [%s] No valid epochs — skipping", subject)
                return None

            epochs_all = np.concatenate(all_epochs, axis=0)         # (N, C, T)
            word_indices_all = np.concatenate(all_word_indices)      # (N,)
            groups_all = np.array(all_groups, dtype=np.int32)        # (N,)

            if epoch_cache_path is not None:
                _save_epoch_cache(
                    epoch_cache_path, epochs_all, groups_all, times, word_indices_all
                )
                logger.info(
                    "  Saved epoch cache (%d words, %d ch) → %s",
                    len(word_indices_all), epochs_all.shape[1], epoch_cache_path.name,
                )

        n_total, n_ch, n_times_val = epochs_all.shape
        n_groups = int(np.unique(groups_all).size)
        logger.info(
            "  %s: %d words, %d channels, %d time bins @ %.1f Hz (%d CV groups/stories)",
            subject, n_total, n_ch, n_times_val, dcfg.target_sfreq, n_groups,
        )

        if n_total < 20:
            logger.warning("  [%s] Too few epochs (%d) — skipping", subject, n_total)
            return None

        emb_indexed = emb[word_indices_all]
        return emb_indexed, epochs_all, groups_all, times, word_indices_all

    # ------------------------------------------------------------------
    # Sequential subject path
    # ------------------------------------------------------------------

    def _run_subject(
        self,
        subject: str,
        ds: MegMascDataset,
        words_df: pd.DataFrame,
        emb: np.ndarray,
    ) -> list[ChannelScore]:
        data = self._load_subject_epoch_data(subject, ds, words_df, emb)
        if data is None:
            return []
        emb_all, epochs_all, groups_all, times, word_indices_all = data
        scores = self._regress_channels(
            subject, emb_all, epochs_all, times, groups_all,
            save_word_predictions=self.cfg.save_word_predictions,
        )
        if self.cfg.save_word_predictions:
            self._save_word_predictions(
                subject, scores,
                words_df=words_df, word_indices=word_indices_all, times=times,
                epochs=epochs_all,
            )
        return scores

    def _save_word_predictions(
        self,
        subject: str,
        scores: list[ChannelScore],
        *,
        words_df: pd.DataFrame,
        word_indices: np.ndarray,
        times: np.ndarray,
        epochs: np.ndarray,
    ) -> None:
        """Save each channel's best-layer AND last-layer out-of-fold predicted
        lag matrix, plus the actual observed EEG (needed to score any post-hoc
        word subset, e.g. an easy/hard DLT-cost split — a prediction alone
        isn't scoreable).

        Only these two layers are ever refit with OOF capture (see
        ``ChannelScore.last_layer_time_course`` / the two-pass MP refit in
        ``_run_subjects_batch_parallel``) — holding a (n_words × n_times)
        array for every layer of e.g. a 49-layer model would be enormous, and
        in practice only the winning layer and the final representation are
        ever compared downstream.

        Mirrors :meth:`broderick.BroderickEegEncoder._save_word_predictions`
        — one compressed ``.npz`` per (subject, model revision): all channels
        stacked as ``(n_channels, n_words, n_times)`` for
        ``oof_predictions_best``, ``oof_predictions_last``, and ``y_true``,
        plus word text / onset / channel index / time-axis / layer-index
        metadata for a later per-word DLT-cost join.
        """
        channels_with_oof = [
            c for c in scores
            if c.time_course is not None and c.time_course.oof_predictions is not None
            and c.last_layer_time_course is not None and c.last_layer_time_course.oof_predictions is not None
        ]
        if not channels_with_oof:
            logger.warning("  save_word_predictions=True but no OOF predictions captured for %s", subject)
            return

        rows = words_df.iloc[word_indices]
        n_words = len(rows)
        best_stacked = np.stack(
            [c.time_course.oof_predictions for c in channels_with_oof], axis=0
        )  # (n_channels, n_words, n_times)
        last_stacked = np.stack(
            [c.last_layer_time_course.oof_predictions for c in channels_with_oof], axis=0
        )  # (n_channels, n_words, n_times)
        if best_stacked.shape[1] != n_words:
            logger.warning(
                "  Word-count mismatch for %s (%d words vs %d OOF rows) — skipping save",
                subject, n_words, best_stacked.shape[1],
            )
            return

        name_to_idx = {c.name: ci for ci, c in enumerate(channels_with_oof)}
        ch_idx = [int(name.replace("ch_", "")) for name in name_to_idx]
        y_true = np.stack(
            [epochs[:, ci, :] for ci in ch_idx], axis=0
        ).astype(np.float32)  # (n_channels, n_words, n_times)

        best_layer_idx = np.array(
            [c.best_layer_index() if c.best_layer_index() is not None else -1 for c in channels_with_oof]
        )
        n_layers_total = len(channels_with_oof[0].layer_time_courses) if channels_with_oof[0].layer_time_courses else -1

        out_dir = word_predictions_dir(self.cfg.output_dir, "megmasc_word_predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_revision = (self.cfg.model.model_revision or "main").replace("/", "_")
        safe_model = self.cfg.model.model_name.replace("/", "_")
        out_path = out_dir / f"{safe_model}_{model_revision}_{subject}.npz"
        np.savez_compressed(
            out_path,
            oof_predictions_best=best_stacked,
            oof_predictions_last=last_stacked,
            y_true=y_true,
            channel_names=np.array([c.name for c in channels_with_oof]),
            best_layer_idx=best_layer_idx,
            last_layer_idx=np.array(n_layers_total - 1),
            words=rows["word"].astype(str).to_numpy(),
            word_idx=np.asarray(word_indices),
            onsets=rows["onset"].to_numpy(),
            times=times,
            model=np.array(self.cfg.model.model_name),
            revision=np.array(self.cfg.model.model_revision or "main"),
            n_params=np.array(self.cfg.model.n_params),
        )
        logger.info("  Saved word-level predictions (best+last layer, %s) → %s", best_stacked.shape, out_path)

    # ------------------------------------------------------------------
    # Flat (subject × layer) parallel GPU path
    # ------------------------------------------------------------------

    def _run_subjects_batch_parallel(
        self,
        batch: list[str],
        ds: MegMascDataset,
        words_df: pd.DataFrame,
        emb: np.ndarray,
        out_dir: Path,
        model_revision: str,
        ch_csv: Path,
        subj_csv: Path,
        tc_csv: Path,
        lw_csv: Path,
        lw_tc_csv: Path,
        gpu_pools: list | None = None,
        n_gpus: int | None = None,
    ) -> list[ChannelScore]:
        """Process a batch of subjects with flat (subject × layer) GPU parallelism.

        Uses spawned processes (not threads) so each worker gets its own GIL.
        While one worker's GPU kernel executes, others run CPU numpy work in
        parallel. Large arrays (embeddings, per-subject EEG) are placed in OS
        shared memory once, read zero-copy by all workers.

        Multi-GPU note: Jean Zay H100 nodes have 4 GPUs. Since --exclusive is
        mandatory for MPS, the entire node is billed regardless. We therefore
        use all visible GPUs: tasks are distributed across GPUs in round-robin
        so each GPU runs ~n_tasks/n_gpus concurrent workers via MPS.

        SLURM header required:
          #SBATCH --gres=gpu:4   (all 4 H100s on the node)
          #SBATCH -C mps         (SLURM starts MPS daemon per GPU automatically)
          #SBATCH --exclusive    (mandatory for MPS; full node billed either way)
        """
        import multiprocessing as _mpp
        from multiprocessing import shared_memory as _shmm
        from concurrent.futures import ProcessPoolExecutor, as_completed

        rcfg = self.cfg.regression
        layerwise = emb.ndim == 3
        n_layers = emb.shape[1] if layerwise else 1

        # Detect available GPUs — SLURM sets CUDA_VISIBLE_DEVICES.
        if n_gpus is None:
            try:
                import torch as _torch
                n_gpus = _torch.cuda.device_count()
            except Exception:
                n_gpus = 1
            n_gpus = max(n_gpus, 1)
        logger.info("[MP Batch] %d GPU(s) visible (CUDA_VISIBLE_DEVICES=%s)",
                    n_gpus, __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "all"))
        _pools_owned = gpu_pools is None  # whether this call must shut down the pools

        # Step 1: load epoch data for all subjects in parallel (I/O-bound).
        # ThreadPoolExecutor is safe here: _load_subject_epoch_data only reads
        # .npz files and slices numpy arrays — no CUDA, no GIL-heavy sections.
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

        def _load_one(subj: str) -> tuple[str, tuple | None]:
            try:
                return subj, self._load_subject_epoch_data(subj, ds, words_df, emb)
            except Exception:
                logger.error("[%s] epoch loading failed", subj, exc_info=True)
                return subj, None

        batch_data: dict[str, tuple] = {}
        n_io_workers = min(len(batch), 8)  # cap at 8 threads; disk I/O saturates quickly
        with ThreadPoolExecutor(max_workers=n_io_workers) as pool:
            futs = {pool.submit(_load_one, s): s for s in batch}
            for fut in _as_completed(futs):
                subj, data = fut.result()
                if data is not None:
                    logger.info("--- Subject: %s --- loaded", subj)
                    batch_data[subj] = data

        if not batch_data:
            return []

        # Steps 2 & 3: place per-subject embeddings AND EEG in shared memory.
        #
        # KEY: emb_all from _load_subject_epoch_data is already indexed to this
        # subject's words (emb[word_indices]). We must NOT use the global `emb`
        # here — it has 58993 rows while the subject's EEG has only N_subj rows,
        # which causes IndexError in cv_ridge_score_lag_matrix_gpu_batched.
        #
        # Embeddings are transposed (n_subj_words, n_layers, D) → (n_layers, n_subj_words, D)
        # so each worker's slice emb_full[li] is a contiguous (n_subj_words, D) block.
        shm_embs: dict[str, _shmm.SharedMemory] = {}
        emb_meta: dict[str, tuple] = {}
        shm_eegs: dict[str, _shmm.SharedMemory] = {}
        eeg_meta: dict[str, tuple] = {}
        # Lightweight metadata kept after the heavy arrays are freed.
        # (n_ch, groups_all, times) — all that's needed for task building and result assembly.
        batch_meta: dict[str, tuple] = {}

        # Stream: pop each subject from batch_data immediately after its shm is built.
        # This ensures only ONE subject's embedding array is live at a time during the
        # loop, rather than holding all N subjects in memory simultaneously alongside
        # the growing shm.  For LLaMA-3.1-8B × 34 subjects this saves ~140 GiB peak RAM.
        for subj in list(batch_data.keys()):
            emb_all, eeg_all, groups_all, times, word_indices = batch_data.pop(subj)

            if layerwise:
                emb_c = np.ascontiguousarray(emb_all.transpose(1, 0, 2).astype(np.float32))
            else:
                emb_c = np.ascontiguousarray(emb_all.astype(np.float32))
            del emb_all  # free immediately — emb_c is the copy that goes into shm
            shm_e = _shmm.SharedMemory(create=True, size=emb_c.nbytes)
            np.ndarray(emb_c.shape, dtype=emb_c.dtype, buffer=shm_e.buf)[:] = emb_c
            shm_embs[subj] = shm_e
            emb_meta[subj] = (shm_e.name, emb_c.shape, str(emb_c.dtype))
            del emb_c

            n_ch_subj = int(eeg_all.shape[1])
            eeg_c = np.ascontiguousarray(eeg_all.astype(np.float32))
            del eeg_all
            shm_y = _shmm.SharedMemory(create=True, size=eeg_c.nbytes)
            np.ndarray(eeg_c.shape, dtype=eeg_c.dtype, buffer=shm_y.buf)[:] = eeg_c
            shm_eegs[subj] = shm_y
            eeg_meta[subj] = (shm_y.name, eeg_c.shape, str(eeg_c.dtype))
            del eeg_c

            batch_meta[subj] = (n_ch_subj, groups_all, times, word_indices)

        # batch_data is now empty (all entries popped above).
        del batch_data
        # emb (the full global embedding array, e.g. 32 GiB for LLaMA-3.1-8B) is a
        # parameter of this function.  Deleting the local name here releases this
        # function's reference; the caller (run()) still holds one, but that reference
        # is not needed during the parallel regression phase and will be freed by the
        # GC after run() processes all batches.  This is the best we can do without a
        # larger refactor of the calling convention.
        del emb

        # Step 4: build flat task list — one dict per (subject, layer).
        # gpu_id assigned in round-robin so tasks spread evenly across all GPUs.
        tasks: list[dict] = []
        task_counter = 0
        for subj, (n_ch_subj, groups_all, times, _word_indices) in batch_meta.items():
            n_splits = min(rcfg.n_splits, int(np.unique(groups_all).size))
            kw = dict(
                n_splits=n_splits,
                alpha=rcfg.alpha,
                scale=rcfg.scale,
                standardize_target=True,
                # First pass: no OOF capture for any layer (cheap — every
                # layer must be scored anyway to find the winning one). A
                # second, targeted pass below refits only the best + last
                # layer per subject with OOF, once the winning layer is known.
                return_oof_predictions=False,
                random_state=rcfg.random_state,
                groups=groups_all,
            )
            emb_shm_name, emb_shm_shape, emb_shm_dtype = emb_meta[subj]
            eeg_shm_name, eeg_shape, eeg_dtype = eeg_meta[subj]
            for li in range(n_layers):
                tasks.append({
                    "subj": subj,
                    "li": li,
                    "gpu_id": task_counter % n_gpus,   # round-robin GPU assignment
                    "times": times,
                    "batched_kw": kw,
                    "emb_shm_name": emb_shm_name,
                    "emb_shape": emb_shm_shape,
                    "emb_dtype": emb_shm_dtype,
                    "emb_layerwise": layerwise,
                    "eeg_shm_name": eeg_shm_name,
                    "eeg_shape": eeg_shape,
                    "eeg_dtype": eeg_dtype,
                })
                task_counter += 1

        # Pool size: n_gpus × workers_per_gpu concurrent tasks.
        # With MPS each GPU can run workers_per_gpu tasks simultaneously.
        # workers_per_gpu=10 → 4 GPUs × 10 = 40 simultaneous tasks.
        # Each worker peaks at ~5 GB on H100 80 GB, so 10 × 5 = 50 GB (62%).
        # Safe because per-GPU pools guarantee exactly workers_per_gpu processes
        # per GPU with no cross-GPU context accumulation.
        workers_per_gpu = self.cfg.workers_per_gpu
        n_workers = min(len(tasks), n_gpus * workers_per_gpu)
        logger.info(
            "[MP Batch] %d subjects × %d layers = %d tasks (%d GPU(s) × %d workers = pool %d)",
            len(batch_meta), n_layers, len(tasks), n_gpus, workers_per_gpu, n_workers,
        )

        # Step 5: submit all tasks; assemble + save each subject as it completes
        layer_results: dict[str, dict[int, list]] = {s: {} for s in batch_meta}
        layer_done: dict[str, int] = {s: 0 for s in batch_meta}
        all_scores: list[ChannelScore] = []

        # One pool per GPU, each worker pre-pinned via _gpu_pool_initializer.
        # This prevents cross-GPU CUDA context accumulation: a worker in pool[0]
        # only ever touches GPU 0 and never creates a context on GPUs 1-3.
        # When gpu_pools is injected by the caller (persistent across batches),
        # we skip creation here and skip shutdown in the finally block.
        if gpu_pools is None:
            ctx = _mpp.get_context("spawn")  # spawn required: CUDA cannot be forked
            gpu_pools = [
                ProcessPoolExecutor(
                    max_workers=workers_per_gpu,
                    mp_context=ctx,
                    initializer=_gpu_pool_initializer,
                    initargs=(gpu_id,),
                )
                for gpu_id in range(n_gpus)
            ]
        futures: dict = {}
        for tc, task in enumerate(tasks):
            gpu_id = tc % n_gpus
            task["gpu_id"] = gpu_id
            fut = gpu_pools[gpu_id].submit(_mp_worker_fit, task)
            futures[fut] = (task["subj"], task["li"])

        # Subjects whose scores are assembled but still need a best+last-layer
        # OOF refit (stage 2 below) before their word predictions can be saved.
        pending_word_pred: dict[str, list[ChannelScore]] = {}

        try:
            for fut in as_completed(futures):
                subj, li = futures[fut]
                try:
                    _, _, tcs = fut.result()
                except Exception:
                    logger.error("[%s] layer %d failed", subj, li, exc_info=True)
                    layer_done[subj] += 1
                    continue

                layer_results[subj][li] = tcs
                layer_done[subj] += 1
                _alphas_ch = np.array(
                    [float(tc.ridge_alpha[0]) for tc in tcs if tc.ridge_alpha is not None],
                    dtype=np.float64,
                )
                if _alphas_ch.size:
                    logger.info(
                        "  [%s] layer %d/%d done (MP) — chosen alpha (%d ch): "
                        "median=%.3g  min=%.3g  max=%.3g",
                        subj, li + 1, n_layers, _alphas_ch.size,
                        float(np.median(_alphas_ch)), float(_alphas_ch.min()), float(_alphas_ch.max()),
                    )
                else:
                    logger.info("  [%s] layer %d/%d done (MP)", subj, li + 1, n_layers)

                # As soon as all layers for a subject finish, assemble and save.
                if layer_done[subj] == n_layers:
                    n_ch, groups_all, subj_times, subj_word_indices = batch_meta[subj]
                    try:
                        scores = self._assemble_scores(
                            subj, layer_results[subj], n_layers, n_ch
                        )
                        all_scores.extend(scores)
                        self._save_subject_scores(
                            subj, scores, out_dir, model_revision,
                            ch_csv, subj_csv, tc_csv, lw_csv, lw_tc_csv,
                        )
                        if self.cfg.save_word_predictions:
                            pending_word_pred[subj] = scores
                    except Exception:
                        logger.error(
                            "[%s] score assembly/save failed", subj, exc_info=True
                        )
                    # Free this subject's raw per-layer fits as soon as it's
                    # done rather than waiting for the whole batch to finish.
                    layer_results.pop(subj, None)

            # Stage 2: for subjects needing word predictions, refit ONLY the
            # subject-level best layer (mean pearsonr across channels, same
            # definition as the checkpoint-level "best_layer" column) and the
            # last layer, this time with OOF capture. Two extra tasks per
            # subject (one if best == last) instead of the full n_layers —
            # the first pass above already had to score every layer anyway,
            # so this adds negligible extra work for the OOF we actually keep.
            if pending_word_pred:
                refit_futures: dict = {}
                refit_plan: dict[str, tuple[int, int]] = {}
                for subj, scores in pending_word_pred.items():
                    n_ch_subj, groups_all, subj_times, subj_word_indices = batch_meta[subj]
                    per_layer_mean = np.nanmean(
                        np.array([
                            [s.pearsonr for s in c.layer_scores] for c in scores if c.layer_scores
                        ]),
                        axis=0,
                    )
                    best_li = int(np.nanargmax(per_layer_mean)) if per_layer_mean.size else 0
                    last_li = n_layers - 1
                    refit_plan[subj] = (best_li, last_li)

                    n_splits = min(rcfg.n_splits, int(np.unique(groups_all).size))
                    oof_kw = dict(
                        n_splits=n_splits, alpha=rcfg.alpha, scale=rcfg.scale,
                        standardize_target=True, return_oof_predictions=True,
                        random_state=rcfg.random_state, groups=groups_all,
                    )
                    emb_shm_name, emb_shm_shape, emb_shm_dtype = emb_meta[subj]
                    eeg_shm_name, eeg_shape, eeg_dtype = eeg_meta[subj]
                    for li in {best_li, last_li}:
                        refit_task = {
                            "subj": subj, "li": li, "gpu_id": 0,
                            "times": subj_times, "batched_kw": oof_kw,
                            "emb_shm_name": emb_shm_name, "emb_shape": emb_shm_shape,
                            "emb_dtype": emb_shm_dtype, "emb_layerwise": layerwise,
                            "eeg_shm_name": eeg_shm_name, "eeg_shape": eeg_shape,
                            "eeg_dtype": eeg_dtype,
                        }
                        gpu_id = len(refit_futures) % n_gpus
                        refit_fut = gpu_pools[gpu_id].submit(_mp_worker_fit, refit_task)
                        refit_futures[refit_fut] = (subj, li)

                refit_results: dict[str, dict[int, list]] = {s: {} for s in pending_word_pred}
                for fut in as_completed(refit_futures):
                    subj, li = refit_futures[fut]
                    try:
                        _, _, tcs = fut.result()
                        refit_results[subj][li] = tcs
                    except Exception:
                        logger.error("[%s] OOF refit layer %d failed", subj, li, exc_info=True)

                for subj, scores in pending_word_pred.items():
                    best_li, last_li = refit_plan[subj]
                    best_tcs = refit_results.get(subj, {}).get(best_li)
                    last_tcs = refit_results.get(subj, {}).get(last_li)
                    if best_tcs is None or last_tcs is None:
                        logger.warning("[%s] missing OOF refit results — skipping word predictions", subj)
                        continue
                    for ci, c in enumerate(scores):
                        c.time_course = best_tcs[ci]
                        c.last_layer_time_course = last_tcs[ci]
                    n_ch, groups_all, subj_times, subj_word_indices = batch_meta[subj]
                    eeg_shm_name, eeg_shape, eeg_dtype = eeg_meta[subj]
                    _shm = shm_eegs[subj]
                    subj_epochs = np.array(
                        np.ndarray(eeg_shape, dtype=eeg_dtype, buffer=_shm.buf)
                    )
                    try:
                        self._save_word_predictions(
                            subj, scores,
                            words_df=words_df, word_indices=subj_word_indices,
                            times=subj_times, epochs=subj_epochs,
                        )
                    except Exception:
                        logger.error("[%s] word-prediction save failed", subj, exc_info=True)
        finally:
            if _pools_owned:
                for p in gpu_pools:
                    p.shutdown(wait=False, cancel_futures=True)
            for shm in shm_embs.values():
                shm.close()
                shm.unlink()
            for shm in shm_eegs.values():
                shm.close()
                shm.unlink()

        return all_scores

    # ------------------------------------------------------------------
    # Score assembly from flat MP results
    # ------------------------------------------------------------------

    def _assemble_scores(
        self,
        subject: str,
        layer_tcs: dict[int, list],   # {layer_idx: [TimeResolvedScore per channel]}
        n_layers: int,
        n_channels: int,
    ) -> list[ChannelScore]:
        layerwise = n_layers > 1

        if layerwise:
            missing = [li for li in range(n_layers) if li not in layer_tcs]
            if missing:
                logger.warning(
                    "[%s] skipping assembly: %d/%d layers missing",
                    subject, len(missing), n_layers,
                )
                return []

        results: list[ChannelScore] = []
        for ci in range(n_channels):
            ch_name = f"ch_{ci:02d}"
            if layerwise:
                per_layer: list[TimeResolvedScore] = [
                    layer_tcs[li][ci] for li in range(n_layers)
                ]
                layer_scores = [tc.to_mean_cvScore() for tc in per_layer]
                best_li = int(np.nanargmax([s.pearsonr for s in layer_scores]))
                results.append(ChannelScore(
                    subject=subject, name=ch_name, region=None,
                    score=layer_scores[-1],
                    time_course=per_layer[best_li],
                    layer_scores=layer_scores,
                    layer_time_courses=per_layer,
                ))
            else:
                tc = layer_tcs[0][ci]
                results.append(ChannelScore(
                    subject=subject, name=ch_name, region=None,
                    score=tc.to_mean_cvScore(),
                    time_course=tc,
                ))
        return results

    # ------------------------------------------------------------------
    # Channel regression dispatch
    # ------------------------------------------------------------------

    def _regress_channels(
        self,
        subject: str,
        emb: np.ndarray,       # (N, [L,] D)
        eeg: np.ndarray,       # (N, C, T)
        times: np.ndarray,     # (T,)
        groups: np.ndarray,    # (N,)
        save_word_predictions: bool = False,
    ) -> list[ChannelScore]:
        # OOF capture isn't wired through the batched GPU path — force the
        # per-channel CPU path when word-level predictions are requested,
        # same restriction as Broderick.
        if not save_word_predictions and self.cfg.regression_device != "cpu" and self.cfg.use_lag_matrix:
            return self._regress_channels_gpu_batched(subject, emb, eeg, times, groups)
        return self._regress_channels_cpu(
            subject, emb, eeg, times, groups, save_word_predictions=save_word_predictions,
        )

    def _regress_channels_gpu_batched(
        self,
        subject: str,
        emb: np.ndarray,
        eeg: np.ndarray,
        times: np.ndarray,
        groups: np.ndarray,
    ) -> list[ChannelScore]:
        """Fit all channels simultaneously via himalaya GPU ridge (one call per layer)."""
        rcfg = self.cfg.regression
        layerwise = emb.ndim == 3
        n_layers = emb.shape[1] if layerwise else 1
        n_channels = eeg.shape[1]
        n_splits = min(rcfg.n_splits, int(np.unique(groups).size))

        logger.info(
            "  [%s] GPU batched: %d words, %d ch, %d lags, %d layers, %d folds",
            subject, emb.shape[0], n_channels, eeg.shape[2], n_layers, n_splits,
        )

        try:
            from himalaya.backend import set_backend as _set_himalaya_backend
            _set_himalaya_backend("torch_cuda")
        except Exception as exc:
            raise RuntimeError(f"himalaya GPU backend unavailable: {exc}") from exc

        import torch

        batched_kw = dict(
            n_splits=n_splits,
            alpha=rcfg.alpha,
            scale=rcfg.scale,
            standardize_target=True,
            random_state=rcfg.random_state,
            groups=groups,
        )

        layer_tcs: list[list[TimeResolvedScore] | None] = [None] * n_layers

        def _fit_layer(li: int) -> tuple[int, list[TimeResolvedScore]]:
            X_li = emb[:, li, :] if layerwise else emb
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                tcs = cv_ridge_score_lag_matrix_gpu_batched(
                    X_li, eeg, times, **batched_kw
                )
            stream.synchronize()
            return li, tcs

        # torch.linalg.eigh (used inside cv_ridge_score_lag_matrix_gpu_batched)
        # does a one-time, non-thread-safe CUDA/MAGMA backend registration on
        # its first call in the process. If two layer-worker threads race to
        # trigger that first call concurrently, PyTorch raises
        # "RuntimeError: lazy wrapper should be called at most once" and the
        # subject's regression is lost (same fix as podcast.py / broderick.py's
        # GPU-batched regression). Force that one-time init here, single
        # threaded, before spawning the pool below.
        _eigh_warmup = torch.eye(2, device=f"cuda:{torch.cuda.current_device()}")
        torch.linalg.eigh(_eigh_warmup)
        del _eigh_warmup

        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        # Sequential (1 worker), NOT the 4 the EEG encoders use: running >=2
        # concurrent CUDA torch.linalg.eigh calls SIGSEGVs (exit 139) for
        # d_model in {2560, 4096} on the Jean Zay H100 stack — it crashed
        # pythia-2.8b/6.9b, gpt-neo-2.7b, gpt-j-6b and opt-6.7b on the first
        # sweep (the _eigh_warmup only fixes the Python lazy-init race, not the
        # deeper cuSOLVER reentrancy fault). With only 11 subjects the wall-time
        # cost of serial layers is acceptable (family scripts get 6 h).
        n_workers = 1
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_fit_layer, li): li for li in range(n_layers)}
            for fut in _as_completed(futures):
                li, tcs = fut.result()
                layer_tcs[li] = tcs
                _alphas_ch = np.array(
                    [float(tc.ridge_alpha[0]) for tc in tcs if tc.ridge_alpha is not None],
                    dtype=np.float64,
                )
                if _alphas_ch.size:
                    logger.info(
                        "  [%s] layer %d/%d done — chosen alpha (%d ch): "
                        "median=%.3g  min=%.3g  max=%.3g",
                        subject, li + 1, n_layers, _alphas_ch.size,
                        float(np.median(_alphas_ch)), float(_alphas_ch.min()), float(_alphas_ch.max()),
                    )
                else:
                    logger.info("  [%s] layer %d/%d done", subject, li + 1, n_layers)

        results: list[ChannelScore] = []
        for ci in range(n_channels):
            ch_name = f"ch_{ci:02d}"
            if layerwise:
                per_layer = [layer_tcs[li][ci] for li in range(n_layers)]  # type: ignore[index]
                layer_scores = [tc.to_mean_cvScore() for tc in per_layer]
                best_li = int(np.nanargmax([s.pearsonr for s in layer_scores]))
                results.append(ChannelScore(
                    subject=subject, name=ch_name, region=None,
                    score=layer_scores[-1],
                    time_course=per_layer[best_li],
                    layer_scores=layer_scores,
                    layer_time_courses=per_layer,
                ))
            else:
                tc = layer_tcs[0][ci]  # type: ignore[index]
                results.append(ChannelScore(
                    subject=subject, name=ch_name, region=None,
                    score=tc.to_mean_cvScore(),
                    time_course=tc,
                ))

        mean_r = float(np.nanmean([r.best_layer_pearsonr() for r in results]))
        logger.info(
            "  [%s] %d channels — mean best-layer r = %.4f (GPU batched)",
            subject, len(results), mean_r,
        )
        return results

    def _regress_channels_cpu(
        self,
        subject: str,
        emb: np.ndarray,
        eeg: np.ndarray,
        times: np.ndarray,
        groups: np.ndarray,
        save_word_predictions: bool = False,
    ) -> list[ChannelScore]:
        rcfg = self.cfg.regression
        n_channels = eeg.shape[1]
        n_splits = min(rcfg.n_splits, int(np.unique(groups).size))

        reg_kwargs = dict(
            alpha=rcfg.alpha,
            scale=rcfg.scale,
            standardize_target=True,
            random_state=rcfg.random_state,
            device="cpu",
        )

        results: list[ChannelScore] = Parallel(
            n_jobs=self.cfg.n_jobs, backend="loky",
        )(
            delayed(_regress_channel_cpu)(
                ci=ci,
                emb=emb,
                y_ci=eeg[:, ci, :],
                times=times,
                groups=groups,
                ch_name=f"ch_{ci:02d}",
                subject=subject,
                n_splits=n_splits,
                reg_kwargs=reg_kwargs,
                save_word_predictions=save_word_predictions,
            )
            for ci in range(n_channels)
        )

        mean_r = float(np.nanmean([r.best_layer_pearsonr() for r in results]))
        logger.info(
            "  [%s] %d channels — mean best-layer r = %.4f",
            subject, len(results), mean_r,
        )
        return results

    # ------------------------------------------------------------------
    # Incremental CSV saving (thread-safe)
    # ------------------------------------------------------------------

    def _save_subject_scores(
        self,
        subj: str,
        scores: list[ChannelScore],
        out_dir: Path,
        model_revision: str,
        ch_csv: Path,
        subj_csv: Path,
        tc_csv: Path,
        lw_csv: Path,
        lw_tc_csv: Path,
    ) -> None:
        if not scores:
            return

        mean_r = float(np.nanmean([c.best_layer_pearsonr() for c in scores]))
        logger.info(
            "  [%s] %d channels, mean best-layer Pearson r = %.4f",
            subj, len(scores), mean_r,
        )

        rows = [c.to_dict() for c in scores]
        for r in rows:
            r["revision"] = model_revision
        subj_df = pd.DataFrame(rows)

        score_cols = ["best_pearsonr", "best_spearmanr", "pearsonr", "spearmanr"]
        srow: dict = {"subject": subj, "revision": model_revision}
        for col in score_cols:
            if col in subj_df:
                srow[f"{col}_mean"] = subj_df[col].mean()
                srow[f"{col}_std"] = subj_df[col].std()
                srow[f"{col}_count"] = int(subj_df[col].count())

        tc_rows: list[dict] = []
        for c in scores:
            tc_rows.extend(c.timecourse_rows())
        for r in tc_rows:
            r["revision"] = model_revision

        lw_rows: list[dict] = []
        for c in scores:
            if c.layer_scores:
                for li, ls in enumerate(c.layer_scores):
                    lw_rows.append({
                        "subject": subj, "channel": c.name,
                        "revision": model_revision, "layer": li,
                        "pearsonr": ls.pearsonr, "spearmanr": ls.spearmanr,
                    })

        # Per-layer mean TRF curve (mean over channels for this subject)
        layer_tc_buf: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        ref_times: np.ndarray | None = None
        for c in scores:
            if c.layer_time_courses is None:
                continue
            for li, ltc in enumerate(c.layer_time_courses):
                if ref_times is None:
                    ref_times = ltc.window_centers
                layer_tc_buf.setdefault(li, []).append((ltc.pearsonr, ltc.spearmanr))

        lw_tc_rows: list[dict] = []
        if layer_tc_buf and ref_times is not None:
            for li, ch_curves in sorted(layer_tc_buf.items()):
                pr_mat = np.stack([pr for pr, _ in ch_curves])
                sr_mat = np.stack([sr for _, sr in ch_curves])
                for ti, t in enumerate(ref_times):
                    lw_tc_rows.append({
                        "subject": subj, "revision": model_revision,
                        "layer": li, "time_s": float(t),
                        "pearsonr_mean": float(np.nanmean(pr_mat[:, ti])),
                        "pearsonr_std": float(np.nanstd(pr_mat[:, ti])),
                        "spearmanr_mean": float(np.nanmean(sr_mat[:, ti])),
                        "spearmanr_std": float(np.nanstd(sr_mat[:, ti])),
                    })

        with _csv_write_lock:
            subj_df.to_csv(ch_csv, mode="a", header=not ch_csv.exists(), index=False)
            pd.DataFrame([srow]).to_csv(
                subj_csv, mode="a", header=not subj_csv.exists(), index=False,
            )
            if tc_rows:
                pd.DataFrame(tc_rows).to_csv(
                    tc_csv, mode="a", header=not tc_csv.exists(), index=False,
                )
            if lw_rows:
                pd.DataFrame(lw_rows).to_csv(
                    lw_csv, mode="a", header=not lw_csv.exists(), index=False,
                )
            if lw_tc_rows:
                pd.DataFrame(lw_tc_rows).to_csv(
                    lw_tc_csv, mode="a", header=not lw_tc_csv.exists(), index=False,
                )
