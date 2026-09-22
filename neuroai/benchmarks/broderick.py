"""Broderick 2018 EEG encoding benchmark.

Predicts bandpass-filtered (1–8 Hz) scalp EEG from LLM hidden states using
per-channel lag-matrix ridge regression with 5-fold cross-validation.

Pipeline (per subject)
----------------------
1. Load all word timings from Praat TextGrids across 20 segments.
2. Extract LLM features for all words (full-text sliding-window forward pass).
3. For each of the 20 EEG runs:
   a. Load .vhdr → bandpass 1–8 Hz → average reference → downsample to 32 Hz.
   b. Extract word-locked epochs [tmin, tmax]; MNE-style baseline correction.
4. Pool epochs across all runs; run index = CV group (5-fold, 4 runs/fold).
5. Per-channel lag-matrix ridge regression → TimeResolvedScore per channel.
   Brain score = mean over lags of median-over-folds Pearson r (one scalar).
6. Layer sweep: repeat steps 2–5 for each transformer layer; track best layer.

Features
--------
LLM hidden states extracted via
:func:`~neuroai.features.llm.contextual_word_embeddings_hf_fulltext`:
the full concatenated audiobook text (words from all 20 segments in order) is
tokenized once and processed in overlapping fixed-length chunks.  Falls back to
per-word extraction for models with slow (non-fast) tokenizers.

Outputs
-------
Per-checkpoint (written to ``output_dir``):
  broderick_checkpoint_scores.csv      — best-layer summary per checkpoint
  broderick_checkpoint_layerwise.csv   — per-layer Pearson r
  broderick_by_channel.csv             — per-channel scores (all subjects)
  broderick_by_subject.csv             — per-subject mean
  broderick_by_region.csv              — per scalp-region mean
  broderick_electrode_timecourse.csv   — time-resolved r per channel
  broderick_by_channel_layerwise.csv   — per-channel × layer Pearson r (scalar)
  broderick_layerwise_timecourse.csv   — TRF curves (mean±std over channels) per layer
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
from ..datasets.broderick import (
    BroderickDataset,
    eeg_vhdr_path,
    load_all_words,
    load_eeg_run,
    load_segment_words,
)
from ..features.cbr_rnn import contextual_word_embeddings_cbr_rnn
from ..features.llm import (
    contextual_word_embeddings_hf,
    contextual_word_embeddings_hf_fulltext,
)
from ..regressors.ridge import (
    CVScore,
    TimeResolvedScore,
    cv_ridge_score_lag_matrix,
    cv_ridge_score_lag_matrix_gpu_batched,
    cv_ridge_score_windowed,
)
from ..datasets.podcast import extract_word_epochs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BroderickDataConfig:
    """Dataset location and epoch-extraction parameters."""

    bids_root: str
    subjects: list[str] | None = None
    n_runs: int = 20
    # Epoch window (s relative to word onset)
    tmin: float = -0.2
    tmax: float = 1.0
    # MNE-style baseline: subtract mean over [baseline_tmin, 0] from each epoch
    baseline: tuple[float, float] | None = (-0.2, 0.0)
    # Preprocessing
    l_freq: float = 1.0
    h_freq: float = 8.0
    target_sfreq: float = 32.0
    average_ref: bool = True
    drop_bads: bool = True
    # Cache paths
    feature_cache_dir: str | None = None   # LLM embeddings (.npz, keyed by model)
    epoch_cache_dir: str | None = None     # preprocessed EEG epochs (.npz, keyed by preproc params)
    # LLM feature options
    context_words: int | None = None
    max_length: int | None = None


@dataclass
class BroderickBenchmarkConfig:
    """Top-level config for the Broderick EEG encoding benchmark."""

    data: BroderickDataConfig
    model: ModelConfig
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    output_dir: str = "outputs"
    layerwise: bool = True
    use_lag_matrix: bool = True
    # Fallback windowed parameters (only when use_lag_matrix=False)
    window_size: float = 0.05
    step_size: float = 0.025
    n_jobs: int = 40
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    regression_device: str = "cpu"
    n_layer_workers: int = 6
    # Cap on layers fit concurrently on the GPU within one subject's batched
    # ridge fit. None = no cap (all n_layers submitted to the ThreadPoolExecutor
    # at once, gated only by n_layer_workers). Needed for wide-hidden-dim models
    # (e.g. opt-66b: 9216-dim, 65 layers) where n_layer_workers concurrent lag
    # matrices on one GPU OOM even with model sharding across GPUs — the ridge
    # fit itself only ever uses one device.
    layer_chunk_size: int | None = None
    # >1 enables flat (subject × layer) GPU parallelism: all subjects in the batch
    # share the GPU pool simultaneously instead of being processed sequentially.
    n_subjects_batch: int = 5
    # Save the best-layer out-of-fold predicted lag matrix per channel
    # (n_words × n_times, float32) alongside the usual scores — needed for
    # post-hoc per-word difficulty splits (e.g. DLT cost). On by default.
    save_word_predictions: bool = True


# ---------------------------------------------------------------------------
# Result dataclasses  (mirrors podcast.py ElectrodeScore / PodcastEncoderResult)
# ---------------------------------------------------------------------------

@dataclass
class ChannelScore:
    """Encoding score for one EEG channel."""

    subject: str
    name: str
    region: str | None
    score: CVScore
    time_course: TimeResolvedScore | None = None
    layer_scores: list[CVScore] | None = None
    layer_time_courses: list[TimeResolvedScore] | None = None

    # x/y/z are None for scalp EEG (no MNI coordinates)
    x: float | None = None
    y: float | None = None
    z: float | None = None
    hemisphere: str | None = None

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
        base = {
            "subject": self.subject,
            "channel": self.name,
            "region": self.region,
        }
        return [
            {**base,
             "time_s": float(tc.window_centers[t]),
             "pearsonr": float(tc.pearsonr[t]),
             "spearmanr": float(tc.spearmanr[t])}
            for t in range(len(tc.window_centers))
        ]

    def to_dict(self, include_layers: bool = False) -> dict:
        d: dict[str, tp.Any] = {
            "subject": self.subject,
            "channel": self.name,
            "region": self.region,
            "hemisphere": self.hemisphere,
            "x": self.x,
            "y": self.y,
            "z": self.z,
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
class BroderickEncoderResult:
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

    def by_region(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if "region" not in df.columns or df["region"].isna().all():
            return pd.DataFrame()
        cols = ["best_pearsonr", "best_spearmanr", "pearsonr", "spearmanr"]
        agg = (
            df.dropna(subset=["region"])
            .groupby("region")[cols]
            .agg(["mean", "std", "count"])
        )
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        return agg.reset_index()


# ---------------------------------------------------------------------------
# EEG epoch cache helpers
# ---------------------------------------------------------------------------

def _epoch_cache_key(dcfg: "BroderickDataConfig") -> str:
    """MD5 of preprocessing hyperparameters — used to version the epoch cache."""
    params: dict = {
        "l_freq": dcfg.l_freq,
        "h_freq": dcfg.h_freq,
        "target_sfreq": dcfg.target_sfreq,
        "tmin": dcfg.tmin,
        "tmax": dcfg.tmax,
        "baseline": list(dcfg.baseline) if dcfg.baseline is not None else None,
        "average_ref": dcfg.average_ref,
    }
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def _load_epoch_cache(path: Path) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """Load cached epochs. Returns (epochs, groups, times, word_indices, ch_names)."""
    d = np.load(path, allow_pickle=True)
    return (
        d["epochs"],
        d["groups"],
        d["times"],
        d["word_indices"],
        d["ch_names"].tolist(),
    )


def _save_epoch_cache(
    path: Path,
    epochs: np.ndarray,
    groups: np.ndarray,
    times: np.ndarray,
    word_indices: np.ndarray,
    ch_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        epochs=epochs.astype(np.float32),
        groups=groups.astype(np.int32),
        times=times.astype(np.float32),
        word_indices=word_indices.astype(np.int32),
        ch_names=np.array(ch_names),
    )


# ---------------------------------------------------------------------------
# Per-channel regression worker (module-level for joblib pickling)
# ---------------------------------------------------------------------------

def _regress_channel(
    ci: int,
    emb: np.ndarray,        # (n_valid_words, [n_layers,] D)
    y_ci: np.ndarray,       # (n_valid_words, n_times)
    times: np.ndarray,      # (n_times,)
    groups: np.ndarray,     # (n_valid_words,)
    ch_name: str,
    meta: dict,
    subject: str,
    use_lag_matrix: bool,
    window_size: float,
    step_size: float,
    reg_kwargs: dict,
    save_word_predictions: bool = False,
) -> ChannelScore:
    layerwise = emb.ndim == 3

    def _best_li(scores: list[CVScore]) -> int:
        vals = np.asarray([s.pearsonr for s in scores], dtype=float)
        return int(np.nanargmax(vals)) if np.any(np.isfinite(vals)) else 0

    rk = {k: v for k, v in reg_kwargs.items()
          if k not in ("window_size", "step_size", "windowed", "use_lag_matrix")}

    if use_lag_matrix:
        if layerwise:
            n_layers = emb.shape[1]
            layer_tcs: list[TimeResolvedScore] = []
            for li in range(n_layers):
                tc = cv_ridge_score_lag_matrix(
                    emb[:, li, :], y_ci, times, groups=groups, **rk,
                )
                layer_tcs.append(tc)
            layer_scores = [tc.to_mean_cvScore() for tc in layer_tcs]
            best_li = _best_li(layer_scores)
            score = layer_scores[-1]
            time_course: TimeResolvedScore | None = layer_tcs[best_li]
            layer_time_courses: list[TimeResolvedScore] | None = layer_tcs
            if save_word_predictions:
                # Refit only the winning layer with OOF predictions kept — avoids
                # holding a (n_words × n_times) array per layer in memory/CSV.
                time_course = cv_ridge_score_lag_matrix(
                    emb[:, best_li, :], y_ci, times,
                    groups=groups, return_oof_predictions=True, **rk,
                )
        else:
            tc = cv_ridge_score_lag_matrix(
                emb, y_ci, times, groups=groups,
                return_oof_predictions=save_word_predictions, **rk,
            )
            score = tc.to_mean_cvScore()
            time_course = tc
            layer_scores = None
            layer_time_courses = None
    else:
        if layerwise:
            n_layers = emb.shape[1]
            layer_tcs_w: list[TimeResolvedScore] = []
            for li in range(n_layers):
                tw = cv_ridge_score_windowed(
                    emb[:, li, :], y_ci, times,
                    window_size=window_size, step_size=step_size,
                    groups=groups, **rk,
                )
                layer_tcs_w.append(tw)
            layer_scores = [tw.to_mean_cvScore() for tw in layer_tcs_w]
            best_li = _best_li(layer_scores)
            score = layer_scores[-1]
            time_course = layer_tcs_w[best_li]
            layer_time_courses = layer_tcs_w
        else:
            tw = cv_ridge_score_windowed(
                emb, y_ci, times,
                window_size=window_size, step_size=step_size,
                groups=groups, **rk,
            )
            score = tw.to_mean_cvScore()
            time_course = tw
            layer_scores = None
            layer_time_courses = None

    return ChannelScore(
        subject=subject,
        name=ch_name,
        region=meta.get("region"),
        hemisphere=meta.get("hemisphere"),
        x=meta.get("x"),
        y=meta.get("y"),
        z=meta.get("z"),
        score=score,
        time_course=time_course,
        layer_scores=layer_scores,
        layer_time_courses=layer_time_courses,
    )


# Lock for safe concurrent CSV appends in the sequential path.
_csv_write_lock = threading.Lock()


def _mp_worker_fit(task: dict) -> tuple[str, int, list]:
    """Multiprocessing worker for one (subject, layer) ridge regression.

    Runs in a separate spawned process with its own Python interpreter and GIL,
    eliminating the GIL contention that caused ThreadPoolExecutor to leave the
    GPU idle between kernel launches.  Attaches to shared memory for the LLM
    embeddings and per-subject EEG epochs (zero-copy reads), then runs himalaya
    GPU ridge.

    Returns (subj, layer_index, list[TimeResolvedScore]).
    """
    from multiprocessing.shared_memory import SharedMemory

    import numpy as _np
    import torch as _torch
    from neuroai.regressors.ridge import cv_ridge_score_lag_matrix_gpu_batched as _fit

    subj: str = task["subj"]
    li: int = task["li"]

    # Route this worker to its assigned GPU so all available GPUs are used.
    if _torch.cuda.is_available():
        _torch.cuda.set_device(task.get("device_id", 0))

    # Attach to shared emb, extract layer slice, then detach immediately.
    # emb is stored transposed as (n_layers, n_words, D) when layerwise so that
    # emb_full[li] is a *contiguous* (n_words, D) slice — avoids 820-KB-strided
    # access across a 9+ GB array when 20 workers run simultaneously (deadlock-like
    # stall observed for Qwen3-14B / Pythia-12B / OPT-13B with d=5120).
    shm_emb = SharedMemory(name=task["emb_shm_name"])
    emb_full = _np.ndarray(task["emb_shape"], dtype=task["emb_dtype"], buffer=shm_emb.buf)
    X_li = _np.array(emb_full[li] if task["emb_layerwise"] else emb_full)
    shm_emb.close()

    # Attach to shared EEG epochs for this subject, copy, then detach.
    shm_eeg = SharedMemory(name=task["eeg_shm_name"])
    eeg = _np.array(_np.ndarray(task["eeg_shape"], dtype=task["eeg_dtype"], buffer=shm_eeg.buf))
    shm_eeg.close()

    tcs = _fit(X_li, eeg, task["times"], **task["batched_kw"])

    # Release PyTorch's CUDA caching allocator back to the MPS pool after each task.
    # Without this, allocations accumulate across sequential tasks in the same worker
    # process and GPU memory grows until OOM (observed: 65 → 80 GB within one batch).
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return subj, li, tcs


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class BroderickEegEncoder:
    """Predict Broderick EEG (1–8 Hz) from LLM hidden states.

    Mirrors :class:`~neuroai.benchmarks.podcast.PodcastEcogEncoder` with
    EEG-specific preprocessing and leave-one-run-out cross-validation.
    """

    def __init__(self, cfg: BroderickBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(self) -> BroderickEncoderResult:
        words_df = self._load_all_words()
        words = words_df["word"].astype(str).tolist()

        emb = self._extract_features(words)
        logger.info(
            "[%s] features shape: %s", self.cfg.model.model_name,
            "×".join(str(d) for d in emb.shape),
        )

        # Free the LLM from GPU immediately — features are cached to disk, model no longer
        # needed. For large models (Pythia-12B, OLMo2-7B, Qwen3-8B) this reclaims 15–30 GB
        # that would otherwise compete with the 20 himalaya ridge worker processes.
        from ..features.llm import clear_hf_model_cache
        clear_hf_model_cache()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

        ds = BroderickDataset(
            self.cfg.data.bids_root,
            subjects=self.cfg.data.subjects,
            drop_bads=self.cfg.data.drop_bads,
            n_runs=self.cfg.data.n_runs,
        )
        subjects = ds.available_subjects()
        if not subjects:
            raise RuntimeError(
                f"No subjects with EEG data found under {self.cfg.data.bids_root}"
            )
        logger.info("[%s] running on %d subjects: %s",
                    self.cfg.model.model_name, len(subjects), subjects)

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ch_csv = out_dir / "broderick_by_channel.csv"
        lw_tc_csv = out_dir / "broderick_layerwise_timecourse.csv"
        model_revision = self.cfg.model.model_revision or "main"

        # Skip subjects already in the incremental CSV for this revision.
        done_subjects: set[str] = set()
        if ch_csv.exists():
            try:
                _ex = pd.read_csv(ch_csv, usecols=["subject", "revision"])
                done_subjects = set(
                    _ex.loc[_ex["revision"] == model_revision, "subject"].unique()
                )
                if done_subjects:
                    logger.info("Skipping %d already-done subject(s): %s",
                                len(done_subjects), sorted(done_subjects))
            except Exception:
                logger.warning("Could not read existing channel CSV; re-running all.",
                               exc_info=True)

        all_scores: list[ChannelScore] = []
        subj_csv = out_dir / "broderick_by_subject.csv"
        reg_csv = out_dir / "broderick_by_region.csv"
        tc_csv = out_dir / "broderick_electrode_timecourse.csv"
        lw_csv = out_dir / "broderick_by_channel_layerwise.csv"

        subjects_todo = [s for s in subjects if s not in done_subjects]
        for s in subjects:
            if s in done_subjects:
                logger.info("--- Subject: %s --- (skipped)", s)

        use_parallel = (
            self.cfg.n_subjects_batch > 1
            and self.cfg.regression_device != "cpu"
            and not self.cfg.save_word_predictions
        )
        if self.cfg.save_word_predictions and self.cfg.n_subjects_batch > 1:
            logger.info(
                "save_word_predictions=True — using the sequential per-subject path "
                "(OOF capture isn't wired through the multiprocess flat-batch path)."
            )

        if use_parallel:
            # Flat parallel: batch subjects, submit all (subject × layer) tasks at once.
            n_batches = (len(subjects_todo) + self.cfg.n_subjects_batch - 1) // self.cfg.n_subjects_batch
            for i in range(0, len(subjects_todo), self.cfg.n_subjects_batch):
                batch = subjects_todo[i : i + self.cfg.n_subjects_batch]
                batch_num = i // self.cfg.n_subjects_batch + 1
                logger.info("=== Parallel batch %d/%d: %s ===", batch_num, n_batches, batch)
                try:
                    batch_scores = self._run_subjects_batch_parallel(
                        batch, ds, words_df, emb, out_dir, model_revision,
                        ch_csv, subj_csv, reg_csv, tc_csv, lw_csv, lw_tc_csv,
                    )
                    all_scores.extend(batch_scores)
                except Exception:
                    logger.error("Parallel batch %s failed", batch, exc_info=True)
        else:
            # Sequential path — one subject at a time (original behaviour).
            for subj in subjects_todo:
                logger.info("--- Subject: %s ---", subj)
                try:
                    scores = self._run_subject(subj, ds, words_df, emb)
                    all_scores.extend(scores)
                    self._save_subject_scores(
                        subj, scores, out_dir, model_revision,
                        ch_csv, subj_csv, reg_csv, tc_csv, lw_csv, lw_tc_csv,
                    )
                except Exception:
                    logger.error("Subject %s failed", subj, exc_info=True)

        # Mean time course across all subjects (read from CSV for consistency)
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
                    mean_tc_csv = out_dir / "broderick_mean_timecourse.csv"
                    mean_tc.to_csv(
                        mean_tc_csv, mode="a",
                        header=not mean_tc_csv.exists(), index=False,
                    )
                    logger.info("Saved mean time course (%d bins) → %s",
                                len(mean_tc), mean_tc_csv)
            except Exception:
                logger.warning("Mean timecourse failed", exc_info=True)

        return BroderickEncoderResult(
            model_name=self.cfg.model.model_name,
            model_revision=self.cfg.model.model_revision,
            n_words=len(words),
            channel_scores=all_scores,
            n_subjects_expected=len(subjects),
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _load_all_words(self) -> pd.DataFrame:
        return load_all_words(self.cfg.data.bids_root, n_runs=self.cfg.data.n_runs)

    def _extract_features(self, words: list[str]) -> np.ndarray:
        dcfg = self.cfg.data
        mcfg = self.cfg.model
        layer_index: str | int = "all" if self.cfg.layerwise else "last"
        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

        if "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower():
            return contextual_word_embeddings_cbr_rnn(
                words,
                mcfg.model_name,
                device=mcfg.device,
                context_words=dcfg.context_words if dcfg.context_words is not None else 48,
                cache_dir=dcfg.feature_cache_dir,
                section_id=0,
                lang="en",
            )

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
                lang="en",
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
                batch_size=mcfg.batch_size,
                cache_dir=dcfg.feature_cache_dir,
                section_id=0,
                lang="en",
                tokenizer_no_revision=self.cfg.tokenizer_no_revision,
                n_params=mcfg.n_params,
                trust_remote_code=self.cfg.trust_remote_code,
                dtype=surp_dtype,
            )
        return emb  # (n_words, [n_layers,] D)

    # ------------------------------------------------------------------
    # Per-subject pipeline
    # ------------------------------------------------------------------

    def _load_subject_epoch_data(
        self,
        subject: str,
        ds: BroderickDataset,
        words_df: pd.DataFrame,
        emb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict, np.ndarray] | None:
        """Load (or compute+cache) EEG epochs for one subject.

        Returns ``(emb_all, epochs_all, groups_all, times, ch_names, channel_meta,
        word_indices_all)`` or ``None`` if the subject should be skipped.
        ``word_indices_all`` indexes into ``words_df`` / ``emb`` (row order
        preserved from epoch extraction) — used to join OOF predictions back
        to word text/onset for the DLT-cost analysis.
        """
        dcfg = self.cfg.data
        available_runs = ds.available_runs(subject)
        if not available_runs:
            logger.warning("  [%s] No EEG runs found — skipping", subject)
            return None

        good_ch = ds.good_channel_names(subject, run=1)
        if not good_ch:
            logger.warning("  [%s] No good channels — skipping", subject)
            return None

        cache_key = _epoch_cache_key(dcfg)
        epoch_cache_path: Path | None = (
            Path(dcfg.epoch_cache_dir) / f"{subject}_{cache_key}.npz"
            if dcfg.epoch_cache_dir else None
        )

        epochs_all: np.ndarray
        groups_all: np.ndarray
        analysis_times: np.ndarray
        word_indices_all: np.ndarray
        ch_names_final: list[str]

        if epoch_cache_path is not None and epoch_cache_path.exists():
            logger.info("  Loading cached epochs: %s", epoch_cache_path.name)
            epochs_all, groups_all, analysis_times, word_indices_all, ch_names_final = (
                _load_epoch_cache(epoch_cache_path)
            )
        else:
            all_epochs: list[np.ndarray] = []
            all_word_indices: list[np.ndarray] = []
            all_groups: list[int] = []
            _analysis_times: np.ndarray | None = None
            n_channels: int | None = None
            ch_names_final = good_ch

            for run in available_runs:
                run_words_df = words_df[words_df["run"] == run].copy()
                if run_words_df.empty:
                    logger.warning("  Run %02d: no words in TextGrid — skipping", run)
                    continue

                word_indices = run_words_df["word_idx"].values
                if word_indices.max() >= len(emb):
                    logger.warning(
                        "  Run %02d: word_idx out of bounds (max=%d, emb size=%d)",
                        run, word_indices.max(), len(emb),
                    )
                    continue

                word_onsets = run_words_df["onset"].values.astype(np.float64)

                try:
                    data, sfreq, ch_names_run = load_eeg_run(
                        dcfg.bids_root, subject, run,
                        l_freq=dcfg.l_freq,
                        h_freq=dcfg.h_freq,
                        target_sfreq=dcfg.target_sfreq,
                        average_ref=dcfg.average_ref,
                        good_channels=good_ch,
                    )
                except Exception as exc:
                    logger.warning("  Run %02d: EEG load failed: %s", run, exc)
                    continue

                if n_channels is None:
                    n_channels = data.shape[0]
                    ch_names_final = ch_names_run

                if data.shape[0] != n_channels:
                    logger.warning(
                        "  Run %02d: channel count mismatch (%d vs %d) — skipping",
                        run, data.shape[0], n_channels,
                    )
                    continue

                epochs, valid_mask, epoch_times = extract_word_epochs(
                    data, sfreq, recording_tmin=0.0,
                    word_onsets=word_onsets,
                    tmin=dcfg.tmin,
                    tmax=dcfg.tmax,
                )
                n_valid = int(valid_mask.sum())
                logger.info("  Run %02d: %d / %d words extracted",
                            run, n_valid, len(word_onsets))

                if n_valid == 0:
                    continue

                t_mask = (epoch_times >= dcfg.tmin) & (epoch_times < dcfg.tmax)
                epochs = epochs[:, :, t_mask]
                t_cropped = epoch_times[t_mask]

                if dcfg.baseline is not None:
                    bl_tmin, bl_tmax = dcfg.baseline
                    bl_mask = (t_cropped >= bl_tmin) & (t_cropped <= bl_tmax)
                    if bl_mask.any():
                        epochs = epochs - epochs[:, :, bl_mask].mean(axis=-1, keepdims=True)

                if _analysis_times is None:
                    _analysis_times = t_cropped
                elif len(_analysis_times) != len(t_cropped):
                    logger.warning("  Run %02d: epoch time axis mismatch — skipping", run)
                    continue

                all_epochs.append(epochs)
                all_word_indices.append(word_indices[valid_mask])
                all_groups.extend([run - 1] * n_valid)

            if not all_epochs:
                logger.warning("  [%s] No valid epochs collected — skipping", subject)
                return None

            epochs_all = np.concatenate(all_epochs, axis=0)
            word_indices_all = np.concatenate(all_word_indices)
            groups_all = np.array(all_groups, dtype=np.int32)
            analysis_times = _analysis_times  # type: ignore[assignment]

            if epoch_cache_path is not None:
                _save_epoch_cache(
                    epoch_cache_path, epochs_all, groups_all,
                    analysis_times, word_indices_all, ch_names_final,
                )
                logger.info("  Saved epoch cache (%d words, %d ch) → %s",
                            len(word_indices_all), epochs_all.shape[1],
                            epoch_cache_path.name)

        emb_all = emb[word_indices_all]
        n_total = epochs_all.shape[0]
        n_ch = epochs_all.shape[1]
        n_times = epochs_all.shape[2]
        logger.info(
            "  %s: %d total word epochs, %d channels, %d time bins @ %.1f Hz  "
            "(%d CV groups / runs)",
            subject, n_total, n_ch, n_times, dcfg.target_sfreq,
            len(np.unique(groups_all)),
        )

        if n_total < 20:
            logger.warning("  [%s] Too few epochs (%d) — skipping", subject, n_total)
            return None

        channel_meta = ds.channel_meta(ch_names_final)
        return emb_all, epochs_all, groups_all, analysis_times, ch_names_final, channel_meta, word_indices_all

    def _run_subject(
        self,
        subject: str,
        ds: BroderickDataset,
        words_df: pd.DataFrame,
        emb: np.ndarray,
    ) -> list[ChannelScore]:
        """Run encoding for one subject (sequential path)."""
        data = self._load_subject_epoch_data(subject, ds, words_df, emb)
        if data is None:
            return []
        emb_all, epochs_all, groups_all, analysis_times, ch_names_final, channel_meta, word_indices_all = data
        scores = self._regress_channels(
            subject, emb_all, epochs_all, analysis_times, groups_all, channel_meta,
        )
        if self.cfg.save_word_predictions:
            self._save_word_predictions(
                subject, scores,
                words_df=words_df, word_indices=word_indices_all, times=analysis_times,
                epochs=epochs_all, ch_names=ch_names_final,
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
        ch_names: list[str],
    ) -> None:
        """Save each channel's best-layer out-of-fold predicted lag matrix
        AND the actual observed EEG (needed to score any post-hoc word subset,
        e.g. an easy/hard DLT-cost split — a prediction alone isn't scoreable).

        Mirrors :meth:`podcast.PodcastEcogEncoder._save_word_predictions` — one
        compressed ``.npz`` per (subject, model revision): all channels stacked
        as ``(n_channels, n_words, n_times)`` for both ``oof_predictions`` and
        ``y_true``, plus word text / onset / channel name / time-axis metadata
        for a later per-word DLT-cost join.
        """
        channels_with_oof = [
            c for c in scores
            if c.time_course is not None and c.time_course.oof_predictions is not None
        ]
        if not channels_with_oof:
            logger.warning("  save_word_predictions=True but no OOF predictions captured for %s", subject)
            return

        rows = words_df.iloc[word_indices]
        n_words = len(rows)
        stacked = np.stack(
            [c.time_course.oof_predictions for c in channels_with_oof], axis=0
        )  # (n_channels, n_words, n_times)
        if stacked.shape[1] != n_words:
            logger.warning(
                "  Word-count mismatch for %s (%d words vs %d OOF rows) — skipping save",
                subject, n_words, stacked.shape[1],
            )
            return

        name_to_idx = {name: i for i, name in enumerate(ch_names)}
        y_true = np.stack(
            [epochs[:, name_to_idx[c.name], :] for c in channels_with_oof], axis=0
        ).astype(np.float32)  # (n_channels, n_words, n_times)

        out_dir = word_predictions_dir(self.cfg.output_dir, "broderick_word_predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_revision = (self.cfg.model.model_revision or "main").replace("/", "_")
        safe_model = self.cfg.model.model_name.replace("/", "_")
        out_path = out_dir / f"{safe_model}_{model_revision}_{subject}.npz"
        np.savez_compressed(
            out_path,
            oof_predictions=stacked,
            y_true=y_true,
            channel_names=np.array([c.name for c in channels_with_oof]),
            words=rows["word"].astype(str).to_numpy(),
            word_idx=np.asarray(word_indices),
            onsets=rows["onset"].to_numpy(),
            times=times,
            model=np.array(self.cfg.model.model_name),
            revision=np.array(self.cfg.model.model_revision or "main"),
            n_params=np.array(self.cfg.model.n_params),
        )
        logger.info("  Saved word-level predictions (%s) → %s", stacked.shape, out_path)

    # ------------------------------------------------------------------
    # Score assembly and CSV saving helpers
    # ------------------------------------------------------------------

    def _assemble_scores_from_layer_tcs(
        self,
        subject: str,
        layer_tcs: dict[int, list],   # {layer_idx: [TimeResolvedScore per channel]}
        n_layers: int,
        ch_names: list[str],
        channel_meta: dict[str, dict],
    ) -> list[ChannelScore]:
        """Build ChannelScore list from completed flat-pool layer results."""
        results: list[ChannelScore] = []
        n_channels = len(ch_names)
        layerwise = n_layers > 1

        # Guard: if any layer failed (OOM-killed worker), skip assembly for this subject
        # rather than crashing with KeyError on the missing layer index.
        if layerwise:
            missing = [li for li in range(n_layers) if li not in layer_tcs]
            if missing:
                logger.warning(
                    "[%s] skipping score assembly: %d/%d layers missing (failed workers): %s",
                    subject, len(missing), n_layers, missing[:10],
                )
                return []

        for ci in range(n_channels):
            ch_name = ch_names[ci] if ci < len(ch_names) else f"ch_{ci}"
            meta = channel_meta.get(ch_name, {})
            if layerwise:
                per_layer: list[TimeResolvedScore] = [layer_tcs[li][ci] for li in range(n_layers)]
                layer_scores = [tc.to_mean_cvScore() for tc in per_layer]
                best_li = int(np.nanargmax([s.pearsonr for s in layer_scores]))
                results.append(ChannelScore(
                    subject=subject,
                    name=ch_name,
                    region=meta.get("region"),
                    hemisphere=meta.get("hemisphere"),
                    x=meta.get("x"), y=meta.get("y"), z=meta.get("z"),
                    score=layer_scores[-1],
                    time_course=per_layer[best_li],
                    layer_scores=layer_scores,
                    layer_time_courses=per_layer,
                ))
            else:
                tc = layer_tcs[0][ci]
                results.append(ChannelScore(
                    subject=subject,
                    name=ch_name,
                    region=meta.get("region"),
                    hemisphere=meta.get("hemisphere"),
                    x=meta.get("x"), y=meta.get("y"), z=meta.get("z"),
                    score=tc.to_mean_cvScore(),
                    time_course=tc,
                ))
        return results

    def _save_subject_scores(
        self,
        subj: str,
        scores: list[ChannelScore],
        out_dir: Path,
        model_revision: str,
        ch_csv: Path,
        subj_csv: Path,
        reg_csv: Path,
        tc_csv: Path,
        lw_csv: Path,
        lw_tc_csv: Path,
    ) -> None:
        """Append all per-subject result CSVs. Thread-safe via _csv_write_lock."""
        if not scores:
            return

        mean_r = float(np.nanmean([c.best_layer_pearsonr() for c in scores]))
        logger.info("  [%s] %d channels, mean best-layer Pearson r = %.4f",
                    subj, len(scores), mean_r)

        rows = [c.to_dict(include_layers=False) for c in scores]
        for r in rows:
            r["revision"] = model_revision
        subj_df = pd.DataFrame(rows)

        score_cols = ["best_pearsonr", "best_spearmanr", "pearsonr", "spearmanr"]
        sagg = subj_df[score_cols].agg(["mean", "std", "count"])
        srow: dict = {"subject": subj, "revision": model_revision}
        for col in score_cols:
            srow[f"{col}_mean"] = sagg.loc["mean", col]
            srow[f"{col}_std"] = sagg.loc["std", col]
            srow[f"{col}_count"] = int(sagg.loc["count", col])

        tc_rows: list[dict] = []
        for c in scores:
            tc_rows.extend(c.timecourse_rows())

        lw_rows: list[dict] = []
        for c in scores:
            if c.layer_scores:
                for li, ls in enumerate(c.layer_scores):
                    lw_rows.append({
                        "subject": subj, "channel": c.name,
                        "revision": model_revision, "layer": li,
                        "pearsonr": ls.pearsonr, "spearmanr": ls.spearmanr,
                    })

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
            logger.info("  Saved %d channel rows → %s", len(subj_df), ch_csv)
            pd.DataFrame([srow]).to_csv(
                subj_csv, mode="a", header=not subj_csv.exists(), index=False,
            )
            if "region" in subj_df.columns and subj_df["region"].notna().any():
                ragg = (
                    subj_df.dropna(subset=["region"])
                    .groupby("region")[score_cols]
                    .agg(["mean", "std", "count"])
                )
                ragg.columns = [f"{c}_{s}" for c, s in ragg.columns]
                ragg = ragg.reset_index()
                ragg.insert(0, "subject", subj)
                ragg.insert(1, "revision", model_revision)
                ragg.to_csv(reg_csv, mode="a", header=not reg_csv.exists(), index=False)
            if tc_rows:
                tc_df = pd.DataFrame(tc_rows)
                tc_df["revision"] = model_revision
                tc_df.to_csv(tc_csv, mode="a", header=not tc_csv.exists(), index=False)
            if lw_rows:
                pd.DataFrame(lw_rows).to_csv(
                    lw_csv, mode="a", header=not lw_csv.exists(), index=False,
                )
            if lw_tc_rows:
                pd.DataFrame(lw_tc_rows).to_csv(
                    lw_tc_csv, mode="a", header=not lw_tc_csv.exists(), index=False,
                )

    # ------------------------------------------------------------------
    # Flat (subject × layer) parallel GPU path
    # ------------------------------------------------------------------

    def _run_subjects_batch_parallel(
        self,
        batch: list[str],
        ds: BroderickDataset,
        words_df: pd.DataFrame,
        emb: np.ndarray,
        out_dir: Path,
        model_revision: str,
        ch_csv: Path,
        subj_csv: Path,
        reg_csv: Path,
        tc_csv: Path,
        lw_csv: Path,
        lw_tc_csv: Path,
    ) -> list[ChannelScore]:
        """Process a batch of subjects with flat (subject × layer) GPU parallelism.

        Uses multiprocessing (spawned processes) instead of threads so each
        worker has its own Python interpreter and GIL.  While one worker's GPU
        kernel executes, other workers run their CPU-side numpy/data-prep code in
        parallel — eliminating the idle gaps that made ThreadPoolExecutor show
        0% GPU utilisation.  Large arrays (emb, per-subject eeg) are placed in
        OS shared memory once and read zero-copy by workers.

        As soon as all layers for one subject complete, its scores are assembled
        and saved incrementally, preserving crash-safety.
        """
        import multiprocessing as _mpp
        from multiprocessing import shared_memory as _shmm
        from concurrent.futures import ProcessPoolExecutor, as_completed

        rcfg = self.cfg.regression
        layerwise = emb.ndim == 3
        n_layers = emb.shape[1] if layerwise else 1

        # Step 1: load epoch data for every subject in the batch
        batch_data: dict[str, tuple] = {}
        for subj in batch:
            logger.info("--- Subject: %s --- (loading epochs)", subj)
            try:
                data = self._load_subject_epoch_data(subj, ds, words_df, emb)
                if data is not None:
                    batch_data[subj] = data
            except Exception:
                logger.error("[%s] epoch loading failed", subj, exc_info=True)

        if not batch_data:
            return []

        # Step 2: put emb and per-subject eeg in shared memory (zero-copy for workers).
        # del the intermediate copy immediately after writing to shared memory so that
        # we never hold the data twice in RAM (critical for large models like GPT-2-XL
        # where emb is ~3.4 GB and a silent second copy causes cgroup OOM kills).
        # Transpose layerwise emb from (n_words, n_layers, D) → (n_layers, n_words, D)
        # so that each worker's slice emb_full[li] is a *contiguous* (n_words, D) block.
        # Without the transpose, emb_full[:, li, :] strides 41×5120×4 = 820 KB between
        # elements; with 20 workers simultaneously reading a 9+ GB shared memory block
        # this causes a kernel TLB / page-fault stall that looks like a deadlock.
        if layerwise:
            emb_c = np.ascontiguousarray(emb.transpose(1, 0, 2).astype(np.float32))
        else:
            emb_c = np.ascontiguousarray(emb.astype(np.float32))
        emb_shm_shape, emb_shm_dtype = emb_c.shape, str(emb_c.dtype)
        shm_emb = _shmm.SharedMemory(create=True, size=emb_c.nbytes)
        np.ndarray(emb_shm_shape, dtype=emb_shm_dtype, buffer=shm_emb.buf)[:] = emb_c
        del emb_c  # data is now in shared memory; free the intermediate copy

        shm_eegs: dict[str, _shmm.SharedMemory] = {}
        eeg_meta: dict[str, tuple] = {}   # subj -> (shm_name, shape, dtype_str)
        for subj, (_, eeg_all, *_rest) in batch_data.items():
            eeg_c = np.ascontiguousarray(eeg_all.astype(np.float32))
            shm = _shmm.SharedMemory(create=True, size=eeg_c.nbytes)
            np.ndarray(eeg_c.shape, dtype=eeg_c.dtype, buffer=shm.buf)[:] = eeg_c
            shm_eegs[subj] = shm
            eeg_meta[subj] = (shm.name, eeg_c.shape, str(eeg_c.dtype))
            del eeg_c  # free immediately

        # Step 3: build flat task list — one dict per (subject, layer)
        tasks: list[dict] = []
        for subj, (emb_all, eeg_all, groups_all, times, ch_names, channel_meta, _word_idx) in batch_data.items():
            n_splits = min(rcfg.n_splits, int(np.unique(groups_all).size))
            kw = dict(
                n_splits=n_splits,
                alpha=rcfg.alpha,
                scale=rcfg.scale,
                standardize_target=True,
                random_state=rcfg.random_state,
                groups=groups_all,
            )
            eeg_shm_name, eeg_shape, eeg_dtype = eeg_meta[subj]
            for li in range(n_layers):
                tasks.append({
                    "subj": subj,
                    "li": li,
                    "times": times,
                    "batched_kw": kw,
                    "emb_shm_name": shm_emb.name,
                    "emb_shape": emb_shm_shape,
                    "emb_dtype": emb_shm_dtype,
                    "emb_layerwise": layerwise,
                    "eeg_shm_name": eeg_shm_name,
                    "eeg_shape": eeg_shape,
                    "eeg_dtype": eeg_dtype,
                })

        # Distribute tasks round-robin across all visible GPUs so that workers
        # spread across devices instead of piling onto cuda:0 (MPS default).
        try:
            import torch as _torch
            n_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 1
        except Exception:
            n_gpus = 1
        n_gpus = max(1, n_gpus)
        for task_idx, task in enumerate(tasks):
            task["device_id"] = task_idx % n_gpus

        n_workers = self.cfg.n_layer_workers
        logger.info(
            "[MP Batch] %d subjects × %d layers = %d tasks  (pool=%d processes, %d GPUs)",
            len(batch_data), n_layers, len(tasks), n_workers, n_gpus,
        )

        # Step 4: submit all tasks and collect results
        layer_results: dict[str, dict[int, list]] = {s: {} for s in batch_data}
        layer_done: dict[str, int] = {s: 0 for s in batch_data}
        all_scores: list[ChannelScore] = []

        ctx = _mpp.get_context("spawn")  # spawn required: CUDA cannot be forked
        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {
                    pool.submit(_mp_worker_fit, task): (task["subj"], task["li"])
                    for task in tasks
                }
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
                    logger.info("  [%s] layer %d/%d done (MP)", subj, li + 1, n_layers)

                    # When all layers for this subject finish, assemble and save immediately
                    if layer_done[subj] == n_layers:
                        _, _, _, _, ch_names, channel_meta, _ = batch_data[subj]
                        try:
                            scores = self._assemble_scores_from_layer_tcs(
                                subj, layer_results[subj], n_layers, ch_names, channel_meta,
                            )
                            all_scores.extend(scores)
                            self._save_subject_scores(
                                subj, scores, out_dir, model_revision,
                                ch_csv, subj_csv, reg_csv, tc_csv, lw_csv, lw_tc_csv,
                            )
                        except Exception:
                            logger.error("[%s] score assembly/save failed", subj, exc_info=True)
        finally:
            # Always release shared memory even on exception/KeyboardInterrupt
            shm_emb.close()
            shm_emb.unlink()
            for shm in shm_eegs.values():
                shm.close()
                shm.unlink()

        return all_scores

    # ------------------------------------------------------------------
    # Per-channel regression dispatch
    # ------------------------------------------------------------------

    def _regress_channels(
        self,
        subject: str,
        emb: np.ndarray,          # (N, [L,] D)
        eeg: np.ndarray,          # (N, C, T)
        times: np.ndarray,        # (T,)
        groups: np.ndarray,       # (N,)
        channel_meta: dict[str, dict],
    ) -> list[ChannelScore]:
        if self.cfg.regression_device != "cpu" and self.cfg.use_lag_matrix:
            return self._regress_channels_gpu_batched(
                subject, emb, eeg, times, groups, channel_meta
            )
        return self._regress_channels_cpu(
            subject, emb, eeg, times, groups, channel_meta
        )

    def _regress_channels_gpu_batched(
        self,
        subject: str,
        emb: np.ndarray,
        eeg: np.ndarray,
        times: np.ndarray,
        groups: np.ndarray,
        channel_meta: dict[str, dict],
    ) -> list[ChannelScore]:
        """Fit all channels in one himalaya GPU call per layer."""
        rcfg = self.cfg.regression
        n_channels = eeg.shape[1]
        channel_names = list(channel_meta.keys())
        layerwise = emb.ndim == 3
        n_layers = emb.shape[1] if layerwise else 1

        # Respect config n_splits — cap at number of unique run groups (20 runs max).
        n_splits = min(rcfg.n_splits, int(np.unique(groups).size))

        logger.info(
            "  [%s] GPU batched: %d words, %d channels, %d lags, %d layers, %d CV folds",
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
            return_oof_predictions=self.cfg.save_word_predictions,
        )

        n_layer_workers = min(self.cfg.n_layer_workers, n_layers)
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
        # subject's regression is lost (same fix as podcast.py's
        # _regress_electrodes_gpu_batched). Force that one-time init here,
        # single threaded, before spawning the pool below.
        _eigh_warmup = torch.eye(2, device=f"cuda:{torch.cuda.current_device()}")
        torch.linalg.eigh(_eigh_warmup)
        del _eigh_warmup

        chunk_size = self.cfg.layer_chunk_size or n_layers
        from concurrent.futures import ThreadPoolExecutor, as_completed
        for chunk_lo in range(0, n_layers, chunk_size):
            chunk_hi = min(chunk_lo + chunk_size, n_layers)
            chunk_workers = min(n_layer_workers, chunk_hi - chunk_lo)
            with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
                futures = {
                    pool.submit(_fit_layer, li): li for li in range(chunk_lo, chunk_hi)
                }
                for fut in as_completed(futures):
                    li, tcs = fut.result()
                    layer_tcs[li] = tcs
                    logger.info("  [%s] layer %d/%d done", subject, li + 1, n_layers)
            if chunk_hi < n_layers:
                torch.cuda.empty_cache()

        results: list[ChannelScore] = []
        for ci in range(n_channels):
            ch_name = channel_names[ci] if ci < len(channel_names) else f"ch_{ci}"
            meta = channel_meta.get(ch_name, {})

            if layerwise:
                per_layer: list[TimeResolvedScore] = [layer_tcs[li][ci] for li in range(n_layers)]  # type: ignore[index]
                layer_scores = [tc.to_mean_cvScore() for tc in per_layer]
                best_li = int(np.nanargmax([s.pearsonr for s in layer_scores]))
                if self.cfg.save_word_predictions:
                    # Keep OOF predictions only for the winning layer — the full
                    # (n_layers × n_words × n_times) set would be enormous.
                    for li, tc_ in enumerate(per_layer):
                        if li != best_li:
                            tc_.oof_predictions = None
                results.append(ChannelScore(
                    subject=subject,
                    name=ch_name,
                    region=meta.get("region"),
                    hemisphere=meta.get("hemisphere"),
                    x=meta.get("x"), y=meta.get("y"), z=meta.get("z"),
                    score=layer_scores[-1],
                    time_course=per_layer[best_li],
                    layer_scores=layer_scores,
                    layer_time_courses=per_layer,
                ))
            else:
                tc = layer_tcs[0][ci]
                results.append(ChannelScore(
                    subject=subject,
                    name=ch_name,
                    region=meta.get("region"),
                    hemisphere=meta.get("hemisphere"),
                    x=meta.get("x"), y=meta.get("y"), z=meta.get("z"),
                    score=tc.to_mean_cvScore(),
                    time_course=tc,
                ))

        mean_r = float(np.nanmean([r.best_layer_pearsonr() for r in results]))
        logger.info("  [%s] %d channels — mean best-layer r = %.4f (GPU batched)",
                    subject, len(results), mean_r)
        return results

    def _regress_channels_cpu(
        self,
        subject: str,
        emb: np.ndarray,
        eeg: np.ndarray,
        times: np.ndarray,
        groups: np.ndarray,
        channel_meta: dict[str, dict],
    ) -> list[ChannelScore]:
        rcfg = self.cfg.regression
        n_channels = eeg.shape[1]
        channel_names = list(channel_meta.keys())
        n_splits = min(rcfg.n_splits, int(np.unique(groups).size))

        reg_kwargs = dict(
            n_splits=n_splits,
            alpha=rcfg.alpha,
            scale=rcfg.scale,
            standardize_target=True,
            random_state=rcfg.random_state,
            device="cpu",
        )

        jobs = (
            delayed(_regress_channel)(
                ci=ci,
                emb=emb,
                y_ci=eeg[:, ci, :],
                times=times,
                groups=groups,
                ch_name=channel_names[ci] if ci < len(channel_names) else f"ch_{ci}",
                meta=channel_meta.get(
                    channel_names[ci] if ci < len(channel_names) else f"ch_{ci}", {}
                ),
                subject=subject,
                use_lag_matrix=self.cfg.use_lag_matrix,
                window_size=self.cfg.window_size,
                step_size=self.cfg.step_size,
                reg_kwargs=reg_kwargs,
                save_word_predictions=self.cfg.save_word_predictions,
            )
            for ci in range(n_channels)
        )

        results: list[ChannelScore] = Parallel(
            n_jobs=self.cfg.n_jobs, backend="loky",
        )(jobs)

        mean_r = float(np.nanmean([r.best_layer_pearsonr() for r in results]))
        logger.info("  [%s] %d channels — mean best-layer r = %.4f",
                    subject, len(results), mean_r)
        return results
