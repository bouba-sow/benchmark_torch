"""Podcast ECoG encoding benchmark — pooled LME per lag (Design B).

Unlike the per-electrode ridge encoder (``benchmarks/podcast.py``), this pools
every subject and electrode into ONE mixed-effects model per lag, with
electrode nested within subject (``(1|subject) + (1+surp|subject:electrode)``)
so no correspondence between electrodes across subjects is assumed. See
``neuroai/regressors/podcast_lme.py`` for the formula and fallback-ladder
details.

Pipeline
--------
1. Load the transcript; get the model's final-layer surprisal from the
   logit-lens cache (already computed by the ridge/surprisal sweep — this
   benchmark is CPU-only and never loads an LLM itself; a cache miss is an
   error, not a silent slow path).
2. Compute word length + wordfreq lexical-frequency baseline covariates
   (same confound-control covariates as ``PodcastBenchmarkConfig.
   compute_unique_surprisal``).
3. Drop the one non-finite word (index 0, no left context for logit-lens).
4. Assign contiguous word-index chunks (shared across all subjects, since
   they all read/heard the same transcript) — the train/held-out split unit.
5. For each subject: load+preprocess word-locked epochs (shared helper,
   ``benchmarks.podcast.load_subject_epochs``), and append
   (subject, electrode, chunk, wlen, freq, surprisal) rows plus that
   electrode's full (n_words, n_times) epoch matrix to a pooled table.
6. Write the pooled table (parquet) and stacked signal (npy, one column per
   lag) to disk once, then fit one LME per lag via a process pool (each lag
   is a fully independent fit — same X design, different Y column) — pymer4/
   rpy2 embeds one R interpreter per process, so parallelism must be
   process-based (spawn), not threaded.

Output: ``podcast_lme_by_lag.csv`` — one row per lag, written incrementally
as each lag's fit completes (crash-safe, and lets you watch progress live).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import typing as tp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.podcast import PodcastDataset, load_transcript
from ..sweep import csv_file_lock
from .base import ModelConfig
from .podcast import load_subject_epochs

logger = logging.getLogger(__name__)

# Fixed column order for podcast_lme_by_lag.csv — every appended row (success
# or crash-fallback) is reindexed to this before writing, so a row with only
# a few keys (e.g. {"lag": li, "error": ...} from a crashed worker) can never
# misalign the CSV's columns (pandas silently shifts short dict-derived rows
# otherwise, corrupting whichever column happens to come second).
#
# "revision" added (checkpoint-collision fix, 2026-08-27): without it, every
# checkpoint of a given model size shared one out_dir/CSV with a resume-check
# keyed only on "lag" (worse than Fedorenko2016's version of this same bug —
# this file had no csv_file_lock at all, not even for concurrent SLURM array
# workers). Same fix shape as N400/Tuckute/LPP/NS-fMRI/Fedorenko2016.
_LAG_ROW_COLUMNS = (
    "lag", "lag_time_s", "beta", "se", "t_stat", "p_value", "loglik", "aic",
    "bic", "baseline_loglik", "delta_loglik_train", "delta_loglik_test",
    "baseline_loglik_test", "loglik_test", "lrt_p_value",
    "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr", "converged",
    "slope_dropped", "n_obs", "n_obs_test", "error", "fit_wall_time_s",
    "revision",
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PodcastLmeDataConfig:
    bids_root: str
    subjects: list[str] | None = None
    tmin: float = -2.0
    tmax: float = 2.0
    tmin_extract: float = -2.1
    downsample_to: float = 32.0
    clip_iqr: float | None = 5.0
    # No pre-onset baseline correction, matching the reference Hasson-lab
    # tutorial (mne.Epochs(..., baseline=None)) — the outcome is z-scored per
    # (subject, electrode) group on train-fold statistics instead, inside
    # fit_podcast_lme_one_lag; see regressors/podcast_lme.py module docstring.
    baseline: tuple[float, float] | None = None
    baseline_mode: str = "logratio"
    electrode_selection_dir: str | None = None
    roi_map_path: str | None = None
    # Per-subject epoch cache (preprocessed ECoG: IQR-clipped, baseline-
    # corrected if enabled, cropped to [tmin, tmax), resampled). Model-
    # independent — every model in a sweep needs the exact same epochs, only
    # the surprisal predictor differs — so this is shared across the whole
    # family instead of every model's job re-reading/re-processing the raw
    # .fif files from scratch. None disables caching (always recompute).
    epoch_cache_dir: str | None = None
    # Surprisal cache — must already exist (this benchmark is CPU-only and
    # never loads an LLM). Same resolution logic as PodcastBenchmarkConfig's
    # feature_cache_dir / logit_lens_surprisal_all_layers.
    feature_cache_dir: str | None = None
    max_length: int | None = None
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    # wordfreq lexical-frequency baseline language (word length is computed
    # directly).
    wordfreq_language: str = "en"


@dataclass
class PodcastLmeConfig:
    # Contiguous word-index chunk size — the train/held-out split unit
    # (words are inherently sequential in one continuous podcast, this is
    # the direct word-index analogue of the NS-fMRI LME's TR-chunk split).
    chunk_words: int = 30
    train_frac: float = 0.75
    split_random_state: int = 0
    verbose: bool = False
    allow_slope_fallback: bool = True
    # Which lags to fit: None = all time bins; else an explicit stride
    # (e.g. 4 → every 4th lag) to cut cost when the full grid is too
    # expensive to run in one job.
    lag_stride: int = 1
    # Mean-window mode: (lo_s, hi_s) -- instead of fitting one LME per lag
    # (32 lags x 110 checkpoints would be ~32x every other dataset's
    # checkpoint-sweep cost), average the response ACROSS lags whose
    # analysis time falls in [lo_s, hi_s] into ONE column and fit exactly
    # ONE LME on that windowed mean -- the standard ERP/ECoG "mean amplitude
    # in a window" convention, not a per-timepoint sweep. Set for a
    # checkpoint-scaling sweep; leave None for the full per-lag grid (a
    # single full-dataset run, not a 110-checkpoint sweep). Mutually
    # exclusive with lag_stride in practice (window mode ignores it).
    window_s: tuple[float, float] | None = None
    # Process-pool size for fitting lags concurrently. Each worker is a
    # separate Python process (spawn) with its own embedded R/pymer4
    # interpreter — rpy2 is not thread-safe, so this must be process-based.
    n_workers: int = 10
    # Save one parquet per lag under output_dir/predictions/ — one row per
    # HELD-OUT test-set (subject, electrode, word_idx) with y_true/yhat_base/
    # yhat_full/baseline_loglik/full_loglik/delta_loglik, for DLT-difficulty-
    # style post-hoc analysis (same convention as the ridge encoder's
    # --save_word_predictions). On by default — pass --no_save_word_predictions
    # to skip if disk usage is a concern (a full lag grid's worth of held-out
    # predictions is large: millions of rows across all lags).
    save_word_predictions: bool = True
    # Separate ROI experiment: fit independently for each mapped left-hemisphere parcel.
    roi_wise: bool = False


@dataclass
class PodcastLmeBenchmarkConfig:
    data: PodcastLmeDataConfig
    model: ModelConfig
    lme: PodcastLmeConfig = field(default_factory=PodcastLmeConfig)
    output_dir: str = "outputs"
    # Grouping label for saved CSVs when HF load repo differs.
    canonical_model_name: str | None = None


# ---------------------------------------------------------------------------
# Per-subject epoch cache — model-independent, shared across a whole family
# sweep. Every model needs the exact same preprocessed ECoG epochs (only the
# surprisal predictor differs), so caching this avoids every model's job
# independently re-reading and re-processing the raw .fif files.
# ---------------------------------------------------------------------------

def _epoch_cfg_hash(dcfg: PodcastLmeDataConfig, n_words: int) -> str:
    # n_words: the length of the (already feat_finite-filtered) onsets array
    # being epoched THIS run. Epochs are only shareable across models when
    # they cover the exact same set of words — but which words survive the
    # non-finite-surprisal filter is genuinely model-dependent (e.g. models
    # whose tokenizer auto-prepends a BOS token, like Llama/Mistral, or
    # CBR-RNN's empty-context-for-word-0 convention, never produce a NaN
    # surprisal for word 0 and so never drop it, unlike GPT-2/GPT-Neo/etc,
    # which do). Cache entries must be split by word count too, or a
    # (word_idx, valid_mask)-sized array from one model gets fed to another
    # model's differently-sized chunk/word_idx arrays downstream — exactly
    # the "boolean index did not match indexed array" crash this fixes.
    key = (
        f"{dcfg.tmin}_{dcfg.tmax}_{dcfg.tmin_extract}_{dcfg.downsample_to}"
        f"_{dcfg.clip_iqr}_{dcfg.baseline}_{dcfg.baseline_mode}"
        f"_{dcfg.electrode_selection_dir}_{n_words}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:10]


def _epoch_cache_path(cache_dir: Path, subject: str, cfg_hash: str) -> Path:
    return cache_dir / f"{subject}_{cfg_hash}.npz"


def _load_subject_epochs_cached(
    ds: PodcastDataset,
    subject: str,
    onsets: np.ndarray,
    dcfg: PodcastLmeDataConfig,
    cache_dir: Path | None,
    cfg_hash: str,
):
    """Cached wrapper around ``podcast.load_subject_epochs``.

    Atomic write with a PID-unique tmp filename — a deterministic tmp name
    (subject + cfg_hash only) would collide when multiple array tasks/models
    cold-start the same cache concurrently, exactly the race that caused
    Nieuwland's "cache write failed" warnings; this avoids that class of bug
    from the start rather than fixing it after the fact.
    """
    cp = _epoch_cache_path(cache_dir, subject, cfg_hash) if cache_dir else None
    if cp is not None and cp.is_file():
        try:
            z = np.load(cp, allow_pickle=False)
            valid_mask = z["valid_mask"].astype(bool)
            # Defense in depth on top of n_words being baked into cfg_hash:
            # if this ever collides anyway (hash truncated to 10 hex chars,
            # or a future field added to the key without updating old
            # entries), fail safe by recomputing rather than silently
            # returning a wrong-length mask to the caller.
            if len(valid_mask) != len(onsets):
                raise ValueError(
                    f"cached valid_mask length {len(valid_mask)} != onsets length {len(onsets)}"
                )
            electrode_meta = json.loads(str(z["electrode_meta_json"]))
            return (
                z["epochs"], valid_mask, z["times"], electrode_meta,
            )
        except Exception as exc:
            logger.warning("%s: epoch cache read failed (%s) — recomputing", subject, exc)

    loaded = load_subject_epochs(
        ds, subject, onsets,
        tmin=dcfg.tmin, tmax=dcfg.tmax, tmin_extract=dcfg.tmin_extract,
        downsample_to=dcfg.downsample_to, clip_iqr=dcfg.clip_iqr,
        baseline=dcfg.baseline, baseline_mode=dcfg.baseline_mode,
        electrode_selection_dir=dcfg.electrode_selection_dir,
    )
    if loaded is None or cp is None:
        return loaded

    epochs, valid_mask, times, electrode_meta = loaded
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.parent / f"{cp.stem}.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(
            tmp,
            epochs=epochs,
            valid_mask=valid_mask,
            times=times,
            electrode_meta_json=json.dumps(electrode_meta),
        )
        tmp.rename(cp)
    except Exception as exc:
        logger.warning("%s: epoch cache write failed — %s", subject, exc)
        tmp.unlink(missing_ok=True)
    return loaded


# ---------------------------------------------------------------------------
# Worker (module-level for spawn-based multiprocessing pickling)
# ---------------------------------------------------------------------------

def _fit_one_lag_worker(
    base_parquet_path: str,
    y_npy_path: str,
    lag_idx: int,
    lag_time_s: float,
    *,
    train_frac: float,
    random_state: int,
    verbose: bool,
    allow_slope_fallback: bool,
    predictions_dir: str | None = None,
    region: str | None = None,
) -> dict[str, tp.Any]:
    """Runs in its own spawned process: load the shared table, fit one lag."""
    import numpy as _np
    import pandas as _pd
    from pathlib import Path as _Path

    from ..regressors.podcast_lme import fit_podcast_lme_one_lag

    t0 = time.time()
    row: dict[str, tp.Any] = {"lag": lag_idx, "lag_time_s": lag_time_s}
    if region is not None:
        row["region"] = region
    try:
        base_df = _pd.read_parquet(base_parquet_path)
        y_all = _np.load(y_npy_path, mmap_mode="r")
        if region is not None:
            roi_mask = base_df["region"].astype(str) == region
            row_indices = _np.flatnonzero(roi_mask.to_numpy())
            base_df = base_df.loc[roi_mask].copy()
            if base_df.empty:
                raise ValueError(f"No observations for region {region!r}")
            y_all = y_all[row_indices]
        y_col = _np.asarray(y_all[:, lag_idx], dtype=_np.float64)
        # Predictors (surprisal/wlen/freq) are already finiteness-filtered
        # once, before the per-lag loop (lag-invariant) — but y is genuinely
        # per-lag and was never checked here. A non-finite ECoG sample at
        # this specific lag (e.g. an edge-of-epoch artifact) would otherwise
        # flow straight into lmer() for this lag only, same bug class fixed
        # in natural_stories_fmri_lme.py's per-(fROI, layer) worker.
        valid = _np.isfinite(y_col)
        n_dropped = int((~valid).sum())
        if n_dropped:
            logger.warning(
                "Lag %d: dropping %d/%d row(s) with non-finite y", lag_idx, n_dropped, len(y_col),
            )
        base_df = base_df.loc[valid].copy()
        base_df["y"] = y_col[valid]

        fit = fit_podcast_lme_one_lag(
            base_df,
            train_frac=train_frac,
            random_state=random_state,
            verbose=verbose,
            allow_slope_fallback=allow_slope_fallback,
            save_per_word=predictions_dir is not None,
        )
        # Written directly from the worker process, not sent back through the
        # ProcessPoolExecutor's pickle channel — a held-out per-word table
        # for a Podcast-scale lag (~9 subjects × ~20 electrodes × ~25% of
        # ~5100 words) is large enough that round-tripping it through IPC
        # would be wasteful when the worker can just write it itself.
        if predictions_dir is not None and fit.per_word is not None:
            pred_path = _Path(predictions_dir) / f"lag{lag_idx:03d}.parquet"
            fit.per_word.insert(0, "lag", lag_idx)
            fit.per_word.to_parquet(pred_path, index=False)
        row.update({
            "beta": fit.beta,
            "se": fit.se,
            "t_stat": fit.t_stat,
            "p_value": fit.p_value,
            "loglik": fit.loglik,
            "aic": fit.aic,
            "bic": fit.bic,
            "baseline_loglik": fit.baseline_loglik,
            "delta_loglik_train": fit.delta_loglik_train,
            "delta_loglik_test": fit.delta_loglik_test,
            "baseline_loglik_test": fit.baseline_loglik_test,
            "loglik_test": fit.loglik_test,
            "lrt_p_value": fit.lrt_p_value,
            "test_pearsonr_full_model": fit.test_pearsonr_full_model,
            "test_spearmanr_full_model": fit.test_spearmanr_full_model,
            "test_pearsonr_baseline_lme": fit.test_pearsonr_baseline_lme,
            "test_spearmanr_baseline_lme": fit.test_spearmanr_baseline_lme,
            "test_delta_pearsonr": fit.test_delta_pearsonr,
            "test_delta_spearmanr": fit.test_delta_spearmanr,
            "converged": fit.converged,
            "slope_dropped": fit.slope_dropped,
            "n_obs": fit.n_obs,
            "n_obs_test": fit.n_obs_test,
            "error": None,
        })
    except Exception as exc:  # noqa: BLE001 — one bad lag must not kill the pool
        logger.error("Region %s lag %d failed", region, lag_idx, exc_info=True)
        row["error"] = str(exc)
    row["fit_wall_time_s"] = time.time() - t0
    return row


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class PodcastLmeBenchmark:
    """Pooled Podcast ECoG LME: one mixed-effects model per lag."""

    def __init__(self, cfg: PodcastLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(self, extra_fields: dict[str, tp.Any] | None = None) -> pd.DataFrame:
        dcfg = self.cfg.data
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        revision = self.cfg.model.model_revision or "main"
        safe_revision = revision.replace("/", "_")

        words_df = load_transcript(dcfg.bids_root)
        words = words_df["word"].astype(str).tolist()
        onsets = words_df["onset"].values.astype(np.float64)

        surprisal = self._load_surprisal(words)
        baseline_feats = self._load_baseline_covariates(words)

        feat_finite = np.isfinite(surprisal) & np.isfinite(baseline_feats).all(axis=1)
        if not feat_finite.all():
            n_dropped = int((~feat_finite).sum())
            logger.warning(
                "Dropping %d word(s) with non-finite features before pooling: indices %s",
                n_dropped, np.where(~feat_finite)[0].tolist(),
            )
        words = [w for w, keep in zip(words, feat_finite) if keep]
        onsets = onsets[feat_finite]
        surprisal = surprisal[feat_finite]
        baseline_feats = baseline_feats[feat_finite]
        word_idx_kept = np.where(feat_finite)[0]

        n_words_feat = len(words)
        chunk_ids_by_orig_idx = word_idx_kept // self.cfg.lme.chunk_words

        ds = PodcastDataset(dcfg.bids_root, subjects=dcfg.subjects, drop_bads=True)
        subjects = ds.available_subjects()
        if not subjects:
            raise RuntimeError(f"No subjects with highgamma .fif found under {dcfg.bids_root}")
        logger.info("Pooling %d subjects, %d words (post-filter)", len(subjects), n_words_feat)

        subj_parts: list[np.ndarray] = []
        elec_parts: list[np.ndarray] = []
        chunk_parts: list[np.ndarray] = []
        word_idx_parts: list[np.ndarray] = []
        wlen_parts: list[np.ndarray] = []
        freq_parts: list[np.ndarray] = []
        surp_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        n_times: int | None = None
        analysis_times: np.ndarray | None = None

        epoch_cache_dir = Path(dcfg.epoch_cache_dir) if dcfg.epoch_cache_dir else None
        epoch_cfg_hash = _epoch_cfg_hash(dcfg, n_words=len(onsets))

        for subj in subjects:
            logger.info("--- Subject: %s ---", subj)
            loaded = _load_subject_epochs_cached(
                ds, subj, onsets, dcfg, epoch_cache_dir, epoch_cfg_hash,
            )
            if loaded is None:
                continue
            epochs, valid_mask, subj_times, electrode_meta = loaded
            if n_times is None:
                n_times = epochs.shape[-1]
                analysis_times = subj_times
            elif epochs.shape[-1] != n_times:
                logger.warning(
                    "  %s has %d time bins (expected %d) — skipping", subj, epochs.shape[-1], n_times,
                )
                continue

            n_valid = int(valid_mask.sum())
            channel_names = list(electrode_meta.keys())
            n_channels = len(channel_names)

            for ci, ch_name in enumerate(channel_names):
                subj_parts.append(np.full(n_valid, subj, dtype=object))
                elec_parts.append(np.full(n_valid, ch_name, dtype=object))
                chunk_parts.append(chunk_ids_by_orig_idx[valid_mask])
                word_idx_parts.append(word_idx_kept[valid_mask])
                wlen_parts.append(baseline_feats[valid_mask, 0])
                freq_parts.append(baseline_feats[valid_mask, 1])
                surp_parts.append(surprisal[valid_mask])
                y_parts.append(epochs[:, ci, :])
            logger.info("  %d channels × %d valid words appended", n_channels, n_valid)

        if not y_parts:
            raise RuntimeError("No subject/electrode data assembled — nothing to fit")

        base_df = pd.DataFrame({
            "subject": np.concatenate(subj_parts),
            "electrode": np.concatenate(elec_parts),
            "chunk": np.concatenate(chunk_parts).astype(str),
            "word_idx": np.concatenate(word_idx_parts).astype(np.int64),
            "wlen": np.concatenate(wlen_parts),
            "freq": np.concatenate(freq_parts),
            "surprisal": np.concatenate(surp_parts),
        })
        y_all = np.concatenate(y_parts, axis=0).astype(np.float32)  # (n_rows, n_times)
        region_names: list[str] = []
        if self.cfg.lme.roi_wise:
            if not dcfg.roi_map_path:
                raise ValueError("Podcast ROI-wise LME requires data.roi_map_path")
            roi_map = pd.read_csv(dcfg.roi_map_path)
            required_map_cols = {"subject", "channel", "region", "hemisphere", "assigned"}
            missing_map = required_map_cols - set(roi_map.columns)
            if missing_map:
                raise ValueError(f"Podcast ROI map missing columns: {sorted(missing_map)}")
            roi_map = roi_map.loc[
                roi_map["assigned"].astype(bool)
                & (roi_map["hemisphere"].astype(str) == "L")
                & roi_map["region"].astype(str).isin(
                    ("IFGorb", "IFG", "MFG", "AntTemp", "PostTemp")
                ),
                ["subject", "channel", "region"],
            ].copy()
            roi_map["subject"] = roi_map["subject"].astype(str)
            roi_map["channel"] = roi_map["channel"].astype(str)
            if roi_map.duplicated(["subject", "channel"]).any():
                raise ValueError("Podcast ROI map has duplicate (subject, channel) assignments")
            base_df["_source_row"] = np.arange(len(base_df), dtype=np.int64)
            mapped = base_df.merge(
                roi_map.rename(columns={"channel": "electrode"}),
                on=["subject", "electrode"], how="inner", sort=False,
            ).sort_values("_source_row")
            mapped_rows = mapped["_source_row"].to_numpy(dtype=np.int64)
            y_all = y_all[mapped_rows]
            base_df = mapped.drop(columns="_source_row").reset_index(drop=True)
            region_names = sorted(base_df["region"].astype(str).unique().tolist())
            if not region_names:
                raise RuntimeError("Podcast ROI map produced no left-hemisphere parcel observations")
            logger.info(
                "Podcast ROI filtering retained %d rows across parcels: %s",
                len(base_df), ", ".join(region_names),
            )
        logger.info(
            "Pooled table: %d rows (%d subject×electrode groups), %d lags",
            len(base_df), base_df.groupby(["subject", "electrode"]).ngroups, n_times,
        )

        window_s = self.cfg.lme.window_s
        if window_s is not None:
            lo, hi = window_s
            win_mask = (analysis_times >= lo) & (analysis_times <= hi)
            n_in_window = int(win_mask.sum())
            if n_in_window == 0:
                raise ValueError(
                    f"window_s={window_s} matches no lags -- analysis_times "
                    f"range [{analysis_times.min():.3f}, {analysis_times.max():.3f}]s"
                )
            logger.info(
                "Mean-window mode: averaging %d lag(s) in [%.3f, %.3f]s into ONE column "
                "(1 fit instead of %d)", n_in_window, lo, hi, n_times,
            )
            y_all = y_all[:, win_mask].mean(axis=1, keepdims=True).astype(np.float32)
            analysis_times = np.array([0.5 * (lo + hi)], dtype=np.float64)
            n_times = 1

        base_parquet_path = out_dir / f"podcast_lme_pooled_table_{safe_revision}.parquet"
        y_npy_path = out_dir / f"podcast_lme_pooled_y_{safe_revision}.npy"
        base_df.to_parquet(base_parquet_path, index=False)
        np.save(y_npy_path, y_all)
        logger.info("Wrote pooled table → %s (%.1f MB)",
                    base_parquet_path, base_parquet_path.stat().st_size / 1e6)
        logger.info("Wrote pooled signal → %s (%.1f MB)",
                    y_npy_path, y_npy_path.stat().st_size / 1e6)
        # Free the in-memory copies — workers read from the files we just wrote.
        del base_df, y_all, subj_parts, elec_parts, chunk_parts, word_idx_parts, wlen_parts, freq_parts, surp_parts, y_parts

        if window_s is not None:
            lag_indices = [0]
        else:
            lag_indices = list(range(0, n_times, max(1, self.cfg.lme.lag_stride)))
        logger.info(
            "Fitting %d lags (stride=%d) with %d worker processes",
            len(lag_indices), self.cfg.lme.lag_stride, self.cfg.lme.n_workers,
        )

        out_csv = out_dir / ("podcast_lme_by_roi.csv" if self.cfg.lme.roi_wise else "podcast_lme_by_lag.csv")
        out_columns = (["region"] if self.cfg.lme.roi_wise else []) + list(_LAG_ROW_COLUMNS) + [
            k for k in (extra_fields or {}) if k not in _LAG_ROW_COLUMNS
        ]
        done_lags: set[int] = set()
        if out_csv.exists():
            try:
                # Full row (not just usecols) — rewritten below with only the
                # error-free rows, so all columns must survive the round trip.
                existing = pd.read_csv(out_csv)
                # Legacy CSVs (pre-revision-fix) have no "revision" column at
                # all -- treat every existing row there as "main" so an old
                # single-checkpoint run isn't spuriously re-fit.
                existing_revision = existing["revision"] if "revision" in existing.columns else "main"
                same_rev = existing.loc[existing_revision == revision]
                # Only rows with no error are genuinely done — a crashed lag
                # (e.g. a worker OOM-killed mid-fit) must be retried, not
                # silently treated as complete.
                good = same_rev.loc[same_rev["error"].isna()]
                if self.cfg.lme.roi_wise:
                    done_lags = set(zip(good["region"].astype(str), good["lag"].astype(int)))
                else:
                    done_lags = set(good["lag"].tolist())
                n_failed = int(same_rev["error"].notna().sum())
                if done_lags:
                    logger.info("Skipping %d already-fit lag(s) for revision=%s", len(done_lags), revision)
                if n_failed:
                    logger.warning(
                        "%d previously-crashed lag(s) found in %s for revision=%s — will retry",
                        n_failed, out_csv, revision,
                    )
                    # Drop the stale crash rows now, so a successful retry
                    # appends a clean row instead of leaving two rows (one
                    # failed, one succeeded) for the same (revision, lag).
                    with csv_file_lock(out_csv):
                        existing_all = pd.read_csv(out_csv)
                        existing_all_revision = (
                            existing_all["revision"] if "revision" in existing_all.columns else "main"
                        )
                        keep = existing_all.loc[
                            (existing_all_revision != revision) | existing_all["error"].isna()
                        ]
                        keep.reindex(columns=out_columns).to_csv(out_csv, index=False)
            except Exception:
                logger.warning("Could not read existing lag CSV; re-fitting all lags.")

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        predictions_dir: Path | None = None
        if self.cfg.lme.save_word_predictions:
            predictions_dir = out_dir / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)

        fit_regions = region_names if self.cfg.lme.roi_wise else [None]
        tasks = [(region, li) for region in fit_regions for li in lag_indices
                 if ((region, li) not in done_lags if self.cfg.lme.roi_wise else li not in done_lags)]
        rows: list[dict] = []
        # max_tasks_per_child: recycle each worker process after ONE lag fit.
        # lme4/rpy2's embedded R interpreter leaks memory across repeated
        # fits within a long-lived process — this held true regardless of
        # worker count (observed OOM kills with 10, 20, and 40 workers,
        # always the pool as a whole dying partway through the lag grid).
        # Since a single fit takes minutes and respawning a worker (fresh
        # Python + R init) costs well under a second, this trades a
        # negligible per-lag overhead for eliminating the leak entirely.
        with ProcessPoolExecutor(
            max_workers=self.cfg.lme.n_workers, mp_context=ctx, max_tasks_per_child=1,
        ) as pool:
            futures = {
                pool.submit(
                    _fit_one_lag_worker,
                    str(base_parquet_path), str(y_npy_path), li, float(analysis_times[li]),
                    train_frac=self.cfg.lme.train_frac,
                    random_state=self.cfg.lme.split_random_state,
                    verbose=self.cfg.lme.verbose,
                    allow_slope_fallback=self.cfg.lme.allow_slope_fallback,
                    predictions_dir=str(predictions_dir) if predictions_dir else None,
                    region=region,
                ): (region, li)
                for region, li in tasks
            }
            for fut in as_completed(futures):
                region, li = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    logger.error("Lag %d worker crashed: %s", li, exc, exc_info=True)
                    row = {"lag": li, "error": str(exc)}
                if region is not None:
                    row["region"] = region
                row["revision"] = revision
                if extra_fields:
                    row.update(extra_fields)
                rows.append(row)
                # Reindex to the fixed column order — a crash-fallback row
                # (only "lag"/"error" keys) must not misalign against the
                # header when appended without its own header.
                with csv_file_lock(out_csv):
                    pd.DataFrame([row]).reindex(columns=out_columns).to_csv(
                        out_csv, mode="a", header=not out_csv.exists(), index=False,
                    )
                logger.info(
                    "  [%s | lag %d, t=%.3fs] done in %.1fs — delta_loglik_test=%s converged=%s%s",
                    region or "pooled", li, row.get("lag_time_s", float("nan")), row.get("fit_wall_time_s", float("nan")),
                    f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
                    row.get("converged"),
                    " (slope dropped)" if row.get("slope_dropped") else "",
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Feature extraction (surprisal from cache, baseline covariates)
    # ------------------------------------------------------------------

    def _load_surprisal(self, words: list[str]) -> np.ndarray:
        """Final-layer logit-lens surprisal, read from the existing cache
        (this benchmark is CPU-only and never loads an LLM itself)."""
        mcfg = self.cfg.model
        dcfg = self.cfg.data

        if "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower():
            # No cache needed — cheap enough (~17M params) to recompute
            # directly on CPU, same bounded moving-window context as the
            # extraction stage (podcast.py::_extract_features).
            from ..datasets.podcast import build_word_context
            from ..features.cbr_rnn import batch_surprisal_cbr_rnn

            contexts = build_word_context(words)
            surp = batch_surprisal_cbr_rnn(
                mcfg.model_name, contexts, words,
                device="cpu", n_params=mcfg.n_params,
            )  # (n_words, 1)
            return surp[:, 0].astype(np.float64)

        from ..features.logit_lens import logit_lens_surprisal_all_layers

        if not dcfg.feature_cache_dir:
            raise ValueError("PodcastLmeBenchmark requires data.feature_cache_dir (surprisal cache)")

        # See logit_lens.py's `dtype` docstring — BabyLM-community's
        # gpt-bert-* models need float32 (their custom DWA accumulator isn't
        # bf16-safe). Only matters on a cache-miss fallback load; the
        # prefetch stage should always leave a warm cache here.
        surp_dtype = None
        if "gpt-bert" in mcfg.model_name.lower():
            import torch

            surp_dtype = torch.float32

        surp_all = logit_lens_surprisal_all_layers(
            words,
            mcfg.model_name,
            revision=mcfg.model_revision,
            device="cpu",
            max_length=dcfg.max_length,
            cache_dir=dcfg.feature_cache_dir,
            section_id=0,
            lang="en",
            tokenizer_no_revision=dcfg.tokenizer_no_revision,
            n_params=mcfg.n_params,
            trust_remote_code=dcfg.trust_remote_code,
            dtype=surp_dtype,
            # This benchmark is CPU-only and never loads an LLM (see
            # feature_cache_dir's docstring above) — a cache miss must raise
            # a clear error here, not silently fall back to a CPU model
            # load, which for the FLA-based trust_remote_code families
            # (stickbreaking/alibi/forgetting/hard-n-gram) crashes with an
            # inscrutable "RuntimeError: 0 active drivers" from Triton
            # trying to import on a GPU-less node.
            require_cache_hit=True,
        )  # (n_words, n_layers, 1) — cache hit expected, no model load
        return surp_all[:, -1, 0].astype(np.float64)

    def _load_baseline_covariates(self, words: list[str]) -> np.ndarray:
        from ..features.word_frequency import compute_wlen_wordfreq_covariates

        return compute_wlen_wordfreq_covariates(
            words, language=self.cfg.data.wordfreq_language,
        ).astype(np.float64)
