"""Broderick 2018 EEG encoding benchmark — LME fit POOLED across all
subjects, one model per LM layer, region-averaged.

Two earlier versions of this pipeline were tried and abandoned:
1. Pooled all 19 subjects into ONE mixed model per layer, with ``subject``
   as a random effect carrying a 4-term correlated slope. Region-averaging
   (collapsing 128 BioSemi channels into 4 anatomical groups via
   ``channel_region``) had already cut the pooled table from ~28M to ~870K
   rows, but in production this STILL failed to complete even a single
   layer fit within a 1-4h walltime — the bottleneck was the
   covariance-search dimensionality from the rich correlated ``subject``
   slope, not data volume (see ``neuroai/regressors/broderick_lme.py``
   module docstring).
2. Fit one LME PER SUBJECT instead (``subject`` never a grouping factor),
   then averaged the per-subject estimates across subjects downstream.
   This avoided the wall, but was never trusted for the report: with no
   single model per layer there was no natural place to attach a
   convergence check, and the per-subject-then-averaged numbers silently
   mixed in blown-up, non-significant fits.

This version pools again, but with ``subject`` as an INTERCEPT-ONLY random
effect (no slope) crossed with ``region`` (also a genuinely crossed factor
here — same 4 labels for every subject) — a completely different
covariance-search cost than version 1's correlated slope:

    baseline: eeg ~ wlen_z + freq_z + (1 | subject) + (1 | region)
    full:     eeg ~ wlen_z + freq_z + surprisal_z + prev_surprisal_z +
                     (1 | subject) + (1 + surprisal_z | region)

This restores ONE fit per layer, so ``converged`` is a real per-layer flag
again (no averaging across per-subject outcomes needed).

Pipeline (per model checkpoint)
--------------------------------
1. Load the canonical word list across all 20 audiobook segments and
   extract per-layer logit-lens surprisal via ONE forward pass PER SEGMENT
   (continuous-text extractor). ``prev_surprisal`` is that same array
   shifted by one word within each segment — no second extraction pass
   needed, same as Fedorenko2016/Podcast.
2. Compute word length + wordfreq covariates for every canonical word.
3. For each subject: load + preprocess that subject's EEG for each
   available run, extract word-locked epochs, baseline-correct, reduce to
   one scalar per (word, channel) via the target window, then average over
   the channels in each of the 4 ``channel_region`` groups — one row per
   (region, word) for THIS subject, tagged with a ``subject`` column.
4. Concatenate every subject's table into ONE shared table (parquet) with
   its shared surprisal/prev-surprisal slice (npy), then fit one POOLED
   LME per layer via a process pool.
5. Append each layer's result to the by-layer CSV the moment it completes
   (crash-safe restart, keyed by layer alone).

Output
------
``broderick_lme_by_layer.csv`` — one row per layer, from the pooled fit —
the primary result table, written incrementally.
"""

from __future__ import annotations

import logging
import time
import typing as tp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.broderick import (
    BroderickDataset,
    channel_region,
    load_all_words,
    load_eeg_run,
)
from ..datasets.podcast import extract_word_epochs
from ..sweep import csv_file_lock
from .base import ModelConfig


def _pw_dir(out_dir):
    """Per-word prediction directory: mirrors ``out_dir`` under $SCRATCH (word_predictions_root)
    so large per-word CSVs never land on $WORK (2026-10-12)."""
    from .base import word_predictions_root

    d = word_predictions_root(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d

logger = logging.getLogger(__name__)

_LAYER_ROW_COLUMNS = (
    "layer", "beta", "se", "t_stat", "p_value", "loglik", "aic", "bic",
    "baseline_loglik", "delta_loglik_train", "delta_loglik_test",
    "baseline_loglik_test", "loglik_test", "lrt_p_value",
    "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr", "converged",
    "slope_dropped", "n_subjects", "n_obs", "n_obs_test", "error", "fit_wall_time_s",
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BroderickLmeDataConfig:
    """Dataset location, EEG preprocessing, and window-reduction options."""

    bids_root: str
    subjects: list[str] | None = None
    n_runs: int = 20
    # EEG scalar window (s relative to word onset) — mean amplitude over this
    # window is the LME's "eeg" outcome, analogous to the N400 datasets'
    # fixed centro-parietal window (default 300-500 ms).
    window_tmin: float = 0.3
    window_tmax: float = 0.5
    # Epoch extraction window must cover both the baseline and the target
    # window above.
    epoch_tmin: float = -0.2
    epoch_tmax: float = 1.0
    baseline: tuple[float, float] | None = (-0.2, 0.0)
    l_freq: float = 1.0
    h_freq: float = 8.0
    target_sfreq: float = 32.0
    average_ref: bool = True
    drop_bads: bool = True
    max_length: int | None = None
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    wordfreq_language: str = "en"
    # Device for the (cheap, ~11K-word) surprisal extraction forward passes.
    device: str = "cpu"
    # Surprisal cache dir. Same two-stage GPU/CPU split as every other pooled
    # LME in this codebase: R (pymer4/lme4) is CPU-only on Jean Zay, so a GPU
    # job populates this cache before the CPU-only LME job reads from it.
    feature_cache_dir: str | None = None


@dataclass
class BroderickLmeConfig:
    train_frac: float = 0.75
    split_random_state: int = 0
    verbose: bool = False
    # Barr et al. (2013)-style simplification ladder if the full model fails
    # to converge — see fit_broderick_lme_one_layer's docstring for the
    # 2-step fallback order (region correlation -> region slope dropped).
    # subject is intercept-only throughout, nothing to simplify there.
    allow_slope_fallback: bool = True
    # Which layers to fit: 1 = all layers; N>1 = every Nth layer.
    layer_stride: int = 1
    # Fit only the final layer (overrides layer_stride).
    last_layer_only: bool = False
    # Process-pool size for fitting layers concurrently (one pooled,
    # all-subjects fit per layer) — each worker is a separate process with
    # its own embedded R/pymer4 interpreter. Each fit is now much heavier
    # (~870K pooled rows vs. a few thousand per-subject rows), so this may
    # need to be lower than the old per-(subject,layer) default depending on
    # available CPU/RAM — test empirically.
    n_workers: int = 10
    # CPU sweep runs are the LME-only half of a two-stage pipeline — a cache
    # miss there should fail fast rather than silently loading a model.
    require_cached_surprisal: bool = False
    # Save each held-out row's observed eeg + baseline/full model predictions
    # + per-observation log-likelihoods to a per-word CSV — needed for
    # post-hoc per-word difficulty splits (e.g. DLT cost), same convention
    # as the behavioral LME datasets. On by default.
    save_word_predictions: bool = True


@dataclass
class BroderickLmeBenchmarkConfig:
    data: BroderickLmeDataConfig
    model: ModelConfig
    lme: BroderickLmeConfig = field(default_factory=BroderickLmeConfig)
    output_dir: str = "outputs"
    canonical_model_name: str | None = None


# ---------------------------------------------------------------------------
# Worker (module-level for spawn-based multiprocessing pickling)
# ---------------------------------------------------------------------------

def _fit_one_layer_worker(
    base_parquet_path: str,
    surprisal_npy_path: str,
    prev_surprisal_npy_path: str,
    layer_idx: int,
    n_subjects: int,
    *,
    train_frac: float,
    random_state: int,
    verbose: bool,
    allow_slope_fallback: bool,
    save_word_predictions: bool = False,
) -> dict[str, tp.Any]:
    """Runs in its own spawned process: load the shared all-subjects table,
    fit ONE pooled LME for this layer."""
    import numpy as _np
    import pandas as _pd

    from ..regressors.broderick_lme import fit_broderick_lme_one_layer

    t0 = time.time()
    row: dict[str, tp.Any] = {"layer": layer_idx, "n_subjects": n_subjects}
    per_word_out: pd.DataFrame | None = None
    try:
        base_df = _pd.read_parquet(base_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        prev_surp_all = _np.load(prev_surprisal_npy_path, mmap_mode="r")
        base_df["surprisal"] = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)
        base_df["prev_surprisal"] = _np.asarray(prev_surp_all[:, layer_idx], dtype=_np.float64)

        fit = fit_broderick_lme_one_layer(
            base_df,
            train_frac=train_frac,
            random_state=random_state,
            verbose=verbose,
            allow_slope_fallback=allow_slope_fallback,
            save_per_word=save_word_predictions,
        )
        if fit.per_word is not None and not fit.per_word.empty:
            per_word_out = fit.per_word.copy()
            per_word_out.insert(0, "layer", layer_idx)
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
    except Exception as exc:  # noqa: BLE001 — one bad fit must not kill the pool
        logger.error("Layer %d failed", layer_idx, exc_info=True)
        row["error"] = str(exc)
    row["fit_wall_time_s"] = time.time() - t0
    return {"row": row, "per_word": per_word_out}


# ---------------------------------------------------------------------------
# Sentence-like grouping for surprisal extraction (mirrors Fedorenko2016LmeBenchmark)
# ---------------------------------------------------------------------------

def _build_segment_groups(words_df: pd.DataFrame) -> list[tuple[int, list[str], list[int]]]:
    """One ``(segment_id, words, word_idxs)`` tuple per audiobook segment."""
    groups: list[tuple[int, list[str], list[int]]] = []
    for seg_id, g in words_df.sort_values(["segment_id", "word_idx"]).groupby("segment_id"):
        words = g["word"].astype(str).tolist()
        idxs = g["word_idx"].astype(int).tolist()
        groups.append((int(seg_id), words, idxs))
    return groups


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class BroderickLmeBenchmark:
    """Broderick EEG LME fit pooled across all subjects, one model per layer, region-averaged."""

    def __init__(self, cfg: BroderickLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def extract_and_cache_surprisal(self) -> tuple[np.ndarray, np.ndarray]:
        """GPU-stage entry point: populate ``data.feature_cache_dir`` only.

        Loads the LLM, computes per-layer logit-lens surprisal for every
        canonical word (one forward pass per segment) and returns — does
        not touch EEG data, does not fit anything. Run once per model (GPU
        job) before the CPU-only ``run()`` LME sweep, same two-stage split
        as every other pooled LME here.
        """
        dcfg = self.cfg.data
        words_df = load_all_words(dcfg.bids_root, n_runs=dcfg.n_runs)
        return self._extract_segment_surprisal(words_df)

    def run(self) -> pd.DataFrame:
        dcfg = self.cfg.data
        lcfg = self.cfg.lme
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = out_dir / "broderick_lme_by_layer.csv"

        words_df = load_all_words(dcfg.bids_root, n_runs=dcfg.n_runs)
        ds = BroderickDataset(
            dcfg.bids_root, subjects=dcfg.subjects, drop_bads=dcfg.drop_bads, n_runs=dcfg.n_runs,
        )
        subjects = ds.available_subjects()

        surprisal_items, prev_surprisal_items = self._extract_segment_surprisal(words_df)
        wlen_freq = self._load_baseline_covariates(words_df["word"].astype(str).tolist())
        n_layers = surprisal_items.shape[1]

        item_finite = (
            np.isfinite(surprisal_items).all(axis=1)
            & np.isfinite(prev_surprisal_items).all(axis=1)
            & np.isfinite(wlen_freq).all(axis=1)
        )
        if not item_finite.all():
            n_dropped = int((~item_finite).sum())
            logger.warning(
                "[%s] Dropping %d/%d word(s) with non-finite features before "
                "pooling (segment-initial words have no left context for "
                "prev_surprisal)", model_name, n_dropped, len(words_df),
            )
        keep_word_idx = set(words_df.loc[item_finite, "word_idx"].astype(int).tolist())
        word_idx_to_pos = {int(w): i for i, w in enumerate(words_df["word_idx"].astype(int).tolist())}

        done_layers: set[int] = set()
        if summary_csv.exists():
            try:
                existing = pd.read_csv(summary_csv)
                if set(_LAYER_ROW_COLUMNS) - set(existing.columns):
                    raise ValueError(
                        f"{summary_csv} has a different schema (missing columns) — "
                        "likely a leftover from an older fitting scheme."
                    )
                good = existing.loc[existing["error"].isna()]
                done_layers = set(int(li) for li in good["layer"])
                n_failed = int(existing["error"].notna().sum())
                if done_layers:
                    logger.info("Skipping %d already-fit layer(s)", len(done_layers))
                if n_failed:
                    logger.warning(
                        "%d previously-crashed layer row(s) found in %s — will retry",
                        n_failed, summary_csv,
                    )
                    good.reindex(columns=_LAYER_ROW_COLUMNS).to_csv(summary_csv, index=False)
            except Exception as exc:
                # Rename the unreadable/wrong-schema file out of the way rather
                # than leaving it in place: the append loop below decides
                # whether to (re-)write a header purely from
                # ``summary_csv.exists()``, so leaving a stale file there
                # would silently produce a mixed-schema CSV (old header,
                # new-schema rows appended underneath).
                bad_path = summary_csv.with_suffix(".stale.csv")
                logger.warning(
                    "Could not use existing by-layer CSV (%s) — re-fitting all "
                    "layers; moved the old file to %s.", exc, bad_path,
                )
                summary_csv.replace(bad_path)

        candidate_layers = [n_layers - 1] if lcfg.last_layer_only else range(0, n_layers, max(1, lcfg.layer_stride))
        layer_indices = [li for li in candidate_layers if li not in done_layers]

        # Build every subject's region table, tag it with a ``subject``
        # column, and concatenate into ONE shared pooled table — one process
        # pool below fits one layer at a time on the full pooled data,
        # rather than the old per-subject file-per-worker split.
        subj_tables: list[pd.DataFrame] = []
        for subject in subjects:
            subj_df = self._build_subject_region_table(subject, ds, words_df, keep_word_idx)
            if subj_df is None or subj_df.empty:
                logger.warning("[%s | %s] no valid rows — skipping", model_name, subject)
                continue
            subj_df["subject"] = subject
            subj_tables.append(subj_df)

        if not subj_tables or not layer_indices:
            logger.warning("[%s] no data or no pending layers — nothing to fit.", model_name)
            return pd.DataFrame()

        pooled_df = pd.concat(subj_tables, ignore_index=True)
        del subj_tables
        n_subjects = pooled_df["subject"].nunique()

        row_pos = pooled_df["word_idx"].map(word_idx_to_pos).to_numpy(dtype=np.int64)
        surprisal_by_row = surprisal_items[row_pos]
        prev_surprisal_by_row = prev_surprisal_items[row_pos]
        pooled_df["wlen"] = wlen_freq[row_pos, 0]
        pooled_df["freq"] = wlen_freq[row_pos, 1]

        logger.info(
            "[%s] pooled table: %d rows (%d subjects, %d regions), %d layer(s) pending",
            model_name, len(pooled_df), n_subjects, pooled_df["region"].nunique(), len(layer_indices),
        )

        base_parquet_path = out_dir / "broderick_lme_table.parquet"
        surprisal_npy_path = out_dir / "broderick_lme_surprisal.npy"
        prev_surprisal_npy_path = out_dir / "broderick_lme_prev_surprisal.npy"
        pooled_df.to_parquet(base_parquet_path, index=False)
        np.save(surprisal_npy_path, surprisal_by_row.astype(np.float32))
        np.save(prev_surprisal_npy_path, prev_surprisal_by_row.astype(np.float32))
        del pooled_df, surprisal_by_row, prev_surprisal_by_row

        logger.info(
            "[%s] Fitting %d layer(s) with %d worker processes",
            model_name, len(layer_indices), lcfg.n_workers,
        )

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        per_word_csv = _pw_dir(out_dir) / "broderick_lme_per_word.csv"
        rows_out: list[dict] = []
        # max_tasks_per_child=1: recycle each worker after ONE fit —
        # lme4/rpy2's embedded R interpreter leaks memory across repeated
        # fits within a long-lived process.
        with ProcessPoolExecutor(
            max_workers=lcfg.n_workers, mp_context=ctx, max_tasks_per_child=1,
        ) as pool:
            futures = {
                pool.submit(
                    _fit_one_layer_worker,
                    str(base_parquet_path), str(surprisal_npy_path), str(prev_surprisal_npy_path),
                    li, n_subjects,
                    train_frac=lcfg.train_frac,
                    random_state=lcfg.split_random_state,
                    verbose=lcfg.verbose,
                    allow_slope_fallback=lcfg.allow_slope_fallback,
                    save_word_predictions=lcfg.save_word_predictions,
                ): li
                for li in layer_indices
            }
            for fut in as_completed(futures):
                li = futures[fut]
                try:
                    result = fut.result()
                    row, per_word = result["row"], result["per_word"]
                except Exception as exc:
                    logger.error("Layer %d worker crashed: %s", li, exc, exc_info=True)
                    row, per_word = {"layer": li, "n_subjects": n_subjects, "error": str(exc)}, None
                rows_out.append(row)
                pd.DataFrame([row]).reindex(columns=_LAYER_ROW_COLUMNS).to_csv(
                    summary_csv, mode="a", header=not summary_csv.exists(), index=False,
                )
                if per_word is not None:
                    with csv_file_lock(per_word_csv):
                        per_word.to_csv(
                            per_word_csv, mode="a", header=not per_word_csv.exists(), index=False,
                        )
                logger.info(
                    "  [layer %d] done in %.1fs — delta_loglik_test=%s converged=%s%s",
                    li, row.get("fit_wall_time_s", float("nan")),
                    f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
                    row.get("converged"),
                    " (slope dropped)" if row.get("slope_dropped") else "",
                )

        return pd.DataFrame(rows_out)

    # ------------------------------------------------------------------
    # Region-averaged EEG table (region x word) for ONE subject
    # ------------------------------------------------------------------

    def _build_subject_region_table(
        self,
        subject: str,
        ds: BroderickDataset,
        words_df: pd.DataFrame,
        keep_word_idx: set[int],
    ) -> pd.DataFrame | None:
        """One row per (region, word) for THIS subject: mean EEG amplitude
        in the target post-onset window, averaged over the channels in
        each of the 4 ``channel_region`` groups (left/right/posterior/
        occipital).
        """
        dcfg = self.cfg.data
        parts: list[pd.DataFrame] = []

        available_runs = ds.available_runs(subject)
        if not available_runs:
            return None
        good_ch = ds.good_channel_names(subject, run=1)
        if not good_ch:
            return None
        regions = [channel_region(ch) for ch in good_ch]
        region_groups: dict[str, list[int]] = {}
        for ci, r in enumerate(regions):
            if r is None:
                continue
            region_groups.setdefault(r, []).append(ci)
        if not region_groups:
            logger.warning("[%s] no channels mapped to a region — skipping", subject)
            return None

        for run in available_runs:
            run_words_df = words_df[
                (words_df["run"] == run) & (words_df["word_idx"].isin(keep_word_idx))
            ]
            if run_words_df.empty:
                continue

            word_idx = run_words_df["word_idx"].to_numpy(dtype=np.int64)
            word_onsets = run_words_df["onset"].to_numpy(dtype=np.float64)

            try:
                data, sfreq, ch_names_run = load_eeg_run(
                    dcfg.bids_root, subject, run,
                    l_freq=dcfg.l_freq, h_freq=dcfg.h_freq,
                    target_sfreq=dcfg.target_sfreq, average_ref=dcfg.average_ref,
                    good_channels=good_ch,
                )
            except Exception as exc:
                logger.warning("[%s] run %02d: EEG load failed: %s", subject, run, exc)
                continue
            if list(ch_names_run) != good_ch:
                logger.warning(
                    "[%s] run %02d: channel order mismatch vs. run-01 — skipping", subject, run,
                )
                continue

            epochs, valid_mask, epoch_times = extract_word_epochs(
                data, sfreq, recording_tmin=0.0, word_onsets=word_onsets,
                tmin=dcfg.epoch_tmin, tmax=dcfg.epoch_tmax,
            )
            if not valid_mask.any():
                continue

            if dcfg.baseline is not None:
                bl_tmin, bl_tmax = dcfg.baseline
                bl_mask = (epoch_times >= bl_tmin) & (epoch_times <= bl_tmax)
                if bl_mask.any():
                    epochs = epochs - epochs[:, :, bl_mask].mean(axis=-1, keepdims=True)

            win_mask = (epoch_times >= dcfg.window_tmin) & (epoch_times < dcfg.window_tmax)
            if not win_mask.any():
                logger.warning(
                    "[%s] run %02d: target window [%.2f, %.2f) outside epoch range — skipping",
                    subject, run, dcfg.window_tmin, dcfg.window_tmax,
                )
                continue
            amp = epochs[:, :, win_mask].mean(axis=-1)  # (n_valid, n_channels)

            kept_word_idx = word_idx[valid_mask]
            for region_name, ch_idxs in region_groups.items():
                region_amp = amp[:, ch_idxs].mean(axis=1)  # (n_valid,)
                parts.append(pd.DataFrame({
                    "region": region_name,
                    "chunk": run,
                    "word_idx": kept_word_idx,
                    "eeg": region_amp,
                }))

        if not parts:
            return None
        long_df = pd.concat(parts, ignore_index=True)
        return long_df.dropna(subset=["eeg"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Baseline covariates (word length + wordfreq Zipf frequency)
    # ------------------------------------------------------------------

    def _load_baseline_covariates(self, words: list[str]) -> np.ndarray:
        from ..features.word_frequency import compute_wlen_wordfreq_covariates

        return compute_wlen_wordfreq_covariates(words, language=self.cfg.data.wordfreq_language)

    # ------------------------------------------------------------------
    # Surprisal extraction (per-layer logit-lens, one model load per segment, disk-cached)
    # ------------------------------------------------------------------

    def _extract_segment_surprisal(self, words_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per-layer surprisal for every canonical word, and its within-segment
        predecessor.

        One forward pass per audiobook segment via
        :func:`logit_lens_surprisal_all_layers` — the continuous-text
        extractor, since Broderick's words live inside a real continuous
        narrative. ``prev_surprisal`` is the SAME extraction's output shifted
        by one word within each segment (NaN at each segment's first word)
        — no second extraction pass needed, same as Fedorenko2016/Podcast.

        CBR-RNN is a special case: bypasses ``AutoModelForCausalLM``
        entirely and has a single hidden-state vector per token (no layer
        dimension), so ``batch_surprisal_cbr_rnn`` is used instead,
        returning a single "layer" of surprisal.

        Returns
        -------
        (surprisal, prev_surprisal), each ``(n_words, n_layers)``, in
        ``word_idx`` order (same order as ``words_df``).
        """
        mcfg = self.cfg.model
        dcfg = self.cfg.data
        is_cbr_rnn = "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower()
        require_cache_hit = self.cfg.lme.require_cached_surprisal

        groups = _build_segment_groups(words_df)
        n_layers: int | None = None
        surprisal_by_idx: dict[int, np.ndarray] = {}
        prev_surprisal_by_idx: dict[int, np.ndarray] = {}

        if is_cbr_rnn:
            from ..features.cbr_rnn import batch_surprisal_cbr_rnn

            n_layers = 1
            for seg_id, words, idxs in groups:
                contexts = [" ".join(words[:wi]) for wi in range(len(words))]
                seg_surp = batch_surprisal_cbr_rnn(
                    mcfg.model_name, contexts, list(words),
                    device=dcfg.device, n_params=mcfg.n_params,
                    cache_dir=dcfg.feature_cache_dir, section_id=seg_id,
                    require_cache_hit=require_cache_hit,
                )[:, 0]
                prev = np.full(len(words), np.nan, dtype=np.float32)
                prev[1:] = seg_surp[:-1]
                for i, wi in enumerate(idxs):
                    surprisal_by_idx[wi] = np.array([seg_surp[i]], dtype=np.float32)
                    prev_surprisal_by_idx[wi] = np.array([prev[i]], dtype=np.float32)
        else:
            from ..features.logit_lens import logit_lens_surprisal_all_layers

            surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None
            kw = dict(
                revision=mcfg.model_revision,
                device=dcfg.device,
                max_length=dcfg.max_length,
                tokenizer_no_revision=dcfg.tokenizer_no_revision,
                n_params=mcfg.n_params,
                trust_remote_code=mcfg.trust_remote_code,
                cache_dir=dcfg.feature_cache_dir,
                untrained=mcfg.untrained,
                untrained_seed=mcfg.untrained_seed,
                require_cache_hit=require_cache_hit,
                dtype=surp_dtype,
            )
            for seg_id, words, idxs in groups:
                arr = logit_lens_surprisal_all_layers(
                    words, mcfg.model_name, section_id=seg_id, **kw,
                )  # (n_words_in_segment, n_layers, 1)
                surp = arr[:, :, 0]
                if n_layers is None:
                    n_layers = surp.shape[1]
                prev = np.full_like(surp, np.nan)
                prev[1:] = surp[:-1]
                for i, wi in enumerate(idxs):
                    surprisal_by_idx[wi] = surp[i]
                    prev_surprisal_by_idx[wi] = prev[i]

        assert n_layers is not None
        word_idx_order = words_df["word_idx"].astype(int).tolist()
        surprisal = np.stack(
            [surprisal_by_idx.get(w, np.full(n_layers, np.nan, dtype=np.float32)) for w in word_idx_order],
            axis=0,
        ).astype(np.float32)
        prev_surprisal = np.stack(
            [prev_surprisal_by_idx.get(w, np.full(n_layers, np.nan, dtype=np.float32)) for w in word_idx_order],
            axis=0,
        ).astype(np.float32)
        return surprisal, prev_surprisal
