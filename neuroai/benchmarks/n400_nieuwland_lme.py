"""Nieuwland N400 EEG encoding benchmark — pooled LME per LM layer.

Unlike the per-subject/per-channel ridge encoder (``benchmarks/eeg_erp.py``,
which predicts the full EEG time course from full hidden-state embeddings),
this pools every subject and electrode into ONE mixed-effects model per
layer, using per-layer logit-lens surprisal (Kuribayashi et al. 2025) as the
predictor of interest — same design as ``benchmarks/n400_szewczyk_lme.py``. Nieuwland's 6 centro-parietal
electrodes are also standard 10-20 scalp locations comparable across
subjects, so electrode is a CROSSED random effect (``1 | electrode``).

Two real differences from the single-lab N400 datasets (see
``neuroai/datasets/nieuwland.py`` and ``neuroai/regressors/lme.py::
n400_lme_formulas`` for details):

- **Multi-lab**: 9 UK labs (GLAS excluded — incompatible montage), subject
  IDs are lab-prefixed and globally unique. Fit with ``include_lab=True``,
  adding a crossed ``(1 | lab)`` random intercept on top of subject/electrode.
- **Raw EEG, not a flat CSV**: N400 amplitude must be epoched from BIDS
  EEG via MNE (``load_all_epochs``, reused unchanged from the ridge encoder,
  with ``zscore=False`` so the LME gets raw amplitude — its per-subject
  random intercepts already absorb baseline offsets). Item discovery for the
  cheap GPU extraction stage uses the new EEG-free ``load_all_items``
  instead (pools ``events.tsv`` across subjects, no raw EEG touch).

Scope: N400 mean-amplitude only (200-500ms window, mean per trial×channel),
same convention as the single-lab N400 datasets — NOT the time-resolved/per-lag design
Podcast's pooled LME or Nieuwland's own ridge encoder use.

Pipeline
--------
1. ``load_all_items`` (EEG-free) → per-layer logit-lens surprisal at the
   noun for all ~160 items, plus the previous word's per-layer surprisal
   (spillover control).
2. Compute word length + wordfreq covariates for the noun.
3. ``load_all_epochs(..., zscore=False)`` (full MNE epoching, cached
   per-subject — reused across every model/layer run) → raw EEG epochs,
   trial metadata (subject, lab, item_number, condition).
4. ``n400_window_mean_per_channel`` → per-trial, per-channel raw N400
   amplitude. Melt to long format (one row per trial × channel), join
   surprisal via ``(item_number, condition) → item_condition_id``.
5. Write the pooled base table (parquet) and the (n_rows, n_layers)
   surprisal / prev_surprisal arrays (npy) to disk once, then fit one LME per
   layer via a process pool (pymer4/rpy2 embeds one R interpreter per
   process, so parallelism must be process-based).

Output: ``nieuwland_lme_by_layer.csv`` — one row per layer, written
incrementally as each layer's fit completes.
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

from ..datasets.nieuwland import (
    DEFAULT_EXCLUDE_LABS,
    N400_CHANNELS,
    N400_TMAX,
    N400_TMIN,
    NieuwlandEpochsConfig,
    load_all_epochs,
    load_all_items,
    n400_window_mean_per_channel,
)
from ..sweep import csv_file_lock
from .base import ModelConfig, word_predictions_dir

logger = logging.getLogger(__name__)

# Fixed column order for nieuwland_lme_by_layer.csv — a crash-fallback row
# (only a few keys) is reindexed to this before writing, so it can never
# misalign the CSV's columns.
_LAYER_ROW_COLUMNS = (
    "layer", "beta", "se", "t_stat", "p_value", "loglik", "aic",
    "bic", "baseline_loglik", "delta_loglik_train", "delta_loglik_test",
    "baseline_loglik_test", "loglik_test", "lrt_p_value",
    "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr", "converged",
    "slope_dropped", "n_obs", "n_obs_test", "error", "fit_wall_time_s", "revision",
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NieuwlandLmeDataConfig:
    bids_root: str | Path = "data/nieuwland"
    subjects: list[str] | None = None
    included_in_paper_only: bool = True
    exclude_labs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_LABS))
    target_channels: list[str] = field(default_factory=lambda: list(N400_CHANNELS))
    target_sfreq: float = 32.0
    epoch_tmin: float = -0.2
    epoch_tmax: float = 0.8
    baseline: tuple[float, float] = (-0.2, 0.0)
    artifact_threshold_uv: float = 100.0
    min_trials: int = 20
    n400_tmin: float = N400_TMIN
    n400_tmax: float = N400_TMAX
    # Per-subject epoch cache — shares the same directory (and, if EEG
    # preprocessing params match configs/embedding/neural/nieuwland.yaml
    # exactly, the same cache entries) as the existing ridge sweep.
    epoch_cache_dir: str | None = None
    max_length: int | None = None
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    # wordfreq Zipf frequency — same source as the single-lab N400 datasets' pooled LME.
    wordfreq_language: str = "en"
    # Device for the (cheap, ~160-item) surprisal extraction forward passes.
    device: str = "cpu"
    # Surprisal cache dir. On Jean Zay, R (pymer4/lme4) is only available on
    # CPU partitions, not GPU — so, like Podcast's pooled
    # LME, extraction and LME fitting must run as separate jobs: a GPU job
    # populates this cache (see ``scripts/run_nieuwland_surprisal_extract.
    # py``), then the CPU-only LME job (``scripts/run_nieuwland_lme_sweep.
    # py``) reads from it. A cache miss on the CPU-only job silently falls
    # back to loading the LLM on CPU (slow, but not wrong) — same soft
    # contract as Podcast's pooled LME.
    feature_cache_dir: str | None = None
    # Path to the paper's ``nieuwland_2018.tsv`` (subject, cloze, lab, n400 (uV),
    # FullText, TargetWords). When set, the pooled table is built from these
    # preprocessed per-trial N400 values (paper-formula mode only) and raw EEG is
    # never epoched.
    paper_n400_tsv: str | None = None


@dataclass
class NieuwlandLmeConfig:
    # Multi-lab dataset — the one real default difference from the single-lab N400 datasets'
    # LmeConfig, which default include_lab=False.
    include_lab: bool = True
    # Fit Nieuwland et al.'s own paper-exact formula (surprisal + lab as a
    # FIXED effect, pure random intercepts for subject/item, no electrode
    # term -- N400 averaged across the 6 centro-parietal channels per trial,
    # matching the paper's already-channel-averaged outcome) instead of this
    # codebase's generic per-channel correlated-random-slope structure. Only
    # possible because ``target_channels`` already defaults to exactly the
    # paper's 6 centro-parietal channels (N400_CHANNELS) -- no separate
    # channel-selection logic needed, just averaging across the (already
    # correctly restricted) channel axis. See
    # neuroai/regressors/n400_pooled_lme.py's fit_n400_lme_one_layer_paper.
    use_paper_formula: bool = True
    train_frac: float = 0.75
    split_random_state: int = 0
    verbose: bool = False
    # Barr et al. (2013)-style simplification ladder if the full model (with
    # the correlated random slope on subject) fails to converge: drop the
    # correlation first, then the slope entirely. Applied identically to
    # every layer so comparisons stay valid.
    allow_slope_fallback: bool = True
    # Which layers to fit: 1 = all layers; N>1 = every Nth layer.
    layer_stride: int = 1
    # If True, fit ONLY the final layer (index n_layers-1), ignoring layer_stride.
    last_layer_only: bool = False
    # Process-pool size for fitting layers concurrently. Nieuwland's pooled
    # table (~300 subjects × 6 channels × ~160 items ≈ 300K+ rows) is bigger
    # than the single-lab N400 datasets', closer to Podcast's regime — default lower
    # (10, Podcast's proven-safe value) rather than the single-lab N400 datasets' 20.
    n_workers: int = 10
    # Save held-out per-trial predictions (one row per test-set subject ×
    # electrode × item) alongside the usual per-layer summary — needed for
    # post-hoc per-word difficulty splits (e.g. DLT cost). Written one
    # parquet file per layer (mirrors Podcast's ``predictions/lag*.parquet``
    # convention) since layers fit concurrently in separate processes. On by
    # default.
    save_word_predictions: bool = True


@dataclass
class NieuwlandLmeBenchmarkConfig:
    data: NieuwlandLmeDataConfig
    model: ModelConfig
    lme: NieuwlandLmeConfig = field(default_factory=NieuwlandLmeConfig)
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
    *,
    include_lab: bool,
    train_frac: float,
    random_state: int,
    verbose: bool,
    allow_slope_fallback: bool,
    save_word_predictions: bool = False,
    per_word_dir: str | None = None,
    use_paper_formula: bool = True,
) -> dict[str, tp.Any]:
    """Runs in its own spawned process: load the shared table, fit one layer."""
    import numpy as _np
    import pandas as _pd

    from ..regressors.n400_pooled_lme import fit_n400_lme_one_layer, fit_n400_lme_one_layer_paper

    t0 = time.time()
    row: dict[str, tp.Any] = {"layer": layer_idx}
    try:
        base_df = _pd.read_parquet(base_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        base_df["surprisal"] = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)

        if use_paper_formula:
            fit = fit_n400_lme_one_layer_paper(
                base_df,
                dataset="nieuwland",
                train_frac=train_frac,
                random_state=random_state,
                verbose=verbose,
                allow_slope_fallback=allow_slope_fallback,
                save_per_word=save_word_predictions,
            )
        else:
            prev_surp_all = _np.load(prev_surprisal_npy_path, mmap_mode="r")
            base_df["prev_surprisal"] = _np.asarray(prev_surp_all[:, layer_idx], dtype=_np.float64)
            fit = fit_n400_lme_one_layer(
                base_df,
                include_lab=include_lab,
                train_frac=train_frac,
                random_state=random_state,
                verbose=verbose,
                allow_slope_fallback=allow_slope_fallback,
                save_per_word=save_word_predictions,
            )
        if save_word_predictions and fit.per_word is not None and per_word_dir is not None:
            _pd.DataFrame(fit.per_word).to_parquet(
                f"{per_word_dir}/layer{layer_idx:03d}.parquet", index=False,
            )
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
    except Exception as exc:  # noqa: BLE001 — one bad layer must not kill the pool
        logger.error("Layer %d failed", layer_idx, exc_info=True)
        row["error"] = str(exc)
    row["fit_wall_time_s"] = time.time() - t0
    return row


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class NieuwlandLmeBenchmark:
    """Pooled Nieuwland N400 LME: one mixed-effects model per LM layer."""

    def __init__(self, cfg: NieuwlandLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def extract_and_cache_surprisal(self) -> tuple[np.ndarray, np.ndarray]:
        """GPU-stage entry point: populate ``data.feature_cache_dir`` only.

        Loads the LLM, computes per-layer logit-lens surprisal for every
        item (noun + previous-word spillover control), and returns — does
        not touch raw EEG (uses the EEG-free ``load_all_items``, not
        ``load_all_epochs``), does not fit anything. Run this once per model
        (GPU job) before the CPU-only ``run()`` LME sweep, same two-stage
        split as Podcast's pooled LME (R/pymer4 is
        CPU-only on Jean Zay, so extraction and fitting can't share a job).
        """
        items = self._load_items()
        contexts = items["sentence_context"].astype(str).tolist()
        targets = items["noun"].astype(str).tolist()
        return self._extract_item_surprisal(contexts, targets)

    def run(self, extra_fields: dict[str, tp.Any] | None = None) -> pd.DataFrame:
        dcfg = self.cfg.data
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        revision = self.cfg.model.model_revision or "main"

        items = self._load_items()
        contexts = items["sentence_context"].astype(str).tolist()
        targets = items["noun"].astype(str).tolist()

        surprisal_items, prev_surprisal_items = self._extract_item_surprisal(contexts, targets)
        wlen_freq = self._load_baseline_covariates(targets)

        item_finite = (
            np.isfinite(surprisal_items).all(axis=1)
            & np.isfinite(prev_surprisal_items).all(axis=1)
            & np.isfinite(wlen_freq).all(axis=1)
        )
        if not item_finite.all():
            n_dropped = int((~item_finite).sum())
            logger.warning(
                "Dropping %d/%d item(s) with non-finite features before pooling",
                n_dropped, len(items),
            )
        keep_item_condition_ids = set(items.loc[item_finite, "item_condition_id"].tolist())

        if dcfg.paper_n400_tsv:
            # Reuse the paper's already-preprocessed per-trial N400 (200-500 ms mean over
            # the 6 centro-parietal channels) instead of re-epoching raw EEG.
            base_df, item_cond_id_kept = self._paper_trials(items, keep_item_condition_ids)
            surprisal_by_row = surprisal_items[item_cond_id_kept]
            prev_surprisal_by_row = prev_surprisal_items[item_cond_id_kept]
            n_layers = surprisal_items.shape[1]
        else:
            # ── Full MNE epoching (expensive, cached per-subject; zscore=False
            #    so random intercepts — not pre-standardization — absorb subject
            #    baseline, matching the single-lab N400 datasets' raw-DV convention) ──────
            epoch_cfg = NieuwlandEpochsConfig(
                bids_root=dcfg.bids_root,
                subjects=dcfg.subjects,
                included_in_paper_only=dcfg.included_in_paper_only,
                exclude_labs=dcfg.exclude_labs,
                target_channels=dcfg.target_channels,
                target_sfreq=dcfg.target_sfreq,
                epoch_tmin=dcfg.epoch_tmin,
                epoch_tmax=dcfg.epoch_tmax,
                baseline=dcfg.baseline,
                artifact_threshold_uv=dcfg.artifact_threshold_uv,
                min_trials=dcfg.min_trials,
                epoch_cache_dir=dcfg.epoch_cache_dir,
                zscore=False,
            )
            epochs_result = load_all_epochs(epoch_cfg)
            Y, meta = epochs_result.Y, epochs_result.meta
            times, channel_names = epochs_result.times, epochs_result.channel_names

            n400_per_ch = n400_window_mean_per_channel(
                Y, times, dcfg.n400_tmin, dcfg.n400_tmax
            )  # (n_trials, n_channels)

            meta = meta.merge(
                items[["item_number", "condition", "item_condition_id"]],
                on=["item_number", "condition"], how="left",
            )

            trial_mask = (
                meta["item_condition_id"].notna().values
                & meta["item_condition_id"].isin(keep_item_condition_ids).values
                & np.isfinite(n400_per_ch).all(axis=1)
            )
            n_kept = int(trial_mask.sum())
            n_channels = len(channel_names)
            logger.info(
                "Nieuwland: %d/%d trials matched a kept item and have finite N400",
                n_kept, len(meta),
            )

            kept_idx = np.where(trial_mask)[0]
            item_cond_id_kept = meta["item_condition_id"].values[kept_idx].astype(np.int64)
            n_layers = surprisal_items.shape[1]

            if self.cfg.lme.use_paper_formula:
                # One row per TRIAL (not trial x channel): N400 averaged across
                # the already-restricted 6 centro-parietal channels
                # (target_channels == N400_CHANNELS by default), matching the
                # paper's own pre-averaged outcome exactly -- no electrode term.
                n400_avg = np.nanmean(n400_per_ch[kept_idx], axis=1).astype(np.float64)
                surprisal_by_row = surprisal_items[item_cond_id_kept]
                prev_surprisal_by_row = prev_surprisal_items[item_cond_id_kept]
                base_df = pd.DataFrame({
                    "subject": meta["subject"].astype(str).values[kept_idx],
                    "lab": meta["lab"].astype(str).values[kept_idx],
                    "frame_id": meta["item_number"].values[kept_idx],
                    "item_id": item_cond_id_kept,
                    "n400": n400_avg,
                })
            else:
                subj_rep = np.repeat(meta["subject"].astype(str).values[kept_idx], n_channels)
                lab_rep = np.repeat(meta["lab"].astype(str).values[kept_idx], n_channels)
                frame_rep = np.repeat(meta["item_number"].values[kept_idx], n_channels)
                elec_rep = np.tile(np.array(channel_names), n_kept)
                n400_rep = n400_per_ch[kept_idx].reshape(-1).astype(np.float64)
                item_cond_id_rep = np.repeat(item_cond_id_kept, n_channels)

                surprisal_by_row = surprisal_items[item_cond_id_rep]           # (n_rows, n_layers)
                prev_surprisal_by_row = prev_surprisal_items[item_cond_id_rep]  # (n_rows, n_layers)

                base_df = pd.DataFrame({
                    "subject": subj_rep,
                    "electrode": elec_rep,
                    "frame_id": frame_rep,
                    "lab": lab_rep,
                    "wlen": wlen_freq[item_cond_id_rep, 0],
                    "freq": wlen_freq[item_cond_id_rep, 1],
                    "n400": n400_rep,
                })
        if "electrode" in base_df.columns:
            n_groups = base_df.groupby(["subject", "electrode"]).ngroups
            logger.info("Pooled table: %d rows (%d subject×electrode groups), %d layers",
                        len(base_df), n_groups, n_layers)
        else:
            logger.info("Pooled table: %d rows (%d subjects, paper formula: no electrode term), %d layers",
                        len(base_df), base_df["subject"].nunique(), n_layers)

        safe_revision = revision.replace("/", "_")
        base_parquet_path = out_dir / f"nieuwland_lme_pooled_table_{safe_revision}.parquet"
        surprisal_npy_path = out_dir / f"nieuwland_lme_pooled_surprisal_{safe_revision}.npy"
        prev_surprisal_npy_path = out_dir / f"nieuwland_lme_pooled_prev_surprisal_{safe_revision}.npy"
        base_df.to_parquet(base_parquet_path, index=False)
        np.save(surprisal_npy_path, surprisal_by_row.astype(np.float32))
        np.save(prev_surprisal_npy_path, prev_surprisal_by_row.astype(np.float32))
        del base_df, surprisal_by_row, prev_surprisal_by_row

        if self.cfg.lme.last_layer_only:
            layer_indices = [n_layers - 1]
        else:
            layer_indices = list(range(0, n_layers, max(1, self.cfg.lme.layer_stride)))
        logger.info(
            "Fitting %d layers (stride=%d) with %d worker processes",
            len(layer_indices), self.cfg.lme.layer_stride, self.cfg.lme.n_workers,
        )

        out_csv = out_dir / "nieuwland_lme_by_layer.csv"
        out_columns = list(_LAYER_ROW_COLUMNS) + [
            k for k in (extra_fields or {}) if k not in _LAYER_ROW_COLUMNS
        ]
        done_layers: set[int] = set()
        if out_csv.exists():
            try:
                existing = pd.read_csv(out_csv)
                existing_revision = existing["revision"] if "revision" in existing.columns else "main"
                same_revision = existing_revision == revision
                good = existing.loc[same_revision & existing["error"].isna()]
                done_layers = set(good["layer"].tolist())
                n_failed = int((same_revision & existing["error"].notna()).sum())
                if done_layers:
                    logger.info("Skipping %d already-fit layer(s) for revision=%s", len(done_layers), revision)
                if n_failed:
                    logger.warning(
                        "%d previously-crashed layer(s) found in %s for revision=%s — will retry",
                        n_failed, out_csv, revision,
                    )
                    keep = existing.loc[~(same_revision & existing["error"].notna())]
                    with csv_file_lock(out_csv):
                        keep.reindex(columns=out_columns).to_csv(out_csv, index=False)
            except Exception:
                logger.warning("Could not read existing layer CSV; re-fitting all layers.")

        per_word_dir = word_predictions_dir(out_dir, "nieuwland_lme_per_word")
        if self.cfg.lme.save_word_predictions:
            per_word_dir.mkdir(parents=True, exist_ok=True)

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        pending = [li for li in layer_indices if li not in done_layers]
        rows: list[dict] = []
        # max_tasks_per_child=1: recycle each worker after ONE layer fit —
        # lme4/rpy2's embedded R interpreter leaks memory across repeated
        # fits within a long-lived process (same mitigation as Podcast's
        # pooled LME).
        with ProcessPoolExecutor(
            max_workers=self.cfg.lme.n_workers, mp_context=ctx, max_tasks_per_child=1,
        ) as pool:
            futures = {
                pool.submit(
                    _fit_one_layer_worker,
                    str(base_parquet_path), str(surprisal_npy_path), str(prev_surprisal_npy_path), li,
                    include_lab=self.cfg.lme.include_lab,
                    train_frac=self.cfg.lme.train_frac,
                    random_state=self.cfg.lme.split_random_state,
                    verbose=self.cfg.lme.verbose,
                    allow_slope_fallback=self.cfg.lme.allow_slope_fallback,
                    save_word_predictions=self.cfg.lme.save_word_predictions,
                    per_word_dir=str(per_word_dir) if self.cfg.lme.save_word_predictions else None,
                    use_paper_formula=self.cfg.lme.use_paper_formula,
                ): li
                for li in pending
            }
            for fut in as_completed(futures):
                li = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    logger.error("Layer %d worker crashed: %s", li, exc, exc_info=True)
                    row = {"layer": li, "error": str(exc)}
                row["revision"] = revision
                if extra_fields:
                    row.update(extra_fields)
                rows.append(row)
                with csv_file_lock(out_csv):
                    pd.DataFrame([row]).reindex(columns=out_columns).to_csv(
                        out_csv, mode="a", header=not out_csv.exists(), index=False,
                    )
                logger.info(
                    "  [layer %d] done in %.1fs — delta_loglik_test=%s converged=%s%s",
                    li, row.get("fit_wall_time_s", float("nan")),
                    f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
                    row.get("converged"),
                    " (slope dropped)" if row.get("slope_dropped") else "",
                )

        return pd.DataFrame(rows)

    def _paper_trials(self, items: pd.DataFrame, keep_ids: set) -> tuple[pd.DataFrame, np.ndarray]:
        """Pooled paper-formula table from the paper's own per-trial N400.

        Trials are matched to ``items`` on (sentence-context prefix, target noun); each
        of the 160 (context, noun) keys must match exactly one item. ``n400`` is
        z-scored globally, like ``scale(n400)`` in the paper's R script (delta-loglik
        is invariant to this scale).
        """
        import re

        if not self.cfg.lme.use_paper_formula:
            raise ValueError("paper_n400_tsv requires lme.use_paper_formula=True")

        def key(text: str) -> str:
            return re.sub(r"[^a-z]", "", str(text).lower())

        paper = pd.read_csv(self.cfg.data.paper_n400_tsv, sep="\t").drop_duplicates()
        paper = paper.dropna(subset=["n400", "FullText", "TargetWords"])
        paper["k"] = paper["FullText"].map(key).str[:20] + "|" + paper["TargetWords"].map(key)
        it = items.copy()
        it["k"] = it["sentence_context"].map(key).str[:20] + "|" + it["noun"].map(key)
        if it["k"].duplicated().any():
            raise ValueError("Item (context, noun) keys are not unique")
        unmatched = set(paper["k"]) - set(it["k"])
        if unmatched:
            raise ValueError(f"{len(unmatched)} paper (context, noun) keys have no matching item")
        merged = paper.merge(
            it[["k", "item_number", "item_condition_id"]], on="k", how="inner",
        )
        merged = merged[merged["item_condition_id"].isin(keep_ids)]
        merged = merged.drop_duplicates(["subject", "item_condition_id"])
        n400 = merged["n400"].to_numpy(dtype=np.float64)
        base_df = pd.DataFrame({
            "subject": merged["subject"].astype(str).to_numpy(),
            "lab": merged["lab"].astype(str).str.lower().to_numpy(),
            "frame_id": merged["item_number"].to_numpy(),
            "item_id": merged["item_condition_id"].to_numpy(dtype=np.int64),
            "n400": (n400 - n400.mean()) / n400.std(),
        })
        logger.info("Paper N400 table: %d trials, %d subjects, %d labs",
                    len(base_df), base_df["subject"].nunique(), base_df["lab"].nunique())
        return base_df, base_df["item_id"].to_numpy(dtype=np.int64)

    # ------------------------------------------------------------------
    # Item loading (EEG-free)
    # ------------------------------------------------------------------

    def _load_items(self) -> pd.DataFrame:
        dcfg = self.cfg.data
        return load_all_items(
            dcfg.bids_root,
            included_in_paper_only=dcfg.included_in_paper_only,
            exclude_labs=dcfg.exclude_labs,
            subjects=dcfg.subjects,
        )

    # ------------------------------------------------------------------
    # Feature extraction (per-layer logit-lens surprisal, baseline covariates)
    # ------------------------------------------------------------------

    def _extract_item_surprisal(
        self, contexts: list[str], targets: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-layer surprisal for the noun AND the preceding word.

        Returns
        -------
        (surprisal, prev_surprisal), each ``(n_items, n_layers)``.
        """
        mcfg = self.cfg.model
        dcfg = self.cfg.data
        is_cbr_rnn = "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower()

        # Previous-word spillover control: target = last word of context,
        # context = everything before it.
        context_words = [c.split() for c in contexts]
        prev_targets = [w[-1] if w else "" for w in context_words]
        prev_contexts = [" ".join(w[:-1]) if w else "" for w in context_words]

        if is_cbr_rnn:
            from ..features.cbr_rnn import batch_surprisal_cbr_rnn

            surprisal = batch_surprisal_cbr_rnn(
                mcfg.model_name, contexts, targets,
                device=dcfg.device, n_params=mcfg.n_params,
            )
            prev_surprisal = batch_surprisal_cbr_rnn(
                mcfg.model_name, prev_contexts, prev_targets,
                device=dcfg.device, n_params=mcfg.n_params,
            )
            return surprisal, prev_surprisal

        from ..features.logit_lens import logit_lens_surprisal_trials

        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None
        kw = dict(
            revision=mcfg.model_revision,
            device=dcfg.device,
            tokenizer_no_revision=dcfg.tokenizer_no_revision,
            n_params=mcfg.n_params,
            trust_remote_code=dcfg.trust_remote_code,
            cache_dir=dcfg.feature_cache_dir,
            dtype=surp_dtype,
        )

        surprisal = logit_lens_surprisal_trials(
            contexts, targets, mcfg.model_name, cache_key="target", **kw
        )
        prev_surprisal = logit_lens_surprisal_trials(
            prev_contexts, prev_targets, mcfg.model_name, cache_key="prev", **kw
        )
        return surprisal, prev_surprisal

    def _load_baseline_covariates(self, targets: list[str]) -> np.ndarray:
        from ..features.word_frequency import compute_wlen_wordfreq_covariates

        return compute_wlen_wordfreq_covariates(
            targets, language=self.cfg.data.wordfreq_language,
        ).astype(np.float64)
