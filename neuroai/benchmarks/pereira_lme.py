"""Pereira et al. (2018) fMRI encoding benchmark — pooled LME per LM layer.

Same overall shape as ``neuroai/benchmarks/tuckute_lme.py`` (isolated
sentences, one BOLD scalar per (subject, roi, sentence), no HRF) — see
``neuroai/regressors/pereira_pooled_lme.py`` for the full formula rationale.
Pereira is in fact the EXACT dataset Lin & Schuler (2506.11338v2) fit their
per-sentence fMRI LME to, so this benchmark follows their design directly
rather than Tuckute's adaptation of it (real ``sentlen`` nuisance, since
Pereira sentences vary in length unlike Tuckute's fixed 6 words).

Pipeline (per model checkpoint)
--------------------------------
1. Load the dataset (``neuroai.datasets.pereira``) — 627 unique sentences
   pooled across the 243sentences/384sentences sub-experiments — and extract
   per-layer logit-lens surprisal via ONE forward pass PER SENTENCE, keeping
   only the LAST word's value per layer (Lin & Schuler's own convention for
   this exact data: "its preceding context has already been taken into
   consideration... the last words of the sentences"). Mean word length /
   mean wordfreq frequency / sentence length are computed the same way, once
   per sentence.
2. Load per-subject, per-ROI-averaged BOLD via
   ``PereiraDataset.get_condition_per_subject`` (language-atlas ROIs only by
   default — 12 canonical labels present in all 10 subjects; see that
   module's docstring for why voxels are averaged within each (subject, roi)
   group first) and melt into long format (one row per subject × ROI ×
   sentence).
3. Broadcast the sentence-level surprisal/covariates onto every
   (subject, roi) row for that sentence.
4. Write the pooled base table (parquet) and the (n_rows, n_layers)
   surprisal array (npy) to disk once, then fit one LME per layer via a
   process pool (pymer4/rpy2 embeds one R interpreter per process, so
   parallelism must be process-based).

Output: ``pereira_lme_by_layer.csv`` — one row per layer, written
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

from ..datasets.pereira import DEFAULT_ATLAS, PereiraDataset
from ..sweep import csv_file_lock
from .base import ModelConfig

logger = logging.getLogger(__name__)

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
class PereiraLmeDataConfig:
    nc_path: str | Path = "data/pereira_fmri/Pereira2018_merged.nc"
    # "language" (12 ROIs, all 10 subjects) is the default — see
    # neuroai/datasets/pereira.py module docstring. Set to "auditory"/"MD"/
    # "DMN"/"visual" for the other 4 atlases if desired.
    atlas: str = DEFAULT_ATLAS
    max_length: int | None = None
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    wordfreq_language: str = "en"
    # Device for the (cheap, 627-sentence) surprisal extraction forward passes.
    device: str = "cpu"
    # Surprisal cache dir — same two-stage GPU/CPU split as every other
    # pooled LME in this codebase: R (pymer4/lme4) is CPU-only on Jean Zay,
    # so a GPU job populates this cache (see
    # scripts/run/surprisal/neural/pereira_extract.py) before the CPU-only LME job
    # (scripts/run/surprisal/neural/pereira_fit.py) reads from it.
    feature_cache_dir: str | None = None


@dataclass
class PereiraLmeConfig:
    train_frac: float = 0.75
    split_random_state: int = 0
    verbose: bool = False
    # Barr et al. (2013)-style simplification ladder if the full model fails
    # to converge — see fit_pereira_lme_one_layer's docstring for the 4-step
    # fallback order.
    allow_slope_fallback: bool = True
    # Which layers to fit: 1 = all layers; N>1 = every Nth layer.
    layer_stride: int = 1
    # If True, fit ONLY the final layer (index n_layers-1), ignoring layer_stride.
    last_layer_only: bool = False
    # Process-pool size for fitting layers concurrently — each worker is a
    # separate process with its own embedded R/pymer4 interpreter.
    n_workers: int = 10
    # Same require_cached_surprisal contract as every other pooled LME sweep
    # script — a surprisal-cache miss on a CPU-only node must raise instead
    # of silently falling back to a model load (crashes on import for the
    # FLA-based trust_remote_code families, which need a live CUDA driver).
    require_cached_surprisal: bool = False
    # Separate last-layer experiment: fit each language-atlas ROI independently.
    roi_wise: bool = False
    # With roi_wise: restrict to these ROI labels (None = all ROIs of the atlas).
    rois: tuple[str, ...] | None = None


@dataclass
class PereiraLmeBenchmarkConfig:
    data: PereiraLmeDataConfig
    model: ModelConfig
    lme: PereiraLmeConfig = field(default_factory=PereiraLmeConfig)
    output_dir: str = "outputs"
    canonical_model_name: str | None = None


# ---------------------------------------------------------------------------
# Worker (module-level for spawn-based multiprocessing pickling)
# ---------------------------------------------------------------------------

def _fit_one_layer_worker(
    base_parquet_path: str,
    surprisal_npy_path: str,
    layer_idx: int,
    *,
    train_frac: float,
    random_state: int,
    verbose: bool,
    allow_slope_fallback: bool,
    roi: str | None = None,
) -> dict[str, tp.Any]:
    """Runs in its own spawned process: load the shared table, fit one layer."""
    import numpy as _np
    import pandas as _pd

    from ..regressors.pereira_pooled_lme import fit_pereira_lme_one_layer

    t0 = time.time()
    row: dict[str, tp.Any] = {"layer": layer_idx}
    if roi is not None:
        row["roi"] = roi
    try:
        base_df = _pd.read_parquet(base_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        base_df["surprisal"] = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)
        if roi is not None:
            base_df = base_df.loc[base_df["roi"].astype(str) == roi].copy()
            if base_df.empty:
                raise ValueError(f"No observations for ROI {roi!r}")

        fit = fit_pereira_lme_one_layer(
            base_df,
            train_frac=train_frac,
            random_state=random_state,
            verbose=verbose,
            allow_slope_fallback=allow_slope_fallback,
            per_roi=roi is not None,
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
        logger.error("ROI %s layer %d failed", roi, layer_idx, exc_info=True)
        row["error"] = str(exc)
    row["fit_wall_time_s"] = time.time() - t0
    return row


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class PereiraLmeBenchmark:
    """Pooled Pereira fMRI LME: one mixed-effects model per LM layer."""

    def __init__(self, cfg: PereiraLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def extract_and_cache_surprisal(self) -> np.ndarray:
        """GPU-stage entry point: populate ``data.feature_cache_dir`` only.

        Loads the LLM, computes per-layer logit-lens surprisal (last word
        only) for every one of the 627 sentences, and returns — does not
        touch BOLD data, does not fit anything. Run this once per model
        (GPU job) before the CPU-only ``run()`` LME sweep, same two-stage
        split as every other pooled LME in this codebase (R/pymer4 is
        CPU-only on Jean Zay).
        """
        ds = PereiraDataset(self.cfg.data.nc_path)
        sentences, item_ids = ds.get_sentences()
        surprisal, _, _, _ = self._extract_sentence_surprisal(sentences, item_ids)
        return surprisal

    def run(self, extra_fields: dict[str, tp.Any] | None = None) -> pd.DataFrame:
        dcfg = self.cfg.data
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        revision = self.cfg.model.model_revision or "main"

        ds = PereiraDataset(dcfg.nc_path)
        sentences, item_ids = ds.get_sentences()

        surprisal_items, sentlen_items, mean_wlen_items, mean_freq_items = (
            self._extract_sentence_surprisal(sentences, item_ids)
        )

        item_finite = (
            np.isfinite(surprisal_items).all(axis=1)
            & np.isfinite(sentlen_items)
            & np.isfinite(mean_wlen_items)
            & np.isfinite(mean_freq_items)
        )
        if not item_finite.all():
            n_dropped = int((~item_finite).sum())
            logger.warning(
                "Dropping %d/%d sentence(s) with non-finite features before pooling",
                n_dropped, len(item_ids),
            )
        keep_item_ids = set(item_ids[item_finite].tolist())
        item_id_to_idx = {iid: i for i, iid in enumerate(item_ids.tolist())}

        per_subject = ds.get_condition_per_subject(dcfg.atlas)
        roi_names = ds.rois(dcfg.atlas)
        n_rois = len(roi_names)
        rows: list[pd.DataFrame] = []
        for subj, (bold, subj_item_ids) in per_subject.items():
            n = len(subj_item_ids)
            rows.append(pd.DataFrame({
                "subject": subj,
                "roi": np.tile(roi_names, n),
                "item_id": np.repeat(subj_item_ids, n_rois),
                "y": bold.reshape(-1).astype(np.float64),
            }))
        long_df = pd.concat(rows, ignore_index=True)

        row_mask = long_df["item_id"].isin(keep_item_ids) & long_df["y"].notna()
        long_df = long_df.loc[row_mask].reset_index(drop=True)

        row_item_idx = long_df["item_id"].map(item_id_to_idx).to_numpy(dtype=np.int64)
        n_layers = surprisal_items.shape[1]
        surprisal_by_row = surprisal_items[row_item_idx]  # (n_rows, n_layers)

        base_df = pd.DataFrame({
            "subject": long_df["subject"].astype(str).values,
            "roi": long_df["roi"].astype(str).values,
            "sentence_id": long_df["item_id"].astype(str).values,
            "sentlen": sentlen_items[row_item_idx],
            "mean_wlen": mean_wlen_items[row_item_idx],
            "mean_freq": mean_freq_items[row_item_idx],
            "y": long_df["y"].astype(float).values,
        })
        logger.info(
            "Pooled table: %d rows (%d subject×ROI groups, %d unique sentences), %d layers",
            len(base_df), base_df.groupby(["subject", "roi"]).ngroups,
            base_df["sentence_id"].nunique(), n_layers,
        )

        safe_revision = revision.replace("/", "_")
        base_parquet_path = out_dir / f"pereira_lme_pooled_table_{safe_revision}.parquet"
        surprisal_npy_path = out_dir / f"pereira_lme_pooled_surprisal_{safe_revision}.npy"
        base_df.to_parquet(base_parquet_path, index=False)
        np.save(surprisal_npy_path, surprisal_by_row.astype(np.float32))
        del base_df, surprisal_by_row

        if self.cfg.lme.last_layer_only:
            layer_indices = [n_layers - 1]
        else:
            layer_indices = list(range(0, n_layers, max(1, self.cfg.lme.layer_stride)))
        logger.info(
            "Fitting %d layers (stride=%d) with %d worker processes",
            len(layer_indices), self.cfg.lme.layer_stride, self.cfg.lme.n_workers,
        )

        out_csv = out_dir / ("pereira_lme_by_roi.csv" if self.cfg.lme.roi_wise else "pereira_lme_by_layer.csv")
        out_columns = (["roi"] if self.cfg.lme.roi_wise else []) + list(_LAYER_ROW_COLUMNS) + [
            k for k in (extra_fields or {}) if k not in _LAYER_ROW_COLUMNS
        ]
        done_layers: set[int] = set()
        if out_csv.exists():
            try:
                existing = pd.read_csv(out_csv)
                existing_revision = existing["revision"] if "revision" in existing.columns else "main"
                same_revision = existing_revision == revision
                good = existing.loc[same_revision & existing["error"].isna()]
                if self.cfg.lme.roi_wise:
                    done_layers = set(zip(good["roi"].astype(str), good["layer"].astype(int)))
                else:
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

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        fit_rois = ([r for r in roi_names if self.cfg.lme.rois is None or r in self.cfg.lme.rois]
                    if self.cfg.lme.roi_wise else [None])
        tasks = [(roi, li) for roi in fit_rois for li in layer_indices
                 if ((roi, li) not in done_layers if self.cfg.lme.roi_wise else li not in done_layers)]
        rows_out: list[dict] = []
        # max_tasks_per_child=1: recycle each worker after ONE layer fit —
        # lme4/rpy2's embedded R interpreter leaks memory across repeated
        # fits within a long-lived process (same mitigation as every other
        # pooled LME in this codebase).
        with ProcessPoolExecutor(
            max_workers=self.cfg.lme.n_workers, mp_context=ctx, max_tasks_per_child=1,
        ) as pool:
            futures = {
                pool.submit(
                    _fit_one_layer_worker,
                    str(base_parquet_path), str(surprisal_npy_path), li,
                    train_frac=self.cfg.lme.train_frac,
                    random_state=self.cfg.lme.split_random_state,
                    verbose=self.cfg.lme.verbose,
                    allow_slope_fallback=self.cfg.lme.allow_slope_fallback,
                    roi=roi,
                ): (roi, li)
                for roi, li in tasks
            }
            for fut in as_completed(futures):
                roi, li = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    logger.error("Layer %d worker crashed: %s", li, exc, exc_info=True)
                    row = {"layer": li, "error": str(exc)}
                if roi is not None:
                    row["roi"] = roi
                row["revision"] = revision
                if extra_fields:
                    row.update(extra_fields)
                rows_out.append(row)
                with csv_file_lock(out_csv):
                    pd.DataFrame([row]).reindex(columns=out_columns).to_csv(
                        out_csv, mode="a", header=not out_csv.exists(), index=False,
                    )
                logger.info(
                    "  [%s | layer %d] done in %.1fs — delta_loglik_test=%s converged=%s%s",
                    roi or "pooled", li, row.get("fit_wall_time_s", float("nan")),
                    f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
                    row.get("converged"),
                    " (slope dropped)" if row.get("slope_dropped") else "",
                )

        return pd.DataFrame(rows_out)

    # ------------------------------------------------------------------
    # Feature extraction (per-layer last-word logit-lens surprisal, baseline covariates)
    # ------------------------------------------------------------------

    def _extract_sentence_surprisal(
        self, sentences: list[str], item_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-layer LAST-WORD surprisal, and sentlen/mean-wlen/mean-freq, for
        every sentence.

        One forward pass per sentence via :func:`logit_lens_surprisal_all_layers`,
        keeping only the final word's per-layer value — see module docstring
        for why (context is already fully absorbed into that word's
        surprisal; this is Lin & Schuler's own convention for this exact
        dataset, not an adaptation).

        Returns
        -------
        (surprisal, sentlen, mean_wlen, mean_freq): surprisal is
        ``(n_sentences, n_layers)``; the other three are ``(n_sentences,)``
        — all in the same order as ``sentences``/``item_ids``.
        """
        from ..features.word_frequency import compute_wlen_wordfreq_covariates

        mcfg = self.cfg.model
        dcfg = self.cfg.data
        is_cbr_rnn = "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower()

        n_sentences = len(sentences)
        sentlen = np.empty(n_sentences, dtype=np.float64)
        mean_wlen = np.empty(n_sentences, dtype=np.float64)
        mean_freq = np.empty(n_sentences, dtype=np.float64)
        sentence_words = [sent.split() for sent in sentences]
        for i, words in enumerate(sentence_words):
            sentlen[i] = float(len(words))
            wlen_freq = compute_wlen_wordfreq_covariates(words, language=dcfg.wordfreq_language)
            mean_wlen[i] = float(np.mean(wlen_freq[:, 0]))
            mean_freq[i] = float(np.mean(wlen_freq[:, 1]))

        if is_cbr_rnn:
            from ..features.cbr_rnn import batch_surprisal_cbr_rnn

            contexts = [" ".join(words[:-1]) for words in sentence_words]
            targets = [words[-1] for words in sentence_words]
            surprisal = batch_surprisal_cbr_rnn(
                mcfg.model_name, contexts, targets,
                device=dcfg.device, n_params=mcfg.n_params,
            ).astype(np.float32)  # (n_sentences, 1)
            return surprisal, sentlen, mean_wlen, mean_freq

        from ..features.logit_lens import logit_lens_surprisal_all_layers

        # See podcast_lme.py's `surp_dtype` — BabyLM-community's gpt-bert-*
        # models need float32 (their custom DWA accumulator isn't bf16-safe).
        surp_dtype: str | None = None
        if "gpt-bert" in mcfg.model_name.lower():
            surp_dtype = "float32"

        kw = dict(
            revision=mcfg.model_revision,
            device=dcfg.device,
            max_length=dcfg.max_length,
            tokenizer_no_revision=dcfg.tokenizer_no_revision,
            n_params=mcfg.n_params,
            trust_remote_code=dcfg.trust_remote_code,
            cache_dir=dcfg.feature_cache_dir,
            dtype=surp_dtype,
            # False by default (the GPU extraction stage needs to fall
            # through to a model load to populate the cache in the first
            # place) — the CPU-only LME sweep script sets
            # lme.require_cached_surprisal=True so a cache miss here raises
            # a clear error instead of crashing inside Triton's import
            # machinery. See PereiraLmeConfig.require_cached_surprisal.
            require_cache_hit=self.cfg.lme.require_cached_surprisal,
        )

        n_layers: int | None = None
        surprisal_rows: list[np.ndarray] = []

        # section_id must be an int (cache key) — item_ids here are string
        # stimulus_ids (e.g. "384sentences.0"), so use the stable positional
        # index in `sentences` (get_sentences() always returns a fixed,
        # item_id-sorted order) instead.
        for i, words in enumerate(sentence_words):
            arr = logit_lens_surprisal_all_layers(
                words, mcfg.model_name, section_id=i, **kw,
            )  # (n_words, n_layers, 1)
            surp = arr[:, :, 0]
            if n_layers is None:
                n_layers = surp.shape[1]
            surprisal_rows.append(surp[-1])  # LAST word only

        assert n_layers is not None
        surprisal = np.stack(surprisal_rows, axis=0).astype(np.float32)  # (n_sentences, n_layers)
        return surprisal, sentlen, mean_wlen, mean_freq
