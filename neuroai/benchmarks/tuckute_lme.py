"""Tuckute et al. (2023) fMRI encoding benchmark — pooled LME per LM layer.

Unlike the per-subject/per-condition ridge checkpoint sweep
(``scripts/run/embedding/neural/tuckute.py``, which evaluates conditions B/D/S
separately and predicts BOLD from full hidden-state embeddings), this pools
subjects, ROIs, and sentences into ONE mixed-effects model per layer, using
per-layer logit-lens surprisal (Kuribayashi et al. 2025) as the predictor of
interest. See ``neuroai/regressors/tuckute_pooled_lme.py`` for the full
formula rationale — summary:

- Predictor = LAST-WORD logit-lens surprisal (not a sentence mean), following
  Lin & Schuler (2025, arXiv:2506.11338) / Hosseini et al. (2024)'s
  convention for this exact data shape (isolated sentences, one BOLD scalar
  per sentence): a single forward pass over the sentence already conditions
  the last word's surprisal on every earlier word.
- No ``prev_surprisal`` spillover term — no equivalent word-to-word
  contamination confound exists for a sentence-integrated BOLD signal.
- Nuisances are MEAN word length / MEAN wordfreq frequency across the
  sentence (Tuckute's sentences are all fixed at 6 words, so the paper's own
  ``sentlen`` nuisance has zero variance here and isn't usable).
- ``roi`` is a CROSSED random effect (fixed canonical ``LANG_ROIS`` labels,
  shared across subjects) — like a scalp electrode, unlike
  Fedorenko2016/Podcast's nested subject-specific electrode grids.
- ``sentence_id`` is a third crossed random effect (absent from Lin &
  Schuler's Pereira formula, which had already aggregated BOLD across
  regions into one scalar) — needed here because Tuckute keeps distinct
  per-ROI responses.

Only ONE condition (default ``"B"``, the 1,000-sentence naturalistic
baseline seen by all 10 subjects) is pooled per run, not all of B/D/S
together — mirroring the ridge checkpoint sweep's own per-condition
evaluation convention. D/S sentences were adversarially optimized to
drive/suppress the language network (not naturally varying in surprisal),
and are each seen by only 2-3 of the 10 subjects — pooling them with B would
both contaminate the surprisal-BOLD relationship with out-of-distribution
sentences and introduce a highly unbalanced subject panel. Set
``data.condition`` to sweep D or S separately if desired.

Pipeline
--------
1. Load the dataset (``neuroai.datasets.tuckute``) and extract per-layer
   logit-lens surprisal for every unique sentence via ONE forward pass PER
   SENTENCE (2,000 sentences total across all conditions, 6 words each,
   always exactly one chunk), keeping only the LAST word's value per layer.
   Mean word length / mean wordfreq frequency are computed the same way,
   once per sentence.
2. Load per-subject, per-ROI BOLD via ``TuckuteDataset.get_condition_per_subject``
   (raw, NOT participant-averaged — matches what the ridge checkpoint sweep
   already uses) and melt into long format (one row per subject × ROI ×
   sentence).
3. Broadcast the sentence-level surprisal/covariates onto every
   (subject, ROI) row for that sentence.
4. Write the pooled base table (parquet) and the (n_rows, n_layers)
   surprisal array (npy) to disk once, then fit one LME per layer via a
   process pool (pymer4/rpy2 embeds one R interpreter per process, so
   parallelism must be process-based).

Output: ``tuckute_lme_by_layer.csv`` — one row per layer, written
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

from ..datasets.tuckute import TuckuteDataset
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
class TuckuteLmeDataConfig:
    data_root: str | Path = "data/tuckute_fmri/osfstorage"
    # "B" (1,000 naturalistic sentences, all 10 subjects) is the default —
    # see module docstring for why D/S aren't pooled in by default.
    condition: str = "B"
    max_length: int | None = None
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    wordfreq_language: str = "en"
    # Device for the (cheap, <=2000-sentence) surprisal extraction forward passes.
    device: str = "cpu"
    # Surprisal cache dir. Same two-stage GPU/CPU split as every other pooled
    # LME in this codebase: R (pymer4/lme4) is CPU-only on Jean Zay, so a GPU
    # job populates this cache (see ``scripts/run/surprisal/neural/tuckute_extract.py``)
    # before the CPU-only LME job (``scripts/run/surprisal/neural/tuckute_fit.py``) reads
    # from it. A cache miss on the CPU-only job silently falls back to
    # loading the LLM on CPU (slow, not wrong) — same soft contract as
    # Podcast/Fedorenko2016.
    feature_cache_dir: str | None = None


@dataclass
class TuckuteLmeConfig:
    train_frac: float = 0.75
    split_random_state: int = 0
    verbose: bool = False
    # Barr et al. (2013)-style simplification ladder if the full model fails
    # to converge — see fit_tuckute_lme_one_layer's docstring for the
    # 4-step fallback order (sentence_id correlation → sentence_id slope →
    # subject correlation → all slopes dropped).
    allow_slope_fallback: bool = True
    # Which layers to fit: 1 = all layers; N>1 = every Nth layer.
    layer_stride: int = 1
    # If True, fit ONLY the final layer (index n_layers-1), ignoring layer_stride.
    last_layer_only: bool = False
    # Process-pool size for fitting layers concurrently — each worker is a
    # separate process with its own embedded R/pymer4 interpreter.
    n_workers: int = 10
    # Same require_cached_surprisal contract as every other pooled LME sweep
    # script (ns_fmri, naturalstories, provo, ...): when the LME stage runs
    # on a CPU-only node, a surprisal cache miss must raise instead of
    # silently falling back to a model load — for the FLA-based
    # trust_remote_code families (stickbreaking/alibi/forgetting/
    # hard-n-gram), that fallback load crashes on import (Triton needs a
    # live CUDA driver just to import, which a CPU node doesn't have).
    require_cached_surprisal: bool = False


@dataclass
class TuckuteLmeBenchmarkConfig:
    data: TuckuteLmeDataConfig
    model: ModelConfig
    lme: TuckuteLmeConfig = field(default_factory=TuckuteLmeConfig)
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
) -> dict[str, tp.Any]:
    """Runs in its own spawned process: load the shared table, fit one layer."""
    import numpy as _np
    import pandas as _pd

    from ..regressors.tuckute_pooled_lme import fit_tuckute_lme_one_layer

    t0 = time.time()
    row: dict[str, tp.Any] = {"layer": layer_idx}
    try:
        base_df = _pd.read_parquet(base_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        base_df["surprisal"] = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)

        fit = fit_tuckute_lme_one_layer(
            base_df,
            train_frac=train_frac,
            random_state=random_state,
            verbose=verbose,
            allow_slope_fallback=allow_slope_fallback,
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

class TuckuteLmeBenchmark:
    """Pooled Tuckute fMRI LME: one mixed-effects model per LM layer."""

    def __init__(self, cfg: TuckuteLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def extract_and_cache_surprisal(self) -> np.ndarray:
        """GPU-stage entry point: populate ``data.feature_cache_dir`` only.

        Loads the LLM, computes per-layer logit-lens surprisal (last word
        only) for every unique sentence in ``data.condition``, and returns —
        does not touch BOLD data, does not fit anything. Run this once per
        model (GPU job) before the CPU-only ``run()`` LME sweep, same
        two-stage split as every other pooled LME in this codebase (R/pymer4
        is CPU-only on Jean Zay).
        """
        ds = TuckuteDataset(self.cfg.data.data_root)
        sentences, _, item_ids = ds.get_condition(self.cfg.data.condition)
        surprisal, _, _ = self._extract_sentence_surprisal(sentences, item_ids)
        return surprisal

    def run(self, extra_fields: dict[str, tp.Any] | None = None) -> pd.DataFrame:
        dcfg = self.cfg.data
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        revision = self.cfg.model.model_revision or "main"

        ds = TuckuteDataset(dcfg.data_root)
        sentences, _, item_ids = ds.get_condition(dcfg.condition)

        surprisal_items, mean_wlen_items, mean_freq_items = self._extract_sentence_surprisal(
            sentences, item_ids,
        )

        item_finite = (
            np.isfinite(surprisal_items).all(axis=1)
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

        item_id_to_idx = {int(iid): i for i, iid in enumerate(item_ids.tolist())}

        per_subject = ds.get_condition_per_subject(dcfg.condition)
        n_rois = len(ds.rois)
        rows: list[pd.DataFrame] = []
        for subj, (_, bold, subj_item_ids) in per_subject.items():
            n = len(subj_item_ids)
            rows.append(pd.DataFrame({
                "subject": subj,
                "roi": np.tile(ds.rois, n),
                "item_id": np.repeat(subj_item_ids, n_rois),
                "y": bold.reshape(-1).astype(np.float64),
            }))
        long_df = pd.concat(rows, ignore_index=True)

        row_mask = long_df["item_id"].isin(keep_item_ids) & long_df["y"].notna()
        long_df = long_df.loc[row_mask].reset_index(drop=True)

        row_item_idx = long_df["item_id"].astype(int).map(item_id_to_idx).to_numpy(dtype=np.int64)
        n_layers = surprisal_items.shape[1]
        surprisal_by_row = surprisal_items[row_item_idx]  # (n_rows, n_layers)

        base_df = pd.DataFrame({
            "subject": long_df["subject"].astype(str).values,
            "roi": long_df["roi"].astype(str).values,
            "sentence_id": long_df["item_id"].astype(str).values,
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
        base_parquet_path = out_dir / f"tuckute_lme_pooled_table_{safe_revision}.parquet"
        surprisal_npy_path = out_dir / f"tuckute_lme_pooled_surprisal_{safe_revision}.npy"
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

        out_csv = out_dir / "tuckute_lme_by_layer.csv"
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

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        pending = [li for li in layer_indices if li not in done_layers]
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
                rows_out.append(row)
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

        return pd.DataFrame(rows_out)

    # ------------------------------------------------------------------
    # Feature extraction (per-layer last-word logit-lens surprisal, baseline covariates)
    # ------------------------------------------------------------------

    def _extract_sentence_surprisal(
        self, sentences: list[str], item_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-layer LAST-WORD surprisal, and mean wlen/freq, for every sentence.

        One forward pass per sentence (6 words each, always exactly one
        chunk) via :func:`logit_lens_surprisal_all_layers`, keeping only the
        final word's per-layer value — see module docstring for why
        (context is already fully absorbed into that word's surprisal).

        Returns
        -------
        (surprisal, mean_wlen, mean_freq): surprisal is ``(n_sentences,
        n_layers)``; mean_wlen/mean_freq are ``(n_sentences,)`` — all in the
        same order as ``sentences``/``item_ids``.
        """
        from ..features.word_frequency import compute_wlen_wordfreq_covariates

        mcfg = self.cfg.model
        dcfg = self.cfg.data
        is_cbr_rnn = "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower()

        n_sentences = len(sentences)
        mean_wlen = np.empty(n_sentences, dtype=np.float64)
        mean_freq = np.empty(n_sentences, dtype=np.float64)
        sentence_words = [sent.split() for sent in sentences]
        for i, words in enumerate(sentence_words):
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
            return surprisal, mean_wlen, mean_freq

        from ..features.logit_lens import logit_lens_surprisal_all_layers

        # See podcast_lme.py's `surp_dtype` — BabyLM-community's gpt-bert-*
        # models need float32 (their custom DWA accumulator isn't bf16-safe).
        # Passed as the STRING "float32", not torch.float32 — this custom
        # trust_remote_code config class doesn't stringify torch_dtype the
        # way vanilla HF configs do, so a real torch.dtype object survives
        # into its to_dict() unconverted and crashes json.dumps() the moment
        # AutoConfig.from_pretrained logs the resolved config (an f-string,
        # `logger.info(f"Model config {config}")` — evaluated eagerly,
        # independent of log level). from_pretrained accepts a string dtype
        # identically to a torch.dtype object for actual weight loading, so
        # this changes nothing about how the model loads.
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
            # False by default (the GPU extraction stage, extract_and_cache_
            # surprisal, needs to fall through to a model load to populate
            # the cache in the first place) — the CPU-only LME sweep script
            # sets lme.require_cached_surprisal=True so a cache miss here
            # raises a clear error instead of crashing inside Triton's
            # import machinery. See TuckuteLmeConfig.require_cached_surprisal.
            require_cache_hit=self.cfg.lme.require_cached_surprisal,
        )

        n_layers: int | None = None
        surprisal_rows: list[np.ndarray] = []

        for words, item_id in zip(sentence_words, item_ids.tolist()):
            arr = logit_lens_surprisal_all_layers(
                words, mcfg.model_name, section_id=int(item_id), **kw,
            )  # (n_words, n_layers, 1)
            surp = arr[:, :, 0]
            if n_layers is None:
                n_layers = surp.shape[1]
            surprisal_rows.append(surp[-1])  # LAST word only

        assert n_layers is not None
        surprisal = np.stack(surprisal_rows, axis=0).astype(np.float32)  # (n_sentences, n_layers)
        return surprisal, mean_wlen, mean_freq
