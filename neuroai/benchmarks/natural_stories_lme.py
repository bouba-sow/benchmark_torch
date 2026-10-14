"""Natural Stories SPR benchmark — trial-level LME (pymer4/lme4), per-layer
logit-lens surprisal sweep.

Same per-layer logit-lens design as the neural/EEG LME datasets
(``n400_nieuwland_lme.py``, ``natural_stories_fmri_lme.py``): word-level
surprisal is extracted ONCE per (model, revision) as ``(n_words, n_layers)``
via ``logit_lens_surprisal_trials`` (Kuribayashi et al. 2025 — reuses the
model's own final norm + unembedding to read out a next-token distribution
from every transformer layer), then one LME is fit per layer via a process
pool (pymer4/rpy2 embeds one R interpreter per process, so parallelism must
be process-based). Previously this benchmark used only the model's final
layer (``batch_surprisal_one_model``), fit once per checkpoint.

Output: ``spr_lme_checkpoint_scores.csv`` — one row per (layer, revision),
written incrementally as each layer's fit completes (crash-safe restart).
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

from ..datasets.natural_stories import NaturalStoriesDataset
from ..features.llm import clear_hf_model_cache
from ..regressors.lme import LmeStyle
from ..sweep import csv_file_lock
from .base import ModelConfig


def _pw_dir(out_dir):
    """Per-word prediction directory: mirrors ``out_dir`` under $SCRATCH (word_predictions_root)
    so large per-word CSVs never land on $WORK (2026-10-12)."""
    from .base import word_predictions_root

    d = word_predictions_root(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d
from .natural_stories import NaturalStoriesDataConfig

logger = logging.getLogger(__name__)

# Fixed column order for spr_lme_checkpoint_scores.csv — a crash-fallback
# row (only a few keys) is reindexed to this before writing, so it can never
# misalign the CSV's columns.
_LAYER_ROW_COLUMNS = (
    "layer", "beta", "se", "t_stat", "p_value", "prev_surprisal_beta",
    "loglik", "aic", "bic", "baseline_loglik", "delta_loglik",
    "delta_loglik_train", "delta_loglik_test", "baseline_loglik_test",
    "loglik_test", "lrt_p_value", "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr", "converged",
    "n_obs", "n_obs_test", "error", "fit_wall_time_s", "revision",
)


@dataclass
class NaturalStoriesLmeConfig:
    """LME options (default: Clark et al. 2025 COLING; ``modelblocks`` for legacy)."""

    style: LmeStyle = "clark2025"
    include_surprisal: bool = True
    include_prev_surprisal: bool = True
    # Zipf lexical frequency nuisance (wordfreq), same convention as the
    # neural LME datasets — replaces the earlier KenLM-based nuisance.
    include_freq: bool = True
    filter_sentence_edges: bool = True
    main_random_slope: bool = False
    compare_baseline: bool = True
    out_of_sample: bool = True
    partition_mod: int = 4
    partition_fit_indices: tuple[int, ...] = (0, 1)
    partition_test_indices: tuple[int, ...] = (2, 3)
    # Legacy ModelBlocks alias
    partition_dev_indices: tuple[int, ...] | None = None
    verbose: bool = False
    # CPU sweep runs are the Stage-2 LME-only half of the two-stage Jean Zay
    # pipeline — they must never need to load a model. A cache miss there
    # should fail fast with an actionable message rather than falling through
    # to a full on-CPU model load, which either crashes outright (FLA/Triton
    # custom kernels require a GPU: alibi/forgetting/stickbreaking) or runs
    # for hours until the job's walltime kills it (babylm).
    require_cached_surprisal: bool = False
    # Which layers to fit: 1 = all layers; N>1 = every Nth layer. Ignored
    # when last_layer_only=True.
    layer_stride: int = 1
    # Fit only the deepest layer (n_layers - 1), skipping layer_stride
    # entirely. Cuts the CPU LME-fitting stage from n_layers separate R/lme4
    # fits down to 1 -- the dominant cost for big models (e.g. Pythia-12b's
    # 37 layers) is this fitting stage, not the GPU surprisal extraction
    # (which still runs for every layer regardless, since it's one forward
    # pass either way). Does not affect surprisal_layers extraction/caching.
    last_layer_only: bool = False
    # Process-pool size for fitting layers concurrently. Each worker is a
    # separate process (spawn) with its own embedded R/pymer4 interpreter —
    # rpy2 is not thread-safe, so this must be process-based.
    n_workers: int = 10
    # Save each held-out word's true value + baseline/full model predictions
    # (yhat_baseline, yhat_full) and per-word delta log-likelihood to
    # spr_lme_per_word.csv, one row per (layer, word) — same DLT-difficulty-
    # analysis convention as the neural LME datasets' SAVE_WORD_PREDICTIONS.
    # On by default.
    save_word_predictions: bool = True


@dataclass
class NaturalStoriesLmeBenchmarkConfig:
    data: NaturalStoriesDataConfig
    model: ModelConfig
    lme: NaturalStoriesLmeConfig = field(default_factory=NaturalStoriesLmeConfig)
    output_dir: str = "outputs"
    # Grouping label for saved CSVs when HF load repo differs (e.g. OLMo early-training).
    canonical_model_name: str | None = None


# ---------------------------------------------------------------------------
# Worker (module-level for spawn-based multiprocessing pickling)
# ---------------------------------------------------------------------------

def _fit_one_layer_worker(
    trials_base_parquet_path: str,
    word_base_parquet_path: str,
    surprisal_npy_path: str,
    layer_idx: int,
    *,
    style: str,
    log_rt: bool,
    main_random_slope: bool,
    compare_baseline: bool,
    out_of_sample: bool,
    partition_mod: int,
    partition_fit_indices: tuple[int, ...],
    partition_test_indices: tuple[int, ...],
    include_freq: bool,
    include_prev_surprisal: bool,
    filter_sentence_edges: bool,
    verbose: bool,
    save_word_predictions: bool = True,
) -> dict[str, tp.Any]:
    """Runs in its own spawned process: build this layer's trial frame and fit.

    Mirrors ``n400_nieuwland_lme.py::_fit_one_layer_worker`` — loads the
    shared word/trial tables + this layer's surprisal column, merges (same
    join ``natural_stories_lme.py`` always did: only ``surprisal``/
    ``prev_surprisal`` come from the word table, everything else — including
    the Zipf frequency nuisance, computed fresh from the ``word`` column
    inside ``fit_spr_lme`` — from the trials table as loaded), and fits.
    """
    import numpy as _np
    import pandas as _pd

    from ..regressors.lme import fit_spr_lme

    t0 = time.time()
    row: dict[str, tp.Any] = {"layer": layer_idx}
    per_word_out: pd.DataFrame | None = None
    try:
        word_base = _pd.read_parquet(word_base_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        word_layer = word_base[["story_id", "zone"]].copy()
        word_layer["surprisal"] = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)
        if include_prev_surprisal and style == "clark2025":
            word_layer = word_layer.sort_values(["story_id", "zone"])
            word_layer["prev_surprisal"] = word_layer.groupby("story_id", sort=False)[
                "surprisal"
            ].shift(1)

        trials_base = _pd.read_parquet(trials_base_parquet_path)
        pred_cols = [c for c in word_layer.columns if c in ("surprisal", "prev_surprisal")]
        merge_cols = ["story_id", "zone"] + pred_cols
        trials = trials_base.merge(word_layer[merge_cols], on=["story_id", "zone"], how="inner")

        sub = trials.dropna(subset=["surprisal", "rt"]).copy()
        fit = fit_spr_lme(
            sub,
            predictor_col="surprisal",
            style=style,
            log_rt=log_rt,
            main_random_slope=main_random_slope,
            compare_baseline=compare_baseline,
            out_of_sample=out_of_sample,
            partition_mod=partition_mod,
            partition_fit_indices=partition_fit_indices,
            partition_test_indices=partition_test_indices,
            include_freq=include_freq,
            include_prev_surprisal=include_prev_surprisal,
            filter_sentence_edges=filter_sentence_edges,
            verbose=verbose,
            layer=layer_idx,
            save_per_word=save_word_predictions,
        )
        row.update({
            "beta": fit.beta,
            "se": fit.se,
            "t_stat": fit.t_stat,
            "p_value": fit.p_value,
            "prev_surprisal_beta": fit.prev_surprisal_beta,
            "loglik": fit.loglik,
            "aic": fit.aic,
            "bic": fit.bic,
            "baseline_loglik": fit.baseline_loglik,
            "delta_loglik": fit.delta_loglik,
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
            "n_obs": fit.n_obs,
            "n_obs_test": fit.n_obs_test,
            "error": None,
        })
        if fit.per_word is not None and not fit.per_word.empty:
            per_word_out = fit.per_word.copy()
            per_word_out.insert(0, "layer", layer_idx)
    except Exception as exc:  # noqa: BLE001 — one bad layer must not kill the pool
        logger.error("Layer %d failed", layer_idx, exc_info=True)
        row["error"] = str(exc)
    row["fit_wall_time_s"] = time.time() - t0
    return {"row": row, "per_word": per_word_out}


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class NaturalStoriesLmeBenchmark:
    """Trial-level SPR LME (Clark 2025 or ModelBlocks-style), one fit per LM layer.

    Pipeline
    --------
    1. Load trial-level RTs (``aggregate=False``) and the word-level table
       (``aggregate=True``, cumulative context per word).
    2. Extract per-layer logit-lens surprisal for every word (one model
       load), disk-cached.
    3. Write the word/trial base tables (parquet) and the ``(n_words,
       n_layers)`` surprisal array (npy) once, then fit one ``lmer`` per
       layer via a process pool.
    4. Append each layer's result to CSV the moment it completes.
    """

    def __init__(self, cfg: NaturalStoriesLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def extract_and_cache_surprisal(self) -> np.ndarray:
        """GPU-stage entry point: populate ``data.feature_cache_dir`` only.

        Loads the LLM, computes per-layer logit-lens surprisal for every
        word, and returns — does not fit anything. Run this once per model
        (GPU job) before the CPU-only ``run()`` LME sweep, same two-stage
        split as Podcast's pooled LME (R/pymer4 is CPU-only on
        Jean Zay, so extraction and fitting can't share a job).
        """
        word_df, _ = self._load_data()
        return self._word_surprisal_all_layers(word_df)

    def run(self, extra_fields: dict[str, tp.Any] | None = None) -> pd.DataFrame:
        """Fit every layer for this (model, revision).

        ``extra_fields`` (e.g. ``{"step": ckpt.step, "tokens_seen": ...}``)
        is merged into every row before it's written — lets a checkpoint
        sweep driver attach its own bookkeeping columns without this class
        needing to know about checkpoints at all.
        """
        dcfg = self.cfg.data
        lme_cfg = self.cfg.lme
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        revision = self.cfg.model.model_revision or "main"
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name

        word_df, trials = self._load_data()

        if not lme_cfg.include_surprisal:
            logger.info("[%s] include_surprisal=False — nothing to fit", model_name)
            return pd.DataFrame()

        log_rt = self.cfg.data.log_rt and lme_cfg.style != "clark2025"

        surprisal_layers = self._word_surprisal_all_layers(word_df)  # (n_words, n_layers)
        n_layers = surprisal_layers.shape[1]
        clear_hf_model_cache()

        # Revision MUST be in these filenames: out_dir is per-MODEL, not
        # per-checkpoint, so a training-step sweep (e.g. "pythia_checkpoints",
        # 11 revisions per model) calls run() once per revision against the
        # SAME out_dir. Without the revision in the filename, concurrent
        # checkpoints of the same model (true both for a SLURM array's %N
        # throttle -- consecutive array indices are the same model's
        # consecutive steps -- and for any other same-model parallelism)
        # race on these paths: one checkpoint's write can land, then a
        # sibling checkpoint's fit reads THAT data instead of its own,
        # silently corrupting results (verified: produced near-identical
        # delta_loglik across supposedly-different training steps in both a
        # SLURM array run and an xargs-parallel run of the sibling
        # ucl_reading_lme.py driver -- same bug pattern, fixed there too).
        safe_revision = revision.replace("/", "_")
        word_base_parquet_path = out_dir / f"ns_lme_word_base_{safe_revision}.parquet"
        trials_base_parquet_path = out_dir / f"ns_lme_trials_base_{safe_revision}.parquet"
        surprisal_npy_path = out_dir / f"ns_lme_surprisal_{safe_revision}.npy"
        word_df[["story_id", "zone"]].to_parquet(word_base_parquet_path, index=False)
        trials.to_parquet(trials_base_parquet_path, index=False)
        np.save(surprisal_npy_path, surprisal_layers.astype(np.float32))
        del surprisal_layers

        if lme_cfg.last_layer_only:
            layer_indices = [n_layers - 1]
        else:
            layer_indices = list(range(0, n_layers, max(1, lme_cfg.layer_stride)))
        logger.info(
            "[%s] Fitting %d layers (stride=%d, last_layer_only=%s) with %d worker processes",
            model_name, len(layer_indices), lme_cfg.layer_stride, lme_cfg.last_layer_only, lme_cfg.n_workers,
        )

        out_csv = out_dir / "spr_lme_checkpoint_scores.csv"
        per_word_csv = _pw_dir(out_dir) / "spr_lme_per_word.csv"

        done_layers: set[int] = set()
        if out_csv.exists():
            try:
                existing = pd.read_csv(out_csv)
                good = existing[(existing["revision"] == revision) & existing["error"].isna()]
                done_layers = set(good["layer"].tolist())
                if done_layers:
                    logger.info(
                        "Skipping %d already-fit layer(s) for revision=%s", len(done_layers), revision,
                    )
            except Exception:
                logger.warning("Could not read existing layer CSV; re-fitting all layers.")

        fit_indices = (
            lme_cfg.partition_dev_indices
            if lme_cfg.partition_dev_indices is not None
            else lme_cfg.partition_fit_indices
        )

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        pending = [li for li in layer_indices if li not in done_layers]
        rows: list[dict] = []
        # max_tasks_per_child=1: recycle each worker after ONE layer fit —
        # lme4/rpy2's embedded R interpreter leaks memory across repeated
        # fits within a long-lived process (same mitigation as every other
        # pooled LME in this codebase).
        with ProcessPoolExecutor(
            max_workers=lme_cfg.n_workers, mp_context=ctx, max_tasks_per_child=1,
        ) as pool:
            futures = {
                pool.submit(
                    _fit_one_layer_worker,
                    str(trials_base_parquet_path), str(word_base_parquet_path), str(surprisal_npy_path), li,
                    style=lme_cfg.style,
                    log_rt=log_rt,
                    main_random_slope=lme_cfg.main_random_slope,
                    compare_baseline=lme_cfg.compare_baseline,
                    out_of_sample=lme_cfg.out_of_sample,
                    partition_mod=lme_cfg.partition_mod,
                    partition_fit_indices=fit_indices,
                    partition_test_indices=lme_cfg.partition_test_indices,
                    include_freq=lme_cfg.include_freq,
                    include_prev_surprisal=lme_cfg.include_prev_surprisal,
                    filter_sentence_edges=lme_cfg.filter_sentence_edges,
                    verbose=lme_cfg.verbose,
                    save_word_predictions=lme_cfg.save_word_predictions,
                ): li
                for li in pending
            }
            for fut in as_completed(futures):
                li = futures[fut]
                try:
                    result = fut.result()
                    row, per_word = result["row"], result["per_word"]
                except Exception as exc:
                    logger.error("Layer %d worker crashed: %s", li, exc, exc_info=True)
                    row, per_word = {"layer": li, "error": str(exc)}, None
                row["revision"] = revision
                if extra_fields:
                    row.update(extra_fields)
                rows.append(row)
                out_columns = list(_LAYER_ROW_COLUMNS) + [
                    k for k in (extra_fields or {}) if k not in _LAYER_ROW_COLUMNS
                ]
                with csv_file_lock(out_csv):
                    pd.DataFrame([row]).reindex(columns=out_columns).to_csv(
                        out_csv, mode="a", header=not out_csv.exists(), index=False,
                    )
                if per_word is not None:
                    per_word = per_word.copy()
                    per_word.insert(0, "revision", revision)
                    per_word.insert(0, "model", model_name)
                    per_word.insert(0, "predictor", "surprisal")
                    with csv_file_lock(per_word_csv):
                        per_word.to_csv(
                            per_word_csv, mode="a", header=not per_word_csv.exists(), index=False,
                        )
                logger.info(
                    "  [layer %d] done in %.1fs — delta_loglik_test=%s converged=%s",
                    li, row.get("fit_wall_time_s", float("nan")),
                    f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
                    row.get("converged"),
                )

        return pd.DataFrame(rows)

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dcfg = self.cfg.data
        root = dcfg.dataset_root

        word_ds = NaturalStoriesDataset(
            root,
            stories=dcfg.stories,
            min_correct=dcfg.min_correct,
            aggregate=True,
        )
        word_df = word_ds.load_words().dropna(subset=["rt_mean"]).reset_index(drop=True)

        trial_ds = NaturalStoriesDataset(
            root,
            stories=dcfg.stories,
            min_correct=dcfg.min_correct,
            aggregate=False,
        )
        trials = trial_ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)
        if trials.empty:
            raise ValueError("No trial-level RT observations after filtering")
        logger.info(
            "NaturalStories LME: %d trials, %d words, %d subjects",
            len(trials),
            len(word_df),
            trials["participant_id"].nunique(),
        )
        return word_df, trials

    @staticmethod
    def _is_cbr_rnn(model_name: str) -> bool:
        """Match both cclark/cbr-rnn AND cclark/cbr-rnn-m (an exact-string
        match against only the base model name previously left cbr-rnn-m
        falling through to the generic HF logit-lens path, which always
        cache-misses since CBR-RNN isn't a real AutoModelForCausalLM — see
        broderick_lme.py, which already uses this same
        substring convention)."""
        name = model_name.lower()
        return "cbr-rnn" in name or "cbr_rnn" in name

    def _word_surprisal_all_layers(self, word_df: pd.DataFrame) -> np.ndarray:
        """Per-layer logit-lens surprisal for every word, one model load.

        CBR-RNN is single-layer — there is no "internal layers" concept to
        sweep, so it bypasses logit-lens entirely and uses plain
        final-output surprisal instead (``batch_surprisal_cbr_rnn``). Its
        output is ``(n_words, 1)``, which fits the ``(n_words, n_layers)``
        convention unchanged (n_layers=1) — the per-layer sweep in ``run()``
        just fits that one "layer".

        Returns
        -------
        np.ndarray  shape (n_words, n_layers).
        """
        mcfg = self.cfg.model
        dcfg = self.cfg.data
        tok_rev: str | None = None
        if mcfg.model_revision and "pythia" in mcfg.model_name.lower():
            tok_rev = "main"

        contexts = word_df["context"].astype(str).tolist()
        targets = word_df["word"].astype(str).tolist()

        is_cbr_rnn = self._is_cbr_rnn(mcfg.model_name)
        try:
            if is_cbr_rnn:
                from ..features.cbr_rnn import batch_surprisal_cbr_rnn

                surp_mat = batch_surprisal_cbr_rnn(
                    mcfg.model_name,
                    contexts,
                    targets,
                    device=mcfg.device,
                    n_params=mcfg.n_params,
                )
                return np.asarray(surp_mat, dtype=np.float64).reshape(len(targets), -1)

            from ..features.logit_lens import logit_lens_surprisal_trials

            surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

            arr = logit_lens_surprisal_trials(
                contexts,
                targets,
                mcfg.model_name,
                revision=mcfg.model_revision,
                device=mcfg.device,
                tokenizer_no_revision=tok_rev is not None,  # was "tok_rev is None" — inverted; only pythia+revision should force the main-branch tokenizer
                n_params=mcfg.n_params,
                trust_remote_code=mcfg.trust_remote_code,
                untrained=mcfg.untrained,
                untrained_seed=mcfg.untrained_seed,
                cache_dir=dcfg.feature_cache_dir,
                cache_key="natural_stories",
                require_cache_hit=self.cfg.lme.require_cached_surprisal,
                dtype=surp_dtype,
            )  # (n_words, n_layers)
            return np.asarray(arr, dtype=np.float64)
        finally:
            if not is_cbr_rnn:
                clear_hf_model_cache()
