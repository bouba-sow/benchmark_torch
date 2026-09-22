"""UCL self-paced reading benchmark — trial-level LME (pymer4/lme4), per-layer
logit-lens surprisal sweep.

Fits a by-subject mixed-effects model (raw RT, Zipf frequency + current-word +
spillover surprisal, random slopes by subject) on the Frank et al. (2013)
broad-coverage self-paced reading corpus. Same per-layer logit-lens design as
the neural/EEG LME datasets and ``natural_stories_lme.py`` (see that module's
docstring for the extraction/fitting rationale): word-level surprisal is
extracted ONCE per (model, revision) as ``(n_words, n_layers)`` via
``logit_lens_surprisal_trials``, then one LME is fit per layer via a process
pool. ``UCLReadingLmeConfig`` (this module) is shared by Provo, GECO, MECO,
OneStop, and ZuCo Reading — its ``layer_stride``/``n_workers`` fields apply to
all of them.

Output: ``ucl_reading_spr_lme_checkpoint_scores.csv`` — one row per (layer, revision),
written incrementally as each layer's fit completes (crash-safe restart).
"""

from __future__ import annotations

import logging
import time
import typing as tp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.ucl_reading import UCLReadingDataset
from ..features.llm import clear_hf_model_cache
from ..sweep import csv_file_lock
from .base import ModelConfig

logger = logging.getLogger(__name__)

def _is_cbr_rnn(model_name: str) -> bool:
    """Match both cclark/cbr-rnn AND cclark/cbr-rnn-m (an exact-string
    match against only the base model name previously left cbr-rnn-m
    falling through to the generic HF logit-lens path, which always
    cache-misses since CBR-RNN isn't a real AutoModelForCausalLM — see
    broderick_lme.py, which already uses this same
    substring convention)."""
    name = model_name.lower()
    return "cbr-rnn" in name or "cbr_rnn" in name

# Fixed column order for <dataset>_lme_by_layer.csv — a crash-fallback row
# (only a few keys) is reindexed to this before writing, so it can never
# misalign the CSV's columns. Shared by every benchmark built on
# UCLReadingLmeConfig/fit_ucl_reading_lme (UCL Reading, Provo, GECO, MECO,
# OneStop, ZuCo Reading).
LAYER_ROW_COLUMNS = (
    "layer", "beta", "se", "t_stat", "p_value", "prev_surprisal_beta",
    "loglik", "aic", "bic", "baseline_loglik", "delta_loglik",
    "delta_loglik_train", "delta_loglik_test", "baseline_loglik_test",
    "loglik_test", "lrt_p_value", "test_pearsonr_full_model", "test_spearmanr_full_model",
    "test_pearsonr_baseline_lme", "test_spearmanr_baseline_lme",
    "test_delta_pearsonr", "test_delta_spearmanr", "converged",
    "n_obs", "n_obs_test", "error", "fit_wall_time_s", "revision",
)


@dataclass
class UCLReadingDataConfig:
    """UCL Reading dataset location and loading options."""

    dataset_root: str
    sentences: list[int] | None = None
    min_accuracy: float = 0.0
    min_rt: float = 100.0
    max_rt: float = 3000.0
    feature_cache_dir: str | None = None
    wordfreq_language: str = "en"


@dataclass
class UCLReadingLmeConfig:
    """LME options (by-subject random slopes, raw RT — Clark-2025-style fit).

    Shared by UCL Reading, Provo, GECO, MECO, OneStop, and ZuCo Reading (they
    all import this class rather than redefining it).
    """

    include_surprisal: bool = True
    include_prev_surprisal: bool = True
    # Zipf lexical frequency nuisance (wordfreq), same convention as the
    # neural LME datasets — replaces the earlier KenLM-based nuisance.
    include_freq: bool = True
    filter_sentence_edges: bool = True
    compare_baseline: bool = True
    out_of_sample: bool = True
    partition_mod: int = 4
    partition_fit_indices: tuple[int, ...] = (0, 1)
    partition_test_indices: tuple[int, ...] = (2, 3)
    verbose: bool = False
    # When True, a surprisal cache miss raises instead of falling back to loading
    # the model and computing from scratch. Intended for CPU-only Stage 2 LME jobs
    # in the two-stage Jean Zay pipeline, where a cache miss should never happen
    # (Stage 1 GPU prefetch is supposed to have already populated it) and silently
    # computing on CPU is both very slow and can crash outright for
    # trust_remote_code models whose custom code unconditionally probes for a GPU
    # driver (e.g. flash-linear-attention/Triton, used by alibi/forgetting/
    # stickbreaking). Sweep scripts set this automatically when device == "cpu";
    # left False by default so direct/local/smoke-test usage is unaffected.
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
    # Explicit layer index list — when set, overrides layer_stride AND
    # last_layer_only entirely. Meant for a cheap SECOND pass, after a main
    # sweep has already identified each model's best layer (max
    # delta_loglik_test) from a first pass fit with
    # save_word_predictions=False: re-fit ONLY {last_layer, best_layer}
    # with save_word_predictions=True, reusing the already-cached base
    # tables/surprisal (same revision), instead of saving predictions for
    # every layer of every model (the dominant disk/IO cost otherwise).
    layers: tuple[int, ...] | None = None
    # Process-pool size for fitting layers concurrently. Each worker is a
    # separate process (spawn) with its own embedded R/pymer4 interpreter —
    # rpy2 is not thread-safe, so this must be process-based.
    n_workers: int = 10
    # Save each held-out word's true value + baseline/full model predictions
    # (yhat_baseline, yhat_full) and per-word delta log-likelihood to
    # <file_prefix>_lme_per_word.csv, one row per (layer, word) — same
    # DLT-difficulty-analysis convention as the neural LME datasets'
    # SAVE_WORD_PREDICTIONS. On by default.
    save_word_predictions: bool = True
    # When True, switches to Kuribayashi et al. 2025's exact UCL/SPR
    # baseline+full predictor set (arXiv:2502.01615) instead of this
    # pipeline's default -- see ucl_reading_lme_formulas' docstring in
    # neuroai/regressors/ucl_reading_lme.py. False by default so GECO/MECO/
    # Provo/OneStop/ZuCo Reading (which share this config class) are
    # unaffected.
    paper_replication: bool = False
    # Oh & Schuler (2023, arXiv:2304.11389) replication check: adds a
    # crossed (1 | sentid) random intercept (their eye-gaze random-effects
    # spec). Combine with include_freq=False, include_prev_surprisal=False
    # to match their fixed-effect set too. False by default.
    add_sentence_intercept: bool = False


@dataclass
class UCLReadingLmeBenchmarkConfig:
    data: UCLReadingDataConfig
    model: ModelConfig
    lme: UCLReadingLmeConfig = field(default_factory=UCLReadingLmeConfig)
    output_dir: str = "outputs"
    # Grouping label for saved CSVs when HF load repo differs (e.g. OLMo early-training).
    canonical_model_name: str | None = None


# ---------------------------------------------------------------------------
# Worker (module-level for spawn-based multiprocessing pickling)
# ---------------------------------------------------------------------------

def fit_one_layer_worker(
    trials_base_parquet_path: str,
    word_base_parquet_path: str,
    surprisal_npy_path: str,
    layer_idx: int,
    *,
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
    wordfreq_language: str = "en",
    merge_key_cols: tuple[str, ...] = ("story_id", "zone"),
    sort_group_col: str = "story_id",
    paper_replication: bool = False,
    add_sentence_intercept: bool = False,
) -> dict[str, tp.Any]:
    """Runs in its own spawned process: build this layer's trial frame and fit.

    Shared by every ``UCLReadingLmeConfig``-based benchmark (UCL Reading,
    Provo, GECO, MECO, OneStop, ZuCo Reading) — each just points it at its
    own parquet/npy paths and merge keys. Mirrors
    ``natural_stories_lme.py::_fit_one_layer_worker``.
    """
    import numpy as _np
    import pandas as _pd

    from ..regressors.ucl_reading_lme import fit_ucl_reading_lme

    t0 = time.time()
    row: dict[str, tp.Any] = {"layer": layer_idx}
    per_word_out: pd.DataFrame | None = None
    try:
        word_base = _pd.read_parquet(word_base_parquet_path)
        surp_all = _np.load(surprisal_npy_path, mmap_mode="r")
        word_layer = word_base.copy()
        word_layer["surprisal"] = _np.asarray(surp_all[:, layer_idx], dtype=_np.float64)
        if include_prev_surprisal:
            word_layer = word_layer.sort_values(list(merge_key_cols))
            word_layer["prev_surprisal"] = word_layer.groupby(sort_group_col, sort=False)[
                "surprisal"
            ].shift(1)

        trials_base = _pd.read_parquet(trials_base_parquet_path)
        pred_cols = [
            c for c in word_layer.columns
            if c in ("surprisal", "prev_surprisal")
        ]
        merge_cols = list(merge_key_cols) + pred_cols
        trials = trials_base.merge(word_layer[merge_cols], on=list(merge_key_cols), how="inner")

        sub = trials.dropna(subset=["surprisal", "rt"]).copy()
        fit = fit_ucl_reading_lme(
            sub,
            predictor_col="surprisal",
            compare_baseline=compare_baseline,
            out_of_sample=out_of_sample,
            partition_mod=partition_mod,
            partition_fit_indices=partition_fit_indices,
            partition_test_indices=partition_test_indices,
            include_freq=include_freq,
            include_prev_surprisal=include_prev_surprisal,
            filter_sentence_edges=filter_sentence_edges,
            wordfreq_language=wordfreq_language,
            verbose=verbose,
            layer=layer_idx,
            save_per_word=save_word_predictions,
            paper_replication=paper_replication,
            add_sentence_intercept=add_sentence_intercept,
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


def run_layer_sweep(
    *,
    out_dir: Path,
    revision: str,
    model_name: str,
    trials: pd.DataFrame,
    word_base: pd.DataFrame,
    surprisal_layers: np.ndarray,
    lme_cfg: UCLReadingLmeConfig,
    scores_csv_name: str,
    per_word_csv_name: str,
    file_prefix: str,
    extra_fields: dict[str, tp.Any] | None = None,
    wordfreq_language: str = "en",
    merge_key_cols: tuple[str, ...] = ("story_id", "zone"),
    sort_group_col: str = "story_id",
) -> pd.DataFrame:
    """Shared driver: write base tables once, fit every layer via a process pool.

    Used by every ``UCLReadingLmeConfig``-based benchmark's ``run()`` — only
    the tables/merge keys/output filenames differ per dataset.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_layers = surprisal_layers.shape[1]

    # Revision MUST be in these filenames: out_dir is per-MODEL, not
    # per-checkpoint, so a training-step sweep (e.g. "pythia_checkpoints",
    # 11 revisions per model) calls this function once per revision against
    # the SAME out_dir. Without the revision in the filename, concurrent
    # checkpoints of the same model (true both for a SLURM array's %N
    # throttle -- consecutive array indices are the same model's consecutive
    # steps -- and for any other same-model parallelism) race on these paths:
    # one checkpoint's write can land, then a sibling checkpoint's fit reads
    # THAT data instead of its own, silently corrupting results (verified:
    # produced near-identical delta_loglik across supposedly-different
    # training steps in both a SLURM array run and an xargs-parallel run).
    safe_revision = revision.replace("/", "_")
    word_base_parquet_path = out_dir / f"{file_prefix}_lme_word_base_{safe_revision}.parquet"
    trials_base_parquet_path = out_dir / f"{file_prefix}_lme_trials_base_{safe_revision}.parquet"
    surprisal_npy_path = out_dir / f"{file_prefix}_lme_surprisal_{safe_revision}.npy"
    word_base.to_parquet(word_base_parquet_path, index=False)
    trials.to_parquet(trials_base_parquet_path, index=False)
    np.save(surprisal_npy_path, surprisal_layers.astype(np.float32))

    if lme_cfg.layers is not None:
        layer_indices = [li for li in lme_cfg.layers if 0 <= li < n_layers]
    elif lme_cfg.last_layer_only:
        layer_indices = [n_layers - 1]
    else:
        layer_indices = list(range(0, n_layers, max(1, lme_cfg.layer_stride)))
    logger.info(
        "[%s] Fitting %d layers (stride=%d, last_layer_only=%s, layers=%s) with %d worker processes",
        model_name, len(layer_indices), lme_cfg.layer_stride, lme_cfg.last_layer_only,
        lme_cfg.layers, lme_cfg.n_workers,
    )

    out_csv = out_dir / scores_csv_name
    per_word_csv = out_dir / per_word_csv_name

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

    # A layer's score row already existing does NOT mean its per-word
    # predictions do too -- e.g. a first pass run with
    # save_word_predictions=False, followed by a second pass with it on for
    # just {last, best} layer (see UCLReadingLmeConfig.layers). Without this
    # check, that second pass would see the layer already in done_layers and
    # skip it entirely, silently writing no predictions at all. Only checked
    # when save_word_predictions is actually requested this run -- otherwise
    # every already-scored layer stays "done" as before.
    already_predicted: set[int] = set()
    if lme_cfg.save_word_predictions and per_word_csv.exists():
        try:
            existing_pw = pd.read_csv(per_word_csv, usecols=["revision", "layer"])
            already_predicted = set(
                existing_pw.loc[existing_pw["revision"] == revision, "layer"].unique().tolist()
            )
        except Exception:
            logger.warning("Could not read existing per-word CSV; will not treat any layer as prediction-complete.")

    def _needs_work(li: int) -> bool:
        if li not in done_layers:
            return True
        return lme_cfg.save_word_predictions and li not in already_predicted

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    out_columns = list(LAYER_ROW_COLUMNS) + [
        k for k in (extra_fields or {}) if k not in LAYER_ROW_COLUMNS
    ]

    def _submit_kwargs(li: int) -> dict:
        return dict(
            trials_base_parquet_path=str(trials_base_parquet_path),
            word_base_parquet_path=str(word_base_parquet_path),
            surprisal_npy_path=str(surprisal_npy_path),
            layer_idx=li,
            compare_baseline=lme_cfg.compare_baseline,
            out_of_sample=lme_cfg.out_of_sample,
            partition_mod=lme_cfg.partition_mod,
            partition_fit_indices=lme_cfg.partition_fit_indices,
            partition_test_indices=lme_cfg.partition_test_indices,
            include_freq=lme_cfg.include_freq,
            include_prev_surprisal=lme_cfg.include_prev_surprisal,
            filter_sentence_edges=lme_cfg.filter_sentence_edges,
            verbose=lme_cfg.verbose,
            save_word_predictions=lme_cfg.save_word_predictions,
            wordfreq_language=wordfreq_language,
            merge_key_cols=merge_key_cols,
            sort_group_col=sort_group_col,
            paper_replication=lme_cfg.paper_replication,
            add_sentence_intercept=lme_cfg.add_sentence_intercept,
        )

    def _write_row(row: dict, per_word: pd.DataFrame | None, *, skip_score_row: bool = False) -> None:
        row = dict(row)
        row["revision"] = revision
        if extra_fields:
            row.update(extra_fields)
        if not skip_score_row:
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
            row["layer"], row.get("fit_wall_time_s", float("nan")),
            f"{row['delta_loglik_test']:.2f}" if row.get("delta_loglik_test") is not None else "NA",
            row.get("converged"),
        )

    def _run_pool(layer_ids: list[int], max_workers: int, *, allow_broken_pool_retry: bool) -> tuple[list[dict], list[int]]:
        """Fit ``layer_ids`` in a fresh pool.

        Returns ``(rows, broken_pool_layers)``. ``rows`` holds a final
        (written) result dict for every layer EXCEPT those in
        ``broken_pool_layers`` when ``allow_broken_pool_retry`` — those are
        left unwritten so the caller can retry them in a new pool instead of
        recording a spurious failure for a layer that was simply an innocent
        bystander when a SIBLING worker crashed (see note below).
        """
        rows: list[dict] = []
        broken: list[int] = []
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx, max_tasks_per_child=1,
        ) as pool:
            futures = {pool.submit(fit_one_layer_worker, **_submit_kwargs(li)): li for li in layer_ids}
            for fut in as_completed(futures):
                li = futures[fut]
                try:
                    result = fut.result()
                    row, per_word = result["row"], result["per_word"]
                except BrokenProcessPool as exc:
                    # A SINGLE worker process dying unexpectedly (segfault in
                    # R/lme4, OOM-kill, ...) poisons the WHOLE
                    # ProcessPoolExecutor: every other still-pending/running
                    # future in the same pool — even layers whose own worker
                    # was healthy and mid-fit — immediately raises this same
                    # BrokenProcessPool, all at the same timestamp. That's why
                    # several unrelated layer numbers were observed "crashing"
                    # simultaneously in practice: only one of them actually
                    # failed; the rest were just sharing its pool. Retrying
                    # those layers in a NEW pool (once) recovers the ones that
                    # were never actually broken.
                    if allow_broken_pool_retry:
                        logger.warning(
                            "Layer %d: pool poisoned by a sibling worker crash (%s) — will retry in a fresh pool.",
                            li, exc,
                        )
                        broken.append(li)
                        continue
                    logger.error("Layer %d worker crashed (no more retries): %s", li, exc, exc_info=True)
                    row, per_word = {"layer": li, "error": str(exc)}, None
                except Exception as exc:
                    logger.error("Layer %d worker crashed: %s", li, exc, exc_info=True)
                    row, per_word = {"layer": li, "error": str(exc)}, None
                rows.append(row)
                # li already had a good score row before this run (we're only
                # here to backfill predictions for it) -- don't duplicate it.
                _write_row(row, per_word, skip_score_row=li in done_layers)
        return rows, broken

    pending = [li for li in layer_indices if _needs_work(li)]
    rows, broken = _run_pool(pending, lme_cfg.n_workers, allow_broken_pool_retry=True)
    if broken:
        logger.info("Retrying %d layer(s) poisoned by a sibling crash: %s", len(broken), broken)
        # Smaller pool for the retry — reduces the odds of the same cascade
        # (one crashed worker taking out several healthy siblings) recurring.
        retry_workers = max(1, min(lme_cfg.n_workers, len(broken)))
        retry_rows, still_broken = _run_pool(broken, retry_workers, allow_broken_pool_retry=False)
        rows.extend(retry_rows)
        assert not still_broken  # allow_broken_pool_retry=False never returns broken layers

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class UCLReadingLmeBenchmark:
    """Trial-level UCL Reading SPR LME, one fit per LM layer.

    Pipeline
    --------
    1. Load word-level (subject-averaged) and trial-level (per-subject) RTs.
    2. Extract per-layer logit-lens surprisal for every word (one model
       load), disk-cached. Zipf frequency (nuisance) is computed fresh from
       the ``word`` column at fit-time, inside ``fit_ucl_reading_lme``.
    3. Write the word/trial base tables (parquet) and the ``(n_words,
       n_layers)`` surprisal array (npy) once, then fit one ``lmer`` per
       layer via a process pool (CPU; only the surprisal extraction step
       uses ``device``).
    """

    def __init__(self, cfg: UCLReadingLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def extract_and_cache_surprisal(self) -> np.ndarray:
        """GPU-stage entry point: populate ``data.feature_cache_dir`` only."""
        word_df, _ = self._load_data()
        return self._word_surprisal_all_layers(word_df)

    def run(self, extra_fields: dict[str, tp.Any] | None = None) -> pd.DataFrame:
        dcfg = self.cfg.data
        lme_cfg = self.cfg.lme
        out_dir = Path(self.cfg.output_dir)

        revision = self.cfg.model.model_revision or "main"
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name

        word_df, trials = self._load_data()

        if not lme_cfg.include_surprisal:
            logger.info("[%s] include_surprisal=False — nothing to fit", model_name)
            return pd.DataFrame()

        surprisal_layers = self._word_surprisal_all_layers(word_df)  # (n_words, n_layers)
        clear_hf_model_cache()

        word_cols = ["story_id", "zone"]
        return run_layer_sweep(
            out_dir=out_dir,
            revision=revision,
            model_name=model_name,
            trials=trials,
            word_base=word_df[word_cols],
            surprisal_layers=surprisal_layers,
            lme_cfg=lme_cfg,
            scores_csv_name="ucl_reading_spr_lme_checkpoint_scores.csv",
            per_word_csv_name="ucl_reading_spr_lme_per_word.csv",
            file_prefix="ucl_reading",
            extra_fields=extra_fields,
            wordfreq_language=dcfg.wordfreq_language,
        )

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dcfg = self.cfg.data
        root = dcfg.dataset_root

        word_ds = UCLReadingDataset(
            root,
            sentences=dcfg.sentences,
            min_accuracy=dcfg.min_accuracy,
            min_rt=dcfg.min_rt,
            max_rt=dcfg.max_rt,
            aggregate=True,
        )
        word_df = word_ds.load_words().dropna(subset=["rt_mean"]).reset_index(drop=True)

        trial_ds = UCLReadingDataset(
            root,
            sentences=dcfg.sentences,
            min_accuracy=dcfg.min_accuracy,
            min_rt=dcfg.min_rt,
            max_rt=dcfg.max_rt,
            aggregate=False,
        )
        trials = trial_ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)
        if trials.empty:
            raise ValueError("No trial-level RT observations after filtering")
        logger.info(
            "UCLReading LME: %d trials, %d words, %d subjects",
            len(trials),
            len(word_df),
            trials["participant_id"].nunique(),
        )
        return word_df, trials

    def _word_surprisal_all_layers(self, word_df: pd.DataFrame) -> np.ndarray:
        """Per-layer logit-lens surprisal for every word, one model load."""
        mcfg = self.cfg.model
        dcfg = self.cfg.data

        if _is_cbr_rnn(mcfg.model_name):
            # CBR-RNN is not a HF causal LM and has no logit-lens cache — see
            # provo_lme.py / natural_stories_lme.py's identical branch.
            from ..features.cbr_rnn import batch_surprisal_cbr_rnn

            contexts = word_df["context"].astype(str).tolist()
            targets = word_df["word"].astype(str).tolist()
            surp_mat = batch_surprisal_cbr_rnn(
                mcfg.model_name, contexts, targets, device=mcfg.device, n_params=mcfg.n_params,
            )
            return np.asarray(surp_mat, dtype=np.float64).reshape(len(targets), -1)

        tok_rev: str | None = None
        if mcfg.model_revision and "pythia" in mcfg.model_name.lower():
            tok_rev = "main"

        contexts = word_df["context"].astype(str).tolist()
        targets = word_df["word"].astype(str).tolist()

        from ..features.logit_lens import logit_lens_surprisal_trials

        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

        try:
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
                cache_key="ucl_reading",
                require_cache_hit=self.cfg.lme.require_cached_surprisal,
                dtype=surp_dtype,
            )  # (n_words, n_layers)
            return np.asarray(arr, dtype=np.float64)
        finally:
            clear_hf_model_cache()
