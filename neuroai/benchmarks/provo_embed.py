"""Provo Corpus eye-tracking benchmark: ridge regression on LLM layer activations.

Predicts word-level reading time (Provo SP/FP/GP) from per-layer contextual
embeddings, using the same per-layer ridge regression as the Podcast iEEG benchmark
but applied to Provo behavioral RT. Complements :mod:`neuroai.benchmarks.provo_lme` (scalar surprisal ->
RT via mixed-effects model): same dataset, same word set, orthogonal predictor
(full hidden state vs. one scalar).

Fit independently **per subject** — each subject's own raw per-word RT trials
get their own ridge CV (same embedding per word is reused across the subjects
who read it; RT is never averaged across subjects). "Subject" plays the role
of "electrode" in ``NeuralBenchmark``. Per-subject scores are logged and saved
individually (``*_by_subject.csv``) as each subject finishes; the headline
per-layer score is the mean across subjects (``*_checkpoint_scores.csv``),
analogous to averaging across electrodes for a neural model score.

Subjects can be fit two ways, controlled by ``subject_n_jobs``:

* ``subject_n_jobs=1`` (default) — subjects processed **serially** in-process;
  layers within one subject are parallelized via a thread pool (``n_jobs``,
  mirrors ``ZuCoEegEncoder._regress_cpu``'s ``prefer="threads"`` layer-
  parallelism). Zero extra memory (threads share the parent's ``emb_sub``),
  but thread-parallelism only speeds up the actual BLAS matrix math inside
  ``cv_ridge_score`` — the Python-level glue around it (fold splitting, group
  masks, building result dicts) stays GIL-bound and serializes across
  threads, so wall time is well short of an ``n_jobs``-fold speedup.
* ``subject_n_jobs>1`` — a **bounded** ``ProcessPoolExecutor`` fits several
  subjects concurrently via a sliding window (never more than
  ``subject_n_jobs`` subjects' embedding arrays in memory at once — see
  ``_iter_subject_data``, which yields one subject at a time rather than
  materializing the whole dataset up front), each worker still using
  ``n_jobs`` threads internally for its own layers. Subjects are fully
  independent (no shared state), so this scales close to linearly with
  ``subject_n_jobs`` — this is how the original (pre-OOM-fix) design worked,
  just capped instead of matching subject count. Memory cost is real: each
  concurrent worker holds its own full ``(n_trials, n_layers, hidden)`` copy,
  so set ``subject_n_jobs`` (and ``n_jobs``, so ``subject_n_jobs * n_jobs``
  stays within your node's core count) based on the model size and RAM you're
  willing to spend — e.g. low (or 1) for a 60B+-param model's huge per-layer
  hidden size, higher for a small model. There is no auto-sizing; this is a
  manual per-family/config knob, same as ``n_jobs`` already is.

Either way, each subject's rows are handed to an optional ``on_subject_done``
callback immediately after that subject finishes (see ``run()``; the process-
pool path fires this as each in-flight subject completes, not only after the
whole pool drains), letting the caller persist them right away — a crash
mid-checkpoint loses at most the in-flight subject(s), not every subject fit
so far. Pass ``existing_subject_rows`` to skip subjects a previous (crashed)
run already completed for this exact checkpoint.

Pipeline
--------
1. ``ProvoDataset`` (aggregate=False) -> one row per (participant_id, story_id,
   zone) trial with raw ``rt`` and ``is_edge``.
2. For each passage, run the model once over the *full* token sequence (including
   the unscoreable passage-initial word, needed for correct context) via
   :func:`neuroai.features.llm.contextual_word_embeddings_hf` with
   ``layer_index="all"`` -> ``(n_tokens, n_layers, hidden)`` at the last-token
   position of each word. Cached to disk per (model, revision, section_id); the
   cache key does not include the Provo measure, so SP/FP/GP sweeps sharing one
   model/checkpoint reuse the same embeddings (mirrors the surprisal cache's
   measure-agnostic sharing — see ``provo_lme.py`` module docstring). Computed
   once and shared across all subjects (embeddings don't depend on the subject).
3. Per subject: ridge regression (nested CV) per layer: ``rt ~ embedding``.
   ``GroupKFold`` on ``story_id`` (leave-passages-out) is the default within
   each subject's own trials — Provo is only 55 passages, and words within one
   passage share overlapping context, so plain shuffled ``KFold`` would let
   correlated neighbours leak across the train/test split.
"""

from __future__ import annotations

import logging
import typing as tp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..datasets.provo import ProvoDataset, passage_word_lists
from ..regressors.ridge import (
    CVScore,
    LayerwiseRidgeResults,
    cv_ridge_score,
    cv_ridge_score_multioutput_gpu,
)
from .base import ModelConfig, RegressionConfig, word_predictions_root, word_predictions_dir
from .subject_average import average_items_across_subjects, fit_averaged_layers, fit_averaged_layers_gpu

logger = logging.getLogger(__name__)


def _fit_subject_layers(
    pid: str, emb_sub: np.ndarray, rt: np.ndarray, groups: np.ndarray | None,
    *,
    model_name: str, measure: str,
    n_splits: int, alpha, scale: bool, random_state: int, print_chosen_alpha: bool,
    thread_n_jobs: int,
    save_word_predictions: bool = False,
    word_ids: pd.DataFrame | None = None,
) -> tuple[list[dict], pd.DataFrame | None]:
    """Fit ridge CV for every layer of one subject's trials.

    Module-level (not a method) so it is picklable and can run either inline
    (serial path) or as a ``ProcessPoolExecutor`` task (bounded subject-level
    parallelism, see module docstring). Layers within the call are themselves
    parallelized via a thread pool (BLAS releases the GIL during the fit).

    When ``save_word_predictions``, the LAST layer is additionally refit with
    ``return_oof_predictions=True`` (a per-target best-layer refit isn't worth
    the extra regression calls here, mirroring ``zuco_reading_embed.py``'s
    convention) and a per-word DataFrame of observed vs. out-of-fold-predicted
    RT is returned alongside the per-layer score rows.
    """
    n_layers = emb_sub.shape[1]

    def _fit_layer(li: int) -> dict:
        s = cv_ridge_score(
            emb_sub[:, li, :], rt,
            n_splits=n_splits, alpha=alpha, scale=scale,
            random_state=random_state, groups=groups,
            print_chosen_alpha=print_chosen_alpha,
        )
        return dict(
            model_name=model_name, measure=measure, participant_id=pid,
            layer=li, layer_rel=li / max(1, n_layers - 1), n_layers=n_layers,
            n_trials=len(rt), **s.to_dict(),
        )

    if thread_n_jobs > 1:
        rows = Parallel(n_jobs=thread_n_jobs, prefer="threads")(
            delayed(_fit_layer)(li) for li in range(n_layers)
        )
    else:
        rows = [_fit_layer(li) for li in range(n_layers)]
    rows.sort(key=lambda r: r["layer"])

    per_word: pd.DataFrame | None = None
    if save_word_predictions:
        last = n_layers - 1
        _, oof, valid_mask = cv_ridge_score(
            emb_sub[:, last, :], rt,
            n_splits=n_splits, alpha=alpha, scale=scale,
            random_state=random_state, groups=groups,
            print_chosen_alpha=False, return_oof_predictions=True,
        )
        per_word = word_ids.loc[valid_mask].reset_index(drop=True).copy()
        per_word["model_name"] = model_name
        per_word["measure"] = measure
        per_word["participant_id"] = pid
        per_word["layer"] = last
        per_word["n_layers"] = n_layers
        per_word["rt_true"] = rt[valid_mask]
        per_word["rt_pred_oof"] = oof

    return rows, per_word


def _fit_subject_layers_gpu(
    pid: str, emb_sub: np.ndarray, rt: np.ndarray, groups: np.ndarray | None,
    *,
    model_name: str, measure: str,
    n_splits: int, alpha, scale: bool, random_state: int,
    n_layer_workers: int,
    save_word_predictions: bool = False,
) -> tuple[list[dict], np.ndarray | None]:
    """Fit ridge CV for every layer of one subject's trials — GPU path.

    Same per-subject contract/return format as :func:`_fit_subject_layers`,
    but each layer's fit runs on GPU via ``cv_ridge_score_multioutput_gpu``
    (himalaya ``torch_cuda`` backend, SVD-based GCV) with a single target
    column (this subject's ``rt``) — mirrors
    ``natural_stories_embed_gpu.py``'s ``_regress``. NOT batched across
    subjects: Provo subjects each read a different item subset (0 of 2656
    items are common to all 84 subjects), so there is no shared row set a
    single SVD could serve for more than one subject at once — the multi-
    output speedup that works for podcast's electrodes (dense, same time
    samples for every electrode) doesn't transfer here. This still moves the
    per-layer linear algebra off CPU sklearn onto GPU, with per-layer CUDA
    streams for concurrency within one subject, but there is no cross-subject
    amortization — call this once per subject, not once for a batch.

    ``r2``/``mse`` come back NaN (matching every other GPU-path caller of
    ``cv_ridge_score_multioutput_gpu`` in this codebase) — ``pearsonr``/
    ``spearmanr`` remain fully populated.

    Requires ``groups`` (GroupKFold on story_id) — the GPU function has no
    ungrouped-KFold fallback, unlike ``cv_ridge_score``.

    When ``save_word_predictions``, every layer's out-of-fold predictions are
    captured for free (``return_oof_predictions=True`` costs nothing extra on
    the GPU path — mirrors ``natural_stories_embed_gpu.py``'s convention) and
    returned as ``(n_layers, n_trials)`` alongside the per-layer score rows.
    """
    import torch

    if groups is None:
        raise ValueError(
            "_fit_subject_layers_gpu requires groups (group_cv_by_story=True); "
            "the GPU ridge path has no ungrouped-KFold fallback."
        )

    n_layers = emb_sub.shape[1]
    Y = rt.reshape(-1, 1).astype(np.float32)
    n_splits_eff = min(n_splits, len(np.unique(groups)))

    reg_kw = dict(
        n_splits=n_splits_eff, alpha=alpha, scale=scale,
        standardize_target=True, random_state=random_state, groups=groups,
    )

    def _fit_layer(li: int) -> tuple[dict, np.ndarray | None]:
        X_li = emb_sub[:, li, :].astype(np.float32)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            if save_word_predictions:
                scores, oof = cv_ridge_score_multioutput_gpu(X_li, Y, return_oof_predictions=True, **reg_kw)
                oof_li = oof[:, 0]
            else:
                scores = cv_ridge_score_multioutput_gpu(X_li, Y, **reg_kw)
                oof_li = None
        stream.synchronize()
        s = scores[0]
        row = dict(
            model_name=model_name, measure=measure, participant_id=pid,
            layer=li, layer_rel=li / max(1, n_layers - 1), n_layers=n_layers,
            n_trials=len(rt), **s.to_dict(),
        )
        return row, oof_li

    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_workers = min(n_layer_workers, n_layers)
    rows: list[dict] = [None] * n_layers  # type: ignore[list-item]
    oof_by_layer: list[np.ndarray | None] = [None] * n_layers
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fit_layer, li): li for li in range(n_layers)}
        for fut in as_completed(futures):
            row, oof_li = fut.result()
            rows[row["layer"]] = row
            oof_by_layer[row["layer"]] = oof_li

    oof_all = np.stack(oof_by_layer, axis=0).astype(np.float32) if save_word_predictions else None
    return rows, oof_all


def _mean_cv_score(rows: list[dict]) -> CVScore:
    """Aggregate one layer's per-subject score rows into a single
    mean-across-subjects :class:`CVScore` (analogous to averaging across
    electrodes)."""
    return CVScore(
        pearsonr=float(np.nanmean([r["pearsonr"] for r in rows])),
        spearmanr=float(np.nanmean([r["spearmanr"] for r in rows])),
        r2=float(np.nanmean([r["r2"] for r in rows])),
        mse=float(np.nanmean([r["mse"] for r in rows])),
        ridge_alpha=float(np.nanmean([r["ridge_alpha"] for r in rows])),
    )


@dataclass
class ProvoEmbedDataConfig:
    """Provo Corpus dataset location, measure, and embedding-extraction options."""

    root: str
    measure: str = "FP"  # "SP" | "FP" | "GP"
    texts: list[int] | None = None
    min_dur: float = 50.0
    max_dur: float = 5000.0
    exclude_doc_edges: bool = True
    # >= longest Provo passage (62 words) so every word's context is the full
    # cumulative passage prefix, matching the cumulative context used for surprisal
    # in provo_lme.py (not a fixed sliding window as in the longer LPP corpus).
    context_words: int = 64
    max_length: int | None = 512
    feature_cache_dir: str | None = None


@dataclass
class ProvoEmbedBenchmarkConfig:
    data: ProvoEmbedDataConfig
    model: ModelConfig
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    # Drop sentence- and passage-edge words (same convention as provo_lme's
    # filter_sentence_edges / UCL Reading / ZuCo Reading).
    filter_sentence_edges: bool = True
    # Leave-passages-out CV (GroupKFold on story_id) within each subject's own
    # trials. False -> plain shuffled KFold (risks within-passage leakage).
    group_cv_by_story: bool = True
    # Skip a subject if they have fewer than this many trials left after
    # filtering (too few for a meaningful GroupKFold CV).
    min_trials_per_subject: int = 20
    # Thread-pool workers for per-LAYER regression within one subject (mirrors
    # ZuCoEegEncoder._regress_cpu's layer-parallelism). Applies whether subjects
    # are processed serially or concurrently — see subject_n_jobs and module
    # docstring.
    n_jobs: int = 10
    # Concurrent subject-processes (ProcessPoolExecutor). 1 (default) = serial,
    # zero extra memory. >1 = real multi-core speedup (subjects are fully
    # independent) at the cost of one embedding-array copy per concurrent
    # worker — size this manually against your node's RAM and n_jobs (total
    # cores used ~= subject_n_jobs * n_jobs); see module docstring.
    subject_n_jobs: int = 1
    # "cpu" (default, sklearn RidgeCV via cv_ridge_score) or "cuda" (GPU,
    # cv_ridge_score_multioutput_gpu per subject — see _fit_subject_layers_gpu
    # docstring for why this is NOT batched across subjects like podcast's
    # electrodes). "cuda" requires group_cv_by_story=True and forces subjects
    # to be processed serially regardless of subject_n_jobs (a per-process
    # CUDA context per subject_n_jobs worker is not worth the complexity —
    # the per-layer CUDA-stream parallelism inside one subject already gives
    # the GPU speedup).
    regression_device: str = "cpu"
    # Thread-pool workers for per-LAYER GPU fits (torch.cuda.Stream each) —
    # only used when regression_device="cuda"; mirrors
    # natural_stories_embed_gpu.py's n_layer_workers.
    n_layer_workers: int = 8
    # Grouping label for saved CSVs when the HF load repo differs from the
    # canonical model id (e.g. OLMo early-training branches).
    canonical_model_name: str | None = None
    # Save each held-out trial's observed RT + out-of-fold-predicted RT, per
    # subject, to a per-word CSV, for post-hoc per-word difficulty splits
    # (e.g. DLT cost). CPU path: LAST layer only (mirrors
    # zuco_reading_embed.py). GPU path: every layer, free (mirrors
    # natural_stories_embed_gpu.py). On by default.
    save_word_predictions: bool = True
    output_dir: str | None = None
    # Directory for per-word/per-item OOF prediction .npz files
    # (save_word_predictions=True) specifically. Defaults to output_dir
    # when unset -- set separately (e.g. to a $SCRATCH path) to keep the
    # many small prediction files off the (smaller, shared) $WORK quota
    # the CSV summaries live on.
    word_predictions_dir: str | None = None
    # Additionally fit a single ridge CV per layer on RT averaged across
    # subjects per (story_id, zone) item (available-case mean — see
    # neuroai.benchmarks.subject_average module docstring), alongside the
    # per-subject fits above. Does not affect the per-subject outputs.
    average_across_subjects: bool = False
    # Drop items with fewer than this many contributing subjects from the
    # averaged-RT target (1 = available-case, keep every item any subject has).
    min_subjects_per_item: int = 1
    # Skip the per-subject ridge fits entirely and only compute the averaged-RT
    # branch (implies average_across_subjects). The two are independent — the
    # average is taken directly off raw per-word RT rows, not off any per-subject
    # fit — so this yields the cheap 1-fit-per-layer grand-average result without
    # paying the N-subject cost. Per-subject outputs (n_subjects, layerwise,
    # summary_df, per_subject_df) come back empty/NaN. Mirrors GecoEmbedBenchmarkConfig.
    average_only: bool = False
    # Ridge fit device for the averaged-RT branch specifically (default: cpu).
    # 'cuda' uses fit_averaged_layers_gpu (himalaya SVD-based GCV) instead of
    # sklearn's nested-CV grid search -- see GecoEmbedBenchmarkConfig's
    # identically-named field for the full rationale. Safe here for the same
    # reason: the averaged branch is always exactly one fit, no cross-subject
    # GPU contention.
    average_ridge_device: str = "cuda"  # himalaya GPU ridge everywhere (2026-10-14)


@dataclass
class ProvoEmbedBenchmarkResult:
    model_name: str
    measure: str
    n_trials: int
    n_subjects: int
    n_layers: int
    layerwise: LayerwiseRidgeResults  # mean-across-subjects score per layer
    flat_score: CVScore  # last-layer mean-across-subjects score
    summary_df: pd.DataFrame  # tidy, one row per layer (mean across subjects)
    per_subject_df: pd.DataFrame  # tidy, one row per (subject, layer)
    n_items_avg: int | None = None  # items in the averaged-RT branch (None if disabled)
    average_layerwise: LayerwiseRidgeResults | None = None  # ridge fit on RT averaged across subjects
    average_summary_df: pd.DataFrame | None = None  # tidy, one row per layer (averaged-RT branch)


class ProvoEmbedBenchmark:
    """Predict Provo reading times from LLM layer activations using ridge
    regression, fit independently per subject.

    Usage
    -----
    .. code-block:: python

        cfg = ProvoEmbedBenchmarkConfig(
            data=ProvoEmbedDataConfig(root="data/provo/osfstorage", measure="FP"),
            model=ModelConfig(model_name="EleutherAI/pythia-160m"),
        )
        result = ProvoEmbedBenchmark(cfg).run()
        print(result.layerwise.best_layer_score().spearmanr)  # mean across subjects
    """

    def __init__(self, cfg: ProvoEmbedBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        *,
        existing_subject_rows: list[dict] | None = None,
        on_subject_done: tp.Callable[[list[dict]], None] | None = None,
    ) -> ProvoEmbedBenchmarkResult:
        """Run the full benchmark.

        Parameters
        ----------
        existing_subject_rows
            Per-subject-per-layer row dicts already computed for this exact
            checkpoint by a previous (crashed/timed-out) run — their subjects
            are skipped and their rows are folded into the returned result
            as-is. ``None`` (default) means start fresh.
        on_subject_done
            Called with one subject's row dicts (all layers) immediately after
            that subject's ridge fits finish, before moving to the next
            subject — use this to persist results incrementally so a crash
            loses at most the in-progress subject.
        """
        trial_df = self._load_trials()
        emb_lookup = self._embed_words(trial_df)
        return self._regress(
            trial_df, emb_lookup,
            existing_subject_rows=existing_subject_rows, on_subject_done=on_subject_done,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_trials(self) -> pd.DataFrame:
        dcfg = self.cfg.data
        ds = ProvoDataset(
            dcfg.root,
            dcfg.measure,
            texts=dcfg.texts,
            min_dur=dcfg.min_dur,
            max_dur=dcfg.max_dur,
            exclude_doc_edges=dcfg.exclude_doc_edges,
            aggregate=False,
        )
        trial_df = ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)
        if self.cfg.filter_sentence_edges and "is_edge" in trial_df.columns:
            trial_df = trial_df.loc[~trial_df["is_edge"].astype(bool)].reset_index(drop=True)
        if trial_df.empty:
            raise ValueError(
                f"No Provo trial observations left after filtering (measure={dcfg.measure})"
            )
        trial_df = trial_df.sort_values(["participant_id", "story_id", "zone"]).reset_index(drop=True)
        logger.info(
            "Provo embed [%s]: %d trials, %d subjects, %d passages",
            dcfg.measure, len(trial_df), trial_df["participant_id"].nunique(),
            trial_df["story_id"].nunique(),
        )
        return trial_df

    def _embed_words(self, trial_df: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
        """Return ``{(story_id, zone): (n_layers, hidden)}`` — one embedding per
        word, computed once and shared across every subject who read it."""
        from ..features.llm import contextual_word_embeddings_hf, contextual_word_embeddings_hf_fulltext

        dcfg = self.cfg.data
        mcfg = self.cfg.model
        # Pythia checkpoints share one tokenizer across all steps (see
        # neuroai.features.surprisal._tokenizer_revision) — load it from the base
        # (non-revisioned) repo rather than the training-step branch.
        tokenizer_no_revision = bool(mcfg.model_revision) and "pythia" in mcfg.model_name.lower()
        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

        passages = passage_word_lists(dcfg.root)
        story_ids = sorted(trial_df["story_id"].unique())
        emb_lookup: dict[tuple[int, int], np.ndarray] = {}
        for story_id in story_ids:
            tokens = passages[int(story_id)]
            # Every Provo passage fits well under max_length in one chunk, so
            # this is a single forward pass per passage instead of one per word
            # (see contextual_word_embeddings_hf_fulltext docstring) — falls back
            # to the slower per-word path only for slow (non-fast) tokenizers.
            try:
                emb = contextual_word_embeddings_hf_fulltext(
                    tokens,
                    mcfg.model_name,
                    layer_index="all",
                    revision=mcfg.model_revision,
                    device=mcfg.device,
                    max_length=dcfg.max_length,
                    cache_dir=dcfg.feature_cache_dir,
                    section_id=int(story_id),
                    lang="en",
                    tokenizer_no_revision=tokenizer_no_revision,
                    n_params=mcfg.n_params,
                    trust_remote_code=mcfg.trust_remote_code,
                    untrained=mcfg.untrained,
                    untrained_seed=mcfg.untrained_seed,
                    dtype=surp_dtype,
                )
            except ValueError as exc:
                if "fast tokenizer" not in str(exc):
                    raise
                logger.warning("Falling back to per-word extraction (slow tokenizer): %s", exc)
                emb = contextual_word_embeddings_hf(
                    tokens,
                    mcfg.model_name,
                    layer_index="all",
                    revision=mcfg.model_revision,
                    device=mcfg.device,
                    context_words=dcfg.context_words,
                    max_length=dcfg.max_length,
                    batch_size=mcfg.batch_size,
                    cache_dir=dcfg.feature_cache_dir,
                    section_id=int(story_id),
                    lang="en",
                    tokenizer_no_revision=tokenizer_no_revision,
                    n_params=mcfg.n_params,
                    trust_remote_code=mcfg.trust_remote_code,
                    untrained=mcfg.untrained,
                    untrained_seed=mcfg.untrained_seed,
                    dtype=surp_dtype,
                )
            for zone in range(1, len(tokens) + 1):
                emb_lookup[(int(story_id), zone)] = emb[zone - 1]

        logger.info(
            "Provo embed: extracted embeddings for %d words across %d passages",
            len(emb_lookup), len(story_ids),
        )
        return emb_lookup

    def _fit_one_subject(
        self, pid: str, emb_sub: np.ndarray, rt: np.ndarray, groups: np.ndarray | None,
        *, model_name: str, measure: str, word_ids: pd.DataFrame | None = None,
    ) -> tuple[list[dict], pd.DataFrame | None]:
        """Fit ridge CV for every layer of one subject's trials (serial path;
        see ``_fit_subject_layers``/``_fit_subject_layers_gpu`` for the shared
        implementations)."""
        rcfg = self.cfg.regression
        save_word_predictions = self.cfg.save_word_predictions
        if self.cfg.regression_device == "cuda":
            rows, oof_all = _fit_subject_layers_gpu(
                pid, emb_sub, rt, groups, model_name=model_name, measure=measure,
                n_splits=rcfg.n_splits, alpha=rcfg.alpha, scale=rcfg.scale,
                random_state=rcfg.random_state, n_layer_workers=self.cfg.n_layer_workers,
                save_word_predictions=save_word_predictions,
            )
            per_word: pd.DataFrame | None = None
            if oof_all is not None and word_ids is not None:
                per_word = word_ids.copy()
                per_word.attrs["oof_all_layers"] = oof_all  # (n_layers, n_trials)
                per_word.attrs["rt_true"] = rt
            return rows, per_word
        return _fit_subject_layers(
            pid, emb_sub, rt, groups, model_name=model_name, measure=measure,
            n_splits=rcfg.n_splits, alpha=rcfg.alpha, scale=rcfg.scale,
            random_state=rcfg.random_state, print_chosen_alpha=rcfg.print_chosen_ridge_alpha,
            thread_n_jobs=self.cfg.n_jobs,
            save_word_predictions=save_word_predictions,
            word_ids=word_ids,
        )

    def _word_predictions_root(self) -> Path:
        """Root dir for per-word/per-item OOF prediction .npz files --
        ``word_predictions_dir`` when set (e.g. a $SCRATCH path, to keep
        these many small files off the $WORK quota the CSV summaries live
        on), else falls back to ``output_dir``."""
        if self.cfg.word_predictions_dir:
            return Path(self.cfg.word_predictions_dir)
        return word_predictions_root(self.cfg.output_dir)

    def _save_word_predictions(self, pid: str, per_word: pd.DataFrame) -> None:
        """Save one subject's observed vs. out-of-fold-predicted RT to a
        per-word ``.npz``. CPU path: last layer only (a ``rt_true``/
        ``rt_pred_oof`` pair of columns, mirrors ``zuco_reading_embed.py``).
        GPU path: every layer (an ``(n_layers, n_trials)`` array stashed in
        ``per_word.attrs``, mirrors ``natural_stories_embed_gpu.py``)."""
        if not (self.cfg.word_predictions_dir or self.cfg.output_dir):
            logger.warning(
                "save_word_predictions=True but output_dir is not set — skipping word-prediction save for %s", pid,
            )
            return
        out_dir = self._word_predictions_root() / "provo_embed_word_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_revision = (self.cfg.model.model_revision or "main").replace("/", "_")
        safe_model = (self.cfg.canonical_model_name or self.cfg.model.model_name).replace("/", "_")
        out_path = out_dir / f"{safe_model}_{model_revision}_{pid}.npz"

        if "oof_all_layers" in per_word.attrs:
            oof_all = per_word.attrs["oof_all_layers"]  # (n_layers, n_trials)
            rt_true = per_word.attrs["rt_true"]
            np.savez_compressed(
                out_path,
                oof_predictions=oof_all,
                rt_true=rt_true.astype(np.float32),
                story_id=per_word["story_id"].to_numpy() if "story_id" in per_word.columns else np.array([]),
                zone=per_word["zone"].to_numpy() if "zone" in per_word.columns else np.array([]),
                words=per_word["word"].astype(str).to_numpy() if "word" in per_word.columns else np.array([]),
                model=np.array(self.cfg.model.model_name),
                revision=np.array(self.cfg.model.model_revision or "main"),
            )
            logger.info("  Saved word-level predictions (%s) -> %s", oof_all.shape, out_path)
        else:
            np.savez_compressed(
                out_path,
                rt_true=per_word["rt_true"].to_numpy(dtype=np.float32),
                rt_pred_oof=per_word["rt_pred_oof"].to_numpy(dtype=np.float32),
                story_id=per_word["story_id"].to_numpy(),
                zone=per_word["zone"].to_numpy(),
                words=per_word["word"].astype(str).to_numpy() if "word" in per_word.columns else np.array([]),
                layer=per_word["layer"].iloc[0] if len(per_word) else -1,
                model=np.array(self.cfg.model.model_name),
                revision=np.array(self.cfg.model.model_revision or "main"),
            )
            logger.info("  Saved word-level predictions (%d rows) -> %s", len(per_word), out_path)

    def _log_subject_result(self, subj_rows: list[dict], *, model_name: str, measure: str, n_layers: int) -> None:
        pid = subj_rows[0]["participant_id"]
        for row in subj_rows:
            logger.info(
                "[%s][%s][subject=%s] layer %d/%d: pearson=%.4f spearman=%.4f alpha=%.4g",
                model_name, measure, pid, row["layer"], n_layers - 1,
                row["pearsonr"], row["spearmanr"], row["ridge_alpha"],
            )
        logger.info(
            "[%s][%s][subject=%s] mean correlation across layers: pearson=%.4f spearman=%.4f",
            model_name, measure, pid,
            float(np.nanmean([r["pearsonr"] for r in subj_rows])),
            float(np.nanmean([r["spearmanr"] for r in subj_rows])),
        )

    def _iter_subject_data(
        self,
        trial_df: pd.DataFrame,
        emb_lookup: dict[tuple[int, int], np.ndarray],
        *,
        model_name: str,
        measure: str,
        done_subjects: set,
    ) -> tp.Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame | None]]:
        """Yield one not-yet-done subject's (pid, embeddings, rt, CV groups,
        word identity) at a time. Lazy on purpose: materializing every
        subject's ``emb_sub`` up front (a full list instead of a generator)
        held all of them in memory simultaneously regardless of
        ``subject_n_jobs`` — e.g. 117 UCL subjects at ~1.4GB each for
        Qwen3-8B is ~170GB, which OOM'd even in serial mode. The caller
        controls how many of these are alive at once (1 for serial,
        ``subject_n_jobs`` for the bounded process-pool window)."""
        save_word_predictions = self.cfg.save_word_predictions
        word_id_cols = [c for c in ("story_id", "zone", "word") if c in trial_df.columns]
        for pid in sorted(trial_df["participant_id"].unique()):
            if pid in done_subjects:
                logger.info("[%s][%s][subject=%s] already computed — skipping", model_name, measure, pid)
                continue
            sub_df = trial_df.loc[trial_df["participant_id"] == pid]
            if len(sub_df) < self.cfg.min_trials_per_subject:
                logger.warning(
                    "[%s][%s][subject=%s] only %d trials (< min_trials_per_subject=%d) — skipping",
                    model_name, measure, pid, len(sub_df), self.cfg.min_trials_per_subject,
                )
                continue
            rt = sub_df["rt"].to_numpy(dtype=float)
            emb_sub = np.stack(
                [emb_lookup[(int(r.story_id), int(r.zone))] for r in sub_df.itertuples()], axis=0,
            )
            groups = sub_df["story_id"].to_numpy() if self.cfg.group_cv_by_story else None
            word_ids = sub_df[word_id_cols].reset_index(drop=True) if save_word_predictions else None
            yield pid, emb_sub, rt, groups, word_ids

    def _regress(
        self,
        trial_df: pd.DataFrame,
        emb_lookup: dict[tuple[int, int], np.ndarray],
        *,
        existing_subject_rows: list[dict] | None = None,
        on_subject_done: tp.Callable[[list[dict]], None] | None = None,
    ) -> ProvoEmbedBenchmarkResult:
        rcfg = self.cfg.regression
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name
        measure = self.cfg.data.measure
        n_layers = next(iter(emb_lookup.values())).shape[0]

        use_gpu_ridge = self.cfg.regression_device == "cuda"
        if use_gpu_ridge:
            if not self.cfg.group_cv_by_story:
                raise ValueError(
                    "regression_device='cuda' requires group_cv_by_story=True "
                    "(the GPU ridge path has no ungrouped-KFold fallback)."
                )
            if self.cfg.subject_n_jobs > 1:
                logger.warning(
                    "regression_device='cuda': ignoring subject_n_jobs=%d — subjects are "
                    "processed serially on GPU (per-layer CUDA streams give the "
                    "parallelism instead; see _fit_subject_layers_gpu docstring).",
                    self.cfg.subject_n_jobs,
                )
            from himalaya.backend import set_backend as _set_backend
            _set_backend("torch_cuda")
            # One-time, non-thread-safe CUDA/MAGMA backend registration on
            # torch.linalg.svd's first call — force it here, single-threaded,
            # before the per-layer thread pool inside any subject fit (same
            # fix as natural_stories_embed_gpu.py / podcast.py).
            import torch
            _svd_warmup = torch.eye(2, device=f"cuda:{torch.cuda.current_device()}")
            torch.linalg.svd(_svd_warmup, full_matrices=False)
            del _svd_warmup

        per_subject_rows: list[dict] = list(existing_subject_rows or [])
        done_subjects = {r["participant_id"] for r in per_subject_rows}

        data_iter = self._iter_subject_data(
            trial_df, emb_lookup, model_name=model_name, measure=measure, done_subjects=done_subjects,
        )

        save_word_predictions = self.cfg.save_word_predictions

        if self.cfg.average_only:
            # Averaged-RT-only mode: skip every per-subject fit. The averaged
            # branch below runs off raw per-word RT rows, so nothing here is
            # needed. Per-subject outputs come back empty/NaN.
            n_items_avg = None
            average_layerwise = None
            average_summary_df = None
            if self.cfg.average_across_subjects:
                n_items_avg, average_layerwise, average_summary_df = self._fit_average(
                    trial_df, emb_lookup, model_name=model_name, measure=measure, n_layers=n_layers,
                )

            return ProvoEmbedBenchmarkResult(
                model_name=model_name,
                measure=measure,
                n_trials=len(trial_df),
                n_subjects=0,
                n_layers=n_layers,
                layerwise=LayerwiseRidgeResults(
                    model_name=model_name, layer_scores=[], layer_indices=[],
                ),
                flat_score=CVScore(
                    pearsonr=float("nan"), spearmanr=float("nan"),
                    r2=float("nan"), mse=float("nan"), ridge_alpha=float("nan"),
                ),
                summary_df=pd.DataFrame(),
                per_subject_df=pd.DataFrame(),
                n_items_avg=n_items_avg,
                average_layerwise=average_layerwise,
                average_summary_df=average_summary_df,
            )

        if use_gpu_ridge or self.cfg.subject_n_jobs <= 1:
            for pid, emb_sub, rt, groups, word_ids in data_iter:
                subj_rows, per_word = self._fit_one_subject(
                    pid, emb_sub, rt, groups, model_name=model_name, measure=measure, word_ids=word_ids,
                )
                self._log_subject_result(subj_rows, model_name=model_name, measure=measure, n_layers=n_layers)
                per_subject_rows.extend(subj_rows)
                if per_word is not None:
                    self._save_word_predictions(pid, per_word)
                if on_subject_done is not None:
                    on_subject_done(subj_rows)
        else:
            with ProcessPoolExecutor(max_workers=self.cfg.subject_n_jobs) as executor:
                def _submit(pid, emb_sub, rt, groups, word_ids):
                    return executor.submit(
                        _fit_subject_layers, pid, emb_sub, rt, groups,
                        model_name=model_name, measure=measure,
                        n_splits=rcfg.n_splits, alpha=rcfg.alpha, scale=rcfg.scale,
                        random_state=rcfg.random_state, print_chosen_alpha=rcfg.print_chosen_ridge_alpha,
                        thread_n_jobs=self.cfg.n_jobs,
                        save_word_predictions=save_word_predictions, word_ids=word_ids,
                    )

                def _collect(future, pid) -> None:
                    subj_rows, per_word = future.result()
                    self._log_subject_result(subj_rows, model_name=model_name, measure=measure, n_layers=n_layers)
                    per_subject_rows.extend(subj_rows)
                    if per_word is not None:
                        self._save_word_predictions(pid, per_word)
                    if on_subject_done is not None:
                        on_subject_done(subj_rows)

                # Bounded sliding window: never more than subject_n_jobs subjects'
                # emb_sub arrays alive (in-flight) at once, unlike submitting
                # everything up front (see _iter_subject_data docstring).
                in_flight: dict = {}
                for pid, emb_sub, rt, groups, word_ids in data_iter:
                    in_flight[_submit(pid, emb_sub, rt, groups, word_ids)] = pid
                    if len(in_flight) >= self.cfg.subject_n_jobs:
                        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                        for future in done:
                            fpid = in_flight.pop(future)
                            _collect(future, fpid)
                for future in list(in_flight):
                    fpid = in_flight.pop(future)
                    _collect(future, fpid)

        n_subjects = len({r["participant_id"] for r in per_subject_rows})
        if n_subjects == 0:
            raise ValueError(
                f"No subject had >= min_trials_per_subject={self.cfg.min_trials_per_subject} "
                f"trials (measure={measure})"
            )

        scores_by_layer: list[list[dict]] = [[] for _ in range(n_layers)]
        for row in per_subject_rows:
            scores_by_layer[row["layer"]].append(row)

        layer_scores = [_mean_cv_score(scores_by_layer[li]) for li in range(n_layers)]
        layerwise = LayerwiseRidgeResults(
            model_name=model_name, layer_scores=layer_scores, layer_indices=list(range(n_layers)),
        )
        best = layerwise.best_layer_score()
        logger.info(
            "[%s][%s] mean-across-subjects best layer Spearman r = %.4f (layer %d, n_subjects=%d)",
            model_name, measure, best.spearmanr, int(np.nanargmax(layerwise.scores_array())), n_subjects,
        )

        summary_rows = [
            dict(
                model_name=model_name, measure=measure, layer=li,
                layer_rel=li / max(1, n_layers - 1), n_layers=n_layers, n_subjects=n_subjects,
                **s.to_dict(),
            )
            for li, s in zip(layerwise.layer_indices, layerwise.layer_scores)
        ]
        summary_df = pd.DataFrame(summary_rows)
        per_subject_df = pd.DataFrame(per_subject_rows)

        n_items_avg = None
        average_layerwise = None
        average_summary_df = None
        if self.cfg.average_across_subjects:
            n_items_avg, average_layerwise, average_summary_df = self._fit_average(
                trial_df, emb_lookup, model_name=model_name, measure=measure, n_layers=n_layers,
            )

        return ProvoEmbedBenchmarkResult(
            model_name=model_name,
            measure=measure,
            n_trials=len(trial_df),
            n_subjects=n_subjects,
            n_layers=n_layers,
            layerwise=layerwise,
            flat_score=layer_scores[-1],
            summary_df=summary_df,
            per_subject_df=per_subject_df,
            n_items_avg=n_items_avg,
            average_layerwise=average_layerwise,
            average_summary_df=average_summary_df,
        )

    def _fit_average(
        self,
        trial_df: pd.DataFrame,
        emb_lookup: dict[tuple[int, int], np.ndarray],
        *,
        model_name: str,
        measure: str,
        n_layers: int,
    ) -> tuple[int, LayerwiseRidgeResults, pd.DataFrame]:
        """Fit one ridge CV per layer on RT averaged across subjects per item
        (available-case mean — see ``neuroai.benchmarks.subject_average``),
        the "grand-average" branch alongside the per-subject fits above."""
        rcfg = self.cfg.regression
        item_df = average_items_across_subjects(
            trial_df, item_key_cols=("story_id", "zone"),
            min_subjects_per_item=self.cfg.min_subjects_per_item,
        )
        if self.cfg.average_ridge_device == "cuda":
            import torch
            from himalaya.backend import set_backend as _set_backend
            _set_backend("torch_cuda")
            _svd_warmup = torch.eye(2, device=f"cuda:{torch.cuda.current_device()}")
            torch.linalg.svd(_svd_warmup, full_matrices=False)
            del _svd_warmup
            rows, per_item = fit_averaged_layers_gpu(
                item_df, emb_lookup,
                item_key_cols=("story_id", "zone"),
                group_col="story_id" if self.cfg.group_cv_by_story else None,
                n_splits=rcfg.n_splits, alpha=rcfg.alpha, scale=rcfg.scale,
                random_state=rcfg.random_state,
                extra_fields=dict(model_name=model_name, measure=measure),
                return_oof_for_layer=(n_layers - 1) if self.cfg.save_word_predictions else None,
                n_layer_workers=self.cfg.n_jobs,
            )
        else:
            rows, per_item = fit_averaged_layers(
                item_df, emb_lookup,
                item_key_cols=("story_id", "zone"),
                group_col="story_id" if self.cfg.group_cv_by_story else None,
                n_splits=rcfg.n_splits, alpha=rcfg.alpha, scale=rcfg.scale,
                random_state=rcfg.random_state, print_chosen_alpha=rcfg.print_chosen_ridge_alpha,
                thread_n_jobs=self.cfg.n_jobs,
                extra_fields=dict(model_name=model_name, measure=measure),
                return_oof_for_layer=(n_layers - 1) if self.cfg.save_word_predictions else None,
            )
        if per_item is not None:
            self._save_average_predictions(per_item)

        layer_scores = [
            CVScore(
                pearsonr=r["pearsonr"], spearmanr=r["spearmanr"],
                r2=r["r2"], mse=r["mse"], ridge_alpha=r["ridge_alpha"],
            )
            for r in sorted(rows, key=lambda r: r["layer"])
        ]
        average_layerwise = LayerwiseRidgeResults(
            model_name=model_name, layer_scores=layer_scores, layer_indices=list(range(n_layers)),
        )
        best = average_layerwise.best_layer_score()
        logger.info(
            "[%s][%s] averaged-RT branch best layer Spearman r = %.4f (layer %d, n_items=%d)",
            model_name, measure, best.spearmanr, int(np.nanargmax(average_layerwise.scores_array())), len(item_df),
        )
        average_summary_df = pd.DataFrame(sorted(rows, key=lambda r: r["layer"]))
        return len(item_df), average_layerwise, average_summary_df

    def _save_average_predictions(self, per_item: pd.DataFrame) -> None:
        """Save observed vs. out-of-fold-predicted averaged-RT (last layer)
        for the ``average_across_subjects`` branch, mirroring
        ``_save_word_predictions`` at the item level."""
        if not (self.cfg.word_predictions_dir or self.cfg.output_dir):
            logger.warning(
                "average_across_subjects=True but output_dir is not set — skipping average-prediction save"
            )
            return
        out_dir = self._word_predictions_root() / "provo_embed_average_word_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_revision = (self.cfg.model.model_revision or "main").replace("/", "_")
        safe_model = (self.cfg.canonical_model_name or self.cfg.model.model_name).replace("/", "_")
        out_path = out_dir / f"{safe_model}_{model_revision}.npz"
        np.savez_compressed(
            out_path,
            rt_true_avg=per_item["rt_true_avg"].to_numpy(dtype=np.float32),
            rt_pred_oof=per_item["rt_pred_oof"].to_numpy(dtype=np.float32),
            n_subjects_avg=per_item["n_subjects_avg"].to_numpy(),
            story_id=per_item["story_id"].to_numpy(),
            zone=per_item["zone"].to_numpy(),
            layer=per_item["layer"].iloc[0] if len(per_item) else -1,
            model=np.array(self.cfg.model.model_name),
            revision=np.array(self.cfg.model.model_revision or "main"),
        )
        logger.info("  Saved averaged-RT predictions (%d items) -> %s", len(per_item), out_path)
