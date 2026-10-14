"""OneStop Corpus eye-tracking benchmark: ridge regression on LLM layer activations.

Predicts word-level reading time (OneStop FP/GP) from per-layer contextual
embeddings, mirroring :mod:`neuroai.benchmarks.provo_embed` (same pattern,
ridge instead of LME) — applied to OneStop instead of Provo. Complements
:mod:`neuroai.benchmarks.onestop_lme` (scalar surprisal -> RT via mixed-effects
model): same dataset, same word set, orthogonal predictor.

**Averaged-subject-only by default** (de Varda, Berzak, Fedorenko & Levy 2026):
the shipped configs / SLURM jobs set ``average_only: true``, so RT is averaged
across subjects per (story_id, zone) word — one "grand-average subject" — and a
single ridge CV is fit per layer (see ``neuroai.benchmarks.subject_average``).
The full per-subject machinery below is retained behind ``average_only=False`` as
an escape hatch; the rest of this docstring describes that path.

Fit independently **per subject** — each subject's own raw per-word RT trials
get their own ridge CV (same embedding per word is reused across the subjects
who read it; RT is never averaged across subjects). "Subject" plays the role
of "electrode" in ``NeuralBenchmark``. Per-subject scores are logged and saved
individually (``*_by_subject.csv``) as each subject finishes; the headline
per-layer score is the mean across subjects (``*_checkpoint_scores.csv``),
analogous to averaging across electrodes for a neural model score.

Ridge fit runs on **CPU** (sklearn ``cv_ridge_score``), not GPU. A GPU-ridge
variant (himalaya, per-layer CUDA streams) was tried and reverted for two
reasons: (1) with only 1-2 GPUs available, subjects can only be fit 1-2 at a
time (GPU-serialized) — CPU's ``ProcessPoolExecutor`` subject-level fan-out
(below) has no such ceiling, it scales with however many cores the SLURM
allocation grants; (2) the GPU path also materializes every layer's embedding
for every word in host memory before regression starts, which OOM'd host RAM
for large-hidden-dim models (e.g. opt-66b: 65 layers x 9216-dim x ~50k words
in float32 is 100GB+) — a problem orthogonal to which engine does the ridge
fit, but one the CPU path's smaller default concurrency happens to avoid
tripping as often.

(Podcast-style subject-batching — stacking every subject as an extra target
column of one shared ridge fit per layer — was tried and reverted too: unlike
podcast's channels/lags, which are always fully populated for every trial
row, OneStop subjects have very different valid (story_id, zone) trial sets
after their own RT-based filtering. The intersection of all 14 subjects'
valid trials collapsed to ~288 words, vs. ~19,762 words/subject on average —
each subject's filtering drops a different slice, and the losses compound
multiplicatively across subjects. NaN-tolerant per-column masking throughout
the fit/scoring math was considered but rejected as too invasive and likely
to erase the speed benefit of batching in the first place.)

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
  ``subject_n_jobs`` subjects' embedding arrays in memory at once — subjects
  are yielded one at a time from a generator rather than materializing the
  whole dataset up front), each worker still using ``n_jobs`` threads
  internally for its own layers. Subjects are fully independent (no shared
  state), so this scales close to linearly with ``subject_n_jobs``. Memory
  cost is real: each concurrent worker holds its own full
  ``(n_trials, n_layers, hidden)`` copy, so set ``subject_n_jobs`` (and
  ``n_jobs``, so ``subject_n_jobs * n_jobs`` stays within your node's core
  count) based on the model size and RAM you're willing to spend — e.g. low
  (or 1) for a 60B+-param model's huge per-layer hidden size, higher for a
  small model. There is no auto-sizing; this is a manual per-family/config
  knob, same as ``n_jobs`` already is.

Either way, each subject's rows are handed to an optional ``on_subject_done``
callback immediately after that subject finishes (see ``run()``; the process-
pool path fires this as each in-flight subject completes, not only after the
whole pool drains), letting the caller persist them right away — a crash
mid-checkpoint loses at most the in-flight subject(s), not every subject fit
so far. Pass ``existing_subject_rows`` to skip subjects a previous (crashed)
run already completed for this exact checkpoint.

Unlike Provo, OneStop's materials table (:func:`neuroai.datasets.onestop.onestop_word_list`)
already enumerates every word of every trial (no reconstruction-from-raw-text
step needed — see :mod:`neuroai.datasets.onestop` module docstring), so the word
list used for embeddings is exactly the word list used for RT, with no
extra bookkeeping.

Pipeline
--------
1. ``OnestopDataset`` (aggregate=False) -> one row per (participant_id, story_id,
   zone) trial with raw ``rt`` and ``is_edge``.
2. For each of OneStop's 330 passages, run the model once over the
   full token sequence via
   :func:`neuroai.features.llm.contextual_word_embeddings_hf` with
   ``layer_index="all"`` -> ``(n_tokens, n_layers, hidden)`` at the last-token
   position of each word. Cached to disk per (model, revision, section_id); the
   cache key does not include the OneStop measure, so FP/GP sweeps sharing one
   model/checkpoint reuse the same embeddings (mirrors the surprisal cache's
   measure-agnostic sharing — see ``onestop_lme.py`` module docstring). Computed
   once and shared across all subjects (embeddings don't depend on the subject).
3. Per subject: ridge regression (nested CV) per layer: ``rt ~ embedding``.
   ``GroupKFold`` on ``story_id`` (leave-passages-out) is the default within
   each subject's own trials — words within one passage share overlapping
   context, so plain shuffled ``KFold`` would let correlated neighbours leak
   across the train/test split.
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

from ..datasets.onestop import OnestopDataset, onestop_word_list
from ..regressors.ridge import CVScore, LayerwiseRidgeResults, cv_ridge_score
from .base import ModelConfig, RegressionConfig, word_predictions_root, word_predictions_dir
from .subject_average import (
    average_items_across_subjects,
    fit_averaged_layers,
    fit_averaged_layers_gpu,
    himalaya_gpu_warmup,
)

logger = logging.getLogger(__name__)


def _fit_subject_layers(
    pid: str, emb_sub: np.ndarray, rt: np.ndarray, groups: np.ndarray | None,
    *,
    model_name: str, measure: str,
    n_splits: int, alpha, scale: bool, random_state: int, print_chosen_alpha: bool,
    thread_n_jobs: int,
    save_word_predictions: bool = False,
    word_ids: pd.DataFrame | None = None,
    layer_offset: int = 0,
    total_n_layers: int | None = None,
) -> tuple[list[dict], pd.DataFrame | None]:
    """Fit ridge CV for every layer of one subject's trials.

    Module-level (not a method) so it is picklable and can run either inline
    (serial path) or as a ``ProcessPoolExecutor`` task (bounded subject-level
    parallelism, see module docstring). Layers within the call are themselves
    parallelized via a thread pool (BLAS releases the GIL during the fit).

    ``emb_sub`` may hold only a **chunk** of layers (see
    ``OnestopEmbedBenchmarkConfig.layer_chunk_size`` / module docstring) —
    ``layer_offset`` is added to the chunk-local layer index ``li`` so the
    ``layer`` field in every returned row is the true global layer number,
    and ``total_n_layers`` (defaulting to ``emb_sub.shape[1]`` when the
    caller isn't chunking) is what gets reported in the ``n_layers`` /
    ``layer_rel`` fields instead of the chunk's own (smaller) layer count.

    When ``save_word_predictions``, the LAST layer **of this chunk** is
    additionally refit with ``return_oof_predictions=True`` (a per-target
    best-layer refit isn't worth the extra regression calls here, mirroring
    ``zuco_reading_embed.py``'s convention) and a per-word DataFrame of
    observed vs. out-of-fold-predicted RT is returned alongside the
    per-layer score rows — the caller only passes ``save_word_predictions=
    True`` for the chunk that contains the true final layer (see
    ``OnestopEmbedBenchmark._regress``), so "last layer of this chunk" and
    "true last layer" coincide in practice.
    """
    n_layers_chunk = emb_sub.shape[1]
    n_layers = total_n_layers if total_n_layers is not None else n_layers_chunk

    def _fit_layer(li: int) -> dict:
        s = cv_ridge_score(
            emb_sub[:, li, :], rt,
            n_splits=n_splits, alpha=alpha, scale=scale,
            random_state=random_state, groups=groups,
            print_chosen_alpha=print_chosen_alpha,
        )
        global_li = li + layer_offset
        return dict(
            model_name=model_name, measure=measure, participant_id=pid,
            layer=global_li, layer_rel=global_li / max(1, n_layers - 1), n_layers=n_layers,
            n_trials=len(rt), **s.to_dict(),
        )

    if thread_n_jobs > 1:
        rows = Parallel(n_jobs=thread_n_jobs, prefer="threads")(
            delayed(_fit_layer)(li) for li in range(n_layers_chunk)
        )
    else:
        rows = [_fit_layer(li) for li in range(n_layers_chunk)]
    rows.sort(key=lambda r: r["layer"])

    per_word: pd.DataFrame | None = None
    if save_word_predictions:
        last = n_layers_chunk - 1
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
        per_word["layer"] = last + layer_offset
        per_word["n_layers"] = n_layers
        per_word["rt_true"] = rt[valid_mask]
        per_word["rt_pred_oof"] = oof

    return rows, per_word


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
class OnestopEmbedDataConfig:
    """OneStop Corpus dataset location, measure, and embedding-extraction options."""

    root: str
    measure: str = "FP"  # "SP" | "FP" | "GP"
    stories: list[int] | None = None  # encoded story_id subset; None = all 330 passages
    min_dur: float = 50.0
    max_dur: float = 5000.0
    exclude_doc_edges: bool = True
    # >= longest OneStop passage word count so every word's context is the full
    # cumulative passage prefix, matching the cumulative context used for
    # surprisal in onestop_lme.py. Only consulted on the slow-tokenizer fallback.
    context_words: int = 512
    # OneStop passages are ~2 paragraphs (~125 words mean); 1024 keeps every
    # passage in a single fast-tokenizer forward chunk (fulltext still
    # sliding-windows anything longer).
    max_length: int | None = 1024
    feature_cache_dir: str | None = None
    # Fixed subject subset (the frozen 120-of-360 list). ``subjects`` is an
    # explicit id list; ``subjects_file`` points at a newline-delimited file
    # (``#`` comments / blank lines stripped). ``subjects`` wins if both set.
    subjects: list[str] | None = None
    subjects_file: str | None = None


@dataclass
class OnestopEmbedBenchmarkConfig:
    data: OnestopEmbedDataConfig
    model: ModelConfig
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    # Drop sentence- and screen-edge words (same convention as onestop_lme's
    # filter_sentence_edges).
    filter_sentence_edges: bool = True
    # Leave-passages-out CV (GroupKFold on story_id) within each subject's own
    # trials. False -> plain shuffled KFold.
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
    # Grouping label for saved CSVs when the HF load repo differs from the
    # canonical model id (e.g. OLMo early-training branches).
    canonical_model_name: str | None = None
    # Save each held-out trial's observed RT + out-of-fold-predicted RT (LAST
    # layer only, per subject — mirrors zuco_reading_embed.py's convention) to
    # a per-word CSV, for post-hoc per-word difficulty splits (e.g. DLT). On
    # by default.
    save_word_predictions: bool = True
    output_dir: str | None = None
    # Directory for per-word/per-item OOF prediction .npz files
    # (save_word_predictions=True) specifically. Defaults to output_dir
    # when unset -- set separately (e.g. to a $SCRATCH path) to keep the
    # many small prediction files off the (smaller, shared) $WORK quota
    # the CSV summaries live on.
    word_predictions_dir: str | None = None
    # Fit this many layers at a time instead of holding every layer's
    # embedding for every word of the whole corpus in memory at once (see
    # module docstring — opt-66b's 65 layers x 9216-dim x ~50k words in
    # float32 is 100GB+, which OOM'd host RAM even at cpus-per-task=48).
    # Each chunk re-reads the already-cached per-passage embeddings (no
    # repeated model forward passes) but only keeps the requested layer
    # slice across the whole corpus, so peak memory scales with the chunk
    # size, not the model's total layer count. None = no chunking (one
    # chunk covering every layer, the original behavior).
    layer_chunk_size: int | None = 16
    # Additionally fit a single ridge CV per layer on RT averaged across
    # subjects per (story_id, zone) item (available-case mean — see
    # neuroai.benchmarks.subject_average module docstring), alongside the
    # per-subject fits above. Does not affect the per-subject outputs; the
    # result is exposed as OnestopEmbedBenchmarkResult.average_summary_df /
    # .average_layerwise / .n_items_avg (all None when disabled).
    average_across_subjects: bool = False
    # Drop items with fewer than this many contributing subjects from the
    # averaged-RT target (1 = available-case, keep every item any subject has).
    min_subjects_per_item: int = 1
    # Skip the per-subject ridge fits entirely and only compute the
    # averaged-RT branch. The two are independent computations (averaging
    # works directly off raw per-word RT rows, not off any per-subject fit
    # result) — for models where per-subject fitting is too expensive to
    # finish in a reasonable job (e.g. opt-66b: 14 subjects x 65 layers of
    # nested ridge CV over ~54k words each), this gets the cheap 1-fit-per-
    # layer averaged result without paying for the 14x per-subject cost.
    # Requires average_across_subjects=True; per-subject outputs
    # (n_subjects, layerwise, summary_df, per_subject_df) are empty/zeroed
    # when this is set.
    average_only: bool = False
    # Ridge fit device for the averaged-RT branch specifically ("cpu" sklearn
    # nested-CV grid search, or "cuda" himalaya SVD-based GCV). The
    # per-subject path stays CPU-only regardless (see module docstring for
    # why GPU-ridge was reverted there — 14-way subject/GPU contention), but
    # the averaged branch is always exactly ONE fit, so that concern doesn't
    # apply, and GPU is dramatically faster for OneStop's ~50k-word corpus x
    # thousands-of-dims embeddings. OneStop family SLURM jobs already reserve a
    # GPU (for host-RAM headroom, see job_onestop_jz_common.sh), so "cuda" here
    # is free — it uses hardware that would otherwise sit idle.
    average_ridge_device: str = "cuda"  # himalaya GPU ridge everywhere (2026-10-14)


@dataclass
class OnestopEmbedBenchmarkResult:
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


class OnestopEmbedBenchmark:
    """Predict OneStop reading times from LLM layer activations using ridge
    regression, fit independently per subject.

    Usage
    -----
    .. code-block:: python

        cfg = OnestopEmbedBenchmarkConfig(
            data=OnestopEmbedDataConfig(root="data/onestop", measure="FP"),
            model=ModelConfig(model_name="EleutherAI/pythia-160m"),
        )
        result = OnestopEmbedBenchmark(cfg).run()
        print(result.layerwise.best_layer_score().spearmanr)  # mean across subjects
    """

    def __init__(self, cfg: OnestopEmbedBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        *,
        existing_subject_rows: list[dict] | None = None,
        on_subject_done: tp.Callable[[list[dict]], None] | None = None,
        on_average_chunk_done: tp.Callable[[list[dict]], None] | None = None,
    ) -> OnestopEmbedBenchmarkResult:
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
            loses at most the in-progress subject. With chunking enabled,
            this fires once per (subject, chunk) instead of once per
            subject — each call still carries only that chunk's rows, which
            the sweep script's incremental-CSV-append usage already handles
            fine (just more, smaller writes).
        on_average_chunk_done
            Averaged-RT branch (``average_across_subjects``/``average_only``)
            analogue of ``on_subject_done``: called with one layer-chunk's
            row dicts immediately after that chunk's ridge fit finishes,
            before starting the next chunk's (expensive) feature extraction.
            Without this, the averaged branch only writes anything after
            every chunk has fit (see ``_finalize_average``, called once after
            the whole ``run()`` loop) — a timeout on a large/slow-extracting
            model loses every already-computed layer's score, even though
            they were fit successfully. Use this to persist each chunk's
            rows as they complete.
        """
        trial_df = self._load_trials()
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name
        measure = self.cfg.data.measure
        n_layers = self._n_layers(trial_df)
        chunk_size = self.cfg.layer_chunk_size or n_layers

        avg_item_df: pd.DataFrame | None = None
        average_rows: list[dict] = []
        if self.cfg.average_across_subjects:
            avg_item_df = average_items_across_subjects(
                trial_df, item_key_cols=("story_id", "zone"),
                min_subjects_per_item=self.cfg.min_subjects_per_item,
            )
            # CV groups = ARTICLE (story_id // 100 drops paragraph and difficulty digits):
            # grouping by story_id let the Adv/Ele versions of one paragraph, which share
            # most of their text, fall on both sides of a fold (2026-10-12 audit).
            avg_item_df["article_group"] = avg_item_df["story_id"].astype(int) // 100
            logger.info(
                "[%s][%s] averaged-RT branch: %d items (available-case mean across subjects)",
                model_name, measure, len(avg_item_df),
            )

        if avg_item_df is not None and self.cfg.average_ridge_device == "cuda":
            himalaya_gpu_warmup()

        per_subject_rows: list[dict] = list(existing_subject_rows or [])
        # A subject only counts as fully done if it already has rows for
        # every global layer — partial coverage (e.g. a crash mid-chunk-
        # sweep) is refit from scratch across all chunks for simplicity/
        # correctness rather than attempting fine-grained per-chunk resume.
        rows_by_subject: dict[str, set[int]] = {}
        for r in per_subject_rows:
            rows_by_subject.setdefault(r["participant_id"], set()).add(r["layer"])
        fully_done_subjects = {pid for pid, layers in rows_by_subject.items() if len(layers) >= n_layers}

        n_chunks = -(-n_layers // chunk_size)  # ceil division
        for chunk_i, lo in enumerate(range(0, n_layers, chunk_size)):
            hi = min(lo + chunk_size, n_layers)
            is_last_chunk = hi == n_layers
            logger.info(
                "[%s][%s] layer chunk %d/%d: [%d:%d) of %d total",
                model_name, measure, chunk_i + 1, n_chunks, lo, hi, n_layers,
            )
            emb_lookup_chunk = self._embed_words(trial_df, layer_lo=lo, layer_hi=hi)
            if not self.cfg.average_only:
                chunk_rows = self._regress_chunk(
                    trial_df, emb_lookup_chunk,
                    layer_offset=lo, total_n_layers=n_layers,
                    save_word_predictions=self.cfg.save_word_predictions and is_last_chunk,
                    done_subjects=fully_done_subjects,
                    on_subject_done=on_subject_done,
                )
                per_subject_rows.extend(chunk_rows)

            if avg_item_df is not None:
                rcfg = self.cfg.regression
                oof_layer = (hi - lo - 1) if (is_last_chunk and self.cfg.save_word_predictions) else None
                if self.cfg.average_ridge_device == "cuda":
                    avg_chunk_rows, avg_per_item = fit_averaged_layers_gpu(
                        avg_item_df, emb_lookup_chunk,
                        item_key_cols=("story_id", "zone"),
                        group_col="article_group" if self.cfg.group_cv_by_story else None,
                        n_splits=rcfg.n_splits, alpha=rcfg.alpha, scale=rcfg.scale,
                        random_state=rcfg.random_state,
                        layer_offset=lo, total_n_layers=n_layers,
                        extra_fields=dict(model_name=model_name, measure=measure),
                        return_oof_for_layer=oof_layer,
                        n_layer_workers=self.cfg.n_jobs,
                    )
                else:
                    avg_chunk_rows, avg_per_item = fit_averaged_layers(
                        avg_item_df, emb_lookup_chunk,
                        item_key_cols=("story_id", "zone"),
                        group_col="article_group" if self.cfg.group_cv_by_story else None,
                        n_splits=rcfg.n_splits, alpha=rcfg.alpha, scale=rcfg.scale,
                        random_state=rcfg.random_state, print_chosen_alpha=rcfg.print_chosen_ridge_alpha,
                        thread_n_jobs=self.cfg.n_jobs,
                        layer_offset=lo, total_n_layers=n_layers,
                        extra_fields=dict(model_name=model_name, measure=measure),
                        return_oof_for_layer=oof_layer,
                    )
                average_rows.extend(avg_chunk_rows)
                if on_average_chunk_done is not None:
                    on_average_chunk_done(avg_chunk_rows)
                if avg_per_item is not None:
                    self._save_average_predictions(avg_per_item)

            del emb_lookup_chunk  # free this chunk's corpus-wide accumulation before the next one loads

        if self.cfg.average_only:
            result = OnestopEmbedBenchmarkResult(
                model_name=model_name, measure=measure, n_trials=len(trial_df), n_subjects=0,
                n_layers=n_layers,
                layerwise=LayerwiseRidgeResults(model_name=model_name, layer_scores=[], layer_indices=[]),
                flat_score=CVScore(pearsonr=float("nan"), spearmanr=float("nan"), r2=float("nan"), mse=float("nan"), ridge_alpha=float("nan")),
                summary_df=pd.DataFrame(), per_subject_df=pd.DataFrame(),
            )
        else:
            result = self._finalize(trial_df, per_subject_rows, n_layers=n_layers, model_name=model_name, measure=measure)
        if avg_item_df is not None:
            result.n_items_avg, result.average_layerwise, result.average_summary_df = self._finalize_average(
                average_rows, n_layers=n_layers, model_name=model_name, measure=measure,
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_subjects(self) -> list[str] | None:
        """The fixed subject subset: ``subjects`` list wins, else parse
        ``subjects_file`` (newline-delimited, ``#`` comments / blank lines
        stripped). Mirrors ``OnestopLmeBenchmark._resolve_subjects``."""
        dcfg = self.cfg.data
        if dcfg.subjects:
            return list(dcfg.subjects)
        if dcfg.subjects_file:
            path = Path(dcfg.subjects_file)
            if not path.is_file():
                raise FileNotFoundError(f"OneStop subjects_file not found: {path}")
            subs = [
                ln.strip()
                for ln in path.read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")
            ]
            if not subs:
                raise ValueError(f"OneStop subjects_file is empty: {path}")
            logger.info("OneStop embed: using fixed %d-subject subset from %s", len(subs), path)
            return subs
        return None

    def _load_trials(self) -> pd.DataFrame:
        dcfg = self.cfg.data
        ds = OnestopDataset(
            dcfg.root,
            dcfg.measure,
            stories=dcfg.stories,
            min_dur=dcfg.min_dur,
            max_dur=dcfg.max_dur,
            exclude_doc_edges=dcfg.exclude_doc_edges,
            aggregate=False,
            subjects=self._resolve_subjects(),
        )
        trial_df = ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)
        if self.cfg.filter_sentence_edges and "is_edge" in trial_df.columns:
            trial_df = trial_df.loc[~trial_df["is_edge"].astype(bool)].reset_index(drop=True)
        if trial_df.empty:
            raise ValueError(
                f"No OneStop trial observations left after filtering (measure={dcfg.measure})"
            )
        trial_df = trial_df.sort_values(["participant_id", "story_id", "zone"]).reset_index(drop=True)
        logger.info(
            "OneStop embed [%s]: %d trials, %d subjects, %d passages",
            dcfg.measure, len(trial_df), trial_df["participant_id"].nunique(),
            trial_df["story_id"].nunique(),
        )
        return trial_df

    def _n_layers(self, trial_df: pd.DataFrame) -> int:
        """Peek at one passage's cached embedding to learn the total layer
        count, without materializing the whole corpus. Used to plan the
        layer-chunk boundaries in ``_regress`` before any chunk is loaded."""
        first_story = int(trial_df["story_id"].iloc[0])
        probe = self._embed_words(trial_df, layer_lo=None, layer_hi=None, story_ids=[first_story])
        return next(iter(probe.values())).shape[0]

    def _embed_words(
        self,
        trial_df: pd.DataFrame,
        *,
        layer_lo: int | None = None,
        layer_hi: int | None = None,
        story_ids: list[int] | None = None,
    ) -> dict[tuple[int, int], np.ndarray]:
        """Return ``{(story_id, zone): (n_layers_or_chunk, hidden)}`` — one
        embedding per word, computed once and shared across every subject
        who read it.

        ``layer_lo``/``layer_hi`` slice to a layer **chunk** (see
        ``OnestopEmbedBenchmarkConfig.layer_chunk_size`` / module docstring):
        each passage's embedding still comes from one cached
        ``contextual_word_embeddings_hf_fulltext`` call (a cache hit after
        the first chunk — no repeated model forward passes), but only the
        requested layer slice is kept in ``emb_lookup``; the full per-passage
        array is discarded (garbage-collected) before moving to the next
        passage, so peak memory during the corpus-wide accumulation scales
        with the chunk size, not the full layer count. ``story_ids`` limits
        which passages are processed (used by ``_n_layers`` to probe just
        one)."""
        from ..features.llm import contextual_word_embeddings_hf, contextual_word_embeddings_hf_fulltext

        dcfg = self.cfg.data
        mcfg = self.cfg.model
        tokenizer_no_revision = bool(mcfg.model_revision) and "pythia" in mcfg.model_name.lower()
        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

        # onestop_word_list() already enumerates every word of every trial (unlike
        # Provo, no separate raw-text reconstruction is needed — see module
        # docstring), so it directly gives the ordered token sequence per story.
        full_words = onestop_word_list(dcfg.root, exclude_doc_edges=dcfg.exclude_doc_edges)
        if dcfg.stories is not None:
            full_words = full_words[full_words["story_id"].isin(dcfg.stories)]

        needed_stories = set(trial_df["story_id"].tolist()) if story_ids is None else set(story_ids)
        emb_lookup: dict[tuple[int, int], np.ndarray] = {}
        for story_id, group in full_words.groupby("story_id", sort=True):
            if story_id not in needed_stories:
                continue
            group = group.sort_values("zone")
            tokens = group["word"].astype(str).tolist()
            zones = group["zone"].tolist()
            # Every OneStop passage fits under max_length in one chunk, so this
            # is a single forward pass per passage instead of one per word (see
            # contextual_word_embeddings_hf_fulltext docstring) — falls back to
            # the slower per-word path only for slow (non-fast) tokenizers.
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
            # Slice to the requested layer chunk (default: keep everything,
            # same as before chunking existed) before storing — the full
            # per-passage `emb` array (every layer) goes out of scope right
            # after this loop iteration, so it's only ever the SLICE that
            # accumulates across all 330 passages in `emb_lookup`, not the
            # full per-layer array (see module docstring / class docstring
            # above for why this matters for opt-66b-sized hidden dims).
            emb_chunk = emb[:, layer_lo:layer_hi, :]
            for i, zone in enumerate(zones):
                emb_lookup[(int(story_id), int(zone))] = emb_chunk[i]

        logger.info(
            "OneStop embed: extracted embeddings for %d words across %d passages"
            "%s",
            len(emb_lookup), len(needed_stories),
            f" (layers [{layer_lo}:{layer_hi}])" if layer_lo is not None or layer_hi is not None else "",
        )
        return emb_lookup

    def _fit_one_subject(
        self, pid: str, emb_sub: np.ndarray, rt: np.ndarray, groups: np.ndarray | None,
        *, model_name: str, measure: str, word_ids: pd.DataFrame | None = None,
        layer_offset: int = 0, total_n_layers: int | None = None,
        save_word_predictions: bool | None = None,
    ) -> tuple[list[dict], pd.DataFrame | None]:
        """Fit ridge CV for every layer of one subject's trials (serial path;
        see ``_fit_subject_layers`` for the shared implementation).

        ``save_word_predictions`` overrides ``self.cfg.save_word_predictions``
        when given — used by chunked ``_regress`` to only request the OOF
        refit on the chunk containing the true final layer."""
        rcfg = self.cfg.regression
        swp = self.cfg.save_word_predictions if save_word_predictions is None else save_word_predictions
        return _fit_subject_layers(
            pid, emb_sub, rt, groups, model_name=model_name, measure=measure,
            n_splits=rcfg.n_splits, alpha=rcfg.alpha, scale=rcfg.scale,
            random_state=rcfg.random_state, print_chosen_alpha=rcfg.print_chosen_ridge_alpha,
            thread_n_jobs=self.cfg.n_jobs,
            save_word_predictions=swp,
            word_ids=word_ids,
            layer_offset=layer_offset, total_n_layers=total_n_layers,
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
        """Save one subject's observed vs. out-of-fold-predicted RT (last
        layer) to a per-word ``.npz``, for post-hoc per-word difficulty
        splits (e.g. DLT). Mirrors ``zuco_reading_embed.py``'s
        ``_save_word_predictions`` convention."""
        if not (self.cfg.word_predictions_dir or self.cfg.output_dir):
            logger.warning(
                "save_word_predictions=True but output_dir is not set — skipping word-prediction save for %s", pid,
            )
            return
        out_dir = self._word_predictions_root() / "onestop_embed_word_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_revision = (self.cfg.model.model_revision or "main").replace("/", "_")
        safe_model = (self.cfg.canonical_model_name or self.cfg.model.model_name).replace("/", "_")
        out_path = out_dir / f"{safe_model}_{model_revision}_{pid}.npz"
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

    def _save_average_predictions(self, per_item: pd.DataFrame) -> None:
        """Save observed vs. out-of-fold-predicted averaged-RT (last layer)
        for the ``average_across_subjects`` branch, mirroring
        ``_save_word_predictions`` at the item level."""
        if not (self.cfg.word_predictions_dir or self.cfg.output_dir):
            logger.warning(
                "average_across_subjects=True but output_dir is not set — skipping average-prediction save"
            )
            return
        out_dir = self._word_predictions_root() / "onestop_embed_average_word_predictions"
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

    def _finalize_average(
        self, average_rows: list[dict], *, n_layers: int, model_name: str, measure: str,
    ) -> tuple[int, LayerwiseRidgeResults, pd.DataFrame]:
        """Aggregate every chunk's averaged-RT rows into a per-layer result,
        analogous to ``_finalize`` for the per-subject branch (but there is
        only one "subject" here — the across-subject average — so no
        cross-subject mean is needed)."""
        rows_by_layer: dict[int, dict] = {r["layer"]: r for r in average_rows}
        n_items_avg = next(iter(rows_by_layer.values()))["n_items"] if rows_by_layer else 0
        layer_scores = [
            CVScore(
                pearsonr=rows_by_layer[li]["pearsonr"], spearmanr=rows_by_layer[li]["spearmanr"],
                r2=rows_by_layer[li]["r2"], mse=rows_by_layer[li]["mse"],
                ridge_alpha=rows_by_layer[li]["ridge_alpha"],
            )
            for li in range(n_layers)
        ]
        average_layerwise = LayerwiseRidgeResults(
            model_name=model_name, layer_scores=layer_scores, layer_indices=list(range(n_layers)),
        )
        best = average_layerwise.best_layer_score()
        logger.info(
            "[%s][%s] averaged-RT branch best layer Spearman r = %.4f (layer %d, n_items=%d)",
            model_name, measure, best.spearmanr, int(np.nanargmax(average_layerwise.scores_array())), n_items_avg,
        )
        average_summary_df = pd.DataFrame(sorted(average_rows, key=lambda r: r["layer"]))
        return n_items_avg, average_layerwise, average_summary_df

    def _log_subject_result(self, subj_rows: list[dict], *, model_name: str, measure: str, n_layers: int) -> None:
        pid = subj_rows[0]["participant_id"]
        logger.info(
            "[%s][%s][subject=%s] mean correlation across layers: pearson=%.4f spearman=%.4f",
            model_name, measure, pid,
            float(np.nanmean([r["pearsonr"] for r in subj_rows])),
            float(np.nanmean([r["spearmanr"] for r in subj_rows])),
        )

    def _regress_chunk(
        self,
        trial_df: pd.DataFrame,
        emb_lookup: dict[tuple[int, int], np.ndarray],
        *,
        layer_offset: int,
        total_n_layers: int,
        save_word_predictions: bool,
        done_subjects: set,
        on_subject_done: tp.Callable[[list[dict]], None] | None = None,
    ) -> list[dict]:
        """Fit every not-fully-done subject on **this layer chunk only**
        (``emb_lookup`` holds just ``[layer_offset, layer_offset + chunk_size)``
        — see ``run()``, which loops this once per chunk). Returns that
        chunk's row list; ``run()`` accumulates rows across chunks and does
        the final cross-layer aggregation once, in ``_finalize``."""
        rcfg = self.cfg.regression
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name
        measure = self.cfg.data.measure
        n_layers_chunk = next(iter(emb_lookup.values())).shape[0]

        chunk_rows: list[dict] = []
        word_id_cols = [c for c in ("story_id", "zone", "word") if c in trial_df.columns]

        def _iter_subject_data() -> tp.Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame | None]]:
            """Yield one not-fully-done subject's data at a time (lazy on
            purpose — materializing every subject's ``emb_sub`` up front would
            hold all of them in memory simultaneously for large models)."""
            for pid in sorted(trial_df["participant_id"].unique()):
                if pid in done_subjects:
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
                groups = (sub_df["story_id"].astype(int) // 100).to_numpy() if self.cfg.group_cv_by_story else None  # article
                word_ids = sub_df[word_id_cols].reset_index(drop=True) if save_word_predictions else None
                yield pid, emb_sub, rt, groups, word_ids

        data_iter = _iter_subject_data()

        if self.cfg.subject_n_jobs <= 1:
            for pid, emb_sub, rt, groups, word_ids in data_iter:
                subj_rows, per_word = self._fit_one_subject(
                    pid, emb_sub, rt, groups, model_name=model_name, measure=measure, word_ids=word_ids,
                    layer_offset=layer_offset, total_n_layers=total_n_layers,
                    save_word_predictions=save_word_predictions,
                )
                self._log_subject_result(subj_rows, model_name=model_name, measure=measure, n_layers=n_layers_chunk)
                chunk_rows.extend(subj_rows)
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
                        layer_offset=layer_offset, total_n_layers=total_n_layers,
                    )

                def _collect(future, pid) -> None:
                    subj_rows, per_word = future.result()
                    self._log_subject_result(subj_rows, model_name=model_name, measure=measure, n_layers=n_layers_chunk)
                    chunk_rows.extend(subj_rows)
                    if per_word is not None:
                        self._save_word_predictions(pid, per_word)
                    if on_subject_done is not None:
                        on_subject_done(subj_rows)

                # Bounded sliding window: never more than subject_n_jobs subjects'
                # emb_sub arrays alive (in-flight) at once.
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

        return chunk_rows

    def _finalize(
        self,
        trial_df: pd.DataFrame,
        per_subject_rows: list[dict],
        *,
        n_layers: int,
        model_name: str,
        measure: str,
    ) -> OnestopEmbedBenchmarkResult:
        """Aggregate every chunk's accumulated rows (one row per (subject,
        global layer)) into the final mean-across-subjects layerwise result —
        same aggregation as the pre-chunking ``_regress`` did in one pass."""
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

        return OnestopEmbedBenchmarkResult(
            model_name=model_name,
            measure=measure,
            n_trials=len(trial_df),
            n_subjects=n_subjects,
            n_layers=n_layers,
            layerwise=layerwise,
            flat_score=layer_scores[-1],
            summary_df=summary_df,
            per_subject_df=per_subject_df,
        )
