"""MECo Corpus (English subset) eye-tracking benchmark: ridge regression on LLM
layer activations.

Predicts word-level reading time (MECo SP/FP/GP) from per-layer contextual
embeddings, mirroring :mod:`neuroai.benchmarks.provo_embed` /
:mod:`neuroai.benchmarks.geco_embed` (same pattern, ridge instead of LME) —
applied to MECo instead of Provo/GECO. Complements
:mod:`neuroai.benchmarks.meco_lme` (scalar surprisal -> RT via mixed-effects
model): same dataset, same word set, orthogonal predictor.

Fit independently **per subject** — each subject's own raw per-word RT trials
get their own ridge CV (same embedding per word is reused across the subjects
who read it; RT is never averaged across subjects). "Subject" plays the role
of "electrode" in ``NeuralBenchmark``. Per-subject scores are logged and saved
individually (``*_by_subject.csv``) as each subject finishes; the headline
per-layer score is the mean across subjects (``*_checkpoint_scores.csv``),
analogous to averaging across electrodes for a neural model score.

(Podcast-style subject-batching — stacking every subject as an extra target
column of one shared ridge fit per layer — was tried and reverted for GECO
and doesn't apply here either: unlike podcast's channels/lags, which are
always fully populated for every trial row, MECo subjects have very different
valid (story_id, zone) trial sets after their own RT-based filtering — each
subject's filtering drops a different slice, and the losses compound
multiplicatively across subjects. See ``geco_embed.py`` module docstring for
the numbers that motivated this call.)

Ridge fit runs on GPU via himalaya (``cv_ridge_score_multioutput_gpu``, one
CUDA stream per layer — same pattern as ``geco_embed.py`` /
``natural_stories_embed_gpu.py`` / ``podcast.py``).
With only 1 GPU allocated, subjects fit serially — CUDA streams only
parallelize across LAYERS within one subject, not across subjects, so
MECo's ~93 subjects would bottleneck badly on 1 device (measured: a 20B-param
model with only 12 subjects took ~44 minutes of GPU-serial ridge fitting
alone on ZuCo Reading). The fix is **not** to fall back to CPU but to request
2 GPUs per job (see the family sbatch scripts) so cross-GPU subject
parallelism (``subject_n_jobs``, below) actually has more than one lane —
per-LAYER parallelism (``n_jobs`` CUDA streams within one subject's fit) is
the default axis, but two subjects can now be in flight at once instead of
one.

When the SLURM allocation requests **more than one GPU**, subjects *can* be
fit in true parallel — one subject per physical device, each on its own CUDA
context, no contention — since by the time ``_regress`` runs, every
embedding is already extracted and cached to host memory (the GPU is only
needed for the ridge SVD from here on). Set ``subject_n_jobs`` to the number
of GPU-pinned worker threads to use (capped at ``torch.cuda.device_count()``
internally); each worker calls ``torch.cuda.set_device()`` once per subject
before fitting (verified thread-local: two threads pinned to different
devices land results on the correct device with no cross-talk). Default
``subject_n_jobs=1`` keeps the original serial-subjects behavior — set it to
match however many GPUs the job actually requests.

Chosen ridge alpha is logged per layer, plus a per-subject median/min/max
summary flagging when layers pin at the grid's edge (see
``natural_stories_embed_gpu.py`` for the same convention — this is what
surfaced the alibi alpha-grid-too-narrow issue).

Like GECO, MECo's word table (:func:`neuroai.datasets.meco.meco_word_list`)
already carries every word's surface form for every ``wordnum`` (no
reconstruction-from-raw-text step needed — see :mod:`neuroai.datasets.meco`
module docstring), so the word list used for embeddings is exactly the word
list used for RT, with no extra bookkeeping.

MECo is a much smaller corpus than Provo/GECO: only 12 passages (English
subset), 136-227 words each. Two settings differ from the Provo/GECO template
as a result:

* ``context_words=220`` — covers the longest passage almost in full (see
  module-level note in ``configs/embedding/behavioral/meco_fp.yaml``).
* the default alpha grid is shifted lower and ``print_chosen_alpha`` should
  be watched closely — far fewer training words per fold (and now per subject
  too) than Provo/GECO means the effective regularization needed can differ a
  lot by model width.

Pipeline
--------
1. ``MecoDataset`` (aggregate=False) -> one row per (participant_id, story_id,
   zone) trial with raw ``rt`` and ``is_edge``.
2. For each of MECo's 12 English passages, run the model once over the full
   token sequence via
   :func:`neuroai.features.llm.contextual_word_embeddings_hf` with
   ``layer_index="all"`` -> ``(n_tokens, n_layers, hidden)`` at the last-token
   position of each word. Cached to disk per (model, revision, section_id); the
   cache key does not include the MECo measure, so SP/FP/GP sweeps sharing one
   model/checkpoint reuse the same embeddings (mirrors the surprisal cache's
   measure-agnostic sharing — see ``meco_lme.py`` module docstring). Computed
   once and shared across all subjects (embeddings don't depend on the subject).
3. Per subject: GPU ridge regression (nested CV, himalaya SVD-based GCV) per
   layer: ``rt ~ embedding``. ``GroupKFold`` on ``story_id`` (leave-passages-
   out) is the default within each subject's own trials.
"""

from __future__ import annotations

import logging
import typing as tp
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.meco import MecoDataset, meco_word_list
from ..regressors.ridge import CVScore, LayerwiseRidgeResults, cv_ridge_score_multioutput_gpu
from .base import ModelConfig, RegressionConfig, word_predictions_root, word_predictions_dir
from .subject_average import average_items_across_subjects, fit_averaged_layers

logger = logging.getLogger(__name__)


def _himalaya_gpu_warmup(n_devices: int = 1) -> None:
    """Register the himalaya torch_cuda backend and force the one-time,
    non-thread-safe CUDA/MAGMA SVD backend registration up front, single-
    threaded — must happen before any per-layer (or, with
    ``subject_n_jobs>1``, per-subject) thread pool starts calling
    ``torch.linalg.svd`` concurrently, or PyTorch raises "lazy wrapper should
    be called at most once" (same fix as podcast.py /
    natural_stories_embed_gpu.py's GPU-batched regression).

    ``n_devices>1`` additionally warms up every device that cross-GPU
    subject parallelism will use — the lazy registration is per-process, not
    per-device, but the first real ``torch.linalg.svd`` call on each device
    still does device-specific driver setup that's safer done up front,
    single-threaded, than racing it inside the subject thread pool."""
    from himalaya.backend import set_backend as _set_backend
    _set_backend("torch_cuda")
    import torch
    orig_device = torch.cuda.current_device()
    for gid in range(n_devices):
        torch.cuda.set_device(gid)
        _svd_warmup = torch.eye(2, device=f"cuda:{gid}")
        torch.linalg.svd(_svd_warmup, full_matrices=False)
        del _svd_warmup
    torch.cuda.set_device(orig_device)


def _fit_subject_layers_gpu(
    pid: str, emb_sub: np.ndarray, rt: np.ndarray, groups: np.ndarray | None,
    *,
    model_name: str, measure: str,
    n_splits: int, alpha, scale: bool, random_state: int,
    n_layer_workers: int,
    save_word_predictions: bool = False,
) -> tuple[list[dict], np.ndarray | None]:
    """Fit GPU ridge CV for every layer of one subject's trials, one CUDA
    stream per layer (layer-level parallelism on the single shared GPU).

    When ``save_word_predictions``, every layer's out-of-fold predictions are
    captured for free (``return_oof_predictions=True`` costs nothing extra on
    the GPU path — mirrors ``natural_stories_embed_gpu.py``'s convention) and
    returned as ``(n_layers, n_trials)`` alongside the per-layer score rows.
    """
    import torch

    n_layers = emb_sub.shape[1]
    Y = rt.reshape(-1, 1).astype(np.float32)

    def _fit_layer(li: int) -> tuple[dict, np.ndarray | None]:
        X_li = emb_sub[:, li, :].astype(np.float32)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            if save_word_predictions:
                scores, oof = cv_ridge_score_multioutput_gpu(
                    X_li, Y, groups=groups, n_splits=n_splits, alpha=alpha,
                    scale=scale, standardize_target=True, random_state=random_state,
                    return_oof_predictions=True,
                )
                oof_li = oof[:, 0]
            else:
                scores = cv_ridge_score_multioutput_gpu(
                    X_li, Y, groups=groups, n_splits=n_splits, alpha=alpha,
                    scale=scale, standardize_target=True, random_state=random_state,
                )
                oof_li = None
        stream.synchronize()
        score = scores[0]
        row = dict(
            model_name=model_name, measure=measure, participant_id=pid,
            layer=li, layer_rel=li / max(1, n_layers - 1), n_layers=n_layers,
            n_trials=len(rt), **score.to_dict(),
        )
        return row, oof_li

    rows: list[dict | None] = [None] * n_layers
    oof_by_layer: list[np.ndarray | None] = [None] * n_layers
    n_workers = min(n_layer_workers, n_layers)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fit_layer, li): li for li in range(n_layers)}
        for fut in as_completed(futures):
            li = futures[fut]
            row, oof_li = fut.result()
            rows[li] = row
            oof_by_layer[li] = oof_li

    rows = [r for r in rows if r is not None]
    chosen_alphas = np.array([r["ridge_alpha"] for r in rows if np.isfinite(r["ridge_alpha"])])
    if chosen_alphas.size:
        alphas_arr = np.asarray(alpha, dtype=np.float64)
        n_at_min = int(np.sum(chosen_alphas <= alphas_arr.min()))
        n_at_max = int(np.sum(chosen_alphas >= alphas_arr.max()))
        logger.info(
            "  [%s][%s][subject=%s] chosen alpha across %d layers: median=%.4g  min=%.4g  max=%.4g"
            "%s%s",
            model_name, measure, pid, chosen_alphas.size,
            float(np.median(chosen_alphas)), float(chosen_alphas.min()), float(chosen_alphas.max()),
            f"  ({n_at_min} layer(s) at grid min)" if n_at_min else "",
            f"  ({n_at_max} layer(s) at grid max)" if n_at_max else "",
        )

    oof_all = np.stack(oof_by_layer, axis=0).astype(np.float32) if save_word_predictions else None
    return rows, oof_all  # type: ignore[return-value]


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
class MecoEmbedDataConfig:
    """MECo Corpus (English subset) dataset location, measure, and
    embedding-extraction options."""

    root: str
    measure: str = "FP"  # "SP" | "FP" | "GP"
    language: str = "en"  # only English is wired up — see neuroai.datasets.meco docstring
    passages: list[int] | None = None
    rda_filename: str = "joint_l1_data_trimmed_version2.1.rda"
    min_dur: float = 50.0
    max_dur: float = 5000.0
    exclude_doc_edges: bool = True
    # Longest MECo (en) passage is 227 words (241 GPT-2 BPE tokens) — 220 covers
    # 11/12 passages fully; the 227-word passage loses its first ~7 words of
    # context, an acceptable edge case for a fixed-window setting.
    context_words: int = 220
    max_length: int | None = 512
    feature_cache_dir: str | None = None


@dataclass
class MecoEmbedBenchmarkConfig:
    data: MecoEmbedDataConfig
    model: ModelConfig
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    # Drop sentence- and passage-edge words (same convention as meco_lme's
    # filter_sentence_edges).
    filter_sentence_edges: bool = True
    # Leave-passages-out CV (GroupKFold on story_id) within each subject's own
    # trials. False -> plain shuffled KFold.
    group_cv_by_story: bool = True
    # Skip a subject if they have fewer than this many trials left after
    # filtering (too few for a meaningful GroupKFold CV).
    min_trials_per_subject: int = 20
    # Concurrent CUDA-stream workers for per-LAYER GPU regression within one
    # subject (see module docstring). Was a CPU thread-pool size before the
    # GPU migration; kept the same field name/config key for compatibility.
    n_jobs: int = 10
    # Number of GPU-pinned worker threads for SUBJECT-level parallelism, one
    # subject fully in flight per physical GPU at a time (each thread pins
    # its own `torch.cuda.set_device()` before fitting — verified thread-
    # local, see module docstring). Capped at `torch.cuda.device_count()`.
    # Set to match however many GPUs the SLURM job actually requests.
    subject_n_jobs: int = 1
    # Grouping label for saved CSVs when the HF load repo differs from the
    # canonical model id (e.g. OLMo early-training branches).
    canonical_model_name: str | None = None
    # Save every layer's out-of-fold predictions + the observed RT (+ word
    # identity), per subject, to a per-checkpoint-per-subject .npz — free on
    # the GPU path (see _fit_subject_layers_gpu). For post-hoc per-word
    # difficulty splits (e.g. DLT cost). On by default.
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
    # branch (implies average_across_subjects). See GecoEmbedBenchmarkConfig /
    # ProvoEmbedBenchmarkConfig for the rationale. Per-subject outputs come back
    # empty/NaN.
    average_only: bool = False


@dataclass
class MecoEmbedBenchmarkResult:
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


class MecoEmbedBenchmark:
    """Predict MECo (English) reading times from LLM layer activations using
    GPU ridge regression, fit independently per subject.

    Usage
    -----
    .. code-block:: python

        cfg = MecoEmbedBenchmarkConfig(
            data=MecoEmbedDataConfig(root="data/meco/.../eye tracking data", measure="FP"),
            model=ModelConfig(model_name="EleutherAI/pythia-160m"),
        )
        result = MecoEmbedBenchmark(cfg).run()
        print(result.layerwise.best_layer_score().spearmanr)  # mean across subjects
    """

    def __init__(self, cfg: MecoEmbedBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        *,
        existing_subject_rows: list[dict] | None = None,
        on_subject_done: tp.Callable[[list[dict]], None] | None = None,
    ) -> MecoEmbedBenchmarkResult:
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
        ds = MecoDataset(
            dcfg.root,
            dcfg.measure,
            language=dcfg.language,
            passages=dcfg.passages,
            min_dur=dcfg.min_dur,
            max_dur=dcfg.max_dur,
            exclude_doc_edges=dcfg.exclude_doc_edges,
            aggregate=False,
            rda_filename=dcfg.rda_filename,
        )
        trial_df = ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)
        if self.cfg.filter_sentence_edges and "is_edge" in trial_df.columns:
            trial_df = trial_df.loc[~trial_df["is_edge"].astype(bool)].reset_index(drop=True)
        if trial_df.empty:
            raise ValueError(
                f"No MECo trial observations left after filtering (measure={dcfg.measure})"
            )
        trial_df = trial_df.sort_values(["participant_id", "story_id", "zone"]).reset_index(drop=True)
        logger.info(
            "MECo embed [%s/%s]: %d trials, %d subjects, %d passages",
            dcfg.language, dcfg.measure, len(trial_df), trial_df["participant_id"].nunique(),
            trial_df["story_id"].nunique(),
        )
        return trial_df

    def _embed_words(self, trial_df: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
        """Return ``{(story_id, zone): (n_layers, hidden)}`` — one embedding per
        word, computed once and shared across every subject who read it."""
        from ..features.llm import contextual_word_embeddings_hf, contextual_word_embeddings_hf_fulltext

        dcfg = self.cfg.data
        mcfg = self.cfg.model
        tokenizer_no_revision = bool(mcfg.model_revision) and "pythia" in mcfg.model_name.lower()
        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

        # meco_word_list() already enumerates every word of every passage (unlike
        # Provo, no separate raw-text reconstruction is needed — see module
        # docstring), so it directly gives the ordered token sequence per story.
        full_words = meco_word_list(
            dcfg.root, language=dcfg.language, exclude_doc_edges=dcfg.exclude_doc_edges,
            rda_filename=dcfg.rda_filename,
        )
        if dcfg.passages is not None:
            full_words = full_words[full_words["story_id"].isin(dcfg.passages)]

        needed_stories = set(trial_df["story_id"].tolist())
        emb_lookup: dict[tuple[int, int], np.ndarray] = {}
        for story_id, group in full_words.groupby("story_id", sort=True):
            if story_id not in needed_stories:
                continue
            group = group.sort_values("zone")
            tokens = group["word"].astype(str).tolist()
            zones = group["zone"].tolist()
            # Every MECo passage fits well under max_length in one chunk, so this
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
                    lang=dcfg.language,
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
                    lang=dcfg.language,
                    tokenizer_no_revision=tokenizer_no_revision,
                    n_params=mcfg.n_params,
                    trust_remote_code=mcfg.trust_remote_code,
                    untrained=mcfg.untrained,
                    untrained_seed=mcfg.untrained_seed,
                    dtype=surp_dtype,
                )
            for i, zone in enumerate(zones):
                emb_lookup[(int(story_id), int(zone))] = emb[i]

        logger.info(
            "MECo embed: extracted embeddings for %d words across %d passages",
            len(emb_lookup), len(needed_stories),
        )
        return emb_lookup

    def _fit_one_subject(
        self, pid: str, emb_sub: np.ndarray, rt: np.ndarray, groups: np.ndarray | None,
        *, model_name: str, measure: str, gpu_id: int | None = None,
        word_ids: pd.DataFrame | None = None,
    ) -> tuple[list[dict], pd.DataFrame | None]:
        """Fit GPU ridge CV for every layer of one subject's trials (one CUDA
        stream per layer; see ``_fit_subject_layers_gpu``).

        ``gpu_id`` pins this subject's fit (and every per-layer CUDA stream
        it spawns) to one physical device — set only when called from a
        cross-GPU subject worker thread (``subject_n_jobs>1``); verified
        thread-local, see module docstring."""
        import torch

        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        rcfg = self.cfg.regression
        rows, oof_all = _fit_subject_layers_gpu(
            pid, emb_sub, rt, groups, model_name=model_name, measure=measure,
            n_splits=rcfg.n_splits, alpha=rcfg.alpha, scale=rcfg.scale,
            random_state=rcfg.random_state, n_layer_workers=self.cfg.n_jobs,
            save_word_predictions=self.cfg.save_word_predictions,
        )
        per_word: pd.DataFrame | None = None
        if oof_all is not None and word_ids is not None:
            per_word = word_ids.copy()
            per_word.attrs["oof_all_layers"] = oof_all  # (n_layers, n_trials)
            per_word.attrs["rt_true"] = rt
        return rows, per_word

    def _word_predictions_root(self) -> Path:
        """Root dir for per-word/per-item OOF prediction .npz files --
        ``word_predictions_dir`` when set (e.g. a $SCRATCH path, to keep
        these many small files off the $WORK quota the CSV summaries live
        on), else falls back to ``output_dir``."""
        if self.cfg.word_predictions_dir:
            return Path(self.cfg.word_predictions_dir)
        return word_predictions_root(self.cfg.output_dir)

    def _save_word_predictions(self, pid: str, per_word: pd.DataFrame) -> None:
        """Save one subject's every-layer out-of-fold predictions + observed
        RT to a per-word ``.npz`` (mirrors ``natural_stories_embed_gpu.py``'s
        ``_save_word_predictions`` convention), for post-hoc per-word
        difficulty splits (e.g. DLT)."""
        if not (self.cfg.word_predictions_dir or self.cfg.output_dir):
            logger.warning(
                "save_word_predictions=True but output_dir is not set — skipping word-prediction save for %s", pid,
            )
            return
        out_dir = self._word_predictions_root() / "meco_embed_word_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_revision = (self.cfg.model.model_revision or "main").replace("/", "_")
        safe_model = (self.cfg.canonical_model_name or self.cfg.model.model_name).replace("/", "_")
        out_path = out_dir / f"{safe_model}_{model_revision}_{pid}.npz"
        oof_all = per_word.attrs["oof_all_layers"]  # (n_layers, n_trials)
        rt_true = per_word.attrs["rt_true"]
        np.savez_compressed(
            out_path,
            oof_predictions=oof_all,                    # (n_layers, n_trials)
            rt_true=rt_true.astype(np.float32),
            story_id=per_word["story_id"].to_numpy() if "story_id" in per_word.columns else np.array([]),
            zone=per_word["zone"].to_numpy() if "zone" in per_word.columns else np.array([]),
            words=per_word["word"].astype(str).to_numpy() if "word" in per_word.columns else np.array([]),
            model=np.array(self.cfg.model.model_name),
            revision=np.array(self.cfg.model.model_revision or "main"),
        )
        logger.info("  Saved word-level predictions (%s) -> %s", oof_all.shape, out_path)

    def _log_subject_result(self, subj_rows: list[dict], *, model_name: str, measure: str, n_layers: int) -> None:
        pid = subj_rows[0]["participant_id"]
        logger.info(
            "[%s][%s][subject=%s] mean correlation across layers: pearson=%.4f spearman=%.4f",
            model_name, measure, pid,
            float(np.nanmean([r["pearsonr"] for r in subj_rows])),
            float(np.nanmean([r["spearmanr"] for r in subj_rows])),
        )

    def _regress(
        self,
        trial_df: pd.DataFrame,
        emb_lookup: dict[tuple[int, int], np.ndarray],
        *,
        existing_subject_rows: list[dict] | None = None,
        on_subject_done: tp.Callable[[list[dict]], None] | None = None,
    ) -> MecoEmbedBenchmarkResult:
        import torch

        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name
        measure = self.cfg.data.measure
        n_layers = next(iter(emb_lookup.values())).shape[0]

        if self.cfg.average_only:
            # Averaged-RT-only mode: skip every per-subject GPU fit (and the
            # himalaya warmup / device bookkeeping below). The averaged branch
            # runs on CPU off raw per-word RT rows. Per-subject outputs are
            # returned empty/NaN. Mirrors GecoEmbedBenchmark.
            n_items_avg = None
            average_layerwise = None
            average_summary_df = None
            if self.cfg.average_across_subjects:
                n_items_avg, average_layerwise, average_summary_df = self._fit_average(
                    trial_df, emb_lookup, model_name=model_name, measure=measure, n_layers=n_layers,
                )
            return MecoEmbedBenchmarkResult(
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

        n_gpus = torch.cuda.device_count()
        n_subject_workers = min(self.cfg.subject_n_jobs, max(1, n_gpus))
        if self.cfg.subject_n_jobs > 1 and n_gpus <= 1:
            logger.warning(
                "subject_n_jobs=%d requested but only %d GPU visible — cross-"
                "GPU subject parallelism needs >1 device (see module "
                "docstring); falling back to serial subjects. Layer "
                "parallelism is still controlled by n_jobs.",
                self.cfg.subject_n_jobs, n_gpus,
            )
        elif n_subject_workers > 1:
            logger.info(
                "subject_n_jobs=%d, %d GPUs visible -> fitting %d subjects "
                "concurrently, one per device.",
                self.cfg.subject_n_jobs, n_gpus, n_subject_workers,
            )

        _himalaya_gpu_warmup(n_devices=n_subject_workers)

        per_subject_rows: list[dict] = list(existing_subject_rows or [])
        done_subjects = {r["participant_id"] for r in per_subject_rows}

        save_word_predictions = self.cfg.save_word_predictions
        word_id_cols = [c for c in ("story_id", "zone", "word") if c in trial_df.columns]

        def _iter_subject_data() -> tp.Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame | None]]:
            """Yield one not-yet-done subject's data at a time (lazy on
            purpose — materializing every subject's ``emb_sub`` up front held
            all of them in memory simultaneously regardless of
            ``subject_n_jobs``, which OOM'd for large models)."""
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

        if n_subject_workers <= 1:
            for pid, emb_sub, rt, groups, word_ids in _iter_subject_data():
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
            # Cross-GPU subject parallelism: one subject in flight per
            # device, round-robin-assigned. Bounded submission (at most
            # n_subject_workers pending futures, refilled from the same lazy
            # generator as each one finishes) keeps the "one subject's
            # emb_sub materialized at a time per in-flight slot" memory
            # property _iter_subject_data was written for, rather than
            # loading every subject's embeddings up front.
            subject_iter = _iter_subject_data()
            pid_of: dict[tp.Any, str] = {}

            def _submit(gpu_id: int) -> tp.Any | None:
                try:
                    pid, emb_sub, rt, groups, word_ids = next(subject_iter)
                except StopIteration:
                    return None
                fut = pool.submit(
                    self._fit_one_subject, pid, emb_sub, rt, groups,
                    model_name=model_name, measure=measure, gpu_id=gpu_id, word_ids=word_ids,
                )
                pid_of[fut] = pid
                return fut

            with ThreadPoolExecutor(max_workers=n_subject_workers) as pool:
                gpu_of: dict[tp.Any, int] = {}
                for gpu_id in range(n_subject_workers):
                    fut = _submit(gpu_id)
                    if fut is not None:
                        gpu_of[fut] = gpu_id

                while gpu_of:
                    done, _ = wait(list(gpu_of.keys()), return_when=FIRST_COMPLETED)
                    for fut in done:
                        gpu_id = gpu_of.pop(fut)
                        fpid = pid_of.pop(fut)
                        subj_rows, per_word = fut.result()
                        self._log_subject_result(subj_rows, model_name=model_name, measure=measure, n_layers=n_layers)
                        per_subject_rows.extend(subj_rows)
                        if per_word is not None:
                            self._save_word_predictions(fpid, per_word)
                        if on_subject_done is not None:
                            on_subject_done(subj_rows)
                        new_fut = _submit(gpu_id)
                        if new_fut is not None:
                            gpu_of[new_fut] = gpu_id

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

        return MecoEmbedBenchmarkResult(
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
        out_dir = self._word_predictions_root() / "meco_embed_average_word_predictions"
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
