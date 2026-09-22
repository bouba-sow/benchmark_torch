"""Natural Stories self-paced reading time benchmark — GPU/himalaya ridge.

GPU-only counterpart to :mod:`neuroai.benchmarks.natural_stories` (CPU
``cv_ridge_score``, sklearn-based). Same data pipeline (word-level mean RTs,
per-story contextual embeddings, leave-N-stories-out grouped CV), but the
ridge fit itself runs on GPU via ``cv_ridge_score_multioutput_gpu`` (himalaya
SVD-based GCV) — one CUDA stream per layer, same pattern as
``podcast.py``. Even though the target here is a single
scalar (RT), the multi-output GPU path still applies cleanly with
``n_targets=1``; the benefit isn't cross-target amortization here but
avoiding a slow CPU ``RidgeCV`` loop over a high-dimensional embedding grid.

No CPU fallback / two-stage split: unlike the LME pipelines (R/pymer4 is
CPU-only on Jean Zay), ridge-on-embeddings has no such constraint — embedding
extraction AND regression both run on the same GPU job, one stage, no
prefetch/CPU split needed.

Chosen ridge alpha is logged per layer (not just a cross-target summary,
since there is only one target here) so the best alpha can be read directly
from the job log.

Out-of-fold predictions are captured for EVERY layer (not just the best/last
layer — cheap here since ``cv_ridge_score_multioutput_gpu``'s
``return_oof_predictions=True`` comes for free alongside the score, no
separate refit needed, unlike the CPU-refit convention used elsewhere in this
codebase) and saved to a per-checkpoint ``.npz`` for post-hoc per-word
difficulty splits (e.g. DLT cost). On by default.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.natural_stories import NaturalStoriesDataset
from ..evaluation.metrics import summarize_results
from ..features.cbr_rnn import contextual_word_embeddings_cbr_rnn
from ..features.llm import clear_hf_model_cache, contextual_word_embeddings_hf
from ..regressors.ridge import CVScore, LayerwiseRidgeResults, cv_ridge_score_multioutput_gpu
from .base import ModelConfig, RegressionConfig, word_predictions_dir

logger = logging.getLogger(__name__)


@dataclass
class NaturalStoriesEmbedGpuDataConfig:
    """Natural Stories dataset location and loading options."""

    dataset_root: str
    stories: list[int] | None = None
    min_correct: int = 0
    context_words: int = 48
    max_length: int = 512
    log_rt: bool = True
    feature_cache_dir: str | None = None


@dataclass
class NaturalStoriesEmbedGpuBenchmarkConfig:
    """Top-level config for the GPU/himalaya Natural Stories SPR benchmark."""

    data: NaturalStoriesEmbedGpuDataConfig
    model: ModelConfig
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    output_dir: str = "outputs"
    layerwise: bool = True
    tokenizer_no_revision: bool = True
    canonical_model_name: str | None = None
    # Parallel CUDA-stream workers for the per-layer ridge sweep (same
    # pattern as podcast.py).
    n_layer_workers: int = 8
    # Save every layer's out-of-fold predictions + the actual observed RT
    # (+ word identity) to a per-checkpoint .npz, for post-hoc per-word
    # difficulty splits (e.g. DLT cost). On by default.
    save_word_predictions: bool = True


@dataclass
class NaturalStoriesEmbedGpuBenchmarkResult:
    """Output of :class:`NaturalStoriesEmbedGpuBenchmark`."""

    model_name: str
    n_words: int
    n_layers: int
    layerwise: LayerwiseRidgeResults
    flat_score: CVScore  # last-layer score, for CSV-schema parity with the CPU pipeline
    summary_df: pd.DataFrame


class NaturalStoriesEmbedGpuBenchmark:
    """Predict Natural Stories reading times from LLM layer activations,
    fit on GPU via himalaya (one CUDA stream per layer).

    Pipeline
    --------
    1. Load word-level mean RTs via :class:`NaturalStoriesDataset`.
    2. Extract contextual embeddings per story (context resets at story boundaries).
    3. Grouped ridge CV by ``story_id`` (default 10-fold LOSO), one GPU
       himalaya fit per layer, all layers run concurrently (one CUDA stream
       each).
    4. Optionally save every layer's out-of-fold predictions for post-hoc
       per-word analysis.
    """

    def __init__(self, cfg: NaturalStoriesEmbedGpuBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(self) -> NaturalStoriesEmbedGpuBenchmarkResult:
        df = self._load_data()
        emb = self._extract_embeddings(df)
        y, groups = self._target_and_groups(df)

        result = self._regress(emb, y, groups, df)
        self._save_results(result)
        return result

    def _load_data(self) -> pd.DataFrame:
        dcfg = self.cfg.data
        ds = NaturalStoriesDataset(
            dcfg.dataset_root,
            stories=dcfg.stories,
            min_correct=dcfg.min_correct,
            aggregate=True,
        )
        df = ds.load_words()
        df = df.dropna(subset=["rt_mean"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid reading-time observations after filtering")
        return df

    def _target_and_groups(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rt = df["rt_mean"].astype(float).values
        if self.cfg.data.log_rt:
            rt = np.log(np.clip(rt, 1.0, None))
        groups = df["story_id"].astype(int).values
        return rt, groups

    def _extract_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        dcfg = self.cfg.data
        mcfg = self.cfg.model
        name = mcfg.model_name.lower()
        is_cbr_rnn = "cbr-rnn" in name or "cbr_rnn" in name
        layer_index: str | int = "all" if self.cfg.layerwise else "last"
        chunks: list[np.ndarray] = []
        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

        for story_id, group in df.groupby("story_id", sort=True):
            words = group["word"].astype(str).tolist()
            if is_cbr_rnn:
                # CBR-RNN bypasses AutoModelForCausalLM entirely and produces
                # a single hidden-state vector per token (no layer
                # dimension) -- same fallback as fmri.py/broderick.py/
                # eeg_erp.py/fedorenko2016.py/podcast.py: treat it as a
                # trivial 1-layer model so it flows through the same
                # layerwise ridge path as every real per-layer model.
                emb = contextual_word_embeddings_cbr_rnn(
                    words,
                    mcfg.model_name,
                    device=mcfg.device,
                    context_words=dcfg.context_words,
                    cache_dir=dcfg.feature_cache_dir,
                    section_id=int(story_id),
                    lang="en",
                )  # (n_words, d_hid)
                if layer_index == "all":
                    emb = emb[:, np.newaxis, :]  # (n_words, 1, d_hid)
            else:
                emb = contextual_word_embeddings_hf(
                    words,
                    mcfg.model_name,
                    layer_index=layer_index,
                    revision=mcfg.model_revision,
                    device=mcfg.device,
                    context_words=dcfg.context_words,
                    max_length=dcfg.max_length,
                    batch_size=mcfg.batch_size,
                    cache_dir=dcfg.feature_cache_dir,
                    section_id=int(story_id),
                    lang="en",
                    tokenizer_no_revision=self.cfg.tokenizer_no_revision,
                    n_params=mcfg.n_params,
                    trust_remote_code=mcfg.trust_remote_code,
                    untrained=mcfg.untrained,
                    untrained_seed=mcfg.untrained_seed,
                    dtype=surp_dtype,
                )
            chunks.append(emb)

        out = np.concatenate(chunks, axis=0)
        logger.info(
            "NaturalStories embeddings: model=%s shape=%s",
            mcfg.model_name,
            "×".join(str(d) for d in out.shape),
        )
        return out

    # ------------------------------------------------------------------
    # GPU ridge sweep — one himalaya fit per layer, one CUDA stream each
    # ------------------------------------------------------------------

    def _regress(
        self,
        emb: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        df: pd.DataFrame,
    ) -> NaturalStoriesEmbedGpuBenchmarkResult:
        rcfg = self.cfg.regression
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name
        n_splits = min(rcfg.n_splits, len(np.unique(groups)))

        if emb.ndim != 3:
            emb = emb[:, None, :]
        n_layers = emb.shape[1]
        Y = y.reshape(-1, 1).astype(np.float32)

        alphas_arr = np.asarray(rcfg.alpha, dtype=np.float64)
        logger.info(
            "[%s] ridge alpha grid: %d candidates, range=[%.4g, %.4g] — %s",
            model_name, alphas_arr.size, alphas_arr.min(), alphas_arr.max(),
            np.array2string(alphas_arr, precision=4),
        )
        logger.info(
            "[%s] %d-fold grouped CV (groups=story_id, %d unique stories)",
            model_name, n_splits, len(np.unique(groups)),
        )

        from himalaya.backend import set_backend as _set_backend
        _set_backend("torch_cuda")
        # torch.linalg.svd (used inside cv_ridge_score_multioutput_gpu) does a
        # one-time, non-thread-safe CUDA/MAGMA backend registration on its
        # first call in the process — force it here, single-threaded, before
        # spawning the per-layer thread pool below (same fix as
        # podcast.py / broderick.py's GPU-batched regression).
        import torch
        _svd_warmup = torch.eye(2, device=f"cuda:{torch.cuda.current_device()}")
        torch.linalg.svd(_svd_warmup, full_matrices=False)
        del _svd_warmup

        reg_kw = dict(
            n_splits=n_splits,
            alpha=rcfg.alpha,
            scale=rcfg.scale,
            standardize_target=True,
            random_state=rcfg.random_state,
            groups=groups,
            return_oof_predictions=True,
        )

        def _fit_layer(li: int) -> tuple[int, CVScore, np.ndarray]:
            X_li = emb[:, li, :].astype(np.float32)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                scores, oof = cv_ridge_score_multioutput_gpu(X_li, Y, **reg_kw)
            stream.synchronize()
            return li, scores[0], oof[:, 0]

        layer_scores: list[CVScore | None] = [None] * n_layers
        oof_by_layer: list[np.ndarray | None] = [None] * n_layers
        n_workers = min(self.cfg.n_layer_workers, n_layers)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_fit_layer, li): li for li in range(n_layers)}
            for fut in as_completed(futures):
                li, score, oof_li = fut.result()
                layer_scores[li] = score
                oof_by_layer[li] = oof_li
                logger.info(
                    "  [%s] layer %d/%d: chosen alpha=%.4g  pearson=%.4f  spearman=%.4f",
                    model_name, li, n_layers - 1,
                    score.ridge_alpha, score.pearsonr, score.spearmanr,
                )

        layerwise = LayerwiseRidgeResults(
            model_name=model_name,
            layer_scores=layer_scores,  # type: ignore[arg-type]
            layer_indices=list(range(n_layers)),
        )
        best = layerwise.best_layer_score()
        logger.info(
            "[%s] best-layer Pearson r = %.4f (layer %d)",
            model_name, best.pearsonr, int(np.nanargmax(layerwise.scores_array())),
        )

        chosen_alphas = np.array(
            [s.ridge_alpha for s in layer_scores if s is not None and np.isfinite(s.ridge_alpha)],
            dtype=np.float64,
        )
        if chosen_alphas.size:
            n_at_min = int(np.sum(chosen_alphas <= alphas_arr.min()))
            n_at_max = int(np.sum(chosen_alphas >= alphas_arr.max()))
            logger.info(
                "[%s] chosen alpha across %d layers: median=%.4g  min=%.4g  max=%.4g"
                "%s%s",
                model_name, chosen_alphas.size,
                float(np.median(chosen_alphas)), float(chosen_alphas.min()), float(chosen_alphas.max()),
                f"  ({n_at_min} layer(s) at grid min — consider extending the grid lower)"
                if n_at_min else "",
                f"  ({n_at_max} layer(s) at grid max — consider extending the grid higher)"
                if n_at_max else "",
            )

        rows = [
            dict(model_name=model_name, layer=li, layer_rel=li / max(1, n_layers - 1), **s.to_dict())
            for li, s in zip(layerwise.layer_indices, layerwise.layer_scores)
        ]
        summary_df = summarize_results(rows)

        if self.cfg.save_word_predictions:
            oof_all = np.stack(oof_by_layer, axis=0).astype(np.float32)  # (n_layers, n_words)
            self._save_word_predictions(model_name, df, y, oof_all)

        return NaturalStoriesEmbedGpuBenchmarkResult(
            model_name=model_name,
            n_words=len(y),
            n_layers=n_layers,
            layerwise=layerwise,
            flat_score=layer_scores[-1],  # type: ignore[index]
            summary_df=summary_df,
        )

    # ------------------------------------------------------------------
    # Per-word, per-layer OOF predictions
    # ------------------------------------------------------------------

    def _save_word_predictions(
        self,
        model_name: str,
        df: pd.DataFrame,
        y_true: np.ndarray,
        oof_all_layers: np.ndarray,  # (n_layers, n_words)
    ) -> None:
        out_dir = word_predictions_dir(self.cfg.output_dir, "naturalstories_embed_word_predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_revision = (self.cfg.model.model_revision or "main").replace("/", "_")
        safe_model = model_name.replace("/", "_")
        out_path = out_dir / f"{safe_model}_{model_revision}.npz"
        np.savez_compressed(
            out_path,
            oof_predictions=oof_all_layers,             # (n_layers, n_words)
            rt_true=y_true.astype(np.float32),           # (n_words,) — log-RT if log_rt=True
            words=df["word"].astype(str).to_numpy(),
            story_id=df["story_id"].to_numpy(),
            zone=df["zone"].to_numpy(),
            model=np.array(model_name),
            revision=np.array(self.cfg.model.model_revision or "main"),
        )
        logger.info(
            "  Saved word-level predictions (%s) -> %s", oof_all_layers.shape, out_path,
        )

    def _save_results(self, result: NaturalStoriesEmbedGpuBenchmarkResult) -> None:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if result.summary_df is not None:
            safe = result.model_name.replace("/", "_")
            path = out_dir / f"{safe}_spr_scores.csv"
            result.summary_df.to_csv(path, index=False)
            logger.info("Saved results → %s", path)


def natural_stories_embed_gpu_ridge_alphas() -> list[float]:
    """Default ridge grid for SPR (high-dimensional embeddings, many words).

    Wider/lower than ``default_ridge_alphas()`` (1e3-1e9): the alibi-family
    GCV sweep was pinning at that grid's floor (alpha=1000 chosen for every
    layer), meaning the true optimum was below the searchable range. 1e-1 to
    1e5 brackets that instead.
    """
    return np.logspace(-1, 5, 7).tolist()
