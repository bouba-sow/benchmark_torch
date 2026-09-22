"""Natural Stories self-paced reading time benchmark.

Predicts aggregate word reading times from LLM hidden states and/or surprisal
using story-level grouped ridge CV (leave-one-story-out by default).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.natural_stories import NaturalStoriesDataset
from ..evaluation.metrics import summarize_results
from ..features.llm import (
    clear_hf_model_cache,
    contextual_word_embeddings_hf,
)
from ..regressors.ridge import (
    CVScore,
    LayerwiseRidgeResults,
    cv_ridge_score,
)
from .base import ModelConfig, RegressionConfig, default_ridge_alphas

logger = logging.getLogger(__name__)


@dataclass
class NaturalStoriesDataConfig:
    """Natural Stories dataset location and loading options."""

    dataset_root: str
    stories: list[int] | None = None
    min_correct: int = 0
    context_words: int = 48
    max_length: int = 512
    log_rt: bool = True
    feature_cache_dir: str | None = None


@dataclass
class NaturalStoriesBenchmarkConfig:
    """Top-level config for the Natural Stories SPR benchmark."""

    data: NaturalStoriesDataConfig
    model: ModelConfig
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    output_dir: str = "outputs"
    include_surprisal: bool = True
    layerwise: bool = True
    tokenizer_no_revision: bool = True


@dataclass
class NaturalStoriesBenchmarkResult:
    """Output of :class:`NaturalStoriesBenchmark`."""

    model_name: str
    n_words: int
    score: CVScore
    layerwise: LayerwiseRidgeResults | None = None
    summary_df: pd.DataFrame | None = None
    surprisal_score: CVScore | None = None


class NaturalStoriesBenchmark:
    """Predict Natural Stories reading times from LLM features.

    Pipeline
    --------
    1. Load word-level mean RTs via :class:`NaturalStoriesDataset`.
    2. Extract contextual embeddings per story (context resets at story boundaries).
    3. Grouped ridge CV by ``story_id`` (default 10-fold LOSO).
    4. Optionally score RT ~ surprisal with the same CV splits.
    """

    def __init__(self, cfg: NaturalStoriesBenchmarkConfig) -> None:
        self.cfg = cfg

    def run(self) -> NaturalStoriesBenchmarkResult:
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
        layer_index: str | int = "all" if self.cfg.layerwise else "last"
        chunks: list[np.ndarray] = []
        surp_dtype = "float32" if "gpt-bert" in mcfg.model_name.lower() else None

        for story_id, group in df.groupby("story_id", sort=True):
            words = group["word"].astype(str).tolist()
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

    def _regress(
        self,
        emb: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        df: pd.DataFrame,
    ) -> NaturalStoriesBenchmarkResult:
        rcfg = self.cfg.regression
        model_name = self.cfg.model.model_name
        n_splits = min(rcfg.n_splits, len(np.unique(groups)))

        if emb.ndim == 3:
            n_layers = emb.shape[1]
            flat_X = emb[:, -1, :]
        else:
            flat_X = emb
            n_layers = 1

        ridge_kw = dict(
            n_splits=n_splits,
            alpha=rcfg.alpha,
            scale=rcfg.scale,
            random_state=rcfg.random_state,
            print_chosen_alpha=rcfg.print_chosen_ridge_alpha,
            groups=groups,
        )

        flat_score = cv_ridge_score(flat_X, y, **ridge_kw)
        logger.info("[%s] activations Pearson r = %.4f", model_name, flat_score.pearsonr)

        layerwise: LayerwiseRidgeResults | None = None
        if emb.ndim == 3:
            layer_scores = [
                cv_ridge_score(emb[:, li, :], y, **ridge_kw) for li in range(n_layers)
            ]
            layerwise = LayerwiseRidgeResults(
                model_name=model_name,
                layer_scores=layer_scores,
                layer_indices=list(range(n_layers)),
            )
            best = layerwise.best_layer_score()
            if np.isfinite(best.pearsonr):
                logger.info(
                    "[%s] best-layer Pearson r = %.4f (layer %d)",
                    model_name,
                    best.pearsonr,
                    int(np.nanargmax(layerwise.scores_array())),
                )

        rows: list[dict] = []
        if layerwise:
            for li, s in zip(layerwise.layer_indices, layerwise.layer_scores):
                rows.append(
                    dict(
                        model_name=model_name,
                        layer=li,
                        layer_rel=li / max(1, n_layers - 1),
                        **s.to_dict(),
                    )
                )
        else:
            rows.append(
                dict(model_name=model_name, layer=0, layer_rel=1.0, **flat_score.to_dict())
            )
        summary_df = summarize_results(rows)

        surprisal_score = self._surprisal_score(y, groups, df) if self.cfg.include_surprisal else None

        return NaturalStoriesBenchmarkResult(
            model_name=model_name,
            n_words=len(y),
            score=flat_score,
            layerwise=layerwise,
            summary_df=summary_df,
            surprisal_score=surprisal_score,
        )

    def _surprisal_score(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        df: pd.DataFrame,
    ) -> CVScore | None:
        mcfg = self.cfg.model
        dcfg = self.cfg.data
        tok_rev: str | None = None
        if mcfg.model_revision and "pythia" in mcfg.model_name.lower():
            tok_rev = "main"

        from ..features.surprisal import (
            batch_surprisal_one_model,
            load_ns_surprisal_cache,
            ns_surprisal_cache_path,
            save_ns_surprisal_cache,
        )

        surp_vec: np.ndarray | None = None
        cache_path = None
        if dcfg.feature_cache_dir:
            cache_path = ns_surprisal_cache_path(
                dcfg.feature_cache_dir,
                mcfg.model_name,
                mcfg.model_revision,
                tok_rev,
                dcfg.stories,
            )
            cached = load_ns_surprisal_cache(cache_path)
            if cached is not None:
                merged = df[["story_id", "zone"]].merge(
                    cached[["story_id", "zone", "surprisal"]],
                    on=["story_id", "zone"],
                    how="left",
                )
                if merged["surprisal"].notna().all():
                    surp_vec = np.asarray(merged["surprisal"], dtype=np.float64)

        if surp_vec is None:
            contexts = df["context"].astype(str).tolist()
            targets = df["word"].astype(str).tolist()
            try:
                surp_mat = batch_surprisal_one_model(
                    mcfg.model_name,
                    contexts,
                    targets,
                    device=mcfg.device,
                    model_revision=mcfg.model_revision,
                    tokenizer_revision=tok_rev,
                    n_params=mcfg.n_params,
                    trust_remote_code=mcfg.trust_remote_code,
                )
                surp_vec = np.asarray(surp_mat, dtype=np.float64).reshape(len(targets), -1)[:, 0]
                if cache_path is not None:
                    try:
                        save_ns_surprisal_cache(cache_path, df, surp_vec)
                    except Exception as exc:
                        logger.warning("Could not save surprisal cache: %s", exc)
            except Exception as exc:
                logger.warning("NaturalStories surprisal failed (%s)", exc)
                return None
            finally:
                clear_hf_model_cache()

        surp_vec = np.asarray(surp_vec, dtype=np.float64)
        valid = np.isfinite(y) & np.isfinite(surp_vec)
        rcfg = self.cfg.regression
        n_splits = min(rcfg.n_splits, len(np.unique(groups)))
        if valid.sum() < n_splits * 2:
            return None

        score = cv_ridge_score(
            surp_vec[valid].reshape(-1, 1),
            y[valid],
            n_splits=n_splits,
            alpha=rcfg.alpha,
            scale=rcfg.scale,
            random_state=rcfg.random_state,
            print_chosen_alpha=rcfg.print_chosen_ridge_alpha,
            groups=groups[valid],
        )
        logger.info(
            "[%s] RT ~ surprisal: Pearson r = %.4f, Spearman r = %.4f",
            mcfg.model_name,
            score.pearsonr,
            score.spearmanr,
        )
        return score

    def _save_results(self, result: NaturalStoriesBenchmarkResult) -> None:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if result.summary_df is not None:
            safe = result.model_name.replace("/", "_")
            path = out_dir / f"{safe}_spr_scores.csv"
            result.summary_df.to_csv(path, index=False)
            logger.info("Saved results → %s", path)


def natural_stories_ridge_alphas() -> list[float]:
    """Default ridge grid for SPR (high-dimensional embeddings, many words)."""
    return default_ridge_alphas()
