"""Pereira et al. (2018) fMRI voxelwise embedding encoding — GPU himalaya ridge.

Companion to ``neuroai/benchmarks/pereira_lme.py`` (which asks "does
surprisal explain language-network BOLD variance on top of nuisances?").
This one asks the encoding-style question: "how well do an LM's hidden-state
EMBEDDINGS predict voxelwise BOLD?" — same question ``neuroai/lpp_encoding.py``
answers for LPP. See ``docs/pereira_embedding_design.md`` for the full design
writeup this implements.

Key differences from the LME pipeline / from LPP's encoder
------------------------------------------------------------
- **Per-subject, not average-subject.** Pereira has no precomputed shared
  voxel space (unlike LPP's l-bg ``average_subject_*.gz`` files) — each
  subject's own AAL-multimask voxel indexing is subject-specific. One ridge
  model is fit PER SUBJECT (10 independent fits per layer); the resulting
  per-voxel correlation scores are averaged across subjects afterward, not
  fit into a shared "average subject" target.
- **Voxelwise, not ROI-averaged.** Unlike the LME pipeline (which averages
  within each of the 12 language-atlas ROIs to get one scalar predictor
  column), this predicts EVERY voxel individually — the standard "encoding
  model" convention — via ``cv_ridge_score_multioutput_gpu``'s multi-output
  solve (all voxels share one SVD per fold, "free" parallelism across
  voxels).
- **Mean-pooled sentence embeddings**, not last-word — see
  ``neuroai/features/hidden_states.py`` module docstring for the literature
  precedent (de Varda, Berzak, Fedorenko & Levy 2026) and why this differs
  from the LME pipeline's last-word SURPRISAL convention (a deliberate,
  separately-justified choice per predictor type, not an inconsistency).
- **5-fold GroupKFold by passage** (``PereiraDataset.get_passage_groups``) —
  the Pereira analogue of LPP's leave-one-section-out grouping (sentences in
  the same 3-4-sentence passage share topical content and must stay on one
  side of a split).
- **Layer parallelism via CUDA streams** (``ThreadPoolExecutor`` + one
  ``torch.cuda.Stream()`` per layer), same pattern as
  the Podcast GPU ridge path.

Pipeline (per model checkpoint)
--------------------------------
1. GPU stage (``extract_and_cache_hidden_states``): one forward pass per
   sentence (627 total), mean-pooled-over-words hidden state per layer,
   cached to disk (``neuroai/features/hidden_states.py``).
2. Encoding stage (``run``): for each subject, gather that subject's seen
   sentences' embeddings (X) and per-voxel language-atlas BOLD (Y); for each
   layer, ``cv_ridge_score_multioutput_gpu`` with 5-fold GroupKFold-by-passage
   (each layer on its own CUDA stream); aggregate per-voxel scores into a
   per-subject mean and a per-ROI breakdown; average across subjects.

Output: ``pereira_embed_by_layer.csv`` — one row per (layer, subject), plus
a ``pereira_embed_by_layer_group_mean.csv`` with the across-subject means.
"""

from __future__ import annotations

import logging
import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.pereira import DEFAULT_ATLAS, PereiraDataset
from ..regressors.ridge import CVScore, cv_ridge_score_multioutput_gpu
from ..sweep import csv_file_lock
from .base import ModelConfig

logger = logging.getLogger(__name__)

_ROW_COLUMNS = (
    "layer", "subject", "n_voxels", "n_obs", "pearsonr_mean", "spearmanr_mean",
    "pearsonr_median", "ridge_alpha_median", "error", "fit_wall_time_s", "revision",
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PereiraEmbedDataConfig:
    nc_path: str | Path = "data/pereira_fmri/Pereira2018_merged.nc"
    atlas: str = DEFAULT_ATLAS
    max_length: int | None = None
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    # Device for the (cheap, 627-sentence) hidden-state extraction forward passes.
    device: str = "cpu"
    # Hidden-state cache dir — same two-stage GPU/CPU-adjacent split as the
    # LME pipeline, though here BOTH stages typically run on GPU (himalaya
    # needs CUDA, unlike the LME's CPU-only R/pymer4 fit).
    feature_cache_dir: str | None = None


@dataclass
class PereiraEmbedConfig:
    # Production grid, narrowed from the wide calibration runs on gpt2
    # (2026-09-01): all 10 subjects' GCV-selected alphas landed in
    # 0.01-10,000, with 9/10 clustering in 1,000-10,000 —
    # np.logspace(1, 6, 8) brackets that with margin on both sides. (The
    # original wide grid, np.logspace(-3, 9, 12), first revealed a real bug
    # -- a handful of scattered-NaN voxels per subject were poisoning
    # cv_ridge_score_multioutput_gpu's shared per-fold GCV scalar, pinning
    # alpha at the grid's ceiling for 8/10 subjects — fixed in
    # PereiraDataset.get_voxelwise_per_subject before this grid was chosen.)
    ridge_alphas: np.ndarray = field(default_factory=lambda: np.logspace(1, 6, 8))
    n_splits: int = 5
    random_state: int = 0
    standardize_target: bool = True
    # Which layers to fit: 1 = all layers; N>1 = every Nth layer.
    layer_stride: int = 1
    # If True, fit ONLY the final layer (index n_layers-1), ignoring layer_stride.
    last_layer_only: bool = False
    # Concurrent CUDA-stream layer workers (ThreadPoolExecutor) — mirrors
    # ZuCoEegEncoder's n_layer_workers.
    n_layer_workers: int = 8
    require_cached_hidden_states: bool = False


@dataclass
class PereiraEmbedBenchmarkConfig:
    data: PereiraEmbedDataConfig
    model: ModelConfig
    embed: PereiraEmbedConfig = field(default_factory=PereiraEmbedConfig)
    output_dir: str = "outputs"
    canonical_model_name: str | None = None


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class PereiraEmbedBenchmark:
    """Voxelwise Pereira fMRI embedding encoding: himalaya GPU ridge, per
    subject, per LM layer, 5-fold GroupKFold-by-passage."""

    def __init__(self, cfg: PereiraEmbedBenchmarkConfig) -> None:
        self.cfg = cfg

    def extract_and_cache_hidden_states(self) -> np.ndarray:
        """GPU-stage entry point: populate ``data.feature_cache_dir`` only.

        Loads the LLM, computes mean-pooled-over-words per-layer hidden
        states for all 627 sentences, and returns — does not touch BOLD
        data, does not fit anything.
        """
        ds = PereiraDataset(self.cfg.data.nc_path)
        sentences, item_ids = ds.get_sentences()
        return self._extract_sentence_hidden_states(sentences, item_ids)

    def run(self, extra_fields: dict[str, tp.Any] | None = None) -> pd.DataFrame:
        dcfg = self.cfg.data
        ecfg = self.cfg.embed
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        revision = self.cfg.model.model_revision or "main"

        ds = PereiraDataset(dcfg.nc_path)
        sentences, item_ids = ds.get_sentences()
        hidden_all = self._extract_sentence_hidden_states(sentences, item_ids)  # (627, n_layers, D)
        n_layers = hidden_all.shape[1]
        item_id_to_idx = {iid: i for i, iid in enumerate(item_ids.tolist())}

        if ecfg.last_layer_only:
            layer_indices = [n_layers - 1]
        else:
            layer_indices = list(range(0, n_layers, max(1, ecfg.layer_stride)))

        per_subject = ds.get_voxelwise_per_subject(dcfg.atlas)
        logger.info(
            "Pereira embed encoding: %d subjects, %d layers (stride=%d), "
            "n_voxels per subject: %s",
            len(per_subject), len(layer_indices), ecfg.layer_stride,
            {s: y.shape[1] for s, (y, _, _) in per_subject.items()},
        )

        out_csv = out_dir / "pereira_embed_by_layer.csv"
        out_columns = list(_ROW_COLUMNS) + [
            k for k in (extra_fields or {}) if k not in _ROW_COLUMNS
        ]
        done: set[tuple[int, str]] = set()
        if out_csv.exists():
            try:
                # dtype={"subject": str} is the actual fix — the file on
                # disk correctly has e.g. "018", but without this pandas
                # infers int64 for the all-numeric-looking column and
                # silently drops the leading zero, which no downstream
                # .astype(str) can recover.
                existing = pd.read_csv(out_csv, dtype={"subject": str})
                existing_revision = existing["revision"] if "revision" in existing.columns else "main"
                same_revision = existing_revision == revision
                good = existing.loc[same_revision & existing["error"].isna()]
                # Subject IDs (e.g. "018") round-trip through CSV as bare
                # ints (pandas infers int64 for an all-numeric-looking
                # string column) unless forced back to str — comparing
                # against the str subject keys used below would silently
                # never match, refitting every already-done subject on
                # every resume (duplicate rows, not wrong values, but
                # wasted compute).
                done = set(zip(good["layer"].tolist(), good["subject"].astype(str).tolist()))
                n_failed = int((same_revision & existing["error"].notna()).sum())
                if done:
                    logger.info("Skipping %d already-fit (layer, subject) pair(s) for revision=%s", len(done), revision)
                if n_failed:
                    keep = existing.loc[~(same_revision & existing["error"].notna())]
                    with csv_file_lock(out_csv):
                        keep.reindex(columns=out_columns).to_csv(out_csv, index=False)
            except Exception:
                logger.warning("Could not read existing embed CSV; re-fitting everything.")

        self._ensure_himalaya_cuda()

        rows_out: list[dict] = []
        for subj, (Y, subj_item_ids, roi_per_voxel) in per_subject.items():
            pending_layers = [li for li in layer_indices if (li, subj) not in done]
            if not pending_layers:
                continue
            row_idx = np.array([item_id_to_idx[i] for i in subj_item_ids.tolist()], dtype=np.int64)
            X = hidden_all[row_idx]  # (n_seen, n_layers, D)
            groups = ds.get_passage_groups(subj_item_ids)

            layer_results = self._fit_layers_parallel(
                X, Y, groups=groups, layer_indices=pending_layers,
            )
            for li, (scores, wall_s, err) in layer_results.items():
                row: dict[str, tp.Any] = {
                    "layer": li, "subject": subj, "n_voxels": Y.shape[1],
                    "n_obs": X.shape[0], "fit_wall_time_s": wall_s, "error": err,
                }
                if err is None:
                    pr = np.array([s.pearsonr for s in scores])
                    sr = np.array([s.spearmanr for s in scores])
                    alphas = np.array([s.ridge_alpha for s in scores])
                    row.update({
                        "pearsonr_mean": float(np.nanmean(pr)),
                        "spearmanr_mean": float(np.nanmean(sr)),
                        "pearsonr_median": float(np.nanmedian(pr)),
                        "ridge_alpha_median": float(np.nanmedian(alphas)),
                    })
                    self._save_voxel_scores(out_dir, subj, li, revision, scores, roi_per_voxel)
                row["revision"] = revision
                if extra_fields:
                    row.update(extra_fields)
                rows_out.append(row)
                with csv_file_lock(out_csv):
                    pd.DataFrame([row]).reindex(columns=out_columns).to_csv(
                        out_csv, mode="a", header=not out_csv.exists(), index=False,
                    )
                logger.info(
                    "  [subject %s | layer %d] done in %.1fs — pearsonr_mean=%s",
                    subj, li, wall_s,
                    f"{row.get('pearsonr_mean'):.4f}" if row.get("pearsonr_mean") is not None else "NA",
                )

        return pd.DataFrame(rows_out)

    # ------------------------------------------------------------------
    # GPU layer-parallel fitting (CUDA streams, mirrors ZuCoEegEncoder)
    # ------------------------------------------------------------------

    def _ensure_himalaya_cuda(self) -> None:
        import torch
        from himalaya.backend import set_backend as _set_backend

        _set_backend("torch_cuda")
        # Force CUDA context init in the main thread before spawning worker
        # threads — same rationale as ZuCoEegEncoder's identical guard.
        torch.cuda.init()
        _svd_warmup = torch.eye(2, device=f"cuda:{torch.cuda.current_device()}")
        torch.linalg.svd(_svd_warmup, full_matrices=False)
        del _svd_warmup

    def _fit_layers_parallel(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        groups: np.ndarray,
        layer_indices: list[int],
    ) -> dict[int, tuple[list[CVScore], float, str | None]]:
        """Fit every layer in ``layer_indices`` concurrently, one CUDA stream each."""
        import torch

        ecfg = self.cfg.embed
        n_workers = min(ecfg.n_layer_workers, len(layer_indices))

        def _fit_one(li: int) -> tuple[int, list[CVScore] | None, float, str | None]:
            t0 = time.time()
            try:
                X_li = X[:, li, :].astype(np.float32)
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    scores = cv_ridge_score_multioutput_gpu(
                        X_li, Y,
                        groups=groups,
                        n_splits=ecfg.n_splits,
                        alpha=ecfg.ridge_alphas,
                        standardize_target=ecfg.standardize_target,
                        random_state=ecfg.random_state,
                    )
                stream.synchronize()
                return li, scores, time.time() - t0, None
            except Exception as exc:  # noqa: BLE001 — one bad layer must not kill the rest
                logger.error("Layer %d failed", li, exc_info=True)
                return li, None, time.time() - t0, str(exc)

        out: dict[int, tuple[list[CVScore], float, str | None]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_fit_one, li): li for li in layer_indices}
            for fut in as_completed(futures):
                li, scores, wall_s, err = fut.result()
                out[li] = (scores, wall_s, err)
        return out

    def _save_voxel_scores(
        self, out_dir: Path, subject: str, layer: int, revision: str,
        scores: list[CVScore], roi_per_voxel: np.ndarray,
    ) -> None:
        """Per-voxel correlation map + per-ROI breakdown for one (subject, layer)."""
        safe_rev = revision.replace("/", "_")
        voxel_dir = out_dir / "voxel_scores"
        voxel_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "voxel_idx": np.arange(len(scores)),
            "roi": roi_per_voxel,
            "pearsonr": [s.pearsonr for s in scores],
            "spearmanr": [s.spearmanr for s in scores],
            "ridge_alpha": [s.ridge_alpha for s in scores],
        })
        df.to_parquet(voxel_dir / f"{subject}_layer{layer}_{safe_rev}.parquet", index=False)

    # ------------------------------------------------------------------
    # Feature extraction (mean-pooled per-layer hidden states)
    # ------------------------------------------------------------------

    def _extract_sentence_hidden_states(
        self, sentences: list[str], item_ids: np.ndarray,
    ) -> np.ndarray:
        """Mean-pooled-over-words per-layer hidden state for every sentence.

        Returns
        -------
        np.ndarray
            Shape ``(n_sentences, n_layers, d_model)``, float32, in the same
            order as ``sentences``/``item_ids``.
        """
        mcfg = self.cfg.model
        dcfg = self.cfg.data
        is_cbr_rnn = "cbr-rnn" in mcfg.model_name.lower() or "cbr_rnn" in mcfg.model_name.lower()
        sentence_words = [sent.split() for sent in sentences]

        if is_cbr_rnn:
            from ..features.cbr_rnn import contextual_word_embeddings_cbr_rnn

            pooled_rows = []
            for words in sentence_words:
                word_vecs = contextual_word_embeddings_cbr_rnn(
                    words, mcfg.model_name, device=dcfg.device,
                )  # (n_words, d_hid)
                pooled_rows.append(word_vecs.mean(axis=0, keepdims=True))  # (1, d_hid)
            pooled = np.stack(pooled_rows, axis=0).astype(np.float32)  # (n_sentences, 1, d_hid)
            return pooled

        from ..features.hidden_states import mean_pooled_hidden_states_all_layers

        n_layers: int | None = None
        rows: list[np.ndarray] = []
        for i, words in enumerate(sentence_words):
            vec = mean_pooled_hidden_states_all_layers(
                words, mcfg.model_name,
                revision=mcfg.model_revision,
                device=dcfg.device,
                max_length=dcfg.max_length,
                tokenizer_no_revision=dcfg.tokenizer_no_revision,
                n_params=mcfg.n_params,
                trust_remote_code=dcfg.trust_remote_code,
                cache_dir=dcfg.feature_cache_dir,
                section_id=i,
                require_cache_hit=self.cfg.embed.require_cached_hidden_states,
            )  # (n_layers, D)
            if n_layers is None:
                n_layers = vec.shape[0]
            rows.append(vec)

        assert n_layers is not None
        return np.stack(rows, axis=0).astype(np.float32)  # (n_sentences, n_layers, D)
