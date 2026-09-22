"""GECO Corpus eye-tracking benchmark — trial-level LME (pymer4/lme4),
per-layer logit-lens surprisal sweep.

Fits the same by-subject mixed-effects model as Provo / UCL Reading / ZuCo
Reading (raw duration, Zipf frequency + current-word + spillover surprisal,
random slopes by subject) on one of GECO's two eye-tracking measures — FP
(first-pass / gaze duration), GP (go-past / regression-path duration; no SP —
see :mod:`neuroai.datasets.geco` module docstring). Reuses
:func:`neuroai.regressors.ucl_reading_lme.fit_ucl_reading_lme` and the shared
layer-sweep driver :func:`neuroai.benchmarks.ucl_reading_lme.run_layer_sweep`
unmodified — see ``natural_stories_lme.py`` / ``ucl_reading_lme.py`` module
docstrings for the per-layer logit-lens extraction/fitting rationale.

Performance note
-----------------
Like Provo's SP/FP/GP, GECO's FP/GP measures share the exact same 588-trial
word set — only the outcome duration column differs. Per-layer word
surprisal is therefore cached **once per (model, checkpoint)**
(cache_key="geco"), not once per measure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.geco import GecoDataset
from ..features.llm import clear_hf_model_cache
from .base import ModelConfig
from .ucl_reading_lme import UCLReadingLmeConfig, run_layer_sweep

logger = logging.getLogger(__name__)

def _is_cbr_rnn(model_name: str) -> bool:
    """Match both cclark/cbr-rnn AND cclark/cbr-rnn-m (an exact-string match
    against only the base model name previously left cbr-rnn-m falling
    through to the generic HF logit-lens path, which always cache-misses
    since CBR-RNN isn't a real AutoModelForCausalLM — see broderick_lme.py, which already use this same substring convention)."""
    name = model_name.lower()
    return "cbr-rnn" in name or "cbr_rnn" in name


@dataclass
class GecoDataConfig:
    """GECO Corpus dataset location, measure, and loading options."""

    root: str
    measure: str = "FP"  # "FP" | "GP"
    trials: list[int] | None = None
    min_dur: float = 50.0
    max_dur: float = 5000.0
    exclude_doc_edges: bool = True
    feature_cache_dir: str | None = None
    wordfreq_language: str = "en"


@dataclass
class GecoLmeBenchmarkConfig:
    data: GecoDataConfig
    model: ModelConfig
    lme: UCLReadingLmeConfig = field(default_factory=UCLReadingLmeConfig)
    output_dir: str = "outputs"
    # Grouping label for saved CSVs when HF load repo differs (e.g. OLMo early-training).
    canonical_model_name: str | None = None


class GecoLmeBenchmark:
    """Trial-level GECO Corpus eye-tracking LME, one fit per LM layer.

    Pipeline
    --------
    1. Load word-level (subject-averaged) and trial-level (per-subject) durations
       for one GECO measure (FP/GP).
    2. Zipf frequency (nuisance) is computed fresh from the ``word`` column
       at fit-time, inside ``fit_ucl_reading_lme`` (shared across measures).
    3. Extract per-layer logit-lens surprisal for every word (one model load,
       shared across FP/GP), disk-cached.
    4. Fit one ``lmer`` per layer via a process pool (CPU; only the surprisal
       extraction step uses ``device``) — identical model formulas to Provo /
       UCL Reading / ZuCo Reading.
    """

    def __init__(self, cfg: GecoLmeBenchmarkConfig) -> None:
        self.cfg = cfg

    def extract_and_cache_surprisal(self) -> np.ndarray:
        """GPU-stage entry point: populate ``data.feature_cache_dir`` only."""
        word_df, _ = self._load_data()
        return self._word_surprisal_all_layers(word_df)

    def run(self, extra_fields: dict | None = None) -> pd.DataFrame:
        dcfg = self.cfg.data
        lme_cfg = self.cfg.lme
        out_dir = Path(self.cfg.output_dir)

        revision = self.cfg.model.model_revision or "main"
        model_name = self.cfg.canonical_model_name or self.cfg.model.model_name

        word_df, trials = self._load_data()

        if not lme_cfg.include_surprisal:
            logger.info("[%s][%s] include_surprisal=False — nothing to fit", model_name, dcfg.measure)
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
            scores_csv_name="geco_lme_checkpoint_scores.csv",
            per_word_csv_name=f"geco_{dcfg.measure}_lme_per_word.csv",
            file_prefix=f"geco_{dcfg.measure}",
            extra_fields={"measure": dcfg.measure, **(extra_fields or {})},
            wordfreq_language=dcfg.wordfreq_language,
        )

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dcfg = self.cfg.data

        word_ds = GecoDataset(
            dcfg.root,
            dcfg.measure,
            trials=dcfg.trials,
            min_dur=dcfg.min_dur,
            max_dur=dcfg.max_dur,
            exclude_doc_edges=dcfg.exclude_doc_edges,
            aggregate=True,
        )
        word_df = word_ds.load_words().dropna(subset=["rt_mean"]).reset_index(drop=True)

        trial_ds = GecoDataset(
            dcfg.root,
            dcfg.measure,
            trials=dcfg.trials,
            min_dur=dcfg.min_dur,
            max_dur=dcfg.max_dur,
            exclude_doc_edges=dcfg.exclude_doc_edges,
            aggregate=False,
        )
        trials = trial_ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)
        if trials.empty:
            raise ValueError(f"No trial-level duration observations after filtering (measure={dcfg.measure})")
        logger.info(
            "GECO LME [%s]: %d trials, %d words, %d subjects",
            dcfg.measure,
            len(trials),
            len(word_df),
            trials["participant_id"].nunique(),
        )
        return word_df, trials

    def _word_surprisal_all_layers(self, word_df: pd.DataFrame) -> np.ndarray:
        """Per-layer logit-lens surprisal for every word, one model load.

        Shared across FP/GP — cache_key="geco" (measure-agnostic), same
        rationale as the old ``geco_surprisal_cache_path`` (no ``measure``
        param).

        The cache is populated (by the GPU prefetch stage,
        ``scripts/run/surprisal/behavioral/geco_extract.py``) over the FULL, measure-
        independent word list from :func:`neuroai.datasets.geco.geco_word_list`
        (every word in the material, regardless of whether any subject
        fixated it — a strict superset of any one measure's word set, chosen
        so the cache is reusable across FP/GP without recomputation). ``word_df``
        here, by contrast, is already measure-filtered (inner-joined against
        fixation data, ``dropna(subset=["rt_mean"])`` in ``_load_data``) and is
        therefore *shorter* than the cached array. Extracting/caching against
        the full list (not ``word_df``) and then selecting out ``word_df``'s
        rows by ``(story_id, zone)`` keeps this call's cache reads aligned
        with what the prefetch stage actually wrote, instead of assuming
        position-for-position equality between two differently-filtered word
        lists (that mismatch previously crashed ``fit_one_layer_worker`` with
        a length-mismatch ValueError once the two diverged in size).
        """
        mcfg = self.cfg.model
        dcfg = self.cfg.data

        if _is_cbr_rnn(mcfg.model_name):
            # CBR-RNN is not a HF causal LM and has no logit-lens cache — see
            # provo_lme.py / natural_stories_lme.py's identical branch.
            # Recompute directly instead of falling through to
            # logit_lens_surprisal_trials, which would hit a guaranteed cache
            # miss (RuntimeError under require_cache_hit=True, since the GPU
            # prefetch stage's cbr_rnn branch never wrote to that cache format
            # in the first place).
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

        from ..datasets.geco import geco_word_list

        full_word_df = geco_word_list(dcfg.root, exclude_doc_edges=dcfg.exclude_doc_edges)
        contexts = full_word_df["context"].astype(str).tolist()
        targets = full_word_df["word"].astype(str).tolist()

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
                cache_key="geco",
                require_cache_hit=self.cfg.lme.require_cached_surprisal,
                dtype=surp_dtype,
            )  # (n_full_words, n_layers)
        finally:
            clear_hf_model_cache()

        full_word_df = full_word_df.reset_index(drop=True)
        full_word_df["_full_idx"] = full_word_df.index
        idx = (
            word_df[["story_id", "zone"]]
            .merge(full_word_df[["story_id", "zone", "_full_idx"]], on=["story_id", "zone"], how="left")
            ["_full_idx"]
        )
        if idx.isna().any():
            missing = int(idx.isna().sum())
            raise ValueError(
                f"GECO surprisal alignment: {missing} word(s) in the measure-filtered "
                "word list have no match in the full word list (story_id/zone mismatch)."
            )
        return np.asarray(arr, dtype=np.float64)[idx.to_numpy(dtype=int)]
