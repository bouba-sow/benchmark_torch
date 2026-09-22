"""ZuCo behavioral reading benchmark — trial-level LME (pymer4/lme4), per-layer
logit-lens surprisal sweep.

Fits the same by-subject mixed-effects model as UCL Reading (raw TRT, Zipf
frequency + current-word + spillover surprisal, random slopes by subject) but on
ZuCo's eye-tracking TRT (total reading time) measure. Reuses
:func:`neuroai.regressors.ucl_reading_lme.fit_ucl_reading_lme` and the shared
layer-sweep driver :func:`neuroai.benchmarks.ucl_reading_lme.run_layer_sweep`
unmodified — see ``natural_stories_lme.py`` / ``ucl_reading_lme.py`` module
docstrings for the per-layer logit-lens extraction/fitting rationale.

Unlike Provo/GECO/MECO/OneStop's measure-agnostic caching (SP/FP/GP share one
word set), ZuCo's SR/NR/TSR tasks are three *distinct* sentence sets, so
per-layer surprisal is cached **per (model, checkpoint, task)**
(cache_key=f"zuco_reading_{task}"), not shared across tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.zuco_reading import ZuCoReadingDataset
from ..features.llm import clear_hf_model_cache
from .base import ModelConfig
from .ucl_reading_lme import UCLReadingLmeConfig, run_layer_sweep

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


@dataclass
class ZuCoReadingDataConfig:
    """ZuCo Reading dataset location, task, and loading options."""

    mat_root: str
    task: str = "NR"  # "SR" | "NR" | "TSR"
    subjects: list[str] | None = None
    min_fixations: int = 1
    min_trt: float = 50.0
    max_trt: float = 5000.0
    feature_cache_dir: str | None = None
    wordfreq_language: str = "en"


@dataclass
class ZuCoReadingLmeBenchmarkConfig:
    data: ZuCoReadingDataConfig
    model: ModelConfig
    lme: UCLReadingLmeConfig = field(default_factory=UCLReadingLmeConfig)
    output_dir: str = "outputs"
    # Grouping label for saved CSVs when HF load repo differs (e.g. OLMo early-training).
    canonical_model_name: str | None = None


class ZuCoReadingLmeBenchmark:
    """Trial-level ZuCo Reading TRT LME, one fit per LM layer.

    Pipeline
    --------
    1. Load word-level (subject-averaged) and trial-level (per-subject) TRTs
       for one ZuCo task (SR/NR/TSR).
    2. Zipf frequency (nuisance) is computed fresh from the ``word`` column
       at fit-time, inside ``fit_ucl_reading_lme``.
    3. Extract per-layer logit-lens surprisal for every word (one model
       load), disk-cached per task.
    4. Fit one ``lmer`` per layer via a process pool (CPU; only the surprisal
       extraction step uses ``device``) — identical model formulas to UCL
       Reading.
    """

    def __init__(self, cfg: ZuCoReadingLmeBenchmarkConfig) -> None:
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
            logger.info("[%s][%s] include_surprisal=False — nothing to fit", model_name, dcfg.task)
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
            scores_csv_name="zuco_reading_trt_lme_checkpoint_scores.csv",
            per_word_csv_name=f"zuco_reading_{dcfg.task}_lme_per_word.csv",
            file_prefix=f"zuco_reading_{dcfg.task}",
            extra_fields={"task": dcfg.task, **(extra_fields or {})},
            wordfreq_language=dcfg.wordfreq_language,
        )

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dcfg = self.cfg.data

        word_ds = ZuCoReadingDataset(
            dcfg.mat_root,
            dcfg.task,
            subjects=dcfg.subjects,
            min_fixations=dcfg.min_fixations,
            min_trt=dcfg.min_trt,
            max_trt=dcfg.max_trt,
            aggregate=True,
        )
        word_df = word_ds.load_words().dropna(subset=["rt_mean"]).reset_index(drop=True)

        trial_ds = ZuCoReadingDataset(
            dcfg.mat_root,
            dcfg.task,
            subjects=dcfg.subjects,
            min_fixations=dcfg.min_fixations,
            min_trt=dcfg.min_trt,
            max_trt=dcfg.max_trt,
            aggregate=False,
        )
        trials = trial_ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)
        if trials.empty:
            raise ValueError(f"No trial-level TRT observations after filtering (task={dcfg.task})")
        logger.info(
            "ZuCo Reading LME [%s]: %d trials, %d words, %d subjects",
            dcfg.task,
            len(trials),
            len(word_df),
            trials["participant_id"].nunique(),
        )
        return word_df, trials

    def _word_surprisal_all_layers(self, word_df: pd.DataFrame) -> np.ndarray:
        """Per-layer logit-lens surprisal for every word, one model load.

        cache_key=f"zuco_reading_{task}" — SR/NR/TSR are distinct sentence
        sets, so (unlike Provo/GECO/MECO/OneStop) the cache is NOT shared
        across tasks.
        """
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
                cache_key=f"zuco_reading_{dcfg.task}",
                require_cache_hit=self.cfg.lme.require_cached_surprisal,
                dtype=surp_dtype,
            )  # (n_words, n_layers)
            return np.asarray(arr, dtype=np.float64)
        finally:
            clear_hf_model_cache()
