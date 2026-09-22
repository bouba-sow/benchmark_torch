"""OneStop Corpus eye-tracking benchmark — trial-level LME (pymer4/lme4),
per-layer logit-lens surprisal sweep.

Fits the same by-subject mixed-effects model as Provo / GECO / MECO / UCL
Reading / ZuCo Reading (raw duration, Zipf frequency + current-word + spillover
surprisal, random slopes by subject) on one of OneStop's three eye-tracking
measures — SP (scan path / total dwell, default off), FP (first-pass / gaze
duration, **default**), GP (go-past / regression-path duration). Reuses
:func:`neuroai.regressors.ucl_reading_lme.fit_ucl_reading_lme` and the shared
layer-sweep driver :func:`neuroai.benchmarks.ucl_reading_lme.run_layer_sweep`
unmodified — see ``natural_stories_lme.py`` / ``ucl_reading_lme.py`` module
docstrings for the per-layer logit-lens extraction/fitting rationale.

Performance note
-----------------
Like Provo's SP/FP/GP, OneStop's three measures share the exact same
330-passage word set — only the outcome duration column differs. Per-layer
word surprisal is therefore cached **once per (model, checkpoint)**
(cache_key="onestop"), not once per measure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.onestop import OnestopDataset
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
class OnestopDataConfig:
    """OneStop Corpus dataset location, measure, and loading options."""

    root: str
    measure: str = "FP"  # "SP" | "FP" | "GP"
    stories: list[int] | None = None
    min_dur: float = 50.0
    max_dur: float = 5000.0
    exclude_doc_edges: bool = True
    feature_cache_dir: str | None = None
    wordfreq_language: str = "en"
    # Fixed subject subset for the LME sweep (120 of 360 — same for every
    # model). ``subjects`` is an explicit list; ``subjects_file`` points at a
    # newline-delimited file (``#`` comments and blank lines ignored), e.g.
    # ``configs/subjects/onestop_subjects_120.txt``. ``subjects`` wins if both are set;
    # both None loads all 360.
    subjects: list[str] | None = None
    subjects_file: str | None = None


@dataclass
class OnestopLmeBenchmarkConfig:
    data: OnestopDataConfig
    model: ModelConfig
    lme: UCLReadingLmeConfig = field(default_factory=UCLReadingLmeConfig)
    output_dir: str = "outputs"
    # Grouping label for saved CSVs when HF load repo differs (e.g. OLMo early-training).
    canonical_model_name: str | None = None


class OnestopLmeBenchmark:
    """Trial-level OneStop Corpus eye-tracking LME, one fit per LM layer.

    Pipeline
    --------
    1. Load word-level (subject-averaged) and trial-level (per-subject) durations
       for one OneStop measure (SP/FP/GP).
    2. Zipf frequency (nuisance) is computed fresh from the ``word`` column
       at fit-time, inside ``fit_ucl_reading_lme`` (shared across measures).
    3. Extract per-layer logit-lens surprisal for every word (one model load,
       shared across SP/FP/GP), disk-cached.
    4. Fit one ``lmer`` per layer via a process pool (CPU; only the surprisal
       extraction step uses ``device``) — identical model formulas to Provo /
       GECO / MECO / UCL Reading / ZuCo Reading.
    """

    def __init__(self, cfg: OnestopLmeBenchmarkConfig) -> None:
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
            scores_csv_name="onestop_lme_checkpoint_scores.csv",
            per_word_csv_name=f"onestop_{dcfg.measure}_lme_per_word.csv",
            file_prefix=f"onestop_{dcfg.measure}",
            extra_fields={"measure": dcfg.measure, **(extra_fields or {})},
            wordfreq_language=dcfg.wordfreq_language,
        )

    def _resolve_subjects(self) -> list[str] | None:
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
            logger.info("OneStop LME: using fixed %d-subject subset from %s", len(subs), path)
            return subs
        return None

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dcfg = self.cfg.data
        subjects = self._resolve_subjects()

        word_ds = OnestopDataset(
            dcfg.root,
            dcfg.measure,
            stories=dcfg.stories,
            min_dur=dcfg.min_dur,
            max_dur=dcfg.max_dur,
            exclude_doc_edges=dcfg.exclude_doc_edges,
            aggregate=True,
            subjects=subjects,
        )
        word_df = word_ds.load_words().dropna(subset=["rt_mean"]).reset_index(drop=True)

        trial_ds = OnestopDataset(
            dcfg.root,
            dcfg.measure,
            stories=dcfg.stories,
            min_dur=dcfg.min_dur,
            max_dur=dcfg.max_dur,
            exclude_doc_edges=dcfg.exclude_doc_edges,
            aggregate=False,
            subjects=subjects,
        )
        trials = trial_ds.load_words().dropna(subset=["rt"]).reset_index(drop=True)
        if trials.empty:
            raise ValueError(f"No trial-level duration observations after filtering (measure={dcfg.measure})")
        logger.info(
            "OneStop LME [%s]: %d trials, %d words, %d subjects",
            dcfg.measure,
            len(trials),
            len(word_df),
            trials["participant_id"].nunique(),
        )
        return word_df, trials

    def _word_surprisal_all_layers(self, word_df: pd.DataFrame) -> np.ndarray:
        """Per-layer logit-lens surprisal for every word, one model load.

        Shared across SP/FP/GP — cache_key="onestop" (measure-agnostic), same
        rationale as the old ``onestop_surprisal_cache_path`` (no ``measure``
        param).

        The cache is populated (by the GPU prefetch stage,
        ``scripts/run/surprisal/behavioral/onestop_extract.py``) over the FULL, measure-
        independent word list from :func:`neuroai.datasets.onestop.onestop_word_list`
        (every word in the 330 passages, regardless of whether any subject
        fixated it), while ``word_df`` here is already measure-filtered
        (inner-joined against fixation data, ``dropna(subset=["rt_mean"])`` in
        ``_load_data``) and therefore *shorter*. Extracting/caching against the
        full list and then selecting out ``word_df``'s rows by
        ``(story_id, zone)`` keeps this call's cache reads aligned with what
        the prefetch stage actually wrote — see ``geco_lme.py``'s identical fix
        for the length-mismatch ValueError this previously caused in
        ``fit_one_layer_worker`` once the two word lists diverged in size.
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

        from ..datasets.onestop import onestop_word_list

        full_word_df = onestop_word_list(dcfg.root, exclude_doc_edges=dcfg.exclude_doc_edges)
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
                cache_key="onestop",
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
                f"OneStop surprisal alignment: {missing} word(s) in the measure-filtered "
                "word list have no match in the full word list (story_id/zone mismatch)."
            )
        return np.asarray(arr, dtype=np.float64)[idx.to_numpy(dtype=int)]
