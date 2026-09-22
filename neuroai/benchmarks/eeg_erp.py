"""Nieuwland EEG N400 encoding benchmark: LLM activations → EEG amplitude.

For each unique (sentence_context, noun) pair we extract the LLM hidden state
of the target noun given its sentence context, then use multi-output ridge
regression to predict the full epoched EEG time course across 6 centro-parietal
channels.

Architecture mirrors the Podcast iEEG benchmark:
  - Himalaya GPU batched ridge: all channels × all timepoints in one call
  - ThreadPoolExecutor layer parallelism: each layer submits an independent
    himalaya call on its own CUDA stream
  - Item-based GroupKFold: generalise to unseen sentences (not subjects)
  - Spearman r is the primary metric; Pearson r saved alongside

Output files (written to ``out_dir/<family>/<model_short>/``):
  nieuwland_checkpoint_scores.csv      — per-checkpoint summary (best layer)
  nieuwland_checkpoint_layerwise.csv   — per-layer scores per checkpoint
  nieuwland_by_channel.csv             — per-channel scores per checkpoint
  nieuwland_by_subject.csv             — per-subject mean scores per checkpoint
  nieuwland_time_course.npz            — full time-resolved scores per layer

Reference
---------
Nieuwland et al. (2018), eLife 7:e33468.
Michaelov & Bergen (2026), Journal of Memory and Language.
"""

from __future__ import annotations

import hashlib
import logging
import typing as tp
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: F401
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr as _spearmanr_fn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from .base import word_predictions_dir

logger = logging.getLogger(__name__)


def _csv_append(path: Path, row: dict) -> None:
    """Append one dict as a CSV row, writing the header if the file is new."""
    import csv
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NieuwlandErpConfig:
    bids_root: str | Path
    # Subject / trial filtering
    subjects: list[str] | None = None   # None = all passing inclusion criteria
    included_in_paper_only: bool = True
    exclude_labs: list[str] = field(default_factory=lambda: ["glas"])
    min_trials: int = 20
    # EEG preprocessing
    target_sfreq: float = 32.0
    epoch_tmin: float = -0.2
    epoch_tmax: float = 0.8
    baseline: tuple[float, float] = (-0.2, 0.0)
    n400_tmin: float = 0.2
    n400_tmax: float = 0.5
    target_channels: list[str] = field(
        default_factory=lambda: ["Cz", "C3", "C4", "Pz", "P3", "P4"]
    )
    artifact_threshold_uv: float = 100.0
    # EEG epoch cache
    epoch_cache_dir: str | Path | None = None
    # LLM features
    hf_model_name: str = "EleutherAI/pythia-1b"
    hf_revision: str | None = None
    hf_device: str = "cpu"
    hf_batch_size: int = 8
    # None → use model config max_position_embeddings
    hf_max_length: int | None = None
    layer_mode: tp.Union[tp.Literal["last", "all"], int] = "all"
    tokenizer_no_revision: bool = True
    trust_remote_code: bool = False
    feature_cache_dir: str | Path | None = None
    # Regression
    n_folds: int = 10
    add_lab_covariate: bool = True
    # Ridge alpha grid — same defaults as podcast
    ridge_alphas: list[float] = field(
        default_factory=lambda: np.logspace(3.5, 8, 5).tolist()
    )
    scale: bool = True
    random_state: int = 0
    regression_device: str = "cpu"
    n_layer_workers: int = 36
    # Output
    out_dir: str | Path = "outputs_jeanzay/nieuwland_encoding"
    n_params: float = 0.0      # informational only (filled by sweep script)
    # Save each channel's best-layer out-of-fold predicted item-mean ERP (from
    # the item-mean regression path, ~80 items) alongside the actual ERP —
    # needed for a post-hoc DLT-cost difficulty split (analogous to Podcast/
    # Broderick's save_word_predictions, keyed by item_number instead of a
    # transcript position).
    save_word_predictions: bool = False


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChannelTimeResolvedScore:
    """Time-resolved Spearman/Pearson r per channel, averaged across CV folds."""
    channel: str
    times: np.ndarray          # (n_times,)
    spearmanr: np.ndarray      # (n_times,) median across folds
    pearsonr: np.ndarray       # (n_times,)
    spearmanr_folds: np.ndarray  # (n_folds, n_times) — full fold distribution

    @property
    def n400_spearmanr(self) -> float:
        """Mean Spearman r in 200–500 ms window."""
        mask = (self.times >= 0.2) & (self.times <= 0.5)
        if not mask.any():
            return float("nan")
        return float(np.nanmean(self.spearmanr[mask]))

    @property
    def peak_spearmanr(self) -> float:
        return float(np.nanmax(self.spearmanr))

    @property
    def peak_time(self) -> float:
        idx = int(np.nanargmax(self.spearmanr))
        return float(self.times[idx])


@dataclass
class ChannelScore:
    """Per-channel result for one (model, layer) combination."""
    name: str
    # Best-layer aggregate
    best_layer: int
    spearmanr: float          # primary metric: mean Spearman r in N400 window
    pearsonr: float
    # Best-layer time course
    time_course: ChannelTimeResolvedScore
    # Per-layer aggregates (list indexed by layer)
    layer_spearmanr: list[float] = field(default_factory=list)
    layer_pearsonr: list[float] = field(default_factory=list)
    # Per-layer time courses
    layer_time_courses: list[ChannelTimeResolvedScore] = field(default_factory=list)


@dataclass
class SubjectScore:
    """Per-subject N400 encoding score (mean across channels, best layer)."""
    subject: str
    lab: str
    spearmanr: float    # mean(ch, t) of per-(ch,t) Spearman r over sentences
    pearsonr: float
    n_trials: int
    # spearmanr(N400_mean(pred), N400_mean(actual)) per channel over sentences,
    # then averaged across channels.
    perchannel_spearmanr: float = float("nan")
    perchannel_pearsonr: float = float("nan")
    # For each trial and each channel: spearmanr(pred[trial, ch, :32], Y[trial, ch, :32])
    # over the 32 ERP timepoints, then averaged over channels, then over trials.
    # Analogous to the iEEG benchmark: measures how well the predicted ERP *shape* matches the actual.
    waveform_spearmanr: float = float("nan")
    waveform_pearsonr: float = float("nan")
    best_layer: int = 0   # layer with highest pc_sr for this subject


@dataclass
class NieuwlandErpResult:
    model_name: str
    revision: str | None
    # Trial-level (option 1): one regression per layer over all ~22k individual trials
    channel_scores: list[ChannelScore]
    subject_scores: list[SubjectScore]
    # Item-mean (option 2): Y averaged across subjects per item before regression
    item_mean_channel_scores: list[ChannelScore] = field(default_factory=list)
    n_subjects: int = 0
    n_trials: int = 0
    n_layers: int = 0
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    channel_names: list[str] = field(default_factory=list)
    # Per-layer pc_sr: mean over subjects of per-subject pc_sr at each layer
    # Shape: (n_layers,) — used for layer profile plots with the main metric
    layer_pc_sr: np.ndarray = field(default_factory=lambda: np.array([]))
    layer_wf_sr: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-channel pc_sr at each layer: (n_layers, n_channels) — for per-channel scaling plots
    layer_pc_sr_ch: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Option 1: trial-level aggregates ──────────────────────────────────────

    @property
    def mean_spearmanr(self) -> float:
        """Mean N400 Spearman r across channels — trial-level regression."""
        vals = [c.spearmanr for c in self.channel_scores if np.isfinite(c.spearmanr)]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def mean_pearsonr(self) -> float:
        vals = [c.pearsonr for c in self.channel_scores if np.isfinite(c.pearsonr)]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def mean_subject_spearmanr(self) -> float:
        """Mean N400 Spearman r across subjects — trial-level regression."""
        vals = [s.spearmanr for s in self.subject_scores if np.isfinite(s.spearmanr)]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def mean_subject_pearsonr(self) -> float:
        vals = [s.pearsonr for s in self.subject_scores if np.isfinite(s.pearsonr)]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def mean_subject_perchannel_spearmanr(self) -> float:
        """Mean across subjects of per-channel N400 Spearman r.

        For each subject: average EEG over N400 window per channel, compute
        spearmanr(pred, actual) over sentences per channel, then average channels.
        Finally average over subjects. This is the most interpretable subject-level score.
        """
        vals = [s.perchannel_spearmanr for s in self.subject_scores
                if np.isfinite(s.perchannel_spearmanr)]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def mean_subject_perchannel_pearsonr(self) -> float:
        vals = [s.perchannel_pearsonr for s in self.subject_scores
                if np.isfinite(s.perchannel_pearsonr)]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def mean_subject_waveform_spearmanr(self) -> float:
        """Mean across subjects of (mean across trials of (mean across channels of
        spearmanr(pred[trial,ch,:32], Y[trial,ch,:32]) over 32 ERP timepoints)).
        Measures how well the predicted ERP shape matches the actual shape per trial.
        """
        vals = [s.waveform_spearmanr for s in self.subject_scores
                if np.isfinite(s.waveform_spearmanr)]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def mean_subject_waveform_pearsonr(self) -> float:
        vals = [s.waveform_pearsonr for s in self.subject_scores
                if np.isfinite(s.waveform_pearsonr)]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def best_layer(self) -> int:
        """Layer with the most votes across subjects (each subject votes for their pc_sr argmax)."""
        layers = [s.best_layer for s in self.subject_scores]
        if not layers:
            return 0
        unique, counts = np.unique(layers, return_counts=True)
        return int(unique[np.argmax(counts)])

    # ── Option 2: item-mean aggregates ────────────────────────────────────────

    @property
    def item_mean_spearmanr(self) -> float:
        """Mean N400 Spearman r across channels — item-mean regression."""
        vals = [c.spearmanr for c in self.item_mean_channel_scores if np.isfinite(c.spearmanr)]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def item_mean_pearsonr(self) -> float:
        vals = [c.pearsonr for c in self.item_mean_channel_scores if np.isfinite(c.pearsonr)]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def item_mean_best_layer(self) -> int:
        if not self.item_mean_channel_scores:
            return 0
        layer_means = np.array([
            np.nanmean([c.layer_spearmanr[li] for c in self.item_mean_channel_scores
                        if li < len(c.layer_spearmanr)])
            for li in range(self.n_layers)
        ])
        return int(np.nanargmax(layer_means))


# ---------------------------------------------------------------------------
# LLM feature extraction for individual sentences
# ---------------------------------------------------------------------------

def _feature_cache_path(
    cache_dir: Path,
    model_name: str,
    revision: str | None,
    layer_mode: tp.Any,
    items_hash: str,
) -> Path:
    rev = revision or "main"
    mode = str(layer_mode)
    key = hashlib.md5(f"{model_name}_{rev}_{mode}_{items_hash}".encode()).hexdigest()[:12]
    safe = model_name.replace("/", "_")
    return cache_dir / f"{safe}_{rev}_{mode}_{key}.npz"


def _items_hash(contexts: list[str], nouns: list[str]) -> str:
    key = "|".join(f"{c}||{n}" for c, n in zip(contexts, nouns))
    return hashlib.md5(key.encode()).hexdigest()[:10]


def extract_sentence_features(
    contexts: list[str],
    nouns: list[str],
    model_name: str,
    *,
    revision: str | None = None,
    layer_mode: tp.Union[tp.Literal["last", "all"], int] = "all",
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int | None = None,
    tokenizer_no_revision: bool = True,
    cache_dir: Path | None = None,
    n_params: float | None = None,
    trust_remote_code: bool = False,
) -> np.ndarray:
    """Extract LLM hidden states for the target noun in each sentence.

    For each (context, noun) pair, builds the full text ``context + " " + noun``,
    tokenizes it, runs the model, and extracts the hidden state of the last token
    of ``noun`` (average across subword tokens if noun spans multiple tokens).

    Parameters
    ----------
    contexts : list of str
        Sentence context up to (but not including) the target noun.
    nouns    : list of str
        Target noun for each trial.

    Returns
    -------
    np.ndarray
        ``(n_items, n_layers, hidden_dim)`` if ``layer_mode="all"``, else
        ``(n_items, hidden_dim)``.
    """
    assert len(contexts) == len(nouns), "contexts and nouns must have equal length"

    # ── Cache lookup ──────────────────────────────────────────────────────────
    ih = _items_hash(contexts, nouns)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = _feature_cache_path(cache_dir, model_name, revision, layer_mode, ih)
        if cp.is_file():
            try:
                z = np.load(cp)
                arr = z["features"]
                n_bad = int(np.sum(~np.isfinite(arr)))
                if n_bad > 0:
                    logger.warning(
                        "Feature cache %s contains %d non-finite values — "
                        "deleting stale cache and re-extracting",
                        cp.name, n_bad,
                    )
                    cp.unlink(missing_ok=True)
                else:
                    logger.debug("Nieuwland features: loaded from cache %s", cp.name)
                    return arr
            except Exception as exc:
                logger.warning("Feature cache read failed (%s) — re-extracting", exc)

    # ── CBR-RNN: custom single-layer RNN, not a HuggingFace AutoModel ────────
    if "cbr-rnn" in model_name.lower() or "cbr_rnn" in model_name.lower():
        from neuroai.features.cbr_rnn import contextual_word_embeddings_cbr_rnn
        logger.info("Nieuwland features: CBR-RNN path for %s on %s", model_name, device)
        full_texts = [f"{c} {n}" for c, n in zip(contexts, nouns)]
        all_words  = [t.split() for t in full_texts]
        # Extract hidden state at the last word (= noun) for each sentence
        feats: list[np.ndarray] = []
        for words in all_words:
            h = contextual_word_embeddings_cbr_rnn(words, model_name, device=device)
            feats.append(h[-1])          # (d_hid,) — last word = noun
        arr = np.stack(feats, axis=0)    # (n_items, d_hid)
        arr = arr[:, np.newaxis, :]      # (n_items, 1, d_hid) — single "layer"
        if cache_dir is not None:
            try:
                np.savez_compressed(cp, features=arr)  # cp defined above in cache-lookup block
            except Exception as exc:
                logger.warning("Feature cache write failed (%s)", exc)
        return arr

    import torch
    from neuroai.features.llm import _get_cached_causal_lm

    logger.info(
        "Nieuwland features: loading %s (rev=%s) on %s",
        model_name, revision or "main", device,
    )

    _erp_load_kw: dict[str, object] = {}
    if trust_remote_code:
        _erp_load_kw["trust_remote_code"] = True
    tokenizer, model = _get_cached_causal_lm(
        model_name,
        revision=revision,
        device=device,
        load_kw=_erp_load_kw,
        tokenizer_no_revision=tokenizer_no_revision,
        n_params=n_params,
    )

    # Resolve max_length from model config if not set
    eff_max_length = max_length
    if eff_max_length is None:
        eff_max_length = getattr(model.config, "max_position_embeddings", 2048)
        logger.debug("Using model max_position_embeddings=%d", eff_max_length)

    n_items = len(contexts)
    full_texts = [f"{c} {n}" for c, n in zip(contexts, nouns)]

    all_features: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            batch_texts = full_texts[start : start + batch_size]
            batch_nouns = nouns[start : start + batch_size]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=eff_max_length,
                return_offsets_mapping=True,
            )
            offset_mapping = enc.pop("offset_mapping")  # (B, L, 2)
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
            # Some trust_remote_code architectures (e.g. GPT-BERT) ignore/default
            # differently on return_dict and hand back a plain tuple instead of a
            # ModelOutput -- same rationale as llm.py's _forward_hidden_states and
            # lpp_llm_features.py's _forward_all_hidden_states. Fall through to the
            # hook-based path below rather than crashing on `.hidden_states` access.
            # hidden_states: tuple of (B, L, D), length = n_layers + 1 (embedding + layers)
            hidden_states = None if isinstance(outputs, tuple) else outputs.hidden_states

            if hidden_states is None:
                # Model ignores output_hidden_states (e.g. stickbreaking) — collect via hooks
                _hs: list[torch.Tensor] = []
                _hk: list = []
                _inner = getattr(model, "model", model)
                _emb = getattr(_inner, "embeddings", None)
                if _emb is not None:
                    _hk.append(_emb.register_forward_hook(
                        lambda _m, _i, _o: _hs.append(_o.detach() if isinstance(_o, torch.Tensor) else _o[0].detach())
                    ))
                for _lyr in getattr(_inner, "layers", []):
                    _hk.append(_lyr.register_forward_hook(
                        lambda _m, _i, _o: _hs.append(_o.detach() if isinstance(_o, torch.Tensor) else _o[0].detach())
                    ))
                with torch.no_grad():
                    model(**enc, use_cache=False)
                for _h in _hk:
                    _h.remove()
                if _hs:
                    hidden_states = tuple(_hs)

            for bi, (text, noun) in enumerate(zip(batch_texts, batch_nouns)):
                noun_start_char = len(text) - len(noun)
                offsets = offset_mapping[bi]  # (L, 2)
                attn = enc["attention_mask"][bi]  # (L,)

                # Find token indices that overlap with the noun span
                noun_token_idx = []
                for ti in range(len(offsets)):
                    if attn[ti].item() == 0:
                        continue
                    ts, te = int(offsets[ti][0]), int(offsets[ti][1])
                    if te > noun_start_char and ts < len(text):
                        noun_token_idx.append(ti)

                if not noun_token_idx:
                    # Fallback: use last non-padding token
                    noun_token_idx = [int(attn.sum().item()) - 1]

                if layer_mode == "all":
                    # Extract from all layers: (n_layers, D)
                    layer_vecs = [
                        hidden_states[li][bi, noun_token_idx, :]
                        .mean(dim=0).cpu().float().numpy()
                        for li in range(len(hidden_states))
                    ]
                    # OPT-350M: embed_proj dim != hidden_size → drop mismatched layers
                    if len({v.shape[0] for v in layer_vecs}) > 1:
                        from collections import Counter
                        modal_dim = Counter(v.shape[0] for v in layer_vecs).most_common(1)[0][0]
                        layer_vecs = [v for v in layer_vecs if v.shape[0] == modal_dim]
                    feats = np.stack(layer_vecs, axis=0)  # (n_layers, D)
                elif layer_mode == "last":
                    li = len(hidden_states) - 1
                    feats = (
                        hidden_states[li][bi, noun_token_idx, :]
                        .mean(dim=0).cpu().float().numpy()
                    )  # (D,)
                else:
                    li = int(layer_mode)
                    feats = (
                        hidden_states[li][bi, noun_token_idx, :]
                        .mean(dim=0).cpu().float().numpy()
                    )  # (D,)

                all_features.append(feats)

    features = np.stack(all_features, axis=0)  # (n_items, n_layers, D) or (n_items, D)

    # Some custom-code models (e.g. ALiBi) produce NaN/Inf activations due to
    # numerical instability.  Replace them with 0 before caching so downstream
    # ridge regression does not crash.
    n_bad = int(np.sum(~np.isfinite(features)))
    if n_bad > 0:
        logger.warning(
            "extract_sentence_features: %d non-finite values in features "
            "(%s, rev=%s) — replacing with 0",
            n_bad, model_name, revision or "main",
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Write cache ───────────────────────────────────────────────────────────
    if cache_dir is not None:
        try:
            np.savez_compressed(cp, features=features)
            logger.debug("Nieuwland features: saved to cache %s", cp.name)
        except Exception as exc:
            logger.warning("Feature cache write failed — %s", exc)

    del model
    try:
        import gc
        import torch
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    return features


# ---------------------------------------------------------------------------
# Ridge regression helpers
# ---------------------------------------------------------------------------

def _build_X(
    features: np.ndarray,   # (n_unique_items, [n_layers,] D)
    meta: pd.DataFrame,     # n_total_trials rows with item_number, condition columns
    items_df: pd.DataFrame, # n_unique_items rows with item_number, condition
    add_lab_covariate: bool,
    layerwise: bool,
) -> np.ndarray | list[np.ndarray]:
    """Map per-item features to per-trial X matrix, optionally add lab covariate.

    Returns
    -------
    If layerwise: list of (n_trials, D+n_labs) arrays, one per layer.
    Else        : (n_trials, D+n_labs) array.
    """
    # Build lookup: (item_number, condition) -> row index in items_df
    item_key = list(zip(
        items_df["item_number"].astype(str),
        items_df["condition"].astype(str),
    ))
    key_to_idx = {k: i for i, k in enumerate(item_key)}

    trial_keys = list(zip(
        meta["item_number"].astype(str),
        meta["condition"].astype(str),
    ))
    trial_idx = np.array([key_to_idx.get(k, 0) for k in trial_keys], dtype=np.int64)

    # Lab one-hot covariate
    if add_lab_covariate and "lab_id" in meta.columns:
        n_labs = int(meta["lab_id"].max()) + 1
        lab_onehot = np.zeros((len(meta), n_labs), dtype=np.float32)
        for i, lid in enumerate(meta["lab_id"].values):
            lab_onehot[i, int(lid)] = 1.0
    else:
        lab_onehot = None

    def _X_for_layer(feat_2d: np.ndarray) -> np.ndarray:
        X = feat_2d[trial_idx]          # (n_trials, D)
        if lab_onehot is not None:
            X = np.hstack([X, lab_onehot])
        return X.astype(np.float32)

    if layerwise:
        n_layers = features.shape[1]
        return [_X_for_layer(features[:, li, :]) for li in range(n_layers)]
    else:
        return _X_for_layer(features)


def _scores_from_predictions(
    Ypred: np.ndarray,   # (n_test, n_channels * n_times)
    Ytest: np.ndarray,   # (n_test, n_channels * n_times)
) -> tuple[np.ndarray, np.ndarray]:
    """Per-output Spearman r and Pearson r."""
    n_out = Ypred.shape[1]
    sr = np.full(n_out, np.nan, dtype=np.float32)
    pr = np.full(n_out, np.nan, dtype=np.float32)
    for j in range(n_out):
        yt, yp = Ytest[:, j], Ypred[:, j]
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            sr[j] = float(_spearmanr_fn(yt, yp).statistic)
            pr[j] = float(pearsonr(yt, yp).statistic)
    return sr, pr


# ---------------------------------------------------------------------------
# Subject selection helper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class NieuwlandErpEncoder:
    """Multi-output ridge encoding: LLM activations → EEG N400 time course.

    Usage
    -----
    cfg = NieuwlandErpConfig(bids_root="data/nieuwland", ...)
    encoder = NieuwlandErpEncoder(cfg)
    result  = encoder.run()
    """

    def __init__(self, cfg: NieuwlandErpConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_epochs(self):
        from ..datasets.nieuwland import NieuwlandEpochsConfig, load_all_epochs, build_unique_items_df
        cfg = self.cfg
        epoch_cfg = NieuwlandEpochsConfig(
            bids_root=cfg.bids_root,
            subjects=cfg.subjects,
            included_in_paper_only=cfg.included_in_paper_only,
            exclude_labs=cfg.exclude_labs,
            target_channels=cfg.target_channels,
            target_sfreq=cfg.target_sfreq,
            epoch_tmin=cfg.epoch_tmin,
            epoch_tmax=cfg.epoch_tmax,
            baseline=cfg.baseline,
            artifact_threshold_uv=cfg.artifact_threshold_uv,
            min_trials=cfg.min_trials,
            epoch_cache_dir=cfg.epoch_cache_dir,
        )
        epochs_result = load_all_epochs(epoch_cfg)
        items_df = build_unique_items_df(epochs_result.meta)
        return epochs_result, items_df

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, items_df: pd.DataFrame) -> np.ndarray:
        cfg = self.cfg
        contexts = items_df["sentence_context"].tolist()
        nouns = items_df["noun"].tolist()
        fcache = Path(cfg.feature_cache_dir) if cfg.feature_cache_dir else None
        return extract_sentence_features(
            contexts,
            nouns,
            cfg.hf_model_name,
            revision=cfg.hf_revision,
            layer_mode=cfg.layer_mode,
            device=cfg.hf_device,
            batch_size=cfg.hf_batch_size,
            max_length=cfg.hf_max_length,
            tokenizer_no_revision=cfg.tokenizer_no_revision,
            cache_dir=fcache,
            n_params=cfg.n_params if cfg.n_params > 0 else None,
            trust_remote_code=cfg.trust_remote_code,
        )

    # ------------------------------------------------------------------
    # Per-subject ridge regression
    # ------------------------------------------------------------------

    def _run_ridge_per_subject(
        self,
        X_layers: list[np.ndarray],   # list[n_layers] of (n_all_trials, D)
        Y: np.ndarray,                # (n_all_trials, n_channels, n_times)
        meta: pd.DataFrame,
        groups: np.ndarray,           # (n_all_trials,) item_number for GroupKFold
        times: np.ndarray,
        channel_names: list[str],
        on_subject_done: tp.Callable[[SubjectScore], None] | None = None,
    ) -> tuple[list[ChannelScore], list[SubjectScore]]:
        """Per-subject ridge regression with incremental saving.

        For each selected subject, fits one Ridge per layer using only that
        subject's ~80 trials with GroupKFold on item_number (same sentence never
        in train and test). All layer-wise metrics are saved to a per-subject
        .npz file immediately after fitting so a SLURM timeout loses at most one
        subject.  On restart, already-saved subjects are loaded from disk and
        skipped.
        """
        import warnings
        from scipy.stats import rankdata

        cfg = self.cfg
        n_all, n_channels, n_times = Y.shape
        n_layers = len(X_layers)
        n_outputs = n_channels * n_times
        alphas = cfg.ridge_alphas
        n400_mask = (times >= cfg.n400_tmin) & (times <= cfg.n400_tmax)

        subject_col = meta["subject"].values.astype(str)
        lab_col     = meta["lab"].values.astype(str) if "lab" in meta.columns else subject_col

        Y_flat = Y.reshape(n_all, n_outputs).astype(np.float32)

        # ── All subjects ──────────────────────────────────────────────────────
        selected = meta["subject"].unique().tolist()
        n_selected = len(selected)

        subj_cache_dir: Path | None = None
        if cfg.out_dir:
            subj_cache_dir = Path(cfg.out_dir) / "subject_timecourses"
            subj_cache_dir.mkdir(parents=True, exist_ok=True)

        # ── Backend ───────────────────────────────────────────────────────────
        try:
            from himalaya.backend import set_backend as _set_backend
            _set_backend("torch_cuda")
            use_himalaya = True
        except Exception as exc:
            logger.warning("himalaya GPU unavailable (%s) — falling back to sklearn", exc)
            use_himalaya = False

        # ── Correlation helpers ───────────────────────────────────────────────
        def _col_spearmanr(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            """Column-wise Spearman r: (n, k) → (k,). Correlates over n."""
            if y_pred.shape[0] < 3:
                return np.full(y_pred.shape[1], np.nan, dtype=np.float32)
            pr = np.apply_along_axis(rankdata, 0, y_pred)
            tr = np.apply_along_axis(rankdata, 0, y_true)
            pr_c = pr - pr.mean(0); tr_c = tr - tr.mean(0)
            num = (pr_c * tr_c).sum(0)
            denom = np.sqrt((pr_c ** 2).sum(0) * (tr_c ** 2).sum(0))
            return np.where(denom > 0, (num / denom).astype(np.float32), np.nan)

        def _col_pearsonr(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            if y_pred.shape[0] < 3:
                return np.full(y_pred.shape[1], np.nan, dtype=np.float32)
            pc = y_pred - y_pred.mean(0); tc = y_true - y_true.mean(0)
            num = (pc * tc).sum(0)
            denom = np.sqrt((pc ** 2).sum(0) * (tc ** 2).sum(0))
            return np.where(denom > 0, (num / denom).astype(np.float32), np.nan)

        def _row_spearmanr(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            """Row-wise Spearman r: (n, t) → (n,). Correlates over t (timepoints)."""
            n, t = y_pred.shape
            if t < 3:
                return np.full(n, np.nan, dtype=np.float32)
            op = np.argsort(y_pred, axis=1); ot = np.argsort(y_true, axis=1)
            rp = np.empty((n, t), dtype=np.float32); rt = np.empty((n, t), dtype=np.float32)
            row_idx = np.arange(n)[:, None]; ranks = np.arange(1, t + 1, dtype=np.float32)
            rp[row_idx, op] = ranks; rt[row_idx, ot] = ranks
            rp_c = rp - rp.mean(1, keepdims=True); rt_c = rt - rt.mean(1, keepdims=True)
            num = (rp_c * rt_c).sum(1)
            denom = np.sqrt((rp_c ** 2).sum(1) * (rt_c ** 2).sum(1))
            return np.where(denom > 0, num / denom, np.nan).astype(np.float32)

        def _row_pearsonr(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            n, t = y_pred.shape
            if t < 3:
                return np.full(n, np.nan, dtype=np.float32)
            pc = y_pred - y_pred.mean(1, keepdims=True)
            tc = y_true - y_true.mean(1, keepdims=True)
            num = (pc * tc).sum(1)
            denom = np.sqrt((pc ** 2).sum(1) * (tc ** 2).sum(1))
            return np.where(denom > 0, num / denom, np.nan).astype(np.float32)

        # ── Accumulation arrays (indexed by position in selected list) ────────
        subj_layer_tc_sr  = np.full((n_selected, n_layers, n_channels, n_times), np.nan, dtype=np.float32)
        subj_layer_tc_pr  = np.full_like(subj_layer_tc_sr, np.nan)
        subj_layer_n400_sr = np.full((n_selected, n_layers), np.nan, dtype=np.float32)
        subj_layer_n400_pr = np.full_like(subj_layer_n400_sr, np.nan)
        # perchannel and waveform scores at every layer (needed to evaluate at global best later)
        subj_layer_pc_sr  = np.full((n_selected, n_layers), np.nan, dtype=np.float32)
        subj_layer_pc_pr  = np.full_like(subj_layer_pc_sr, np.nan)
        subj_layer_wf_sr  = np.full((n_selected, n_layers), np.nan, dtype=np.float32)
        subj_layer_wf_pr  = np.full_like(subj_layer_wf_sr, np.nan)
        # per-channel pc_sr at every layer: (n_selected, n_layers, n_channels)
        subj_layer_pc_sr_ch = np.full((n_selected, n_layers, n_channels), np.nan, dtype=np.float32)
        subj_n_trials = np.zeros(n_selected, dtype=np.int64)
        subj_labs = ["unknown"] * n_selected

        logger.info(
            "Nieuwland per-subject ridge: %d/%d subjects selected, %d layers, "
            "%d ch × %d t, himalaya=%s",
            n_selected, len(np.unique(subject_col)), n_layers, n_channels, n_times, use_himalaya,
        )

        # ── Per-subject loop ──────────────────────────────────────────────────
        for si, subj in enumerate(selected):
            mask = subject_col == subj
            idx  = np.where(mask)[0]
            n_s  = len(idx)
            subj_n_trials[si] = n_s
            subj_labs[si] = str(lab_col[idx[0]]) if n_s else "unknown"

            subj_safe = subj.replace("/", "_").replace(" ", "_")
            tc_path = (subj_cache_dir / f"{subj_safe}.npz") if subj_cache_dir else None

            # ── Try loading from cache ────────────────────────────────────────
            if tc_path is not None and tc_path.exists():
                try:
                    d = np.load(tc_path, allow_pickle=True)
                    subj_layer_tc_sr[si]   = d["layer_tc_sr"]
                    subj_layer_tc_pr[si]   = d["layer_tc_pr"]
                    subj_layer_n400_sr[si] = d["layer_n400_sr"]
                    subj_layer_n400_pr[si] = d["layer_n400_pr"]
                    subj_layer_pc_sr[si]   = d["layer_pc_sr"]
                    subj_layer_pc_pr[si]   = d["layer_pc_pr"]
                    subj_layer_wf_sr[si]   = d["layer_wf_sr"]
                    subj_layer_wf_pr[si]   = d["layer_wf_pr"]
                    if "layer_pc_sr_ch" in d:
                        subj_layer_pc_sr_ch[si] = d["layer_pc_sr_ch"]
                    subj_n_trials[si] = int(d["n_trials"])
                    subj_labs[si]     = str(d["lab"])
                    best_li_c = int(np.nanargmax(subj_layer_pc_sr[si]))
                    logger.info(
                        "  [%d/%d] %s (%d trials) — from cache  best_layer=%d  pc_sr=%.4f  wf_sr=%.4f",
                        si + 1, n_selected, subj, subj_n_trials[si],
                        best_li_c,
                        float(subj_layer_pc_sr[si, best_li_c]),
                        float(subj_layer_wf_sr[si, best_li_c]),
                    )
                    # still notify so the by_subject CSV stays complete on restart
                    if on_subject_done is not None:
                        on_subject_done(SubjectScore(
                            subject=subj, lab=subj_labs[si],
                            spearmanr=float(subj_layer_n400_sr[si, best_li_c]),
                            pearsonr=float(subj_layer_n400_pr[si, best_li_c]),
                            n_trials=int(subj_n_trials[si]),
                            perchannel_spearmanr=float(subj_layer_pc_sr[si, best_li_c]),
                            perchannel_pearsonr=float(subj_layer_pc_pr[si, best_li_c]),
                            waveform_spearmanr=float(subj_layer_wf_sr[si, best_li_c]),
                            waveform_pearsonr=float(subj_layer_wf_pr[si, best_li_c]),
                            best_layer=best_li_c,
                        ))
                    continue
                except Exception as exc:
                    logger.warning("%s: cache load failed (%s) — refitting", subj, exc)

            if n_s < 3:
                logger.warning("  [%d/%d] %s: only %d trials — skipping", si + 1, n_selected, subj, n_s)
                continue

            # ── Per-subject GroupKFold ────────────────────────────────────────
            groups_s   = groups[idx]
            n_splits_s = min(cfg.n_folds, len(np.unique(groups_s)))
            gkf_s      = GroupKFold(n_splits=n_splits_s)
            splits_s   = list(gkf_s.split(X_layers[0][idx], Y_flat[idx], groups_s))

            # Scale X using this subject's own first training fold
            tr0_s = splits_s[0][0]
            X_scaled_s: list[np.ndarray] = []
            for li in range(n_layers):
                Xl = X_layers[li][idx].astype(np.float32)
                if cfg.scale and len(tr0_s) > 1:
                    sc = RobustScaler()
                    sc.fit(Xl[tr0_s])
                    Xl = sc.transform(Xl)
                X_scaled_s.append(Xl)

            Y_s = Y_flat[idx]   # (n_s, n_outputs)

            # ── Fit per layer, collect OOF for this subject ───────────────────
            layer_oof_s: dict[int, np.ndarray] = {}
            all_fold_alphas: list[float] = []   # across all layers — for INFO summary
            for li in range(n_layers):
                oof_s = np.full((n_s, n_outputs), np.nan, dtype=np.float32)
                Xl_s  = X_scaled_s[li]
                fold_alphas: list[float] = []
                for fi, (tr_idx, te_idx) in enumerate(splits_s):
                    Xtr, Xte = Xl_s[tr_idx], Xl_s[te_idx]
                    Ytr = Y_s[tr_idx]
                    if use_himalaya and len(Xtr) > 4:
                        from himalaya.ridge import RidgeCV as _HimRCV
                        from himalaya.backend import get_backend as _gb
                        _b = _gb()
                        n_inner = max(2, min(3, len(Xtr) // 5))
                        hm = _HimRCV(alphas=alphas, cv=n_inner)
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", message="Solving ridge is slower",
                                category=UserWarning,
                            )
                            hm.fit(_b.asarray(Xtr), _b.asarray(Ytr))
                        oof_s[te_idx] = _b.to_numpy(hm.predict(_b.asarray(Xte)))
                        best_a = _b.to_numpy(hm.best_alphas_)
                        fold_alphas.append(float(np.median(best_a)))
                        unique_a, counts_a = np.unique(best_a, return_counts=True)
                        logger.debug(
                            "    %s layer %d fold %d/%d: alpha median=%.3g  "
                            "min=%.3g  max=%.3g  dist=%s",
                            subj, li, fi + 1, n_splits_s,
                            float(np.median(best_a)), float(np.min(best_a)), float(np.max(best_a)),
                            {f"{a:.3g}": int(c) for a, c in zip(unique_a.tolist(), counts_a.tolist())},
                        )
                    else:
                        from sklearn.linear_model import RidgeCV as _SkRCV
                        n_inner = max(2, min(3, len(Xtr) // 5))
                        hm = _SkRCV(alphas=alphas, cv=n_inner)
                        hm.fit(Xtr, Ytr)
                        oof_s[te_idx] = hm.predict(Xte).astype(np.float32)
                        fold_alphas.append(float(hm.alpha_))
                        logger.debug(
                            "    %s layer %d fold %d/%d: alpha=%.3g",
                            subj, li, fi + 1, n_splits_s, hm.alpha_,
                        )
                layer_oof_s[li] = oof_s
                all_fold_alphas.extend(fold_alphas)
                if fold_alphas:
                    logger.debug(
                        "  %s layer %d/%d: alpha median over folds=%.3g  (range %.3g–%.3g)",
                        subj, li + 1, n_layers,
                        float(np.median(fold_alphas)), min(fold_alphas), max(fold_alphas),
                    )

            if all_fold_alphas:
                logger.info(
                    "  %s alpha summary (%d layers × %d folds): "
                    "median=%.3g  min=%.3g  max=%.3g",
                    subj, n_layers, n_splits_s,
                    float(np.median(all_fold_alphas)),
                    min(all_fold_alphas),
                    max(all_fold_alphas),
                )

            # ── Compute all layer-wise metrics for this subject ───────────────
            layer_pc_sr_s = np.full(n_layers, np.nan, dtype=np.float32)
            layer_pc_pr_s = np.full_like(layer_pc_sr_s, np.nan)
            layer_wf_sr_s = np.full(n_layers, np.nan, dtype=np.float32)
            layer_wf_pr_s = np.full_like(layer_wf_sr_s, np.nan)

            for li in range(n_layers):
                pred_s = layer_oof_s[li]          # (n_s, n_outputs)
                sr_li  = _col_spearmanr(pred_s, Y_s).reshape(n_channels, n_times)
                pr_li  = _col_pearsonr(pred_s, Y_s).reshape(n_channels, n_times)
                subj_layer_tc_sr[si, li] = sr_li
                subj_layer_tc_pr[si, li] = pr_li
                if n400_mask.any():
                    subj_layer_n400_sr[si, li] = float(np.nanmean(sr_li[:, n400_mask]))
                    subj_layer_n400_pr[si, li] = float(np.nanmean(pr_li[:, n400_mask]))

                pred_3d = pred_s.reshape(n_s, n_channels, n_times)
                Y_3d    = Y_s.reshape(n_s, n_channels, n_times)

                if n_s >= 3 and n400_mask.any():
                    pred_ch = pred_3d[:, :, n400_mask].mean(2)
                    Y_ch    = Y_3d[:, :, n400_mask].mean(2)
                    pc_sr_ch = _col_spearmanr(pred_ch, Y_ch)   # (n_channels,)
                    pc_pr_ch = _col_pearsonr(pred_ch, Y_ch)
                    subj_layer_pc_sr_ch[si, li] = pc_sr_ch
                    layer_pc_sr_s[li] = float(np.nanmean(pc_sr_ch))
                    layer_pc_pr_s[li] = float(np.nanmean(pc_pr_ch))

                if n_times >= 3:
                    wf_sr = np.zeros((n_s, n_channels), dtype=np.float32)
                    wf_pr = np.zeros((n_s, n_channels), dtype=np.float32)
                    for ci in range(n_channels):
                        wf_sr[:, ci] = _row_spearmanr(pred_3d[:, ci, :], Y_3d[:, ci, :])
                        wf_pr[:, ci] = _row_pearsonr(pred_3d[:, ci, :], Y_3d[:, ci, :])
                    layer_wf_sr_s[li] = float(np.nanmean(wf_sr))
                    layer_wf_pr_s[li] = float(np.nanmean(wf_pr))

            subj_layer_pc_sr[si] = layer_pc_sr_s
            subj_layer_pc_pr[si] = layer_pc_pr_s
            subj_layer_wf_sr[si] = layer_wf_sr_s
            subj_layer_wf_pr[si] = layer_wf_pr_s

            # ── Save per-subject .npz (atomic write via .tmp) ─────────────────
            if tc_path is not None:
                tmp = tc_path.parent / (tc_path.stem + ".tmp.npz")
                try:
                    np.savez_compressed(
                        tmp,
                        layer_tc_sr=subj_layer_tc_sr[si],   # (n_layers, n_ch, n_t) — full window
                        layer_tc_pr=subj_layer_tc_pr[si],
                        layer_n400_sr=subj_layer_n400_sr[si],
                        layer_n400_pr=subj_layer_n400_pr[si],
                        layer_pc_sr=layer_pc_sr_s,          # (n_layers,)
                        layer_pc_pr=layer_pc_pr_s,
                        layer_wf_sr=layer_wf_sr_s,
                        layer_wf_pr=layer_wf_pr_s,
                        layer_pc_sr_ch=subj_layer_pc_sr_ch[si],  # (n_layers, n_channels)
                        n_trials=np.array(n_s),
                        lab=np.array(subj_labs[si]),
                        times=times,
                        channel_names=np.array(channel_names),
                    )
                    tmp.rename(tc_path)
                except Exception as exc:
                    logger.warning("%s: could not save cache (%s)", subj, exc)

            # ── Log progress and call callback ────────────────────────────────
            best_li_s = int(np.nanargmax(layer_pc_sr_s))   # per-subject best: argmax of pc_sr
            ss_partial = SubjectScore(
                subject=subj, lab=subj_labs[si],
                spearmanr=float(subj_layer_n400_sr[si, best_li_s]),
                pearsonr=float(subj_layer_n400_pr[si, best_li_s]),
                n_trials=int(n_s),
                perchannel_spearmanr=float(layer_pc_sr_s[best_li_s]),
                perchannel_pearsonr=float(layer_pc_pr_s[best_li_s]),
                waveform_spearmanr=float(layer_wf_sr_s[best_li_s]),
                waveform_pearsonr=float(layer_wf_pr_s[best_li_s]),
                best_layer=best_li_s,
            )
            logger.info(
                "  [%d/%d] %s (%d trials): best_layer=%d  pc_sr=%.4f  wf_sr=%.4f",
                si + 1, n_selected, subj, n_s,
                best_li_s, ss_partial.perchannel_spearmanr, ss_partial.waveform_spearmanr,
            )
            if on_subject_done is not None:
                on_subject_done(ss_partial)

        # ── Aggregate channel scores (mean over selected subjects) ────────────
        mean_tc_sr = np.nanmean(subj_layer_tc_sr, axis=0)  # (n_layers, n_ch, n_t)
        mean_tc_pr = np.nanmean(subj_layer_tc_pr, axis=0)

        channel_scores: list[ChannelScore] = []
        for ci, ch in enumerate(channel_names):
            layer_tcs: list[ChannelTimeResolvedScore] = []
            layer_n400_sr_l: list[float] = []
            layer_n400_pr_l: list[float] = []
            for li in range(n_layers):
                sr_t = mean_tc_sr[li, ci]
                pr_t = mean_tc_pr[li, ci]
                n400_sr_v = float(np.nanmean(sr_t[n400_mask])) if n400_mask.any() else float("nan")
                n400_pr_v = float(np.nanmean(pr_t[n400_mask])) if n400_mask.any() else float("nan")
                layer_n400_sr_l.append(n400_sr_v)
                layer_n400_pr_l.append(n400_pr_v)
                layer_tcs.append(ChannelTimeResolvedScore(
                    channel=ch, times=times,
                    spearmanr=sr_t, pearsonr=pr_t,
                    spearmanr_folds=np.array([]),
                ))
            best_li = int(np.nanargmax(layer_n400_sr_l)) if layer_n400_sr_l else 0
            channel_scores.append(ChannelScore(
                name=ch, best_layer=best_li,
                spearmanr=layer_n400_sr_l[best_li] if layer_n400_sr_l else float("nan"),
                pearsonr=layer_n400_pr_l[best_li] if layer_n400_pr_l else float("nan"),
                time_course=layer_tcs[best_li],
                layer_spearmanr=layer_n400_sr_l, layer_pearsonr=layer_n400_pr_l,
                layer_time_courses=layer_tcs,
            ))

        # ── Final per-subject scores — each at their own best layer ──────────
        # Voted global best = mode of per-subject best layers (argmax of pc_sr per subject)
        subj_best_layers = np.array([
            int(np.nanargmax(subj_layer_pc_sr[si])) if not np.all(np.isnan(subj_layer_pc_sr[si])) else 0
            for si in range(n_selected)
        ], dtype=np.int64)
        unique_l, counts_l = np.unique(subj_best_layers, return_counts=True)
        voted_best_layer = int(unique_l[np.argmax(counts_l)])
        vote_dist = {int(l): int(c) for l, c in zip(unique_l.tolist(), counts_l.tolist())}
        logger.info(
            "Global best layer (voted): %d  distribution=%s",
            voted_best_layer, vote_dist,
        )

        subject_scores: list[SubjectScore] = []
        for si, subj in enumerate(selected):
            bli = int(subj_best_layers[si])
            subject_scores.append(SubjectScore(
                subject=subj, lab=subj_labs[si],
                spearmanr=float(subj_layer_n400_sr[si, bli]),
                pearsonr=float(subj_layer_n400_pr[si, bli]),
                n_trials=int(subj_n_trials[si]),
                perchannel_spearmanr=float(subj_layer_pc_sr[si, bli]),
                perchannel_pearsonr=float(subj_layer_pc_pr[si, bli]),
                waveform_spearmanr=float(subj_layer_wf_sr[si, bli]),
                waveform_pearsonr=float(subj_layer_wf_pr[si, bli]),
                best_layer=bli,
            ))

        # Per-layer pc_sr and wf_sr: mean over subjects at each layer
        layer_pc_sr    = np.nanmean(subj_layer_pc_sr,    axis=0)   # (n_layers,)
        layer_wf_sr    = np.nanmean(subj_layer_wf_sr,    axis=0)
        layer_pc_sr_ch = np.nanmean(subj_layer_pc_sr_ch, axis=0)   # (n_layers, n_channels)

        return channel_scores, subject_scores, layer_pc_sr, layer_wf_sr, layer_pc_sr_ch

    # ------------------------------------------------------------------
    # Option 2 — item-mean regression
    # ------------------------------------------------------------------

    def _run_ridge_item_mean(
        self,
        X_layers: list[np.ndarray],   # list[n_layers] of (n_all_trials, D)
        Y: np.ndarray,                # (n_all_trials, n_channels, n_times)
        meta: pd.DataFrame,
        times: np.ndarray,
        channel_names: list[str],
    ) -> list[ChannelScore]:
        """Ridge on item-averaged EEG (Y averaged across subjects per sentence).

        Collapses 22k trials → 80 unique items before regression, giving much
        higher SNR in Y at the cost of fewer data points.  Results are directly
        comparable to Michaelov & Bergen (2022).
        """
        import warnings
        from scipy.stats import rankdata

        cfg = self.cfg
        n_all, n_channels, n_times = Y.shape
        n_layers = len(X_layers)
        n_outputs = n_channels * n_times
        alphas = cfg.ridge_alphas
        n400_mask = (times >= cfg.n400_tmin) & (times <= cfg.n400_tmax)

        item_numbers = meta["item_number"].values
        unique_items = np.array(sorted(np.unique(item_numbers).tolist()))
        n_items = len(unique_items)
        item_to_pos = {it: i for i, it in enumerate(unique_items)}

        Y_flat_all = Y.reshape(n_all, n_outputs).astype(np.float32)

        # Average Y across subjects for each item
        Y_item = np.zeros((n_items, n_outputs), dtype=np.float32)
        counts = np.zeros(n_items, dtype=np.int64)
        for t in range(n_all):
            p = item_to_pos[item_numbers[t]]
            Y_item[p] += Y_flat_all[t]
            counts[p] += 1
        Y_item /= np.maximum(counts[:, np.newaxis], 1)

        # X_unique: take the first occurrence per item (features are identical
        # for all subjects who saw the same sentence)
        X_unique_layers: list[np.ndarray] = []
        for li in range(n_layers):
            Xl = X_layers[li].astype(np.float32)
            X_u = np.empty((n_items, Xl.shape[1]), dtype=np.float32)
            seen = np.zeros(n_items, dtype=bool)
            for t in range(n_all):
                p = item_to_pos[item_numbers[t]]
                if not seen[p]:
                    X_u[p] = Xl[t]
                    seen[p] = True
            X_unique_layers.append(X_u)

        # GroupKFold on items (each item is its own group → leave-items-out CV)
        n_splits = min(cfg.n_folds, n_items)
        groups_unique = unique_items
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X_unique_layers[0], Y_item, groups_unique))

        logger.info(
            "Nieuwland item-mean ridge: %d items, %d layers, "
            "%d ch × %d t, %d folds",
            n_items, n_layers, n_channels, n_times, n_splits,
        )

        # Scale X on first fold's training data
        tr0 = splits[0][0]
        X_scaled: list[np.ndarray] = []
        for li in range(n_layers):
            Xu = X_unique_layers[li]
            if cfg.scale and len(tr0) > 1:
                sc = RobustScaler()
                sc.fit(Xu[tr0])
                Xu = sc.transform(Xu)
            X_scaled.append(Xu)

        try:
            from himalaya.backend import set_backend as _set_backend
            _set_backend("torch_cuda")
            use_himalaya = True
        except Exception:
            use_himalaya = False

        # Fit each layer, collect OOF predictions over the 80 items
        item_oof: dict[int, np.ndarray] = {}
        for li in range(n_layers):
            oof = np.full((n_items, n_outputs), np.nan, dtype=np.float32)
            Xu = X_scaled[li]
            for tr_idx, te_idx in splits:
                Xtr, Xte = Xu[tr_idx], Xu[te_idx]
                Ytr = Y_item[tr_idx]
                if use_himalaya and len(Xtr) > 4:
                    from himalaya.ridge import RidgeCV as _HimRCV
                    from himalaya.backend import get_backend as _gb
                    _b = _gb()
                    n_inner = max(2, min(5, len(Xtr) // 5))
                    hm = _HimRCV(alphas=alphas, cv=n_inner)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message="Solving ridge is slower",
                            category=UserWarning,
                        )
                        hm.fit(_b.asarray(Xtr), _b.asarray(Ytr))
                    oof[te_idx] = _b.to_numpy(hm.predict(_b.asarray(Xte)))
                else:
                    from sklearn.linear_model import RidgeCV as _SkRCV
                    n_inner = max(2, min(5, len(Xtr) // 5))
                    hm = _SkRCV(alphas=alphas, cv=n_inner)
                    hm.fit(Xtr, Ytr)
                    oof[te_idx] = hm.predict(Xte).astype(np.float32)
            item_oof[li] = oof
            logger.debug("  [item-mean] layer %d/%d fitted", li + 1, n_layers)

        # Compute per-channel Spearman/Pearson r over the 80 OOF items
        def _col_spearmanr(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            if y_pred.shape[0] < 3:
                return np.full(y_pred.shape[1], np.nan, dtype=np.float32)
            pr = np.apply_along_axis(rankdata, 0, y_pred)
            tr = np.apply_along_axis(rankdata, 0, y_true)
            pr_c = pr - pr.mean(0); tr_c = tr - tr.mean(0)
            num = (pr_c * tr_c).sum(0)
            denom = np.sqrt((pr_c ** 2).sum(0) * (tr_c ** 2).sum(0))
            return np.where(denom > 0, (num / denom).astype(np.float32), np.nan)

        def _col_pearsonr(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            if y_pred.shape[0] < 3:
                return np.full(y_pred.shape[1], np.nan, dtype=np.float32)
            pc = y_pred - y_pred.mean(0); tc = y_true - y_true.mean(0)
            num = (pc * tc).sum(0)
            denom = np.sqrt((pc ** 2).sum(0) * (tc ** 2).sum(0))
            return np.where(denom > 0, (num / denom).astype(np.float32), np.nan)

        # layer × channel × time scores over the 80 items
        layer_tc_sr = np.full((n_layers, n_channels, n_times), np.nan, dtype=np.float32)
        layer_tc_pr = np.full_like(layer_tc_sr, np.nan)
        for li in range(n_layers):
            sr = _col_spearmanr(item_oof[li], Y_item).reshape(n_channels, n_times)
            pr = _col_pearsonr(item_oof[li], Y_item).reshape(n_channels, n_times)
            layer_tc_sr[li] = sr
            layer_tc_pr[li] = pr

        channel_scores: list[ChannelScore] = []
        for ci, ch in enumerate(channel_names):
            layer_n400_sr: list[float] = []
            layer_n400_pr: list[float] = []
            layer_tcs: list[ChannelTimeResolvedScore] = []
            for li in range(n_layers):
                sr_t = layer_tc_sr[li, ci]
                pr_t = layer_tc_pr[li, ci]
                n400_sr = float(np.nanmean(sr_t[n400_mask])) if n400_mask.any() else float("nan")
                n400_pr = float(np.nanmean(pr_t[n400_mask])) if n400_mask.any() else float("nan")
                layer_n400_sr.append(n400_sr)
                layer_n400_pr.append(n400_pr)
                layer_tcs.append(ChannelTimeResolvedScore(
                    channel=ch, times=times,
                    spearmanr=sr_t, pearsonr=pr_t,
                    spearmanr_folds=np.array([]),
                ))
            best_li = int(np.nanargmax(layer_n400_sr)) if layer_n400_sr else 0
            channel_scores.append(ChannelScore(
                name=ch, best_layer=best_li,
                spearmanr=layer_n400_sr[best_li] if layer_n400_sr else float("nan"),
                pearsonr=layer_n400_pr[best_li] if layer_n400_pr else float("nan"),
                time_course=layer_tcs[best_li],
                layer_spearmanr=layer_n400_sr, layer_pearsonr=layer_n400_pr,
                layer_time_courses=layer_tcs,
            ))

        if cfg.save_word_predictions:
            self._save_item_predictions(
                channel_names=channel_names, channel_scores=channel_scores,
                item_oof=item_oof, Y_item=Y_item, unique_items=unique_items,
                n_channels=n_channels, n_times=n_times, meta=meta, times=times,
            )

        return channel_scores

    def _save_item_predictions(
        self,
        *,
        channel_names: list[str],
        channel_scores: list[ChannelScore],
        item_oof: dict[int, np.ndarray],   # {layer: (n_items, n_channels*n_times)}
        Y_item: np.ndarray,                # (n_items, n_channels*n_times)
        unique_items: np.ndarray,          # (n_items,) item_number
        n_channels: int,
        n_times: int,
        meta: pd.DataFrame,
        times: np.ndarray,
    ) -> None:
        """Save each channel's best-layer out-of-fold item-mean ERP prediction
        AND the actual item-mean ERP (needed to score any post-hoc word/item
        subset, e.g. an easy/hard DLT-cost split — a prediction alone isn't
        scoreable). Mirrors podcast.py/broderick.py's save_word_predictions,
        keyed by item_number instead of a transcript word_idx.
        """
        cfg = self.cfg

        oof_stack = np.stack([
            item_oof[cs.best_layer][:, ci * n_times:(ci + 1) * n_times]
            for ci, cs in enumerate(channel_scores)
        ], axis=0).astype(np.float32)  # (n_channels, n_items, n_times)
        y_stack = np.stack([
            Y_item[:, ci * n_times:(ci + 1) * n_times]
            for ci in range(n_channels)
        ], axis=0).astype(np.float32)  # (n_channels, n_items, n_times)

        # Representative sentence_context/noun per item_number (first occurrence —
        # DLT cost at the critical noun is structural, so it doesn't depend on
        # which of the two conditions' noun happens to be picked; see module notes).
        first_per_item = (
            meta[["item_number", "sentence_context", "noun"]]
            .dropna(subset=["item_number"])
            .drop_duplicates(subset=["item_number"], keep="first")
            .set_index("item_number")
        )
        contexts = [str(first_per_item["sentence_context"].get(it, "")) for it in unique_items]
        nouns = [str(first_per_item["noun"].get(it, "")) for it in unique_items]

        out_dir = word_predictions_dir(cfg.out_dir, "nieuwland_word_predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_revision = (cfg.hf_revision or "main").replace("/", "_")
        safe_model = cfg.hf_model_name.replace("/", "_")
        out_path = out_dir / f"{safe_model}_{model_revision}.npz"
        np.savez_compressed(
            out_path,
            oof_predictions=oof_stack,
            y_true=y_stack,
            channel_names=np.array(channel_names),
            item_number=np.asarray(unique_items),
            sentence_context=np.array(contexts),
            noun=np.array(nouns),
            times=times,
            model=np.array(cfg.hf_model_name),
            revision=np.array(cfg.hf_revision or "main"),
            n_params=np.array(cfg.n_params),
        )
        logger.info("  Saved item-level predictions (%s) → %s", oof_stack.shape, out_path)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> NieuwlandErpResult:
        cfg = self.cfg
        logger.info(
            "NieuwlandErpEncoder: %s (rev=%s)", cfg.hf_model_name, cfg.hf_revision or "main"
        )

        # ── Load epochs ───────────────────────────────────────────────────────
        epochs_result, items_df = self._load_epochs()
        Y = epochs_result.Y          # (n_trials, n_channels, n_times)
        meta = epochs_result.meta
        times = epochs_result.times
        channel_names = epochs_result.channel_names

        # ── Extract LLM features ──────────────────────────────────────────────
        features = self._extract_features(items_df)
        # features: (n_unique_items, n_layers, D) or (n_unique_items, D)
        layerwise = features.ndim == 3
        n_layers = features.shape[1] if layerwise else 1
        if not layerwise:
            features = features[:, np.newaxis, :]  # (n, 1, D)

        # ── Build per-trial X (one per layer) ────────────────────────────────
        # No lab covariate: each subject has their own regression, so between-lab
        # amplitude differences are absorbed at the subject level.
        X_layers = _build_X(
            features, meta, items_df,
            add_lab_covariate=False,
            layerwise=True,
        )

        # Groups for item-based GroupKFold (split on sentence, not subject)
        groups = meta["item_number"].astype(float).fillna(-1).values.astype(np.int64)

        # ── Per-subject ridge regression ─────────────────────────────────────
        # Append one CSV row per subject as soon as it finishes.  On restart,
        # subjects already in the CSV are skipped to avoid duplicate rows.
        by_subject_csv: Path | None = None
        written_subjects: set[str] = set()
        revision_key = cfg.hf_revision or "main"
        if cfg.out_dir:
            by_subject_csv = Path(cfg.out_dir) / "nieuwland_by_subject.csv"
            by_subject_csv.parent.mkdir(parents=True, exist_ok=True)
            if by_subject_csv.exists():
                try:
                    _existing = pd.read_csv(by_subject_csv)
                    if "revision" in _existing.columns and "subject" in _existing.columns:
                        written_subjects = set(
                            _existing.loc[_existing["revision"] == revision_key, "subject"]
                        )
                except Exception:
                    pass

        def _on_subject_done(ss: SubjectScore) -> None:
            if by_subject_csv is None or ss.subject in written_subjects:
                return
            _csv_append(by_subject_csv, {
                "model": cfg.hf_model_name,
                "revision": revision_key,
                "subject": ss.subject,
                "lab": ss.lab,
                "n_trials": ss.n_trials,
                "best_layer": ss.best_layer,
                "perchannel_spearmanr": ss.perchannel_spearmanr,
                "perchannel_pearsonr": ss.perchannel_pearsonr,
                "waveform_spearmanr": ss.waveform_spearmanr,
                "waveform_pearsonr": ss.waveform_pearsonr,
            })
            written_subjects.add(ss.subject)

        channel_scores, subject_scores, layer_pc_sr, layer_wf_sr, layer_pc_sr_ch = self._run_ridge_per_subject(
            X_layers, Y, meta, groups, times, channel_names,
            on_subject_done=_on_subject_done,
        )

        item_mean_channel_scores = self._run_ridge_item_mean(
            X_layers, Y, meta, times, channel_names,
        )

        return NieuwlandErpResult(
            model_name=cfg.hf_model_name,
            revision=cfg.hf_revision,
            channel_scores=channel_scores,
            subject_scores=subject_scores,
            item_mean_channel_scores=item_mean_channel_scores,
            n_subjects=epochs_result.n_subjects,
            n_trials=len(meta),
            n_layers=n_layers,
            times=times,
            channel_names=channel_names,
            layer_pc_sr=layer_pc_sr,
            layer_wf_sr=layer_wf_sr,
            layer_pc_sr_ch=layer_pc_sr_ch,
        )


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

def result_to_checkpoint_row(
    result: NieuwlandErpResult,
    *,
    step: int = 0,
    tokens_seen: int = 0,
    n_params: float = 0.0,
) -> dict[str, tp.Any]:
    """One summary row per checkpoint for nieuwland_checkpoint_scores.csv."""
    row: dict[str, tp.Any] = {
        "model": result.model_name,
        "revision": result.revision or "main",
        "step": step,
        "tokens_seen": tokens_seen,
        "n_params": n_params,
        "n_subjects": result.n_subjects,
        "n_trials": result.n_trials,
        "n_layers": result.n_layers,
        "best_layer": result.best_layer,
        # Metric 3 — perchannel: mean(subj) of mean(ch) spearmanr(N400_mean(pred), N400_mean(Y)) over sentences
        "mean_subject_perchannel_spearmanr": result.mean_subject_perchannel_spearmanr,
        "mean_subject_perchannel_pearsonr": result.mean_subject_perchannel_pearsonr,
        # Metric 4 — waveform: mean(subj) of mean(trials) of mean(ch) of spearmanr(pred[t], Y[t]) over timepoints
        "mean_subject_waveform_spearmanr": result.mean_subject_waveform_spearmanr,
        "mean_subject_waveform_pearsonr": result.mean_subject_waveform_pearsonr,
        # Option 2 — item-mean regression (Y averaged across subjects per item)
        "item_mean_best_layer": result.item_mean_best_layer,
        "item_mean_spearmanr": result.item_mean_spearmanr,
        "item_mean_pearsonr": result.item_mean_pearsonr,
    }
    # Per-channel pc_sr at best (voted) layer — primary per-channel metric
    has_ch_pc = (result.layer_pc_sr_ch.ndim == 2
                 and result.layer_pc_sr_ch.shape[0] > result.best_layer)
    for ci, ch in enumerate(result.channel_names):
        row[f"ch_{ch}_pc_sr"] = (
            float(result.layer_pc_sr_ch[result.best_layer, ci]) if has_ch_pc else float("nan")
        )
    # Per-channel scores — option 1 (trial-level, mean over subjects of timecourse N400 mean)
    for cs in result.channel_scores:
        row[f"ch_{cs.name}_spearmanr"] = cs.spearmanr
        row[f"ch_{cs.name}_pearsonr"] = cs.pearsonr
    # Per-channel scores — option 2 (item-mean)
    for cs in result.item_mean_channel_scores:
        row[f"ch_{cs.name}_item_mean_spearmanr"] = cs.spearmanr
        row[f"ch_{cs.name}_item_mean_pearsonr"] = cs.pearsonr
    return row


def result_to_layerwise_rows(
    result: NieuwlandErpResult,
    *,
    step: int = 0,
    tokens_seen: int = 0,
    n_params: float = 0.0,
) -> list[dict[str, tp.Any]]:
    """One row per layer for nieuwland_checkpoint_layerwise.csv."""
    rows = []
    has_pc  = result.layer_pc_sr.size == result.n_layers
    has_wf  = result.layer_wf_sr.size == result.n_layers
    for li in range(result.n_layers):
        sr_vals = [
            cs.layer_spearmanr[li]
            for cs in result.channel_scores
            if li < len(cs.layer_spearmanr)
        ]
        pr_vals = [
            cs.layer_pearsonr[li]
            for cs in result.channel_scores
            if li < len(cs.layer_pearsonr)
        ]
        rows.append({
            "model": result.model_name,
            "revision": result.revision or "main",
            "step": step,
            "tokens_seen": tokens_seen,
            "n_params": n_params,
            "layer": li,
            "mean_spearmanr": float(np.nanmean(sr_vals)) if sr_vals else float("nan"),
            "mean_pearsonr": float(np.nanmean(pr_vals)) if pr_vals else float("nan"),
            "pc_sr": float(result.layer_pc_sr[li]) if has_pc else float("nan"),
            "wf_sr": float(result.layer_wf_sr[li]) if has_wf else float("nan"),
        })
    return rows


def result_to_channel_rows(
    result: NieuwlandErpResult,
    *,
    step: int = 0,
    tokens_seen: int = 0,
    n_params: float = 0.0,
) -> list[dict[str, tp.Any]]:
    """One row per channel for nieuwland_by_channel.csv."""
    rows = []
    for cs in result.channel_scores:
        rows.append({
            "model": result.model_name,
            "revision": result.revision or "main",
            "step": step,
            "tokens_seen": tokens_seen,
            "n_params": n_params,
            "channel": cs.name,
            "best_layer": cs.best_layer,
            "spearmanr": cs.spearmanr,
            "pearsonr": cs.pearsonr,
            "peak_spearmanr": cs.time_course.peak_spearmanr,
            "peak_time_ms": cs.time_course.peak_time * 1000.0,
            "n400_spearmanr": cs.time_course.n400_spearmanr,
        })
    return rows


def result_to_subject_rows(
    result: NieuwlandErpResult,
    *,
    step: int = 0,
    tokens_seen: int = 0,
    n_params: float = 0.0,
) -> list[dict[str, tp.Any]]:
    """One row per subject for nieuwland_by_subject.csv."""
    rows = []
    for ss in result.subject_scores:
        rows.append({
            "model": result.model_name,
            "revision": result.revision or "main",
            "step": step,
            "tokens_seen": tokens_seen,
            "n_params": n_params,
            "subject": ss.subject,
            "lab": ss.lab,
            "n_trials": ss.n_trials,
            "best_layer": ss.best_layer,
            "perchannel_spearmanr": ss.perchannel_spearmanr,
            "perchannel_pearsonr": ss.perchannel_pearsonr,
            "waveform_spearmanr": ss.waveform_spearmanr,
            "waveform_pearsonr": ss.waveform_pearsonr,
        })
    return rows


def save_time_courses(
    result: NieuwlandErpResult,
    out_path: Path,
) -> None:
    """Save per-layer time-resolved Spearman r curves as .npz."""
    n_layers = result.n_layers
    n_ch = len(result.channel_scores)
    n_t = len(result.times)

    sr_arr = np.full((n_layers, n_ch, n_t), np.nan, dtype=np.float32)
    pr_arr = np.full((n_layers, n_ch, n_t), np.nan, dtype=np.float32)

    for ci, cs in enumerate(result.channel_scores):
        for li in range(n_layers):
            if li < len(cs.layer_time_courses):
                tc = cs.layer_time_courses[li]
                sr_arr[li, ci, :] = tc.spearmanr
                pr_arr[li, ci, :] = tc.pearsonr

    np.savez_compressed(
        out_path,
        spearmanr=sr_arr,          # (n_layers, n_channels, n_times)
        pearsonr=pr_arr,
        times=result.times,
        channel_names=np.array(result.channel_names),
        model=result.model_name,
        revision=result.revision or "main",
    )
