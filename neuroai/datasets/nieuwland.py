"""Nieuwland et al. (2018) multi-site N400 EEG dataset loader.

Loads epoched EEG trials for the critical noun conditions (noun_expected,
noun_unexpected) from the BIDS dataset at data/nieuwland.

Nine UK laboratories recorded 334 participants reading English sentences
word-by-word (RSVP) in a pre-registered replication of DeLong et al. (2005).

Reference
---------
Nieuwland, M.S., et al. (2018). Large-scale replication study reveals a limit
on probabilistic prediction in language comprehension. eLife, 7, e33468.
DOI: 10.7554/eLife.33468 | NEMAR: nm000228
"""

from __future__ import annotations

import hashlib
import logging
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("WARNING")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 6 centro-parietal channels used for N400 analysis (Nieuwland et al. 2018).
N400_CHANNELS: tuple[str, ...] = ("Cz", "C3", "C4", "Pz", "P3", "P4")

# N400 scoring window (seconds, post-stimulus).
N400_TMIN: float = 0.2
N400_TMAX: float = 0.5

# Critical trial types (target nouns only; articles excluded).
NOUN_TRIAL_TYPES: tuple[str, ...] = ("noun_expected", "noun_unexpected")

# GLAS uses BioSemi-128 A1-H16 naming — incompatible with standard 10-20.
# Excluded by default; set exclude_labs=[] to attempt GLAS loading.
DEFAULT_EXCLUDE_LABS: tuple[str, ...] = ("glas",)

# EEG file extensions per lab (BrainVision header or BDF).
_LAB_EXT: dict[str, str] = {
    "birm": ".vhdr",
    "bris": ".vhdr",
    "edin": ".bdf",
    "glas": ".bdf",
    "kent": ".vhdr",
    "lond": ".bdf",
    "oxfo": ".bdf",
    "stir": ".bdf",
    "york": ".vhdr",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NieuwlandEpochsConfig:
    bids_root: str | Path
    subjects: list[str] | None = None
    included_in_paper_only: bool = True
    exclude_labs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_LABS))
    target_channels: list[str] = field(default_factory=lambda: list(N400_CHANNELS))
    target_sfreq: float = 32.0
    epoch_tmin: float = -0.2
    epoch_tmax: float = 0.8
    baseline: tuple[float, float] = (-0.2, 0.0)
    artifact_threshold_uv: float = 100.0
    min_trials: int = 20
    epoch_cache_dir: str | Path | None = None
    # Z-score per subject (removes between-subject amplitude scale) before
    # pooling — the ridge encoder needs this. The pooled-LME path
    # (benchmarks/n400_nieuwland_lme.py) wants raw amplitude instead: its
    # per-subject random intercepts already absorb baseline offsets, so
    # pre-standardizing would double up. Set False for that path.
    zscore: bool = True
    # Downsample the continuous raw signal to target_sfreq before epoching.
    # The ridge encoder needs a fixed, shared sample rate across subjects for
    # its per-lag/per-timepoint analysis. The pooled-LME path only ever reads
    # out ONE scalar per trial (the N400-window mean) — it never needs a
    # specific sample rate, and skipping the resample gives a window mean
    # computed from ~10x more native-rate samples (finer, less noisy) while
    # also skipping resample()'s FFT-based cost entirely. Set False for that
    # path; ridge encoder keeps the default True.
    resample: bool = True


@dataclass
class NieuwlandEpochsResult:
    """Pooled epoched EEG across all subjects."""
    Y: np.ndarray                 # (n_total_trials, n_channels, n_times)
    meta: pd.DataFrame            # per-trial metadata (subject, item_number, condition, lab, …)
    times: np.ndarray             # (n_times,) seconds relative to noun onset
    channel_names: list[str]      # actual channel names found (subset of target)
    n_subjects: int
    sfreq: float


# ---------------------------------------------------------------------------
# Participants helper
# ---------------------------------------------------------------------------

def load_participants(bids_root: str | Path) -> pd.DataFrame:
    """Load participants.tsv from the Nieuwland BIDS root."""
    path = Path(bids_root) / "participants.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["included_in_paper"] = df["included_in_paper"].str.lower().isin(
        ("true", "1", "yes")
    )
    return df


# ---------------------------------------------------------------------------
# EEG file detection
# ---------------------------------------------------------------------------

def _find_eeg_file(bids_root: Path, subject: str) -> Path | None:
    """Return path to the delong task EEG file for one subject (any extension)."""
    eeg_dir = bids_root / subject / "eeg"
    if not eeg_dir.is_dir():
        return None
    for ext in (".bdf", ".vhdr", ".cnt"):
        p = eeg_dir / f"{subject}_task-delong_eeg{ext}"
        if p.is_file():
            return p
    return None


def _read_raw(eeg_path: Path) -> mne.io.BaseRaw:
    """Load raw EEG with MNE, auto-detecting format from extension."""
    ext = eeg_path.suffix.lower()
    if ext == ".bdf":
        raw = mne.io.read_raw_bdf(str(eeg_path), preload=False, verbose=False)
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(eeg_path), preload=False, verbose=False)
    elif ext == ".cnt":
        raw = mne.io.read_raw_cnt(str(eeg_path), preload=False, verbose=False)
    else:
        raise ValueError(f"Unsupported EEG format: {ext}")
    return raw


# ---------------------------------------------------------------------------
# Events loading from BIDS events.tsv
# ---------------------------------------------------------------------------

def _load_events_df(bids_root: Path, subject: str) -> pd.DataFrame:
    """Load the delong task events.tsv for one subject."""
    path = bids_root / subject / "eeg" / f"{subject}_task-delong_events.tsv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["sample"] = pd.to_numeric(df["sample"], errors="coerce")
    df["item_number"] = pd.to_numeric(df["item_number"], errors="coerce")
    df["expected_cloze"] = pd.to_numeric(df["expected_cloze"], errors="coerce")
    df["unexpected_cloze"] = pd.to_numeric(df["unexpected_cloze"], errors="coerce")
    return df


def _noun_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to critical noun trials only."""
    return events_df[events_df["trial_type"].isin(NOUN_TRIAL_TYPES)].copy()


# ---------------------------------------------------------------------------
# Per-subject epoch extraction
# ---------------------------------------------------------------------------

def load_subject_epochs(
    bids_root: Path,
    subject: str,
    *,
    target_channels: list[str],
    target_sfreq: float,
    epoch_tmin: float,
    epoch_tmax: float,
    baseline: tuple[float, float],
    artifact_threshold_uv: float,
    min_trials: int,
    resample: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, list[str]] | None:
    """Epoch EEG for one subject around noun onsets.

    Returns
    -------
    (epochs_np, trial_meta, times, channel_names) or None on failure.
    epochs_np : (n_trials, n_channels, n_times)  float32
    trial_meta: DataFrame with item_number, condition, sentence_context, noun, …
    times     : (n_times,) seconds
    channel_names: list of channel names actually extracted
    """
    eeg_path = _find_eeg_file(bids_root, subject)
    if eeg_path is None:
        logger.debug("%s: no EEG file found", subject)
        return None

    events_df = _load_events_df(bids_root, subject)
    if events_df.empty:
        logger.debug("%s: no events found", subject)
        return None

    noun_df = _noun_events(events_df)
    if len(noun_df) < min_trials:
        logger.debug("%s: only %d noun trials < %d", subject, len(noun_df), min_trials)
        return None

    # ── Load raw, pick channels, resample ────────────────────────────────────
    try:
        raw = _read_raw(eeg_path)
    except Exception as exc:
        logger.warning("%s: failed to read EEG — %s", subject, exc)
        return None

    # Find available target channels
    raw_ch_names = [ch.upper() for ch in raw.ch_names]
    available = [
        raw.ch_names[raw_ch_names.index(ch.upper())]
        for ch in target_channels
        if ch.upper() in raw_ch_names
    ]
    if not available:
        logger.warning(
            "%s: none of target channels %s found (raw has %s…)",
            subject, target_channels, raw.ch_names[:6],
        )
        return None
    if len(available) < len(target_channels):
        logger.warning(
            "%s: only %d/%d target channels available: %s",
            subject, len(available), len(target_channels), available,
        )

    raw.pick(available)
    raw.load_data(verbose=False)

    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, npad="auto", verbose=False)

    # ── Build MNE events array from events.tsv sample column ─────────────────
    sfreq = raw.info["sfreq"]
    # Re-compute samples at target_sfreq from onset times (avoids rounding from
    # original sfreq when sample column was generated at a different rate).
    sample_ids = np.round(noun_df["onset"].values * sfreq).astype(np.int64)
    event_id = 1
    mne_events = np.column_stack([
        sample_ids,
        np.zeros(len(sample_ids), dtype=np.int64),
        np.full(len(sample_ids), event_id, dtype=np.int64),
    ])

    # ── Epoch ─────────────────────────────────────────────────────────────────
    try:
        epochs = mne.Epochs(
            raw,
            mne_events,
            event_id=event_id,
            tmin=epoch_tmin,
            tmax=epoch_tmax,
            baseline=baseline,
            preload=True,
            reject={"eeg": artifact_threshold_uv * 1e-6},
            flat={"eeg": 1e-9},
            verbose=False,
            on_missing="ignore",
        )
    except Exception as exc:
        logger.warning("%s: epoching failed — %s", subject, exc)
        return None

    if len(epochs) < min_trials:
        logger.debug(
            "%s: %d epochs after rejection < %d", subject, len(epochs), min_trials
        )
        return None

    # Map kept epochs back to noun_df rows via the drop log
    kept_mask = np.array([len(log) == 0 for log in epochs.drop_log])
    noun_meta = noun_df.iloc[kept_mask].copy().reset_index(drop=True)

    # ── Extract actual noun text per trial ───────────────────────────────────
    # noun column = the word actually shown (expected_noun for expected, etc.)
    noun_meta["noun"] = noun_meta.apply(_extract_noun, axis=1)

    epochs_np = epochs.get_data(copy=False).astype(np.float32)  # (n, C, T)
    times = epochs.times.astype(np.float32)
    channel_names = epochs.ch_names

    logger.debug(
        "%s: %d/%d trials kept, %d channels, %.0f Hz",
        subject, len(epochs), len(noun_df), len(channel_names), sfreq,
    )
    del raw, epochs
    return epochs_np, noun_meta, times, channel_names


def _extract_noun(row: pd.Series) -> str:
    """Return the noun actually presented to the subject for this trial."""
    if row["trial_type"] == "noun_expected":
        return str(row.get("expected_noun", ""))
    return str(row.get("unexpected_noun", ""))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _epochs_cache_path(cache_dir: Path, subject: str, cfg_hash: str) -> Path:
    return cache_dir / f"{subject}_{cfg_hash}.npz"


def _cfg_hash(cfg: NieuwlandEpochsConfig) -> str:
    key = (
        f"{cfg.target_channels}_{cfg.target_sfreq}_{cfg.epoch_tmin}"
        f"_{cfg.epoch_tmax}_{cfg.baseline}_{cfg.artifact_threshold_uv}"
    )
    # Appended only for zscore=False so the default (zscore=True) hash is
    # byte-identical to before this flag existed — any epoch cache already
    # populated by the ridge encoder stays valid. Raw-amplitude callers
    # (the pooled LME) get their own distinct cache entries instead of
    # silently reinterpreting z-scored data as raw.
    if not cfg.zscore:
        key += "_raw"
    # Same reasoning for resample=False: native-rate epochs have a different
    # times axis and sample count than resampled ones, so they must never
    # share a cache entry with resample=True (default) epochs.
    if not cfg.resample:
        key += "_native"
    return hashlib.md5(key.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Main loader: all subjects pooled
# ---------------------------------------------------------------------------

def load_all_epochs(
    cfg: NieuwlandEpochsConfig,
) -> NieuwlandEpochsResult:
    """Load and pool noun epochs across all subjects.

    Subjects with fewer than ``cfg.min_trials`` kept epochs are skipped.
    EEG data is z-scored per subject before pooling to normalise between-
    subject amplitude differences.

    Returns
    -------
    NieuwlandEpochsResult
        Y         : (n_total_trials, n_channels, n_times) float32, z-scored per subject
        meta      : per-trial DataFrame with columns
                    subject, lab, item_number, condition, sentence_context, noun,
                    expected_cloze, unexpected_cloze, list, included_in_paper
        times     : (n_times,) float32 in seconds
        channel_names : list of str
        n_subjects: int
        sfreq     : float
    """
    bids_root = Path(cfg.bids_root)
    participants = load_participants(bids_root)

    # Filter participants
    if cfg.included_in_paper_only:
        participants = participants[participants["included_in_paper"]]
    if cfg.exclude_labs:
        excl = [e.lower() for e in cfg.exclude_labs]
        participants = participants[~participants["lab"].str.lower().isin(excl)]
    if cfg.subjects is not None:
        participants = participants[
            participants["participant_id"].isin(cfg.subjects)
        ]

    subjects = participants["participant_id"].tolist()
    logger.info(
        "Nieuwland: loading %d subjects (included_in_paper=%s, exclude_labs=%s)",
        len(subjects), cfg.included_in_paper_only, cfg.exclude_labs,
    )

    cache_dir = Path(cfg.epoch_cache_dir) if cfg.epoch_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = _cfg_hash(cfg)

    load_kwargs = dict(
        target_channels=cfg.target_channels,
        target_sfreq=cfg.target_sfreq,
        epoch_tmin=cfg.epoch_tmin,
        epoch_tmax=cfg.epoch_tmax,
        baseline=cfg.baseline,
        artifact_threshold_uv=cfg.artifact_threshold_uv,
        min_trials=cfg.min_trials,
    )

    all_Y: list[np.ndarray] = []
    all_meta: list[pd.DataFrame] = []
    times: np.ndarray | None = None
    channel_names: list[str] | None = None
    n_subjects = 0

    for subj in subjects:
        # ── Cache lookup ──────────────────────────────────────────────────────
        if cache_dir is not None:
            cp = _epochs_cache_path(cache_dir, subj, cfg_hash)
            if cp.is_file():
                try:
                    z = np.load(cp, allow_pickle=True)
                    Y_s = z["Y"]
                    meta_s = pd.DataFrame(z["meta"].tolist())
                    t_s = z["times"]
                    ch_s = list(z["channel_names"])
                    if times is None:
                        times = t_s
                        channel_names = ch_s
                    all_Y.append(Y_s)
                    all_meta.append(meta_s)
                    n_subjects += 1
                    logger.debug("%s: loaded from cache (%d trials)", subj, len(Y_s))
                    continue
                except Exception as exc:
                    logger.warning("%s: cache read failed (%s) — reloading", subj, exc)

        # ── Load from raw BIDS ────────────────────────────────────────────────
        result = load_subject_epochs(bids_root, subj, **load_kwargs)
        if result is None:
            continue
        Y_s, meta_s, t_s, ch_s = result

        # Align to common channel set (first valid subject defines the set)
        if channel_names is None:
            times = t_s
            channel_names = ch_s
        else:
            # Keep only channels present in the common set
            common = [c for c in channel_names if c in ch_s]
            if not common:
                logger.warning("%s: no channel overlap with common set — skipping", subj)
                continue
            if common != channel_names:
                idx = [ch_s.index(c) for c in common]
                Y_s = Y_s[:, idx, :]
                channel_names = common

        # Per-subject z-score across trials (removes between-subject amplitude
        # scale) — skipped when cfg.zscore=False (see NieuwlandEpochsConfig.zscore).
        if cfg.zscore:
            Y_s = _zscore_subject(Y_s)

        # Attach subject-level metadata
        meta_s = meta_s.copy()
        meta_s["subject"] = subj
        prow = participants[participants["participant_id"] == subj].iloc[0]
        meta_s["lab"] = prow["lab"]
        meta_s["included_in_paper"] = prow["included_in_paper"]

        all_Y.append(Y_s)
        all_meta.append(meta_s)
        n_subjects += 1

        # ── Write cache (atomic: write to .tmp then rename) ───────────────────
        if cache_dir is not None:
            cp = _epochs_cache_path(cache_dir, subj, cfg_hash)
            # numpy appends .npz automatically, so temp path must already end in .npz
            tmp = cp.parent / (cp.stem + ".tmp.npz")
            try:
                np.savez_compressed(
                    tmp,
                    Y=Y_s,
                    meta=np.array(meta_s.to_dict("records"), dtype=object),
                    times=t_s,
                    channel_names=np.array(ch_s),
                )
                tmp.rename(cp)
            except Exception as exc:
                logger.warning("%s: cache write failed — %s", subj, exc)
                tmp.unlink(missing_ok=True)

    if not all_Y:
        raise RuntimeError("No subjects loaded — check bids_root and exclude_labs.")

    Y_all = np.concatenate(all_Y, axis=0)
    meta_all = pd.concat(all_meta, ignore_index=True)

    # ── Encode lab as integer (for covariate in regression) ──────────────────
    labs = meta_all["lab"].unique().tolist()
    meta_all["lab_id"] = meta_all["lab"].map({l: i for i, l in enumerate(sorted(labs))})

    logger.info(
        "Nieuwland: %d subjects, %d total trials, %d channels, %.0f Hz",
        n_subjects, len(Y_all), len(channel_names or []), cfg.target_sfreq,
    )
    return NieuwlandEpochsResult(
        Y=Y_all,
        meta=meta_all,
        times=times if times is not None else np.array([]),
        channel_names=channel_names or [],
        n_subjects=n_subjects,
        sfreq=cfg.target_sfreq,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zscore_subject(Y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score epochs along trial axis (axis=0) per channel-time point."""
    mean = Y.mean(axis=0, keepdims=True)
    std = Y.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((Y - mean) / std).astype(np.float32, copy=False)


def build_unique_items_df(meta: pd.DataFrame) -> pd.DataFrame:
    """Extract the unique (item_number, condition, sentence_context, noun) pairs.

    Returns one row per unique (item_number, condition) pair, with columns:
    item_number, condition, sentence_context, noun, expected_cloze,
    unexpected_cloze, item_condition_id (0-based row index — used to index
    into a per-item feature/surprisal array, same convention as the other N400 adapters
    ``item_condition_id``). Used for LLM feature extraction (features are
    shared across subjects).
    """
    cols = ["item_number", "condition", "sentence_context", "noun"]
    present = [c for c in cols if c in meta.columns]
    items = (
        meta[present]
        .drop_duplicates(subset=["item_number", "condition"])
        .sort_values(["item_number", "condition"])
        .reset_index(drop=True)
    )
    items["item_condition_id"] = items.index.astype(np.int64)
    return items


def n400_window_mean(Y: np.ndarray, times: np.ndarray,
                     tmin: float = N400_TMIN, tmax: float = N400_TMAX) -> np.ndarray:
    """Mean amplitude in the N400 window, averaged across channels.

    Parameters
    ----------
    Y     : (n_trials, n_channels, n_times)
    times : (n_times,)

    Returns
    -------
    (n_trials,) float32 — mean N400 amplitude per trial.
    """
    mask = (times >= tmin) & (times <= tmax)
    return Y[:, :, mask].mean(axis=(1, 2))


def n400_window_mean_per_channel(
    Y: np.ndarray, times: np.ndarray,
    tmin: float = N400_TMIN, tmax: float = N400_TMAX,
) -> np.ndarray:
    """Mean amplitude in the N400 window, per channel (no channel averaging).

    Same window-mean logic as :func:`n400_window_mean`, but keeps the channel
    axis — needed by the pooled LME (``benchmarks/n400_nieuwland_lme.py``),
    which models electrode as a crossed random effect rather than averaging
    it away.

    Parameters
    ----------
    Y     : (n_trials, n_channels, n_times)
    times : (n_times,)

    Returns
    -------
    (n_trials, n_channels) float32.
    """
    mask = (times >= tmin) & (times <= tmax)
    return Y[:, :, mask].mean(axis=2)


# ---------------------------------------------------------------------------
# Item-only loading (no raw EEG / epoching) — for the pooled-LME GPU stage
# ---------------------------------------------------------------------------

def load_all_items(
    bids_root: str | Path,
    *,
    included_in_paper_only: bool = True,
    exclude_labs: tp.Sequence[str] = DEFAULT_EXCLUDE_LABS,
    subjects: list[str] | None = None,
) -> pd.DataFrame:
    """Pool unique (item_number, condition, sentence_context, noun) pairs
    from every subject's ``events.tsv`` — WITHOUT touching raw EEG.

    ``build_unique_items_df`` needs ``meta``, which today only exists after
    :func:`load_all_epochs` has epoched every subject's raw EEG (expensive,
    MNE-based). The sentence/noun text is already in each subject's
    ``events.tsv`` (read by :func:`_load_events_df`), so item discovery for
    the pooled LME's GPU extraction stage (``benchmarks/n400_nieuwland_lme.
    py::extract_and_cache_surprisal``) doesn't need to pay that cost — it can
    just pool ``events.tsv`` across subjects. Any single subject may not have
    seen every item (Latin-square counterbalanced), so all subjects are read
    (still cheap: small per-subject TSV files, no EEG parsing).

    Returns
    -------
    pd.DataFrame — same columns/shape as ``build_unique_items_df``'s output
    (``item_number, condition, sentence_context, noun, item_condition_id``).
    """
    bids_root = Path(bids_root)
    participants = load_participants(bids_root)

    if included_in_paper_only:
        participants = participants[participants["included_in_paper"]]
    if exclude_labs:
        excl = [e.lower() for e in exclude_labs]
        participants = participants[~participants["lab"].str.lower().isin(excl)]
    if subjects is not None:
        participants = participants[participants["participant_id"].isin(subjects)]

    all_noun_dfs: list[pd.DataFrame] = []
    for subj in participants["participant_id"].tolist():
        events_df = _load_events_df(bids_root, subj)
        if events_df.empty:
            continue
        noun_df = _noun_events(events_df)
        if noun_df.empty:
            continue
        noun_df = noun_df.copy()
        noun_df["noun"] = noun_df.apply(_extract_noun, axis=1)
        all_noun_dfs.append(noun_df)

    if not all_noun_dfs:
        raise RuntimeError(f"No events.tsv trials found under {bids_root}")

    meta = pd.concat(all_noun_dfs, ignore_index=True)
    logger.info(
        "Nieuwland items (EEG-free): pooled %d subjects' events.tsv → %d trial rows",
        len(all_noun_dfs), len(meta),
    )
    return build_unique_items_df(meta)
