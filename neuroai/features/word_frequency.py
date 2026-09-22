"""Word length + wordfreq lexical-frequency baseline covariates.

Used as the confound baseline (Oh et al. 2024's frequency/length story for
the inverse-scaling effect) for the Podcast ridge unique-contribution check
and the Podcast pooled LME. ``wordfreq``'s own tokenizer handles trailing
punctuation gracefully (``"middle."`` and ``"middle"`` return the same
value), so no cleaning is needed before lookup.
"""

from __future__ import annotations

import numpy as np


def compute_wlen_wordfreq_covariates(
    words: list[str],
    *,
    language: str = "en",
) -> np.ndarray:
    """Word length + wordfreq Zipf frequency for a flat, ordered word list.

    Zipf frequency is a log10-based per-billion-words frequency scale
    (roughly 0-8 for English; higher = more frequent). Unlike the KenLM
    lexical surprisal this replaces, no model load or disk cache is needed —
    ``zipf_frequency`` is a fast in-memory lookup.

    Returns
    -------
    np.ndarray  shape (n_words, 2): [wlen, zipf_freq]
    """
    from wordfreq import zipf_frequency

    wlen = np.array([len(w) for w in words], dtype=np.float64)
    freq = np.array([zipf_frequency(w.lower(), language) for w in words], dtype=np.float64)
    return np.stack([wlen, freq], axis=1).astype(np.float32)
