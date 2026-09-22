"""Fedorenko2016 word-level iEEG tabular dataset helpers.

Dataset files expected under ``data/fedorenko2016``:
  - stimuli.csv
  - neural_wide.csv
  - electrodes.csv
  - sentences.csv

Rows are already word-aligned (one row per word stimulus). Neural targets are
scalar responses per electrode column (no time axis), so this dataset is used
with scalar ridge regression (not lag-matrix TRF).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Fedorenko2016Dataset:
    """Loader/validator for the Fedorenko2016 benchmark tables."""

    root: str | Path

    def _path(self, name: str) -> Path:
        return Path(self.root) / name

    def load_stimuli(self) -> pd.DataFrame:
        p = self._path("stimuli.csv")
        if not p.is_file():
            raise FileNotFoundError(f"Missing stimuli table: {p}")
        return pd.read_csv(p)

    def load_neural_wide(self) -> pd.DataFrame:
        p = self._path("neural_wide.csv")
        if not p.is_file():
            raise FileNotFoundError(f"Missing neural_wide table: {p}")
        return pd.read_csv(p)

    def load_electrodes(self) -> pd.DataFrame:
        p = self._path("electrodes.csv")
        if not p.is_file():
            raise FileNotFoundError(f"Missing electrodes table: {p}")
        return pd.read_csv(p)

    def load_sentences(self) -> pd.DataFrame:
        p = self._path("sentences.csv")
        if not p.is_file():
            raise FileNotFoundError(f"Missing sentences table: {p}")
        return pd.read_csv(p)

    def load_word_table(self) -> pd.DataFrame:
        """Return a single table with word metadata + electrode targets.

        The returned DataFrame has one row per stimulus word with columns from
        ``stimuli.csv`` and all ``electrode_*`` columns from ``neural_wide.csv``.
        """
        stimuli = self.load_stimuli()
        neural_wide = self.load_neural_wide()

        if len(stimuli) != len(neural_wide):
            raise ValueError(
                f"stimuli/neural_wide row mismatch: {len(stimuli)} vs {len(neural_wide)}"
            )
        if "stimulus_id" not in stimuli.columns or "stimulus_id" not in neural_wide.columns:
            raise ValueError("Both stimuli.csv and neural_wide.csv must include stimulus_id")

        left = stimuli.sort_values("stimulus_id").reset_index(drop=True)
        right = neural_wide.sort_values("stimulus_id").reset_index(drop=True)
        if not np.array_equal(left["stimulus_id"].to_numpy(), right["stimulus_id"].to_numpy()):
            raise ValueError("stimulus_id mismatch between stimuli.csv and neural_wide.csv")

        electrode_cols = [c for c in right.columns if c.startswith("electrode_")]
        if not electrode_cols:
            raise ValueError("No electrode_* columns found in neural_wide.csv")

        out = left.copy()
        out[electrode_cols] = right[electrode_cols].to_numpy()
        return out

