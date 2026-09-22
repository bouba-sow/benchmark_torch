"""Shared infrastructure for checkpoint sweep scripts.

Imported by every ``run_*_checkpoint_sweep.py`` and future dataset sweep scripts.
Contains only generic sweep mechanics — no dataset or modality knowledge.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import sys
import typing as tp
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class FlushingStreamHandler(logging.StreamHandler):
    """Flush stdout after every log record so SLURM .out files update live."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.flush()
        except OSError:
            pass
        sys.stdout.flush()


def setup_logging() -> None:
    """Configure root logger for sweep scripts (line-buffered, flushing handler).

    Call once at module level, before any other logging calls.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except OSError:
                pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[FlushingStreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------------
# Checkpoint bookkeeping
# ---------------------------------------------------------------------------

class CheckpointItem(NamedTuple):
    """One HF weight snapshot in a training-step sweep."""
    revision: str    # HF branch / tag / commit SHA (e.g. "step8000", "main")
    step: int        # optimizer step (0 for final / main)
    tokens_seen: int # approximate tokens seen at this checkpoint


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def safe_model_dir(model_id: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe directory name.

    Example: ``"EleutherAI/pythia-1b"`` → ``"EleutherAI_pythia-1b"``
    """
    return model_id.replace("/", "_")


# ---------------------------------------------------------------------------
# GPU / memory cleanup
# ---------------------------------------------------------------------------

def free_gpu() -> None:
    """Release unused GPU memory and run the Python GC between model runs."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Incremental CSV helpers (concurrency-safe via fcntl file lock)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def csv_file_lock(csv_path: Path) -> tp.Generator[None, None, None]:
    """Exclusive file lock around CSV writes for concurrent SLURM array workers.

    Creates a ``<csv>.lock`` sentinel file; all workers sharing the same output
    directory should use this context manager before any read-modify-write on the CSV.
    """
    import fcntl

    lock_path = csv_path.with_suffix(f"{csv_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def load_existing_csv(csv_path: Path) -> pd.DataFrame:
    """Load a sweep CSV, returning an empty DataFrame if the file is missing or corrupt."""
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "Could not read %s: %s — starting fresh", csv_path, exc
            )
    return pd.DataFrame()


def already_computed(df: pd.DataFrame, revision: str) -> bool:
    """Return True if ``revision`` already has a row in the sweep CSV."""
    if df.empty or "revision" not in df.columns:
        return False
    return bool((df["revision"] == revision).any())


def append_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one result row to a CSV, creating it if it does not exist.

    Uses :func:`csv_file_lock` so concurrent SLURM workers do not corrupt the file.
    """
    with csv_file_lock(csv_path):
        df_new = pd.DataFrame([row])
        if csv_path.exists():
            df_new.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(csv_path, index=False)
