"""Le Petit Prince (OpenNeuro ds003643) — Sprint 1 data contract.

Passive **audio** listening: there is **no word-level reaction time** in this
corpus. Encoding targets are **fMRI** only (wired in later sprints).

**Timelines**

* Annotation timelines are keyed by ``section ∈ {1,…,9}`` (story segment), as
  in ``annotation/<LANG>/lpp<LANG>_word_information.csv``.
* BIDS ``run-*`` indices are **scanner acquisition** labels. We align them to
  ``section`` by **sorting** runs numerically and pairing in order with
  ``1..N`` (``N`` = number of runs found). When ``N ≠ 9``, downstream code
  must handle missing segments — :func:`discover_lp_timelines` emits a warning.

**Events (Sprint 1)**

* Primary units are **words** with ``onset`` / ``offset`` on the stimulus
  timeline (seconds) and metadata (POS, frequency, etc.).
* **Sentence-level** structure lives in separate files (e.g. one constituency
  tree per line in ``lppEN_tree.csv``). Use :func:`load_sentence_trees` and
  optional :func:`apply_lbg_word_table_fixes` when replicating published
  pipelines.

**Scanner timing (Li et al., Sci Data 2022)**

Per the LPPC-fMRI data descriptor, each run starts with a **scanner trigger**,
then **8 s (4 TRs)** of silence before **auditory stimulus onset** (TR = 2 s).
Preprocessing **discards the first 4 volumes** per run for T1 equilibrium
(AFNI pipeline); ME-ICA denoised derivatives are resampled to 2 mm isotropic
MNI (``space-MNIColin27`` in preprocessed BOLD). Word ``onset``/``offset`` in
the annotation CSVs are on the **stimulus/audio** timeline within each section —
Sprint 2 must combine this with volume indexing explicitly.

**References:** `doi:10.1038/s41597-022-01625-7` — OpenNeuro
`doi:10.18112/openneuro.ds003643` (check your snapshot version; the paper cites
v2.0.0). Maintainer code: https://github.com/jixing-li/lpp_data ,
https://github.com/chrplr/lpp-paradigm .
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

LPP_TR_SECONDS: float = 2.0
LPP_N_ANNOTATION_SECTIONS: int = 9
# Scanner trigger → auditory stimulus onset (Sci Data paper; 4 TRs at 2 s).
LPP_TRIGGER_TO_STIMULUS_SEC: float = 8.0
# First volumes excluded in AFNI functional preprocessing (Sci Data paper).
LPP_DISCARD_VOLUMES_PREPROC: int = 4

_LANG_TO_TASK: dict[str, str] = {"en": "lppEN", "fr": "lppFR", "cn": "lppCN"}
_LANG_TO_DIR: dict[str, str] = {"en": "EN", "fr": "FR", "cn": "CN"}

_RUN_INDEX_RE = re.compile(r"_run-(\d+)_")


def language_to_bids_task(lang: str) -> str:
    key = lang.lower()
    if key not in _LANG_TO_TASK:
        raise ValueError(f"Unsupported LPP language {lang!r}; expected one of {set(_LANG_TO_TASK)}")
    return _LANG_TO_TASK[key]


def annotation_language_dir(lang: str) -> str:
    key = lang.lower()
    if key not in _LANG_TO_DIR:
        raise ValueError(f"Unsupported LPP language {lang!r}")
    return _LANG_TO_DIR[key]


def word_information_path(bids_root: Path | str, lang: str) -> Path:
    root = Path(bids_root)
    sub = annotation_language_dir(lang)
    return root / "annotation" / sub / f"lpp{sub}_word_information.csv"


def tree_corpus_path(bids_root: Path | str, lang: str) -> Path:
    root = Path(bids_root)
    sub = annotation_language_dir(lang)
    return root / "annotation" / sub / f"lpp{sub}_tree.csv"


def load_sentence_trees(bids_root: Path | str, lang: str) -> list[str]:
    """One sentence / constituency tree per line (story order)."""
    path = tree_corpus_path(bids_root, lang)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def load_word_information_table(
    bids_root: Path | str,
    lang: str,
    *,
    apply_lbg_fixes: bool = False,
) -> pd.DataFrame:
    """Load ``lpp*_word_information.csv`` as a DataFrame.

    Columns include at least: word, lemma, onset, offset, logfreq, pos, section,
    top_down, bottom_up, left_corner.

    Parameters
    ----------
    apply_lbg_fixes
        If True, apply the same row drops / replacements as in
        ``l-bg/llms_brain_lateralization`` ``extract_llm_activations.py`` for
        closer replication (English + French only).
    """
    path = word_information_path(bids_root, lang)
    df = pd.read_csv(path, index_col=0)
    if "word" in df.columns and "lemma" in df.columns:
        # EN annotation only (31 rows, all in section in {1,4,5,6,7,9}): the
        # "n't" half of a split contraction (e.g. "hasn't" -> word="has" then
        # a second row for "not") has an empty surface "word" field, read by
        # pandas as a float NaN rather than an empty string. lemma still has
        # the real content ("not") for these rows, so recover it there rather
        # than let a bare NaN reach `" ".join(words)` downstream (crashes:
        # `TypeError: sequence item N: expected str instance, float found`)
        # or silently turn into the literal 4-character string "nan" via a
        # blanket str()/astype(str) call.
        missing = df["word"].isna()
        if missing.any():
            df.loc[missing, "word"] = df.loc[missing, "lemma"]
    key = lang.lower()
    if apply_lbg_fixes:
        df = _apply_lbg_fixes(df.copy(), key)
    return df


def _apply_lbg_fixes(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Mirror optional fixes from llms_brain_lateralization."""
    if lang == "en":
        drop_ix = [3919, 6775, 6781]
        df = df.drop(index=[i for i in drop_ix if i in df.index], errors="ignore")
    elif lang == "fr":
        fixes: dict[int, str] = {
            3332: "de",
            3379: "trois",
            3405: "trois",
            4587: "l",
            5325: "la",
            5326: "première",
            5328: "habitée",
            11257: "À",
            12249: "il",
        }
        for ix, w in fixes.items():
            if ix in df.index:
                df.loc[ix, "word"] = w
        drop_ix = [338, 1204, 3333]
        df = df.drop(index=[i for i in drop_ix if i in df.index], errors="ignore")
    elif lang == "cn":
        pass
    else:
        raise ValueError(lang)
    return df


def words_for_section(df: pd.DataFrame, section: int) -> pd.DataFrame:
    """Rows of the word table for one story section (1-based)."""
    return df.loc[df["section"] == section].copy()


def parse_bids_run_index(filename: str) -> int | None:
    m = _RUN_INDEX_RE.search(filename)
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class LppBoldRun:
    """One functional run on disk."""

    bids_run: int
    path: Path


@dataclass(frozen=True)
class LppTimeline:
    """One encoding timeline: section index + BOLD file + subject."""

    subject: str
    language: str
    section: int
    bids_run: int
    bold_path: Path
    tr_seconds: float = LPP_TR_SECONDS


def discover_bold_runs(
    bids_root: Path | str,
    subject: str,
    lang: str,
    *,
    source: Literal["raw", "derivatives"] = "raw",
    echo: int | None = 2,
) -> list[LppBoldRun]:
    """List BOLD NIfTIs for ``sub-<ID>`` and language task, sorted by run.

    Parameters
    ----------
    source
        ``raw`` — multi-echo files under ``sub-*/func/``; keep one echo if
        ``echo`` is set (default **2** = mid echo for English in sidecars).
        ``derivatives`` — preprocessed ``*_desc-preproc_bold.nii.gz`` under
        ``derivatives/sub-*/func/`` (single file per run, no echo dimension).
    echo
        Used only for ``source='raw'``. ``None`` keeps **all** echoes (caller
        must disambiguate).
    """
    root = Path(bids_root)
    task = language_to_bids_task(lang)
    if source == "derivatives":
        func_dir = root / "derivatives" / subject / "func"
        pattern = f"{subject}_task-{task}_run-*_desc-preproc_bold.nii.gz"
        paths = sorted(func_dir.glob(pattern))
        if not paths:
            pattern2 = f"{subject}_task-{task}_run-*_*_desc-preproc_bold.nii.gz"
            paths = sorted(func_dir.glob(pattern2))
    else:
        func_dir = root / subject / "func"
        pattern = f"{subject}_task-{task}_run-*_bold.nii.gz"
        paths = sorted(func_dir.glob(pattern))
        if echo is not None:
            paths = [p for p in paths if f"_echo-{echo}_" in p.name]

    keyed: list[tuple[int, Path]] = []
    for p in paths:
        r = parse_bids_run_index(p.name)
        if r is None:
            continue
        keyed.append((r, p))
    keyed.sort(key=lambda t: t[0])
    return [LppBoldRun(bids_run=r, path=p) for r, p in keyed]


def discover_lpp_derivative_subjects(bids_root: Path | str, lang: str) -> list[str]:
    """List ``sub-<LANG>…`` IDs that have a ``derivatives/<id>/`` folder (preprocessed BOLD).

    Used for **average-subject** fMRI matching l-bg: average over everyone with
    derivatives, when the YAML ``subjects`` key is omitted.
    """
    root = Path(bids_root) / "derivatives"
    if not root.is_dir():
        return []
    key = lang.lower()
    prefix = {"en": "sub-EN", "fr": "sub-FR", "cn": "sub-CN"}.get(key)
    if prefix is None:
        raise ValueError(f"Unsupported language {lang!r} for derivative subject discovery")
    return sorted(p.name for p in root.glob(f"{prefix}*") if p.is_dir())


def discover_lp_timelines(
    bids_root: Path | str,
    subject: str,
    lang: str,
    *,
    source: Literal["raw", "derivatives"] = "raw",
    echo: int | None = 2,
) -> list[LppTimeline]:
    """Pair sorted BOLD runs with annotation ``section`` 1..N.

    See module docstring for alignment assumptions.
    """
    runs = discover_bold_runs(
        bids_root, subject, lang, source=source, echo=echo
    )
    if not runs:
        warnings.warn(
            f"No BOLD files found for {subject} task {language_to_bids_task(lang)} "
            f"(source={source!r}, echo={echo}).",
            stacklevel=2,
        )
        return []
    n = len(runs)
    if n != LPP_N_ANNOTATION_SECTIONS:
        warnings.warn(
            f"{subject}: expected {LPP_N_ANNOTATION_SECTIONS} runs for full LPP "
            f"alignment, found {n}. Section indices 1..{n} are still assigned in "
            "sorted run order.",
            stacklevel=2,
        )
    timelines: list[LppTimeline] = []
    for sec, br in enumerate(runs, start=1):
        timelines.append(
            LppTimeline(
                subject=subject,
                language=lang.lower(),
                section=sec,
                bids_run=br.bids_run,
                bold_path=br.path,
            )
        )
    return timelines


def build_word_events_frame(
    words_df: pd.DataFrame,
    *,
    language: str,
) -> pd.DataFrame:
    """Normalize word rows to a NeuralSet-style event table (no RT, no Ieeg/Fmri).

    Columns: type, start, duration, text, language, lemma, pos, section,
    logfreq, top_down, bottom_up, left_corner.

    Sprint 2+ will append neural modality rows and feed NeuralSet.
    """
    rows: list[dict[str, object]] = []
    lang_lower = language.lower()
    for rec in words_df.itertuples(index=True):
        word = getattr(rec, "word", None)
        if not isinstance(word, str) or word.strip() == "" or word.strip() == " ":
            continue
        onset = float(getattr(rec, "onset"))
        offset = float(getattr(rec, "offset"))
        duration = max(0.0, offset - onset)
        section = int(getattr(rec, "section"))
        rows.append(
            {
                "type": "Word",
                "start": onset,
                "duration": duration,
                "text": word,
                "language": lang_lower,
                "lemma": getattr(rec, "lemma", None),
                "pos": getattr(rec, "pos", None),
                "section": section,
                "logfreq": getattr(rec, "logfreq", None),
                "top_down": getattr(rec, "top_down", None),
                "bottom_up": getattr(rec, "bottom_up", None),
                "left_corner": getattr(rec, "left_corner", None),
            }
        )
    return pd.DataFrame(rows)


def timeline_word_events(
    bids_root: Path | str,
    subject: str,
    lang: str,
    timeline: LppTimeline,
    *,
    apply_lbg_fixes: bool = False,
) -> pd.DataFrame:
    """Word events for one :class:`LppTimeline` (one section / run)."""
    df = load_word_information_table(bids_root, lang, apply_lbg_fixes=apply_lbg_fixes)
    sec_df = words_for_section(df, timeline.section)
    return build_word_events_frame(sec_df, language=lang)


def summarize_nifti_run(path: Path) -> dict[str, object]:
    """Spatial dimensions and time length from NIfTI header (requires nibabel)."""
    import nibabel as nib  # lazy: optional for callers that only use tables

    img = nib.load(str(path))
    shape = img.shape
    if len(shape) < 4:
        return {"path": path, "shape": shape, "n_volumes": None, "error": "not 4D"}
    *_, t = shape
    return {"path": path, "shape": shape, "n_volumes": int(t)}


__all__ = [
    "LPP_DISCARD_VOLUMES_PREPROC",
    "LPP_N_ANNOTATION_SECTIONS",
    "LPP_TR_SECONDS",
    "LPP_TRIGGER_TO_STIMULUS_SEC",
    "LppBoldRun",
    "LppTimeline",
    "annotation_language_dir",
    "build_word_events_frame",
    "discover_bold_runs",
    "discover_lpp_derivative_subjects",
    "discover_lp_timelines",
    "language_to_bids_task",
    "load_sentence_trees",
    "load_word_information_table",
    "parse_bids_run_index",
    "summarize_nifti_run",
    "timeline_word_events",
    "tree_corpus_path",
    "word_information_path",
    "words_for_section",
]
