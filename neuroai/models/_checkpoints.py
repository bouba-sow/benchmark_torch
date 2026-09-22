"""Pure-Python checkpoint metadata helpers — no torch/neuralset dependency."""

from __future__ import annotations

import re
import typing as tp

OLMO2_STAGE1_REVISION_RE = re.compile(r"^stage1-step(\d+)-tokens(\d+)B$")
OLMO2_STEP_REVISION_RE = re.compile(r"^step(\d+)-tokens(\d+)B$")

PYTHIA_MODELS = [
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]

PYTHIA_CHECKPOINT_STEPS: list[int] = [
    0, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000,
]

GPT2_MODELS = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
]

OLMO2_0425_1B_DEFAULT_REVISIONS: list[str] = [
    "stage1-step0-tokens0B",
    "stage1-step300-tokens1B",
    "stage1-step10000-tokens21B",
    "stage1-step20000-tokens42B",
    "stage1-step40000-tokens84B",
    "stage1-step80000-tokens168B",
    "stage1-step200000-tokens420B",
    "stage1-step400000-tokens839B",
    "stage1-step1000000-tokens2098B",
    "stage1-step1907359-tokens4001B",
]

OLMO2_1124_7B_DEFAULT_REVISIONS: list[str] = [
    "stage1-step150-tokens1B",
    "stage1-step2000-tokens9B",
    "stage1-step10000-tokens42B",
    "stage1-step20000-tokens84B",
    "stage1-step40000-tokens168B",
    "stage1-step99000-tokens416B",
    "stage1-step200000-tokens839B",
    "stage1-step500000-tokens2098B",
    "stage1-step928646-tokens3896B",
]

OLMO2_1124_13B_DEFAULT_REVISIONS: list[str] = [
    "stage1-step0-tokens0B",
    "stage1-step1000-tokens9B",
    "stage1-step5000-tokens42B",
    "stage1-step10000-tokens84B",
    "stage1-step20000-tokens168B",
    "stage1-step50000-tokens420B",
    "stage1-step100000-tokens839B",
    "stage1-step250000-tokens2098B",
    "stage1-step596057-tokens5001B",
]


def pythia_checkpoint_revision(step: int) -> str:
    return f"step{step}"


def olmo2_stage1_metrics(revision: str) -> tp.Optional[tuple[int, int]]:
    m = OLMO2_STAGE1_REVISION_RE.match(revision.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)) * 1_000_000_000


def olmo2_checkpoint_metrics(revision: str) -> tp.Optional[tuple[int, int]]:
    rev = revision.strip()
    if rev in ("", "main"):
        return 0, 0
    parsed = olmo2_stage1_metrics(rev)
    if parsed is not None:
        return parsed
    m = OLMO2_STEP_REVISION_RE.match(rev)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)) * 1_000_000_000
