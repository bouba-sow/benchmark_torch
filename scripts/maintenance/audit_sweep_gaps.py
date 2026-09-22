#!/usr/bin/env python3
"""Audit outputs_jeanzay/ for missing or incomplete checkpoint/LME/embed sweep
results, and print ready-to-run sbatch commands to fill every gap.

For each known dataset x sweep-type combo, compares the models the registry
(``neuroai/models/registry.py``) says a family SHOULD have against which
models actually have a non-trivial score CSV on disk (a header-only or
missing CSV counts as not done -- a model directory existing is not enough,
several sweeps have crashed empty stubs).

Usage:
    .venv/bin/python scripts/maintenance/audit_sweep_gaps.py                # everything
    .venv/bin/python scripts/maintenance/audit_sweep_gaps.py --dataset geco  # filter by key substring
    .venv/bin/python scripts/maintenance/audit_sweep_gaps.py --commands-only # just the submit.py lines

This intentionally hardcodes the on-disk layout per dataset+sweep-type (see SPECS
below) rather than inferring it -- output directories vary (measure-nested dirs,
dlt/oos roots, flat vs per-family) enough that guessing would silently miss results.
The printed commands go through ``scripts/submit.py`` (stage table:
``scripts/slurm/stages.yaml``). When a new sweep type is added, add a SweepSpec here.

Broderick LME is deliberately excluded (see SKIPPED_KEYS): its by-layer fits
are non-convergent and it is not registered in the benchmark's surprisal track.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))

from neuroai.models.registry import flatten_family_tasks  # noqa: E402

ALL_FAMILIES: tuple[str, ...] = (
    "gpt2", "gptneo", "opt", "mamba", "mamba2", "pythia", "olmo2", "qwen3",
    "llama32", "mistral", "cbr_rnn", "stickbreaking", "alibi", "forgetting",
    "hard_2gram", "hard_3gram", "hard_5gram", "babylm",
    # 2026-09-12 additions (memory-/architecturally-constrained families at
    # web scale) -- were missing from this list entirely, so every one of
    # their gaps across every dataset was silently invisible to this audit.
    "bloom", "rwkv4", "fox", "molm",
)

# Sweeps deliberately excluded from the audit (by SweepSpec.key) -- not
# tracked/prioritized right now, so their gaps are just noise here.
#
# Only broderick_lme (non-convergent fits) remains deliberately out of scope.
SKIPPED_KEYS: frozenset[str] = frozenset({
    "broderick_lme",
})


def _non_trivial_csv(path: Path) -> bool:
    """True if ``path`` exists and has at least one data row past the header."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                if any(cell.strip() for cell in row):
                    return True
    except (StopIteration, OSError, csv.Error):
        return False
    return False


def _compress_indices(idxs: list[int]) -> str:
    """[1,3,4,5,7,8,9] -> '1,3-5,7-9' (SLURM --array style)."""
    if not idxs:
        return ""
    idxs = sorted(set(idxs))
    parts: list[str] = []
    start = prev = idxs[0]
    for i in idxs[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = i
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


@dataclass
class SweepSpec:
    key: str
    label: str
    out_dir: str                      # relative to repo root; already dlt/measure-resolved
    score_name: str                   # filename searched for (rglob) under <out_dir>/<family>/<model_dir_safe>/
    stage_templates: tuple[str, ...]  # 1 (single-stage) or 2 (two-stage) sbatch templates; {family} {array}
    families: tuple[str, ...] = ALL_FAMILIES
    env_prefix: str = ""              # e.g. "GECO_SWEEP_ROOT=$PWD/outputs_jeanzay/geco/lme_sweep_dlt "
    note: str | None = None
    also_check_flat: bool = False     # also probe <out_dir>/<model_dir_safe>/ (no family
                                       # subdir) in addition to the nested path above -- set
                                       # for sweeps whose runner script's --out_dir is the
                                       # bare sweep root (e.g. LPP: job_lpp_jz_common.sh calls
                                       # run/embedding/neural/lpp.py --out_dir "${SWEEP_ROOT}",
                                       # no per-family subdir), where results land flat and the
                                       # nested path is never written by the current pipeline.


SPECS: list[SweepSpec] = [
    SweepSpec("broderick_ckpt", "Broderick checkpoint",
        "outputs_jeanzay/broderick/embedding_sweep", "broderick_checkpoint_scores.csv",
        ("python scripts/submit.py broderick embedding fit --family {family} --indices {array} --submit",)),
    SweepSpec("broderick_lme", "Broderick LME",
        "outputs_jeanzay/broderick/surprisal_sweep", "broderick_lme_by_layer.csv",
        ("python scripts/submit.py broderick surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py broderick surprisal fit --family {family} --indices {array} --submit")),
    SweepSpec("fedorenko2016_ckpt", "Fedorenko2016 checkpoint",
        "outputs_jeanzay/fedorenko2016/embedding_sweep", "fedorenko2016_checkpoint_scores.csv",
        ("python scripts/submit.py fedorenko2016 embedding fit --family {family} --indices {array} --submit",)),
    SweepSpec("fedorenko2016_lme", "Fedorenko2016 LME",
        "outputs_jeanzay/fedorenko2016/surprisal_sweep", "fedorenko2016_lme_by_layer.csv",
        ("python scripts/submit.py fedorenko2016 surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py fedorenko2016 surprisal fit --family {family} --indices {array} --submit")),
    SweepSpec("geco_embed_fp", "GECO embed (FP)",
        "outputs_jeanzay/geco/embedding_sweep/fp", "geco_embed_checkpoint_scores.csv",
        ("python scripts/submit.py geco embedding fit --family {family} --indices {array} --measure FP --submit",)),
    SweepSpec("geco_lme", "GECO LME (dlt)",
        "outputs_jeanzay/geco/surprisal_sweep/fp", "geco_lme_checkpoint_scores.csv",
        ("python scripts/submit.py geco surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py geco surprisal fit --family {family} --indices {array} --measure FP --submit")),
    SweepSpec("naturalstories_lme", "NaturalStories LME (dlt)",
        "outputs_jeanzay/naturalstories/surprisal_sweep", "spr_lme_checkpoint_scores.csv",
        ("python scripts/submit.py naturalstories surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py naturalstories surprisal fit --family {family} --indices {array} --submit")),
    SweepSpec("naturalstories_embed_gpu", "NaturalStories embed (GPU)",
        "outputs_jeanzay/naturalstories/embedding_sweep", "spr_checkpoint_scores.csv",
        ("python scripts/submit.py naturalstories embedding fit --family {family} --indices {array} --submit",)),
    SweepSpec("nieuwland_ckpt", "Nieuwland checkpoint (dlt)",
        "outputs_jeanzay/nieuwland/embedding_sweep", "nieuwland_checkpoint_scores.csv",
        ("python scripts/submit.py nieuwland embedding fit --family {family} --indices {array} --submit",)),
    SweepSpec("nieuwland_lme", "Nieuwland LME",
        "outputs_jeanzay/nieuwland/surprisal_sweep", "nieuwland_lme_by_layer.csv",
        ("python scripts/submit.py nieuwland surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py nieuwland surprisal fit --family {family} --indices {array} --submit"),
        note="babylm idx2 (Jtapsa/moep) may be DELIBERATELY excluded (unvalidated hook fallback) -- confirm before submitting babylm."),
    SweepSpec("ns_fmri_ckpt", "ns_fmri checkpoint",
        "outputs_jeanzay/ns_fmri/embedding_sweep", "ns_fmri_checkpoint_scores.csv",
        ("python scripts/submit.py ns_fmri embedding fit --family {family} --indices {array} --submit",)),
    SweepSpec("ns_fmri_lme", "ns_fmri LME",
        "outputs_jeanzay/ns_fmri/surprisal_sweep", "ns_fmri_lme_checkpoint_scores.csv",
        ("python scripts/submit.py ns_fmri surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py ns_fmri surprisal fit --family {family} --indices {array} --submit")),
    SweepSpec("podcast_ckpt", "Podcast checkpoint (dlt)",
        "outputs_jeanzay/podcast/embedding_sweep", "podcast_checkpoint_scores.csv",
        ("python scripts/submit.py podcast embedding fit --family {family} --indices {array} --submit",)),
    SweepSpec("podcast_lme", "Podcast LME",
        "outputs_jeanzay/podcast/surprisal_sweep", "podcast_lme_by_lag.csv",
        ("python scripts/submit.py podcast surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py podcast surprisal fit --family {family} --indices {array} --submit")),
    SweepSpec("lpp_ckpt", "LPP checkpoint",
        "outputs_jeanzay/lpp/embedding_sweep", "lpp_checkpoint_scores.csv",
        ("python scripts/submit.py lpp embedding fit --family {family} --indices {array} --submit",),
        also_check_flat=True,
        note="results land flat at <out_dir>/<model_dir_safe>/ (no per-family subdir); the nested path some older runs used is also checked."),
    SweepSpec("lpp_lme", "LPP LME",
        "outputs_jeanzay/lpp/surprisal_sweep", "lpp_lme_by_layer.csv",
        ("python scripts/submit.py lpp surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py lpp surprisal fit --family {family} --indices {array} --submit")),
    SweepSpec("meco_embed_fp", "MeCo embed (FP)",
        "outputs_jeanzay/meco/embedding_sweep/fp", "meco_embed_checkpoint_scores.csv",
        ("python scripts/submit.py meco embedding fit --family {family} --indices {array} --measure FP --submit",)),
    SweepSpec("meco_lme_fp", "MeCo LME (dlt, FP)",
        "outputs_jeanzay/meco/surprisal_sweep/fp", "meco_lme_checkpoint_scores.csv",
        ("python scripts/submit.py meco surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py meco surprisal fit --family {family} --indices {array} --measure FP --submit")),
    SweepSpec("provo_embed_fp", "Provo embed (FP)",
        "outputs_jeanzay/provo/embedding_sweep/fp", "provo_embed_checkpoint_scores.csv",
        ("python scripts/submit.py provo embedding fit --family {family} --indices {array} --measure FP --submit",)),
    SweepSpec("provo_lme_fp", "Provo LME (dlt, FP)",
        "outputs_jeanzay/provo/surprisal_sweep/fp", "provo_lme_checkpoint_scores.csv",
        ("python scripts/submit.py provo surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py provo surprisal fit --family {family} --indices {array} --measure FP --submit")),
    SweepSpec("tuckute_ckpt", "Tuckute checkpoint",
        "outputs_jeanzay/tuckute/embedding_sweep", "tuckute_scores.csv",
        ("python scripts/submit.py tuckute embedding fit --family {family} --indices {array} --submit",)),
    SweepSpec("tuckute_lme", "Tuckute LME",
        "outputs_jeanzay/tuckute/surprisal_sweep", "tuckute_lme_by_layer.csv",
        ("python scripts/submit.py tuckute surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py tuckute surprisal fit --family {family} --indices {array} --submit")),
    SweepSpec("ucl_reading_embed", "UCL Reading embed",
        "outputs_jeanzay/ucl_reading/embedding_sweep", "ucl_reading_embed_checkpoint_scores.csv",
        ("python scripts/submit.py ucl_reading embedding fit --family {family} --indices {array} --submit",)),
    SweepSpec("ucl_reading_lme", "UCL Reading LME (dlt)",
        "outputs_jeanzay/ucl_reading/surprisal_sweep", "ucl_reading_spr_lme_checkpoint_scores.csv",
        ("python scripts/submit.py ucl_reading surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py ucl_reading surprisal fit --family {family} --indices {array} --submit")),
    SweepSpec("zuco_reading_embed_nr", "ZuCo Reading embed (NR)",
        "outputs_jeanzay/zuco_reading/embedding_sweep/nr", "zuco_reading_embed_checkpoint_scores.csv",
        ("python scripts/submit.py zuco_reading embedding fit --family {family} --indices {array} --task NR --submit",)),
    SweepSpec("zuco_reading_lme_nr", "ZuCo Reading LME (dlt, NR)",
        "outputs_jeanzay/zuco_reading/surprisal_sweep/nr", "zuco_reading_trt_lme_checkpoint_scores.csv",
        ("python scripts/submit.py zuco_reading surprisal extract --family {family} --indices {array} --task NR --submit",
        "python scripts/submit.py zuco_reading surprisal fit --family {family} --indices {array} --task NR --submit")),
    SweepSpec("szewczyk_ckpt", "Szewczyk checkpoint",
        "outputs_jeanzay/szewczyk/embedding_sweep", "szewczyk_checkpoint_scores.csv",
        ("python scripts/submit.py szewczyk embedding fit --family {family} --indices {array} --submit",)),
    SweepSpec("szewczyk_lme", "Szewczyk LME",
        "outputs_jeanzay/szewczyk/surprisal_sweep", "szewczyk_lme_by_layer.csv",
        ("python scripts/submit.py szewczyk surprisal extract --family {family} --indices {array} --submit",
        "python scripts/submit.py szewczyk surprisal fit --family {family} --indices {array} --submit")),
    SweepSpec("megmasc_ckpt", "MEG-MASC (Gwilliams) checkpoint",
        "outputs_jeanzay/megmasc/embedding_sweep", "megmasc_by_channel_layerwise.csv",
        ("python scripts/submit.py megmasc embedding fit --family {family} --indices {array} --submit",)),
]

def _audit_spec(spec: SweepSpec) -> list[tuple[str, list[int], int]]:
    """Returns [(family, missing_array_indices, n_expected), ...] for families with any gap."""
    out_root = REPO_ROOT / spec.out_dir
    gaps: list[tuple[str, list[int], int]] = []
    for family in spec.families:
        try:
            tasks = flatten_family_tasks(family)
        except KeyError:
            continue
        missing: list[int] = []
        for t in tasks:
            candidate_dirs = [out_root / family / t.model_dir_safe]
            if spec.also_check_flat:
                candidate_dirs.append(out_root / t.model_dir_safe)
            ok = any(
                d.exists() and any(_non_trivial_csv(p) for p in d.rglob(spec.score_name))
                for d in candidate_dirs
            )
            if not ok:
                missing.append(t.array_index)
        if missing:
            gaps.append((family, missing, len(tasks)))
    return gaps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="only audit specs whose key/label contains this substring")
    ap.add_argument("--commands-only", action="store_true", help="print just the sbatch lines, no prose")
    args = ap.parse_args()

    specs = [s for s in SPECS if s.key not in SKIPPED_KEYS]
    if args.dataset:
        needle = args.dataset.lower()
        specs = [s for s in specs if needle in s.key.lower() or needle in s.label.lower()]
        if not specs:
            print(f"No spec matches '{args.dataset}'.", file=sys.stderr)

    total_gap_families = 0
    total_specs_with_gaps = 0
    all_commands: list[str] = []

    for spec in specs:
        gaps = _audit_spec(spec)
        if not gaps:
            continue
        total_specs_with_gaps += 1
        total_gap_families += len(gaps)
        if not args.commands_only:
            print(f"\n## {spec.label}  ({spec.out_dir}/)")
            if spec.note:
                print(f"   note: {spec.note}")
        for family, missing, n_expected in gaps:
            array = _compress_indices(missing)
            cmds = [spec.env_prefix + t.format(family=family, array=array) for t in spec.stage_templates]
            if not args.commands_only:
                status = "missing" if len(missing) == n_expected else "partial"
                print(f"   - {family}: {status} ({len(missing)}/{n_expected} missing)")
                for c in cmds:
                    print(f"       {c}")
            all_commands.extend(cmds)

    if args.commands_only:
        for c in dict.fromkeys(all_commands):  # de-dup, keep order
            print(c)
        return

    print(f"\n{'=' * 60}")
    print(f"{total_specs_with_gaps} sweep(s) with gaps, {total_gap_families} family-level gap rows total.")
    if total_specs_with_gaps == 0:
        print("Everything audited is complete.")


if __name__ == "__main__":
    main()
