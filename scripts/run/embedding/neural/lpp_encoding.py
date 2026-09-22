#!/usr/bin/env python3
"""Run Le Petit Prince fMRI encoding (GloVe or HuggingFace + HRF + ridge, LOSO).

This script can also load its settings from a YAML config file instead of
individual CLI flags.  When ``--config`` is given, all YAML values are used
as defaults which can still be overridden by explicit CLI flags.

Example (GloVe, CLI flags)
--------------------------
.venv/bin/python scripts/run/embedding/neural/lpp_encoding.py \\
  --bids_root data/lpp_fmri \\
  --subject sub-EN114 --lang en \\
  --bold_source derivatives --out_dir outputs/lpp_encoding_en114

Example (HF + all layers + lateralization)
------------------------------------------
.venv/bin/python scripts/run/embedding/neural/lpp_encoding.py \\
  --bids_root data/lpp_fmri \\
  --subject sub-EN114 --lang en \\
  --feature_mode huggingface --hf_model EleutherAI/pythia-1b \\
  --layer_mode all --lateralization \\
  --out_dir outputs/lpp_pythia1b_layerwise

Example (YAML config)
---------------------
.venv/bin/python scripts/run/embedding/neural/lpp_encoding.py \\
  --config configs/embedding/neural/lpp_encoding.yaml \\
  --subject sub-EN114 --out_dir outputs/lpp_encoding_yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import yaml

from neuroai.datasets.le_petit_prince import discover_lpp_derivative_subjects
from neuroai.lpp_encoding import (
    LppFmriEncoder,
    LppFmriEncodingConfig,
    layerwise_summary_dataframe,
    result_to_dataframe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _load_yaml_defaults(config_path: Path) -> dict:
    """Load YAML config and return a flat dict of defaults for CLI."""
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # Flatten top-level keys for direct use as arg defaults
    defaults: dict = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            defaults[k] = v
    return defaults


def main() -> None:
    p = argparse.ArgumentParser(
        description="LPP voxelwise encoding (Sprint 2–3). "
                    "Use --config to load settings from a YAML file."
    )
    # YAML config (optional; CLI flags override YAML values)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (e.g. configs/embedding/neural/lpp_encoding.yaml). "
             "CLI flags take precedence over YAML values.",
    )
    p.add_argument("--bids_root", type=Path, default=None)
    p.add_argument("--subject", type=str, default=None)
    p.add_argument("--lang", type=str, default=None, choices=("en", "fr", "cn"))
    p.add_argument(
        "--bold_source", type=str, default=None, choices=("raw", "derivatives"),
    )
    p.add_argument("--echo", type=int, default=None)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument(
        "--feature_mode", type=str, default=None, choices=("glove", "huggingface"),
    )
    p.add_argument("--hf_model", type=str, default=None,
                   help="HuggingFace model id when --feature_mode huggingface")
    p.add_argument("--hf_revision", type=str, default=None,
                   help="Optional weights revision (e.g. Pythia step tag)")
    p.add_argument(
        "--hf_tokenizer_no_revision",
        action="store_true",
        default=None,
        help="Load tokenizer from base model (no revision), matching l-bg behavior.",
    )
    p.add_argument("--hf_device", type=str, default=None, help="cuda or cpu")
    p.add_argument("--hf_batch_size", type=int, default=None)
    p.add_argument("--hf_context_words", type=int, default=None)
    p.add_argument("--hf_max_length", type=int, default=None)
    p.add_argument(
        "--layer_mode", type=str, default=None,
        help='Layer selection: "last" (default), "all" (per-layer), or int N.',
    )
    p.add_argument(
        "--lateralization", action="store_true", default=None,
        help="Compute left/right hemisphere Pearson r and asymmetry index.",
    )
    p.add_argument(
        "--picture_nuisance", action="store_true", default=None,
        help="Add section-1 picture-block nuisances (EN/CN; Li et al. 2022)",
    )
    p.add_argument("--save_maps", type=Path, default=None,
                   help="Directory for per-fold Pearson correlation NIfTIs")
    p.add_argument("--max_voxels", type=int, default=None)
    p.add_argument(
        "--fmri_unit",
        type=str,
        default=None,
        choices=("single_subject", "average_subject"),
        help="average_subject = group-mean BOLD like l-bg fit_average_subject.py",
    )
    p.add_argument(
        "--average_subjects",
        type=str,
        default=None,
        help="Comma-separated BIDS subject ids for group mean (average_subject only).",
    )
    p.add_argument(
        "--average_subject_precomputed_dir",
        type=Path,
        default=None,
        help=(
            "Directory with l-bg precomputed average subject files "
            "(average_subject_run-0..8.gz)."
        ),
    )
    p.add_argument("--max_runs", type=int, default=None)
    p.add_argument("--trim_start", type=int, default=None)
    p.add_argument("--trim_end", type=int, default=None)
    p.add_argument("--smoothing_fwhm", type=float, default=None)
    p.add_argument("--high_pass_hz", type=float, default=None)
    p.add_argument("--hrf_model", type=str, default=None)
    p.add_argument("--strict_lbg_parity", action="store_true", default=None)
    p.add_argument("--min_word_duration_sec", type=float, default=None)
    p.add_argument("--lbg_fixes", action="store_true", default=None)
    p.add_argument("--mask_runs", type=int, default=None)
    p.add_argument(
        "--feature_cache_dir", type=Path, default=None,
        help="Directory to cache pre-extracted LLM word features.",
    )
    p.add_argument(
        "--untrained", action="store_true", default=None,
        help="Use randomly initialised weights (same architecture) instead of "
             "pretrained checkpoints -- untrained baseline for brain scores.",
    )
    p.add_argument("--untrained_seed", type=int, default=None)
    args = p.parse_args()

    # --- Merge YAML defaults with CLI overrides ---
    yaml_defaults: dict = {}
    if args.config is not None:
        yaml_defaults = _load_yaml_defaults(args.config)

    def _get(attr: str, fallback):
        val = getattr(args, attr, None)
        return val if val is not None else yaml_defaults.get(attr, fallback)

    bids_root = _get("bids_root", None)
    subject = _get("subject", None)
    out_dir_val = _get("out_dir", None)

    if bids_root is None:
        p.error("--bids_root is required (or set bids_root in YAML config).")
    if out_dir_val is None:
        p.error("--out_dir is required (or set out_dir in YAML config).")

    lang = _get("lang", "en")
    bold_source = _get("bold_source", "derivatives")
    echo_val = _get("echo", 2)
    feature_mode = _get("feature_mode", "glove")
    hf_model = _get("hf_model", yaml_defaults.get("hf_model_name", "openai-community/gpt2"))
    hf_revision = _get("hf_revision", yaml_defaults.get("hf_revision", None))
    hf_device = _get("hf_device", yaml_defaults.get("hf_device", "cpu"))
    hf_batch_size = _get("hf_batch_size", yaml_defaults.get("hf_batch_size", 4))
    hf_tokenizer_no_revision = bool(
        _get(
            "hf_tokenizer_no_revision",
            yaml_defaults.get("hf_tokenizer_no_revision", False),
        )
    )
    hf_context_words = _get("hf_context_words", yaml_defaults.get("hf_context_words", 48))
    hf_max_length = _get("hf_max_length", yaml_defaults.get("hf_max_length", 512))
    hf_extraction_mode = _get(
        "hf_extraction_mode", yaml_defaults.get("hf_extraction_mode", "context_window")
    )
    hf_lbg_story_text_zip = _get(
        "hf_lbg_story_text_zip", yaml_defaults.get("hf_lbg_story_text_zip", None)
    )
    hf_lbg_stride_delta = int(
        _get("hf_lbg_stride_delta", yaml_defaults.get("hf_lbg_stride_delta", 64))
    )
    hf_lbg_max_length = _get(
        "hf_lbg_max_length", yaml_defaults.get("hf_lbg_max_length", None)
    )
    fmri_unit_raw = _get("fmri_unit", yaml_defaults.get("fmri_unit", "single_subject"))
    fmri_unit = str(fmri_unit_raw).lower().strip()
    if fmri_unit not in ("single_subject", "average_subject"):
        p.error(f"fmri_unit must be single_subject or average_subject, got {fmri_unit!r}")

    avg_cli = getattr(args, "average_subjects", None)
    avg_yaml = yaml_defaults.get("average_subject_subjects")
    avg_precomputed_dir = _get(
        "average_subject_precomputed_dir",
        yaml_defaults.get("average_subject_precomputed_dir", None),
    )
    avg_subjects: tuple[str, ...] | None = None
    if avg_cli:
        avg_subjects = tuple(s.strip() for s in str(avg_cli).split(",") if s.strip())
    elif isinstance(avg_yaml, list) and avg_yaml:
        avg_subjects = tuple(str(s) for s in avg_yaml)

    if fmri_unit == "average_subject":
        discovered_subjects = tuple(
            discover_lpp_derivative_subjects(Path(bids_root), str(lang))
        )
        if avg_precomputed_dir is None and not avg_subjects:
            avg_subjects = discovered_subjects
        if avg_precomputed_dir is None and not avg_subjects:
            p.error(
                "fmri_unit=average_subject: set --average_subjects or YAML "
                "average_subject_subjects, or add derivatives/sub-<LANG>* folders."
            )
        if subject is None:
            if avg_subjects:
                subject = avg_subjects[0]
            else:
                if discovered_subjects:
                    subject = discovered_subjects[0]
                else:
                    p.error(
                        "fmri_unit=average_subject with precomputed averages still "
                        "needs one available subject in derivatives/ to resolve run↔section."
                    )
    elif subject is None:
        p.error("--subject is required (or set subject in YAML config).")

    max_voxels = _get("max_voxels", yaml_defaults.get("max_voxels", 4000))
    rv_frac_raw = yaml_defaults.get("reliable_voxel_fraction")
    reliable_voxel_fraction = (
        float(rv_frac_raw) if rv_frac_raw is not None else None
    )
    rv_path_raw = yaml_defaults.get("reliable_voxel_scores_path")
    reliable_voxel_scores_path = (
        Path(rv_path_raw) if rv_path_raw not in (None, "", "null") else None
    )
    max_runs = _get("max_runs", yaml_defaults.get("max_runs", None))
    trim_start = _get("trim_start", yaml_defaults.get("trim_trs_start", 10))
    trim_end = _get("trim_end", yaml_defaults.get("trim_trs_end", 10))
    smoothing_fwhm = _get("smoothing_fwhm", yaml_defaults.get("smoothing_fwhm", 4.0))
    lbg_fixes = bool(_get("lbg_fixes", yaml_defaults.get("apply_lbg_fixes", False)))
    mask_runs = _get("mask_runs", yaml_defaults.get("mask_from_runs", 3))
    high_pass_hz = _get("high_pass_hz", yaml_defaults.get("high_pass_hz", 1.0 / 128.0))
    hrf_model = _get("hrf_model", yaml_defaults.get("hrf_model", "glover"))
    strict_lbg_parity = bool(
        _get("strict_lbg_parity", yaml_defaults.get("strict_lbg_parity", False))
    )
    min_word_duration_sec = float(
        _get("min_word_duration_sec", yaml_defaults.get("min_word_duration_sec", 0.05))
    )
    ridge_alphas_raw = yaml_defaults.get("ridge_alphas", None)
    ridge_alphas = (
        [float(a) for a in ridge_alphas_raw]
        if ridge_alphas_raw is not None
        else None
    )
    picture_nuisance = bool(_get("picture_nuisance", yaml_defaults.get("picture_nuisance_regressors", False)))
    save_maps = _get("save_maps", yaml_defaults.get("save_corr_maps_dir", None))
    feature_cache_dir = _get("feature_cache_dir", yaml_defaults.get("feature_cache_dir", None))
    lateralization = bool(_get("lateralization", yaml_defaults.get("compute_lateralization", False)))
    hf_untrained = bool(_get("untrained", yaml_defaults.get("hf_untrained", False)))
    hf_untrained_seed = int(_get("untrained_seed", yaml_defaults.get("hf_untrained_seed", 0)))
    if hf_untrained:
        # Keep untrained-baseline caches/outputs fully separate from trained runs.
        out_dir_val = Path(str(out_dir_val).rstrip("/") + "_untrained")
        if feature_cache_dir is not None:
            feature_cache_dir = Path(str(feature_cache_dir).rstrip("/") + "_untrained")

    # Parse layer_mode (string or int)
    layer_mode_raw = _get("layer_mode", yaml_defaults.get("layer_mode", "last"))
    if layer_mode_raw is None or layer_mode_raw == "last":
        layer_mode = "last"
    elif layer_mode_raw == "all":
        layer_mode = "all"
    else:
        try:
            layer_mode = int(layer_mode_raw)
        except (ValueError, TypeError):
            layer_mode = "last"

    rev = hf_revision
    if rev in ("", "none", "None"):
        rev = None

    cfg = LppFmriEncodingConfig(
        bids_root=Path(bids_root),
        subject=str(subject),
        lang=str(lang),
        bold_source=bold_source,
        echo=None if bold_source == "derivatives" else echo_val,
        feature_mode=feature_mode,
        hf_model_name=str(hf_model),
        hf_revision=rev,
        hf_device=str(hf_device),
        hf_batch_size=int(hf_batch_size),
        hf_tokenizer_no_revision=hf_tokenizer_no_revision,
        hf_context_words=int(hf_context_words),
        hf_max_length=int(hf_max_length),
        hf_extraction_mode=str(hf_extraction_mode),
        hf_lbg_story_text_zip=Path(hf_lbg_story_text_zip)
        if hf_lbg_story_text_zip is not None
        else None,
        hf_lbg_stride_delta=hf_lbg_stride_delta,
        hf_lbg_max_length=int(hf_lbg_max_length)
        if hf_lbg_max_length is not None
        else None,
        layer_mode=layer_mode,
        feature_cache_dir=feature_cache_dir,
        picture_nuisance_regressors=picture_nuisance,
        hrf_model=str(hrf_model),
        trim_trs_start=int(trim_start),
        trim_trs_end=int(trim_end),
        ridge_alphas=ridge_alphas,
        max_voxels=max_voxels,
        reliable_voxel_fraction=reliable_voxel_fraction,
        reliable_voxel_scores_path=reliable_voxel_scores_path,
        max_runs=max_runs,
        smoothing_fwhm=float(smoothing_fwhm) if smoothing_fwhm is not None else None,
        high_pass_hz=float(high_pass_hz) if high_pass_hz is not None else None,
        strict_lbg_parity=strict_lbg_parity,
        min_word_duration_sec=float(min_word_duration_sec),
        apply_lbg_fixes=lbg_fixes,
        mask_from_runs=mask_runs,
        save_corr_maps_dir=save_maps,
        compute_lateralization=lateralization,
        fmri_unit=fmri_unit,
        average_subject_subjects=avg_subjects if fmri_unit == "average_subject" else None,
        average_subject_precomputed_dir=Path(avg_precomputed_dir)
        if avg_precomputed_dir is not None
        else None,
        hf_untrained=hf_untrained,
        hf_untrained_seed=hf_untrained_seed,
    )
    out_dir_path = Path(out_dir_val)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    encoder = LppFmriEncoder(cfg)

    if cfg.layer_mode == "all" and cfg.feature_mode == "huggingface":
        # Per-layer analysis
        lw_result = encoder.run_layerwise()
        from neuroai.lpp_encoding import layerwise_result_to_dataframe
        runs_df = layerwise_result_to_dataframe(lw_result)
        summary_df = layerwise_summary_dataframe(lw_result)
        sub_tag = lw_result.subject.replace("/", "_")
        runs_csv = out_dir_path / f"{sub_tag}_lpp_layerwise_runs.csv"
        summary_csv = out_dir_path / f"{sub_tag}_lpp_layerwise_summary.csv"
        runs_df.to_csv(runs_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        summary_txt = out_dir_path / f"{sub_tag}_lpp_layerwise_summary.txt"
        summary_txt.write_text(
            f"subject={lw_result.subject}\n"
            f"lang={lw_result.lang}\n"
            f"n_layers={lw_result.n_layers}\n"
            f"best_layer={lw_result.best_layer}\n"
            f"overall_mean_pearson={lw_result.overall_mean_pearson:.6f}\n"
            f"overall_lateralization_index={lw_result.overall_lateralization_index:.6f}\n",
            encoding="utf-8",
        )
        logger.info("Wrote %s, %s and %s", runs_csv, summary_csv, summary_txt)
        logger.info(
            "Best layer: %d  mean_pearson=%.4f  lat=%.4f",
            lw_result.best_layer, lw_result.overall_mean_pearson,
            lw_result.overall_lateralization_index,
        )
    else:
        # Single-layer / GloVe analysis
        result = encoder.run_leave_one_run_out()
        df = result_to_dataframe(result)
        sub_tag = result.subject.replace("/", "_")
        csv_path = out_dir_path / f"{sub_tag}_lpp_encoding_runs.csv"
        df.to_csv(csv_path, index=False)
        summary_path = out_dir_path / f"{sub_tag}_lpp_encoding_summary.txt"
        summary_path.write_text(
            f"subject={result.subject}\n"
            f"lang={result.lang}\n"
            f"feature_mode={cfg.feature_mode}\n"
            f"layer_mode={cfg.layer_mode}\n"
            f"overall_mean_pearson={result.overall_mean_pearson:.6f}\n"
            f"overall_mean_pearson_left={result.overall_mean_pearson_left:.6f}\n"
            f"overall_mean_pearson_right={result.overall_mean_pearson_right:.6f}\n"
            f"overall_lateralization_index={result.overall_lateralization_index:.6f}\n",
            encoding="utf-8",
        )
        logger.info("Wrote %s and %s", csv_path, summary_path)
        logger.info(
            "Overall mean Pearson: %.4f  left=%.4f  right=%.4f  lat=%.4f",
            result.overall_mean_pearson,
            result.overall_mean_pearson_left,
            result.overall_mean_pearson_right,
            result.overall_lateralization_index,
        )


if __name__ == "__main__":
    main()
