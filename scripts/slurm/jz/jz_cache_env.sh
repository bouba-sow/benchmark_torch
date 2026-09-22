# Jean Zay lustre cache layout — source from login scripts or SLURM job headers.
#
#   export SCRATCH=/lustre/fsn1/projects/rech/ldh/una68ug
#   export WORK=/lustre/fswork/projects/rech/ldh/una68ug
#   source scripts/jz_cache_env.sh
#
# All benchmark/tool caches live under $SCRATCH/.cache (not $WORK).

if [[ -z "${SCRATCH:-}" ]]; then
  echo "jz_cache_env.sh: SCRATCH is not set — skipping lustre cache exports." >&2
  return 0 2>/dev/null || exit 0
fi

export BENCHMARK_ROOT="${BENCHMARK_ROOT:-${WORK}/benchmark}"
export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$SCRATCH/.cache}"
export LPP_FEATURE_CACHE_DIR="${LPP_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/lpp_features}"
export NS_FEATURE_CACHE_DIR="${NS_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/naturalstories_features}"
export PODCAST_FEATURE_CACHE_DIR="${PODCAST_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/podcast_features}"
export PODCAST_EPOCH_CACHE_DIR="${PODCAST_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/podcast_epochs}"
export BRODERICK_FEATURE_CACHE_DIR="${BRODERICK_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_features}"
export BRODERICK_EPOCH_CACHE_DIR="${BRODERICK_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/broderick_epochs}"
export NIEUWLAND_BIDS_ROOT="${NIEUWLAND_BIDS_ROOT:-data/nieuwland}"
export NIEUWLAND_FEATURE_CACHE_DIR="${NIEUWLAND_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_features}"
export NIEUWLAND_EPOCH_CACHE_DIR="${NIEUWLAND_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/nieuwland_epochs}"
export MEGMASC_BIDS_ROOT="${MEGMASC_BIDS_ROOT:-$SCRATCH/data/gwilliams}"
export MEGMASC_FEATURE_CACHE_DIR="${MEGMASC_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_features}"
export MEGMASC_EPOCH_CACHE_DIR="${MEGMASC_EPOCH_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_epochs}"
export MEGMASC_LME_FEATURE_CACHE_DIR="${MEGMASC_LME_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/megmasc_lme_features}"
export WHISPER_CACHE_DIR="${WHISPER_CACHE_DIR:-$SCRATCH/.cache/whisper}"
export ZUCO_FEATURE_CACHE_DIR="${ZUCO_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/zuco_features}"
export ZUCO_MAT_ROOT="${ZUCO_MAT_ROOT:-${BENCHMARK_ROOT:-${WORK}/benchmark}/data/zuco_reading/osfstorage}"
export NS_FMRI_FEATURE_CACHE_DIR="${NS_FMRI_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/ns_fmri_features}"
export NS_FMRI_DATA_ROOT="${NS_FMRI_DATA_ROOT:-${BENCHMARK_ROOT:-${WORK}/benchmark}/data/naturalstories_fmri/osfstorage}"
export FEDORENKO_FEATURE_CACHE_DIR="${FEDORENKO_FEATURE_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/fedorenko2016_features}"

#export UV_CACHE_DIR="${UV_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/uv}"
#export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/pip}"
export TORCH_HOME="${TORCH_HOME:-$BENCHMARK_CACHE_ROOT/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$BENCHMARK_CACHE_ROOT/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BENCHMARK_CACHE_ROOT/triton}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Optional: datasets / nilearn (override if you relocated them)
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${WORK:-$HOME}/hf_datasets}"
export NILEARN_DATA="${NILEARN_DATA:-${WORK:-$HOME}/nilearn_data}"

module load r/4.5.2
module load gcc/11.5.0

mkdir -p \
  "$BENCHMARK_CACHE_ROOT" \
  "$LPP_FEATURE_CACHE_DIR" \
  "$NS_FEATURE_CACHE_DIR" \
  "$PODCAST_FEATURE_CACHE_DIR" \
  "$PODCAST_EPOCH_CACHE_DIR" \
  "$BRODERICK_FEATURE_CACHE_DIR" \
  "$BRODERICK_EPOCH_CACHE_DIR" \
  "$NIEUWLAND_FEATURE_CACHE_DIR" \
  "$NIEUWLAND_EPOCH_CACHE_DIR" \
  "$MEGMASC_FEATURE_CACHE_DIR" \
  "$MEGMASC_EPOCH_CACHE_DIR" \
  "$WHISPER_CACHE_DIR" \
  "$ZUCO_FEATURE_CACHE_DIR" \
  "$NS_FMRI_FEATURE_CACHE_DIR" \
  "$FEDORENKO_FEATURE_CACHE_DIR" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  "$TRITON_CACHE_DIR" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE"
