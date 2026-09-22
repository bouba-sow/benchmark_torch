#!/usr/bin/env bash
# Prefetch Hugging Face weights for ALL benchmark models used in the neural alignment sweep.
#
# Models with training-step revisions (Pythia, OLMo-2):
#   - Pythia deduped:
#   - OLMo-2: steps match OLMO2_*_DEFAULT_REVISIONS in your sweep config
#     * OLMo-2-0425-1B  : revision format  stage1-stepXXX-tokensYYYB
#     * OLMo-2-1124-7B  : revision format  stepXXX-tokensYYYB  (every 1000 steps)
#     * OLMo-2-1124-13B : revision format  stepXXX-tokensYYYB  (every 1000 steps)
#
# Models downloaded at main only (no public intermediate checkpoints):
#   GPT-2, GPT-Neo/J/NeoX, OPT, Mamba, Mamba-2, RWKV-7
#
# Needs internet, so run it on a login node or via sbatch (compute nodes run HF_HUB_OFFLINE=1 and
# have no internet). Idempotent: already-cached repos/revisions are skipped, so re-run it after
# adding models or checkpoints.
#
#   bash scripts/slurm/torch/utils/setup/prefetch_models_hf.sh                       # login node
#   sbatch --account=torch_pr_287_general --partition=cpu_short --cpus-per-task=8 --time=03:30:00 \
#          --output=outputs_torch/logs/prefetch_hf_%j.out scripts/slurm/torch/utils/setup/prefetch_models_hf.sh
#   # cpu_short caps near 4h in practice (docs/torch_cluster.md); for a longer run use
#   # --partition=cs instead (no hard per-job cap, real limit comes from the CPU QOS).
#
# Also fetches the Nilearn Harvard-Oxford atlas and warms the LPP parcel cache (needed offline by the
# LPP ROI code); skip that with PREFETCH_ATLAS=0.
#
# NOTE: Set HF_TOKEN in your environment or in ~/.huggingface/token.
#       Do NOT hardcode tokens in this file.
set -euo pipefail

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if ! command -v hf >/dev/null 2>&1; then
    echo "error: 'hf' not found. Install huggingface_hub and use the hf CLI." >&2
    exit 1
fi

if [[ -z "${HF_HOME:-}" ]]; then
    if [[ -n "${SCRATCH:-}" ]]; then
        export HF_HOME="$SCRATCH/hf_cache"
    else
        echo "error: set HF_HOME or SCRATCH before running (downloads would use ~/.cache and fill HOME)." >&2
        exit 1
    fi
fi
REPO="${REPO:-$SCRATCH/Documents/benchmark}"
cd "${REPO}"

if [[ -z "${SCRATCH:-}" ]]; then
  echo "error: set SCRATCH (lustre project dir) before prefetch." >&2
  exit 1
fi
# shellcheck source=scripts/slurm/torch/torch_cache_env.sh
source "${REPO}/scripts/slurm/torch/torch_cache_env.sh"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "warning: HF_TOKEN is not set. Private or gated models may fail." >&2
fi

echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_XET_HIGH_PERFORMANCE=0

# ---------------------------------------------------------------------------
# Helper: download with exponential-backoff retry (handles 429/503 rate limits)
# ---------------------------------------------------------------------------
_failed_repos=()

download() {
    local repo="$1"
    local rev="${2:-main}"
    local max_attempts=5 attempt=1 delay=10

    while (( attempt <= max_attempts )); do
        echo "[attempt $attempt/$max_attempts] hf download ${repo} --revision ${rev}"
        if hf download "$repo" --revision "$rev" 2>&1; then
            return 0
        fi
        if (( attempt < max_attempts )); then
            echo "  → retry in ${delay}s (rate limit or transient error)..." >&2
            sleep "$delay"
            delay=$(( delay * 2 ))   # 10 → 20 → 40 → 80 → 160 s
        fi
        attempt=$(( attempt + 1 ))
    done

    echo "ERROR: failed to download ${repo} @ ${rev} after ${max_attempts} attempts" >&2
    _failed_repos+=("${repo} @ ${rev}")
    return 1
}

# ===========================================================================
# 1. GPT-2 family  (main only)
# ===========================================================================
#echo "### GPT-2 family ###"
#GPT2_REPOS=(
#    "openai-community/gpt2"           # 124M  (also available as "gpt2")
#    "openai-community/gpt2-medium"    # 355M
#    "openai-community/gpt2-large"     # 774M
#    "openai-community/gpt2-xl"        # 1.6B
#)
#for repo in "${GPT2_REPOS[@]}"; do
#    download "${repo}" main
#done
#
## ===========================================================================
## 2. GPT-Neo / GPT-J / GPT-NeoX family  (main only)
## ===========================================================================
#echo
#echo "### GPT-Neo / GPT-J / GPT-NeoX family (EleutherAI) ###"
#GPTNEO_REPOS=(
#    "EleutherAI/gpt-neo-125m"
#    "EleutherAI/gpt-neo-1.3B"
#    "EleutherAI/gpt-neo-2.7B"
#    "EleutherAI/gpt-j-6b"
#    "EleutherAI/gpt-neox-20b"
#)
#for repo in "${GPTNEO_REPOS[@]}"; do
#    download "${repo}" main
#done
#
## ===========================================================================
## 3. OPT family  (main only)
## ===========================================================================
#echo
#echo "### OPT family (Meta) ###"
#OPT_REPOS=(
#    "facebook/opt-125m"
#    "facebook/opt-350m"
#    "facebook/opt-1.3b"
#    "facebook/opt-2.7b"
#    "facebook/opt-6.7b"
#   "facebook/opt-13b"
#    "facebook/opt-30b"
#    "facebook/opt-66b"
#)
#for repo in "${OPT_REPOS[@]}"; do
#    download "${repo}" main
#done
#
## ===========================================================================
## 4. Mamba family  (main only — use -hf variants for transformers compatibility)
## ===========================================================================
#echo
#echo "### Mamba family (state-spaces) ###"
## The -hf suffix variants are natively compatible with transformers (MambaForCausalLM).
## They require: pip install causal-conv1d mamba-ssm
#MAMBA_REPOS=(
#    "state-spaces/mamba-130m-hf"
#    "state-spaces/mamba-370m-hf"
#    "state-spaces/mamba-790m-hf"
#    "state-spaces/mamba-1.4b-hf"
#    "state-spaces/mamba-2.8b-hf"
#)
#for repo in "${MAMBA_REPOS[@]}"; do
#    download "${repo}" main
#done
#
## ===========================================================================
## 5. Mamba-2 family  (main only)
## ===========================================================================
#echo
#echo "### Mamba-2 family (state-spaces) ###"
#MAMBA2_REPOS=(
#    "AntonV/mamba2-130m-hf"
#    "AntonV/mamba2-370m-hf"
#    "AntonV/mamba2-780m-hf"
#    "AntonV/mamba2-1.3b-hf"
#    "AntonV/mamba2-2.7b-hf"
#)
#for repo in "${MAMBA2_REPOS[@]}"; do
#    download "${repo}" main
#done
#
## ===========================================================================
## 6. RWKV-7 family  (main only)
##    fla-hub variants: transformers-compatible, require flash-linear-attention
##    and trust_remote_code=True
## ===========================================================================
#echo
#echo "### RWKV-7 family (fla-hub) ###"
#RWKV7_REPOS=(
#    "fla-hub/rwkv7-191M-world"
#    "fla-hub/rwkv7-430M-world"
#    "fla-hub/rwkv7-1.5B-world"
#    "fla-hub/rwkv7-2.9B-world"
#)
##for repo in "${RWKV7_REPOS[@]}"; do
##    download "${repo}" main
##done
#
# ===========================================================================
# 7. Pythia deduped  (main + training-step revisions)
#    Keep STEPS in sync with LPP_PYTHIA_STEPS in neuroai/models/registry.py
# ===========================================================================
echo
echo "### Pythia deduped (EleutherAI) — main + step revisions ###"
PYTHIA_REPOS=(
    "EleutherAI/pythia-14m-deduped"
    "EleutherAI/pythia-31m-deduped"
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-160m-deduped"
    "EleutherAI/pythia-410m-deduped"
    "EleutherAI/pythia-1b-deduped"
    "EleutherAI/pythia-1.4b-deduped"
    "EleutherAI/pythia-2.8b-deduped"
    "EleutherAI/pythia-6.9b-deduped"
    "EleutherAI/pythia-12b-deduped"
)
PYTHIA_STEPS=(
    0
    128
    256
    512
    2000
    8000
    16000
    32000
    64000
    128000
    143000
)

echo "Prefetching main (tokenizer snapshot) per Pythia repo ..."
#for repo in "${PYTHIA_REPOS[@]}"; do
#    download "${repo}" main
#done

echo "Prefetching Pythia training-step revisions ..."
#for repo in "${PYTHIA_REPOS[@]}"; do
#    for step in "${PYTHIA_STEPS[@]}"; do
#        download "${repo}" "step${step}"
#    done
#done
# ===========================================================================
# 8. OLMo-2  (main + training-step revisions)
#
#    OLMo-2-0425-1B  : stage1-stepXXX-tokensYYYB
#      Keep OLMO2_0425_1B_REVISIONS in sync with OLMO2_0425_1B_DEFAULT_REVISIONS
#
#    OLMo-2-1124-7B / 13B : stepXXX-tokensYYYB  (every 1000 steps up to final)
#      Keep the step lists in sync with OLMO2_1124_*_DEFAULT_REVISIONS
# ===========================================================================
#echo
#echo "### OLMo-2 (AllenAI) — main + step revisions ###"

# --- OLMo-2-0425-1B ---
# Revision format: stage1-stepXXX-tokensYYYB
# Update this list to match OLMO2_0425_1B_DEFAULT_REVISIONS in your sweep config.
#OLMO2_1B_EARLY_TRAINING_REPO="allenai/OLMo-2-0425-1B-early-training"
#OLMO2_1B_EARLY_TRAINING_REVISIONS=(
#    #"main",
#    "stage1-step0-tokens0B"
#    "stage1-step1000-tokens3B"
#    "stage1-step2000-tokens5B"  
#    "stage1-step4000-tokens9B"
#    "stage1-step8000-tokens17B"
#    "stage1-step16000-tokens34B"  
#    "stage1-step30000-tokens63B"
#)
#
##download "${OLMO2_1B_EARLY_TRAINING_REPO}" main
#for rev in "${OLMO2_1B_EARLY_TRAINING_REVISIONS[@]}"; do
#    download "${OLMO2_1B_EARLY_TRAINING_REPO}" "${rev}"
#done
#
#OLMO2_1B_REPO="allenai/OLMo-2-0425-1B"
#OLMO2_1B_REVISIONS=(
#    #"main",
#    "stage1-step60000-tokens126B"
#    "stage1-step120000-tokens252B"
#    "stage1-step260000-tokens546B"
#    "stage1-step520000-tokens1091B"
#    "stage1-step1040000-tokens2182B"
#    "stage1-step1500000-tokens3146B"
#    "stage1-step1907359-tokens4001B"
#)
##download "${OLMO2_1B_REPO}" main
#for rev in "${OLMO2_1B_REVISIONS[@]}"; do
#    download "${OLMO2_1B_REPO}" "${rev}"
#done
#
## Early checkpoints also live on a separate HF repo (same revision names).
#OLMO2_1B_EARLY_REPO="allenai/OLMo-2-0425-1B-early-training"
#OLMO2_1B_EARLY_REVISIONS=(
#    "stage1-step0-tokens0B"
#    "stage1-step1000-tokens3B"
#    "stage1-step2000-tokens5B"
#    "stage1-step4000-tokens9B"
#    "stage1-step8000-tokens17B"
#    "stage1-step16000-tokens34B"
#    "stage1-step30000-tokens63B"
#)
#for rev in "${OLMO2_1B_EARLY_REVISIONS[@]}"; do
#    download "${OLMO2_1B_EARLY_REPO}" "${rev}"
#done

# --- OLMo-2-1124-7B ---
# Revision format: stepXXX-tokensYYYB  (every 1000 steps)
# Update this list to match OLMO2_1124_7B_DEFAULT_REVISIONS in your sweep config.
#OLMO2_7B_REPO="allenai/OLMo-2-1124-7B"
#OLMO2_7B_REVISIONS=(
#    "stage1-step1000-tokens5B"
#    "stage1-step2000-tokens9B"
#    "stage1-step4000-tokens17B"
#    "stage1-step8000-tokens34B"
#    "stage1-step16000-tokens68B"
#    "stage1-step32000-tokens135B"
#    "stage1-step64000-tokens269B"
#    "stage1-step128000-tokens537B"
#    "stage1-step256000-tokens1074B"   
#    "stage1-step528000-tokens2215B"
#    "stage1-step700000-tokens2937B"
#    "stage1-step928646-tokens3896B" # final checkpoint (~2T tokens)
#)
#download "${OLMO2_7B_REPO}" main
#for rev in "${OLMO2_7B_REVISIONS[@]}"; do
#    download "${OLMO2_7B_REPO}" "${rev}"
#done

# --- OLMo-2-1124-13B ---
# Same revision format as 7B.
# Update this list to match OLMO2_1124_13B_DEFAULT_REVISIONS in your sweep config.
#OLMO2_13B_REPO="allenai/OLMo-2-1124-13B"
#OLMO2_13B_REVISIONS=(
#    "stage1-step0-tokens0B"
#    "stage1-step1000-tokens9B"
#    "stage1-step2000-tokens17B"
#    "stage1-step4000-tokens34B"
#    "stage1-step8000-tokens68B"
#    "stage1-step16000-tokens135B"
#    "stage1-step32000-tokens269B"
#    "stage1-step64000-tokens537B"
#    "stage1-step128000-tokens1074B"
#    "stage1-step143000-tokens1200B"
#    "stage1-step256000-tokens2148B"
#    "stage1-step350000-tokens2937B"
#    "stage1-step420000-tokens3524B"
#    "stage1-step477000-tokens4002B"
#    'stage1-step520000-tokens4363B'
#    'stage1-step596057-tokens5001B'
#)
#download "${OLMO2_13B_REPO}" main
#for rev in "${OLMO2_13B_REVISIONS[@]}"; do
#    download "${OLMO2_13B_REPO}" "${rev}"
#done

# ---Qwen3 ---
#echo "### Qwen3 family (main only) ###"
#QWEN3_REPOS=(
#    "Qwen/Qwen3-0.6B-Base"   # also available as "Qwen/Qwen3-1.5B-v0.9.0"
#    "Qwen/Qwen3-1.7B-Base"   # also available as "Qwen/Qwen3-1.7B-v0.9.0"
#    "Qwen/Qwen3-4B-Base"   # also available as "Qwen/Qwen3-3.5B-v0.9.0"
#    "Qwen/Qwen3-8B-Base"   # also available as "Qwen/Qwen3-7B-v0.9.0"
#    "Qwen/Qwen3-14B-Base"   # also available as "Qwen/Qwen3-14B-v0.9.0"
#)
#for repo in "${QWEN3_REPOS[@]}"; do
#    download "${repo}" main
#done
#
## Llama3.2 family of models
#echo
#echo "### Llama3.2 family (Meta) ###"
#LLAMA32_REPOS=(
#    "meta-llama/Llama-3.2-1B"
#    "meta-llama/Llama-3.2-3B"
#    "meta-llama/Llama-3.1-8B"
#)
#for repo in "${LLAMA32_REPOS[@]}"; do
#    download "${repo}" main
#done
#
## Ministral-3 family — multimodal checkpoints; loaded via AutoModelForVision2Seq
## with .language_model extraction (see neuroai/features/llm.py _get_cached_causal_lm)
#echo
#echo "### Mistral family (Mistral AI) ###"
#MISTRAL_REPOS=(
#    mistralai/Ministral-3-3B-Base-2512
#    mistralai/Ministral-3-8B-Base-2512
#    mistralai/Ministral-3-14B-Base-2512
#)
#for repo in "${MISTRAL_REPOS[@]}"; do
#    download "${repo}" main
#done
# ===========================================================================
# 9. Stickbreaking family  (Lanni-ni, main only, trust_remote_code=True)
#    Memory-constrained models: 2-layer (~27.8M) and 4-layer (~48.1M) variants,
#    trained on The Pile or BabyLM (10M / 100M tokens).
# ===========================================================================
echo
echo "### Stickbreaking family (Lanni-ni) ###"
STICKBREAKING_REPOS=(
    "Lanni-ni/stickbreaking_pile_2layer"
    "Lanni-ni/stickbreaking_pile_4layer"
    "Lanni-ni/stickbreaking_babylm_10m_2layer"
    "Lanni-ni/stickbreaking_babylm_10m_4layer"
    "Lanni-ni/stickbreaking_babylm_100m_2layer"
    "Lanni-ni/stickbreaking_babylm_100m_4layer"
)
#for repo in "${STICKBREAKING_REPOS[@]}"; do
#    download "${repo}" main
#done

# ===========================================================================
# 10. ALiBi family  (Lanni-ni, main only, trust_remote_code=True)
#     Same corpus/layer variants as stickbreaking; ALiBi positional bias.
# ===========================================================================
echo
echo "### ALiBi family (Lanni-ni) ###"
ALIBI_REPOS=(
    "Lanni-ni/alibi_pile_2layer"
    "Lanni-ni/alibi_pile_4layer"
    "Lanni-ni/alibi_babylm_10m_2layer"
    "Lanni-ni/alibi_babylm_10m_4layer"
    "Lanni-ni/alibi_babylm_100m_2layer"
    "Lanni-ni/alibi_babylm_100m_4layer"
)
#for repo in "${ALIBI_REPOS[@]}"; do
#    download "${repo}" main
#done

# ===========================================================================
# 11. Forgetting family  (Lanni-ni, main only, trust_remote_code=True)
#     Forgetting-softmax attention (Fox et al.); same corpus/layer variants.
# ===========================================================================
echo
echo "### Forgetting family (Lanni-ni) ###"
FORGETTING_REPOS=(
    "Lanni-ni/forgetting_pile_2layer"
    "Lanni-ni/forgetting_pile_4layer"
    "Lanni-ni/forgetting_babylm_10m_2layer"
    "Lanni-ni/forgetting_babylm_10m_4layer"
    "Lanni-ni/forgetting_babylm_100m_2layer"
    "Lanni-ni/forgetting_babylm_100m_4layer"
)
#for repo in "${FORGETTING_REPOS[@]}"; do
#    download "${repo}" main
#done

# ===========================================================================
# 11b. Hard n-gram family  (Lanni-ni, main only, trust_remote_code=True)
#      Fixed local sliding-window attention baselines (hard_{2,3,5}gram);
#      same corpus/layer variants as stickbreaking/alibi/forgetting. Loaded
#      via forgetting_transformer.model.sliding_window — needs the same
#      PYTHONPATH=forgetting-transformer/src as the forgetting family.
# ===========================================================================
echo
echo "### Hard n-gram family (Lanni-ni) ###"
HARD_2GRAM_REPOS=(
    "Lanni-ni/hard_2gram_pile_2layer"
    "Lanni-ni/hard_2gram_pile_4layer"
    "Lanni-ni/hard_2gram_babylm_10m_2layer"
    "Lanni-ni/hard_2gram_babylm_10m_4layer"
    "Lanni-ni/hard_2gram_babylm_100m_2layer"
    "Lanni-ni/hard_2gram_babylm_100m_4layer"
)
HARD_3GRAM_REPOS=(
    "Lanni-ni/hard_3gram_pile_2layer"
    "Lanni-ni/hard_3gram_pile_4layer"
    "Lanni-ni/hard_3gram_babylm_10m_2layer"
    "Lanni-ni/hard_3gram_babylm_10m_4layer"
    "Lanni-ni/hard_3gram_babylm_100m_2layer"
    "Lanni-ni/hard_3gram_babylm_100m_4layer"
)
HARD_5GRAM_REPOS=(
    "Lanni-ni/hard_5gram_pile_2layer"
    "Lanni-ni/hard_5gram_pile_4layer"
    "Lanni-ni/hard_5gram_babylm_10m_2layer"
    "Lanni-ni/hard_5gram_babylm_10m_4layer"
    "Lanni-ni/hard_5gram_babylm_100m_2layer"
    "Lanni-ni/hard_5gram_babylm_100m_4layer"
)
#for repo in "${HARD_2GRAM_REPOS[@]}" "${HARD_3GRAM_REPOS[@]}" "${HARD_5GRAM_REPOS[@]}"; do
#    download "${repo}" main
#done

# ===========================================================================
# 11c. CBR-RNN family  (cclark, main only, custom PyTorchModelHubMixin arch —
#      not AutoModelForCausalLM, no trust_remote_code needed. Also needs the
#      tokenizer prefetched separately since AutoTokenizer.from_pretrained is
#      called directly, not covered by the `hf download` weights-only fetch.)
# ===========================================================================
echo
echo "### CBR-RNN family (cclark) ###"
CBR_RNN_REPOS=(
    "cclark/cbr-rnn"
    "cclark/cbr-rnn-m"
)
#for repo in "${CBR_RNN_REPOS[@]}"; do
#    download "${repo}" main
#done


# ===========================================================================
# 12. BabyLM 2025 shared-task models
# ===========================================================================
echo
echo "### BabyLM 2025 models ###"
BABYLM_REPOS=(
    "colinglab/CLASS_IT-140M"
    "llm-slice/blm-gpt2s-90M-s42_901M-s42_submission"
    "Jtapsa/moep"
)
#for repo in "${BABYLM_REPOS[@]}"; do
#    download "${repo}" main
#done

# ===========================================================================
# 12b. BabyLM 2026 baseline models (BabyLM-community, main only)
#      GPT2-Strict{,-Small} / Strict{,-Small}-Interaction: plain GPT-2 configs.
#      gpt-bert 100m/10m variants: custom GPTBERT arch, trust_remote_code=True.
#      flamingo-old: custom Flamingo arch (OPT backbone + dino-vitb16 vision
#      tower loaded inside the modeling code), trust_remote_code=True.
# ===========================================================================
echo
echo "### BabyLM 2026 baseline models (BabyLM-community) ###"
BABYLM_2026_REPOS=(
    "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
    "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"
    "BabyLM-community/BabyLM-2026-Baseline-Strict-Interaction"
    "BabyLM-community/BabyLM-2026-Baseline-Strict-Small-Interaction"
    "BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus"
    "BabyLM-community/babylm-baseline-100m-gpt-bert-mixed"
    "BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus"
    "BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus"
    "BabyLM-community/babylm-baseline-10m-gpt-bert-mixed"
    "BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus"
    "BabyLM-community/babylm-multimodal-baseline-flamingo-old"
)
#for repo in "${BABYLM_2026_REPOS[@]}"; do
#    download "${repo}" main
#done

# ===========================================================================
# 13. BLOOM family (bigscience, main only) — ALiBi at real scale, same
#     mechanism as the Lanni-ni alibi_* checkpoints above but web-scale
#     trained. Standard AutoModelForCausalLM, no trust_remote_code needed.
# ===========================================================================
echo
echo "### BLOOM family (bigscience) ###"
BLOOM_REPOS=(
    "bigscience/bloom-560m"
    "bigscience/bloom-1b1"
    "bigscience/bloom-1b7"
    "bigscience/bloom-3b"
    "bigscience/bloom-7b1"
    # "bigscience/bloom"  # 176B — excluded per user request (too large)
)
for repo in "${BLOOM_REPOS[@]}"; do
    download "${repo}" main
done

# ===========================================================================
# 14. RWKV-4 family (RWKV, Pile-trained, main only) — fixed-size recurrent
#     state (bounded memory by construction, no attention), same idea as
#     CBR-RNN above but at real scale.
# ===========================================================================
echo
echo "### RWKV-4 family (RWKV, Pile) ###"
RWKV4_REPOS=(
    "RWKV/rwkv-4-169m-pile"
    "RWKV/rwkv-4-430m-pile"
    "RWKV/rwkv-4-1b5-pile"
    "RWKV/rwkv-4-3b-pile"
    "RWKV/rwkv-4-7b-pile"
    "RWKV/rwkv-4-14b-pile"
)
for repo in "${RWKV4_REPOS[@]}"; do
    download "${repo}" main
done

# ===========================================================================
# 15. Forgetting Transformer (FoX) family (zhixuan-lin, main only) —
#     content-sensitive forget gate (Lin et al. 2025), same mechanism as the
#     Lanni-ni forgetting_* checkpoints above but ~15x their scale ceiling.
#     "Pro"/"LLaMA" variants and matched vanilla-transformer baselines, all
#     760M @ longcrawl64-48B.
# ===========================================================================
echo
echo "### Forgetting Transformer (FoX) family (zhixuan-lin) ###"
FOX_REPOS=(
    "zhixuan-lin/fox-llama-760m-longcrawl64-48b"
    "zhixuan-lin/fox-pro-760m-longcrawl64-48b"
    "zhixuan-lin/transformer-llama-760m-longcrawl64-48b"
    "zhixuan-lin/transformer-pro-760m-longcrawl64-48b"
)
for repo in "${FOX_REPOS[@]}"; do
    download "${repo}" main
done

# ===========================================================================
# 16. ModuleFormer / MoLM family (IBM, main only) — stick-breaking attention
#     (sparse MoE), same mechanism as the Lanni-ni stickbreaking_* checkpoints
#     above but at real scale (4B/8B total params, MoE-sparse active params).
#     Custom ModuleFormer arch — likely needs trust_remote_code=True to load.
# ===========================================================================
echo
echo "### ModuleFormer / MoLM family (IBM) ###"
MOLM_REPOS=(
    "ibm/MoLM-350M-4B"
    "ibm/MoLM-700M-4B"
    "ibm/MoLM-700M-8B"
)
for repo in "${MOLM_REPOS[@]}"; do
    download "${repo}" main
done

if [[ "${PREFETCH_ATLAS:-1}" == "1" ]]; then
    echo
    echo "### Nilearn Harvard-Oxford atlas + LPP parcel cache ###"
    export NILEARN_DATA="${NILEARN_DATA:-$SCRATCH/.cache/nilearn}"
    export LBG_ROOT="${LBG_ROOT:-$REPO/data}"
    mkdir -p "$NILEARN_DATA"
    if command -v uv >/dev/null 2>&1; then _PY=(uv run --frozen --no-sync --no-dev python); else _PY=(python); fi
    "${_PY[@]}" - <<'PY' || _failed_repos+=("nilearn Harvard-Oxford atlas")
import os
from pathlib import Path
from nilearn.datasets import fetch_atlas_harvard_oxford
data_dir = Path(os.environ["NILEARN_DATA"])
atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm", symmetric_split=True, data_dir=str(data_dir))
print("atlas maps:", atlas["maps"])
mask = Path(os.environ["LBG_ROOT"]) / "masks" / "mask_lpp_en.nii.gz"
if mask.is_file():
    from neuroai.roi.lpp import load_harvard_oxford_parcelization
    ho = load_harvard_oxford_parcelization(mask)
    print(f"parcel cache warmed: {ho.n_regions} regions, {ho.left_masks.shape[1]} voxels")
else:
    print(f"(LPP brain mask not found at {mask}; atlas fetched, parcel cache not warmed)")
PY
fi

echo
echo "=== Prefetch complete ==="
echo "Models cached to: $HF_HUB_CACHE"
if (( ${#_failed_repos[@]} > 0 )); then
    echo
    echo "WARNING: the following downloads failed after all retries:"
    printf "  %s\n" "${_failed_repos[@]}"
    echo "Re-run this script or download them manually before submitting GPU jobs."
    exit 1
fi