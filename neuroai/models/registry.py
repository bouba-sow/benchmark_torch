"""LPP checkpoint-sweep model registry (aligned with ``scripts/prefetch_models_hf.sh``).

Families with training-step checkpoints (Pythia, OLMo-2-1B) expose every prefetched
revision; all other families evaluate ``main`` only.

Use :func:`flatten_family_tasks` for SLURM array sizing (models × checkpoints).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._checkpoints import (
    PYTHIA_CHECKPOINT_STEPS,
    olmo2_checkpoint_metrics,
    pythia_checkpoint_revision,
)

# Keep in sync with PYTHIA_STEPS in scripts/prefetch_models_hf.sh
LPP_PYTHIA_STEPS: list[int] = [
    0,
    128,
    256,
    512,
    2000,
    8000,
    16000,
    32000,
    64000,
    128000,
    143000,
]

# OLMo-2-0425-1B — keep in sync with OLMO2_1B_REVISIONS in prefetch_models_hf.sh
OLMO2_1B_REPO = "allenai/OLMo-2-0425-1B"
OLMO2_1B_EARLY_REPO = "allenai/OLMo-2-0425-1B-early-training"

# Early checkpoints live on the separate early-training repo; later ones on the main repo.
LPP_OLMO2_1B_EARLY_REVISIONS: tuple[str, ...] = (
    #"stage1-step0-tokens0B",
    #"stage1-step1000-tokens3B",
    #"stage1-step2000-tokens5B",
    #"stage1-step4000-tokens9B",
    #"stage1-step8000-tokens17B",
    #"stage1-step16000-tokens34B",
    #"stage1-step30000-tokens63B",
)

LPP_OLMO2_1B_REVISIONS: list[str] = [
    #*LPP_OLMO2_1B_EARLY_REVISIONS,
    #"stage1-step60000-tokens126B",
    #"stage1-step120000-tokens252B",
    #"stage1-step260000-tokens546B",
    #"stage1-step520000-tokens1091B",
    #"stage1-step1040000-tokens2182B",
    #"stage1-step1500000-tokens3146B",
    "stage1-step1907359-tokens4001B",
]

LPP_OLMO2_1B_REVISION_MODEL_IDS: dict[str, str] = {
    rev: OLMO2_1B_EARLY_REPO for rev in LPP_OLMO2_1B_EARLY_REVISIONS
}

LPP_OLMO2_7B_REVISIONS: list[str] = [
    #"stage1-step1000-tokens5B",
    #"stage1-step2000-tokens9B",
    #"stage1-step4000-tokens17B",
    #"stage1-step8000-tokens34B",
    #"stage1-step16000-tokens68B",
    #"stage1-step32000-tokens135B",
    #"stage1-step64000-tokens269B",
    #"stage1-step128000-tokens537B",
    #"stage1-step256000-tokens1074B",
    #"stage1-step528000-tokens2215B",
    #"stage1-step700000-tokens2937B",
    "stage1-step928646-tokens3896B",
]

LPP_OLMO2_13B_REVISIONS: list[str] = [
    #"stage1-step0-tokens0B",
    #"stage1-step1000-tokens9B",
    #"stage1-step2000-tokens17B",
    #"stage1-step4000-tokens34B",
    #"stage1-step8000-tokens68B",
    #"stage1-step16000-tokens135B",
    #"stage1-step32000-tokens269B",
    #"stage1-step64000-tokens537B",
    #"stage1-step128000-tokens1074B",
    #"stage1-step143000-tokens1200B",
    #"stage1-step256000-tokens2148B",
    #"stage1-step350000-tokens2937B",
    #"stage1-step420000-tokens3524B",
    #"stage1-step477000-tokens4002B",
    #"stage1-step520000-tokens4363B",
    "stage1-step596057-tokens5001B",
]

@dataclass(frozen=True)
class LppModelSpec:
    """One HuggingFace model in an LPP scaling family."""

    model_id: str
    label: str
    n_layers: int
    n_params: float
    batch_size: int
    checkpoint_kind: str  # "pythia" | "olmo2" | "main"
    steps: list[int] | None = None
    revisions: list[str] | None = None
    revision_model_ids: dict[str, str] | None = None
    trust_remote_code: bool = False


def _main_only(
    model_id: str,
    label: str,
    n_layers: int,
    n_params: float,
    batch_size: int,
    trust_remote_code: bool = False,
) -> LppModelSpec:
    return LppModelSpec(
        model_id=model_id,
        label=label,
        n_layers=n_layers,
        n_params=n_params,
        batch_size=batch_size,
        checkpoint_kind="main",
        trust_remote_code=trust_remote_code,
    )


# ---------------------------------------------------------------------------
# Families (only models whose prefetch download is not commented out)
# ---------------------------------------------------------------------------

LPP_FAMILIES: dict[str, list[LppModelSpec]] = {
    "gpt2": [
        _main_only("openai-community/gpt2", "GPT-2", 13, 124e6, 16),
        _main_only("openai-community/gpt2-medium", "GPT-2-medium", 25, 355e6, 12),
        _main_only("openai-community/gpt2-large", "GPT-2-large", 37, 774e6, 8),
        _main_only("openai-community/gpt2-xl", "GPT-2-xl", 49, 1.6e9, 4),
    ],
    "gptneo": [
        _main_only("EleutherAI/gpt-neo-125m", "GPT-Neo-125M", 13, 125e6, 16),
        _main_only("EleutherAI/gpt-neo-1.3B", "GPT-Neo-1.3B", 25, 1.3e9, 12),
        _main_only("EleutherAI/gpt-neo-2.7B", "GPT-Neo-2.7B", 33, 2.7e9, 8),
        _main_only("EleutherAI/gpt-j-6b", "GPT-J-6B", 29, 6.0e9, 4),
        _main_only("EleutherAI/gpt-neox-20b", "GPT-NeoX-20B", 45, 20.0e9, 2),
    ],
    "opt": [
        _main_only("facebook/opt-125m", "OPT-125M", 13, 125e6, 16),
        # 24, not 25 -- opt-350m is the one OPT size with word_embed_proj_dim
        # (512) != hidden_size (1024), so its embedding-layer hidden state
        # can't be stacked with the rest and gets dropped; verified against
        # the actual extracted shape (ZuCo reading embed sweep), not just
        # the usual num_hidden_layers+1 convention every other OPT size follows.
        _main_only("facebook/opt-350m", "OPT-350M", 24, 350e6, 12),
        _main_only("facebook/opt-1.3b", "OPT-1.3B", 25, 1.3e9, 12),
        _main_only("facebook/opt-2.7b", "OPT-2.7B", 33, 2.7e9, 8),
        _main_only("facebook/opt-6.7b", "OPT-6.7B", 33, 6.7e9, 4),
        _main_only("facebook/opt-13b", "OPT-13B", 41, 13.0e9, 2),
        _main_only("facebook/opt-30b", "OPT-30B", 49, 30.0e9, 1),
        _main_only("facebook/opt-66b", "OPT-66B", 65, 66.0e9, 1),
    ],
    "mamba": [
        _main_only("state-spaces/mamba-130m-hf", "Mamba-130M", 25, 130e6, 16),
        _main_only("state-spaces/mamba-370m-hf", "Mamba-370M", 49, 370e6, 12),
        _main_only("state-spaces/mamba-790m-hf", "Mamba-790M", 49, 790e6, 8),
        _main_only("state-spaces/mamba-1.4b-hf", "Mamba-1.4B", 49, 1.4e9, 8),
        _main_only("state-spaces/mamba-2.8b-hf", "Mamba-2.8B", 61, 2.8e9, 4),
    ],
    "mamba2": [
        _main_only("AntonV/mamba2-130m-hf", "Mamba2-130M", 25, 130e6, 16),
        _main_only("AntonV/mamba2-370m-hf", "Mamba2-370M", 49, 370e6, 12),
        _main_only("AntonV/mamba2-780m-hf", "Mamba2-780M", 49, 780e6, 8),
        _main_only("AntonV/mamba2-1.3b-hf", "Mamba2-1.3B", 49, 1.3e9, 8),
        _main_only("AntonV/mamba2-2.7b-hf", "Mamba2-2.7B", 61, 2.7e9, 4),
    ],
    # Training-step sweep (11 milestone steps: LPP_PYTHIA_STEPS) for every
    # Pythia size -- kept as a SEPARATE family from "pythia" below (rather
    # than flipping "pythia" itself to checkpoint_kind="pythia") so every
    # other pipeline reading LPP_FAMILIES["pythia"] keeps getting main-only
    # behavior unaffected. Used for perplexity-matched behavioral LME sweeps
    # (naturalstories_spr/geco/meco/provo/ucl_reading/zuco_reading) -- see
    # scripts/slurm/jz/surprisal/<dataset>/<dataset>_pythia_checkpoints_*.sh.
    # Matches the pre-existing "pythia_checkpoints" convention already used
    # by the embedding sweeps (EXCLUDED_FAMILY_DIRS in neuroai/analysis/benchmark_common.py).
    "pythia_checkpoints": [
        LppModelSpec("EleutherAI/pythia-14m-deduped", "Pythia-14M", 6, 14.0e6, 64, "pythia"),
        LppModelSpec("EleutherAI/pythia-31m-deduped", "Pythia-31M", 6, 31.0e6, 64, "pythia"),
        LppModelSpec("EleutherAI/pythia-70m-deduped", "Pythia-70M-deduped", 6, 70.0e6, 64, "pythia"),
        LppModelSpec("EleutherAI/pythia-160m-deduped", "Pythia-160M-deduped", 12, 160.0e6, 32, "pythia"),
        LppModelSpec("EleutherAI/pythia-410m-deduped", "Pythia-410M-deduped", 24, 410.0e6, 16, "pythia"),
        LppModelSpec("EleutherAI/pythia-1b-deduped", "Pythia-1B-deduped", 17, 1.0e9, 8, "pythia"),
        LppModelSpec("EleutherAI/pythia-1.4b-deduped", "Pythia-1.4B-deduped", 25, 1.4e9, 6, "pythia"),
        LppModelSpec("EleutherAI/pythia-2.8b-deduped", "Pythia-2.8B-deduped", 33, 2.8e9, 4, "pythia"),
        LppModelSpec("EleutherAI/pythia-6.9b-deduped", "Pythia-6.9B-deduped", 33, 6.9e9, 2, "pythia"),
        LppModelSpec("EleutherAI/pythia-12b-deduped", "Pythia-12B-deduped", 37, 12.0e9, 2, "pythia"),
    ],
    # Main checkpoint only (final, fully-trained model — the HF "main" branch
    # for every EleutherAI Pythia-deduped repo, no stepN revision needed).
    "pythia": [
        _main_only("EleutherAI/pythia-14m-deduped", "Pythia-14M", 6, 14.0e6, 64),
        _main_only("EleutherAI/pythia-31m-deduped", "Pythia-31M", 6, 31.0e6, 64),
        _main_only("EleutherAI/pythia-70m-deduped", "Pythia-70M-deduped", 6, 70.0e6, 64),
        _main_only("EleutherAI/pythia-160m-deduped", "Pythia-160M-deduped", 12, 160.0e6, 32),
        _main_only("EleutherAI/pythia-410m-deduped", "Pythia-410M-deduped", 24, 410.0e6, 16),
        _main_only("EleutherAI/pythia-1b-deduped", "Pythia-1B-deduped", 17, 1.0e9, 8),
        _main_only("EleutherAI/pythia-1.4b-deduped", "Pythia-1.4B-deduped", 25, 1.4e9, 6),
        _main_only("EleutherAI/pythia-2.8b-deduped", "Pythia-2.8B-deduped", 33, 2.8e9, 4),
        _main_only("EleutherAI/pythia-6.9b-deduped", "Pythia-6.9B-deduped", 33, 6.9e9, 2),
        _main_only("EleutherAI/pythia-12b-deduped", "Pythia-12B-deduped", 37, 12.0e9, 2),
    ],
    "olmo2": [
        LppModelSpec(
            OLMO2_1B_REPO,
            "OLMo2-1B",
            17,
            1.0e9,
            8,
            "olmo2",
            revisions=list(LPP_OLMO2_1B_REVISIONS),
            revision_model_ids=dict(LPP_OLMO2_1B_REVISION_MODEL_IDS),
        ),
        LppModelSpec(
            "allenai/OLMo-2-1124-7B",
            "OLMo2-7B",
            32,
            7.0e9,
            4,
            "olmo2",
            revisions=list(LPP_OLMO2_7B_REVISIONS),
        ),
        LppModelSpec(
            "allenai/OLMo-2-1124-13B",
            "OLMo2-13B",
            40,
            13.0e9,
            2,
            "olmo2",
            revisions=list(LPP_OLMO2_13B_REVISIONS),
        ),
    ],
    "qwen3": [
        _main_only("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B", 29, 0.6e9, 16),
        _main_only("Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B", 29, 1.7e9, 12),
        _main_only("Qwen/Qwen3-4B-Base", "Qwen3-4B", 37, 4.0e9, 8),
        _main_only("Qwen/Qwen3-8B-Base", "Qwen3-8B", 37, 8.0e9, 4),
        _main_only("Qwen/Qwen3-14B-Base", "Qwen3-14B", 41, 14.0e9, 2),
    ],
    "llama32": [
        _main_only("meta-llama/Llama-3.2-1B", "Llama-3.2-1B", 17, 1.0e9, 12),
        _main_only("meta-llama/Llama-3.2-3B", "Llama-3.2-3B", 29, 3.0e9, 8),
        _main_only("meta-llama/Llama-3.1-8B", "Llama-3.1-8B", 33, 8.0e9, 4),
    ],
    "mistral": [
        _main_only(
            "mistralai/Ministral-3-3B-Base-2512",
            "Ministral-3-3B",
            27,   # not 29 -- actual hidden_states count is 2 short of the
                  # usual num_hidden_layers+1 convention; verified against
                  # the real extracted embedding shape (ZuCo reading embed
                  # sweep), not just the HF config's declared layer count.
            3.0e9,
            8,
        ),
        _main_only(
            "mistralai/Ministral-3-8B-Base-2512",
            "Ministral-3-8B",
            35,   # not 37 -- same discrepancy as Ministral-3-3B above.
            8.0e9,
            4,
        ),
        _main_only(
            "mistralai/Ministral-3-14B-Base-2512",
            "Ministral-3-14B",
            41,
            14.0e9,
            2,
        ),
    ],
    # BLOOM (bigscience): ALiBi at real scale — same positional mechanism as
    # the Lanni-ni alibi_* family below, but web-scale trained (main only,
    # standard AutoModelForCausalLM, no trust_remote_code). n_layers verified
    # via an actual output_hidden_states forward pass on bloom-560m (Jean Zay
    # login node, 2026-09-12): 25 hidden states = num_hidden_layers(24) + 1,
    # the standard convention — applied to the rest of the family from their
    # config.json n_layer. 176B ("bigscience/bloom") excluded per user
    # request (too large to be worth the compute).
    "bloom": [
        _main_only("bigscience/bloom-560m", "BLOOM-560M", 25, 560e6, 16),
        _main_only("bigscience/bloom-1b1", "BLOOM-1.1B", 25, 1.1e9, 12),
        _main_only("bigscience/bloom-1b7", "BLOOM-1.7B", 25, 1.7e9, 8),
        _main_only("bigscience/bloom-3b", "BLOOM-3B", 31, 3.0e9, 8),
        _main_only("bigscience/bloom-7b1", "BLOOM-7.1B", 31, 7.1e9, 4),
    ],
    # RWKV-4 (RWKV, Pile-trained): fixed-size recurrent state (bounded memory
    # by construction, no attention) — same idea as CBR-RNN below but at real
    # scale. Main only, natively supported by transformers (RwkvForCausalLM,
    # no trust_remote_code). n_layers verified via an actual
    # output_hidden_states forward pass on rwkv-4-169m-pile (Jean Zay login
    # node, 2026-09-12): 13 hidden states = num_hidden_layers(12) + 1 — applied
    # to the rest of the family from their config.json num_hidden_layers.
    "rwkv4": [
        _main_only("RWKV/rwkv-4-169m-pile", "RWKV4-169M", 13, 169e6, 32),
        _main_only("RWKV/rwkv-4-430m-pile", "RWKV4-430M", 25, 430e6, 16),
        _main_only("RWKV/rwkv-4-1b5-pile", "RWKV4-1.5B", 25, 1.5e9, 8),
        _main_only("RWKV/rwkv-4-3b-pile", "RWKV4-3B", 33, 3.0e9, 8),
        _main_only("RWKV/rwkv-4-7b-pile", "RWKV4-7B", 33, 7.0e9, 4),
        _main_only("RWKV/rwkv-4-14b-pile", "RWKV4-14B", 41, 14.0e9, 2),
    ],
    # Forgetting Transformer (FoX) / matched vanilla-Transformer baselines
    # (zhixuan-lin, Lin et al. 2025): same ForgettingTransformerForCausalLM /
    # TransformerForCausalLM classes (and the same vendored
    # forgetting-transformer/src loader) as the Lanni-ni forgetting_* family
    # below, at ~15x that family's scale ceiling. trust_remote_code=True;
    # these use Triton/flash-linear-attention kernels and only load on GPU
    # (confirmed on Jean Zay login node, 2026-09-12 — CPU load raises "0
    # active drivers" from triton).
    #
    # n_layers verified on an H100 (Jean Zay, 2026-09-12,
    # scripts/maintenance/verify_fox_layers.py): ALL FOUR give 25 hidden
    # states = num_hidden_layers(24) + 1 — the *standard* convention, NOT the
    # raw-count convention used for the Lanni-ni forgetting_*/
    # stickbreaking_*/alibi_* families below (that earlier assumption was
    # wrong; do not copy it here).
    #
    # transformer-llama-760m and transformer-pro-760m (use_rope=True in their
    # config — fox-llama/fox-pro have use_rope=False, since Forgetting
    # Attention has its own decay mechanism and never hits this code) both
    # initially crashed with two real bugs in the vendored
    # forgetting-transformer/src, now patched there (not upstream, not fla):
    #  1. modeling_transformer.py / modeling_forgetting_transformer.py called
    #     `self.rotary(q, k, seqlen_offset, max_seqlen)` positionally, but the
    #     pinned fla commit's RotaryEmbedding.forward signature is
    #     (q, k, seqlen_offset=0, cu_seqlens=None, max_seqlen=None) — the 3rd
    #     positional slot is cu_seqlens, not max_seqlen, so max_seqlen (an
    #     int) landed in cu_seqlens and crashed on `cu_seqlens.shape`. Fixed
    #     by passing max_seqlen as a keyword at both call sites.
    #  2. modeling_transformer.py's vanilla TransformerModel.forward still
    #     called the removed `DynamicCache.from_legacy_cache`/
    #     `.to_legacy_cache` API (transformers 5.13.1 dropped legacy tuple
    #     caches) — modeling_forgetting_transformer.py already had this same
    #     fix applied (commented out); mirrored it here.
    # Both are now GPU-verified end to end (config -> model -> forward pass).
    "fox": [
        _main_only("zhixuan-lin/fox-llama-760m-longcrawl64-48b", "FoX-Llama-760M", 25, 760e6, 8, trust_remote_code=True),
        _main_only("zhixuan-lin/fox-pro-760m-longcrawl64-48b", "FoX-Pro-760M", 25, 760e6, 8, trust_remote_code=True),
        _main_only("zhixuan-lin/transformer-llama-760m-longcrawl64-48b", "Transformer-Llama-760M", 25, 760e6, 8, trust_remote_code=True),
        _main_only("zhixuan-lin/transformer-pro-760m-longcrawl64-48b", "Transformer-Pro-760M", 25, 760e6, 8, trust_remote_code=True),
    ],
    # ModuleFormer / MoLM (IBM): stick-breaking attention with a sparse MoE
    # FFN — same mechanism as the Lanni-ni stickbreaking_* family below, at
    # real scale. IBM never published this as a standard HF trust_remote_code
    # repo (no auto_map in config.json, "moduleformer" isn't a registered
    # AutoConfig model_type on transformers 5.13.1 — confirmed on the Jean Zay
    # login node, 2026-09-12, "Transformers does not recognize this
    # architecture"), so it's loaded via a direct-class loader
    # (neuroai/features/llm.py::_load_moduleformer_direct, same treatment as
    # _load_forgetting_direct) against a vendored copy of
    # github.com/IBM/ModuleFormer at moduleformer/moduleformer/ (config's
    # ModuleFormerOnnxConfig dropped — depends on the since-removed
    # transformers.onnx submodule and isn't needed for feature extraction;
    # modeling code's self.get_head_mask(...) call patched inline —
    # ModuleUtilsMixin.get_head_mask no longer exists in transformers 5.13.1).
    #
    # n_layers verified on an H100 (Jean Zay, 2026-09-12, GPU job 2060588,
    # scripts/maintenance/verify_molm_layers.py) via an actual output_hidden_states
    # forward pass on REAL loaded weights (all three): 25/25/49 hidden states
    # = num_hidden_layers(24/24/48) + 1 — the standard convention, same as
    # bloom/rwkv4/fox above. MoLM-700M-8B ships as a sharded
    # pytorch_model-*.bin checkpoint (no single-file model.safetensors/
    # pytorch_model.bin) — _load_moduleformer_direct merges shards via the
    # index's weight_map. n_params below is the ACTIVE-parameter count (the
    # model-name prefix, e.g. 350M of the 4B total in "MoLM-350M-4B") since
    # that's the compute-comparable figure for a sparse MoE — verified total
    # params (incl. inactive experts) were 4,142,526,464 / 4,142,526,464 /
    # 8,182,046,720; swap n_params to those if the scaling analysis should
    # key on stored size instead.
    "molm": [
        _main_only("ibm/MoLM-350M-4B", "MoLM-350M-4B", 25, 350e6, 8, trust_remote_code=True),
        _main_only("ibm/MoLM-700M-4B", "MoLM-700M-4B", 25, 700e6, 8, trust_remote_code=True),
        _main_only("ibm/MoLM-700M-8B", "MoLM-700M-8B", 49, 700e6, 4, trust_remote_code=True),
    ],
    # CBR-RNN: custom architecture (PyTorchModelHubMixin, not AutoModelForCausalLM).
    # Surprisal is computed via neuroai.features.cbr_rnn.batch_surprisal_cbr_rnn.
    # n_params measured exactly by a one-off counting script on Jean
    # Zay (2026-07-13, since removed): cbr-rnn and cbr-rnn-m are both 17,360,688 params exactly
    # (same architecture/size, presumably differing only in training data/seed).
    "cbr_rnn": [
        _main_only("cclark/cbr-rnn", "CBR-RNN", 1, 17_360_688, 64),
        _main_only("cclark/cbr-rnn-m", "CBR-RNN-M", 1, 17_360_688, 64),
    ],
    # Memory-constrained Lanni-ni models (Fox et al.).
    # AutoModelForCausalLM with trust_remote_code=True — no custom loader needed.
    # Param counts measured exactly by a one-off counting script (since removed) on
    # Jean Zay (2026-07-09).
    "stickbreaking": [
        _main_only("Lanni-ni/stickbreaking_pile_2layer",        "StickBreaking-Pile-2L",        2, 27_840_256, 64, trust_remote_code=True),
        _main_only("Lanni-ni/stickbreaking_pile_4layer",        "StickBreaking-Pile-4L",        4, 48_053_376, 64, trust_remote_code=True),
        _main_only("Lanni-ni/stickbreaking_babylm_10m_2layer",  "StickBreaking-BabyLM10M-2L",  2, 27_840_256, 64, trust_remote_code=True),
        _main_only("Lanni-ni/stickbreaking_babylm_10m_4layer",  "StickBreaking-BabyLM10M-4L",  4, 48_053_376, 64, trust_remote_code=True),
        _main_only("Lanni-ni/stickbreaking_babylm_100m_2layer", "StickBreaking-BabyLM100M-2L", 2, 27_840_256, 64, trust_remote_code=True),
        _main_only("Lanni-ni/stickbreaking_babylm_100m_4layer", "StickBreaking-BabyLM100M-4L", 4, 48_053_376, 64, trust_remote_code=True),
    ],
    "alibi": [
        _main_only("Lanni-ni/alibi_pile_2layer",        "ALiBi-Pile-2L",        2, 27_447_040, 64, trust_remote_code=True),
        _main_only("Lanni-ni/alibi_pile_4layer",        "ALiBi-Pile-4L",        4, 45_694_080, 64, trust_remote_code=True),
        _main_only("Lanni-ni/alibi_babylm_10m_2layer",  "ALiBi-BabyLM10M-2L",  2, 27_447_040, 64, trust_remote_code=True),
        _main_only("Lanni-ni/alibi_babylm_10m_4layer",  "ALiBi-BabyLM10M-4L",  4, 45_694_080, 64, trust_remote_code=True),
        _main_only("Lanni-ni/alibi_babylm_100m_2layer", "ALiBi-BabyLM100M-2L", 2, 27_447_040, 64, trust_remote_code=True),
        _main_only("Lanni-ni/alibi_babylm_100m_4layer", "ALiBi-BabyLM100M-4L", 4, 45_694_080, 64, trust_remote_code=True),
    ],
    # forgetting_pile_{2,4}layer fail to load with trust_remote_code (real
    # mlp.gate_proj/up_proj shape mismatch vs. the local forgetting_transformer
    # modeling code — see the 2026-07 counting logs). Counts below are still
    # the pre-existing ALiBi-based approximation, unconfirmed until that load
    # bug is fixed and the models can actually be measured.
    "forgetting": [
        _main_only("Lanni-ni/forgetting_pile_2layer",        "Forgetting-Pile-2L",        2, 27.4e6, 64, trust_remote_code=True),
        _main_only("Lanni-ni/forgetting_pile_4layer",        "Forgetting-Pile-4L",        4, 45.7e6, 64, trust_remote_code=True),
        _main_only("Lanni-ni/forgetting_babylm_10m_2layer",  "Forgetting-BabyLM10M-2L",  2, 27.4e6, 64, trust_remote_code=True),
        _main_only("Lanni-ni/forgetting_babylm_10m_4layer",  "Forgetting-BabyLM10M-4L",  4, 45.7e6, 64, trust_remote_code=True),
        _main_only("Lanni-ni/forgetting_babylm_100m_2layer", "Forgetting-BabyLM100M-2L", 2, 27.4e6, 64, trust_remote_code=True),
        _main_only("Lanni-ni/forgetting_babylm_100m_4layer", "Forgetting-BabyLM100M-4L", 4, 45.7e6, 64, trust_remote_code=True),
    ],
    # Hard n-gram (fixed local sliding-window) attention baselines. Measured
    # by a one-off counting script (since removed) on Jean Zay (2026-07-13), after
    # fixing the tie_word_embeddings bug in the downloaded trust_remote_code
    # (configuration_sliding_window.py never forwarded tie_word_embeddings to
    # super().__init__(), so it was silently forced True regardless of the
    # checkpoint's real config, and — separately — the checkpoints have no
    # real saved lm_head.weight at all, so the correct fix is tying it to the
    # input embedding; see neuroai/features/llm.py's _get_cached_causal_lm for
    # the runtime repair). True parameter counts with lm_head correctly tied:
    # 14,970,624 @ 2-layer/256-hidden, 28,750,464 @ 4-layer/384-hidden — lower
    # than stickbreaking/ALiBi/forgetting since this family has no separate
    # lm_head at all (fully tied) and no learned decay/bias params.
    "hard_2gram": [
        _main_only("Lanni-ni/hard_2gram_pile_2layer",        "Hard2gram-Pile-2L",        2, 14_970_624, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_2gram_pile_4layer",        "Hard2gram-Pile-4L",        4, 28_750_464, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_2gram_babylm_10m_2layer",  "Hard2gram-BabyLM10M-2L",  2, 14_970_624, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_2gram_babylm_10m_4layer",  "Hard2gram-BabyLM10M-4L",  4, 28_750_464, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_2gram_babylm_100m_2layer", "Hard2gram-BabyLM100M-2L", 2, 14_970_624, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_2gram_babylm_100m_4layer", "Hard2gram-BabyLM100M-4L", 4, 28_750_464, 64, trust_remote_code=True),
    ],
    "hard_3gram": [
        _main_only("Lanni-ni/hard_3gram_pile_2layer",        "Hard3gram-Pile-2L",        2, 14_970_624, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_3gram_pile_4layer",        "Hard3gram-Pile-4L",        4, 28_750_464, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_3gram_babylm_10m_2layer",  "Hard3gram-BabyLM10M-2L",  2, 14_970_624, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_3gram_babylm_10m_4layer",  "Hard3gram-BabyLM10M-4L",  4, 28_750_464, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_3gram_babylm_100m_2layer", "Hard3gram-BabyLM100M-2L", 2, 14_970_624, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_3gram_babylm_100m_4layer", "Hard3gram-BabyLM100M-4L", 4, 28_750_464, 64, trust_remote_code=True),
    ],
    "hard_5gram": [
        _main_only("Lanni-ni/hard_5gram_pile_2layer",        "Hard5gram-Pile-2L",        2, 14_970_624, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_5gram_pile_4layer",        "Hard5gram-Pile-4L",        4, 28_750_464, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_5gram_babylm_10m_2layer",  "Hard5gram-BabyLM10M-2L",  2, 14_970_624, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_5gram_babylm_10m_4layer",  "Hard5gram-BabyLM10M-4L",  4, 28_750_464, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_5gram_babylm_100m_2layer", "Hard5gram-BabyLM100M-2L", 2, 14_970_624, 64, trust_remote_code=True),
        _main_only("Lanni-ni/hard_5gram_babylm_100m_4layer", "Hard5gram-BabyLM100M-4L", 4, 28_750_464, 64, trust_remote_code=True),
    ],
    # ---------------------------------------------------------------------------
    # BabyLM 2025 shared-task models
    # ---------------------------------------------------------------------------
    # CLASS-IT: Llama-style causal LM trained on BabyLM 100M corpus (~140 M params)
    # blm-gpt2s: GPT-2 small variant, custom vocab (16k), trained on BabyLM (~90 M params)
    # moep: Mixture-of-Experts Parallel LM, BabyLM 100M, trust_remote_code=True (~80 M params)
    #
    # BabyLM 2026 baselines (BabyLM-community org). Param counts measured exactly
    # by instantiating each config via AutoModelForCausalLM.from_config (no
    # weight download needed) on 2026-07-27:
    #   GPT2-Strict{,-Small} / Strict{,-Small}-Interaction: identical GPT-2 config
    #     (12 layers, 768 hidden, 16k vocab) -> 98,425,344 params.
    #   gpt-bert 100m/10m variants (causal-focus/mixed/masked-focus): custom
    #     GPTBERT arch, trust_remote_code=True; 100m configs -> 119,454,532
    #     params, 10m configs -> 33,046,084 params (both 12 layers).
    #   flamingo-old: custom Flamingo arch (OPT text backbone + gated
    #     cross-attention + facebook/dino-vitb16 vision tower, auto-loaded from
    #     its own pretrained weights inside the modeling code), trust_remote_code=True
    #     -> 255,487,500 params total (12 text-decoder layers).
    # BabyLM 2025 shared-task GPT-2 baseline, WITH training checkpoints (unlike
    # the 2026 GPT2-Strict baselines above, which only publish a main branch).
    # 29 revisions on HF ("main" + chck_1M..chck_9M, chck_10M..chck_90M,
    # chck_100M..chck_1000M -- named by millions of training tokens seen; NOT
    # necessarily evenly spaced, and "main" is a DIFFERENT commit from
    # chck_1000M -- likely an early-stopped/best-eval checkpoint, not simply
    # the final training step). Same architecture as GPT2-Strict-2026 (12
    # transformer layers, 768 hidden, 16k vocab) -> 98,425,344 params
    # (verified 2026-08-26 via AutoModelForCausalLM.from_config). Reuses the
    # "olmo2" checkpoint_kind purely for its generic explicit-revision-list
    # mechanism (checkpoints={"type": "revisions", "revisions": [...]}) --
    # this is NOT an OLMo2 model; olmo2_checkpoint_metrics() harmlessly
    # returns None for these "chck_NM"-style names (falls back to step=0,
    # tokens_seen=0 -- parse "chck_NM" separately if you need the tokens-seen
    # axis). Tokenizer files (tokenizer.json/merges.txt) are present on every
    # checkpoint branch (verified for chck_1M), so no "tokenizer always on
    # main" override is needed here (contrast Pythia's tok_rev="main" special
    # case). Prefetched to $SCRATCH/hf_cache 2026-08-26 via
    # `.venv/bin/hf download BabyLM-community/babylm-baseline-100m-gpt2
    # --revision <rev>` for each revision (works directly from a Jean Zay
    # login/compute shell -- no prepost-partition srun needed, egress to
    # huggingface.co is not blocked here).
    "babylm_gpt2_checkpoints": [
        LppModelSpec(
            "BabyLM-community/babylm-baseline-100m-gpt2", "BabyLM2025-GPT2-100M", 13, 98_425_344, 16,
            "olmo2",
            revisions=[
                "main",
                "chck_1M", "chck_2M", "chck_3M", "chck_4M", "chck_5M",
                "chck_6M", "chck_7M", "chck_8M", "chck_9M",
                "chck_10M", "chck_20M", "chck_30M", "chck_40M", "chck_50M",
                "chck_60M", "chck_70M", "chck_80M", "chck_90M",
                "chck_100M", "chck_200M", "chck_300M", "chck_400M", "chck_500M",
                "chck_600M", "chck_700M", "chck_800M", "chck_900M", "chck_1000M",
            ],
        ),
    ],
    "babylm": [
        _main_only("colinglab/CLASS_IT-140M",                                    "CLASS-IT-140M",    13,  140e6, 16),
        _main_only("llm-slice/blm-gpt2s-90M-s42_901M-s42_submission",           "BLM-GPT2s-90M",    13,   90e6, 16),
        # 11, not 13 -- actual hidden_states count is 2 short of the usual
        # convention (likely an MoE-gating quirk in this custom modeling
        # code); verified against the real extracted embedding shape (ZuCo
        # reading embed sweep).
        _main_only("Jtapsa/moep",                                                "MoEP-80M",         11,   80e6, 16, trust_remote_code=True),
        _main_only("BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict",               "GPT2-Strict-2026",            13, 98_425_344, 16),
        _main_only("BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small",         "GPT2-StrictSmall-2026",       13, 98_425_344, 16),
        # Excluded: checkpoints are broken, not a pipeline bug. Verified by
        # direct inspection (2026-08-11) — identical GPT2 architecture/tokenizer
        # to the working GPT2-Strict siblings, no NaN/Inf in any weight, and the
        # same catastrophic loss (~20 nats/token on real corpus text, reproduced
        # identically in fp32/fp16/bf16 so it's not a precision artifact). The
        # models are wildly overconfident on the *wrong* continuation (e.g.
        # logit ~80 on an unrelated token while the true next word gets near-zero
        # probability), even though greedy generation still looks locally
        # plausible. This drives corpus-level perplexity to 1e8-1e25, which is
        # not a meaningful/comparable number. Likely an upstream issue with
        # these BabyLM "Interaction"-track submissions specifically.
        # _main_only("BabyLM-community/BabyLM-2026-Baseline-Strict-Interaction",        "Strict-Interaction-2026",     13, 98_425_344, 16),
        # _main_only("BabyLM-community/BabyLM-2026-Baseline-Strict-Small-Interaction",  "StrictSmall-Interaction-2026",13, 98_425_344, 16),
        _main_only("BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus",     "GPTBERT-100M-CausalFocus",    13, 119_454_532, 16, trust_remote_code=True),
        _main_only("BabyLM-community/babylm-baseline-100m-gpt-bert-mixed",            "GPTBERT-100M-Mixed",          13, 119_454_532, 16, trust_remote_code=True),
        _main_only("BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus",     "GPTBERT-100M-MaskedFocus",    13, 119_454_532, 16, trust_remote_code=True),
        _main_only("BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus",      "GPTBERT-10M-CausalFocus",     13, 33_046_084, 16, trust_remote_code=True),
        _main_only("BabyLM-community/babylm-baseline-10m-gpt-bert-mixed",             "GPTBERT-10M-Mixed",           13, 33_046_084, 16, trust_remote_code=True),
        _main_only("BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus",      "GPTBERT-10M-MaskedFocus",     13, 33_046_084, 16, trust_remote_code=True),
        # Excluded: from_pretrained crashes on transformers 5.13.1 (the custom
        # modeling code calls ViTModel.from_pretrained("facebook/dino-vitb16")
        # internally while the outer load is on a meta-device context, which
        # this transformers version now forbids). Param count (255,487,500,
        # from_config + real dino-vitb16 vision weights) was still verified.
        # _main_only("BabyLM-community/babylm-multimodal-baseline-flamingo-old", "Flamingo-Multimodal-255M", 13, 255_487_500, 8, trust_remote_code=True),
    ],
}

LPP_FAMILY_NAMES: tuple[str, ...] = tuple(LPP_FAMILIES.keys())


def family_array_size(family: str, *, untrained: bool = False) -> int:
    """Number of SLURM array tasks = sum over models of checkpoint count.

    ``untrained=True`` collapses each model to a single task (random init
    doesn't depend on training-step revision, so sweeping checkpoints would
    just recompute the same random weights under different revision labels).
    """
    return len(flatten_family_tasks(family, untrained=untrained))


def model_spec_to_cfg(spec: LppModelSpec, registry_family: str) -> dict[str, Any]:
    """Convert a registry entry to a ``run_lpp_checkpoint_sweep`` model dict."""
    cfg: dict[str, Any] = {
        "model_id": spec.model_id,
        "label": spec.label,
        "n_layers": spec.n_layers,
        "n_params": spec.n_params,
        "batch_size": spec.batch_size,
        "family": spec.checkpoint_kind,
        "registry_family": registry_family,
        "trust_remote_code": spec.trust_remote_code,
    }
    if spec.checkpoint_kind == "pythia":
        cfg["checkpoints"] = {
            "type": "steps",
            "steps": spec.steps or list(LPP_PYTHIA_STEPS),
        }
    elif spec.checkpoint_kind == "olmo2":
        cfg["checkpoints"] = {
            "type": "revisions",
            "revisions": spec.revisions or [],
        }
        if spec.revision_model_ids:
            cfg["revision_model_ids"] = dict(spec.revision_model_ids)
    else:
        cfg["checkpoints"] = {"type": "main"}
    return cfg


def resolve_load_model_id(model_cfg: dict[str, Any], revision: str) -> str:
    """HF repo to load for ``revision`` (may differ from canonical ``model_id``)."""
    overrides = model_cfg.get("revision_model_ids") or {}
    return overrides.get(revision, model_cfg["model_id"])


def registry_model_configs(
    family: str | None = None,
) -> list[dict[str, Any]]:
    """All model dicts, optionally filtered to one registry family."""
    if family is not None:
        if family not in LPP_FAMILIES:
            raise KeyError(f"Unknown LPP family {family!r}; choose from {LPP_FAMILY_NAMES}")
        specs = LPP_FAMILIES[family]
        return [model_spec_to_cfg(s, family) for s in specs]
    out: list[dict[str, Any]] = []
    for fam, specs in LPP_FAMILIES.items():
        out.extend(model_spec_to_cfg(s, fam) for s in specs)
    return out


@dataclass(frozen=True)
class FamilyTask:
    """One SLURM array element: one model at one checkpoint."""

    registry_family: str
    model_id: str
    label: str
    revision: str
    step: int
    tokens_seen: int
    array_index: int
    load_model_id: str

    @property
    def model_dir_safe(self) -> str:
        return self.model_id.replace("/", "_")


def build_checkpoint_items(model_cfg: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Return (revision, step, tokens_seen) tuples for a model config dict."""
    family = model_cfg.get("family", "main")
    ckpt_cfg = model_cfg.get("checkpoints", {})
    ckpt_type = ckpt_cfg.get("type", "steps")

    if family == "olmo2" or ckpt_type == "revisions":
        out: list[tuple[str, int, int]] = []
        for rev in ckpt_cfg.get("revisions") or []:
            parsed = olmo2_checkpoint_metrics(rev)
            if parsed is None:
                out.append((rev, 0, 0))
            else:
                step, tokens = parsed
                out.append((rev, step, tokens))
        return out

    if ckpt_type == "main" or family == "main":
        return [("main", 0, 0)]

    steps = ckpt_cfg.get("steps", PYTHIA_CHECKPOINT_STEPS)
    tok_per_step = 2_097_152
    return [
        (pythia_checkpoint_revision(s), s, s * tok_per_step) for s in steps
    ]


def flatten_family_tasks(family: str, *, untrained: bool = False) -> list[FamilyTask]:
    """Flat task list for SLURM array index → (model, checkpoint).

    ``untrained=True`` keeps only the first checkpoint per model — see
    :func:`family_array_size`.
    """
    if family not in LPP_FAMILIES:
        raise KeyError(f"Unknown LPP family {family!r}; choose from {LPP_FAMILY_NAMES}")
    tasks: list[FamilyTask] = []
    idx = 0
    for spec in LPP_FAMILIES[family]:
        cfg = model_spec_to_cfg(spec, family)
        items = build_checkpoint_items(cfg)
        if untrained:
            items = items[:1]
        for revision, step, tokens in items:
            tasks.append(
                FamilyTask(
                    registry_family=family,
                    model_id=spec.model_id,
                    label=spec.label,
                    revision=revision,
                    step=step,
                    tokens_seen=tokens,
                    array_index=idx,
                    load_model_id=resolve_load_model_id(cfg, revision),
                )
            )
            idx += 1
    return tasks


def resolve_family_task(family: str, array_index: int, *, untrained: bool = False) -> FamilyTask:
    tasks = flatten_family_tasks(family, untrained=untrained)
    if array_index < 0 or array_index >= len(tasks):
        raise IndexError(
            f"array_index={array_index} out of range for family {family!r} "
            f"(size={len(tasks)})"
        )
    return tasks[array_index]
