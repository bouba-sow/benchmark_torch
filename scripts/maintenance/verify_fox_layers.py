#!/usr/bin/env python3
"""One-off: verify n_layers (len(output_hidden_states)) for the zhixuan-lin
FoX / matched-Transformer checkpoints added to neuroai/models/registry.py's
"fox" family (2026-09-12).

These use Triton/flash-linear-attention kernels (forgetting-transformer/src)
and only load on a CUDA device -- confirmed failing with "0 active drivers"
when attempted on the Jean Zay login node (CPU only). Run on a GPU node:

    sbatch (removed one-off script)

Only instantiates from config (random-initialized) -- no need to load real
weights, since output_hidden_states length depends only on the architecture,
not the weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "forgetting-transformer" / "src"))

import torch
from huggingface_hub import hf_hub_download

REPOS = [
    "zhixuan-lin/fox-llama-760m-longcrawl64-48b",
    "zhixuan-lin/fox-pro-760m-longcrawl64-48b",
    "zhixuan-lin/transformer-llama-760m-longcrawl64-48b",
    "zhixuan-lin/transformer-pro-760m-longcrawl64-48b",
]

_IGNORED_CONFIG_KEYS = {"architectures", "auto_map", "transformers_version", "model_type", "torch_dtype"}


def _load_config_and_model(repo: str):
    cfg_path = hf_hub_download(repo, "config.json")
    raw = json.load(open(cfg_path))
    arch = raw.get("architectures", [None])[0]
    cfg_kwargs = {k: v for k, v in raw.items() if k not in _IGNORED_CONFIG_KEYS}

    if arch == "TransformerForCausalLM":
        from forgetting_transformer.model.transformer.configuration_transformer import (
            TransformerConfig,
        )
        from forgetting_transformer.model.transformer.modeling_transformer import (
            TransformerForCausalLM,
        )

        config = TransformerConfig(**cfg_kwargs)
        model = TransformerForCausalLM(config)
    else:
        from forgetting_transformer.model.forgetting_transformer.configuration_forgetting_transformer import (
            ForgettingTransformerConfig,
        )
        from forgetting_transformer.model.forgetting_transformer.modeling_forgetting_transformer import (
            ForgettingTransformerForCausalLM,
        )

        config = ForgettingTransformerConfig(**cfg_kwargs)
        model = ForgettingTransformerForCausalLM(config)

    return raw, arch, config, model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    for repo in REPOS:
        try:
            raw, arch, config, model = _load_config_and_model(repo)
            model = model.to(device).eval()
            ids = torch.randint(0, 1000, (1, 8), device=device)
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True)
            hs = out.hidden_states if hasattr(out, "hidden_states") else out[-1]
            print(
                f"{repo}: architecture={arch} num_hidden_layers(cfg)={raw.get('num_hidden_layers')} "
                f"n_hidden_states={len(hs)}"
            )
            del model
        except Exception as exc:  # noqa: BLE001 -- keep checking the rest of REPOS
            print(f"{repo}: FAILED ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
