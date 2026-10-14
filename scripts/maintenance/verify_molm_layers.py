#!/usr/bin/env python3
"""One-off: verify n_layers (len(output_hidden_states)) and real-weight
loading for the ibm/MoLM-* checkpoints added to neuroai/models/registry.py's
"molm" family (2026-09-12), via the new
neuroai.features.llm._load_moduleformer_direct loader.

MoLM-350M-4B / MoLM-700M-4B / MoLM-700M-8B have up to 8B *total* (sparse MoE)
params -- random-initializing the full model in fp32 was OOM-killed on the
Jean Zay login node (~16GB for the smallest one alone), so this runs on a
GPU node with real weights loaded in bfloat16:

    sbatch (removed one-off script)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))

import torch

from neuroai.features.llm import _load_moduleformer_direct

REPOS = [
    "ibm-research/MoLM-350M-4B",
    "ibm-research/MoLM-700M-4B",
    "ibm-research/MoLM-700M-8B",
]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    for repo in REPOS:
        try:
            model = _load_moduleformer_direct(repo, {})
            n_params = sum(p.numel() for p in model.parameters())
            model = model.to(dtype=torch.bfloat16, device=device).eval()
            ids = torch.randint(0, 1000, (1, 8), device=device)
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True)
            hs = out.hidden_states if hasattr(out, "hidden_states") else out[-1]
            print(
                f"{repo}: n_layer(cfg)={model.config.n_layer} n_hidden_states={len(hs)} "
                f"n_params_total={n_params:,}"
            )
            del model
        except Exception as exc:  # noqa: BLE001 -- keep checking the rest of REPOS
            print(f"{repo}: FAILED ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
