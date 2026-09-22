# Dependencies and vendored code

* **Main environment**: `uv sync` (Python 3.12, CUDA 12.4 PyTorch wheels). `mamba-ssm` and
  `causal-conv1d` are compiled wheels for Python 3.12 (see `pyproject.toml`).
* **R** with `lme4`/`lmerTest`, used through `pymer4`, for the surprisal track.
* **DLT environment** (`.venv-dlt`): spaCy, separate from the main one.
* **Vendored, patched libraries** (kept in place because the editable install and the model
  loaders import them):
  * `forgetting-transformer/`: patched for the `fla` `RotaryEmbedding.forward` argument order
    and the removed `DynamicCache.from_legacy_cache` API (FoX / forgetting-transformer models).
  * `moduleformer/`: IBM ModuleFormer, patched to drop the `transformers.onnx` dependency and
    to inline the removed `get_head_mask` (MoLM). Loaded by
    `neuroai/features/llm.py::_load_moduleformer_direct`.
  * `stickbreaking-attention/`: stick-breaking attention models.
* Add the upstream revision and licence of each vendored tree here before release.
