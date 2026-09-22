"""Vendored from https://github.com/IBM/ModuleFormer (2026-09-12), trimmed
to what neuroai's feature-extraction loader needs (config + causal LM only;
ONNX export config dropped, see configuration_moduleformer.py's docstring).
"""

from .configuration_moduleformer import ModuleFormerConfig
from .modeling_moduleformer import ModuleFormerForCausalLM, ModuleFormerModel

__all__ = ["ModuleFormerConfig", "ModuleFormerForCausalLM", "ModuleFormerModel"]
