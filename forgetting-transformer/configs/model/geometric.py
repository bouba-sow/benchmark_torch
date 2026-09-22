from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from typing import List, Any, Optional

from . import ModelConfig


@dataclass
class GeometricArgConfig:
    _target_: str = "forgetting_transformer.model.geometric.configuration_geometric.GeometricConfig"
    vocab_size: int = MISSING  # Should be provided programmatically
    hidden_size: int = MISSING
    hidden_ratio: int = 4
    intermediate_size: Optional[int] = None
    num_hidden_layers: int = MISSING
    num_heads: int = MISSING
    num_kv_heads: Optional[int] = None
    hidden_act: str = "swish"
    window_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    initializer_range: float = 0.02
    elementwise_affine: Optional[bool] = True
    norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    fuse_norm: bool = True
    fuse_cross_entropy: bool = True
    use_rope: bool = False
    # Geometric specific parameters
    use_geometric_normalize: bool = True  # Whether to normalize geometric attention weights
    qk_norm: bool = False
    qk_norm_share_param_across_head: bool = False
    use_k_shift: bool = False
    use_v_shift: bool = False




@dataclass
class GeometricConfig(ModelConfig):
    _target_: str = "forgetting_transformer.model.geometric.modeling_geometric.GeometricForCausalLM"
    config: GeometricArgConfig = GeometricArgConfig()