# -*- coding: utf-8 -*-

from transformers import PretrainedConfig


class StickbreakingConfig(PretrainedConfig):
    model_type = "stickbreaking"
    
    def __init__(
        self,
        vocab_size: int = 50277,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: int = None,
        hidden_act: str = "swish",
        intermediate_size: int = None,
        hidden_ratio: int = 4,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        # Stickbreaking specific
        attend_current: bool = False,  # Whether to attend to current position
        normalize_attention: bool = True,  # Whether to normalize attention weights
        # Optional features (same as other models)
        use_rope: bool = False,
        rope_base: float = 500000.0,
        qk_norm: bool = False,
        qk_norm_share_param_across_head: bool = False,
        use_k_shift: bool = False,
        use_v_shift: bool = False,
        window_size: int = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_ratio = hidden_ratio
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        
        # Stickbreaking specific
        self.attend_current = attend_current
        self.normalize_attention = normalize_attention
        
        # Optional features
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.qk_norm = qk_norm
        self.qk_norm_share_param_across_head = qk_norm_share_param_across_head
        self.use_k_shift = use_k_shift
        self.use_v_shift = use_v_shift
        self.window_size = window_size
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )