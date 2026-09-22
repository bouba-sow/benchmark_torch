from transformers import PretrainedConfig


class DynamicSlidingWindowConfig(PretrainedConfig):
    model_type = "dynamic_sliding_window"
    
    def __init__(
        self,
        vocab_size=50304,
        hidden_size=768,
        intermediate_size=None,
        hidden_ratio=4,
        num_hidden_layers=12,
        num_heads=12,
        num_kv_heads=None,
        hidden_act="swish",
        max_position_embeddings=2048,
        initializer_range=0.02,
        norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        attention_bias=False,
        fuse_norm=True,
        fuse_cross_entropy=True,
        use_rope=False,
        # Dynamic sliding window parameters
        use_dynamic_window=False,
        window_num_epochs=10,
        window_initial_lambda=1.0,
        window_decay_rate=0.6,
        context_length=512,  # needed to compute window size from lambda
        qk_norm=False,
        qk_norm_share_param_across_head=False,
        use_k_shift=False,
        use_v_shift=False,
        elementwise_affine=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_ratio * hidden_size
        self.hidden_ratio = hidden_ratio
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_rope = use_rope
        
        # Dynamic sliding window
        self.use_dynamic_window = use_dynamic_window
        self.window_num_epochs = window_num_epochs
        self.window_initial_lambda = window_initial_lambda
        self.window_decay_rate = window_decay_rate
        self.context_length = context_length
        
        self.qk_norm = qk_norm
        self.qk_norm_share_param_across_head = qk_norm_share_param_across_head
        self.use_k_shift = use_k_shift
        self.use_v_shift = use_v_shift
        self.elementwise_affine = elementwise_affine

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
