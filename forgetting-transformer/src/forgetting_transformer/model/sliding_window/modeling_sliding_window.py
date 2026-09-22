import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_sliding_window import SlidingWindowConfig
from forgetting_transformer.ops.sliding_window_attention_std import sliding_window_attention_std


class SlidingWindowAttention(nn.Module):
    def __init__(self, config: SlidingWindowConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_kv_heads or self.num_heads
        self.window_size = config.window_size
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, T, H = hidden_states.shape
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)
        
        # Sliding window attention
        attn_output = sliding_window_attention_std(
            q, k, v,
            head_first=False,
            window_size=self.window_size,
        )
        
        # Output projection
        attn_output = attn_output.reshape(B, T, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output, None


class SlidingWindowMLP(nn.Module):
    def __init__(self, config: SlidingWindowConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SlidingWindowDecoderLayer(nn.Module):
    def __init__(self, config: SlidingWindowConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.attn = SlidingWindowAttention(config, layer_idx)
        self.mlp = SlidingWindowMLP(config)
        
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps, elementwise_affine=config.elementwise_affine)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps, elementwise_affine=config.elementwise_affine)
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, None


class SlidingWindowModel(PreTrainedModel):
    config_class = SlidingWindowConfig
    _no_split_modules = ["SlidingWindowDecoderLayer"]  # ← 关键修复1
    
    def __init__(self, config: SlidingWindowConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            SlidingWindowDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, elementwise_affine=config.elementwise_affine)
        
        self.gradient_checkpointing = False
        self.post_init()
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        
        for decoder_layer in self.layers:
            hidden_states, _ = decoder_layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


class SlidingWindowForCausalLM(PreTrainedModel):
    config_class = SlidingWindowConfig
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["SlidingWindowDecoderLayer"]  # ← 关键修复2
    
    def __init__(self, config):
        super().__init__(config)
        self.model = SlidingWindowModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Return per-token loss with shape [B, T-1]
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            # Reshape to [B, T-1]
            B, T = shift_logits.size(0), shift_logits.size(1)
            loss = loss.view(B, T)
            
            # Pad last position to make shape [B, T] instead of [B, T-1]
            loss = F.pad(loss, (0, 1), value=0.0)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}