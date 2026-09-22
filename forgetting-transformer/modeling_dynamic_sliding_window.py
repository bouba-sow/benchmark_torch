import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_dynamic_sliding_window import DynamicSlidingWindowConfig


class DynamicSlidingWindowAttention(nn.Module):
    def __init__(self, config: DynamicSlidingWindowConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_kv_heads or self.num_heads
        
        # Dynamic window parameters
        self.use_dynamic_window = config.use_dynamic_window
        self.context_length = config.context_length
        self.current_epoch = 0
        self.current_window_size = self.context_length  # default: full context
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
    def _get_dynamic_window_size(self):
        """
        Compute window size from lambda using same formula as ALiBi/FG.
        lambda_t = initial_lambda * decay_rate ^ current_epoch
        window_t = max(1, round(context_length ^ (1 - lambda_t)))
        """
        if not self.use_dynamic_window:
            return self.context_length
            
        lambda_t = self.config.window_initial_lambda * (self.config.window_decay_rate ** self.current_epoch)
        # Clamp lambda to [0, 1]
        lambda_t = max(0.0, min(1.0, lambda_t))
        # Convert lambda to window size
        # When lambda=1: window = context^0 = 1 (most constrained)
        # When lambda=0: window = context^1 = full context (unconstrained)
        window = max(1, round(self.context_length ** (1.0 - lambda_t)))
        return window
    
    def update_epoch(self, epoch):
        """Update current epoch and recompute window size."""
        self.current_epoch = epoch
        self.current_window_size = self._get_dynamic_window_size()
        
    def _make_sliding_window_mask(self, seq_len, device):
        """
        Create a causal mask with sliding window constraint.
        Each token can only attend to tokens within the window.
        """
        # Standard causal mask: lower triangular
        # row i can attend to columns 0..i
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Causal: col <= row
        causal_mask = col_idx <= row_idx
        
        # Window: row - col < window_size (distance within window)
        window_mask = (row_idx - col_idx) < self.current_window_size
        
        # Combine: must be both causal and within window
        mask = causal_mask & window_mask
        
        return mask  # [seq_len, seq_len]
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, T, H = hidden_states.shape
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        # Apply sliding window causal mask
        sw_mask = self._make_sliding_window_mask(T, hidden_states.device)
        sw_mask = sw_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        
        attn_weights = attn_weights.masked_fill(~sw_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output, None


class DynamicSlidingWindowMLP(nn.Module):
    def __init__(self, config: DynamicSlidingWindowConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DynamicSlidingWindowDecoderLayer(nn.Module):
    def __init__(self, config: DynamicSlidingWindowConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.attn = DynamicSlidingWindowAttention(config, layer_idx)
        self.mlp = DynamicSlidingWindowMLP(config)
        
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


class DynamicSlidingWindowModel(PreTrainedModel):
    config_class = DynamicSlidingWindowConfig
    _no_split_modules = ["DynamicSlidingWindowDecoderLayer"]
    
    def __init__(self, config: DynamicSlidingWindowConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            DynamicSlidingWindowDecoderLayer(config, layer_idx)
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


class DynamicSlidingWindowForCausalLM(PreTrainedModel):
    config_class = DynamicSlidingWindowConfig
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["DynamicSlidingWindowDecoderLayer"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = DynamicSlidingWindowModel(config)
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
    
    def update_epoch(self, epoch):
        """Update epoch for all attention layers to change window size."""
        lambda_t = self.config.window_initial_lambda * (self.config.window_decay_rate ** epoch)
        lambda_t = max(0.0, min(1.0, lambda_t))
        window = max(1, round(self.config.context_length ** (1.0 - lambda_t)))
        
        for layer in self.model.layers:
            layer.attn.update_epoch(epoch)
        
        return lambda_t, window
    
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
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            B, T = shift_logits.size(0), shift_logits.size(1)
            loss = loss.view(B, T)
            loss = F.pad(loss, (0, 1), value=0.0)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
