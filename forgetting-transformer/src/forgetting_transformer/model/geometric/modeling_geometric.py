# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from fla.modules import FusedCrossEntropyLoss, RMSNorm
from fla.modules.layernorm import group_norm_fn
from fla.modules.activations import swiglu_linear

from fla.modules import RotaryEmbedding
from einops import rearrange

# 动态导入配置类
try:
    from .configuration_geometric import GeometricConfig
except (ImportError, ValueError):
    try:
        from configuration_geometric import GeometricConfig
    except ImportError:
        from forgetting_transformer.model.geometric.configuration_geometric import GeometricConfig

# 🔥 导入geometric attention
from forgetting_transformer.ops.geometric_attention_final import geometric_attention

logger = logging.get_logger(__name__)


class ShiftLinear(nn.Module):
    """
    Data-dependent token shift (from forgetting transformer)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        bias: bool,
        shift_bias: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        assert self.output_dim % self.num_heads == 0

        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.shift_proj = nn.Linear(input_dim, num_heads, bias=shift_bias)

    def forward(self, x: torch.Tensor, shift_state: Optional[torch.Tensor]) -> torch.Tensor:
        # 简化版本：不使用shift（geometric不需要）
        return self.linear(x)


class GroupRMSNorm(nn.Module):
    """Group RMSNorm for multi-head normalization"""
    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return group_norm_fn(x, self.num_groups, self.weight, self.eps)


class GeometricAttention(nn.Module):
    """
    Geometric Attention Layer
    基于 "The Neural Data Router" 论文实现
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        window_size: Optional[int] = None,
        max_position_embeddings: int = 2048,
        use_rope: bool = False,
        rope_base: float = 500000.0,
        qk_norm: bool = False,
        qk_norm_share_param_across_head: bool = False,
        use_k_shift: bool = False,
        use_v_shift: bool = False,
        use_geometric_normalize: bool = True,
        norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        layer_idx: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            - hidden_size: dimension of hidden representations
            - num_heads: number of attention heads
            - num_kv_heads: (optional) For GQA, number of key-value heads
            - window_size: (optional) used for sliding window
            - max_position_embeddings: maximum sequence length
            - use_rope: whether to use rotary embeddings
            - rope_base: base for RoPE
            - qk_norm: Whether to use qk_norm
            - qk_norm_share_param_across_head: In QK-norm, whether to share params
            - use_k_shift: Whether to use data-dependent key shift  
            - use_v_shift: Whether to use data-dependent value shift
            - use_geometric_normalize: Whether to normalize geometric attention weights
            - norm_eps: epsilon for normalization
            - initializer_range: standard deviation for initialization
            - layer_idx: The block index of this layer (for KV-cache)
        """
        super().__init__()

        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            raise NotImplementedError("GQA has not been tested.")
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.use_geometric_normalize = use_geometric_normalize

        # QKV projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        if use_k_shift:
            self.k_proj = ShiftLinear(self.hidden_size, self.kv_dim, self.num_heads, bias=False)
        else:
            self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        if use_v_shift:
            self.v_proj = ShiftLinear(self.hidden_size, self.kv_dim, self.num_heads, bias=False)
        else:
            self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.use_k_shift = use_k_shift
        self.use_v_shift = use_v_shift

        # RoPE (optional)
        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)
        else:
            self.rotary = None

        # QK normalization (optional)
        self.qk_norm = qk_norm
        self.qk_norm_share_param_across_head = qk_norm_share_param_across_head
        if qk_norm:
            if self.qk_norm_share_param_across_head:
                self.q_norm = RMSNorm(self.head_dim)
                self.k_norm = RMSNorm(self.head_dim)
            else:
                self.q_norm = GroupRMSNorm(num_groups=self.num_heads, hidden_size=self.hidden_size, eps=norm_eps)
                self.k_norm = GroupRMSNorm(num_groups=self.num_heads, hidden_size=self.hidden_size, eps=norm_eps)

        self.initializer_range = initializer_range
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of geometric attention
        """
        batch_size, q_len, _ = hidden_states.size()
        
        # Geometric attention不使用shift，设为None
        key_shift_state = None
        value_shift_state = None

        # QKV projections
        q = self.q_proj(hidden_states)
        if self.use_k_shift:
            k = self.k_proj(hidden_states, key_shift_state)
        else:
            k = self.k_proj(hidden_states)
        if self.use_v_shift:
            v = self.v_proj(hidden_states, value_shift_state)
        else:
            v = self.v_proj(hidden_states)

        # QK normalization (optional)
        if self.qk_norm and (not self.qk_norm_share_param_across_head):
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)

        # Reshape for multi-head
        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(k, '... (h d) -> ... h d', h=self.num_kv_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_kv_heads)

        if self.qk_norm and (self.qk_norm_share_param_across_head):
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)

        # RoPE (optional)
        seqlen_offset, max_seqlen = 0, q.shape[1]
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx) if hasattr(past_key_values, 'get_seq_length') else 0
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                seqlen_offset = (seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1])
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        
        if self.rotary is not None:
            q, k = self.rotary(q, k, seqlen_offset, max_seqlen)

        # Update KV cache if needed
        if past_key_values is not None and use_cache:
            # 使用标准的DynamicCache接口
            if hasattr(past_key_values, 'update'):
                k_cache = rearrange(k, 'b t h d -> b h t d')
                v_cache = rearrange(v, 'b t h d -> b h t d')
                past_key_values.update(k_cache, v_cache, self.layer_idx)
            # 注意：这里不需要重新赋值k和v，因为我们在训练时不使用cache

        # Handle GQA (if enabled)
        if self.num_kv_groups > 1:
            k = rearrange(k.unsqueeze(-2).repeat(1, 1, 1, self.num_kv_groups, 1), 'b t h g d -> b t (h g) d')
            v = rearrange(v.unsqueeze(-2).repeat(1, 1, 1, self.num_kv_groups, 1), 'b t h g d -> b t (h g) d')

        # 🔥 Geometric Attention (核心)
        if attention_mask is not None:
            B, T = attention_mask.size()
            seq_start = T - attention_mask.sum(dim=-1)
            o = geometric_attention(
                q, k, v,
                head_first=False,
                seq_start=seq_start,
                sm_scale=1 / math.sqrt(self.head_dim),
                normalize=self.use_geometric_normalize,
            )
        else:
            o = geometric_attention(
                q, k, v,
                head_first=False,
                sm_scale=1 / math.sqrt(self.head_dim),
                normalize=self.use_geometric_normalize,
            )

        # Reshape output
        o = o.reshape(batch_size, q_len, self.hidden_size)

        # Output projection
        o = self.o_proj(o)

        # Attention weights (if requested)
        attentions = None
        if output_attentions:
            # 简化版：不返回详细的attention weights
            attentions = None

        return o, attentions, past_key_values


class GeometricMLP(nn.Module):
    """
    MLP层 (与ForgettingTransformer完全相同)
    """
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[float] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        self.hidden_act = hidden_act

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * y)


class GeometricBlock(nn.Module):
    """
    Transformer Block with Geometric Attention
    """
    def __init__(self, config: GeometricConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.norm_eps
        )
        self.attn = GeometricAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            window_size=config.window_size,
            max_position_embeddings=config.max_position_embeddings,
            use_rope=config.use_rope,
            rope_base=config.rope_base,
            qk_norm=config.qk_norm,
            qk_norm_share_param_across_head=config.qk_norm_share_param_across_head,
            use_k_shift=config.use_k_shift,
            use_v_shift=config.use_v_shift,
            use_geometric_normalize=config.use_geometric_normalize,
            norm_eps=config.norm_eps,
            initializer_range=config.initializer_range,
            layer_idx=layer_idx
        )

        self.mlp_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.norm_eps
        )
        self.mlp = GeometricMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # Attention block with residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP block with residual  
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)
        return outputs


class GeometricPreTrainedModel(PreTrainedModel):
    config_class = GeometricConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["GeometricBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GeometricModel(GeometricPreTrainedModel):
    """
    Geometric Transformer Model
    """
    def __init__(self, config: GeometricConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GeometricBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        hidden_states = inputs_embeds

        # Layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            past_key_values = layer_outputs[2]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GeometricForCausalLM(GeometricPreTrainedModel):
    """
    Geometric Transformer for Causal Language Modeling
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GeometricModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Model forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        loss = None
        if labels is not None:
            if self.config.fuse_cross_entropy:
                loss_fct = FusedCrossEntropyLoss(inplace_backward=True, reduction='none')
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='none')
            logits = self.lm_head(hidden_states)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = loss.view(*labels.size())  # Reshape to [batch, seq_len]
            del logits
            logits = None
        else:
            logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
