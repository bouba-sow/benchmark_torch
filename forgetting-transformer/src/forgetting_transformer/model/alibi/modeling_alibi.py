# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from fla.modules import FusedCrossEntropyLoss, RMSNorm, RotaryEmbedding
from torch.nn import functional as F
from fla.modules.activations import swiglu_linear
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from einops import rearrange

# 动态导入配置类以支持本地和HuggingFace Hub加载
try:
    from .configuration_alibi import AlibiConfig
except (ImportError, ValueError):
    try:
        from configuration_alibi import AlibiConfig
    except ImportError:
        from forgetting_transformer.model.alibi.configuration_alibi import AlibiConfig

from functools import partial

logger = logging.get_logger(__name__)


class Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        window_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        rope_base: float = 500000.0,
        use_rope: bool = False,
        use_alibi: bool = True,
        layer_idx: int = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)
        else:
            self.rotary = None

        if use_alibi:
            # ⭐ 改进：使用GPTNeoX的方式，slopes存为1D向量
            slopes = torch.tensor(self._get_slopes(self.num_heads), dtype=torch.float32)
            self.register_buffer("alibi_slopes", slopes, persistent=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        pass

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        B, T, _ = hidden_states.size()
        q = rearrange(self.q_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_kv_heads)

        seqlen_offset = 0
        max_seqlen = q.shape[1]
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        if self.rotary is not None:
            q, k = self.rotary(q, k, seqlen_offset, max_seqlen)

        q = rearrange(q, 'b t h d -> b h t d')
        k = rearrange(k, 'b t h d -> b h t d')
        v = rearrange(v, 'b t h d -> b h t d')

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        B, H, Tq, Dh = q.shape
        Tk = k.size(2)

        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # ⭐ 改进：使用GPTNeoX的高效ALiBi计算方式
        if hasattr(self, "alibi_slopes"):
            # GPTNeoX方式：slopes @ positions
            # slopes: [H] → [H, 1]
            # positions: [Tk] → [1, Tk]
            # result: [H, Tk] → [1, H, 1, Tk] → [B, H, Tq, Tk]
            positions = torch.arange(Tk, device=scores.device, dtype=torch.float32)
            alibi_slopes = self.alibi_slopes.view(H, 1).to(scores.device)  # [H, 1]
            alibi_bias = torch.matmul(alibi_slopes, positions.unsqueeze(0))  # [H, Tk]
            alibi_bias = alibi_bias.view(1, H, 1, Tk).expand(B, -1, Tq, -1)  # [B, H, Tq, Tk]
            scores = scores + alibi_bias.to(scores.dtype)

        # Causal mask：基于绝对位置
        pos_q = seqlen_offset + torch.arange(Tq, device=scores.device)
        pos_k = torch.arange(Tk, device=scores.device)
        causal_mask = (pos_k.unsqueeze(0) > pos_q.unsqueeze(1))
        scores = scores.masked_fill(causal_mask.view(1, 1, Tq, Tk), float('-inf'))

        # Padding mask
        if attention_mask is not None and attention_mask.shape[-1] == Tk:
            pad_mask = (attention_mask == 0).view(B, 1, 1, Tk)
            scores = scores.masked_fill(pad_mask, float('-inf'))

        # Window mask
        if self.window_size is not None:
            past_too_far = (pos_k.view(1, Tk) < (pos_q.view(Tq, 1) - (self.window_size - 1)))
            scores = scores.masked_fill(past_too_far.view(1, 1, Tq, Tk), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        o = torch.matmul(attn, v)
        o = rearrange(o, 'b h t d -> b t (h d)')
        o = self.o_proj(o)

        attentions = attn if output_attentions else None
        return o, attentions, past_key_values

    def _get_slopes(self, n):
        """
        Get slopes for ALiBi positional embedding
        Based on the original ALiBi paper and GPTNeoX implementation
        
        Returns negative slopes that will be multiplied by position indices
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )
        
        # ⭐ 返回负的slopes（与GPTNeoX一致）
        # 这样可以直接 scores = scores + slopes * positions
        return [-x for x in slopes]


class TransformerMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ) -> 'TransformerMLP':
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
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

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        return swiglu_linear(
            gate, y,
            self.down_proj.weight.to(y.dtype),
            self.down_proj.bias.to(y.dtype) if self.down_proj.bias is not None else self.down_proj.bias
        )


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            window_size=config.window_size,
            use_alibi=config.use_alibi,
            max_position_embeddings=config.max_position_embeddings,
            rope_base=config.rope_base,
            use_rope=config.use_rope,
            layer_idx=layer_idx
        )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )

    def forward_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        return hidden_states, attentions, past_key_values

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ):
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        gradient_checkpointing: bool = False
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        if gradient_checkpointing:
            forward_attn = partial(torch.utils.checkpoint.checkpoint, self.forward_attn, use_reentrant=False)
            forward_mlp = partial(torch.utils.checkpoint.checkpoint, self.forward_mlp, use_reentrant=False)
        else:
            forward_attn = self.forward_attn
            forward_mlp = self.forward_mlp

        hidden_states, attentions, past_key_values = forward_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions
        )

        hidden_states = forward_mlp(
            hidden_states,
            residual,
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class TransformerPreTrainedModel(PreTrainedModel):

    config_class = AlibiConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['TransformerBlock']

    def __init__(self, config, *inputs, **kwargs):
        # 动态修复 config_class 以支持远程代码加载
        if hasattr(config, '__class__'):
            config_module = config.__class__.__module__
            if 'transformers_modules' in config_module or config_module == 'configuration_alibi':
                self.__class__.config_class = config.__class__
        super().__init__(config, *inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class AlibiModel(TransformerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn(
                "`AlibiModel` does not support output attention weights now, so `output_attentions` is set to `False`."
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                if past_key_values is None:
                    past_key_values = DynamicCache()
                else:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                gradient_checkpointing=self.gradient_checkpointing and self.training
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class AlibiForCausalLM(TransformerPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = AlibiModel(config)
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

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids.contiguous()}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]

        loss = None
        if labels is not None:
            if self.config.fuse_cross_entropy:
                loss_fct = FusedCrossEntropyLoss(inplace_backward=True, reduction='none')
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='none')
            logits = self.lm_head(hidden_states)
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = loss.view(*labels.size())
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