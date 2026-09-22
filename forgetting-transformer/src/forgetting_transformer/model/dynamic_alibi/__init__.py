# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_dynamic_alibi import DynamicAlibiConfig
from .modeling_dynamic_alibi import (
    DynamicAlibiForCausalLM, DynamicAlibiModel)

AutoConfig.register(DynamicAlibiConfig.model_type, DynamicAlibiConfig)
AutoModel.register(DynamicAlibiConfig, DynamicAlibiModel)
AutoModelForCausalLM.register(DynamicAlibiConfig, DynamicAlibiForCausalLM)

__all__ = ['DynamicAlibiConfig', 'DynamicAlibiForCausalLM', 'DynamicAlibiModel']
