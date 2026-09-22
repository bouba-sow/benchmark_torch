# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_alibi import AlibiConfig
from .modeling_alibi import (
    AlibiForCausalLM, AlibiModel)

AutoConfig.register(AlibiConfig.model_type, AlibiConfig)
AutoModel.register(AlibiConfig, AlibiModel)
AutoModelForCausalLM.register(AlibiConfig, AlibiForCausalLM)



__all__ = ['AlibiConfig', 'AlibiForCausalLM', 'AlibiModel']
