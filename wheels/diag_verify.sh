#!/bin/bash
set -euo pipefail
source /lustre/fswork/projects/rech/ldh/una68ug/benchmark/.venv/bin/activate
python -c "
import mamba_ssm
print('mamba_ssm imported OK, version', mamba_ssm.__version__, 'MambaLMHeadModel=', mamba_ssm.MambaLMHeadModel)
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
print('selective_state_update OK')
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
print('selective_scan_fn / mamba_inner_fn OK')
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
print('causal_conv1d_fn / causal_conv1d_update OK')
from transformers.utils.import_utils import is_mamba_ssm_available, is_causal_conv1d_available
print('is_mamba_ssm_available:', is_mamba_ssm_available())
print('is_causal_conv1d_available:', is_causal_conv1d_available())
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('state-spaces/mamba-130m-hf', torch_dtype=torch.bfloat16).cuda()
print('model loaded, is_fast_path in use - check no warning above')
import flash_linear_attention
print('flash_linear_attention OK', flash_linear_attention.__version__ if hasattr(flash_linear_attention,\"__version__\") else '?')
"
