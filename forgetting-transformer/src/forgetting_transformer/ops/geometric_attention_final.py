"""
Geometric Attention - CUDA加速版本 (支持FP16)
"""

import math
import torch
from einops import rearrange
from typing import Optional

# 尝试导入CUDA版本
try:
    from forgetting_transformer.ops.geometric_attention.cuda_interface import (
        load_extension, 
        geometric_attention_activation,
    )
    load_extension()
    HAS_CUDA = True
    print("✅ Using CUDA geometric attention (with FP16 support)")
except Exception as e:
    HAS_CUDA = False
    print(f"⚠️  CUDA not available: {e}")


def geometric_attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    head_first: bool = False,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    normalize: bool = True,
) -> torch.Tensor:
    if not HAS_CUDA:
        raise RuntimeError("CUDA not available")
    
    # ⭐ 保存原始dtype
    original_dtype = q.dtype
    needs_cast = original_dtype == torch.float16
    
    # ⭐ 如果是FP16，转成FP32
    if needs_cast:
        q = q.float()
        k = k.float()
        v = v.float()
    
    # Rearrange
    if not head_first:
        q = rearrange(q, "b t h d -> b h t d")
        k = rearrange(k, "b t h d -> b h t d")
        v = rearrange(v, "b t h d -> b h t d")
    
    B, H, T_q, D = q.shape
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    
    # Attention scores
    logits = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    
    # CUDA kernel (FP32)
    attn_weights = geometric_attention_activation(
        logits, mask=None, pos_offset=0, normalize=normalize
    )
    
    # Apply to values
    output = torch.matmul(attn_weights, v)
    
    # Rearrange back
    if not head_first:
        output = rearrange(output, "b h t d -> b t h d")
    
    # ⭐ 转回原始dtype
    if needs_cast:
        output = output.to(original_dtype)
    
    return output


def geometric_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    head_first: bool = False,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """自动选择CUDA或Python"""
    
    if HAS_CUDA and q.is_cuda:
        try:
            return geometric_attention_cuda(
                q, k, v, head_first=head_first,
                seq_start=seq_start, sm_scale=sm_scale,
                normalize=normalize
            )
        except Exception as e:
            # 不打印太多警告，会刷屏
            pass
    
    # Fallback
    from forgetting_transformer.ops.geometric_attention_std import geometric_attention_std
    return geometric_attention_std(
        q, k, v, head_first=head_first,
        seq_start=seq_start, sm_scale=sm_scale,
        normalize=normalize
    )
