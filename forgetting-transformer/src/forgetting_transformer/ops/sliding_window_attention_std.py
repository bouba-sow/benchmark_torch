"""
Sliding Window / Hard Attention
Based on "Context Limitations Make Neural Language Models More Human-Like"
(Kuribayashi et al., 2022)
"""

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


def sliding_window_attention_std(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    head_first: bool = False,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    window_size: int = 2,  # 默认2-gram（看前1个token）
) -> torch.Tensor:
    """
    Sliding Window Attention
    
    硬截断：只能attend到最近window_size个token
    """
    
    if not head_first:
        q = rearrange(q, "b t h d -> b h t d")
        k = rearrange(k, "b t h d -> b h t d")
        v = rearrange(v, "b t h d -> b h t d")
    
    B, H, T_q, D = q.shape
    T_k = k.shape[2]
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    
    # Compute logits
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    
    # Create sliding window mask
    mask = create_sliding_window_mask(T_q, T_k, window_size, device=q.device)
    logits = logits.masked_fill(~mask, float('-inf'))
    
    # Seq start mask
    if seq_start is not None:
        seq_mask = torch.arange(T_k, device=q.device)[None, None, None, :] < seq_start[None, :, None, None]
        logits = logits.masked_fill(seq_mask, float('-inf'))
    
    # Standard softmax
    weights = F.softmax(logits, dim=-1)
    
    # Apply to values (cast back down from the fp32 softmax to v's dtype)
    out = torch.matmul(weights.to(v.dtype), v)
    
    if not head_first:
        out = rearrange(out, "b h t d -> b t h d")
    
    return out


def create_sliding_window_mask(
    T_q: int,
    T_k: int,
    window_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    创建sliding window mask
    
    window_size=1: 只看前1个token (2-gram)
    window_size=2: 只看前2个token (3-gram)
    """
    # 基础causal mask
    mask = torch.tril(torch.ones(T_q, T_k, dtype=torch.bool, device=device))
    
    # 应用window限制
    if window_size > 0 and window_size < T_k:
        for i in range(T_q):
            # 只保留 [i-window_size+1, i] 范围
            start = max(0, i - window_size + 1)
            if start > 0:
                mask[i, :start] = False
    
    return mask[None, None, :, :]  # [1, 1, T_q, T_k]