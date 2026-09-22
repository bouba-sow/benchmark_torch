"""
Geometric Attention - 标准 Softmax 版本
基于论文 "The Neural Data Router" (Csordás et al., 2022)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


def geometric_attention_std(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    head_first: bool = False,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    标准 Softmax 版本的 Geometric Attention
    
    Args:
        q: Query tensor [B, T, H, D] or [B, H, T, D] if head_first
        k: Key tensor [B, T, H, D] or [B, H, T, D] if head_first
        v: Value tensor [B, T, H, D] or [B, H, T, D] if head_first
        head_first: 是否head维度在前
        seq_start: 序列起始位置 [B]
        sm_scale: scaling factor，默认 1/sqrt(D)
        normalize: 是否归一化attention weights
    
    Returns:
        output: [B, T, H, D] or [B, H, T, D] if head_first
    """
    
    # Rearrange to head_first format
    if not head_first:
        q = rearrange(q, "b t h d -> b h t d")
        k = rearrange(k, "b t h d -> b h t d")
        v = rearrange(v, "b t h d -> b h t d")
    
    B, H, T_q, D = q.shape
    T_k = k.shape[2]
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    
    # Step 1: 计算 content-based logits
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    # logits: [B, H, T_q, T_k]
    
    # Step 2: Mask diagonal (不允许attend到自己)
    if T_q == T_k:
        diag_mask = torch.eye(T_q, dtype=torch.bool, device=q.device)
        logits = logits.masked_fill(diag_mask[None, None, :, :], float('-inf'))
    
    # Step 3: 处理 seq_start mask
    if seq_start is not None:
        seq_mask = torch.arange(T_k, device=q.device)[None, None, None, :] < seq_start[None, :, None, None]
        logits = logits.masked_fill(seq_mask, float('-inf'))
    
    # Step 4: Causal mask (如果需要)
    # 注意：geometric attention论文中没有causal，如果你的任务需要可以取消注释
    # P_SEQ = T_k - T_q
    # causal_mask = torch.triu(torch.ones((T_q, T_k), dtype=torch.bool, device=q.device), diagonal=P_SEQ + 1)
    # logits = logits.masked_fill(causal_mask[None, None, :, :], float('-inf'))
    
    # Step 5: Geometric weighting (核心算法)
    attn_weights = geometric_weighting(logits, normalize=normalize)
    
    # Step 6: 应用attention到values
    out = torch.matmul(attn_weights.to(v.dtype), v)
    
    if not head_first:
        out = rearrange(out, "b h t d -> b t h d")
    
    return out


def geometric_weighting(
    logits: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    计算geometric attention weights
    
    实现论文中的 Equation 7:
    A[i,j] = P[i,j] * ∏(1 - P[i,k]) for k closer to i than j
    
    Args:
        logits: [B, H, T_q, T_k] attention logits
        normalize: 是否归一化
    
    Returns:
        weights: [B, H, T_q, T_k] attention weights
    """
    B, H, T_q, T_k = logits.shape
    
    # Step 1: Sigmoid to get matching probabilities
    P = torch.sigmoid(logits)  # [B, H, T_q, T_k]
    
    # Step 2: 使用 log-space 计算（数值稳定）
    log_P = torch.log(P + 1e-10)
    log_one_minus_P = torch.log(1.0 - P + 1e-10)
    
    # Step 3: 简化版本 - 使用cumsum实现几何分布
    # 这是一个高效的近似，避免了显式的循环
    
    # 对于每个位置i，计算其左侧所有位置的log(1-P)累积和
    log_decay_left = log_one_minus_P.cumsum(dim=-1)
    
    # 计算weights（简化版）
    # 完整版本需要根据距离动态选择区间，这里用一个高效近似
    weights = torch.exp(log_P + log_decay_left.roll(1, dims=-1))
    
    # 第一个位置特殊处理（没有左侧元素）
    # 避免inplace操作
    weights_first = P[:, :, :, :1]  # 获取第一列
    weights = torch.cat([weights_first, weights[:, :, :, 1:]], dim=-1)
    
    # Step 4: 归一化（可选）
    if normalize:
        weights = F.normalize(weights, p=1, dim=-1)
    
    # 处理NaN（如果所有位置都是-inf）
    weights = torch.nan_to_num(weights, 0.0)
    
    return weights


def geometric_weighting_full(
    logits: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    完整版geometric weighting（更慢但更准确）
    
    仅在需要最高精度时使用，训练时建议用上面的简化版
    """
    B, H, T_q, T_k = logits.shape
    device = logits.device
    
    P = torch.sigmoid(logits)
    log_P = torch.log(P + 1e-10)
    log_one_minus_P = torch.log(1.0 - P + 1e-10)
    
    # 初始化weights
    weights = torch.zeros_like(P)
    
    # 对每个(i,j)计算geometric weight
    for i in range(T_q):
        for j in range(T_k):
            # 找出比j更接近i的所有位置k
            if i < j:
                # 向右看：closer positions are [i+1, ..., j-1]
                closer_positions = range(i + 1, j)
            elif i > j:
                # 向左看：closer positions are [j+1, ..., i-1]
                closer_positions = range(j + 1, i)
            else:
                # i == j (对角线)，已经在外面mask掉了
                continue
            
            # 计算 ∏(1 - P[i,k]) in log-space
            log_prod = sum(log_one_minus_P[:, :, i, k] for k in closer_positions) if closer_positions else 0.0
            
            # weights[i,j] = P[i,j] * ∏(1 - P[i,k])
            weights[:, :, i, j] = torch.exp(log_P[:, :, i, j] + log_prod)
    
    if normalize:
        weights = F.normalize(weights, p=1, dim=-1)
    
    weights = torch.nan_to_num(weights, 0.0)
    
    return weights 