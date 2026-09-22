"""
Vanilla Transformer 的标准 Softmax Attention
用于替换 flash_attn 的实现
"""
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple

def vanilla_attention_std(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    标准 Softmax Attention，兼容 flash_attn_func 的输入格式
    
    Args:
        q, k, v: [batch, seq_len, num_heads, head_dim] 格式
        causal: 是否使用因果mask
        window_size: 滑动窗口大小 (left, right)，(-1, -1) 表示无限制
        sm_scale: softmax 缩放因子
    
    Returns:
        output: [batch, seq_len, num_heads, head_dim] 格式
    """
    B, T_q, H, D = q.shape
    T_k = k.shape[1]
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    
    # 转换为 [B, H, T, D] 格式进行计算
    q = rearrange(q, 'b t h d -> b h t d')
    k = rearrange(k, 'b t h d -> b h t d')
    v = rearrange(v, 'b t h d -> b h t d')
    
    # 计算 attention scores
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    
    # Causal mask
    if causal:
        P_SEQ = T_k - T_q  # 处理 KV cache 的情况
        causal_mask = torch.triu(
            torch.ones((T_q, T_k), dtype=torch.bool, device=q.device),
            diagonal=P_SEQ + 1
        )
        scores = scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))
    
    # Window mask (sliding window attention)
    if window_size is not None and window_size != (-1, -1):
        left_window, right_window = window_size
        window_mask = torch.ones((T_q, T_k), dtype=torch.bool, device=q.device)
        for i in range(T_q):
            # 计算每个查询位置的有效窗口范围
            start = max(0, i - left_window)
            end = min(T_k, i + right_window + 1)
            window_mask[i, start:end] = False
        scores = scores.masked_fill(window_mask[None, None, :, :], float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, 0.0)
    
    # Apply attention to values
    output = torch.matmul(attn_weights.to(v.dtype), v)
    
    # 转换回 [B, T, H, D] 格式
    output = rearrange(output, 'b h t d -> b t h d')
    
    return output


def vanilla_attention_varlen_std(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    变长序列的标准 Softmax Attention，兼容 flash_attn_varlen_func
    
    Args:
        q: [total_q_tokens, num_heads, head_dim]
        k: [total_k_tokens, num_kv_heads, head_dim]
        v: [total_k_tokens, num_kv_heads, head_dim]
        cu_seqlens_q: 累积序列长度 [batch_size + 1]
        cu_seqlens_k: 累积序列长度 [batch_size + 1]
        max_seqlen_q: 最大查询序列长度
        max_seqlen_k: 最大键值序列长度
    
    Returns:
        output: [total_q_tokens, num_heads, head_dim]
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    H = q.shape[1]
    D = q.shape[2]
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    
    outputs = []
    
    # 逐批次处理
    for b in range(batch_size):
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b+1].item()
        k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b+1].item()
        
        if q_start == q_end:  # 空序列
            continue
            
        # 提取当前批次的 q, k, v
        q_b = q[q_start:q_end]  # [T_q, H, D]
        k_b = k[k_start:k_end]  # [T_k, H, D] 
        v_b = v[k_start:k_end]  # [T_k, H, D]
        
        T_q = q_b.shape[0]
        T_k = k_b.shape[0]
        
        # 转换为 [H, T, D] 格式
        q_b = rearrange(q_b, 't h d -> h t d')
        k_b = rearrange(k_b, 't h d -> h t d')
        v_b = rearrange(v_b, 't h d -> h t d')
        
        # 计算 attention scores
        scores = torch.matmul(q_b.float(), k_b.float().transpose(-2, -1)) * sm_scale
        
        # Causal mask
        if causal:
            P_SEQ = T_k - T_q
            causal_mask = torch.triu(
                torch.ones((T_q, T_k), dtype=torch.bool, device=q.device),
                diagonal=P_SEQ + 1
            )
            scores = scores.masked_fill(causal_mask[None, :, :], float('-inf'))
        
        # Window mask
        if window_size is not None and window_size != (-1, -1):
            left_window, right_window = window_size
            window_mask = torch.ones((T_q, T_k), dtype=torch.bool, device=q.device)
            for i in range(T_q):
                start = max(0, i - left_window)
                end = min(T_k, i + right_window + 1)
                window_mask[i, start:end] = False
            scores = scores.masked_fill(window_mask[None, :, :], float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        
        # Apply attention
        output_b = torch.matmul(attn_weights.to(v_b.dtype), v_b)
        
        # 转换回 [T, H, D] 格式
        output_b = rearrange(output_b, 'h t d -> t h d')
        outputs.append(output_b)
    
    # 拼接所有批次的输出
    output = torch.cat(outputs, dim=0)
    
    return output
