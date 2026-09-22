import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

from huggingface_hub import PyTorchModelHubMixin


class CbrRnnCell(nn.Module):
    def __init__(self, d_model, d_qk, d_v, d_hid):
        super().__init__()
        self.d_qk = d_qk
        self.d_v = d_v
        
        self.weight_q = nn.Linear(d_model+d_hid, d_qk)
        d_inter = d_v + d_qk + d_model + d_hid
        self.weight_inter = nn.Linear(d_inter, d_inter)
        self.inter_norm = nn.LayerNorm(d_inter)
        self.weight_k = nn.Linear(d_inter, d_qk)
        self.weight_v = nn.Linear(d_inter, d_v)
        self.weight_h = nn.Linear(d_inter, d_hid)
        self.h_norm = nn.LayerNorm(d_hid)
        self.tanh = nn.Tanh()

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return (x * cos) + (x_rotated * sin)

    def attend(self, q_t: torch.Tensor, K_history: torch.Tensor, V_history: torch.Tensor) -> torch.Tensor:
        q_t_unsqueezed = q_t.unsqueeze(1) # [batch, 1, d_qk]
        K_transposed = K_history.permute(1, 2, 0) # [batch, d_qk, time]
        
        # Q @ K^T
        scores = torch.bmm(q_t_unsqueezed, K_transposed) / math.sqrt(self.d_qk) # [batch, 1, time]
        attn_weights = F.softmax(scores, dim=-1)
        # Multiply by V
        V_transposed = V_history.permute(1, 0, 2) # [batch, time, d_v]
        c_t = torch.bmm(attn_weights, V_transposed).squeeze(1) # [batch, d_v]
        
        return c_t

    def compute_mem_update(self, K: torch.Tensor, V: torch.Tensor) \
          -> torch.Tensor:
        # edge case where there is no history (time == 1)
        if K.size(0) <= 1:
            return torch.zeros(K.size(1), device=K.device, dtype=K.dtype)

        K_prev = K[:-1, ...].permute(1, 2, 0) # [batch, d_qk, time-1]
        V_prev = V[:-1, ...].permute(1, 0, 2) # [batch, time-1, d_v]

        k_curr = K[-1] # [batch, d_qk]
        v_curr = V[-1] # [batch, d_v]

        k_norm = torch.norm(k_curr, dim=-1) # [batch]
        k_curr = k_curr.unsqueeze(1) # [batch, 1, d_qk]

        scores = torch.bmm(k_curr, K_prev) / math.sqrt(self.d_qk) # [batch, 1, time-1]
        attn_weights = F.softmax(scores, dim=-1) # [batch, 1, time-1]
        v_pred = torch.bmm(attn_weights, V_prev).squeeze(1) # [batch, d_v]

        v_delta = v_curr - v_pred # [batch, d_v]
        v_norm = torch.norm(v_delta, dim=-1) # [batch]
        # frobenius norm is product of k_norm and v_norm
        mem_update = k_norm * v_norm # [batch]
        return mem_update


class CbrRnnJITLoop(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, inputs: torch.Tensor, h_init: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # inputs: [seq_len, batch, d_model]
        batch_size = inputs.size(1)
        
        k_list = torch.jit.annotate(List[torch.Tensor], [])
        v_list = torch.jit.annotate(List[torch.Tensor], [])
        hiddens = torch.jit.annotate(List[torch.Tensor], [])
        mem_updates = torch.jit.annotate(List[torch.Tensor], [])

        h_t = h_init

        for i in range(inputs.size(0)):
            x_t = inputs[i] # [batch, d_model]
            
            q_input = torch.concat([x_t, h_t], dim=1) # [batch, d_model+d_hid]
            q_raw = self.layer.weight_q(q_input) # [batch, d_qk]
            # tanh activation and layer normalization are not applied
            # to query, key, and value
            q_t = self.layer.apply_rope(q_raw, cos[i], sin[i]) # [batch, d_qk]

            # strictly past attention
            if i == 0:
                # Edge Case: Nothing in the cache yet. 
                # Context is an empty zero vector.
                c_t = torch.zeros(batch_size, self.layer.d_v, device=inputs.device)
            else:
                # Stack the past history (0 to t-1)
                K_past = torch.stack(k_list) # [past_time, batch, d_qk]
                V_past = torch.stack(v_list) # [past_time, batch, d_v]
                
                # Perform cued retrieval against strictly past tokens
                c_t = self.layer.attend(q_t, K_past, V_past) # [batch, d_v]
            
            inter_input = torch.concat([x_t, c_t, q_t, h_t], dim=1) # [batch, d_inter]
            inter = self.layer.weight_inter(inter_input) # [batch, d_inter]
            inter = self.layer.inter_norm(inter)
            inter_output = self.layer.tanh(inter)
            k_raw = self.layer.weight_k(inter_output) # [batch, d_qk]
            k_t = self.layer.apply_rope(k_raw, cos[i], sin[i]) # [batch, d_qk]
            v_t = self.layer.weight_v(inter_output) # [batch, d_v]
            
            # Append to Cache
            k_list.append(k_t)
            v_list.append(v_t)

            mem_update = self.layer.compute_mem_update(
                torch.stack(k_list), torch.stack(v_list)
            )
            
            # compute new hidden state
            hid = self.layer.weight_h(inter_output) # [batch, d_hid]
            hid = self.layer.h_norm(hid)
            h_t = self.layer.tanh(hid)
            hiddens.append(h_t)
            mem_updates.append(mem_update)
        return torch.stack(hiddens), torch.stack(mem_updates) # [seq_len, batch, d_hid], [seq_len, batch]


#class CbrRnn(nn.Module):
class CbrRnn(
    nn.Module,
    PyTorchModelHubMixin
):
    def __init__(self, vocab_size, embed_dim, d_qk, d_v, d_hid, max_seq_len, rope_base):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.d_qk = d_qk
        self.d_v = d_v
        self.d_hid = d_hid
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.rnn = torch.jit.script(CbrRnnJITLoop(CbrRnnCell(embed_dim, d_qk, d_v, d_hid))) 
        self.decoder = nn.Linear(d_hid, vocab_size)

        # RoPE initialization
        freqs = 1.0 / (rope_base ** (torch.arange(0, d_qk, 2).float() / d_qk))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs).float()
        freqs_full = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos", freqs_full.cos().unsqueeze(1))
        self.register_buffer("sin", freqs_full.sin().unsqueeze(1))

    def forward(self, x, hidden=None):
        # x: [batch, seq_len] -> [seq_len, batch]
        x = x.t() 
        seq_len = x.size(0)
        batch_size = x.size(1)
        
        x_emb = self.embedding(x)

        # Initialize standard RNN hidden vectors if none provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.d_hid, device=x.device)

        # concatenate embedding with hidden state
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        
        hiddens, mem_updates = self.rnn(x_emb, hidden, cos, sin)
        rnn_out = hiddens.transpose(0, 1) # [batch, seq_len, d_hid]
        logits = self.decoder(rnn_out)
        
        return logits, hiddens, mem_updates
