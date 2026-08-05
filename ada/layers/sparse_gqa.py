"""Dynamic Sparse Grouped Query Attention (GQA) sublayer.

Standard GQA (Ainslie et al. 2023) reduces KV heads relative to query heads.
On top of that, this layer computes an input-dependent sparsity mask via
`ada.routing.sparse_attention_mask` so that attention compute concentrates on
positions estimated to matter, rather than a fixed pattern (sliding window,
etc). When `config.use_dynamic_sparse_attn` is False, the mask degenerates
to standard dense causal attention — this is what the "fixed_attn" ablation
variant in the roadmap uses (see configs/ablation_fixed_attn.yaml).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ada.config import ADAConfig
from ada.routing.sparse_attention_mask import build_dynamic_sparse_mask


class DynamicSparseGQA(nn.Module):
    def __init__(self, config: ADAConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_q_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_dim // config.num_query_heads
        self.group_size = config.num_query_heads // config.num_kv_heads
        self.use_dynamic_sparse = config.use_dynamic_sparse_attn
        self.sparsity_target = config.attn_sparsity_target

        self.q_proj = nn.Linear(config.hidden_dim, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_dim, bias=False)

        self.last_mask_density: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, l, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(b, l, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, l, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, l, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads to match query head groups (GQA).
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (b, h, l, l)

        causal_mask = torch.triu(torch.ones(l, l, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        if self.use_dynamic_sparse:
            sparse_mask, density = build_dynamic_sparse_mask(
                q, k, target_density=1.0 - self.sparsity_target
            )
            scores = scores.masked_fill(~sparse_mask, float("-inf"))
            self.last_mask_density = density
        else:
            self.last_mask_density = torch.tensor(1.0, device=hidden_states.device)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)  # rows that are fully masked (shouldn't happen post-causal, but safe)
        out = torch.matmul(attn, v)  # (b, h, l, head_dim)

        out = out.transpose(1, 2).reshape(b, l, self.num_q_heads * self.head_dim)
        return self.out_proj(out)
