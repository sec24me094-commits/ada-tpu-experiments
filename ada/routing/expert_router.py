"""Top-k token-choice expert routing (Shazeer et al. 2017 style gating)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ExpertRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: (n, hidden_dim) flattened tokens
        returns:
            topk_weights: (n, top_k) softmax weight for each chosen expert
            topk_idx:     (n, top_k) chosen expert indices
            router_probs: (n, num_experts) full softmax distribution (for the
                           load-balance auxiliary loss)
        """
        logits = self.gate(x)  # (n, num_experts)
        router_probs = F.softmax(logits, dim=-1)
        topk_weights, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return topk_weights, topk_idx, router_probs
