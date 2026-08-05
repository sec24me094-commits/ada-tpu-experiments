"""Sparse Mixture-of-Experts sublayer: top-k token-choice routing over N
expert FFNs, plus the auxiliary load-balance loss (Switch/GShard-style,
Fedus et al. 2022).

Implementation is dense-compute (every expert is evaluated on every token,
then masked) rather than true sparse dispatch. This is the correct choice
for correctness testing and small scale; it is NOT the memory/throughput
profile discussed in the TRC proposal's "MoE memory footprint" section —
real training at ADA-Small/Base scale should route via `torch.gather` /
`index_select` dispatch (or an XLA-friendly einsum-based dispatch on TPU)
so that per-device memory reflects only the activated experts' compute, not
a claim this reference module makes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ada.config import ADAConfig
from ada.routing.expert_router import ExpertRouter


class Expert(nn.Module):
    def __init__(self, hidden_dim: int, expert_hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_hidden_dim, bias=False)
        self.fc2 = nn.Linear(expert_hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class MoELayer(nn.Module):
    def __init__(self, config: ADAConfig):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.router = ExpertRouter(config.hidden_dim, config.moe_num_experts, config.moe_top_k)
        self.experts = nn.ModuleList(
            [Expert(config.hidden_dim, config.expert_hidden_dim) for _ in range(config.moe_num_experts)]
        )
        self.last_balance_loss: torch.Tensor | None = None
        self.last_expert_utilization: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch, seq_len, hidden_dim)
        returns:       (batch, seq_len, hidden_dim)
        """
        b, l, d = hidden_states.shape
        flat = hidden_states.reshape(b * l, d)

        topk_weights, topk_idx, router_probs = self.router(flat)  # (n, k), (n, k), (n, num_experts)

        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            mask = (topk_idx == e)  # (n, k)
            if not mask.any():
                continue
            token_mask = mask.any(dim=-1)  # (n,)
            weight = (topk_weights * mask).sum(dim=-1)  # (n,), 0 where expert e not selected
            expert_out = self.experts[e](flat[token_mask])
            out[token_mask] += expert_out * weight[token_mask].unsqueeze(-1)

        self.last_balance_loss = self._load_balance_loss(router_probs, topk_idx)
        self.last_expert_utilization = router_probs.mean(dim=0).detach()

        return out.reshape(b, l, d)

    def _load_balance_loss(self, router_probs: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer-style auxiliary loss: num_experts * sum_e(f_e * P_e).

        f_e = fraction of tokens whose top-1 choice is expert e
        P_e = mean router probability mass assigned to expert e
        Minimized when routing is uniform across experts.
        """
        n = router_probs.shape[0]
        top1 = topk_idx[:, 0]
        f = torch.zeros(self.num_experts, device=router_probs.device)
        f.scatter_add_(0, top1, torch.ones(n, device=router_probs.device))
        f = f / n
        P = router_probs.mean(dim=0)
        return self.num_experts * torch.sum(f * P)
