"""Adaptive Depth Controller: per-token soft/hard early-exit gating.

At each block exit, a lightweight 2-layer MLP produces a scalar "continue"
score in [0, 1] per token. During training this is used as a soft residual
weight (fully differentiable — see `forward(..., hard=False)`); at inference
it becomes a hard threshold decision (`hard=True`) so exited tokens skip
remaining blocks entirely (the actual compute saving).

This module does not itself implement inference-time compute skipping (that
requires model-level control flow — see ADAModel.forward) — it only computes
the routing decision and the associated depth-budget loss term.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveDepthController(nn.Module):
    def __init__(self, hidden_dim: int, controller_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, controller_hidden),
            nn.GELU(),
            nn.Linear(controller_hidden, 1),
        )

    def forward(self, hidden_states: torch.Tensor, hard: bool = False, threshold: float = 0.5):
        """
        hidden_states: (batch, seq_len, hidden_dim)
        returns:
            continue_prob: (batch, seq_len) in [0, 1] — probability/weight of
                            continuing to the next block
            continue_mask: (batch, seq_len) bool — only meaningful when hard=True;
                            True = token continues, False = token exits here
        """
        logits = self.net(hidden_states).squeeze(-1)  # (b, l)
        continue_prob = torch.sigmoid(logits)

        if hard:
            continue_mask = continue_prob > threshold
        else:
            continue_mask = torch.ones_like(continue_prob, dtype=torch.bool)

        return continue_prob, continue_mask


def depth_budget_loss(continue_probs: list[torch.Tensor], depth_budget_tau: float) -> torch.Tensor:
    """L_depth: penalizes mean routing depth exceeding the target budget tau.

    continue_probs: list of (batch, seq_len) tensors, one per block, each the
                     continue_prob returned by AdaptiveDepthController.
    depth_budget_tau: target average fraction of total layers used (0, 1].
    """
    if not continue_probs:
        return torch.tensor(0.0)
    stacked = torch.stack(continue_probs, dim=0)  # (num_layers, b, l)
    mean_depth_fraction = stacked.mean()
    return torch.relu(mean_depth_fraction - depth_budget_tau) ** 2
