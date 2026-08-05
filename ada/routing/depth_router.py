"""Depth-routing analytics helpers.

The actual gating decision lives in `ada.layers.adaptive_depth`; this module
holds pure functions for summarizing routing behavior across a batch/run
(used by ADAModel to populate `output.depth_stats`, and by
notebooks/02_routing_visualization.ipynb).
"""

from __future__ import annotations

import torch


def per_token_exit_depth(continue_masks: list[torch.Tensor]) -> torch.Tensor:
    """Given per-block hard continue_mask tensors (True = continues past this
    block), return the exit depth (1-indexed block at which each token last
    was still active) per token.

    continue_masks: list of length num_layers, each (batch, seq_len) bool.
    returns: (batch, seq_len) long tensor of exit depths in [1, num_layers].
    """
    if not continue_masks:
        raise ValueError("continue_masks must be non-empty")
    stacked = torch.stack(continue_masks, dim=0).long()  # (num_layers, b, l)
    # A token's exit depth = 1 + number of blocks after which it still continued.
    exit_depth = stacked.sum(dim=0) + 1
    num_layers = stacked.shape[0]
    return exit_depth.clamp(max=num_layers)


def mean_depth(continue_masks: list[torch.Tensor]) -> float:
    return per_token_exit_depth(continue_masks).float().mean().item()


def expert_utilization_entropy(expert_utilization: torch.Tensor) -> float:
    """Shannon entropy of the mean expert-utilization distribution, in nats.
    Used against the pre-flight success criterion in the roadmap:
    "Utilization entropy > 0.8 x log(N)".
    """
    p = expert_utilization.clamp_min(1e-9)
    p = p / p.sum()
    return float(-(p * p.log()).sum().item())
