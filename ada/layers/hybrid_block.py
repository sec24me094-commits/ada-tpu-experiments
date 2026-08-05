"""A single ADA block: Mamba-2 SSD -> MoE -> Dynamic Sparse GQA -> depth gate,
each wrapped with a pre-norm residual connection. Any of the three
processing sublayers can be disabled via config (used by the ablation
variants); the depth gate is always computed but only affects behavior when
`config.use_adaptive_depth` is True (see ADAModel for how the mask is used).
"""

from __future__ import annotations

import torch
from torch import nn

from ada.config import ADAConfig
from ada.layers.adaptive_depth import AdaptiveDepthController
from ada.layers.mamba2_ssd import Mamba2SSD
from ada.layers.moe import MoELayer
from ada.layers.sparse_gqa import DynamicSparseGQA


class ADAHybridBlock(nn.Module):
    def __init__(self, config: ADAConfig):
        super().__init__()
        self.config = config

        self.use_ssm = config.use_ssm
        self.use_moe = config.use_moe

        if self.use_ssm:
            self.ssm_norm = nn.LayerNorm(config.hidden_dim)
            self.ssm = Mamba2SSD(config)

        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        self.attn = DynamicSparseGQA(config)

        if self.use_moe:
            self.moe_norm = nn.LayerNorm(config.hidden_dim)
            self.moe = MoELayer(config)
        else:
            # Ablation ("no_moe"): a single dense FFN takes its place so the
            # block still has a feed-forward sublayer, just without routing.
            self.moe_norm = nn.LayerNorm(config.hidden_dim)
            self.dense_ffn = nn.Sequential(
                nn.Linear(config.hidden_dim, config.expert_hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(config.expert_hidden_dim, config.hidden_dim, bias=False),
            )

        self.depth_controller = AdaptiveDepthController(config.hidden_dim, config.depth_controller_hidden)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, hard_depth: bool = False):
        if self.use_ssm:
            hidden_states = hidden_states + self.dropout(self.ssm(self.ssm_norm(hidden_states)))

        hidden_states = hidden_states + self.dropout(self.attn(self.attn_norm(hidden_states)))

        if self.use_moe:
            hidden_states = hidden_states + self.dropout(self.moe(self.moe_norm(hidden_states)))
            balance_loss = self.moe.last_balance_loss
        else:
            hidden_states = hidden_states + self.dropout(self.dense_ffn(self.moe_norm(hidden_states)))
            balance_loss = None

        continue_prob, continue_mask = self.depth_controller(hidden_states, hard=hard_depth)

        return hidden_states, continue_prob, continue_mask, balance_loss
