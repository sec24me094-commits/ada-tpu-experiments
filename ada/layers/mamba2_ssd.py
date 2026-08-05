"""Mamba-2 State Space Duality (SSD) sublayer.

NOTE ON SCOPE: this is a correctness-first, framework-only implementation of
selective state-space recurrence (input-dependent A/B/C, per Gu & Dao 2023 /
Dao & Gu 2024). It uses a plain sequential scan so it runs anywhere (CPU,
GPU, TPU/XLA) without custom kernels, which makes it useful for unit tests,
shape/ correctness checks, and small-scale CPU debugging.

It is NOT the chunked semiseparable-matrix SSD training algorithm that makes
Mamba-2 fast at scale. Before running real training jobs (Phase 1+ in the
roadmap), replace `_scan` with either:
  - the official `mamba_ssm` CUDA kernels (GPU), or
  - a chunked/parallel-scan formulation compiled under XLA (TPU) — see
    tpu/pytorch_xla_adapter.py for where that hook belongs.
Swapping the scan implementation should not require changing this module's
public interface (`Mamba2SSD.forward`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ada.config import ADAConfig


class Mamba2SSD(nn.Module):
    def __init__(self, config: ADAConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.state_dim = config.ssm_state_dim
        self.num_heads = config.ssm_num_heads
        self.head_dim = config.hidden_dim // config.ssm_num_heads

        # Input projection -> (x, z, B, C, dt) packed together, split below.
        proj_dim = self.hidden_dim + self.hidden_dim + 2 * self.state_dim + self.num_heads
        self.in_proj = nn.Linear(config.hidden_dim, proj_dim, bias=False)

        # Short depthwise causal conv over x, standard in Mamba to mix local context
        # before the recurrence (kernel size = config.ssm_conv_kernel).
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=config.ssm_conv_kernel,
            groups=self.hidden_dim,
            padding=config.ssm_conv_kernel - 1,
            bias=True,
        )

        # A is per-head, negative (stability), parameterized in log-space.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.num_heads))

        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch, seq_len, hidden_dim)
        returns:       (batch, seq_len, hidden_dim)
        """
        _batch, seq_len, _ = hidden_states.shape
        proj = self.in_proj(hidden_states)  # (b, l, proj_dim)
        x, z, B, C, dt = torch.split(
            proj,
            [self.hidden_dim, self.hidden_dim, self.state_dim, self.state_dim, self.num_heads],
            dim=-1,
        )

        # Causal depthwise conv over x, then SiLU.
        x = x.transpose(1, 2)  # (b, hidden_dim, l)
        x = self.conv1d(x)[..., :seq_len]
        x = F.silu(x.transpose(1, 2))  # (b, l, hidden_dim)

        dt = F.softplus(dt + self.dt_bias)  # (b, l, num_heads), > 0
        A = -torch.exp(self.A_log)  # (num_heads,), negative for stability

        y = self._scan(x, dt, A, B, C)  # (b, l, hidden_dim)
        y = y + x * self.D.repeat_interleave(self.head_dim)

        y = self.norm(y) * F.silu(z)
        return self.out_proj(y)

    def _scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential selective scan (reference implementation, O(seq_len) python loop).

        x:  (b, l, hidden_dim)  reshaped to (b, l, num_heads, head_dim)
        dt: (b, l, num_heads)
        A:  (num_heads,)
        B, C: (b, l, state_dim) — shared across heads (Mamba-2 style)
        """
        b, l, _ = x.shape
        x = x.view(b, l, self.num_heads, self.head_dim)

        # Discretize: dA = exp(dt * A), per (b, l, head)
        dA = torch.exp(dt * A)  # (b, l, num_heads)

        state = x.new_zeros(b, self.num_heads, self.head_dim, self.state_dim)
        outputs = []
        for t in range(l):
            dBx = torch.einsum("bh,bhd,bn->bhdn", dt[:, t], x[:, t], B[:, t])
            state = state * dA[:, t].unsqueeze(-1).unsqueeze(-1) + dBx
            y_t = torch.einsum("bhdn,bn->bhd", state, C[:, t])
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)  # (b, l, num_heads, head_dim)
        return y.reshape(b, l, self.num_heads * self.head_dim)
