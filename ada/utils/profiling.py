"""FLOP estimation and throughput measurement.

`estimate_flops_per_token` uses the standard dense-Transformer approximation
(~6 * activated_params per token for a forward+backward pass, Kaplan et al.
2020 / Chinchilla), adjusted for MoE's activated (not total) parameter
count. This is an estimate for planning/reporting (e.g. filling in the TRC
proposal's "[FILL: FLOP estimate]" tables) — not a measured value. For a
measured value, wrap a training step with `torch.profiler` or, on TPU, the
XLA profiler (see tpu/tpu_profiling.md).
"""

from __future__ import annotations

import time
from typing import cast

import torch
from torch import nn

from ada.config import ADAConfig
from ada.layers.hybrid_block import ADAHybridBlock
from ada.model import ADAModel


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_activated_params(config: ADAConfig, model: ADAModel) -> int:
    """Approximate activated (not total) params per token: dense sublayers
    count fully; MoE experts count only top_k / num_experts of expert params.

    Takes the concrete ADAModel (not generic nn.Module) since it inspects
    `model.blocks[i].moe.experts`, which only exists on ADA's block type.
    """
    total = count_total_params(model)
    if not config.use_moe:
        return total

    expert_params = 0
    for block in cast(list[ADAHybridBlock], list(model.blocks)):
        if block.use_moe:
            expert_params += sum(p.numel() for e in block.moe.experts for p in e.parameters())
    non_expert = total - expert_params
    activated_expert = expert_params * (config.moe_top_k / config.moe_num_experts)
    return int(non_expert + activated_expert)


def estimate_flops_per_token(activated_params: int) -> int:
    """~6 * N FLOPs per token for forward + backward (standard approximation)."""
    return 6 * activated_params


def estimate_total_training_flops(config: ADAConfig, model: ADAModel) -> int:
    activated = count_activated_params(config, model)
    return estimate_flops_per_token(activated) * config.training_tokens


def measure_throughput(model: nn.Module, input_ids: torch.Tensor, num_iters: int = 10, warmup: int = 2) -> float:
    """Returns tokens/second for repeated forward passes of `input_ids` on
    whatever device the model + input already live on.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)

        start = time.perf_counter()
        for _ in range(num_iters):
            model(input_ids)
        elapsed = time.perf_counter() - start

    tokens_per_iter = input_ids.numel()
    return (tokens_per_iter * num_iters) / elapsed
