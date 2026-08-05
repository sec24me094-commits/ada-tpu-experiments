"""AdamW optimizer + cosine-with-warmup schedule, configured from ADAConfig.

Weight decay is excluded from biases, LayerNorm parameters, and embeddings,
per common practice (GPT-3 / Chinchilla style).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def build_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name.lower() or "bias" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
    )


def build_lr_scheduler(optimizer: torch.optim.Optimizer, config, total_steps: int):
    warmup_steps = config.warmup_steps
    min_lr_ratio = config.min_learning_rate / config.learning_rate

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
