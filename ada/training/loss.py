"""L_total = L_LM + lambda_depth * L_depth + lambda_balance * L_balance

Matches the training objective in the research proposal (Section 5.1 /
README "Adaptive Depth Routing"). Each term is also returned individually so
callers can log them separately (they get noisy/uninformative if only the
sum is tracked).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ada.config import ADAConfig
from ada.model import ADAOutput


@dataclass
class LossOutput:
    total: torch.Tensor
    lm_loss: torch.Tensor
    depth_loss: torch.Tensor
    balance_loss: torch.Tensor


def compute_loss(output: ADAOutput, labels: torch.Tensor, config: ADAConfig) -> LossOutput:
    """
    output.logits: (batch, seq_len, vocab_size)
    labels:        (batch, seq_len) long tensor, next-token targets
                    (already shifted by the caller; use -100 to ignore a position)
    """
    logits = output.logits[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()

    lm_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )

    depth_loss = output.depth_loss if output.depth_loss is not None else torch.tensor(0.0, device=lm_loss.device)

    if output.balance_losses:
        balance_loss = torch.stack(output.balance_losses).mean()
    else:
        balance_loss = torch.tensor(0.0, device=lm_loss.device)

    total = lm_loss + config.lambda_depth * depth_loss + config.lambda_balance * balance_loss

    return LossOutput(total=total, lm_loss=lm_loss, depth_loss=depth_loss, balance_loss=balance_loss)
