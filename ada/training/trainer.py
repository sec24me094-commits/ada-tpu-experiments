"""Minimal single-device training loop.

This is intentionally small — it covers the mechanics (forward, loss,
backward, grad clip, step, log) needed to validate the model trains at all
on CPU/GPU. Multi-host TPU training uses a separate entry point
(tpu/distributed_train.py) that wraps this same `TrainStep` logic under
PyTorch/XLA's parallel loader and `xm.mark_step()` — see that file's
docstring for why the loops are kept separate rather than branching here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from ada.config import ADAConfig
from ada.training.loss import LossOutput, compute_loss
from ada.training.optimizer import build_lr_scheduler, build_optimizer
from ada.utils.checkpointing import save_checkpoint
from ada.utils.logging import StepLogger


@dataclass
class TrainerState:
    step: int = 0
    tokens_seen: int = 0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ADAConfig,
        output_dir: str,
        device: str = "cpu",
        wandb_project: str | None = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir

        total_steps = max(1, config.training_tokens // config.batch_size_tokens)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_lr_scheduler(self.optimizer, config, total_steps)
        self.state = TrainerState()
        self.logger = StepLogger(project=wandb_project, config=config)

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> LossOutput:
        self.model.train()
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        output = self.model(input_ids, inference=False)
        loss_out = compute_loss(output, labels, self.config)

        self.optimizer.zero_grad(set_to_none=True)
        loss_out.total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        self.state.step += 1
        self.state.tokens_seen += input_ids.numel()
        return loss_out

    def fit(
        self,
        data_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
        max_steps: int,
        log_every: int = 10,
        checkpoint_every: int = 1000,
    ) -> None:
        for input_ids, labels in data_iter:
            loss_out = self.train_step(input_ids, labels)

            if self.state.step % log_every == 0:
                self.logger.log(
                    step=self.state.step,
                    tokens_seen=self.state.tokens_seen,
                    lr=self.scheduler.get_last_lr()[0],
                    lm_loss=loss_out.lm_loss.item(),
                    depth_loss=loss_out.depth_loss.item(),
                    balance_loss=loss_out.balance_loss.item(),
                    total_loss=loss_out.total.item(),
                )

            if self.state.step % checkpoint_every == 0:
                save_checkpoint(self.model, self.optimizer, self.state.step, self.output_dir)

            if self.state.step >= max_steps:
                break
