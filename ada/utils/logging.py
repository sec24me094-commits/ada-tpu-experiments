"""Structured step logging. Always prints to stdout; additionally logs to
Weights & Biases if `wandb_project` is given and the `wandb` package is
importable and the environment is logged in (falls back to stdout-only
otherwise, so unit tests / CI never require W&B credentials).
"""

from __future__ import annotations

from typing import Any


class StepLogger:
    def __init__(self, project: str | None, config: Any = None):
        self.use_wandb = False
        if project:
            try:
                import wandb

                wandb.init(project=project, config=config.__dict__ if config else None)
                self.wandb = wandb
                self.use_wandb = True
            except Exception:  # noqa: BLE001 — intentionally broad: any wandb init
                # failure (missing package, no credentials, offline, etc.) should
                # degrade to stdout-only logging rather than crash training.
                self.use_wandb = False

    def log(self, step: int, **metrics: Any) -> None:
        summary = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"[step {step}] {summary}")
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
