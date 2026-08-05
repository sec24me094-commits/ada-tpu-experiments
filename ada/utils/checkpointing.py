"""Checkpoint save/load.

Saves model + optimizer + step so training can resume after TPU preemption
(see the Risk Register in the project roadmap: "TPU preemption — High
probability — Auto-resume from checkpoint built in"). Local-disk by default;
Phase 1+ should point `output_dir` at a `gs://` FUSE-mounted path or swap in
a GCS-aware save (e.g. via `tensorflow.io.gfile` or `google-cloud-storage`)
without changing this function's signature.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"step_{step:08d}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    # Convenience symlink/copy pointing at the most recent checkpoint, so
    # auto-resume logic can always load `latest.pt` without listing the dir.
    latest_path = os.path.join(output_dir, "latest.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        latest_path,
    )
    return path


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer | None, path: str) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0)


def resume_if_available(model: nn.Module, optimizer: torch.optim.Optimizer, output_dir: str) -> int:
    latest_path = os.path.join(output_dir, "latest.pt")
    if os.path.exists(latest_path):
        return load_checkpoint(model, optimizer, latest_path)
    return 0
