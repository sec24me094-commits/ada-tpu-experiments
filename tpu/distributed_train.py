#!/usr/bin/env python
"""Multi-host TPU training entry point (PyTorch/XLA).

STATUS: stub / untested. Mirrors scripts/train.py's CLI but wraps the loop
with `xm.mark_step()` and (for multi-host) `torch_xla.distributed.xla_multiprocessing`,
per the roadmap's Week 1 milestone: "xm.mark_step() integration throughout
training loop — target: no 'graph too large' warnings."

Usage (single TPU VM, e.g. v4-8):
    python tpu/distributed_train.py \
        --config configs/ada_nano.yaml \
        --data_path gs://bucket/tokenized/dataset \
        --output_dir gs://bucket/checkpoints/ada-nano \
        --tpu_cores 8

This has not been run against a TPU — treat it as a documented starting
point (see tpu/README.md) rather than a tested implementation.
"""

from __future__ import annotations

import argparse

from ada import ADAConfig
from ada.training.loss import compute_loss
from tpu.pytorch_xla_adapter import build_model_for_xla, mark_step


def _train_loop(index, config: ADAConfig, args):
    import torch
    import torch_xla.core.xla_model as xm

    model = build_model_for_xla(config)
    device = xm.xla_device()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.weight_decay,
    )

    # TODO(Phase 3, Week 1): replace with the GCS streaming pipeline
    # (data/preprocess/*.py output, read via a torch_xla-compatible
    # ParallelLoader) instead of synthetic data.
    max_steps = args.max_steps or max(1, config.training_tokens // config.batch_size_tokens)

    for step in range(max_steps):
        input_ids = torch.randint(0, config.vocab_size, (config.micro_batch_size, 512), device=device)
        labels = input_ids.clone()

        output = model(input_ids, inference=False)
        loss_out = compute_loss(output, labels, config)

        optimizer.zero_grad(set_to_none=True)
        loss_out.total.backward()
        xm.optimizer_step(optimizer)  # all-reduces gradients across TPU cores, then steps
        mark_step()

        if step % args.log_every == 0:
            xm.master_print(f"[step {step}] loss={loss_out.total.item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tpu_cores", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    config = ADAConfig.from_yaml(args.config)

    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(_train_loop, args=(config, args), nprocs=args.tpu_cores)


if __name__ == "__main__":
    main()
