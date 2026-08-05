#!/usr/bin/env python
"""Main training entry point.

Usage:
    python scripts/train.py \
        --config configs/ada_nano.yaml \
        --data_path /path/to/tokenized/dataset \
        --output_dir checkpoints/ada-nano \
        --wandb_project ada-research

The data pipeline is intentionally left pluggable: this script expects
`data_path` to be a directory readable by `ada.utils` dataset loading (see
data/preprocess/pack.py for the expected packed-sequence format). Until a
real tokenized dataset exists, pass `--synthetic` to sanity-check the full
train loop against random data (this is what CI's model_tests workflow does).
"""

from __future__ import annotations

import argparse
import os

import torch

from ada import ADAConfig, ADAModel
from ada.training.trainer import Trainer
from ada.utils.checkpointing import resume_if_available


def synthetic_data_iter(config: ADAConfig, device: str, num_batches: int):
    for _ in range(num_batches):
        input_ids = torch.randint(0, config.vocab_size, (config.micro_batch_size, min(config.max_seq_len, 512)))
        labels = input_ids.clone()
        yield input_ids.to(device), labels.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file")
    parser.add_argument("--data_path", default=None, help="Path to tokenized/packed dataset directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_steps", type=int, default=None, help="Override steps derived from training_tokens")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir/latest.pt if present")
    parser.add_argument("--synthetic", action="store_true", help="Train on random data (smoke test only)")
    args = parser.parse_args()

    config = ADAConfig.from_yaml(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    config.to_yaml(os.path.join(args.output_dir, "config.yaml"))

    model = ADAModel(config)
    trainer = Trainer(model, config, args.output_dir, device=args.device, wandb_project=args.wandb_project)

    if args.resume:
        resumed_step = resume_if_available(trainer.model, trainer.optimizer, args.output_dir)
        trainer.state.step = resumed_step
        print(f"Resumed from step {resumed_step}")

    max_steps = args.max_steps or max(1, config.training_tokens // config.batch_size_tokens)

    if args.synthetic or args.data_path is None:
        if not args.synthetic:
            print("WARNING: --data_path not given; falling back to --synthetic random data.")
        data_iter = synthetic_data_iter(config, args.device, max_steps + 1)
    else:
        raise NotImplementedError(
            "Real dataset loading isn't wired up yet — see data/preprocess/ for the "
            "tokenize -> filter -> dedupe -> pack pipeline this should read from, "
            "and data/README.md for the expected on-disk format. Use --synthetic "
            "for a smoke test in the meantime."
        )

    trainer.fit(
        data_iter,
        max_steps=max_steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
