#!/usr/bin/env python
"""Export a trained ADA checkpoint to a HuggingFace-Hub-pushable format.

Usage:
    python scripts/export_to_hf.py --checkpoint checkpoints/ada-nano/latest.pt \
        --config configs/ada_nano.yaml --repo_id imal1uqhr/ada-nano-v45 [--push]

This writes config.json + pytorch_model.bin + a minimal model card into
--output_dir, matching what ADAModel.from_pretrained (see ada/model.py) will
eventually expect to load. --push additionally uploads via huggingface_hub;
omit it to just produce the local export directory.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from ada import ADAConfig, ADAModel
from ada.utils.checkpointing import load_checkpoint

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
  - state-space-model
  - mixture-of-experts
  - adaptive-computation
---

# {name}

Adaptive Depth Architecture (ADA) checkpoint. See
https://github.com/sec24me094-commits/ada-tpu-experiments for the full
implementation, training scripts, and paper.

Trained for `{step}` optimizer steps on `{tokens}` tokens
(config: `{config_name}`).

**Status:** checkpoint auto-exported by `scripts/export_to_hf.py` —
replace this card with real evaluation numbers before treating it as a
release artifact.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="hf_export")
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    config = ADAConfig.from_yaml(args.config)
    model = ADAModel(config)
    step = load_checkpoint(model, None, args.checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write(
            MODEL_CARD_TEMPLATE.format(
                name=args.repo_id or config.name,
                step=step,
                tokens=step * config.batch_size_tokens,
                config_name=config.name,
            )
        )

    print(f"Exported to {args.output_dir}")

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo_id is required with --push")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, exist_ok=True)
        api.upload_folder(folder_path=args.output_dir, repo_id=args.repo_id)
        print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
