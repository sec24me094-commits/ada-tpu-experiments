#!/usr/bin/env python
"""Compute a FLOP / parameter-count profile for a given config, without
training. Useful for filling in the TRC proposal's "[FILL: FLOP estimate]"
tables and for tuning hidden_dim/num_layers to hit a target parameter count.

Usage:
    python scripts/profile_flops.py --config configs/ada_nano.yaml
"""

from __future__ import annotations

import argparse

import torch

from ada import ADAConfig, ADAModel
from ada.utils.profiling import (
    count_activated_params,
    count_total_params,
    estimate_flops_per_token,
    estimate_total_training_flops,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Actually allocate weights (needed for --throughput-style checks). "
        "Default builds on torch's meta device so large configs (e.g. ada_base, "
        "which needs ~15GB+ of real RAM) can still be profiled for param counts "
        "on modest machines.",
    )
    args = parser.parse_args()

    config = ADAConfig.from_yaml(args.config)
    device = "cpu" if args.materialize else "meta"
    with torch.device(device):
        model = ADAModel(config)

    total = count_total_params(model)
    activated = count_activated_params(config, model)
    flops_per_token = estimate_flops_per_token(activated)
    total_flops = estimate_total_training_flops(config, model)

    print(f"config: {config.name}")
    print(f"total params:      {total / 1e6:,.1f}M  (~{total * 4 / 1e9:.1f} GB in fp32)")
    print(f"activated params:  {activated / 1e6:,.1f}M  (per forward pass, MoE-aware)")
    print(f"~FLOPs/token:      {flops_per_token:,}")
    print(f"training_tokens:   {config.training_tokens:,}")
    print(f"~total train FLOPs: {total_flops:.3e}")


if __name__ == "__main__":
    main()
