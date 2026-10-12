#!/usr/bin/env python
"""Evaluate a checkpoint: perplexity on a held-out set + routing analytics.

Usage:
    python scripts/evaluate.py --config configs/ada_nano.yaml \
        --checkpoint checkpoints/ada-nano/latest.pt --data_path /path/to/val

Downstream zero-shot tasks (LAMBADA, HellaSwag, ARC, WinoGrande) are handled
via the lm-evaluation-harness integration — see eval/harness_config.yaml and
eval/README.md — not by this script, which only covers ADA-specific
perplexity + routing metrics that the harness doesn't know how to compute.
"""

from __future__ import annotations

import argparse
import math

import torch

from ada import ADAConfig, ADAModel
from ada.utils.checkpointing import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    config = ADAConfig.from_yaml(args.config)
    model = ADAModel(config).to(args.device)
    load_checkpoint(model, None, args.checkpoint)
    model.eval()

    if args.synthetic or args.data_path is None:
        print("WARNING: no --data_path given; evaluating on random data as a smoke test only.")
        input_ids = torch.randint(0, config.vocab_size, (4, min(config.max_seq_len, 512)), device=args.device)
    else:
        raise NotImplementedError("Wire up real held-out data loading — see data/README.md.")

    with torch.no_grad():
        output = model(input_ids, inference=True, return_routing_stats=True)
        logits = output.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        ppl = math.exp(loss.item())

    print(f"loss={loss.item():.4f} perplexity={ppl:.2f}")
    print(f"mean_depth={output.depth_stats.get('mean_depth'):.2f} / {config.num_layers}")
    if "expert_entropy" in output.depth_stats:
        print(f"expert_utilization_entropy={output.depth_stats['expert_entropy']:.3f} "
              f"(target > {0.8 * math.log(config.moe_num_experts):.3f})")
    if "sparse_mask_density" in output.depth_stats:
        print(f"sparse_mask_density={output.depth_stats['sparse_mask_density']:.3f}")


if __name__ == "__main__":
    main()
