#!/usr/bin/env python
"""Launch the full 6-variant ablation matrix (Phase 2 in the roadmap) as
sequential (or, with --parallel, background-parallel) subprocess calls to
scripts/train.py, one per config in configs/ablation_*.yaml plus
transformer_nano_baseline.yaml and ada_nano.yaml (the "Full ADA" row).

Usage:
    python scripts/ablation_launcher.py --config_dir configs/ --output_root checkpoints/ablation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ABLATION_CONFIGS = [
    "ada_nano.yaml",                 # Full ADA
    "ablation_no_ssm.yaml",
    "ablation_no_moe.yaml",
    "ablation_fixed_attn.yaml",
    "ablation_fixed_depth.yaml",
    "transformer_nano_baseline.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--output_root", default="checkpoints/ablation")
    parser.add_argument("--extra_train_args", default="", help="Extra args forwarded to train.py, e.g. '--synthetic'")
    parser.add_argument("--parallel", action="store_true", help="Launch all variants as background processes")
    args = parser.parse_args()

    procs = []
    for cfg_name in ABLATION_CONFIGS:
        cfg_path = Path(args.config_dir) / cfg_name
        variant = cfg_path.stem
        out_dir = Path(args.output_root) / variant
        cmd = [
            sys.executable, "scripts/train.py",
            "--config", str(cfg_path),
            "--output_dir", str(out_dir),
        ] + args.extra_train_args.split()

        print("Launching:", " ".join(cmd))
        if args.parallel:
            procs.append(subprocess.Popen(cmd))
        else:
            subprocess.run(cmd, check=True)

    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
