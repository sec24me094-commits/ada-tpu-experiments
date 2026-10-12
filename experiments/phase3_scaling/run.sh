#!/usr/bin/env bash
# Phase 3: Scaling Study — ADA-Small / ADA-Base vs matched Transformer baselines.
# STATUS: smoke test only (--synthetic).
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/train.py --config configs/ada_small.yaml \
    --output_dir checkpoints/phase3/ada-small --synthetic --max_steps 50

python scripts/train.py --config configs/ada_base.yaml \
    --output_dir checkpoints/phase3/ada-base --synthetic --max_steps 50
