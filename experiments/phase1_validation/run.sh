#!/usr/bin/env bash
# Phase 1: Architecture Validation — ADA-Nano vs Transformer-Nano baseline.
# STATUS: smoke test only (--synthetic). Swap in --data_path once a real
# tokenized/packed dataset exists (see data/README.md).
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/train.py --config configs/ada_nano.yaml \
    --output_dir checkpoints/phase1/ada-nano --synthetic --max_steps 100

python scripts/train.py --config configs/transformer_nano_baseline.yaml \
    --output_dir checkpoints/phase1/transformer-nano --synthetic --max_steps 100

python scripts/evaluate.py --config configs/ada_nano.yaml \
    --checkpoint checkpoints/phase1/ada-nano/latest.pt --synthetic
