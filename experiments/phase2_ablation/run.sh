#!/usr/bin/env bash
# Phase 2: Ablation Study — 6-variant matrix.
# STATUS: smoke test only (--synthetic).
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/ablation_launcher.py \
    --config_dir configs/ \
    --output_root checkpoints/phase2_ablation \
    --extra_train_args "--synthetic --max_steps 100"
