# Changelog

All notable changes to this project are documented in this file.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Initial repository scaffold: `ada/` core library layout, `configs/`,
  `scripts/`, `experiments/`, `tpu/`, `data/`, `eval/`, `docs/`, `tests/`.
- `ADAConfig` dataclass and YAML loading.
- Skeleton implementations of Mamba-2 SSD, MoE, Dynamic Sparse GQA, and the
  Adaptive Depth Controller layers.
- `ADAModel` assembling the hybrid blocks end-to-end with a working forward
  pass on random input (CPU-testable).
- Training loss (`L_LM + lambda_depth * L_depth + lambda_balance * L_balance`)
  and a minimal training loop.
- Reference configs for ADA-Nano / ADA-Small / ADA-Base and the ablation
  matrix (no-SSM, no-MoE, fixed-attention, fixed-depth) plus a matched
  Transformer baseline config.
- CI workflow (lint + unit tests), issue templates, PR template.

### Status
- v4.5 architecture: implemented, CPU-testable, not yet TPU-adapted.
- TPU/XLA adaptation: not started (see `tpu/README.md`).
- Experimental validation: not started — pending TRC resource allocation.

## [0.1.0] - 2026-08-04
- Repository created from the ADA project design documents.
