# Adaptive Depth Architecture (ADA)

<p align="center">
  <a href="https://github.com/sec24me094-commits/ada-tpu-experiments/actions"><img src="https://github.com/sec24me094-commits/ada-tpu-experiments/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/></a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776ab.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"/>
</p>

ADA (Adaptive Depth Architecture) is a hybrid language model architecture
that combines Mamba-2 State Space Duality, Mixture-of-Experts sparse
routing, and Dynamic Sparse Grouped Query Attention under a unified
adaptive depth controller. ADA dynamically allocates computation per
token — routing simple tokens through fewer layers and complex tokens
through more — with the goal of better perplexity per training FLOP than
fixed-depth Transformer baselines.

**Status: v0.1 scaffold.** The architecture is implemented in PyTorch and
passes unit tests (forward pass, backward pass, all ablation-variant
combinations) on CPU with small configs. It has **not** been trained at
any real scale, adapted for TPU/XLA, or validated experimentally — that is
the work this repository exists to do next. See
[Known Scope Limitations](#known-scope-limitations) before assuming any
number below is a real result.

## Table of Contents
- [Why ADA](#why-ada)
- [Architecture](#architecture)
- [Known Scope Limitations](#known-scope-limitations)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Experiments](#experiments)
- [Repository Layout](#repository-layout)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Why ADA

Standard language models apply identical compute to every token in every
layer. This is an allocation inefficiency: "the" does not require the same
reasoning depth as a rare technical term or a long-range coreference.

| Problem | Solution | ADA Component |
|---|---|---|
| O(n²) global attention cost | Selective state spaces | Mamba-2 SSD |
| Fixed parameter activation | Sparse expert routing | MoE |
| Uniform local attention compute | Input-dependent sparsity | Dynamic Sparse GQA |
| Fixed processing depth per token | Per-token routing | Adaptive Depth Controller |

ADA integrates all four into a single architecture. The central research
question — whether combining them produces efficiency advantages beyond
the sum of the parts — is what the experiments in this repo are designed
to answer, once TPU resources are available (see the TRC application
materials this repo was scaffolded from).

## Architecture

An ADA model is `embedding -> N x ADA Block -> output head`. Each block
runs three sublayers in sequence — Mamba-2 SSD, MoE, Dynamic Sparse GQA —
followed by an Adaptive Depth Controller that decides whether each token
continues to the next block or exits early. See
[`docs/architecture.md`](docs/architecture.md) for the full writeup,
including the training objective and the ablation matrix.

```
L_total = L_LM + lambda_depth * L_depth + lambda_balance * L_balance
```

## Known Scope Limitations

This is a first, CPU-testable reference implementation, not a
performance-tuned one. Specifically:

- **Mamba-2 SSD** (`ada/layers/mamba2_ssd.py`) uses a sequential Python-loop
  scan — correct, but not the chunked/parallel-scan formulation Mamba-2
  needs for real training throughput.
- **MoE** (`ada/layers/moe.py`) is dense-compute (every expert runs on
  every token, then masked) rather than true sparse dispatch — correct,
  but doesn't reflect the per-device memory profile real MoE training
  needs.
- **Adaptive depth** computes correct per-token routing decisions, but
  does not yet skip compute for exited tokens at inference — see
  `ada/model.py`'s module docstring.
- **TPU/XLA adaptation** (`tpu/`) is documented but untested — there's no
  TPU in the environment this was built in.
- **Real data loading** isn't wired up — `scripts/train.py` only supports
  `--synthetic` (random token) training right now; `data/preprocess/`
  has the intended tokenize → filter → dedupe → pack pipeline, but no
  dataset has been run through it.

None of this blocks development — CI runs the real test suite against
these modules — but don't treat a `--synthetic` training run or the
reference config sizes as validated results.

## Installation

### Requirements
- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (GPU) or PyTorch/XLA (TPU)

### From Source
```bash
git clone https://github.com/sec24me094-commits/ada-tpu-experiments
cd ada-tpu-experiments
pip install -e .
```

### With Development Dependencies
```bash
pip install -e ".[dev]"
```

### For TPU
```bash
pip install -e ".[tpu]"
# See tpu/README.md for TPU VM setup
```

## Quick Start

```python
from ada import ADAModel, ADAConfig

config = ADAConfig.from_yaml("configs/ada_nano.yaml")
model = ADAModel(config)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

import torch
input_ids = torch.randint(0, config.vocab_size, (2, 512))  # batch=2, seq_len=512
output = model(input_ids, return_routing_stats=True)

print(output.logits.shape)  # (2, 512, vocab_size)
print(f"Mean routing depth: {output.depth_stats['mean_depth']:.2f} / {config.num_layers}")
```

Check any config's actual parameter count and estimated FLOPs (safe to run
on any machine — builds on torch's `meta` device, no real memory
allocated) with:
```bash
python scripts/profile_flops.py --config configs/ada_nano.yaml
```

## Training

### Smoke test (no real data required)
```bash
python scripts/train.py --config configs/ada_nano.yaml \
    --output_dir checkpoints/ada-nano --synthetic --max_steps 20
```

### Single-device / multi-GPU / TPU
See [`docs/training_guide.md`](docs/training_guide.md) for the full set of
entry points (`scripts/train.py`, `torchrun`, `tpu/distributed_train.py`)
and [`docs/hyperparameter_guide.md`](docs/hyperparameter_guide.md) for what
every config field means.

## Inference

`ADAModel.from_pretrained` isn't implemented yet — no checkpoint exists.
See [`docs/inference_guide.md`](docs/inference_guide.md) for the intended
API and current status once one does.

## Experiments

Reproduction scripts for all three research phases (architecture
validation, ablation study, scaling study) live in `experiments/` — see
[`experiments/README.md`](experiments/README.md). All are currently smoke
tests against synthetic data pending TPU resource allocation.

## Repository Layout

```
ada-tpu-experiments/
├── ada/            # Core library (import ada)
├── scripts/        # Training, eval, export entry points
├── configs/        # YAML configs for all model sizes + ablations
├── experiments/    # Reproducible experiment scripts
├── tpu/            # TPU-specific setup and training (untested)
├── data/           # Dataset preprocessing pipeline
├── eval/           # Evaluation harness configuration
├── docs/           # Architecture docs and guides
├── tests/          # Unit and integration tests
└── notebooks/      # Interactive walkthroughs (placeholders)
```

## Roadmap

### v0.1 — Current
- [x] Core ADA architecture implementation (Mamba-2 SSD + MoE + Dynamic
      Sparse GQA + Adaptive Depth), CPU-testable
- [x] PyTorch training loop with routing analytics
- [x] Unit test suite (18 tests, all ablation combinations)
- [ ] Real dataset loading
- [ ] TPU/XLA compatibility (PyTorch/XLA adapter) — pending TRC allocation
- [ ] ADA-Nano validation run
- [ ] Ablation study (6 variants)
- [ ] ADA-Small and ADA-Base scaling study

### Long-term
- [ ] JAX/Flax implementation for native TPU performance
- [ ] Chunked/parallel-scan Mamba-2 SSD kernel
- [ ] Inference-time compute-skipping for adaptive depth
- [ ] Extended context length with hierarchical SSM routing

## Citation

If you use ADA in your research, see [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

## Acknowledgements

This research is conducted independently at Sri Sairam Engineering
College, Chennai, India. Compute resources for experimental validation
requested from Google TPU Research Cloud (TRC).

ADA builds on ideas from:
- Mamba-2 — Dao & Gu (2024)
- EleutherAI lm-evaluation-harness — evaluation framework
- HuggingFace Transformers — model hub integration
