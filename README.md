# ADA: Adaptive Depth Architecture for TPU Experiments

Adaptive Depth Architecture (ADA) is a research-oriented language model implementation designed around a hybrid stack of:

- Mamba-2 state-space modeling for long-context sequence mixing
- sparse Mixture-of-Experts (MoE) routing for conditional compute
- dynamic sparse grouped-query attention
- adaptive per-token depth control

The project is structured as a TPU-focused experimental codebase for exploring efficient large-model training and routing behavior in a Cloud TPU environment.

## Project status

This repository is a research prototype and experimental scaffold. The core model architecture and training harness are implemented, but several TPU-specific and data pipeline components remain under active development.

Current status highlights:

- Architecture and training loop are in place
- Synthetic training works as a smoke test
- Real dataset pipeline is not yet complete
- Real TPU execution is not yet validated on hardware
- Model checkpoint export/loading is not fully implemented yet

This repo is best understood as an implementation and research platform for the ADA roadmap, not yet as a fully productionized training stack.

## Why ADA?

Transformer attention has quadratic scaling in sequence length, which becomes costly for long-context training and inference. ADA combines the efficiency of Mamba-2 recurrence with the conditional computation of MoE routing and sparse attention masks to pursue a more scalable hybrid architecture.

The design is intended to explore:

- sub-quadratic sequence mixing with SSMs
- sparse expert activation for improved compute efficiency
- adaptive depth gating to reduce unnecessary compute
- TPU-friendly execution patterns and XLA compatibility

## Architecture overview

The model follows the pattern:

```text
input tokens -> embedding -> N x ADAHybridBlock -> final norm -> language-model head
```

Each hybrid block contains:

1. Mamba-2 SSD layer
2. MoE layer
3. dynamic sparse GQA attention layer
4. adaptive depth controller

The repo also includes ablation configurations to evaluate the effect of removing or replacing individual pieces of the architecture.

See:

- `docs/architecture.md` for the architecture design and known limitations
- `ada/model.py` for the model forward pass
- `ada/layers/` for the block implementations
- `configs/` for concrete experiment configuration files

## Repository layout

```text
ada-tpu-experiments/
├── ada/                     # core model implementation
│   ├── layers/              # Mamba-2, MoE, attention, depth-routing components
│   ├── routing/             # routing and mask logic
│   ├── training/            # training logic and objectives
│   ├── utils/               # checkpointing, logging, profiling
│   ├── config.py            # model config
│   └── model.py             # ADAModel entry point
├── configs/                 # YAML experiment configs
├── data/                    # dataset card and preprocessing pipeline stubs
├── docs/                    # architecture and training guidance docs
├── eval/                    # evaluation harness and benchmark layout
├── experiments/             # phased experiment scripts and expected outputs
├── notebooks/               # architecture and routing notebooks
├── scripts/                 # training, evaluation, export, profiling scripts
├── tests/                   # unit tests for key components
├── tpu/                     # TPU setup and XLA adaptation scaffolding
├── LICENSE
├── README.md
├── pyproject.toml
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CITATION.cff
└── .github/
```

## Installation

This project targets Python 3.11+ and installs as a standard package.

```bash
git clone https://github.com/sec24me094-commits/ada-tpu-experiments.git
cd ada-tpu-experiments
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional extras:

```bash
pip install -e .[tpu]
pip install -e .[gpu]
pip install -e .[dev]
```

## Quick start

The repository currently supports a synthetic smoke-test training path while real data and TPU execution are being prepared.

```bash
python scripts/train.py \
  --config configs/ada_nano.yaml \
  --output_dir checkpoints/ada-nano \
  --synthetic
```

This launches the training harness with random data to validate the flow end-to-end without a full dataset.

## Training and evaluation

The training entry point expects either:

- a real packed dataset via `--data_path`, or
- `--synthetic` for smoke tests

Example:

```bash
python scripts/train.py \
  --config configs/ada_nano.yaml \
  --data_path /path/to/tokenized/dataset \
  --output_dir checkpoints/ada-nano
```

The repo also includes:

- `scripts/evaluate.py` for model-specific evaluation metrics
- `eval/harness_config.yaml` for downstream benchmark configuration
- `experiments/` for phase-based validation and ablation workflows

## TPU roadmap

The TPU work is intentionally documented as staged and incomplete. The `tpu/` directory includes setup and adaptation scaffolding for Cloud TPU workflows, but it is not yet a verified runtime deployment.

The planned progression includes:

1. XLA/PyTorch compatibility for `ADAModel`
2. static-shape adaptation and tracing cleanup
3. BF16 and distributed training setup
4. GCS-backed streaming data pipeline
5. TPU pre-flight and scaling validation

See:

- `tpu/README.md`
- `docs/training_guide.md`
- `docs/inference_guide.md`

## Data pipeline

The repository includes a preprocessing pipeline intended for real training data, but that pipeline is not yet active against real dataset material.

Expected order:

```bash
python data/preprocess/filter.py --input <raw> --output <filtered>
python data/preprocess/deduplicate.py --input <filtered> --output <deduped>
python data/preprocess/tokenize.py --input <deduped> --output <tokenized> --tokenizer <name_or_path>
python data/preprocess/pack.py --input <tokenized> --output <packed> --seq_len 2048
```

Documentation for the dataset card and current pipeline expectations lives under:

- `data/README.md`
- `data/DATASET_CARD.md`

## Experiments

The repo includes a staged experiment plan for benchmarking ADA against baselines.

- `experiments/phase1_validation` — architecture validation
- `experiments/phase2_ablation` — ablation matrix
- `experiments/phase3_scaling` — scaling study

These are scaffolded and designed to become real benchmark drivers once data and TPU access are ready.

## Development and contribution

Contributions are welcome in the form of architecture improvements, TPU integration work, evaluation additions, bug fixes, and experiment validation.

Please review:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- repository workflow files in `.github/workflows/`

## License

This project is available under the Apache 2.0 License. See the `LICENSE` file for details.

## Citation

The repository includes `CITATION.cff` for attribution guidance.

## Summary

ADA is a research-first hybrid language model codebase that brings together Mamba-2, MoE, sparse attention, and adaptive depth in a TPU-oriented design. It is a strong foundation for experimentation and iteration, but it is still explicitly in the prototype and roadmap stage rather than a fully validated production model stack.
