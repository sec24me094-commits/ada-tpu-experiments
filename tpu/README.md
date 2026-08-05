# TPU Setup & Training

Status: **not started**. This directory is scaffolding for Phase 3 of the
roadmap ("TPU Adaptation & Pre-Flight", Week 1-2, see `../roadmap.md` /
project roadmap doc) — it documents what needs to happen once TRC grants
resource access, but none of it has been run against real TPU hardware yet.

## Setup

```bash
# On a fresh TPU VM (see setup_tpu.sh for the exact commands):
bash tpu/setup_tpu.sh
```

## Order of operations (per the roadmap's Week 1 checklist)

1. `pytorch_xla_adapter.py` — get `ADAModel` running a forward pass on TPU
   under PyTorch/XLA without XLA errors (`ADA-Nano` scale first).
2. Eliminate dynamic shapes / retracing — pad sequences to fixed lengths,
   audit any `.item()`/`.nonzero()`/data-dependent control flow (the
   `AdaptiveDepthController`'s hard-threshold branch and the top-k MoE
   dispatch in `ada/layers/moe.py` are the two most likely offenders — both
   currently use boolean masking that is fine for training with soft depth
   gating, but must be replaced with static-shape gather/scatter before
   `inference=True` hard-exit paths run under XLA).
3. Add `xm.mark_step()` calls through the training loop (`distributed_train.py`).
4. Enable BF16 training.
5. Build the GCS streaming data pipeline.
6. Run the 100-step ADA-Nano pre-flight (loss decreasing, routing stats
   logging, no OOM — success criteria in the roadmap).

## Files

| File | Purpose | Status |
|---|---|---|
| `setup_tpu.sh` | TPU VM environment setup | stub |
| `pytorch_xla_adapter.py` | PyTorch/XLA compatibility layer for `ADAModel` | stub |
| `distributed_train.py` | Multi-host TPU training entry point | stub |
| `tpu_profiling.md` | XLA profiler usage notes | stub |

None of this has been tested on real TPU hardware — there is no TPU in this
development environment. Treat every file here as a documented starting
point, not a verified implementation.
