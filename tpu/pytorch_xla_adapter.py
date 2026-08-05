"""PyTorch/XLA compatibility layer for ADAModel.

STATUS: stub / untested — there is no TPU in this development environment,
so nothing here has been run against real XLA hardware. This module exists
to document the intended shape of the adaptation, per roadmap Phase 3 Week 1:
"ADA-Nano forward pass completes on TPU v4-8 without XLA errors."

Known trouble spots to check first (see tpu/README.md for the full list):
  - `ada/layers/mamba2_ssd.py`'s `_scan` uses a Python `for t in range(l)`
    loop. Under `torch.compile`/XLA this will unroll into `l` ops, which is
    fine for tracing correctness at small seq_len but will be slow and
    memory-heavy; the roadmap's own risk register flags this as needing
    "chunked/parallel scan formulation compiled under XLA" (see that file's
    module docstring).
  - `ada/layers/moe.py`'s expert loop (`for e in range(self.num_experts)`)
    with boolean masking is a data-independent shape (good for XLA) but
    wastes compute (every expert runs on every token). Fine for a first
    pre-flight pass; revisit once throughput is measured.
  - `ada/layers/adaptive_depth.py` + `ADAModel.forward(..., inference=True)`
    currently computes hard masks but does not skip compute — this keeps
    shapes fully static (good for XLA) at the cost of not yet realizing the
    inference speedup. Do not "fix" this by making shapes data-dependent
    without first confirming XLA can trace it; prefer a fixed-capacity
    gather (pad/truncate to a static max per-block token count) if/when
    real compute-skipping is implemented.
"""

from __future__ import annotations

from torch import nn

from ada.config import ADAConfig
from ada.model import ADAModel


def get_xla_device():
    """Returns the XLA device, raising a clear error if torch_xla isn't
    installed (e.g. when developing on CPU/GPU without a TPU)."""
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as e:
        raise ImportError(
            "torch_xla is not installed. Install the `tpu` extra "
            "(`pip install -e '.[tpu]'`) on a TPU VM — see tpu/setup_tpu.sh."
        ) from e
    return xm.xla_device()


def build_model_for_xla(config: ADAConfig) -> nn.Module:
    """Construct an ADAModel and move it to the XLA device.

    Intentionally thin right now — as XLA-specific issues are found (see
    module docstring), this is the place to patch them in (e.g. swapping
    Mamba2SSD's `_scan` for a chunked implementation) without touching the
    CPU/GPU-facing `ada/` package.
    """
    device = get_xla_device()
    model = ADAModel(config)
    return model.to(device)


def mark_step():
    """Thin wrapper around `xm.mark_step()` so callers don't need a direct
    torch_xla import (keeps ada/training/trainer.py TPU-agnostic; the
    TPU-specific loop in distributed_train.py calls this after each step)."""
    import torch_xla.core.xla_model as xm

    xm.mark_step()
