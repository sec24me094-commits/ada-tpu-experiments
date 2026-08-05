# Inference Guide

## Status

`ADAModel.from_pretrained` is **not implemented** — no checkpoints exist yet
(see `ada/model.py`; it raises `NotImplementedError` with a pointer to
`scripts/export_to_hf.py`, which documents the intended export format).

## Once a checkpoint exists

```python
from ada import ADAConfig, ADAModel
from ada.utils.checkpointing import load_checkpoint

config = ADAConfig.from_yaml("configs/ada_nano.yaml")
model = ADAModel(config)
load_checkpoint(model, optimizer=None, path="checkpoints/ada-nano/latest.pt")
model.eval()

import torch
input_ids = torch.randint(0, config.vocab_size, (1, 32))
output = model(input_ids, inference=True, return_routing_stats=True)
print(output.depth_stats["mean_depth"], "/", config.num_layers)
```

## Routing analysis at inference

`output.depth_stats` (when `return_routing_stats=True`) contains:
- `per_token_depth`: `(batch, seq_len)` — exit depth per token
- `mean_depth`: scalar
- `expert_utilization` / `expert_entropy`: MoE routing stats
- `sparse_mask_density`: mean Dynamic Sparse GQA mask density

## Generation (`model.generate(...)`)

Not implemented in this scaffold — the README's Quick Start snippet
(`model.generate(input_ids, max_new_tokens=50)`) documents the intended API,
but autoregressive decoding with KV-caching across the SSM's recurrent
state + attention's KV cache + adaptive depth's per-token exit is
nontrivial to get right and is left as a dedicated follow-up once training
produces a checkpoint worth generating from.

## Important caveat on hard early-exit

Setting `inference=True` computes hard depth-routing *decisions* correctly,
but does not yet skip compute for exited tokens — every token still runs
through every block. This means `mean_depth` and `per_token_depth` are
trustworthy for *analysis* (RQ2 in the research proposal), but inference
latency will not yet reflect the efficiency the architecture is designed
for. See `ada/model.py`'s module docstring and `tpu/pytorch_xla_adapter.py`
for the plan.
