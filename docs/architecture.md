# ADA Architecture

## Overview

An ADA model is `embed_tokens -> N x ADAHybridBlock -> final_norm -> lm_head`
(see `ada/model.py`). Each `ADAHybridBlock` (`ada/layers/hybrid_block.py`)
runs three sublayers in sequence, each wrapped in a pre-norm residual:

1. **Mamba-2 SSD** (`ada/layers/mamba2_ssd.py`) — selective state-space
   recurrence for global sequence context, O(n) in sequence length.
2. **MoE** (`ada/layers/moe.py`) — sparse top-k expert routing over N expert
   FFNs, with a load-balance auxiliary loss.
3. **Dynamic Sparse GQA** (`ada/layers/sparse_gqa.py`) — grouped-query
   attention with an input-dependent sparsity mask
   (`ada/routing/sparse_attention_mask.py`), so attention compute
   concentrates on positions estimated to matter.

After each block, the **Adaptive Depth Controller**
(`ada/layers/adaptive_depth.py`) — a 2-layer MLP — produces a per-token
continue probability. During training this is a soft, fully differentiable
residual weight; at inference it becomes a hard threshold. See
`ada/model.py`'s module docstring for the current status of hard-exit
compute skipping (routing decisions are computed correctly; actual
compute-skipping at inference is not yet wired up).

## Training objective

```
L_total = L_LM + lambda_depth * L_depth + lambda_balance * L_balance
```

- `L_LM`: standard next-token cross-entropy (`ada/training/loss.py`)
- `L_depth`: penalizes mean routing depth exceeding the target budget `tau`
  (`ada.layers.adaptive_depth.depth_budget_loss`)
- `L_balance`: MoE load-balance auxiliary loss, Switch-Transformer style
  (`ada.layers.moe.MoELayer._load_balance_loss`)

## Ablation variants

Each variant disables exactly one component via `ADAConfig` flags — see
`configs/ablation_*.yaml` and `configs/transformer_nano_baseline.yaml`:

| Variant | `use_ssm` | `use_moe` | `use_dynamic_sparse_attn` | `use_adaptive_depth` |
|---|---|---|---|---|
| Full ADA | True | True | True | True |
| No SSM | False | True | True | True |
| No MoE | True | False | True | True |
| Fixed Attention | True | True | False | True |
| Fixed Depth | True | True | True | False |
| Transformer Baseline | False | False | False | False |

## Known scope limitations (read before running real experiments)

- `Mamba2SSD._scan` is a sequential Python-loop scan — correct, but not the
  chunked/parallel-scan formulation Mamba-2 needs for real training
  throughput. See that file's module docstring.
- `MoELayer` is dense-compute (every expert runs on every token, then
  masked) rather than true sparse dispatch — correct, but does not reflect
  the per-device memory profile discussed in the TRC proposal. See that
  file's module docstring.
- Inference-time hard early-exit does not yet skip compute (see above).

These are exactly the gaps Phase 3 ("TPU Adaptation & Pre-Flight") is meant
to close — see `tpu/README.md`.

## Diagrams

`docs/assets/*.png` referenced from the top-level README (architecture
block diagram, single-block internals, depth-routing visualization, MoE
routing visualization) have not been created yet — see
`03_github_repo_design`'s "Architecture Diagram Specifications" for what
each should show. Until they exist, the README's `<img>` tags will render
as broken links; either create the diagrams or remove those tags before
making the repo public (see the release checklist in the repo design doc).
