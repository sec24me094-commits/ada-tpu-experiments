# Phase 2 — Expected Outputs

Status: Target ablation metrics and FLOP profiles computed; answers RQ3 and RQ6.

## Ablation Matrix & Pre-Flight Projections

All variants are trained across 5.0B tokens on FineWeb-Edu under identical training hyperparameter schedules:

| Variant | Target Perplexity | Total Training FLOPs | Activated Params | Est. Peak Memory (BF16) | Throughput Target (tok/s/core) |
|---|---|---|---|---|---|
| **Full ADA** | **23.8 – 24.5** | $3.75 \times 10^{18}$ | 125.1M | ~2.4 GB | ~14,200 |
| **No SSM** | 26.2 – 27.0 | $4.74 \times 10^{18}$ | 158.0M | ~2.9 GB | ~11,800 |
| **No MoE** | 27.5 – 28.3 | $3.75 \times 10^{18}$ | 125.0M | ~1.8 GB | ~15,600 |
| **Fixed Attention** | 24.8 – 25.5 | $5.45 \times 10^{18}$ | 181.7M | ~3.1 GB | ~10,400 |
| **Fixed Depth** | 24.2 – 24.9 | $5.45 \times 10^{18}$ | 181.7M | ~3.1 GB | ~10,200 |
| **Transformer Baseline**| 25.0 – 25.8 | $3.04 \times 10^{18}$ | 101.3M | ~1.6 GB | ~16,800 |

### Hypotheses to Validate (RQ3 & RQ6)
1. **SSM Impact (Full ADA vs No SSM)**: Removing Mamba-2 SSD degrades long-range perplexity by $+2.0$ points while requiring $+26\%$ more FLOPs due to quadratic attention reliance.
2. **MoE Specialization (Full ADA vs No MoE)**: MoE sparse routing provides the highest parameter-efficiency gain ($>3.0$ PPL improvement at matched FLOPs).
3. **Adaptive Depth Efficiency (Full ADA vs Fixed Depth)**: Dynamic token routing reduces effective layer traversal by $25\%$ ($\tau = 0.75$), achieving equivalent or superior perplexity with $31\%$ lower training FLOPs ($3.75 \times 10^{18}$ vs $5.45 \times 10^{18}$).
