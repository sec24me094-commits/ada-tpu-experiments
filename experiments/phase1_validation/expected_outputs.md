# Phase 1 — Expected Outputs

Status: Target benchmarks defined; empirical validation scheduled on Cloud TPU v4-8.

## Target Success Criteria
- **Loss Convergence**: ADA-Nano achieves stable monotonic training loss decrease across 5B tokens ($\mathcal{L}_{\text{total}} < 2.85$).
- **Perplexity Target**: Validation perplexity $\le 24.5$ on FineWeb-Edu 100M-token validation split, matching or improving upon the compute-matched Transformer-Nano baseline ($25.2 \pm 0.3$) while activating 57% fewer parameters per forward pass (125.1M vs 295.0M).
- **Depth Routing Distribution**: Balanced depth usage with mean exit depth $\bar{d} \approx 9.0 \pm 1.0$ layers (budget $\tau = 0.75 \times 12$). Routing collapse avoidance: $< 5\%$ of tokens exiting at depth $\le 2$ on non-trivial sequence positions.
- **MoE Load Balancing**: Expert utilization entropy $H > 1.66$ ($0.8 \times \ln(8)$) across all 8 experts by step 100, confirming uniform expert dispatch.
- **XLA Compilation**: 0 graph recompilations and 0 "graph too large" XLA warnings after step 3 on TPU v4-8.

Upon completion of Phase 1 runs on TRC hardware, replace this file with the generated WandB training curves and checkpoint evaluation logs.
