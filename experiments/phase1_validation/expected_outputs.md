# Phase 1 — Expected Outputs

Status: not yet run against real data. Success criteria, from the roadmap:

- ADA-Nano achieves stable training loss convergence.
- Perplexity within [FILL FROM PAPER: expected range] of the Transformer
  baseline.
- Routing collapse does not occur (>95% of tokens exiting at depth <= 2 is
  a failure condition).
- Utilization entropy > 0.8 * log(num_experts) at step 100.
- No XLA "graph too large" warnings after step 3 (once run on TPU).

Populate this file with actual loss curves / final numbers once Phase 1
runs against real hardware and data.
