# TPU Profiling Notes

Status: not started — no TPU access yet. Placeholder for how throughput
baselines (roadmap Phase 3, "Throughput baseline measurement", Week 2) will
be captured once TPU access exists.

## Planned approach

1. Wrap a training step range with the XLA profiler:
   ```python
   import torch_xla.debug.profiler as xp
   server = xp.start_server(9012)
   # ... run a handful of training steps ...
   ```
2. Capture a trace with `xplane`/TensorBoard's `capture_profile` and inspect
   in TensorBoard's `profile` tab for:
   - step time breakdown (compile vs. execute)
   - any repeated "graph too large" or retracing warnings (roadmap success
     criterion: none after step 3)
   - MXU/HBM utilization per op — expect Mamba-2 SSD's scan and MoE's expert
     dispatch to dominate; compare against the dense-Transformer baseline.
3. Record tokens/sec/device at ADA-Nano scale as the throughput baseline
   deliverable.

## Known risk (from the project risk register)

"XLA dynamic shape retracing — Medium probability / Medium impact —
mitigation: sequence padding + shape audit before Phase 1 launch." The main
suspects for dynamic shapes in the current implementation are flagged in
`tpu/README.md`'s step 2.
