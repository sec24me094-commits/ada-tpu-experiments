# Evaluation

Two layers of evaluation:

1. **ADA-specific metrics** (perplexity, routing analytics) —
   `scripts/evaluate.py` at the repo root. Not covered by the harness below.
2. **Standard zero-shot downstream benchmarks** (LAMBADA, HellaSwag,
   ARC-Easy, ARC-Challenge, WinoGrande) — via
   [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
   configured in `harness_config.yaml`.

## Running the harness

Status: not yet wired up — `ADAModel.from_pretrained` isn't implemented
until a checkpoint exists (see `ada/model.py`). Once it is, the harness
integration will need a thin `lm_eval.api.model.LM` subclass wrapping
`ADAModel` (harness convention); add it under this directory as
`ada_lm_eval_wrapper.py` rather than modifying `ada/model.py` directly.

```bash
# once the wrapper exists:
lm_eval --model ada_wrapper \
    --model_args checkpoint=checkpoints/ada-nano/latest.pt,config=configs/ada_nano.yaml \
    --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,winogrande \
    --output_path eval/results/
```

## Custom benchmarks

`custom_benchmarks/` is a placeholder for any evaluation set specific to
this project (referenced as `[FILL: any custom eval sets from your paper]`
in the research proposal) — none exist yet.
