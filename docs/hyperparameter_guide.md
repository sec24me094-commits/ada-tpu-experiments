# Hyperparameter Guide

Reference: `ada/config.py` (`ADAConfig`). Every field below maps 1:1 to a
YAML key in `configs/*.yaml`.

## Sizing
| Field | Meaning |
|---|---|
| `hidden_dim` | Model width |
| `num_layers` | Number of `ADAHybridBlock`s |
| `vocab_size` | Tokenizer vocabulary size — must match the tokenizer used in `data/preprocess/tokenize.py` |
| `max_seq_len` | Maximum training/inference sequence length |

## Mamba-2 SSD
| Field | Meaning |
|---|---|
| `ssm_state_dim` | SSM hidden state dimension per head |
| `ssm_num_heads` | Number of SSM heads (`hidden_dim` should be divisible by this) |
| `ssm_conv_kernel` | Depthwise causal conv kernel size before the scan |

## MoE
| Field | Meaning |
|---|---|
| `moe_num_experts` | Total experts per MoE layer |
| `moe_top_k` | Experts activated per token |
| `moe_expert_hidden_mult` | Expert FFN hidden size = `hidden_dim * mult` |
| `lambda_balance` | Weight on the load-balance auxiliary loss |

## Dynamic Sparse GQA
| Field | Meaning |
|---|---|
| `num_query_heads` / `num_kv_heads` | GQA head config — `num_query_heads` must be divisible by `num_kv_heads` |
| `attn_sparsity_target` | Target fraction of causal-visible positions *masked out* on average |
| `use_dynamic_sparse_attn` | `False` -> dense causal attention (the "fixed_attn" ablation) |

## Adaptive Depth
| Field | Meaning |
|---|---|
| `depth_controller_hidden` | Hidden width of the 2-layer gating MLP |
| `depth_budget_tau` | Target average fraction of layers used |
| `lambda_depth` | Weight on the depth-budget regularizer |
| `use_adaptive_depth` | `False` -> fixed depth, every token uses every block (the "fixed_depth" ablation) |

## Optimization
| Field | Meaning |
|---|---|
| `learning_rate` / `min_learning_rate` | Peak LR / cosine-decay floor |
| `warmup_steps` | Linear warmup steps before cosine decay |
| `weight_decay`, `adam_beta1`, `adam_beta2` | AdamW settings |
| `batch_size_tokens` | Tokens per optimizer step (global, across grad accumulation / data parallelism) |
| `micro_batch_size` | Sequences per forward pass on one device |
| `training_tokens` | Total training token budget — determines `max_steps = training_tokens // batch_size_tokens` |
| `grad_clip` | Gradient norm clipping threshold |

## Tuning workflow

1. Pick `hidden_dim` / `num_layers` for a target parameter count.
2. `python scripts/profile_flops.py --config configs/your_config.yaml` to
   check actual total/activated params and estimated FLOPs.
3. Adjust and repeat until within range of the target (ADA-Nano ~100M,
   ADA-Small ~300M, ADA-Base ~1B, per the roadmap).
