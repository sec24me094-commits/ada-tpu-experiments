"""ADAConfig: all hyperparameters for an ADA model in one place.

Configs are stored as YAML in configs/*.yaml and loaded via
`ADAConfig.from_yaml(path)`. Every architectural or training knob referenced
in the ADA research proposal / roadmap should have a field here so a run is
fully reproducible from its config file alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ADAConfig:
    # --- Identity ---
    name: str = "ada-nano"

    # --- Core sizing ---
    vocab_size: int = 32000
    hidden_dim: int = 768
    num_layers: int = 12
    max_seq_len: int = 2048

    # --- Mamba-2 SSD sublayer ---
    use_ssm: bool = True
    ssm_state_dim: int = 128
    ssm_num_heads: int = 8
    ssm_head_dim: int = 96  # hidden_dim / ssm_num_heads by convention
    ssm_conv_kernel: int = 4

    # --- MoE sublayer ---
    use_moe: bool = True
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_expert_hidden_mult: float = 4.0  # expert FFN hidden = hidden_dim * mult
    lambda_balance: float = 0.01

    # --- Dynamic Sparse GQA sublayer ---
    use_dynamic_sparse_attn: bool = True  # False => fixed (dense) GQA, used by the "fixed_attn" ablation
    num_query_heads: int = 12
    num_kv_heads: int = 4
    attn_sparsity_target: float = 0.5  # fraction of positions masked out, on average

    # --- Adaptive depth controller ---
    use_adaptive_depth: bool = True  # False => fixed depth (all layers, every token), the "fixed_depth" ablation
    depth_controller_hidden: int = 128
    depth_budget_tau: float = 0.75  # target average fraction of layers used
    lambda_depth: float = 0.01

    # --- Regularization / init ---
    dropout: float = 0.0
    initializer_range: float = 0.02

    # --- Training ---
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_steps: int = 500
    batch_size_tokens: int = 262_144  # tokens per optimizer step
    micro_batch_size: int = 8
    training_tokens: int = 5_000_000_000
    grad_clip: float = 1.0
    seed: int = 0

    # --- Bookkeeping / extras (free-form, not read by the model) ---
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- Derived / convenience ----
    @property
    def expert_hidden_dim(self) -> int:
        return int(self.hidden_dim * self.moe_expert_hidden_mult)

    def validate(self) -> None:
        if self.hidden_dim % self.num_query_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_query_heads ({self.num_query_heads})"
            )
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for GQA grouping"
            )
        if self.use_moe and self.moe_top_k > self.moe_num_experts:
            raise ValueError("moe_top_k cannot exceed moe_num_experts")

    # ---- IO ----
    @classmethod
    def from_yaml(cls, path: str | Path) -> ADAConfig:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        extra = {k: v for k, v in raw.items() if k not in cls.__dataclass_fields__}
        if extra:
            known["extra"] = extra
        cfg = cls(**known)
        cfg.validate()
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
