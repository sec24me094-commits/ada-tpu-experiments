"""ADAModel: embedding -> N x ADAHybridBlock -> output head, with per-token
adaptive-depth early exit.

Training-time forward pass keeps depth gating soft (differentiable) so every
token passes through every block, weighted by its continue probability at
each exit — see `hard_depth=False` (the default). This matches the roadmap's
"soft rather than hard depth gating" design decision (docs/03_github_repo_design).

Inference-time hard early exit (tokens actually skip remaining blocks) is a
model-level control-flow optimization that is NOT implemented in this
reference forward pass yet — `forward(..., hard_depth=True)` computes hard
masks and reports depth stats, but still runs every block on every token, so
it is correct for measuring routing behavior (RQ2) but not yet a real
inference speedup. Wiring actual compute-skipping (gather/scatter by
continue_mask between blocks) is TPU/XLA-shape-sensitive and belongs in
Phase 3 (tpu/pytorch_xla_adapter.py) rather than this reference module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn

from ada.config import ADAConfig
from ada.layers.adaptive_depth import depth_budget_loss
from ada.layers.hybrid_block import ADAHybridBlock
from ada.routing.depth_router import expert_utilization_entropy, per_token_exit_depth


@dataclass
class ADAOutput:
    logits: torch.Tensor
    depth_stats: dict[str, Any] = field(default_factory=dict)
    balance_losses: list[torch.Tensor] = field(default_factory=list)
    depth_loss: torch.Tensor | None = None


class ADAModel(nn.Module):
    def __init__(self, config: ADAConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([ADAHybridBlock(config) for _ in range(config.num_layers)])

        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        # Weight-tie the output projection to the input embedding (standard
        # practice, e.g. GPT-2) — halves the embedding parameter cost, which
        # matters more than usual here since MoE already spends heavily on
        # expert params. Must be set up before self.apply(_init_weights)
        # below (also after both are constructed with matching shape).
        self.lm_head.weight = self.embed_tokens.weight

        # NOTE: the design docs describe a dedicated early-exit head that
        # projects a token to output space when it exits before the final
        # block. Since this reference forward pass doesn't yet skip compute
        # for exited tokens (see the module docstring), every token reaches
        # `final_norm` / `lm_head` regardless of its routing decision, so
        # there is no separate early-exit head parameter here yet — adding
        # one with no gradient path would just be dead weight. Add it back
        # here when real compute-skipping is implemented (tpu/ Phase 3).

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        inference: bool = False,
        return_routing_stats: bool = False,
    ) -> ADAOutput:
        """
        input_ids: (batch, seq_len) long tensor of token ids
        inference: if True, depth gating uses hard thresholds (see module docstring
                   for the caveat on this not yet skipping compute)
        """
        hidden_states = self.embed_dropout(self.embed_tokens(input_ids))

        continue_probs: list[torch.Tensor] = []
        continue_masks: list[torch.Tensor] = []
        balance_losses: list[torch.Tensor] = []

        for block in self.blocks:
            hidden_states, continue_prob, continue_mask, balance_loss = block(
                hidden_states, hard_depth=inference
            )
            continue_probs.append(continue_prob)
            continue_masks.append(continue_mask)
            if balance_loss is not None:
                balance_losses.append(balance_loss)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        depth_loss = None
        if self.config.use_adaptive_depth:
            depth_loss = depth_budget_loss(continue_probs, self.config.depth_budget_tau)

        depth_stats: dict[str, Any] = {}
        if return_routing_stats or inference:
            per_token_depth = per_token_exit_depth(continue_masks)
            depth_stats["per_token_depth"] = per_token_depth
            depth_stats["mean_depth"] = per_token_depth.float().mean().item()

            # nn.ModuleList.__iter__ is typed as returning plain nn.Module, so
            # cast to the concrete block type for attribute access below.
            typed_blocks = cast(list[ADAHybridBlock], list(self.blocks))

            utils: list[torch.Tensor] = []
            for b in typed_blocks:
                if b.use_moe and b.moe.last_expert_utilization is not None:
                    utils.append(b.moe.last_expert_utilization)
            if utils:
                mean_util = torch.stack(utils, dim=0).mean(dim=0)
                depth_stats["expert_utilization"] = mean_util
                depth_stats["expert_entropy"] = expert_utilization_entropy(mean_util)

            densities: list[torch.Tensor] = []
            for b in typed_blocks:
                if b.attn.last_mask_density is not None:
                    densities.append(b.attn.last_mask_density)
            if densities:
                depth_stats["sparse_mask_density"] = torch.stack(
                    [d if d.dim() == 0 else d.mean() for d in densities]
                ).mean().item()

        return ADAOutput(logits=logits, depth_stats=depth_stats, balance_losses=balance_losses, depth_loss=depth_loss)

    def set_inference_config(self, inference_config) -> None:
        """Compatibility shim matching the README's documented API
        (`ADAInferenceConfig(depth_threshold=...)`). Thresholds are applied
        per-block via AdaptiveDepthController's `threshold` kwarg; wiring a
        stored threshold through here is left as a small follow-up once
        actual compute-skipping inference is implemented (see module docstring).
        """
        self._inference_config = inference_config

    @classmethod
    def from_pretrained(cls, path_or_repo_id: str) -> ADAModel:
        raise NotImplementedError(
            "from_pretrained is not implemented yet — no checkpoints exist. "
            "See scripts/export_to_hf.py for the intended HF export format "
            "once a checkpoint is trained (Phase 1+ in the roadmap)."
        )
