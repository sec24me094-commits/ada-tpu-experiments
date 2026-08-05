"""Input-dependent sparsity mask for Dynamic Sparse GQA.

Reference strategy: use the query/key dot-product magnitude itself as the
relevance proxy, and keep the top `target_density` fraction of causal
positions per query row. This is a simple, differentiable-free (mask is
detached) approach suitable as a v1 baseline; the roadmap's "content-adaptive
sparsity" ambition (see 03_github_repo_design / architecture docs) may
warrant a learned relevance head instead — swap that in here without
changing the call signature.
"""

from __future__ import annotations

import math

import torch


def build_dynamic_sparse_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    target_density: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: (batch, num_heads, seq_len, head_dim) — already causal-consistent
          (i.e. this function does not add the causal mask; the caller does).
    target_density: fraction of causal-visible positions to keep per query
                     row, in (0, 1].

    returns:
        mask:    (batch, num_heads, seq_len, seq_len) bool, True = attend
        density: scalar tensor, actual achieved density (for logging /
                 routing analytics, e.g. output.depth_stats['sparse_mask_density'])
    """
    b, h, l, d = q.shape
    with torch.no_grad():
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)  # (b, h, l, l)
        causal = torch.triu(torch.ones(l, l, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

        num_visible = torch.arange(1, l + 1, device=q.device)  # causal-visible count per row
        keep_count = torch.clamp((num_visible.float() * target_density).ceil().long(), min=1)

        # Rank-based threshold per row: keep top `keep_count[i]` scores in row i.
        sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
        # Gather the score at the keep_count-th position as the per-row threshold.
        idx = (keep_count - 1).clamp(max=l - 1).view(1, 1, l, 1).expand(b, h, l, 1)
        threshold = torch.gather(sorted_scores, dim=-1, index=idx)  # (b, h, l, 1)

        mask = scores >= threshold
        mask = mask & ~causal  # never attend to future positions regardless of threshold ties

    density = mask.float().sum() / (~causal).float().sum().clamp_min(1.0) / (b * h)
    return mask, density
