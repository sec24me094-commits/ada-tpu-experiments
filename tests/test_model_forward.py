import copy
import itertools

import torch

from ada.model import ADAModel, ADAOutput


def test_forward_shape(tiny_config):
    model = ADAModel(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
    out = model(input_ids)
    assert isinstance(out, ADAOutput)
    assert out.logits.shape == (2, 6, tiny_config.vocab_size)


def test_inference_mode_populates_depth_stats(tiny_config):
    model = ADAModel(tiny_config)
    model.eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
    with torch.no_grad():
        out = model(input_ids, inference=True, return_routing_stats=True)
    assert "mean_depth" in out.depth_stats
    assert 1 <= out.depth_stats["mean_depth"] <= tiny_config.num_layers
    assert "expert_entropy" in out.depth_stats


def test_all_ablation_variants_run(tiny_config):
    """Every combination of the four toggleable components should produce a
    valid forward pass — this is what the ablation matrix in configs/ relies on."""
    for use_ssm, use_moe, use_sparse, use_depth in itertools.product([True, False], repeat=4):
        cfg = copy.deepcopy(tiny_config)
        cfg.use_ssm = use_ssm
        cfg.use_moe = use_moe
        cfg.use_dynamic_sparse_attn = use_sparse
        cfg.use_adaptive_depth = use_depth
        model = ADAModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
        out = model(input_ids)
        assert out.logits.shape == (1, 5, cfg.vocab_size)
