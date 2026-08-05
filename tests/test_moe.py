import torch

from ada.layers.moe import MoELayer


def test_forward_shape(tiny_config):
    layer = MoELayer(tiny_config)
    x = torch.randn(2, 5, tiny_config.hidden_dim)
    out = layer(x)
    assert out.shape == x.shape


def test_balance_loss_is_scalar_and_finite(tiny_config):
    layer = MoELayer(tiny_config)
    x = torch.randn(2, 5, tiny_config.hidden_dim)
    layer(x)
    assert layer.last_balance_loss is not None
    assert layer.last_balance_loss.dim() == 0
    assert torch.isfinite(layer.last_balance_loss)


def test_expert_utilization_sums_to_one(tiny_config):
    layer = MoELayer(tiny_config)
    x = torch.randn(2, 5, tiny_config.hidden_dim)
    layer(x)
    util = layer.last_expert_utilization
    assert util.shape == (tiny_config.moe_num_experts,)
    assert torch.isclose(util.sum(), torch.tensor(1.0), atol=1e-4)
