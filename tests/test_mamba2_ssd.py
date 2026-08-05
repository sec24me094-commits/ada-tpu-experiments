import torch

from ada.layers.mamba2_ssd import Mamba2SSD


def test_forward_shape(tiny_config):
    layer = Mamba2SSD(tiny_config)
    x = torch.randn(2, 6, tiny_config.hidden_dim)
    out = layer(x)
    assert out.shape == x.shape


def test_causal_no_future_leakage(tiny_config):
    """Changing a future token must not change an earlier position's output."""
    layer = Mamba2SSD(tiny_config)
    layer.eval()
    x = torch.randn(1, 6, tiny_config.hidden_dim)
    x2 = x.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])  # perturb only the tail

    with torch.no_grad():
        out1 = layer(x)
        out2 = layer(x2)

    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5)


def test_backward_runs(tiny_config):
    layer = Mamba2SSD(tiny_config)
    x = torch.randn(2, 6, tiny_config.hidden_dim, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
