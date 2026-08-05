import torch

from ada.layers.sparse_gqa import DynamicSparseGQA


def test_forward_shape(tiny_config):
    layer = DynamicSparseGQA(tiny_config)
    x = torch.randn(2, 6, tiny_config.hidden_dim)
    out = layer(x)
    assert out.shape == x.shape


def test_dense_fallback_when_disabled(tiny_config):
    tiny_config.use_dynamic_sparse_attn = False
    layer = DynamicSparseGQA(tiny_config)
    x = torch.randn(2, 6, tiny_config.hidden_dim)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isclose(layer.last_mask_density, torch.tensor(1.0))


def test_causal_no_future_leakage(tiny_config):
    layer = DynamicSparseGQA(tiny_config)
    layer.eval()
    x = torch.randn(1, 6, tiny_config.hidden_dim)
    x2 = x.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])

    with torch.no_grad():
        out1 = layer(x)
        out2 = layer(x2)

    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5)
