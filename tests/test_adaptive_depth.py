import torch

from ada.layers.adaptive_depth import AdaptiveDepthController, depth_budget_loss


def test_forward_shapes(tiny_config):
    controller = AdaptiveDepthController(tiny_config.hidden_dim, tiny_config.depth_controller_hidden)
    x = torch.randn(2, 5, tiny_config.hidden_dim)
    continue_prob, continue_mask = controller(x, hard=False)
    assert continue_prob.shape == (2, 5)
    assert (continue_prob >= 0).all() and (continue_prob <= 1).all()
    assert continue_mask.all()  # soft mode: everyone "continues" (weighting is via continue_prob)


def test_hard_thresholding(tiny_config):
    controller = AdaptiveDepthController(tiny_config.hidden_dim, tiny_config.depth_controller_hidden)
    x = torch.randn(2, 5, tiny_config.hidden_dim)
    _, continue_mask = controller(x, hard=True, threshold=0.5)
    assert continue_mask.dtype == torch.bool


def test_depth_budget_loss_zero_when_under_budget():
    probs = [torch.zeros(2, 5) for _ in range(3)]  # mean depth fraction = 0
    loss = depth_budget_loss(probs, depth_budget_tau=0.5)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_depth_budget_loss_positive_when_over_budget():
    probs = [torch.ones(2, 5) for _ in range(3)]  # mean depth fraction = 1
    loss = depth_budget_loss(probs, depth_budget_tau=0.5)
    assert loss.item() > 0
