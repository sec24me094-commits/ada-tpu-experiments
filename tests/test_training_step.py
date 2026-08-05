import copy

import torch

from ada.model import ADAModel
from ada.training.loss import compute_loss
from ada.training.optimizer import build_lr_scheduler, build_optimizer


def test_single_training_step_reduces_or_computes_loss(tiny_config):
    model = ADAModel(tiny_config)
    optimizer = build_optimizer(model, tiny_config)
    scheduler = build_lr_scheduler(optimizer, tiny_config, total_steps=100)

    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
    labels = input_ids.clone()

    output = model(input_ids)
    loss_out = compute_loss(output, labels, tiny_config)

    assert torch.isfinite(loss_out.total)

    optimizer.zero_grad(set_to_none=True)
    loss_out.total.backward()

    # Confirm gradients actually flow to every top-level parameter group.
    no_grad_params = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad_params, f"Parameters with no gradient: {no_grad_params}"

    torch.nn.utils.clip_grad_norm_(model.parameters(), tiny_config.grad_clip)
    optimizer.step()
    scheduler.step()


def test_params_actually_change_after_step(tiny_config):
    model = ADAModel(tiny_config)
    before = copy.deepcopy(model.state_dict())

    optimizer = build_optimizer(model, tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
    labels = input_ids.clone()

    output = model(input_ids)
    loss_out = compute_loss(output, labels, tiny_config)
    optimizer.zero_grad(set_to_none=True)
    loss_out.total.backward()
    optimizer.step()

    after = model.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed
