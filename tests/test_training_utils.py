import torch
from torch import nn

from risk_detection.model import freeze


def test_freeze_disables_gradients_on_all_parameters():
    module = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    assert all(p.requires_grad for p in module.parameters())

    freeze(module)

    assert all(not p.requires_grad for p in module.parameters())


def test_frozen_module_does_not_accumulate_gradients():
    module = nn.Linear(3, 1)
    freeze(module)

    output = module(torch.randn(2, 3))
    output.sum().backward()

    assert module.weight.grad is None
    assert module.bias.grad is None
