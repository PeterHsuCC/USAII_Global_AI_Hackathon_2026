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

    # requires_grad=True on the input (not the frozen params) gives the graph
    # something to backward through -- without it, output has no grad_fn at
    # all and backward() fails before it can check the module's own gradients.
    x = torch.randn(2, 3, requires_grad=True)
    output = module(x)
    output.sum().backward()

    assert module.weight.grad is None
    assert module.bias.grad is None
