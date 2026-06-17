import math

import pytest
import torch

from risk_detection.model import (
    behavior_loss,
    binary_review_loss,
    cyberbullying_loss,
    masked_multitask_loss,
)


def test_cyberbullying_loss_matches_hand_calculation():
    p = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    y = torch.tensor([0, 1])

    loss = cyberbullying_loss(p, y)

    expected = torch.tensor([-math.log(0.9), -math.log(0.8)])
    assert torch.allclose(loss, expected, atol=1e-5)


def test_binary_review_loss_matches_hand_calculation():
    p = torch.tensor([0.9, 0.2])
    y = torch.tensor([1.0, 0.0])

    loss = binary_review_loss(p, y)

    expected = torch.tensor([-math.log(0.9), -math.log(0.8)])
    assert torch.allclose(loss, expected, atol=1e-5)


def test_behavior_loss_matches_hand_calculation():
    b = torch.tensor([[0.9, 0.1, 0.5, 0.5, 0.5, 0.5]])
    y = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    loss = behavior_loss(b, y)

    expected = torch.tensor([-2 * math.log(0.9) - 4 * math.log(0.5)])
    assert torch.allclose(loss, expected, atol=1e-5)


def test_masked_multitask_loss_ignores_unlabeled_samples():
    losses = {
        "cb": torch.tensor([1.0, 2.0, 3.0]),
        "g": torch.tensor([0.5, 0.5, 0.5]),
    }
    masks = {
        "cb": torch.tensor([1.0, 1.0, 0.0]),
        "g": torch.tensor([0.0, 0.0, 0.0]),  # no sample has a grooming label
    }

    total = masked_multitask_loss(losses, masks)

    # cb: (1*1.0 + 1*2.0 + 0*3.0) / max(1,2) = 1.5; g: 0 / max(1,0) = 0
    assert total.item() == pytest.approx(1.5)


def test_masked_multitask_loss_applies_task_weights():
    losses = {"cb": torch.tensor([2.0]), "g": torch.tensor([4.0])}
    masks = {"cb": torch.tensor([1.0]), "g": torch.tensor([1.0])}

    total = masked_multitask_loss(losses, masks, task_weights={"cb": 1.0, "g": 0.5})

    assert total.item() == pytest.approx(1.0 * 2.0 + 0.5 * 4.0)
