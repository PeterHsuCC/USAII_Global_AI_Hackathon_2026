import pytest
import torch

from risk_detection.model import ConversationEncoder


def test_output_shape_matches_d():
    d = 8
    encoder = ConversationEncoder(d=d)
    h = torch.randn(5, d)

    z, alpha = encoder.encode(h)

    assert z.shape == (d,)
    assert alpha.shape == (5,)


def test_attention_weights_sum_to_one():
    d = 8
    encoder = ConversationEncoder(d=d)
    h = torch.randn(4, d)

    _, alpha = encoder.encode(h)

    assert torch.allclose(alpha.sum(), torch.tensor(1.0), atol=1e-5)


def test_batched_forward_respects_lengths():
    d = 8
    encoder = ConversationEncoder(d=d)
    h = torch.randn(2, 5, d)
    lengths = torch.tensor([5, 2])

    z, alpha = encoder(h, lengths=lengths)

    assert z.shape == (2, d)
    assert torch.allclose(alpha[1, 2:], torch.zeros(3), atol=1e-6)
    assert torch.allclose(alpha[1, :2].sum(), torch.tensor(1.0), atol=1e-5)


def test_rejects_odd_d():
    with pytest.raises(ValueError):
        ConversationEncoder(d=7)


def test_empty_window_returns_zero_vector():
    d = 8
    encoder = ConversationEncoder(d=d)
    h = torch.zeros(0, d)

    z, alpha = encoder.encode(h)

    assert z.shape == (d,)
    assert alpha.shape == (0,)
