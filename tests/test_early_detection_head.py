import torch

from risk_detection.model import EarlyDetectionHead


def test_input_dimension_is_dz_plus_26_by_default():
    head = EarlyDetectionHead(d_z=768)

    assert head.W_e.in_features == 768 + 11 + 15


def test_output_shapes():
    head = EarlyDetectionHead(d_z=8, safety_dim=11, history_dim=15)
    z = torch.randn(8)
    safety = torch.rand(11)
    h_prev = torch.rand(15)

    s_e = head(z, safety, h_prev)

    assert s_e.shape == ()
    assert 0.0 <= s_e.item() <= 1.0


def test_supports_batched_input():
    head = EarlyDetectionHead(d_z=8, safety_dim=11, history_dim=15)
    z = torch.randn(4, 8)
    safety = torch.rand(4, 11)
    h_prev = torch.rand(4, 15)

    s_e = head(z, safety, h_prev)

    assert s_e.shape == (4,)


def test_h_prev_actually_influences_the_score():
    torch.manual_seed(0)
    head = EarlyDetectionHead(d_z=8, safety_dim=11, history_dim=15)
    z = torch.randn(8)
    safety = torch.rand(11)

    s_e_low_history = head(z, safety, torch.zeros(15))
    s_e_high_history = head(z, safety, torch.ones(15))

    assert s_e_low_history.item() != s_e_high_history.item()
