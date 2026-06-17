import torch

from risk_detection.model import GroomingHead


def test_output_shapes():
    head = GroomingHead(d_z=8, safety_dim=11)
    z = torch.randn(8)
    safety = torch.rand(11)

    s_g, b_t = head(z, safety)

    assert s_g.shape == ()
    assert b_t.shape == (6,)


def test_outputs_are_in_unit_interval():
    head = GroomingHead(d_z=8, safety_dim=11)
    z = torch.randn(8) * 10  # large magnitude to stress-test sigmoid saturation
    safety = torch.rand(11)

    s_g, b_t = head(z, safety)

    assert 0.0 <= s_g.item() <= 1.0
    assert torch.all(b_t >= 0.0) and torch.all(b_t <= 1.0)


def test_supports_batched_input():
    head = GroomingHead(d_z=8, safety_dim=11)
    z = torch.randn(4, 8)
    safety = torch.rand(4, 11)

    s_g, b_t = head(z, safety)

    assert s_g.shape == (4,)
    assert b_t.shape == (4, 6)
