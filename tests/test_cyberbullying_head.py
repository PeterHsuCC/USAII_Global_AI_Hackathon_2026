import torch

from risk_detection.model import CyberbullyingHead, max_mean_top3


def test_stage2_reproduces_stage1_at_init():
    head = CyberbullyingHead(d=8, d_z=8)
    h = torch.randn(5, 8)
    z = torch.randn(8)

    p1 = head.forward_stage1(h)
    p2 = head.forward_stage2(h, z)

    assert torch.allclose(p1, p2, atol=1e-6)


def test_risk_is_one_minus_nonbullying_prob():
    head = CyberbullyingHead(d=8, d_z=8, non_bullying_index=1)
    h = torch.randn(3, 8)

    p = head.forward_stage1(h)
    risk = head.risk(p)

    assert torch.allclose(risk, 1.0 - p[:, 1])


def test_window_score_matches_max_mean_top3():
    head = CyberbullyingHead(d=8, d_z=8)
    risk = torch.tensor([0.9, 0.1, 0.5, 0.4, 0.95])

    assert torch.allclose(head.window_score(risk), max_mean_top3(risk))


def test_max_mean_top3_formula():
    values = torch.tensor([0.9, 0.1, 0.5, 0.4, 0.95])
    expected = 0.6 * 0.95 + 0.4 * ((0.95 + 0.9 + 0.5) / 3)

    assert torch.allclose(max_mean_top3(values), torch.tensor(expected), atol=1e-5)
