import torch

from risk_detection.model import EmotionScoreHead


def test_output_shape_and_range():
    head = EmotionScoreHead()
    m_t = torch.rand(5)

    s_m = head(m_t)

    assert s_m.shape == ()
    assert 0.0 <= s_m.item() <= 1.0


def test_supports_batched_input():
    head = EmotionScoreHead()
    m_t = torch.rand(4, 5)

    s_m = head(m_t)

    assert s_m.shape == (4,)
