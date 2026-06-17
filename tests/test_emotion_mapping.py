import pytest
import torch

from risk_detection.model import MAPPED_EMOTION_NAMES, map_emotions


def test_mapping_formula_matches_hand_calculation():
    label_to_index = {
        "fear": 0,
        "sadness": 1,
        "anger": 2,
        "nervousness": 3,
        "grief": 4,
        "caring": 5,
        "love": 6,
    }
    g_t = torch.tensor([0.8, 0.5, 0.3, 0.6, 0.2, 0.4, 0.9])
    d_t = 0.7

    m_t = map_emotions(g_t, label_to_index, d_t, lam=0.1)

    expected_fear = 0.8
    expected_sadness = 0.5
    expected_anger = 0.3
    expected_distress = 0.30 * 0.8 + 0.30 * 0.6 + 0.20 * 0.2 + 0.20 * 0.5
    expected_dependency_proxy = 0.5 * 0.4 + 0.5 * 0.9
    expected_dependency = 0.1 * expected_dependency_proxy + 0.9 * d_t

    assert m_t.shape == (5,)
    assert MAPPED_EMOTION_NAMES == ("fear", "sadness", "anger", "distress", "dependency")
    assert m_t[0].item() == pytest.approx(expected_fear)
    assert m_t[1].item() == pytest.approx(expected_sadness)
    assert m_t[2].item() == pytest.approx(expected_anger)
    assert m_t[3].item() == pytest.approx(expected_distress)
    assert m_t[4].item() == pytest.approx(expected_dependency)


def test_lambda_near_zero_makes_dependency_t_dominate():
    label_to_index = {
        "fear": 0,
        "sadness": 1,
        "anger": 2,
        "nervousness": 3,
        "grief": 4,
        "caring": 5,
        "love": 6,
    }
    g_t = torch.ones(7)  # caring/love proxy maxed out at 1.0

    m_t_low_d = map_emotions(g_t, label_to_index, d_t=0.0, lam=0.1)
    m_t_high_d = map_emotions(g_t, label_to_index, d_t=1.0, lam=0.1)

    assert m_t_low_d[4].item() == pytest.approx(0.1 * 1.0 + 0.9 * 0.0)
    assert m_t_high_d[4].item() == pytest.approx(0.1 * 1.0 + 0.9 * 1.0)


def test_missing_label_raises_key_error():
    label_to_index = {"fear": 0}
    g_t = torch.tensor([0.5])

    with pytest.raises(KeyError):
        map_emotions(g_t, label_to_index, d_t=0.5)
