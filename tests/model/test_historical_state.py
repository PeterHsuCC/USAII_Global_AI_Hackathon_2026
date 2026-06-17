import pytest
import torch

from risk_detection.model import HistoricalRiskState


def test_default_state_is_all_zero_with_fifteen_dimensions():
    state = HistoricalRiskState()

    vector = state.to_vector()

    assert vector.shape == (15,)
    assert torch.allclose(vector, torch.zeros(15))


def test_to_vector_preserves_component_order():
    behavior_frequency = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    smoothed_behavior = torch.tensor([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    state = HistoricalRiskState(
        accumulated_risk=0.5,
        risk_trend=-0.2,
        persistence=0.3,
        behavior_frequency=behavior_frequency,
        smoothed_behavior=smoothed_behavior,
    )

    vector = state.to_vector()

    assert vector[0].item() == pytest.approx(0.5)
    assert vector[1].item() == pytest.approx(-0.2)
    assert vector[2].item() == pytest.approx(0.3)
    assert torch.allclose(vector[3:9], behavior_frequency)
    assert torch.allclose(vector[9:15], smoothed_behavior)


def test_rejects_wrong_behavior_frequency_shape():
    with pytest.raises(ValueError):
        HistoricalRiskState(behavior_frequency=torch.zeros(5))


def test_rejects_wrong_smoothed_behavior_shape():
    with pytest.raises(ValueError):
        HistoricalRiskState(smoothed_behavior=torch.zeros(7))
