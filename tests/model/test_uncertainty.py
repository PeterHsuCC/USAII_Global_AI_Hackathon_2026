import pytest
import torch
from torch import nn

from risk_detection.model import (
    currently_operable_review,
    enable_mc_dropout,
    human_review_required,
    mc_dropout_stats,
)


def test_mc_dropout_stats_with_deterministic_predict_fn():
    def predict_fn():
        return torch.tensor(0.42)

    estimate = mc_dropout_stats(predict_fn, n=5)

    assert estimate.mean.item() == pytest.approx(0.42)
    assert estimate.variance.item() == pytest.approx(0.0)
    assert estimate.uncertainty.item() == pytest.approx(0.0)
    assert estimate.confidence.item() == pytest.approx(1.0)


def test_mc_dropout_stats_matches_hand_calculation():
    values = iter([0.2, 0.4, 0.6, 0.8])

    def predict_fn():
        return torch.tensor(next(values))

    estimate = mc_dropout_stats(predict_fn, n=4)

    mean = (0.2 + 0.4 + 0.6 + 0.8) / 4
    variance = sum((v - mean) ** 2 for v in (0.2, 0.4, 0.6, 0.8)) / 4
    uncertainty = min(1.0, 4 * variance)

    assert estimate.mean.item() == pytest.approx(mean, abs=1e-6)
    assert estimate.variance.item() == pytest.approx(variance, abs=1e-6)
    assert estimate.uncertainty.item() == pytest.approx(uncertainty, abs=1e-6)
    assert estimate.confidence.item() == pytest.approx(1 - uncertainty, abs=1e-6)


def test_uncertainty_is_clamped_to_one():
    values = iter([0.0, 1.0, 0.0, 1.0])  # variance = 0.25 -> 4*0.25 = 1.0 exactly

    def predict_fn():
        return torch.tensor(next(values))

    estimate = mc_dropout_stats(predict_fn, n=4)

    assert estimate.uncertainty.item() == pytest.approx(1.0)
    assert estimate.confidence.item() == pytest.approx(0.0)


def test_mc_dropout_stats_supports_batched_predictions():
    batches = iter([torch.tensor([0.2, 0.8]), torch.tensor([0.4, 0.6])])

    def predict_fn():
        return next(batches)

    estimate = mc_dropout_stats(predict_fn, n=2)

    assert estimate.mean.shape == (2,)
    assert torch.allclose(estimate.mean, torch.tensor([0.3, 0.7]))


def test_mc_dropout_stats_rejects_invalid_n():
    with pytest.raises(ValueError):
        mc_dropout_stats(lambda: torch.tensor(0.5), n=0)


def test_enable_mc_dropout_only_activates_dropout_layers():
    module = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 1))
    module.eval()

    enable_mc_dropout(module)

    assert module.training is False
    assert module[0].training is False
    assert module[1].training is True
    assert module[2].training is False


@pytest.mark.parametrize(
    "r_hat, confidence, s_r_tilde, expected",
    [
        (0.71, 0.9, 0.1, True),  # high overall score (strictly above 0.7)
        (0.1, 0.59, 0.1, True),  # low confidence (strictly below 0.6)
        (0.1, 0.9, 0.8, True),  # rule score at threshold (>= 0.8 triggers)
        (0.5, 0.8, 0.5, False),  # all comfortably below thresholds
        (0.7, 0.6, 0.79, False),  # exactly at the non-inclusive boundaries
    ],
)
def test_human_review_required(r_hat, confidence, s_r_tilde, expected):
    decision = human_review_required(r_hat=r_hat, confidence=confidence, s_r_tilde=s_r_tilde)

    assert bool(decision.item()) is expected


def test_human_review_required_threat_phrase_override():
    # All other signals comfortably below threshold -- only the single
    # severe rule should force review.
    decision = human_review_required(
        r_hat=0.1, confidence=0.9, s_r_tilde=0.2, q_threat_phrase=True
    )
    assert bool(decision.item()) is True


def test_human_review_required_defaults_threat_phrase_to_false():
    decision = human_review_required(r_hat=0.1, confidence=0.9, s_r_tilde=0.2)
    assert bool(decision.item()) is False


@pytest.mark.parametrize(
    "s_r_tilde, q_threat_phrase, expected",
    [
        (0.8, False, True),  # rule score at threshold
        (0.2, True, True),  # single severe rule alone
        (0.2, False, False),  # neither condition met
        (0.79, False, False),  # just below the non-inclusive boundary
    ],
)
def test_currently_operable_review(s_r_tilde, q_threat_phrase, expected):
    decision = currently_operable_review(s_r_tilde=s_r_tilde, q_threat_phrase=q_threat_phrase)
    assert bool(decision.item()) is expected


def test_currently_operable_review_ignores_score_and_confidence():
    # Demonstrates the Section 19.5 point directly: a sky-high R_hat_t (if
    # it were passed in, which this function doesn't even accept) cannot
    # influence this decision -- only the rule-based terms can.
    decision = currently_operable_review(s_r_tilde=0.0, q_threat_phrase=False)
    assert bool(decision.item()) is False
