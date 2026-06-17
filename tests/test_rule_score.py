import pytest

from risk_detection import RuleSignals, rule_safety_score


def test_uniform_weights_average_active_signals():
    signals = RuleSignals(
        secret_request=True,
        contact_migration=False,
        age_reference=False,
        image_request=True,
        threat_phrase=False,
    )

    # S_r = 1+0+0+1+0 = 2; sum(rho) = 5; S~_r = min(1, 2/5) = 0.4
    assert rule_safety_score(signals) == pytest.approx(0.4)


def test_all_signals_active_saturates_at_one():
    signals = RuleSignals(True, True, True, True, True)

    assert rule_safety_score(signals) == pytest.approx(1.0)


def test_no_signals_active_is_zero():
    signals = RuleSignals(False, False, False, False, False)

    assert rule_safety_score(signals) == 0.0


def test_custom_weights_emphasize_severe_signals():
    signals = RuleSignals(
        secret_request=False,
        contact_migration=False,
        age_reference=False,
        image_request=True,
        threat_phrase=False,
    )
    weights = [1.0, 1.0, 1.0, 5.0, 1.0]  # emphasize image_request

    # S_r = 5; sum(rho) = 9; S~_r = 5/9
    assert rule_safety_score(signals, weights=weights) == pytest.approx(5 / 9)


def test_rejects_mismatched_weight_length():
    signals = RuleSignals(False, False, False, False, False)

    with pytest.raises(ValueError):
        rule_safety_score(signals, weights=[1.0, 1.0])


def test_rejects_zero_total_weight():
    signals = RuleSignals(False, False, False, False, False)

    with pytest.raises(ValueError):
        rule_safety_score(signals, weights=[0.0] * 5)
