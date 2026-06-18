import torch

from risk_detection.model import EarlyWarningThresholds, EarlyWarningTracker, compute_warning
from risk_detection.model.state.historical_state import HistoricalRiskState


def _state(e, t, p):
    return HistoricalRiskState(accumulated_risk=e, risk_trend=t, persistence=p)


def test_high_and_persistent_triggers_even_with_flat_trend():
    # Plateaued risk: high and persistent, but T_t ~ 0 -- this is exactly
    # the case a pure AND-of-three conjunction would miss (Section 9.1).
    assert compute_warning(e_t=0.76, t_t=0.0, p_t=0.6) is True


def test_rising_and_elevated_triggers_before_plateau():
    assert compute_warning(e_t=0.55, t_t=0.2, p_t=0.1) is True


def test_low_risk_does_not_trigger():
    assert compute_warning(e_t=0.2, t_t=0.0, p_t=0.1) is False


def test_high_risk_alone_without_persistence_does_not_trigger():
    assert compute_warning(e_t=0.9, t_t=0.0, p_t=0.1) is False


def test_custom_thresholds_are_respected():
    strict = EarlyWarningThresholds(tau_e=0.9, tau_p=0.9, tau_t=0.5, tau_e_prime=0.9)
    assert compute_warning(e_t=0.76, t_t=0.0, p_t=0.6, thresholds=strict) is False


def test_tracker_latches_and_stays_triggered_after_risk_drops():
    tracker = EarlyWarningTracker()

    assert tracker.update(_state(0.2, 0.0, 0.1)) is False
    assert tracker.triggered is False

    assert tracker.update(_state(0.76, 0.0, 0.6)) is True
    assert tracker.triggered is True

    # Risk drops back down, but the warning stays latched.
    assert tracker.update(_state(0.1, -0.6, 0.0)) is True
    assert tracker.triggered is True


def test_tracker_reset_clears_the_latch():
    tracker = EarlyWarningTracker()
    tracker.update(_state(0.76, 0.0, 0.6))
    assert tracker.triggered is True

    tracker.reset()

    assert tracker.triggered is False
    assert tracker.update(_state(0.1, 0.0, 0.0)) is False
