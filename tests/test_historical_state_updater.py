import pytest
import torch

from risk_detection.model import HistoricalStateUpdater, precursor_risk, trend_label


def test_precursor_risk_formula():
    assert precursor_risk(s_g=0.8, s_cb=0.4) == pytest.approx(0.7 * 0.8 + 0.3 * 0.4)


@pytest.mark.parametrize(
    "t_t, expected",
    [
        (0.06, "increasing"),
        (0.05, "stable"),
        (0.0, "stable"),
        (-0.05, "stable"),
        (-0.06, "decreasing"),
    ],
)
def test_trend_label_thresholds(t_t, expected):
    assert trend_label(t_t) == expected


def test_initial_state_is_zero():
    updater = HistoricalStateUpdater()

    assert updater.state.accumulated_risk == 0.0
    assert updater.state.risk_trend == 0.0
    assert updater.state.persistence == 0.0
    assert torch.equal(updater.state.smoothed_behavior, torch.zeros(6))


def test_accumulated_risk_and_trend_recurrence():
    updater = HistoricalStateUpdater()

    # r_1 = 0.7*1 + 0.3*1 = 1.0 -> E_1 = 0.7*0 + 0.3*1 = 0.3; T_1 = 0.3 - 0 = 0.3
    state1 = updater.update(s_g=1.0, s_cb=1.0, b_t=torch.zeros(6))
    assert state1.accumulated_risk == pytest.approx(0.3)
    assert state1.risk_trend == pytest.approx(0.3)
    assert updater.current_trend_label == "increasing"

    # r_2 = 0 -> E_2 = 0.7*0.3 + 0.3*0 = 0.21; T_2 = 0.21 - 0.3 = -0.09
    state2 = updater.update(s_g=0.0, s_cb=0.0, b_t=torch.zeros(6))
    assert state2.accumulated_risk == pytest.approx(0.21)
    assert state2.risk_trend == pytest.approx(-0.09)
    assert updater.current_trend_label == "decreasing"


def test_persistence_counts_high_risk_windows_within_n():
    updater = HistoricalStateUpdater(persistence_window=3)

    # r values: 1.0, 0.0, 1.0, 0.0 (using s_g == s_cb == r so r_t == that value)
    for r in (1.0, 0.0, 1.0, 0.0):
        state = updater.update(s_g=r, s_cb=r, b_t=torch.zeros(6))

    # n_t = min(3, 4) = 3; last 3 r's are [0.0, 1.0, 0.0] -> 1 of 3 above 0.5
    assert state.persistence == pytest.approx(1 / 3)


def test_persistence_n_t_grows_until_window_fills():
    updater = HistoricalStateUpdater(persistence_window=5)

    state = updater.update(s_g=1.0, s_cb=1.0, b_t=torch.zeros(6))
    assert state.persistence == pytest.approx(1.0)  # n_t = min(5,1) = 1, one hit

    state = updater.update(s_g=0.0, s_cb=0.0, b_t=torch.zeros(6))
    assert state.persistence == pytest.approx(0.5)  # n_t = min(5,2) = 2, one of two


def test_behavior_frequency_matches_recent_window():
    updater = HistoricalStateUpdater(persistence_window=3)
    ones = torch.ones(6)
    zeros = torch.zeros(6)

    # behavior history: ones, zeros, ones, zeros -> deque keeps last 3: [zeros, ones, zeros]
    for b in (ones, zeros, ones, zeros):
        state = updater.update(s_g=0.0, s_cb=0.0, b_t=b)

    assert torch.allclose(state.behavior_frequency, torch.full((6,), 1 / 3))


def test_smoothed_behavior_recurrence():
    updater = HistoricalStateUpdater()
    ones = torch.ones(6)
    zeros = torch.zeros(6)

    # B_bar_1 = 0.7*0 + 0.3*1 = 0.3
    state1 = updater.update(s_g=0.0, s_cb=0.0, b_t=ones)
    assert torch.allclose(state1.smoothed_behavior, torch.full((6,), 0.3))

    # B_bar_2 = 0.7*0.3 + 0.3*0 = 0.21
    state2 = updater.update(s_g=0.0, s_cb=0.0, b_t=zeros)
    assert torch.allclose(state2.smoothed_behavior, torch.full((6,), 0.21))


def test_reset_clears_state_and_history():
    updater = HistoricalStateUpdater(persistence_window=3)
    updater.update(s_g=1.0, s_cb=1.0, b_t=torch.ones(6))
    updater.update(s_g=1.0, s_cb=1.0, b_t=torch.ones(6))

    updater.reset()

    assert updater.state.accumulated_risk == 0.0
    state = updater.update(s_g=0.0, s_cb=0.0, b_t=torch.zeros(6))
    assert state.persistence == 0.0  # n_t = 1, single non-high-risk window


def test_rejects_invalid_persistence_window():
    with pytest.raises(ValueError):
        HistoricalStateUpdater(persistence_window=0)


def test_rejects_wrong_behavior_shape():
    updater = HistoricalStateUpdater()
    with pytest.raises(ValueError):
        updater.update(s_g=0.5, s_cb=0.5, b_t=torch.zeros(5))
