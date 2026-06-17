from collections import deque
from dataclasses import dataclass, field

import torch

BEHAVIOR_DIM = 6
STATE_DIM = 3 + 2 * BEHAVIOR_DIM  # E_t, T_t, P_t, C_t, B_bar_t = 1+1+1+6+6 = 15

DEFAULT_PERSISTENCE_WINDOW = 10  # n: prototype default, tunable
HIGH_RISK_THRESHOLD = 0.5
TREND_THRESHOLD = 0.05

TREND_INCREASING = "increasing"
TREND_STABLE = "stable"
TREND_DECREASING = "decreasing"


@dataclass
class HistoricalRiskState:
    """H_t = [E_t, T_t, P_t, C_t, B_bar_t] in R^15 (Section 7).

    accumulated_risk (E_t): accumulated risk, scalar in [0,1].
    risk_trend (T_t): risk trend, scalar in [-1,1].
    persistence (P_t): high-risk persistence, scalar in [0,1].
    behavior_frequency (C_t): recent behavior frequency, vector in [0,1]^6.
    smoothed_behavior (B_bar_t): smoothed behavior state, vector in [0,1]^6.

    Defaults to the zero state (H_0 = 0). See HistoricalStateUpdater for the
    Section 8 recurrences that advance this state over time.
    """

    accumulated_risk: float = 0.0
    risk_trend: float = 0.0
    persistence: float = 0.0
    behavior_frequency: torch.Tensor = field(default_factory=lambda: torch.zeros(BEHAVIOR_DIM))
    smoothed_behavior: torch.Tensor = field(default_factory=lambda: torch.zeros(BEHAVIOR_DIM))

    def __post_init__(self) -> None:
        if tuple(self.behavior_frequency.shape) != (BEHAVIOR_DIM,):
            raise ValueError(f"behavior_frequency must have shape ({BEHAVIOR_DIM},)")
        if tuple(self.smoothed_behavior.shape) != (BEHAVIOR_DIM,):
            raise ValueError(f"smoothed_behavior must have shape ({BEHAVIOR_DIM},)")

    def to_vector(self) -> torch.Tensor:
        """Flatten into the 15-dim H_t = [E_t, T_t, P_t, C_t, B_bar_t]."""
        scalars = torch.tensor(
            [self.accumulated_risk, self.risk_trend, self.persistence],
            dtype=self.behavior_frequency.dtype,
        )
        return torch.cat([scalars, self.behavior_frequency, self.smoothed_behavior])


def precursor_risk(s_g: float, s_cb: float) -> float:
    """r_t = 0.7 * S_g(t) + 0.3 * S_cb(t) (Section 8.1)."""
    return 0.7 * float(s_g) + 0.3 * float(s_cb)


def trend_label(t_t: float, threshold: float = TREND_THRESHOLD) -> str:
    """trend_t = increasing if T_t > threshold; stable if |T_t| <= threshold;
    decreasing if T_t < -threshold (Section 8.3)."""
    if t_t > threshold:
        return TREND_INCREASING
    if t_t < -threshold:
        return TREND_DECREASING
    return TREND_STABLE


class HistoricalStateUpdater:
    """Maintains H_t across a conversation and applies the Section 8 update
    equations (8.1-8.6) at every new time step t.

    persistence_window (n) is the number of most recent time steps used for
    the persistence (P_t, Section 8.4) and behavior frequency (C_t, Section
    8.5) running statistics; n_t = min(n, t) falls out naturally from the
    bounded history below.
    """

    def __init__(self, persistence_window: int = DEFAULT_PERSISTENCE_WINDOW):
        if persistence_window < 1:
            raise ValueError("persistence_window must be >= 1")
        self.persistence_window = persistence_window
        self.state = HistoricalRiskState()  # H_0 = 0
        self._risk_history: deque[float] = deque(maxlen=persistence_window)
        self._behavior_history: deque[torch.Tensor] = deque(maxlen=persistence_window)

    def reset(self) -> None:
        """Reset to H_0 = 0 and clear the rolling history, e.g. at the start
        of a new conversation."""
        self.state = HistoricalRiskState()
        self._risk_history.clear()
        self._behavior_history.clear()

    def update(self, s_g: float, s_cb: float, b_t: torch.Tensor) -> HistoricalRiskState:
        """Advance the state by one time step given the current S_g(t),
        S_cb(t), and B_t; returns the new H_t."""
        b_t = torch.as_tensor(b_t, dtype=torch.float32)
        if tuple(b_t.shape) != (BEHAVIOR_DIM,):
            raise ValueError(f"b_t must have shape ({BEHAVIOR_DIM},)")

        previous = self.state

        # 8.1 Immediate precursor risk
        r_t = precursor_risk(s_g, s_cb)
        self._risk_history.append(r_t)

        # 8.2 Accumulated risk
        e_t = 0.7 * previous.accumulated_risk + 0.3 * r_t

        # 8.3 Risk trend
        t_t = e_t - previous.accumulated_risk

        # 8.4 Persistence: n_t = len(self._risk_history) == min(n, t)
        p_t = sum(1.0 for r in self._risk_history if r > HIGH_RISK_THRESHOLD) / len(
            self._risk_history
        )

        # 8.5 Behavior frequency
        self._behavior_history.append(b_t)
        stacked = torch.stack(list(self._behavior_history))  # (n_t, 6)
        c_t = (stacked > HIGH_RISK_THRESHOLD).float().mean(dim=0)

        # 8.6 Smoothed behavior state
        b_bar_t = 0.7 * previous.smoothed_behavior + 0.3 * b_t

        self.state = HistoricalRiskState(
            accumulated_risk=e_t,
            risk_trend=t_t,
            persistence=p_t,
            behavior_frequency=c_t,
            smoothed_behavior=b_bar_t,
        )
        return self.state

    @property
    def current_trend_label(self) -> str:
        """Readable label for the current risk trend (Section 8.3)."""
        return trend_label(self.state.risk_trend)
