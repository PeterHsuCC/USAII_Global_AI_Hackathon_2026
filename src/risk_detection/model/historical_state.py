from dataclasses import dataclass, field

import torch

BEHAVIOR_DIM = 6
STATE_DIM = 3 + 2 * BEHAVIOR_DIM  # E_t, T_t, P_t, C_t, B_bar_t = 1+1+1+6+6 = 15


@dataclass
class HistoricalRiskState:
    """H_t = [E_t, T_t, P_t, C_t, B_bar_t] in R^15 (Section 7).

    accumulated_risk (E_t): accumulated risk, scalar in [0,1].
    risk_trend (T_t): risk trend, scalar in [-1,1].
    persistence (P_t): high-risk persistence, scalar in [0,1].
    behavior_frequency (C_t): recent behavior frequency, vector in [0,1]^6.
    smoothed_behavior (B_bar_t): smoothed behavior state, vector in [0,1]^6.

    Update equations for these components are out of scope here; this is
    just the state container with the shape and ordering the report
    specifies. Defaults to the zero state.
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
