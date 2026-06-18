from dataclasses import dataclass

from .historical_state import HistoricalRiskState


@dataclass(frozen=True)
class EarlyWarningThresholds:
    """Prototype thresholds for Warning_t (Section 9.1). Tunable on
    validation data, not fixed constants."""

    tau_e: float = 0.65
    tau_p: float = 0.5
    tau_t: float = 0.05
    tau_e_prime: float = 0.5


DEFAULT_THRESHOLDS = EarlyWarningThresholds()


def compute_warning(
    e_t: float,
    t_t: float,
    p_t: float,
    thresholds: EarlyWarningThresholds = DEFAULT_THRESHOLDS,
) -> bool:
    """Warning_t = 1(E_t > tau_E and P_t > tau_P) or 1(T_t > tau_T and
    E_t > tau_E') (Section 9.1).

    The first disjunct catches high, persistent risk that has plateaued
    (T_t ~ 0); the second catches rapidly escalating risk before it
    plateaus. A pure conjunction of all three conditions would silently
    turn off once T_t settles near zero even while E_t and P_t stay high.
    """
    high_and_persistent = e_t > thresholds.tau_e and p_t > thresholds.tau_p
    rising_and_elevated = t_t > thresholds.tau_t and e_t > thresholds.tau_e_prime
    return high_and_persistent or rising_and_elevated


class EarlyWarningTracker:
    """Latches Warning_t (Section 9.1): once triggered, stays triggered
    until reset() (e.g. after human review) or a new conversation starts,
    rather than being re-evaluated fresh on every window."""

    def __init__(self, thresholds: EarlyWarningThresholds = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds
        self._latched = False

    def update(self, state: HistoricalRiskState) -> bool:
        """Feed the latest H_t; returns the (possibly latched) warning."""
        if not self._latched:
            self._latched = compute_warning(
                state.accumulated_risk, state.risk_trend, state.persistence, self.thresholds
            )
        return self._latched

    @property
    def triggered(self) -> bool:
        return self._latched

    def reset(self) -> None:
        """Clear the latch, e.g. after human review or at a new conversation."""
        self._latched = False
