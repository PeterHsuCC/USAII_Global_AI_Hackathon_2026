import torch
from torch import nn

from .grooming_head import DEFAULT_SAFETY_DIM
from .historical_state import STATE_DIM


class EarlyDetectionHead(nn.Module):
    """S_e(t) = sigmoid(W_e [z_t ; F_t^safe ; H_{t-1}] + b_e).

    Combines the current conversation representation, safety-only
    structured signals, and the *previous* historical state to estimate
    whether risk patterns are persistent or escalating. Emotion features
    are excluded so the Safety Score uses only behavioral and rule-based
    signals.

    Input dimension is d_z + safety_dim + history_dim = d_z + 11 + 15 =
    d_z + 26 for the prototype defaults.

    Callers must always pass H_{t-1}, never H_t: the historical state must
    be advanced to H_t only *after* S_e(t) has been computed, which is what
    eliminates the circular dependency between S_e(t) and the historical
    state. This head has no way to enforce that itself -- it just consumes
    whatever h_prev tensor it is given -- so the calling pipeline is
    responsible for the ordering.
    """

    def __init__(
        self,
        d_z: int = 768,
        safety_dim: int = DEFAULT_SAFETY_DIM,
        history_dim: int = STATE_DIM,
    ):
        super().__init__()
        self.d_z = d_z
        self.safety_dim = safety_dim
        self.history_dim = history_dim
        input_dim = d_z + safety_dim + history_dim

        self.W_e = nn.Linear(input_dim, 1)

    def forward(
        self, z: torch.Tensor, safety_features: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        """z: (..., d_z); safety_features: (..., safety_dim) = F_t^safe;
        h_prev: (..., history_dim) = H_{t-1}. Returns S_e(t) in [0,1],
        shape (...,)."""
        context = torch.cat([z, safety_features, h_prev], dim=-1)
        return torch.sigmoid(self.W_e(context)).squeeze(-1)
