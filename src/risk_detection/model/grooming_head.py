import torch
from torch import nn

DEFAULT_SAFETY_DIM = 11

BEHAVIOR_NAMES = (
    "rapid_trust_building",
    "secrecy",
    "isolation",
    "emotional_dependency",
    "sexual_escalation",
    "coercion",
)


class GroomingHead(nn.Module):
    """S_g(t) = sigmoid(W_g [z_t ; F_t^safe] + b_g);
    B_t = sigmoid(W_b [z_t ; F_t^safe] + b_b) in [0,1]^6.

    The six behavior outputs correspond, in order, to BEHAVIOR_NAMES: rapid
    trust-building, secrecy, isolation, emotional dependency, sexual
    escalation, and coercion.
    """

    def __init__(self, d_z: int = 768, safety_dim: int = DEFAULT_SAFETY_DIM):
        super().__init__()
        self.d_z = d_z
        self.safety_dim = safety_dim
        self.num_behaviors = len(BEHAVIOR_NAMES)
        input_dim = d_z + safety_dim

        self.W_g = nn.Linear(input_dim, 1)
        self.W_b = nn.Linear(input_dim, self.num_behaviors)

    def forward(
        self, z: torch.Tensor, safety_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (..., d_z) conversation representation; safety_features:
        (..., safety_dim) F_t^safe = [L_t ; Q_t].

        Returns (S_g, B_t): S_g is (...,) scalar grooming score in [0,1];
        B_t is (..., 6) behavior vector in [0,1]^6.
        """
        context = torch.cat([z, safety_features], dim=-1)
        s_g = torch.sigmoid(self.W_g(context)).squeeze(-1)
        b_t = torch.sigmoid(self.W_b(context))
        return s_g, b_t
