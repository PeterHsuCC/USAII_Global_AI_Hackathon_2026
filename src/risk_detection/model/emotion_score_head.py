import torch
from torch import nn

from .emotion_mapping import MAPPED_EMOTION_NAMES

NUM_MAPPED_EMOTIONS = len(MAPPED_EMOTION_NAMES)


class EmotionScoreHead(nn.Module):
    """S_m(t) = sigmoid(b_m + theta^T M_t) (Section 10.2).

    theta = [theta_Fear, theta_Sadness, theta_Anger, theta_Distress,
    theta_Dep] are learned emotion weights and b_m is a learned bias.
    Larger positive weights indicate a stronger learned relationship
    between that emotion dimension and emotion-related review risk.
    Distress and Dependency are project-specific composite signals
    (Section 10.2); their weights should only be interpreted once the
    underlying mapping has been validated with labeled data.
    """

    def __init__(self, num_mapped_emotions: int = NUM_MAPPED_EMOTIONS):
        super().__init__()
        self.linear = nn.Linear(num_mapped_emotions, 1)

    def forward(self, m_t: torch.Tensor) -> torch.Tensor:
        """m_t: (..., 5) mapped emotion vector. Returns S_m(t) in [0,1],
        shape (...,)."""
        return torch.sigmoid(self.linear(m_t)).squeeze(-1)
