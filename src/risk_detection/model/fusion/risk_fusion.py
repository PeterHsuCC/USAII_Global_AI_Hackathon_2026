from dataclasses import dataclass

import torch
from torch import nn


class SafetyScoreFusion(nn.Module):
    """S_safety(t) = sigmoid(b_s + v_cb*S_cb + v_g*S_g + v_e*S_e + v_r*S~_r
    + v_ge*S_g*S_e) (Section 11, Stage 1).

    A small logistic regression over [S_cb, S_g, S_e, S~_r, S_g*S_e].
    Learned via Stage 3 calibration (Section 12.3) on held-out data, with
    the shared encoder and task heads frozen.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)  # order: [S_cb, S_g, S_e, S~_r, S_g*S_e]

    def forward(
        self,
        s_cb: torch.Tensor,
        s_g: torch.Tensor,
        s_e: torch.Tensor,
        s_r_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs are (...,) scalars (one score per window). Returns
        S_safety(t) in [0,1], shape (...,)."""
        s_ge = s_g * s_e
        features = torch.stack([s_cb, s_g, s_e, s_r_tilde, s_ge], dim=-1)
        return torch.sigmoid(self.linear(features)).squeeze(-1)


class OverallScoreFusion(nn.Module):
    """R_t = sigmoid(b_o + w_s*S_safety(t) + w_m*S_emotion(t)) (Section 11,
    Stage 2).

    A separate logistic regression on top of the Safety and Emotion
    scores -- a distinct learned weight set from SafetyScoreFusion, so the
    two branches stay independent until this final combination.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # order: [S_safety, S_emotion]

    def forward(self, s_safety: torch.Tensor, s_emotion: torch.Tensor) -> torch.Tensor:
        """s_safety, s_emotion: (...,). Returns R_t in [0,1], shape (...,)."""
        features = torch.stack([s_safety, s_emotion], dim=-1)
        return torch.sigmoid(self.linear(features)).squeeze(-1)


class PrototypeSafetyScoreFusion(nn.Module):
    """Prototype Safety Score: sigmoid(b_s + v_cb*S_cb + v_g*S_g + v_r*S~_r)
    (Section 11), omitting S_e and S_g*S_e.

    The full SafetyScoreFusion above requires S_e, but the Early Detection
    Head is not yet trained (Section 9.1/19.5) -- feeding its
    framework-initialized output into a calibration-bound linear layer
    would mix a real signal with noise. Use this 3-feature variant until
    S_e is trained, at which point SafetyScoreFusion is the target.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # order: [S_cb, S_g, S~_r]

    def forward(
        self,
        s_cb: torch.Tensor,
        s_g: torch.Tensor,
        s_r_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs are (...,) scalars (one score per window). Returns
        the prototype S_safety(t) in [0,1], shape (...,)."""
        features = torch.stack([s_cb, s_g, s_r_tilde], dim=-1)
        return torch.sigmoid(self.linear(features)).squeeze(-1)


@dataclass
class FusedScores:
    safety_score: torch.Tensor  # S_safety(t)
    emotion_score: torch.Tensor  # S_emotion(t) = S_m(t)
    overall_score: torch.Tensor  # R_t


class PrototypeRiskFusion(nn.Module):
    """Two-stage fusion using the prototype Safety Score (no S_e). Same
    Stage 2 (OverallScoreFusion) as the full RiskFusion -- only Stage 1
    changes. This is the variant `IntegratedInferencePipeline` should use
    until the Early Detection Head is trained (Section 19.5)."""

    def __init__(self):
        super().__init__()
        self.safety_fusion = PrototypeSafetyScoreFusion()
        self.overall_fusion = OverallScoreFusion()

    def forward(
        self,
        s_cb: torch.Tensor,
        s_g: torch.Tensor,
        s_r_tilde: torch.Tensor,
        s_emotion: torch.Tensor,
    ) -> FusedScores:
        s_safety = self.safety_fusion(s_cb, s_g, s_r_tilde)
        r_t = self.overall_fusion(s_safety, s_emotion)
        return FusedScores(safety_score=s_safety, emotion_score=s_emotion, overall_score=r_t)


class RiskFusion(nn.Module):
    """Two-stage hierarchical fusion (Section 11). Stage 1 produces the
    Safety Score from the behavioral/rule-based task scores; Stage 2
    produces the Overall Score from the Safety Score and the
    already-computed Emotion Score. Each stage has its own learned weight
    set, so the Safety and Emotion branches stay score-level independent
    until this final combination.

    The three outputs are exactly the report's three dashboard scores:
    Safety Score, Emotion Score, and Overall Score.
    """

    def __init__(self):
        super().__init__()
        self.safety_fusion = SafetyScoreFusion()
        self.overall_fusion = OverallScoreFusion()

    def forward(
        self,
        s_cb: torch.Tensor,
        s_g: torch.Tensor,
        s_e: torch.Tensor,
        s_r_tilde: torch.Tensor,
        s_emotion: torch.Tensor,
    ) -> FusedScores:
        s_safety = self.safety_fusion(s_cb, s_g, s_e, s_r_tilde)
        r_t = self.overall_fusion(s_safety, s_emotion)
        return FusedScores(safety_score=s_safety, emotion_score=s_emotion, overall_score=r_t)
