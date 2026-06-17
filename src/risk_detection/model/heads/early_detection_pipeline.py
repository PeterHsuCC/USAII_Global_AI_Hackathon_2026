from dataclasses import dataclass

import torch

from ...conversation import ConversationWindow
from ...signals.safety_features import SafetyFeatureExtractor, SafetyFeatures
from ..encoder.conversation_encoder import ConversationEncoder
from .early_detection_head import EarlyDetectionHead
from ..state.historical_state import HistoricalRiskState
from ..encoder.message_encoder import MessageEncoder


@dataclass
class EarlyDetectionResult:
    early_detection_score: torch.Tensor  # S_e(t), scalar
    safety_features: SafetyFeatures
    attention_weights: torch.Tensor  # alpha_i, shape (T,)


class EarlyDetectionPipeline:
    """Ties the Conversation Encoder (Section 4.2) and Structured Signal
    Extraction (Section 3) to the Early Detection Head (Section 9) for one
    rolling Conversation Window.

    score() takes h_prev = H_{t-1} explicitly, by design: this pipeline
    never reads or advances a HistoricalStateUpdater itself, so callers
    cannot accidentally pass H_t. Advancing to H_t is the caller's job and
    must happen only after this score has been computed, per Section 9's
    one-step delay.
    """

    def __init__(
        self,
        message_encoder: MessageEncoder,
        conversation_encoder: ConversationEncoder,
        early_detection_head: EarlyDetectionHead,
        safety_feature_extractor: SafetyFeatureExtractor | None = None,
    ):
        self.message_encoder = message_encoder
        self.conversation_encoder = conversation_encoder
        self.early_detection_head = early_detection_head
        self.safety_feature_extractor = safety_feature_extractor or SafetyFeatureExtractor()

    @torch.no_grad()
    def score(
        self, window: ConversationWindow, h_prev: HistoricalRiskState
    ) -> EarlyDetectionResult:
        h = self.message_encoder.encode_window(window)
        if h.shape[0] == 0:
            return EarlyDetectionResult(h.new_zeros(()), SafetyFeatures.zero(), h.new_zeros(0))

        z, alpha = self.conversation_encoder.encode(h)
        features = self.safety_feature_extractor.extract(window)
        safety_tensor = torch.tensor(features.to_vector(), dtype=z.dtype, device=z.device)
        h_prev_tensor = h_prev.to_vector().to(dtype=z.dtype, device=z.device)

        s_e = self.early_detection_head(z, safety_tensor, h_prev_tensor)
        return EarlyDetectionResult(s_e, features, alpha)
