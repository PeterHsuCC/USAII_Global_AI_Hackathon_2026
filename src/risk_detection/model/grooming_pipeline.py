from dataclasses import dataclass

import torch

from ..conversation import ConversationWindow
from ..signals.safety_features import SafetyFeatureExtractor, SafetyFeatures
from .conversation_encoder import ConversationEncoder
from .grooming_head import GroomingHead
from .message_encoder import MessageEncoder


@dataclass
class GroomingResult:
    grooming_score: torch.Tensor  # S_g(t), scalar
    behaviors: torch.Tensor  # B_t, shape (6,)
    safety_features: SafetyFeatures
    attention_weights: torch.Tensor  # alpha_i, shape (T,)


class GroomingPipeline:
    """Ties the Conversation Encoder (Section 4.2) and Structured Signal
    Extraction (Section 3) to the Online Grooming Head (Section 6) for one
    rolling Conversation Window."""

    def __init__(
        self,
        message_encoder: MessageEncoder,
        conversation_encoder: ConversationEncoder,
        grooming_head: GroomingHead,
        safety_feature_extractor: SafetyFeatureExtractor | None = None,
    ):
        self.message_encoder = message_encoder
        self.conversation_encoder = conversation_encoder
        self.grooming_head = grooming_head
        self.safety_feature_extractor = safety_feature_extractor or SafetyFeatureExtractor()

    @torch.no_grad()
    def score(self, window: ConversationWindow) -> GroomingResult:
        h = self.message_encoder.encode_window(window)
        if h.shape[0] == 0:
            zero_score = h.new_zeros(())
            zero_behaviors = h.new_zeros(self.grooming_head.num_behaviors)
            return GroomingResult(zero_score, zero_behaviors, SafetyFeatures.zero(), h.new_zeros(0))

        z, alpha = self.conversation_encoder.encode(h)
        features = self.safety_feature_extractor.extract(window)
        safety_tensor = torch.tensor(features.to_vector(), dtype=z.dtype, device=z.device)
        s_g, b_t = self.grooming_head(z, safety_tensor)
        return GroomingResult(s_g, b_t, features, alpha)
