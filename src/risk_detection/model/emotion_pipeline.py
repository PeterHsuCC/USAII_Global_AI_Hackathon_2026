from dataclasses import dataclass

import torch

from ..conversation import ConversationWindow
from ..signals.emotional_dependency import EmotionalDependencyExtractor
from .aggregation import max_mean_top3
from .emotion_classifier import GoEmotionsClassifier
from .emotion_mapping import DEFAULT_LAMBDA, MAPPED_EMOTION_NAMES, map_emotions
from .emotion_score_head import EmotionScoreHead


@dataclass
class EmotionResult:
    emotion_score: torch.Tensor  # S_m(t), scalar
    mapped_emotions: torch.Tensor  # M_t, shape (5,)
    window_emotions: torch.Tensor  # G_t, shape (d_G,)
    per_message_emotions: torch.Tensor  # G_i, shape (T, d_G) -- for Section 14 evidence extraction
    dependency_signal: float  # D_t


class EmotionPipeline:
    """Ties the GoEmotions classifier, window aggregation (the same
    max-MeanTop3 weighting as the Cyberbullying Head), the project-specific
    emotion mapping (phi), and the learned logistic Emotion Score head
    together for one rolling Conversation Window (Section 10.2).

    Runs independently of the Safety Branch -- it does not take z_t or
    F_t^safe as input, matching Section 3's note that the Emotion Branch
    uses a separate model pipeline combined only at the final score-fusion
    stage.
    """

    def __init__(
        self,
        emotion_classifier: GoEmotionsClassifier,
        emotion_score_head: EmotionScoreHead,
        dependency_extractor: EmotionalDependencyExtractor | None = None,
        lam: float = DEFAULT_LAMBDA,
    ):
        self.emotion_classifier = emotion_classifier
        self.emotion_score_head = emotion_score_head
        self.dependency_extractor = dependency_extractor or EmotionalDependencyExtractor()
        self.lam = lam

    @torch.no_grad()
    def score(self, window: ConversationWindow) -> EmotionResult:
        g_i = self.emotion_classifier.encode_window(window)  # (T, d_G)
        if g_i.shape[0] == 0:
            zero_g = g_i.new_zeros(self.emotion_classifier.d_g)
            zero_m = g_i.new_zeros(len(MAPPED_EMOTION_NAMES))
            return EmotionResult(g_i.new_zeros(()), zero_m, zero_g, g_i, 0.0)

        g_t = max_mean_top3(g_i.transpose(0, 1))  # (d_G,), aggregated per dimension over messages
        d_t = self.dependency_extractor.extract(window)
        m_t = map_emotions(g_t, self.emotion_classifier.label_to_index, d_t, lam=self.lam)
        s_m = self.emotion_score_head(m_t)
        return EmotionResult(s_m, m_t, g_t, g_i, d_t)
