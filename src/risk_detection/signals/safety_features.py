from dataclasses import dataclass

from ..conversation import ConversationWindow
from .llm_safety import LLMSafetySignalExtractor, LLMSafetySignals
from .rules import RuleSignalExtractor, RuleSignals


@dataclass
class SafetyFeatures:
    """F_t^safe = [L_t ; Q_t] in R^11 -- the combined safety feature vector used
    by the Safety Branch. Emotion signals (M_t) are deliberately excluded; they
    are produced by a separate Emotion Branch pipeline and combined only at the
    final score-fusion stage."""

    llm_signals: LLMSafetySignals
    rule_signals: RuleSignals

    def to_vector(self) -> list[float]:
        return self.llm_signals.to_vector() + self.rule_signals.to_vector()

    @classmethod
    def zero(cls) -> "SafetyFeatures":
        """All-zero F_t^safe, e.g. for an empty Conversation Window where
        there is nothing to extract signals from."""
        return cls(
            llm_signals=LLMSafetySignals(
                secrecy=0.0,
                isolation=0.0,
                dependency=0.0,
                sexual_escalation=0.0,
                threat=0.0,
                coercion=0.0,
            ),
            rule_signals=RuleSignals(
                secret_request=False,
                contact_migration=False,
                age_reference=False,
                image_request=False,
                threat_phrase=False,
            ),
        )


class SafetyFeatureExtractor:
    def __init__(
        self,
        llm_extractor: LLMSafetySignalExtractor | None = None,
        rule_extractor: RuleSignalExtractor | None = None,
    ):
        self.llm_extractor = llm_extractor or LLMSafetySignalExtractor()
        self.rule_extractor = rule_extractor or RuleSignalExtractor()

    def extract(self, window: ConversationWindow) -> SafetyFeatures:
        return SafetyFeatures(
            llm_signals=self.llm_extractor.extract(window),
            rule_signals=self.rule_extractor.extract(window),
        )
