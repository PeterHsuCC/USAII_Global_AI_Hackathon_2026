from typing import Any

try:
    import anthropic
except ImportError:
    anthropic = None
from pydantic import BaseModel

from ..conversation import ConversationWindow

DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are a structured signal extractor for a child-safety conversation \
risk detection system. You analyze a short window of chat messages and output calibrated \
risk-indicator scores. You do not make accusations, diagnoses, or final judgments -- your \
output is one input among several to a downstream model that a human analyst reviews.

For the given conversation window, score each of the following six dimensions in the range \
[0, 1], where 0 means the indicator is absent and 1 means it is strongly and unambiguously \
present:

- secrecy: pressure or requests to keep the conversation, a relationship, or specific topics \
hidden from parents, guardians, or other people.
- isolation: language that discourages or undermines the other person's other relationships \
(family, friends) or encourages them to rely solely on this conversation.
- dependency: the speaker fostering emotional reliance in the other person (e.g. "I'm the \
only one who understands you", "you need me").
- sexual_escalation: sexual or romantic content, innuendo, or escalation that is inappropriate \
to the apparent context of the conversation.
- threat: explicit or implicit threats, intimidation, or coercive consequences.
- coercion: pressure, manipulation, or persistence aimed at extracting an action, information, \
or compliance against the other person's evident wishes.

Base your scores only on the provided conversation window. Respond with calibrated scores \
reflecting your confidence, not binary judgments."""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class LLMSafetySignals(BaseModel):
    """L_t = [secrecy, isolation, dependency, sexual_escalation, threat, coercion] in [0,1]^6."""

    secrecy: float
    isolation: float
    dependency: float
    sexual_escalation: float
    threat: float
    coercion: float

    def to_vector(self) -> list[float]:
        return [
            _clamp01(self.secrecy),
            _clamp01(self.isolation),
            _clamp01(self.dependency),
            _clamp01(self.sexual_escalation),
            _clamp01(self.threat),
            _clamp01(self.coercion),
        ]


def _format_transcript(window: ConversationWindow) -> str:
    lines = [f"[t={m.relative_time:.1f}] {m.speaker_id}: {m.text}" for m in window]
    return "Conversation window:\n" + "\n".join(lines)


class LLMSafetySignalExtractor:
    def __init__(self, client: Any | None = None, model: str = DEFAULT_MODEL):
        if client is None:
            if anthropic is None:
                raise ImportError(
                    "anthropic is required to use LLMSafetySignalExtractor. "
                    "Install it with: pip install anthropic"
                )
            client = anthropic.Anthropic()

        self.client = client
        self.model = model
    def extract(self, window: ConversationWindow) -> LLMSafetySignals:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _format_transcript(window)}],
            output_format=LLMSafetySignals,
        )
        return response.parsed_output
