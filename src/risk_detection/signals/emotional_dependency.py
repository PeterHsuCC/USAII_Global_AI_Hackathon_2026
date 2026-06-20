from typing import Any

try:
    import anthropic
except ImportError:
    anthropic = None
from pydantic import BaseModel

from ..conversation import ConversationWindow
from .llm_safety import LLMRefusalError

DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are a structured signal extractor for a child-safety conversation \
risk detection system. You analyze a short window of chat messages and score how strongly \
one of the speakers expresses first-person emotional reliance or dependency on the other \
person -- for example, phrases like "I only need you", "I cannot cope without you", or \
"you're the only one I have". This is about that speaker's own expressed emotional state, \
not the other party's behavior toward them.

Score this dependency signal in [0, 1], where 0 means no such expression is present and 1 \
means it is strong and unambiguous. Base your score only on the provided conversation \
window."""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class EmotionalDependencySignal(BaseModel):
    """D_t: a separate LLM-extracted signal targeting first-person
    expressions of emotional reliance (Section 10.2). Distinct from the
    grooming-oriented "dependency" dimension in L_t, which captures the
    perpetrator's manipulation pattern rather than the victim's emotional
    state."""

    dependency: float

    def value(self) -> float:
        return _clamp01(self.dependency)


def _format_transcript(window: ConversationWindow) -> str:
    lines = [f"[t={m.relative_time:.1f}] {m.speaker_id}: {m.text}" for m in window]
    return "Conversation window:\n" + "\n".join(lines)


class EmotionalDependencyExtractor:
    def __init__(self, client: Any | None = None, model: str = DEFAULT_MODEL):
        if client is None:
            if anthropic is None:
                raise ImportError(
                    "anthropic is required to use EmotionalDependencyExtractor. "
                    "Install it with: pip install anthropic"
                )
            client = anthropic.Anthropic()

        self.client = client
        self.model = model

    def extract(self, window: ConversationWindow) -> float:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _format_transcript(window)}],
            output_format=EmotionalDependencySignal,
        )
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise LLMRefusalError(category)
        if response.parsed_output is None:
            raise RuntimeError(
                f"LLM returned no parsed output (stop_reason={response.stop_reason!r}); "
                "cannot extract the emotional-dependency signal for this window"
            )
        return response.parsed_output.value()
