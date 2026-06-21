import re
from dataclasses import dataclass

from ..conversation import ConversationWindow

RULE_SIGNAL_NAMES = (
    "secret_request",
    "contact_migration",
    "age_reference",
    "image_request",
    "threat_phrase",
)

_SECRET_REQUEST = re.compile(
    r"\b(don'?t tell|do not tell|keep (this|it) (a )?secret|just between us|our little secret)\b",
    re.IGNORECASE,
)
_CONTACT_MIGRATION = re.compile(
    r"\b(snapchat|instagram|whatsapp|kik|discord|telegram|text me|call me|"
    r"phone number|add me on)\b",
    re.IGNORECASE,
)
_AGE_REFERENCE = re.compile(
    r"\b(how old are you|what'?s your age|i'?m \d{1,2}\s*(years?\s*old|yo)\b|"
    r"in (middle|high) school|what grade (are you )?in)\b",
    re.IGNORECASE,
)
_IMAGE_REQUEST = re.compile(
    r"\b(send (me )?(a )?(pic|picture|photo|selfie)|send nudes|video call|show me your)\b",
    re.IGNORECASE,
)
_THREAT_PHRASE = re.compile(
    r"\b(i'?ll (hurt|kill|find) you|"
    r"i(?:'m| am)? (?:going to|gonna|will) (hurt|kill|find) you|"
    r"or else|you'?ll regret|i know where you live)\b",
    re.IGNORECASE,
)

_PATTERNS = {
    "secret_request": _SECRET_REQUEST,
    "contact_migration": _CONTACT_MIGRATION,
    "age_reference": _AGE_REFERENCE,
    "image_request": _IMAGE_REQUEST,
    "threat_phrase": _THREAT_PHRASE,
}


@dataclass
class RuleSignals:
    """Q_t = [secret_request, contact_migration, age_reference, image_request,
    threat_phrase] in {0,1}^5."""

    secret_request: bool
    contact_migration: bool
    age_reference: bool
    image_request: bool
    threat_phrase: bool

    def to_vector(self) -> list[float]:
        return [
            float(self.secret_request),
            float(self.contact_migration),
            float(self.age_reference),
            float(self.image_request),
            float(self.threat_phrase),
        ]


@dataclass
class RuleEvidence:
    """Per-rule triggered message indices within a Conversation Window,
    keyed by rule name (RULE_SIGNAL_NAMES). A rule may fire across multiple
    messages -- every triggering message is kept, not just the first
    (Section 14: "if rule j = image request is detected in messages 3 and
    9, both IDs are recorded")."""

    triggered_message_indices: dict[str, list[int]]

    def union_indices(self) -> list[int]:
        """E_t^rule = union over active rules j of TriggeredMessageIDs(j),
        sorted ascending (Section 14)."""
        indices: set[int] = set()
        for message_ids in self.triggered_message_indices.values():
            indices.update(message_ids)
        return sorted(indices)


class RuleSignalExtractor:
    def extract_evidence(self, window: ConversationWindow) -> RuleEvidence:
        """Per-message rule matches -- the basis for both the window-level
        RuleSignals (extract()) and Section 14's rule evidence."""
        triggered: dict[str, list[int]] = {name: [] for name in RULE_SIGNAL_NAMES}
        for i, message in enumerate(window):
            for name, pattern in _PATTERNS.items():
                if pattern.search(message.text):
                    triggered[name].append(i)
        return RuleEvidence(triggered_message_indices=triggered)

    def extract(self, window: ConversationWindow) -> RuleSignals:
        triggered = self.extract_evidence(window).triggered_message_indices
        return RuleSignals(
            secret_request=bool(triggered["secret_request"]),
            contact_migration=bool(triggered["contact_migration"]),
            age_reference=bool(triggered["age_reference"]),
            image_request=bool(triggered["image_request"]),
            threat_phrase=bool(triggered["threat_phrase"]),
        )
