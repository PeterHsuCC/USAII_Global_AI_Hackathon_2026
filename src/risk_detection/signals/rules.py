import re
from dataclasses import dataclass

from ..conversation import ConversationWindow

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
    r"\b(i'?ll (hurt|kill|find) you|or else|you'?ll regret|i know where you live)\b",
    re.IGNORECASE,
)


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


class RuleSignalExtractor:
    def extract(self, window: ConversationWindow) -> RuleSignals:
        text = " ".join(m.text for m in window)
        return RuleSignals(
            secret_request=bool(_SECRET_REQUEST.search(text)),
            contact_migration=bool(_CONTACT_MIGRATION.search(text)),
            age_reference=bool(_AGE_REFERENCE.search(text)),
            image_request=bool(_IMAGE_REQUEST.search(text)),
            threat_phrase=bool(_THREAT_PHRASE.search(text)),
        )
