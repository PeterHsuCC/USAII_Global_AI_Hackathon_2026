"""Best-effort regex-based PII redaction (v6 Section 1 / Section 3).

Not NER/ML-based: catches emails, URLs, phone-like digit sequences, and
@handles, but will miss plain names, addresses, and other free-text PII.
This limitation is surfaced verbatim via the Explainability Service's data
limitations output, not hidden (plan decision #6).
"""

import re
from dataclasses import dataclass

_URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Every separator is optional, not just the leading ones -- a bare digit
# run like "5551234567" or "15551234567" (no punctuation at all, exactly
# the format someone fires off mid-chat) previously matched none of the
# four patterns here and passed straight into redacted_content untouched.
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,2}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\w)")
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{2,}")

# Order matters: URL/EMAIL consume '@'/'.' before HANDLE could misfire on them.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("URL", _URL_RE),
    ("EMAIL", _EMAIL_RE),
    ("PHONE", _PHONE_RE),
    ("HANDLE", _HANDLE_RE),
)


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    categories_found: tuple[str, ...]


def redact_text(text: str) -> RedactionResult:
    redacted = text
    found: list[str] = []
    for label, pattern in _PATTERNS:
        if pattern.search(redacted):
            found.append(label)
        redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return RedactionResult(redacted_text=redacted, categories_found=tuple(found))