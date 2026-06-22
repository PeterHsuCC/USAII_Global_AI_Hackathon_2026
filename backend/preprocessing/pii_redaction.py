"""Regex- and NER-based PII redaction (v6 Section 1 / Section 3).

Catches emails, URLs, phone-like digit sequences, @handles, and US-style
street addresses via regex, plus person names and place names via a
general-purpose NER model (dslim/bert-base-NER). A live case containing two
plain names and a street address confirmed none of the four were redacted
under the old regex-only version, and that the module's own prior claim --
"this limitation is surfaced verbatim via the Explainability Service's data
limitations output" -- was aspirational, not actual (no such entry ever
appeared in a real API response). PII_REDACTION_SCOPE_LIMITATION below is
now genuinely included in every case's data_limitations
(anonymize.py's prepare_conversation()), and is reworded to describe the
current, still-imperfect coverage rather than the old, fully-absent one.

NER is not perfect (informal chat text, uncommon names, and names without
enough surrounding context can still be missed; the model can also produce
false positives on ordinary capitalized words) and is therefore a
best-effort improvement, not a guarantee -- consistent with this module's
existing "best-effort" framing for the regex patterns below.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("risk_platform")

PII_REDACTION_SCOPE_LIMITATION = (
    "PII redaction combines regex (emails, URLs, phone-like digit sequences, "
    "@handles, US-style street addresses) with a general-purpose NER model "
    "for person and place names. Neither is exhaustive: regex only matches "
    "the structured formats above, and NER can still miss uncommon names or "
    "names with little surrounding context, or occasionally redact an "
    "ordinary capitalized word by mistake. Treat redaction as a strong "
    "best-effort reduction in exposed personal information, not a guarantee "
    "that none remains."
)

# Set when the NER model could not be loaded (e.g. no network on first run,
# since the model is downloaded from HuggingFace Hub the first time it's
# used) -- surfaced via ner_is_available() so anonymize.py can add a visible
# data_limitations entry instead of silently falling back to regex-only,
# which is exactly the kind of silent gap that prompted this rewrite.
_ner_pipeline = None
_ner_load_failed = False

_NER_MODEL_NAME = "dslim/bert-base-NER"
_NER_ENTITY_LABELS = {"PER": "PERSON", "LOC": "LOCATION"}


def _get_ner_pipeline():
    global _ner_pipeline, _ner_load_failed
    if _ner_pipeline is not None or _ner_load_failed:
        return _ner_pipeline
    try:
        from transformers import pipeline

        _ner_pipeline = pipeline("token-classification", model=_NER_MODEL_NAME, aggregation_strategy="simple")
    except Exception:
        logger.warning("Could not load PII NER model %s; falling back to regex-only redaction", _NER_MODEL_NAME)
        _ner_load_failed = True
    return _ner_pipeline


def ner_is_available() -> bool:
    """True once the NER model has been loaded successfully. Returns False
    (without forcing a load attempt) before first use, so a case submitted
    before any redaction has run yet doesn't trigger an avoidable load just
    to answer this -- callers that need an authoritative answer should call
    redact_text() first, which always attempts to load it as needed."""
    return _ner_pipeline is not None


def reset_ner_pipeline_for_testing() -> None:
    """Test-only hook to force a fresh load attempt (e.g. after
    monkeypatching the model name or simulating a load failure)."""
    global _ner_pipeline, _ner_load_failed
    _ner_pipeline = None
    _ner_load_failed = False


_URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Every separator is optional, not just the leading ones -- a bare digit
# run like "5551234567" or "15551234567" (no punctuation at all, exactly
# the format someone fires off mid-chat) previously matched none of the
# four patterns here and passed straight into redacted_content untouched.
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,2}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\w)")
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{2,}")
# US-style street address: a house number, 1-5 words of street name, a
# street-type suffix, and an optional ", City, ST ZIP" tail. Run before NER
# below so NER doesn't try to tag fragments of an already-structured address
# as a PERSON/LOCATION entity in its own right.
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Za-z0-9'.-]+\s+){1,5}"
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Lane|Ln|Road|Rd|Way|"
    r"Court|Ct|Place|Pl|Circle|Cir|Terrace|Ter|Highway|Hwy)\.?"
    r"(?:\s*,?\s*[A-Za-z\s]{2,20},?\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?)?",
    re.IGNORECASE,
)

# Order matters: URL/EMAIL consume '@'/'.' before HANDLE could misfire on
# them, and ADDRESS runs before the NER pass above it relies on.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("URL", _URL_RE),
    ("EMAIL", _EMAIL_RE),
    ("PHONE", _PHONE_RE),
    ("HANDLE", _HANDLE_RE),
    ("ADDRESS", _ADDRESS_RE),
)


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    categories_found: tuple[str, ...]


def redact_text(text: str) -> RedactionResult:
    """Finds every span to redact -- regex matches and NER entities alike --
    against the ORIGINAL text first, then substitutes all of them in a
    single pass. Running NER on already-regex-substituted text (e.g. after
    "a@b.com" became "[REDACTED_EMAIL]") let the model occasionally tag a
    fragment of that placeholder marker itself as a false-positive entity,
    producing a corrupted, nested "[REDACTED_[REDACTED_LOCATION]MAIL]" --
    caught by a regression test, fixed by never feeding NER anything this
    function has already rewritten."""
    spans: list[tuple[int, int, str]] = []
    for label, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), label))

    ner = _get_ner_pipeline()
    if ner is not None:
        for entity in ner(text):
            if entity["entity_group"] in _NER_ENTITY_LABELS:
                spans.append((entity["start"], entity["end"], _NER_ENTITY_LABELS[entity["entity_group"]]))

    if not spans:
        return RedactionResult(redacted_text=text, categories_found=())

    # Stable sort by start position only: regex spans were appended before
    # NER spans above, so a regex/NER tie at the same start keeps the regex
    # label -- structured regex matches (e.g. a full street address) are
    # more reliable than an NER span that might only cover part of the same
    # text, so regex should win any overlap, not just an exact-start tie.
    spans.sort(key=lambda s: s[0])

    selected: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, label in spans:
        if start < last_end:
            continue  # overlaps a previously-selected, higher-priority span
        selected.append((start, end, label))
        last_end = end

    pieces: list[str] = []
    cursor = 0
    for start, end, label in selected:
        pieces.append(text[cursor:start])
        pieces.append(f"[REDACTED_{label}]")
        cursor = end
    pieces.append(text[cursor:])

    # Unique categories present, not one entry per occurrence -- matches the
    # original contract (e.g. two phone numbers in one message still report
    # categories_found == ("PHONE",), not ("PHONE", "PHONE")).
    found: list[str] = []
    for _, _, label in selected:
        if label not in found:
            found.append(label)

    return RedactionResult(redacted_text="".join(pieces), categories_found=tuple(found))
