"""Speaker-local remapping, ordering, relative-time conversion, and
completeness checks (v6 Section 14 / Section 15.8).

Runs once at case submission, before the analysis job is queued -- per
Section 3's diagram, AI preprocessing/PII redaction happens before the
durable job queue, not inside the worker.
"""

import string
from dataclasses import dataclass
from datetime import datetime

from backend.preprocessing.pii_redaction import redact_text

_ALPHABET = string.ascii_uppercase


@dataclass(frozen=True)
class RawMessage:
    speaker_external_id: str
    text: str
    timestamp: datetime | None = None


@dataclass(frozen=True)
class PreparedMessage:
    message_sequence: int
    speaker_local_id: str
    redacted_content: str | None
    relative_time: float
    redaction_categories: tuple[str, ...]


@dataclass(frozen=True)
class PreparationResult:
    messages: tuple[PreparedMessage, ...]
    data_limitations: tuple[str, ...]


def _local_speaker_id(index: int) -> str:
    """0 -> SPEAKER_A, 25 -> SPEAKER_Z, 26 -> SPEAKER_AA, ..."""
    letters: list[str] = []
    n = index
    while True:
        n, rem = divmod(n, 26)
        letters.append(_ALPHABET[rem])
        if n == 0:
            break
        n -= 1
    return "SPEAKER_" + "".join(reversed(letters))


def prepare_conversation(raw_messages: list[RawMessage]) -> PreparationResult:
    if not raw_messages:
        return PreparationResult(messages=(), data_limitations=("empty_conversation",))

    limitations: list[str] = []

    has_timestamps = all(m.timestamp is not None for m in raw_messages)
    if has_timestamps:
        ordered = sorted(raw_messages, key=lambda m: m.timestamp)  # type: ignore[arg-type, return-value]
        base_time = ordered[0].timestamp
    else:
        limitations.append("missing_timestamps_used_submission_order")
        ordered = raw_messages
        base_time = None

    speaker_map: dict[str, str] = {}
    prepared: list[PreparedMessage] = []

    for sequence, raw in enumerate(ordered):
        local_id = speaker_map.setdefault(raw.speaker_external_id, _local_speaker_id(len(speaker_map)))

        redaction = redact_text(raw.text)
        content = redaction.redacted_text if redaction.redacted_text.strip() else None

        relative_time = (raw.timestamp - base_time).total_seconds() if has_timestamps else float(sequence)  # type: ignore[operator]

        prepared.append(
            PreparedMessage(
                message_sequence=sequence,
                speaker_local_id=local_id,
                redacted_content=content,
                relative_time=relative_time,
                redaction_categories=redaction.categories_found,
            )
        )

    if len(speaker_map) < 2:
        limitations.append("single_speaker_conversation")

    return PreparationResult(messages=tuple(prepared), data_limitations=tuple(limitations))