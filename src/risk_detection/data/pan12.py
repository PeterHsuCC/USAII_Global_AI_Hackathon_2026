"""PAN12 dataset loader.

Reads the "datapack" JSON produced by ``data/PAN12/create_datapack.py``
(ported from the eSPD-datasets pipeline) plus the PAN12 Problem 2 ground
truth (per-line "suspicious" annotations) and exposes them as structured,
serializable objects that plug into the existing ``Message`` /
``ConversationWindow`` / ``RuleSignalExtractor`` / ``LLMSafetySignalExtractor``
pipeline.

ChatCoder2 and PANC are intentionally out of scope: PANC is built from CC2
positive chats + PAN12 negative chats, and this project does not have CC2
access (manual application required). PAN12/VTPAN are used standalone.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from ..conversation import ConversationWindow, Message

GROOMING_LABEL_SOURCE = "pan12_predator_identity"
DEFAULT_DATAPACK_ID = "PAN12"
PROBLEM2_FILENAME = "pan12-sexual-predator-identification-groundtruth-problem2.txt"


def default_pan12_dir() -> Path:
    """Best-effort default location for the locally-generated PAN12 datapacks
    (see data/README.md). Override the ``pan12_dir`` argument of `load_split`
    when the dataset lives elsewhere."""

    return Path(__file__).resolve().parents[3] / "data" / "PAN12"


def parse_problem2(path: Path) -> dict[str, list[int]]:
    """Parse the PAN12 "Problem 2" ground truth file.

    Each line is ``<conversation_id>\\t<line_number>``, marking one message
    inside that conversation as evidence of predatory behavior. A
    conversation may have several suspicious lines. Returns
    conversation_id -> sorted, deduplicated list of 1-based line numbers
    (matching ``PAN12Message.message_index``).
    """

    suspicious: dict[str, list[int]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conversation_id, line_num = line.split("\t")
            suspicious[conversation_id].append(int(line_num))
    return {cid: sorted(set(nums)) for cid, nums in suspicious.items()}


def load_predator_id_list(path: Path) -> list[str]:
    """Read a one-author-id-per-line PAN12 predator list file."""

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def overlapping_predator_ids(
    train_predator_ids: Iterable[str], test_predator_ids: Iterable[str]
) -> set[str]:
    """Author ids flagged as predators in *both* PAN12 official splits.

    PAN12's official train/test predator lists are not strictly
    identity-disjoint (2 ids overlap as of the 2012-05-01 release). Use this
    to build a stricter identity-disjoint evaluation subset alongside the
    official split, rather than silently trusting the official split for
    unseen-author generalization claims.
    """

    return set(train_predator_ids) & set(test_predator_ids)


@dataclass(frozen=True)
class PAN12Message:
    """One message inside a PAN12 conversation.

    Keeps full provenance, including the persistent PAN12 author_id; see
    `ConversationSample.to_window` for how this gets remapped to a
    window-local placeholder before reaching the encoder.
    """

    conversation_id: str
    message_id: str
    message_index: int  # 1-based PAN12 line number
    author_id: str
    text: str
    timestamp_ms: float
    relative_time: float  # seconds since this conversation's first message
    is_predator_author: bool
    is_problem2_evidence: bool


@dataclass(frozen=True)
class PAN12Conversation:
    """A full PAN12 conversation with PAN12-derived weak labels attached.

    `conversation_label` is derived from author identity (PAN12's official
    predator list) -- it is a *derived weak conversation label*, not a
    human judgement about any specific message (see `label_source`).
    `first_suspicious_index`/`suspicious_message_indices` come from the
    Problem 2 ground truth where available and are a *weak onset proxy*,
    not a complete grooming-phase annotation: Problem 2 marks representative
    evidence lines, not necessarily the earliest point at which risk was
    actually present, and a positive conversation without a Problem 2 entry
    is not necessarily free of early signal.
    """

    conversation_id: str
    official_split: str  # "train" | "test"
    conversation_label: int  # 1 if any author in the conversation is a listed predator
    label_source: str
    predator_ids: tuple[str, ...]
    messages: tuple[PAN12Message, ...]
    has_problem2_annotation: bool
    suspicious_message_indices: tuple[int, ...]
    first_suspicious_index: int | None

    def __len__(self) -> int:
        return len(self.messages)


def _build_conversation(
    conversation_id: str,
    chat: Mapping,
    official_split: str,
    suspicious_by_conversation: Mapping[str, list[int]],
) -> PAN12Conversation:
    suspicious_indices = tuple(suspicious_by_conversation.get(conversation_id, ()))
    suspicious_set = set(suspicious_indices)

    # DynamicArray serializes to a list indexed by lineNum-1; lines dropped
    # during datapack creation (missing author/body/time) leave `None` gaps.
    raw_messages = [m for m in chat["content"] if m is not None]
    first_ts = raw_messages[0]["time"] if raw_messages else 0.0

    messages: list[PAN12Message] = []
    predator_ids: set[str] = set()
    for m in raw_messages:
        if m["isFromPredator"]:
            predator_ids.add(m["author"])
        messages.append(
            PAN12Message(
                conversation_id=conversation_id,
                message_id=f"{conversation_id}_{m['lineNum']}",
                message_index=m["lineNum"],
                author_id=m["author"],
                text=m["body"],
                timestamp_ms=m["time"],
                # PAN12 timestamps only carry time-of-day (no date), anchored
                # to 1970-01-01: a conversation spanning midnight can make
                # this go non-monotonic. Kept as-is rather than guessed at.
                relative_time=(m["time"] - first_ts) / 1000.0,
                is_predator_author=m["isFromPredator"],
                is_problem2_evidence=m["lineNum"] in suspicious_set,
            )
        )

    return PAN12Conversation(
        conversation_id=conversation_id,
        official_split=official_split,
        conversation_label=int(chat["className"] == "predator"),
        label_source=GROOMING_LABEL_SOURCE,
        predator_ids=tuple(sorted(predator_ids)),
        messages=tuple(messages),
        has_problem2_annotation=bool(suspicious_indices),
        suspicious_message_indices=suspicious_indices,
        first_suspicious_index=min(suspicious_indices) if suspicious_indices else None,
    )


def iter_conversations(
    datapack_path: Path,
    official_split: str,
    suspicious_by_conversation: Mapping[str, list[int]] | None = None,
) -> Iterator[PAN12Conversation]:
    """Stream `PAN12Conversation` objects from a generated datapack JSON.

    Loads the whole JSON file into memory at once (the datapacks are
    ~200-450MB on disk; budget a few GB of RAM), but yields conversations
    one at a time so callers aren't forced to also hold the full
    materialized object graph.
    """

    suspicious_by_conversation = suspicious_by_conversation or {}
    with open(datapack_path, "r", encoding="utf-8") as f:
        datapack = json.load(f)
    for conversation_id, chat in datapack["chats"].items():
        yield _build_conversation(conversation_id, chat, official_split, suspicious_by_conversation)


def load_split(
    pan12_dir: Path,
    official_split: str,
    datapack_id: str = DEFAULT_DATAPACK_ID,
    problem2_filename: str = PROBLEM2_FILENAME,
) -> Iterator[PAN12Conversation]:
    """Convenience entry point: locate the datapack + Problem 2 file under
    `pan12_dir` (see `default_pan12_dir`) and stream conversations for one
    official split ("train" or "test")."""

    datapack_path = pan12_dir / "datapacks" / f"datapack-{datapack_id}-{official_split}.json"
    problem2_path = pan12_dir / "raw_dataset" / problem2_filename
    suspicious = parse_problem2(problem2_path) if problem2_path.exists() else {}
    return iter_conversations(datapack_path, official_split, suspicious)


@dataclass(frozen=True)
class ConversationSample:
    """One trainable unit: a slice of a `PAN12Conversation` plus its weak
    labels. `rule_features`/`llm_features` are left for the caller to fill
    in later (via `RuleSignalExtractor` / a cached `LLMSafetySignalExtractor`)
    so the same samples can back the Text-only / Text+Rules / Linear Hybrid
    Grooming ablations without building three separate datasets.
    """

    conversation_id: str
    official_split: str
    window_kind: str  # "full" | "fixed" | "prefix"
    window_start: int  # 0-based position into the conversation's messages
    window_end: int  # exclusive
    messages: tuple[PAN12Message, ...]
    label: int
    label_source: str
    contains_suspicious_evidence: bool
    first_suspicious_index: int | None
    rule_features: tuple[float, ...] | None = None
    llm_features: tuple[float, ...] | None = None

    def to_window(self) -> ConversationWindow:
        """Builds the encoder-facing window with PAN12's persistent
        global author hashes remapped to window-local placeholders
        (SPEAKER_A, SPEAKER_B, ...) by order of first appearance, so the
        encoder never sees a persistent identity it could memorize across
        conversations (Section 2)."""
        speaker_map = _local_speaker_map(self.messages)
        window = ConversationWindow(k=max(1, len(self.messages)))
        for m in self.messages:
            window.add(
                Message(
                    speaker_id=speaker_map[m.author_id],
                    text=m.text,
                    relative_time=m.relative_time,
                )
            )
        return window


def _local_speaker_map(messages: Sequence["PAN12Message"]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for m in messages:
        if m.author_id not in mapping:
            mapping[m.author_id] = _speaker_placeholder(len(mapping))
    return mapping


def _speaker_placeholder(index: int) -> str:
    if index < 26:
        return f"SPEAKER_{chr(ord('A') + index)}"
    return f"SPEAKER_{index}"  # PAN12 conversations essentially never have >26 authors


def _make_sample(
    conversation: PAN12Conversation, start: int, end: int, window_kind: str
) -> ConversationSample:
    window_messages = conversation.messages[start:end]
    contains_evidence = any(m.is_problem2_evidence for m in window_messages)
    return ConversationSample(
        conversation_id=conversation.conversation_id,
        official_split=conversation.official_split,
        window_kind=window_kind,
        window_start=start,
        window_end=end,
        messages=window_messages,
        label=conversation.conversation_label,
        label_source=conversation.label_source,
        contains_suspicious_evidence=contains_evidence,
        first_suspicious_index=conversation.first_suspicious_index,
    )


def full_conversation_samples(
    conversations: Iterable[PAN12Conversation],
) -> Iterator[ConversationSample]:
    """One sample per conversation, covering every message. The phase-1
    Grooming Head ablations (Text-only / Text+Rules / Linear Hybrid) use
    this or `fixed_window_samples`."""

    for conv in conversations:
        if not conv.messages:
            continue
        yield _make_sample(conv, 0, len(conv.messages), "full")


def fixed_window_samples(
    conversations: Iterable[PAN12Conversation],
    window_size: int,
    stride: int | None = None,
) -> Iterator[ConversationSample]:
    """Fixed-size windows with a configurable stride (defaults to
    non-overlapping). Conversations shorter than `window_size` yield a
    single short window covering everything they have."""

    stride = stride or window_size
    for conv in conversations:
        n = len(conv.messages)
        if n == 0:
            continue
        if n <= window_size:
            yield _make_sample(conv, 0, n, "fixed")
            continue
        for start in range(0, n - window_size + 1, stride):
            yield _make_sample(conv, start, start + window_size, "fixed")


def prefix_window_samples(
    conversations: Iterable[PAN12Conversation],
    fractions: Sequence[float] = (0.1, 0.2, 0.3),
) -> Iterator[ConversationSample]:
    """Prefix windows at given fractions of conversation length (at least 1
    message each). Not used for Grooming Head training -- kept for the
    future weakly-supervised Early Detection ablation built on the Problem 2
    onset proxy (see `PAN12Conversation` docstring for its limits)."""

    for conv in conversations:
        n = len(conv.messages)
        if n == 0:
            continue
        for frac in fractions:
            end = max(1, round(n * frac))
            yield _make_sample(conv, 0, end, "prefix")
