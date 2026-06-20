from datetime import datetime, timedelta

from backend.preprocessing.anonymize import RawMessage, prepare_conversation


def test_empty_conversation_flagged():
    result = prepare_conversation([])
    assert result.messages == ()
    assert "empty_conversation" in result.data_limitations


def test_speaker_remap_is_local_and_consistent():
    raw = [
        RawMessage(speaker_external_id="UserA", text="hi"),
        RawMessage(speaker_external_id="UserB", text="hello"),
        RawMessage(speaker_external_id="UserA", text="how are you"),
    ]
    result = prepare_conversation(raw)
    assert [m.speaker_local_id for m in result.messages] == ["SPEAKER_A", "SPEAKER_B", "SPEAKER_A"]


def test_missing_timestamps_uses_submission_order_and_flags_limitation():
    raw = [
        RawMessage(speaker_external_id="A", text="first"),
        RawMessage(speaker_external_id="B", text="second"),
    ]
    result = prepare_conversation(raw)
    assert [m.message_sequence for m in result.messages] == [0, 1]
    assert [m.relative_time for m in result.messages] == [0.0, 1.0]
    assert "missing_timestamps_used_submission_order" in result.data_limitations


def test_timestamps_reorder_and_produce_relative_seconds():
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    raw = [
        RawMessage(speaker_external_id="A", text="second", timestamp=t0 + timedelta(seconds=30)),
        RawMessage(speaker_external_id="B", text="first", timestamp=t0),
    ]
    result = prepare_conversation(raw)
    assert [m.redacted_content for m in result.messages] == ["first", "second"]
    assert [m.relative_time for m in result.messages] == [0.0, 30.0]
    assert "missing_timestamps_used_submission_order" not in result.data_limitations


def test_single_speaker_flagged():
    raw = [RawMessage(speaker_external_id="A", text="hi"), RawMessage(speaker_external_id="A", text="hello")]
    result = prepare_conversation(raw)
    assert "single_speaker_conversation" in result.data_limitations


def test_redaction_applied_per_message():
    raw = [RawMessage(speaker_external_id="A", text="email me at a@b.com")]
    result = prepare_conversation(raw)
    assert result.messages[0].redacted_content == "email me at [REDACTED_EMAIL]"
    assert result.messages[0].redaction_categories == ("EMAIL",)


def test_blank_message_becomes_none():
    raw = [RawMessage(speaker_external_id="A", text="   ")]
    result = prepare_conversation(raw)
    assert result.messages[0].redacted_content is None