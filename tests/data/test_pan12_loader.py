import json

from risk_detection.data import (
    fixed_window_samples,
    full_conversation_samples,
    iter_conversations,
    load_predator_id_list,
    overlapping_predator_ids,
    parse_problem2,
    prefix_window_samples,
)
from risk_detection.data.pan12 import GROOMING_LABEL_SOURCE
from risk_detection.signals.rules import RuleSignalExtractor


def _msg(line, author, body, time_ms, is_predator):
    return {
        "type": "message",
        "author": author,
        "body": body,
        "time": time_ms,
        "isFromPredator": is_predator,
        "labels": [],
        "lineNum": line,
    }


def _write_datapack(tmp_path, chats):
    path = tmp_path / "datapack-PAN12-train.json"
    path.write_text(json.dumps({"chats": chats}), encoding="utf-8")
    return path


def test_parse_problem2(tmp_path):
    path = tmp_path / "problem2.txt"
    path.write_text("conv1\t3\nconv1\t1\nconv1\t3\nconv2\t7\n", encoding="utf-8")

    result = parse_problem2(path)

    assert result == {"conv1": [1, 3], "conv2": [7]}


def test_load_predator_id_list(tmp_path):
    path = tmp_path / "predators.txt"
    path.write_text("aaa\nbbb\n\n", encoding="utf-8")

    assert load_predator_id_list(path) == ["aaa", "bbb"]


def test_overlapping_predator_ids():
    assert overlapping_predator_ids(["a", "b"], ["b", "c"]) == {"b"}


def test_iter_conversations_builds_labels_and_skips_none_gaps(tmp_path):
    chats = {
        "predator_chat": {
            "className": "predator",
            "content": [
                _msg(1, "victim", "hi", 1000.0, False),
                None,  # DynamicArray gap (line dropped during datapack creation)
                _msg(3, "groomer", "send a pic", 3000.0, True),
            ],
        },
        "clean_chat": {
            "className": "non-predator",
            "content": [
                _msg(1, "a", "hey", 5000.0, False),
                _msg(2, "b", "sup", 5500.0, False),
            ],
        },
    }
    datapack_path = _write_datapack(tmp_path, chats)
    suspicious = {"predator_chat": [3]}

    conversations = {c.conversation_id: c for c in iter_conversations(datapack_path, "train", suspicious)}

    predator_conv = conversations["predator_chat"]
    assert predator_conv.conversation_label == 1
    assert predator_conv.label_source == GROOMING_LABEL_SOURCE
    assert predator_conv.predator_ids == ("groomer",)
    assert len(predator_conv.messages) == 2  # None gap skipped
    assert predator_conv.has_problem2_annotation is True
    assert predator_conv.suspicious_message_indices == (3,)
    assert predator_conv.first_suspicious_index == 3

    first, second = predator_conv.messages
    assert first.relative_time == 0.0
    assert second.relative_time == 2.0  # (3000 - 1000) ms -> 2.0s
    assert first.is_problem2_evidence is False
    assert second.is_problem2_evidence is True
    assert second.message_index == 3
    assert second.message_id == "predator_chat_3"

    clean_conv = conversations["clean_chat"]
    assert clean_conv.conversation_label == 0
    assert clean_conv.predator_ids == ()
    assert clean_conv.has_problem2_annotation is False
    assert clean_conv.first_suspicious_index is None


def test_full_conversation_samples(tmp_path):
    chats = {
        "c1": {
            "className": "predator",
            "content": [_msg(1, "a", "hi", 0.0, False), _msg(2, "b", "bye", 100.0, True)],
        }
    }
    conversations = list(iter_conversations(_write_datapack(tmp_path, chats), "train"))

    samples = list(full_conversation_samples(conversations))

    assert len(samples) == 1
    sample = samples[0]
    assert sample.window_kind == "full"
    assert sample.window_start == 0
    assert sample.window_end == 2
    assert sample.label == 1
    assert len(sample.messages) == 2
    assert sample.rule_features is None
    assert sample.llm_features is None


def test_fixed_window_samples_short_and_long_conversations(tmp_path):
    short_chat = {
        "className": "non-predator",
        "content": [_msg(i + 1, "a", f"m{i}", float(i), False) for i in range(1)],
    }
    long_chat = {
        "className": "non-predator",
        "content": [_msg(i + 1, "a", f"m{i}", float(i), False) for i in range(5)],
    }
    conversations = list(
        iter_conversations(_write_datapack(tmp_path, {"short": short_chat, "long": long_chat}), "train")
    )

    samples = list(fixed_window_samples(conversations, window_size=2, stride=2))
    by_conv = {}
    for s in samples:
        by_conv.setdefault(s.conversation_id, []).append((s.window_start, s.window_end))

    assert by_conv["short"] == [(0, 1)]  # shorter than window_size -> single full window
    assert by_conv["long"] == [(0, 2), (2, 4)]  # stride=2, last partial window dropped


def test_prefix_window_samples(tmp_path):
    chat = {
        "className": "predator",
        "content": [_msg(i + 1, "a", f"m{i}", float(i), False) for i in range(10)],
    }
    conversations = list(iter_conversations(_write_datapack(tmp_path, {"c": chat}), "train"))

    samples = list(prefix_window_samples(conversations, fractions=(0.1, 0.5)))

    ends = sorted(s.window_end for s in samples)
    assert ends == [1, 5]
    assert all(s.window_start == 0 for s in samples)


def test_sample_to_window_feeds_existing_rule_extractor(tmp_path):
    chat = {
        "className": "predator",
        "content": [
            _msg(1, "a", "hi there", 0.0, False),
            _msg(2, "b", "don't tell your parents, add me on snapchat", 10.0, True),
        ],
    }
    conversations = list(iter_conversations(_write_datapack(tmp_path, {"c": chat}), "train"))
    sample = next(full_conversation_samples(conversations))

    window = sample.to_window()
    signals = RuleSignalExtractor().extract(window)

    assert signals.secret_request is True
    assert signals.contact_migration is True
    assert signals.age_reference is False


def test_to_window_remaps_speaker_ids_to_local_placeholders(tmp_path):
    chat = {
        "className": "non-predator",
        "content": [
            _msg(1, "0158d0d6781fc4d493f243d4caa49747", "hi", 0.0, False),
            _msg(2, "97964e7a9e8eb9cf78f2e4d7b2ff34c7", "hey", 1.0, False),
            _msg(3, "0158d0d6781fc4d493f243d4caa49747", "what's up", 2.0, False),
        ],
    }
    conversations = list(iter_conversations(_write_datapack(tmp_path, {"c": chat}), "train"))
    sample = next(full_conversation_samples(conversations))

    window = sample.to_window()
    speaker_ids = [m.speaker_id for m in window]

    # First-appearance order: the first author seen becomes SPEAKER_A, the
    # second SPEAKER_B; PAN12's persistent global author hash never reaches
    # the window (Section 2).
    assert speaker_ids == ["SPEAKER_A", "SPEAKER_B", "SPEAKER_A"]
    assert all(not sid.startswith("0158") and not sid.startswith("97964") for sid in speaker_ids)


def test_to_window_remapping_is_independent_per_conversation(tmp_path):
    # Conversation "c2"'s first author happens to be the same PAN12 hash as
    # "c1"'s second author -- each window's remapping must be local to that
    # window, not a global hash-to-placeholder table.
    chats = {
        "c1": {
            "className": "non-predator",
            "content": [_msg(1, "author_x", "hi", 0.0, False), _msg(2, "author_y", "hey", 1.0, False)],
        },
        "c2": {
            "className": "non-predator",
            "content": [_msg(1, "author_y", "yo", 0.0, False), _msg(2, "author_x", "sup", 1.0, False)],
        },
    }
    conversations = {c.conversation_id: c for c in iter_conversations(_write_datapack(tmp_path, chats), "train")}
    samples = {s.conversation_id: s for s in full_conversation_samples(conversations.values())}

    speakers_c1 = [m.speaker_id for m in samples["c1"].to_window()]
    speakers_c2 = [m.speaker_id for m in samples["c2"].to_window()]

    assert speakers_c1 == ["SPEAKER_A", "SPEAKER_B"]
    assert speakers_c2 == ["SPEAKER_A", "SPEAKER_B"]
