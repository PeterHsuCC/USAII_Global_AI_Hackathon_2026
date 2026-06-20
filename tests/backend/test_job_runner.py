from backend.model_runtime.job_runner import JobMessage, risk_level_from_score, run_analysis


def test_run_analysis_with_stub_components():
    messages = [
        JobMessage(speaker_local_id="SPEAKER_A", redacted_content="our little secret ok?", message_sequence=0),
        JobMessage(speaker_local_id="SPEAKER_B", redacted_content="ok i promise", message_sequence=1),
        JobMessage(speaker_local_id="SPEAKER_A", redacted_content="add me on snapchat", message_sequence=2),
    ]
    outcome = run_analysis(messages, window_size=5)

    assert 0.0 <= outcome.result.overall_score.item() <= 1.0
    assert outcome.window_message_sequences == (0, 1, 2)
    assert outcome.rule_evidence.triggered_message_indices["secret_request"] == [0]
    assert outcome.rule_evidence.triggered_message_indices["contact_migration"] == [2]


def test_run_analysis_only_uses_last_window_size_messages():
    messages = [
        JobMessage(speaker_local_id="A", redacted_content="first", message_sequence=0),
        JobMessage(speaker_local_id="B", redacted_content="second", message_sequence=1),
        JobMessage(speaker_local_id="A", redacted_content="third", message_sequence=2),
    ]
    outcome = run_analysis(messages, window_size=2)
    assert outcome.window_message_sequences == (1, 2)


def test_run_analysis_handles_none_content():
    messages = [
        JobMessage(speaker_local_id="A", redacted_content=None, message_sequence=0),
        JobMessage(speaker_local_id="B", redacted_content="hello", message_sequence=1),
    ]
    outcome = run_analysis(messages, window_size=5)
    assert outcome.window_message_sequences == (0, 1)


def test_risk_level_thresholds():
    assert risk_level_from_score(0.9) == "critical"
    assert risk_level_from_score(0.6) == "high"
    assert risk_level_from_score(0.3) == "medium"
    assert risk_level_from_score(0.1) == "low"