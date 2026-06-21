from backend.model_runtime.job_runner import JobMessage, build_windows, risk_level_from_score, run_analysis


def test_run_analysis_with_stub_components():
    messages = [
        JobMessage(speaker_local_id="SPEAKER_A", redacted_content="our little secret ok?", message_sequence=0),
        JobMessage(speaker_local_id="SPEAKER_B", redacted_content="ok i promise", message_sequence=1),
        JobMessage(speaker_local_id="SPEAKER_A", redacted_content="add me on snapchat", message_sequence=2),
    ]
    outcome = run_analysis(messages, window_size=5)

    assert 0.0 <= outcome.result.overall_score.item() <= 1.0
    assert outcome.window_count == 1
    assert outcome.window_message_sequences == (0, 1, 2)
    assert outcome.analyzed_message_sequences == (0, 1, 2)
    assert outcome.rule_evidence.triggered_message_indices["secret_request"] == [0]
    assert outcome.rule_evidence.triggered_message_indices["contact_migration"] == [2]


def test_build_windows_splits_non_overlapping_when_over_window_size():
    messages = [JobMessage(speaker_local_id="A", redacted_content=f"m{i}", message_sequence=i) for i in range(30)]
    windows = build_windows(messages, window_size=12)

    assert len(windows) == 3
    assert [seqs for _, seqs in windows] == [
        tuple(range(0, 12)),
        tuple(range(12, 24)),
        tuple(range(24, 30)),
    ]


def test_build_windows_single_window_when_under_window_size():
    messages = [JobMessage(speaker_local_id="A", redacted_content="hi", message_sequence=0)]
    windows = build_windows(messages, window_size=12)

    assert len(windows) == 1
    assert windows[0][1] == (0,)


def test_run_analysis_covers_every_message_across_multiple_windows():
    """A case over window_size must now be split into multiple windows and
    fully covered -- not silently dropped to just the last window_size
    messages (the old single-shot behavior)."""
    messages = [
        JobMessage(speaker_local_id="A", redacted_content="first", message_sequence=0),
        JobMessage(speaker_local_id="B", redacted_content="second", message_sequence=1),
        JobMessage(speaker_local_id="A", redacted_content="third", message_sequence=2),
    ]
    outcome = run_analysis(messages, window_size=2)

    assert outcome.window_count == 2
    assert outcome.analyzed_message_sequences == (0, 1, 2)


def test_run_analysis_discloses_multi_window_split():
    messages = [
        JobMessage(speaker_local_id="A", redacted_content=f"message {i}", message_sequence=i) for i in range(5)
    ]
    outcome = run_analysis(messages, window_size=2)

    assert outcome.window_count == 3
    assert any("split into 3 sequential 2-message windows" in limitation for limitation in outcome.extra_limitations)


def test_run_analysis_does_not_disclose_split_for_a_single_window():
    messages = [
        JobMessage(speaker_local_id="A", redacted_content="first", message_sequence=0),
        JobMessage(speaker_local_id="B", redacted_content="second", message_sequence=1),
    ]
    outcome = run_analysis(messages, window_size=5)

    assert outcome.window_count == 1
    assert not any("split into" in limitation for limitation in outcome.extra_limitations)


def test_run_analysis_merges_rule_evidence_across_windows():
    """secret_request lands in window 1 (position 0 -> sequence 0);
    image_request lands in window 2 (position 1 -> sequence 3). Each
    window-local position must be translated via THAT window's own
    sequence mapping before merging, not treated as a shared index space."""
    messages = [
        JobMessage(speaker_local_id="A", redacted_content="this has to be our little secret ok?", message_sequence=0),
        JobMessage(speaker_local_id="B", redacted_content="ok i promise", message_sequence=1),
        JobMessage(speaker_local_id="A", redacted_content="hello", message_sequence=2),
        JobMessage(speaker_local_id="B", redacted_content="can you send me a pic of yourself", message_sequence=3),
    ]
    outcome = run_analysis(messages, window_size=2)

    assert outcome.window_count == 2
    assert outcome.rule_evidence.triggered_message_indices["secret_request"] == [0]
    assert outcome.rule_evidence.triggered_message_indices["image_request"] == [3]
    assert outcome.safety_features.rule_signals.secret_request is True
    assert outcome.safety_features.rule_signals.image_request is True


def test_run_analysis_ors_human_review_required_across_windows():
    """A severe rule trigger (threat_phrase) in an early window must flag
    the whole case for review even if a later window -- whichever one ends
    up "representative" by overall_score -- has no rule triggers of its
    own. Section 13's rule-based review condition is not latched like
    Warningt, so job_runner.py must OR it across windows itself."""
    messages = [
        JobMessage(speaker_local_id="A", redacted_content="i am going to kill you", message_sequence=0),
        JobMessage(speaker_local_id="B", redacted_content="please stop", message_sequence=1),
        JobMessage(speaker_local_id="A", redacted_content="hello", message_sequence=2),
        JobMessage(speaker_local_id="B", redacted_content="how are you", message_sequence=3),
    ]
    outcome = run_analysis(messages, window_size=2)

    assert outcome.window_count == 2
    assert outcome.result.human_review_required is True


def test_run_analysis_discloses_message_exceeding_encoder_token_limit():
    """The stub encoder's max_position_embeddings is 32 (model_runtime/
    loader.py); a message that fragments into more stub tokens than that
    must be disclosed as partially analyzed by the text/emotion encoders,
    even though the LLM/rule branches still saw it in full."""
    long_message = "this is a long message " * 20  # fragments into well over 32 stub tokens
    messages = [JobMessage(speaker_local_id="A", redacted_content=long_message, message_sequence=0)]

    outcome = run_analysis(messages, window_size=5)

    assert any("exceeded the trained text encoder's" in limitation for limitation in outcome.extra_limitations)


def test_run_analysis_does_not_disclose_token_overflow_for_short_messages():
    messages = [JobMessage(speaker_local_id="A", redacted_content="hi", message_sequence=0)]
    outcome = run_analysis(messages, window_size=5)

    assert not any("exceeded the trained text encoder's" in limitation for limitation in outcome.extra_limitations)


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
