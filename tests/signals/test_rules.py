from risk_detection import ConversationWindow, Message, RuleSignalExtractor


def _window(*texts: str) -> ConversationWindow:
    window = ConversationWindow(k=len(texts) or 1)
    for i, text in enumerate(texts):
        window.add(Message(speaker_id="a", text=text, relative_time=float(i)))
    return window


def test_no_signals_on_benign_conversation():
    window = _window("hey how was school today", "it was fine, just homework")
    signals = RuleSignalExtractor().extract(window)

    assert signals.to_vector() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_detects_secret_request():
    window = _window("this has to be our little secret, ok?")
    signals = RuleSignalExtractor().extract(window)

    assert signals.secret_request is True


def test_detects_contact_migration():
    window = _window("add me on snapchat so we can talk more")
    signals = RuleSignalExtractor().extract(window)

    assert signals.contact_migration is True


def test_detects_age_reference():
    window = _window("how old are you anyway?")
    signals = RuleSignalExtractor().extract(window)

    assert signals.age_reference is True


def test_age_reference_requires_explicit_age_phrasing():
    window = _window("i'm 5 minutes away from your place")
    signals = RuleSignalExtractor().extract(window)

    assert signals.age_reference is False


def test_detects_image_request():
    window = _window("can you send me a pic of yourself")
    signals = RuleSignalExtractor().extract(window)

    assert signals.image_request is True


def test_detects_threat_phrase():
    window = _window("send it or else you'll regret it")
    signals = RuleSignalExtractor().extract(window)

    assert signals.threat_phrase is True


def test_evidence_records_every_triggering_message_for_a_rule():
    window = _window(
        "send me a pic",  # index 0: image_request
        "no thanks",  # index 1: nothing
        "come on just send a picture",  # index 2: image_request again
    )

    evidence = RuleSignalExtractor().extract_evidence(window)

    assert evidence.triggered_message_indices["image_request"] == [0, 2]
    assert evidence.triggered_message_indices["secret_request"] == []


def test_evidence_union_indices_combines_multiple_rules():
    window = _window(
        "our little secret ok?",  # index 0: secret_request
        "add me on snapchat",  # index 1: contact_migration
        "totally unrelated message",  # index 2: nothing
    )

    evidence = RuleSignalExtractor().extract_evidence(window)

    assert evidence.union_indices() == [0, 1]


def test_extract_is_consistent_with_extract_evidence():
    window = _window("our little secret ok?", "ok i promise")
    extractor = RuleSignalExtractor()

    signals = extractor.extract(window)
    evidence = extractor.extract_evidence(window)

    assert signals.secret_request == bool(evidence.triggered_message_indices["secret_request"])
