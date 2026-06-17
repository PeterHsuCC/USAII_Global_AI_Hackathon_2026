from _tiny_emotion_classifier import make_tiny_emotion_classifier

from risk_detection import ConversationWindow, Message
from risk_detection.model import GoEmotionsClassifier


def _make_classifier() -> GoEmotionsClassifier:
    tokenizer, model = make_tiny_emotion_classifier(hidden_size=8)
    return GoEmotionsClassifier(tokenizer=tokenizer, encoder=model)


def test_neutral_label_is_excluded():
    classifier = _make_classifier()

    assert "neutral" not in classifier.label_names
    assert classifier.d_g == 8  # 9 labels in the checkpoint, minus neutral


def test_label_to_index_covers_every_label_name():
    classifier = _make_classifier()

    assert set(classifier.label_to_index) == {name.lower() for name in classifier.label_names}
    assert classifier.label_to_index["fear"] < classifier.d_g


def test_forward_returns_probabilities_in_unit_interval():
    classifier = _make_classifier()

    g = classifier(["i am scared", "i love you"])

    assert g.shape == (2, classifier.d_g)
    assert (g >= 0).all() and (g <= 1).all()


def test_encode_window_matches_message_count():
    classifier = _make_classifier()
    window = ConversationWindow(k=3)
    window.add(Message(speaker_id="a", text="i am scared", relative_time=0.0))
    window.add(Message(speaker_id="b", text="i care about you", relative_time=1.0))

    g = classifier.encode_window(window)

    assert g.shape == (2, classifier.d_g)


def test_encode_empty_window_returns_empty_tensor():
    classifier = _make_classifier()
    window = ConversationWindow(k=3)

    g = classifier.encode_window(window)

    assert g.shape == (0, classifier.d_g)
