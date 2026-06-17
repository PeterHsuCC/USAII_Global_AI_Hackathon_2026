from _tiny_bert import make_tiny_bert

from risk_detection import ConversationWindow, Message
from risk_detection.model import MessageEncoder


def _make_encoder() -> MessageEncoder:
    tokenizer, model = make_tiny_bert(hidden_size=8)
    return MessageEncoder(tokenizer=tokenizer, encoder=model)


def test_encode_returns_one_vector_per_message():
    encoder = _make_encoder()
    h = encoder(["a", "b"], ["hello world", "test secret"])

    assert h.shape == (2, encoder.d)


def test_encode_window_matches_message_count():
    encoder = _make_encoder()
    window = ConversationWindow(k=3)
    window.add(Message(speaker_id="a", text="hello", relative_time=0.0))
    window.add(Message(speaker_id="b", text="world", relative_time=1.0))

    h = encoder.encode_window(window)

    assert h.shape == (2, encoder.d)


def test_encode_empty_window_returns_empty_tensor():
    encoder = _make_encoder()
    window = ConversationWindow(k=3)

    h = encoder.encode_window(window)

    assert h.shape == (0, encoder.d)
