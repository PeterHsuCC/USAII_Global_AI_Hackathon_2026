from _tiny_bert import make_tiny_bert

from risk_detection import ConversationWindow, Message
from risk_detection.model import (
    ConversationEncoder,
    CyberbullyingHead,
    CyberbullyingPipeline,
    MessageEncoder,
)


def _make_pipeline() -> CyberbullyingPipeline:
    tokenizer, model = make_tiny_bert(hidden_size=8)
    message_encoder = MessageEncoder(tokenizer=tokenizer, encoder=model)
    conversation_encoder = ConversationEncoder(d=8)
    head = CyberbullyingHead(d=8, d_z=8)
    return CyberbullyingPipeline(message_encoder, conversation_encoder, head)


def test_score_returns_one_risk_value_per_message():
    pipeline = _make_pipeline()
    window = ConversationWindow(k=5)
    for i, text in enumerate(["hello", "you are stupid", "stop bullying me"]):
        window.add(Message(speaker_id="a", text=text, relative_time=float(i)))

    result = pipeline.score(window)

    assert result.per_message_risk.shape == (3,)
    assert result.attention_weights.shape == (3,)
    assert 0.0 <= result.window_score.item() <= 1.0


def test_score_on_empty_window():
    pipeline = _make_pipeline()
    window = ConversationWindow(k=5)

    result = pipeline.score(window)

    assert result.per_message_risk.shape == (0,)
    assert result.window_score.item() == 0.0
