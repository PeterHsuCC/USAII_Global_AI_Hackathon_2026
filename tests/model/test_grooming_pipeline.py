import torch
from _tiny_bert import make_tiny_bert

from risk_detection import ConversationWindow, LLMSafetySignals, Message, RuleSignalExtractor
from risk_detection.signals.safety_features import SafetyFeatureExtractor
from risk_detection.model import (
    ConversationEncoder,
    GroomingHead,
    GroomingPipeline,
    MessageEncoder,
)


class _StubLLMExtractor:
    def extract(self, window):
        return LLMSafetySignals(
            secrecy=0.8,
            isolation=0.2,
            dependency=0.3,
            sexual_escalation=0.0,
            threat=0.0,
            coercion=0.4,
        )


def _make_pipeline() -> GroomingPipeline:
    tokenizer, model = make_tiny_bert(hidden_size=8)
    message_encoder = MessageEncoder(tokenizer=tokenizer, encoder=model)
    conversation_encoder = ConversationEncoder(d=8)
    grooming_head = GroomingHead(d_z=8, safety_dim=11)
    safety_extractor = SafetyFeatureExtractor(
        llm_extractor=_StubLLMExtractor(),
        rule_extractor=RuleSignalExtractor(),
    )
    return GroomingPipeline(message_encoder, conversation_encoder, grooming_head, safety_extractor)


def test_score_returns_scalar_and_six_behaviors():
    pipeline = _make_pipeline()
    window = ConversationWindow(k=5)
    window.add(Message(speaker_id="a", text="our little secret ok?", relative_time=0.0))
    window.add(Message(speaker_id="b", text="ok i promise", relative_time=1.0))

    result = pipeline.score(window)

    assert result.grooming_score.shape == ()
    assert result.behaviors.shape == (6,)
    assert 0.0 <= result.grooming_score.item() <= 1.0
    assert result.safety_features.rule_signals.secret_request is True


def test_score_on_empty_window_short_circuits():
    pipeline = _make_pipeline()
    window = ConversationWindow(k=5)

    result = pipeline.score(window)

    assert result.grooming_score.item() == 0.0
    assert torch.equal(result.behaviors, torch.zeros(6))
