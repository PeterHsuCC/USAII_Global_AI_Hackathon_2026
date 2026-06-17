import torch
from _tiny_bert import make_tiny_bert

from risk_detection import ConversationWindow, LLMSafetySignals, Message, RuleSignalExtractor
from risk_detection.signals.safety_features import SafetyFeatureExtractor
from risk_detection.model import (
    ConversationEncoder,
    EarlyDetectionHead,
    EarlyDetectionPipeline,
    HistoricalRiskState,
    MessageEncoder,
)


class _StubLLMExtractor:
    def extract(self, window):
        return LLMSafetySignals(
            secrecy=0.7,
            isolation=0.3,
            dependency=0.2,
            sexual_escalation=0.1,
            threat=0.0,
            coercion=0.4,
        )


def _make_pipeline() -> EarlyDetectionPipeline:
    tokenizer, model = make_tiny_bert(hidden_size=8)
    message_encoder = MessageEncoder(tokenizer=tokenizer, encoder=model)
    conversation_encoder = ConversationEncoder(d=8)
    early_detection_head = EarlyDetectionHead(d_z=8, safety_dim=11, history_dim=15)
    safety_extractor = SafetyFeatureExtractor(
        llm_extractor=_StubLLMExtractor(),
        rule_extractor=RuleSignalExtractor(),
    )
    return EarlyDetectionPipeline(message_encoder, conversation_encoder, early_detection_head, safety_extractor)


def _window() -> ConversationWindow:
    window = ConversationWindow(k=5)
    window.add(Message(speaker_id="a", text="our little secret ok?", relative_time=0.0))
    window.add(Message(speaker_id="b", text="ok i promise", relative_time=1.0))
    return window


def test_score_returns_scalar_in_unit_interval():
    pipeline = _make_pipeline()

    result = pipeline.score(_window(), h_prev=HistoricalRiskState())

    assert result.early_detection_score.shape == ()
    assert 0.0 <= result.early_detection_score.item() <= 1.0


def test_h_prev_changes_the_score():
    pipeline = _make_pipeline()
    window = _window()

    zero_history = HistoricalRiskState()
    high_history = HistoricalRiskState(
        accumulated_risk=0.9,
        risk_trend=0.5,
        persistence=0.9,
        behavior_frequency=torch.ones(6),
        smoothed_behavior=torch.ones(6),
    )

    score_with_zero_history = pipeline.score(window, h_prev=zero_history).early_detection_score
    score_with_high_history = pipeline.score(window, h_prev=high_history).early_detection_score

    assert score_with_zero_history.item() != score_with_high_history.item()


def test_score_on_empty_window_short_circuits():
    pipeline = _make_pipeline()
    window = ConversationWindow(k=5)

    result = pipeline.score(window, h_prev=HistoricalRiskState())

    assert result.early_detection_score.item() == 0.0
