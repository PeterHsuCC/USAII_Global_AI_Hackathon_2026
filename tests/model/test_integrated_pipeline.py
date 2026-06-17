import torch
from _tiny_bert import make_tiny_bert
from _tiny_emotion_classifier import make_tiny_emotion_classifier

from risk_detection import ConversationWindow, LLMSafetySignals, Message, RuleSignalExtractor
from risk_detection.signals.safety_features import SafetyFeatureExtractor
from risk_detection.model import (
    ConversationEncoder,
    CyberbullyingHead,
    EarlyDetectionHead,
    EmotionScoreHead,
    GoEmotionsClassifier,
    GroomingHead,
    HistoricalRiskState,
    HistoricalStateUpdater,
    IntegratedInferencePipeline,
    MessageEncoder,
    RiskFusion,
)


class _StubLLMExtractor:
    def extract(self, window):
        return LLMSafetySignals(
            secrecy=0.6,
            isolation=0.2,
            dependency=0.3,
            sexual_escalation=0.0,
            threat=0.0,
            coercion=0.4,
        )


class _StubDependencyExtractor:
    def extract(self, window):
        return 0.5


def _make_pipeline(d: int = 8, persistence_window: int = 5) -> IntegratedInferencePipeline:
    tokenizer, bert = make_tiny_bert(hidden_size=d)
    emo_tokenizer, emo_model = make_tiny_emotion_classifier(hidden_size=d)

    return IntegratedInferencePipeline(
        message_encoder=MessageEncoder(tokenizer=tokenizer, encoder=bert),
        conversation_encoder=ConversationEncoder(d=d),
        cyberbullying_head=CyberbullyingHead(d=d, d_z=d),
        grooming_head=GroomingHead(d_z=d, safety_dim=11),
        early_detection_head=EarlyDetectionHead(d_z=d, safety_dim=11, history_dim=15),
        emotion_classifier=GoEmotionsClassifier(tokenizer=emo_tokenizer, encoder=emo_model),
        emotion_score_head=EmotionScoreHead(),
        risk_fusion=RiskFusion(),
        historical_state_updater=HistoricalStateUpdater(persistence_window=persistence_window),
        safety_feature_extractor=SafetyFeatureExtractor(
            llm_extractor=_StubLLMExtractor(),
            rule_extractor=RuleSignalExtractor(),
        ),
        dependency_extractor=_StubDependencyExtractor(),
        mc_dropout_passes=3,  # small N for fast tests
    )


def _window() -> ConversationWindow:
    window = ConversationWindow(k=5)
    window.add(Message(speaker_id="a", text="our little secret ok?", relative_time=0.0))
    window.add(Message(speaker_id="b", text="ok i promise", relative_time=1.0))
    window.add(Message(speaker_id="a", text="add me on snapchat", relative_time=2.0))
    return window


def test_process_returns_full_dashboard_result():
    pipeline = _make_pipeline()

    result = pipeline.process(_window())

    assert 0.0 <= result.safety_score.item() <= 1.0
    assert 0.0 <= result.emotion_score.item() <= 1.0
    assert 0.0 <= result.overall_score.item() <= 1.0
    assert set(result.component_scores) == {"cyberbullying", "grooming", "early_predator", "rule_score"}
    assert result.risk_trend_label in {"increasing", "stable", "decreasing"}
    assert len(result.evidence.cyberbullying) <= 3
    assert len(result.evidence.rule) >= 1  # "our little secret" / "add me on snapchat" both fire
    assert isinstance(result.human_review_required, bool)
    assert 0.0 <= result.uncertainty_estimate.confidence.item() <= 1.0
    assert len(result.limitations) > 0


def test_historical_state_advances_across_successive_calls():
    pipeline = _make_pipeline()
    window = _window()

    state_before = pipeline.historical_state_updater.state
    result1 = pipeline.process(window)
    assert pipeline.historical_state_updater.state is result1.historical_state
    assert result1.historical_state is not state_before

    result2 = pipeline.process(window)
    assert result2.historical_state.accumulated_risk != 0.0


def test_early_detection_uses_h_prev_not_h_t():
    pipeline = _make_pipeline()
    window = _window()

    pipeline.historical_state_updater.state = HistoricalRiskState(
        accumulated_risk=0.9,
        risk_trend=0.5,
        persistence=0.9,
        behavior_frequency=torch.ones(6),
        smoothed_behavior=torch.ones(6),
    )
    result_high_prior = pipeline.process(window)

    pipeline.historical_state_updater.reset()
    result_zero_prior = pipeline.process(window)

    assert (
        result_high_prior.component_scores["early_predator"].item()
        != result_zero_prior.component_scores["early_predator"].item()
    )


def test_modules_restored_to_eval_after_processing():
    pipeline = _make_pipeline()

    pipeline.process(_window())

    assert pipeline.message_encoder.training is False
    assert pipeline.conversation_encoder.training is False
    assert pipeline.cyberbullying_head.training is False
    assert pipeline.grooming_head.training is False
    assert pipeline.early_detection_head.training is False
    assert pipeline.risk_fusion.training is False
    assert pipeline.emotion_classifier.training is False
