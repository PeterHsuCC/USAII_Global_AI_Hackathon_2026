import torch
from _tiny_bert import make_tiny_bert
from _tiny_emotion_classifier import make_tiny_emotion_classifier

from risk_detection import ConversationWindow, LLMSafetySignals, Message, RuleSignalExtractor
from risk_detection.signals.safety_features import SafetyFeatureExtractor
from risk_detection.model import (
    ConversationEncoder,
    CyberbullyingHead,
    EmotionScoreHead,
    GoEmotionsClassifier,
    GroomingHead,
    HistoricalRiskState,
    HistoricalStateUpdater,
    EarlyWarningTracker,
    IntegratedInferencePipeline,
    MessageEncoder,
    PrototypeRiskFusion,
    risk_level,
    to_dashboard_dict,
)


class _StubLLMExtractor:
    def extract(self, window):
        return LLMSafetySignals(
            secrecy=0.6, isolation=0.2, dependency=0.3, sexual_escalation=0.0, threat=0.0, coercion=0.4
        )


class _StubDependencyExtractor:
    def extract(self, window):
        return 0.5


def _make_pipeline(d: int = 8) -> IntegratedInferencePipeline:
    tokenizer, bert = make_tiny_bert(hidden_size=d)
    emo_tokenizer, emo_model = make_tiny_emotion_classifier(hidden_size=d)
    return IntegratedInferencePipeline(
        message_encoder=MessageEncoder(tokenizer=tokenizer, encoder=bert),
        conversation_encoder=ConversationEncoder(d=d),
        cyberbullying_head=CyberbullyingHead(d=d, d_z=d),
        grooming_head=GroomingHead(d_z=d, safety_dim=11),
        emotion_classifier=GoEmotionsClassifier(tokenizer=emo_tokenizer, encoder=emo_model),
        emotion_score_head=EmotionScoreHead(),
        risk_fusion=PrototypeRiskFusion(),
        historical_state_updater=HistoricalStateUpdater(persistence_window=5),
        safety_feature_extractor=SafetyFeatureExtractor(
            llm_extractor=_StubLLMExtractor(), rule_extractor=RuleSignalExtractor()
        ),
        dependency_extractor=_StubDependencyExtractor(),
        mc_dropout_passes=3,
    )


def _window() -> ConversationWindow:
    window = ConversationWindow(k=5)
    window.add(Message(speaker_id="a", text="our little secret ok?", relative_time=0.0))
    window.add(Message(speaker_id="b", text="ok i promise", relative_time=1.0))
    window.add(Message(speaker_id="a", text="add me on snapchat", relative_time=2.0))
    return window


def test_risk_level_buckets():
    assert risk_level(0.9) == "high"
    assert risk_level(0.7) == "high"
    assert risk_level(0.5) == "medium"
    assert risk_level(0.4) == "medium"
    assert risk_level(0.1) == "low"


def test_to_dashboard_dict_matches_section_16_shape():
    pipeline = _make_pipeline()
    result = pipeline.process(_window())

    d = to_dashboard_dict(result)

    assert d["output_status"] == "illustrative_unvalidated_example"
    assert isinstance(d["overall_risk"], float)
    assert d["risk_level"] in {"high", "medium", "low"}
    assert set(d["summary_scores"]) == {"safety_score", "emotion_score", "overall_score"}
    assert set(d["component_scores"]) == {"cyberbullying", "grooming", "rule_score"}
    assert set(d["early_warning"]) == {
        "triggered", "method", "accumulated_risk", "risk_trend", "risk_trend_label", "persistence"
    }
    assert d["early_warning"]["method"] == "persistence_based_baseline"
    emotion_report = d["emotion_report"]
    assert emotion_report["score_status"] == "illustrative_placeholder"
    assert set(emotion_report["signals"]) == {"fear", "sadness", "anger", "distress", "dependency"}
    assert len(emotion_report["primary_emotions"]) == 3
    assert isinstance(emotion_report["interpretation"], str)
    assert set(d["evidence_messages"]) == {"cyberbullying", "conversation", "rule", "emotion"}
    assert isinstance(d["uncertainty"], float)
    assert isinstance(d["confidence"], float)
    assert isinstance(d["human_review_required"], bool)


def test_to_dashboard_dict_primary_emotions_ranked_descending():
    pipeline = _make_pipeline()
    result = pipeline.process(_window())

    d = to_dashboard_dict(result)
    signals = d["emotion_report"]["signals"]
    primary = d["emotion_report"]["primary_emotions"]

    values = [signals[name] for name in primary]
    assert values == sorted(values, reverse=True)


def test_to_dashboard_dict_status_fields_are_overridable():
    pipeline = _make_pipeline()
    result = pipeline.process(_window())

    d = to_dashboard_dict(
        result,
        output_status="validated_result",
        emotion_score_status="calibrated",
        emotion_model_type="trained_logistic_emotion_head",
    )

    assert d["output_status"] == "validated_result"
    assert d["emotion_report"]["score_status"] == "calibrated"
    assert d["emotion_report"]["model_type"] == "trained_logistic_emotion_head"


def test_to_dashboard_dict_reflects_warning_and_review():
    tracker = EarlyWarningTracker()
    tracker.update(HistoricalRiskState(accumulated_risk=0.9, risk_trend=0.0, persistence=0.9))
    pipeline = _make_pipeline()
    pipeline.early_warning_tracker = tracker

    result = pipeline.process(_window())
    d = to_dashboard_dict(result)

    assert d["early_warning"]["triggered"] is True
    assert d["human_review_required"] is True
