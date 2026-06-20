"""Builds a fresh per-job pipeline around the shared heavy ModelComponents
and runs one case's analysis (v6 Section 3: AI Analysis Workers).

Single-shot semantics: one ConversationWindow built from the case's
(already redacted/anonymized) messages, one `.process()` call -- the same
semantics frontend/app.py's placeholder demo already used. Genuinely
incremental multi-call historical tracking across a conversation's real-
time lifetime is a deeper model-integration question, out of scope here.

Per the plan's correctness note: HistoricalStateUpdater/EarlyWarningTracker
are constructed fresh per call so risk history never leaks between cases;
only the expensive frozen/trained modules in ModelComponents are shared.
"""

from dataclasses import dataclass

from risk_detection import ConversationWindow, Message
from risk_detection.model import EarlyWarningTracker, HistoricalStateUpdater, IntegratedInferencePipeline, IntegratedInferenceResult
from risk_detection.signals.rules import RuleEvidence, RuleSignalExtractor
from risk_detection.signals.safety_features import SafetyFeatures

from backend.config import settings
from backend.model_runtime.loader import ModelComponents, get_model_components

RISK_LEVEL_THRESHOLDS = (
    # Illustrative thresholds (the doc specifies the 4 risk_level values but
    # not where to draw the lines against a continuous score) -- same
    # "illustrative default" framing the doc uses elsewhere (Section 6.1, 10).
    (0.75, "critical"),
    (0.50, "high"),
    (0.25, "medium"),
)


@dataclass(frozen=True)
class JobMessage:
    speaker_local_id: str
    redacted_content: str | None
    message_sequence: int


@dataclass(frozen=True)
class AnalysisOutcome:
    result: IntegratedInferenceResult
    rule_evidence: RuleEvidence
    safety_features: SafetyFeatures
    window_message_sequences: tuple[int, ...]  # window index -> original message_sequence
    extra_limitations: tuple[str, ...]


def risk_level_from_score(score: float) -> str:
    for threshold, label in RISK_LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "low"


def build_window(messages: list[JobMessage], window_size: int) -> tuple[ConversationWindow, tuple[int, ...]]:
    windowed = messages[-window_size:]
    window = ConversationWindow(k=window_size)
    for msg in windowed:
        window.add(Message(speaker_id=msg.speaker_local_id, text=msg.redacted_content or "", relative_time=0.0))
    return window, tuple(msg.message_sequence for msg in windowed)


def run_analysis(
    messages: list[JobMessage],
    components: ModelComponents | None = None,
    window_size: int | None = None,
) -> AnalysisOutcome:
    components = components or get_model_components()
    window_size = window_size if window_size is not None else settings.default_window_size

    pipeline = IntegratedInferencePipeline(
        message_encoder=components.message_encoder,
        conversation_encoder=components.conversation_encoder,
        cyberbullying_head=components.cyberbullying_head,
        grooming_head=components.grooming_head,
        emotion_classifier=components.emotion_classifier,
        emotion_score_head=components.emotion_score_head,
        risk_fusion=components.risk_fusion,
        historical_state_updater=HistoricalStateUpdater(),
        early_warning_tracker=EarlyWarningTracker(),
        safety_feature_extractor=components.safety_feature_extractor,
        dependency_extractor=components.dependency_extractor,
        grooming_message_encoder=components.grooming_message_encoder,
        grooming_conversation_encoder=components.grooming_conversation_encoder,
    )

    window, window_message_sequences = build_window(messages, window_size)
    result = pipeline.process(window)

    # Re-derived directly from the window rather than threaded out of
    # process() (which only exposes the unioned EvidenceBundle.rule indices,
    # not the per-rule breakdown the Explainability Service needs).
    rule_extractor: RuleSignalExtractor = components.safety_feature_extractor.rule_extractor
    rule_evidence = rule_extractor.extract_evidence(window)
    safety_features = components.safety_feature_extractor.extract(window)

    return AnalysisOutcome(
        result=result,
        rule_evidence=rule_evidence,
        safety_features=safety_features,
        window_message_sequences=window_message_sequences,
        extra_limitations=components.extra_limitations,
    )