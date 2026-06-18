from ..emotion.emotion_mapping import MAPPED_EMOTION_NAMES
from .integrated_pipeline import IntegratedInferenceResult

DEFAULT_OUTPUT_STATUS = "illustrative_unvalidated_example"
DEFAULT_EMOTION_SCORE_STATUS = "illustrative_placeholder"
DEFAULT_EMOTION_MODEL_TYPE = "uncalibrated_prototype_emotion_layer"

RISK_LEVEL_HIGH_THRESHOLD = 0.7
RISK_LEVEL_MEDIUM_THRESHOLD = 0.4
PRIMARY_EMOTION_THRESHOLD = 0.5
PRIMARY_EMOTION_TOP_K = 3


def risk_level(overall_risk: float) -> str:
    """Prototype bucketing of R_t into a readable risk_level (Section 16).
    Thresholds are prototype settings and must be tuned with validation
    data, like every other threshold in this report."""
    if overall_risk >= RISK_LEVEL_HIGH_THRESHOLD:
        return "high"
    if overall_risk >= RISK_LEVEL_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _interpretation(signals: dict[str, float], threshold: float = PRIMARY_EMOTION_THRESHOLD) -> str:
    """A minimal templated description, not a learned summary -- consistent
    with `emotion_score_status` defaulting to illustrative (Section 19.5)."""
    elevated = [name for name, value in signals.items() if value >= threshold]
    if not elevated:
        return "No elevated emotional signals were detected."
    return f"Elevated emotional {', '.join(elevated)} were detected."


def to_dashboard_dict(
    result: IntegratedInferenceResult,
    output_status: str = DEFAULT_OUTPUT_STATUS,
    emotion_score_status: str = DEFAULT_EMOTION_SCORE_STATUS,
    emotion_model_type: str = DEFAULT_EMOTION_MODEL_TYPE,
) -> dict:
    """Section 16: the exact dashboard JSON shape this report documents,
    built from one `IntegratedInferencePipeline.process()` result.

    `output_status`/`emotion_score_status`/`emotion_model_type` default to
    today's reality (Section 19.5: Grooming/EmotionScoreHead/RiskFusion are
    not yet trained or calibrated). Override them once the corresponding
    components are actually trained, rather than leaving a stale
    "illustrative" label on a genuinely validated result.
    """
    overall = result.overall_score.item()
    signals = dict(zip(MAPPED_EMOTION_NAMES, result.mapped_emotions.tolist()))
    primary_emotions = sorted(signals, key=signals.get, reverse=True)[:PRIMARY_EMOTION_TOP_K]

    return {
        "output_status": output_status,
        "overall_risk": overall,
        "risk_level": risk_level(overall),
        "summary_scores": {
            "safety_score": result.safety_score.item(),
            "emotion_score": result.emotion_score.item(),
            "overall_score": overall,
        },
        "component_scores": {name: value.item() for name, value in result.component_scores.items()},
        "early_warning": {
            "triggered": result.early_warning.triggered,
            "method": result.early_warning.method,
            "accumulated_risk": result.early_warning.accumulated_risk,
            "risk_trend": result.early_warning.risk_trend,
            "risk_trend_label": result.early_warning.risk_trend_label,
            "persistence": result.early_warning.persistence,
        },
        "emotion_report": {
            "emotion_score": result.emotion_score.item(),
            "score_status": emotion_score_status,
            "signals": signals,
            "primary_emotions": primary_emotions,
            "interpretation": _interpretation(signals),
            "model_type": emotion_model_type,
        },
        "evidence_messages": {
            "cyberbullying": result.evidence.cyberbullying,
            "conversation": result.evidence.conversation,
            "rule": result.evidence.rule,
            "emotion": result.evidence.emotion,
        },
        "uncertainty": result.uncertainty_estimate.uncertainty.item(),
        "confidence": result.uncertainty_estimate.confidence.item(),
        "human_review_required": result.human_review_required,
    }
