"""Converts structured model and rule outputs into analyst-facing evidence
(v6 Section 9). Assembles the 5 documented output types from structured
inputs; does not generate free-form summaries.

Severity-per-rule and the LLM-signal "triggered" threshold are this
backend's own judgment calls -- the doc asks for "Rule ID, severity" and
"Signal name, score, source" but doesn't define either scale, so these are
documented here rather than left as undocumented magic numbers.
"""

import json
import math
from dataclasses import asdict, dataclass

from risk_detection.model.emotion.emotion_mapping import MAPPED_EMOTION_NAMES

from backend.model_runtime.job_runner import AnalysisOutcome, risk_level_from_score


def _finite_or(value: float, fallback: float) -> float:
    """NaN/Inf must never reach the API response: NaN compares False against
    every threshold (e.g. `confidence < confidence_threshold` in
    human_review_required), so an unnoticed NaN would silently evade the
    human-review trigger rather than visibly fail safe. Substituting the
    most-uncertain value here means a non-finite result always reads as
    "treat this as maximally uncertain", not as a confident answer."""
    return value if math.isfinite(value) else fallback


def _operable_risk_score(
    cyberbullying_score: float,
    grooming_score: float,
    rule_safety_score: float,
    llm_signals,
) -> float:
    """max() across every score that does NOT depend on an untrained head:
    the trained Cyberbullying/Grooming Heads, the rule-based safety score,
    and the six LLM-extracted safety signals. Deliberately excludes
    emotion_component_score (EmotionScoreHead is untrained, same defect as
    RiskFusion) and overall_score/safety_score (RiskFusion itself) --
    found live, on a 31-message case where every one of those scores was
    low but risk_level still read "high" off RiskFusion's noise. A single
    elevated signal is enough to raise this max, mirroring Section 13's
    existing single-rule Review_t override rather than averaging signals
    together and diluting one severe one."""
    return max(
        cyberbullying_score,
        grooming_score,
        rule_safety_score,
        llm_signals.secrecy,
        llm_signals.isolation,
        llm_signals.dependency,
        llm_signals.sexual_escalation,
        llm_signals.threat,
        llm_signals.coercion,
    )

ATTENTION_DISCLAIMER = (
    "Attention weights indicate which messages the model focused on. They "
    "are not causal explanations and must not be presented as legal or "
    "clinical evidence."
)

# Section 9.3's exact required wording -- non-removable wherever an
# explainability output is shown.
MANDATORY_DISCLAIMER = (
    "This output does not identify individuals as offenders or victims, "
    "make clinical or legal conclusions, automatically report cases, "
    "or trigger enforcement actions. A qualified human must review "
    "all cases before any organizational action is taken."
)

RULE_SEVERITY = {
    "threat_phrase": "high",
    "image_request": "high",
    "contact_migration": "medium",
    "secret_request": "medium",
    "age_reference": "low",
}

LLM_SIGNAL_TRIGGER_THRESHOLD = 0.3


@dataclass(frozen=True)
class TriggeredSignal:
    name: str
    score: float
    source: str  # "llm" | "rule"
    message_sequences: tuple[int, ...]


@dataclass(frozen=True)
class RuleEvidenceItem:
    rule_id: str
    severity: str
    matched_message_sequence: int
    redacted_evidence_span: str | None


@dataclass(frozen=True)
class ModelEvidence:
    high_risk_message_sequences: tuple[int, ...]
    attention_focus_message_sequences: tuple[int, ...]
    cyberbullying_component_score: float
    grooming_component_score: float
    # S~_r(t) (Section 13): the rule safety score itself, not just which
    # named rules fired -- this is the exact number `human_review_required`
    # checks against the 0.8 review threshold, and (unlike
    # emotion_component_score below) it's purely rule-based, with no
    # untrained learned weights involved.
    rule_safety_score: float
    # S_emotion(t) (Section 10.2/16) -- like cyberbullying/grooming above,
    # but routed through EmotionScoreHead, which (per data_limitations
    # below) is untrained, so this is illustrative, not a calibrated score.
    emotion_component_score: float
    # M_t = [Fear, Sadness, Anger, Distress, Dependency] (Section 10.2):
    # unlike emotion_component_score, these come from the frozen pretrained
    # GoEmotions classifier plus a fixed (not learned) mapping, so they're
    # real signal, not a placeholder -- src/risk_detection/model/inference/
    # dashboard.py's to_dashboard_dict() already computes this exact shape
    # for the "Section 16" dashboard JSON, just never wired into this API.
    mapped_emotions: dict[str, float]
    attention_disclaimer: str = ATTENTION_DISCLAIMER


@dataclass(frozen=True)
class ConfidenceAndUncertainty:
    risk_level: str
    confidence: float
    uncertainty: float
    mc_dropout_variance: float
    # max() across cyberbullying/grooming/rule-safety scores and the six
    # LLM-extracted safety signals -- everything that does NOT depend on
    # an untrained head (RiskFusion, EmotionScoreHead) -- bucketed with
    # the same thresholds as risk_level above (job_runner.py's
    # RISK_LEVEL_THRESHOLDS), so "high" means the same thing either way.
    # Shown alongside risk_level for comparison; does not feed
    # human_review_required, which keeps its own existing, separately
    # validated rule/LLM-based condition (Section 13).
    operable_risk_score: float
    operable_risk_level: str


@dataclass(frozen=True)
class ExplainabilityOutput:
    triggered_signals: tuple[TriggeredSignal, ...]
    rule_evidence: tuple[RuleEvidenceItem, ...]
    model_evidence: ModelEvidence
    confidence_and_uncertainty: ConfidenceAndUncertainty
    data_limitations: tuple[str, ...]
    disclaimer: str = MANDATORY_DISCLAIMER

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _window_index_to_sequence(outcome: AnalysisOutcome, window_index: int) -> int | None:
    if 0 <= window_index < len(outcome.window_message_sequences):
        return outcome.window_message_sequences[window_index]
    return None


def _redacted_span(messages_by_sequence: dict[int, str | None], sequence: int | None) -> str | None:
    if sequence is None:
        return None
    return messages_by_sequence.get(sequence)


def build_explainability(
    outcome: AnalysisOutcome,
    *,
    messages_by_sequence: dict[int, str | None],
    risk_level: str,
    preprocessing_limitations: tuple[str, ...] = (),
) -> ExplainabilityOutput:
    rule_evidence_items: list[RuleEvidenceItem] = []
    triggered_signals: list[TriggeredSignal] = []

    # outcome.rule_evidence is already merged across every window the case
    # was split into (job_runner.py's _merge_rule_evidence_across_windows),
    # with values translated to absolute message_sequence numbers -- unlike
    # outcome.result.evidence below, these are not window-local positions,
    # so no _window_index_to_sequence translation is needed here.
    for rule_id, sequences in outcome.rule_evidence.triggered_message_indices.items():
        if sequences:
            triggered_signals.append(
                TriggeredSignal(name=rule_id, score=1.0, source="rule", message_sequences=tuple(sequences))
            )
        for sequence in sequences:
            rule_evidence_items.append(
                RuleEvidenceItem(
                    rule_id=rule_id,
                    severity=RULE_SEVERITY.get(rule_id, "low"),
                    matched_message_sequence=sequence,
                    redacted_evidence_span=_redacted_span(messages_by_sequence, sequence),
                )
            )

    llm_signals = outcome.safety_features.llm_signals
    for name in ("secrecy", "isolation", "dependency", "sexual_escalation", "threat", "coercion"):
        score = getattr(llm_signals, name)
        if score >= LLM_SIGNAL_TRIGGER_THRESHOLD:
            # No per-message attribution is available for LLM-extracted
            # signals (the extractor scores the whole window, not
            # individual messages), unlike rule-sourced signals above.
            triggered_signals.append(TriggeredSignal(name=name, score=score, source="llm", message_sequences=()))

    evidence = outcome.result.evidence
    high_risk_sequences = tuple(
        seq for seq in (_window_index_to_sequence(outcome, i) for i in evidence.cyberbullying) if seq is not None
    )
    attention_sequences = tuple(
        seq for seq in (_window_index_to_sequence(outcome, i) for i in evidence.conversation) if seq is not None
    )

    cyberbullying_score = float(outcome.result.component_scores["cyberbullying"].item())
    grooming_score = float(outcome.result.component_scores["grooming"].item())
    rule_score = float(outcome.result.component_scores["rule_score"].item())

    model_evidence = ModelEvidence(
        high_risk_message_sequences=high_risk_sequences,
        attention_focus_message_sequences=attention_sequences,
        # Window-level component scores, not per-message: the pipeline does
        # not expose per-message classifier scores without a redundant
        # second forward pass, so this is scoped to what's actually
        # available rather than overclaiming a finer granularity.
        cyberbullying_component_score=cyberbullying_score,
        grooming_component_score=grooming_score,
        rule_safety_score=rule_score,
        emotion_component_score=float(outcome.result.emotion_score.item()),
        mapped_emotions=dict(zip(MAPPED_EMOTION_NAMES, (float(v) for v in outcome.result.mapped_emotions.tolist()))),
    )

    operable_score = _operable_risk_score(cyberbullying_score, grooming_score, rule_score, llm_signals)

    uncertainty = outcome.result.uncertainty_estimate
    confidence_and_uncertainty = ConfidenceAndUncertainty(
        risk_level=risk_level,
        confidence=_finite_or(float(uncertainty.confidence.item()), 0.0),
        uncertainty=_finite_or(float(uncertainty.uncertainty.item()), 1.0),
        mc_dropout_variance=_finite_or(float(uncertainty.variance.item()), 0.25),
        operable_risk_score=operable_score,
        operable_risk_level=risk_level_from_score(operable_score),
    )

    data_limitations = tuple(outcome.result.limitations) + outcome.extra_limitations + preprocessing_limitations

    return ExplainabilityOutput(
        triggered_signals=tuple(triggered_signals),
        rule_evidence=tuple(rule_evidence_items),
        model_evidence=model_evidence,
        confidence_and_uncertainty=confidence_and_uncertainty,
        data_limitations=data_limitations,
    )