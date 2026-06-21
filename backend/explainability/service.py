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

from backend.model_runtime.job_runner import AnalysisOutcome


def _finite_or(value: float, fallback: float) -> float:
    """NaN/Inf must never reach the API response: NaN compares False against
    every threshold (e.g. `confidence < confidence_threshold` in
    human_review_required), so an unnoticed NaN would silently evade the
    human-review trigger rather than visibly fail safe. Substituting the
    most-uncertain value here means a non-finite result always reads as
    "treat this as maximally uncertain", not as a confident answer."""
    return value if math.isfinite(value) else fallback

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
    attention_disclaimer: str = ATTENTION_DISCLAIMER


@dataclass(frozen=True)
class ConfidenceAndUncertainty:
    risk_level: str
    confidence: float
    uncertainty: float
    mc_dropout_variance: float


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

    model_evidence = ModelEvidence(
        high_risk_message_sequences=high_risk_sequences,
        attention_focus_message_sequences=attention_sequences,
        # Window-level component scores, not per-message: the pipeline does
        # not expose per-message classifier scores without a redundant
        # second forward pass, so this is scoped to what's actually
        # available rather than overclaiming a finer granularity.
        cyberbullying_component_score=float(outcome.result.component_scores["cyberbullying"].item()),
        grooming_component_score=float(outcome.result.component_scores["grooming"].item()),
    )

    uncertainty = outcome.result.uncertainty_estimate
    confidence_and_uncertainty = ConfidenceAndUncertainty(
        risk_level=risk_level,
        confidence=_finite_or(float(uncertainty.confidence.item()), 0.0),
        uncertainty=_finite_or(float(uncertainty.uncertainty.item()), 1.0),
        mc_dropout_variance=_finite_or(float(uncertainty.variance.item()), 0.25),
    )

    data_limitations = tuple(outcome.result.limitations) + outcome.extra_limitations + preprocessing_limitations

    return ExplainabilityOutput(
        triggered_signals=tuple(triggered_signals),
        rule_evidence=tuple(rule_evidence_items),
        model_evidence=model_evidence,
        confidence_and_uncertainty=confidence_and_uncertainty,
        data_limitations=data_limitations,
    )