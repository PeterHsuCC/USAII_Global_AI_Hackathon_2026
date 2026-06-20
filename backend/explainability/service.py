"""Converts structured model and rule outputs into analyst-facing evidence
(v6 Section 9). Assembles the 5 documented output types from structured
inputs; does not generate free-form summaries.

Severity-per-rule and the LLM-signal "triggered" threshold are this
backend's own judgment calls -- the doc asks for "Rule ID, severity" and
"Signal name, score, source" but doesn't define either scale, so these are
documented here rather than left as undocumented magic numbers.
"""

import json
from dataclasses import asdict, dataclass

from backend.model_runtime.job_runner import AnalysisOutcome

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

    for rule_id, window_indices in outcome.rule_evidence.triggered_message_indices.items():
        sequences = tuple(
            seq for seq in (_window_index_to_sequence(outcome, i) for i in window_indices) if seq is not None
        )
        if sequences:
            triggered_signals.append(
                TriggeredSignal(name=rule_id, score=1.0, source="rule", message_sequences=sequences)
            )
        for window_index in window_indices:
            sequence = _window_index_to_sequence(outcome, window_index)
            rule_evidence_items.append(
                RuleEvidenceItem(
                    rule_id=rule_id,
                    severity=RULE_SEVERITY.get(rule_id, "low"),
                    matched_message_sequence=sequence if sequence is not None else window_index,
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
        confidence=float(uncertainty.confidence.item()),
        uncertainty=float(uncertainty.uncertainty.item()),
        mc_dropout_variance=float(uncertainty.variance.item()),
    )

    data_limitations = tuple(outcome.result.limitations) + outcome.extra_limitations + preprocessing_limitations

    return ExplainabilityOutput(
        triggered_signals=tuple(triggered_signals),
        rule_evidence=tuple(rule_evidence_items),
        model_evidence=model_evidence,
        confidence_and_uncertainty=confidence_and_uncertainty,
        data_limitations=data_limitations,
    )