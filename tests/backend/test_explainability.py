import json

import torch

from backend.explainability.service import MANDATORY_DISCLAIMER, build_explainability
from backend.model_runtime.job_runner import JobMessage, risk_level_from_score, run_analysis


def _run() -> tuple:
    messages = [
        JobMessage(speaker_local_id="SPEAKER_A", redacted_content="our little secret ok?", message_sequence=10),
        JobMessage(speaker_local_id="SPEAKER_B", redacted_content="ok i promise", message_sequence=11),
        JobMessage(speaker_local_id="SPEAKER_A", redacted_content="add me on snapchat", message_sequence=12),
    ]
    outcome = run_analysis(messages, window_size=5)
    messages_by_sequence = {m.message_sequence: m.redacted_content for m in messages}
    return outcome, messages_by_sequence


def test_explainability_includes_mandatory_disclaimer():
    outcome, messages_by_sequence = _run()
    output = build_explainability(
        outcome,
        messages_by_sequence=messages_by_sequence,
        risk_level=risk_level_from_score(outcome.result.overall_score.item()),
    )
    assert output.disclaimer == MANDATORY_DISCLAIMER


def test_rule_evidence_maps_back_to_original_message_sequence():
    outcome, messages_by_sequence = _run()
    output = build_explainability(
        outcome,
        messages_by_sequence=messages_by_sequence,
        risk_level="medium",
    )
    secret_request_items = [r for r in output.rule_evidence if r.rule_id == "secret_request"]
    assert len(secret_request_items) == 1
    assert secret_request_items[0].matched_message_sequence == 10
    assert secret_request_items[0].redacted_evidence_span == "our little secret ok?"
    assert secret_request_items[0].severity == "medium"

    contact_items = [r for r in output.rule_evidence if r.rule_id == "contact_migration"]
    assert contact_items[0].matched_message_sequence == 12


def test_triggered_signals_include_rule_sourced_signal():
    outcome, messages_by_sequence = _run()
    output = build_explainability(outcome, messages_by_sequence=messages_by_sequence, risk_level="medium")
    rule_signals = [s for s in output.triggered_signals if s.source == "rule"]
    names = {s.name for s in rule_signals}
    assert "secret_request" in names
    assert "contact_migration" in names
    for signal in rule_signals:
        assert signal.message_sequences  # rule-sourced signals carry message attribution


def test_model_evidence_carries_attention_disclaimer():
    outcome, messages_by_sequence = _run()
    output = build_explainability(outcome, messages_by_sequence=messages_by_sequence, risk_level="medium")
    assert "not causal" in output.model_evidence.attention_disclaimer


def test_data_limitations_include_preprocessing_flags_passed_in():
    outcome, messages_by_sequence = _run()
    output = build_explainability(
        outcome,
        messages_by_sequence=messages_by_sequence,
        risk_level="medium",
        preprocessing_limitations=("single_speaker_conversation",),
    )
    assert "single_speaker_conversation" in output.data_limitations


def test_to_json_round_trips():
    outcome, messages_by_sequence = _run()
    output = build_explainability(outcome, messages_by_sequence=messages_by_sequence, risk_level="medium")
    parsed = json.loads(output.to_json())
    assert parsed["disclaimer"] == MANDATORY_DISCLAIMER
    assert "rule_evidence" in parsed


def test_non_finite_uncertainty_is_sanitized_to_fail_safe_values():
    """NaN must never reach the API response: it compares False against
    every human_review_required threshold (e.g. confidence < threshold), so
    an unnoticed NaN would silently bypass the review trigger rather than
    visibly fail safe."""
    outcome, messages_by_sequence = _run()
    outcome.result.uncertainty_estimate.confidence = torch.tensor(float("nan"))
    outcome.result.uncertainty_estimate.uncertainty = torch.tensor(float("nan"))
    outcome.result.uncertainty_estimate.variance = torch.tensor(float("nan"))

    output = build_explainability(outcome, messages_by_sequence=messages_by_sequence, risk_level="medium")

    assert output.confidence_and_uncertainty.confidence == 0.0
    assert output.confidence_and_uncertainty.uncertainty == 1.0
    assert output.confidence_and_uncertainty.mc_dropout_variance == 0.25

    parsed = json.loads(output.to_json())
    assert parsed["confidence_and_uncertainty"]["confidence"] == 0.0