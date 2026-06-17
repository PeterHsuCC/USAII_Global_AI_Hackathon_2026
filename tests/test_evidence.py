import torch

from risk_detection import RuleEvidence
from risk_detection.model import (
    EmotionScoreHead,
    attention_evidence,
    cyberbullying_evidence,
    emotion_evidence,
    extract_evidence,
    rule_evidence,
    top_k_indices,
)

_LABEL_TO_INDEX = {
    "fear": 0,
    "sadness": 1,
    "anger": 2,
    "nervousness": 3,
    "grief": 4,
    "caring": 5,
    "love": 6,
}


def test_top_k_indices_descending_order():
    values = torch.tensor([0.1, 0.9, 0.5, 0.95, 0.2])

    assert top_k_indices(values, k=3) == [3, 1, 2]  # 0.95, 0.9, 0.5


def test_top_k_indices_clamped_to_available_count():
    values = torch.tensor([0.5, 0.9])

    assert sorted(top_k_indices(values, k=3)) == [0, 1]


def test_top_k_indices_empty_input():
    assert top_k_indices(torch.tensor([])) == []


def test_cyberbullying_evidence_matches_top_k_indices():
    risk = torch.tensor([0.1, 0.9, 0.3])

    assert cyberbullying_evidence(risk, k=2) == top_k_indices(risk, k=2)


def test_attention_evidence_matches_top_k_indices():
    alpha = torch.tensor([0.05, 0.6, 0.35])

    assert attention_evidence(alpha, k=2) == top_k_indices(alpha, k=2)


def test_rule_evidence_unions_across_active_rules():
    evidence = RuleEvidence(
        triggered_message_indices={
            "secret_request": [1, 4],
            "contact_migration": [],
            "age_reference": [],
            "image_request": [3, 9],
            "threat_phrase": [4],
        }
    )

    assert rule_evidence(evidence) == [1, 3, 4, 9]


def test_emotion_evidence_picks_highest_scoring_messages():
    head = EmotionScoreHead()
    g_i = torch.rand(5, 7)

    indices = emotion_evidence(g_i, _LABEL_TO_INDEX, d_t=0.5, emotion_score_head=head, k=2)

    assert len(indices) == 2
    assert all(0 <= i < 5 for i in indices)


def test_extract_evidence_bundle_matches_each_component():
    head = EmotionScoreHead()
    per_message_risk = torch.tensor([0.1, 0.9, 0.3])
    attention_weights = torch.tensor([0.2, 0.5, 0.3])
    rule_ev = RuleEvidence(
        triggered_message_indices={
            "secret_request": [0],
            "contact_migration": [],
            "age_reference": [],
            "image_request": [],
            "threat_phrase": [],
        }
    )
    g_i = torch.rand(3, 7)

    bundle = extract_evidence(per_message_risk, attention_weights, rule_ev, g_i, _LABEL_TO_INDEX, 0.5, head)

    assert bundle.cyberbullying == top_k_indices(per_message_risk, 3)
    assert bundle.conversation == top_k_indices(attention_weights, 3)
    assert bundle.rule == [0]
    assert len(bundle.emotion) == 3
