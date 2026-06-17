from dataclasses import dataclass

import torch

from ..signals.rules import RuleEvidence
from .emotion_mapping import DEFAULT_LAMBDA, map_emotions
from .emotion_score_head import EmotionScoreHead

DEFAULT_TOP_K = 3


def top_k_indices(values: torch.Tensor, k: int = DEFAULT_TOP_K) -> list[int]:
    """TopKIndices({x_i}, k): indices of the k largest values, in
    descending order (Section 14)."""
    if values.numel() == 0:
        return []
    k = min(k, values.numel())
    return torch.topk(values, k).indices.tolist()


def cyberbullying_evidence(per_message_risk: torch.Tensor, k: int = DEFAULT_TOP_K) -> list[int]:
    """E_t^cb = TopKIndices({p_cb,i^risk}, k)."""
    return top_k_indices(per_message_risk, k)


def attention_evidence(attention_weights: torch.Tensor, k: int = DEFAULT_TOP_K) -> list[int]:
    """E_t^conv = TopKIndices({alpha_i}, k). Attention indicates model
    focus, not causal or legal proof -- all evidence must be interpreted
    by a human analyst."""
    return top_k_indices(attention_weights, k)


def rule_evidence(evidence: RuleEvidence) -> list[int]:
    """E_t^rule = union over active rules j of TriggeredMessageIDs(j)."""
    return evidence.union_indices()


def per_message_emotion_scores(
    per_message_emotions: torch.Tensor,
    label_to_index: dict[str, int],
    d_t: torch.Tensor | float,
    emotion_score_head: EmotionScoreHead,
    lam: float = DEFAULT_LAMBDA,
) -> torch.Tensor:
    """S_emotion,i = sigmoid(b_m + theta^T M_i), where M_i = phi(G_i, D_t)
    for every message i (Section 14). Reuses the per-message G_i already
    computed during the Section 10.2 aggregation step (no second
    classifier pass) and the already-learned EmotionScoreHead weights.

    Note: the report writes M_i = phi(G_i, D_i) with a per-message D_i,
    but D_t is only ever extracted once per window (a single LLM call
    over the whole window, Section 10.2) rather than once per message.
    This broadcasts that single D_t to every message instead of issuing
    one additional LLM call per message.
    """
    m_i = map_emotions(per_message_emotions, label_to_index, d_t, lam=lam)
    return emotion_score_head(m_i)


def emotion_evidence(
    per_message_emotions: torch.Tensor,
    label_to_index: dict[str, int],
    d_t: torch.Tensor | float,
    emotion_score_head: EmotionScoreHead,
    lam: float = DEFAULT_LAMBDA,
    k: int = DEFAULT_TOP_K,
) -> list[int]:
    """E_t^emo = TopKIndices({S_emotion,i}, k)."""
    scores = per_message_emotion_scores(per_message_emotions, label_to_index, d_t, emotion_score_head, lam=lam)
    return top_k_indices(scores, k)


@dataclass
class EvidenceBundle:
    """Section 14's evidence indices for one Conversation Window.

    LLM evidence (message spans returned by structured extraction) is not
    included here: it requires extending the Section 3.1 LLMSafetySignals
    schema to additionally return supporting spans, which is a prompt and
    schema design decision not specified by Section 14's formulas.
    """

    cyberbullying: list[int]
    conversation: list[int]
    rule: list[int]
    emotion: list[int]


def extract_evidence(
    per_message_risk: torch.Tensor,
    attention_weights: torch.Tensor,
    rule_evidence_obj: RuleEvidence,
    per_message_emotions: torch.Tensor,
    label_to_index: dict[str, int],
    d_t: torch.Tensor | float,
    emotion_score_head: EmotionScoreHead,
    lam: float = DEFAULT_LAMBDA,
    k: int = DEFAULT_TOP_K,
) -> EvidenceBundle:
    """Assembles all four Section 14 evidence types for one Conversation
    Window into a single bundle."""
    return EvidenceBundle(
        cyberbullying=cyberbullying_evidence(per_message_risk, k),
        conversation=attention_evidence(attention_weights, k),
        rule=rule_evidence(rule_evidence_obj),
        emotion=emotion_evidence(per_message_emotions, label_to_index, d_t, emotion_score_head, lam=lam, k=k),
    )
