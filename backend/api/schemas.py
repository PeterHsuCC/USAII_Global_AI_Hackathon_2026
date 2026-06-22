"""Pydantic request/response models for the API layer."""

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


# Operational abuse/payload-size guards only -- NOT a proxy for what the
# trained encoders actually see. The real-mode MessageEncoder/
# GoEmotionsClassifier still truncate at 128 tokens per message
# (model_runtime/loader.py); that's a model-coverage limitation surfaced via
# the explainability service's data_limitations (job_runner.py), not an API
# rejection -- rejecting a long but legitimate message (e.g. a threat letter)
# would lose evidence instead of just analyzing it partially.
MAX_MESSAGE_TEXT_LENGTH = 3_000

# Cases over this many messages are split into multiple sequential
# windows for analysis (job_runner.py's build_windows), not just the most
# recent RISK_PLATFORM_WINDOW_SIZE -- but still bounded here so one
# submission can't queue an unbounded number of model-inference passes.
# Together with MAX_MESSAGE_TEXT_LENGTH above, this already bounds total
# per-submission text to MAX_MESSAGES_PER_CASE * MAX_MESSAGE_TEXT_LENGTH --
# a separate total-text-length check would never be able to fire on its own.
MAX_MESSAGES_PER_CASE = 120


class MessageIn(BaseModel):
    speaker: str
    text: str = Field(max_length=MAX_MESSAGE_TEXT_LENGTH)
    timestamp: datetime | None = None


class CaseSubmitRequest(BaseModel):
    priority: Literal["standard", "urgent"] = "standard"
    messages: list[MessageIn] = Field(min_length=1, max_length=MAX_MESSAGES_PER_CASE)


class CaseSubmitResponse(BaseModel):
    case_id: uuid.UUID
    job_id: uuid.UUID
    status: str


class CaseSummary(BaseModel):
    case_id: uuid.UUID
    status: str
    priority: str
    source_type: str
    created_at: datetime
    updated_at: datetime
    retention_until: date
    assigned_analyst_id: uuid.UUID | None
    submitted_by: uuid.UUID

    model_config = {"from_attributes": True}


class CaseListResponse(BaseModel):
    cases: list[CaseSummary]


class TriggeredSignalOut(BaseModel):
    name: str
    score: float
    source: Literal["llm", "rule"]
    message_sequences: tuple[int, ...]


class RuleEvidenceItemOut(BaseModel):
    rule_id: str
    severity: str
    matched_message_sequence: int
    redacted_evidence_span: str | None


class ModelEvidenceOut(BaseModel):
    high_risk_message_sequences: tuple[int, ...]
    attention_focus_message_sequences: tuple[int, ...]
    cyberbullying_component_score: float
    grooming_component_score: float
    # Added after some Result rows were already persisted with older
    # evidence_json that predates these keys -- default to None/empty so
    # historical cases stay viewable instead of 500ing on read.
    rule_safety_score: float | None = None
    emotion_component_score: float | None = None
    mapped_emotions: dict[str, float] = Field(default_factory=dict)
    attention_disclaimer: str


class ConfidenceAndUncertaintyOut(BaseModel):
    risk_level: str
    confidence: float
    uncertainty: float
    mc_dropout_variance: float
    # Optional/defaulted for the same reason as ModelEvidenceOut's
    # rule_safety_score/emotion_component_score above: older persisted
    # evidence_json predates these keys.
    operable_risk_score: float | None = None
    operable_risk_level: str | None = None


class ExplainabilityOut(BaseModel):
    triggered_signals: tuple[TriggeredSignalOut, ...]
    rule_evidence: tuple[RuleEvidenceItemOut, ...]
    model_evidence: ModelEvidenceOut
    confidence_and_uncertainty: ConfidenceAndUncertaintyOut
    data_limitations: tuple[str, ...]
    disclaimer: str


class ResultOut(BaseModel):
    result_id: uuid.UUID
    case_id: uuid.UUID
    risk_level: str
    confidence: float | None
    human_review_required: bool
    created_at: datetime
    explainability: ExplainabilityOut | None


class DecisionOut(BaseModel):
    decision_id: uuid.UUID
    case_id: uuid.UUID
    analyst_id: uuid.UUID
    decision_type: str
    rationale: str | None
    referral_status: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CaseMessageOut(BaseModel):
    message_sequence: int
    speaker_local_id: str
    redacted_content: str | None

    model_config = {"from_attributes": True}


class CaseDetailResponse(BaseModel):
    case: CaseSummary
    results: list[ResultOut]
    decisions: list[DecisionOut]
    messages: list[CaseMessageOut]


class DecisionSubmitRequest(BaseModel):
    decision_type: Literal["refer", "close", "monitor", "more_info"]
    rationale: str | None = None


class AuditEventOut(BaseModel):
    audit_id: uuid.UUID
    case_id: uuid.UUID | None
    actor_id: uuid.UUID
    actor_role: str
    event_type: str
    event_timestamp: datetime
    request_id: uuid.UUID | None

    model_config = {"from_attributes": True}


class AuditListResponse(BaseModel):
    events: list[AuditEventOut]
    chain_valid: bool


class DLQEntryOut(BaseModel):
    dlq_id: uuid.UUID
    job_id: uuid.UUID
    case_id: uuid.UUID
    failure_stage: str
    error_category: str
    attempt_count: int
    model_version: str | None
    last_failure_at: datetime
    resolution_status: str | None

    model_config = {"from_attributes": True}


class DLQListResponse(BaseModel):
    entries: list[DLQEntryOut]
    alert: bool


class MetricsResponse(BaseModel):
    counters: dict[str, int]
    open_dlq_depth: int


class RetentionSweepResponse(BaseModel):
    deleted: int
    deferred: int