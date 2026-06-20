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


class MessageIn(BaseModel):
    speaker: str
    text: str
    timestamp: datetime | None = None


class CaseSubmitRequest(BaseModel):
    priority: Literal["standard", "urgent"] = "standard"
    messages: list[MessageIn] = Field(min_length=1)


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
    attention_disclaimer: str


class ConfidenceAndUncertaintyOut(BaseModel):
    risk_level: str
    confidence: float
    uncertainty: float
    mc_dropout_variance: float


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


class CaseDetailResponse(BaseModel):
    case: CaseSummary
    results: list[ResultOut]
    decisions: list[DecisionOut]


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