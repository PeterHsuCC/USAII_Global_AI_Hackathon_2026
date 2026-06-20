"""ORM models for every table in v6 Section 15 (Data Schema).

Column names/types/constraints mirror the doc 1:1 where it specifies them;
a few columns are added because the doc explicitly defers them ("Column
types and constraints are illustrative; final definitions require
implementation review") but the running system needs them -- each is
commented with why.
"""

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base
from backend.db.state_machine import ALL_STATUSES, SUBMITTED


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# 15.12 Organizations
# ---------------------------------------------------------------------------
class Organization(Base):
    __tablename__ = "organizations"

    organization_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)

    users: Mapped[list["User"]] = relationship(back_populates="organization")
    cases: Mapped[list["Case"]] = relationship(back_populates="organization")


# ---------------------------------------------------------------------------
# 15.13 Users
# ---------------------------------------------------------------------------
ROLE_VALUES = (
    "analyst",
    "senior_reviewer",
    "safeguarding_specialist",
    "organization_admin",
    "system_admin",
    "auditor",
)


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organizations.organization_id"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)

    # Not in the doc's minimum schema (it has no auth-credential table at
    # all); required to actually authenticate users.
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    # Hook for Section 4.3 "MFA required in production deployments" -- not
    # wired to a real TOTP/SMS provider in this build (plan decision #8).
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (CheckConstraint(f"role IN {ROLE_VALUES!r}", name="ck_users_role"),)

    organization: Mapped[Organization] = relationship(back_populates="users")


# ---------------------------------------------------------------------------
# 15.2 Cases
# ---------------------------------------------------------------------------
class Case(Base):
    __tablename__ = "cases"

    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organizations.organization_id"), nullable=False
    )
    submitted_by: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.user_id"), nullable=False)
    assigned_analyst_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.user_id"))
    status: Mapped[str] = mapped_column(Text, nullable=False, default=SUBMITTED)
    priority: Mapped[str] = mapped_column(Text, nullable=False, default="standard")
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    retention_until: Mapped[date] = mapped_column(Date, nullable=False)
    hold_status: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        CheckConstraint("priority IN ('standard', 'urgent')", name="ck_cases_priority"),
        CheckConstraint("source_type IN ('file_upload', 'api')", name="ck_cases_source_type"),
        CheckConstraint(f"status IN {ALL_STATUSES!r}", name="ck_cases_status"),
    )

    organization: Mapped[Organization] = relationship(back_populates="cases")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="case")
    files: Mapped[list["CaseFile"]] = relationship(back_populates="case")
    jobs: Mapped[list["AnalysisJob"]] = relationship(back_populates="case")
    results: Mapped[list["Result"]] = relationship(back_populates="case")
    decisions: Mapped[list["AnalystDecision"]] = relationship(back_populates="case")


# ---------------------------------------------------------------------------
# 15.7 Conversations
# ---------------------------------------------------------------------------
class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("cases.case_id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)

    # Not in the doc's minimum schema -- Section 9.1's "data limitations"
    # output type is sourced partly from preprocessing (Section 14:
    # completeness checks, missing timestamps), which only runs once at
    # submission time; this is where that result is kept so the
    # Explainability Service can read it back later, in a different
    # request/process than the one that did the preprocessing.
    preprocessing_limitations_json: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("case_id", "conversation_id", name="uq_conversations_case_conversation"),)

    case: Mapped[Case] = relationship(back_populates="conversations")
    messages: Mapped[list["CaseMessage"]] = relationship(back_populates="conversation")


# ---------------------------------------------------------------------------
# 15.8 Case Messages
# ---------------------------------------------------------------------------
class CaseMessage(Base):
    __tablename__ = "case_messages"

    message_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    conversation_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    message_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    speaker_local_id: Mapped[str] = mapped_column(Text, nullable=False)
    redacted_content: Mapped[str | None] = mapped_column(Text)
    redacted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    retention_until: Mapped[date] = mapped_column(Date, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["case_id", "conversation_id"],
            ["conversations.case_id", "conversations.conversation_id"],
        ),
        CheckConstraint("message_sequence >= 0", name="ck_case_messages_sequence"),
        UniqueConstraint("conversation_id", "message_sequence", name="uq_case_messages_conversation_sequence"),
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


# ---------------------------------------------------------------------------
# 15.9 Case Files
# ---------------------------------------------------------------------------
class CaseFile(Base):
    __tablename__ = "case_files"

    file_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("cases.case_id"), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(Text, nullable=False)
    redaction_status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    retention_until: Mapped[date] = mapped_column(Date, nullable=False)

    case: Mapped[Case] = relationship(back_populates="files")


# ---------------------------------------------------------------------------
# 15.10 Model Artifacts
# ---------------------------------------------------------------------------
class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    artifact_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    checksum: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)


# ---------------------------------------------------------------------------
# 15.14 Analysis Jobs
# ---------------------------------------------------------------------------
JOB_STATUS_VALUES = ("queued", "running", "succeeded", "failed")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    job_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("cases.case_id"), nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="queued")
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    rule_version: Mapped[str] = mapped_column(Text, nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(Text, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)

    __table_args__ = (CheckConstraint(f"status IN {JOB_STATUS_VALUES!r}", name="ck_analysis_jobs_status"),)

    case: Mapped[Case] = relationship(back_populates="jobs")


# ---------------------------------------------------------------------------
# 15.3 Results
# ---------------------------------------------------------------------------
RISK_LEVEL_VALUES = ("low", "medium", "high", "critical")


class Result(Base):
    __tablename__ = "results"

    result_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("cases.case_id"), nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    rule_version: Mapped[str] = mapped_column(Text, nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(Text, nullable=False)
    risk_level: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric(4, 3))
    human_review_required: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)

    # Not an explicit doc table -- Section 9 describes the Explainability
    # Service's 5 output types as derived from "structured model and rule
    # outputs"; this column is where that structured output is persisted
    # alongside the result it was derived from (small JSON, well under the
    # 100MB object-storage threshold in Section 15.11).
    evidence_json: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(f"risk_level IN {RISK_LEVEL_VALUES!r}", name="ck_results_risk_level"),
        CheckConstraint("confidence BETWEEN 0.000 AND 1.000", name="ck_results_confidence"),
    )

    case: Mapped[Case] = relationship(back_populates="results")


# ---------------------------------------------------------------------------
# 15.4 Analyst Decisions
# ---------------------------------------------------------------------------
DECISION_TYPE_VALUES = ("refer", "close", "monitor", "more_info")
REFERRAL_STATUS_VALUES = ("pending", "submitted", "acknowledged")


class AnalystDecision(Base):
    __tablename__ = "analyst_decisions"

    decision_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("cases.case_id"), nullable=False)
    analyst_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.user_id"), nullable=False)
    decision_type: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)
    referral_status: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)

    __table_args__ = (
        CheckConstraint(f"decision_type IN {DECISION_TYPE_VALUES!r}", name="ck_decisions_type"),
        CheckConstraint(f"referral_status IN {REFERRAL_STATUS_VALUES!r} OR referral_status IS NULL", name="ck_decisions_referral_status"),
    )

    case: Mapped[Case] = relationship(back_populates="decisions")


# ---------------------------------------------------------------------------
# 15.5 Audit Log -- append-only, never updated/deleted by application code.
# ---------------------------------------------------------------------------
class AuditLog(Base):
    __tablename__ = "audit_log"

    audit_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    case_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    actor_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    actor_role: Mapped[str] = mapped_column(Text, nullable=False)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    request_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    metadata_hash: Mapped[str | None] = mapped_column(Text)
    previous_hash: Mapped[str | None] = mapped_column(Text)

    # Not in the doc's minimum schema. event_timestamp alone is not a safe
    # ordering key once events can be written from more than one thread
    # (the HTTP request thread and the background job-worker thread both
    # call record_audit_event): two events can land in the same
    # microsecond, and ORDER BY event_timestamp does not then guarantee
    # the order rows were actually chained in. This monotonic counter,
    # assigned under the same lock as the hash chain, is the real
    # insertion order and is what verify_chain() must sort by.
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)


# ---------------------------------------------------------------------------
# 15.6 DLQ Error Metadata
# ---------------------------------------------------------------------------
FAILURE_STAGE_VALUES = ("validation", "preprocessing", "inference", "explainability")
ERROR_CATEGORY_VALUES = ("transient", "permanent", "resource", "unknown")
DLQ_RESOLUTION_VALUES = ("investigating", "redriven", "closed")


class DLQErrorMetadata(Base):
    __tablename__ = "dlq_error_metadata"

    dlq_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    job_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("analysis_jobs.job_id"), nullable=False)
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("cases.case_id"), nullable=False)
    failure_stage: Mapped[str] = mapped_column(Text, nullable=False)
    error_category: Mapped[str] = mapped_column(Text, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str | None] = mapped_column(Text)
    last_failure_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    resolution_status: Mapped[str | None] = mapped_column(Text, default="investigating")

    __table_args__ = (
        CheckConstraint(f"failure_stage IN {FAILURE_STAGE_VALUES!r}", name="ck_dlq_failure_stage"),
        CheckConstraint(f"error_category IN {ERROR_CATEGORY_VALUES!r}", name="ck_dlq_error_category"),
    )