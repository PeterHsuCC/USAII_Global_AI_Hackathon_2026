"""DLQ failure classification (v6 Section 8.1) and operational helpers
(Section 8.3/8.4)."""

import random
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.config import settings
from backend.db.models import AnalysisJob, Case, DLQErrorMetadata
from backend.db.state_machine import CLOSED, QUEUED, transition

TRANSIENT = "transient"
PERMANENT = "permanent"
RESOURCE = "resource"
UNKNOWN = "unknown"

INVESTIGATING = "investigating"
REDRIVEN = "redriven"
RESOLVED_CLOSED = "closed"


class DLQEntryAlreadyResolvedError(Exception):
    """Section 8.3's resolution_status (investigating | redriven | closed)
    is a one-way state machine: an entry already redriven or closed must not
    be redriven/closed again (e.g. a double-click, or a slow first request
    retried by the client). Without this guard, redrive_dlq_entry would
    happily re-enqueue the job and reset attempt_count a second time."""

    def __init__(self, current: str) -> None:
        super().__init__(f"DLQ entry already resolved (resolution_status={current!r})")
        self.current = current

# Section 8.1: only transient/resource failures are retried; permanent and
# unknown go straight to the DLQ.
RETRYABLE_CATEGORIES = frozenset({TRANSIENT, RESOURCE})


def classify_exception(exc: Exception) -> str:
    message = str(exc).lower()
    if "out of memory" in message or ("cuda" in message and "memory" in message):
        return RESOURCE
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return TRANSIENT
    if isinstance(exc, (ValueError, KeyError, TypeError, LookupError)):
        return PERMANENT
    return UNKNOWN


def backoff_delay_seconds(attempt: int) -> float:
    """Exponential backoff with jitter (Section 7.3)."""
    base = settings.retry_base_delay_seconds * (2 ** max(attempt - 1, 0))
    return base + random.uniform(0, base * 0.5)


def create_dlq_entry(
    session: Session,
    *,
    job_id: uuid.UUID,
    case_id: uuid.UUID,
    failure_stage: str,
    error_category: str,
    attempt_count: int,
    model_version: str | None,
) -> DLQErrorMetadata:
    entry = DLQErrorMetadata(
        job_id=job_id,
        case_id=case_id,
        failure_stage=failure_stage,
        error_category=error_category,
        attempt_count=attempt_count,
        model_version=model_version,
        last_failure_at=datetime.now(timezone.utc),
        resolution_status="investigating",
    )
    session.add(entry)
    session.flush()
    return entry


def open_dlq_depth(session: Session) -> int:
    return session.execute(
        select(func.count()).select_from(DLQErrorMetadata).where(DLQErrorMetadata.resolution_status == "investigating")
    ).scalar_one()


def is_over_alert_threshold(session: Session) -> bool:
    """Section 8.4: alert when DLQ depth exceeds a configurable threshold."""
    return open_dlq_depth(session) > settings.dlq_alert_threshold


def redrive_dlq_entry(session: Session, entry: DLQErrorMetadata) -> AnalysisJob:
    """Section 8.2: 'Correct transient/configuration problem -> Redrive to
    main queue.' Resets attempt_count, giving the job a fresh retry budget
    on the assumption the underlying problem has been fixed."""
    if entry.resolution_status != INVESTIGATING:
        raise DLQEntryAlreadyResolvedError(entry.resolution_status)

    case = session.get(Case, entry.case_id)
    job = session.get(AnalysisJob, entry.job_id)

    transition(case, QUEUED)
    job.status = "queued"
    job.attempt_count = 0
    entry.resolution_status = REDRIVEN
    session.commit()
    return job


def close_dlq_entry_as_invalid(session: Session, entry: DLQErrorMetadata) -> None:
    """Section 8.2: 'Mark input as invalid -> Notify submitting analyst.'
    Notification delivery itself is out of scope here; the case status
    change and audit trail are what this backend is responsible for."""
    if entry.resolution_status != INVESTIGATING:
        raise DLQEntryAlreadyResolvedError(entry.resolution_status)

    case = session.get(Case, entry.case_id)
    transition(case, CLOSED)
    entry.resolution_status = RESOLVED_CLOSED
    session.commit()