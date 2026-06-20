"""System Admin operational endpoints: DLQ investigation/redrive (v6
Section 8.2/8.4) and a metrics snapshot standing in for Section 12."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.api.schemas import DLQEntryOut, DLQListResponse, MetricsResponse, RetentionSweepResponse
from backend.auth.dependencies import CurrentUser, require_permission
from backend.auth.roles import MAINTAIN_PLATFORM
from backend.db.models import Case, DLQErrorMetadata
from backend.jobs.dlq import (
    DLQEntryAlreadyResolvedError,
    close_dlq_entry_as_invalid,
    is_over_alert_threshold,
    open_dlq_depth,
    redrive_dlq_entry,
)
from backend.jobs.queue import job_queue
from backend.monitoring.metrics import snapshot
from backend.retention.deletion import run_retention_sweep

router = APIRouter(prefix="/admin", tags=["admin"])


def _get_org_scoped_dlq_entry(db: Session, dlq_id: uuid.UUID, organization_id: uuid.UUID) -> DLQErrorMetadata:
    # Only called from the redrive/close mutation endpoints below (never a
    # plain read), so locking unconditionally is safe -- see the for_update
    # note on get_case_for_org for the same Postgres-correct/SQLite-best-effort
    # caveat.
    entry = db.execute(
        select(DLQErrorMetadata)
        .join(Case, Case.case_id == DLQErrorMetadata.case_id)
        .where(DLQErrorMetadata.dlq_id == dlq_id, Case.organization_id == organization_id)
        .with_for_update()
    ).scalar_one_or_none()
    if entry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "DLQ entry not found")
    return entry


@router.get("/dlq", response_model=DLQListResponse)
def list_dlq_entries(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_permission(MAINTAIN_PLATFORM)),
) -> DLQListResponse:
    entries = db.execute(
        select(DLQErrorMetadata)
        .join(Case, Case.case_id == DLQErrorMetadata.case_id)
        .where(Case.organization_id == current_user.organization_id, DLQErrorMetadata.resolution_status == "investigating")
        .order_by(DLQErrorMetadata.last_failure_at)
    ).scalars().all()
    return DLQListResponse(entries=[DLQEntryOut.model_validate(e) for e in entries], alert=is_over_alert_threshold(db))


@router.post("/dlq/{dlq_id}/redrive", response_model=DLQEntryOut)
def redrive(
    dlq_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_permission(MAINTAIN_PLATFORM)),
) -> DLQEntryOut:
    entry = _get_org_scoped_dlq_entry(db, dlq_id, current_user.organization_id)
    try:
        job = redrive_dlq_entry(db, entry)
    except DLQEntryAlreadyResolvedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    job_queue.enqueue(job.job_id)
    return DLQEntryOut.model_validate(entry)


@router.post("/dlq/{dlq_id}/close", response_model=DLQEntryOut)
def close_as_invalid(
    dlq_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_permission(MAINTAIN_PLATFORM)),
) -> DLQEntryOut:
    entry = _get_org_scoped_dlq_entry(db, dlq_id, current_user.organization_id)
    try:
        close_dlq_entry_as_invalid(db, entry)
    except DLQEntryAlreadyResolvedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return DLQEntryOut.model_validate(entry)


@router.get("/metrics", response_model=MetricsResponse)
def metrics(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_permission(MAINTAIN_PLATFORM)),
) -> MetricsResponse:
    return MetricsResponse(counters=snapshot(), open_dlq_depth=open_dlq_depth(db))


@router.post("/retention/run", response_model=RetentionSweepResponse)
def run_retention(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_permission(MAINTAIN_PLATFORM)),
) -> RetentionSweepResponse:
    result = run_retention_sweep(db, organization_id=current_user.organization_id)
    return RetentionSweepResponse(**result)