"""Case Service (submission) and Query Service (read) -- v6 Section 3.

Two of the three request paths the doc's architecture describes; the third
(Review Service) is in review.py. AI preprocessing (redaction,
anonymization) runs synchronously here, before the job is queued -- the
worker only ever sees already-redacted case_messages rows.
"""

import json
import uuid
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api.deps import get_db, new_request_id
from backend.api.schemas import (
    CaseDetailResponse,
    CaseListResponse,
    CaseSubmitRequest,
    CaseSubmitResponse,
    CaseSummary,
    DecisionOut,
    ExplainabilityOut,
    ResultOut,
)
from backend.audit.service import record_audit_event
from backend.auth.dependencies import CurrentUser, require_permission
from backend.auth.roles import SUBMIT_CASE, VIEW_ASSIGNED_CASES, can_access_case
from backend.config import settings
from backend.db.models import AnalysisJob, Case, CaseMessage, Conversation, Result
from backend.db.queries import get_case_for_org, list_cases_for_org
from backend.db.state_machine import PRIVACY_PROCESSING, QUEUED, SUBMITTED, VALIDATING, transition
from backend.jobs.queue import job_queue
from backend.model_runtime.loader import PREPROCESSING_VERSION, RULE_VERSION, current_model_version
from backend.monitoring.metrics import increment
from backend.preprocessing.anonymize import RawMessage, prepare_conversation
from backend.rate_limit.limiter import (
    RateLimitExceeded,
    acquire_concurrent_job_slot,
    check_job_submission,
    check_read,
    release_concurrent_job_slot,
)

router = APIRouter(prefix="/cases", tags=["cases"])


def _retry_after_response(exc: RateLimitExceeded) -> HTTPException:
    return HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        detail=str(exc),
        headers={"Retry-After": str(int(exc.retry_after_seconds) + 1)},
    )


@router.post("", response_model=CaseSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_case(
    payload: CaseSubmitRequest,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_permission(SUBMIT_CASE)),
) -> CaseSubmitResponse:
    try:
        check_job_submission(current_user.user_id, current_user.organization_id)
        acquire_concurrent_job_slot(current_user.user_id)
    except RateLimitExceeded as exc:
        raise _retry_after_response(exc) from exc

    # acquire_concurrent_job_slot is normally released by process_job_once
    # once a queued job reaches a terminal state -- but no job exists yet
    # at this point, so any failure before it's queued (a bad payload,
    # a DB error on commit, ...) must release the slot itself, or it leaks
    # until process restart (bounded by rate_limit_concurrent_jobs_per_analyst,
    # so the analyst is eventually locked out of submitting anything).
    try:
        retention_until = date.today() + timedelta(days=settings.redacted_conversation_retention_days)

        case = Case(
            organization_id=current_user.organization_id,
            submitted_by=current_user.user_id,
            status=SUBMITTED,
            priority=payload.priority,
            source_type="api",
            retention_until=retention_until,
        )
        db.add(case)
        db.flush()

        transition(case, VALIDATING)

        raw_messages = [
            RawMessage(speaker_external_id=m.speaker, text=m.text, timestamp=m.timestamp) for m in payload.messages
        ]
        prepared = prepare_conversation(raw_messages)

        transition(case, PRIVACY_PROCESSING)

        conversation = Conversation(
            case_id=case.case_id,
            preprocessing_limitations_json=json.dumps(list(prepared.data_limitations)),
        )
        db.add(conversation)
        db.flush()

        for msg in prepared.messages:
            db.add(
                CaseMessage(
                    case_id=case.case_id,
                    conversation_id=conversation.conversation_id,
                    message_sequence=msg.message_sequence,
                    speaker_local_id=msg.speaker_local_id,
                    redacted_content=msg.redacted_content,
                    retention_until=retention_until,
                )
            )

        job = AnalysisJob(
            case_id=case.case_id,
            status="queued",
            model_version=current_model_version(),
            rule_version=RULE_VERSION,
            preprocessing_version=PREPROCESSING_VERSION,
        )
        db.add(job)

        transition(case, QUEUED)

        record_audit_event(
            db,
            actor_id=current_user.user_id,
            actor_role=current_user.role,
            event_type="case_submitted",
            case_id=case.case_id,
            request_id=new_request_id(),
        )
        db.commit()
    except Exception:
        release_concurrent_job_slot(current_user.user_id)
        raise
    increment("case_submitted")

    job_queue.enqueue(job.job_id)

    return CaseSubmitResponse(case_id=case.case_id, job_id=job.job_id, status=case.status)


@router.get("", response_model=CaseListResponse)
def list_cases(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_permission(VIEW_ASSIGNED_CASES)),
) -> CaseListResponse:
    try:
        check_read(current_user.user_id)
    except RateLimitExceeded as exc:
        raise _retry_after_response(exc) from exc

    all_cases = list_cases_for_org(db, current_user.organization_id)
    visible = [c for c in all_cases if can_access_case(current_user.role, current_user.user_id, c)]
    return CaseListResponse(cases=[CaseSummary.model_validate(c) for c in visible])


def _result_to_out(result: Result) -> ResultOut:
    explainability = ExplainabilityOut.model_validate(json.loads(result.evidence_json)) if result.evidence_json else None
    return ResultOut(
        result_id=result.result_id,
        case_id=result.case_id,
        risk_level=result.risk_level,
        confidence=float(result.confidence) if result.confidence is not None else None,
        human_review_required=result.human_review_required,
        created_at=result.created_at,
        explainability=explainability,
    )


@router.get("/{case_id}", response_model=CaseDetailResponse)
def get_case(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_permission(VIEW_ASSIGNED_CASES)),
) -> CaseDetailResponse:
    try:
        check_read(current_user.user_id)
    except RateLimitExceeded as exc:
        raise _retry_after_response(exc) from exc

    case = get_case_for_org(db, case_id, current_user.organization_id)
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")

    is_referred = any(d.decision_type == "refer" for d in case.decisions)
    if not can_access_case(current_user.role, current_user.user_id, case, is_referred=is_referred):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not authorized to view this case")

    return CaseDetailResponse(
        case=CaseSummary.model_validate(case),
        results=[_result_to_out(r) for r in case.results],
        decisions=[DecisionOut.model_validate(d) for d in case.decisions],
    )