"""Review Service (v6 Section 3, Section 15.4 Analyst Decisions).

The doc doesn't define a separate "start review" endpoint, so submitting a
decision against a case still in Ready for Review implicitly advances it
to Under Review first -- the analyst submitting a decision is exactly the
review the state name describes.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api.deps import get_db, new_request_id
from backend.api.schemas import DecisionOut, DecisionSubmitRequest
from backend.audit.service import record_audit_event
from backend.auth.dependencies import CurrentUser, require_permission
from backend.auth.roles import SUBMIT_DECISION, can_access_case
from backend.db.models import AnalystDecision
from backend.db.queries import get_case_for_org
from backend.db.state_machine import CLOSED, MORE_INFO_REQUIRED, READY_FOR_REVIEW, REFERRED, UNDER_REVIEW, transition
from backend.monitoring.metrics import increment
from backend.preprocessing.pii_redaction import redact_text

router = APIRouter(prefix="/cases", tags=["review"])

_STATUS_BY_DECISION = {
    "refer": REFERRED,
    "close": CLOSED,
    "more_info": MORE_INFO_REQUIRED,
    # "monitor" deliberately absent: Section 15.4 says it does not
    # transition case status.
}


@router.post("/{case_id}/decisions", response_model=DecisionOut, status_code=status.HTTP_201_CREATED)
def submit_decision(
    case_id: uuid.UUID,
    payload: DecisionSubmitRequest,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_permission(SUBMIT_DECISION)),
) -> DecisionOut:
    # for_update=True: lock the row for this transaction so a second
    # concurrent decision on the same case can't both pass the transition
    # check below against the same stale status (Section 15.4).
    case = get_case_for_org(db, case_id, current_user.organization_id, for_update=True)
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")

    is_referred = any(d.decision_type == "refer" for d in case.decisions)
    if not can_access_case(current_user.role, current_user.user_id, case, is_referred=is_referred):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not authorized to act on this case")

    if case.status == READY_FOR_REVIEW:
        transition(case, UNDER_REVIEW)

    new_status = _STATUS_BY_DECISION.get(payload.decision_type)
    if new_status is not None:
        try:
            transition(case, new_status)
        except Exception as exc:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"Case in status {case.status!r} cannot receive a {payload.decision_type!r} decision"
            ) from exc

    rationale = redact_text(payload.rationale).redacted_text if payload.rationale else None

    decision = AnalystDecision(
        case_id=case.case_id,
        analyst_id=current_user.user_id,
        decision_type=payload.decision_type,
        rationale=rationale,
        referral_status="pending" if payload.decision_type == "refer" else None,
    )
    db.add(decision)

    record_audit_event(
        db,
        actor_id=current_user.user_id,
        actor_role=current_user.role,
        event_type="decision_recorded",
        case_id=case.case_id,
        request_id=new_request_id(),
    )
    db.commit()
    db.refresh(decision)
    increment(f"decision_recorded.{payload.decision_type}")

    return DecisionOut.model_validate(decision)