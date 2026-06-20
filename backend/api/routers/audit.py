"""Auditor read-only access to audit metadata (v6 Section 4.2)."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.api.schemas import AuditEventOut, AuditListResponse
from backend.audit.service import verify_chain
from backend.auth.dependencies import CurrentUser, require_permission
from backend.auth.roles import READ_AUDIT_LOG
from backend.db.queries import list_audit_events_for_org

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=AuditListResponse)
def list_audit_events(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_permission(READ_AUDIT_LOG)),
) -> AuditListResponse:
    events = list_audit_events_for_org(db, current_user.organization_id)
    return AuditListResponse(
        events=[AuditEventOut.model_validate(e) for e in events],
        chain_valid=verify_chain(db),
    )