"""Retention/deletion flow (v6 Section 10.1).

Retention period expires -> check for legal/safeguarding hold -> if a hold
is active, defer deletion and log the deferral; otherwise delete the
conversation content and write a deletion audit event.

No real cache layer exists in this build to purge cache entries from
(plan decision #5 -- rate-limit/job-status state is in-memory but not
case-keyed), so that step of the doc's flow has nothing to do here.

Nothing schedules this automatically (no cron in this build); it's
triggered via POST /admin/retention/run.
"""

import uuid
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.audit.service import SYSTEM_ACTOR_ID, SYSTEM_ACTOR_ROLE, record_audit_event
from backend.db.models import Case, CaseMessage


def run_retention_sweep(
    session: Session, *, organization_id: uuid.UUID | None = None, today: date | None = None
) -> dict[str, int]:
    today = today or date.today()
    stmt = select(Case).where(Case.retention_until <= today)
    if organization_id is not None:
        stmt = stmt.where(Case.organization_id == organization_id)
    expired_cases = session.execute(stmt).scalars().all()

    deleted = 0
    deferred = 0
    for case in expired_cases:
        if case.hold_status:
            deferred += 1
            record_audit_event(
                session,
                actor_id=SYSTEM_ACTOR_ID,
                actor_role=SYSTEM_ACTOR_ROLE,
                event_type="retention_deletion_deferred",
                case_id=case.case_id,
            )
            continue

        messages = session.execute(select(CaseMessage).where(CaseMessage.case_id == case.case_id)).scalars().all()
        for message in messages:
            session.delete(message)

        deleted += 1
        record_audit_event(
            session,
            actor_id=SYSTEM_ACTOR_ID,
            actor_role=SYSTEM_ACTOR_ROLE,
            event_type="retention_deletion_completed",
            case_id=case.case_id,
        )

    session.commit()
    return {"deleted": deleted, "deferred": deferred}