"""Organization-scoped read helpers.

Plan decision #2: SQLite has no Row-Level Security, so every read that
could cross an organization boundary goes through one of these functions
instead of an ad hoc query, keeping the `organization_id` filter mandatory
rather than something each router has to remember to add.
"""

import uuid

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.db.models import AuditLog, Case, User


def get_case_for_org(session: Session, case_id: uuid.UUID, organization_id: uuid.UUID) -> Case | None:
    return session.execute(
        select(Case).where(Case.case_id == case_id, Case.organization_id == organization_id)
    ).scalar_one_or_none()


def list_cases_for_org(
    session: Session,
    organization_id: uuid.UUID,
    *,
    status: str | None = None,
    priority: str | None = None,
) -> list[Case]:
    stmt = select(Case).where(Case.organization_id == organization_id)
    if status is not None:
        stmt = stmt.where(Case.status == status)
    if priority is not None:
        stmt = stmt.where(Case.priority == priority)
    stmt = stmt.order_by(Case.created_at.desc())
    return list(session.execute(stmt).scalars().all())


def get_user_by_email(session: Session, email: str) -> User | None:
    return session.execute(select(User).where(User.email == email)).scalar_one_or_none()


def list_audit_events_for_org(session: Session, organization_id: uuid.UUID) -> list[AuditLog]:
    """audit_log has no organization_id column (Section 15.5's minimum
    schema doesn't define multi-tenant scoping for it), so this scopes by
    joining case_id -> cases.organization_id for case-scoped events, or
    actor_id -> users.organization_id for system/auth events with no case
    (case_id IS NULL). An event with neither a matching case nor a
    resolvable actor (e.g. a login attempt against an email that doesn't
    exist) is attributable to no organization and won't appear in any
    org-scoped view -- a deliberate, documented gap, not an oversight.
    """
    stmt = (
        select(AuditLog)
        .outerjoin(Case, Case.case_id == AuditLog.case_id)
        .outerjoin(User, User.user_id == AuditLog.actor_id)
        .where(or_(Case.organization_id == organization_id, User.organization_id == organization_id))
        .order_by(AuditLog.event_timestamp)
    )
    return list(session.execute(stmt).scalars().all())