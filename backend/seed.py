"""Creates a demo organization with one user per role, for local runs.

    python -m backend.seed
"""

from sqlalchemy.orm import Session

from backend.audit.service import record_audit_event
from backend.auth.security import hash_password
from backend.db.base import SessionLocal, create_all
from backend.db.models import Organization, User

DEMO_ORG_NAME = "Demo Safeguarding Org"
DEMO_PASSWORD = "ChangeMe123!"

DEMO_USERS = (
    ("analyst@demo.org", "analyst"),
    ("senior.reviewer@demo.org", "senior_reviewer"),
    ("safeguarding@demo.org", "safeguarding_specialist"),
    ("org.admin@demo.org", "organization_admin"),
    ("system.admin@demo.org", "system_admin"),
    ("auditor@demo.org", "auditor"),
)


def seed(session: Session) -> Organization:
    existing = session.query(Organization).filter_by(name=DEMO_ORG_NAME).one_or_none()
    if existing is not None:
        return existing

    org = Organization(name=DEMO_ORG_NAME)
    session.add(org)
    session.flush()

    hashed = hash_password(DEMO_PASSWORD)
    first_user = None
    for email, role in DEMO_USERS:
        user = User(organization_id=org.organization_id, role=role, email=email, hashed_password=hashed)
        session.add(user)
        session.flush()
        if first_user is None:
            first_user = user

    record_audit_event(
        session,
        actor_id=first_user.user_id,
        actor_role=first_user.role,
        event_type="system_seeded",
        case_id=None,
    )
    session.commit()
    return org


def main() -> None:
    create_all()
    with SessionLocal() as session:
        org = seed(session)
        print(f"Seeded organization {org.name!r} ({org.organization_id})")
        for email, role in DEMO_USERS:
            print(f"  {role:<25} {email}  password={DEMO_PASSWORD}")


if __name__ == "__main__":
    main()