import uuid
from datetime import date, datetime, timezone

from backend.db.models import Case, Organization, User
from backend.db.queries import get_case_for_org, get_user_by_email, list_cases_for_org
from backend.db.state_machine import SUBMITTED


def _make_org_with_case(session, org_name: str) -> tuple[Organization, Case]:
    org = Organization(name=org_name)
    session.add(org)
    session.flush()

    user = User(organization_id=org.organization_id, role="analyst", email=f"{org_name}@demo.org", hashed_password="x")
    session.add(user)
    session.flush()

    case = Case(
        organization_id=org.organization_id,
        submitted_by=user.user_id,
        status=SUBMITTED,
        priority="standard",
        source_type="api",
        retention_until=date(2030, 1, 1),
    )
    session.add(case)
    session.flush()
    return org, case


def test_get_case_for_org_returns_none_across_orgs(db_session):
    org_a, case_a = _make_org_with_case(db_session, "org-a")
    org_b, _case_b = _make_org_with_case(db_session, "org-b")
    db_session.commit()

    assert get_case_for_org(db_session, case_a.case_id, org_a.organization_id) is not None
    assert get_case_for_org(db_session, case_a.case_id, org_b.organization_id) is None


def test_list_cases_for_org_does_not_leak_other_orgs(db_session):
    org_a, case_a = _make_org_with_case(db_session, "org-a")
    org_b, case_b = _make_org_with_case(db_session, "org-b")
    db_session.commit()

    cases_a = list_cases_for_org(db_session, org_a.organization_id)
    cases_b = list_cases_for_org(db_session, org_b.organization_id)

    assert [c.case_id for c in cases_a] == [case_a.case_id]
    assert [c.case_id for c in cases_b] == [case_b.case_id]


def test_get_user_by_email_is_global_lookup_for_login(db_session):
    org, _case = _make_org_with_case(db_session, "org-a")
    db_session.commit()

    user = get_user_by_email(db_session, "org-a@demo.org")
    assert user is not None
    assert user.organization_id == org.organization_id
    assert get_user_by_email(db_session, "nobody@demo.org") is None