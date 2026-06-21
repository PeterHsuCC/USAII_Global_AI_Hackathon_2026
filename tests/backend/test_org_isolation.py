import dataclasses
import uuid
from datetime import date, datetime, timezone

from backend.config import settings
from backend.db.models import AnalysisJob, Case, Organization, User
from backend.db.queries import get_case_for_org, get_user_by_email, list_cases_for_org
from backend.db.state_machine import SUBMITTED
from backend.jobs.dlq import create_dlq_entry, is_over_alert_threshold, open_dlq_depth


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


def test_dlq_depth_is_org_scoped_when_organization_id_is_given(db_session, monkeypatch):
    monkeypatch.setattr("backend.jobs.dlq.settings", dataclasses.replace(settings, dlq_alert_threshold=0))

    org_a, _case_a = _make_org_with_case(db_session, "org-a")
    org_b, case_b = _make_org_with_case(db_session, "org-b")
    job_b = AnalysisJob(
        case_id=case_b.case_id, status="failed", model_version="v1", rule_version="v1", preprocessing_version="v1"
    )
    db_session.add(job_b)
    db_session.flush()
    create_dlq_entry(
        db_session,
        job_id=job_b.job_id,
        case_id=case_b.case_id,
        failure_stage="inference",
        error_category="unknown",
        attempt_count=3,
        model_version="v1",
    )
    db_session.commit()

    # Org B genuinely has an open DLQ entry over threshold...
    assert open_dlq_depth(db_session, organization_id=org_b.organization_id) == 1
    assert is_over_alert_threshold(db_session, organization_id=org_b.organization_id) is True

    # ...but org A's admin must not see it, scoped or not.
    assert open_dlq_depth(db_session, organization_id=org_a.organization_id) == 0
    assert is_over_alert_threshold(db_session, organization_id=org_a.organization_id) is False

    # The unscoped (organization_id=None) call is the platform-ops-wide view
    # used internally by process_job_once -- still sees every tenant.
    assert open_dlq_depth(db_session) == 1


def test_get_user_by_email_is_global_lookup_for_login(db_session):
    org, _case = _make_org_with_case(db_session, "org-a")
    db_session.commit()

    user = get_user_by_email(db_session, "org-a@demo.org")
    assert user is not None
    assert user.organization_id == org.organization_id
    assert get_user_by_email(db_session, "nobody@demo.org") is None