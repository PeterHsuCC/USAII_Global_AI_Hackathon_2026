import uuid
from datetime import date

from backend.db.models import AuditLog, Case, CaseMessage, Conversation, Organization, Result, User
from backend.db.state_machine import READY_FOR_REVIEW
from backend.retention.deletion import run_retention_sweep


def _make_expired_case(session, *, hold_status: bool = False) -> Case:
    org = Organization(name=f"org-{uuid.uuid4()}")
    session.add(org)
    session.flush()

    user = User(organization_id=org.organization_id, role="analyst", email=f"{uuid.uuid4()}@demo.org", hashed_password="x")
    session.add(user)
    session.flush()

    case = Case(
        organization_id=org.organization_id,
        submitted_by=user.user_id,
        status=READY_FOR_REVIEW,
        priority="standard",
        source_type="api",
        retention_until=date(2020, 1, 1),  # already expired
        hold_status=hold_status,
    )
    session.add(case)
    session.flush()

    conversation = Conversation(case_id=case.case_id)
    session.add(conversation)
    session.flush()

    session.add(
        CaseMessage(
            case_id=case.case_id,
            conversation_id=conversation.conversation_id,
            message_sequence=0,
            speaker_local_id="SPEAKER_A",
            redacted_content="our little secret ok?",
            retention_until=date(2020, 1, 1),
        )
    )

    session.add(
        Result(
            case_id=case.case_id,
            model_version="v1",
            rule_version="v1",
            preprocessing_version="v1",
            risk_level="medium",
            confidence=0.5,
            human_review_required=False,
            evidence_json='{"redacted_evidence_span": "our little secret ok?"}',
        )
    )
    session.commit()
    return case


def test_retention_sweep_deletes_messages_and_clears_result_evidence(db_session):
    case = _make_expired_case(db_session)

    result = run_retention_sweep(db_session)

    assert result == {"deleted": 1, "deferred": 0}
    assert db_session.query(CaseMessage).filter_by(case_id=case.case_id).count() == 0

    stored_result = db_session.query(Result).filter_by(case_id=case.case_id).one()
    assert stored_result.evidence_json is None
    # Non-content metadata is kept, not wiped wholesale.
    assert stored_result.risk_level == "medium"

    events = db_session.query(AuditLog).filter_by(case_id=case.case_id, event_type="retention_deletion_completed").all()
    assert len(events) == 1


def test_retention_sweep_defers_cases_on_hold(db_session):
    case = _make_expired_case(db_session, hold_status=True)

    result = run_retention_sweep(db_session)

    assert result == {"deleted": 0, "deferred": 1}
    # Hold means nothing gets touched.
    assert db_session.query(CaseMessage).filter_by(case_id=case.case_id).count() == 1
    stored_result = db_session.query(Result).filter_by(case_id=case.case_id).one()
    assert stored_result.evidence_json is not None

    events = db_session.query(AuditLog).filter_by(case_id=case.case_id, event_type="retention_deletion_deferred").all()
    assert len(events) == 1


def test_retention_sweep_is_org_scoped(db_session):
    case_a = _make_expired_case(db_session)
    case_b = _make_expired_case(db_session)

    result = run_retention_sweep(db_session, organization_id=case_a.organization_id)

    assert result == {"deleted": 1, "deferred": 0}
    assert db_session.query(CaseMessage).filter_by(case_id=case_a.case_id).count() == 0
    # Case B belongs to a different org and must be untouched by this sweep.
    assert db_session.query(CaseMessage).filter_by(case_id=case_b.case_id).count() == 1
