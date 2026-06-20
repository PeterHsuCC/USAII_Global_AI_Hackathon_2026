import uuid

from sqlalchemy.orm import sessionmaker

from backend.audit.service import record_audit_event, verify_chain
from backend.db.models import AuditLog


def test_single_event_chain_is_valid(db_session):
    actor_id = uuid.uuid4()
    record_audit_event(db_session, actor_id=actor_id, actor_role="analyst", event_type="case_submitted")
    db_session.commit()

    assert verify_chain(db_session) is True
    row = db_session.query(AuditLog).one()
    assert row.previous_hash is None
    assert row.metadata_hash is not None


def test_chain_links_consecutive_events(db_session):
    actor_id = uuid.uuid4()
    first = record_audit_event(db_session, actor_id=actor_id, actor_role="analyst", event_type="case_submitted")
    second = record_audit_event(db_session, actor_id=actor_id, actor_role="analyst", event_type="decision_recorded")
    db_session.commit()

    assert second.previous_hash == first.metadata_hash
    assert verify_chain(db_session) is True


def test_tampered_row_breaks_chain(db_session):
    actor_id = uuid.uuid4()
    record_audit_event(db_session, actor_id=actor_id, actor_role="analyst", event_type="case_submitted")
    record_audit_event(db_session, actor_id=actor_id, actor_role="analyst", event_type="decision_recorded")
    db_session.commit()

    tampered = db_session.query(AuditLog).filter_by(event_type="case_submitted").one()
    tampered.event_type = "case_submitted_TAMPERED"
    db_session.commit()

    assert verify_chain(db_session) is False


def test_chain_continues_correctly_from_a_second_independent_session(db_session):
    """record_audit_event derives previous_hash/sequence_number fresh from
    the DB on every call rather than trusting an in-memory cache, so a
    second session (standing in for a second process touching the same
    DB, e.g. the seed script running once while the server is already up)
    must still chain correctly off of whatever the first session wrote.
    """
    actor_id = uuid.uuid4()
    record_audit_event(db_session, actor_id=actor_id, actor_role="analyst", event_type="case_submitted")
    db_session.commit()

    OtherSession = sessionmaker(bind=db_session.get_bind())
    with OtherSession() as other_session:
        second = record_audit_event(other_session, actor_id=actor_id, actor_role="analyst", event_type="decision_recorded")
        other_session.commit()
        second_previous_hash = second.previous_hash  # read before the session closes and detaches `second`

    first = db_session.query(AuditLog).filter_by(event_type="case_submitted").one()
    assert second_previous_hash == first.metadata_hash
    assert verify_chain(db_session) is True


def test_record_audit_event_commits_before_releasing_the_lock(db_session):
    """The lock must cover the commit, not just the flush -- otherwise a
    second session that can't see this one's uncommitted insert would
    derive the same previous_hash/sequence_number and fork the chain. This
    calls record_audit_event with no explicit commit by the caller (relying
    on the function's own internal commit, which is the actual fix) and
    then has a second, independent session chain off of it immediately."""
    actor_id = uuid.uuid4()
    first = record_audit_event(db_session, actor_id=actor_id, actor_role="analyst", event_type="case_submitted")
    # Deliberately no db_session.commit() here.

    OtherSession = sessionmaker(bind=db_session.get_bind())
    with OtherSession() as other_session:
        second = record_audit_event(other_session, actor_id=actor_id, actor_role="analyst", event_type="decision_recorded")
        second_previous_hash = second.previous_hash
        second_sequence_number = second.sequence_number

    assert second_previous_hash == first.metadata_hash
    assert second_sequence_number == first.sequence_number + 1