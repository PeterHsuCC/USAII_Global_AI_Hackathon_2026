import asyncio
import uuid
from datetime import date

import pytest

from backend.db.models import AnalysisJob, Case, CaseMessage, Conversation, DLQErrorMetadata, Organization, Result, User
from backend.db.state_machine import DLQ_INVESTIGATION, QUEUED, READY_FOR_REVIEW
from backend.jobs.queue import AnalysisJobQueue, STATUS_DLQ, STATUS_RETRY_SCHEDULED, STATUS_SUCCEEDED, process_job_once
from backend.model_runtime.loader import MODEL_VERSION_STUB


def _make_case_with_job(session) -> tuple[Case, AnalysisJob]:
    org = Organization(name="org")
    session.add(org)
    session.flush()

    user = User(organization_id=org.organization_id, role="analyst", email=f"{uuid.uuid4()}@demo.org", hashed_password="x")
    session.add(user)
    session.flush()

    case = Case(
        organization_id=org.organization_id,
        submitted_by=user.user_id,
        status=QUEUED,
        priority="standard",
        source_type="api",
        retention_until=date(2030, 1, 1),
    )
    session.add(case)
    session.flush()

    conversation = Conversation(case_id=case.case_id)
    session.add(conversation)
    session.flush()

    for seq, (speaker, text) in enumerate([("SPEAKER_A", "our little secret ok?"), ("SPEAKER_B", "ok i promise")]):
        session.add(
            CaseMessage(
                case_id=case.case_id,
                conversation_id=conversation.conversation_id,
                message_sequence=seq,
                speaker_local_id=speaker,
                redacted_content=text,
                retention_until=date(2030, 1, 1),
            )
        )

    job = AnalysisJob(
        case_id=case.case_id,
        status="queued",
        model_version=MODEL_VERSION_STUB,
        rule_version="rules-v1",
        preprocessing_version="preprocessing-v1",
    )
    session.add(job)
    session.flush()
    session.commit()
    return case, job


def test_successful_job_writes_result_and_advances_case(db_session):
    case, job = _make_case_with_job(db_session)

    outcome = process_job_once(db_session, job.job_id)

    assert outcome.status == STATUS_SUCCEEDED
    db_session.refresh(case)
    db_session.refresh(job)
    assert case.status == READY_FOR_REVIEW
    assert job.status == "succeeded"

    results = db_session.query(Result).filter_by(case_id=case.case_id).all()
    assert len(results) == 1
    assert results[0].risk_level in {"low", "medium", "high", "critical"}
    assert results[0].evidence_json is not None


def test_transient_failure_retries_then_eventually_dlqs(db_session, monkeypatch):
    case, job = _make_case_with_job(db_session)

    def _raise_transient(*args, **kwargs):
        raise ConnectionError("upstream timeout")

    monkeypatch.setattr("backend.jobs.queue.run_analysis", _raise_transient)

    outcome1 = process_job_once(db_session, job.job_id)
    assert outcome1.status == STATUS_RETRY_SCHEDULED
    db_session.refresh(case)
    db_session.refresh(job)
    assert case.status == QUEUED
    assert job.attempt_count == 1

    outcome2 = process_job_once(db_session, job.job_id)
    assert outcome2.status == STATUS_RETRY_SCHEDULED
    db_session.refresh(job)
    assert job.attempt_count == 2

    outcome3 = process_job_once(db_session, job.job_id)
    assert outcome3.status == STATUS_DLQ
    db_session.refresh(case)
    db_session.refresh(job)
    assert case.status == DLQ_INVESTIGATION
    assert job.status == "failed"

    dlq_rows = db_session.query(DLQErrorMetadata).filter_by(job_id=job.job_id).all()
    assert len(dlq_rows) == 1
    assert dlq_rows[0].error_category == "transient"
    assert dlq_rows[0].attempt_count == 3


def test_permanent_failure_goes_straight_to_dlq_without_retry(db_session, monkeypatch):
    case, job = _make_case_with_job(db_session)

    def _raise_permanent(*args, **kwargs):
        raise ValueError("invalid input schema")

    monkeypatch.setattr("backend.jobs.queue.run_analysis", _raise_permanent)

    outcome = process_job_once(db_session, job.job_id)

    assert outcome.status == STATUS_DLQ
    db_session.refresh(case)
    db_session.refresh(job)
    assert case.status == DLQ_INVESTIGATION
    assert job.attempt_count == 1

    dlq_rows = db_session.query(DLQErrorMetadata).filter_by(job_id=job.job_id).all()
    assert dlq_rows[0].error_category == "permanent"


def test_already_succeeded_job_is_skipped(db_session):
    case, job = _make_case_with_job(db_session)
    process_job_once(db_session, job.job_id)

    from backend.jobs.queue import STATUS_SKIPPED

    outcome = process_job_once(db_session, job.job_id)
    assert outcome.status == STATUS_SKIPPED


def test_async_queue_drains_enqueued_job(db_sessionmaker):
    with db_sessionmaker() as session:
        case, job = _make_case_with_job(session)
        case_id, job_id = case.case_id, job.job_id

    async def _drive() -> None:
        queue = AnalysisJobQueue(session_factory=db_sessionmaker)
        queue.start()
        queue.enqueue(job_id)
        await asyncio.wait_for(queue._queue.join(), timeout=10)
        await queue.stop()

    asyncio.run(_drive())

    with db_sessionmaker() as session:
        refreshed_case = session.get(Case, case_id)
        refreshed_job = session.get(AnalysisJob, job_id)
        assert refreshed_case.status == READY_FOR_REVIEW
        assert refreshed_job.status == "succeeded"


def test_replay_unfinished_jobs_re_enqueues_queued_and_running(db_sessionmaker):
    with db_sessionmaker() as session:
        _case_a, job_a = _make_case_with_job(session)
        case_b, job_b = _make_case_with_job(session)
        job_b.status = "running"
        session.commit()
        job_a_id, job_b_id = job_a.job_id, job_b.job_id

    queue = AnalysisJobQueue(session_factory=db_sessionmaker)
    count = queue.replay_unfinished_jobs()

    assert count == 2
    assert queue._queue.qsize() == 2