"""In-process async job queue + worker (v6 Section 3 Durable Job Queue /
AI Analysis Workers, Section 7.3 Reliability Controls, Section 8 DLQ flow).

"Durable" here means `analysis_jobs` rows in the DB are the source of
truth; the asyncio.Queue is just the in-process dispatch mechanism,
repopulated from the DB on startup via replay_unfinished_jobs(). This is
not a distributed broker -- see plan decision #4.

process_job_once() does exactly one attempt and is synchronous/DB-only, so
it's directly unit-testable without an event loop. AnalysisJobQueue is the
thin async wrapper that calls it via asyncio.to_thread and schedules
retries.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.audit.service import SYSTEM_ACTOR_ID, SYSTEM_ACTOR_ROLE, record_audit_event
from backend.config import settings
from backend.rate_limit.limiter import release_concurrent_job_slot
from backend.db.base import SessionLocal
from backend.db.models import AnalysisJob, Case, CaseMessage, Conversation, Result
from backend.db.state_machine import ANALYZING, DLQ_INVESTIGATION, PROCESSING_FAILED, QUEUED, READY_FOR_REVIEW, transition
from backend.explainability.service import build_explainability
from backend.jobs.dlq import RETRYABLE_CATEGORIES, backoff_delay_seconds, classify_exception, create_dlq_entry, is_over_alert_threshold
from backend.model_runtime.job_runner import JobMessage, risk_level_from_score, run_analysis
from backend.monitoring.metrics import increment

log = logging.getLogger("risk_platform.jobs")

STATUS_SUCCEEDED = "succeeded"
STATUS_RETRY_SCHEDULED = "retry_scheduled"
STATUS_DLQ = "dlq"
STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class JobAttemptOutcome:
    status: str
    retry_delay_seconds: float | None = None


def _load_case_messages(session: Session, case_id: uuid.UUID) -> list[JobMessage]:
    rows = session.execute(
        select(CaseMessage).where(CaseMessage.case_id == case_id).order_by(CaseMessage.message_sequence)
    ).scalars().all()
    return [
        JobMessage(speaker_local_id=row.speaker_local_id, redacted_content=row.redacted_content, message_sequence=row.message_sequence)
        for row in rows
    ]


def _preprocessing_limitations(session: Session, case_id: uuid.UUID) -> tuple[str, ...]:
    conversations = session.execute(select(Conversation).where(Conversation.case_id == case_id)).scalars().all()
    limitations: list[str] = []
    for conv in conversations:
        if conv.preprocessing_limitations_json:
            limitations.extend(json.loads(conv.preprocessing_limitations_json))
    return tuple(limitations)


def process_job_once(session: Session, job_id: uuid.UUID) -> JobAttemptOutcome:
    job = session.get(AnalysisJob, job_id)
    if job is None or job.status == STATUS_SUCCEEDED:
        return JobAttemptOutcome(status=STATUS_SKIPPED)

    case = session.get(Case, job.case_id)
    if case is None:
        return JobAttemptOutcome(status=STATUS_SKIPPED)

    job.status = "running"
    job.attempt_count += 1
    try:
        transition(case, ANALYZING)
    except Exception:
        pass  # already past Analyzing (e.g. a redriven job resuming) -- not fatal
    session.commit()

    try:
        messages = _load_case_messages(session, case.case_id)
        outcome = run_analysis(messages)

        risk_level = risk_level_from_score(outcome.result.overall_score.item())
        messages_by_sequence = {m.message_sequence: m.redacted_content for m in messages}
        explainability = build_explainability(
            outcome,
            messages_by_sequence=messages_by_sequence,
            risk_level=risk_level,
            preprocessing_limitations=_preprocessing_limitations(session, case.case_id),
        )

        result_row = Result(
            case_id=case.case_id,
            model_version=job.model_version,
            rule_version=job.rule_version,
            preprocessing_version=job.preprocessing_version,
            risk_level=risk_level,
            confidence=float(outcome.result.uncertainty_estimate.confidence.item()),
            human_review_required=outcome.result.human_review_required,
            evidence_json=explainability.to_json(),
        )
        session.add(result_row)

        job.status = STATUS_SUCCEEDED
        transition(case, READY_FOR_REVIEW)
        record_audit_event(
            session,
            actor_id=SYSTEM_ACTOR_ID,
            actor_role=SYSTEM_ACTOR_ROLE,
            event_type="case_analysis_completed",
            case_id=case.case_id,
        )
        session.commit()
        release_concurrent_job_slot(case.submitted_by)
        increment("case_analysis_completed")
        return JobAttemptOutcome(status=STATUS_SUCCEEDED)

    except Exception as exc:
        session.rollback()
        # Re-fetch: rollback expired the ORM objects committed above.
        job = session.get(AnalysisJob, job_id)
        case = session.get(Case, job.case_id)

        category = classify_exception(exc)
        log.warning("Analysis job %s failed (attempt %d, category=%s): %s", job_id, job.attempt_count, category, exc)

        if category in RETRYABLE_CATEGORIES and job.attempt_count < settings.max_job_attempts:
            transition(case, PROCESSING_FAILED)
            transition(case, QUEUED)
            job.status = "queued"
            session.commit()
            return JobAttemptOutcome(status=STATUS_RETRY_SCHEDULED, retry_delay_seconds=backoff_delay_seconds(job.attempt_count))

        transition(case, PROCESSING_FAILED)
        job.status = "failed"
        session.commit()

        transition(case, DLQ_INVESTIGATION)
        create_dlq_entry(
            session,
            job_id=job.job_id,
            case_id=case.case_id,
            failure_stage="inference",
            error_category=category,
            attempt_count=job.attempt_count,
            model_version=job.model_version,
        )
        record_audit_event(
            session,
            actor_id=SYSTEM_ACTOR_ID,
            actor_role=SYSTEM_ACTOR_ROLE,
            event_type="dlq_entry_created",
            case_id=case.case_id,
        )
        session.commit()
        increment("dlq_entry_created")

        if is_over_alert_threshold(session):
            log.warning("DLQ depth alert: open DLQ entries exceed configured threshold")

        release_concurrent_job_slot(case.submitted_by)
        return JobAttemptOutcome(status=STATUS_DLQ)


class AnalysisJobQueue:
    def __init__(self, session_factory=SessionLocal) -> None:
        self._queue: asyncio.Queue[uuid.UUID] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._session_factory = session_factory

    def enqueue(self, job_id: uuid.UUID) -> None:
        self._queue.put_nowait(job_id)

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def replay_unfinished_jobs(self) -> int:
        with self._session_factory() as session:
            job_ids = session.execute(
                select(AnalysisJob.job_id).where(AnalysisJob.status.in_(("queued", "running")))
            ).scalars().all()
        for job_id in job_ids:
            self.enqueue(job_id)
        return len(job_ids)

    def _process_in_new_session(self, job_id: uuid.UUID) -> JobAttemptOutcome:
        # Session creation AND use must happen in the same thread -- a
        # SQLAlchemy/DBAPI session handed across threads (e.g. created
        # here in the event loop thread, then operated on from a
        # to_thread() worker thread) is not safe and previously caused
        # committed status changes to appear to vanish under StaticPool.
        with self._session_factory() as session:
            return process_job_once(session, job_id)

    async def _run(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                outcome = await asyncio.to_thread(self._process_in_new_session, job_id)
                if outcome.status == STATUS_RETRY_SCHEDULED:
                    asyncio.create_task(self._delayed_requeue(job_id, outcome.retry_delay_seconds or 0.0))
            except Exception:
                log.exception("Unhandled error processing job %s", job_id)
            finally:
                self._queue.task_done()

    async def _delayed_requeue(self, job_id: uuid.UUID, delay_seconds: float) -> None:
        await asyncio.sleep(delay_seconds)
        self.enqueue(job_id)


job_queue = AnalysisJobQueue()