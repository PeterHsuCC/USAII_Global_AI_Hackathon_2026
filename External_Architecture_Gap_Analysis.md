# External Architecture Gap Analysis

Comparison of `backend/` (the implementation of `Final_External_Architecture_Conclusion_v6.docx`) against that document, section by section. Verified by reading the actual code (file/line references below), not by assumption. Generated 2026-06-20.

Items already documented as deliberate technical substitutions at delivery time (MFA, Redis, PostgreSQL RLS, Celery/Redis durable queue, Prometheus/Grafana, PagerDuty/Slack) are **not** repeated here — see the "Architecture and what's substituted" table in `README.md`.

## Functional gaps (higher impact)

1. **Urgent cases are not prioritized.** (Doc Section 6: "prioritize urgent human-review work") `Case.priority` (standard/urgent) is captured at submission and returned in API responses, but `AnalysisJobQueue` (`backend/jobs/queue.py`) uses a plain FIFO `asyncio.Queue`. `priority` is never read when deciding dequeue order — urgent cases are processed in submission order, same as standard ones.

2. **No Organization Admin management endpoints.** (Doc Section 4.2: "Manage members and roles") `backend/api/routers/admin.py` only has DLQ/metrics/retention endpoints, gated to `system_admin`. The only way to create a user is `backend/seed.py`, a one-time script — there is no API for an org admin to add members, change roles, or deactivate an account.

3. **No logout endpoint or session revocation.** (Doc Section 4.3: "Session revocation on logout, role change, or account suspension") `backend/api/routers/auth.py` has only `/login` and `/refresh`. Logout is client-side only (frontend clears local tokens); the JWT remains valid until expiry. No audit event is recorded for logout. `User` has no `is_active`/`is_suspended` column, so there is no suspension mechanism either.

4. **File upload / object storage is not wired up.** (Doc Section 3, 15.9, 15.10) The `CaseFile` and `ModelArtifact` tables exist in `backend/db/models.py` but nothing writes to them — there is no upload endpoint. `source_type` allows `'file_upload'` per the CHECK constraint, but `submit_case` (`backend/api/routers/cases.py`) always hardcodes `source_type="api"`.

5. **No bulk export endpoint.** (Doc Section 6.1: "Bulk export | 2/hour | requires re-authentication") `check_bulk_export` exists in `backend/rate_limit/limiter.py` but is never called anywhere — there is no export endpoint for it to guard.

6. **No re-authentication for sensitive operations.** (Doc Section 4.3: "Re-authentication required for sensitive operations (referral submission, bulk export)") Submitting a `refer` decision only requires a valid bearer token, same as any other request.

7. **`VALIDATION_FAILED` is unreachable.** (Doc Section 15.1 state machine) `submit_case` always proceeds `SUBMITTED -> VALIDATING -> PRIVACY_PROCESSING -> QUEUED` with no domain-level validation that could fail. The state exists in `backend/db/state_machine.py` but no code path ever transitions into it.

## Operational completeness gaps (medium impact)

8. **Most Section 12.1 alert conditions are missing.** Only DLQ-depth alerting is implemented (`is_over_alert_threshold` in `backend/jobs/dlq.py`). Urgent-case queue wait time, API 5xx error rate, model score distribution drift, and audit-log write failure have no corresponding alerting logic.

9. **No result caching by input hash.** (Doc Section 5.1) Identical conversations are re-analyzed from scratch every time; no cache keyed by redacted-input hash + model/rule/preprocessing versions exists.

10. **No security headers middleware.** Only CORS is configured in `backend/main.py` — no X-Content-Type-Options, X-Frame-Options, or Content-Security-Policy.

11. **No API versioning scheme** (e.g. a `/v1` prefix).

12. **No request body size limit.**

13. **No idempotency-key handling.** A duplicate `POST /cases` or `POST /cases/{id}/decisions` is processed as two independent requests.

14. **No circuit breaker** around any external call (e.g. the LLM safety extractor in `real` mode).

15. **Structured logs bind only `request_id`, not `case_id` or `trace_id`.** (Doc Section 12) The audit log does record `case_id` separately, but the general JSON logs in `backend/monitoring/logging_setup.py` don't, so app logs can't be filtered end-to-end by case from log content alone.

16. **DLQ has no "max wait before escalation" or assignee field**, and no distinction between error categories that could auto-redrive vs. require manual review — every redrive today is a manual admin action.

## Suggested priority if addressed

If picking a subset to fix next: **#1 (priority queue)**, **#2 (org admin endpoints)**, **#3 (logout/suspension)**, and **#7 (a real validation-failure path)** affect behavior the document describes explicitly as core to the workflow. The rest are operational hardening that matter more as the system scales past a hackathon demo.

## Bugs found and fixed via code audit (2026-06-20, not gaps vs. the doc)

The items above are *missing* behavior relative to the doc. The items below are different in kind: the doc's design was correct, but the implementation didn't fully realize it. Found by reading the actual code (not assumed), fixed the same day, each with a regression test. Listed here for traceability since they were found while validating the doc's design intent, not as additions to the comparison above.

1. **Audit log hash-chain fork under concurrent writes.** `backend/audit/service.py`. The in-process lock covered the read+flush but released before `commit()`; since every caller uses its own DB session/connection, a second thread could acquire the lock before the first transaction was durable, read the same "last row", and chain a new entry off the same `previous_hash`/`sequence_number` (the docstring's own claimed guarantee didn't hold). Fixed by committing inside the locked section. `tests/backend/test_audit_hash_chain.py`.

2. **Two contradictory decisions could both be recorded on one case.** `backend/api/routers/review.py` / `backend/db/queries.py`. `submit_decision` read `case.status` with no row lock, so two concurrent decisions on the same case could both pass the transition check before either committed. Fixed with `get_case_for_org(..., for_update=True)` (Doc Section 15.4) — correct and effective on the documented Postgres deployment; the current SQLite substitution doesn't enforce row-level locking, so this is best-effort there, not a complete guarantee (documented in code).

3. **DLQ entries could be redriven or closed more than once.** `backend/jobs/dlq.py`. The doc's own `resolution_status` schema (Section 8.3: `investigating | redriven | closed`) implies a one-way state machine, but `redrive_dlq_entry`/`close_dlq_entry_as_invalid` never checked the current value before overwriting it — a double-click could re-enqueue the same job twice or reset `attempt_count` after closure. Fixed with a `DLQEntryAlreadyResolvedError` guard (admin endpoints now return 409). `tests/backend/test_job_queue_retry_dlq.py`.

4. **A worker crash mid-job silently burned a retry attempt with no trace.** `backend/jobs/queue.py`. `attempt_count` is incremented and committed *before* `run_analysis()` runs; if the process is killed during analysis (not a caught exception), `replay_unfinished_jobs()` just re-enqueued the job with no audit/DLQ record of the lost attempt. Fixed with `_recover_running_job()`, which routes a crash-recovered job through the same classify-and-record path as a caught failure (`error_category="unknown"`), sending it to DLQ if the crash used the last retry. `tests/backend/test_job_queue_retry_dlq.py`.

5. **`/admin/metrics` and `/admin/dlq` leaked another organization's DLQ depth/alert status.** `backend/jobs/dlq.py` / `backend/api/routers/admin.py`. `open_dlq_depth`/`is_over_alert_threshold` queried `DLQErrorMetadata` with no join to `Case` and no `organization_id` filter at all — any `system_admin` could see every tenant's open-failure count, contrasted with `list_dlq_entries`'s own entry list two lines away, which *was* correctly org-scoped. The platform-ops-wide alert check inside `process_job_once` (Section 12.1: "Platform operations team") is intentionally global and was left unscoped; only the two per-org admin endpoints needed the fix. Fixed by adding an optional `organization_id` filter. `tests/backend/test_org_isolation.py`.

6. **A failed case submission could permanently leak a concurrency slot.** `backend/api/routers/cases.py`. `acquire_concurrent_job_slot` is taken before any `AnalysisJob` row exists; everything after it (preprocessing, DB writes, the final commit) had no exception handling, so any failure there left the slot held forever (it's normally released by `process_job_once` once a job reaches a terminal state — but no job was ever queued). Bounded by `rate_limit_concurrent_jobs_per_analyst` (default 3), so three such failures locked an analyst out of submitting anything until process restart. Fixed by releasing the slot in an `except` block around the submission body. `tests/backend/test_end_to_end_lifecycle.py`.

7. **PII phone-number redaction missed bare digit runs with no punctuation.** `backend/preprocessing/pii_redaction.py`. Every separator in `_PHONE_RE` was mandatory, so a number sent with zero punctuation (`"text me 5551234567"`, or with a country code, `"15551234567"`) matched none of the four redaction patterns and was written to `CaseMessage.redacted_content` completely unredacted — exactly the format a contact-migration attempt is likely to use. Fixed by making every separator optional. `tests/backend/test_pii_redaction.py`.

8. **A validation error on `POST /cases` could echo back unredacted conversation text.** `backend/main.py`. FastAPI's default 422 handler echoes each rejected field's raw `input` value; for a *missing required field* (e.g. a message sent with no `speaker`), Pydantic's `input` for that error is the whole sibling dict — including that message's real, unredacted `text` — not just the missing field. Fixed with a `RequestValidationError` handler that strips `input` from every error entry before the response leaves the server. `tests/backend/test_end_to_end_lifecycle.py`.

9. **Retention deletion left redacted conversation text behind in `Result.evidence_json`.** `backend/retention/deletion.py`. The doc's deletion flow (Section 10.1) calls for "derived temporary artifacts" to be removed alongside conversation content, but the sweep only deleted `CaseMessage` rows — `Result.evidence_json` (which embeds `redacted_evidence_span`, an actual redacted message snippet, per Section 9's rule-evidence schema) was never touched, so a case's content remained queryable via its `Result` row even after a `retention_deletion_completed` audit event said otherwise. Fixed by clearing `evidence_json` (not the rest of the row — `risk_level`/`confidence`/etc. are metadata, not content, kept for the same accountability reasons `AnalystDecision` rows are). `tests/backend/test_retention_deletion.py` (previously had zero test coverage at all).

Also reviewed and confirmed correct (no bug): JWT algorithm pinning, refresh/access token type separation, bcrypt password hashing, every by-ID case/DLQ fetch's organization scoping outside the two leaks above, and `HistoricalStateUpdater`/`EarlyWarningTracker` cross-conversation isolation (`backend/model_runtime/job_runner.py` builds both fresh per job, per its own docstring's correctness note).