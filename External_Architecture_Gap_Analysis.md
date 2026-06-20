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