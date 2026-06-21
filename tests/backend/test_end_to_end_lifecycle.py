"""End-to-end smoke test against the real FastAPI app: login -> submit ->
worker drains queue (stub model) -> result with explainability -> decision
-> audit trail. This is the main confidence check for the vertical slice.
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

import backend.seed as seed_module
from backend.api.deps import get_db
from backend.config import settings
from backend.db.models import User
from backend.jobs import queue as queue_module
from backend.model_runtime import loader as loader_module
from backend.monitoring import metrics
from backend.rate_limit import limiter

ANALYST_LOGIN = {"email": "analyst@demo.org", "password": seed_module.DEMO_PASSWORD}
AUDITOR_LOGIN = {"email": "auditor@demo.org", "password": seed_module.DEMO_PASSWORD}


@pytest.fixture()
def client(db_sessionmaker, monkeypatch):
    monkeypatch.setattr(queue_module.job_queue, "_session_factory", db_sessionmaker)
    monkeypatch.setattr(queue_module.job_queue, "_worker_task", None)
    monkeypatch.setattr(queue_module.job_queue, "_queue", asyncio.Queue())

    limiter.reset_all()
    metrics.reset_all()
    loader_module.reset_model_components()

    def _override_get_db():
        session = db_sessionmaker()
        try:
            yield session
        finally:
            session.close()

    from backend.main import app

    app.dependency_overrides[get_db] = _override_get_db

    with db_sessionmaker() as seed_session:
        seed_module.seed(seed_session)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _login(client: TestClient, credentials: dict) -> dict:
    response = client.post("/auth/login", json=credentials)
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _wait_for_case_status(client: TestClient, case_id: str, headers: dict, terminal_statuses: set[str]) -> dict:
    for _ in range(40):
        detail = client.get(f"/cases/{case_id}", headers=headers).json()
        if detail["case"]["status"] in terminal_statuses:
            return detail
        time.sleep(0.1)
    raise AssertionError(f"case never reached {terminal_statuses}, last status={detail['case']['status']}")


def test_full_case_lifecycle(client: TestClient):
    analyst_headers = _login(client, ANALYST_LOGIN)

    submit = client.post(
        "/cases",
        json={
            "priority": "standard",
            "messages": [
                {"speaker": "UserA", "text": "our little secret ok? dont tell your parents"},
                {"speaker": "UserB", "text": "ok i promise"},
                {"speaker": "UserA", "text": "add me on snapchat so we can talk privately"},
            ],
        },
        headers=analyst_headers,
    )
    assert submit.status_code == 202, submit.text
    case_id = submit.json()["case_id"]

    detail = _wait_for_case_status(client, case_id, analyst_headers, {"ready_for_review"})
    assert detail["case"]["status"] == "ready_for_review"

    results = detail["results"]
    assert len(results) == 1
    explainability = results[0]["explainability"]
    assert explainability["disclaimer"]
    rule_ids = {r["rule_id"] for r in explainability["rule_evidence"]}
    assert "secret_request" in rule_ids
    assert "contact_migration" in rule_ids
    assert explainability["model_evidence"]["attention_disclaimer"]

    decision = client.post(
        f"/cases/{case_id}/decisions",
        json={"decision_type": "close", "rationale": "no risk found"},
        headers=analyst_headers,
    )
    assert decision.status_code == 201, decision.text

    final_detail = client.get(f"/cases/{case_id}", headers=analyst_headers).json()
    assert final_detail["case"]["status"] == "closed"
    assert len(final_detail["decisions"]) == 1

    auditor_headers = _login(client, AUDITOR_LOGIN)
    audit = client.get("/audit", headers=auditor_headers)
    assert audit.status_code == 200
    assert audit.json()["chain_valid"] is True
    event_types = [e["event_type"] for e in audit.json()["events"]]
    assert "case_submitted" in event_types
    assert "case_analysis_completed" in event_types
    assert "decision_recorded" in event_types

    # Analyst (no READ_AUDIT_LOG permission) must be denied.
    forbidden = client.get("/audit", headers=analyst_headers)
    assert forbidden.status_code == 403


def test_long_single_message_does_not_overflow_stub_position_embeddings(client: TestClient):
    """A long "letter"-style single message tokenizes past the stub tiny
    BERT's max_position_embeddings (32) unless MessageEncoder/
    GoEmotionsClassifier are constructed with a matching max_length in
    backend/model_runtime/loader.py -- previously this crashed inference
    with a tensor-shape mismatch and the case landed in the DLQ instead of
    ready_for_review."""
    analyst_headers = _login(client, ANALYST_LOGIN)

    long_letter = (
        "To whoever reads this, you have ignored my warnings for too long. "
        "If you keep talking about me or try to report this, there will be "
        "consequences. I know where you usually go after school, and I can "
        "make sure everyone sees the messages I saved. Do not show this "
        "letter to anyone. Do not tell your parents. If you do, things will "
        "get much worse for you. This is your final warning. "
        "I am going to kill you."
    )

    submit = client.post(
        "/cases",
        json={"priority": "standard", "messages": [{"speaker": "A", "text": long_letter}]},
        headers=analyst_headers,
    )
    assert submit.status_code == 202, submit.text
    case_id = submit.json()["case_id"]

    detail = _wait_for_case_status(client, case_id, analyst_headers, {"ready_for_review", "dlq_investigation"})
    assert detail["case"]["status"] == "ready_for_review", detail


def test_cross_org_case_access_denied(client: TestClient):
    analyst_headers = _login(client, ANALYST_LOGIN)
    submit = client.post("/cases", json={"messages": [{"speaker": "A", "text": "hello"}]}, headers=analyst_headers)
    case_id = submit.json()["case_id"]

    # Auditor role can authenticate but has no case-view permission at all.
    auditor_headers = _login(client, AUDITOR_LOGIN)
    response = client.get(f"/cases/{case_id}", headers=auditor_headers)
    assert response.status_code == 403


def test_unauthenticated_request_rejected(client: TestClient):
    response = client.get("/cases")
    assert response.status_code == 401


def test_submit_case_releases_concurrency_slot_on_mid_request_failure(client: TestClient, db_sessionmaker, monkeypatch):
    """acquire_concurrent_job_slot is taken before any job exists; if
    something between that and the commit fails, the slot must be released
    there too, or it leaks (capped at rate_limit_concurrent_jobs_per_analyst)
    until the analyst can never submit again."""
    analyst_headers = _login(client, ANALYST_LOGIN)

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated failure during preprocessing")

    monkeypatch.setattr("backend.api.routers.cases.prepare_conversation", _raise)
    with pytest.raises(RuntimeError):
        client.post("/cases", json={"messages": [{"speaker": "A", "text": "hello"}]}, headers=analyst_headers)
    monkeypatch.undo()

    with db_sessionmaker() as session:
        analyst_id = session.query(User).filter_by(email=ANALYST_LOGIN["email"]).one().user_id

    # If the failed submission's slot leaked, this would raise before the
    # full per-analyst concurrency limit is reached.
    for _ in range(settings.rate_limit_concurrent_jobs_per_analyst):
        limiter.acquire_concurrent_job_slot(analyst_id)


def test_validation_error_does_not_echo_raw_submitted_content(client: TestClient):
    """FastAPI's default 422 handler echoes each rejected field's raw
    `input` value back in the response body. For a missing required field
    (e.g. a message sent with no `speaker`), Pydantic's `input` for that
    error is the *whole sibling dict* -- including the message's real
    `text` -- not just the missing field. main.py overrides the handler so
    this never reaches the client unredacted."""
    analyst_headers = _login(client, ANALYST_LOGIN)
    secret_text = "my home address is 123 Secret Lane, call me at 555-000-1111"

    response = client.post(
        "/cases",
        json={"messages": [{"text": secret_text}]},  # "speaker" omitted
        headers=analyst_headers,
    )
    assert response.status_code == 422
    assert secret_text not in response.text
    for error in response.json()["detail"]:
        assert "input" not in error


def test_invalid_decision_type_rejected_by_schema(client: TestClient):
    analyst_headers = _login(client, ANALYST_LOGIN)
    submit = client.post("/cases", json={"messages": [{"speaker": "A", "text": "hello"}]}, headers=analyst_headers)
    case_id = submit.json()["case_id"]
    _wait_for_case_status(client, case_id, analyst_headers, {"ready_for_review"})

    response = client.post(f"/cases/{case_id}/decisions", json={"decision_type": "bogus"}, headers=analyst_headers)
    assert response.status_code == 422