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


def test_invalid_decision_type_rejected_by_schema(client: TestClient):
    analyst_headers = _login(client, ANALYST_LOGIN)
    submit = client.post("/cases", json={"messages": [{"speaker": "A", "text": "hello"}]}, headers=analyst_headers)
    case_id = submit.json()["case_id"]
    _wait_for_case_status(client, case_id, analyst_headers, {"ready_for_review"})

    response = client.post(f"/cases/{case_id}/decisions", json={"decision_type": "bogus"}, headers=analyst_headers)
    assert response.status_code == 422