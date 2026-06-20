Hybrid Explainable Multi-Task Conversation Risk Detection System

Privacy-preserving AI decision support for detecting early conversational patterns associated with cyberbullying, online grooming, coercion, sextortion, and digital exploitation.

USAII Global AI Hackathon 2026
Direction A: Safe Passage — AI Against Digital Exploitation

Overview

Digital exploitation rarely appears as one obvious message. Risk may emerge gradually through repeated secrecy requests, isolation, emotional dependency, rapid trust-building, sexual escalation, threats, or coercive language.

This project proposes a hybrid, explainable, and privacy-preserving conversation risk detection system that combines structured LLM signal extraction, hierarchical multi-task learning, deterministic safety rules, independent emotion analysis, temporal risk tracking, score fusion with calibration planned, uncertainty estimation, and human review.

The system supports trained analysts and safeguarding organizations. It does not automatically accuse, diagnose, report, or take enforcement action against any person.

## Running the External System Layer

The AI pipeline (`src/risk_detection/`) sits behind a FastAPI backend (`backend/`) that implements the external system layer: analyst authentication/RBAC, case submission and review workflow, an asynchronous job queue, an append-only audit log, and an Explainability Service. From the repo root, with dependencies installed (`pip install -r requirements.txt`):

```bash
# 1. Start the backend (creates backend/risk_platform.db on first run)
uvicorn backend.main:app --reload

# 2. Seed a demo organization with one user per role (prints passwords)
python -m backend.seed

# 3. Start the analyst dashboard
streamlit run frontend/app.py
```

Set `RISK_PLATFORM_API_URL` for the frontend if the backend isn't at `http://localhost:8000`.

### Model runtime modes

- **`stub`** (default): tiny randomly-initialized encoders and zero-signal LLM/dependency extractors. No network access, fast, fully deterministic — this is also what the automated test suite uses.
- **`real`**: loads `trained_weights/message_encoder.pt` + `cyberbullying_head.pt` for the cyberbullying signal and the Variant B `grooming_*_B.pt` checkpoint trio (its own dedicated message/conversation encoder pair, per `IntegratedInferencePipeline`'s `grooming_message_encoder`/`grooming_conversation_encoder` params) for the grooming signal, plus the real Claude-backed safety/dependency extractors (requires `ANTHROPIC_API_KEY`). Variant C (LLM-augmented) was never trained, since no `ANTHROPIC_API_KEY` was available at training time — see `backend/model_runtime/loader.py` for the full explanation. Enable with:

  ```bash
  RISK_PLATFORM_MODEL_MODE=real uvicorn backend.main:app
  ```

### Architecture and what's substituted

`backend/` implements the design in `Final_External_Architecture_Conclusion_v6.docx` (Section 13's "Hackathon MVP" scope, expanded to also cover RBAC, the DLQ, audit hash-chaining, rate limiting, and the Explainability Service) on a lightweight stack instead of the doc's full production stack:

| Production target (per the doc) | This deployment |
|---|---|
| PostgreSQL + Row-Level Security | SQLite with a Postgres-shaped schema; organization isolation enforced at the query layer (`backend/db/queries.py`) instead of DB-level RLS |
| Redis (rate limits, job-status/version cache) | In-memory counters behind the same call shapes (`backend/rate_limit/`, `backend/monitoring/metrics.py`) |
| Durable broker (Celery/RQ + Redis) | In-process `asyncio.Queue`; the `analysis_jobs` table is the real source of truth and is replayed on restart (`backend/jobs/queue.py`) |
| Prometheus/Grafana, PagerDuty/Slack | In-memory counters at `GET /admin/metrics`; alerts are structured log lines |
| ML/NER-based PII redaction | Regex-based redaction (emails, phones, URLs, @handles) — a documented data limitation, not a hidden gap (`backend/preprocessing/pii_redaction.py`) |
| MFA | Schema hook only (`User.mfa_enabled`); not wired to a real provider |
| WAF + load balancer | Single FastAPI/uvicorn instance |

RBAC permission checks, the case status state machine, audit log hash-chaining, DLQ failure classification and redrive, the Explainability Service's five output types with mandatory disclaimers, and rate limiting are all implemented as specified — only the underlying infrastructure is lighter weight. Run the backend test suite with `pytest tests/backend/ -v`.