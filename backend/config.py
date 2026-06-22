"""Environment-driven settings for the external system layer.

No pydantic-settings dependency added; this project's convention (per
src/risk_detection/data/pan12.py) favors plain dataclasses over extra deps.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from backend.paths import PROJECT_ROOT

# Mirrors scripts/train_grooming.py's own load_dotenv call. Without this,
# ANTHROPIC_API_KEY (and anything else only set in .env, not the real shell
# environment) is invisible to the backend process -- LLMSafetySignalExtractor/
# EmotionalDependencyExtractor construct `anthropic.Anthropic()` with no
# explicit api_key, which reads os.environ directly, so a case submitted in
# real mode failed with "Could not resolve authentication method" even
# though `.env` had the key (caught via a live real-mode test, not assumed).
# Must run before Settings() below, since its fields read os.environ at
# construction time.
load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Storage technology decision (v6 Section 3): PostgreSQL is the documented
    # system of record. This deployment substitutes SQLite (hackathon-permitted
    # per v6 Section 13 / companion DB report) with a Postgres-shaped schema.
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "RISK_PLATFORM_DATABASE_URL",
            f"sqlite:///{PROJECT_ROOT / 'backend' / 'risk_platform.db'}",
        )
    )

    # Auth / JWT (v6 Section 4.3 Session and Token Policy).
    jwt_secret: str = field(
        default_factory=lambda: os.environ.get(
            "RISK_PLATFORM_JWT_SECRET", "dev-only-insecure-secret-change-me"
        )
    )
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = field(default_factory=lambda: _env_int("RISK_PLATFORM_ACCESS_TOKEN_MINUTES", 20))
    refresh_token_hours: int = field(default_factory=lambda: _env_int("RISK_PLATFORM_REFRESH_TOKEN_HOURS", 24))

    # Model runtime mode: "stub" (fast, deterministic, offline; used by tests),
    # "local" (real trained_weights/ checkpoints, zero-cost LLM stand-ins --
    # no ANTHROPIC_API_KEY needed), or "real" (also calls the real,
    # billed Claude-backed LLM/dependency extractors).
    model_runtime_mode: str = field(
        default_factory=lambda: os.environ.get("RISK_PLATFORM_MODEL_MODE", "stub")
    )
    trained_weights_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "trained_weights")
    default_window_size: int = field(default_factory=lambda: _env_int("RISK_PLATFORM_WINDOW_SIZE", 12))

    # Job queue / retry policy (v6 Section 7.3, 8.1).
    max_job_attempts: int = field(default_factory=lambda: _env_int("RISK_PLATFORM_MAX_JOB_ATTEMPTS", 3))
    retry_base_delay_seconds: float = 0.5

    # Rate limiting (v6 Section 6.1 illustrative limits).
    rate_limit_job_submit_per_analyst_per_min: int = 5
    rate_limit_job_submit_per_org_per_min: int = 20
    rate_limit_concurrent_jobs_per_analyst: int = 3
    rate_limit_reads_per_analyst_per_min: int = 120
    rate_limit_bulk_export_per_org_per_hour: int = 2

    # Retention defaults (v6 Section 10 table); organization-configurable in
    # principle, fixed defaults here for the hackathon build.
    redacted_conversation_retention_days: int = field(
        default_factory=lambda: _env_int("RISK_PLATFORM_RETENTION_DAYS", 90)
    )

    # DLQ operational requirement (v6 Section 8.4): alert when depth exceeds threshold.
    dlq_alert_threshold: int = field(default_factory=lambda: _env_int("RISK_PLATFORM_DLQ_ALERT_THRESHOLD", 10))

    enable_mfa: bool = field(default_factory=lambda: _env_bool("RISK_PLATFORM_ENABLE_MFA", False))


settings = Settings()