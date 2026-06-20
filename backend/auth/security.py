"""Password hashing and JWT issuance/verification (v6 Section 4.3)."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from backend.config import settings

ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


@dataclass(frozen=True)
class DecodedToken:
    user_id: uuid.UUID
    organization_id: uuid.UUID
    role: str
    token_type: str


def _encode(user_id: uuid.UUID, organization_id: uuid.UUID, role: str, token_type: str, expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "org": str(organization_id),
        "role": role,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: uuid.UUID, organization_id: uuid.UUID, role: str) -> str:
    return _encode(
        user_id, organization_id, role, ACCESS_TOKEN_TYPE, timedelta(minutes=settings.access_token_minutes)
    )


def create_refresh_token(user_id: uuid.UUID, organization_id: uuid.UUID, role: str) -> str:
    return _encode(
        user_id, organization_id, role, REFRESH_TOKEN_TYPE, timedelta(hours=settings.refresh_token_hours)
    )


class TokenError(Exception):
    pass


def decode_token(token: str, expected_type: str | None = None) -> DecodedToken:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc

    token_type = payload.get("type")
    if expected_type is not None and token_type != expected_type:
        raise TokenError(f"Expected a {expected_type} token, got {token_type!r}")

    return DecodedToken(
        user_id=uuid.UUID(payload["sub"]),
        organization_id=uuid.UUID(payload["org"]),
        role=payload["role"],
        token_type=token_type,
    )