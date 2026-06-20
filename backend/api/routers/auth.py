"""Auth endpoints (v6 Section 4.3 Session and Token Policy)."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.api.schemas import LoginRequest, RefreshRequest, TokenResponse
from backend.audit.service import SYSTEM_ACTOR_ID, record_audit_event
from backend.auth.security import (
    REFRESH_TOKEN_TYPE,
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from backend.db.models import User
from backend.db.queries import get_user_by_email
from backend.monitoring.metrics import increment

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = get_user_by_email(db, payload.email)

    if user is None or not verify_password(payload.password, user.hashed_password):
        # Section 4.3: "All authentication events written to the audit log" --
        # even failures, using a placeholder actor when the email itself
        # doesn't resolve to a real user.
        record_audit_event(
            db,
            actor_id=user.user_id if user else SYSTEM_ACTOR_ID,
            actor_role=user.role if user else "unknown",
            event_type="login_failed",
        )
        db.commit()
        increment("login_failed")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")

    access_token = create_access_token(user.user_id, user.organization_id, user.role)
    refresh_token = create_refresh_token(user.user_id, user.organization_id, user.role)

    record_audit_event(db, actor_id=user.user_id, actor_role=user.role, event_type="login_succeeded")
    db.commit()
    increment("login_succeeded")

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)) -> TokenResponse:
    try:
        decoded = decode_token(payload.refresh_token, expected_type=REFRESH_TOKEN_TYPE)
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid or expired refresh token: {exc}") from exc

    user = db.get(User, decoded.user_id)
    if user is None or user.organization_id != decoded.organization_id or user.role != decoded.role:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token no longer valid for this user")

    access_token = create_access_token(user.user_id, user.organization_id, user.role)
    refresh_token = create_refresh_token(user.user_id, user.organization_id, user.role)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)