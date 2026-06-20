"""FastAPI auth dependencies implementing v6 Section 4.1's check order.

Steps 1-2 (authenticated user, role permission check) are enforced here.
Steps 3-4 (organization ownership, case-assignment/scope check) are
necessarily per-resource and are applied in the routers using
`current_user.organization_id` to scope queries and
`backend.auth.roles.can_access_case` for case-level scope.
"""

import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.auth.roles import has_permission
from backend.auth.security import ACCESS_TOKEN_TYPE, TokenError, decode_token
from backend.db.models import User

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    user_id: uuid.UUID
    organization_id: uuid.UUID
    role: str


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")

    try:
        decoded = decode_token(credentials.credentials, expected_type=ACCESS_TOKEN_TYPE)
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid or expired token: {exc}") from exc

    user = db.get(User, decoded.user_id)
    if user is None or user.organization_id != decoded.organization_id or user.role != decoded.role:
        # Role/org changed since the token was issued (Section 4.3 revocation
        # on role change) -- force re-auth rather than trusting stale claims.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token no longer valid for this user")

    return CurrentUser(user_id=user.user_id, organization_id=user.organization_id, role=user.role)


def require_permission(permission: str):
    def _check(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not has_permission(current_user.role, permission):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, f"Role {current_user.role!r} lacks permission {permission!r}"
            )
        return current_user

    return _check