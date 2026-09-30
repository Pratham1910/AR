"""
Auth dependencies (Project.md #45): `get_current_user` resolves the bearer
token to a real `User` row (never trusting a role claimed in the token alone
without checking the DB still has that user), and `require_role(...)` is a
reusable per-endpoint permission gate.
"""

import uuid

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import InvalidTokenError, decode_access_token
from app.models.enums import UserRole
from app.models.user import User

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        payload = decode_access_token(credentials.credentials)
    except InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {exc}") from exc

    user = db.get(User, uuid.UUID(payload["sub"]))
    if user is None:
        raise HTTPException(status_code=401, detail="Token refers to a user that no longer exists")
    return user


def require_role(*allowed_roles: UserRole):
    """
    Usage: `Depends(require_role(UserRole.ADMIN, UserRole.ENGINEER))`. Raises
    403 (not 401 — the caller IS authenticated, just not permitted) if the
    current user's role isn't one of `allowed_roles`.
    """

    def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role {current_user.role.value!r} is not permitted; requires one of "
                f"{[r.value for r in allowed_roles]}",
            )
        return current_user

    return _check
