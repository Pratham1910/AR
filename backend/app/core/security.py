"""
Minimal JWT auth (Project.md #45, Phase 8's RBAC requirement) — deliberately
small: email-only login, no password/SSO (§72's "choose the smallest
architecture that satisfies the requirement," and neither was ever
requested). This exists so `require_role()` (app/api/deps.py) has something
real to check, not to be a production identity system — replacing the login
mechanism later doesn't change anything downstream, since everything else
only depends on "there is a current User with a role."
"""

from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import get_settings

_settings = get_settings()


def create_access_token(user_id: str, role: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=_settings.jwt_expires_minutes)
    payload = {"sub": user_id, "role": role, "exp": expires_at}
    return jwt.encode(payload, _settings.jwt_secret_key, algorithm=_settings.jwt_algorithm)


class InvalidTokenError(ValueError):
    pass


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, _settings.jwt_secret_key, algorithms=[_settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc
