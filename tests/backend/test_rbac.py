"""
Tests require_role() directly against plain (unpersisted) User objects —
it's a pure function of `current_user.role`, so no DB or HTTP layer is
needed to verify its logic.
"""

import pytest
from fastapi import HTTPException

from app.api.deps import require_role
from app.models.enums import UserRole
from app.models.user import User


def _user(role: UserRole) -> User:
    return User(email="someone@example.com", role=role)


def test_allows_a_permitted_role():
    check = require_role(UserRole.ADMIN, UserRole.ENGINEER)
    user = _user(UserRole.ENGINEER)

    result = check(current_user=user)

    assert result is user


def test_rejects_a_role_not_in_the_allowed_list():
    check = require_role(UserRole.ADMIN, UserRole.ENGINEER)
    user = _user(UserRole.OPERATOR)

    with pytest.raises(HTTPException) as exc_info:
        check(current_user=user)

    assert exc_info.value.status_code == 403


def test_single_role_gate():
    check = require_role(UserRole.QA_INSPECTOR)
    assert check(current_user=_user(UserRole.QA_INSPECTOR)) is not None

    with pytest.raises(HTTPException):
        check(current_user=_user(UserRole.ADMIN))
