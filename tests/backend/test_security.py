import uuid

import pytest

from app.core.security import InvalidTokenError, create_access_token, decode_access_token


def test_token_round_trips_user_id_and_role():
    user_id = str(uuid.uuid4())
    token = create_access_token(user_id, "admin")

    payload = decode_access_token(token)

    assert payload["sub"] == user_id
    assert payload["role"] == "admin"


def test_garbage_token_raises_invalid_token_error():
    with pytest.raises(InvalidTokenError):
        decode_access_token("not.a.real.jwt")


def test_tampered_token_is_rejected():
    token = create_access_token(str(uuid.uuid4()), "admin")
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(InvalidTokenError):
        decode_access_token(tampered)
