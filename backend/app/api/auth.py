"""
Minimal auth API (Project.md #45, Phase 8) — see app/core/security.py's
docstring for exactly how minimal: email-only, no password. `register` only
exists because there is no other way to create the first User row (no admin
bootstrap was in scope); a real deployment should gate or remove it.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import create_access_token
from app.models.enums import UserRole
from app.models.user import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str
    role: UserRole


class LoginRequest(BaseModel):
    email: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID
    role: UserRole


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    existing = db.query(User).filter(User.email == payload.email).one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="A user with this email already exists")

    user = User(email=payload.email, role=payload.role)
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(str(user.id), user.role.value)
    return TokenResponse(access_token=token, user_id=user.id, role=user.role)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email).one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="No user with this email — register first")

    token = create_access_token(str(user.id), user.role.value)
    return TokenResponse(access_token=token, user_id=user.id, role=user.role)
