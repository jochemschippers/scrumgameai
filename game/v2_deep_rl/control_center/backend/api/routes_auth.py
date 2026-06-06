"""
Authentication Route Handler.

This module provides routes and utility functions for user authentication, specifically JWT-based
authentication for the Control Center API. It validates user credentials via the backend database
and signs/decodes JSON Web Tokens (JWT) for secure HTTP sessions.

Key Components:
  - Token Signature: Signs user identifiers and roles using HMAC-SHA256.
  - Expiration Window: Tokens are set to expire after a default window (7 days).
  - Endpoint (`/auth/login`): Verifies password hashes and returns signed Bearer tokens.

Connections:
  - Imports: `authenticate_user` from `storage.jobs_db`.
  - Exported Utility: `decode_token` is called by FastAPI global middleware in `app.py`.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel

from storage.jobs_db import authenticate_user

router = APIRouter(prefix="/auth", tags=["auth"])

# Set JWT_SECRET_KEY in your environment — keep it secret, keep it safe.
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "scrum-game-dev-secret-change-in-prod")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24 * 7  # tokens last 7 days


class LoginRequest(BaseModel):
    """Payload representing a login request."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Payload representing signed JWT token response."""
    access_token: str
    token_type: str
    username: str
    role: str


def create_access_token(username: str, role: str) -> str:
    """Create a signed JWT access token for a given user and role."""
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(
        {"sub": username, "role": role, "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token, returning its payload dict or None."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    """Authenticate a user and generate a new JWT session token."""
    user = authenticate_user(body.username, body.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = create_access_token(user["username"], user["role"])
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=user["username"],
        role=user["role"],
    )
