"""
FastAPI Security Dependencies.

This module provides reusable route dependencies to handle access control, role authorization,
and user parsing by decapsulating the client's Bearer JWT payload.

Key Functions:
  - `get_current_user`: Checks for a valid JWT signature and retrieves user context.
  - `require_admin`: Restricts write operations (e.g. initiating training, deleting database entries) to admin users.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.routes_auth import decode_token

_bearer = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """Decode the JWT and return its payload, or raise 401."""
    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return payload


def require_admin(
    payload: dict = Depends(get_current_user),
) -> dict:
    """Allow only admin-role tokens through; raise 403 for guests."""
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required. Guests can view but not modify.",
        )
    return payload
