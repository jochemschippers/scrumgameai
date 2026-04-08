from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.play_service import advance_session, create_session, get_session, list_sessions


router = APIRouter(prefix="/play", tags=["play"])


@router.get("/session")
def get_play_sessions():
    """List active in-memory play sessions."""
    return {"items": list_sessions()}


@router.post("/session")
def post_play_session(payload: dict):
    """Create one new parallel-seat play session."""
    try:
      return create_session(payload)
    except ValueError as error:
      raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/session/{session_id}")
def get_play_session(session_id: str):
    """Return one play session."""
    try:
      return get_session(session_id)
    except ValueError as error:
      raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/session/{session_id}/action")
def post_play_action(session_id: str, payload: dict | None = None):
    """Advance a play session by one round, optionally with a human action."""
    try:
      return advance_session(session_id, payload or {})
    except ValueError as error:
      raise HTTPException(status_code=404, detail=str(error)) from error
