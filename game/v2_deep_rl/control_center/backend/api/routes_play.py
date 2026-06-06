"""
Interactive Game Play Route Controller.

This module exposes endpoints to manage in-memory interactive play sessions.
It allows the frontend to run live simulations of the Scrum Game. Users can step through turns,
make choices (e.g. continue or switch products), roll dice, draw incidents, and watch automated agents
(model-based policy, random, or heuristic controllers) execute their steps.

Key Endpoints:
  - `POST /play/session`: Starts a new play session with specific seats (Human, Model, Heuristic, or Random AI).
  - `GET /play/session/{session_id}`: Returns the current state, logs, and valid actions for a session.
  - `POST /play/session/{session_id}/action`: Commits the human's action and advances the game board simulation.

Connections:
  - Imports: Session management routines from `services.play_service`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.play_service import advance_session, create_session, get_session, list_sessions


router = APIRouter(prefix="/play", tags=["play"])


def _status_for_error(message: str) -> int:
    """Return appropriate HTTP status code based on error message text."""
    return 404 if "was not found" in message else 400



@router.get("/session")
def get_play_sessions():
    """List active in-memory play sessions."""
    return {"items": list_sessions()}


@router.post("/session")
def post_play_session(payload: dict):
    """Create one new play session."""
    try:
      return create_session(payload)
    except ValueError as error:
      raise HTTPException(status_code=_status_for_error(str(error)), detail=str(error)) from error


@router.get("/session/{session_id}")
def get_play_session(session_id: str):
    """Return one play session."""
    try:
      return get_session(session_id)
    except ValueError as error:
      raise HTTPException(status_code=_status_for_error(str(error)), detail=str(error)) from error


@router.post("/session/{session_id}/action")
def post_play_action(session_id: str, payload: dict | None = None):
    """Advance a play session by one round, optionally with a human action."""
    try:
      return advance_session(session_id, payload or {})
    except ValueError as error:
      raise HTTPException(status_code=_status_for_error(str(error)), detail=str(error)) from error
