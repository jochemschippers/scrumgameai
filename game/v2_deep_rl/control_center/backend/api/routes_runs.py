"""
Training Run Catalog Route Controller.

This module exposes endpoints to view and analyze historical and active DQN Training Runs.
A training run is a structured folder generated on disk containing training parameters, CSV metrics,
log output, and saved checkpoints. This controller provides summaries, chart progress, and computes
performance ratings.

Key Endpoints:
  - `GET /runs`: Lists all completed or active training run directories on disk.
  - `GET /runs/{run_id}`: Fetches configurations, metrics, and associated checkpoints for a single run.
  - `GET /runs/{run_id}/progress`: Extracts historical CSV lines to render learning curves in the UI.
  - `GET /runs/{run_id}/rating`: Analyzes evaluation logs to assign a quality score and letter grade (A-F).

Connections:
  - Imports: Catalog operations from `services.catalog_service` and autopilot analyzer from `services.training_autopilot`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.catalog_service import list_runs, get_run, get_run_progress
from services.training_autopilot import compute_run_rating

router = APIRouter(tags=["runs"])


@router.get("/runs")
def get_runs():
    """List timestamped training runs and summary metadata."""
    return {"items": list_runs()}


@router.get("/runs/{run_id}")
def get_run_details(run_id: str):
    """Return one run with metadata, metrics, configs, and checkpoint list."""
    run_payload = get_run(run_id)
    if run_payload is None:
        raise HTTPException(status_code=404, detail=f"Run `{run_id}` was not found.")
    return run_payload


@router.get("/runs/{run_id}/progress")
def get_run_progress_route(run_id: str):
    """Return persisted training progress and chart data for one run."""
    payload = get_run_progress(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Run `{run_id}` was not found.")
    return payload


@router.get("/runs/{run_id}/rating")
def get_run_rating(run_id: str):
    """Compute a 0–100 quality score and letter grade for a run from its evaluation history."""
    try:
        return compute_run_rating(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
