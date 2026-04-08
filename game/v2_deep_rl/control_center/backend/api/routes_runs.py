from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.catalog_service import list_runs, get_run

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
