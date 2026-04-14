from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.training_autopilot import (
    analyze_run,
    clear_stop_request,
    get_autopilot_history,
    get_settings,
    is_stop_requested,
    probe_ai_advisor,
    request_stop_after_cycle,
    run_autopilot,
    save_settings,
)

router = APIRouter(prefix="/autopilot", tags=["autopilot"])


class AutopilotRunRequest(BaseModel):
    dry_run: bool = False


class AutopilotSettingsPayload(BaseModel):
    logic_enabled: bool | None = None
    ai_enabled: bool | None = None


@router.get("/analyze/{run_id}")
def analyze_run_endpoint(run_id: str):
    """
    Dry-run analysis: classify the training state of a completed run and
    return the recommended action without writing to disk or enqueuing a job.
    """
    try:
        return analyze_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/run/{run_id}")
def run_autopilot_endpoint(run_id: str, body: AutopilotRunRequest = AutopilotRunRequest()):
    """
    Run the autopilot on a completed run: classify state, write decision record,
    and enqueue the next training job (unless dry_run=true or action is 'stop').
    """
    try:
        return run_autopilot(run_id, dry_run=body.dry_run)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/history/{run_id}")
def autopilot_history_endpoint(run_id: str):
    """Return all recorded autopilot decisions for a run, oldest first."""
    return {"items": get_autopilot_history(run_id)}


@router.post("/test-ai")
def test_ai_endpoint():
    """
    Call the AI advisor with dummy plateau metrics and return its raw response.
    Use this to verify the NVIDIA API key and model are reachable before relying on it.
    """
    return probe_ai_advisor()


@router.get("/status")
def autopilot_status_endpoint():
    """Return whether a stop-after-cycle request is currently pending."""
    return {"stop_requested": is_stop_requested()}


@router.get("/settings")
def get_autopilot_settings_endpoint():
    """Return current logic_enabled and ai_enabled toggles."""
    return get_settings()


@router.post("/settings")
def update_autopilot_settings_endpoint(body: AutopilotSettingsPayload):
    """Update logic_enabled and/or ai_enabled toggles."""
    return save_settings(body.model_dump(exclude_none=True))


@router.post("/stop-after-cycle")
def request_stop_endpoint():
    """
    Request the autopilot to stop after the current training block finishes.
    The current job keeps running; the next autopilot cycle will not enqueue a new job.
    """
    request_stop_after_cycle()
    return {"stop_requested": True}


@router.delete("/stop-after-cycle")
def clear_stop_endpoint():
    """Clear a pending stop-after-cycle request so the autopilot resumes."""
    clear_stop_request()
    return {"stop_requested": False}
