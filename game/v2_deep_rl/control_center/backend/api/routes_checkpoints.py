from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from services.checkpoint_service import (
    get_checkpoint_compatibility,
    list_checkpoints,
    resolve_checkpoint_path,
)


router = APIRouter(tags=["checkpoints"])


@router.get("/checkpoints")
def get_checkpoints():
    """List managed and legacy checkpoints with metadata summaries."""
    return {"items": list_checkpoints()}


@router.get("/checkpoints/{checkpoint_id:path}/compatibility")
def get_checkpoint_compatibility_route(
    checkpoint_id: str,
    game_config_id: str = Query(..., description="Game config asset id or path."),
):
    """Return strict-resume and fine-tune compatibility for one checkpoint and config."""
    try:
        return get_checkpoint_compatibility(checkpoint_id, game_config_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/checkpoints/{checkpoint_id:path}/download")
def download_checkpoint_route(checkpoint_id: str):
    """Download a tracked checkpoint file directly."""
    try:
        checkpoint_path = resolve_checkpoint_path(checkpoint_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return FileResponse(path=checkpoint_path, filename=checkpoint_path.name, media_type="application/octet-stream")
