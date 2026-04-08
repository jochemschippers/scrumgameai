from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from services.checkpoint_service import get_checkpoint_compatibility, list_checkpoints


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
