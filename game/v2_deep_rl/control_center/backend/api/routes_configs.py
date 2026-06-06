"""
Configuration Catalog Route Controller.

This module exposes endpoints to manage game rule configurations and training hyperparameter presets.
It allows listing, loading, saving, validating, and deleting both default/bundled configurations
and custom, user-defined configuration files.

Key Endpoints:
  - `GET /configs/game` / `GET /configs/training`: Lists all available configuration catalog items.
  - `POST /configs/game` / `POST /configs/training`: Persists config files (requires admin privileges).
  - `POST /configs/game/validate` / `POST /configs/training/validate`: Validates syntax and parses derived properties (like signatures).
  - `DELETE /configs/game/{id}` / `DELETE /configs/training/{id}`: Deletes custom configs (requires admin privileges).

Connections:
  - Imports: Catalog service handlers from `services.catalog_service`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import require_admin

from services.catalog_service import (
    delete_game_config_asset,
    delete_training_config_asset,
    get_game_config,
    get_training_config,
    list_game_configs,
    list_training_configs,
    save_game_config_asset,
    save_training_config_asset,
    validate_game_config_asset,
    validate_training_config_asset,
)

router = APIRouter(prefix="/configs", tags=["configs"])


@router.get("/game")
def get_game_configs():
    """List bundled and managed game config assets."""
    return {"items": list_game_configs()}


@router.get("/game/{config_id:path}")
def get_game_config_details(config_id: str):
    """Return one game config asset and its full canonical payload."""
    try:
        return get_game_config(config_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/game", dependencies=[Depends(require_admin)])
def post_game_config(payload: dict):
    """Create or update one managed game config asset. Requires admin role."""
    try:
        return save_game_config_asset(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/game/validate")
def post_game_config_validate(payload: dict):
    """Validate one game config draft and return derived metadata."""
    try:
        return validate_game_config_asset(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.delete("/game/{config_id:path}", dependencies=[Depends(require_admin)])
def delete_game_config(config_id: str):
    """Delete one managed custom game config asset. Requires admin role."""
    try:
        return delete_game_config_asset(config_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/training")
def get_training_configs():
    """List bundled and managed training config assets."""
    return {"items": list_training_configs()}


@router.get("/training/{config_id:path}")
def get_training_config_details(config_id: str):
    """Return one training config asset and its full canonical payload."""
    try:
        return get_training_config(config_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/training", dependencies=[Depends(require_admin)])
def post_training_config(payload: dict):
    """Create or update one managed training config asset. Requires admin role."""
    try:
        return save_training_config_asset(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/training/validate")
def post_training_config_validate(payload: dict):
    """Validate one training config draft and return derived metadata."""
    try:
        return validate_training_config_asset(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.delete("/training/{config_id:path}", dependencies=[Depends(require_admin)])
def delete_training_config(config_id: str):
    """Delete one managed custom training config asset. Requires admin role."""
    try:
        return delete_training_config_asset(config_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
