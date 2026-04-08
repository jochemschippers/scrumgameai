from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.testing_service import compare_checkpoints, evaluate_checkpoint


router = APIRouter(prefix="/testing", tags=["testing"])


@router.post("/evaluate")
def post_evaluate_checkpoint(payload: dict):
    """Run one greedy seeded evaluation batch for a checkpoint."""
    try:
        return evaluate_checkpoint(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/compare")
def post_compare_checkpoints(payload: dict):
    """Run one side-by-side greedy comparison for two checkpoints."""
    try:
        return compare_checkpoints(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
