"""
Checkpoint Evaluation & Comparison Route Controller.

This module exposes endpoints to run synchronous evaluations and comparative tests on trained models.
Unlike long-running training jobs, these endpoints quickly run a fixed, seeded batch of game simulations using
greedy action-selection to assess policy performance or compare two models side-by-side.

Key Endpoints:
  - `POST /testing/evaluate`: Evaluates a single model checkpoint on a specific config for a fixed set of episodes.
  - `POST /testing/compare`: Runs two checkpoints side-by-side under identical random seeds to measure win/draw rates,
    average sprint completion, and financial return differences.

Connections:
  - Imports: Performance calculation handlers from `services.testing_service`.
"""

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
