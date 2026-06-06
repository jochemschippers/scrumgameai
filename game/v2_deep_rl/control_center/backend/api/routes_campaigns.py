"""
Campaign Management Route Controller.

This module exposes endpoints to manage Robustness Evaluation Campaigns.
A campaign evaluates a trained DQN model across a randomized batch of game rule variations
(e.g., varying starting money, scrum targets, penalty bounds) to identify the model's
robustness threshold under drift or scenario modifications.

Key Endpoints:
  - `POST /campaigns`: Starts a new campaign for a given run ID with a targeted number of rule variations.
  - `GET /campaigns`: Lists metadata for all registered campaigns.
  - `POST /campaigns/{campaign_id}/stop`: Gracefully terminates an active evaluation campaign.
  - `POST /campaigns/{campaign_id}/escalate`: Escalates an existing campaign (e.g. running extra evaluation episodes for fine-grained validation).

Connections:
  - Imports: Campaign coordinator functions from `services.campaign_service`.
  - Guards: `require_admin` restricts mutation endpoints to authorized users.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import require_admin

from services.campaign_service import (
    create_campaign,
    escalate_campaign,
    get_campaign,
    list_campaigns,
    stop_campaign,
)


router = APIRouter(prefix="/campaigns", tags=["campaigns"])


class CreateCampaignRequest(BaseModel):
    """Payload representing a campaign creation request."""
    run_id: str
    max_variations: int = Field(default=5, ge=1, le=20)


@router.post("", dependencies=[Depends(require_admin)])
def post_create_campaign(body: CreateCampaignRequest) -> dict:
    """Create a new evaluation campaign for a completed run."""
    campaign_id = create_campaign(body.run_id, max_variations=body.max_variations)
    return get_campaign(campaign_id)


@router.get("")
def get_list_campaigns() -> list[dict]:
    """Retrieve metadata summaries for all robustness campaigns."""
    return list_campaigns()


@router.get("/{campaign_id}")
def get_one_campaign(campaign_id: str) -> dict:
    """Retrieve detailed execution history for one campaign."""
    try:
        return get_campaign(campaign_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id!r} not found") from exc


@router.post("/{campaign_id}/stop", dependencies=[Depends(require_admin)])
def post_stop_campaign(campaign_id: str) -> dict:
    """Halt an active robustness campaign."""
    try:
        stop_campaign(campaign_id)
        return get_campaign(campaign_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id!r} not found") from exc


@router.post("/{campaign_id}/escalate", dependencies=[Depends(require_admin)])
def post_escalate_campaign(campaign_id: str) -> dict:
    """Escalate a completed campaign to run more iterations or larger validation scopes."""
    try:
        escalate_campaign(campaign_id)
        return get_campaign(campaign_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id!r} not found") from exc

