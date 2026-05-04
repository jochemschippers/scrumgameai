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
    run_id: str
    max_variations: int = Field(default=5, ge=1, le=20)


@router.post("", dependencies=[Depends(require_admin)])
def post_create_campaign(body: CreateCampaignRequest) -> dict:
    campaign_id = create_campaign(body.run_id, max_variations=body.max_variations)
    return get_campaign(campaign_id)


@router.get("")
def get_list_campaigns() -> list[dict]:
    return list_campaigns()


@router.get("/{campaign_id}")
def get_one_campaign(campaign_id: str) -> dict:
    try:
        return get_campaign(campaign_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id!r} not found") from exc


@router.post("/{campaign_id}/stop", dependencies=[Depends(require_admin)])
def post_stop_campaign(campaign_id: str) -> dict:
    try:
        stop_campaign(campaign_id)
        return get_campaign(campaign_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id!r} not found") from exc


@router.post("/{campaign_id}/escalate", dependencies=[Depends(require_admin)])
def post_escalate_campaign(campaign_id: str) -> dict:
    try:
        escalate_campaign(campaign_id)
        return get_campaign(campaign_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id!r} not found") from exc
