from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from jobs.queue_manager import enqueue_train_job
from services.app_paths import ARTIFACTS_DIR, RUNS_DIR
from services.campaign_variation_generator import generate_variation


CAMPAIGNS_DIR = ARTIFACTS_DIR / "campaigns"


def _ensure_dir() -> None:
    CAMPAIGNS_DIR.mkdir(parents=True, exist_ok=True)


def _campaign_path(campaign_id: str) -> Path:
    return CAMPAIGNS_DIR / f"{campaign_id}.json"


def _read_campaign(campaign_id: str) -> dict:
    path = _campaign_path(campaign_id)
    if not path.exists():
        raise FileNotFoundError(f"Campaign not found: {campaign_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_campaign(data: dict) -> None:
    _ensure_dir()
    _campaign_path(data["campaign_id"]).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_campaign_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    return f"campaign_{ts}_{uuid.uuid4().hex[:6]}"


def create_campaign(base_run_id: str, max_variations: int = 5) -> str:
    _ensure_dir()
    campaign_id = _make_campaign_id()
    _write_campaign(
        {
            "campaign_id": campaign_id,
            "status": "running",
            "base_run_id": base_run_id,
            "current_run_id": base_run_id,
            "variations_completed": 0,
            "max_variations": int(max_variations),
            "escalate_mode": False,
            "variation_history": [],
        }
    )
    return campaign_id


def get_campaign(campaign_id: str) -> dict:
    return _read_campaign(campaign_id)


def list_campaigns() -> list[dict]:
    _ensure_dir()
    campaigns = []
    for path in sorted(CAMPAIGNS_DIR.glob("campaign_*.json")):
        try:
            campaigns.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return campaigns


def stop_campaign(campaign_id: str) -> None:
    data = _read_campaign(campaign_id)
    data["status"] = "stopped"
    _write_campaign(data)


def get_campaign_for_run(run_id: str) -> dict | None:
    for campaign in list_campaigns():
        if campaign.get("status") != "running":
            continue
        if campaign.get("base_run_id") == run_id or campaign.get("current_run_id") == run_id:
            return campaign
    return None


def _next_run_id(campaign: dict) -> str:
    return f"{campaign['base_run_id']}_cv{campaign['variations_completed'] + 1}"


def _save_varied_config(campaign_id: str, variation_index: int, config_dict: dict) -> Path:
    _ensure_dir()
    path = CAMPAIGNS_DIR / f"{campaign_id}_game_config_v{variation_index}.json"
    path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    return path


def on_run_stopped(run_id: str, metrics: dict) -> None:
    campaign = get_campaign_for_run(run_id)
    if campaign is None:
        return
    _generate_and_queue_next(campaign, run_id, metrics, escalate=False)


def _generate_and_queue_next(campaign: dict, from_run_id: str, metrics: dict, escalate: bool) -> None:
    cid = campaign["campaign_id"]
    variation_index = campaign["variations_completed"] + 1
    run_dir = RUNS_DIR / from_run_id

    try:
        config_dict = json.loads((run_dir / "game_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        config_dict = {}

    checkpoint_path = run_dir / "checkpoints" / "latest_scrum_model.pth"
    new_config_dict, changes, reason = generate_variation(
        config_dict,
        metrics,
        variation_index=variation_index,
        escalate=escalate,
    )
    config_path = _save_varied_config(cid, variation_index, new_config_dict)

    new_run_id = _next_run_id(campaign)
    job = enqueue_train_job(
        {
            "run_name": new_run_id,
            "game_config_path": str(config_path),
            "resume_from": str(checkpoint_path),
            "resume_mode": "strict",
            "resume_episodes_mode": "incremental",
            "autopilot_after_completion": True,
            "autopilot_context": {},
        }
    )
    new_run_dir = job.get("run_dir", "")
    actual_run_id = Path(new_run_dir).name if new_run_dir else new_run_id

    updated = dict(campaign)
    updated["current_run_id"] = actual_run_id
    updated["variations_completed"] = variation_index
    updated["variation_history"] = list(campaign.get("variation_history", [])) + [
        {
            "index": variation_index,
            "from_run": from_run_id,
            "to_run": actual_run_id,
            "changes": changes,
            "reason": reason,
            "escalate": escalate,
        }
    ]
    if variation_index >= int(updated.get("max_variations", 0)):
        updated["status"] = "completed"
    else:
        updated["status"] = "running"
    _write_campaign(updated)


def escalate_campaign(campaign_id: str) -> None:
    campaign = _read_campaign(campaign_id)
    last_run_id = campaign["current_run_id"]
    metrics = {
        "latest_reward": 0,
        "bankruptcy_rate": 0,
        "average_ending_money": 0,
        "invalid_action_rate": 0,
        "reward_cv": 0,
    }

    campaign["status"] = "running"
    campaign["escalate_mode"] = True
    campaign["max_variations"] = int(campaign.get("variations_completed", 0)) + 1
    _write_campaign(campaign)

    _generate_and_queue_next(campaign, last_run_id, metrics, escalate=True)
