from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _patch_dir(campaigns_dir):
    return patch("services.campaign_service.CAMPAIGNS_DIR", campaigns_dir)


def _base_config_dict() -> dict:
    return {
        "schema_version": "2.0",
        "config_name": "Test",
        "config_description": "",
        "players_count": 1,
        "product_names": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
        "max_turns": 6,
        "starting_money": 25000,
        "ring_value": 10000,
        "cost_continue": 0,
        "cost_switch_mid": 5000,
        "cost_switch_after": 2500,
        "mandatory_loan_amount": 50000,
        "loan_interest": 5000,
        "penalty_negative": -100,
        "penalty_positive": 0,
        "daily_scrums_per_sprint": 3,
        "daily_scrum_target": 3,
        "board_ring_values": [[0] * 7 for _ in range(4)],
        "board_features": [[3] * 7 for _ in range(4)],
        "dice_rules": [{"min_features": 1, "max_features": 4, "dice_count": 1, "dice_sides": 6}],
        "refinement": {"active": False, "product_rules": []},
        "incident": {
            "active": True,
            "allow_player_specific_incidents": False,
            "draw_probability": 1.0,
            "severity_multiplier": 1.0,
            "cards": [],
        },
        "reserved_fields": {},
    }


def _make_run_dir(runs_dir: Path, run_id: str, game_config: dict | None = None) -> Path:
    run_dir = runs_dir / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "latest_scrum_model.pth").touch()
    (run_dir / "checkpoints" / "best_scrum_model.pth").touch()
    (run_dir / "game_config.json").write_text(json.dumps(game_config or _base_config_dict()), encoding="utf-8")
    return run_dir


def test_create_campaign_writes_json(tmp_path):
    from services.campaign_service import create_campaign

    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_2026-04-20_1400", max_variations=5)

    data = json.loads((d / f"{cid}.json").read_text(encoding="utf-8"))
    assert data["campaign_id"] == cid
    assert data["base_run_id"] == "run_2026-04-20_1400"
    assert data["current_run_id"] == "run_2026-04-20_1400"
    assert data["max_variations"] == 5
    assert data["variations_completed"] == 0
    assert data["status"] == "running"
    assert data["variation_history"] == []
    assert data["escalate_mode"] is False


def test_get_campaign_returns_data(tmp_path):
    from services.campaign_service import create_campaign, get_campaign

    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_abc", max_variations=3)
        result = get_campaign(cid)

    assert result["campaign_id"] == cid
    assert result["max_variations"] == 3


def test_get_campaign_unknown_raises(tmp_path):
    from services.campaign_service import get_campaign

    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d), pytest.raises(FileNotFoundError):
        get_campaign("nonexistent")


def test_list_campaigns_returns_all(tmp_path):
    from services.campaign_service import create_campaign, list_campaigns

    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid1 = create_campaign("run_a", max_variations=3)
        cid2 = create_campaign("run_b", max_variations=5)
        results = list_campaigns()

    ids = [c["campaign_id"] for c in results]
    assert cid1 in ids and cid2 in ids


def test_stop_campaign_sets_status(tmp_path):
    from services.campaign_service import create_campaign, get_campaign, stop_campaign

    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_x", max_variations=5)
        stop_campaign(cid)
        result = get_campaign(cid)

    assert result["status"] == "stopped"


def test_get_campaign_for_run_finds_by_base_run_id(tmp_path):
    from services.campaign_service import create_campaign, get_campaign_for_run

    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_base", max_variations=5)
        result = get_campaign_for_run("run_base")

    assert result is not None
    assert result["campaign_id"] == cid


def test_get_campaign_for_run_returns_none_when_stopped(tmp_path):
    from services.campaign_service import create_campaign, get_campaign_for_run, stop_campaign

    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_stopped", max_variations=5)
        stop_campaign(cid)
        result = get_campaign_for_run("run_stopped")

    assert result is None


def test_get_campaign_for_run_finds_by_current_run_id(tmp_path):
    from services.campaign_service import create_campaign, get_campaign, get_campaign_for_run, _write_campaign

    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_base", max_variations=5)
        data = {**get_campaign(cid), "current_run_id": "run_base_v1"}
        _write_campaign(data)
        result = get_campaign_for_run("run_base_v1")

    assert result is not None
    assert result["campaign_id"] == cid


def test_on_run_stopped_queues_next_job(tmp_path):
    from services.campaign_service import create_campaign, get_campaign, on_run_stopped

    d = tmp_path / "campaigns"
    d.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dir(runs, "run_base")

    with _patch_dir(d), patch("services.campaign_service.RUNS_DIR", runs), patch(
        "services.campaign_service.generate_variation",
        return_value=(_base_config_dict(), {"starting_money": 28000}, "test reason"),
    ), patch(
        "services.campaign_service.enqueue_train_job",
        return_value={"id": 1, "run_dir": str(runs / "run_base_cv1")},
    ) as mock_enqueue:
        cid = create_campaign("run_base", max_variations=5)
        on_run_stopped("run_base", {"latest_reward": 50000.0})
        result = get_campaign(cid)

    mock_enqueue.assert_called_once()
    assert result["variations_completed"] == 1
    assert result["status"] == "running"
    assert result["variation_history"][0]["reason"] == "test reason"


def test_on_run_stopped_completes_at_max_variations(tmp_path):
    from services.campaign_service import create_campaign, get_campaign, on_run_stopped

    d = tmp_path / "campaigns"
    d.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dir(runs, "run_base")

    with _patch_dir(d), patch("services.campaign_service.RUNS_DIR", runs), patch(
        "services.campaign_service.generate_variation",
        return_value=(_base_config_dict(), {}, "reason"),
    ), patch(
        "services.campaign_service.enqueue_train_job",
        return_value={"id": 1, "run_dir": str(runs / "run_base_cv1")},
    ):
        cid = create_campaign("run_base", max_variations=1)
        on_run_stopped("run_base", {"latest_reward": 50000.0})
        result = get_campaign(cid)

    assert result["variations_completed"] == 1
    assert result["status"] == "completed"


def test_on_run_stopped_no_op_when_no_campaign(tmp_path):
    from services.campaign_service import on_run_stopped

    d = tmp_path / "campaigns"
    d.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dir(runs, "run_orphan")

    with _patch_dir(d), patch("services.campaign_service.RUNS_DIR", runs), patch(
        "services.campaign_service.enqueue_train_job"
    ) as mock_enqueue:
        on_run_stopped("run_orphan", {"latest_reward": 0})

    mock_enqueue.assert_not_called()


def test_escalate_queues_job_with_escalate_mode(tmp_path):
    from services.campaign_service import create_campaign, escalate_campaign, get_campaign

    d = tmp_path / "campaigns"
    d.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dir(runs, "run_final_v5")

    with _patch_dir(d), patch("services.campaign_service.RUNS_DIR", runs):
        cid = create_campaign("run_base", max_variations=5)
        data = get_campaign(cid)
        data["status"] = "completed"
        data["current_run_id"] = "run_final_v5"
        data["variations_completed"] = 5
        (d / f"{cid}.json").write_text(json.dumps(data), encoding="utf-8")

        with patch(
            "services.campaign_service.generate_variation",
            return_value=(_base_config_dict(), {"max_turns": 8}, "escalate reason"),
        ) as mock_gen, patch(
            "services.campaign_service.enqueue_train_job",
            return_value={"id": 2, "run_dir": str(runs / "run_final_v5_esc1")},
        ):
            escalate_campaign(cid)
            result = get_campaign(cid)

    assert result["status"] == "completed"
    assert result["escalate_mode"] is True
    assert mock_gen.call_args.kwargs["escalate"] is True
