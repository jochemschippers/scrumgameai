"""Test campaign variation generator behavior."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


# Handle base config dict.
def _base_config_dict():
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
        "dice_rules": [
            {"min_features": 1, "max_features": 4, "dice_count": 1, "dice_sides": 6},
            {"min_features": 5, "max_features": 8, "dice_count": 1, "dice_sides": 8},
        ],
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


# Verify clamp starting money within safe bounds.
def test_clamp_starting_money_within_safe_bounds():
    from services.campaign_variation_generator import clamp_diff

    result = clamp_diff({"starting_money": 40000}, _base_config_dict(), escalate=False)

    assert result["starting_money"] == 31250


# Verify clamp max turns within safe bounds.
def test_clamp_max_turns_within_safe_bounds():
    from services.campaign_variation_generator import clamp_diff

    result = clamp_diff({"max_turns": 20}, _base_config_dict(), escalate=False)

    assert result["max_turns"] == 8


# Verify clamp incident draw probability.
def test_clamp_incident_draw_probability():
    from services.campaign_variation_generator import clamp_diff

    result = clamp_diff({"incident_draw_probability": 0.1}, _base_config_dict(), escalate=False)

    assert result["incident_draw_probability"] == 0.5


# Verify escalate allows wider starting money.
def test_escalate_allows_wider_starting_money():
    from services.campaign_variation_generator import clamp_diff

    result = clamp_diff({"starting_money": 39000}, _base_config_dict(), escalate=True)

    assert result["starting_money"] == 39000


# Verify clamp dice sides.
def test_clamp_dice_sides():
    from services.campaign_variation_generator import clamp_diff

    result = clamp_diff({"dice_rule_0_dice_sides": 15}, _base_config_dict(), escalate=False)

    assert result["dice_rule_0_dice_sides"] == 8


# Verify apply diff to config dict.
def test_apply_diff_to_config_dict():
    from services.campaign_variation_generator import apply_diff_to_config

    result = apply_diff_to_config(
        _base_config_dict(),
        {
            "starting_money": 28000,
            "incident_draw_probability": 0.8,
            "dice_rule_0_dice_sides": 8,
        },
    )

    assert result["starting_money"] == 28000
    assert result["incident"]["draw_probability"] == 0.8
    assert result["dice_rules"][0]["dice_sides"] == 8
    assert result["max_turns"] == 6


# Verify generate with mocked ai returns new config.
def test_generate_with_mocked_ai_returns_new_config():
    from services.campaign_variation_generator import generate_variation

    ai_response = json.dumps(
        {
            "changes": {"starting_money": 30000, "incident_draw_probability": 0.8},
            "reason": "High bankruptcy rate - more starting capital may help",
        }
    )
    mock_choice = MagicMock()
    mock_choice.message.content = ai_response
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("services.campaign_variation_generator.OpenAI", return_value=mock_client):
        new_config, changes, reason = generate_variation(
            _base_config_dict(),
            {
                "latest_reward": 50000.0,
                "bankruptcy_rate": 0.40,
                "average_ending_money": 30000.0,
                "invalid_action_rate": 0.05,
                "reward_cv": 0.15,
            },
            variation_index=1,
            escalate=False,
        )

    assert new_config["starting_money"] == 30000
    assert new_config["incident"]["draw_probability"] == 0.8
    assert "bankruptcy" in reason.lower()
    assert "starting_money" in changes


# Verify generate falls back on ai failure.
def test_generate_falls_back_on_ai_failure():
    from services.campaign_variation_generator import generate_variation

    base = _base_config_dict()

    with patch("services.campaign_variation_generator.OpenAI", side_effect=Exception("network error")):
        new_config, changes, reason = generate_variation(
            base,
            {"latest_reward": 0, "bankruptcy_rate": 0.5, "average_ending_money": 0},
            variation_index=1,
            escalate=False,
        )

    assert new_config == base
    assert changes == {}
    assert "fallback" in reason.lower()
