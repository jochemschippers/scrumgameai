"""Test play service shared behavior."""

from __future__ import annotations

import pytest
import sys

for module_name in ("shared_match_runner", "match_runner", "scrum_game_env"):
    sys.modules.pop(module_name, None)

from services import play_service


# Handle setup function.
def setup_function():
    play_service.PLAY_SESSIONS.clear()


# Verify create shared session rejects invalid seat counts.
def test_create_shared_session_rejects_invalid_seat_counts():
    with pytest.raises(ValueError, match="At least one seat"):
        play_service.create_session({"mode": "shared", "game_config_id": "default_game_config", "seats": []})

    with pytest.raises(ValueError, match="at most 4"):
        play_service.create_session(
            {
                "mode": "shared",
                "game_config_id": "default_game_config",
                "seats": [{"type": "random"} for _ in range(5)],
            }
        )


# Verify create shared session rejects multiple humans.
def test_create_shared_session_rejects_multiple_humans():
    with pytest.raises(ValueError, match="at most one human"):
        play_service.create_session(
            {
                "mode": "shared",
                "game_config_id": "default_game_config",
                "seats": [{"type": "human"}, {"type": "human"}],
            }
        )


# Verify create shared session rejects model without checkpoint.
def test_create_shared_session_rejects_model_without_checkpoint():
    with pytest.raises(ValueError, match="requires checkpoint_id"):
        play_service.create_session(
            {
                "mode": "shared",
                "game_config_id": "default_game_config",
                "seats": [{"type": "model", "profile_name": "expert"}],
            }
        )


# Verify create and advance shared ai session.
def test_create_and_advance_shared_ai_session():
    session = play_service.create_session(
        {
            "mode": "shared",
            "game_config_id": "default_game_config",
            "base_seed": 42,
            "seats": [{"type": "random", "display_name": "AI 1"}, {"type": "heuristic", "display_name": "AI 2"}],
        }
    )

    assert session["mode"] == "shared"
    assert len(session["seats"]) == 2
    assert session["board"]["products"]
    assert session["standings"]

    advanced = play_service.advance_session(session["id"], {})

    assert advanced["round_number"] == 2
    assert len(advanced["turn_log"]) == 2


# Verify legacy human action payload is supported.
def test_legacy_human_action_payload_is_supported():
    session = play_service.create_session(
        {
            "mode": "shared",
            "game_config_id": "default_game_config",
            "base_seed": 42,
            "seats": [{"type": "human", "display_name": "Player"}],
        }
    )

    advanced = play_service.advance_session(session["id"], {"human_action": 0})

    assert advanced["round_number"] == 2
    assert len(advanced["turn_log"]) == 1
