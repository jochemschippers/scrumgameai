from __future__ import annotations

import sys

import pytest


for module_name in ("shared_match_runner", "match_runner", "scrum_game_env"):
    sys.modules.pop(module_name, None)

from config_manager import load_game_config  # noqa: E402
from match_runner import HeuristicController, HumanController, RandomController  # noqa: E402
from shared_match_runner import (  # noqa: E402
    all_shared_seats_done,
    board_payload,
    play_shared_round,
    start_shared_match,
    standings,
)


def test_shared_match_supports_one_to_four_seats():
    config = load_game_config()
    for count in range(1, 5):
        match_state = start_shared_match(
            config,
            [RandomController(display_name=f"AI {index}") for index in range(count)],
            base_seed=100,
        )
        assert match_state["mode"] == "shared"
        assert len(match_state["seats"]) == count
        assert [seat["id"] for seat in match_state["seats"]] == [
            f"seat_{index + 1}" for index in range(count)
        ]


def test_sprint_completion_is_shared_across_seats():
    config = load_game_config()
    match_state = start_shared_match(
        config,
        [HumanController(display_name="Player"), HeuristicController(display_name="AI")],
        base_seed=42,
    )
    first_seat = match_state["seats"][0]
    first_seat["env"]._play_daily_scrums = lambda _features: {"daily_scrums": [], "net_result": 0}

    play_shared_round(match_state, {"human_actions": {"seat_1": 0}})

    assert match_state["board_state"]["product_next_sprints"][0] == 2
    assert match_state["seats"][1]["state"]["target_next_sprints"][0] == 2
    assert first_seat["state"]["current_money"] != match_state["seats"][1]["state"]["current_money"]


def test_human_round_waits_for_human_action():
    config = load_game_config()
    match_state = start_shared_match(
        config,
        [HumanController(display_name="Player"), RandomController(display_name="AI")],
        base_seed=42,
    )

    with pytest.raises(ValueError, match="Human action required"):
        play_shared_round(match_state, {})

    assert match_state["round_number"] == 1
    assert match_state["turn_log"] == []


def test_ai_only_round_advances_and_standings_render():
    config = load_game_config()
    match_state = start_shared_match(
        config,
        [RandomController(display_name="AI 1"), HeuristicController(display_name="AI 2")],
        base_seed=42,
    )

    play_shared_round(match_state, {})

    assert match_state["round_number"] == 2
    assert len(match_state["turn_log"]) == 2
    assert len(standings(match_state)) == 2
    assert board_payload(match_state)["products"]
    assert not all_shared_seats_done(match_state)


def test_shared_match_rotates_first_actor_each_round():
    config = load_game_config()
    match_state = start_shared_match(
        config,
        [RandomController(display_name="AI 1"), RandomController(display_name="AI 2")],
        base_seed=42,
    )

    play_shared_round(match_state, {})
    play_shared_round(match_state, {})

    assert [row["seat_id"] for row in match_state["turn_log"][:4]] == [
        "seat_1",
        "seat_2",
        "seat_2",
        "seat_1",
    ]
