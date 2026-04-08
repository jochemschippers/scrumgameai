from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import uuid

from .catalog_service import list_game_configs
from .checkpoint_service import get_checkpoint_by_id
from .app_paths import ensure_engine_import_path

ensure_engine_import_path()

from checkpoint_utils import load_agent_from_checkpoint  # noqa: E402
from config_manager import load_game_config  # noqa: E402
from match_runner import (  # noqa: E402
    HeuristicController,
    HumanController,
    ModelController,
    RandomController,
    all_seats_done,
    play_round,
    start_parallel_match,
    valid_actions_for_state,
)


PLAY_SESSIONS: dict[str, dict] = {}


def _resolve_game_config(game_config_id: str):
    for item in list_game_configs():
        if item["id"] == game_config_id or item["path"] == game_config_id:
            return load_game_config(item["path"]), item
    raise ValueError(f"Game config `{game_config_id}` was not found.")


@lru_cache(maxsize=16)
def _cached_agent(checkpoint_path: str, game_config_path: str):
    agent, _, metadata = load_agent_from_checkpoint(
        checkpoint_path,
        game_config=load_game_config(game_config_path),
        strict_signature=False,
    )
    return agent, metadata


def _controller_from_payload(payload: dict, game_config_path: str):
    controller_type = payload.get("type")
    display_name = payload.get("display_name")

    if controller_type == "human":
        return HumanController(display_name=display_name or "Human")
    if controller_type == "random":
        return RandomController(display_name=display_name or "Random AI")
    if controller_type == "heuristic":
        return HeuristicController(display_name=display_name or "Heuristic AI")
    if controller_type == "model":
        checkpoint_id = payload.get("checkpoint_id")
        if not checkpoint_id:
            raise ValueError("Model controller requires checkpoint_id.")
        checkpoint = get_checkpoint_by_id(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint `{checkpoint_id}` was not found.")
        agent, _ = _cached_agent(checkpoint["path"], game_config_path)
        return ModelController(
            agent=agent,
            profile_name=payload.get("profile_name", "expert"),
            display_name=display_name or "Checkpoint AI",
        )
    raise ValueError(f"Unknown controller type: {controller_type}")


def _seat_payload(seat: dict) -> dict:
    env = seat["env"]
    state = seat["state"]
    return {
        "controller": {
            "type": seat["controller"].controller_type,
            "display_name": seat["controller"].display_name,
            **({"profile_name": seat["controller"].profile_name} if hasattr(seat["controller"], "profile_name") else {}),
        },
        "seed": seat["seed"],
        "done": seat["done"],
        "total_reward": seat["total_reward"],
        "terminal_reason": seat["terminal_reason"],
        "state": {
            "current_money": state["current_money"],
            "current_product": state["current_product"],
            "current_sprint": state["current_sprint"],
            "features_required": state["features_required"],
            "sprint_value": state["sprint_value"],
            "remaining_turns": state["remaining_turns"],
            "expected_value": state["expected_value"],
            "win_probability": state["win_probability"],
            "current_product_completed": state["current_product_completed"],
        },
        "valid_actions": [
            {
                "action_id": action_id,
                "label": env.action_name(action_id),
            }
            for action_id in valid_actions_for_state(env, state)
        ],
        "steps": seat["steps"],
    }


def _session_payload(session_id: str, match_state: dict) -> dict:
    return {
        "id": session_id,
        "base_seed": match_state["base_seed"],
        "round_number": match_state["round_number"],
        "done": all_seats_done(match_state),
        "seats": [_seat_payload(seat) for seat in match_state["seats"]],
    }


def list_sessions() -> list[dict]:
    return [_session_payload(session_id, match_state) for session_id, match_state in PLAY_SESSIONS.items()]


def create_session(payload: dict) -> dict:
    game_config, game_config_item = _resolve_game_config(payload["game_config_id"])
    controllers_payload = payload.get("controllers") or []
    if not controllers_payload:
        raise ValueError("At least one controller is required.")

    controllers = [
        _controller_from_payload(controller_payload, game_config_item["path"])
        for controller_payload in controllers_payload
    ]
    match_state = start_parallel_match(
        game_config=game_config,
        controllers=controllers,
        base_seed=int(payload.get("base_seed", 42)),
    )

    session_id = str(uuid.uuid4())
    PLAY_SESSIONS[session_id] = match_state
    return _session_payload(session_id, match_state)


def get_session(session_id: str) -> dict:
    match_state = PLAY_SESSIONS.get(session_id)
    if match_state is None:
        raise ValueError(f"Play session `{session_id}` was not found.")
    return _session_payload(session_id, match_state)


def advance_session(session_id: str, payload: dict | None = None) -> dict:
    match_state = PLAY_SESSIONS.get(session_id)
    if match_state is None:
        raise ValueError(f"Play session `{session_id}` was not found.")

    human_action = None if payload is None else payload.get("human_action")
    play_round(match_state, human_action=human_action)
    return _session_payload(session_id, match_state)
