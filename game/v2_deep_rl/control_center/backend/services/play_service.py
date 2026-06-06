"""
Interactive Play Session Coordinator Service.

This service manages active, in-memory play sessions representing live matches of the Scrum Game.
It supports both parallel mode (independent boards per player) and shared board mode (players compete
on a unified global board with a shared incident deck). It maps client requests to play controllers,
caches neural network policies for AI agents, and advances simulation rounds.

Key Flow:
  1. Session creation: Spawns environments and attaches controllers (Human, Model AI, Heuristic, or Random).
  2. Action evaluation: Evaluates model actions or processes human actions submitted from the UI.
  3. State translation: Serializes complex gym observations, standings, and turn history logs.

Connections:
  - Imports: `list_game_configs` from `services.catalog_service`, and match runners from `play.match_runner` / `play.shared_match_runner`.
  - Used by: `api/routes_play.py` to serve active sessions.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import uuid

from .catalog_service import list_game_configs
from .checkpoint_service import get_checkpoint_by_id
from .app_paths import ensure_engine_import_path

ensure_engine_import_path()

# torch-dependent engine imports are deferred so the API server starts without torch.
# They are imported on first use inside each function that needs them.


PLAY_SESSIONS: dict[str, dict] = {}


def _resolve_game_config(game_config_id: str):
    """Load and resolve a game configuration and its catalog item by ID."""
    from config.config_manager import load_game_config  # noqa: E402
    for item in list_game_configs():
        if item["id"] == game_config_id or item["path"] == game_config_id:
            return load_game_config(item["path"]), item
    raise ValueError(f"Game config `{game_config_id}` was not found.")


@lru_cache(maxsize=16)
def _cached_agent(checkpoint_path: str, game_config_path: str):
    """Load and cache the inference policy/agent for a specific checkpoint path."""
    from rl.checkpoint_utils import load_agent_for_inference  # noqa: E402
    from config.config_manager import load_game_config  # noqa: E402
    agent, _, metadata = load_agent_for_inference(
        checkpoint_path,
        game_config=load_game_config(game_config_path),
        strict_signature=False,
    )
    return agent, metadata


def _controller_from_payload(payload: dict, game_config_path: str):
    """Factory function to build a player controller (Human, Random, Heuristic, or Model) from request details."""
    from play.match_runner import (  # noqa: E402
        HeuristicController, HumanController, ModelController, RandomController,
    )
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


def _valid_actions_for_state(env, state):
    """Compute the set of valid action IDs for the environment's current step state."""
    from play.match_runner import valid_actions_for_state  # noqa: E402
    return valid_actions_for_state(env, state)


def _seat_payload(seat: dict) -> dict:
    """Format and serialize an individual player seat's state, parameters, and valid actions."""
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
            "target_next_sprints": state["target_next_sprints"],
        },
        "valid_actions": [
            {
                "action_id": action_id,
                "label": env.action_name(action_id),
            }
            for action_id in _valid_actions_for_state(env, state)
        ],
        "steps": seat["steps"],
    }


def _shared_session_payload(session_id: str, match_state: dict) -> dict:
    """Format the dashboard/API state payload for a shared-board multiplayer game session."""
    from play.shared_match_runner import (  # noqa: E402
        all_shared_seats_done,
        board_payload,
        seat_payload,
        standings,
    )
    return {
        "id": session_id,
        "mode": "shared",
        "base_seed": match_state["base_seed"],
        "round_number": match_state["round_number"],
        "done": all_shared_seats_done(match_state),
        "board": board_payload(match_state),
        "seats": [seat_payload(seat) for seat in match_state["seats"]],
        "standings": standings(match_state),
        "turn_log": list(match_state.get("turn_log", [])),
        "round_incidents": list(match_state.get("round_incidents", [])),
    }


def _session_payload(session_id: str, match_state: dict) -> dict:
    """Determine play mode and compile the appropriate game session state payload."""
    if match_state.get("mode") == "shared":
        return _shared_session_payload(session_id, match_state)
    from play.match_runner import all_seats_done  # noqa: E402
    return {
        "id": session_id,
        "mode": "parallel",
        "base_seed": match_state["base_seed"],
        "round_number": match_state["round_number"],
        "done": all_seats_done(match_state),
        "seats": [_seat_payload(seat) for seat in match_state["seats"]],
    }


def list_sessions() -> list[dict]:
    """Retrieve lists of all active parallel or shared match play session states."""
    return [_session_payload(session_id, match_state) for session_id, match_state in PLAY_SESSIONS.items()]


def _seat_payloads_from_request(payload: dict) -> list[dict]:
    """Extract and validate the list of seats/controllers specified in a session creation request."""
    seats_payload = payload.get("seats")
    if seats_payload is None:
        seats_payload = payload.get("controllers") or []
    if not isinstance(seats_payload, list):
        raise ValueError("Seats must be a list.")
    if len(seats_payload) < 1:
        raise ValueError("At least one seat is required.")
    if len(seats_payload) > 4:
        raise ValueError("Shared play supports at most 4 seats.")
    human_count = sum(1 for seat in seats_payload if seat.get("type") == "human")
    if human_count > 1:
        raise ValueError("Shared play supports at most one human seat.")
    return seats_payload


def create_session(payload: dict) -> dict:
    """Initialize a new parallel or shared game session with configured player controllers."""
    game_config, game_config_item = _resolve_game_config(payload["game_config_id"])
    mode = payload.get("mode", "shared")
    controllers_payload = _seat_payloads_from_request(payload)

    controllers = [
        _controller_from_payload(controller_payload, game_config_item["path"])
        for controller_payload in controllers_payload
    ]
    if mode == "parallel":
        from play.match_runner import start_parallel_match  # noqa: E402
        match_state = start_parallel_match(
            game_config=game_config,
            controllers=controllers,
            base_seed=int(payload.get("base_seed", 42)),
        )
        match_state["mode"] = "parallel"
    elif mode == "shared":
        from play.shared_match_runner import start_shared_match  # noqa: E402
        match_state = start_shared_match(
            game_config=game_config,
            controllers=controllers,
            base_seed=int(payload.get("base_seed", 42)),
        )
    else:
        raise ValueError(f"Unknown play mode: {mode}")

    session_id = str(uuid.uuid4())
    PLAY_SESSIONS[session_id] = match_state
    return _session_payload(session_id, match_state)


def get_session(session_id: str) -> dict:
    """Retrieve an active play session by its session ID."""
    match_state = PLAY_SESSIONS.get(session_id)
    if match_state is None:
        raise ValueError(f"Play session `{session_id}` was not found.")
    return _session_payload(session_id, match_state)


def advance_session(session_id: str, payload: dict | None = None) -> dict:
    """Advance the session by one round, feeding in human action if provided."""
    match_state = PLAY_SESSIONS.get(session_id)
    if match_state is None:
        raise ValueError(f"Play session `{session_id}` was not found.")

    if match_state.get("mode") == "shared":
        from play.shared_match_runner import play_shared_round  # noqa: E402
        play_shared_round(match_state, payload or {})
    else:
        from play.match_runner import play_round  # noqa: E402
        human_action = None if payload is None else payload.get("human_action")
        play_round(match_state, human_action=human_action)
    return _session_payload(session_id, match_state)
