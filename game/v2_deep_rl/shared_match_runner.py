from __future__ import annotations

from copy import deepcopy
from typing import Any

from match_runner import Controller, valid_actions_for_state
from scrum_game_env import ScrumGameEnv


def _copy_board_from_env(env: ScrumGameEnv) -> dict[str, Any]:
    return {
        "product_next_sprints": list(env.product_next_sprints),
        "refinement_feature_deltas": deepcopy(env.refinement_feature_deltas),
        "incident_value_deltas": deepcopy(env.incident_value_deltas),
        "incident_value_overrides": deepcopy(env.incident_value_overrides),
        "incident_deck": env.incident_deck,
        "incident_active": int(env.incident_active),
        "current_incident_id": int(env.current_incident_id),
        "current_incident_name": env.current_incident_name,
        "current_incident_scope": float(env.current_incident_scope),
    }


def _sync_board_into_env(env: ScrumGameEnv, board_state: dict[str, Any]) -> None:
    env.product_next_sprints = list(board_state["product_next_sprints"])
    env.refinement_feature_deltas = deepcopy(board_state["refinement_feature_deltas"])
    env.incident_value_deltas = deepcopy(board_state["incident_value_deltas"])
    env.incident_value_overrides = deepcopy(board_state["incident_value_overrides"])
    env.incident_deck = board_state["incident_deck"]
    env.incident_active = int(board_state.get("incident_active", 0))
    env.current_incident_id = int(board_state.get("current_incident_id", 0))
    env.current_incident_name = board_state.get("current_incident_name", "None")
    env.current_incident_scope = float(board_state.get("current_incident_scope", 0.0))
    env._refresh_observation_fields()


def _capture_board_after_turn(env: ScrumGameEnv, board_state: dict[str, Any]) -> dict[str, Any]:
    updated = _copy_board_from_env(env)
    # Keep the shared incident deck identity from the acting env because draw()
    # mutates draw/discard piles.
    updated["incident_deck"] = env.incident_deck or board_state.get("incident_deck")
    return updated


def _create_shared_seat(
    seat_id: str,
    controller: Controller,
    game_config,
    board_state: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    env = ScrumGameEnv(game_config=game_config)
    env.reset(seed=seed)
    _sync_board_into_env(env, board_state)
    return {
        "id": seat_id,
        "controller": controller,
        "seed": seed,
        "env": env,
        "state": env._get_state(),
        "done": False,
        "steps": [],
        "total_reward": 0.0,
        "terminal_reason": "",
    }


def start_shared_match(game_config, controllers: list[Controller], base_seed: int = 42) -> dict[str, Any]:
    """Create one shared-board match with private player finances."""
    template_env = ScrumGameEnv(game_config=game_config)
    template_env.reset(seed=base_seed)
    board_state = _copy_board_from_env(template_env)
    seats = [
        _create_shared_seat(
            seat_id=f"seat_{index + 1}",
            controller=controller,
            game_config=game_config,
            board_state=board_state,
            seed=base_seed + index,
        )
        for index, controller in enumerate(controllers)
    ]
    return {
        "mode": "shared",
        "game_config": game_config,
        "base_seed": base_seed,
        "round_number": 1,
        "board_state": board_state,
        "seats": seats,
        "turn_log": [],
    }


def _record_shared_step(match_state: dict[str, Any], seat: dict[str, Any], action: int, reward, done, info) -> None:
    row = {
        "round": match_state["round_number"],
        "seat_id": seat["id"],
        "controller": seat["controller"].display_name,
        "action_id": action,
        "action": info["action_name"],
        "outcome": info["result"],
        "reward": reward,
        "bank": info["ending_money"],
        "payout": info.get("payout", 0),
        "product": info.get("product_name", ""),
        "sprint": info.get("played_sprint"),
        "terminal": info.get("terminal_reason", ""),
        "incident": info.get("incident_card_name", "None"),
        "refinement": info.get("refinement_effect", "none"),
    }
    seat["steps"].append(row)
    match_state["turn_log"].append(row)
    seat["total_reward"] += reward
    seat["done"] = done
    seat["terminal_reason"] = info.get("terminal_reason", "")


def _human_actions_by_seat(payload: dict[str, Any] | None) -> dict[str, int]:
    if not payload:
        return {}
    raw_actions = payload.get("human_actions")
    if isinstance(raw_actions, dict):
        return {str(seat_id): int(action) for seat_id, action in raw_actions.items()}
    if payload.get("human_action") is not None:
        return {"seat_1": int(payload["human_action"])}
    return {}


def play_shared_round(match_state: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Advance every active seat by one turn in seat order."""
    human_actions = _human_actions_by_seat(payload)
    waiting_humans = [
        seat for seat in match_state["seats"]
        if not seat["done"] and seat["controller"].controller_type == "human"
    ]
    missing_humans = [seat["id"] for seat in waiting_humans if seat["id"] not in human_actions]
    if missing_humans:
        raise ValueError(f"Human action required for {', '.join(missing_humans)}.")

    for seat in match_state["seats"]:
        if seat["done"]:
            continue

        env = seat["env"]
        controller = seat["controller"]
        _sync_board_into_env(env, match_state["board_state"])
        state = env._get_state()

        if controller.controller_type == "human":
            action = human_actions[seat["id"]]
        else:
            action = controller.choose_action(state, env)

        next_state, reward, done, info = env.step(action)
        match_state["board_state"] = _capture_board_after_turn(env, match_state["board_state"])
        _sync_board_into_env(env, match_state["board_state"])
        seat["state"] = next_state
        _record_shared_step(match_state, seat, action, reward, done, info)

    for seat in match_state["seats"]:
        if not seat["done"]:
            _sync_board_into_env(seat["env"], match_state["board_state"])
            seat["state"] = seat["env"]._get_state()

    match_state["round_number"] += 1
    return match_state


def all_shared_seats_done(match_state: dict[str, Any]) -> bool:
    return all(seat["done"] for seat in match_state["seats"])


def standings(match_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seat in match_state["seats"]:
        state = seat["state"]
        rows.append(
            {
                "seat_id": seat["id"],
                "controller": seat["controller"].display_name,
                "type": seat["controller"].controller_type,
                "total_reward": round(float(seat["total_reward"]), 2),
                "ending_money": state["current_money"],
                "turns_played": len(seat["steps"]),
                "done": seat["done"],
                "terminal": seat["terminal_reason"],
            }
        )
    return sorted(rows, key=lambda row: (row["ending_money"], row["total_reward"]), reverse=True)


def board_payload(match_state: dict[str, Any]) -> dict[str, Any]:
    env = match_state["seats"][0]["env"] if match_state["seats"] else ScrumGameEnv(game_config=match_state["game_config"])
    _sync_board_into_env(env, match_state["board_state"])
    products = []
    for product_id, product_name in enumerate(env.product_names, start=1):
        next_sprint = env.product_next_sprints[product_id - 1]
        cells = []
        for sprint_id in range(1, env.sprints_per_product + 1):
            product_index = product_id - 1
            sprint_index = sprint_id - 1
            base_value = env.board_ring_values[product_index][sprint_index] * env.ring_value
            override = env.incident_value_overrides[product_index][sprint_index]
            incident_delta = env.incident_value_deltas[product_index][sprint_index]
            sprint_value = override if override is not None else base_value + incident_delta
            refinement_delta = env.refinement_feature_deltas[product_index][sprint_index]
            features_required = max(1, env.board_features[product_index][sprint_index] + refinement_delta)
            cells.append(
                {
                    "sprint": sprint_id,
                    "completed": sprint_id < next_sprint,
                    "active": sprint_id == next_sprint and next_sprint <= env.sprints_per_product,
                    "base_value": base_value,
                    "sprint_value": sprint_value,
                    "base_features": env.board_features[product_index][sprint_index],
                    "features_required": features_required,
                    "incident_delta": incident_delta,
                    "incident_override": override,
                    "refinement_delta": refinement_delta,
                }
            )
        products.append(
            {
                "product_id": product_id,
                "name": product_name,
                "next_sprint": next_sprint,
                "completed": next_sprint > env.sprints_per_product,
                "cells": cells,
            }
        )
    return {
        "products": products,
        "incident": {
            "active": bool(match_state["board_state"].get("incident_active")),
            "id": match_state["board_state"].get("current_incident_id", 0),
            "name": match_state["board_state"].get("current_incident_name", "None"),
        },
    }


def seat_payload(seat: dict[str, Any]) -> dict[str, Any]:
    env = seat["env"]
    state = seat["state"]
    return {
        "id": seat["id"],
        "controller": {
            "type": seat["controller"].controller_type,
            "display_name": seat["controller"].display_name,
            **({"profile_name": seat["controller"].profile_name} if hasattr(seat["controller"], "profile_name") else {}),
        },
        "seed": seat["seed"],
        "done": seat["done"],
        "total_reward": round(float(seat["total_reward"]), 2),
        "terminal_reason": seat["terminal_reason"],
        "state": {
            "current_money": state["current_money"],
            "current_product": state["current_product"],
            "current_sprint": state["current_sprint"],
            "features_required": state["features_required"],
            "sprint_value": state["sprint_value"],
            "loan_active": state["loan_active"],
            "interest_due": state["interest_due"],
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
