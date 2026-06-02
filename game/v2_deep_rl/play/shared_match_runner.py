from __future__ import annotations

from copy import deepcopy
import random
from typing import Any

from game_runtime.scrum_game_env import ScrumGameEnv
from play.match_runner import Controller, valid_actions_for_state


def _copy_board_from_env(env: ScrumGameEnv) -> dict[str, Any]:
    return {
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


def _step_without_incident(env: ScrumGameEnv, action: int):
    incidents_active = env.incidents_active
    env.incidents_active = False
    try:
        return env.step(action)
    finally:
        env.incidents_active = incidents_active


def _apply_round_incident(match_state: dict[str, Any]) -> None:
    """Resolve the single Wait/Incident phase after all seats act."""
    seats = match_state["seats"]
    if not seats:
        return

    env = seats[0]["env"]
    private_progress = list(env.product_next_sprints)
    private_current_product = env.current_product
    _sync_board_into_env(env, match_state["board_state"])

    try:
        # Incidents mutate the shared board, not one player's private progress.
        # Treat all configured cells as globally future board cells.
        env.product_next_sprints = [1] * env.products_count
        env.current_product = 1
        if (
            env.incidents_active
            and env.incident_deck is not None
            and random.random() <= env.incident_draw_probability
        ):
            incident_card = env.incident_deck.draw()
            incident_card.apply_effect(env)
            env.incident_active = 1
            env.current_incident_id = incident_card.card_id
            env.current_incident_name = incident_card.name
            env.current_incident_scope = env._encode_incident_scope(incident_card)
            match_state["round_incidents"].append(
                {
                    "round": match_state["round_number"],
                    "id": incident_card.card_id,
                    "name": incident_card.name,
                    "description": incident_card.description,
                }
            )
        else:
            env.incident_active = 0
            env.current_incident_id = 0
            env.current_incident_name = "None"
            env.current_incident_scope = 0.0
        match_state["board_state"] = _capture_board_after_turn(env, match_state["board_state"])
    finally:
        env.product_next_sprints = private_progress
        env.current_product = private_current_product
        _sync_board_into_env(env, match_state["board_state"])


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
        "round_incidents": [],
    }


def _record_shared_step(match_state: dict[str, Any], seat: dict[str, Any], action: int, reward, done, info) -> None:
    daily_scrums = info.get("daily_scrums", []) or []
    total_rolled = sum(int(scrum.get("roll_total", 0)) for scrum in daily_scrums)
    target_total = 0
    if daily_scrums:
        target_total = len(daily_scrums) * int(seat["env"].daily_scrum_target)
    variance = info.get("net_result", 0)
    planning_penalty = abs(int(variance or 0)) * (
        seat["env"].penalty_negative if int(variance or 0) <= 0 else seat["env"].penalty_positive
    )
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
        "dice": {
            "features_required": info.get("features_required"),
            "dice_label": (
                f"{daily_scrums[0].get('dice_count')}x D{daily_scrums[0].get('dice_sides')}"
                if daily_scrums else "-"
            ),
            "daily_scrums": daily_scrums,
            "total_rolled": total_rolled,
            "target_total": target_total,
            "variance": variance,
            "planning_penalty": planning_penalty,
            "payout": info.get("payout", 0),
            "switch_cost_paid": info.get("switch_cost_paid", 0),
            "continue_cost_paid": info.get("continue_cost_paid", 0),
            "interest_paid": info.get("interest_paid", 0),
        },
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


def _seats_in_round_order(match_state: dict[str, Any]) -> list[dict[str, Any]]:
    seats = match_state["seats"]
    if not seats:
        return []
    offset = (int(match_state["round_number"]) - 1) % len(seats)
    return seats[offset:] + seats[:offset]


def play_shared_round(match_state: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Advance every active seat by one turn, rotating the first actor each round."""
    human_actions = _human_actions_by_seat(payload)
    waiting_humans = [
        seat for seat in match_state["seats"]
        if not seat["done"] and seat["controller"].controller_type == "human"
    ]
    missing_humans = [seat["id"] for seat in waiting_humans if seat["id"] not in human_actions]
    if missing_humans:
        raise ValueError(f"Human action required for {', '.join(missing_humans)}.")

    for seat in _seats_in_round_order(match_state):
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

        next_state, reward, done, info = _step_without_incident(env, action)
        match_state["board_state"] = _capture_board_after_turn(env, match_state["board_state"])
        _sync_board_into_env(env, match_state["board_state"])
        seat["state"] = next_state
        _record_shared_step(match_state, seat, action, reward, done, info)

    if any(row["round"] == match_state["round_number"] for row in match_state["turn_log"]):
        _apply_round_incident(match_state)

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
    seat_positions = {}
    for seat in match_state["seats"]:
        state = seat["state"]
        if seat["done"]:
            continue
        product_id = int(state["current_product"])
        sprint_id = int(state["current_sprint"])
        seat_positions.setdefault((product_id, sprint_id), []).append(seat["id"])

    products = []
    for product_id, product_name in enumerate(env.product_names, start=1):
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
            completed_for_all = bool(match_state["seats"]) and all(
                seat["done"] or int(seat["state"]["target_next_sprints"][product_id - 1]) > sprint_id
                for seat in match_state["seats"]
            )
            cells.append(
                {
                    "sprint": sprint_id,
                    "completed": completed_for_all,
                    "active": bool(seat_positions.get((product_id, sprint_id))),
                    "active_seats": seat_positions.get((product_id, sprint_id), []),
                    "base_value": base_value,
                    "sprint_value": max(0, sprint_value),
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
                "next_sprint": None,
                "completed": False,
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
        "round_incidents": list(match_state.get("round_incidents", [])),
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
            "target_next_sprints": state["target_next_sprints"],
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
