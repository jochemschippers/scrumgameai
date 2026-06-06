"""
Parallel Match Orchestrator for the Scrum Game.

This module simulates matches where multiple players (seats) compete in parallel.
Each seat runs in its own independent ScrumGameEnv using a specific Controller.

Why Parallel Matches?
  - Allows comparing the performance of different algorithms (Heuristic AI, Random AI, Deep RL Model AI)
    on identical layouts and configurations under the same base seed.

Controller Types:
  - HumanController: Placeholder for interactive players whose actions are fed from a user interface.
  - RandomController: Baseline AI that picks a legal action uniformly at random.
  - HeuristicController: Analytical baseline AI that calculates the immediate expected value of each product
    minus its switch/continue costs, greedily selecting the highest payout.
  - ModelController: Deep RL AI driven by a trained DQNAgent under a specific difficulty profile.

Connections:
  - Used by: Streamlit dashboard (`dashboard.py`) and FastAPI control center backend
    to simulate benchmark comparisons.
  - Interfaces with: `rl.dqn_agent.DQNAgent` and `play.deployment_profiles`.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from game_runtime.scrum_game_env import ScrumGameEnv
from play.deployment_profiles import choose_profile_action
from rl.dqn_agent import encode_state


def valid_actions_for_state(env: ScrumGameEnv, state: dict[str, Any]) -> list[int]:
    """
    Determine the list of legal action IDs for the given state.
    
    Rules:
      - Action 0 (continue) is only valid if the current product has uncompleted sprints.
      - Switching (action 1..N) is only valid if the target product is not completed.
      - Switching to the currently active product is invalid (handled as a continue instead).
    """
    valid_actions = []
    current_product = int(state["current_product"])

    # 1. Can we continue? Only if current product has sprints remaining
    if not state["current_product_completed"]:
        valid_actions.append(0)

    # 2. Can we switch? Loop over other products and check if they are incomplete
    for product_id in range(1, env.products_count + 1):
        if product_id == current_product:
            continue
        if state["target_is_completed"][product_id - 1]:
            continue
        valid_actions.append(product_id)

    # Fallback to 0 if all products are somehow complete, to avoid empty action lists
    return valid_actions or [0]


@dataclass
class Controller:
    """Abstract base class representing a Scrum Game player/decision-maker."""
    
    controller_type: str
    display_name: str

    def choose_action(self, state: dict[str, Any], env: ScrumGameEnv) -> int:
        """Choose and return an action index."""
        raise NotImplementedError


@dataclass
class HumanController(Controller):
    """Interactive controller whose actions are supplied by human input in a UI."""
    
    controller_type: str = "human"
    display_name: str = "Human"

    def choose_action(self, state: dict[str, Any], env: ScrumGameEnv) -> int:
        """Choose action for the HumanController. Always raises a RuntimeError, as human actions are input via UI."""
        raise RuntimeError("HumanController actions must be provided by the UI.")


@dataclass
class RandomController(Controller):
    """Baseline AI that picks a random action from the list of legal options."""
    
    controller_type: str = "random"
    display_name: str = "Random AI"

    def choose_action(self, state: dict[str, Any], env: ScrumGameEnv) -> int:
        """Randomly select a legal action from the set of valid actions."""
        valid_actions = valid_actions_for_state(env, state)
        return random.choice(valid_actions)


@dataclass
class HeuristicController(Controller):
    """
    Greedy analytical AI.
    Calculates expected value minus transition costs for all legal actions and picks the best one.
    """
    
    controller_type: str = "heuristic"
    display_name: str = "Heuristic AI"

    def choose_action(self, state: dict[str, Any], env: ScrumGameEnv) -> int:
        """Deterministically choose the action yielding the highest expected value net of transition costs."""
        valid_actions = valid_actions_for_state(env, state)
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Calculate current product score: E[Value] - Continue Cost
        current_score = float(state["expected_value"]) - float(env.cost_continue)
        best_action = 0
        best_score = current_score

        # Loop over other products and calculate their E[Value] minus Switch Cost
        for action in valid_actions:
            if action == 0:
                continue
            # Switch cost depends on whether the current product is completed
            switch_cost = env.cost_switch_after if state["target_is_completed"][state["current_product"] - 1] else env.cost_switch_mid
            candidate_score = float(state["target_expected_values"][action - 1]) - float(switch_cost)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_action = action

        return best_action


@dataclass
class ModelController(Controller):
    """
    AI Controller driven by a trained Deep RL model (DQNAgent) using difficulty profiles.
    """
    
    agent: Any = None
    profile_name: str = "expert"
    controller_type: str = "model"
    display_name: str = "Checkpoint AI"

    def choose_action(self, state: dict[str, Any], env: ScrumGameEnv) -> int:
        """Select action using the underlying trained DQN policy matched to the active difficulty profile."""
        # Encode state dict into normalized vector
        state_vector = encode_state(state, env)
        # Delegate selection to profile helper
        return choose_profile_action(
            self.agent,
            state_vector,
            profile_name=self.profile_name,
            valid_actions=valid_actions_for_state(env, state),
        )


def create_match_seat(controller: Controller, game_config, seed: int) -> dict[str, Any]:
    """
    Initialize one player seat with its own environment instance and tracking lists.
    """
    env = ScrumGameEnv(game_config=game_config)
    initial_state = env.reset(seed=seed)
    return {
        "controller": controller,
        "seed": seed,
        "env": env,
        "state": initial_state,
        "done": False,
        "steps": [],
        "total_reward": 0.0,
        "terminal_reason": "",
    }


def start_parallel_match(game_config, controllers: list[Controller], base_seed: int = 42) -> dict[str, Any]:
    """
    Prepare a parallel match container for multiple seats.
    Each seat gets a seed relative to the `base_seed` so environments are unique but deterministic.
    """
    seats = [
        create_match_seat(controller, game_config=game_config, seed=base_seed + index)
        for index, controller in enumerate(controllers)
    ]
    return {
        "game_config": game_config,
        "base_seed": base_seed,
        "round_number": 1,
        "seats": seats,
    }


def _record_step(seat: dict[str, Any], action: int, reward: float, done: bool, info: dict[str, Any]):
    """Record turn results for post-match statistics."""
    seat["steps"].append(
        {
            "Round": len(seat["steps"]) + 1,
            "Controller": seat["controller"].display_name,
            "Action": info["action_name"],
            "Outcome": info["result"],
            "Reward": reward,
            "Bank": info["ending_money"],
            "Terminal": info.get("terminal_reason", ""),
        }
    )
    seat["total_reward"] += reward
    seat["done"] = done
    seat["terminal_reason"] = info.get("terminal_reason", "")


def play_round(match_state: dict[str, Any], human_action: int | None = None) -> dict[str, Any]:
    """
    Advance all uncompleted seats by exactly one turn.
    For seats using a HumanController, consumes the `human_action` parameter.
    """
    for seat in match_state["seats"]:
        if seat["done"]:
            continue

        controller = seat["controller"]
        env = seat["env"]
        state = seat["state"]

        if controller.controller_type == "human":
            if human_action is None:
                continue
            action = human_action
        else:
            action = controller.choose_action(state, env)

        next_state, reward, done, info = env.step(action)
        seat["state"] = next_state
        _record_step(seat, action, reward, done, info)

    match_state["round_number"] += 1
    return match_state


def all_seats_done(match_state: dict[str, Any]) -> bool:
    """Return True if all players have hit a terminal state (bankruptcy or completed all products)."""
    return all(seat["done"] for seat in match_state["seats"])


def run_full_auto_match(match_state: dict[str, Any]) -> dict[str, Any]:
    """
    Run the match automatically until all seats complete,
    pausing if a human seat requires user input.
    """
    while not all_seats_done(match_state):
        human_seats = [
            seat for seat in match_state["seats"]
            if seat["controller"].controller_type == "human" and not seat["done"]
        ]
        if human_seats:
            break
        play_round(match_state)
    return match_state


def build_standings_dataframe(match_state: dict[str, Any]) -> pd.DataFrame:
    """
    Convert the match standings into a sorted scoreboard Pandas DataFrame.
    Sorts players by Ending Cash, then by Cumulative Reward.
    """
    import pandas as pd
    rows = []
    for seat in match_state["seats"]:
        rows.append(
            {
                "Controller": seat["controller"].display_name,
                "Type": seat["controller"].controller_type,
                "Total Reward": round(seat["total_reward"], 2),
                "Ending Money": seat["state"]["current_money"],
                "Turns Played": len(seat["steps"]),
                "Done": seat["done"],
                "Terminal": seat["terminal_reason"],
            }
        )
    standings = pd.DataFrame(rows)
    if standings.empty:
        return standings
    return standings.sort_values(
        by=["Ending Money", "Total Reward"],
        ascending=False,
    ).reset_index(drop=True)


def build_match_log_dataframe(match_state: dict[str, Any]) -> pd.DataFrame:
    """
    Combine all seat step records into a single long-form Pandas DataFrame for visualization.
    """
    import pandas as pd
    rows = []
    for seat in match_state["seats"]:
        rows.extend(seat["steps"])
    if not rows:
        return pd.DataFrame(
            columns=["Round", "Controller", "Action", "Outcome", "Reward", "Bank", "Terminal"]
        )
    return pd.DataFrame(rows)

