"""
Domain Randomization Engine for the Scrum Game.

This module provides functions to randomly vary game config parameters (costs, starting
money, loan values, dice counts, incident frequencies) within configurable bounds.

Why Domain Randomization?
  - In deep reinforcement learning, training an agent on a single, fixed environment
    can lead to overfitting (e.g. the agent only learns how to play with exactly $25k starting money).
  - Randomizing these parameters forces the agent to learn robust policies that adapt
    to varying financial pressures, costs, and game lengths.

Key Constraints:
  - We do NOT randomize the number of products, number of sprints, or layout shapes.
    Doing so would change the dimensions of the observation space and action space,
    which would crash the PyTorch neural network.

Connections:
  - Imported by: `training.train_dqn` (if `rule_randomization_enabled` is set in TrainingConfig,
    it calls `sample_game_config` periodically to randomize the environment rules)
  - Imported by: `evaluation.evaluate_ddqn_robustness` (samples multiple test scenarios to test model robustness)
"""

from __future__ import annotations

import random
from typing import Any

from config.config_manager import GameConfig

# Default min/max boundaries for each randomized parameter.
# These ranges ensure the game remains playable (e.g., max turns is between 4 and 10).
DEFAULT_RULE_RANDOMIZATION_BOUNDS: dict[str, Any] = {
    "starting_money": [15000, 45000],
    "max_turns": [4, 10],
    "cost_continue": [0, 2000],
    "cost_switch_mid": [0, 10000],
    "cost_switch_after": [0, 5000],
    "mandatory_loan_amount": [30000, 80000],
    "loan_interest": [1000, 10000],
    "penalty_negative": [500, 2500],
    "penalty_positive": [0, 2500],
    "incident_draw_probability": [0.2, 1.0],
    "incident_severity_multiplier": [0.5, 2.0],
    "dice_sides": [4, 20],
    "dice_count": [1, 4],
}


def _range(bounds: dict[str, Any], key: str) -> tuple[float, float]:
    """Helper to extract a low/high float range for a parameter, enforcing order."""
    value = bounds.get(key, DEFAULT_RULE_RANDOMIZATION_BOUNDS[key])
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Rule-randomization bound `{key}` must be a two-item range.")
    low = float(value[0])
    high = float(value[1])
    if low > high:
        low, high = high, low
    return low, high


def _rand_int(rng: random.Random, bounds: dict[str, Any], key: str) -> int:
    """Sample an integer uniformly within bounds[key]."""
    low, high = _range(bounds, key)
    return int(rng.randint(int(round(low)), int(round(high))))


def _rand_float(rng: random.Random, bounds: dict[str, Any], key: str) -> float:
    """Sample a float uniformly within bounds[key]."""
    low, high = _range(bounds, key)
    return float(rng.uniform(low, high))


def merged_rule_randomization_bounds(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Merge default bounds with user-defined overrides.
    
    Args:
        overrides: Dictionary of key -> [min, max] overrides.
        
    Returns:
        dict: Merged bounds configuration.
    """
    result = dict(DEFAULT_RULE_RANDOMIZATION_BOUNDS)
    if overrides:
        result.update(overrides)
    return result


def sample_game_config(
    base_config: GameConfig,
    rng: random.Random,
    bounds: dict[str, Any] | None = None,
    *,
    config_name: str | None = None,
) -> GameConfig:
    """
    Generate a new GameConfig template with randomized economy and rule variables.
    
    This is used to perform Domain Randomization in RL training.
    
    Key invariant:
      Does not alter:
        - `product_names` (and count of products)
        - `board_ring_values` shape
        - `board_features` shape
        - `dice_rules` count (the rule matching thresholds stay constant, only sides/counts vary)
      This maintains static shapes for Gymnasium observations and DQN action indices.
      
    Args:
        base_config: The template config containing fixed layout and lists.
        rng: Python Random instance (to support seeded evaluation runs).
        bounds: Randomization min/max range limits.
        config_name: Custom name prefix for the randomized config.
        
    Returns:
        GameConfig: A newly constructed, fully validated GameConfig instance.
    """
    resolved_bounds = merged_rule_randomization_bounds(bounds)
    payload = base_config.to_dict()

    payload["config_name"] = config_name or f"{payload.get('config_name', 'Config')} Randomized"
    
    # 1. Randomize basic finances
    payload["starting_money"] = _rand_int(rng, resolved_bounds, "starting_money")
    payload["max_turns"] = _rand_int(rng, resolved_bounds, "max_turns")
    
    # 2. Randomize action switch/continue costs
    payload["cost_continue"] = _rand_int(rng, resolved_bounds, "cost_continue")
    payload["cost_switch_mid"] = _rand_int(rng, resolved_bounds, "cost_switch_mid")
    payload["cost_switch_after"] = _rand_int(rng, resolved_bounds, "cost_switch_after")
    
    # 3. Randomize debt/loan variables
    payload["mandatory_loan_amount"] = _rand_int(rng, resolved_bounds, "mandatory_loan_amount")
    payload["loan_interest"] = _rand_int(rng, resolved_bounds, "loan_interest")
    payload["penalty_negative"] = _rand_int(rng, resolved_bounds, "penalty_negative")
    payload["penalty_positive"] = _rand_int(rng, resolved_bounds, "penalty_positive")

    # 4. Randomize incident frequency and severity multiplier
    incident = payload.setdefault("incident", {})
    incident["draw_probability"] = round(_rand_float(rng, resolved_bounds, "incident_draw_probability"), 3)
    incident["severity_multiplier"] = round(_rand_float(rng, resolved_bounds, "incident_severity_multiplier"), 3)

    # 5. Randomize dice counts and sides for the various features-based roll regimes
    dice_sides_low, dice_sides_high = _range(resolved_bounds, "dice_sides")
    dice_count_low, dice_count_high = _range(resolved_bounds, "dice_count")
    for rule in payload.get("dice_rules", []):
        rule["dice_sides"] = int(rng.randint(int(round(dice_sides_low)), int(round(dice_sides_high))))
        rule["dice_count"] = int(rng.randint(int(round(dice_count_low)), int(round(dice_count_high))))

    # Construction validates the config using `validate_game_config()`
    return GameConfig.from_dict(payload)

