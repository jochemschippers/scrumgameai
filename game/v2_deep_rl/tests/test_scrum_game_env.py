"""Unit tests for the ScrumGameEnv environment simulator."""

import sys
from pathlib import Path
import pytest

# Bypass the fake environment mock in conftest.py
if "game_runtime.scrum_game_env" in sys.modules:
    del sys.modules["game_runtime.scrum_game_env"]

from game_runtime.scrum_game_env import ScrumGameEnv
from config.config_manager import load_game_config
from services.app_paths import DEFAULT_GAME_CONFIG_PATH


@pytest.fixture
def default_config():
    """Load the default game configuration for testing."""
    return load_game_config(DEFAULT_GAME_CONFIG_PATH)


def test_env_initialization(default_config):
    """Verify the environment initializes correctly from the GameConfig properties."""
    env = ScrumGameEnv(game_config=default_config)
    assert env.starting_money == default_config.starting_money
    assert env.max_turns == default_config.max_turns
    assert env.products_count == len(default_config.product_names)
    assert env.sprints_per_product == len(default_config.board_ring_values[0])


def test_env_reset(default_config):
    """Verify that resetting the environment yields the correct starting observations."""
    env = ScrumGameEnv(game_config=default_config)
    state = env.reset(seed=42)
    
    assert isinstance(state, dict)
    assert "current_money" in state
    assert "current_product" in state
    assert "current_sprint" in state
    assert "expected_value" in state
    assert "win_probability" in state
    assert "remaining_turns" in state
    assert "target_next_sprints" in state
    
    assert state["current_money"] == env.starting_money
    assert state["current_product"] == 1
    assert state["current_sprint"] == 1
    assert state["remaining_turns"] == env.max_turns


def test_env_step_transition(default_config):
    """Verify that taking a valid step advances the environment turn and changes state."""
    env = ScrumGameEnv(game_config=default_config)
    env.reset(seed=100)
    
    initial_money = env.current_money
    action = 0  # Continue on product 1
    
    next_state, reward, done, info = env.step(action)
    
    assert env.turn_count == 1
    assert next_state["remaining_turns"] == env.max_turns - 1
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert info["action"] == action


def test_env_invalid_action_handling(default_config):
    """Verify that invalid actions (like switching to the current product) are flagged and penalized."""
    env = ScrumGameEnv(game_config=default_config)
    env.reset(seed=100)
    
    # Action 1 corresponds to switching to Product 1, which is already the active product.
    next_state, reward, done, info = env.step(1)
    
    assert info["invalid_action"] is True
    assert info["invalid_action_reason"] == "self_switch"
    assert reward < 0  # Should be penalized for invalid actions


def test_env_bankruptcy_terminal(default_config):
    """Verify that dropping below €0 immediately triggers a bankruptcy terminal state."""
    env = ScrumGameEnv(game_config=default_config)
    env.reset(seed=100)
    
    # Artificially force money below zero so low that no sprint payout can rescue it
    env.current_money = -1_000_000
    
    # Step the environment
    next_state, reward, done, info = env.step(0)
    
    assert done is True
    assert info.get("terminal_reason") == "bankruptcy"


def test_env_max_turns_terminal(default_config):
    """Verify that exceeding the turn count limit triggers a max_turns_reached terminal state."""
    env = ScrumGameEnv(game_config=default_config)
    env.reset(seed=100)
    
    # Force turn count to max limit
    env.turn_count = env.max_turns
    
    next_state, reward, done, info = env.step(0)
    
    assert done is True
    assert info.get("terminal_reason") == "max_turns_reached"
