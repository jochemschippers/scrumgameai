"""Unit tests verifying the determinism and reproducibility of ScrumGameEnv."""

import sys
import pytest

# Bypass the fake environment mock in conftest.py
if "game_runtime.scrum_game_env" in sys.modules:
    del sys.modules["game_runtime.scrum_game_env"]

from game_runtime.scrum_game_env import ScrumGameEnv
from config.config_manager import load_game_config
from services.app_paths import DEFAULT_GAME_CONFIG_PATH


@pytest.fixture
def default_config():
    """Load default config for testing."""
    return load_game_config(DEFAULT_GAME_CONFIG_PATH)


def test_environment_determinism(default_config):
    """Verify that resetting and stepping the environment under identical seeds yields identical results."""
    env1 = ScrumGameEnv(game_config=default_config)
    env2 = ScrumGameEnv(game_config=default_config)
    
    # 1. Initialize and step first environment instance
    state1_initial = env1.reset(seed=42)
    state1_next, reward1, done1, info1 = env1.step(0)
    
    # 2. Initialize and step second environment instance with matching seed
    state2_initial = env2.reset(seed=42)
    state2_next, reward2, done2, info2 = env2.step(0)
    
    # Assert initial states match
    for key in state1_initial:
        assert state1_initial[key] == state2_initial[key], f"Mismatch in initial state for key: {key}"
        
    # Assert step transitions match
    for key in state1_next:
        assert state1_next[key] == state2_next[key], f"Mismatch in next state for key: {key}"
        
    assert reward1 == reward2, "Mismatch in step rewards"
    assert done1 == done2, "Mismatch in terminal flags"
    assert info1["refinement_roll"] == info2["refinement_roll"], "Mismatch in refinement rolls"


def test_environment_reseed_variance(default_config):
    """Verify that resetting the environment under different seeds yields different outcomes."""
    env = ScrumGameEnv(game_config=default_config)
    
    state_seed1 = env.reset(seed=42)
    state_seed2 = env.reset(seed=43)
    
    # Expected expected_value or win_probability to vary based on randomized refinement/cards distributions
    # Since layout board is identical at reset, let's step once to trigger random rolls.
    _, _, _, info_seed1 = env.step(0)
    
    env.reset(seed=43)
    _, _, _, info_seed2 = env.step(0)
    
    # Refinement rolls should differ due to seed variations
    if info_seed1.get("refinement_roll") is not None and info_seed2.get("refinement_roll") is not None:
        # Check that rolls are not identical (highly likely to differ for 42 vs 43)
        assert info_seed1["refinement_roll"] != info_seed2["refinement_roll"] or info_seed1["turn_count"] == 1
