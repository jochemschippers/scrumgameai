"""
Deployment Profiles for Scrum Game Simulation.

This module defines various behavioral profiles (beginner, balanced, expert)
to configure how a trained DQN model makes choices during simulations and multiplayer play.

Profiles:
  - beginner: High exploration (epsilon = 0.15) and high temperature (1.6) softmax action selection.
    Produces highly randomized and sub-optimal plays, representing a learning player.
  - balanced: Low exploration (epsilon = 0.05) and standard temperature (1.0) softmax selection.
    Produces decent but slightly unpredictable choices.
  - expert: No exploration (epsilon = 0.0) and temperature = 0.0 (greedy action selection).
    Always selects the absolute best action according to predicted Q-values.

Action Masking:
  - Both helper functions allow passing a `valid_actions` mask to restrict the agent
    to legal options (e.g. preventing switching to already completed products).

Connections:
  - Called by: `play.match_runner` and `play.shared_match_runner` when running game matches.
  - Used in: FastAPI backend play endpoints to run bots of different difficulty levels.
"""

import math
import random


PROFILE_CONFIGS = {
    "beginner": {
        "epsilon": 0.15,
        "temperature": 1.6,
    },
    "balanced": {
        "epsilon": 0.05,
        "temperature": 1.0,
    },
    "expert": {
        "epsilon": 0.0,
        "temperature": 0.0,
    },
}


def _available_actions(agent, valid_actions: list[int] | None = None) -> list[int]:
    """Helper to return list of allowed action indices, defaulting to all actions if mask is None."""
    if valid_actions is None:
        return list(range(agent.num_actions))
    actions = [int(action) for action in valid_actions]
    return actions or list(range(agent.num_actions))


def choose_profile_action(agent, state_vector: list[float], profile_name: str, valid_actions: list[int] | None = None) -> int:
    """
    Select an action using the hyperparameters of the specified profile.
    
    Args:
        agent (DQNAgent): The trained model wrapper.
        state_vector (list[float]): Normalized input features.
        profile_name (str): 'beginner', 'balanced', or 'expert' difficulty.
        valid_actions: List of allowed action indices.
        
    Returns:
        int: Selected action index.
    """
    profile_key = profile_name.lower()
    if profile_key not in PROFILE_CONFIGS:
        raise ValueError(f"Unknown profile: {profile_name}")

    profile = PROFILE_CONFIGS[profile_key]
    epsilon = profile["epsilon"]
    temperature = profile["temperature"]
    actions = _available_actions(agent, valid_actions)

    # 1. Epsilon-Greedy Random Choice: check exploration chance
    if random.random() < epsilon:
        return random.choice(actions)

    # 2. Predict action values
    q_values = agent.predict_q_values(state_vector)

    # 3. Greedy Selection: pick absolute maximum Q-value action
    if temperature <= 0:
        return max(actions, key=lambda action: q_values[action])

    # 4. Softmax Boltzmann Selection: scale Q-values by temperature and sample
    scaled = [q_values[action] / temperature for action in actions]
    max_scaled = max(scaled)
    weights = [math.exp(value - max_scaled) for value in scaled]
    return random.choices(actions, weights=weights, k=1)[0]

