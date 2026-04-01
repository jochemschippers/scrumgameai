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


def choose_profile_action(agent, state_vector, profile_name):
    """Choose an action according to one of the deployment profiles."""
    profile_key = profile_name.lower()
    if profile_key not in PROFILE_CONFIGS:
        raise ValueError(f"Unknown profile: {profile_name}")

    profile = PROFILE_CONFIGS[profile_key]
    epsilon = profile["epsilon"]
    temperature = profile["temperature"]

    if random.random() < epsilon:
        return random.randint(0, agent.num_actions - 1)

    return agent.choose_action_with_temperature(state_vector, temperature=temperature)
