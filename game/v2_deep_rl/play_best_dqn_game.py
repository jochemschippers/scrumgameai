from pathlib import Path

import torch

from deployment_profiles import choose_profile_action
from dqn_agent import DQNAgent, encode_state
from scrum_game_env import ScrumGameEnv


def load_dqn_checkpoint(model_path="artifacts/checkpoints/best_scrum_model.pth"):
    """Load the trained DQN checkpoint used for the final demo."""
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    env = ScrumGameEnv()
    agent = DQNAgent(
        state_dim=len(encode_state(env.reset(seed=42), env)),
        num_actions=env.num_actions,
        learning_rate=0.0005,
        gamma=0.85,
    )
    try:
        state_dict = torch.load(checkpoint_path, map_location=agent.device)
        agent.policy_network.load_state_dict(state_dict)
        agent.target_network.load_state_dict(state_dict)
    except RuntimeError as error:
        raise RuntimeError(
            "The selected checkpoint is incompatible with the advanced 8-action Double DQN model. "
            "Train a new checkpoint with `py train_dqn.py` in `game/v2_deep_rl`."
        ) from error
    agent.policy_network.eval()
    agent.target_network.eval()
    return agent


def product_name(product_id):
    """Convert a numeric product id to a readable label."""
    return f"Product {product_id}"


def action_name(action):
    """Convert the action id to a readable demo label."""
    if action == 0:
        return "Continue"
    return f"Switch to Product {action}"


def result_name(info):
    """Return a readable turn outcome label."""
    if "success" not in info:
        return "Switch"
    return "Success" if info["success"] else "Failure"


def play_demo_game(model_path="artifacts/checkpoints/best_scrum_model.pth", seed=42, profile_name="expert"):
    """Play one full game using one of the deployment profiles."""
    env = ScrumGameEnv()
    agent = load_dqn_checkpoint(model_path=model_path)

    state = env.reset(seed=seed)
    state_vector = encode_state(state, env)
    done = False
    turn_number = 1

    print("Final DQN Demo Game")
    print(f"Model: {model_path}")
    print(f"Profile: {profile_name}")
    print("")

    while not done:
        current_product = state["current_product"]
        action = choose_profile_action(agent, state_vector, profile_name=profile_name)
        next_state, reward, done, info = env.step(action)

        print(
            f"Turn {turn_number}: {product_name(current_product)} - "
            f"Action: {action_name(action)}"
        )
        print(
            f"Dice Results: {result_name(info)} - "
            f"Current Bank: {next_state['current_money']}"
        )
        print(
            f"Win Probability: {state['win_probability']:.3f} - Reward: {reward}"
        )
        print(f"Expected Value: {state['expected_value']:.2f}")

        if "net_result" in info:
            print(f"Net Scrum Result: {info['net_result']}")

        if info.get("loan_triggered"):
            print(
                f"Mandatory Loan Triggered: Yes - "
                f"Interest Due Next Turn: {next_state['interest_due']}"
            )

        if done and "terminal_reason" in info:
            print(f"Game Over: {info['terminal_reason']}")

        print("")

        state = next_state
        state_vector = encode_state(state, env)
        turn_number += 1


if __name__ == "__main__":
    play_demo_game()
