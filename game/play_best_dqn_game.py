from pathlib import Path

import torch

from dqn_agent import DQNAgent, encode_state
from scrum_game_env import ScrumGameEnv


def load_dqn_checkpoint(model_path="artifacts/deep_rl/checkpoints/best_scrum_model.pth"):
    """Load the trained DQN checkpoint used for the final demo."""
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    agent = DQNAgent(
        state_dim=8,
        num_actions=2,
        learning_rate=0.0005,
        gamma=0.85,
    )
    state_dict = torch.load(checkpoint_path, map_location=agent.device)
    agent.policy_network.load_state_dict(state_dict)
    agent.target_network.load_state_dict(state_dict)
    agent.policy_network.eval()
    agent.target_network.eval()
    return agent


def product_name(product_id):
    """Convert a numeric product id to a readable label."""
    return f"Product {product_id}"


def action_name(action):
    """Convert the action id to a readable demo label."""
    return "Continue" if action == 0 else "Switch"


def result_name(info):
    """Return a readable turn outcome label."""
    if "success" not in info:
        return "Switch"
    return "Success" if info["success"] else "Failure"


def play_demo_game(model_path="artifacts/deep_rl/checkpoints/best_scrum_model.pth", seed=42):
    """Play one full greedy game using the final DQN checkpoint."""
    env = ScrumGameEnv()
    agent = load_dqn_checkpoint(model_path=model_path)

    state = env.reset(seed=seed)
    state_vector = encode_state(state, env)
    done = False
    turn_number = 1

    print("Final DQN Demo Game")
    print(f"Model: {model_path}")
    print("Epsilon: 0.0")
    print("")

    while not done:
        current_product = state[1]
        action = agent.choose_action(state_vector, epsilon=0.0)
        next_state, reward, done, info = env.step(action)

        print(
            f"Turn {turn_number}: {product_name(current_product)} - "
            f"Action: {action_name(action)}"
        )
        print(
            f"Dice Results: {result_name(info)} - "
            f"Current Bank: {next_state[0]}"
        )
        print(
            f"Win Probability: {state[7]:.3f} - Reward: {reward}"
        )

        if "net_result" in info:
            print(f"Net Scrum Result: {info['net_result']}")

        if info.get("loan_triggered"):
            print(
                f"Mandatory Loan Triggered: Yes - "
                f"Interest Due Next Turn: {next_state[6]}"
            )

        if done and "terminal_reason" in info:
            print(f"Game Over: {info['terminal_reason']}")

        print("")

        state = next_state
        state_vector = encode_state(state, env)
        turn_number += 1


if __name__ == "__main__":
    play_demo_game()
