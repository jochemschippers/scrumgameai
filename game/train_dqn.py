import csv
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from dqn_agent import DQNAgent, encode_state
from model_utils import save_metrics_json
from scrum_game_env import ScrumGameEnv


def ensure_deep_rl_directories():
    """Create the deep-RL artifact folders used by training."""
    checkpoint_dir = Path("artifacts/deep_rl/checkpoints")
    plot_dir = Path("artifacts/deep_rl/plots")
    report_dir = Path("artifacts/deep_rl/reports")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, plot_dir, report_dir


def initialize_training_log(log_path):
    """Create or overwrite the live DQN training log CSV."""
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "epsilon",
                "episode_reward",
                "rolling_average_reward",
                "mean_recent_loss",
                "replay_buffer_size",
            ]
        )


def append_training_log(
    log_path,
    episode,
    epsilon,
    episode_reward,
    rolling_average_reward,
    mean_recent_loss,
    replay_buffer_size,
):
    """Append one summarized training record for the dashboard."""
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                episode,
                epsilon,
                episode_reward,
                rolling_average_reward,
                mean_recent_loss,
                replay_buffer_size,
            ]
        )


def rolling_average(values, window_size=500):
    """Compute a rolling mean for the DQN learning curve."""
    if not values:
        return []

    smoothed = []
    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        window = values[start_index : index + 1]
        smoothed.append(sum(window) / len(window))
    return smoothed


def epsilon_by_episode(
    episode,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay_episodes=400000,
):
    """
    Linearly decay epsilon very slowly across the first 400,000 episodes.

    This gives the DQN enough time to explore the fixed classical board before
    settling into a more greedy policy.
    """
    if episode >= epsilon_decay_episodes:
        return epsilon_min

    progress = episode / float(epsilon_decay_episodes)
    return epsilon_start - (epsilon_start - epsilon_min) * progress


def evaluate_dqn_agent(agent, num_episodes=1000, seed=1042):
    """
    Evaluate the DQN greedily with epsilon fixed at 0.

    Evaluation uses a separate environment instance and no optimization steps.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    env = ScrumGameEnv()
    evaluation_rewards = []

    for episode in range(num_episodes):
        state = env.reset(seed=seed + episode)
        state_vector = encode_state(state, env)
        done = False
        cumulative_reward = 0

        while not done:
            action = agent.choose_action(state_vector, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            next_state_vector = encode_state(next_state, env)

            cumulative_reward += reward
            state_vector = next_state_vector

        evaluation_rewards.append(cumulative_reward)

    average_reward = sum(evaluation_rewards) / len(evaluation_rewards)
    return evaluation_rewards, average_reward


def save_training_plot(training_rewards, output_path):
    """Save the DQN rolling-average training curve."""
    smoothed_rewards = rolling_average(training_rewards, window_size=500)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label="Rolling Average Reward (500 episodes)", linewidth=2)
    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_dqn_agent(
    num_episodes=500000,
    learning_rate=0.0005,
    gamma=0.85,
    checkpoint_interval=10000,
    seed=42,
):
    """Train a DQN agent on the Scrum Game environment."""
    random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_dir, plot_dir, report_dir = ensure_deep_rl_directories()
    log_path = report_dir / "logs.csv"
    initialize_training_log(log_path)

    env = ScrumGameEnv()
    agent = DQNAgent(
        state_dim=8,
        num_actions=2,
        learning_rate=learning_rate,
        gamma=gamma,
        replay_capacity=100000,
        batch_size=64,
        target_update_frequency=1000,
    )

    training_rewards = []
    training_losses = []
    checkpoint_path = checkpoint_dir / "best_scrum_model.pth"

    for episode in range(1, num_episodes + 1):
        state = env.reset(seed=seed + episode)
        state_vector = encode_state(state, env)
        done = False
        cumulative_reward = 0

        epsilon = epsilon_by_episode(episode - 1)

        while not done:
            action = agent.choose_action(state_vector, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)
            next_state_vector = encode_state(next_state, env)

            agent.store_transition(state_vector, action, reward, next_state_vector, done)
            loss = agent.train_step()

            if loss is not None:
                training_losses.append(loss)

            cumulative_reward += reward
            state_vector = next_state_vector

        training_rewards.append(cumulative_reward)

        if episode % 100 == 0:
            recent_rewards = training_rewards[-100:]
            recent_losses = training_losses[-100:] if training_losses else []
            append_training_log(
                log_path=log_path,
                episode=episode,
                epsilon=epsilon,
                episode_reward=cumulative_reward,
                rolling_average_reward=sum(recent_rewards) / len(recent_rewards),
                mean_recent_loss=(sum(recent_losses) / len(recent_losses)) if recent_losses else 0.0,
                replay_buffer_size=len(agent.replay_buffer),
            )

        if episode % checkpoint_interval == 0:
            torch.save(agent.policy_network.state_dict(), checkpoint_path)
            print(
                f"Checkpoint saved at episode {episode}: {checkpoint_path}"
            )

    # Save one final copy even if the total episode count changes later.
    torch.save(agent.policy_network.state_dict(), checkpoint_path)

    plot_path = plot_dir / "dqn_training_curve.png"
    save_training_plot(training_rewards, output_path=plot_path)

    evaluation_rewards, average_reward = evaluate_dqn_agent(agent, num_episodes=1000, seed=seed + 1000)

    save_metrics_json(
        {
            "model": "DQN",
            "training_episodes": num_episodes,
            "evaluation_episodes": len(evaluation_rewards),
            "average_reward_per_episode": average_reward,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "checkpoint_path": str(checkpoint_path),
            "plot_path": str(plot_path),
            "log_path": str(log_path),
            "final_epsilon": epsilon_by_episode(num_episodes - 1),
            "mean_training_reward": sum(training_rewards) / len(training_rewards),
            "mean_training_loss": (sum(training_losses) / len(training_losses)) if training_losses else None,
        },
        str(report_dir / "dqn_metrics.json"),
    )

    return agent, training_rewards, evaluation_rewards, average_reward, checkpoint_path, plot_path, log_path


def main():
    """Train the DQN agent with the requested production hyperparameters."""
    agent, training_rewards, evaluation_rewards, average_reward, checkpoint_path, plot_path, log_path = train_dqn_agent()

    print(f"Training episodes completed: {len(training_rewards)}")
    print(f"Evaluation episodes completed: {len(evaluation_rewards)}")
    print(f"Average reward per episode during evaluation: {average_reward:.2f}")
    print(f"Saved model checkpoint to: {checkpoint_path}")
    print(f"Saved training curve to: {plot_path}")
    print(f"Saved live training log to: {log_path}")


if __name__ == "__main__":
    main()
