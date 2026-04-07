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
    checkpoint_dir = Path("artifacts/checkpoints")
    plot_dir = Path("artifacts/plots")
    report_dir = Path("artifacts/reports")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, plot_dir, report_dir


def initialize_training_log(log_path, num_actions):
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
                "average_loan_duration",
                "bankruptcy_count",
                "average_ending_money",
                "invalid_action_count",
            ]
            + [f"action_{action_id}_count" for action_id in range(num_actions)]
        )


def initialize_evaluation_log(log_path):
    """Create or overwrite the periodic evaluation CSV."""
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "average_reward",
                "bankruptcy_rate",
                "average_ending_money",
                "average_loan_duration",
                "invalid_action_rate",
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
    average_loan_duration,
    bankruptcy_count,
    average_ending_money,
    invalid_action_count,
    action_counts,
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
                average_loan_duration,
                bankruptcy_count,
                average_ending_money,
                invalid_action_count,
            ]
            + list(action_counts)
        )


def append_evaluation_log(log_path, episode, metrics):
    """Append one periodic evaluation record."""
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                episode,
                metrics["average_reward"],
                metrics["bankruptcy_rate"],
                metrics["average_ending_money"],
                metrics["average_loan_duration"],
                metrics["invalid_action_rate"],
            ]
        )


def rolling_average(values, window_size=500):
    """Compute a rolling mean for the DDQN learning curve."""
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
    epsilon_decay_episodes=450000,
):
    """
    Linearly decay epsilon very slowly across the first 450,000 episodes.

    The 8-action branch needs more exploration than the earlier binary-action setup.
    """
    if episode >= epsilon_decay_episodes:
        return epsilon_min

    progress = episode / float(epsilon_decay_episodes)
    return epsilon_start - (epsilon_start - epsilon_min) * progress


def evaluate_dqn_agent(agent, num_episodes=1000, seed=1042):
    """
    Evaluate the DDQN greedily with epsilon fixed at 0.

    Evaluation uses a separate environment instance and no optimization steps.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    env = ScrumGameEnv()
    evaluation_rewards = []
    ending_monies = []
    bankruptcy_count = 0
    loan_durations = []
    action_counts = [0] * env.num_actions
    invalid_action_count = 0

    for episode in range(num_episodes):
        state = env.reset(seed=seed + episode)
        state_vector = encode_state(state, env)
        done = False
        cumulative_reward = 0

        while not done:
            action = agent.choose_action(state_vector, epsilon=0.0)
            action_counts[action] += 1
            next_state, reward, done, info = env.step(action)
            next_state_vector = encode_state(next_state, env)
            if info.get("invalid_action"):
                invalid_action_count += 1

            cumulative_reward += reward
            state_vector = next_state_vector

            if done and info.get("terminal_reason") == "bankruptcy":
                bankruptcy_count += 1
                ending_monies.append(next_state["current_money"])
                loan_durations.append(env.turns_with_loan)
            elif done:
                ending_monies.append(next_state["current_money"])
                loan_durations.append(env.turns_with_loan)

        evaluation_rewards.append(cumulative_reward)

    return {
        "rewards": evaluation_rewards,
        "average_reward": sum(evaluation_rewards) / len(evaluation_rewards),
        "average_ending_money": sum(ending_monies) / len(ending_monies),
        "bankruptcy_rate": bankruptcy_count / len(evaluation_rewards),
        "average_loan_duration": sum(loan_durations) / len(loan_durations),
        "invalid_action_rate": invalid_action_count / max(sum(action_counts), 1),
        "action_counts": action_counts,
    }


def save_training_plot(training_rewards, output_path):
    """Save the DDQN rolling-average training curve."""
    smoothed_rewards = rolling_average(training_rewards, window_size=500)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label="Rolling Average Reward (500 episodes)", linewidth=2)
    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.title("Double DQN Training Curve")
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
    evaluation_interval=10000,
    evaluation_episodes=100,
    seed=42,
):
    """Train a Double DQN agent on the advanced Scrum Game environment."""
    random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_dir, plot_dir, report_dir = ensure_deep_rl_directories()

    env = ScrumGameEnv()
    initial_state = env.reset(seed=seed)
    state_dim = len(encode_state(initial_state, env))
    num_actions = env.num_actions

    log_path = report_dir / "logs.csv"
    evaluation_log_path = report_dir / "evaluation_history.csv"
    initialize_training_log(log_path, num_actions)
    initialize_evaluation_log(evaluation_log_path)

    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        replay_capacity=100000,
        batch_size=128,
        target_update_frequency=2000,
    )

    training_rewards = []
    training_losses = []
    episode_loan_durations = []
    bankruptcy_flags = []
    ending_monies = []
    recent_action_counts = []
    invalid_action_flags = []

    best_average_reward = float("-inf")
    best_checkpoint_path = checkpoint_dir / "best_scrum_model.pth"

    for episode in range(1, num_episodes + 1):
        state = env.reset(seed=seed + episode)
        state_vector = encode_state(state, env)
        done = False
        cumulative_reward = 0
        bankruptcy_this_episode = 0
        invalid_actions_this_episode = 0
        episode_action_counts = [0] * num_actions

        epsilon = epsilon_by_episode(episode - 1)

        while not done:
            action = agent.choose_action(state_vector, epsilon=epsilon)
            episode_action_counts[action] += 1

            next_state, reward, done, info = env.step(action)
            next_state_vector = encode_state(next_state, env)
            if info.get("invalid_action"):
                invalid_actions_this_episode += 1

            agent.store_transition(state_vector, action, reward, next_state_vector, done)
            loss = agent.train_step()

            if loss is not None:
                training_losses.append(loss)

            cumulative_reward += reward
            state_vector = next_state_vector

            if done and info.get("terminal_reason") == "bankruptcy":
                bankruptcy_this_episode = 1

        training_rewards.append(cumulative_reward)
        episode_loan_durations.append(env.turns_with_loan)
        bankruptcy_flags.append(bankruptcy_this_episode)
        ending_monies.append(env.current_money)
        recent_action_counts.append(episode_action_counts)
        invalid_action_flags.append(invalid_actions_this_episode)

        if episode % 100 == 0:
            recent_rewards = training_rewards[-100:]
            recent_losses = training_losses[-100:] if training_losses else []
            recent_loan_durations = episode_loan_durations[-100:]
            recent_bankruptcies = bankruptcy_flags[-100:]
            recent_ending_monies = ending_monies[-100:]
            recent_invalid_actions = invalid_action_flags[-100:]
            block_action_counts = [0] * num_actions
            for action_counts in recent_action_counts[-100:]:
                for action_id, count in enumerate(action_counts):
                    block_action_counts[action_id] += count

            append_training_log(
                log_path=log_path,
                episode=episode,
                epsilon=epsilon,
                episode_reward=cumulative_reward,
                rolling_average_reward=sum(recent_rewards) / len(recent_rewards),
                mean_recent_loss=(sum(recent_losses) / len(recent_losses)) if recent_losses else 0.0,
                replay_buffer_size=len(agent.replay_buffer),
                average_loan_duration=(sum(recent_loan_durations) / len(recent_loan_durations)) if recent_loan_durations else 0.0,
                bankruptcy_count=sum(recent_bankruptcies),
                average_ending_money=(sum(recent_ending_monies) / len(recent_ending_monies)) if recent_ending_monies else 0.0,
                invalid_action_count=sum(recent_invalid_actions),
                action_counts=block_action_counts,
            )

        if episode % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode:06d}.pth"
            torch.save(agent.policy_network.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at episode {episode}: {checkpoint_path}")

        if episode % evaluation_interval == 0:
            evaluation_metrics = evaluate_dqn_agent(agent, num_episodes=evaluation_episodes, seed=seed + 100000 + episode)
            append_evaluation_log(evaluation_log_path, episode, evaluation_metrics)

            if evaluation_metrics["average_reward"] > best_average_reward:
                best_average_reward = evaluation_metrics["average_reward"]
                torch.save(agent.policy_network.state_dict(), best_checkpoint_path)
                print(
                    f"Updated best model at episode {episode}: {best_checkpoint_path} "
                    f"(avg reward {best_average_reward:.2f})"
                )

    if not best_checkpoint_path.exists():
        torch.save(agent.policy_network.state_dict(), best_checkpoint_path)

    plot_path = plot_dir / "dqn_training_curve.png"
    save_training_plot(training_rewards, output_path=plot_path)

    best_agent = DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        replay_capacity=100000,
        batch_size=128,
        target_update_frequency=2000,
        device=agent.device,
    )
    best_state_dict = torch.load(best_checkpoint_path, map_location=best_agent.device)
    best_agent.policy_network.load_state_dict(best_state_dict)
    best_agent.target_network.load_state_dict(best_state_dict)
    best_agent.policy_network.eval()
    best_agent.target_network.eval()

    final_evaluation = evaluate_dqn_agent(best_agent, num_episodes=1000, seed=seed + 1000)

    save_metrics_json(
        {
            "model": "Double DQN",
            "training_episodes": num_episodes,
            "evaluation_episodes": len(final_evaluation["rewards"]),
            "average_reward_per_episode": final_evaluation["average_reward"],
            "average_ending_money": final_evaluation["average_ending_money"],
            "bankruptcy_rate": final_evaluation["bankruptcy_rate"],
            "average_loan_duration": final_evaluation["average_loan_duration"],
            "invalid_action_rate": final_evaluation["invalid_action_rate"],
            "learning_rate": learning_rate,
            "gamma": gamma,
            "state_dim": state_dim,
            "num_actions": num_actions,
            "checkpoint_path": str(best_checkpoint_path),
            "plot_path": str(plot_path),
            "log_path": str(log_path),
            "evaluation_log_path": str(evaluation_log_path),
            "final_epsilon": epsilon_by_episode(num_episodes - 1),
            "mean_training_reward": sum(training_rewards) / len(training_rewards),
            "mean_training_loss": (sum(training_losses) / len(training_losses)) if training_losses else None,
            "best_intermediate_evaluation_reward": best_average_reward,
        },
        str(report_dir / "dqn_metrics.json"),
    )

    return best_agent, training_rewards, final_evaluation, best_checkpoint_path, plot_path, log_path, evaluation_log_path


def main():
    """Train the advanced Double DQN agent with the requested production hyperparameters."""
    _, training_rewards, final_evaluation, checkpoint_path, plot_path, log_path, evaluation_log_path = train_dqn_agent()

    print(f"Training episodes completed: {len(training_rewards)}")
    print(f"Evaluation episodes completed: {len(final_evaluation['rewards'])}")
    print(f"Average reward per episode during evaluation: {final_evaluation['average_reward']:.2f}")
    print(f"Average ending money during evaluation: {final_evaluation['average_ending_money']:.2f}")
    print(f"Bankruptcy rate during evaluation: {final_evaluation['bankruptcy_rate']:.3f}")
    print(f"Saved best model checkpoint to: {checkpoint_path}")
    print(f"Saved training curve to: {plot_path}")
    print(f"Saved live training log to: {log_path}")
    print(f"Saved periodic evaluation log to: {evaluation_log_path}")


if __name__ == "__main__":
    main()
