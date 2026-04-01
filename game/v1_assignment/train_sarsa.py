import json
from pathlib import Path
import matplotlib.pyplot as plt

from model_utils import save_metrics_json, save_q_table
from sarsa_agent import SarsaAgent
from scrum_game_env import ScrumGameEnv


def rolling_average(values, window_size=100):
    """
    Compute a rolling average so the training curve is easier to interpret.

    Reinforcement learning rewards are noisy, so smoothing helps reveal the
    long-term learning trend instead of focusing on one lucky or unlucky episode.
    """
    if not values:
        return []

    smoothed_values = []

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        window = values[start_index : index + 1]
        smoothed_values.append(sum(window) / len(window))

    return smoothed_values


def train_sarsa_agent(
    num_episodes=100000,
    alpha=0.05,
    gamma=0.85,
    epsilon_start=1.0,
    epsilon_decay=0.99993,
    epsilon_min=0.05,
):
    """
    Train a SARSA agent on the Scrum Game environment.

    SARSA is on-policy, so the next action must be selected before the
    learning update can be applied.
    """
    env = ScrumGameEnv()
    agent = SarsaAgent(alpha=alpha, gamma=gamma)

    epsilon = epsilon_start
    training_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        cumulative_reward = 0

        # SARSA needs the first action before entering the transition loop
        # because each update uses the action actually selected in the next state.
        action = agent.choose_action(state, epsilon)

        while not done:
            next_state, reward, done, info = env.step(action)
            cumulative_reward += reward

            if done:
                # Terminal states do not have a valid next action to continue with,
                # so we pass None and let the agent treat the target as reward only.
                agent.learn(state, action, reward, next_state, next_action=None, done=True)
            else:
                next_action = agent.choose_action(next_state, epsilon)
                agent.learn(state, action, reward, next_state, next_action, done=False)
                state = next_state
                action = next_action

        training_rewards.append(cumulative_reward)

        # Epsilon decays gradually so training shifts from exploration toward exploitation.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return agent, training_rewards


def evaluate_sarsa_agent(agent, num_episodes=1000):
    """
    Evaluate the trained SARSA agent with strict evaluation isolation.

    Critical evaluation rules:
    - epsilon is fixed at 0, so the policy is purely greedy
    - learn() is never called
    """
    env = ScrumGameEnv()
    evaluation_rewards = []
    epsilon = 0.0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        cumulative_reward = 0

        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            cumulative_reward += reward
            state = next_state

        evaluation_rewards.append(cumulative_reward)

    average_reward = sum(evaluation_rewards) / len(evaluation_rewards)
    return evaluation_rewards, average_reward


def save_training_plot(training_rewards, output_path="artifacts/plots/sarsa_training_curve.png", window_size=100):
    """Plot and save the rolling average reward curve for the report."""
    smoothed_rewards = rolling_average(training_rewards, window_size=window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label=f"Rolling Average Reward ({window_size} episodes)", linewidth=2)
    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.title("SARSA Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Train the SARSA agent, evaluate it, and save the learning curve."""
    agent, training_rewards = train_sarsa_agent()
    evaluation_rewards, average_reward = evaluate_sarsa_agent(agent)

    plot_path = "artifacts/plots/sarsa_training_curve.png"
    q_table_path = save_q_table("sarsa", agent.q_table)
    save_training_plot(training_rewards, output_path=plot_path)

    # JSON object keys must be strings, so the tuple state keys are serialized.
    serializable_q_table = {
        str(state): q_values for state, q_values in agent.q_table.items()
    }
    final_model_path = Path("artifacts/models/final_sarsa_model.json")
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    with final_model_path.open("w", encoding="utf-8") as file:
        json.dump(serializable_q_table, file, indent=2)

    save_metrics_json(
        {
            "model": "SARSA",
            "training_episodes": len(training_rewards),
            "evaluation_episodes": len(evaluation_rewards),
            "average_reward_per_episode": average_reward,
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "plot_path": plot_path,
            "q_table_path": q_table_path,
            "final_model_path": str(final_model_path),
        },
        "artifacts/reports/sarsa_metrics.json",
    )

    print(f"Training episodes completed: {len(training_rewards)}")
    print(f"Evaluation episodes completed: {len(evaluation_rewards)}")
    print(f"Average reward per episode during evaluation: {average_reward:.2f}")
    print(f"Saved training curve to: {plot_path}")
    print(f"Saved Q-table artifact to: {q_table_path}")
    print(f"Saved final SARSA model to: {final_model_path}")


if __name__ == "__main__":
    main()
