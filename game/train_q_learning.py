import matplotlib.pyplot as plt

from model_utils import save_metrics_json, save_q_table
from q_learning_agent import QLearningAgent
from scrum_game_env import ScrumGameEnv


def rolling_average(values, window_size=100):
    """
    Compute a simple rolling average for smoother reward visualization.

    A rolling average is useful in RL because single-episode rewards can vary
    heavily due to exploration and environment randomness.
    """
    if not values:
        return []

    smoothed_values = []

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        window = values[start_index : index + 1]
        smoothed_values.append(sum(window) / len(window))

    return smoothed_values


def train_q_learning_agent(
    num_episodes=25000,
    alpha=0.05,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_decay=0.9997,
    epsilon_min=0.05,
):
    """
    Train a Q-Learning agent on the Scrum Game environment.

    During this phase, exploration is enabled and learning updates are allowed.
    """
    env = ScrumGameEnv()
    agent = QLearningAgent(alpha=alpha, gamma=gamma)

    epsilon = epsilon_start
    training_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        cumulative_reward = 0

        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            cumulative_reward += reward
            state = next_state

        training_rewards.append(cumulative_reward)

        # Epsilon decays gradually so the agent explores early on and then
        # becomes increasingly greedy as its Q-values become more reliable.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return agent, training_rewards


def evaluate_q_learning_agent(agent, num_episodes=1000):
    """
    Evaluate the trained agent with strict isolation from learning.

    Critical evaluation rules:
    - epsilon is fixed at 0, so the agent acts greedily.
    - learn() is never called, so no policy updates happen during evaluation.
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

            # No learning is allowed during evaluation episodes.
            cumulative_reward += reward
            state = next_state

        evaluation_rewards.append(cumulative_reward)

    average_reward = sum(evaluation_rewards) / len(evaluation_rewards)
    return evaluation_rewards, average_reward


def save_training_plot(training_rewards, output_path="artifacts/plots/q_learning_training_curve.png", window_size=100):
    """Plot and save the rolling average learning curve for the report."""
    smoothed_rewards = rolling_average(training_rewards, window_size=window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label=f"Rolling Average Reward ({window_size} episodes)", linewidth=2)
    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.title("Q-Learning Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Train the Q-Learning agent, evaluate it, and save the learning curve."""
    agent, training_rewards = train_q_learning_agent()
    evaluation_rewards, average_reward = evaluate_q_learning_agent(agent)

    plot_path = "artifacts/plots/q_learning_training_curve.png"
    q_table_path = save_q_table("q_learning", agent.q_table)
    save_training_plot(training_rewards, output_path=plot_path)
    save_metrics_json(
        {
            "model": "Q-Learning",
            "training_episodes": len(training_rewards),
            "evaluation_episodes": len(evaluation_rewards),
            "average_reward_per_episode": average_reward,
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "plot_path": plot_path,
            "q_table_path": q_table_path,
        },
        "artifacts/reports/q_learning_metrics.json",
    )

    print(f"Training episodes completed: {len(training_rewards)}")
    print(f"Evaluation episodes completed: {len(evaluation_rewards)}")
    print(f"Average reward per episode during evaluation: {average_reward:.2f}")
    print(f"Saved training curve to: {plot_path}")
    print(f"Saved Q-table artifact to: {q_table_path}")


if __name__ == "__main__":
    main()
