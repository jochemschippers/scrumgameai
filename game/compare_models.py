import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from baseline_agent import evaluate_baseline_agent
from dqn_agent import DQNAgent, encode_state
from model_utils import save_metrics_csv, save_metrics_json, save_q_table, save_text_report
from scrum_game_env import ScrumGameEnv
from train_mc import evaluate_mc_agent, train_mc_agent
from train_q_learning import evaluate_q_learning_agent, train_q_learning_agent
from train_sarsa import evaluate_sarsa_agent, train_sarsa_agent


def summarize_rewards(model_name, rewards, model_type, q_table_size=None, artifact_path=None, plot_path=None):
    """Create a compact metrics dictionary for one model."""
    average_reward = sum(rewards) / len(rewards)

    return {
        "model": model_name,
        "model_type": model_type,
        "average_reward_per_episode": round(average_reward, 2),
        "reward_std_dev": round(statistics.pstdev(rewards), 2),
        "best_episode_reward": round(max(rewards), 2),
        "worst_episode_reward": round(min(rewards), 2),
        "episodes_evaluated": len(rewards),
        "q_table_size": q_table_size if q_table_size is not None else "N/A",
        "artifact_path": artifact_path or "N/A",
        "plot_path": plot_path or "N/A",
    }


def save_comparison_plot(rows, output_path="artifacts/plots/model_comparison.png"):
    """Save a simple bar chart comparing average evaluation reward by model."""
    model_names = [row["model"] for row in rows]
    average_rewards = [row["average_reward_per_episode"] for row in rows]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, average_rewards)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Average Reward Per Episode")
    plt.title("Model Comparison on Evaluation Episodes")
    plt.tight_layout()

    for bar, value in zip(bars, average_rewards):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )

    plt.savefig(output_path)
    plt.close()
    return output_path


def build_report_summary(rows):
    """Create a report-ready Markdown summary of the comparison results."""
    sorted_rows = sorted(rows, key=lambda row: row["average_reward_per_episode"], reverse=True)
    best_model = sorted_rows[0]
    dqn_row = next((row for row in rows if row["model"] == "DQN"), None)

    lines = [
        "# Model Comparison Summary",
        "",
        "## Evaluation Protocol",
        "- Training and evaluation were separated into different episode phases.",
        "- During evaluation, epsilon was fixed at 0.0 for pure exploitation.",
        "- During evaluation, no learning updates were allowed.",
        "- The baseline agent used a fixed heuristic and did not learn.",
        "",
        "## Results Table",
        "",
        "| Model | Avg Reward | Std Dev | Best | Worst | Q-Table Size |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in sorted_rows:
        lines.append(
            f"| {row['model']} | {row['average_reward_per_episode']:.2f} | "
            f"{row['reward_std_dev']:.2f} | {row['best_episode_reward']:.2f} | "
            f"{row['worst_episode_reward']:.2f} | {row['q_table_size']} |"
        )

    lines.extend(
        [
            "",
            "## Hyperparameter Summary",
            "- Q-Learning: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.",
            "- SARSA: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.",
            "- Monte Carlo: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.",
            "- DQN: learning_rate=0.0005, gamma=0.85, replay_buffer=100,000, epsilon decayed slowly over 400,000 of 500,000 training episodes.",
            "- Baseline: no learning, fixed heuristic policy.",
            "",
            "## Model Choice Justification",
            "- Q-Learning was included as a standard off-policy temporal-difference baseline.",
            "- SARSA was included because its on-policy updates can produce safer behavior in stochastic environments.",
            "- Monte Carlo was included to compare full-episode return learning against temporal-difference methods.",
            "- DQN was included to test whether a neural network with replay memory could learn a stronger policy than the tabular methods.",
            "- The heuristic baseline was included to show whether learned behavior outperformed a simple rule-based policy.",
            "",
            "## Robustness And Limitations",
            "- The Scrum Game environment is stochastic because sprint success depends on random outcomes, so reward variance is expected.",
            "- Large reward standard deviations indicate that policy performance is sensitive to chance and should be discussed as a robustness limitation.",
            "- If a training curve improves and then degrades late in training, that suggests instability or partial overfitting to recent stochastic experiences.",
            "- A practical mitigation is repeated training over multiple random seeds and reporting the mean and spread across runs.",
            "",
            "## Bias Discussion",
            "- The environment encodes design assumptions such as loan penalties, sprint values, and failure penalties.",
            "- These assumptions can bias which strategies appear optimal, so results should be interpreted as valid for this game design rather than all Scrum settings.",
            "- If environment reward settings change, the ranking of agents may also change.",
            "",
            "## Current Best Model",
            f"- Based on the latest comparison run, the best average evaluation reward was achieved by {best_model['model']}.",
        ]
    )

    if dqn_row is not None:
        lines.extend(
            [
                "",
                "## Selected Deployment Model",
                "- The final selected deployment model for the project is DQN.",
                f"- The saved deployment artifact is `{dqn_row['artifact_path']}`.",
                "- DQN was selected because it is the final deep-RL production model with dashboard support, checkpointing, and a dedicated demo runner.",
            ]
        )

    return "\n".join(lines)


def load_dqn_checkpoint(checkpoint_path="artifacts/deep_rl/checkpoints/best_scrum_model.pth"):
    """Load the trained DQN checkpoint if it exists."""
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        return None

    agent = DQNAgent(
        state_dim=8,
        num_actions=2,
        learning_rate=0.0005,
        gamma=0.85,
    )
    state_dict = torch.load(checkpoint_file, map_location=agent.device)
    agent.policy_network.load_state_dict(state_dict)
    agent.target_network.load_state_dict(state_dict)
    agent.policy_network.eval()
    agent.target_network.eval()
    return agent


def evaluate_dqn_checkpoint(agent, num_episodes=1000, seed=42):
    """Evaluate a loaded DQN checkpoint greedily."""
    random.seed(seed)
    torch.manual_seed(seed)

    env = ScrumGameEnv()
    rewards = []

    for episode in range(num_episodes):
        state = env.reset(seed=seed + episode)
        state_vector = encode_state(state, env)
        done = False
        cumulative_reward = 0

        while not done:
            action = agent.choose_action(state_vector, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            state_vector = encode_state(next_state, env)
            cumulative_reward += reward

        rewards.append(cumulative_reward)

    average_reward = sum(rewards) / len(rewards)
    return rewards, average_reward


def main():
    """Train and compare the baseline, tabular agents, and the saved DQN model."""
    results = []

    random.seed(42)
    baseline_rewards, baseline_average = evaluate_baseline_agent(num_episodes=1000)
    results.append(
        summarize_rewards(
            "Baseline",
            baseline_rewards,
            model_type="Heuristic",
        )
    )

    random.seed(42)
    q_agent, q_training_rewards = train_q_learning_agent()
    q_eval_rewards, q_average = evaluate_q_learning_agent(q_agent, num_episodes=1000)
    q_plot_path = "artifacts/plots/q_learning_training_curve.png"
    q_artifact_path = save_q_table("q_learning", q_agent.q_table)
    from train_q_learning import save_training_plot

    save_training_plot(q_training_rewards, output_path=q_plot_path)
    results.append(
        summarize_rewards(
            "Q-Learning",
            q_eval_rewards,
            model_type="TD Off-Policy",
            q_table_size=len(q_agent.q_table),
            artifact_path=q_artifact_path,
            plot_path=q_plot_path,
        )
    )

    random.seed(42)
    sarsa_agent, sarsa_training_rewards = train_sarsa_agent()
    sarsa_eval_rewards, sarsa_average = evaluate_sarsa_agent(sarsa_agent, num_episodes=1000)
    sarsa_plot_path = "artifacts/plots/sarsa_training_curve.png"
    sarsa_artifact_path = save_q_table("sarsa", sarsa_agent.q_table)
    from train_sarsa import save_training_plot as save_sarsa_plot

    save_sarsa_plot(sarsa_training_rewards, output_path=sarsa_plot_path)
    results.append(
        summarize_rewards(
            "SARSA",
            sarsa_eval_rewards,
            model_type="TD On-Policy",
            q_table_size=len(sarsa_agent.q_table),
            artifact_path=sarsa_artifact_path,
            plot_path=sarsa_plot_path,
        )
    )

    random.seed(42)
    mc_agent, mc_training_rewards = train_mc_agent()
    mc_eval_rewards, mc_average = evaluate_mc_agent(mc_agent, num_episodes=1000)
    mc_plot_path = "artifacts/plots/mc_training_curve.png"
    mc_artifact_path = save_q_table("mc", mc_agent.q_table)
    from train_mc import save_training_plot as save_mc_plot

    save_mc_plot(mc_training_rewards, output_path=mc_plot_path)
    results.append(
        summarize_rewards(
            "Monte Carlo",
            mc_eval_rewards,
            model_type="Full-Episode",
            q_table_size=len(mc_agent.q_table),
            artifact_path=mc_artifact_path,
            plot_path=mc_plot_path,
        )
    )

    dqn_checkpoint_path = "artifacts/deep_rl/checkpoints/best_scrum_model.pth"
    dqn_plot_path = "artifacts/deep_rl/plots/dqn_training_curve.png"
    dqn_agent = load_dqn_checkpoint(dqn_checkpoint_path)
    dqn_average = None
    if dqn_agent is not None:
        dqn_eval_rewards, dqn_average = evaluate_dqn_checkpoint(dqn_agent, num_episodes=1000, seed=42)
        results.append(
            summarize_rewards(
                "DQN",
                dqn_eval_rewards,
                model_type="Deep RL",
                q_table_size="N/A",
                artifact_path=dqn_checkpoint_path,
                plot_path=dqn_plot_path,
            )
        )

    comparison_plot_path = save_comparison_plot(results)
    report_summary = build_report_summary(results)

    save_metrics_csv(results, "artifacts/reports/model_comparison.csv")
    save_metrics_json(
        {
            "baseline_average_reward": baseline_average,
            "q_learning_average_reward": q_average,
            "sarsa_average_reward": sarsa_average,
            "mc_average_reward": mc_average,
            "dqn_average_reward": dqn_average,
            "comparison_plot_path": comparison_plot_path,
            "results": results,
        },
        "artifacts/reports/model_comparison.json",
    )
    save_text_report(report_summary, "artifacts/reports/model_comparison_summary.md")

    print("Model comparison complete.")
    for row in sorted(results, key=lambda item: item["average_reward_per_episode"], reverse=True):
        print(f"{row['model']}: {row['average_reward_per_episode']:.2f}")
    print(f"Saved comparison plot to: {comparison_plot_path}")
    print("Saved results to: artifacts/reports/model_comparison.csv")
    print("Saved report summary to: artifacts/reports/model_comparison_summary.md")


if __name__ == "__main__":
    main()
