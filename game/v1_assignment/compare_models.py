import random
import statistics

import matplotlib.pyplot as plt

from baseline_agent import evaluate_baseline_agent
from model_utils import save_metrics_csv, save_metrics_json, save_q_table, save_text_report
from train_mc import evaluate_mc_agent, save_training_plot as save_mc_plot, train_mc_agent
from train_q_learning import (
    evaluate_q_learning_agent,
    save_training_plot as save_q_plot,
    train_q_learning_agent,
)
from train_sarsa import evaluate_sarsa_agent, save_training_plot as save_sarsa_plot, train_sarsa_agent


def summarize_rewards(model_name, rewards, model_type, q_table_size=None, artifact_path=None, plot_path=None):
    """Create one compact metrics row for the assignment comparison table."""
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
    """Save a bar chart comparing the assignment-track models."""
    model_names = [row["model"] for row in rows]
    average_rewards = [row["average_reward_per_episode"] for row in rows]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, average_rewards)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Average Reward Per Episode")
    plt.title("Assignment Track Model Comparison")
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
    """Create a report-ready Markdown summary for the assignment deliverable."""
    sorted_rows = sorted(rows, key=lambda row: row["average_reward_per_episode"], reverse=True)
    best_model = sorted_rows[0]

    lines = [
        "# Assignment Track Model Comparison",
        "",
        "## Scope",
        "- This comparison belongs to `game/v1_assignment` and keeps the assignment-safe tabular work separate from the advanced DQN experiments.",
        "- The compared policies are Baseline, Q-Learning, SARSA, and Monte Carlo.",
        "- The advanced DQN work lives separately in `../v2_deep_rl` and is intentionally excluded from this table.",
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
            "- SARSA: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes for this fair comparison run.",
            "- Monte Carlo: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.",
            "- Baseline: no learning, fixed always-continue heuristic.",
            "",
            "## Model Choice Justification",
            "- Q-Learning is the standard off-policy temporal-difference baseline.",
            "- SARSA is included because on-policy updates can produce safer behavior in stochastic environments.",
            "- Monte Carlo is included to contrast full-episode return learning with temporal-difference updates.",
            "- The heuristic baseline shows whether learned behavior is better than a simple hand-written policy.",
            "",
            "## Notes",
            f"- In this comparison run, the strongest assignment-track model was {best_model['model']}.",
            "- If you also want to discuss the advanced DQN extension, treat it as a separate `v2_deep_rl` experiment rather than part of the assignment-safe table.",
        ]
    )

    return "\n".join(lines)


def print_results_table(rows):
    """Print a compact terminal table sorted by average reward."""
    header = (
        f"{'Model':<15}"
        f"{'Avg Reward':>15}"
        f"{'Std Dev':>12}"
        f"{'Best':>12}"
        f"{'Worst':>12}"
    )
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    for row in sorted(rows, key=lambda item: item["average_reward_per_episode"], reverse=True):
        print(
            f"{row['model']:<15}"
            f"{row['average_reward_per_episode']:>15.2f}"
            f"{row['reward_std_dev']:>12.2f}"
            f"{row['best_episode_reward']:>12.2f}"
            f"{row['worst_episode_reward']:>12.2f}"
        )

    print(divider)


def main():
    """Train and compare the assignment-safe baseline and tabular RL models."""
    results = []
    train_episodes = 25000
    eval_episodes = 1000

    random.seed(42)
    baseline_rewards, _ = evaluate_baseline_agent(num_episodes=eval_episodes)
    results.append(
        summarize_rewards(
            "Baseline",
            baseline_rewards,
            model_type="Heuristic",
        )
    )

    random.seed(42)
    q_agent, q_training_rewards = train_q_learning_agent(num_episodes=train_episodes)
    q_eval_rewards, _ = evaluate_q_learning_agent(q_agent, num_episodes=eval_episodes)
    q_plot_path = "artifacts/plots/q_learning_training_curve.png"
    q_artifact_path = save_q_table("q_learning", q_agent.q_table)
    save_q_plot(q_training_rewards, output_path=q_plot_path)
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
    sarsa_agent, sarsa_training_rewards = train_sarsa_agent(
        num_episodes=train_episodes,
        alpha=0.05,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_decay=0.9997,
        epsilon_min=0.05,
    )
    sarsa_eval_rewards, _ = evaluate_sarsa_agent(sarsa_agent, num_episodes=eval_episodes)
    sarsa_plot_path = "artifacts/plots/sarsa_training_curve.png"
    sarsa_artifact_path = save_q_table("sarsa", sarsa_agent.q_table)
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
    mc_agent, mc_training_rewards = train_mc_agent(num_episodes=train_episodes)
    mc_eval_rewards, _ = evaluate_mc_agent(mc_agent, num_episodes=eval_episodes)
    mc_plot_path = "artifacts/plots/mc_training_curve.png"
    mc_artifact_path = save_q_table("mc", mc_agent.q_table)
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

    plot_path = save_comparison_plot(results)
    report_text = build_report_summary(results)

    save_metrics_csv(results, "artifacts/reports/model_comparison.csv")
    save_metrics_json({"results": results}, "artifacts/reports/model_comparison.json")
    save_text_report(report_text, "artifacts/reports/model_comparison_summary.md")

    print_results_table(results)
    print(f"Saved comparison plot to: {plot_path}")
    print("Saved comparison table to: artifacts/reports/model_comparison.csv")
    print("Saved comparison summary to: artifacts/reports/model_comparison_summary.md")


if __name__ == "__main__":
    main()
