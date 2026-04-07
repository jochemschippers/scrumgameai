import statistics

from model_utils import save_metrics_csv, save_metrics_json, save_text_report
from train_dqn import evaluate_dqn_agent, train_dqn_agent


def evaluate_across_seeds(
    seeds,
    train_episodes=500000,
    evaluation_episodes=1000,
):
    """Train and evaluate the advanced DDQN model across multiple fixed seeds."""
    results = []

    for seed in seeds:
        agent, _, _, _, _, _, _ = train_dqn_agent(
            num_episodes=train_episodes,
            seed=seed,
        )
        evaluation = evaluate_dqn_agent(agent, num_episodes=evaluation_episodes, seed=seed + 1000)

        results.append(
            {
                "seed": seed,
                "average_reward": evaluation["average_reward"],
                "average_ending_money": evaluation["average_ending_money"],
                "bankruptcy_rate": evaluation["bankruptcy_rate"],
                "average_loan_duration": evaluation["average_loan_duration"],
                "invalid_action_rate": evaluation["invalid_action_rate"],
                "best_episode_reward": max(evaluation["rewards"]),
                "worst_episode_reward": min(evaluation["rewards"]),
            }
        )

    return results


def summarize_results(results):
    """Compute aggregate statistics for the robustness report."""
    average_rewards = [row["average_reward"] for row in results]
    ending_monies = [row["average_ending_money"] for row in results]
    bankruptcy_rates = [row["bankruptcy_rate"] for row in results]
    loan_durations = [row["average_loan_duration"] for row in results]
    invalid_rates = [row["invalid_action_rate"] for row in results]

    return {
        "seeds": [row["seed"] for row in results],
        "mean_average_reward": statistics.mean(average_rewards),
        "std_average_reward": statistics.stdev(average_rewards) if len(average_rewards) > 1 else 0.0,
        "best_case_average_reward": max(average_rewards),
        "worst_case_average_reward": min(average_rewards),
        "mean_average_ending_money": statistics.mean(ending_monies),
        "mean_bankruptcy_rate": statistics.mean(bankruptcy_rates),
        "mean_average_loan_duration": statistics.mean(loan_durations),
        "mean_invalid_action_rate": statistics.mean(invalid_rates),
    }


def build_report_text(results, summary):
    """Create a Markdown robustness summary."""
    lines = [
        "# DDQN Robustness Evaluation",
        "",
        "## Protocol",
        f"- Seeds: {summary['seeds']}",
        "- Each seed trains a fresh advanced DDQN model.",
        "- Evaluation is performed after training with a separate fixed evaluation seed offset.",
        "",
        "## Aggregate Results",
        f"- Mean average reward: {summary['mean_average_reward']:.2f}",
        f"- Standard deviation: {summary['std_average_reward']:.2f}",
        f"- Best-case average reward: {summary['best_case_average_reward']:.2f}",
        f"- Worst-case average reward: {summary['worst_case_average_reward']:.2f}",
        f"- Mean ending money: {summary['mean_average_ending_money']:.2f}",
        f"- Mean bankruptcy rate: {summary['mean_bankruptcy_rate']:.3f}",
        f"- Mean loan duration: {summary['mean_average_loan_duration']:.2f}",
        f"- Mean invalid action rate: {summary['mean_invalid_action_rate']:.4f}",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Avg Reward | Best | Worst | Avg Ending Money | Bankruptcy Rate | Loan Duration | Invalid Action Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        lines.append(
            f"| {row['seed']} | {row['average_reward']:.2f} | {row['best_episode_reward']:.2f} | "
            f"{row['worst_episode_reward']:.2f} | {row['average_ending_money']:.2f} | "
            f"{row['bankruptcy_rate']:.3f} | {row['average_loan_duration']:.2f} | {row['invalid_action_rate']:.4f} |"
        )

    return "\n".join(lines)


def print_summary(summary):
    """Print a compact terminal summary."""
    print("DDQN Robustness Evaluation")
    print("-------------------------")
    print(f"Seeds: {summary['seeds']}")
    print(f"Mean Average Reward: {summary['mean_average_reward']:.2f}")
    print(f"Std Average Reward: {summary['std_average_reward']:.2f}")
    print(f"Best-Case Average Reward: {summary['best_case_average_reward']:.2f}")
    print(f"Worst-Case Average Reward: {summary['worst_case_average_reward']:.2f}")
    print(f"Mean Ending Money: {summary['mean_average_ending_money']:.2f}")
    print(f"Mean Bankruptcy Rate: {summary['mean_bankruptcy_rate']:.3f}")
    print(f"Mean Loan Duration: {summary['mean_average_loan_duration']:.2f}")
    print(f"Mean Invalid Action Rate: {summary['mean_invalid_action_rate']:.4f}")


def main():
    """Run the five-seed DDQN robustness protocol and save report artifacts."""
    seeds = [42, 123, 999, 2026, 31415]
    results = evaluate_across_seeds(seeds=seeds)
    summary = summarize_results(results)
    report_text = build_report_text(results, summary)

    save_metrics_csv(results, "artifacts/reports/ddqn_robustness.csv")
    save_metrics_json(
        {
            "summary": summary,
            "per_seed_results": results,
        },
        "artifacts/reports/ddqn_robustness.json",
    )
    save_text_report(report_text, "artifacts/reports/ddqn_robustness.md")

    print_summary(summary)
    print("Saved robustness CSV to: artifacts/reports/ddqn_robustness.csv")
    print("Saved robustness JSON to: artifacts/reports/ddqn_robustness.json")
    print("Saved robustness Markdown to: artifacts/reports/ddqn_robustness.md")


if __name__ == "__main__":
    main()
