import argparse
from datetime import datetime
from pathlib import Path
import statistics

from model_utils import save_metrics_csv, save_metrics_json, save_text_report
from train_dqn import evaluate_dqn_agent, train_dqn_agent


BASE_DIR = Path(__file__).resolve().parent
ROBUSTNESS_DIR = BASE_DIR / "artifacts" / "robustness"


def create_robustness_output_dir():
    """Create a unique output folder for one robustness batch."""
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = datetime.now().strftime("robustness_%Y-%m-%d_%H%M")
    candidate = ROBUSTNESS_DIR / base_name
    suffix = 1
    while candidate.exists():
        candidate = ROBUSTNESS_DIR / f"{base_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


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


def build_report_text(results, summary, train_episodes, evaluation_episodes):
    """Create a Markdown robustness summary."""
    lines = [
        "# DDQN Robustness Evaluation",
        "",
        "## Protocol",
        f"- Seeds: {summary['seeds']}",
        f"- Training episodes per seed: {train_episodes}",
        f"- Evaluation episodes per seed: {evaluation_episodes}",
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


def parse_args():
    """Parse CLI options for robustness evaluation."""
    parser = argparse.ArgumentParser(description="Run the 5-seed DDQN robustness evaluation.")
    parser.add_argument("--episodes", type=int, default=500000, help="Training episodes per seed.")
    parser.add_argument("--evaluation-episodes", type=int, default=1000, help="Evaluation episodes per seed.")
    return parser.parse_args()


def main():
    """Run the five-seed DDQN robustness protocol and save report artifacts."""
    args = parse_args()
    seeds = [42, 123, 999, 2026, 31415]
    results = evaluate_across_seeds(
        seeds=seeds,
        train_episodes=args.episodes,
        evaluation_episodes=args.evaluation_episodes,
    )
    summary = summarize_results(results)
    report_text = build_report_text(results, summary, args.episodes, args.evaluation_episodes)
    output_dir = create_robustness_output_dir()
    csv_path = output_dir / "ddqn_robustness.csv"
    json_path = output_dir / "ddqn_robustness.json"
    report_path = output_dir / "ddqn_robustness.md"

    save_metrics_csv(results, str(csv_path))
    save_metrics_json(
        {
            "summary": summary,
            "per_seed_results": results,
        },
        str(json_path),
    )
    save_text_report(report_text, str(report_path))

    print_summary(summary)
    print(f"Saved robustness CSV to: {csv_path}")
    print(f"Saved robustness JSON to: {json_path}")
    print(f"Saved robustness Markdown to: {report_path}")


if __name__ == "__main__":
    main()
