import random
import statistics

import torch

from deployment_profiles import PROFILE_CONFIGS
from train_dqn import evaluate_dqn_agent, train_dqn_agent


def evaluate_across_seeds(
    seeds,
    train_episodes=500000,
    evaluation_episodes=1000,
):
    """Train and evaluate the advanced DDQN model across multiple seeds."""
    results = []

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)

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
            }
        )

    return results


def print_summary(results):
    """Print a compact robustness summary table."""
    average_rewards = [row["average_reward"] for row in results]
    ending_monies = [row["average_ending_money"] for row in results]
    bankruptcy_rates = [row["bankruptcy_rate"] for row in results]
    loan_durations = [row["average_loan_duration"] for row in results]

    print("DDQN Robustness Evaluation")
    print("-------------------------")
    print(f"Seeds: {[row['seed'] for row in results]}")
    print(f"Mean Average Reward: {statistics.mean(average_rewards):.2f}")
    print(f"Std Average Reward: {statistics.stdev(average_rewards) if len(average_rewards) > 1 else 0.0:.2f}")
    print(f"Mean Ending Money: {statistics.mean(ending_monies):.2f}")
    print(f"Mean Bankruptcy Rate: {statistics.mean(bankruptcy_rates):.3f}")
    print(f"Mean Loan Duration: {statistics.mean(loan_durations):.2f}")
    print("")
    print("Deployment Profiles available:")
    for profile_name in PROFILE_CONFIGS:
        print(f"- {profile_name}")


def main():
    """Run the five-seed DDQN robustness protocol."""
    seeds = [42, 43, 44, 45, 46]
    results = evaluate_across_seeds(seeds=seeds)
    print_summary(results)


if __name__ == "__main__":
    main()
