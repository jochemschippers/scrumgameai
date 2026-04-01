import random
import statistics

from baseline_agent import evaluate_baseline_agent
from train_mc import evaluate_mc_agent, train_mc_agent
from train_q_learning import evaluate_q_learning_agent, train_q_learning_agent
from train_sarsa import evaluate_sarsa_agent, train_sarsa_agent


def evaluate_model_across_seeds(
    model_name,
    seeds,
    train_fn=None,
    eval_fn=None,
    train_episodes=25000,
    eval_episodes=1000,
):
    """
    Train and evaluate one model across multiple random seeds.

    For learning agents, training and evaluation are seeded separately so the
    evaluation does not simply continue from the exact random stream used
    during training.
    """
    seed_results = []

    for seed in seeds:
        if train_fn is None:
            # The baseline does not learn, so it only needs evaluation episodes.
            random.seed(seed)
            _, average_reward = evaluate_baseline_agent(num_episodes=eval_episodes)
        else:
            random.seed(seed)
            agent, _ = train_fn(num_episodes=train_episodes)

            # Use a deterministic but different evaluation seed per run.
            random.seed(seed + 1000)
            _, average_reward = eval_fn(agent, num_episodes=eval_episodes)

        seed_results.append(average_reward)

    mean_average_reward = statistics.mean(seed_results)
    std_average_reward = statistics.stdev(seed_results) if len(seed_results) > 1 else 0.0

    return {
        "model": model_name,
        "seed_rewards": seed_results,
        "mean_average_reward": mean_average_reward,
        "std_average_reward": std_average_reward,
    }


def print_results_table(results):
    """Print a clean table of robustness results."""
    header = (
        f"{'Model':<15}"
        f"{'Mean Avg Reward':>18}"
        f"{'Std Dev':>14}"
        f"{'Seeds':>10}"
    )
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    for result in sorted(results, key=lambda item: item["mean_average_reward"], reverse=True):
        print(
            f"{result['model']:<15}"
            f"{result['mean_average_reward']:>18.2f}"
            f"{result['std_average_reward']:>14.2f}"
            f"{len(result['seed_rewards']):>10}"
        )

    print(divider)


def main():
    """Run the robustness evaluation across five fixed random seeds."""
    seeds = [42, 43, 44, 45, 46]
    train_episodes = 25000
    eval_episodes = 1000

    results = []

    results.append(
        evaluate_model_across_seeds(
            model_name="Baseline",
            seeds=seeds,
            train_fn=None,
            eval_fn=None,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
        )
    )

    results.append(
        evaluate_model_across_seeds(
            model_name="Q-Learning",
            seeds=seeds,
            train_fn=train_q_learning_agent,
            eval_fn=evaluate_q_learning_agent,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
        )
    )

    results.append(
        evaluate_model_across_seeds(
            model_name="SARSA",
            seeds=seeds,
            train_fn=train_sarsa_agent,
            eval_fn=evaluate_sarsa_agent,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
        )
    )

    results.append(
        evaluate_model_across_seeds(
            model_name="Monte Carlo",
            seeds=seeds,
            train_fn=train_mc_agent,
            eval_fn=evaluate_mc_agent,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
        )
    )

    print("Robustness evaluation across seeds:", seeds)
    print_results_table(results)

    print("\nPer-seed average rewards:")
    for result in results:
        rounded_rewards = ", ".join(f"{reward:.2f}" for reward in result["seed_rewards"])
        print(f"{result['model']}: [{rounded_rewards}]")


if __name__ == "__main__":
    main()
