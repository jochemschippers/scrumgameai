import random

from train_sarsa import evaluate_sarsa_agent, train_sarsa_agent


def tune_sarsa_discount_factor(
    gamma_values=None,
    alpha=0.05,
    train_episodes=25000,
    eval_episodes=1000,
    seed=42,
):
    """
    Evaluate several SARSA discount factors under a fixed training setup.

    A fixed random seed is used for each gamma value so the comparison is as
    fair as possible and each run starts from the same random conditions.
    """
    if gamma_values is None:
        gamma_values = [0.85, 0.90, 0.95, 0.99]

    results = []

    for gamma in gamma_values:
        random.seed(seed)
        agent, _ = train_sarsa_agent(
            num_episodes=train_episodes,
            alpha=alpha,
            gamma=gamma,
        )

        # Evaluation uses a different but deterministic seed so that evaluation
        # is repeatable while remaining isolated from the training random stream.
        random.seed(seed + 1000)
        _, average_reward = evaluate_sarsa_agent(agent, num_episodes=eval_episodes)

        results.append(
            {
                "gamma": gamma,
                "average_reward": average_reward,
            }
        )

    return results


def print_results(results):
    """Print the tuning results in a clean table."""
    header = f"{'Gamma':<10}{'Average Reward':>20}"
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    for result in sorted(results, key=lambda item: item["average_reward"], reverse=True):
        print(f"{result['gamma']:<10.2f}{result['average_reward']:>20.2f}")

    print(divider)


def main():
    """Run a small grid search over SARSA gamma values."""
    gamma_values = [0.85, 0.90, 0.95, 0.99]
    results = tune_sarsa_discount_factor(gamma_values=gamma_values)

    print("SARSA gamma tuning")
    print("Fixed settings: alpha=0.05, training episodes=25000, evaluation episodes=1000, seed=42")
    print_results(results)


if __name__ == "__main__":
    main()
