from scrum_game_env import ScrumGameEnv


class BaselineAgent:
    """Simple heuristic baseline agent with no learning logic."""

    def __init__(self):
        """The baseline agent uses a fixed non-learning policy."""

    def act(self, state):
        """
        Select an action using a fixed heuristic policy.

        Policy:
        - Always continue the current sprint.

        Loans are now handled automatically by the environment when required,
        so the baseline no longer chooses loan-related actions.
        """
        return 0


def evaluate_baseline_agent(num_episodes=1000):
    """
    Run a fixed-policy evaluation loop and return episode rewards.

    This is a true baseline evaluation: the agent never updates a policy,
    never stores values, and never learns from experience.
    """
    env = ScrumGameEnv()
    agent = BaselineAgent()
    episode_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        cumulative_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            cumulative_reward += reward
            state = next_state

        episode_rewards.append(cumulative_reward)

    average_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Average reward per episode over {num_episodes} episodes: {average_reward:.2f}")
    return episode_rewards, average_reward


if __name__ == "__main__":
    evaluate_baseline_agent()
