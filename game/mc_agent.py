import random

from scrum_game_env import discretize_state


class MCAgent:
    """
    Tabular Monte Carlo control agent for the Scrum Game environment.

    Unlike Q-Learning and SARSA, Monte Carlo methods do not update after every
    single transition. Instead, they wait until the full episode finishes and
    then learn from the complete sequence of rewards.
    """

    def __init__(self, alpha=0.05, gamma=0.95):
        # alpha controls how strongly newly observed returns change old estimates.
        self.alpha = alpha

        # gamma discounts future rewards when computing the return G.
        self.gamma = gamma

        # The Q-table maps each discretized state to a list of action values.
        self.q_table = {}

        # The refactored Scrum Game environment exposes two actions:
        # 0 = continue sprint, 1 = switch product.
        self.num_actions = 2

    def choose_action(self, state, epsilon):
        """
        Choose an action using epsilon-greedy exploration.

        With probability epsilon:
            sample a random action for exploration.

        Otherwise:
            exploit the action with the highest Q-value.
        """
        discrete_state = discretize_state(self._state_to_dict(state))
        self._ensure_state_exists(discrete_state)

        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        q_values = self.q_table[discrete_state]
        return max(range(self.num_actions), key=lambda action: q_values[action])

    def learn(self, episode_history):
        """
        Update the Q-table from a full episode using a First-Visit Monte Carlo rule.

        episode_history is expected to be a list of:
            (state, action, reward)

        We iterate backward so we can build the discounted return:
            G = reward_t + gamma * G

        First-Visit MC means each state-action pair is updated only the first time
        it appears in the episode when viewed from the start of the episode.
        Because we iterate backward, that corresponds to updating only the last
        occurrence seen in the reversed pass.
        """
        discounted_return = 0.0
        visited_state_actions = set()

        for state, action, reward in reversed(episode_history):
            discounted_return = reward + self.gamma * discounted_return

            discrete_state = discretize_state(self._state_to_dict(state))
            self._ensure_state_exists(discrete_state)

            state_action_key = (discrete_state, action)

            # In a reversed pass, skipping duplicates ensures only one update per
            # state-action pair for the full episode, which is a First-Visit MC style.
            if state_action_key in visited_state_actions:
                continue

            visited_state_actions.add(state_action_key)

            current_q = self.q_table[discrete_state][action]

            # The update moves the Q-value estimate toward the observed full return G.
            self.q_table[discrete_state][action] = current_q + self.alpha * (
                discounted_return - current_q
            )

    def _ensure_state_exists(self, discrete_state):
        """Initialize unseen states with zero action values."""
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = [0.0] * self.num_actions

    def _state_to_dict(self, state):
        """
        Normalize states into dictionary form before discretization.

        This allows the agent to work with either tuple states returned by the
        environment or raw dictionaries prepared elsewhere in the pipeline.
        """
        if isinstance(state, dict):
            return state

        return {
            "current_money": state[0],
            "current_product": state[1],
            "current_sprint": state[2],
            "features_required": state[3],
            "sprint_value": state[4],
            "loan_active": state[5],
            "interest_due": state[6],
        }
