import random

from scrum_game_env import discretize_state


class QLearningAgent:
    """
    Tabular Q-Learning agent for the Scrum Game environment.

    The Q-table is a dictionary:
        key   -> discretized state tuple
        value -> list of Q-values, one for each discrete action
    """

    def __init__(self, alpha=0.1, gamma=0.95):
        # alpha controls how aggressively new experience overrides old estimates.
        self.alpha = alpha

        # gamma controls how much the agent values future reward versus immediate reward.
        self.gamma = gamma

        # The Q-table starts empty and grows as the agent visits new states.
        self.q_table = {}

        # The refactored Scrum Game environment exposes two actions:
        # 0 = continue sprint, 1 = switch product.
        self.num_actions = 2

    def choose_action(self, state, epsilon):
        """
        Choose an action using an epsilon-greedy exploration strategy.

        With probability epsilon:
            take a random action to explore the environment.

        With probability (1 - epsilon):
            take the action with the highest estimated Q-value.
        """
        discrete_state = discretize_state(self._state_to_dict(state))
        self._ensure_state_exists(discrete_state)

        # Exploration is required during training so the agent can discover
        # better actions instead of getting stuck with early estimates.
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        # Exploitation means selecting the action with the highest learned value.
        # max() with a key function returns the index of the best Q-value.
        q_values = self.q_table[discrete_state]
        return max(range(self.num_actions), key=lambda action: q_values[action])

    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the standard Q-learning Bellman equation.

        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]

        For terminal states:
            there is no future reward term, so the target becomes just `reward`.
        """
        discrete_state = discretize_state(self._state_to_dict(state))
        discrete_next_state = discretize_state(self._state_to_dict(next_state))

        self._ensure_state_exists(discrete_state)
        self._ensure_state_exists(discrete_next_state)

        current_q = self.q_table[discrete_state][action]

        if done:
            # Once an episode ends, there is no next action to evaluate.
            target = reward
        else:
            # Q-learning is off-policy: it uses the best possible next action value,
            # not necessarily the action the current policy would sample next.
            best_next_q = max(self.q_table[discrete_next_state])
            target = reward + self.gamma * best_next_q

        # The temporal-difference error measures how far the old estimate is
        # from the new Bellman target, and alpha controls the update size.
        td_error = target - current_q
        self.q_table[discrete_state][action] = current_q + self.alpha * td_error

    def _ensure_state_exists(self, discrete_state):
        """Create a zero-initialized action-value list for unseen states."""
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = [0.0] * self.num_actions

    def _state_to_dict(self, state):
        """
        Normalize a state into dictionary form before discretization.

        This makes the agent compatible with either:
        - raw dictionaries, or
        - tuple states returned directly by the environment.
        """
        if isinstance(state, dict):
            return {
                "current_money": state["current_money"],
                "current_product": state["current_product"],
                "current_sprint": state["current_sprint"],
                "features_required": state["features_required"],
                "sprint_value": state["sprint_value"],
                "loan_active": state["loan_active"],
                "interest_due": state["interest_due"],
                "win_probability": state.get("win_probability", 0.0),
            }

        return {
            "current_money": state[0],
            "current_product": state[1],
            "current_sprint": state[2],
            "features_required": state[3],
            "sprint_value": state[4],
            "loan_active": state[5],
            "interest_due": state[6],
            "win_probability": state[7] if len(state) > 7 else 0.0,
        }
