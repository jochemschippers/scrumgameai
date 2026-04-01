import random

from scrum_game_env import discretize_state


class SarsaAgent:
    """
    Tabular SARSA agent for the Scrum Game environment.

    SARSA is an on-policy method, which means it updates Q-values using the
    action the current policy actually chooses in the next state.
    """

    def __init__(self, alpha=0.05, gamma=0.95):
        # alpha determines how quickly new experience changes old Q-value estimates.
        self.alpha = alpha

        # gamma determines how strongly the agent values future rewards.
        self.gamma = gamma

        # The Q-table maps each discretized state to one Q-value per action.
        self.q_table = {}

        # The refactored Scrum Game environment exposes two actions:
        # 0 = continue sprint, 1 = switch product.
        self.num_actions = 2

    def choose_action(self, state, epsilon):
        """
        Choose an action using an epsilon-greedy policy.

        With probability epsilon, the agent explores by selecting a random action.
        Otherwise, it exploits the best Q-value currently known for the state.
        """
        discrete_state = discretize_state(self._state_to_dict(state))
        self._ensure_state_exists(discrete_state)

        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        q_values = self.q_table[discrete_state]
        return max(range(self.num_actions), key=lambda action: q_values[action])

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Update the Q-table using the SARSA Bellman update rule.

        SARSA update:
        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]

        The key difference from Q-learning is that SARSA uses the actual next
        action selected by the current policy, making it an on-policy method.

        For terminal states:
        - there is no future action value to bootstrap from
        - the target becomes just the immediate reward
        """
        discrete_state = discretize_state(self._state_to_dict(state))
        self._ensure_state_exists(discrete_state)

        current_q = self.q_table[discrete_state][action]

        if done:
            # If the episode has ended, there is no next state-action pair to use.
            target = reward
        else:
            discrete_next_state = discretize_state(self._state_to_dict(next_state))
            self._ensure_state_exists(discrete_next_state)

            # SARSA is on-policy, so it bootstraps from the Q-value of the
            # actual next action chosen under the current epsilon-greedy policy.
            next_q = self.q_table[discrete_next_state][next_action]
            target = reward + self.gamma * next_q

        # Temporal-difference learning updates the old estimate toward the target.
        td_error = target - current_q
        self.q_table[discrete_state][action] = current_q + self.alpha * td_error

    def _ensure_state_exists(self, discrete_state):
        """Initialize unseen states with zero Q-values for all actions."""
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = [0.0] * self.num_actions

    def _state_to_dict(self, state):
        """
        Normalize either tuple states or dictionary states into a dictionary.

        This keeps the agent flexible while still using the shared
        discretize_state() helper from the environment module.
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
