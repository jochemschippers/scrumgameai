import ast
import json

from scrum_game_env import ScrumGameEnv, discretize_state


class FinalSarsaPolicy:
    """Greedy policy wrapper around a saved SARSA Q-table."""

    def __init__(self, model_path="final_sarsa_model.json"):
        self.q_table = self._load_q_table(model_path)
        self.num_actions = 2

    def choose_action(self, state, epsilon=0.0):
        """
        Choose the greedy action for the current state.

        epsilon is accepted for API clarity, but this demo uses epsilon = 0.
        """
        discrete_state = discretize_state(state)
        self._ensure_state_exists(discrete_state)
        q_values = self.q_table[discrete_state]
        return max(range(self.num_actions), key=lambda action: q_values[action])

    def _ensure_state_exists(self, discrete_state):
        """Fallback to zero values if the loaded table does not contain a state."""
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = [0.0] * self.num_actions

    def _load_q_table(self, model_path):
        """Load the JSON Q-table and convert stringified tuple keys back to tuples."""
        with open(model_path, "r", encoding="utf-8") as file:
            raw_q_table = json.load(file)

        loaded_q_table = {}
        for state_key, q_values in raw_q_table.items():
            loaded_q_table[ast.literal_eval(state_key)] = q_values

        return loaded_q_table


def product_name(product_id):
    """Convert a product ID to a readable demo label."""
    return f"Product {product_id}"


def action_name(action):
    """Convert the action ID to a readable action label."""
    return "Continue" if action == 0 else "Switch"


def result_name(info):
    """Map the environment info to a simple success/failure description."""
    if "success" not in info:
        return "Switch"
    return "Success" if info["success"] else "Failure"


def play_demo_game(model_path="final_sarsa_model.json", seed=42):
    """Play one full greedy demo game using the saved final SARSA model."""
    env = ScrumGameEnv()
    policy = FinalSarsaPolicy(model_path=model_path)

    state = env.reset(seed=seed)
    done = False
    turn_number = 1
    epsilon = 0.0

    print("Final SARSA Demo Game")
    print(f"Model: {model_path}")
    print(f"Epsilon: {epsilon}")
    print("")

    while not done:
        current_product = state[1]
        action = policy.choose_action(state, epsilon=epsilon)

        next_state, reward, done, info = env.step(action)

        print(
            f"Turn {turn_number}: {product_name(current_product)} - "
            f"Action: {action_name(action)}"
        )
        print(
            f"Dice Results: {result_name(info)} - "
            f"Current Bank: {next_state[0]}"
        )

        if "net_result" in info:
            print(f"Net Scrum Result: {info['net_result']} - Reward: {reward}")
        else:
            print(f"Reward: {reward}")

        if info.get("loan_triggered"):
            print(
                f"Mandatory Loan Triggered: Yes - "
                f"Interest Due Next Turn: {next_state[6]}"
            )

        if done and "terminal_reason" in info:
            print(f"Game Over: {info['terminal_reason']}")

        print("")

        state = next_state
        turn_number += 1


if __name__ == "__main__":
    play_demo_game()
