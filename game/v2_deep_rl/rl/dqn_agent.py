"""
Double Deep Q-Network (Double DQN) Agent for the Scrum Game.

This module implements the reinforcement learning agent, its neural network,
the experience replay buffer, and state vector encoding helpers.

Double DQN (DDQN) Architecture:
  - Policy (Online) Network: Used to choose actions and trained via backpropagation.
  - Target Network: A delayed copy of the policy network, used to compute target Q-values.
  - Rationale: Using two separate networks helps prevent Q-value overestimation bias,
    stabilizing gradient updates during training.

State Encoding and Context-Aware RL:
  - `encode_state` transforms the state dictionary into a flat list of normalized floats in [0, 1].
  - `_encode_rule_context` appends active game rule parameters (costs, loan limits, penalty settings)
    directly into the observation vector. This allows a single neural network model to generalize
    and play effectively even when game rules are randomized (Domain Randomization).

Connections:
  - Drives: `game_runtime.scrum_game_env.ScrumGameEnv` by selecting action indices.
  - Orchestrated by: `training.train_dqn` (which gathers transitions, runs step updates, and calls optimizer).
  - Saved/Loaded by: `rl.checkpoint_utils` to save/restore model parameters.
"""

from collections import deque
import math
import random

import torch
from torch import nn


class ReplayBuffer:
    """
    Experience Replay memory buffer. Stores agent transitions to break correlation
    between consecutive environment steps by sampling batches uniformly at random.
    """

    def __init__(self, capacity: int = 100000):
        """Initialize the buffer with a maximum capacity."""
        self.buffer = deque(maxlen=capacity)

    def push(self, state: list[float], action: int, reward: float, next_state: list[float], done: bool):
        """Append a new transition tuple to the buffer, evicting the oldest if capacity is reached."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """Randomly sample a batch of transitions for training."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

    def state_dict(self) -> dict:
        """Return a snapshot dictionary representing the buffer content for checkpoint serialization."""
        return {
            "capacity": self.buffer.maxlen,
            "buffer": list(self.buffer),
        }

    def load_state_dict(self, state: dict):
        """Restore replay memory from a checkpoint state dictionary."""
        capacity = int(state.get("capacity") or self.buffer.maxlen or 100000)
        self.buffer = deque(state.get("buffer", []), maxlen=capacity)


class QNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) representing the Q-value function.
    Given a state vector input, maps it to the expected Q-value for each discrete action.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Set up the linear network layers.
        
        Args:
            input_dim: Length of the state observation vector.
            output_dim: Action space size (1 + number of products).
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action Q-values for the input tensor batch."""
        return self.network(x)


class DQNAgent:
    """
    Double DQN Agent managing network instances, action selection, and optimization updates.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        learning_rate: float = 0.0005,
        gamma: float = 0.85,
        replay_capacity: int = 100000,
        batch_size: int = 128,
        target_update_frequency: int = 2000,
        device: str | None = None,
    ):
        """
        Initialize the agent with training hyperparameters and policy/target networks.
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma  # Discount factor for future rewards
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # The active policy network (updated every train step)
        self.policy_network = QNetwork(input_dim=state_dim, output_dim=num_actions).to(self.device)
        # The target network (updated periodically by copying policy network weights)
        self.target_network = QNetwork(input_dim=state_dim, output_dim=num_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        # Huber Loss (Smooth L1 Loss) is less sensitive to outliers than MSE
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.training_steps = 0

    def training_state_dict(self) -> dict:
        """Snapshot optimizer and replay state for full continuation checkpoints."""
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "training_steps": self.training_steps,
        }

    def load_training_state_dict(self, state: dict, include_replay: bool = True):
        """Restore optimizer state, replay memory, and training counter from a snapshot."""
        optimizer_state = state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        if include_replay and state.get("replay_buffer") is not None:
            self.replay_buffer.load_state_dict(state["replay_buffer"])

        if state.get("training_steps") is not None:
            self.training_steps = int(state["training_steps"])

    def choose_action(self, state_vector: list[float], epsilon: float) -> int:
        """
        Select action using Epsilon-Greedy exploration strategy.
        
        Args:
            state_vector: Flattened, normalized observation vector.
            epsilon: Current exploration rate (chance of picking a random action).
            
        Returns:
            int: Action index (0 = continue, 1..N = switch to product N).
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        q_values = self.predict_q_values(state_vector)
        return max(range(self.num_actions), key=lambda action: q_values[action])

    def choose_action_with_temperature(self, state_vector: list[float], temperature: float = 1.0) -> int:
        """
        Sample action from Q-values using Softmax temperature scaling.
        Lower temperature makes the agent more deterministic (choosing the best action);
        higher temperature increases randomness.
        """
        q_values = self.predict_q_values(state_vector)

        if temperature <= 0:
            return max(range(self.num_actions), key=lambda action: q_values[action])

        # Softmax computation with numerical stability (subtracting max)
        scaled = [value / temperature for value in q_values]
        max_scaled = max(scaled)
        weights = [math.exp(value - max_scaled) for value in scaled]
        weight_sum = sum(weights)
        probabilities = [weight / weight_sum for weight in weights]
        return random.choices(range(self.num_actions), weights=probabilities, k=1)[0]

    def predict_q_values(self, state_vector: list[float]) -> list[float]:
        """Compute the expected Q-value for each action in the given state."""
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_network(state_tensor).squeeze(0).cpu().tolist()
        return q_values

    def store_transition(self, state: list[float], action: int, reward: float, next_state: list[float], done: bool):
        """Append transition to experience replay memory."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """
        Sample a mini-batch from the ReplayBuffer and perform a Double DQN optimization step.
        
        Double DQN Target Formulation:
          Y_t = R_{t+1} + gamma * Q_target(S_{t+1}, argmax_a Q_policy(S_{t+1}, a))
          
        Steps:
          1. Predict Q(S, A) using Policy Network.
          2. Find best action A* for S_next using Policy Network: A* = argmax_a Q_policy(S_next, a).
          3. Evaluate Q_target(S_next, A*) using Target Network.
          4. Compute target reward sum and compare using Smooth L1 Loss.
          5. Update policy weights, clip gradients to prevent exploding gradients,
             and periodically synchronize Target Network.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Cast to PyTorch tensors
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Get policy predictions for actions taken: Q_policy(S, A)
        current_q_values = self.policy_network(states_tensor).gather(1, actions_tensor).squeeze(1)

        # Double DQN target computation:
        with torch.no_grad():
            # Action selection: argmax_a Q_policy(S_next, a)
            next_policy_actions = self.policy_network(next_states_tensor).argmax(dim=1, keepdim=True)
            # Action evaluation: Q_target(S_next, selected_action)
            next_target_values = self.target_network(next_states_tensor).gather(1, next_policy_actions).squeeze(1)
            # Bellman backup: Target Q = R + gamma * Q_target * (1 - done)
            target_q_values = rewards_tensor + self.gamma * next_target_values * (1.0 - dones_tensor)

        # Loss calculation
        loss = self.loss_function(current_q_values, target_q_values)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm to 10.0 to stabilize training (prevent parameter explosions)
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Target network update interval check
        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        return float(loss.item())


def encode_state(state: dict, env) -> list[float]:
    """
    Flatten and normalize the environment state dictionary into a vector of float values.
    
    Why Normalize?
      Raw money values range from -50k to 200k, while binary flags are 0 or 1.
      Neural networks perform best when input features have similar scales (mean near 0, range near 0..1).
      This function maps all values to a [0, 1] scale relative to physical game limits.
      
    Args:
        state: State dictionary returned by `ScrumGameEnv._get_state()`.
        env: The active environment instance (used to retrieve max interest, severity, etc.).
        
    Returns:
        list[float]: A flat list of normalized input features.
    """
    # 1. Normalize individual player's finance and progress values
    current_money = float(state["current_money"]) / max(env.mandatory_loan_amount * 4, 1)
    current_product = float(state["current_product"] - 1) / max(env.products_count - 1, 1)
    current_sprint = float(state["current_sprint"]) / max(env.sprints_per_product, 1)
    features_required = float(state["features_required"]) / 4.0
    sprint_value = float(state["sprint_value"]) / max(env.max_visible_sprint_value, 1)
    loan_active = float(bool(state["loan_active"]))
    interest_due = float(state["interest_due"]) / max(env.max_interest_reference, 1)
    win_probability = float(state["win_probability"])
    expected_value = float(state["expected_value"]) / max(env.max_visible_sprint_value, 1)
    remaining_turns = float(state["remaining_turns"]) / max(env.max_turns, 1)
    is_last_sprint = float(state["is_last_sprint"])
    debt_ratio = min(max(float(state["debt_ratio"]), 0.0), 2.0) / 2.0  # Clip to [0, 1] range representing [0, 2.0]
    switch_is_free = float(state["switch_is_free"])
    
    # 2. Normalize active incident card details
    incident_active = float(state["incident_active"])
    current_incident_id = float(state["current_incident_id"]) / 500.0
    current_incident_scope = float(state["current_incident_scope"])
    current_incident_delta = float(state["current_incident_delta"]) / max(env.max_visible_sprint_value, 1)
    current_refinement_delta = (
        float(state["current_refinement_delta"]) + env.max_refinement_reference
    ) / max(env.max_refinement_reference * 2, 1)
    current_product_completed = float(state["current_product_completed"])

    # Assemble base features
    vector = [
        current_money,
        current_product,
        current_sprint,
        features_required,
        sprint_value,
        loan_active,
        interest_due,
        win_probability,
        expected_value,
        remaining_turns,
        is_last_sprint,
        debt_ratio,
        switch_is_free,
        incident_active,
        current_incident_id,
        current_incident_scope,
        current_incident_delta,
        current_refinement_delta,
        current_product_completed,
    ]

    # 3. Append state information for ALL products (so the agent sees other products' status too)
    for next_sprint in state["target_next_sprints"]:
        vector.append(float(next_sprint) / max(env.sprints_per_product, 1))

    for features in state["target_features_required"]:
        vector.append(float(features) / 4.0)

    for tsv in state["target_sprint_values"]:
        vector.append(float(tsv) / max(env.max_visible_sprint_value, 1))

    for twp in state["target_win_probabilities"]:
        vector.append(float(twp))

    for tev in state["target_expected_values"]:
        vector.append(float(tev) / max(env.max_visible_sprint_value, 1))

    for completed_flag in state["target_is_completed"]:
        vector.append(float(completed_flag))

    for incident_delta in state["target_incident_deltas"]:
        vector.append(float(incident_delta) / max(env.max_visible_sprint_value, 1))

    for refinement_delta in state["target_refinement_deltas"]:
        vector.append((float(refinement_delta) + env.max_refinement_reference) / max(env.max_refinement_reference * 2, 1))

    for incident_flag in state["target_incident_flags"]:
        vector.append(float(incident_flag))

    # 4. Append game rule constants (allows model generalization across domain randomizations)
    vector.extend(_encode_rule_context(env))

    return vector


def _encode_rule_context(env) -> list[float]:
    """
    Encode static game rule parameters as input features.
    
    This context vector informs the network of costs, loans, penalties,
    and probabilities of the active environment configuration, enabling a
    single policy to play successfully under varying (domain-randomized) rules.
    """
    max_money_reference = max(env.mandatory_loan_amount * 2, env.starting_money, 1)
    max_cost_reference = max(env.mandatory_loan_amount, env.cost_switch_mid, env.cost_switch_after, 1)
    max_penalty_reference = max(abs(env.penalty_negative), abs(env.penalty_positive), 1)

    features = [
        float(env.starting_money) / max_money_reference,
        float(env.max_turns) / 15.0,
        float(env.cost_continue) / max_cost_reference,
        float(env.cost_switch_mid) / max_cost_reference,
        float(env.cost_switch_after) / max_cost_reference,
        float(env.mandatory_loan_amount) / max_money_reference,
        float(env.loan_interest) / max(env.mandatory_loan_amount, 1),
        float(env.penalty_negative) / max_penalty_reference,
        float(env.penalty_positive) / max_penalty_reference,
        float(env.daily_scrums_per_sprint) / 10.0,
        float(env.daily_scrum_target) / 30.0,
        float(env.incident_draw_probability),
        float(env.incident_severity_multiplier) / 3.0,
        float(env.incidents_active),
        float(env.refinements_active),
    ]

    # Append features/dice counts mapping rules
    for rule in env.dice_rules:
        features.append(float(rule.dice_count) / 4.0)
        features.append(float(rule.dice_sides) / 20.0)

    return features

