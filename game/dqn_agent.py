from collections import deque
import random

import torch
from torch import nn


class ReplayBuffer:
    """Fixed-size replay memory for DQN transitions."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Simple MLP that maps state vectors to two action-values."""

    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """PyTorch DQN agent for the Scrum Game environment."""

    def __init__(
        self,
        state_dim=8,
        num_actions=2,
        learning_rate=0.0005,
        gamma=0.85,
        replay_capacity=100000,
        batch_size=64,
        target_update_frequency=1000,
        device=None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_network = QNetwork(input_dim=state_dim, output_dim=num_actions).to(self.device)
        self.target_network = QNetwork(input_dim=state_dim, output_dim=num_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.training_steps = 0

    def choose_action(self, state_vector, epsilon):
        """Select an action using epsilon-greedy exploration."""
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Add one transition to replay memory."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Sample a batch from replay memory and update the policy network."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q_values = self.policy_network(states_tensor).gather(1, actions_tensor).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(dim=1).values
            target_q_values = rewards_tensor + self.gamma * next_q_values * (1.0 - dones_tensor)

        loss = self.loss_function(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        return float(loss.item())


def encode_state(state, env):
    """
    Convert the raw environment tuple into a normalized float vector.

    The encoding keeps the original seven state features but scales them to
    ranges that are easier for an MLP to learn from.
    """
    current_money, current_product, current_sprint, features_required, sprint_value, loan_active, interest_due, win_probability = state

    return [
        float(current_money) / max(env.mandatory_loan_amount * 4, 1),
        float(current_product - 1) / max(env.products_count - 1, 1),
        float(current_sprint - 1) / max(env.sprints_per_product - 1, 1),
        float(features_required) / 4.0,
        float(sprint_value) / max(env.ring_value * 7, 1),
        float(bool(loan_active)),
        float(interest_due) / max(env.loan_interest * 4, 1),
        float(win_probability),
    ]
