# Scrum Game Deep RL: Assessment Q&A Cheat Sheet

This guide presents direct answers to typical questions that assessors might ask about your codebase and the underlying reinforcement learning theory.

---

### Q1: Can you explain standard DQN (Deep Q-Network)?
**Answer**:
- **Concept**: DQN combines Q-Learning (a tabular reinforcement learning method) with Deep Neural Networks. Instead of maintaining a huge table of states and actions, DQN uses a neural network to approximate the action-value function $Q(s, a; \theta)$, which estimates the expected future return for taking action $a$ in state $s$.
- **Input/Output**: The network takes the state vector (in our case, the normalized money, active product, sprints, etc.) as input, and outputs a Q-value for each possible action (continue, switch to product 1, switch to product 2, etc.).
- **Action Selection**: The agent selects the action with the highest Q-value ($\text{argmax}_a Q(s, a)$), or chooses a random action with probability $\epsilon$ (epsilon-greedy exploration).
- **Goal**: Minimize the Mean Squared Bellman Error between the network's prediction and the target value:
  $$\text{Loss} = \mathbb{E} \left[ \left( R + \gamma \max_{a'} Q(S', a'; \theta^-) - Q(S, A; \theta) \right)^2 \right]$$

---

### Q2: What is the difference between standard DQN and Double DQN (DDQN)? Why did you use DDQN?
**Answer**:
- **Standard DQN Overestimation**: Standard DQN uses the same network parameters to select the best action and evaluate its value in the next state: $\max_{a} Q(S', a; \theta)$. Because Q-value estimates are noisy, taking the maximum of noisy values introduces a positive **maximization bias** (the agent overestimates how good states are).
- **Double DQN Decoupling**: DDQN solves this by using two separate sets of network parameters: the online network ($\theta$) and the target network ($\theta^-$):
  1. **Selection (Online Network)**: Find which action is best: $a^* = \text{argmax}_a Q(S', a; \theta)$.
  2. **Evaluation (Target Network)**: Evaluate that action: target $= R + \gamma Q(S', a^*; \theta^-)$.
- **Why we used it**: Decoupling action selection and evaluation stabilizes training, prevents Q-values from exploding, and leads to a much more robust and stable policy.

---

### Q3: What is the Replay Buffer? Why is it crucial for training?
**Answer**:
- **What it is**: The Replay Buffer is a memory cache that stores the agent's past experiences as transitions: $(s, a, r, s', \text{done})$. During training, instead of learning only from the immediate step, we sample a random batch of transitions from this buffer to perform gradient updates.
- **Why it is crucial**:
  1. **Breaks Temporal Correlation**: Sequential steps in a game are highly correlated (state $s_{t+1}$ depends directly on $s_t$). Neural networks assume training samples are Independent and Identically Distributed (IID). Sampling randomly from the buffer breaks this correlation.
  2. **Data Efficiency**: Every step the agent takes is saved and reused multiple times for training updates, maximizing the value of every simulation roll.
  3. **Prevents Policy Oscillations**: It prevents the agent from forgetting older experiences when learning new ones.

---

### Q4: Why do you have a Target Network, and how is it updated?
**Answer**:
- **Why we have it**: If we use the same network to compute both the current $Q(s, a)$ and the target value $R + \gamma \max_{a'} Q(s', a')$, the target shifts constantly with every single gradient update. This is like a dog chasing its own tail—training becomes highly unstable. The target network ($\theta^-$) provides stable target values.
- **How it is updated**: The target network is a copy of the online network. It is updated periodically in one of two ways:
  - **Hard Update**: Copying the online network weights directly every $C$ steps (e.g., target_update_frequency = 2000).
  - **Soft Update**: Slowly blending the weights every step: $\theta^- \leftarrow \tau \theta + (1 - \tau)\theta^-$ where $\tau \ll 1$.
  Our codebase implements periodic target updates to ensure stability.

---

### Q5: How does your Gymnasium-style environment represent states and actions?
**Answer**:
- **State Representation**: The raw observation is a dictionary tracking metrics like cash balance, current product, current sprint index, required features, win probability, expected value, loan states, and completed status lists. The `encode_state` function flattens and normalizes these values into a single vector of float values between $-1.0$ and $1.0$ so the neural network can ingest it stably.
- **Action Space**: It is a Discrete action space of size $1 + N$ (where $N$ is the number of products). Action `0` is to continue on the active product. Actions `1` to `N` represent switching to that specific product.

---

### Q6: Explain the reward shaping logic. Why not just give +1 for winning and -1 for losing?
**Answer**:
- **Sparse Reward Problem**: A simple +1/-1 reward is "sparse"—the agent might play thousands of turns randomly before accidentally winning or losing, receiving no feedback in between. Learning would take millions of episodes.
- **Shaped Reward**: We guide the agent at each step by adding intermediate rewards:
  - **Wealth Delta**: The raw money difference ($\Delta \text{money} = \text{new\_money} - \text{old\_money}$). This aligns the agent with cash flow management.
  - **Sprint Success Bonus**: $+5000$ when a sprint is successfully completed, encouraging delivery velocity.
  - **Debt Penalty**: A penalty of $-100 \times \text{turns\_with\_loan}$ to encourage the agent to pay off loans quickly.
  - **Invalid Action Penalty**: $-2000$ if the agent chooses an illegal action, teaching it the rules of the board.

---

### Q7: What is Domain Randomization, and why is it implemented?
**Answer**:
- **What it is**: During training, the environment randomizes game configuration parameters (starting money, max turns, switch costs, or card values) within bounds every $N$ episodes.
- **Why it is implemented**: It forces the agent to learn generalized strategies rather than memorizing the optimal path for a single static board layout. This makes the agent robust to rule changes, ensuring it performs well in randomized test configurations (evaluated in `evaluate_ddqn_robustness.py`).

---

### Q8: How does the environment compute exact win probabilities without simulation?
**Answer**:
- **Standard Method**: Roll dice 10,000 times (Monte Carlo simulation) and count how many times the sum matches the target. This is too slow to run inside the step loop.
- **Our Method**: We compute the exact probability distribution of the sum of dice rolls using **Discrete Convolutions** of their probability mass functions (PMFs) at class level and cache the result. The win probability is look up instantly in $O(1)$ from `_win_prob_cache`, making the simulation step loop extremely fast.

---

### Q9: Explain the deferred imports pattern in the backend.
**Answer**:
- **Answer**: Heavy libraries like PyTorch and Matplotlib take a long time to import and require large ML dependencies. If we imported them globally, the lightweight FastAPI dashboard server would take several seconds to boot and would fail to start entirely in environments that lack PyTorch. We defer these imports inside the functions that actually execute ML tasks, keeping the web API fast and lightweight.
