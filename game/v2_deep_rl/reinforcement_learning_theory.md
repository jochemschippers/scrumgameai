# Scrum Game AI: Deep Reinforcement Learning & Mathematical Theory

This document provides a rigorous theoretical overview of the reinforcement learning algorithms, mathematical proofs, and environment calculations used in the Scrum Game (v2) Deep RL project. Use this guide to prepare for theoretical questions during your assessment.

---

## 1. Markov Decision Processes (MDP) & Game Modeling

Any reinforcement learning task is formulated as a **Markov Decision Process (MDP)**, defined by the 5-tuple $(S, A, P, R, \gamma)$:

1. **State Space ($S$)**: The set of all valid environment configurations. In our codebase, the state $s \in S$ is represented by a dictionary of financial status, active product, sprint indices, refinement values, and win probabilities.
2. **Action Space ($A$)**: The set of all possible decisions the agent can make. In our model, $A = \{0, 1, \dots, N\}$:
   - $a = 0$: Continue working on the current product.
   - $a \in [1, N]$: Switch to a different product $i$, paying the corresponding switch cost.
3. **Transition Probability Function ($P$)**: The probability $P(s' \mid s, a)$ of moving to state $s'$ after action $a$ in state $s$. In the Scrum Game, transitions are stochastic, governed by dice rolls and randomized incident card draws.
4. **Reward Function ($R$)**: The mapping $R(s, a, s') \to \mathbb{R}$ that provides feedback.
5. **Discount Factor ($\gamma$)**: A scalar $\gamma \in [0, 1)$ determining the present value of future rewards. In our configuration, $\gamma = 0.85$, signifying that immediate cash flow is heavily prioritized over long-term steps.

### State Normalization & Encoding
Before feeding the state $s$ into the neural network, `encode_state` normalizes all features to a $[-1.0, 1.0]$ or $[0.0, 1.0]$ range. Neural networks optimize parameters much more stably when inputs have similar scales (preventing exploding gradients):
$$\text{Normalized Money} = \frac{\text{current\_money}}{\text{max\_money\_scale}}$$

---

## 2. Deep Q-Learning & Double DQN (DDQN)

### The Bellman Optimality Equation
The goal of the agent is to find an optimal policy $\pi^*$ that maximizes the expected cumulative discounted reward:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

The action-value function $Q^\pi(s, a)$ represents the expected return starting from state $s$, taking action $a$, and following policy $\pi$ thereafter. The optimal action-value function $Q^*(s, a)$ satisfies the **Bellman Optimality Equation**:
$$Q^*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \;\middle|\; S_t = s, A_t = a \right]$$

### The Maximization Bias & Standard DQN
In standard Deep Q-Networks (DQN), the network weights $\theta$ are updated to minimize the mean squared Bellman error. The target value $Y_t^{DQN}$ is computed using the same network parameter set $\theta_t$ for both selecting and evaluating the next action:
$$Y_t^{DQN} = R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a; \theta_t)$$

Because the target is determined by taking the maximum over noisy, estimated Q-values, standard DQN systematically suffers from **maximization bias** (overestimating Q-values). This causes training instability and slow convergence.

### Double DQN (DDQN) Solution
Double DQN (van Hasselt et al., 2015) decouples the action selection from the action evaluation. It uses two separate sets of network parameters: the online network ($\theta_t$) and the target network ($\theta_t^-$):
1. **Selection**: Select the best action $a^*$ in the next state using the online network $\theta_t$:
   $$a^* = \text{argmax}_{a} Q(S_{t+1}, a; \theta_t)$$
2. **Evaluation**: Evaluate the selected action $a^*$ using the target network $\theta_t^-$:
   $$Y_t^{DoubleQ} = R_{t+1} + \gamma Q\left(S_{t+1}, \text{argmax}_{a} Q(S_{t+1}, a; \theta_t); \theta_t^-\right)$$

By selecting the action with one network and evaluating it with another, the probability of propagating accidental positive estimation errors is significantly reduced, stabilizing training.

---

## 3. Environment Mathematics: Discrete Convolutions

To calculate the exact win probability of a sprint, the environment must determine the probability that the sum of $n$ discrete dice rolls is strictly less than or equal to a target daily scrum threshold. 

Rather than running slow Monte Carlo simulations (which are stochastic and slow down the RL step loop), the environment precomputes the exact probability distribution using **Discrete Convolutions**.

### The Convolution Math
Let $X_1, X_2, \dots, X_k$ be independent random variables representing the results of $k$ individual dice rolls. The probability mass function (PMF) of a single die with $d$ sides (e.g., $d=6$) is:
$$P(X = x) = \begin{cases} \frac{1}{d} & \text{if } x \in \{1, 2, \dots, d\} \\ 0 & \text{otherwise} \end{cases}$$

The PMF of the sum of two dice $S_2 = X_1 + X_2$ is the convolution of their individual PMFs:
$$P(S_2 = s) = (P_{X_1} * P_{X_2})[s] = \sum_{y=-\infty}^{\infty} P_{X_1}(y) P_{X_2}(s - y)$$

For $k$ dice, we perform the convolution $k-1$ times recursively. Once we have the combined PMF of the sum $S_k$, the exact win probability for a target daily scrum threshold $T$ is the Cumulative Distribution Function (CDF):
$$P(S_k \le T) = \sum_{s=k}^{T} P(S_k = s)$$

This exact mathematical probability is cached at the class level in `_win_prob_cache`, allowing instant $O(1)$ lookups during simulation steps.

---

## 4. Exploration & Optimization Mechanics

### Epsilon-Greedy Exploration Decay
To balance exploration (learning new strategies) and exploitation (using what has already been learned), we use an $\epsilon$-greedy strategy. The exploration probability $\epsilon$ decays linearly over time:
$$\epsilon_t = \max \left( \epsilon_{min}, \epsilon_{start} - t \times \frac{\epsilon_{start} - \epsilon_{min}}{\text{epsilon\_decay\_episodes}} \right)$$

### Replay Buffer
To train the neural network, transitions $(s_t, a_t, r_{t+1}, s_{t+1}, d_t)$ are stored in a **Replay Buffer**. During the backpropagation step, a random batch of transitions is sampled uniformly from the buffer. This is crucial because:
1. It breaks the **temporal correlation** of sequential steps, satisfying the Independent and Identically Distributed (IID) assumption of gradient descent.
2. It allows transitions to be reused multiple times for weight updates, increasing data efficiency.

---

## 5. Reward Shaping & Game Theory

If the reward is purely sparse (e.g., $+1$ for winning the game, $-1$ for losing), the agent will take a very long time to learn because it rarely encounters a positive signal. We shape the reward to guide the agent:
$$R_{\text{shaped}} = \Delta \text{money} + R_{\text{success\_bonus}} - R_{\text{debt\_penalty}} - R_{\text{invalid\_action\_penalty}}$$

- **Wealth Delta**: The raw cash change $\Delta \text{money} = \text{new\_money} - \text{old\_money}$. This aligns the agent with the core business objective: maximizing profitability.
- **Success Bonus**: $+5000$ upon completing a sprint. This encourages finishing sprints quickly to maximize velocity.
- **Debt Penalty**: $-100 \times \text{turns\_with\_loan}$. This applies progressive financial pressure to clear loans quickly without double-penalizing the agent since interest payments are already deducted from cash.
- **Invalid Action Penalty**: $-2000$ for selecting illegal actions (e.g., self-switching). This teaches the model the rules of the board.

---

## 6. Autopilot Hyperparameter Diagnostics

During autopilot cycles, the system analyzes logs to classify training behaviors:
- **Plateau Detection**: If the rolling average reward changes by less than 2% over a window of evaluation epochs (when exploration is low, $\epsilon \le 0.10$), the agent is considered to have hit a plateau.
- **Coefficient of Variation (CV)**: Used to assess training noise:
  $$\text{CV} = \frac{\sigma}{\mu}$$
  Where $\mu$ is the mean reward over the window and $\sigma$ is the standard deviation. A high CV ($>0.20$) indicates high noise variance, prompting the autopilot to reduce the learning rate to stabilize convergence.
