# Scrum Game RL Project - Domain Knowledge

## 1. Project Overview
We are building a Reinforcement Learning (RL) agent to act as a Non-Player Character (NPC) in a digital board game called the "Scrum Game" for Witan Entertainment. The game revolves around managing software sprints, balancing financial risk, and avoiding bankruptcy.

## 2. Deliverable B Strict Requirements
[cite_start]This project is graded under an adapted RL rubric [cite: 292-299]. The following rules are absolute:
* [cite_start]**No Traditional Train/Test Split:** We use Training Episodes (to learn the policy) and Evaluation Episodes (to measure performance) [cite: 300-305].
* [cite_start]**No Data Leakage:** Absolutely NO learning ($\alpha = 0$, $\epsilon = 0$) can occur during Evaluation Episodes [cite: 306-308].
* **Baseline:** The baseline is NOT an accuracy score. [cite_start]It is the average reward of a simple Heuristic or Random policy [cite: 310-317].
* [cite_start]**Models:** We must implement and compare at least 3 models (e.g., Q-Learning, SARSA, Monte Carlo) [cite: 318-324].

## 3. Environment Architecture
The environment must be structured like a standard OpenAI Gym environment (`reset` and `step` functions).

### [cite_start]State Space [cite: 335, 339]
The observation the agent receives at the start of its turn:
* `current_money` (integer)
* `current_product` (categorical ID)
* `current_sprint` (integer, e.g., 1 to 4)
* `features_required` (integer, determines dice roll risk)
* `sprint_value` (integer, potential payout)
* `loan_active` (boolean, 1 if in debt, 0 otherwise)
* `interest_due` (integer, penalty deducted per turn if loan_active)

### [cite_start]Action Space [cite: 335, 340]
The agent has 3 discrete actions:
* `0`: Continue Sprint (Rolls dice based on `features_required`. Success = +value, Fail = penalty).
* `1`: Switch Product (Costs a fixed fee, resets sprint progress).
* `2`: Take Loan (Adds a large sum of money to avoid bankruptcy, but sets `loan_active` to True and incurs constant `interest_due`).

### [cite_start]Reward Function [cite: 335, 341]
The scalar feedback signal:
* Positive reward for net financial gain after a successful sprint.
* Negative reward for failed sprints or switching costs.
* Continuous negative reward each turn a loan is active.
* Massive negative penalty for bankruptcy (money < 0).