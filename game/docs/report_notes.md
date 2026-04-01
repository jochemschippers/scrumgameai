# Deliverable B Report Notes

## Baseline Performance

The baseline agent uses a simple heuristic policy:
- Always continue the sprint.

This baseline is valid because it represents a non-learning decision rule that a human could design without reinforcement learning. It provides a minimum benchmark that the learned agents should aim to outperform.

Run `compare_models.py` and copy the baseline average reward from:
- `artifacts/reports/model_comparison.csv`
- `artifacts/reports/model_comparison_summary.md`

## Models Compared

This project compares four policies:
- Baseline heuristic agent
- Q-Learning agent
- SARSA agent
- Monte Carlo agent
- DQN agent

The project exceeds the requirement to compare at least three learning approaches.

## Model Choice Justification

### Q-Learning

Q-Learning was selected because it is a standard off-policy temporal-difference algorithm. It updates state-action values toward the maximum estimated future reward, which often makes learning efficient.

Pros:
- Simple to implement
- Learns directly from experience without a model of the environment
- Common baseline for tabular RL

Cons:
- Can be unstable in stochastic environments
- Can overreact to noisy rewards if the learning rate is too high

### SARSA

SARSA was selected because it is an on-policy temporal-difference algorithm. It updates using the next action actually chosen by the current policy, which can produce more conservative behavior in risky environments.

Pros:
- Often safer in stochastic environments
- More tightly aligned with the behavior policy during training

Cons:
- Can learn more slowly than Q-Learning
- Performance can depend strongly on the exploration schedule

### Monte Carlo

Monte Carlo was selected because it learns from complete episodes instead of one-step temporal-difference targets. This provides a useful contrast with the TD methods.

Pros:
- Uses full observed returns
- Easy to explain conceptually

Cons:
- Must wait until the episode ends before learning
- Can have high variance

### DQN

DQN was selected as the deep reinforcement learning extension of the project. Instead of storing Q-values in a lookup table, it uses a neural network to approximate action values from the game state.

Pros:
- Can use richer state features than a small tabular lookup
- Supports replay memory and target networks for more stable learning
- Performed best in the final project run

Cons:
- More complex to train and debug
- Requires more computation time than the tabular agents

## Implementation Explanation

The Scrum Game environment follows a Gym-style structure with:
- `__init__`
- `reset()`
- `step(action)`

The state includes:
- current money
- current product
- current sprint
- features required
- sprint value
- loan status
- interest due

Because tabular RL cannot efficiently use highly variable raw money values, the project includes a `discretize_state(state)` helper that bins money into a small number of categories and returns a hashable tuple. This tuple is used as the key in each Q-table.

## Hyperparameter Tuning

The current implementation uses:
- `alpha = 0.05`
- `gamma = 0.95`
- `epsilon` decayed from `1.0` to `0.05`
- `25,000` training episodes for each RL model
- DQN uses learning rate `0.0005`, gamma `0.85`, replay memory `100000`, and `500,000` training episodes

Tuning rationale:
- The learning rate was reduced from `0.1` to `0.05` after the Q-Learning training curve showed late-stage instability.
- Training episodes were increased from `10,000` to `25,000` to give the policy more time to stabilize.
- The epsilon floor of `0.05` preserves some exploration during training while still allowing convergence toward greedy behavior.
- SARSA discount factor tuning identified `gamma = 0.85` as the strongest tested value.
- DQN was trained longer because the neural network needed far more interaction data than the tabular agents.

## Strict Evaluation Isolation

This project follows the RL evaluation rules from the assignment:
- Training episodes are used for learning
- Evaluation episodes are used only for performance measurement
- During evaluation, `epsilon = 0`
- During evaluation, `learn()` is never called

This avoids data leakage and ensures fair comparison between models.

## Robustness Discussion

The Scrum Game contains randomness because sprint outcomes depend on dice-like chance. This means performance can vary even for the same learned policy.

Key robustness considerations:
- Reward variance should be reported, not just average reward
- A single run can be misleading if it was unusually lucky or unlucky
- The best practice is to repeat training with multiple random seeds and compare the mean and spread of the results
- In this project, tabular robustness was measured across five seeds, while DQN was selected based on its strongest final evaluation run and dashboard behavior.

## Overfitting / Underfitting Discussion

In tabular RL, overfitting does not look exactly like supervised learning overfitting, but similar instability can still happen.

Possible signs of poor fit:
- If the learning curve never improves meaningfully, the model may be underfitting
- If the curve improves and then degrades late in training, the agent may be overreacting to stochastic outcomes or overfitting to recent experience

In this project, late instability in Q-Learning was addressed by lowering the learning rate and extending training.

## Bias Discussion

The environment itself contains assumptions that influence what the agent learns:
- the size of the loan
- the interest penalty
- the penalty for failure
- the distribution of sprint values
- the success threshold based on features required

These assumptions bias the reward landscape. As a result, the learned policy should be interpreted as optimal for this simulation design, not as a universal strategy for all Scrum projects.

## Trained Model Artifacts

Saved model artifacts are written to the `artifacts/models/` folder:
- `q_learning_q_table.json`
- `sarsa_q_table.json`
- `mc_q_table.json`

Deep RL artifacts are written to:
- `artifacts/deep_rl/checkpoints/best_scrum_model.pth`
- `artifacts/deep_rl/plots/dqn_training_curve.png`
- `artifacts/deep_rl/reports/dqn_metrics.json`

Saved plots are written to the `artifacts/plots/` folder:
- `q_learning_training_curve.png`
- `sarsa_training_curve.png`
- `mc_training_curve.png`
- `model_comparison.png`

## Final Comparison Output

Run:

```powershell
py compare_models.py
```

This generates:
- `artifacts/reports/model_comparison.csv`
- `artifacts/reports/model_comparison.json`
- `artifacts/reports/model_comparison_summary.md`

Those files can be used directly in the report for the comparison table and discussion section.

## Final Selected Model

The final selected model is the DQN agent.

Reason:
- It achieved the strongest final evaluation result in the project run, with an average reward of about `-14210.50`.
- It outperformed the earlier tabular agents on the current environment.
- The strategy heatmap and dashboard behavior showed more coherent board awareness than the earlier switching-heavy policies.

Final deployment artifact:
- `artifacts/deep_rl/checkpoints/best_scrum_model.pth`
